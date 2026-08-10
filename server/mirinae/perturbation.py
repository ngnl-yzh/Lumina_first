"""적대적 섭동 — PGD + SNR 정규화 + 마스킹 투영.

PGD의 본질은 gradient가 아니라 **투영**이다. "제약 안에서 최대한 강하게"가 목표이므로
매 스텝 제약 위로 삐져나온 것을 도로 눌러넣는 연산이 성능을 결정한다.

D02 검토에서 확인된 결함 2건이 여기 반영되어 있다. 둘 다 조용히 틀리는 종류다.

  ① 정규화 순서 — 정규화를 루프 **밖**에 두면 섭동이 마스킹 임계값 밖으로 밀려난다.
     실측에서 샘플의 94.9%가 위반, 최대 14.3배 초과였다. "안 들린다"는 보장이 사라진다.
     따라서 정규화는 루프 **안**, 투영 **앞**에 온다.

  ② 투영식 0-나눗셈 — `D/abs(D)`는 무음 구간에서 NaN을 만들고,
     NaN은 ISTFT에서 프레임 전체로 번져 **에러 없이** 최적화가 멈춘다.

구현 중 같은 부류의 결함 1건을 추가로 발견해 함께 고쳤다.

  ③ δ 초기화 — 설계도 의사코드는 `delta = zeros_like(x)`로 시작한다.
     그런데 δ=0은 cos_sim(E(x+δ), E(x)) = 1, 즉 **목적함수의 최댓값이자 정류점**이다.
     실측 gradient 크기가 δ=0에서 1.0e-07, 랜덤 초기화에서 4.1e-01로 6자릿수 차이가 났고,
     δ=0에서는 전체 샘플의 22.8%가 gradient 정확히 0이라 sign(0)=0으로 움직이지도 않는다.
     첫 스텝 방향이 사실상 float32 반올림 오차로 정해진다. 역시 에러가 나지 않는 결함이다.
     표준 PGD가 랜덤 초기화를 쓰는 이유가 이것이므로 그렇게 고쳤다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .config import EPS, PGDConfig, SAMPLE_RATE, BAND_TAPER_HZ
from .encoder import EncoderEnsemble, SpeakerEncoder, cosine_similarity
from .psychoacoustic import MaskingModel
from .vad import speech_mask


# ── 제약 연산자 ───────────────────────────────────────────────────────────────

def normalize_snr(delta: torch.Tensor, x: torch.Tensor, target_db: float) -> torch.Tensor:
    """섭동의 크기를 목표 SNR로 고정한다.

    SNR = 10·log10(P_x / P_δ) 이므로 ‖δ‖ = ‖x‖ / 10^(SNR/20).
    이 다음에 오는 마스킹 투영이 크기를 더 깎기만 하므로,
    **최종 SNR은 항상 목표 이상(= 더 조용함)**이 된다. 그래서 전역 SNR을 따로 재측정해 보고한다.
    """
    x_norm = torch.sqrt(torch.clamp((x ** 2).sum(), min=EPS))
    d_norm = torch.sqrt(torch.clamp((delta ** 2).sum(), min=EPS))
    target_norm = x_norm / (10.0 ** (target_db / 20.0))
    return delta * (target_norm / d_norm)


def project_masking(
    delta: torch.Tensor,
    thr_mag: torch.Tensor,
    ratio: float,
    model: MaskingModel,
    repeats: int = 2,
) -> torch.Tensor:
    """마스킹 임계값 아래로 섭동을 눌러넣는다. 위상은 보존하고 크기만 깎는다.

    :param repeats: 투영을 몇 번 반복할지. 1회로는 불변식이 완전히 서지 않는다 —
        아래 `enforce_masking_bound` 설명 참조. 2회면 대부분의 초과가 사라지고
        비용은 STFT 왕복 한 번뿐이라 기본값으로 둔다.
    """
    n = delta.shape[-1]
    bound = thr_mag * ratio

    for _ in range(max(1, repeats)):
        D = model.stft(delta)
        mag = D.abs()

        # 결함 ② 수정 — 무음 구간(mag≈0)에서 0으로 나누지 않는다.
        scaled = D / torch.clamp(mag, min=EPS) * torch.minimum(mag, bound)
        D = torch.where(mag > EPS, scaled, torch.zeros_like(D))
        delta = model.istft(D, length=n)

    return delta


def enforce_masking_bound(
    delta: torch.Tensor,
    thr_mag: torch.Tensor,
    ratio: float,
    model: MaskingModel,
) -> torch.Tensor:
    """불변식을 **강제로** 성립시킨다 — 전역 스칼라 축소.

    왜 필요한가.
    STFT 도메인에서 크기를 깎은 스펙트로그램은 대개 *일관적이지 않다*.
    즉 그런 STFT를 갖는 시간축 신호가 존재하지 않는다. ISTFT를 거치면
    프레임 간 overlap-add로 에너지가 번져 일부 bin이 다시 임계값을 넘는다.
    실측: 투영만으로는 200회를 반복해도 최대 초과가 +8 dB 수준에서 수렴한다.

    전역 스칼라로 줄이면 모든 bin이 같은 비율로 내려가므로 불변식이 반드시 선다.
    대신 초과가 가장 큰 bin 하나가 전체 섭동 크기를 결정하므로 방어가 약해진다.
    **"안 들린다"를 보장할 것인가, 방어 강도를 유지하고 초과량을 정직하게 보고할 것인가** —
    이건 청취 평가로 결정할 사안이라 설정으로 열어 둔다. (PGDConfig.enforce_masking)
    """
    with torch.no_grad():
        mag = model.stft(delta).abs()
        bound = torch.clamp(thr_mag * ratio, min=EPS)
        over = float((mag / bound).max())
        if over > 1.0:
            delta = delta / over
    return delta


def band_limit(
    delta: torch.Tensor,
    low_hz: float,
    high_hz: float,
    sample_rate: int = SAMPLE_RATE,
    taper_hz: float = BAND_TAPER_HZ,
) -> torch.Tensor:
    """통화 대역(300~3400 Hz) 밖을 제거한다. 밖은 코덱에서 어차피 사라진다.

    브릭월은 시간축 링잉을 만들므로 전이대역을 조금 둔다.
    """
    n = delta.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=delta.device)

    # 전이대역은 **통과대역 안쪽**에 둔다. 바깥에 두면 300 Hz 미만에도 에너지가 남아
    # "대역 밖 < −60 dB"가 깨진다. 안쪽에 두면 대역 밖 응답이 정확히 0이다.
    resp = torch.ones_like(freqs)
    resp = resp * torch.clamp((freqs - low_hz) / taper_hz, 0.0, 1.0)
    resp = resp * torch.clamp((high_hz - freqs) / taper_hz, 0.0, 1.0)

    spec = torch.fft.rfft(delta) * resp
    return torch.fft.irfft(spec, n=n)


# ── 결과 ──────────────────────────────────────────────────────────────────────

@dataclass
class PerturbResult:
    delta: torch.Tensor
    protected: torch.Tensor
    srs: float                              # 최종 화자 유사도 (낮을수록 성공)
    srs_initial: float
    snr_db: float                           # 실측 전역 SNR
    steps: int
    srs_trace: list[float] = field(default_factory=list)
    per_encoder_srs: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"SRS {self.srs_initial:.3f} → {self.srs:.3f} · "
            f"SNR {self.snr_db:.1f} dB · {self.steps}스텝"
        )


# ── PGD 본체 ──────────────────────────────────────────────────────────────────

def pgd_perturbation(
    x: torch.Tensor,
    encoders: EncoderEnsemble | SpeakerEncoder,
    cfg: PGDConfig | None = None,
    masking_model: MaskingModel | None = None,
    vad_mask: torch.Tensor | None = None,
    trace_every: int = 0,
    seed: int = 0,
    ref_embeds: list[torch.Tensor] | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> PerturbResult:
    """한 청크에 대한 섭동 δ를 계산한다.

    :param x: (n,) 16 kHz 모노 파형
    :param trace_every: >0이면 그 간격마다 SRS를 기록한다 (수렴 곡선 리포트용)
    :param seed: δ 초기화 난수 시드. 실험 재현성을 위해 청크 인덱스를 넘긴다.
    """
    cfg = cfg or PGDConfig()
    if isinstance(encoders, SpeakerEncoder):
        encoders = EncoderEnsemble([encoders])

    device = x.device
    model = masking_model or MaskingModel(
        SAMPLE_RATE, cfg.n_fft, cfg.hop_length, device=device
    )
    thr_mag, _ = model.threshold(x)
    mask = vad_mask if vad_mask is not None else speech_mask(x)

    def heard(w: torch.Tensor) -> torch.Tensor:
        """상대방이 실제로 듣게 될 신호.

        통화 채널의 지배적인 성분은 **대역 제한**이다 —
        `codec_test.py` 실측에서 대역 제한만으로 이미 방어가 무너졌고
        8 kHz 리샘플과 G.711 양자화는 그 위에 거의 아무것도 더하지 않았다
        (SRS 0.8420 → 0.8405 → 0.8346).

        그래서 미분 가능한 대역 제한 하나로 채널을 근사한다.
        μ-law 양자화는 미분이 안 되고, 실측상 기여도 작아 루프에 넣지 않는다.
        최종 검증은 `codec_test.py`가 **진짜 코덱**으로 한다 — 근사로 최적화하고
        근사로 검증하면 아무것도 증명하지 못한다.
        """
        if not cfg.channel_aware:
            return w
        return band_limit(w, cfg.band_low_hz, cfg.band_high_hz)

    # 기준 임베딩 — 최적화 내내 고정이므로 한 번만 계산한다.
    # channel_aware면 **채널을 통과한 원본**이 기준이 된다. 이게 핵심이다 —
    # 전대역 원본에서 멀어져 봐야 상대는 채널 통과본을 듣는다.
    #
    # `ref_embeds`가 주어지면 그것을 쓴다. **청크 분할에서 결정적이다.**
    #
    # 청크마다 자기 임베딩을 기준으로 삼으면 각자 다른 방향으로 밀어내고,
    # 복제 모델이 파일 전체를 볼 때 그 방향들이 서로 상쇄된다. 실측:
    #
    #   청크별 기준 (2초씩 따로)   전체 SRS 0.8045
    #   전체 발화 한 번에          전체 SRS 0.2658   ← 같은 60스텝·같은 SNR
    #
    # 공통 기준을 넘기면 모든 청크가 **같은 방향**을 향한다.
    if ref_embeds is not None:
        refs = [e.detach() for e in ref_embeds]
    else:
        with torch.no_grad():
            refs = [e.detach() for e in encoders.embeddings(heard(x))]

    # 결함 ③ 수정 — 정류점(δ=0)에서 출발하지 않는다.
    # 난수는 CPU에서 뽑아 장비가 바뀌어도 같은 결과가 나오게 한다.
    g_init = torch.Generator(device="cpu").manual_seed(seed)
    delta0 = torch.randn(x.shape, generator=g_init, dtype=x.dtype).to(device)
    delta0 = normalize_snr(delta0, x, cfg.target_snr_db)
    delta0 = band_limit(delta0, cfg.band_low_hz, cfg.band_high_hz) * mask
    delta = delta0.detach().clone().requires_grad_(True)

    trace: list[float] = []

    # 시간 예산 — 마감을 넘길 것 같으면 남은 스텝을 포기한다.
    #
    # 마지막 스텝을 절반만 돌고 끊으면 δ가 제약을 만족하지 않은 채 남는다.
    # 그래서 **스텝 경계에서만** 끊고, 한 스텝 더 돌 여유가 있는지를 직전 스텝의
    # 소요 시간으로 예측한다. 예측이 빗나가도 한 스텝만큼만 초과한다.
    import time as _time

    deadline = (_time.perf_counter() + cfg.time_budget_sec) \
        if cfg.time_budget_sec else None
    last_step_sec = 0.0
    steps_run = 0

    for step in range(cfg.steps):
        if deadline is not None and step > 0:
            if _time.perf_counter() + last_step_sec > deadline:
                break
        step_t0 = _time.perf_counter()

        embeds = encoders.embeddings(heard(x + delta))

        # untargeted — 원본에서 멀어지기만 하면 된다. 타깃 화자가 필요 없어
        # 구현이 단순하고 이중 용도 우려도 낮다.
        loss = torch.stack([
            -(1.0 - cosine_similarity(emb, ref)) for emb, ref in zip(embeds, refs)
        ]).sum()

        (grad,) = torch.autograd.grad(loss, delta)

        with torch.no_grad():
            delta.add_(-cfg.alpha * grad.sign())

            # ── 순서가 핵심이다 ──────────────────────────────────────────────
            delta.copy_(normalize_snr(delta, x, cfg.target_snr_db))   # ① 크기 고정
            delta.copy_(project_masking(delta, thr_mag,
                                        cfg.masking_ratio, model))    # ② 제약
            delta.copy_(band_limit(delta, cfg.band_low_hz,
                                   cfg.band_high_hz) * mask)          # ③ 대역·발화 구간

        steps_run = step + 1
        last_step_sec = _time.perf_counter() - step_t0
        if on_step is not None:
            on_step(steps_run, cfg.steps)

        if trace_every and (step % trace_every == 0 or step == cfg.steps - 1):
            with torch.no_grad():
                trace.append(float(cosine_similarity(
                    encoders.embeddings(heard(x + delta))[0], refs[0])))

    with torch.no_grad():
        delta_final = delta.detach()
        if cfg.enforce_masking:
            delta_final = enforce_masking_bound(delta_final, thr_mag,
                                                cfg.masking_ratio, model)
        protected = x + delta_final
        # 보고되는 SRS도 채널을 통과한 기준으로 잰다. 최적화 목표와 보고 지표가
        # 다르면 "무엇을 달성했는지"가 흐려진다.
        per_enc = {
            name: float(cosine_similarity(emb, ref))
            for name, emb, ref in zip(encoders.names,
                                      encoders.embeddings(heard(protected)), refs)
        }
        srs_initial = 1.0     # 무섭동 기준선. 원본 대 원본이므로 정의상 1 (C-C 대조군이 확인)
        snr = snr_db(x, delta_final)

    return PerturbResult(
        delta=delta_final,
        protected=protected,
        srs=per_enc[encoders.names[0]],
        srs_initial=srs_initial,
        snr_db=snr,
        # 실제로 돈 스텝 수. 시간 예산에 걸려 중단됐으면 cfg.steps보다 작다 —
        # 보고 값이 설정 값과 다를 수 있다는 것이 이 필드의 존재 이유다.
        steps=steps_run or cfg.steps,
        srs_trace=trace,
        per_encoder_srs=per_enc,
    )


def snr_db(x: torch.Tensor, delta: torch.Tensor) -> float:
    """실측 SNR. 청크마다 정규화하면 전체 SNR이 흔들리므로 최종 파형에서 다시 잰다."""
    p_x = float((x ** 2).sum())
    p_d = float((delta ** 2).sum())
    return 10.0 * torch.log10(torch.tensor(max(p_x, EPS) / max(p_d, EPS))).item()
