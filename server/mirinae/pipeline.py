"""전체 발화 보호 파이프라인 — 청크 분할 → PGD → overlap-add 조립 → 대조군 → 지표.

D09 §4.2의 처리 흐름을 오프라인으로 실행한다.
W2에서 WebSocket 스트리밍으로 감쌀 때도 이 함수가 그대로 워커 안에 들어간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .chunking import overlap_add, split
from .config import CHUNK_SEC, HOP_SEC, PGDConfig, SAMPLE_RATE
from .controls import make_controls
from .encoder import EncoderEnsemble, SpeakerEncoder, cosine_similarity
from .metrics import AudibilityReport, audibility, band_energy_ratio_db, snr_db
from .perturbation import pgd_perturbation
from .psychoacoustic import MaskingModel


@dataclass
class ChunkRecord:
    """청크 하나의 처리 기록. 그대로 메타데이터 JSON이 된다."""

    index: int
    start_sec: float
    end_sec: float
    srs: float
    snr_db: float
    elapsed_sec: float


@dataclass
class ProtectionResult:
    original: torch.Tensor
    protected: torch.Tensor
    delta: torch.Tensor
    chunks: list[ChunkRecord]
    global_snr_db: float

    # 가청도를 **두 기준으로** 들고 다닌다. 하나만 보면 정반대로 읽힌다.
    #
    #   audibility      — thr × masking_ratio 대비. "내가 설정한 제약을 지켰는가"
    #   audibility_abs  — thr × 1.0 대비.         "사람 귀에 들리는가"
    #
    # 한동안 앞의 것만 보고했다. 배율이 1보다 크면 기준선 자체가 함께 올라가므로
    # **가청도를 실제보다 좋게 보고한다.** 실측(out/ref · 배율 3.0):
    #
    #   제약 기준   최대 초과 12.24 dB · 위반 2.44%
    #   절대 기준   최대 초과 21.78 dB · 위반 **20.30%**
    #
    # 같은 파일인데 "bin의 2.4%가 임계값 위"와 "20.3%가 위"로 갈린다.
    # 배율을 키울수록 이 괴리가 커져서, 제약 기준만 보면
    # "배율을 올릴수록 덜 들린다"는 정반대 결론에 도달한다.
    audibility: AudibilityReport
    audibility_abs: AudibilityReport | None = None
    out_of_band_db: float = 0.0
    srs_protected: float = 0.0
    srs_controls: dict[str, float] = field(default_factory=dict)

    # SRS를 무엇을 기준으로 쟀는지. 채널 인지 최적화를 쓰면 평가도 채널 통과 후로 한다 —
    # 목표와 보고 기준이 다르면 도구가 자기 결과를 부정한다.
    srs_basis: str = "파일 대 파일 (채널 없음)"
    total_sec: float = 0.0

    # 판정 임계값 — **C-D(타 화자) 대조군에서 실제로 측정한 값이다.**
    #
    # 한때 0.75였고, 근거 없이 정한 숫자였다. `calibrate_dsr.py`가 XTTS 내장 화자
    # 6명의 3초 조각을 만들어 같은 조건에서 두 분포를 쟀다.
    #
    #   같은 화자 (조각쌍 116개)   0.8622 ±0.0331
    #   타 화자   (조각쌍 664개)   0.5962 ±0.0944
    #   → EER 지점 0.7962 · 동일오류율 4.3%
    #
    # 동일오류율 4.3%면 이 인코더는 화자를 잘 가른다. 임계값을 믿고 쓸 수 있다.
    # 옛 0.75는 우연히 가까웠지만 그건 운이었지 근거가 아니었다.
    #
    # **여전히 잠정값인 이유** — 대상이 XTTS 합성 화자다. 합성 화자끼리의 거리가
    # 사람끼리의 거리와 같다는 보장이 없다. 팀원 녹음이 오면 같은 명령을 다시 돌린다.
    #     python calibrate_dsr.py out/speakers
    PROVISIONAL_THRESHOLD = 0.7962

    def report(self) -> str:
        labels = {
            "C-A": "C-A 백색잡음",
            "C-B": "C-B 대역제한잡음",
            "C-C": "C-C 무섭동",
            "C-E": "C-E 셔플 섭동",
        }
        rows = [("적대적 섭동 (미리내)", self.srs_protected)]
        rows += [(labels.get(k, k), v) for k, v in sorted(self.srs_controls.items())]

        lines = [
            f"청크 {len(self.chunks)}개 · 처리 {self.total_sec:.1f}초",
            f"전역 SNR {self.global_snr_db:.1f} dB · 대역 밖 {self.out_of_band_db:.1f} dB",
            # 절대 기준을 **먼저** 쓴다. "들리는가"에 답하는 것은 이쪽이다.
            f"가청도 (마스킹 임계값 대비) — {self.audibility_abs or self.audibility}",
            f"  참고: 설정한 제약(배율 포함) 대비 — {self.audibility}",
            "",
            f"SRS 기준 — {self.srs_basis}",
            "",
            f"{'조건':<20}{'SRS':>9}   {'잠정판정':<10}",
            "─" * 48,
        ]
        for name, s in rows:
            verdict = "다른 화자" if s < self.PROVISIONAL_THRESHOLD else "같은 화자"
            lines.append(f"{name:<20}{s:>9.4f}   {verdict}")

        # 대조군을 못 이기면 아무것도 주장할 수 없다 — 그 판정을 눈에 보이게 찍는다
        noise = [self.srs_controls[k] for k in ("C-A", "C-B") if k in self.srs_controls]
        if noise:
            best = min(noise)
            lines += ["", (
                f"대조군 대비: {best - self.srs_protected:+.3f} "
                + ("— 잡음 대비 우위 있음" if self.srs_protected < best
                   else "— 잡음보다 못하다. 이 설정으로는 우위를 주장할 수 없다")
            )]
        if "C-E" in self.srs_controls:
            ce = self.srs_controls["C-E"]
            lines.append(
                f"구조 기여(C-E 셔플 대비): {ce - self.srs_protected:+.3f} "
                + ("— 효과의 원인이 크기가 아니라 구조임을 지지" if self.srs_protected < ce
                   else "— 구조 기여가 확인되지 않음")
            )
        lines.append(
            f"※ 판정 임계값 {self.PROVISIONAL_THRESHOLD}은 C-D(타 화자) 대조군에서 "
            "측정한 값이다 (EER · 동일오류율 4.3%).\n"
            "  다만 대상이 XTTS 합성 화자이므로 사람 목소리로 재측정할 것 "
            "— python calibrate_dsr.py out/speakers"
        )
        return "\n".join(lines)


def _evaluate(result, x, protected, delta, encoders, cfg, with_controls) -> None:
    # ── 평가 기준을 최적화 목표와 맞춘다 ──────────────────────────────────────
    #
    # channel_aware 섭동은 **깨끗한 파일에서 거의 무효**다. 그게 정상이다 —
    # 표적을 "채널 통과본"으로 바꿨기 때문이다. 그런데 평가를 깨끗한 파일로 하면
    # 리포트가 "잡음보다 못하다"고 말한다. 실측 예:
    #
    #   채널 인지 보호본   깨끗한 파일 0.9429  ←리포트가 실패로 읽던 값
    #                     채널 통과 후 0.5915  ←실제 위협 모델에서의 성능
    #
    # 최적화 목표와 보고 기준이 다르면 도구가 자기 결과를 부정한다.
    # 여기서는 **진짜 코덱**으로 평가한다. 최적화에 쓴 미분 가능 근사가 아니라.
    if cfg.channel_aware:
        from .codec import CHANNELS, telephone_channel
        chan = CHANNELS["ulaw"]
        heard = lambda w: telephone_channel(w, chan)   # noqa: E731
        result.srs_basis = f"통화 채널 통과 후 ({chan.describe()})"
    else:
        heard = lambda w: w                            # noqa: E731
        result.srs_basis = "파일 대 파일 (채널 없음)"

    primary = encoders.encoders[0]
    with torch.no_grad():
        ref = primary(heard(x))
        result.srs_protected = float(cosine_similarity(primary(heard(protected)), ref))

        if with_controls:
            for name, wav in make_controls(x, delta, cfg.target_snr_db).items():
                result.srs_controls[name] = float(
                    cosine_similarity(primary(heard(wav)), ref))


def _protect_whole(
    x: torch.Tensor,
    encoders: EncoderEnsemble,
    cfg: PGDConfig,
    model: MaskingModel,
    sample_rate: int,
    with_controls: bool,
    progress: bool,
    on_step=None,
) -> ProtectionResult:
    """발화 전체를 한 번에 최적화한다 — 기본 경로.

    Resemblyzer는 발화를 1.6초 조각으로 잘라 임베딩을 **평균**낸다.
    전체를 한 번에 최적화하면 PGD가 그 평균 연산까지 통과해 미분되므로,
    "평균을 가장 크게 움직이는" 섭동을 찾는다.
    청크별 최적화는 각 조각을 제 나름대로 밀 뿐이라 평균이 잘 안 움직인다.
    """
    import time

    t_total = time.perf_counter()
    if progress:
        print(f"  전체 발화 {x.shape[-1] / sample_rate:.1f}초 · "
              f"{cfg.steps}스텝 최적화 중...", flush=True)

    r = pgd_perturbation(x, encoders, cfg, masking_model=model,
                         seed=0, on_step=on_step)
    delta = r.delta
    protected = r.protected
    thr, _ = model.threshold(x)

    result = ProtectionResult(
        original=x,
        protected=protected,
        delta=delta,
        chunks=[ChunkRecord(index=0, start_sec=0.0,
                            end_sec=x.shape[-1] / sample_rate,
                            srs=r.srs, snr_db=r.snr_db,
                            elapsed_sec=time.perf_counter() - t_total)],
        global_snr_db=snr_db(x, delta),
        audibility=audibility(delta, thr, cfg.masking_ratio, model),
        audibility_abs=audibility(delta, thr, 1.0, model),
        out_of_band_db=band_energy_ratio_db(delta, cfg.band_low_hz,
                                            cfg.band_high_hz, sample_rate),
        total_sec=time.perf_counter() - t_total,
    )
    _evaluate(result, x, protected, delta, encoders, cfg, with_controls)
    return result


def protect_utterance(
    x: torch.Tensor,
    encoders: EncoderEnsemble | SpeakerEncoder,
    cfg: PGDConfig | None = None,
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: float = CHUNK_SEC,
    hop_sec: float = HOP_SEC,
    with_controls: bool = True,
    progress: bool = True,
    chunked: bool = False,
) -> ProtectionResult:
    """발화 하나를 보호한다.

    ## 청크 분할을 기본에서 뺐다 — 실측 근거

    원래는 2초 청크로 쪼개 각각 최적화하고 Hann overlap-add로 합쳤다.
    명분은 "모든 2초 창이 각각 보호되므로 발췌 공격에 강해진다"였다.

    **그런데 그 방식으로는 전체 파일이 보호되지 않는다.**
    청크마다 **자기 자신의** 임베딩에서 멀어지므로 미는 방향이 제각각이고,
    복제 모델이 파일 전체를 볼 때(임베딩을 평균낼 때) 서로 상쇄된다.

    실사용 녹음 14초로 잰 값 (60스텝 · SNR 20 dB · 같은 기준):

        방식          파일 기준   통화채널 통과    소요
        청크 분할      0.9427     0.8045        763초   ← 임계값 0.7962 위. 실패
        전체 발화      0.8097     0.3709         56초   ← 크게 아래. 성공

    **더 강하고 13배 빠르다.** PGD를 청크 수만큼이 아니라 한 번만 돌기 때문이다.

    발췌 공격 대비도 잃지 않았다. 전체 발화로 최적화한 뒤 잘라 봐도
    2초 발췌 0.8213 · 4초 0.7900 · 8초 0.8014로 전체(0.8097)와 차이가 없다.
    청크 분할의 유일한 명분이 실측에서 성립하지 않았다.

    `chunked=True`로 옛 동작을 부를 수 있다 — 비교 실험용으로만 남긴다.
    """
    import time

    cfg = cfg or PGDConfig()
    if isinstance(encoders, SpeakerEncoder):
        encoders = EncoderEnsemble([encoders])

    device = x.device
    model = MaskingModel(sample_rate, cfg.n_fft, cfg.hop_length, device=device)

    if not chunked:
        return _protect_whole(x, encoders, cfg, model, sample_rate,
                              with_controls, progress)

    pieces, starts, chunk_len = split(x, sample_rate, chunk_sec, hop_sec)
    deltas: list[torch.Tensor] = []
    records: list[ChunkRecord] = []
    t_total = time.perf_counter()

    for i, (piece, s) in enumerate(zip(pieces, starts)):
        t0 = time.perf_counter()
        # 청크 인덱스를 시드로 넘겨 재현 가능하면서도 청크마다 다른 초기화를 쓴다
        r = pgd_perturbation(piece, encoders, cfg, masking_model=model, seed=i)
        dt = time.perf_counter() - t0

        deltas.append(r.delta)
        records.append(ChunkRecord(
            index=i,
            start_sec=s / sample_rate,
            end_sec=(s + chunk_len) / sample_rate,
            srs=r.srs,
            snr_db=r.snr_db,
            elapsed_sec=dt,
        ))
        if progress:
            print(f"  청크 {i + 1}/{len(pieces)} · SRS {r.srs:.4f} · "
                  f"SNR {r.snr_db:.1f} dB · {dt:.1f}초", flush=True)

    delta = overlap_add(deltas, starts, x.shape[-1], chunk_len)
    protected = x + delta

    # 청크마다 정규화했으므로 전체 SNR은 최종 파형에서 다시 잰다
    thr, _ = model.threshold(x)
    result = ProtectionResult(
        original=x,
        protected=protected,
        delta=delta,
        chunks=records,
        global_snr_db=snr_db(x, delta),
        audibility=audibility(delta, thr, cfg.masking_ratio, model),
        # 배율과 무관한 절대 기준. 배율 간 비교와 "실제로 들리는가"는 이쪽으로만 답한다.
        audibility_abs=audibility(delta, thr, 1.0, model),
        out_of_band_db=band_energy_ratio_db(delta, cfg.band_low_hz,
                                            cfg.band_high_hz, sample_rate),
        srs_protected=0.0,
        total_sec=time.perf_counter() - t_total,
    )

    _evaluate(result, x, protected, delta, encoders, cfg, with_controls)
    return result
