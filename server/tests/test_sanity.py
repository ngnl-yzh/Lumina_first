"""단위 테스트 — D09 §6.2 표를 그대로 구현한다.

계획서가 "구현 첫날 작성"이라고 못박은 이유가 있다. 여기서 잡는 결함은 전부
**에러 없이 조용히 틀리는** 종류라, 나중에 쓰면 이미 잘못된 수치로 실험을 끝낸 뒤가 된다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirinae.chunking import overlap_add, split                      # noqa: E402
from mirinae.config import EPS, PGDConfig, SAMPLE_RATE               # noqa: E402
from mirinae.metrics import (                                        # noqa: E402
    audibility, band_energy_ratio_db, wilson_ci,
)
from mirinae.perturbation import (                                   # noqa: E402
    band_limit, enforce_masking_bound, normalize_snr, project_masking, snr_db,
)
from mirinae.psychoacoustic import MaskingModel                      # noqa: E402
from mirinae.vad import speech_mask                                  # noqa: E402


# ── 픽스처 ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> MaskingModel:
    return MaskingModel()


def synth_speech(n_sec: float = 2.0, silence: bool = True) -> torch.Tensor:
    """음성 비슷한 테스트 신호. 무음 구간을 일부러 포함시킨다 — 0-나눗셈이 거기서 난다."""
    n = int(SAMPLE_RATE * n_sec)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE

    # 기본주파수 + 배음. 포먼트 흉내로 진폭을 다르게 준다.
    sig = torch.zeros(n)
    for k, amp in enumerate([1.0, 0.6, 0.35, 0.2, 0.1], start=1):
        sig = sig + amp * torch.sin(2 * math.pi * 130.0 * k * t)
    sig = sig * (0.5 + 0.5 * torch.sin(2 * math.pi * 3.0 * t))       # 음절 포락선
    sig = sig / sig.abs().max() * 0.5

    if silence:
        sig[: n // 8] = 0.0                                           # 완전 무음 구간
        sig[-n // 8:] = 0.0
    return sig


# ── 1. 마스킹 불변식 ──────────────────────────────────────────────────────────

def test_masking_invariant_enforced(model):
    """abs(stft(δ)) ≤ thr×ratio 가 전 bin에서 성립해야 한다 — 강제 모드.

    잡는 결함 — 정규화가 제약을 깨는 문제.
    정규화를 루프 밖에 두면 실측 94.9%의 샘플이 이 단언을 깬다.

    설계도(D09 §6.2)의 단언은 이 형태다. 다만 STFT 투영만으로는 성립하지 않으므로
    `enforce_masking_bound`(전역 축소)를 거쳐야 한다. 아래 테스트가 그 차이를 계측한다.
    """
    x = synth_speech()
    thr, _ = model.threshold(x)
    ratio = 0.75

    delta = torch.randn_like(x) * 0.05
    delta = project_masking(delta, thr, ratio, model)
    delta = enforce_masking_bound(delta, thr, ratio, model)

    mag = model.stft(delta).abs()
    bound = thr * ratio
    tol = 1e-5 * float(bound.max())
    assert bool((mag <= bound + tol).all()), (
        f"최대 초과 {float((mag - bound).max()):.3e} (허용 {tol:.3e})"
    )


def test_audibility_is_relative_to_the_ratio_you_pass(model):
    """`audibility`의 기준선은 `thr × ratio`다 — **배율마다 자기 기준이 달라진다.**

    이걸 모르고 배율별 초과량을 나란히 놓으면 결론이 정확히 반대가 된다.
    배율을 키우면 기준선도 함께 올라가서 초과량이 **줄어들기** 때문이다.
    실측(스윕): 배율 0.75에서 제약 기준 16.1 dB, 배율 10.0에서 12.4 dB.
    그대로 읽으면 "배율을 올릴수록 덜 들린다"가 되는데 절대 기준으로는 정반대다
    (위반 bin 0.74% → 25.4%).

    D09가 마스킹 배율을 "청취 평가로 결정"하도록 열어둔 파라미터인데,
    그 결정을 뒷받침해야 할 지표가 배율 간 비교에 쓸 수 없는 값이었다.
    이 테스트가 그 성질을 명시적으로 못박는다.
    """
    x = synth_speech()
    thr, _ = model.threshold(x)
    delta = torch.randn_like(x) * 0.01

    loose = audibility(delta, thr, 10.0, model)    # 느슨한 제약
    strict = audibility(delta, thr, 1.0, model)    # 절대 기준

    # 같은 δ인데 기준선이 다르므로 초과량이 다르게 나온다.
    assert loose.max_excess_db < strict.max_excess_db, (
        "배율을 키웠는데 초과량이 줄지 않았다 — 기준선이 배율을 따라가지 않는다"
    )
    assert loose.violation_ratio <= strict.violation_ratio

    # 차이는 정확히 20·log10(ratio)다. 우연이 아니라 정의에서 나온다.
    assert strict.max_excess_db - loose.max_excess_db == pytest.approx(
        20.0 * math.log10(10.0), abs=1e-3
    )


def test_telephone_channel_preserves_length_and_band():
    """통화 채널이 길이를 보존하고 대역 밖을 실제로 잘라내는가.

    길이가 어긋나면 이후 SRS 비교가 **채널 효과가 아니라 정렬 어긋남**을 잰다.
    조용히 틀리는 종류라 여기서 못박는다.
    """
    from mirinae.codec import CHANNELS, telephone_channel

    x = synth_speech()
    for key, cfg in CHANNELS.items():
        y = telephone_channel(x, cfg)
        assert y.shape == x.shape, f"{key}: 길이가 바뀌었다 {y.shape} != {x.shape}"
        assert torch.isfinite(y).all(), f"{key}: NaN/Inf가 생겼다"

    # 절대 수준이 아니라 **감쇠량**을 본다.
    # 이 테스트 신호는 130 Hz 기본주파수라 가장 강한 성분(130·260 Hz)이 통과대역 아래에 있다.
    # 필터가 정상이어도 남은 대역 밖 비율이 -20 dB 근방에 머문다 —
    # 절대 임계값으로 단언하면 신호 탓에 실패하고, 그 실패는 필터에 대해 아무것도 말해주지 않는다.
    before = band_energy_ratio_db(x, 300.0, 3400.0, SAMPLE_RATE)
    after = band_energy_ratio_db(telephone_channel(x, CHANNELS["ulaw"]),
                                 300.0, 3400.0, SAMPLE_RATE)
    assert after < before - 15.0, (
        f"대역 밖 에너지가 {before:.1f} → {after:.1f} dB로 거의 안 줄었다 — 필터가 안 걸렸다"
    )


def test_channel_aware_targets_the_channel_passed_signal():
    """channel_aware는 **채널 통과본**을 기준으로 최적화해야 한다.

    잡는 결함 — 전대역 원본을 표적으로 삼으면 통화 경로에서 방어가 무너진다.
    실측: 무처리 SRS 0.6342가 G.711 협대역 통과 후 0.8346으로 되돌아왔다
    (판정 임계값 0.7962 위). 섭동은 남아 있었다(잔존 101% · 구조 상관 0.989) —
    지워진 게 아니라 표적이 틀렸던 것이다.

    여기서는 두 설정이 **서로 다른 δ를 만드는지**만 확인한다.
    실제 개선 여부는 `channel_ab.py`가 진짜 코덱으로 측정한다.
    """
    from mirinae.encoder import SpeakerEncoder
    from mirinae.perturbation import pgd_perturbation

    x = synth_speech()
    enc = SpeakerEncoder()
    off = pgd_perturbation(x, enc, PGDConfig(steps=3, channel_aware=False), seed=0)
    on = pgd_perturbation(x, enc, PGDConfig(steps=3, channel_aware=True), seed=0)

    # 같은 시드인데 목적함수가 다르므로 δ가 달라야 한다.
    diff = float((off.delta - on.delta).abs().max())
    assert diff > 1e-6, "channel_aware가 최적화 목표를 바꾸지 않았다"


def test_pipeline_reports_both_audibility_scales():
    """파이프라인 결과가 두 기준을 모두 들고 있어야 한다.

    절대 기준만 "들리는가"에 답한다. 한동안 제약 기준만 보고했고,
    배율 3.0 실측에서 위반 비율을 2.44%로 보고했는데 실제(절대 기준)는 20.30%였다.
    """
    from mirinae.pipeline import ProtectionResult

    names = ProtectionResult.__dataclass_fields__
    assert "audibility" in names
    assert "audibility_abs" in names, (
        "절대 기준 가청도가 없다 — 배율이 1보다 크면 가청도를 실제보다 좋게 보고한다"
    )


def test_masking_projection_reduces_violation(model):
    """투영이 실제로 초과분을 눌러넣는지 — 강제 없이 계측만 한다.

    STFT 투영 후 ISTFT를 거치면 스펙트로그램 비일관성 때문에 일부 bin이 다시 넘는다.
    그 잔여량은 **감춰서는 안 되는 수치**이므로 여기서 상한을 걸어 회귀를 막는다.
    (실측 기준: 투영 전 위반 37% → 투영 후 10% 미만)
    """
    x = synth_speech()
    thr, _ = model.threshold(x)
    ratio = 0.75
    bound = thr * ratio

    def violation(sig: torch.Tensor) -> float:
        return float((model.stft(sig).abs() > bound).to(torch.float32).mean())

    # (a) 최악 입력 — 전대역 백색잡음. 무음 구간에도 에너지가 있어 제약이 가장 빡빡하다.
    raw = torch.randn_like(x) * 0.05
    before = violation(raw)
    after = violation(project_masking(raw, thr, ratio, model))
    assert after < before, "투영이 위반을 줄이지 못했다"
    assert after < 0.30, f"최악 입력 투영 후 위반 bin {after * 100:.1f}% — 회귀 의심"

    # (b) 실제 루프 조건 — 대역 제한 + 발화 마스크까지 걸린 상태.
    #     이쪽이 운영 중 실제로 나오는 값이다. (실측 8% 수준)
    mask = speech_mask(x)
    real = band_limit(project_masking(raw, thr, ratio, model), 300.0, 3400.0) * mask
    real_violation = violation(real)
    assert real_violation < 0.12, (
        f"실제 조건 위반 bin {real_violation * 100:.1f}% — 회귀 의심"
    )


# ── 2. 투영 NaN ───────────────────────────────────────────────────────────────

def test_projection_no_nan(model):
    """무음을 포함한 입력에 200스텝을 돌려도 NaN이 없어야 한다.

    잡는 결함 — `D/abs(D)`의 0-나눗셈.
    NaN은 ISTFT에서 프레임 전체로 번지고, 예외를 던지지 않아 최적화가 조용히 멈춘다.
    """
    x = synth_speech(silence=True)
    thr, _ = model.threshold(x)

    delta = torch.zeros_like(x)
    for _ in range(200):
        delta = delta + torch.randn_like(x) * 1e-4
        delta = normalize_snr(delta, x, 20.0)
        delta = project_masking(delta, thr, 0.75, model)
        delta = band_limit(delta, 300.0, 3400.0)

    assert not bool(torch.isnan(delta).any()), "NaN 발생"
    assert not bool(torch.isinf(delta).any()), "Inf 발생"


def test_projection_no_nan_on_pure_silence(model):
    """완전 무음 입력 — 0-나눗셈이 가장 확실하게 터지는 경계 조건."""
    x = torch.zeros(int(SAMPLE_RATE * 1.0))
    thr, _ = model.threshold(x)
    out = project_masking(torch.zeros_like(x), thr, 0.75, model)
    assert not bool(torch.isnan(out).any())


# ── 3. SNR 목표 ───────────────────────────────────────────────────────────────

def test_snr_target():
    """|측정 − 목표| < 0.5 dB. 잡는 결함 — dB 스케일 혼용(10log10 vs 20log10)."""
    x = synth_speech()
    for target in (15.0, 20.0, 25.0, 30.0):
        delta = normalize_snr(torch.randn_like(x) * 0.01, x, target)
        assert abs(snr_db(x, delta) - target) < 0.5, f"목표 {target} dB 미달성"


def test_snr_never_below_target_after_projection(model):
    """투영은 크기를 깎기만 하므로 최종 SNR은 목표 이상(= 더 조용함)이어야 한다.

    이 성질 때문에 전역 SNR을 최종 파형에서 다시 재어 보고한다.
    """
    x = synth_speech()
    thr, _ = model.threshold(x)
    delta = normalize_snr(torch.randn_like(x) * 0.01, x, 20.0)
    delta = project_masking(delta, thr, 0.75, model)
    assert snr_db(x, delta) >= 20.0 - 0.5


# ── 4. 대역 제한 ──────────────────────────────────────────────────────────────

def test_band_limit():
    """대역 밖 에너지 < −60 dB. 잡는 결함 — 대역 제한 누락."""
    x = synth_speech()
    delta = band_limit(torch.randn_like(x) * 0.01, 300.0, 3400.0)
    ratio_db = band_energy_ratio_db(delta, 300.0, 3400.0, SAMPLE_RATE)
    assert ratio_db < -60.0, f"대역 밖 에너지 {ratio_db:.1f} dB"


# ── 5. VAD 페이드 ─────────────────────────────────────────────────────────────

def test_vad_fade():
    """마스크 경계의 1차 차분에 계단식 불연속이 없어야 한다. 잡는 결함 — 클릭 노이즈."""
    x = synth_speech(silence=True)
    mask = speech_mask(x)

    max_jump = float(mask.diff().abs().max())
    fade_samples = int(SAMPLE_RATE * 8.0 / 1000.0)

    # 페이드가 걸렸다면 한 샘플당 변화량은 대략 1/fade_samples 수준이어야 한다
    assert max_jump < 3.0 / fade_samples, f"경계 점프 {max_jump:.4f} — 페이드 미적용"
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


# ── 6. overlap-add ────────────────────────────────────────────────────────────

def test_overlap_add():
    """균일한 δ를 넣으면 합성 후에도 진폭이 보존되어야 한다.

    잡는 결함 — 청크 창 보정 오류. 특히 겹칠 상대가 없는 양 끝에서 진폭이 죽는다.
    """
    n = int(SAMPLE_RATE * 7.3)                 # 청크로 딱 안 떨어지는 길이를 일부러 쓴다
    x = torch.ones(n)
    pieces, starts, chunk = split(x)
    out = overlap_add(pieces, starts, n, chunk)

    err = float((out - 1.0).abs().max())
    assert err < 1e-5, f"진폭 오차 {err:.2e} — 양 끝 창 보정 확인"


def test_split_covers_tail():
    """꼬리 구간이 버려지지 않아야 한다."""
    n = int(SAMPLE_RATE * 7.3)
    _, starts, chunk = split(torch.zeros(n))
    assert starts[-1] + chunk >= n


# ── 7. STFT 왕복 ──────────────────────────────────────────────────────────────

def test_stft_roundtrip():
    """100회 왕복 상대오차 < 1e-10. 잡는 결함 — COLA 위반.

    float32로는 이 정밀도가 안 나오므로 float64로 잰다.
    목적이 '수치 정밀도 측정'이 아니라 **창/홉 조합이 COLA를 만족하는지** 확인하는 것이라
    이렇게 해야 결함이 드러난다. (PGD 루프 자체는 인코더 때문에 float32로 돈다.)
    """
    torch.manual_seed(0)
    n_fft, hop = 512, 128
    x = torch.randn(SAMPLE_RATE * 2, dtype=torch.float64)
    win = torch.hann_window(n_fft, dtype=torch.float64)

    y = x.clone()
    for _ in range(100):
        X = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win,
                       center=True, return_complex=True)
        y = torch.istft(X, n_fft=n_fft, hop_length=hop, window=win,
                        center=True, length=len(x))

    # 경계는 창 합이 1이 아니므로 제외하고 본다
    edge = n_fft
    rel = float((y[edge:-edge] - x[edge:-edge]).abs().max() /
                x[edge:-edge].abs().max())
    assert rel < 1e-10, f"상대오차 {rel:.2e} — COLA 조건 확인"


# ── 8. 신뢰구간 ───────────────────────────────────────────────────────────────

def test_wilson_ci_matches_design_doc():
    """설계도에 적힌 두 값을 그대로 재현하는지 확인한다.

    n=30에서 0/30 → [0.0, 11.4] · 6/6 → [61, 100]
    """
    lo, hi = wilson_ci(0, 30)
    assert lo == 0.0
    assert abs(hi * 100 - 11.4) < 0.3, f"상한 {hi * 100:.1f}%"

    lo, hi = wilson_ci(6, 6)
    assert abs(lo * 100 - 61.0) < 1.0, f"하한 {lo * 100:.1f}%"
    assert hi == pytest.approx(1.0)


def test_wilson_ci_narrows_with_n():
    """표본이 늘면 구간이 좁아져야 한다 — n을 병기하는 이유."""
    w6 = wilson_ci(6, 6)
    w30 = wilson_ci(30, 30)
    assert (w30[1] - w30[0]) < (w6[1] - w6[0])
