"""전역 파라미터 — D09 §05 확정값 표를 그대로 옮긴다.

여기 있는 값을 바꾸면 설계도와 어긋난다. 바꿀 때는 D09도 함께 고친다.
"""

from dataclasses import dataclass

import torch

# ── 오디오 ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000

# ── 청크 파이프라인 (D09 §04) ─────────────────────────────────────────────────
CHUNK_SEC = 2.0          # Resemblyzer partial utterance 1.6초 → 온전한 임베딩 1.25개
HOP_SEC = 1.0            # 50% 겹침. Hann overlap-add로 경계 클릭음 방지

# ── STFT ──────────────────────────────────────────────────────────────────────
# hop = n_fft/4 로 두어 Hann 창의 COLA 조건을 만족시킨다.
# 위반하면 ISTFT 왕복에서 진폭이 틀어지고 test_stft_roundtrip이 잡는다.
N_FFT = 512
HOP_LENGTH = 128

# ── PGD (D09 §05) ─────────────────────────────────────────────────────────────
PGD_STEPS = 200          # 서버 여유가 커서 표준 100의 2배
#
# alpha는 sign(gradient) 스텝 크기다. 매 스텝 정규화가 크기를 되돌리므로
# "한 스텝에 방향을 얼마나 바꾸는가"를 정한다.
# 절제 실험(합성 신호 · 50스텝)에서 1e-4는 전 배율 구간에서 1e-3보다 뚜렷하게 열등했다.
#   배율 0.75  SRS 0.987(1e-4) vs 0.906(1e-3)
#   배율 3.00  SRS 0.787       vs 0.462
#   배율 100   SRS 0.641       vs 0.351
# → 1e-3을 기본값으로 둔다. **실제 사람 목소리로 다시 스윕해 확정할 것.**
PGD_ALPHA = 1e-3

# ── 지각 제약 ─────────────────────────────────────────────────────────────────
TARGET_SNR_DB = 20.0     # 목표 SNR ≥ 20 dB. 정규화는 루프 안, 투영 앞
#
# 마스킹 배율은 D09에서 "청취 평가 후 결정"으로 열려 있는 유일한 파라미터다.
#
# 자체 절제 실험이 그 판단을 뒷받침한다(합성 신호 · 50스텝 · sweep_params.py).
#   배율 0.75 → SRS 0.906.  같은 SNR 대역제한잡음(0.733)보다 **나쁘다.**
#   배율 3.00 → SRS 0.462.  잡음 대비 뚜렷한 우위.
#   배율 100  → SRS 0.351.  사실상 마스킹 제약 없음.
# 즉 0.75에서는 "그냥 잡음 아니냐"에 답할 수 없다. 대조군을 이기려면 최소 3 근방이 필요하고,
# 그 지점은 섭동이 마스킹 임계값을 넘는다는 뜻이라 **가청도와 정면으로 맞바꾸는 구간**이다.
#
# 잠정값을 3.0으로 둔다. 확정은 두 단계를 거친다.
#   ① 사람 목소리로 배율 스윕 재측정 (합성 신호 결과는 방향만 보여줄 뿐이다)
#   ② 청취 평가로 가청도 상한을 먼저 정하고 거기서 역산 (D07 W1 · 4h)
MASKING_RATIO = 3.0

BAND_LOW_HZ = 300.0      # 통화 대역. 밖은 코덱에서 어차피 제거된다
BAND_HIGH_HZ = 3400.0
BAND_TAPER_HZ = 50.0     # 브릭월 링잉 방지용 전이대역

# ── VAD ───────────────────────────────────────────────────────────────────────
VAD_FRAME_MS = 10.0      # 에너지 기반 10 ms
VAD_FADE_MS = 8.0        # 경계 5~10 ms 페이드 — delta*mask 계단 끊김이 클릭음이 된다
VAD_REL_THRESH_DB = 30.0 # 프레임 최대 에너지 대비 -30 dB 아래를 무음으로 본다

# ── 수치 안정 ─────────────────────────────────────────────────────────────────
EPS = 1e-12              # 투영식 0-나눗셈 방지. 이 값이 없으면 무음 구간에서 NaN이 난다


def default_device() -> torch.device:
    """실구현은 RTX 3060, 개발 PC는 CPU. 어느 쪽이든 같은 코드가 돈다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PGDConfig:
    """PGD 1회 실행에 필요한 값 묶음. 실험 시 이 객체만 바꿔 스윕한다."""

    steps: int = PGD_STEPS
    alpha: float = PGD_ALPHA
    target_snr_db: float = TARGET_SNR_DB
    masking_ratio: float = MASKING_RATIO
    band_low_hz: float = BAND_LOW_HZ
    band_high_hz: float = BAND_HIGH_HZ
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH

    # STFT 투영만으로는 마스킹 불변식이 완전히 서지 않는다(perturbation.enforce_masking_bound 참조).
    # True면 마지막에 전역 축소로 불변식을 강제한다 — 들리지 않음이 보장되는 대신 방어가 약해진다.
    # False면 강도를 유지하고 초과량을 metrics.audibility로 정직하게 보고한다.
    # 어느 쪽을 쓸지는 청취 평가로 정한다.
    enforce_masking: bool = False
