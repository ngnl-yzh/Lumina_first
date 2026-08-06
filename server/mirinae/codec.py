"""통화 채널 시뮬레이션 — 섭동이 전화를 통과하는가.

## 왜 이게 가장 중요한 검증인가

미리내는 **전화 통화용**이다. 그런데 지금까지의 모든 SRS 수치는
**파일 대 파일**로 잰 값이다. 실제 경로에는 통화 채널이 끼어 있다.

    보호된 음성 → [8 kHz 다운샘플 · G.711 양자화 · 대역 제한] → 상대방 → 녹음 → 복제

`server/README.md`가 이 위험을 알고 있었다 — "협대역 8 kHz에서 최대 33%p 손실 예상".
**예상만 적혀 있고 측정된 적이 없다.** 여기가 무너지면 모드 2는 실사용에서 무의미하다.

이유가 구조적이다. 모드 2는 심리음향 마스킹 **아래**로 섭동을 숨긴다.
그런데 음성 코덱은 정확히 "안 들리는 성분"을 버리도록 설계되어 있다 —
**숨긴 원리와 버리는 원리가 같다.** 그래서 코덱이 섭동을 지울 것이라고 예상할 이유가 충분하다.
D08·D09의 `MediaRecorder` 명세를 AudioWorklet raw PCM으로 바꾼 것도 같은 이유였다.

## 무엇을 구현하는가

PSTN/VoIP 경로에서 실제로 일어나는 것만 넣는다. 추측으로 왜곡을 추가하지 않는다.

| 단계 | 근거 |
|---|---|
| 대역 제한 300~3400 Hz | ITU-T G.712 전화 대역 |
| 8 kHz 리샘플 왕복 | 협대역 통화의 표준 표본화 |
| G.711 μ-law 8비트 양자화 | 북미·한국 PSTN 표준 코덱 |
| (선택) A-law | 유럽 표준. 비교용 |

G.711을 쓰는 이유는 **가장 관대한 조건**이기 때문이다. AMR·Opus 같은 저비트레이트
코덱은 심리음향 모델을 적극적으로 써서 훨씬 많이 버린다.
**G.711에서 이미 섭동이 사라진다면 다른 코덱에서는 볼 것도 없다.**
반대로 G.711을 통과해도 AMR을 통과한다는 보장은 없다 — 낙관적 상한으로 읽어야 한다.
"""

from __future__ import annotations

import array
import audioop
from dataclasses import dataclass

import numpy as np
import torch
from scipy import signal as sps

from .config import BAND_HIGH_HZ, BAND_LOW_HZ, EPS, SAMPLE_RATE

TELEPHONE_RATE = 8_000


@dataclass(frozen=True)
class ChannelConfig:
    """통화 채널 설정. 하나씩 꺼가며 어느 단계가 섭동을 지우는지 분리할 수 있다."""

    band_limit: bool = True
    resample_8k: bool = True
    codec: str = "ulaw"          # "ulaw" | "alaw" | "none"
    name: str = "G.711 μ-law 협대역"

    def describe(self) -> str:
        parts = []
        if self.band_limit:
            parts.append(f"{BAND_LOW_HZ:.0f}~{BAND_HIGH_HZ:.0f} Hz")
        if self.resample_8k:
            parts.append("8 kHz")
        if self.codec != "none":
            parts.append(self.codec)
        return " · ".join(parts) or "무처리"


def _to_int16(x: np.ndarray) -> tuple[bytes, float]:
    """float → int16 PCM. 클리핑을 피하려고 스케일을 기억해 뒀다가 되돌린다.

    스케일을 고정값으로 두면 조용한 신호가 양자화 잡음에 묻혀
    **코덱 탓이 아닌 손실**이 섞인다.
    """
    peak = float(np.abs(x).max())
    scale = (0.99 / peak) if peak > EPS else 1.0
    pcm = np.clip(x * scale, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes(), scale


def _from_int16(raw: bytes, scale: float) -> np.ndarray:
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    return arr / scale if scale != 0 else arr


def bandpass(x: np.ndarray, sample_rate: int,
             low: float = BAND_LOW_HZ, high: float = BAND_HIGH_HZ) -> np.ndarray:
    """전화 대역 통과. 8차 Butterworth를 zero-phase로 건다.

    `perturbation.band_limit`과 다른 구현을 일부러 쓴다 —
    같은 FFT 마스크를 재사용하면 "우리가 만든 대역 제한을 우리가 다시 걸었다"가 되어
    실제 통신망 필터와의 차이를 못 본다.
    """
    nyq = sample_rate / 2.0
    lo = max(low / nyq, 1e-6)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return x
    sos = sps.butter(8, [lo, hi], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def telephone_channel(x: torch.Tensor, cfg: ChannelConfig | None = None,
                      sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """통화 채널을 왕복시킨다. 입력과 같은 길이·표본화율로 돌려준다."""
    cfg = cfg or ChannelConfig()
    device, dtype = x.device, x.dtype
    y = x.detach().cpu().numpy().astype(np.float32)
    n_in = len(y)

    if cfg.band_limit:
        y = bandpass(y, sample_rate)

    if cfg.resample_8k:
        n_8k = int(round(n_in * TELEPHONE_RATE / sample_rate))
        y = sps.resample_poly(y, TELEPHONE_RATE, sample_rate).astype(np.float32)
        y = y[:n_8k]

    if cfg.codec != "none":
        raw, scale = _to_int16(y)
        if cfg.codec == "ulaw":
            raw = audioop.ulaw2lin(audioop.lin2ulaw(raw, 2), 2)
        elif cfg.codec == "alaw":
            raw = audioop.alaw2lin(audioop.lin2alaw(raw, 2), 2)
        else:
            raise ValueError(f"모르는 코덱: {cfg.codec}")
        y = _from_int16(raw, scale)

    if cfg.resample_8k:
        y = sps.resample_poly(y, sample_rate, TELEPHONE_RATE).astype(np.float32)

    # 리샘플 왕복에서 길이가 한두 샘플 어긋난다. 맞춰 두지 않으면
    # 이후 SRS 비교가 길이 차이 때문에 흔들린다.
    if len(y) < n_in:
        y = np.pad(y, (0, n_in - len(y)))
    y = y[:n_in]
    return torch.as_tensor(y, dtype=dtype, device=device)


# 비교용 프리셋. 어느 단계가 섭동을 지우는지 분리해서 본다.
CHANNELS: dict[str, ChannelConfig] = {
    "none": ChannelConfig(False, False, "none", "무처리 (기준선)"),
    "band": ChannelConfig(True, False, "none", "대역 제한만"),
    "8k": ChannelConfig(True, True, "none", "대역 제한 + 8 kHz"),
    "ulaw": ChannelConfig(True, True, "ulaw", "G.711 μ-law 협대역 (전체)"),
    "alaw": ChannelConfig(True, True, "alaw", "G.711 A-law 협대역 (전체)"),
}


def surviving_ratio(delta: torch.Tensor, delta_after: torch.Tensor) -> float:
    """채널 통과 후 섭동이 얼마나 남았는가 (에너지 비).

    1.0이면 그대로, 0에 가까우면 지워졌다는 뜻이다.
    SRS만 보면 "방어가 약해졌다"까지만 알 수 있고
    **섭동 자체가 사라진 것인지 방향이 틀어진 것인지** 구별할 수 없다.
    """
    p_before = float((delta ** 2).sum())
    p_after = float((delta_after ** 2).sum())
    return p_after / max(p_before, EPS)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """두 파형의 정규화 상관. 섭동의 **구조**가 살아남았는지 본다.

    에너지가 남아 있어도 구조가 깨지면 적대적 효과는 사라진다 —
    C-E(셔플) 대조군이 보여준 그대로다.
    """
    av = a - a.mean()
    bv = b - b.mean()
    denom = float(torch.sqrt((av ** 2).sum() * (bv ** 2).sum()))
    return float((av * bv).sum()) / max(denom, EPS)
