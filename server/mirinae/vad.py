"""발화 구간 검출 — 에너지 기반 10 ms + 페이드.

무음 구간에 섭동을 넣으면 배경이 조용한 만큼 그대로 들린다. 그래서 마스크로 잘라낸다.
그런데 `delta * mask`를 그냥 곱하면 경계에서 파형이 계단식으로 끊겨 **클릭음**이 생긴다.
5~10 ms 페이드가 그 문제를 없앤다. (D09 §4.3)
"""

from __future__ import annotations

import torch

from .config import EPS, SAMPLE_RATE, VAD_FADE_MS, VAD_FRAME_MS, VAD_REL_THRESH_DB


def speech_mask(
    x: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: float = VAD_FRAME_MS,
    fade_ms: float = VAD_FADE_MS,
    rel_thresh_db: float = VAD_REL_THRESH_DB,
) -> torch.Tensor:
    """샘플 단위 발화 마스크를 만든다. 값은 0~1이고 경계는 매끄럽다.

    :param rel_thresh_db: 최대 프레임 에너지 대비 몇 dB 아래를 무음으로 볼지.
    :return: x와 같은 길이의 float 텐서.
    """
    n = x.shape[-1]
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = (n + frame - 1) // frame

    pad = n_frames * frame - n
    padded = torch.nn.functional.pad(x, (0, pad))
    frames = padded.reshape(n_frames, frame)

    energy_db = 10.0 * torch.log10(torch.clamp((frames ** 2).mean(dim=1), min=EPS))
    gate = (energy_db >= energy_db.max() - rel_thresh_db).to(x.dtype)

    # 프레임 게이트를 샘플로 펼친다
    mask = gate.repeat_interleave(frame)[:n]

    # 경계 페이드 — 이동평균 한 번이면 계단이 사라진다.
    fade = max(1, int(sample_rate * fade_ms / 1000.0))
    if fade > 1:
        kernel = torch.ones(1, 1, fade, dtype=mask.dtype, device=mask.device) / fade
        smoothed = torch.nn.functional.conv1d(
            torch.nn.functional.pad(mask[None, None, :], (fade // 2, fade - fade // 2 - 1),
                                    mode="replicate"),
            kernel,
        )
        mask = smoothed[0, 0]
    return torch.clamp(mask, 0.0, 1.0)


def speech_ratio(mask: torch.Tensor) -> float:
    """발화로 판정된 비율. 리포트에 싣는다."""
    return float((mask > 0.5).to(torch.float32).mean())
