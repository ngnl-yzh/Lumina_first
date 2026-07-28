"""대조군 생성 — 대조군 없이는 아무것도 주장할 수 없다.

자체 검증에서 확인된 사실: MFCC 통계 대리 인코더 기준으로 **SNR 20 dB 백색잡음만 넣어도
화자 유사도가 0.138까지** 떨어졌다. 서로 다른 화자 간 기준선이 0.80이므로
잡음만으로도 "다른 사람" 판정이 난다. 대조군 없이 DSR을 보고하면
"그냥 잡음 아니냐"에 답할 수 없고, 모든 수치가 무의미해진다.

각 대조군은 정확히 하나의 반박을 막는다.
"""

from __future__ import annotations

import torch

from .config import BAND_HIGH_HZ, BAND_LOW_HZ, EPS, SAMPLE_RATE
from .perturbation import band_limit

# C-E 블록 셔플 길이.
# 샘플 단위로 섞으면 스펙트럼까지 백색으로 변해 C-A와 구별이 없어진다.
# 블록으로 섞어야 크기·스펙트럼은 유지되고 **신호와의 정렬(구조)만** 파괴된다.
SHUFFLE_BLOCK_MS = 30.0

CONTROL_DESCRIPTIONS = {
    "C-A": "백색잡음 · 동일 SNR · 전대역 — “그냥 잡음 아닌가”",
    "C-B": "대역제한 잡음 · 동일 SNR · 300~3400 Hz — “대역 제한이 효과의 전부 아닌가”",
    "C-C": "무섭동 원본 · δ=0 — 인코더 정상 작동 확인 (SRS ≈ 1.0)",
    "C-D": "타 화자 음성 — “다른 사람” 판정 기준선. DSR 임계값의 근거",
    "C-E": "셔플 섭동 · 최적화된 δ를 시간축 블록 셔플 — “구조가 원인인가 크기가 원인인가”",
}


def _scale_to_snr(noise: torch.Tensor, x: torch.Tensor, target_db: float) -> torch.Tensor:
    x_norm = torch.sqrt(torch.clamp((x ** 2).sum(), min=EPS))
    n_norm = torch.sqrt(torch.clamp((noise ** 2).sum(), min=EPS))
    return noise * (x_norm / (10.0 ** (target_db / 20.0)) / n_norm)


def white_noise(x: torch.Tensor, target_db: float,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """C-A. 동일 SNR 전대역 백색잡음."""
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return _scale_to_snr(noise, x, target_db)


def bandlimited_noise(x: torch.Tensor, target_db: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """C-B. 섭동과 같은 대역으로 제한한 잡음. 대역 제한만의 효과를 분리한다."""
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    noise = band_limit(noise, BAND_LOW_HZ, BAND_HIGH_HZ)
    return _scale_to_snr(noise, x, target_db)


def shuffle_perturbation(
    delta: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    block_ms: float = SHUFFLE_BLOCK_MS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """C-E. 최적화된 δ의 블록 순서만 뒤섞는다.

    발표에서 가장 강한 카드다. 크기와 스펙트럼은 거의 그대로인데 구조만 파괴되므로,
    이때 DSR이 크게 떨어지면 **적대적 최적화가 만든 구조가 효과의 원인**임이 증명된다.
    비용이 거의 0이면서 대조군 질문 전체를 막는다.
    """
    n = delta.shape[-1]
    block = max(1, int(sample_rate * block_ms / 1000.0))
    n_blocks = n // block
    if n_blocks < 2:
        return delta.clone()

    head = delta[: n_blocks * block].reshape(n_blocks, block)
    perm = torch.randperm(n_blocks, device=delta.device, generator=generator)
    shuffled = head[perm].reshape(-1)
    return torch.cat([shuffled, delta[n_blocks * block:]])


def make_controls(
    x: torch.Tensor,
    delta: torch.Tensor,
    target_snr_db: float,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """같은 발화에 대해 대조군을 한 번에 생성한다.

    이 함수가 있기 때문에 팀원이 30분 말하면 **대조군까지 갖춰진 n=30 데이터셋**이
    그대로 쌓인다. "표본이 적다"와 "대조군이 없다"는 두 약점이 동시에 해결된다.

    C-D(타 화자)는 다른 사람의 음성이 필요하므로 여기서 만들지 않는다.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    def cpu_gen(fn):
        # 재현성을 위해 난수는 CPU에서 뽑고 옮긴다 (CUDA generator는 장비마다 다르다)
        out = fn(x.cpu(), target_snr_db, g)
        return out.to(x.device)

    return {
        "C-A": x + cpu_gen(white_noise),
        "C-B": x + cpu_gen(bandlimited_noise),
        "C-C": x.clone(),
        "C-E": x + shuffle_perturbation(delta.cpu(), generator=g).to(x.device),
    }
