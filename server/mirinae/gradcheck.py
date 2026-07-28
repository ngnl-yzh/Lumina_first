"""gradient 검증 — 수치미분 대조.

모드 2 전체가 "인코더에서 파형까지 gradient가 닿는다"는 전제 위에 서 있다.
그 전제가 깨지면 PGD는 아무 방향으로나 걸어다니고, 섭동은 그냥 잡음이 된다.
그런데 **에러는 나지 않는다.** 그래서 계측한다.

mel 추출을 torch로 다시 구현했으므로 두 가지를 확인해야 한다.
  ① 우리 mel이 librosa mel과 같은 값인가 (같은 인코더를 공략하고 있는가)
  ② 해석적 gradient가 수치미분과 일치하는가 (방향이 맞는가)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .encoder import SpeakerEncoder, cosine_similarity


@dataclass
class GradCheckReport:
    n_probes: int
    rel_error: float          # 목표 < 1e-3
    correlation: float        # 목표 > 0.999
    analytic_norm: float
    numeric_norm: float

    def ok(self, rel_tol: float = 1e-3, corr_tol: float = 0.999) -> bool:
        return self.rel_error < rel_tol and self.correlation > corr_tol

    def __str__(self) -> str:
        return (
            f"probe {self.n_probes}개 · 상대오차 {self.rel_error:.2e} · "
            f"상관 {self.correlation:.6f}"
        )


def check_gradient(
    encoder: SpeakerEncoder,
    x: torch.Tensor,
    n_probes: int = 48,
    h: float = 1e-5,
    seed: int = 0,
    probe_snr_db: float = 30.0,
    double_precision: bool = True,
) -> GradCheckReport:
    """손실의 해석적 gradient를 중심차분과 대조한다.

    전 샘플을 다 보면 32,000번 forward가 필요하므로 무작위 표본만 찍는다.
    표본이라도 방향이 틀어지면 상관계수가 즉시 떨어진다.

    **δ=0에서 재면 안 된다.** 거기는 cos_sim이 최댓값 1인 정류점이라 해석·수치 gradient가
    둘 다 0에 수렴하고, 남는 것은 부동소수점 잡음뿐이라 비교가 무의미해진다.
    실제 최적화가 지나가는 지점(SNR 30 dB 근방)에서 잰다.

    :param double_precision: float32에서는 중심차분의 최적 상대오차가 4.4e-03에서 바닥을 친다
        (h를 더 줄이면 반올림이 지배). 목표인 1e-3에 닿으려면 float64가 필요하다.
        검증 전용 경로라 비용은 문제되지 않는다. PGD 루프 자체는 그대로 float32로 돈다.
    """
    import copy

    from .perturbation import normalize_snr

    device = x.device
    if double_precision:
        encoder = copy.deepcopy(encoder).double()
        x = x.double()

    with torch.no_grad():
        ref = encoder(x).detach()

    def loss_of(delta: torch.Tensor) -> torch.Tensor:
        # PGD와 같은 손실을 쓴다 — untargeted, 원본에서 멀어지는 방향
        return -(1.0 - cosine_similarity(encoder(x + delta), ref))

    g_init = torch.Generator(device="cpu").manual_seed(seed)
    base = normalize_snr(
        torch.randn(x.shape, generator=g_init, dtype=x.dtype).to(device),
        x, probe_snr_db,
    ).detach()

    delta = base.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(loss_of(delta), delta)

    # 발화 구간에서 뽑는다. 무음에서는 gradient가 0이라 대조가 무의미하다.
    energy = x.abs()
    candidates = torch.nonzero(energy > 0.05 * float(energy.max())).flatten()
    if len(candidates) < n_probes:
        candidates = torch.arange(len(x), device=device)

    g = torch.Generator().manual_seed(seed)
    idx = candidates[torch.randperm(len(candidates), generator=g)[:n_probes]]

    numeric = torch.zeros(len(idx), dtype=x.dtype)
    with torch.no_grad():
        for j, i in enumerate(idx):
            probe = base.clone()
            probe[i] = base[i] + h
            plus = loss_of(probe)
            probe[i] = base[i] - h
            minus = loss_of(probe)
            numeric[j] = (plus - minus) / (2 * h)

    a = analytic[idx].detach().cpu().numpy()
    n = numeric.cpu().numpy()

    denom = max(float(np.linalg.norm(a)), 1e-12)
    rel = float(np.linalg.norm(a - n) / denom)
    corr = float(np.corrcoef(a, n)[0, 1]) if np.std(a) > 0 and np.std(n) > 0 else 0.0

    return GradCheckReport(
        n_probes=len(idx),
        rel_error=rel,
        correlation=corr,
        analytic_norm=float(np.linalg.norm(a)),
        numeric_norm=float(np.linalg.norm(n)),
    )
