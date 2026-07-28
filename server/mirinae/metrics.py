"""지표 — SNR · 가청도 · SRS · DSR + 신뢰구간.

보고 형식의 원칙: **n · 신뢰구간 · 대조군 대비** 세 가지가 항상 함께 나온다.
표본 6개일 때 DSR 100%의 95% CI는 [61%, 100%]지만 n=30이면 [88%, 100%]로 좁아진다.
숫자 하나만 쓰면 그 차이가 감춰진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import EPS
from .encoder import cosine_similarity
from .psychoacoustic import MaskingModel


# ── 신호 품질 ─────────────────────────────────────────────────────────────────

def snr_db(x: torch.Tensor, delta: torch.Tensor) -> float:
    p_x = float((x ** 2).sum())
    p_d = float((delta ** 2).sum())
    return 10.0 * math.log10(max(p_x, EPS) / max(p_d, EPS))


@dataclass
class AudibilityReport:
    """섭동이 마스킹 임계값을 얼마나 넘는가.

    0에 가까울수록 "안 들린다"는 보장이 지켜진 것이다.
    정규화 순서 결함이 되살아나면 여기서 즉시 드러난다.
    """

    max_excess_db: float          # 가장 크게 초과한 bin의 초과량
    violation_ratio: float        # 임계값을 넘은 bin 비율
    mean_excess_db: float         # 넘은 bin들의 평균 초과량

    def ok(self, tol_db: float = 0.5) -> bool:
        return self.max_excess_db <= tol_db

    def __str__(self) -> str:
        return (
            f"최대 초과 {self.max_excess_db:+.2f} dB · "
            f"위반 bin {self.violation_ratio * 100:.2f}% · "
            f"평균 초과 {self.mean_excess_db:+.2f} dB"
        )


def audibility(
    delta: torch.Tensor,
    thr_mag: torch.Tensor,
    ratio: float,
    model: MaskingModel,
) -> AudibilityReport:
    with torch.no_grad():
        mag = model.stft(delta).abs()
        bound = torch.clamp(thr_mag * ratio, min=EPS)
        excess_db = 20.0 * torch.log10(torch.clamp(mag, min=EPS) / bound)

        over = excess_db > 0
        n_over = int(over.sum())
        return AudibilityReport(
            max_excess_db=float(excess_db.max()),
            violation_ratio=n_over / excess_db.numel(),
            mean_excess_db=float(excess_db[over].mean()) if n_over else 0.0,
        )


def band_energy_ratio_db(
    delta: torch.Tensor,
    low_hz: float,
    high_hz: float,
    sample_rate: int,
) -> float:
    """대역 밖 에너지 / 전체 에너지 (dB). 대역 제한이 실제로 걸렸는지 확인한다."""
    with torch.no_grad():
        spec = torch.fft.rfft(delta)
        freqs = torch.fft.rfftfreq(delta.shape[-1], d=1.0 / sample_rate,
                                   device=delta.device)
        power = spec.abs() ** 2
        outside = ((freqs < low_hz) | (freqs > high_hz))
        return 10.0 * math.log10(
            max(float(power[outside].sum()), EPS) / max(float(power.sum()), EPS)
        )


# ── 화자 유사도 ───────────────────────────────────────────────────────────────

def srs(encoder, wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    """SRS — 임베딩 코사인 유사도. 튜닝에 쓴다."""
    with torch.no_grad():
        return float(cosine_similarity(encoder(wav_a), encoder(wav_b)))


def dsr(srs_values: list[float], threshold: float) -> float:
    """DSR — 화자검증기가 "다른 사람"으로 판정한 비율. 보고에 쓴다.

    :param threshold: C-D(타 화자) 대조군에서 얻은 판정 기준선.
    """
    if not srs_values:
        return 0.0
    return sum(1 for s in srs_values if s < threshold) / len(srs_values)


# ── 신뢰구간 ──────────────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 95% 신뢰구간 (Wilson score).

    정규근사(Wald)는 0%나 100%에서 폭이 0이 되어 버려 쓸 수 없다.
    DSR은 극단값이 자주 나오므로 Wilson을 쓴다.
    n=30에서 0/30이면 [0.0%, 11.4%], 6/6이면 [61.0%, 100%]가 나온다.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class ConditionResult:
    """DSR 보고 한 줄. 어떤 조건도 이 형식을 벗어나지 않는다."""

    name: str
    n: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_ci(self.successes, self.n)


def format_dsr_table(results: list[ConditionResult], baseline: str = "C-B") -> str:
    """D09 §06이 못박은 보고 형식 그대로 출력한다."""
    base = next((r for r in results if r.name == baseline), None)

    lines = [
        f"{'조건':<16}{'n':>4}{'DSR':>9}{'95% CI':>18}{'vs ' + baseline:>12}",
        "─" * 60,
    ]
    for r in results:
        lo, hi = r.ci
        if base is None or r.name == baseline:
            delta = "기준" if r.name == baseline else "—"
        else:
            delta = f"{(r.rate - base.rate) * 100:+.1f}%p"
        lines.append(
            f"{r.name:<16}{r.n:>4}{r.rate * 100:>8.1f}%"
            f"{f'[{lo * 100:.1f}, {hi * 100:.1f}]':>18}{delta:>12}"
        )
    return "\n".join(lines)
