"""모드 2 청크 워커 — 적체 시 스텝 자동 감축.

hop 1.0초마다 청크가 들어오는데 처리가 그보다 느리면 큐가 무한히 쌓인다.
RTX 3060에서 200스텝이 0.36초라 여유율 64%지만, 그건 계산상 값이다.
실제로는 다른 부하나 느린 회선 때문에 밀릴 수 있다.

밀릴 때 선택지는 둘뿐이다 — **청크를 버리거나, 스텝을 줄이거나.**
청크를 버리면 그 구간이 통째로 무방비가 되므로 스텝을 줄인다.
방어가 약해지는 건 같지만 구멍이 뚫리지는 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .config import HOP_SEC, PGDConfig
from .encoder import EncoderEnsemble, SpeakerEncoder
from .perturbation import PerturbResult, pgd_perturbation
from .psychoacoustic import MaskingModel

# 큐 길이별 스텝 배율. 밀릴수록 공격적으로 줄인다.
BACKLOG_LADDER: list[tuple[int, float]] = [
    (2, 1.00),    # 큐 2 이하 — 정상
    (4, 0.50),    # 큐 3~4  — 절반
    (6, 0.25),    # 큐 5~6  — 1/4
]
# 이 아래로는 안 내린다. 한때 20이었고 "더 줄이면 잡음과 구별이 없어진다"는
# **추정**이었다. `steps_budget.py`로 실제로 쟀다 (2초 청크 · 배율 3.0 · n=10 ·
# 통화 채널 통과 후 평가 · 대조군 C-B 0.6914 ±0.0049).
#
#   스텝   SRS              대조군 대비        초/청크   실시간(hop 1.0초)
#     5   0.8460 ±0.0143   ✗ 잡음보다 나쁨      0.43     OK
#    10   0.6785 ±0.0116   ? 우위 미확인        0.63     OK
#    15   0.6033 ±0.0073   ✓ 유의하게 우세      0.91     OK   ← 하한
#    20   0.5672 ±0.0075   ✓                   1.36     초과
#
# 두 가지가 드러났다.
#   ① **10스텝은 대조군과 구별되지 않는다.** 방어의 진짜 바닥은 10과 15 사이에 있다.
#   ② **20스텝은 CPU에서 실시간을 못 맞춘다.** 하한이 높으면 큐가 계속 밀린다.
#
# 15가 "유의하게 우세한 최소 스텝"이면서 실시간 예산 안에 드는 유일한 지점이다.
# 다만 0.91초는 1.0초 예산에 여유가 거의 없다 — 측정 자체가 부하에 흔들리므로
# CPU에서는 경계선으로 봐야 한다. GPU가 있으면 이 고민이 사라진다.
MIN_STEPS = 15

# 청크 시간 예산 = hop × 이 비율.
#
# 큐 사다리만으로는 부족하다 — **이미 밀린 뒤에야** 줄이기 때문이다.
# 200스텝을 기본으로 두면 느린 장비에서 첫 청크부터 17초가 걸리고,
# 사다리가 반응할 무렵엔 큐가 이미 수십 개다.
#
# 시간 예산은 선제적이다. 첫 청크부터 마감을 지키고,
# 빠른 장비에서는 그냥 걸리지 않으므로 200스텝을 다 돈다.
# 0.7은 전사·전송·overlap-add에 30%를 남긴다는 뜻이다.
TIME_BUDGET_RATIO = 0.7


@dataclass
class ChunkJob:
    seq: int
    audio: torch.Tensor


@dataclass
class ChunkOutput:
    seq: int
    delta: torch.Tensor
    srs: float
    snr_db: float
    steps_used: int
    elapsed_sec: float
    degraded: bool = False


@dataclass
class WorkerStats:
    processed: int = 0
    degraded: int = 0
    max_queue: int = 0
    total_sec: float = 0.0
    steps_history: list[int] = field(default_factory=list)

    def report(self) -> str:
        avg = self.total_sec / self.processed if self.processed else 0.0
        line = (f"청크 {self.processed}개 · 평균 {avg:.2f}초 · 최대 큐 {self.max_queue}")
        if self.degraded:
            # 조용히 넘어가면 안 된다. 방어가 약해진 청크가 몇 개인지 보고에 남는다.
            line += f" · **스텝 감축 {self.degraded}개**"
        return line


class ChunkWorker:
    """청크 큐를 처리한다. 큐 길이에 따라 스텝을 조절한다."""

    def __init__(
        self,
        encoders: EncoderEnsemble | SpeakerEncoder,
        cfg: PGDConfig | None = None,
        masking_model: MaskingModel | None = None,
    ) -> None:
        if isinstance(encoders, SpeakerEncoder):
            encoders = EncoderEnsemble([encoders])
        self.encoders = encoders
        self.cfg = cfg or PGDConfig()
        self.masking = masking_model
        self.stats = WorkerStats()

        # 청크 하나에 허용하는 시간. hop보다 짧게 잡아 전사·전송·조립에 여유를 남긴다.
        # 이 값을 넘기면 남은 PGD 스텝을 포기한다 — 늦은 완벽한 섭동보다
        # 제때 도착한 조금 약한 섭동이 낫다. 통화는 기다려 주지 않는다.
        self.time_budget = HOP_SEC * TIME_BUDGET_RATIO

    def steps_for(self, queue_len: int) -> int:
        for limit, scale in BACKLOG_LADDER:
            if queue_len <= limit:
                return max(MIN_STEPS, int(self.cfg.steps * scale))
        return MIN_STEPS

    def process(self, job: ChunkJob, queue_len: int = 0) -> ChunkOutput:
        # 큐 사다리는 **반응형**이다 — 이미 밀린 뒤에야 줄인다.
        # 그래서 시간 예산을 함께 건다. 예산은 마감(hop)보다 짧게 잡아
        # 전사·전송·조립에 쓸 여유를 남긴다. 이쪽이 **선제적**이라
        # 첫 청크부터 마감을 지킨다.
        steps = self.steps_for(queue_len)
        degraded = steps < self.cfg.steps

        budget = self.cfg.time_budget_sec or self.time_budget
        cfg = PGDConfig(**{**vars(self.cfg), "steps": steps,
                           "time_budget_sec": budget})
        if self.masking is None:
            self.masking = MaskingModel(device=job.audio.device,
                                        n_fft=cfg.n_fft, hop_length=cfg.hop_length)

        t0 = time.perf_counter()
        r: PerturbResult = pgd_perturbation(
            job.audio, self.encoders, cfg, masking_model=self.masking, seed=job.seq
        )
        dt = time.perf_counter() - t0

        # 실제로 돈 스텝을 기록한다. 시간 예산에 걸리면 설정 값보다 적다 —
        # 이 차이를 감추면 "왜 방어가 약하지"에 답할 수 없다.
        actual = r.steps
        degraded = degraded or actual < steps

        self.stats.processed += 1
        self.stats.total_sec += dt
        self.stats.max_queue = max(self.stats.max_queue, queue_len)
        self.stats.steps_history.append(actual)
        if degraded:
            self.stats.degraded += 1

        return ChunkOutput(
            seq=job.seq, delta=r.delta, srs=r.srs, snr_db=r.snr_db,
            steps_used=actual, elapsed_sec=dt, degraded=degraded,
        )
