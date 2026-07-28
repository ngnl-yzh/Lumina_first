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

from .config import PGDConfig
from .encoder import EncoderEnsemble, SpeakerEncoder
from .perturbation import PerturbResult, pgd_perturbation
from .psychoacoustic import MaskingModel

# 큐 길이별 스텝 배율. 밀릴수록 공격적으로 줄인다.
BACKLOG_LADDER: list[tuple[int, float]] = [
    (2, 1.00),    # 큐 2 이하 — 정상
    (4, 0.50),    # 큐 3~4  — 절반
    (6, 0.25),    # 큐 5~6  — 1/4
]
MIN_STEPS = 20    # 이 아래로는 안 내린다. 더 줄이면 잡음과 구별이 없어진다


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

    def steps_for(self, queue_len: int) -> int:
        for limit, scale in BACKLOG_LADDER:
            if queue_len <= limit:
                return max(MIN_STEPS, int(self.cfg.steps * scale))
        return MIN_STEPS

    def process(self, job: ChunkJob, queue_len: int = 0) -> ChunkOutput:
        steps = self.steps_for(queue_len)
        degraded = steps < self.cfg.steps

        cfg = PGDConfig(**{**vars(self.cfg), "steps": steps})
        if self.masking is None:
            self.masking = MaskingModel(device=job.audio.device,
                                        n_fft=cfg.n_fft, hop_length=cfg.hop_length)

        t0 = time.perf_counter()
        r: PerturbResult = pgd_perturbation(
            job.audio, self.encoders, cfg, masking_model=self.masking, seed=job.seq
        )
        dt = time.perf_counter() - t0

        self.stats.processed += 1
        self.stats.total_sec += dt
        self.stats.max_queue = max(self.stats.max_queue, queue_len)
        self.stats.steps_history.append(steps)
        if degraded:
            self.stats.degraded += 1

        return ChunkOutput(
            seq=job.seq, delta=r.delta, srs=r.srs, snr_db=r.snr_db,
            steps_used=steps, elapsed_sec=dt, degraded=degraded,
        )
