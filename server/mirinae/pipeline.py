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
    audibility: AudibilityReport
    out_of_band_db: float
    srs_protected: float
    srs_controls: dict[str, float] = field(default_factory=dict)
    total_sec: float = 0.0

    # 판정 임계값. 본래 C-D(타 화자) 대조군에서 얻어야 하는 값이라
    # 타 화자 음성이 없는 동안에는 잠정값이다. 절대 판정으로 읽으면 안 된다.
    PROVISIONAL_THRESHOLD = 0.75

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
            f"가청도 — {self.audibility}",
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
            f"※ 판정 임계값 {self.PROVISIONAL_THRESHOLD}은 잠정값이다. "
            "C-D(타 화자) 대조군으로 확정할 것."
        )
        return "\n".join(lines)


def protect_utterance(
    x: torch.Tensor,
    encoders: EncoderEnsemble | SpeakerEncoder,
    cfg: PGDConfig | None = None,
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: float = CHUNK_SEC,
    hop_sec: float = HOP_SEC,
    with_controls: bool = True,
    progress: bool = True,
) -> ProtectionResult:
    """발화 하나를 청크 단위로 보호한다.

    청크별로 독립 최적화하고 Hann overlap-add로 합친다.
    이렇게 하면 **모든 2초 창이 각각 보호**되므로 발췌 공격에 강해진다.
    """
    import time

    cfg = cfg or PGDConfig()
    if isinstance(encoders, SpeakerEncoder):
        encoders = EncoderEnsemble([encoders])

    device = x.device
    model = MaskingModel(sample_rate, cfg.n_fft, cfg.hop_length, device=device)

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
        out_of_band_db=band_energy_ratio_db(delta, cfg.band_low_hz,
                                            cfg.band_high_hz, sample_rate),
        srs_protected=0.0,
        total_sec=time.perf_counter() - t_total,
    )

    primary = encoders.encoders[0]
    with torch.no_grad():
        ref = primary(x)
        result.srs_protected = float(cosine_similarity(primary(protected), ref))

        if with_controls:
            for name, wav in make_controls(x, delta, cfg.target_snr_db).items():
                result.srs_controls[name] = float(cosine_similarity(primary(wav), ref))

    return result
