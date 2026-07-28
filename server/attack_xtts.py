"""XTTS-v2 화이트박스 보호 — **.venv-xtts 전용.**

전이 실패에 대한 대응이다. Resemblyzer로 만든 섭동은 XTTS로 넘어가지 않았다
(보호본 복제 SRS 0.9135, 대조군 C-B 0.8806보다도 못함).
표적을 XTTS 자체 인코더로 바꿔 화이트박스로 공략한다.

PGD 본체·마스킹·대조군은 mirinae 패키지를 그대로 쓴다.
바뀌는 것은 손실을 계산하는 인코더뿐이다 — 그래서 코드가 이만큼 짧다.

사용:
    .venv-xtts\\Scripts\\python attack_xtts.py out/reference.wav -o out/xtts
    .venv-xtts\\Scripts\\python attack_xtts.py out/reference.wav -o out/xtts --steps 200 --ratio 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

from mirinae.chunking import overlap_add, split                  # noqa: E402
from mirinae.config import PGDConfig, SAMPLE_RATE                # noqa: E402
from mirinae.controls import make_controls                       # noqa: E402
from mirinae.encoder import EncoderEnsemble, cosine_similarity   # noqa: E402
from mirinae.metrics import audibility, band_energy_ratio_db, snr_db  # noqa: E402
from mirinae.perturbation import pgd_perturbation                # noqa: E402
from mirinae.psychoacoustic import MaskingModel                  # noqa: E402
from xtts_encoder import build_target, load_xtts                 # noqa: E402


def load_wav_16k(path: Path) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = float(np.abs(wav).max())
    if peak > 1.0:
        wav = wav / peak
    return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · XTTS 화이트박스 보호")
    p.add_argument("input")
    p.add_argument("-o", "--out", default="out/xtts")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--ratio", type=float, default=PGDConfig.masking_ratio)
    p.add_argument("--snr", type=float, default=PGDConfig.target_snr_db)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--gpt-cond", action="store_true",
                   help="GPT 조건 latent까지 함께 공략 (느리지만 강하다)")
    args = p.parse_args()

    print("XTTS 모델 로드 중...", flush=True)
    model, device = load_xtts()
    targets = build_target(model, device, use_gpt_cond=args.gpt_cond)
    ensemble = EncoderEnsemble(targets)
    print(f"공략 표적: {', '.join(ensemble.names)}")
    print(f"장치: {device}")

    x = load_wav_16k(Path(args.input))
    if args.seconds and x.shape[-1] > SAMPLE_RATE * args.seconds:
        x = x[: int(SAMPLE_RATE * args.seconds)]
    x = x.to(device)
    print(f"입력: {args.input} · {x.shape[-1] / SAMPLE_RATE:.1f}초")

    cfg = PGDConfig(steps=args.steps, masking_ratio=args.ratio, target_snr_db=args.snr)
    print(f"설정: {cfg.steps}스텝 · 마스킹 배율 {cfg.masking_ratio} · "
          f"목표 SNR {cfg.target_snr_db} dB\n")

    masking = MaskingModel(device=device)
    pieces, starts, chunk_len = split(x)
    deltas = []
    for i, piece in enumerate(pieces):
        r = pgd_perturbation(piece, ensemble, cfg, masking_model=masking, seed=i)
        deltas.append(r.delta)
        print(f"  청크 {i + 1}/{len(pieces)} · SRS {r.srs:.4f}", flush=True)

    delta = overlap_add(deltas, starts, x.shape[-1], chunk_len)
    protected = x + delta

    # ── 측정 ──────────────────────────────────────────────────────────────────
    thr, _ = masking.threshold(x)
    primary = targets[0]
    with torch.no_grad():
        ref = primary(x)
        srs_protected = float(cosine_similarity(primary(protected), ref))
        controls = make_controls(x, delta, cfg.target_snr_db)
        srs_controls = {
            k: float(cosine_similarity(primary(v), ref)) for k, v in controls.items()
        }

    aud = audibility(delta, thr, cfg.masking_ratio, masking)
    global_snr = snr_db(x, delta)
    oob = band_energy_ratio_db(delta, cfg.band_low_hz, cfg.band_high_hz, SAMPLE_RATE)

    print(f"\n전역 SNR {global_snr:.1f} dB · 대역 밖 {oob:.1f} dB")
    print(f"가청도 — {aud}\n")
    print(f"{'조건':<22}{'XTTS 인코더 SRS':>16}")
    print("─" * 40)
    print(f"{'적대적 섭동 (미리내)':<22}{srs_protected:>16.4f}")
    for k in sorted(srs_controls):
        print(f"{k:<22}{srs_controls[k]:>16.4f}")

    best_noise = min(srs_controls.get(k, 1.0) for k in ("C-A", "C-B"))
    print(f"\n대조군 대비: {best_noise - srs_protected:+.4f} "
          + ("— 우위 있음" if srs_protected < best_noise else "— 잡음보다 못하다"))

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, wav: torch.Tensor) -> None:
        sf.write(str(out / name), wav.detach().cpu().numpy(), SAMPLE_RATE)

    write("original.wav", x)
    write("protected.wav", protected)
    write("delta.wav", delta)
    for k, v in controls.items():
        write(f"control_{k}.wav", v)

    (out / "report.json").write_text(json.dumps({
        "target": ensemble.names,
        "config": {"steps": cfg.steps, "masking_ratio": cfg.masking_ratio,
                   "target_snr_db": cfg.target_snr_db},
        "global_snr_db": global_snr,
        "out_of_band_db": oob,
        "audibility": {"max_excess_db": aud.max_excess_db,
                       "violation_ratio": aud.violation_ratio},
        "srs_xtts_encoder": {"protected": srs_protected, **srs_controls},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n출력: {out.resolve()}")
    print("\n다음 — 실제 복제로 확인 (메인 venv):")
    print(f"  python clone_test.py {out}/original.wav {out}/protected.wav --controls {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
