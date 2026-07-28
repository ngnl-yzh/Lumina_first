"""파라미터 스윕 — 마스킹 배율과 alpha를 대조군 기준으로 평가한다.

D07 W1의 "마스킹 배율 확정" 작업에 쓰는 도구다.

핵심은 **절대 SRS가 아니라 대조군 대비**다.
같은 SNR의 백색잡음·대역제한잡음보다 낫지 않으면 적대적 최적화를 했다고 주장할 수 없다.
그래서 매 실행에 C-A·C-B 기준선을 함께 찍는다.

사용:
    python sweep_params.py 목소리.wav
    python sweep_params.py 목소리.wav --steps 200 --ratios 1,3,10 --alphas 1e-3,3e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
from mirinae.controls import bandlimited_noise, white_noise
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.metrics import audibility
from mirinae.perturbation import pgd_perturbation
from mirinae.psychoacoustic import MaskingModel
from mirinae.vad import speech_mask


def parse_list(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · 마스킹 배율/alpha 스윕")
    p.add_argument("input", nargs="?", help="입력 WAV (생략 시 합성 신호)")
    p.add_argument("--steps", type=int, default=50,
                   help="스윕은 스텝을 줄여 빠르게 돈다. 확정값은 200으로 재확인할 것")
    p.add_argument("--ratios", type=parse_list, default=[0.75, 1.5, 3.0, 10.0])
    p.add_argument("--alphas", type=parse_list, default=[1e-4, 1e-3, 3e-3])
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=2.0,
                   help="긴 파일은 앞에서 이만큼만 잘라 쓴다 (청크 1개 기준 비교)")
    p.add_argument("-o", "--out", default="sweep_result.json")
    args = p.parse_args()

    device = default_device()
    if args.input:
        from protect import load_wav
        x = load_wav(Path(args.input))
        source = str(args.input)
    else:
        from protect import synth_wav
        x = synth_wav(args.seconds)
        source = "synth"
        print("※ 합성 신호로 돈다. 여기서 나온 수치는 방향만 보여줄 뿐 결과가 아니다.\n")

    n = int(SAMPLE_RATE * args.seconds)
    x = x[:n].to(device)

    encoder = SpeakerEncoder(device=device)
    model = MaskingModel(device=device)
    thr, _ = model.threshold(x)
    mask = speech_mask(x)

    with torch.no_grad():
        ref = encoder(x)

    # ── 기준선 — 이걸 못 이기면 아무 주장도 할 수 없다 ────────────────────────
    g = torch.Generator(device="cpu").manual_seed(0)
    baselines: dict[str, float] = {}
    for label, fn in (("C-A 백색잡음", white_noise), ("C-B 대역제한잡음", bandlimited_noise)):
        noise = fn(x.cpu(), args.snr, g).to(device)
        with torch.no_grad():
            baselines[label] = float(cosine_similarity(encoder(x + noise), ref))

    print(f"입력: {source} · {args.seconds}초 · 목표 SNR {args.snr} dB · {args.steps}스텝")
    print(f"장치: {device}\n")
    print("기준선 (대조군)")
    for label, srs in baselines.items():
        print(f"  {label:<20}SRS {srs:.4f}")
    best_baseline = min(baselines.values())
    print(f"  → 이겨야 할 값: {best_baseline:.4f}\n")

    # ── 스윕 ──────────────────────────────────────────────────────────────────
    print(f"{'배율':>8}{'alpha':>10}{'SRS':>9}{'대조군대비':>11}"
          f"{'SNR dB':>9}{'최대초과dB':>12}{'위반%':>8}{'초':>7}")
    print("─" * 74)

    rows = []
    for ratio in args.ratios:
        for alpha in args.alphas:
            cfg = PGDConfig(steps=args.steps, masking_ratio=ratio,
                            alpha=alpha, target_snr_db=args.snr)
            t0 = time.perf_counter()
            r = pgd_perturbation(x, encoder, cfg, masking_model=model, vad_mask=mask)
            dt = time.perf_counter() - t0
            aud = audibility(r.delta, thr, ratio, model)

            wins = r.srs < best_baseline
            mark = "✓" if wins else "✗"
            print(f"{ratio:>8.2f}{alpha:>10.0e}{r.srs:>9.4f}"
                  f"{mark + f' {best_baseline - r.srs:+.3f}':>11}"
                  f"{r.snr_db:>9.1f}{aud.max_excess_db:>12.1f}"
                  f"{aud.violation_ratio * 100:>8.2f}{dt:>7.1f}")

            rows.append({
                "masking_ratio": ratio, "alpha": alpha, "srs": r.srs,
                "beats_baseline": wins, "snr_db": r.snr_db,
                "max_excess_db": aud.max_excess_db,
                "violation_ratio": aud.violation_ratio, "elapsed_sec": dt,
            })

    winners = [r for r in rows if r["beats_baseline"]]
    print()
    if winners:
        best = min(winners, key=lambda r: r["srs"])
        print(f"최저 SRS: 배율 {best['masking_ratio']} · alpha {best['alpha']:.0e} "
              f"→ SRS {best['srs']:.4f} (가청도 최대 초과 {best['max_excess_db']:.1f} dB)")
        print("가청도 초과가 클수록 들릴 위험이 커진다. 청취 평가로 상한을 정한 뒤 역산할 것.")
    else:
        print("대조군을 이긴 설정이 없다. 이 상태로는 '적대적 섭동의 우위'를 주장할 수 없다.")

    Path(args.out).write_text(json.dumps({
        "source": source, "steps": args.steps, "target_snr_db": args.snr,
        "baselines": baselines, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
