"""딥보이스 탐지기 검증 — **쓰기 전에 재는 것이 목적이다.**

D08이 P1으로 두고 이렇게 경고했다.
  "ASVspoof 모델은 학습에 없던 합성 방식에 일반화가 약하다. XTTS-v2를 못 잡을 수 있다."

그러니 탐지기를 붙였다고 끝이 아니다. **우리가 실제로 방어하려는 합성 방식(XTTS-v2)을
잡는지**를 확인해야 하고, 못 잡으면 P1 원칙대로 제외하고 한계로 적어야 한다.

이 스크립트는 두 가지를 잰다.
  ① 합성 음성을 합성으로 판정하는가 (재현율)
  ② 사람 음성을 사람으로 판정하는가 (오탐률)

②는 **진짜 사람 목소리 없이는 측정할 수 없다.** 없으면 그렇다고 출력한다.
숫자를 채워 넣는 것보다 빈칸을 남기는 편이 낫다.

사용:
    python benchmark_deepvoice.py --synthetic out/reference.wav
    python benchmark_deepvoice.py --synthetic out/xtts2/*.wav --human 녹음폴더/
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from mirinae.mode1.deepvoice import DEFAULT_THRESHOLD, DeepvoiceDetector


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out += sorted(p.glob("*.wav"))
        else:
            out += [Path(g) for g in sorted(glob.glob(pat))]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="딥보이스 탐지기 검증")
    p.add_argument("--synthetic", nargs="*", default=[],
                   help="합성 음성 (fake로 판정되어야 함)")
    p.add_argument("--human", nargs="*", default=[],
                   help="사람 음성 (real로 판정되어야 함)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--model", default=None,
                   help="다른 사전학습 탐지기로 교체해 비교한다 (HF 모델 id)")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    from protect import load_wav

    det = (DeepvoiceDetector(model_name=args.model, threshold=args.threshold)
           if args.model else DeepvoiceDetector(threshold=args.threshold))
    print(f"모델: {det.model_name}")
    print(f"임계값: {args.threshold} (이상이면 '딥보이스 의심')\n")

    results: dict[str, list[dict]] = {}

    for group, patterns, expect_fake in (
        ("합성 (XTTS 등)", args.synthetic, True),
        ("사람", args.human, False),
    ):
        files = expand(patterns)
        if not files:
            continue

        print(f"── {group} · {len(files)}개 " + "─" * 30)
        rows = []
        for f in files:
            wav = load_wav(f).numpy().astype(np.float32)
            r = det.score(wav)
            hit = r.is_synthetic if expect_fake else not r.is_synthetic
            mark = "✓" if hit else "✗"
            print(f"  {mark} {f.name:<34} fake={r.fake_prob:.4f}  {r.label()}"
                  f"  ({r.duration_sec:.1f}초 · 창 {r.n_windows})")
            rows.append({"file": str(f), "fake_prob": r.fake_prob,
                         "label": r.label(), "correct": hit})

        correct = sum(1 for r in rows if r["correct"])
        rate = correct / len(rows)
        metric = "재현율" if expect_fake else "정상 판정률"
        # 표본이 적으므로 점 추정만 쓰면 안 된다. 16개에서 3/16이면 [6.6%, 43.0%]다.
        from mirinae.metrics import wilson_ci
        lo, hi = wilson_ci(correct, len(rows))
        print(f"  → {metric} {correct}/{len(rows)} = {rate * 100:.1f}% "
              f"[{lo * 100:.1f}, {hi * 100:.1f}]\n")
        results[group] = rows

    # ── 판정 ──────────────────────────────────────────────────────────────────
    print("=" * 56)
    syn = results.get("합성 (XTTS 등)", [])
    hum = results.get("사람", [])

    if not syn:
        print("합성 음성 표본이 없다. 재현율을 잴 수 없다.")
        return 1

    recall = sum(1 for r in syn if r["correct"]) / len(syn)

    if recall >= 0.8:
        print(f"재현율 {recall * 100:.0f}% — XTTS 합성을 잡는다.")
    elif recall >= 0.5:
        print(f"재현율 {recall * 100:.0f}% — 절반 정도만 잡는다. 보조 신호로만 쓸 것.")
    else:
        print(f"재현율 {recall * 100:.0f}% — **XTTS를 못 잡는다.**")
        print("  D08이 예고한 일반화 실패다. P1 원칙대로 제외하고 한계로 기술할 것.")

    if not hum:
        print()
        print("⚠ 사람 음성 표본이 없어 **오탐률을 측정하지 못했다.**")
        print("  재현율만으로는 탐지기를 쓸 수 없다 — 전부 '가짜'라고 답해도 재현율은 100%다.")
        print("  팀원 녹음을 --human 으로 넣어 다시 돌릴 것.")
    else:
        fpr = 1 - sum(1 for r in hum if r["correct"]) / len(hum)
        print(f"오탐률 {fpr * 100:.0f}% — 사람 음성을 합성으로 잘못 본 비율")
        if fpr > 0.1:
            print("  오탐이 너무 많다. 정상 통화에서 위험도가 부풀려진다.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
