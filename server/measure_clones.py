"""이미 생성된 복제음을 측정한다 — 재생성 없이 통계만 다시 낸다.

XTTS 생성은 조건당 1분 넘게 걸린다. 파라미터를 바꾸지 않았는데 다시 돌리는 것은 낭비이고,
중간에 끊긴 실행의 부분 결과를 살리는 용도로도 필요하다.

사용:
    python measure_clones.py out/stat/clone --original out/xtts2/original.wav
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from mirinae.config import default_device
from mirinae.encoder import SpeakerEncoder, cosine_similarity


def label_of(path: Path) -> str:
    """clone_from_<원본이름>_<반복번호>.wav → 조건 라벨."""
    m = re.match(r"clone_from_(.+?)_(\d+)$", path.stem)
    stem = m.group(1) if m else path.stem
    return {
        "original": "원본",
        "protected": "보호본",
        "control_C-A": "C-A",
        "control_C-B": "C-B",
        "control_C-C": "C-C",
        "control_C-E": "C-E",
    }.get(stem, stem)


def main() -> int:
    p = argparse.ArgumentParser(description="생성된 복제음 통계 측정")
    p.add_argument("clone_dir")
    p.add_argument("--original", required=True, help="비교 기준이 되는 원본 WAV")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    from protect import load_wav

    device = default_device()
    encoder = SpeakerEncoder(device=device)
    ref = encoder(load_wav(Path(args.original)).to(device)).detach()

    groups: dict[str, list[float]] = defaultdict(list)
    for wav_path in sorted(Path(args.clone_dir).glob("clone_from_*.wav")):
        with torch.no_grad():
            wav = load_wav(wav_path).to(device)
            groups[label_of(wav_path)].append(
                float(cosine_similarity(encoder(wav), ref))
            )

    if not groups:
        print("복제음을 찾지 못했다.")
        return 1

    print(f"{'조건':<12}{'n':>3}{'평균':>10}{'표준편차':>10}{'±95% CI':>11}{'최소~최대':>18}")
    print("─" * 66)

    stats: dict[str, dict] = {}
    for label in sorted(groups, key=lambda k: (k != "원본", k != "보호본", k)):
        v = groups[label]
        m = statistics.fmean(v)
        sd = statistics.stdev(v) if len(v) > 1 else float("nan")
        half = 1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        stats[label] = {"n": len(v), "mean": m, "sd": sd, "ci_half": half,
                        "samples": v}
        sd_s = "—" if math.isnan(sd) else f"{sd:.4f}"
        ci_s = "—" if math.isnan(half) else f"±{half:.4f}"
        print(f"{label:<12}{len(v):>3}{m:>10.4f}{sd_s:>10}{ci_s:>11}"
              f"{f'{min(v):.3f}~{max(v):.3f}':>18}")

    base = stats.get("원본", {}).get("mean")
    prot = stats.get("보호본", {}).get("mean")

    print()
    if base is not None and prot is not None:
        drop = base - prot
        # 두 조건의 CI가 겹치는지 — 겹치면 차이를 주장할 수 없다
        b, pr = stats["원본"], stats["보호본"]
        overlap = (
            not math.isnan(b["ci_half"]) and not math.isnan(pr["ci_half"])
            and (base - b["ci_half"]) <= (prot + pr["ci_half"])
        )
        print(f"보호에 의한 하락: {drop:+.4f}")
        print(f"원본  95% CI: [{base - b['ci_half']:.4f}, {base + b['ci_half']:.4f}]")
        print(f"보호본 95% CI: [{prot - pr['ci_half']:.4f}, {prot + pr['ci_half']:.4f}]")
        print()
        if overlap:
            print("판정: **구간이 겹친다 — 차이를 주장할 수 없다.**")
            print("  방어 효과가 없다는 뜻이 아니라, 이 표본으로는 있는지 없는지 알 수 없다는 뜻이다.")
        else:
            print("판정: 구간이 분리된다 — 하락이 생성 편차로는 설명되지 않는다.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
