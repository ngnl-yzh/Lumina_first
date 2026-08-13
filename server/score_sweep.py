"""GPT-SoVITS 방어 스윕 채점 — 2단계.

`sweep_gsv.py`가 만든 조건별 복제음을 화자 검증기 셋으로 잰다.
임계값은 `eval_thresholds.py`가 화자 6명으로 구한 값을 그대로 쓴다.

**기준선을 먼저 본다.** 각 조건마다 원본 참조로 만든 복제음이
임계값 위여야 그 줄의 측정이 성립한다. 원본조차 복제되지 않으면
방어가 통한 게 아니라 **그 조건에서는 애초에 복제가 안 되는 것**이다 —
전송 열화 조건(mp3·전화대역·절단)에서 특히 이걸 구분해야 한다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from eval_dsr import ENCODERS, THRESHOLDS, emb, load, sim  # noqa: E402


def main() -> int:
    root = Path("out/sweep")
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dirs:
        print("조건이 없다 — 먼저 sweep_gsv.py를 돌린다.")
        return 1

    plan = {}
    pj = root / "plan.json"
    if pj.exists():
        plan = {c["name"]: c for c in json.loads(pj.read_text(encoding="utf-8"))}

    print("GPT-SoVITS 복제 방어 — 15조건")
    print("임계값 — " + " · ".join(f"{n} {THRESHOLDS[n]}" for n, _ in ENCODERS))
    print("=" * 92)
    print(f"{'조건':<14}{'참조':<11}{'Resem':>9}{'ECAPA':>9}{'WavLM':>9}"
          f"{'저지':>7}  비고")
    print("-" * 92)

    n_ok = n_valid = 0
    rows = []
    for d in dirs:
        src = load(str(d / "original.wav"))
        refs = [emb(e, src) for _, e in ENCODERS]
        vals = {}
        for tag in ("original", "protected"):
            f = d / f"clone_from_{tag}_0.wav"
            if not f.exists():
                continue
            y = load(str(f))
            vals[tag] = [sim(refs[i], emb(e, y))
                         for i, (_, e) in enumerate(ENCODERS)]

        if "original" not in vals or "protected" not in vals:
            continue

        base_ok = any(v >= THRESHOLDS[n]
                      for (n, _), v in zip(ENCODERS, vals["original"]))
        blocked = all(v < THRESHOLDS[n]
                      for (n, _), v in zip(ENCODERS, vals["protected"]))
        n_valid += base_ok
        n_ok += base_ok and blocked

        note = "" if base_ok else "← 기준선 미달, 측정 불성립"
        for tag in ("original", "protected"):
            mark = ("저지" if blocked else "실패") if tag == "protected" else ""
            cells = "".join(
                f"{v:>9.4f}" for v in vals[tag])
            head = d.name if tag == "original" else ""
            print(f"{head:<14}{tag:<11}{cells}{mark:>7}  "
                  f"{note if tag == 'original' else ''}")
        cond = plan.get(d.name, {}).get("cond", {})
        cond = (cond or {}).get("speaker", {}).get("gpt-sovits")
        rows.append({"name": d.name, "baseline_ok": base_ok,
                     "blocked": bool(blocked), "cond": cond,
                     "original": vals["original"], "protected": vals["protected"]})
        print("-" * 92)

    print(f"\n측정이 성립한 조건 {n_valid}개 중 **저지 {n_ok}개** "
          f"({100 * n_ok / max(n_valid, 1):.0f}%)")
    print("저지 = 세 검증기가 **전부** 임계값 아래.")
    (root / "scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
