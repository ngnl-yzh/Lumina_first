"""C-D 대조군 교정 — "다른 사람"이 SRS 얼마인지 실제로 잰다.

## 이 도구가 없으면 SRS를 해석할 수 없다

    보호본의 SRS가 0.46이다. → 좋은 건가?

답할 수 없다. **서로 다른 두 사람의 SRS가 얼마인지 모르기 때문이다.**
0.75라면 0.46은 "남남보다 더 멀어졌다"는 강한 주장이고,
0.20이라면 "여전히 같은 사람으로 들린다"는 뜻이다. 정반대 결론이 같은 숫자에서 나온다.

`metrics.dsr()`은 임계값을 인자로 받는데, 그 임계값을 만들 대조군이 없었다.
`pipeline.ProtectionResult.PROVISIONAL_THRESHOLD = 0.75`도 근거 없는 잠정값이고
주석에 "절대 판정으로 읽으면 안 된다"고 적혀 있다. 이 도구가 그 근거를 만든다.

## 무엇을 재는가

같은 문장을 서로 다른 화자로 발음한 음성들의 **모든 쌍**에 대해 SRS를 잰다.
쌍이 n(n-1)/2개 나오므로 6명이면 15쌍이다. 그 분포가 "다른 사람" 기준선이다.

같은 화자를 다른 구간에서 자른 것끼리도 재서 **같은 사람 기준선**을 함께 낸다.
둘 사이가 벌어져 있어야 인코더가 화자를 구별한다고 말할 수 있다.
겹치면 인코더 자체를 못 믿는다는 뜻이고, 그러면 모든 SRS 수치가 무의미해진다.

## 두 분포를 **같은 조건에서** 재야 한다

처음 구현은 같은 화자를 3초 조각으로, 타 화자를 전체 파일로 쟀다. 이건 틀렸다.
임베딩은 입력이 길수록 안정되고 유사도가 높게 나온다 —
길이가 다르면 "같은 사람이라 비슷한 것"과 "길어서 비슷한 것"을 구별할 수 없다.
지금은 **모든 음성을 같은 길이 조각으로 자른 뒤** 같은 풀에서 쌍을 만든다.
조각 두 개가 같은 화자에서 왔으면 같은 화자 쌍, 다른 화자에서 왔으면 타 화자 쌍이다.

## 임계값을 어디로 잡는가 — EER

**동일오류율(EER)** 지점을 쓴다. 같은 사람을 남으로 잘못 보는 비율과
남을 같은 사람으로 잘못 보는 비율이 같아지는 지점이다. 화자검증의 표준 지표다.

타 화자 분포의 상위 백분위(p95)도 함께 찍지만 **임계값으로 쓰지 않는다.**
p95는 "가장 닮은 남남 쌍"에 맞춘 느슨한 기준이라 우리 결과를 유리하게 만든다.
자기 성과를 좋아 보이게 하는 방향으로 기준선을 고르는 것은 측정이 아니다.

## 한계

대상이 XTTS 합성 화자다. 합성 화자끼리의 거리가 사람끼리의 거리와 같다는 보장이 없다.
여기서 나온 임계값도 **잠정값**이며 팀원 녹음으로 다시 재야 한다.
그래도 근거 없는 0.75보다는 낫다 — 적어도 어디서 나온 숫자인지 말할 수 있다.

사용:
    python calibrate_dsr.py out/speakers
    python calibrate_dsr.py out/speakers --compare out/synth_200/protected.wav
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from mirinae.config import SAMPLE_RATE, default_device
from mirinae.encoder import SpeakerEncoder, cosine_similarity

# 임베딩이 안정되는 최소 길이. Resemblyzer partial utterance가 1.6초다.
SEGMENT_SEC = 3.0
PERCENTILE = 95


def load(path: Path) -> torch.Tensor:
    from protect import load_wav
    return load_wav(path)


def segments(x: torch.Tensor, seg_sec: float = SEGMENT_SEC) -> list[torch.Tensor]:
    """같은 화자에서 겹치지 않는 조각을 뽑는다. 같은 사람 기준선용."""
    n = int(SAMPLE_RATE * seg_sec)
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def eer_threshold(same: list[float], diff: list[float]) -> tuple[float, float]:
    """동일오류율 지점의 임계값과 그때의 오류율.

    임계값 t에서
      FRR(같은 사람을 남으로 봄) = same 중 t 미만의 비율
      FAR(남을 같은 사람으로 봄) = diff 중 t 이상의 비율
    둘이 같아지는 t를 찾는다. 후보는 관측된 값들 사이의 중간점이면 충분하다.
    """
    if not same or not diff:
        return (float("nan"), float("nan"))
    cands = sorted(set(same) | set(diff))
    best_t, best_gap, best_rate = cands[0], float("inf"), 1.0
    for i in range(len(cands)):
        t = cands[i] if i == 0 else (cands[i - 1] + cands[i]) / 2.0
        frr = sum(1 for v in same if v < t) / len(same)
        far = sum(1 for v in diff if v >= t) / len(diff)
        gap = abs(frr - far)
        if gap < best_gap:
            best_t, best_gap, best_rate = t, gap, (frr + far) / 2.0
    return (best_t, best_rate)


def describe(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "p95": percentile(vals, 95),
        "samples": vals,
    }


def show(d: dict) -> None:
    if not d.get("n"):
        print(f"  {d['name']:<22} 표본 없음")
        return
    print(f"  {d['name']:<22} n={d['n']:<4} 평균 {d['mean']:.4f} ±{d['sd']:.4f}   "
          f"범위 [{d['min']:.4f}, {d['max']:.4f}]   p95 {d['p95']:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description="C-D 대조군 교정 — DSR 임계값 산출")
    p.add_argument("speakers", nargs="?", default="out/speakers",
                   help="화자별 WAV가 든 폴더 (make_speakers.py 산출물)")
    p.add_argument("--compare", nargs="*", default=[],
                   help="이 파일들의 SRS를 기준선과 나란히 놓는다")
    p.add_argument("--reference", default=None,
                   help="--compare 대상의 원본. 지정하면 원본 대비 SRS를 낸다")
    p.add_argument("-o", "--out", default="out/dsr_calibration.json")
    args = p.parse_args()

    folder = Path(args.speakers)
    wavs = sorted(folder.glob("*.wav"))
    if len(wavs) < 2:
        print(f"화자 음성이 2개 미만이다 ({folder}). "
              f"먼저 .venv-xtts\\Scripts\\python make_speakers.py 를 돌릴 것.")
        return 1

    device = default_device()
    encoder = SpeakerEncoder(device=device)
    print(f"장치: {device} · 화자 {len(wavs)}명\n")

    # ── 모든 음성을 같은 길이 조각으로 자른다 ─────────────────────────────────
    # 같은 조건에서 재지 않으면 "같은 사람이라 비슷한 것"과 "길어서 비슷한 것"이 섞인다.
    pool: list[tuple[str, torch.Tensor]] = []
    for w in wavs:
        x = load(w).to(device)
        with torch.no_grad():
            for s in segments(x):
                pool.append((w.stem, encoder(s)))

    same: list[float] = []
    diff: list[float] = []
    pair_scores: dict[tuple[str, str], list[float]] = {}
    for (na, ea), (nb, eb) in itertools.combinations(pool, 2):
        v = float(cosine_similarity(ea, eb))
        if na == nb:
            same.append(v)
        else:
            diff.append(v)
            key = (na, nb) if na < nb else (nb, na)
            pair_scores.setdefault(key, []).append(v)

    d_same = describe("같은 화자 (다른 조각)", same)
    d_diff = describe("타 화자 (C-D)", diff)

    print(f"조각 {len(pool)}개 · 각 {SEGMENT_SEC}초 — 두 분포를 같은 조건에서 잰다\n")
    print("기준선")
    show(d_same)
    show(d_diff)

    if not diff:
        print("\n타 화자 쌍이 없다.")
        return 1

    threshold, eer = eer_threshold(same, diff)
    p95 = percentile(diff, PERCENTILE)
    gap = d_same["mean"] - d_diff["mean"]

    print(f"\n인코더 판별력 — 같은 화자와 타 화자 평균 차이 {gap:.4f}")
    overlap = d_diff["max"] >= d_same["min"]
    if overlap:
        print("  ※ 두 분포가 겹친다. 인코더가 화자를 완전히 가르지 못한다는 뜻이므로")
        print("     아래 임계값과 이를 쓰는 모든 SRS 해석을 조심해서 읽어야 한다.")
    else:
        print("  두 분포가 겹치지 않는다. 인코더가 화자를 구별한다.")

    print(f"\n**DSR 판정 임계값 (EER 지점): {threshold:.4f}   동일오류율 {eer*100:.1f}%**")
    print(f"  보호본의 SRS가 이 값보다 낮으면 '다른 사람'으로 판정됐다고 말할 수 있다.")
    print(f"  참고 — 타 화자 상위 {PERCENTILE}백분위는 {p95:.4f}지만 임계값으로 쓰지 않는다.")
    print(f"         '가장 닮은 남남 쌍'에 맞춘 느슨한 기준이라 우리 결과를 유리하게 만든다.")
    print(f"  `pipeline.PROVISIONAL_THRESHOLD`의 현재 값 0.75와 비교할 것.")
    if eer > 0.15:
        print(f"\n  ※ 동일오류율 {eer*100:.1f}%는 높다. 이 인코더는 이 화자들을 잘 못 가른다.")
        print(f"     임계값을 절대 판정으로 쓰지 말고 대조군 **상대 비교**에 쓸 것.")

    print("\n타 화자 쌍 평균")
    for (na, nb), vals in sorted(pair_scores.items(),
                                 key=lambda kv: -statistics.fmean(kv[1])):
        print(f"  {na} ↔ {nb}   {statistics.fmean(vals):.4f}  (조각쌍 {len(vals)}개)")

    # ── 비교 대상 ─────────────────────────────────────────────────────────────
    compared = []
    if args.compare:
        ref_emb = None
        if args.reference:
            with torch.no_grad():
                ref_emb = encoder(load(Path(args.reference)).to(device))
        print("\n비교 대상")
        print("  ※ 이들은 전체 길이로 재므로 위 조각 기준선보다 유사도가 높게 나온다.")
        print("     같은 편향이 보호본과 대조군에 똑같이 걸리므로 **상대 비교**는 유효하다.")
        for c in args.compare:
            cp = Path(c)
            if not cp.exists():
                print(f"  {cp} — 없음")
                continue
            with torch.no_grad():
                e = encoder(load(cp).to(device))
            v = float(cosine_similarity(e, ref_emb)) if ref_emb is not None else float("nan")
            verdict = "타 화자 판정" if v < threshold else "같은 화자로 남음"
            print(f"  {cp.name:<28} SRS {v:.4f}  → {verdict}")
            compared.append({"path": str(cp), "srs": v, "below_threshold": v < threshold})

    payload = {
        "n_speakers": len(wavs),
        "segment_sec": SEGMENT_SEC,
        "n_segments": len(pool),
        "threshold": threshold,
        "threshold_rule": "EER",
        "eer": eer,
        "p95_different": p95,
        "same_speaker": d_same,
        "different_speaker": d_diff,
        "pairs": [{"a": a, "b": b, "srs_mean": statistics.fmean(v), "n": len(v)}
                  for (a, b), v in pair_scores.items()],
        "distributions_overlap": overlap,
        "compared": compared,
        "caveat": ("XTTS 합성 화자 기반 잠정값. 사람 목소리로 재측정 필요. "
                   "비교 대상은 전체 길이라 조각 기준선보다 유사도가 높게 나오는 편향이 있다."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
