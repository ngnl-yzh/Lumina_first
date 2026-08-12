"""준실시간 보호 — 청크로 나눠 최적화해도 살아남는가.

## 왜 이걸 재나

실시간 보호의 유일하게 남은 길이다. 사전 계산 δ를 그냥 더하는 방식은
측정으로 부정됐다(`eval_universal.py` — 다른 발화에서 0.9075, 잡음 0.7720보다 나쁘다).

남은 방식은 **짧은 지연을 받아들이고 그 구간을 즉석 최적화**하는 것이다.
1~2초 버퍼를 두고 채워지는 대로 보호해 내보낸다.

여기엔 알려진 함정이 하나 있다. 예전에 **화자 검증기**를 표적으로 이걸 했다가
실패했다 — 청크마다 자기 임베딩에서 멀어지므로 미는 방향이 제각각이고,
복제기가 파일 전체를 볼 때 서로 상쇄된다(SRS 0.8045, 임계값 위).

**복제기를 표적으로 하면 다를까?** 재본 적이 없다. 그게 이 스크립트다.

## 상쇄를 막는 방법도 함께 잰다

방향이 제각각인 것이 원인이라면, **방향을 하나로 고정**하면 된다.
첫 청크에서 목표 방향을 한 번 정하고 이후 청크는 전부 그쪽으로 민다.

    독립  각 청크가 자기 조건에서 멀어진다        → 방향이 제각각
    공유  모든 청크가 **같은 목표 방향**으로 간다  → 더해진다

## 꼬리 처리

녹음을 멈추면 마지막 구간이 청크 길이에 못 미친다. 그것도 보호해야 한다 —
사용자가 마지막에 한 말이 가장 중요할 수 있다. 짧아도 그대로 최적화한다.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


def load(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


def chunks_of(x: torch.Tensor, n: int, min_tail: int) -> list[torch.Tensor]:
    """청크로 자른다. **꼬리를 버리지 않는다.**

    녹음을 멈추면 마지막 구간이 청크 길이에 못 미친다. 사용자가 마지막에
    한 말이 가장 중요할 수 있으므로 짧아도 그대로 최적화한다.
    다만 너무 짧으면(min_tail 미만) 앞 청크에 붙여 함께 처리한다 —
    0.2초짜리를 따로 최적화해 봐야 의미가 없다.
    """
    out = []
    i = 0
    while i < len(x):
        piece = x[i:i + n]
        i += n
        if len(piece) < min_tail and out:
            out[-1] = torch.cat([out[-1], piece])
        else:
            out.append(piece)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="준실시간 청크 보호 측정")
    p.add_argument("input")
    p.add_argument("--target", default="xtts,gsv")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--chunk", type=float, default=2.0, help="버퍼 길이(초) = 지연")
    args = p.parse_args()

    from attack_cloner import TARGETS, MultiTarget, attack, _cos

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(load(args.input)[: int(SAMPLE_RATE * args.seconds)])
    names = [t.strip() for t in args.target.split(",") if t.strip()]
    built = [TARGETS[t](device) for t in names]
    tgt = built[0] if len(built) == 1 else MultiTarget(built)
    multi = len(built) > 1
    cond = tgt.all_conditioning if multi else (lambda w: [tgt.conditioning(w)])

    with torch.no_grad():
        ref_full = cond(x)

    def score(w: torch.Tensor) -> list[float]:
        with torch.no_grad():
            c = cond(torch.clamp(w, -1.0, 1.0))
        return [float(_cos(ci.speaker, ri.speaker)) for ci, ri in zip(c, ref_full)]

    n = int(SAMPLE_RATE * args.chunk)
    pieces = chunks_of(x, n, min_tail=int(SAMPLE_RATE * 0.5))
    lens = " · ".join(f"{len(p) / SAMPLE_RATE:.1f}초" for p in pieces)
    print(f"입력 {len(x) / SAMPLE_RATE:.1f}초 → 청크 {len(pieces)}개 ({lens})")
    print(f"표적 {args.target} · 청크당 {args.steps}스텝 · 버퍼 {args.chunk}초\n")

    results = {}

    # ── ① 전체 발화 일괄 (기준선) ────────────────────────────────────────────
    t0 = time.perf_counter()
    whole, _ = attack(x, tgt, steps=args.steps * len(pieces), snr_db=args.snr,
                      prosody_weight=2.0, masking_ratio=3.0, progress=False)
    results["전체 일괄 (기준선)"] = (score(whole), time.perf_counter() - t0)

    # ── ② 청크 독립 최적화 ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    out = []
    for piece in pieces:
        pr, _ = attack(piece, tgt, steps=args.steps, snr_db=args.snr,
                       prosody_weight=2.0, masking_ratio=3.0, progress=False)
        out.append(pr)
    results["청크 독립"] = (score(torch.cat(out)), time.perf_counter() - t0)

    # ── ③ 청크 + 공유 방향 ───────────────────────────────────────────────────
    # 첫 청크가 찾은 δ의 **방향**을 이후 청크의 출발점으로 쓴다.
    # 방향이 제각각이라 상쇄되는 것이 문제였으므로, 같은 골짜기에서 출발시킨다.
    t0 = time.perf_counter()
    out = []
    seed = None
    for piece in pieces:
        pr, _ = attack(piece, tgt, steps=args.steps, snr_db=args.snr,
                       prosody_weight=2.0, masking_ratio=3.0, progress=False,
                       init_delta=None if seed is None else seed[:len(piece)])
        out.append(pr)
        if seed is None:
            d = pr - piece
            seed = d.repeat(int(np.ceil(len(x) / len(d))))[:len(x)]
    results["청크 + 공유 방향"] = (score(torch.cat(out)), time.perf_counter() - t0)

    head = "".join(f"{t.name:>14}" for t in built)
    print(f"{'방식':22}{head}{'소요':>10}")
    print("-" * (22 + 14 * len(built) + 10))
    for label, (vals, sec) in results.items():
        print(f"{label:22}" + "".join(f"{v:14.4f}" for v in vals) + f"{sec:9.0f}초")
    print("-" * (22 + 14 * len(built) + 10))
    print("  복제기 내부 화자 조건 — 낮을수록 다른 화자로 인식된다.")
    print("  청크 방식이 전체 일괄에 가까워야 준실시간이 성립한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
