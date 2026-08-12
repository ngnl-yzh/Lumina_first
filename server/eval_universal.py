"""사전 계산 섭동이 **다른 발화**에도 통하는가 — 실시간 보호의 관문.

## 왜 이걸 먼저 재나

실시간 보호는 "통화 중에 최적화한다"가 아니다. 그 방식은 이미 실패했다 —
청크마다 미는 방향이 제각각이라 복제기가 파일 전체를 볼 때 상쇄된다.

되는 방식은 **사전 계산 + 실시간 적용**이다.

    [등록 · 1회 · 오프라인]          [통화 중 · 실시간]
    사용자 목소리 녹음        →      δ 패턴을 마이크 입력에 더하기만
    δ 패턴 산출 (수 분)              덧셈 한 번 — GPU도 필요 없다

그런데 여기엔 검증되지 않은 가정이 하나 있다.
**등록 때 읽은 문장과 통화 내용이 다르다.** δ는 등록 발화에 맞춰 만들어졌는데,
전혀 다른 말을 할 때도 통할까?

**이 질문에 답하지 못하면 실시간은 성립하지 않는다.** 그래서 이걸 먼저 잰다.

## 어떻게 재나

한 화자의 긴 녹음을 둘로 자른다.

    앞부분(A)  — δ를 만드는 데 쓴다 (등록 단계에 해당)
    뒷부분(B)  — **한 번도 보지 않은 구간** (통화 내용에 해당)

A로 만든 δ를 B에 얹고, 복제기 내부 화자 조건이 B에서도 밀리는지 본다.
길이가 다르면 δ를 이어 붙여 채운다 — 실시간에서도 그렇게 쓰게 된다.

비교 기준 셋을 함께 잰다.

    B 원본            아무것도 안 한 것
    B + δ(A)         **사전 계산 섭동** — 이게 통해야 실시간이 된다
    B + δ(B)         B로 직접 만든 δ. 상한선이다
    B + 잡음         같은 세기의 잡음. 하한선이다 — 여기보다 나아야 의미가 있다
"""

from __future__ import annotations

import argparse
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


def tile_to(delta: torch.Tensor, n: int) -> torch.Tensor:
    """δ를 목표 길이만큼 이어 붙인다.

    실시간에서는 δ가 고정 길이 패턴이고 통화는 길이가 정해져 있지 않다.
    그래서 되풀이해 쓰게 된다. 여기서도 같은 방식으로 만들어 재야
    측정이 실사용과 어긋나지 않는다.
    """
    if len(delta) >= n:
        return delta[:n]
    reps = int(np.ceil(n / len(delta)))
    return delta.repeat(reps)[:n]


def snr_match(delta: torch.Tensor, x: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig = float(torch.mean(x ** 2))
    target = sig / (10.0 ** (snr_db / 10.0))
    cur = float(torch.mean(delta ** 2))
    return delta if cur <= 1e-12 else delta * float((target / cur) ** 0.5)


def main() -> int:
    p = argparse.ArgumentParser(description="사전 계산 섭동의 발화 간 전이 측정")
    p.add_argument("input", help="한 화자의 긴 녹음")
    p.add_argument("--target", default="xtts,gsv")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--split", type=float, default=8.0, help="A/B 각각의 길이(초)")
    args = p.parse_args()

    from attack_cloner import TARGETS, MultiTarget, attack, _cos

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load(args.input)
    n = int(SAMPLE_RATE * args.split)
    if len(x) < 2 * n:
        raise SystemExit(f"녹음이 짧다 — {2 * args.split:.0f}초 이상 필요")
    A = torch.from_numpy(np.ascontiguousarray(x[:n]))
    B = torch.from_numpy(np.ascontiguousarray(x[n:2 * n]))
    print(f"입력 {len(x) / SAMPLE_RATE:.1f}초 → A(등록) {args.split:.0f}초 · "
          f"B(미지) {args.split:.0f}초\n")

    names = [t.strip() for t in args.target.split(",") if t.strip()]
    built = [TARGETS[t](device) for t in names]
    tgt = built[0] if len(built) == 1 else MultiTarget(built)
    multi = len(built) > 1
    cond = tgt.all_conditioning if multi else (lambda w: [tgt.conditioning(w)])

    def shift(base: torch.Tensor, test: torch.Tensor) -> list[float]:
        with torch.no_grad():
            r, c = cond(base), cond(test)
        return [float(_cos(ci.speaker, ri.speaker)) for ci, ri in zip(c, r)]

    print(f"A로 δ 산출 중 ({args.steps}스텝)...")
    protA, _ = attack(A, tgt, steps=args.steps, snr_db=args.snr,
                      prosody_weight=2.0, masking_ratio=3.0, progress=False)
    dA = protA - A

    print(f"B로 δ 산출 중 (상한선 비교용)...")
    protB, _ = attack(B, tgt, steps=args.steps, snr_db=args.snr,
                      prosody_weight=2.0, masking_ratio=3.0, progress=False)

    rng = np.random.default_rng(0)
    noise = torch.from_numpy(rng.standard_normal(len(B)).astype(np.float32))
    conds = {
        "B 원본": B,
        "B + δ(A)  ← 사전 계산": B + snr_match(tile_to(dA, len(B)), B, args.snr),
        "B + δ(B)  ← 상한선": protB,
        "B + 잡음   ← 하한선": B + snr_match(noise, B, args.snr),
    }

    head = "".join(f"{t.name:>14}" for t in built)
    print(f"\n{'조건':26}{head}")
    print("-" * (26 + 14 * len(built)))
    for label, w in conds.items():
        vals = shift(B, torch.clamp(w, -1.0, 1.0))
        print(f"{label:26}" + "".join(f"{v:14.4f}" for v in vals))
    print("-" * (26 + 14 * len(built)))
    print("  복제기 내부 화자 조건 — 낮을수록(음수면 더욱) 다른 화자로 인식된다.")
    print("  **δ(A)가 잡음보다 확실히 낮아야** 사전 계산이 성립하고 실시간이 가능하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
