"""스텝 수 대 방어 강도 — GPU가 정말 필요한가에 답하기 위한 측정.

## 답해야 할 질문

설계도는 200스텝을 상정하고 "CPU 51초 · RTX 3060 0.36초"를 적어 두었다.
그래서 "GPU가 있어야 한다"로 읽힌다. 그런데 그 문장에는 빠진 것이 있다 —
**스텝을 줄이면 방어가 얼마나 약해지는가.**

`worker.py`는 큐가 밀리면 스텝을 자동으로 줄인다(`BACKLOG_LADDER`).
즉 **스텝 감축은 이미 정상 동작 경로**인데, 감축했을 때 무엇을 잃는지 잰 적이 없다.
"감축 12개"라고 보고는 하지만 그게 방어가 반토막 났다는 뜻인지
거의 그대로라는 뜻인지 알 수 없었다.

## 무엇을 재는가

스텝별로 **SRS(방어 강도)와 실제 소요 시간**을 함께 낸다.
실시간 판정 기준은 hop 1.0초다 — 1초마다 새 청크가 오므로
청크 하나를 1초 안에 끝내야 큐가 밀리지 않는다.

대조군(C-B 대역제한잡음)도 같이 찍는다. 그걸 못 이기면 스텝이 몇이든 의미가 없다.

사용:
    python steps_budget.py out/reference.wav --repeat 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from clone_test import mean_ci
from mirinae.codec import CHANNELS, telephone_channel
from mirinae.config import HOP_SEC, PGDConfig, SAMPLE_RATE, default_device
from mirinae.controls import bandlimited_noise
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.perturbation import pgd_perturbation
from mirinae.pipeline import ProtectionResult
from mirinae.psychoacoustic import MaskingModel
from mirinae.vad import speech_mask

THRESHOLD = ProtectionResult.PROVISIONAL_THRESHOLD


def main() -> int:
    p = argparse.ArgumentParser(description="스텝 수 대 방어 강도 · 실시간 여유")
    p.add_argument("input", nargs="?", default="out/reference.wav")
    p.add_argument("--steps", type=lambda s: [int(v) for v in s.split(",")],
                   default=[5, 10, 20, 40, 80, 160, 200])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--offset-sec", type=float, default=1.0)
    p.add_argument("-o", "--out", default="out/steps_budget.json")
    args = p.parse_args()

    device = default_device()
    from protect import load_wav
    x = load_wav(Path(args.input))
    s = int(SAMPLE_RATE * args.offset_sec)
    x = x[s:s + int(SAMPLE_RATE * args.seconds)].to(device)

    encoder = SpeakerEncoder(device=device)
    model = MaskingModel(device=device)
    mask = speech_mask(x)
    chan = CHANNELS["ulaw"]

    # 평가는 통화 채널 통과 후로 한다 — 최적화 목표와 같은 기준이어야 한다.
    x_ch = telephone_channel(x, chan)
    with torch.no_grad():
        ref = encoder(x_ch)

    base = []
    for k in range(args.repeat):
        g = torch.Generator(device="cpu").manual_seed(1000 + k)
        y = x + bandlimited_noise(x.cpu(), 20.0, g).to(device)
        with torch.no_grad():
            base.append(float(cosine_similarity(encoder(telephone_channel(y, chan)), ref)))
    bmean, bsd, bhalf = mean_ci(base)

    print(f"입력: {args.input} · {args.seconds}초 · 배율 {args.ratio} · 장치 {device}")
    print(f"평가: 통화 채널 통과 후 · 판정 임계값 {THRESHOLD} · 조건당 {args.repeat}회")
    print(f"대조군 C-B {bmean:.4f} ±{bhalf:.4f} — 이걸 못 이기면 스텝이 몇이든 의미 없다")
    print(f"실시간 기준: hop {HOP_SEC}초 — 청크 하나를 {HOP_SEC}초 안에 끝내야 큐가 안 밀린다\n")

    print(f"{'스텝':>5}{'SRS':>9}{'95%CI':>9}{'판정':>9}{'대조군':>8}"
          f"{'초/청크':>9}{'실시간':>8}")
    print("─" * 60)

    rows = []
    for steps in args.steps:
        cfg = PGDConfig(steps=steps, masking_ratio=args.ratio)
        vals, times = [], []
        for k in range(args.repeat):
            t0 = time.perf_counter()
            r = pgd_perturbation(x, encoder, cfg, masking_model=model,
                                 vad_mask=mask, seed=k)
            times.append(time.perf_counter() - t0)
            with torch.no_grad():
                vals.append(float(cosine_similarity(
                    encoder(telephone_channel(r.protected, chan)), ref)))

        m, sd, half = mean_ci(vals)
        dt = statistics.fmean(times)
        verdict = "타 화자" if m < THRESHOLD else "같은 화자"

        # 대조군 우위는 **신뢰구간 분리**로 판정한다. 점 추정 비교는 측정이 아니다 —
        # `sweep_params.py`에 적용한 규칙을 여기에도 똑같이 건다.
        sep = (m + half) < (bmean - bhalf)
        beats = ("✓" if sep else ("?" if m < bmean else "✗"))
        rt = "OK" if dt <= HOP_SEC else f"×{dt / HOP_SEC:.0f}"
        print(f"{steps:>5}{m:>9.4f}{f'±{half:.4f}':>9}{verdict:>9}{beats:>8}"
              f"{dt:>9.2f}{rt:>8}")
        rows.append({"steps": steps, "srs_mean": m, "srs_ci_half": half,
                     "sec_per_chunk": dt, "below_threshold": m < THRESHOLD,
                     "beats_control": bool(sep), "ci_separated": bool(sep),
                     "realtime_ok": dt <= HOP_SEC})

    print()
    print("  ✓ 대조군보다 유의하게 낮음 (신뢰구간 분리)")
    print("  ? 점 추정은 낮지만 구간이 겹친다 — **우위 미확인**")
    print("  ✗ 대조군보다 높다\n")

    usable = [r for r in rows if r["below_threshold"] and r["beats_control"]]
    if usable:
        cheapest = min(usable, key=lambda r: r["steps"])
        print(f"방어가 성립하는 최소 스텝: **{cheapest['steps']}스텝** "
              f"(SRS {cheapest['srs_mean']:.4f} · {cheapest['sec_per_chunk']:.2f}초/청크)")
        if cheapest["realtime_ok"]:
            print(f"  → 이 장치에서 **실시간 가능**하다. GPU 없이 된다.")
        else:
            need = cheapest["sec_per_chunk"] / HOP_SEC
            print(f"  → 실시간에는 **{need:.0f}배** 부족하다. "
                  f"오프라인 보호는 되지만 통화 중 실시간 주입은 안 된다.")
    else:
        print("어느 스텝에서도 대조군을 이기면서 임계값 아래로 못 간다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "device": str(device), "ratio": args.ratio, "hop_sec": HOP_SEC,
        "control_cb": bmean, "threshold": THRESHOLD, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
