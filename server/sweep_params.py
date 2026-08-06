"""파라미터 스윕 — 마스킹 배율과 alpha를 대조군 기준으로 평가한다.

D07 W1의 "마스킹 배율 확정" 작업에 쓰는 도구다.

핵심은 **절대 SRS가 아니라 대조군 대비**다.
같은 SNR의 백색잡음·대역제한잡음보다 낫지 않으면 적대적 최적화를 했다고 주장할 수 없다.
그래서 매 실행에 C-A·C-B 기준선을 함께 찍는다.

## 반복 측정이 필요한 이유

이 스윕은 한때 조건당 **1회**만 쟀다. 그 수치로 마스킹 배율 3.0을 잠정 확정했다.
그런데 PGD는 δ를 랜덤 초기화하고 대조군 잡음도 난수다 — **같은 설정이 같은 값을 내지 않는다.**
`clone_test.py`가 복제 단계에서 이미 같은 함정을 확인했다:
같은 파일을 두 번 복제했더니 0.019가 벌어졌는데, 보호 효과는 0.027~0.047이었다.
효과와 흔들림이 같은 자릿수면 n=1 비교는 측정이 아니라 착시다.

그래서 조건마다 시드를 바꿔 여러 번 돌리고 **평균과 95% 신뢰구간**을 낸다.
대조군 우위 판정도 점 추정 비교가 아니라 **신뢰구간이 겹치지 않는가**로 한다.
겹치면 "우위 미확인"이지 "우위 없음"이 아니다 — n을 늘려야 한다는 뜻이다.

사용:
    python sweep_params.py 목소리.wav
    python sweep_params.py 목소리.wav --repeat 8 --steps 200 --ratios 1,3,10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from clone_test import mean_ci
from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
from mirinae.controls import bandlimited_noise, white_noise
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.metrics import audibility
from mirinae.perturbation import pgd_perturbation
from mirinae.psychoacoustic import MaskingModel
from mirinae.vad import speech_mask


def parse_list(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def separated(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """두 (평균, sd, ci반폭)의 95% 신뢰구간이 겹치지 않는가.

    겹치면 "차이 없음"이 아니라 **"이 표본 수로는 구별 못 함"**이다.
    둘을 같은 말로 쓰면 n이 부족한 실험이 음성 결과로 둔갑한다.
    """
    if math.isnan(a[2]) or math.isnan(b[2]):
        return False
    return (a[0] + a[2]) < (b[0] - b[2]) or (b[0] + b[2]) < (a[0] - a[2])


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
    p.add_argument("--repeat", type=int, default=5,
                   help="조건당 반복 횟수. 1로 두면 신뢰구간을 낼 수 없다")
    p.add_argument("--offset-sec", type=float, default=0.0,
                   help="입력에서 잘라낼 시작 위치. 무음 구간을 피할 때 쓴다")
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

    start = int(SAMPLE_RATE * args.offset_sec)
    n = int(SAMPLE_RATE * args.seconds)
    x = x[start:start + n].to(device)

    encoder = SpeakerEncoder(device=device)
    model = MaskingModel(device=device)
    thr, _ = model.threshold(x)
    mask = speech_mask(x)

    with torch.no_grad():
        ref = encoder(x)

    R = max(1, args.repeat)

    print(f"입력: {source} · {args.seconds}초 · 목표 SNR {args.snr} dB · {args.steps}스텝")
    print(f"장치: {device} · 조건당 {R}회 반복\n")
    if R == 1:
        print("※ --repeat 1이다. 신뢰구간을 낼 수 없어 대조군 비교가 무의미하다.\n")

    # ── 기준선 — 이걸 못 이기면 아무 주장도 할 수 없다 ────────────────────────
    # 대조군 잡음도 난수다. 한 번만 뽑으면 그 한 번이 유난히 세거나 약할 수 있다.
    baselines: dict[str, dict] = {}
    for label, fn in (("C-A 백색잡음", white_noise), ("C-B 대역제한잡음", bandlimited_noise)):
        vals = []
        for k in range(R):
            g = torch.Generator(device="cpu").manual_seed(1000 + k)
            noise = fn(x.cpu(), args.snr, g).to(device)
            with torch.no_grad():
                vals.append(float(cosine_similarity(encoder(x + noise), ref)))
        m, sd, half = mean_ci(vals)
        baselines[label] = {"n": R, "mean": m, "sd": sd, "ci_half": half,
                            "samples": vals}

    print("기준선 (대조군)")
    for label, s in baselines.items():
        ci = "—" if math.isnan(s["ci_half"]) else f"±{s['ci_half']:.4f}"
        print(f"  {label:<20}SRS {s['mean']:.4f} {ci}  (n={s['n']})")

    # 가장 강한 대조군을 기준으로 삼는다. 약한 쪽을 이겨봐야 의미가 없다.
    best_label = min(baselines, key=lambda k: baselines[k]["mean"])
    bstat = baselines[best_label]
    b_tuple = (bstat["mean"], bstat["sd"], bstat["ci_half"])
    print(f"  → 이겨야 할 대조군: {best_label} {bstat['mean']:.4f}\n")

    # ── 스윕 ──────────────────────────────────────────────────────────────────
    # 가청도를 두 가지로 낸다. 하나만 보면 정반대로 읽힌다 — 아래 설명 참조.
    print(f"{'배율':>7}{'alpha':>9}{'SRS 평균':>10}{'95%CI':>9}{'대조군대비':>12}"
          f"{'절대초과dB':>11}{'절대위반%':>10}{'제약초과dB':>11}{'초/회':>7}")
    print("─" * 95)

    rows = []
    for ratio in args.ratios:
        for alpha in args.alphas:
            cfg = PGDConfig(steps=args.steps, masking_ratio=ratio,
                            alpha=alpha, target_snr_db=args.snr)
            srs_vals, excess_vals, viol_vals, snr_vals = [], [], [], []
            abs_excess_vals, abs_viol_vals = [], []
            t0 = time.perf_counter()
            for k in range(R):
                # 시드를 바꿔야 δ 랜덤 초기화가 달라진다. 안 바꾸면 R번 같은 값이 나온다.
                r = pgd_perturbation(x, encoder, cfg, masking_model=model,
                                     vad_mask=mask, seed=k)
                # 자기 제약(thr × ratio) 대비 — "설정한 제약을 지켰는가"
                aud = audibility(r.delta, thr, ratio, model)
                # 절대 마스킹 임계값(ratio=1) 대비 — **배율 간 비교는 이것으로만 가능하다**
                aud_abs = audibility(r.delta, thr, 1.0, model)
                srs_vals.append(r.srs)
                excess_vals.append(aud.max_excess_db)
                viol_vals.append(aud.violation_ratio)
                abs_excess_vals.append(aud_abs.max_excess_db)
                abs_viol_vals.append(aud_abs.violation_ratio)
                snr_vals.append(r.snr_db)
            dt = (time.perf_counter() - t0) / R

            stat = mean_ci(srs_vals)
            sep = separated(stat, b_tuple)
            better = stat[0] < bstat["mean"]
            # 점 추정이 낮아도 신뢰구간이 겹치면 "우위 확인"이라고 쓰지 않는다.
            mark = ("✓" if better else "✗") if sep else "?"
            ci = "—" if math.isnan(stat[2]) else f"±{stat[2]:.4f}"
            delta = f"{mark} {bstat['mean'] - stat[0]:+.3f}"

            print(f"{ratio:>7.2f}{alpha:>9.0e}{stat[0]:>10.4f}{ci:>9}"
                  f"{delta:>12}"
                  f"{statistics.fmean(abs_excess_vals):>11.1f}"
                  f"{statistics.fmean(abs_viol_vals) * 100:>10.2f}"
                  f"{statistics.fmean(excess_vals):>11.1f}{dt:>7.1f}")

            rows.append({
                "masking_ratio": ratio, "alpha": alpha, "n": R,
                "srs_mean": stat[0], "srs_sd": stat[1], "srs_ci_half": stat[2],
                "srs_samples": srs_vals,
                "beats_baseline": bool(sep and better),
                "ci_separated": bool(sep),
                "snr_db_mean": statistics.fmean(snr_vals),
                # 절대 = 마스킹 임계값(ratio 1) 대비. 배율 간 비교는 이것만 유효하다.
                "abs_max_excess_db_mean": statistics.fmean(abs_excess_vals),
                "abs_violation_ratio_mean": statistics.fmean(abs_viol_vals),
                # 제약 = thr × ratio 대비. 설정한 제약을 지켰는지만 말해준다.
                "constraint_max_excess_db_mean": statistics.fmean(excess_vals),
                "constraint_violation_ratio_mean": statistics.fmean(viol_vals),
                "elapsed_sec_per_run": dt,
            })

    print("\n  ✓ 대조군보다 유의하게 낮음 (신뢰구간 분리)")
    print("  ? 점 추정은 낮지만 신뢰구간이 겹친다 — **우위 미확인**. n을 늘려야 한다")
    print("  ✗ 대조군보다 높다")
    print()
    print("  가청도를 두 가지로 낸다. 섞어 읽으면 결론이 정반대가 된다.")
    print("    절대초과dB  — 마스킹 임계값(배율 1) 대비. **배율 간 비교는 이것만 유효하다**")
    print("    제약초과dB  — thr × 그 배율 대비. 설정한 제약을 지켰는지만 말해준다")
    print("  제약 쪽은 배율이 커지면 기준선도 같이 커져서 초과량이 **줄어든다.**")
    print("  그것만 보면 '배율을 키울수록 덜 들린다'는 정반대 결론이 나온다.\n")

    winners = [r for r in rows if r["beats_baseline"]]
    unclear = [r for r in rows if r["ci_separated"] is False]
    if winners:
        best = min(winners, key=lambda r: r["srs_mean"])
        print(f"최저 SRS: 배율 {best['masking_ratio']} · alpha {best['alpha']:.0e} "
              f"→ {best['srs_mean']:.4f} ±{best['srs_ci_half']:.4f} "
              f"(마스킹 임계값 대비 최대 초과 {best['abs_max_excess_db_mean']:.1f} dB · "
              f"위반 {best['abs_violation_ratio_mean']*100:.1f}%)")
        print("가청도 초과가 클수록 들릴 위험이 커진다. 청취 평가로 상한을 정한 뒤 역산할 것.")
    else:
        print("대조군을 유의하게 이긴 설정이 없다. "
              "이 상태로는 '적대적 섭동의 우위'를 주장할 수 없다.")
    if unclear:
        print(f"판정 보류 {len(unclear)}건 — 신뢰구간이 겹쳐 구별되지 않았다. "
              f"n={R}으로는 부족하다는 뜻이다.")

    Path(args.out).write_text(json.dumps({
        "source": source, "steps": args.steps, "target_snr_db": args.snr,
        "repeat": R, "baselines": baselines, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
