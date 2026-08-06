"""채널 인지 최적화 A/B — 통화 경로에서 방어가 살아남는가.

`codec_test.py`가 문제를 보였다. 섭동은 **전대역 원본**의 임베딩에서 멀어지도록
최적화되는데, 상대방이 듣는 것은 통화 채널을 통과한 신호다. 채널이 원본을 자르면
그 임베딩이 이동하고 최적화한 방향은 더 이상 적대적이지 않다.

    무처리 SRS 0.6342 → G.711 협대역 통과 후 0.8346 (판정 임계값 위로 복귀)

`PGDConfig.channel_aware`가 손실을 `encoder(channel(x+δ))` 대 `encoder(channel(x))`로
바꾼다. **원리가 맞아 보인다고 채택하면 안 되므로** 같은 시드·같은 설정에서
켬/끔을 나란히 돌려 실제 채널로 검증한다.

검증은 반드시 **진짜 코덱**(`mirinae.codec`)으로 한다.
최적화에 쓴 미분 가능한 근사로 검증하면 아무것도 증명하지 못한다.

사용:
    python channel_ab.py out/reference.wav --repeat 5 --steps 50
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from clone_test import mean_ci
from mirinae.codec import CHANNELS, correlation, surviving_ratio, telephone_channel
from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
from mirinae.controls import bandlimited_noise, white_noise
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.perturbation import pgd_perturbation
from mirinae.pipeline import ProtectionResult
from mirinae.psychoacoustic import MaskingModel
from mirinae.vad import speech_mask

THRESHOLD = ProtectionResult.PROVISIONAL_THRESHOLD


def main() -> int:
    p = argparse.ArgumentParser(description="채널 인지 최적화 A/B")
    p.add_argument("input", nargs="?", default="out/reference.wav")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--offset-sec", type=float, default=1.0)
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--channel", default="ulaw", choices=list(CHANNELS))
    p.add_argument("-o", "--out", default="out/channel_ab.json")
    args = p.parse_args()

    device = default_device()
    from protect import load_wav
    x = load_wav(Path(args.input))
    s = int(SAMPLE_RATE * args.offset_sec)
    x = x[s:s + int(SAMPLE_RATE * args.seconds)].to(device)

    encoder = SpeakerEncoder(device=device)
    model = MaskingModel(device=device)
    mask = speech_mask(x)
    chan = CHANNELS[args.channel]

    # 채널을 통과한 원본이 **평가 기준**이다. 상대방은 이것을 듣는다.
    x_ch = telephone_channel(x, chan)
    with torch.no_grad():
        ref_clean = encoder(x)
        ref_ch = encoder(x_ch)
        self_srs = float(cosine_similarity(ref_ch, ref_clean))

    print(f"입력: {args.input} · {args.seconds}초 · 배율 {args.ratio} · {args.steps}스텝")
    print(f"채널: {chan.name} [{chan.describe()}] · 조건당 {args.repeat}회")
    print(f"채널 자체 손실 — 원본↔채널통과원본 SRS {self_srs:.4f}")
    print(f"판정 임계값 {THRESHOLD}\n")

    # ── 대조군도 같은 채널로 평가한다 ─────────────────────────────────────────
    base: dict[str, list[float]] = {"C-A 백색잡음": [], "C-B 대역제한잡음": []}
    for k in range(args.repeat):
        g = torch.Generator(device="cpu").manual_seed(1000 + k)
        for label, fn in (("C-A 백색잡음", white_noise),
                          ("C-B 대역제한잡음", bandlimited_noise)):
            y = x + fn(x.cpu(), 20.0, g).to(device)
            with torch.no_grad():
                base[label].append(
                    float(cosine_similarity(encoder(telephone_channel(y, chan)), ref_ch)))

    print("대조군 (채널 통과 후)")
    for label, vals in base.items():
        m, sd, half = mean_ci(vals)
        print(f"  {label:<18}SRS {m:.4f} ±{half:.4f}")
    beat = min(statistics.fmean(v) for v in base.values())
    print(f"  → 이겨야 할 값: {beat:.4f}\n")

    # ── A/B ───────────────────────────────────────────────────────────────────
    rows = []
    print(f"{'설정':<14}{'채널 후 SRS':>13}{'95%CI':>10}{'판정':>10}"
          f"{'채널 전':>10}{'잔존%':>8}{'상관':>8}")
    print("─" * 76)

    for aware in (False, True):
        cfg = PGDConfig(steps=args.steps, masking_ratio=args.ratio,
                        channel_aware=aware)
        after, before, keeps, corrs = [], [], [], []
        for k in range(args.repeat):
            r = pgd_perturbation(x, encoder, cfg, masking_model=model,
                                 vad_mask=mask, seed=k)
            y_ch = telephone_channel(r.protected, chan)
            d_ch = y_ch - x_ch
            with torch.no_grad():
                after.append(float(cosine_similarity(encoder(y_ch), ref_ch)))
                before.append(float(cosine_similarity(encoder(r.protected), ref_clean)))
            keeps.append(surviving_ratio(r.delta, d_ch))
            corrs.append(correlation(r.delta, d_ch))

        m, sd, half = mean_ci(after)
        verdict = "타 화자" if m < THRESHOLD else "같은 화자"
        label = "채널 인지 켬" if aware else "채널 인지 끔"
        print(f"{label:<14}{m:>13.4f}{f'±{half:.4f}':>10}{verdict:>10}"
              f"{statistics.fmean(before):>10.4f}"
              f"{statistics.fmean(keeps) * 100:>8.1f}{statistics.fmean(corrs):>8.3f}")
        rows.append({
            "channel_aware": aware, "n": args.repeat,
            "srs_after_channel_mean": m, "srs_after_channel_ci_half": half,
            "srs_after_channel_samples": after,
            "srs_before_channel_mean": statistics.fmean(before),
            "surviving_ratio_mean": statistics.fmean(keeps),
            "delta_correlation_mean": statistics.fmean(corrs),
            "beats_control": m < beat,
            "below_threshold": m < THRESHOLD,
        })

    off, on = rows[0], rows[1]
    gain = off["srs_after_channel_mean"] - on["srs_after_channel_mean"]
    print()
    print(f"채널 인지로 얻은 것: SRS {gain:+.4f} "
          f"({off['srs_after_channel_mean']:.4f} → {on['srs_after_channel_mean']:.4f})")
    sep = (off["srs_after_channel_mean"] - off["srs_after_channel_ci_half"]) > \
          (on["srs_after_channel_mean"] + on["srs_after_channel_ci_half"])
    if not sep:
        print("  ※ 신뢰구간이 겹친다 — **개선 미확인**. n을 늘려야 한다.")
    elif on["below_threshold"]:
        print("  신뢰구간이 분리되고 임계값 아래다 — 통화 경로에서 방어가 살아남는다.")
    else:
        print("  유의하게 나아졌지만 여전히 임계값 위다 — 방향은 맞으나 부족하다.")
    if not on["beats_control"]:
        print(f"  주의: 채널 통과 후에도 대조군({beat:.4f})을 못 이긴다. "
              f"'그냥 잡음 아니냐'에 답할 수 없다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "input": args.input, "channel": args.channel, "ratio": args.ratio,
        "steps": args.steps, "threshold": THRESHOLD,
        "channel_self_srs": self_srs,
        "controls": {k: statistics.fmean(v) for k, v in base.items()},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
