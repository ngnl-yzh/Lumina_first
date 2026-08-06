"""통화 채널 통과 후 방어가 살아남는가 — 모드 2의 가장 중요한 미검증 전제.

지금까지의 모든 SRS는 **파일 대 파일**로 잰 값이다.
실제 경로에는 통화 채널이 끼어 있고, `README.md`는 "협대역 8 kHz에서 최대 33%p 손실"을
**예상으로만** 적어 두었다. 측정된 적이 없다.

여기가 무너지면 모드 2는 실사용에서 무의미하다. 그리고 무너질 이유가 구조적이다 —
모드 2는 심리음향 마스킹 **아래**로 섭동을 숨기는데, 음성 코덱은 정확히
"안 들리는 성분"을 버리도록 설계되어 있다. **숨긴 원리와 버리는 원리가 같다.**

## 무엇을 비교하는가

채널을 단계별로 켜 가며 **어디서** 섭동이 죽는지 분리한다.
그리고 매번 대조군(C-A/C-B)도 같은 채널에 통과시킨다 —
채널이 섭동만 지우는지, 잡음도 똑같이 지우는지 구별해야 하기 때문이다.

지표를 세 가지로 낸다. 하나만으로는 무슨 일이 일어났는지 알 수 없다.

| 지표 | 답하는 질문 |
|---|---|
| SRS | 방어가 아직 통하는가 |
| 잔존 에너지비 | 섭동이 물리적으로 남아 있는가 |
| 상관계수 | 섭동의 **구조**가 살아남았는가 |

에너지가 남아도 구조가 깨지면 적대적 효과는 사라진다 — C-E(셔플) 대조군이 보여준 그대로다.

사용:
    python codec_test.py out/ref
    python codec_test.py out/ref --repeat 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from clone_test import mean_ci
from mirinae.codec import CHANNELS, correlation, surviving_ratio, telephone_channel
from mirinae.config import SAMPLE_RATE, default_device
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.pipeline import ProtectionResult

# C-D 대조군에서 얻은 판정 임계값 (calibrate_dsr.py)
THRESHOLD = ProtectionResult.PROVISIONAL_THRESHOLD


def load(path: Path) -> torch.Tensor:
    from protect import load_wav
    return load_wav(path)


def main() -> int:
    p = argparse.ArgumentParser(description="통화 채널 통과 후 방어 유효성")
    p.add_argument("folder", nargs="?", default="out/ref",
                   help="protect.py 산출 폴더 (original/protected/control_*.wav)")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="앞에서 이만큼만 쓴다 (0이면 전체)")
    p.add_argument("-o", "--out", default="out/codec_test.json")
    args = p.parse_args()

    folder = Path(args.folder)
    orig_path = folder / "original.wav"
    if not orig_path.exists():
        print(f"{orig_path} 가 없다. 먼저 protect.py를 돌릴 것.")
        return 1

    device = default_device()
    encoder = SpeakerEncoder(device=device)

    def cut(x: torch.Tensor) -> torch.Tensor:
        if args.seconds > 0:
            return x[: int(SAMPLE_RATE * args.seconds)]
        return x

    x = cut(load(orig_path)).to(device)
    with torch.no_grad():
        ref = encoder(x)

    # 비교 대상 — 보호본과 대조군을 **같은 채널에** 통과시킨다.
    targets: dict[str, torch.Tensor] = {}
    for label, fname in (("보호본", "protected.wav"),
                         ("C-A 백색잡음", "control_C-A.wav"),
                         ("C-B 대역제한잡음", "control_C-B.wav"),
                         ("C-E 셔플 섭동", "control_C-E.wav")):
        fp = folder / fname
        if fp.exists():
            targets[label] = cut(load(fp)).to(device)

    if "보호본" not in targets:
        print(f"{folder}/protected.wav 가 없다.")
        return 1

    print(f"입력: {folder} · {len(x) / SAMPLE_RATE:.1f}초 · 장치 {device}")
    print(f"판정 임계값 {THRESHOLD} (C-D 대조군에서 측정)\n")

    rows = []
    for key, cfg in CHANNELS.items():
        print(f"── {cfg.name}  [{cfg.describe()}]")
        # 원본도 같은 채널을 통과시켜야 한다. 안 그러면 채널 자체가 만든 변화가
        # 방어 효과로 잘못 계상된다.
        x_ch = telephone_channel(x, cfg)
        with torch.no_grad():
            ref_ch = encoder(x_ch)
            sanity = float(cosine_similarity(ref_ch, ref))
        print(f"   원본↔채널통과원본 SRS {sanity:.4f}   ← 채널 자체가 만든 손실")

        for label, y in targets.items():
            delta = y - x
            y_ch = telephone_channel(y, cfg)
            delta_ch = y_ch - x_ch

            with torch.no_grad():
                srs = float(cosine_similarity(encoder(y_ch), ref_ch))
            keep = surviving_ratio(delta, delta_ch)
            corr = correlation(delta, delta_ch)
            verdict = "타 화자" if srs < THRESHOLD else "같은 화자"

            print(f"   {label:<16} SRS {srs:.4f} ({verdict:<5})  "
                  f"잔존 {keep * 100:5.1f}%  상관 {corr:.3f}")
            rows.append({
                "channel": key, "channel_name": cfg.name, "target": label,
                "srs": srs, "surviving_ratio": keep, "delta_correlation": corr,
                "below_threshold": srs < THRESHOLD,
                "channel_self_srs": sanity,
            })
        print()

    # ── 요약 — 채널 전후로 방어가 얼마나 무너졌나 ────────────────────────────
    base = {r["target"]: r for r in rows if r["channel"] == "none"}
    full = {r["target"]: r for r in rows if r["channel"] == "ulaw"}
    print("=" * 72)
    print("통화 채널 통과 전후 (무처리 → G.711 μ-law 협대역)")
    print("=" * 72)
    print(f"  {'대상':<16}{'SRS 전':>9}{'SRS 후':>9}{'변화':>9}   {'섭동 잔존':>10}{'구조 상관':>10}")
    print("  " + "-" * 66)
    for label in targets:
        b, f = base.get(label), full.get(label)
        if not (b and f):
            continue
        print(f"  {label:<16}{b['srs']:>9.4f}{f['srs']:>9.4f}"
              f"{f['srs'] - b['srs']:>+9.4f}   {f['surviving_ratio']*100:>9.1f}%"
              f"{f['delta_correlation']:>10.3f}")

    pb, pf = base.get("보호본"), full.get("보호본")
    if pb and pf:
        print()
        if pf["below_threshold"]:
            print(f"  보호본은 채널 통과 후에도 임계값 아래다 ({pf['srs']:.4f} < {THRESHOLD}).")
            print("  → 통화 경로에서도 방어가 살아남는다.")
        else:
            print(f"  **보호본이 채널 통과 후 임계값 위로 올라왔다 "
                  f"({pb['srs']:.4f} → {pf['srs']:.4f}).**")
            print("  → 통화 경로에서 방어가 무너진다. 실사용 조건에서는 효과가 없다는 뜻이다.")
        if pf["delta_correlation"] < 0.5:
            print(f"  섭동 구조 상관이 {pf['delta_correlation']:.3f}로 낮다 — "
                  f"코덱이 섭동의 **모양**을 바꿨다. 에너지가 남아도 적대적 효과는 사라진다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"folder": str(folder), "threshold": THRESHOLD, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
