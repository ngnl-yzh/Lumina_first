"""섭동이 실제로 주입됐는지 검증한다 — "돌아갔다"와 "됐다"는 다르다.

`protect.py`가 오류 없이 끝나도 섭동이 안 들어갔을 수 있다.
무음 delta가 저장되거나, 파일을 잘못 덮어썼거나, VAD 마스크가 전부 0이거나,
GPU를 쓴다고 생각했는데 CPU로 돌았거나 — 전부 **조용히 틀리는** 종류다.

이 도구는 `protect.py` 출력 폴더를 받아 여섯 가지를 확인한다.

| 확인 | 무엇을 잡는가 |
|---|---|
| ① 파일 무결성 | protected = original + delta 인가. 아니면 엉뚱한 파일을 보고 있는 것 |
| ② 섭동 존재 | delta가 무음이 아닌가. SNR이 목표 근처인가 |
| ③ 대역 제한 | 300~3400 Hz 밖으로 새지 않았는가 |
| ④ 무음 보존 | 말이 없는 구간에 섭동이 안 들어갔는가 (VAD 마스크가 걸렸는가) |
| ⑤ 지각 예산 | 마스킹 임계값을 얼마나 넘었는가 — "안 들린다"를 주장할 수 있는가 |
| ⑥ 실제 효과 | 화자 유사도가 떨어졌는가. 대조군보다 나은가 |

⑥까지 통과해야 "주입됐고 효과가 있다"고 말할 수 있다.
①~⑤만 통과하면 "신호는 들어갔지만 효과는 미확인"이다.

사용:
    python verify_injection.py out/demo
    python verify_injection.py out/demo --channel      # 통화 채널 통과 후로 평가
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirinae.codec import CHANNELS, telephone_channel
from mirinae.config import (
    BAND_HIGH_HZ, BAND_LOW_HZ, SAMPLE_RATE, TARGET_SNR_DB, default_device,
)
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.metrics import audibility, band_energy_ratio_db, snr_db
from mirinae.pipeline import ProtectionResult
from mirinae.psychoacoustic import MaskingModel

THRESHOLD = ProtectionResult.PROVISIONAL_THRESHOLD

OK, WARN, FAIL = "OK  ", "주의", "실패"


def mark(passed: bool, warn: bool = False) -> str:
    return WARN if warn else (OK if passed else FAIL)


def load(path: Path) -> torch.Tensor:
    from protect import load_wav
    return load_wav(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="섭동 주입 검증")
    ap.add_argument("folder", nargs="?", default="out/demo",
                    help="protect.py 출력 폴더")
    # 앱(모드 2)에서 받은 파일에는 delta.wav도 대조군도 없다.
    # 그 경우 δ를 두 파일의 차이로 구하고, 대조군이 필요한 검사는 건너뛴다.
    ap.add_argument("--original", help="원본 WAV — 폴더 대신 파일 두 개로 검증")
    ap.add_argument("--protected", help="보호본 WAV")
    ap.add_argument("--channel", action="store_true",
                    help="통화 채널(300~3400 Hz · 8 kHz · G.711) 통과 후로 평가")
    ap.add_argument("--silence-db", type=float, default=-45.0,
                    help="이 아래를 무음 구간으로 본다")
    args = ap.parse_args()

    pair_mode = bool(args.original and args.protected)
    if pair_mode:
        d = Path(args.protected).parent
        orig_p, prot_p = Path(args.original), Path(args.protected)
        for p in (orig_p, prot_p):
            if not p.exists():
                print(f"{p} 가 없다.")
                return 1
        x, y = load(orig_p), load(prot_p)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        delta = y - x            # δ를 직접 갖고 있지 않으니 차이로 구한다
    else:
        d = Path(args.folder)
        orig_p, prot_p, delta_p = (d / "original.wav", d / "protected.wav",
                                   d / "delta.wav")
        for p in (orig_p, prot_p, delta_p):
            if not p.exists():
                print(f"{p} 가 없다. "
                      f"먼저 `python protect.py 목소리.wav -o {d}` 를 돌리거나, "
                      f"앱에서 받은 파일이면 --original / --protected 로 넘길 것.")
                return 1
        x, y, delta = load(orig_p), load(prot_p), load(delta_p)
        n = min(len(x), len(y), len(delta))
        x, y, delta = x[:n], y[:n], delta[:n]

    device = default_device()

    print(f"입력: {orig_p.name} · {prot_p.name}" if pair_mode else f"폴더: {d}")
    print(f"길이: {n / SAMPLE_RATE:.1f}초 · 장치: {device}")
    if device.type != "cuda":
        print("  ※ CUDA가 아니다. GPU 장비인데 이게 뜨면 torch가 CPU 빌드다 (SETUP.md 1단계)")
    print()

    report = {}
    rp = d / "report.json"
    if rp.exists():
        report = json.loads(rp.read_text(encoding="utf-8"))
        cfg = report.get("config", {})
        print(f"생성 설정: 배율 {cfg.get('masking_ratio')} · {cfg.get('steps')}스텝 "
              f"· 목표 SNR {cfg.get('target_snr_db')} dB")
        print()

    fails = 0
    warns = 0

    # ── ① 파일 무결성 ─────────────────────────────────────────────────────────
    print("① 파일 무결성 — protected = original + delta 인가")
    if pair_mode:
        print("   —     δ를 두 파일의 차이로 구했으므로 정의상 성립. 검사 생략")
    else:
        resid = float((y - (x + delta)).abs().max())
        scale = max(float(y.abs().max()), 1e-12)
        ok = resid / scale < 1e-3
        print(f"   {mark(ok)}  최대 잔차 {resid:.2e} (신호 대비 {resid / scale * 100:.4f}%)")
        if not ok:
            print("        → 세 파일이 서로 맞지 않는다. 다른 실행의 산출물이 섞였을 수 있다.")
            fails += 1
    print()

    # ── ② 섭동 존재 ───────────────────────────────────────────────────────────
    print("② 섭동이 실제로 있는가")
    d_rms = float(torch.sqrt((delta ** 2).mean()))
    x_rms = float(torch.sqrt((x ** 2).mean()))
    snr = snr_db(x, delta)
    alive = d_rms > 1e-6
    print(f"   {mark(alive)}  delta RMS {d_rms:.3e}  (원본 {x_rms:.3e})")
    if not alive:
        print("        → delta가 사실상 무음이다. **섭동이 주입되지 않았다.**")
        fails += 1
    near = abs(snr - TARGET_SNR_DB) <= 3.0
    print(f"   {mark(near, warn=not near)}  SNR {snr:.1f} dB  (목표 {TARGET_SNR_DB} dB)")
    if not near:
        print("        → 목표에서 3 dB 넘게 벗어났다. 정규화가 제대로 안 걸렸을 수 있다.")
        warns += 1
    print()

    # ── ③ 대역 제한 ───────────────────────────────────────────────────────────
    print("③ 통화 대역 안에 있는가")
    oob = band_energy_ratio_db(delta, BAND_LOW_HZ, BAND_HIGH_HZ, SAMPLE_RATE)
    ok = oob < -30.0
    print(f"   {mark(ok, warn=not ok and oob < -20)}  대역 밖 에너지 {oob:.1f} dB "
          f"({BAND_LOW_HZ:.0f}~{BAND_HIGH_HZ:.0f} Hz 기준)")
    if oob >= -20:
        print("        → 대역 제한이 안 걸렸다. 통화망에서 잘려 나갈 성분이 많다.")
        fails += 1
    elif not ok:
        warns += 1
    print()

    # ── ④ 무음 보존 ───────────────────────────────────────────────────────────
    print("④ 말이 없는 구간에 섭동이 안 들어갔는가")
    win = int(SAMPLE_RATE * 0.02)
    nb = n // win
    xe = (x[:nb * win].reshape(nb, win) ** 2).mean(dim=1)
    de = (delta[:nb * win].reshape(nb, win) ** 2).mean(dim=1)
    thr_lin = 10.0 ** (args.silence_db / 10.0) * float(xe.max())
    quiet = xe < thr_lin
    if int(quiet.sum()) == 0:
        print(f"   {WARN}  무음 구간이 없다 — 이 항목은 판정 불가")
        warns += 1
    else:
        leak = float(de[quiet].mean()) / max(float(de[~quiet].mean()), 1e-20)
        ok = leak < 0.05
        print(f"   {mark(ok, warn=not ok and leak < 0.2)}  무음 구간 섭동 에너지가 "
              f"발화 구간의 {leak * 100:.1f}%  (무음 프레임 {int(quiet.sum())}/{nb})")
        if leak >= 0.2:
            print("        → VAD 마스크가 안 걸렸다. 조용한 데서 섭동만 들린다.")
            fails += 1
        elif not ok:
            warns += 1
    print()

    # ── ⑤ 지각 예산 ───────────────────────────────────────────────────────────
    print("⑤ 마스킹 임계값을 얼마나 넘었는가 — '안 들린다'를 주장할 수 있는가")
    model = MaskingModel(device=x.device)
    thr, _ = model.threshold(x)
    a = audibility(delta, thr, 1.0, model)
    quiet_enough = a.violation_ratio < 0.05
    print(f"   {mark(quiet_enough, warn=not quiet_enough)}  "
          f"최대 초과 {a.max_excess_db:.1f} dB · 위반 bin {a.violation_ratio * 100:.1f}%")
    if not quiet_enough:
        print(f"        → bin의 {a.violation_ratio * 100:.0f}%가 임계값 위다. "
              f"**'안 들린다'고 단언하면 안 된다.**")
        print("        → 배율을 낮추면 줄지만 방어도 함께 약해진다 (DEMO.md 준비 2)")
        warns += 1
    print()

    # ── ⑥ 실제 효과 ───────────────────────────────────────────────────────────
    label = "통화 채널 통과 후" if args.channel else "파일 기준"
    print(f"⑥ 화자 유사도가 실제로 떨어졌는가 ({label})")

    def heard(w: torch.Tensor) -> torch.Tensor:
        return telephone_channel(w, CHANNELS["ulaw"]) if args.channel else w

    enc = SpeakerEncoder(device=device)
    with torch.no_grad():
        ref = enc(heard(x.to(device)))
        srs_prot = float(cosine_similarity(enc(heard(y.to(device))), ref))

        controls = {}
        for name, fn in (("C-A 백색잡음", "control_C-A.wav"),
                         ("C-B 통화대역잡음", "control_C-B.wav"),
                         ("C-C 무섭동", "control_C-C.wav"),
                         ("C-E 셔플 섭동", "control_C-E.wav")):
            cp = d / fn
            if cp.exists():
                cw = load(cp)[:n].to(device)
                controls[name] = float(cosine_similarity(enc(heard(cw)), ref))

    cc = controls.get("C-C 무섭동")
    if cc is not None:
        sane = cc > 0.99
        print(f"   {mark(sane)}  C-C 무섭동 {cc:.4f}  ← 인코더 정상 확인 (1.0이어야 한다)")
        if not sane:
            print("        → 인코더가 이상하다. 아래 수치를 믿을 수 없다.")
            fails += 1

    # 판정은 **임계값 기준**이어야 한다.
    #
    # 한때 `srs_prot < 0.95`였다. 실사용 파일이 0.9466으로 들어왔을 때
    # 이 검사가 OK를 냈다 — 방어가 사실상 없는데 통과시킨 것이다.
    # "0.95만 넘지 않으면 된다"는 근거 없는 기준이었고,
    # 실제로 의미 있는 선은 C-D 대조군에서 측정한 판정 임계값이다.
    below = srs_prot < THRESHOLD
    near = srs_prot < THRESHOLD + 0.05
    print(f"   {mark(below, warn=not below and near)}  보호본 {srs_prot:.4f}  "
          f"(판정 임계값 {THRESHOLD} — 이 아래면 '다른 화자')")
    if not below:
        gap = srs_prot - THRESHOLD
        print(f"        → 임계값보다 {gap:+.4f} 높다. **'다른 화자'로 판정되지 않는다.**")
        if srs_prot > 0.9:
            print("        → 0.9를 넘는다는 것은 최적화가 사실상 안 됐다는 뜻이다.")
            print("           PGD 스텝이 몇 번 돌았는지 확인할 것 "
                  "(앱 화면의 '스텝', 또는 서버 로그)")
        fails += 1

    beat = [k for k, v in controls.items()
            if k.endswith("잡음") and srs_prot < v]
    noise = [v for k, v in controls.items() if k.endswith("잡음")]
    if noise:
        best = min(noise)
        won = srs_prot < best
        print(f"   {mark(won, warn=not won)}  잡음 대조군 최저 {best:.4f} 대비 "
              f"{best - srs_prot:+.4f}")
        if not won:
            print("        → 같은 세기 잡음보다 못하다. '그냥 잡음 아니냐'에 답할 수 없다.")
            warns += 1

    ce = controls.get("C-E 셔플 섭동")
    if ce is not None:
        structural = ce - srs_prot > 0.05
        print(f"   {mark(structural, warn=not structural)}  "
              f"C-E 셔플 {ce:.4f} 대비 {ce - srs_prot:+.4f}  "
              f"← 크기가 아니라 구조가 원인인가")
        if not structural:
            print("        → 셔플해도 효과가 같다. 적대적 최적화의 기여가 확인되지 않는다.")
            warns += 1
    print()

    # ── 종합 ──────────────────────────────────────────────────────────────────
    print("=" * 62)
    if fails:
        print(f"실패 {fails}건 · 주의 {warns}건 — **주입이 정상적으로 되지 않았다.**")
    elif warns:
        print(f"실패 0건 · 주의 {warns}건 — 주입은 됐고 효과도 있다. 위 주의 항목을 읽을 것.")
    else:
        print("전 항목 통과 — 섭동이 설계대로 주입됐고 효과도 확인된다.")
    print("=" * 62)

    if not args.channel:
        print("\n실제 위협 모델(통화 녹취)에서의 성능은 --channel 로 다시 볼 것.")
    print("귀로도 확인하려면 delta.wav를 직접 들어본다 — 섭동만 담긴 파일이다.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
