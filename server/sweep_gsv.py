"""GPT-SoVITS 방어를 15개 조건에서 되풀이해 잰다 — 1단계: 만들기.

## 왜 되풀이하나

한 번 통한 것은 **우연일 수 있다.** 250스텝·SNR 20 dB·배율 3.0에서
팀원 목소리 하나를 막았다고 해서 "실제로 음성을 보호한다"고 말할 수는 없다.
그 말이 서려면 세 가지를 각각 확인해야 한다.

    ① 설정을 바꿔도 통하나      — 스텝·SNR·배율을 흔든다
    ② 다른 사람에게도 통하나    — 화자를 바꾼다
    ③ 전송을 거쳐도 남나        — mp3·전화대역·잡음·절단

특히 ③이 실전을 가른다. 공격자는 원본 파일을 그대로 받지 않는다.
카카오톡을 거치고, 유튜브를 거치고, 통화 녹음으로 남는다.
그 과정에서 섭동이 씻겨 나가면 **저장 시점에 넣은 보호는 무의미하다.**

## 어떻게 재나

조건마다 이렇게 한다.

    원본 ──[섭동 주입]──→ 보호본 ──[전송 열화]──→ 공격자가 받는 파일
                                                        │
                                        GPT-SoVITS 참조로 투입
                                                        ↓
                                                   복제된 음성 → 채점

**기준선을 같은 열화에 통과시킨다.** mp3를 거친 원본이 복제되지 않는다면
그건 방어가 아니라 mp3가 한 일이다. 둘을 갈라야 한다.

## 2단계

이 스크립트는 wav만 만든다. 채점은 `score_sweep.py`가 한다 —
화자 검증기 셋(Resemblyzer·ECAPA·WavLM)이 다른 가상환경에 있다.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

SR = 16000


# ─────────────────────────────────────────────────────────────
# 전송 열화 — 공격자가 손에 넣는 파일은 원본이 아니다
# ─────────────────────────────────────────────────────────────

def deg_none(x: np.ndarray) -> np.ndarray:
    return x


def deg_mp3(x: np.ndarray, kbps: int = 64) -> np.ndarray:
    """mp3 왕복. 메신저·SNS 업로드가 하는 일이다."""
    import io
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        src, mid, dst = Path(d) / "a.wav", Path(d) / "b.mp3", Path(d) / "c.wav"
        sf.write(str(src), x, SR)
        for cmd in ([("-i", str(src), "-b:a", f"{kbps}k", str(mid))],
                    [("-i", str(mid), "-ar", str(SR), str(dst))]):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd[0]],
                           check=True)
        y, _ = sf.read(str(dst), dtype="float32")
    return np.ascontiguousarray(y[:len(x)] if len(y) > len(x) else
                                np.pad(y, (0, len(x) - len(y))))


def deg_phone(x: np.ndarray) -> np.ndarray:
    """8 kHz 전화 대역 왕복. 통화 녹음이 하는 일이다."""
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(SR, 8000)
    down = resample_poly(x, 8000 // g, SR // g)
    up = resample_poly(down, SR // g, 8000 // g).astype(np.float32)
    return np.ascontiguousarray(up[:len(x)] if len(up) > len(x) else
                                np.pad(up, (0, len(x) - len(up))))


def deg_noise(x: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """백색잡음. 마이크·환경이 하는 일이다."""
    rng = np.random.default_rng(0)
    n = rng.standard_normal(len(x)).astype(np.float32)
    p_x, p_n = float(np.mean(x ** 2)), float(np.mean(n ** 2))
    scale = np.sqrt(p_x / (p_n * 10 ** (snr_db / 10)))
    return np.ascontiguousarray(x + scale * n)


def deg_crop(x: np.ndarray, seconds: float = 3.0) -> np.ndarray:
    """앞 3초만 잘라 쓴다. 공격자는 짧은 조각으로도 복제를 시도한다.

    **이게 가장 어려운 조건이다.** 섭동은 전체 구간에 걸쳐 계산됐는데
    그중 일부만 남는다. 남은 조각 안에서도 효과가 살아 있어야 한다.
    """
    n = int(seconds * SR)
    return np.ascontiguousarray(x[:n] if len(x) > n else x)


def deg_worst(x: np.ndarray) -> np.ndarray:
    """겹쳐 건다 — mp3 → 전화대역 → 잡음 → 2초 절단.

    실제로 이런 일이 일어난다. 통화가 녹음되고(전화대역),
    메신저로 넘어가고(mp3), 스피커로 재생된 걸 다시 담고(잡음),
    쓸 만한 대목만 잘라 쓴다(절단). **가장 가혹한 조건이다.**
    """
    return deg_crop(deg_noise(deg_phone(deg_mp3(x)), 18.0), 2.0)


DEGRADE = {"none": deg_none, "mp3": deg_mp3, "phone": deg_phone,
           "noise": deg_noise, "crop": deg_crop, "worst": deg_worst}


# ─────────────────────────────────────────────────────────────
# 15개 조건
# ─────────────────────────────────────────────────────────────

def default_plan() -> list[dict]:
    """(이름, 화자, 스텝, SNR, 배율, 열화)."""
    real = "out/team/voice.wav"
    base = dict(steps=250, snr=20.0, mask=3.0, deg="none", eot=False)

    plan: list[dict] = []

    # ① 설정을 바꿔도 통하나 — 실제 목소리 1명
    for steps in (100, 150, 250, 400):
        plan.append({**base, "name": f"A-{steps}스텝", "src": real, "steps": steps})
    for snr in (16.0, 24.0):
        plan.append({**base, "name": f"A-SNR{snr:.0f}", "src": real, "snr": snr})
    for mask in (2.0, 4.0):
        plan.append({**base, "name": f"A-배율{mask}", "src": real, "mask": mask})

    # ② 다른 사람에게도 통하나
    for i in (1, 2, 3):
        plan.append({**base, "name": f"B-화자{i:02d}",
                     "src": f"out/speakers/spk{i:02d}.wav"})

    # ③ 전송을 거쳐도 남나 — 실제 목소리, 최적 설정
    for deg in ("mp3", "phone", "noise", "crop"):
        plan.append({**base, "name": f"C-{deg}", "src": real, "deg": deg})

    # ④ 1차 15회에서 배운 것을 합친다 — 실험 16~23
    #
    # 실패한 5개는 **전부 WavLM 하나만** 임계값 위였다.
    # WavLM을 공격 표적에 넣으면 곧장 뚫리지만 그러면 WavLM은
    # 더 이상 독립된 심판이 아니다 — 검증기 쪽에서 세 번 겪은 과적합이다.
    #
    # 그래서 **표적은 GPT-SoVITS 하나로 두고** 설정만 합친다.
    # 통과한 조건에서 WavLM이 가장 낮았던 값들을 모은다.
    #
    #     400스텝 (0.8268) · SNR 24 (0.7815) · 배율 4.0 (0.8485)
    #
    # SNR 24가 16보다 나았다는 게 뜻밖이다 — 섭동을 **키우는** 게 아니라
    # 청각 마스킹이 허용하는 자리에 **정확히 놓는** 게 중요하다는 뜻이다.
    hard = dict(steps=400, snr=24.0, mask=4.0)
    plan.append({**base, **hard, "name": "D-강화", "src": real})
    for deg in ("mp3", "phone", "noise", "crop"):
        plan.append({**base, **hard, "name": f"D-강화-{deg}",
                     "src": real, "deg": deg})
    for i in (1, 2, 3):
        plan.append({**base, **hard, "name": f"D-강화-화자{i:02d}",
                     "src": f"out/speakers/spk{i:02d}.wav"})

    # ⑤ 열화를 최적화 안으로 — 실험 24~31 (EOT)
    #
    # ④는 실패했다. 손잡이를 돌려서는 안 된다는 것을 확인한 셈이다.
    # 이제 매 스텝 무작위 열화를 걸고 그 상태에서 손실을 내린다.
    eot = dict(steps=400, snr=20.0, mask=3.0, eot=True)
    plan.append({**base, **eot, "name": "E-EOT", "src": real})
    for deg in ("mp3", "phone", "noise", "crop"):
        plan.append({**base, **eot, "name": f"E-EOT-{deg}", "src": real, "deg": deg})
    for i in (1, 2, 3):
        plan.append({**base, **eot, "name": f"E-EOT-화자{i:02d}",
                     "src": f"out/speakers/spk{i:02d}.wav"})

    # ⑥ 확인 — 실험 32~40. EOT가 운이 아닌지 본다.
    #
    # 새 화자 3명, 더 싼 설정 둘, 그리고 열화를 **겹친** 최악 조건.
    # 여기까지 서면 배포 설정을 정할 수 있다.
    eot_b = dict(snr=20.0, mask=3.0, eot=True)
    for st in (150, 250):
        plan.append({**base, **eot_b, "steps": st,
                     "name": f"F-EOT{st}", "src": real})
    for i in (4, 5, 6):
        plan.append({**base, **eot_b, "steps": 400,
                     "name": f"F-EOT-화자{i:02d}",
                     "src": f"out/speakers/spk{i:02d}.wav"})
    plan.append({**base, **eot_b, "steps": 400,
                 "name": "F-EOT-최악", "src": real, "deg": "worst"})
    plan.append({**base, **eot_b, "steps": 250,
                 "name": "F-EOT250-최악", "src": real, "deg": "worst"})
    for i in (1, 2):
        plan.append({**base, **eot_b, "steps": 400,
                     "name": f"F-EOT-화자{i:02d}-최악",
                     "src": f"out/speakers/spk{i:02d}.wav", "deg": "worst"})

    # ⑦ 겹침 EOT — 실험 41~45. F에서 4단 겹침이 남았다.
    g = dict(steps=400, snr=20.0, mask=3.0, eot=True)
    plan.append({**base, **g, "name": "G-겹침", "src": real})
    plan.append({**base, **g, "name": "G-겹침-최악", "src": real, "deg": "worst"})
    plan.append({**base, **g, "name": "G-겹침-phone", "src": real, "deg": "phone"})
    plan.append({**base, **g, "name": "G-겹침-화자06",
                 "src": "out/speakers/spk06.wav"})
    plan.append({**base, **g, "name": "G-겹침-화자02-최악",
                 "src": "out/speakers/spk02.wav", "deg": "worst"})

    return plan


# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS 방어 15조건 스윕 (1단계)")
    ap.add_argument("-o", "--out", default="out/sweep")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--only", default="", help="이름에 이 문자열이 든 조건만")
    args = ap.parse_args()

    import attack_cloner as ac
    from clone_gsv import GsvCloner, load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = [c for c in default_plan() if args.only in c["name"]]

    print(f"장치 {device} · 조건 {len(plan)}개")
    print("모델을 한 번만 올린다 — 조건마다 다시 올리면 시간이 안 맞는다.")
    target = ac.GptSovitsTarget(device)
    cloner = GsvCloner(device)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    for i, c in enumerate(plan, 1):
        t0 = time.time()
        d = root / c["name"]
        d.mkdir(exist_ok=True)
        print(f"\n[{i}/{len(plan)}] {c['name']}  "
              f"{Path(c['src']).name} · {c['steps']}스텝 · "
              f"SNR {c['snr']:.0f} · 배율 {c['mask']} · 열화 {c['deg']}")

        x = load(c["src"])[:int(args.seconds * SR)]

        # ① 섭동 주입
        prot_t, final = ac.attack(torch.from_numpy(x), target,
                                  steps=c["steps"], snr_db=c["snr"],
                                  prosody_weight=0.0, masking_ratio=c["mask"],
                                  progress=False, eot=c.get("eot", False))
        prot = np.ascontiguousarray(prot_t.numpy())
        cond = final.get(target.name, final)

        # ② 전송 열화 — **원본에도 똑같이 적용한다.**
        #    기준선을 같은 조건에 통과시켜야 방어와 열화를 가를 수 있다.
        fn = DEGRADE[c["deg"]]
        # 열화 함수가 float64를 돌려주면 모델이 dtype으로 막는다 — 되돌린다.
        x_d = np.ascontiguousarray(fn(x), dtype=np.float32)
        prot_d = np.ascontiguousarray(fn(prot), dtype=np.float32)

        sf.write(str(d / "original.wav"), x_d, SR)
        sf.write(str(d / "protected.wav"), prot_d, SR)

        # ③ 복제 — 내용은 언제나 깨끗한 원본에서, 음색만 바뀐다
        for tag, ref in (("original", x_d), ("protected", prot_d)):
            wav = cloner.clone(x, ref)
            sf.write(str(d / f"clone_from_{tag}_0.wav"), wav, 32000)

        dt = time.time() - t0
        print(f"    복제기 조건 {cond if cond is not None else '?'} · {dt:.0f}초")
        log.append({**c, "cond": cond, "seconds": round(dt, 1)})
        (root / "plan.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료 — {root.resolve()}")
    print("다음: .venv/Scripts/python score_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
