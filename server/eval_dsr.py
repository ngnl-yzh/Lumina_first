"""복제 저지율(DSR) 재측정 — 앙상블이 실제 복제까지 막는가.

전이성 진단으로 원인은 찾았다. 인코더 두 개에서 임계값 아래로 내렸다.
그런데 **인코더 두 개를 이겼다고 세 번째가 따라오리라는 보장은 없다.**
실제 복제 모델(XTTS-v2)이 만든 음성을 원본 화자와 비교해야 답이 나온다.

채점을 인코더 두 개로 한다. 하나로만 재면 그 인코더에 대한 과적합을
다시 보게 된다 — 이번에 배운 것이 그것이다.
"""
import sys, pathlib, statistics, warnings, numpy as np, soundfile as sf, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from mirinae.encoder import (SpeakerEncoder, EcapaEncoder, WavlmEncoder,
                             cosine_similarity)

THRESHOLD = 0.7962          # C-D 대조군에서 측정한 EER 지점

SR = 16000

def load(p):
    """**반드시 16 kHz로 맞춰서** 읽는다.

    XTTS-v2는 24 kHz로 출력한다. 그대로 인코더에 넣으면 음높이와 속도가
    1.5배 어긋난 채로 채점된다 — 첫 측정에서 원본 복제조차 ECAPA 0.12가
    나왔다. 우리 방어가 통한 것처럼 보였지만 **복제 자체가 실패한 것도
    아니고 측정이 틀린 것**이었다. 기준선을 먼저 보지 않았으면 그대로
    'DSR 100%'라고 적었을 것이다.
    """
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)

ENCODERS = [("Resemblyzer", SpeakerEncoder()), ("ECAPA", EcapaEncoder()),
            ("WavLM", WavlmEncoder())]

def emb(enc, x):
    with torch.no_grad():
        return enc(torch.from_numpy(np.ascontiguousarray(x)))

def sim(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

conds = [("단독 (Res)", "out/cmp_single", "out/clone_single"),
         ("2개 (Res+ECAPA)", "out/ens_smoke",  "out/clone_ens"),
         ("3개 200스텝", "out/ens3_200", "out/clone_ens3")]

print(f"복제 모델 XTTS-v2 · 조건당 5회 · 판정 임계값 {THRESHOLD} · 전부 {SR} Hz로 맞춤")
print("=" * 84)
head = "".join(f"{n:>16}" for n, _ in ENCODERS)
print(f"{'보호 방식':18} {'복제 대상':10}{head} {'저지':>6}")
print("-" * 84)
for label, src, clone_dir in conds:
    ref_p = pathlib.Path(src) / "original.wav"
    if not ref_p.exists():
        print(f"{label:18} (보호본 없음 — 건너뜀)"); print("-" * 84); continue
    ref = load(ref_p)
    refs = [emb(e, ref) for _, e in ENCODERS]
    for tag in ("original", "protected"):
        files = sorted(pathlib.Path(clone_dir).glob(f"clone_from_{tag}_*.wav"))
        if not files: continue
        cols = [[] for _ in ENCODERS]
        blocked = 0
        for f in files:
            c = load(f)
            vals = [sim(refs[i], emb(e, c)) for i, (_, e) in enumerate(ENCODERS)]
            for i, v in enumerate(vals): cols[i].append(v)
            blocked += all(v < THRESHOLD for v in vals)
        def ci(v):
            return 0.0 if len(v) < 2 else 1.96 * statistics.stdev(v) / len(v) ** 0.5
        cells = "".join(f"{statistics.mean(c):9.4f}±{ci(c):.3f}" for c in cols)
        name = label if tag == "protected" else ""
        print(f"{name:18} {tag:10}{cells} {blocked/len(files)*100:5.0f}%")
    print("-" * 84)
print()
print("  저지 = **모든** 인코더가 임계값 아래여야 인정한다.")
print("  하나로만 재면 그 인코더에 대한 과적합을 다시 보게 된다.")
print()
print("  ※ 기준선을 먼저 볼 것 — **원본 복제**가 임계값 위여야 이 측정이 성립한다.")
print("     원본조차 복제되지 않으면 방어 효과를 논할 수 없다.")
