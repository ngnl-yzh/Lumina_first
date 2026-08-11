"""인코더별 판정 임계값 — 하나의 숫자를 셋에 쓰면 안 된다.

지금까지 0.7962를 세 인코더에 똑같이 썼다. 그 값은 **Resemblyzer**에서
다른 화자 쌍으로 잰 EER 지점이다. 인코더마다 유사도 분포가 다르므로
그대로 쓰면 판정이 어긋난다.

실제로 WavLM은 다른 화자끼리도 0.9를 넘겼다 — 그 척도에서 0.7962는
"거의 모든 쌍이 타 화자"라는 뜻이 되어 버린다.

여기서는 화자 6명의 조각쌍으로 **같은 화자 / 다른 화자** 분포를 각각 재고,
동일오류율(EER) 지점을 인코더별로 뽑는다.
"""
import itertools, pathlib, sys, warnings, numpy as np, soundfile as sf, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from mirinae.encoder import SpeakerEncoder, EcapaEncoder, WavlmEncoder

SR, SEG = 16000, 3 * 16000

def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != SR:
        from math import gcd; from scipy.signal import resample_poly
        g = gcd(int(sr), SR); x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)

def segments(x, n=6):
    """조각으로 자른다. 같은 화자 쌍을 만들려면 한 파일에서 여러 조각이 필요하다."""
    out = []
    for i in range(n):
        s = i * SEG
        if s + SEG <= len(x): out.append(x[s:s + SEG])
    return out

files = sorted(pathlib.Path("out/speakers").glob("spk*.wav"))
if len(files) < 2:
    print("화자 파일이 부족하다"); sys.exit(1)
segs = {f.stem: segments(load(f)) for f in files}
segs = {k: v for k, v in segs.items() if v}
print(f"화자 {len(segs)}명 · 조각 {sum(len(v) for v in segs.values())}개\n")

for name, enc in [("Resemblyzer", SpeakerEncoder()), ("ECAPA", EcapaEncoder()),
                  ("WavLM", WavlmEncoder())]:
    E = {k: [enc(torch.from_numpy(s)).detach() for s in v] for k, v in segs.items()}
    def cs(a, b): return float(torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)))
    same = [cs(a, b) for v in E.values() for a, b in itertools.combinations(v, 2)]
    diff = [cs(a, b) for k1, k2 in itertools.combinations(E, 2)
            for a in E[k1] for b in E[k2]]
    # EER — 같은 화자를 놓치는 비율과 다른 화자를 붙이는 비율이 같아지는 지점
    best, gap = None, 9e9
    for t in np.linspace(min(diff + same), max(diff + same), 400):
        far = sum(d >= t for d in diff) / len(diff)      # 다른 화자를 같다고
        frr = sum(s < t for s in same) / len(same)       # 같은 화자를 다르다고
        if abs(far - frr) < gap: gap, best = abs(far - frr), (t, far, frr)
    t, far, frr = best
    print(f"{name:12} 같은화자 {np.mean(same):.4f} · 다른화자 {np.mean(diff):.4f} "
          f"→ **임계값 {t:.4f}** (EER {(far + frr) / 2 * 100:.1f}%)")
    print(f"{'':12} 쌍 {len(same)}·{len(diff)}개")
