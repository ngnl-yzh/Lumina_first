"""전이성 진단 — Resemblyzer에 맞춘 섭동이 다른 인코더에도 통하는가.

DSR이 0%인 이유로 가장 유력한 가설이다. 한 인코더의 임베딩 공간에서만
멀어지게 최적화하면, 구조가 다른 모델은 여전히 같은 화자로 듣는다.
복제 모델(XTTS·GPT-SoVITS)은 Resemblyzer를 쓰지 않는다.
"""
import sys, pathlib, warnings, numpy as np, soundfile as sf, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from mirinae.encoder import SpeakerEncoder, cosine_similarity
from mirinae.codec import telephone_channel

pairs = []
for d in sorted(pathlib.Path("out").rglob("original.wav")):
    prot = d.parent / "protected.wav"
    if prot.exists():
        pairs.append((d, prot))
if not pairs:
    print("보호본 쌍 없음"); sys.exit()

def load(p):
    x, sr = sf.read(p, dtype="float32")
    return x.mean(1) if x.ndim > 1 else x

# ── 인코더 2: ECAPA-TDNN (VoxCeleb, 구조·특징 모두 다름) ────────────────────
from speechbrain.inference.speaker import EncoderClassifier
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/ecapa", run_opts={"device": "cpu"})

def ecapa_emb(x):
    with torch.no_grad():
        e = ecapa.encode_batch(torch.from_numpy(x).unsqueeze(0))
    return e.squeeze()

res = SpeakerEncoder()
def res_emb(x):
    with torch.no_grad():
        return res(torch.from_numpy(x))

print(f"{'파일':28} {'Resemblyzer':>12} {'ECAPA-TDNN':>12}")
print("-" * 56)
rows = []
for orig_p, prot_p in pairs[:6]:
    o, p = load(orig_p), load(prot_p)
    n = min(len(o), len(p)); o, p = o[:n], p[:n]
    r = float(cosine_similarity(res_emb(o), res_emb(p)))
    a = float(torch.nn.functional.cosine_similarity(
        ecapa_emb(o).unsqueeze(0), ecapa_emb(p).unsqueeze(0)))
    rows.append((r, a))
    print(f"{str(orig_p.parent)[-26:]:28} {r:12.4f} {a:12.4f}")
if rows:
    mr = sum(r for r, _ in rows)/len(rows); ma = sum(a for _, a in rows)/len(rows)
    print("-" * 56)
    print(f"{'평균':28} {mr:12.4f} {ma:12.4f}")
    print()
    print(f"  최적화 대상(Resemblyzer): {mr:.4f}  ← 판정 임계값 0.7962 아래면 성공")
    print(f"  미학습 인코더(ECAPA)     : {ma:.4f}  ← 여기가 높으면 **전이 실패**")
