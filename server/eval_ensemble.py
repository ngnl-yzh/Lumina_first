"""인코더 2개 대 3개 — Resemblyzer가 버티는 방향이 막히는가."""
import sys, time, warnings, numpy as np, soundfile as sf, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from mirinae.encoder import (SpeakerEncoder, EcapaEncoder, WavlmEncoder,
                             EncoderEnsemble, cosine_similarity)
from mirinae.pipeline import protect_utterance
from mirinae.config import PGDConfig
from mirinae.codec import telephone_channel

x, sr = sf.read("out/ref/original.wav", dtype="float32")
if x.ndim > 1: x = x.mean(1)
x = np.ascontiguousarray(x[: 16000 * 8])
res, eca, wav = SpeakerEncoder(), EcapaEncoder(), WavlmEncoder()
T = lambda a: a if torch.is_tensor(a) else torch.from_numpy(np.ascontiguousarray(a))
def srs(a, b, enc):
    with torch.no_grad(): return float(cosine_similarity(enc(T(a)), enc(T(b))))

import os
STEPS=int(os.environ.get("STEPS","40"))
cfg = PGDConfig(steps=STEPS)
runs = [(f"3개 {STEPS}스텝", [res, eca, wav])]
print(f"입력 {len(x)/16000:.1f}초 · 40스텝 · 통화 채널 통과 후\n")
print(f"{'앙상블':20} {'Res':>9} {'ECAPA':>9} {'WavLM':>9} {'초':>6}")
print("-" * 58)
for label, encs in runs:
    t0 = time.perf_counter()
    r = protect_utterance(T(x), EncoderEnsemble(encs), cfg, sample_rate=16000,
                          with_controls=False, progress=False)
    el = time.perf_counter() - t0
    co, cp = telephone_channel(T(x)), telephone_channel(T(r.protected))
    print(f"{label:20} {srs(co,cp,res):9.4f} {srs(co,cp,eca):9.4f} "
          f"{srs(co,cp,wav):9.4f} {el:6.0f}")
print("-" * 58)
print("  판정 임계값 0.7962 — 세 열 모두 아래여야 한다")
