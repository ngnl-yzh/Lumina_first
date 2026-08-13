"""GPT-SoVITS로 실제 복제해 방어를 확인한다 — 모드 3의 핵심 증거.

## 왜 이 도구가 필요한가

지금까지 실제 복제 검증은 **XTTS-v2**로만 했다. GPT-SoVITS는 **표적**으로만
썼다 — 섭동을 계산하려고 `ref_enc`를 불러왔을 뿐, 그 모델로 목소리를 만들어
본 적이 없다.

그런데 프로젝트가 성공이라고 말하려면 이 문장이 서야 한다.

    "GPT-SoVITS에 원본을 주면 그 사람 목소리가 나오고,
     보호본을 주면 **다른 사람 목소리가 나온다**."

표적으로 삼은 모델과 복제에 쓴 모델이 같아야 그 문장이 성립한다.
이 도구가 그 자리를 메운다.

## 어떻게 복제하나 — 음색만 바꾼다

GPT-SoVITS의 전체 추론은 텍스트를 받아 GPT가 의미 토큰을 만들고 SoVITS가
소리로 바꾸는 두 단계다. 여기서는 **뒷단만** 쓴다.

    내용(무엇을 말하나)  ←  고정된 원본 발화의 SSL 특징
    음색(누구 목소리인가) ←  참조 음성의 `ref_enc` 벡터   ← 여기가 바뀐다

같은 내용을 **원본 참조**와 **보호본 참조**로 각각 만들어 비교한다.
텍스트 프런트엔드(한국어 g2p·BERT)가 필요 없어 가볍고, 무엇보다
**음색 이외의 변수가 없다** — 두 결과의 차이는 오직 참조 음성 차이다.

이건 축소가 아니라 **더 깨끗한 대조 실험**이다. 전체 추론을 쓰면
GPT의 무작위성이 섞여 무엇 때문에 달라졌는지 말하기 어려워진다.

## 무엇으로 판정하나

만들어진 두 음성을 화자 검증기 셋으로 원본 화자와 비교한다.
임계값은 `eval_thresholds.py`가 화자 6명으로 잰 값이다.

**기준선을 먼저 본다.** 원본 참조로 만든 복제음이 임계값 위여야
이 측정이 성립한다 — 원본조차 복제되지 않으면 방어를 논할 수 없다.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

SR = 16000
GSV_SR = 32000
HUBERT_SR = 16000


def load(path: str, sr: int = SR) -> np.ndarray:
    x, orig = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if orig != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig), sr)
        x = resample_poly(x, sr // g, int(orig) // g).astype(np.float32)
    return np.ascontiguousarray(x)


class GsvCloner:
    """GPT-SoVITS의 음색 변환 경로. 참조 음성만 바꿔 같은 내용을 다시 만든다."""

    def __init__(self, device: torch.device) -> None:
        root = Path(__file__).parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "GPT_SoVITS"))
        from huggingface_hub import snapshot_download
        from GPT_SoVITS.module.models import SynthesizerTrn
        from transformers import HubertModel

        base = Path(snapshot_download(
            "lj1995/GPT-SoVITS",
            allow_patterns=["s2G488k.pth", "chinese-hubert-base/*"]))
        ckpt = torch.load(base / "s2G488k.pth", map_location="cpu",
                          weights_only=False)
        hps = ckpt["config"]

        def g(o, k, d=None):
            return o[k] if isinstance(o, dict) else getattr(o, k, d)

        data, model = g(hps, "data"), g(hps, "model")
        mdict = model if isinstance(model, dict) else vars(model)
        self.vq = SynthesizerTrn(
            g(data, "filter_length") // 2 + 1, 32,
            n_speakers=g(data, "n_speakers", 0) or 0,
            version="v1", **mdict,
        )
        self.vq.load_state_dict(ckpt["weight"], strict=False)
        self.vq.to(device).eval()

        # SSL 특징 추출기 — "무엇을 말하나"를 담당한다.
        self.hubert = HubertModel.from_pretrained(
            str(base / "chinese-hubert-base")).to(device).eval()

        for m in (self.vq, self.hubert):
            for p in m.parameters():
                p.requires_grad_(False)

        self.device = device
        self.n_fft = g(data, "filter_length")
        self.hop = g(data, "hop_length")
        self.win = g(data, "win_length")
        self.window = torch.hann_window(self.win).to(device)

    def _spec(self, w32: torch.Tensor) -> torch.Tensor:
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(w32.unsqueeze(1), (pad, pad),
                                    mode="reflect").squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop,
                          win_length=self.win, window=self.window,
                          center=False, onesided=True, return_complex=True)
        return torch.sqrt(torch.abs(spec) ** 2 + 1e-8)

    @torch.no_grad()
    def clone(self, content16k: np.ndarray, reference16k: np.ndarray) -> np.ndarray:
        """`reference`의 음색으로 `content`의 내용을 다시 만든다."""
        import torchaudio

        c = torch.from_numpy(content16k).unsqueeze(0).to(self.device)
        r = torch.from_numpy(reference16k).unsqueeze(0).to(self.device)

        # ① 내용 — SSL 특징에서 의미 토큰을 뽑는다.
        ssl = self.hubert(c).last_hidden_state.transpose(1, 2)
        # extract_latent는 (양자화기, B, T)로 전치해 돌려준다.
        # decode는 (B, 양자화기, T)를 기대하므로 되돌린다.
        codes = self.vq.extract_latent(ssl).transpose(0, 1)

        # ② 음색 — 참조 음성의 스펙트로그램에서 뽑는다. **여기만 바뀐다.**
        r32 = torchaudio.functional.resample(r, SR, GSV_SR)
        refer = self._spec(r32)

        # text는 길이만 쓰이고 내용은 무시된다 — 자리표를 넣는다.
        wav = self.vq.decode(codes,
                             torch.zeros(1, 1, dtype=torch.long, device=self.device),
                             refer)
        return wav.squeeze().detach().cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser(
        description="GPT-SoVITS로 원본·보호본을 각각 복제해 비교한다")
    p.add_argument("original", help="원본 wav")
    p.add_argument("protected", help="보호본 wav")
    p.add_argument("--content", default=None,
                   help="복제음이 말할 내용의 출처. 기본은 원본 자신")
    p.add_argument("-o", "--out", default="out/clone_gsv")
    p.add_argument("--repeat", type=int, default=3,
                   help="반복 횟수. 결정적 경로라 1회로도 되지만 확인용으로 둔다")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig = load(args.original)
    prot = load(args.protected)
    content = load(args.content) if args.content else orig
    n = min(len(orig), len(prot))
    orig, prot = orig[:n], prot[:n]

    print(f"장치 {device} · 참조 {n / SR:.1f}초 · 내용 {len(content) / SR:.1f}초")
    print("GPT-SoVITS 로드 중...")
    cl = GsvCloner(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "original.wav"), orig, SR)
    sf.write(str(out / "protected.wav"), prot, SR)

    for tag, ref in (("original", orig), ("protected", prot)):
        for i in range(args.repeat):
            print(f"  [{tag}] 복제 {i + 1}/{args.repeat}...")
            wav = cl.clone(content, ref)
            sf.write(str(out / f"clone_from_{tag}_{i}.wav"), wav, GSV_SR)

    print(f"\n출력: {out.resolve()}")
    print("다음 — 채점:")
    print(f"  python eval_dsr.py   (표에 '{out.name}' 조건을 추가한 뒤)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
