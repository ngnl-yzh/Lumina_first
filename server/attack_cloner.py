"""복제 모델을 직접 공격한다 — 화자 검증기가 아니라 **생성기**를 표적으로.

## 왜 방향을 바꿨나

지금까지는 화자 검증기(Resemblyzer·ECAPA·WavLM)를 표적으로 삼았다.
검증기 쪽 숫자는 잘 내려갔다 — SRS 0.49~0.66. 그런데 **DSR은 네 번 다 0%**였다.

    인코더 1→2→3개 · 40→200스텝 · 채널표적→원본표적
    네 축을 바꿔 네 번 쟀고 전부 저지 0%

원인이 분명했다. 복제 모델은 우리가 공격한 검증기를 **쓰지 않는다.**
자기 방식으로 참조 음성에서 화자 특징을 뽑고, 그 과정에서 우리 섭동이 씻겨 나간다.

**검증기를 속이는 것과 생성기를 막는 것은 다른 문제다.**
이 스크립트는 후자를 한다.

## 무엇을 표적으로 삼나

XTTS-v2는 참조 음성에서 **두 갈래**로 화자 정보를 뽑는다.

| 경로 | 무엇 | 쓰이는 곳 |
|---|---|---|
| `hifigan_decoder.speaker_encoder` | 512차원 화자 임베딩 (16 kHz) | 보코더 — **음색** |
| `get_gpt_cond_latents` | mel 기반 조건 잠재 (1024×T) | GPT — **말투·운율** |

둘 다 순수 torch 모듈이라 파형까지 역전파가 닿는다.
`@torch.inference_mode()`는 바깥 래퍼에 붙어 있어 `__wrapped__`로 벗기면 된다.

**둘을 동시에 민다.** 음색만 밀면 GPT가 운율로 화자를 복원하고,
운율만 밀면 보코더가 음색을 복원한다 — 검증기 하나만 공격했을 때와 같은 실패다.

## 일반화

이 구조는 모델에 종속되지 않는다. 어떤 제로샷 복제기든
**참조 음성 → 화자 조건**을 뽑는 미분 가능한 경로가 있고, 거기가 표적이다.
GPT-SoVITS면 `cnhubert` + SoVITS `ref_enc`가 같은 자리다.
`ClonerTarget`을 하나 더 구현하면 같은 최적화 루프가 그대로 돈다.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
XTTS_SR = 22050          # XTTS가 참조 음성을 읽는 표본화율


# ── 표적 인터페이스 ───────────────────────────────────────────────────────────

@dataclass
class Conditioning:
    """복제기가 참조 음성에서 뽑은 화자 조건."""

    speaker: torch.Tensor        # 음색 임베딩
    prosody: torch.Tensor | None  # 말투·운율 잠재 (없는 모델도 있다)


class ClonerTarget:
    """복제 모델의 화자 조건화 경로. 모델마다 하나씩 구현한다."""

    name = "?"

    def conditioning(self, wav16k: torch.Tensor) -> Conditioning:
        raise NotImplementedError


class XttsTarget(ClonerTarget):
    """XTTS-v2 — 보코더 화자 임베딩 + GPT 조건 잠재."""

    name = "xtts-v2"

    def __init__(self, device: torch.device) -> None:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        path, _, _ = ModelManager().download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2")
        cfg = XttsConfig()
        cfg.load_json(os.path.join(path, "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=path, eval=True)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

        # inference_mode 래퍼를 벗겨야 gradient가 흐른다.
        gpt = self.model.get_gpt_cond_latents
        self._gpt_cond = getattr(gpt, "__wrapped__", None)

    def conditioning(self, wav16k: torch.Tensor) -> Conditioning:
        import torchaudio

        w = wav16k.to(self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)

        # ① 보코더 화자 임베딩 — 16 kHz를 그대로 먹는다.
        #    resnet 내부가 squeeze_()로 in-place를 쓰므로 복제해서 넘긴다.
        spk = self.model.hifigan_decoder.speaker_encoder.forward(
            w.clone(), l2_norm=True)

        # ② GPT 조건 잠재 — XTTS는 22.05 kHz로 참조를 읽는다.
        pros = None
        if self._gpt_cond is not None:
            w22 = torchaudio.functional.resample(w, SAMPLE_RATE, XTTS_SR)
            pros = self._gpt_cond(self.model, w22, XTTS_SR, length=6, chunk_length=6)
        return Conditioning(speaker=spk.flatten(), prosody=None if pros is None
                            else pros.flatten())


class GptSovitsTarget(ClonerTarget):
    """GPT-SoVITS — 한국어 복제의 사실상 표준.

    XTTS와 **구조가 다르다.** 여기가 중요하다 — 한 모델만 막으면
    그 모델에 대한 과적합이지 방어가 아니라는 것을 검증기 쪽에서 세 번 겪었다.

    | | XTTS-v2 | GPT-SoVITS |
    |---|---|---|
    | 음색 | HiFi-GAN 화자 인코더 (ResNet, 파형 입력) | `ref_enc` MelStyleEncoder (**선형 스펙트로그램** 입력) |
    | 내용 | GPT 조건 잠재 (mel) | cnhubert SSL 특징 |
    | 표본화율 | 22.05 kHz 참조 | 32 kHz 스펙트로그램 |

    `ref_enc`는 참조 음성의 **선형 스펙트로그램**에서 음색 벡터를 뽑는다.
    XTTS의 파형 입력 인코더와 겹치는 곳이 거의 없으므로,
    둘을 함께 공격하면 "구조가 다른 두 모델에 동시에 통하는 방향"이 된다.
    """

    name = "gpt-sovits"
    GSV_SR = 32000          # SoVITS가 쓰는 표본화율

    def __init__(self, device: torch.device) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "GPT_SoVITS"))
        from huggingface_hub import snapshot_download
        from GPT_SoVITS.module.models import SynthesizerTrn

        root = Path(snapshot_download("lj1995/GPT-SoVITS",
                                      allow_patterns=["s2G488k.pth"]))
        ckpt = torch.load(root / "s2G488k.pth", map_location="cpu",
                          weights_only=False)
        hps = ckpt["config"]
        # 체크포인트의 config는 dict일 수도 HParams일 수도 있다.
        def g(o, k, d=None):
            return o[k] if isinstance(o, dict) else getattr(o, k, d)
        data, model = g(hps, "data"), g(hps, "model")
        mdict = model if isinstance(model, dict) else vars(model)
        self.model = SynthesizerTrn(
            g(data, "filter_length") // 2 + 1,
            32,                                # segment_size는 추론에 안 쓰인다
            n_speakers=g(data, "n_speakers", 0) or 0,
            version="v1",          # s2G488k는 v1 체크포인트다. v2로 만들면
            **mdict,               # ref_enc 입력 차원이 1025→704로 어긋난다
        )
        # 우리는 ref_enc만 쓴다. 텍스트 임베딩 등 나머지는 없어도 된다.
        missing, unexpected = self.model.load_state_dict(ckpt["weight"], strict=False)
        need = [k for k in missing if k.startswith("ref_enc")]
        if need:
            raise RuntimeError(f"ref_enc 가중치가 비었다: {need[:3]}")
        self.model.to(device).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.device = device
        self.n_fft = g(data, "filter_length")
        self.hop = g(data, "hop_length")
        self.win = g(data, "win_length")
        self.register = torch.hann_window(self.win).to(device)

    def _spec(self, w32: torch.Tensor) -> torch.Tensor:
        """선형 스펙트로그램 — GPT-SoVITS의 `spectrogram_torch`와 같은 설정."""
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(w32.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.register, center=False, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        return torch.sqrt(torch.abs(spec) ** 2 + 1e-8)

    def conditioning(self, wav16k: torch.Tensor) -> Conditioning:
        import torchaudio

        w = wav16k.to(self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        w32 = torchaudio.functional.resample(w, SAMPLE_RATE, self.GSV_SR)
        spec = self._spec(w32)
        # ref_enc는 (B, C, T) 스펙트로그램을 먹고 음색 벡터를 낸다.
        # 마스크는 넘기지 않는다 — `mask.int() == 0`을 "가림"으로 쓰는 구조라
        # zeros를 넘기면 전 구간이 가려져 출력이 상수가 되고 gradient가 끊긴다.
        ge = self.model.ref_enc(spec * 1.0, None)
        return Conditioning(speaker=ge.flatten(), prosody=None)


class MultiTarget(ClonerTarget):
    """여러 복제기를 **동시에** 공격한다.

    한 모델만 막으면 그 모델에 대한 과적합이다 — 검증기 쪽에서 세 번 겪었다
    (단독→ECAPA 0.8131, 2개→WavLM 0.9620). 복제기라고 다를 이유가 없다.

    XTTS와 GPT-SoVITS는 음색을 뽑는 방식이 다르다 —
    파형 입력 ResNet 대 선형 스펙트로그램 입력 MelStyleEncoder.
    둘을 함께 밀면 **구조가 다른 두 모델에 동시에 통하는 방향**만 남는다.
    """

    name = "multi"

    def __init__(self, targets: list[ClonerTarget]) -> None:
        self.targets = targets
        self.model = targets[0].model      # device 조회용

    def conditioning(self, wav16k: torch.Tensor) -> Conditioning:
        raise NotImplementedError("MultiTarget은 all_conditioning을 쓴다")

    def all_conditioning(self, wav16k: torch.Tensor) -> list[Conditioning]:
        return [t.conditioning(wav16k) for t in self.targets]


class WavlmVerifierTarget(ClonerTarget):
    """WavLM-SV — **복제기가 아니라 화자 검증기**다. 표적에 넣는 이유가 따로 있다.

    실제 사람 목소리로 재보니 두 검증기(Resemblyzer·ECAPA)는 넘겼는데
    WavLM만 0.9538로 남았다 — 원본(0.9271)보다도 높다.

    오늘 네 번 확인한 사실이 그대로 적용된다. **표적에 없는 모델은 안 밀린다.**
    XTTS와 GPT-SoVITS만 표적으로 삼았으니 WavLM이 안 밀리는 것은 당연하다.

    ## 방법론 주의 — 이걸 넣으면 WavLM은 독립 증거가 아니다

    WavLM은 지금 **채점자**로 쓰고 있다. 그걸 표적에 넣으면 자기 채점표를
    직접 최적화하는 셈이라, 숫자는 좋아지는데 방어가 나아졌다는 보장은 없다.

    그래서 이 표적을 쓸 때는 **WavLM 점수를 독립 근거로 인용하면 안 된다.**
    남는 독립 증거는 실제 복제음이다 — XTTS로 복제해 만든 음성이
    원본 화자와 얼마나 닮았는지가 최종 판정이다.
    """

    name = "wavlm-sv"

    def __init__(self, device: torch.device) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from mirinae.encoder import WavlmEncoder

        self.enc = WavlmEncoder(device=device)
        self.model = self.enc.model
        self.device = device

    def conditioning(self, wav16k: torch.Tensor) -> Conditioning:
        return Conditioning(speaker=self.enc(wav16k).flatten(), prosody=None)


TARGETS = {"xtts": XttsTarget, "gsv": GptSovitsTarget, "wavlm": WavlmVerifierTarget}


# ── 최적화 ────────────────────────────────────────────────────────────────────

def _mask_project(delta: torch.Tensor, thr_mag: torch.Tensor, ratio: float,
                  model) -> torch.Tensor:
    """섭동을 **마스킹 임계값 아래로** 눌러 넣는다.

    SNR만 맞추면 세기는 같아도 **들릴 자리**에 에너지가 몰린다.
    실제로 그랬다 — 마스킹 없이 SNR 20 dB로 맞춘 섭동의 임계값 위반이
    67.3%였다(마스킹을 쓴 쪽은 3.2%). 같은 세기인데 한쪽만 들린다.

    주파수별로 허용 상한을 정하고 넘는 성분만 눌러 담는다.
    위상은 건드리지 않는다 — 위상을 바꾸면 최적화가 찾은 방향이 무너진다.
    """
    spec = model.stft(delta)
    mag = torch.abs(spec)
    bound = torch.clamp(thr_mag * ratio, min=1e-10)
    scale = torch.clamp(bound / torch.clamp(mag, min=1e-10), max=1.0)
    return model.istft(spec * scale, length=delta.shape[-1])


def _snr_project(delta: torch.Tensor, x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """섭동 세기를 목표 SNR로 맞춘다. 세기 제약은 이것 하나로 통일한다."""
    sig = float(torch.mean(x ** 2))
    target = sig / (10.0 ** (snr_db / 10.0))
    cur = float(torch.mean(delta ** 2))
    if cur <= 1e-12:
        return delta
    return delta * float((target / cur) ** 0.5)


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0]


def attack(x: torch.Tensor, target: ClonerTarget, steps: int = 120,
           snr_db: float = 20.0, prosody_weight: float = 1.0,
           masking_ratio: float | None = 3.0,
           progress: bool = True,
           on_step=None,
           init_delta: torch.Tensor | None = None,
           eot: bool = False) -> tuple[torch.Tensor, dict]:
    """복제기의 화자 조건에서 멀어지도록 파형을 민다.

    손실은 **원본 조건과의 코사인 유사도**다. 두 경로를 함께 내린다 —
    한쪽만 밀면 다른 쪽이 화자를 복원한다.
    """
    device = next(iter(getattr(target, "model").parameters())).device
    x = x.to(device)
    multi = isinstance(target, MultiTarget)
    cond = target.all_conditioning if multi else (lambda w: [target.conditioning(w)])
    with torch.no_grad():
        refs = cond(x)

    model = thr = None
    if masking_ratio is not None:
        from mirinae.psychoacoustic import MaskingModel
        model = MaskingModel()
        thr, _ = model.threshold(x.cpu())
        thr = thr.to(device)

    # 시작점. 준실시간 청크 보호에서 **앞 청크가 찾은 방향**을 물려받는 데 쓴다.
    # 청크마다 0에서 출발하면 미는 방향이 제각각이 되고, 복제기가 파일 전체를
    # 볼 때 서로 상쇄된다 — 검증기 표적으로 이미 겪은 실패다.
    if init_delta is None:
        delta = torch.zeros_like(x, requires_grad=True)
    else:
        d0 = init_delta.to(device)
        if len(d0) < len(x):
            d0 = d0.repeat(int(np.ceil(len(x) / len(d0))))[:len(x)]
        delta = d0[:len(x)].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=1e-3)

    hist = {"speaker": [], "prosody": []}
    def total_loss(w: torch.Tensor) -> tuple[torch.Tensor, list[float], list[float]]:
        """모든 표적의 화자·운율 유사도를 함께 내린다."""
        curs = cond(w)
        loss = torch.zeros((), device=device)
        sp, pr = [], []
        for c, r in zip(curs, refs):
            l_s = _cos(c.speaker, r.speaker)
            loss = loss + l_s
            sp.append(float(l_s.detach()))
            if c.prosody is not None and r.prosody is not None:
                l_p = _cos(c.prosody, r.prosody)
                loss = loss + prosody_weight * l_p
                pr.append(float(l_p.detach()))
        return loss, sp, pr

    def eot_view(w: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        """공격자가 실제로 손에 넣을 법한 형태로 한 번 망가뜨린다.

        **왜 최적화 안에서 하나.** 깨끗한 파일에서만 최적화한 섭동은
        mp3·전화대역·잡음을 거치면 씻겨 나간다. 15회 실험에서 그걸 봤다 —
        열화 없는 조건은 막았는데 `C-noise`·`C-phone`이 뚫렸다.

        손잡이(스텝·SNR·배율)를 돌려도 낫지 않았다. 오히려 mp3는
        **나빠졌다**(0.8431 → 0.9010). 강도의 문제가 아니라 **강건성**의
        문제이기 때문이다. 그러면 열화를 손실 안으로 들여와야 한다.

        매 스텝 무작위로 하나를 골라 적용한다. 그래야 섭동이
        "이 한 가지 열화"가 아니라 **열화 전반**을 견디는 방향으로 자란다.
        전부 미분 가능해야 해서 mp3는 대역제한+양자화로 근사한다.
        """
        import torchaudio.functional as AF

        # 절반은 **두 겹**으로 건다. 실제로는 전화 녹음이 mp3로 넘어가고
        # 그걸 다시 담는 식으로 겹쳐 오기 때문이다.
        if int(torch.randint(0, 2, (1,), generator=g)) == 1:
            return _one(_one(w, g), g)
        return _one(w, g)

    def _one(w: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        import torchaudio.functional as AF

        k = int(torch.randint(0, 5, (1,), generator=g, device="cpu"))
        if k == 0:                                   # 그대로
            return w
        if k == 1:                                   # 전화 대역 3.4 kHz
            return AF.lowpass_biquad(w, SAMPLE_RATE, 3400.0)
        if k == 2:                                   # 잡음 (SNR 20~30 dB)
            snr = 20.0 + 10.0 * float(torch.rand(1, generator=g))
            n = torch.randn(w.shape, generator=g, device="cpu").to(w.device)
            scale = (w.detach().pow(2).mean() /
                     (n.pow(2).mean() * 10 ** (snr / 10))).sqrt()
            return w + scale * n
        if k == 3:                                   # 앞뒤를 잘라 낸 조각
            n = w.shape[-1]
            span = max(int(0.4 * n), SAMPLE_RATE * 3)
            if span >= n:
                return w
            s = int(torch.randint(0, n - span, (1,), generator=g))
            return w[..., s:s + span]
        # k == 4 — mp3 근사. 세기를 양자화해 미세한 값을 지운다.
        band = AF.lowpass_biquad(w, SAMPLE_RATE, 7000.0)
        q = 1.0 / 512
        return band + (torch.round(band / q) * q - band).detach()

    gen = torch.Generator().manual_seed(0) if eot else None

    for step in range(steps):
        if on_step is not None:
            on_step(step + 1, steps)      # 앱 진행 막대를 여기서 움직인다
        w = x + delta
        loss, _, _ = total_loss(eot_view(w, gen) if eot else w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if model is not None:
                delta.data = _mask_project(delta.data, thr, masking_ratio, model)
            delta.data = _snr_project(delta.data, x, snr_db)
            delta.data = torch.clamp(x + delta.data, -1.0, 1.0) - x
        if progress and (step % 10 == 0 or step == steps - 1):
            with torch.no_grad():
                _, sp, pr = total_loss(x + delta)
            names = ([t.name for t in target.targets] if multi else [target.name])
            cells = " · ".join(f"{n} {v:+.4f}" for n, v in zip(names, sp))
            tail = f" | 운율 {pr[0]:+.4f}" if pr else ""
            print(f"  [{step + 1:4}/{steps}] {cells}{tail}")

    with torch.no_grad():
        protected = torch.clamp(x + delta, -1.0, 1.0)
        _, sp, pr = total_loss(protected)
        names = ([t.name for t in target.targets] if multi else [target.name])
        final = {
            "speaker": dict(zip(names, sp)),
            "prosody": (pr[0] if pr else None),
            "snr_db": float(10 * torch.log10(
                torch.mean(x ** 2) / torch.clamp(torch.mean((protected - x) ** 2),
                                                 min=1e-12))),
        }
    return protected.cpu(), final


def main() -> int:
    p = argparse.ArgumentParser(
        description="복제 모델의 화자 조건화 경로를 직접 공격한다")
    p.add_argument("input")
    p.add_argument("-o", "--out", default="out/attack")
    p.add_argument("--target", default="xtts",
                   help="쉼표로 여러 개 — 예: xtts,gsv (동시 공격)")
    p.add_argument("--eot", action="store_true",
                   help="열화(전화대역·잡음·절단·mp3)를 최적화 안에 넣는다")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--prosody-weight", type=float, default=1.0,
                   help="운율 경로 가중치. 0이면 음색만 민다")
    p.add_argument("--masking-ratio", type=float, default=3.0,
                   help=("심리음향 마스킹 배율. 0이면 마스킹을 끈다 — "
                         "SNR만 맞추면 같은 세기라도 들릴 자리에 몰린다"
                         "(실측 위반율 67.3% 대 3.2%)."))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device} · 표적: {args.target}")

    x, sr = sf.read(args.input, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    if args.seconds:
        x = x[: int(SAMPLE_RATE * args.seconds)]
    xt = torch.from_numpy(np.ascontiguousarray(x))
    print(f"입력: {args.input} · {len(x) / SAMPLE_RATE:.1f}초")

    names = [n.strip() for n in args.target.split(",") if n.strip()]
    built = [TARGETS[n](device) for n in names]
    target = built[0] if len(built) == 1 else MultiTarget(built)
    print(f"{args.steps}스텝 · 목표 SNR {args.snr} dB · 운율 가중치 {args.prosody_weight}")
    protected, final = attack(xt, target, steps=args.steps, snr_db=args.snr,
                              prosody_weight=args.prosody_weight,
                              masking_ratio=args.masking_ratio or None,
                              eot=args.eot)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "original.wav"), x, SAMPLE_RATE)
    sf.write(str(out / "protected.wav"), protected.numpy(), SAMPLE_RATE)

    pr = final["prosody"]
    print()
    print("복제기 내부 조건 (원본 대비 코사인 · 낮을수록 다른 화자로 인식)")
    for n, v in final["speaker"].items():
        print(f"  화자 임베딩 [{n}]  {v:+.4f}")
    print(f"  운율 잠재    {pr:+.4f}" if pr is not None else "  운율 잠재    —")
    print(f"  SNR         {final['snr_db']:.2f} dB")
    print(f"\n출력: {out.resolve()}")
    print("다음 — 실제 복제로 확인:")
    print(f"  .venv-xtts\\Scripts\\python clone_test.py {out}/original.wav "
          f"{out}/protected.wav --repeat 5 -o out/clone_attack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
