"""화자 인코더 — 임베딩 + gradient 핸들.

모드 2가 공략하는 표적이 바로 이 인코더다. 제로샷 복제는 여기서 나온 벡터를
조건으로 음성을 생성하므로, **벡터만 틀어지면 합성 결과 전체가 다른 목소리가 된다.**

문제는 Resemblyzer의 기본 경로가 gradient를 끊는다는 것이다.
`wav_to_mel_spectrogram`이 librosa(numpy)로 mel을 뽑기 때문에 파형까지 역전파가 닿지 않는다.
그래서 mel 추출을 torch로 다시 구현하되, **필터뱅크는 librosa에서 그대로 가져와**
수치가 어긋날 여지를 없앤다. 일치 여부는 `verify_against_resemblyzer()`로 검증한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import EPS, SAMPLE_RATE, default_device

# Resemblyzer hparams와 반드시 일치해야 한다 (resemblyzer/hparams.py).
MEL_WINDOW_LENGTH_MS = 25
MEL_WINDOW_STEP_MS = 10
MEL_N_CHANNELS = 40
PARTIALS_N_FRAMES = 160          # 1.6초 — 청크 길이 2.0초를 결정한 근거
AUDIO_NORM_TARGET_DBFS = -30.0

N_FFT_MEL = int(SAMPLE_RATE * MEL_WINDOW_LENGTH_MS / 1000)   # 400
HOP_MEL = int(SAMPLE_RATE * MEL_WINDOW_STEP_MS / 1000)       # 160


class SpeakerEncoder(torch.nn.Module):
    """Resemblyzer VoiceEncoder를 미분 가능한 경로로 감싼다.

    파라미터 1.4M · 임베딩 256차원. 가벼워서 200스텝을 감당한다 — 1차 최적화 담당.
    """

    name = "resemblyzer"
    embedding_dim = 256

    def __init__(self, device: torch.device | None = None, verbose: bool = False) -> None:
        super().__init__()
        from resemblyzer import VoiceEncoder   # import 비용이 커서 지연 로드한다

        self.device = device or default_device()
        self.model = VoiceEncoder(device=self.device, verbose=verbose)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)            # 우리가 최적화하는 것은 입력이지 가중치가 아니다

        # librosa의 Slaney mel 필터뱅크를 그대로 쓴다. 직접 구현하면 미세하게 어긋난다.
        import librosa

        fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT_MEL, n_mels=MEL_N_CHANNELS,
            fmin=0.0, fmax=SAMPLE_RATE / 2, htk=False, norm="slaney",
        )
        self.register_buffer("mel_fb", torch.as_tensor(fb, dtype=torch.float32,
                                                       device=self.device))
        self.register_buffer("window", torch.hann_window(N_FFT_MEL, periodic=True,
                                                         device=self.device))

    # ── mel ───────────────────────────────────────────────────────────────────

    def mel_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """librosa.feature.melspectrogram과 동일한 결과를 미분 가능하게 낸다.

        주의 — Resemblyzer는 **로그를 취하지 않은** 선형 power mel을 그대로 먹는다.
        여기서 log를 씌우면 임베딩이 전혀 다른 값이 된다.

        :return: (n_frames, 40)
        """
        spec = torch.stft(
            wav, n_fft=N_FFT_MEL, hop_length=HOP_MEL, win_length=N_FFT_MEL,
            window=self.window, center=True, pad_mode="constant", return_complex=True,
        )
        power = spec.abs() ** 2                      # (bins, frames)
        mel = self.mel_fb @ power                    # (40, frames)
        return mel.transpose(0, 1)                   # (frames, 40)

    # ── 임베딩 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _partial_slices(n_frames: int, rate: float = 1.3,
                        min_coverage: float = 0.75) -> list[tuple[int, int]]:
        """Resemblyzer.compute_partial_slices의 mel 슬라이스 부분을 옮긴 것.

        2.0초 청크(약 201프레임)면 [0,160]과 [77,237] 두 조각이 나온다 — 온전한 1.25개.
        """
        frame_step = max(1, int(round((SAMPLE_RATE / HOP_MEL) / rate)))
        steps = max(1, n_frames - PARTIALS_N_FRAMES + frame_step + 1)

        slices: list[tuple[int, int]] = []
        for i in range(0, steps, frame_step):
            slices.append((i, i + PARTIALS_N_FRAMES))

        # 마지막 조각이 너무 짧으면 버린다(단, 최소 한 개는 남긴다)
        if len(slices) > 1:
            last_start, last_end = slices[-1]
            coverage = (n_frames - last_start) / (last_end - last_start)
            if coverage < min_coverage:
                slices.pop()
        return slices

    def normalize_volume(self, wav: torch.Tensor) -> torch.Tensor:
        """Resemblyzer.preprocess_wav의 음량 정규화(-30 dBFS, 증폭만).

        공격자도 복제 전에 이 전처리를 거친다. 최적화 시점에 같은 조건을 맞춰 둔다.
        스칼라 곱이라 gradient는 그대로 흐른다.
        """
        rms = torch.sqrt(torch.clamp((wav ** 2).mean(), min=EPS))
        dbfs_change = AUDIO_NORM_TARGET_DBFS - 20.0 * torch.log10(rms)
        # 분기 판정에만 값을 쓰고 gradient는 아래 곱셈으로 흘린다
        if float(dbfs_change.detach()) < 0:
            return wav                                # increase_only=True
        return wav * (10.0 ** (dbfs_change / 20.0))

    def forward(self, wav: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """파형 → L2 정규화된 화자 임베딩 (256,).

        `wav.requires_grad`가 켜져 있으면 파형까지 gradient가 닿는다. 이게 이 클래스의 존재 이유다.
        """
        if normalize:
            wav = self.normalize_volume(wav)

        mel = self.mel_spectrogram(wav)               # (frames, 40)
        n_frames = mel.shape[0]
        slices = self._partial_slices(n_frames)

        need = max(end for _, end in slices)
        if need > n_frames:                            # 마지막 조각을 0으로 채운다
            mel = F.pad(mel, (0, 0, 0, need - n_frames))

        batch = torch.stack([mel[s:e] for s, e in slices])   # (n_partials, 160, 40)
        partial_embeds = self.model(batch)                   # (n_partials, 256), 각각 L2=1

        embed = partial_embeds.mean(dim=0)
        return embed / torch.clamp(torch.norm(embed), min=EPS)

    def embed(self, wav: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.forward(wav, normalize=normalize)

    # ── 검증 ──────────────────────────────────────────────────────────────────

    def verify_against_resemblyzer(self, wav_np: np.ndarray) -> dict[str, float]:
        """torch mel이 librosa mel과 같은 값을 내는지 실제로 확인한다.

        여기가 어긋나면 우리가 최적화하는 대상이 실제 인코더가 아니게 되고,
        섭동은 아무 데도 통하지 않는다. 조용히 틀리는 종류의 결함이라 반드시 계측한다.
        """
        from resemblyzer.audio import wav_to_mel_spectrogram

        ref = wav_to_mel_spectrogram(wav_np.astype(np.float32))          # (frames, 40)
        with torch.no_grad():
            got = self.mel_spectrogram(
                torch.as_tensor(wav_np, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        n = min(len(ref), len(got))
        ref, got = ref[:n], got[:n]
        denom = max(float(np.abs(ref).max()), EPS)
        return {
            "n_frames_ref": float(len(ref)),
            "n_frames_got": float(len(got)),
            "max_abs_err": float(np.abs(ref - got).max()),
            "rel_err": float(np.abs(ref - got).max() / denom),
            "corr": float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]),
        }


class EcapaEncoder(torch.nn.Module):
    """ECAPA-TDNN (SpeechBrain · VoxCeleb) — **구조가 다른** 두 번째 인코더.

    ## 왜 붙였나

    모드 2는 SRS를 0.37까지 떨어뜨리는데 **실제 복제 저지율(DSR)은 0%** 였다.
    화자 검증기는 속이는데 복제 모델은 안 속는다. 전이성 진단으로 원인이 나왔다.

        보호본 out/ref        Resemblyzer 0.6342   ECAPA-TDNN 0.8131
                             (판정 임계값 0.7962)

    **한쪽은 통과하고 다른 쪽은 실패한다.** Resemblyzer의 임베딩 공간에서만
    멀어지도록 최적화했으니 당연한 결과다 — 복제 모델은 Resemblyzer를 쓰지 않는다.
    한 모델만 속이는 것은 방어가 아니라 그 모델에 대한 과적합이다.

    ## 왜 이 모델인가

    Resemblyzer와 겹치는 곳이 최소한이어야 앙상블의 의미가 있다.

    | | Resemblyzer (GE2E) | ECAPA-TDNN |
    |---|---|---|
    | 구조 | 3층 LSTM | TDNN + SE-Res2Block + attentive pooling |
    | 특징 | 선형 power mel 40 | log mel-fbank 80 |
    | 학습 | LibriSpeech 등 | VoxCeleb 1+2 |
    | 임베딩 | 256차원 | 192차원 |

    ## 미분 경로

    `encode_batch()`는 내부에서 `torch.no_grad()`를 쓸 수 있고 길이 정규화도
    끼어든다. 그래서 하위 모듈(`compute_features` → `mean_var_norm` →
    `embedding_model`)을 직접 호출한다. 셋 다 순수 torch 모듈이라
    파형까지 역전파가 닿는다. 일치 여부는 `verify_against_speechbrain()`가 확인한다.
    """

    name = "ecapa"
    embedding_dim = 192

    @staticmethod
    def _drop_broken_lazy_modules() -> None:
        """SpeechBrain의 **해결 불가능한 선택 의존성**이 터지지 않게 막는다.

        speechbrain은 k2 같은 선택적 의존성을 `LazyModule`로 등록한다.
        그런데 `inspect.getmodule()`처럼 sys.modules를 훑으며
        `hasattr(module, "__file__")`를 부르는 코드가 그것을 건드리면
        `__getattr__`이 실제 import를 시도하다 ImportError를 던진다.

        librosa가 내부에서 스택을 순회하므로 **인코더와 아무 상관 없는 곳에서
        터진다.** 실제로 세 번 겪었다 — 리샘플링에서 한 번, PGD 파이프라인에서 두 번.

        sys.modules에서 지우는 것으로는 안 된다. 부모 패키지가 다시 만들어 낸다.
        그래서 `ensure_module` 자체를 감싸 **실패하면 예외 대신 빈 모듈**을
        돌려주게 한다. 우리는 k2를 쓰지 않으므로 잃는 기능이 없다.
        """
        try:
            from speechbrain.utils import importutils
        except Exception:                             # noqa: BLE001
            return
        lazy = getattr(importutils, "LazyModule", None)
        if lazy is None or getattr(lazy, "_mirinae_patched", False):
            return

        import types as _types

        original = lazy.ensure_module

        def safe_ensure_module(self, *args, **kwargs):     # noqa: ANN001
            try:
                return original(self, *args, **kwargs)
            except Exception:                             # noqa: BLE001
                stub = _types.ModuleType(str(getattr(self, "target", "lazy")))
                stub.__file__ = "<speechbrain 선택 의존성 미설치>"
                return stub

        lazy.ensure_module = safe_ensure_module
        lazy._mirinae_patched = True

    def __init__(self, device: torch.device | None = None,
                 savedir: str = "models/ecapa") -> None:
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device or default_device()
        self.clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": str(self.device)},
        )
        self._drop_broken_lazy_modules()
        for m in self.clf.mods.values():
            m.eval()
            for prm in m.parameters():
                prm.requires_grad_(False)

    def forward(self, wav: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """(n_samples,) 파형 → (192,) 임베딩. gradient가 파형까지 닿는다."""
        x = wav.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feats = self.clf.mods.compute_features(x)
        feats = self.clf.mods.mean_var_norm(
            feats, torch.ones(x.shape[0], device=self.device))
        emb = self.clf.mods.embedding_model(
            feats, torch.ones(x.shape[0], device=self.device))
        emb = emb.squeeze(0).squeeze(0)
        return F.normalize(emb, dim=0) if normalize else emb

    def verify_against_speechbrain(self, wav: np.ndarray, tol: float = 1e-3) -> float:
        """공식 경로와 우리 경로가 같은 임베딩을 내는지 확인한다.

        미분 가능하게 다시 짠 경로는 **반드시 원본과 대조해야 한다.**
        어긋나 있으면 엉뚱한 방향으로 최적화하면서도 숫자는 좋아 보인다.
        """
        with torch.no_grad():
            official = self.clf.encode_batch(
                torch.as_tensor(wav).unsqueeze(0)).squeeze()
            official = F.normalize(official, dim=0)
            ours = self(torch.as_tensor(wav))
        gap = float(torch.max(torch.abs(official - ours)))
        if gap > tol:
            raise AssertionError(f"ECAPA 미분 경로가 공식 경로와 어긋난다: {gap:.2e}")
        return gap


class WavlmEncoder(torch.nn.Module):
    """WavLM-SV (Microsoft · 자기지도 트랜스포머) — **세 번째** 인코더.

    ## 왜 셋째가 필요했나

    Resemblyzer + ECAPA 앙상블로 실제 복제음의 ECAPA 유사도를
    0.7667 → 0.6766까지 밀었다(신뢰구간 분리). 방향은 맞았다.
    그런데 **Resemblyzer 쪽이 0.9010으로 버텨** 저지율은 여전히 0%였다.

    인코더 둘로는 부족하다는 뜻이다. 그래서 셋째를 붙이되,
    앞의 둘과 최대한 겹치지 않는 것을 고른다.

    | | Resemblyzer | ECAPA-TDNN | **WavLM-SV** |
    |---|---|---|---|
    | 구조 | 3층 LSTM | TDNN + SE-Res2Block | **트랜스포머 12층** |
    | 특징 | 선형 mel 40 | log fbank 80 | **원시 파형(CNN 프런트엔드)** |
    | 학습 | 지도 (GE2E) | 지도 (VoxCeleb) | **자기지도 후 미세조정** |
    | 임베딩 | 256 | 192 | 512 |

    **특징 추출부터 다르다** — WavLM은 mel을 거치지 않고 파형을 직접 먹는다.
    mel 영역에서만 통하는 섭동은 여기서 걸러진다.

    ## 미분 경로

    `WavLMForXVector`는 파형 텐서를 그대로 받고 내부가 전부 torch 모듈이라
    별도 우회가 필요 없다. 다만 학습된 정규화(zero-mean unit-var)를 직접
    적용해야 특징 추출기와 값이 맞는다.
    """

    name = "wavlm"
    embedding_dim = 512

    def __init__(self, device: torch.device | None = None,
                 model_id: str = "microsoft/wavlm-base-plus-sv") -> None:
        super().__init__()
        from transformers import WavLMForXVector

        self.device = device or default_device()
        self.model = WavLMForXVector.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)

    def forward(self, wav: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """(n_samples,) 파형 → (512,) 임베딩. 16 kHz를 가정한다."""
        x = wav.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # 특징 추출기가 하는 정규화를 그대로 재현한다 (do_normalize=True).
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            x.var(dim=-1, keepdim=True) + 1e-7)
        emb = self.model(input_values=x).embeddings.squeeze(0)
        return F.normalize(emb, dim=0) if normalize else emb


class EncoderEnsemble(torch.nn.Module):
    """여러 인코더의 손실을 합산한다 — 전이성 확보용 (AntiFake 노선).

    한 모델만 속이면 방어가 아니라 과적합이다. 전이성 진단이 그것을 확인했다 —
    Resemblyzer만 보고 최적화한 보호본이 ECAPA-TDNN에는 0.8131로 남았다
    (판정 임계값 0.7962). 그래서 `build_ensemble()`이 기본으로 둘을 함께 쓴다.
    """

    def __init__(self, encoders: list[SpeakerEncoder]) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("인코더가 최소 하나는 있어야 한다")
        self.encoders = torch.nn.ModuleList(encoders)

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.encoders]

    def embeddings(self, wav: torch.Tensor) -> list[torch.Tensor]:
        return [e(wav) for e in self.encoders]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SRS — 두 임베딩의 코사인 유사도. 1에 가까울수록 같은 화자."""
    return torch.dot(a, b) / torch.clamp(torch.norm(a) * torch.norm(b), min=EPS)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - cosine_similarity(a, b)


def build_ensemble(device: torch.device | None = None,
                   with_ecapa: bool = True,
                   with_wavlm: bool = True) -> EncoderEnsemble:
    """기본 앙상블 — Resemblyzer + ECAPA-TDNN.

    ECAPA를 못 불러오면(패키지 부재·가중치 다운로드 실패) Resemblyzer 단독으로
    떨어지되 **조용히 넘어가지 않는다.** 단독으로 돌면 전이성이 없다는 것이
    측정으로 확인된 사실이라, 그 상태를 모르고 쓰면 안 된다.
    """
    encoders: list[torch.nn.Module] = [SpeakerEncoder(device=device)]
    import warnings

    optional = []
    if with_ecapa:
        optional.append(("ECAPA-TDNN", EcapaEncoder))
    if with_wavlm:
        optional.append(("WavLM-SV", WavlmEncoder))
    for label, cls in optional:
        try:
            encoders.append(cls(device=device))
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(
                f"{label}을 불러오지 못해 앙상블에서 빠집니다: {exc}. "
                "인코더가 적을수록 다른 복제 모델로 전이되지 않습니다 "
                "(단독 실측 ECAPA 0.8131 · 둘 실측 Resemblyzer 0.9010).",
                RuntimeWarning, stacklevel=2)
    return EncoderEnsemble(encoders)
