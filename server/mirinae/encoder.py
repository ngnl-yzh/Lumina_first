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


class EncoderEnsemble(torch.nn.Module):
    """여러 인코더의 손실을 합산한다 — 전이성 확보용 (AntiFake 노선).

    한 모델만 속이면 방어가 아니라 과적합이다. 지금은 Resemblyzer 단독으로 돌지만
    ECAPA-TDNN을 추가할 자리를 처음부터 열어 둔다.
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
