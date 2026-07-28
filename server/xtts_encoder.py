"""XTTS-v2 화자 인코더 어댑터 — **.venv-xtts 전용.**

블랙박스 전이가 실패했으므로 표적을 바꾼다.

D09 §01은 XTTS-v2 conditioning을 "비공개 · gradient 접근 여부가 최대 리스크"로 적어두고
미확인이면 블랙박스 전이 공격이라고 봤다. 실제로 전이는 실패했다
(보호본 복제 SRS 0.9135 — 잡음 대조군보다도 못함).

그런데 **XTTS-v2는 가중치가 공개되어 있다.** 이미 로컬에 받아둔 모델을 열어보면
화자 임베딩 경로가 그대로 노출된다. 블랙박스로 둘 이유가 없다.

    hifigan_decoder.speaker_encoder(wav_16k, l2_norm=True) → (1, 512)

이건 파형을 직접 받는 신경망이므로 gradient가 파형까지 닿는다.
전이를 기대하는 대신 **실제 표적을 직접 공략**한다.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

os.environ.setdefault("COQUI_TOS_AGREED", "1")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SR = 16_000        # speaker_encoder가 받는 표본화율


def load_xtts(device: torch.device | None = None):
    """XTTS 모델 본체를 로드한다. TTS.api 래퍼 안쪽의 raw 모델이 필요하다."""
    from TTS.api import TTS

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = TTS(MODEL).to(device)

    # TTS.api → synthesizer → tts_model (Xtts)
    model = getattr(getattr(api, "synthesizer", None), "tts_model", None)
    if model is None:
        raise RuntimeError(
            "XTTS 모델 본체를 찾지 못했다. coqui-tts 버전이 바뀌어 경로가 달라졌을 수 있다. "
            "api.synthesizer.tts_model 경로를 확인할 것."
        )
    return model, device


class XttsSpeakerEncoder(nn.Module):
    """XTTS 내부 화자 인코더를 미분 가능한 형태로 감싼다.

    Resemblyzer 래퍼(mirinae.encoder.SpeakerEncoder)와 같은 인터페이스를 갖추어
    EncoderEnsemble에 그대로 꽂을 수 있게 한다 — `name`, `__call__(wav) -> (D,)`.
    """

    name = "xtts_speaker_encoder"

    def __init__(self, model=None, device: torch.device | None = None) -> None:
        super().__init__()
        if model is None:
            model, device = load_xtts(device)
        self.device = device or torch.device("cpu")
        self.model = model

        enc = getattr(getattr(model, "hifigan_decoder", None), "speaker_encoder", None)
        if enc is None:
            raise RuntimeError(
                "hifigan_decoder.speaker_encoder를 찾지 못했다. "
                "모델 구조가 바뀌었는지 확인할 것."
            )
        self.encoder = enc
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)     # 최적화 대상은 입력이지 가중치가 아니다

        self.embedding_dim = self._probe_dim()

    def _probe_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, XTTS_SR, device=self.device)
            out = self.encoder.forward(dummy, l2_norm=True)
        return int(out.reshape(-1).shape[0])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(n,) 16 kHz 파형 → L2 정규화된 화자 임베딩 (D,).

        `wav.requires_grad`가 켜져 있으면 파형까지 gradient가 흐른다.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        emb = self.encoder.forward(wav.to(self.device), l2_norm=True)
        return emb.reshape(-1)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward(wav)


class XttsGptCondEncoder(nn.Module):
    """GPT conditioning latent 경로 — **이쪽이 진짜 표적일 가능성이 높다.**

    speaker_encoder만 공략했을 때 그 임베딩은 SRS 0.4649까지 무너졌는데
    실제 복제음은 원본과 0.8933으로 거의 그대로였다.
    즉 XTTS는 화자 임베딩 하나로 목소리를 정하지 않는다.
    GPT 조건 latent가 음색·운율을 함께 실어 나르므로 여기도 같이 흔들어야 한다.

    speaker_encoder(16 kHz 파형)와 달리 이쪽은 22.05 kHz mel을 쓴다.
    resample과 mel 모두 torch 연산이라 gradient는 파형까지 그대로 흐른다.
    """

    name = "xtts_gpt_cond"

    # XTTS의 wav_to_mel_cloning 호출 인자 (Xtts.get_gpt_cond_latents와 동일하게 맞춘다)
    MEL_KWARGS = dict(
        n_fft=4096, hop_length=1024, win_length=4096, power=2,
        normalized=False, sample_rate=22_050, f_min=0, f_max=8000, n_mels=80,
    )

    def __init__(self, model, device: torch.device) -> None:
        super().__init__()
        from TTS.tts.models.xtts import wav_to_mel_cloning

        self._wav_to_mel = wav_to_mel_cloning
        self.model = model
        self.device = device
        self.mel_norms = model.mel_stats.to(device)
        self.embedding_dim = 0

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        import torchaudio

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav22 = torchaudio.functional.resample(wav, XTTS_SR, 22_050)
        mel = self._wav_to_mel(wav22, mel_norms=self.mel_norms, **self.MEL_KWARGS)
        latent = self.model.gpt.get_style_emb(mel.to(self.device))   # (1, 1024, T)
        # 시간축 평균 → 화자 임베딩과 같은 형태의 벡터로 만든다.
        # 코사인 손실을 그대로 쓸 수 있어야 하므로 (D,)로 편다.
        return latent.transpose(1, 2).mean(dim=1).reshape(-1)


def build_target(model, device: torch.device, use_gpt_cond: bool = False) -> list[nn.Module]:
    """공략 표적 목록. 기본은 speaker_encoder 하나."""
    targets: list[nn.Module] = [XttsSpeakerEncoder(model, device)]
    if use_gpt_cond:
        try:
            targets.append(XttsGptCondEncoder(model, device))
        except Exception as e:      # 경로가 바뀌었어도 speaker_encoder만으로 진행한다
            print(f"  ※ GPT cond 경로 사용 불가 ({e}) — speaker_encoder만 공략한다")
    return targets
