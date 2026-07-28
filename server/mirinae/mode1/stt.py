"""STT — faster-whisper.

D08 §03 파이프라인 3단계.

한국어 전사 정확도가 이 모드 전체의 상한을 정한다.
아무리 스코어러가 정교해도 "안전계좌"가 "안전 개좌"로 들리면 거기서 끝난다.
매처의 자모 근사매칭이 그걸 일부 흡수하지만, 애초에 덜 틀리는 편이 낫다.

정확도를 올리는 수단이 세 가지 있고 전부 지연과 맞바꾼다.
  ① 모델 크기 — base → small 이 한국어에서 가장 체감이 크다
  ② initial_prompt — 나올 법한 단어를 미리 알려준다. 우리는 패턴 DB가 있으니 공짜다
  ③ beam_size — 1보다 크면 후보를 더 넓게 본다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# base는 한국어에서 눈에 띄게 약하다. small이 기본값으로 적당하다.
# GPU가 있으면 medium까지 올릴 만하다 — RTX 3060에서 실시간을 유지한다.
DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ko"
DEFAULT_BEAM_SIZE = 3

# initial_prompt는 토큰 상한(약 224)이 있어 182개를 다 넣을 수 없다.
# 중요도 순으로 잘라 넣는다.
PROMPT_MAX_CHARS = 220


@dataclass
class Transcript:
    text: str
    start: float = 0.0
    end: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def is_reliable(self) -> bool:
        """신뢰할 수 없는 전사는 버린다.

        무음 구간에서 Whisper는 "감사합니다", "시청해주셔서 감사합니다" 같은
        환각 문장을 만들어낸다. 이게 패턴 DB에 걸리면 곧바로 오탐이 되므로 여기서 거른다.
        """
        return self.no_speech_prob < 0.6 and bool(self.text.strip())


def build_initial_prompt(db=None, max_chars: int = PROMPT_MAX_CHARS) -> str:
    """패턴 DB에서 Whisper에게 미리 알려줄 어휘를 만든다.

    **왜 효과가 있나.** Whisper는 앞 문맥을 보고 다음 토큰을 고른다.
    "안전계좌"를 미리 보여주면 비슷한 발음 후보 중 그쪽 확률이 올라간다.
    우리가 잡아야 하는 바로 그 단어들의 전사 정확도가 오르는 것이다.

    **부작용도 있다.** 프롬프트에 있는 단어를 없는데도 만들어낼 수 있다.
    그래서 고위험 신호(C1~C5)와 가중치 높은 단계의 대표 표현만 넣고,
    흔한 일반 명사는 넣지 않는다. 끄고 싶으면 initial_prompt=""로 두면 된다.
    """
    if db is None:
        from .patterns import load_db

        db = load_db()

    terms: list[str] = [c.text for c in db.criticals]

    # 가중치가 높은 단계부터 대표 표현을 담는다
    for sid in sorted(db.stages, key=lambda s: -db.stages[s].weight):
        for kw in db.stages[sid].keywords[:4]:
            if kw.text not in terms:
                terms.append(kw.text)

    out: list[str] = []
    total = 0
    for t in terms:
        if total + len(t) + 2 > max_chars:
            break
        out.append(t)
        total += len(t) + 2
    return ", ".join(out)


class SpeechToText:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        initial_prompt: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        # None이면 패턴 DB에서 자동 생성. 빈 문자열이면 프롬프트를 쓰지 않는다.
        self.initial_prompt = (
            build_initial_prompt() if initial_prompt is None else initial_prompt
        )
        self._device = device
        self._compute_type = compute_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from faster_whisper import WhisperModel

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            # float16은 CUDA에서만. CPU에서 지정하면 그대로 죽는다.
            compute = self._compute_type or ("float16" if device == "cuda" else "int8")
            self._model = WhisperModel(self.model_size, device=device,
                                       compute_type=compute)
        return self._model

    def transcribe(self, wav: np.ndarray, sample_rate: int = 16_000) -> list[Transcript]:
        """16 kHz 모노 float32 배열 → 전사 조각들."""
        if sample_rate != 16_000:
            raise ValueError(f"16 kHz만 받는다 (받은 값 {sample_rate})")

        segments, _ = self.model.transcribe(
            wav.astype(np.float32),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,               # VAD는 이미 앞단에서 했다
            condition_on_previous_text=False,  # 환각이 다음 발화로 번지는 것을 막는다
            initial_prompt=self.initial_prompt or None,
        )
        return [
            Transcript(
                text=s.text.strip(),
                start=s.start, end=s.end,
                avg_logprob=getattr(s, "avg_logprob", 0.0),
                no_speech_prob=getattr(s, "no_speech_prob", 0.0),
            )
            for s in segments
        ]

    def transcribe_text(self, wav: np.ndarray, sample_rate: int = 16_000) -> str:
        parts = [t.text for t in self.transcribe(wav, sample_rate) if t.is_reliable]
        return " ".join(parts).strip()


class NullSTT:
    """STT 없이 파이프라인을 돌려보기 위한 대체물.

    시연 리허설에서 STT가 문제인지 스코어러가 문제인지 가르는 데 쓴다.
    """

    def __init__(self, script: list[str] | None = None) -> None:
        self.script = list(script or [])
        self.index = 0

    def transcribe_text(self, wav, sample_rate: int = 16_000) -> str:
        if self.index >= len(self.script):
            return ""
        out = self.script[self.index]
        self.index += 1
        return out
