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

# initial_prompt는 토큰 상한(약 224)이 있어 전 항목을 다 넣을 수 없다.
# 중요도 순으로 잘라 넣는다.
PROMPT_MAX_CHARS = 220

# hotwords 기본 활성. **실측 근거가 있다.**
#
# 실사용 화면에서 오인식이 눈에 띄었다("카드가 1시 정지되어", "해외 결치 시도").
# `stt_tune.py`로 통화 채널 + 배경잡음 18 dB 조건에서 설정을 비교했다
# (시나리오 3건 · 사기 1 · 정상 2).
#
#   설정                 CER     키워드 적중   환각 키워드   초/통화
#   기본                29.0%      57%         0개       5.6
#   hotwords           15.8%      57%         0개       5.5   ← 채택
#   initial_prompt     15.0%      64%         0개       7.0
#   beam 5             28.5%      64%         0개       5.3
#
# **hotwords는 지연 비용 없이 CER을 절반으로 줄인다.**
# initial_prompt는 CER이 근소하게 더 낫지만 25% 느리고, 사용자가 지연도 문제로
# 지적했으므로 택하지 않았다.
#
# 깨끗한 음성에서는 차이가 거의 없다(CER 7.2% → 6.9%). **열화된 조건에서만 효과가 난다** —
# 그리고 실제 통화가 바로 그 조건이다. 예전 측정이 "효과 없음"으로 결론 난 이유가
# 이것이라고 본다. 다만 표본이 3건이라 넓은 재측정이 필요하다.
#
# ## 범위 — 전부 넣으면 안 된다
#
# 실사용에서 "자산"이 "사산"으로 들린 사례를 받아 범위를 스윕했다.
# 전부 넣으면 디코딩이 오염된다 — 800자에서 "범죄에"가 "검주의"로,
# "수사가"가 "주사가"로 바뀌었다(CER 20.2%로 최악).
# 어휘를 미는 것은 다른 단어를 밀어내는 것이기도 하다.
#
# ## 표본이 적어 설정 순위가 흔들린다 — 확정값이 아니다
#
# 시나리오 3건과 4건에서 두 번 쟀는데 **순위가 크게 뒤바뀌었다.**
# 예: "타임스탬프 없음"이 CER 16.0%·적중 86%에서 30.6%·적중 68%로.
#
# 두 실행에서 **일관된 것은 이것뿐이다.**
#
#   설정              실행A(3건)              실행B(4건)
#   hotwords 없음     CER 30.5% · 적중 57%    CER 43.1% · 적중 38%
#   critical만        CER 17.2% · 적중 57%    CER 18.0% · 적중 49%
#   상위6 · 400자     CER 20.0% · 적중 71%    CER 17.4% · 적중 59%   ← 채택
#
#   ① hotwords가 없으면 확실히 나쁘다. 두 실행 모두 큰 차이로.
#   ② 범위는 CER과 키워드 적중이 **서로 다른 답을 준다.**
#      CER만 보면 critical만이 낫고, 키워드 적중은 상위6이 두 실행 모두 높다.
#      **키워드 적중을 택한다** — 탐지율을 직접 정하는 것은 이쪽이다.
#      CER은 사람이 화면에서 읽는 품질이고, 매처는 자모 근사매칭으로 오차를 흡수한다.
#
# beam·타임스탬프·vad_filter·문맥유지는 실행마다 순위가 뒤집히고 일부는
# **환각 키워드를 만든다**(정답에 없는 위험 표현이 전사에 생긴다 = 곧 오탐).
# 구별되지 않는 것을 바꾸면 안 되므로 전부 기본값을 유지한다.
#
# **표본 4건은 여전히 적다.** 이 값들은 잠정이며, 실사용 녹음이 쌓이면 다시 잰다.
#
# ## hotwords로 못 고치는 것도 있다
#
# **"자산"은 어떤 설정으로도 복구되지 않았다.** 명시적으로 넣어도 "사산"이었다.
# 그런 단어는 매처의 자모 근사매칭이 받는다 —
# "자산 보호 신청"↔"사산 보호 신청"은 초성 1개 차이라 잡힌다.
# 화면 글자가 어색해도 판정은 정상이라는 뜻이다.
HOTWORDS_MAX_CHARS = 400
HOTWORDS_PER_STAGE = 6


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

    **기본값은 꺼짐이다. 켜기 전에 반드시 재볼 것.**

    이론은 이렇다. Whisper는 앞 문맥을 보고 다음 토큰을 고르므로,
    "안전계좌"를 미리 보여주면 비슷한 발음 후보 중 그쪽 확률이 올라간다.
    우리가 잡아야 하는 바로 그 단어들의 전사 정확도가 오를 것이다.

    그런데 실측이 그 이론을 지지하지 않았다 (transcribe_test.py, 19.7초 한국어 발화).

        프롬프트 없음   5.22초   CER 83.7%
        프롬프트 있음  12.69초   CER 83.7%

    **지연이 2.4배가 되는데 정확도는 그대로였다.** 개입 지연 목표가 1.5초인데
    전사에만 그만큼을 더 쓰는 것은 감당할 수 없다.

    부작용도 관찰됐다. VAD로 짧게 자른 조각에서 프롬프트에 있던 "은행 직원"이
    없는데도 전사에 나타났다. 프롬프트 어휘가 곧 위험 키워드라서,
    이 환각은 **그대로 오탐이 된다.**

    다만 위 측정은 표본 하나이고 그 음성 자체가 상태가 나빴다(합성 화자의 한국어).
    사람 목소리 5~10개로 다시 재서 판단할 것. 효과가 확인되면 그때 켠다.
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


def build_hotwords(db=None, max_chars: int = HOTWORDS_MAX_CHARS,
                   per_stage: int = HOTWORDS_PER_STAGE) -> str:
    """Whisper에 미리 알려줄 위험 어휘. `initial_prompt`과 다른 경로로 들어간다.

    프롬프트는 "앞선 대화"로 주입되어 디코딩 전체에 영향을 주고 지연을 늘린다.
    hotwords는 해당 어휘의 확률만 밀어주므로 **비용이 거의 없다** —
    실측에서 기본과 같은 속도로 CER이 절반이 됐다.

    담는 순서가 중요하다. 상한이 있으므로 **틀리면 가장 치명적인 것부터** 넣는다.
    critical은 하나만 잘못 들려도 위험도 하한이 안 걸린다.
    """
    if db is None:
        from .patterns import load_db

        db = load_db()

    terms: list[str] = [c.text for c in db.criticals]
    for sid in sorted(db.stages, key=lambda s: -db.stages[s].weight):
        for kw in db.stages[sid].keywords[:per_stage]:
            if kw.text not in terms:
                terms.append(kw.text)

    out: list[str] = []
    total = 0
    for t in terms:
        if total + len(t) + 1 > max_chars:
            break
        out.append(t)
        total += len(t) + 1
    return " ".join(out)


class SpeechToText:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        # 기본은 프롬프트 없음. 켜려면 build_initial_prompt()를 직접 넘긴다.
        # (근거는 build_initial_prompt의 설명 참조 — 지연이 늘고 이득이 hotwords보다 작다)
        self.initial_prompt = initial_prompt or ""
        # hotwords는 **기본 활성**이다. 근거는 위 HOTWORDS_MAX_CHARS 주석의 실측표.
        # ""를 명시적으로 넘기면 끌 수 있다.
        self.hotwords = build_hotwords() if hotwords is None else hotwords
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
            hotwords=self.hotwords or None,
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
