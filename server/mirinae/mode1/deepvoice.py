"""딥보이스 탐지 — 합성 음성 여부 판정.

D08 §03 파이프라인 4단계이자 **P1 항목**이다. 실패하면 제외하고 한계로 기술한다.

설계도가 미리 경고한 위험이 있다.
  "ASVspoof 계열 모델은 학습에 없던 합성 방식에 일반화가 약하다.
   XTTS-v2를 못 잡을 수 있다."
그래서 이 모듈은 **쓰기 전에 반드시 측정**해야 한다. `benchmark_deepvoice.py`가 그 일을 한다.

판정 결과는 단독으로 쓰지 않는다. 가족사칭 경로(B)의 위험도에 +0.10을 얹는 보조 신호다.
딥보이스가 주로 결합하는 유형이 가족·지인 사칭이기 때문이고,
탐지가 틀려도 패턴 매칭이 주된 판단을 계속하게 하기 위해서다.

주의 — 모드 2와의 상호작용:
  모드 2로 보호한 음성을 이 탐지기에 넣으면 "가짜"로 판정될 수 있다.
  적대적 섭동이 탐지기가 찾는 바로 그 종류의 인공물이기 때문이다.
  시연에서 두 모드를 교차하지 않는 이유가 이것이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000

# 실제로 로드되는 것을 확인한 모델. 라벨은 {0:'fake', 1:'real'}, 16 kHz 입력.
DEFAULT_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"

# 이 값을 넘으면 "합성 의심". 스코어러의 가중 반영 임계값과 같은 값을 쓴다.
DEFAULT_THRESHOLD = 0.7

# 긴 발화는 창으로 잘라 본다. wav2vec2 계열은 입력이 길수록 느려지고,
# 통화 중에는 일부 구간만 합성일 수도 있다.
WINDOW_SEC = 4.0
HOP_SEC = 2.0

# 너무 짧으면 판정이 불안정하다. 이 아래는 아예 판정하지 않는다.
MIN_SEC = 1.0


@dataclass
class DeepvoiceResult:
    """판정 결과.

    `usable`이 False면 점수를 쓰지 않는다 — 모르는 것을 0이나 1로 채우면
    그게 곧 오탐이 된다.
    """

    fake_prob: float
    usable: bool
    n_windows: int
    duration_sec: float
    note: str = ""

    @property
    def is_synthetic(self) -> bool:
        return self.usable and self.fake_prob >= DEFAULT_THRESHOLD

    def label(self) -> str:
        if not self.usable:
            return "판정 불가"
        if self.fake_prob >= DEFAULT_THRESHOLD:
            return "딥보이스 의심"
        if self.fake_prob >= 0.5:
            return "합성음 가능성"
        return "정상 음성"


class DeepvoiceDetector:
    """사전학습 합성음 탐지기 래퍼.

    모델 로드가 무거우므로 지연 로드한다. 모드 2만 쓰는 실행에서
    수백 MB를 올릴 이유가 없다.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._device = device
        self._pipe = None
        self._fake_key: str | None = None

    @property
    def pipe(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            dev = self._device
            if dev is None:
                dev = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("audio-classification", model=self.model_name,
                                  device=dev)

            # 라벨 이름은 모델마다 다르다. 'fake'/'spoof' 계열을 찾아 고정한다.
            labels = list(getattr(self._pipe.model.config, "id2label", {}).values())
            for cand in labels:
                if cand.lower() in {"fake", "spoof", "synthetic", "ai", "generated"}:
                    self._fake_key = cand
                    break
            if self._fake_key is None:
                raise RuntimeError(
                    f"모델 {self.model_name}의 라벨에서 'fake'에 해당하는 것을 못 찾았다: {labels}"
                )
        return self._pipe

    def _window_prob(self, wav: np.ndarray) -> float:
        out = self.pipe({"raw": wav.astype(np.float32), "sampling_rate": SAMPLE_RATE})
        for row in out:
            if row["label"] == self._fake_key:
                return float(row["score"])
        return 0.0

    def score(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> DeepvoiceResult:
        """16 kHz 모노 float32 → 합성 확률.

        긴 발화는 4초 창으로 나눠 보고 **최댓값**을 쓴다.
        통화 일부만 합성인 경우를 놓치지 않기 위해서다.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"16 kHz만 받는다 (받은 값 {sample_rate})")

        dur = len(wav) / SAMPLE_RATE
        if dur < MIN_SEC:
            return DeepvoiceResult(0.0, False, 0, dur,
                                   f"{MIN_SEC}초 미만은 판정하지 않는다")

        win = int(SAMPLE_RATE * WINDOW_SEC)
        hop = int(SAMPLE_RATE * HOP_SEC)

        if len(wav) <= win:
            starts = [0]
        else:
            starts = list(range(0, len(wav) - win + 1, hop))
            if starts[-1] + win < len(wav):
                starts.append(len(wav) - win)

        probs = [self._window_prob(wav[s:s + win]) for s in starts]
        return DeepvoiceResult(
            fake_prob=max(probs),
            usable=True,
            n_windows=len(probs),
            duration_sec=dur,
        )


class NullDetector:
    """탐지기를 끄고 싶을 때 쓴다.

    P1이므로 검증에 실패하면 이걸로 갈아끼우고 한계로 기술한다.
    코드에서 분기를 없애려고 만든 것이다.
    """

    def score(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> DeepvoiceResult:
        return DeepvoiceResult(0.0, False, 0, len(wav) / sample_rate, "탐지기 비활성")
