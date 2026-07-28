"""VAD 발화 분할 — 침묵 0.5초를 문장 경계로 본다.

**고정 청크로 자르면 키워드가 쪼개진다.** 이것이 모드 1과 모드 2가
분할 방식을 달리하는 이유다.

  "가족에게도 말하지 마세요" (약 2.2초)를 2초에서 자르면
  "가족에게도 말하지 마" + "세요"가 된다.
  C2 고위험 신호인 "말하지 마세요"가 경계에 걸려 양쪽 어디서도 안 잡힌다.

모드 2는 청크마다 독립이라 고정 2초로 잘라도 되지만,
모드 1은 의미 단위가 보존되어야 하므로 발화 단위로 자른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 10.0
SILENCE_MS = 500.0          # 이만큼 조용하면 문장이 끝난 것으로 본다
MIN_UTTERANCE_MS = 300.0    # 이보다 짧으면 기침·잡음으로 보고 버린다
MAX_UTTERANCE_MS = 15_000.0  # 쉬지 않고 말하는 경우 강제로 끊는다
PRE_ROLL_MS = 150.0         # 발화 시작 직전을 조금 포함해 첫 음절이 잘리지 않게 한다


@dataclass
class Utterance:
    audio: np.ndarray
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class StreamingVAD:
    """스트리밍 입력에서 발화 단위를 잘라낸다.

    폰에서 오는 오디오는 조각조각 들어오므로 상태를 들고 있어야 한다.
    `push()`가 발화 경계를 만나는 순간에만 Utterance를 돌려준다.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: float = FRAME_MS,
        silence_ms: float = SILENCE_MS,
        min_utterance_ms: float = MIN_UTTERANCE_MS,
        max_utterance_ms: float = MAX_UTTERANCE_MS,
        energy_threshold_db: float = -45.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame = max(1, int(sample_rate * frame_ms / 1000))
        self.silence_frames = int(silence_ms / frame_ms)
        self.min_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_samples = int(sample_rate * max_utterance_ms / 1000)
        self.pre_roll = int(sample_rate * PRE_ROLL_MS / 1000)
        self.energy_threshold = 10.0 ** (energy_threshold_db / 10.0)

        self._buf = np.zeros(0, dtype=np.float32)
        self._voiced: list[np.ndarray] = []
        self._silence_run = 0
        self._in_speech = False
        self._consumed = 0          # 전체 스트림에서 소비한 샘플 수 (타임스탬프용)
        self._start_sample = 0
        self._tail = np.zeros(0, dtype=np.float32)   # pre-roll용 직전 조각

    def push(self, chunk: np.ndarray) -> list[Utterance]:
        """오디오 조각을 넣고, 완성된 발화가 있으면 돌려준다."""
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        out: list[Utterance] = []

        n_frames = len(self._buf) // self.frame
        for i in range(n_frames):
            frame = self._buf[i * self.frame:(i + 1) * self.frame]
            energy = float(np.mean(frame ** 2))
            is_speech = energy > self.energy_threshold

            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._start_sample = self._consumed
                    if len(self._tail):     # 첫 음절이 잘리지 않게 앞을 조금 붙인다
                        self._voiced.append(self._tail)
                        self._start_sample -= len(self._tail)
                self._voiced.append(frame)
                self._silence_run = 0
            elif self._in_speech:
                self._voiced.append(frame)      # 문장 안의 짧은 쉼일 수 있다
                self._silence_run += 1
                if self._silence_run >= self.silence_frames:
                    utt = self._flush()
                    if utt:
                        out.append(utt)
            else:
                self._tail = frame              # 무음 중에는 pre-roll만 갱신

            self._consumed += self.frame

            # 쉬지 않고 말하면 강제로 끊는다 — 안 그러면 개입이 무한정 밀린다
            if self._in_speech and sum(len(v) for v in self._voiced) >= self.max_samples:
                utt = self._flush()
                if utt:
                    out.append(utt)

        self._buf = self._buf[n_frames * self.frame:]
        return out

    def _flush(self) -> Utterance | None:
        audio = np.concatenate(self._voiced) if self._voiced else np.zeros(0, np.float32)
        self._voiced = []
        self._silence_run = 0
        self._in_speech = False

        if len(audio) < self.min_samples:
            return None
        return Utterance(
            audio=audio,
            start_sec=self._start_sample / self.sample_rate,
            end_sec=(self._start_sample + len(audio)) / self.sample_rate,
        )

    def finish(self) -> Utterance | None:
        """스트림 종료 — 남은 발화를 마저 내보낸다."""
        return self._flush() if self._in_speech else None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._voiced = []
        self._silence_run = 0
        self._in_speech = False
        self._consumed = 0
        self._tail = np.zeros(0, dtype=np.float32)
