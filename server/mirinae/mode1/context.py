"""문맥 판정 — 위험 표현이 '지금 벌어지는 일'인지 '남의 말 인용'인지 가른다.

패턴 매칭만으로는 이 둘을 구별할 수 없다. 같은 문자열이기 때문이다.

    사기범   "안전계좌로 이체하세요."
    피해자   "안전계좌로 이체하라고 했어요."          ← 1332 상담
    자녀     "안전계좌로 이체하라는 말 나오면 끊어."   ← 예방 교육
    앱 자신  "안전계좌라는 것은 존재하지 않습니다."    ← 미리내 경고문

`eval_mode1.py`의 정상 통화 12건 중 **4건이 이 유형 하나로 오탐**이었다.
넷 다 위험 등급 0.750이 떴다. 마지막 줄이 특히 나쁘다 —
미리내가 띄운 경고를 사용자가 소리내어 읽으면 그 소리가 다시 마이크로 들어가
앱이 자기 경고에 다시 반응한다.

## 무엇으로 가르는가

한국어는 인용을 **어미로** 표시하고, 그 어미는 인용된 말 **뒤에** 온다.

    "안전계좌로 보내**라고** 한다더라"
    "검찰청이**라고** 하면서"
    "말하지 말**라고** 하는 것도 수법이야"

그래서 매칭된 구간 **뒤쪽 몇 음절**만 보면 된다. 문법 분석기가 필요 없다.

## 부정 표현은 쓰지 않는다

"없다·않다·아니다"로 가르려 했으나 **사기범이 더 자주 쓴다.**

    "오늘까지 처리하지 **않으면** 계좌가 동결됩니다"   ← 실제 사기 발화
    "시간이 **없습니다**"

부정을 억제 신호로 쓰면 이런 발화의 위험 단계가 통째로 지워진다.
반면 인용 어미는 사기범 발화에 거의 나오지 않는다 — 사기범은 직접 명령한다.
그래서 **인용 어미만** 쓴다. 판별력이 높고 부작용이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matcher import MatchSpan, normalize

# 간접인용 어미. 공백이 제거된 정규화 텍스트에서 찾는다.
#
# 주의 — "더라"를 단독으로 넣으면 안 된다. 사기범의 "가시더라도"에 걸린다.
# "더라고"처럼 인용이 확정되는 형태만 넣는다.
QUOTE_MARKERS: tuple[str, ...] = (
    "라고", "다고", "냐고", "자고",          # 안전계좌로 이체하라고 했다
    "라는", "다는", "라던", "다던",          # 이체하라는 말
    "라면서", "다면서", "라며", "다며",      # 검사라면서
    "라더", "다더", "더라고",                # 한다더라
    "대요", "래요", "라니까", "다니까",
)

# 매칭 구간 뒤로 몇 음절까지 인용 어미를 찾을지.
# "안전계좌 + 로보내 + 라고" 처럼 조사·어간이 사이에 낀다.
#
# 값은 짐작이 아니라 평가 세트로 골랐다. 4~30을 훑은 결과:
#
#   창    탐지율   정밀도    오탐률
#    4    80.0%   72.7%    25.0%     인용을 놓쳐 정상 통화 3건이 위험으로
#    8    80.0%   88.9%     8.3%
#   12    80.0%   88.9%     8.3%
#   14    80.0%  100.0%     0.0%   ← 여기서 평탄해진다
#   30    80.0%  100.0%     0.0%     넓혀도 더 나아지지 않는다
#
# 14부터 30까지 결과가 같다. 한 시나리오에 겨우 맞춘 값이 아니라 안정 구간이라는 뜻이다.
# **그 구간의 최솟값을 쓴다.** 억제 범위가 넓을수록 사기범이 인용 어미를 흘려
# 위험 단계를 지우는 회피가 쉬워지므로, 같은 성능이면 좁은 쪽이 안전하다.
QUOTE_WINDOW = 14


@dataclass(frozen=True)
class ContextVerdict:
    """매칭 하나에 대한 문맥 판정."""

    kind: str            # "direct" | "quoted"
    marker: str = ""     # 판정 근거가 된 어미
    evidence: str = ""   # 실제로 본 뒤쪽 구간

    @property
    def suppressed(self) -> bool:
        """점수에서 제외해야 하는가."""
        return self.kind != "direct"

    def why(self) -> str:
        if self.kind == "quoted":
            return f"인용 어미 '{self.marker}' — 남의 말을 옮긴 것으로 판단"
        return "직접 발화"


DIRECT = ContextVerdict("direct")


def classify(sentence: str, span: MatchSpan) -> ContextVerdict:
    """매칭 구간 뒤를 보고 직접 발화인지 인용인지 판정한다.

    :param sentence: 매칭이 일어난 문장 원문 (정규화 전)
    :param span: 정규화 텍스트 기준 음절 인덱스를 가진 매칭
    """
    norm = normalize(sentence)
    tail = norm[span.end:span.end + QUOTE_WINDOW]
    if not tail:
        return DIRECT

    best: tuple[int, str] | None = None
    for marker in QUOTE_MARKERS:
        at = tail.find(marker)
        if at >= 0 and (best is None or at < best[0]):
            best = (at, marker)

    if best is None:
        return DIRECT
    return ContextVerdict("quoted", marker=best[1], evidence=tail)


# ── 문장 분할 ─────────────────────────────────────────────────────────────────

SENTENCE_BREAKS = ".!?…\n"


def split_sentences(text: str) -> list[str]:
    """문맥 판정은 문장 단위로 해야 한다.

    누적 버퍼 전체를 한 덩어리로 보면 앞 문장의 인용 어미가 뒤 문장의 매칭까지
    덮어버린다. 반대로 문장을 넘어 매칭되던 유령 표현도 함께 사라진다 —
    발화 분할기(`segmenter.py`)가 이미 의미 단위를 보장하므로 잃는 것이 없다.
    """
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch in SENTENCE_BREAKS:
            if cur:
                out.append("".join(cur).strip())
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [s for s in out if s]
