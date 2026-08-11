"""행위 프레임 — 명사가 아니라 "무엇을 어디로 누가 시키는가"를 본다.

## 왜 만들었나

검증셋 10차를 적대적으로 설계했더니 탐지율이 45.5%로 떨어졌다.
그런데 남은 실패를 뜯어보니 어휘 공백이 아니었다.
**같은 단어가 사기에도 정상에도 나오고, 갈리는 것은 관계였다.**

    사기  "지금 화면에 뜨는 거 설치하시면 제가 원격으로 봐드릴게요"   0.000 안전
    정상  "제가 원격제어로 봐드릴게요 사내 프로그램이요"              0.750 위험

**정확히 반대로 나왔다.** `원격제어`라는 명사를 단독 고위험으로 두었기 때문이다.
사기범이 그 단어를 피해 부드럽게 말하니 빠져나갔고, 정상 IT 지원이
정확한 용어를 쓰니 걸렸다. 어휘를 더 넣어도 미탐만 풀리고 오탐은 남는다.

같은 결함을 이미 두 번 고쳤다.

| | 잘못 본 것 | 실제 판별자 |
|---|---|---|
| C7 통장 | "통장이랑 도장"이라는 **명사쌍** | 지참(정상) 대 전송(사기) |
| C8 계좌 | 계좌를 말한다는 **사실** | 공식 경로 안내(정상) 대 차단(사기) |

세 번 반복되면 층위 문제다. 그래서 프레임으로 옮긴다.

## 프레임이란

보이스피싱은 결국 **한 가지 행위**를 시켜야 성립한다 —
피해자의 자산이나 기기 통제권을 사기범이 지정한 곳으로 옮기는 것.
그 지시는 세 자리로 이루어져 있다.

    [무엇을]        [어디로 / 누구에게]        [누가 시키는가]
     자금·통장·기기   화자가 지금 지정하는 곳     화자 → 청자

**세 자리가 다 차야 위험하다.** 하나만 보면 정상과 구별되지 않는다 —
`인출`도 `설치`도 `계좌`도 그 자체로는 일상어다.

## 왜 단계 점수가 아니라 별도 층위인가

단계(S1~S8)는 **무엇이 언급됐는가**를 센다. 프레임은 **그것들이 어떻게
묶였는가**를 본다. 같은 단어가 두 단계에 걸쳐 있어도 단계 점수는 늘지만
(합이 아니라 max라 늘지도 않는다) 조합 신호는 만들어지지 않는다.

10차 V-S2가 그랬다 — "인출하신 다음에 저희 직원한테 건네주시면"은
인출도 S5, 전달도 S5라 한 단계에 뭉개졌다. 그런데 **인출과 대면 전달이
이어지는 것**은 대면편취형의 정의 그 자체다. 프레임은 그 이어짐을 본다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .matcher import MatchSpan, normalize

# ── 자리별 어휘 (닫힌 집합) ───────────────────────────────────────────────────

#: 현금화 행위. 어간만 적으면 활용형은 뒤의 `_ACT_TAIL`이 받는다.
WITHDRAW = ("인출", "찾으시", "찾아서", "빼시", "빼서", "출금")

#: 넘길 수 있는 **물건**. 대면편취는 현금만 가져가지 않는다 —
#: 통장·체크카드·도장을 그대로 받아 가는 유형이 고령층에서 특히 많다.
#: 11차 R11-F12에서 "통장이랑 도장 준비해 두세요 ... 직원이 방문하면
#: 건네주시면 됩니다"를 놓쳤다. 프레임이 인출만 보고 있었다.
ASSET_ITEM = ("통장", "체크카드", "현금", "돈", "도장", "카드")

#: 넘기는 행위 — 대면편취의 두 번째 자리.
HANDOVER = ("건네", "전달", "드리면", "주시면", "맡기", "가져다")

#: 넘기는 상대 — 사기범 쪽 사람.
RECIPIENT = ("직원", "수사관", "담당자", "기사", "사람", "저희쪽", "저희측")

#: 기기 통제권을 넘기는 행위.
GRANT = ("설치", "깔아", "눌러", "누르", "실행", "허용", "승인")

#: 원격 접근.
REMOTE = ("원격", "팀뷰어", "애니데스크", "화면공유", "원격제어")

#: **출처를 알 수 없는** 대상. 프레임의 핵심 자리다.
#: 공식 스토어·사내 배포와 갈리는 지점이 여기다.
UNKNOWN_SOURCE = (
    "화면에뜨는", "지금뜨는", "뜨는거", "문자로보내드린", "문자로온",
    "제가보내드린", "보내드린링크", "링크에서", "링크눌러", "안내해드리는",
)

#: 행위 뒤에 붙는 어미. 지시·조건 어미만 인정한다 —
#: "인출했어요"(서술)와 "인출하시면"(지시)을 가르기 위해서다.
_ACT_TAIL = r"(?:하[시셔]|해서|하신|하시면|하세요|하십|시면|셔서|신다음|한다음|한뒤|하고)"

#: 행위가 **금지**되면 프레임이 아니다.
#: "문자로 보내드린 링크는 저희가 보내지 않으니 **누르지 마세요**"는
#: 사기를 막는 경고인데 설치 유도로 잡혔다. 행위 어간 뒤를 내다본다.
_NOT_PROHIBITED = r"(?!지\s*[마말않못])"

#: 두 자리 사이에 끼어들 수 있는 음절 수.
SLOT_GAP = 12


def _alt(items: tuple[str, ...]) -> str:
    return "|".join(re.escape(x) for x in sorted(items, key=len, reverse=True))


@dataclass(frozen=True)
class Frame:
    """행위 프레임 하나."""

    id: str
    kind: str            # "critical" | "stage"
    target: str          # critical id 또는 단계 id
    label: str
    regex: re.Pattern[str]
    rationale: str = ""

    def find(self, sentence: str) -> MatchSpan | None:
        m = self.regex.search(normalize(sentence))
        if m is None:
            return None
        return MatchSpan(self.label, "frame", m.start(), m.end(), 0)


def _cash_handover_frame() -> Frame:
    """현금을 찾아서 **사람에게 건네라** — 대면편취형의 정의.

    전체 피해의 66.9%가 이 유형인데, 인출과 전달이 각각 S5 하나로만 세어져
    조합 신호가 만들어지지 않았다(10차 V-S2에서 0.000).

    정상 금융기관은 인출한 현금을 자기 직원에게 건네라고 하지 않는다.
    은행 창구가 아니라 **밖에서 만나는 사람**이기 때문이다.
    """
    pat = (f"(?:{_alt(WITHDRAW)}){_ACT_TAIL}"
           f"[가-힣\\s]{{0,{SLOT_GAP}}}?"
           f"(?:{_alt(RECIPIENT)})"
           f"[가-힣]{{0,4}}?"
           f"(?:{_alt(HANDOVER)})")
    return Frame(
        id="F-대면편취", kind="critical", target="C9",
        label="인출 후 사람에게 전달",
        regex=re.compile(pat),
        rationale=("현금 인출과 대면 전달이 이어지는 구성. 어느 정상 기관도 "
                   "인출한 현금을 자기 직원에게 건네라고 하지 않는다. "
                   "전체 피해의 66.9%가 이 유형이다."),
    )


def _unknown_install_frame() -> Frame:
    """**출처를 알 수 없는 것**을 설치·실행시킨다.

    `원격제어`라는 명사를 단독 고위험으로 두었더니 사기범이 그 단어를 피해
    빠져나가고(10차 V-S6) 정상 IT 지원이 걸렸다(V-B10).

    판별자는 원격이냐가 아니라 **무엇을 설치시키느냐**다.
    사내 배포 프로그램과 "화면에 뜨는 거"는 다르다.
    """
    src, grant = _alt(UNKNOWN_SOURCE), _alt(GRANT)
    pat = (f"(?:(?:{src})[가-힣]{{0,{SLOT_GAP}}}?(?:{grant}){_NOT_PROHIBITED}"
           f"|(?:{grant}){_NOT_PROHIBITED}[가-힣]{{0,4}}?(?:{src}))")
    return Frame(
        id="F-불명출처설치", kind="critical", target="C5",
        label="출처 불명 설치 유도",
        regex=re.compile(pat),
        rationale=("출처를 밝히지 않은 것을 설치·실행하게 하는 구성. "
                   "정상 기관은 공식 스토어나 사내 배포 경로를 안내한다. "
                   "'화면에 뜨는 거', '문자로 보내드린 링크'가 그 자리를 채우면 "
                   "설치 대상이 무엇인지 청자가 확인할 방법이 없다."),
    )


def _remote_grant_frame() -> Frame:
    """원격 접근 **권한을 넘기게 시킨다**.

    "제가 원격제어로 봐드릴게요"(화자가 한다)와
    "설치하시면 제가 원격으로 봐드릴게요"(청자에게 시킨다)는 다르다.
    방향이 판별자다 — 경어법 규칙(`morphology.py`)과 같은 원리다.
    """
    pat = (f"(?:{_alt(GRANT)}){_ACT_TAIL}"
           f"[가-힣\\s]{{0,{SLOT_GAP}}}?"
           f"(?:{_alt(REMOTE)})")
    return Frame(
        id="F-원격권한", kind="critical", target="C5",
        label="설치 지시 후 원격 접근",
        regex=re.compile(pat),
        rationale=("설치·허용을 **시킨 뒤** 원격으로 접근하겠다는 구성. "
                   "정상 IT 지원은 이미 깔린 사내 도구를 쓰거나 자기가 조치한다. "
                   "명사 '원격제어'만 보면 정상 지원이 걸리고 사기범은 "
                   "'원격으로'라고만 말해 빠져나간다."),
    )


def build_frames() -> list[Frame]:
    return [_cash_handover_frame(), _unknown_install_frame(), _remote_grant_frame()]


# ── 통화 전체에 걸쳐 채워지는 프레임 ─────────────────────────────────────────
#
# 위의 프레임들은 문장 단위로 돈다. 그런데 사기범이 자리를 **나눠서** 말하면
# 걸리지 않는다. 11차 R11-F12가 그랬다.
#
#     "제가 대신 신청해 드릴 테니 통장이랑 도장 준비해 두세요"   ← 자산 준비
#     "이따 직원이 방문하면 건네주시면 됩니다"                    ← 대면 전달
#
# 두 마디에 나뉘어 있어서 문장 단위 프레임이 통째로 비켜 갔다.
# 자연스러운 말하기 방식이지 회피가 아니다 — 한 문장에 다 넣는 쪽이 오히려 부자연스럽다.
#
# 그래서 **자리를 따로 세고 통화 단위로 합친다.** 증거 누적과 같은 구조다.
# 어느 마디에서 채워졌든 통화가 끝날 때 자리가 다 차 있으면 프레임이 성립한다.

#: 자산을 손에 쥐게 만드는 자리.
SLOT_PREPARE = re.compile(
    f"(?:(?:{_alt(WITHDRAW)}){_ACT_TAIL}"
    f"|(?:{_alt(ASSET_ITEM)})[가-힣]{{0,3}}?(?:준비|챙기|가지고|들고|찾아))"
)

#: 사기범 쪽 사람에게 넘기는 자리.
SLOT_HANDOVER = re.compile(
    f"(?:{_alt(RECIPIENT)})[가-힣]{{0,6}}?(?:{_alt(HANDOVER)})"
)

#: 통화 단위 프레임 — (자리 이름, 정규식) 목록과 성립 조건.
CALL_FRAMES: tuple[tuple[str, str, tuple[tuple[str, re.Pattern[str]], ...]], ...] = (
    ("C9", "자산 준비 후 대면 전달",
     (("준비", SLOT_PREPARE), ("전달", SLOT_HANDOVER))),
)


def call_frame_slots(sentence: str) -> list[tuple[str, str, MatchSpan]]:
    """이 문장이 채우는 자리를 돌려준다. (프레임 id, 자리 이름, 구간)"""
    norm = normalize(sentence)
    out = []
    for fid, _label, slots in CALL_FRAMES:
        for name, rx in slots:
            m = rx.search(norm)
            if m:
                out.append((fid, name, MatchSpan(name, "frame", m.start(), m.end(), 0)))
    return out


def call_frame_complete(filled: dict[str, set[str]]) -> list[str]:
    """자리가 다 찬 프레임의 대상 id."""
    done = []
    for fid, _label, slots in CALL_FRAMES:
        if filled.get(fid, set()) >= {name for name, _ in slots}:
            done.append(fid)
    return done
