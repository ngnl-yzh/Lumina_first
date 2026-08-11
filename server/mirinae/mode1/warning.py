"""경고 메시지 생성 — 탐지가 아니라 멈추게 하는 것.

일반적인 탐지 설계에는 없는 모듈이다. 있어야 하는 이유는 하나다.
**탐지에 성공해도 피해자가 멈추지 않으면 실패다.**

피해자는 이미 주의가 사기범에게 완전히 붙들린 심리적 강압 상태여서
"보이스피싱이 의심됩니다" 같은 일반적 알림은 통하지 않는다.
D08 §05의 5원칙은 각 심리 기제를 하나씩 겨냥해 역산한 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .patterns import PatternDB
from .scorer import ScoreResult

# 대항 권위 문구 — critical 신호별로 정확히 반박한다.
COUNTER_AUTHORITY: dict[str, str] = {
    "C1": "안전계좌는 존재하지 않습니다.",
    "C2": "수사기관은 가족에게 알리지 말라고 하지 않습니다.",
    "C3": "은행에 말하지 말라는 것은 사기입니다.",
    "C4": "수사기관은 전화로 돈을 요구하지 않습니다.",
    "C5": "원격제어 앱을 깔면 계좌를 빼앗깁니다.",
    "C6": "문자로 온 링크는 누르지 마세요. 앱은 공식 스토어에서만 받으세요.",
    "C7": "통장 사진을 요구하는 곳은 없습니다. 보내지 마세요.",
    "C8": "고지서에 적힌 계좌로만 내세요. 전화로 부르는 계좌는 아닙니다.",
}

# 경로별 교차검증 문구 — 누구에게 확인해야 하는지가 유형마다 다르다.
CROSS_CHECK: dict[str, str] = {
    "A": "끊고 나서 112로 직접 전화해 확인하세요.",
    "B": "끊고 나서 자녀분께 직접 전화해 확인하세요.",
    "C": "끊고 나서 거래 은행 영업점에 직접 확인하세요.",
    "*": "끊고 나서 112로 직접 전화해 확인하세요.",
}

DEFAULT_COUNTER = "검찰·경찰·금융감독원은 절대로 돈을 옮기라고 하지 않습니다."


@dataclass
class WarningMessage:
    """다섯 원칙이 각각 어느 줄에 해당하는지 추적한다.

    시연에서 "이 문장은 왜 이렇게 썼나"에 바로 답할 수 있어야 한다.
    """

    quote: str                                  # 원칙1 구체성
    counter: list[str] = field(default_factory=list)   # 원칙4 대항 권위
    control: str = ""                           # 원칙2 통제감 회복
    cross_check: str = ""                       # 원칙3 교차검증 복원
    action: str = ""                            # 원칙5 행동 단순화
    tts_tokens: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [self.quote, *self.counter, "", self.control]
        if self.cross_check:
            out.append(f" → {self.cross_check}")
        if self.action:
            out.append(f" → {self.action}")
        return out

    def text(self) -> str:
        return "\n".join(self.lines())

    def screen_lines(self) -> list[str]:
        """화면 표시용 — 고령층 대상이라 한 줄 15자 내외로 끊는다."""
        return [ln for ln in self.lines() if ln.strip()]


def build_warning(result: ScoreResult, db: PatternDB) -> WarningMessage:
    """스코어 결과에서 개입 메시지를 만든다.

    핵심은 원칙1이다. "의심됩니다"가 아니라 **방금 들린 말을 그대로 인용**한다.
    용어도 사기범이 쓴 말 그대로 쓴다 — "안전계좌"를 "가짜 계좌"로 번역하면
    피해자 머릿속의 연결이 끊겨 각성을 깨지 못한다.
    """
    # 인용할 표현 고르기 — critical이 최우선, 없으면 가장 위험한 단계의 매칭
    quoted: str | None = None
    counters: list[str] = []

    for cid in result.criticals:
        crit = next((c for c in db.criticals if c.id == cid), None)
        if crit:
            quoted = quoted or crit.text
            counters.append(COUNTER_AUTHORITY.get(cid, DEFAULT_COUNTER))

    # critical의 `text`는 이미 기본 표현이라 음성 뱅크에 그대로 있다.
    spoken = quoted

    if quoted is None:
        # critical이 없으면 가중치가 가장 높은 히트 단계에서 가져온다
        best = max(
            (sid for sid in result.matched),
            key=lambda s: db.stages[s].weight,
            default=None,
        )
        if best:
            quoted = result.matched[best][0]
            spoken = _base_text(db, best, quoted)

    if not counters:
        counters = [DEFAULT_COUNTER]

    quote = (f"지금 통화에서 “{quoted}”{_quote_tail(quoted)}"
             if quoted else "지금 통화가 보이스피싱으로 의심됩니다.")

    return WarningMessage(
        quote=quote,
        counter=counters[:2],          # 원칙5 — 늘어놓지 않는다
        control="지금 전화를 끊으셔도 아무 일도 생기지 않습니다.",
        cross_check=CROSS_CHECK.get(result.route.id, CROSS_CHECK["*"]),
        action="확인이 어려우면 112로 전화하세요.",
        tts_tokens=_tts_tokens(spoken, result),
    )


HANGUL_BASE, HANGUL_LAST = 0xAC00, 0xD7A3

QUOTE_TAIL_NO_JONG = " 라는 말이 나왔습니다."     # 받침 없음 — "안전계좌라는"
QUOTE_TAIL_JONG = "이라는 말이 나왔습니다."       # 받침 있음 — "발설하면이라는"


def has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는가. 조사 선택에 쓴다."""
    if not word:
        return False
    code = ord(word[-1])
    if HANGUL_BASE <= code <= HANGUL_LAST:
        return (code - HANGUL_BASE) % 28 != 0
    return False          # 숫자·영문은 받침 없는 쪽으로 읽는 편이 자연스럽다


def _quote_tail(word: str | None) -> str:
    """인용 뒤에 붙는 조사.

    종단 실행에서 잡혔다 — 개입 문구가 `“발설하면” 라는 말이 나왔습니다`로 나왔다.
    "발설하면"은 받침이 있으므로 "이라는"이 맞다.
    화면에도 뜨고 음성으로도 읽히는 문장이라 어색하면 바로 티가 난다.
    개입은 피해자를 각성시키는 것이 목적인데, 문장이 어설프면 신뢰를 잃는다.
    """
    return QUOTE_TAIL_JONG if has_final_consonant(word or "") else QUOTE_TAIL_NO_JONG


def _base_text(db: PatternDB, sid: str, form: str) -> str:
    """매칭된 표현을 DB의 **기본 표현**으로 되돌린다 — 음성 전용.

    화면과 음성을 일부러 다르게 처리한다.

      화면 — 실제로 매칭된 표현 그대로. 원칙1(구체성)이 요구하는 것이다.
      음성 — 기본 표현. 이유가 둘이다.

    **① 음성 뱅크에는 기본 표현만 있다.** 변형까지 넣으면 조각이 두 배가 되고,
    없는 조각을 요청하면 그 자리에서 소리가 안 난다 —
    정작 경고가 필요한 순간에 조용해지는 실패다. 실측에서 5건 중 3건이 그랬다
    ("송금"·"현금으로 찾"·"팀뷰어"가 뱅크에 없었다).

    **② 변형에는 STT 오차형이 섞여 있다.** "안전개좌"는 전사 오류를 흡수하려고
    넣은 항목이지 사기범이 실제로 한 말이 아니다. 그대로 읽으면 잘못된 발음을
    사용자에게 들려주게 된다.
    """
    stage = db.stages.get(sid)
    if not stage:
        return form
    for kw in stage.keywords:
        if form in kw.all_forms():
            return kw.text
    return form


def _tts_tokens(quoted: str | None, result: ScoreResult) -> list[str]:
    """사전 생성 TTS 뱅크에서 꺼낼 조각 목록.

    실시간 TTS는 지연과 실패 위험이 있다. 182개 키워드 음성을 오프라인에서 미리 만들어 두고
    고정 문장 프레임과 조립하면 런타임 지연이 사실상 0이고 발음 품질도 미리 검수할 수 있다.
    시연 안정성에서 결정적 차이가 난다.
    """
    tokens = ["frame_intro"]
    if quoted:
        tokens.append(f"kw::{quoted}")
    # 조사가 두 가지이므로 조각도 두 개다. 하나만 만들어 두면 나머지 경우에 무음이 된다.
    tokens.append("frame_quote_tail_jong" if has_final_consonant(quoted or "")
                  else "frame_quote_tail")
    for cid in result.criticals[:2]:
        tokens.append(f"counter::{cid}")
    if not result.criticals:
        tokens.append("counter::default")
    tokens += ["frame_control", f"crosscheck::{result.route.id}", "frame_action"]
    return tokens


def tts_bank_manifest(db: PatternDB) -> dict[str, str]:
    """오프라인에서 미리 생성해야 할 음성 조각 목록.

    이 매니페스트를 TTS에 한 번 돌려 두면 런타임에는 조립만 한다.
    """
    manifest: dict[str, str] = {
        "frame_intro": "지금 통화에서",
        "frame_quote_tail": QUOTE_TAIL_NO_JONG.strip(),
        "frame_quote_tail_jong": QUOTE_TAIL_JONG.strip(),
        "frame_control": "지금 전화를 끊으셔도 아무 일도 생기지 않습니다.",
        "frame_action": "확인이 어려우면 112로 전화하세요.",
        "counter::default": DEFAULT_COUNTER,
    }
    for cid, line in COUNTER_AUTHORITY.items():
        manifest[f"counter::{cid}"] = line
    for rid, line in CROSS_CHECK.items():
        manifest[f"crosscheck::{rid}"] = line

    # 인용될 수 있는 모든 표현 — critical + 전 단계 기본 표현
    for c in db.criticals:
        manifest[f"kw::{c.text}"] = c.text
    for stage in db.stages.values():
        for kw in stage.keywords:
            manifest[f"kw::{kw.text}"] = kw.text
    return manifest
