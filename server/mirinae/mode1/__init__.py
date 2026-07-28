"""미리내 · 모드 1 — 보이스피싱 위험 경고.

D08(모드 1 설계도) 구현.

설계 원리 한 줄: **개별 단어가 아니라 단계의 조합으로 식별한다.**
정상 통화에서 "계좌"는 흔하다. 그러나 검찰청 + 대포통장 + 체포영장 +
말하지 마세요 + 안전계좌가 한 통화에 모두 나올 확률은 사실상 0이다.

모드 2와의 결정적 차이는 **상태 유지**다.
모드 2는 청크마다 독립(stateless)이지만, 모드 1은 S1이 0:30에 S4가 2:00에 나온다.
통화 전체를 누적해야 단계 커버리지가 성립한다. 최신 발화만 채점하면
커버리지가 영원히 1/N에 머물러 점수가 오르지 않는다.
"""

from .matcher import Matcher, normalize, to_jamo
from .patterns import PatternDB, Stage, load_db
from .router import Route, route_of
from .scorer import ScoreResult, Scorer

__all__ = [
    "Matcher", "normalize", "to_jamo",
    "PatternDB", "Stage", "load_db",
    "Route", "route_of",
    "Scorer", "ScoreResult",
]
