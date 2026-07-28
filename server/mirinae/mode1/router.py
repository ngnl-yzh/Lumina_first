"""유형 경로 라우팅 — 유형마다 전개가 다르다.

일반적인 탐지 설계에는 없는 모듈이다. 있어야 하는 이유가 D08 §4.1에 수치로 나와 있다.

가족사칭 "S7 + S5"는 **이 프로젝트가 방어하려는 바로 그 시나리오**인데,
8단계 전체를 분모로 쓰면 0.200 "낮음" 판정이 나온다.
적은 단계만 쓰는 가족사칭형이 구조적으로 불리하기 때문이다.
경로별 정규화가 이 편향을 없앤다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .patterns import COVERAGE_BONUS_MAX, PatternDB, STAGE_WEIGHTS


@dataclass(frozen=True)
class Route:
    id: str
    name: str
    share: str
    stages: tuple[str, ...]
    pair_id: str

    @property
    def denominator(self) -> float:
        """경로별 이론 최대 점수.

        단계 가중치 합 + 커버리지 보너스 최대(1.5).
        이 값으로 나누기 때문에 단계 수가 적은 경로도 1.0까지 도달할 수 있다.
        """
        return sum(STAGE_WEIGHTS[s] for s in self.stages) + COVERAGE_BONUS_MAX


ROUTES: dict[str, Route] = {
    "A": Route("A", "정부기관 사칭", "31.1% · 광주·전남 급증",
               ("S1", "S2", "S3", "S4", "S5", "S6"), "P1"),
    "B": Route("B", "가족·지인 사칭", "33.7% · 딥보이스 주 표적",
               ("S7", "S3", "S4", "S5", "S6"), "P3"),
    "C": Route("C", "대출빙자", "35.2% · 딥보이스 결합도 낮음",
               ("S8", "S5", "S6"), "P2"),
}

# 라우팅 실패 시 폴백 — 전 단계 활성. 무판정보다는 낫다.
FALLBACK = Route("*", "판정 실패 · 전 단계", "—",
                 ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"), "")

# 각 경로를 지목하는 진입 단계. 이 단계가 잡히면 그 유형으로 본다.
ENTRY_STAGE: dict[str, str] = {"S1": "A", "S7": "B", "S8": "C"}


def route_of(stage_hits: dict[str, float]) -> Route:
    """지금까지 잡힌 단계로 유형을 판정한다.

    진입 단계가 여러 개 잡히면 **점수가 높은 쪽**을 택한다.
    예컨대 "검찰청"과 "엄마 나야"가 같이 나오면 정상적인 통화가 아니라
    둘 중 하나가 오탐이므로, 강하게 잡힌 쪽을 믿는다.
    """
    candidates = [
        (stage_hits.get(sid, 0.0), rid)
        for sid, rid in ENTRY_STAGE.items()
        if stage_hits.get(sid, 0.0) > 0
    ]
    if not candidates:
        return FALLBACK
    return ROUTES[max(candidates)[1]]


def route_max(route: Route) -> float:
    return route.denominator
