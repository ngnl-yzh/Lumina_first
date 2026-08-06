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
#
# ## 분모를 줄여보려다 그만둔 기록
#
# 진입 단계(S1/S7/S8)가 안 잡히면 8단계 전체가 분모(7.90)라 점수가 크게 희석된다.
# 통화 중간부터 듣기 시작하거나 도입부를 STT가 놓치면 불리하다는 뜻이라,
# "관측된 단계를 가장 잘 덮는 경로를 고르는" 적응형 폴백을 만들어 재봤다.
#
#   현재 (전 단계)   탐지율 81.0%  오탐률 0.0%   경고 0건
#   적응형 폴백      탐지율 81.0%  오탐률 0.0%   **경고 2건** (B-13 부동산 계약금 · B-14 중고거래)
#
# **탐지는 하나도 안 늘고 정상 통화 경고만 늘었다.** 채택하지 않는다.
# 이유는 분명하다 — 진입 단계가 없는 통화는 대개 증거 자체가 적다.
# 그런 상태에서 분모를 줄이면 "계좌로 이체" + "오늘까지" 두 개만 잡혀도
# 커버리지가 1.0이 되어 점수가 치솟는다. 부동산 계약금과 중고거래가 정확히 그 모양이다.
#
# 실제로 R-A-02(공개 녹취 발췌) 미탐의 원인도 폴백이 아니었다.
# 같은 증거를 경로 A로 채점해도 0.252 → 0.325로, 임계값 0.75에 한참 못 미친다.
# 원인은 **잡힌 단계가 2개뿐**이라는 것이고, 그건 분모로 풀 문제가 아니다.
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
