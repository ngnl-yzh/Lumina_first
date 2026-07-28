"""위험도 스코어링 — 누적 전사 버퍼 전체로 채점한다.

**최신 발화만 보면 안 된다.** 이것이 모드 2와의 결정적 차이다.
S1이 0:30에, S4가 2:00에 나온다. 통화 전체를 누적해야 단계 커버리지가 성립하고,
최신 발화만 채점하면 커버리지가 영원히 1/N에 머물러 점수가 오르지 않는다.

D08 §04 의사코드에 표시된 수정 3건이 반영되어 있다.
  ① 빈 시퀀스 — `max(..., default=0.0)` 없이는 매칭 0건에서 ValueError
  ② 커버리지 상시 최대 — `if hit[s] > 0` 조건이 없으면 cov가 항상 1.0
  ③ 포화 — 8단계 전체가 아니라 **경로별 이론 최대**로 정규화
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .matcher import Matcher
from .patterns import PatternDB
from .router import Route, route_of

# 위험도 등급 경계 (D03 §4.2)
THRESHOLD_WARN = 0.45      # 주의
THRESHOLD_ALERT = 0.75     # 위험 — 즉시 개입

CRITICAL_FLOOR = 0.75      # 단독·조합 고위험 신호가 걸리면 최소 이 점수
BENIGN_PENALTY = 0.15      # 정상 문맥 지시어 1개당 감점
COVERAGE_BONUS_SCALE = 1.5
DEEPVOICE_BONUS = 0.10     # 딥보이스 탐지 결과를 S7 경로에 반영 (P1 과제)


@dataclass
class ScoreResult:
    score: float
    route: Route
    stage_hits: dict[str, float]
    matched: dict[str, list[str]] = field(default_factory=dict)
    criticals: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    benign_hits: list[str] = field(default_factory=list)
    intervene: bool = False

    @property
    def level(self) -> str:
        if self.score >= THRESHOLD_ALERT:
            return "위험"
        if self.score >= THRESHOLD_WARN:
            return "주의"
        return "안전"

    @property
    def coverage(self) -> float:
        active = self.route.stages
        return sum(1 for s in active if self.stage_hits.get(s, 0) > 0) / len(active)

    def why(self) -> str:
        """근거 패널 — "왜 이 점수인가"에 답한다.

        점수만 보여주면 사용자도 심사위원도 믿지 않는다.
        """
        lines = [
            f"위험도 {self.score:.3f} ({self.level}) · 경로 {self.route.id} {self.route.name}",
            f"단계 커버리지 {self.coverage * 100:.0f}%",
        ]
        for sid in self.route.stages:
            hits = self.matched.get(sid, [])
            if hits:
                lines.append(f"  {sid} — {', '.join(hits[:4])}")
        if self.criticals:
            lines.append(f"  단독 고위험 발동: {', '.join(self.criticals)}")
        if self.pairs:
            lines.append(f"  조합 고위험 발동: {', '.join(self.pairs)}")
        if self.benign_hits:
            lines.append(f"  정상 문맥 감점: {', '.join(self.benign_hits)}")
        return "\n".join(lines)


class Scorer:
    def __init__(self, db: PatternDB, matcher: Matcher | None = None) -> None:
        self.db = db
        self.matcher = matcher or Matcher()

    # ── 단계 매칭 ─────────────────────────────────────────────────────────────

    def stage_hits(self, text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        """단계별 최고 점수와 매칭된 표현.

        **합산이 아니라 max**를 쓴다. 같은 단계의 표현을 여러 번 말했다고 해서
        위험이 그만큼 커지는 것은 아니다. 합산하면 반복 언급이 과대평가된다.
        """
        hits: dict[str, float] = {}
        matched: dict[str, list[str]] = {}

        for sid, stage in self.db.stages.items():
            best = 0.0
            found: list[str] = []
            for kw in stage.keywords:
                for form in kw.all_forms():
                    if self.matcher.match(text, form):
                        best = max(best, kw.score)      # ① default=0.0 대신 초기값 0
                        found.append(form)
                        break
            hits[sid] = best
            if found:
                matched[sid] = found
        return hits, matched

    # ── 고위험 신호 ───────────────────────────────────────────────────────────

    def find_criticals(self, text: str) -> list[str]:
        out = []
        for c in self.db.criticals:
            if any(self.matcher.match(text, f) for f in c.all_forms()):
                out.append(c.id)
        return out

    def find_pairs(self, hits: dict[str, float]) -> list[str]:
        return [
            p.id for p in self.db.pairs
            if all(hits.get(s, 0.0) > 0 for s in p.stages)
        ]

    def find_benign(self, text: str) -> list[str]:
        """benign은 **정확 일치만** 쓴다 — 비대칭 설계.

        근사매칭을 허용하면 위험 표현이 benign으로 잘못 걸려 점수를 깎는다.
        놓치는 비용보다 잘못 깎는 비용이 크다.
        """
        return self.matcher.find_all(text, self.db.benign, exact_only=True)

    # ── 본체 ──────────────────────────────────────────────────────────────────

    def score(
        self,
        buffer_text: str,
        route: Route | None = None,
        deepvoice_score: float = 0.0,
        deepvoice_threshold: float = 0.7,
    ) -> ScoreResult:
        """누적 전사 버퍼 전체를 채점한다."""
        hits, matched = self.stage_hits(buffer_text)
        route = route or route_of(hits)
        active = route.stages

        # 단계 가중 점수
        stage_score = sum(
            hits.get(s, 0.0) * self.db.stages[s].weight for s in active
        )

        # ② 커버리지 — 0보다 큰 단계만 계수한다. 조건이 없으면 항상 1.0이 된다.
        cov = sum(1 for s in active if hits.get(s, 0.0) > 0) / len(active)
        bonus = (cov ** 2) * COVERAGE_BONUS_SCALE

        benign_hits = self.find_benign(buffer_text)
        penalty = len(benign_hits) * BENIGN_PENALTY

        # ③ 경로별 이론 최대로 정규화 — 8단계 전체로 나누면 가족사칭형이 불리해진다
        raw = (stage_score + bonus) / route.denominator - penalty
        score = min(1.0, max(0.0, raw))

        criticals = self.find_criticals(buffer_text)
        pairs = self.find_pairs(hits)
        if criticals or pairs:
            score = max(score, CRITICAL_FLOOR)

        # 딥보이스 탐지 결과를 가족사칭 경로에 가중 (P1 · 실패 시 제외 가능)
        if route.id == "B" and deepvoice_score > deepvoice_threshold:
            score = min(1.0, score + DEEPVOICE_BONUS)

        return ScoreResult(
            score=score,
            route=route,
            stage_hits=hits,
            matched=matched,
            criticals=criticals,
            pairs=pairs,
            benign_hits=benign_hits,
            intervene=score >= THRESHOLD_ALERT,
        )


class CallState:
    """통화 하나의 누적 상태.

    모드 1이 stateful인 이유가 이 클래스에 있다.
    발화가 들어올 때마다 버퍼에 붙이고 **전체를 다시 채점**한다.
    """

    def __init__(self, scorer: Scorer) -> None:
        self.scorer = scorer
        self.utterances: list[str] = []
        self.last: ScoreResult | None = None
        self.intervened = False

    @property
    def buffer_text(self) -> str:
        return " ".join(self.utterances)

    def add_utterance(self, text: str, deepvoice_score: float = 0.0) -> ScoreResult:
        self.utterances.append(text)
        self.last = self.scorer.score(self.buffer_text, deepvoice_score=deepvoice_score)
        return self.last

    def should_intervene(self) -> bool:
        """개입은 통화당 한 번만. 반복 재생은 오히려 각성을 방해한다."""
        if self.last and self.last.intervene and not self.intervened:
            self.intervened = True
            return True
        return False

    def reset(self) -> None:
        self.utterances.clear()
        self.last = None
        self.intervened = False
