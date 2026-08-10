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

from .context import classify, split_sentences
from .matcher import Matcher, MatchSpan
from .morphology import MorphRule, build_rules
from .patterns import PatternDB
from .router import Route, route_of

# 위험도 등급 경계 (D03 §4.2)
THRESHOLD_WARN = 0.45      # 주의
THRESHOLD_ALERT = 0.75     # 위험 — 즉시 개입

CRITICAL_FLOOR = 0.75      # 단독·조합 고위험 신호가 걸리면 최소 이 점수
BENIGN_PENALTY = 0.15      # 정상 문맥 지시어 1개당 감점

# 정상 문맥 감점의 **총량 상한**. 이게 없으면 감점 자체가 회피 수단이 된다.
#
# benign 목록의 상당수는 사기범도 쓰는 말이다. SBS가 공개한 실제 녹취에서
# 사기범이 "주민등록번호, 계좌번호, 통장 비밀번호 절대로 말씀하시면 안 되시고"라고
# 말한다 — 예방 문구를 흉내내 신뢰를 얻는 수법이다.
# '본인 확인 절차'·'112에 신고'·'1332'·'영업점 방문'도 마찬가지로 사기범 화법에 흔하다.
#
# 회피 평가 F-E-02에서 실제로 뚫렸다. 예방 문구 6개를 흘리자 감점이 0.90까지 쌓여
# C1·C2·P1이 전부 발동한 통화가 **0.000 "안전"**으로 나왔다.
# 정상 통화를 살리려고 넣은 규칙이 그대로 사기범의 도구가 된 것이다.
#
# 감점은 **약한 증거**로만 쓴다. 진짜 정상 통화는 인용 판정(`context.py`)이 걸러낸다.
BENIGN_PENALTY_MAX = 0.30

# critical 신호의 근사매칭 허용 편집거리 상한.
#
# **판정 결과가 무거운 신호일수록 더 엄격한 증거를 요구한다.**
# 일반 단계 키워드는 하나 틀려도 다른 단계와 합산되어 완만하게 반영되지만,
# C1~C5는 하나만 걸려도 위험도를 0.75로 **밀어올린다.** 같은 허용치를 쓸 이유가 없다.
#
# STT 오차가 끼면 정상 통화가 망가진 채 들어오고, 헐거운 근사매칭이 거기 걸린다.
# 오차 25% 구간에서 오탐률이 오르던 주된 경로가 이것이었다.
CRITICAL_MAX_EDIT = 1

# ── 시도했다가 **뺀 것** — 근사매칭 증거 가중 ────────────────────────────────
#
# "정확히 들린 표현과 비슷해서 걸린 표현을 같은 증거로 볼 이유가 없다"는 생각으로
# 편집거리별 가중치({0: 1.00, 1: 0.85, 2: 0.70})를 넣어 단계 점수에 곱해 봤다.
# 원리는 맞아 보였지만 **효과가 정확히 0이었다.**
#
#   오차율    가중 켬 (탐지/오탐)      가중 끔 (탐지/오탐)
#     5%      76.0% / 5.2%            76.0% / 5.2%
#    10%      70.8% / 3.8%            70.8% / 3.8%
#    15%      71.4% / 7.3%            71.4% / 7.3%
#    20%      67.2% / 6.9%            67.2% / 6.9%
#    25%      65.6% / 10.8%           65.6% / 10.8%
#
# 다섯 구간 전부 소수점까지 같았다. 이유는 분명하다 — **위험 판정은 critical·pair가 만드는
# 이진 하한이 지배하는데, 가중치는 등급 점수 경로에만 걸린다.** 하한은 "걸렸나
# 안 걸렸나"라서 증거의 세기를 반영할 자리가 없다.
#
# 측정해서 효과가 없는 코드는 남기지 않는다. 테스트할 수 없는 분기는 썩는다.
# 같은 착상을 하한 쪽에 적용한 것이 위의 `CRITICAL_MAX_EDIT`이고, 그쪽은 실제로
# 오탐률을 낮췄다(오차 25%에서 17.8% → 13.9%, 탐지율 손실 없음).
COVERAGE_BONUS_SCALE = 1.5
DEEPVOICE_BONUS = 0.10     # 딥보이스 탐지 결과를 S7 경로에 반영 (P1 과제)


def _dedupe(items: list[str]) -> list[str]:
    """순서를 지키며 중복 제거. 근거 패널에 같은 표현이 여러 번 뜨지 않게 한다."""
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


@dataclass
class Evidence:
    """통화 하나에서 지금까지 모은 증거. 발화가 들어올 때마다 **병합**한다.

    누적 버퍼를 매번 통째로 다시 채점하던 것을 이 구조가 대체한다.
    버퍼는 뒤에만 붙으므로 앞쪽 발화의 판정은 바뀌지 않는다 —
    다시 계산할 이유가 없었다.

    실측(발화 30건 시점, 채점 1회):

        구 매처 · 전체 재채점    60.7초
        신 매처 · 전체 재채점     2.14초
        신 매처 · 증거 누적       발화당 일정 (아래 참조)

    통화가 길어져도 발화당 비용이 늘지 않는 것이 핵심이다.
    30분 통화의 마지막 발화도 첫 발화와 같은 비용으로 채점된다.

    `first_seen`은 단계가 **처음 잡힌 발화 번호**다. 전개 순서를 보려면 필요하다.
    """

    hits: dict[str, float] = field(default_factory=dict)
    matched: dict[str, list[str]] = field(default_factory=dict)
    suppressed: dict[str, list[str]] = field(default_factory=dict)
    criticals: list[str] = field(default_factory=list)
    benign: list[str] = field(default_factory=list)
    specific: dict[str, bool] = field(default_factory=dict)
    first_seen: dict[str, int] = field(default_factory=dict)
    n_units: int = 0

    def merge(self, other: "Evidence") -> "Evidence":
        """새 발화의 증거를 흡수한다. 단계 점수는 max, 나머지는 합집합."""
        # 0점 단계도 기록한다. 값은 같지만 키가 빠지면 전체 재채점 결과와
        # 딕셔너리가 달라져, 두 경로를 비교하는 검증이 불가능해진다.
        for sid, v in other.hits.items():
            self.hits[sid] = max(self.hits.get(sid, 0.0), v)
        for sid, forms in other.matched.items():
            self.matched[sid] = _dedupe(self.matched.get(sid, []) + forms)
        for sid, forms in other.suppressed.items():
            self.suppressed[sid] = _dedupe(self.suppressed.get(sid, []) + forms)
        for sid, v in other.specific.items():
            self.specific[sid] = self.specific.get(sid, False) or v
        self.criticals = _dedupe(self.criticals + other.criticals)
        self.benign = _dedupe(self.benign + other.benign)
        for sid, idx in other.first_seen.items():
            self.first_seen.setdefault(sid, idx)      # 처음 본 시점을 지킨다
        self.n_units += other.n_units
        return self


@dataclass
class ScoreResult:
    score: float
    route: Route
    stage_hits: dict[str, float]
    matched: dict[str, list[str]] = field(default_factory=dict)
    criticals: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    benign_hits: list[str] = field(default_factory=list)
    suppressed: dict[str, list[str]] = field(default_factory=dict)
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
        if self.suppressed:
            drops = ", ".join(f"{k}({','.join(v[:2])})"
                              for k, v in sorted(self.suppressed.items()))
            lines.append(f"  인용으로 판단해 제외: {drops}")
        return "\n".join(lines)


class Scorer:
    def __init__(self, db: PatternDB, matcher: Matcher | None = None,
                 morph: list[MorphRule] | None = None) -> None:
        self.db = db
        self.matcher = matcher or Matcher()
        self.morph = build_rules() if morph is None else morph

    # ── 단계 매칭 ─────────────────────────────────────────────────────────────

    def stage_hits(
        self, text: str
    ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, list[str]],
               dict[str, bool]]:
        """단계별 최고 점수와 매칭된 표현, 그리고 문맥으로 제외된 표현.

        **합산이 아니라 max**를 쓴다. 같은 단계의 표현을 여러 번 말했다고 해서
        위험이 그만큼 커지는 것은 아니다. 합산하면 반복 언급이 과대평가된다.

        매칭은 **문장 단위**로 한다. 인용문 안의 표현은 세지 않는다.
        "안전계좌로 이체하라고 했어요"는 피해자의 신고이지 사기범의 지시가 아니다.
        판정 근거는 `context.py`에 있다.
        """
        sentences = split_sentences(text)
        hits: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        suppressed: dict[str, list[str]] = {}
        specific: dict[str, bool] = {}

        for sid, stage in self.db.stages.items():
            best = 0.0
            found: list[str] = []
            skipped: list[str] = []
            has_specific = False
            for kw in stage.keywords:
                got = self._first_direct_hit(sentences, kw.all_forms(), skipped,
                                             exact_only=kw.exact_only)
                if got:
                    form, _span = got
                    best = max(best, kw.score)     # ① default=0.0 대신 초기값 0
                    found.append(form)
                    if not kw.generic:
                        has_specific = True
            hits[sid] = best
            specific[sid] = has_specific
            if found:
                matched[sid] = found
            if skipped:
                suppressed[sid] = _dedupe(skipped)

        # 형태 규칙 — 표면형을 나열하는 대신 구성을 잡는다.
        # 키워드와 같은 문맥 판정(인용·고지·서술)을 그대로 통과시킨다.
        for rule in self.morph:
            for sent in sentences:
                span = rule.find(sent)
                if span is None:
                    continue
                if classify(sent, span).suppressed:
                    suppressed.setdefault(rule.target, []).append(rule.label)
                    continue
                hits[rule.target] = max(hits.get(rule.target, 0.0), rule.score)
                specific[rule.target] = True
                matched.setdefault(rule.target, []).append(rule.label)
                break
        for sid in list(matched):
            matched[sid] = _dedupe(matched[sid])
        for sid in list(suppressed):
            suppressed[sid] = _dedupe(suppressed[sid])
        return hits, matched, suppressed, specific

    def _first_direct_hit(self, sentences: list[str], forms: list[str],
                          skipped: list[str],
                          max_distance: int | None = None,
                          exact_only: bool = False
                          ) -> tuple[str, MatchSpan] | None:
        """인용이 아닌 첫 매칭을 (표현, 매칭구간)으로 돌려준다.

        구간이 필요한 이유는 두 가지다 — 문맥 판정이 위치를 쓰고,
        점수 가중이 편집거리를 쓴다.
        인용으로 걸러진 것은 `skipped`에 쌓아 근거 패널에서 보여준다.
        """
        for form in forms:
            for sent in sentences:
                span = self.matcher.find_span(sent, form, exact_only=exact_only,
                                              max_distance=max_distance)
                if span is None and not exact_only:
                    # 연속 매칭이 실패했을 때만 어절 분리 매칭을 시도한다.
                    # 한국어는 어절 사이에 조사·부사가 자유롭게 끼어들어
                    # "기존 대출을 먼저 상환하셔야"가 "기존 대출 상환"과 끊긴다.
                    span = self.matcher.find_gapped(sent, form,
                                                    max_distance=max_distance)
                if span is None:
                    continue
                if classify(sent, span).suppressed:
                    skipped.append(form)
                    continue
                return form, span
        return None

    # ── 고위험 신호 ───────────────────────────────────────────────────────────

    def find_criticals(self, text: str) -> tuple[list[str], list[str]]:
        """단독 고위험 신호. 인용된 것은 제외하고 별도로 보고한다."""
        sentences = split_sentences(text)
        out: list[str] = []
        skipped: list[str] = []
        for c in self.db.criticals:
            drop: list[str] = []
            if self._first_direct_hit(sentences, c.all_forms(), drop,
                                      max_distance=CRITICAL_MAX_EDIT):
                out.append(c.id)
            elif drop:
                skipped.append(c.id)
        return out, _dedupe(skipped)

    def find_pairs(self, hits: dict[str, float],
                   specific: dict[str, bool] | None = None) -> list[str]:
        """조합 신호. `needs_specific` 단계는 generic 표현만으로는 성립하지 않는다."""
        spec = specific or {}
        out = []
        for p in self.db.pairs:
            if not all(hits.get(s, 0.0) > 0 for s in p.stages):
                continue
            if any(not spec.get(s, True) for s in p.needs_specific):
                continue
            # 쌍의 **어느 단계에도** 구체적 증거가 없으면 성립하지 않는다.
            #
            # 검증셋 7차에서 렌터카 안내가 위험이 됐다(Y-B-05) —
            # "선납금"(S5)과 "보증금 먼저 결제"(S8) 둘 다 generic인데 P2가 발동했다.
            # 정상 상거래 용어 두 개가 만나 하한 0.75를 강제한 것이다.
            #
            # 5차에서는 "S5는 반드시 specific"이라는 더 강한 규칙을 시도했다가
            # 회피 시나리오 2건을 잃고 되돌렸다. 이건 그보다 약하다 —
            # 한쪽만 구체적이면 통과한다. 회피 시나리오는 안전계좌·원격제어 같은
            # 구체 표현을 쓰므로 영향받지 않는다.
            if not any(spec.get(s, False) for s in p.stages):
                continue
            out.append(p.id)
        return out

    def find_benign(self, text: str) -> list[str]:
        """benign은 **정확 일치만** 쓴다 — 비대칭 설계.

        근사매칭을 허용하면 위험 표현이 benign으로 잘못 걸려 점수를 깎는다.
        놓치는 비용보다 잘못 깎는 비용이 크다.
        """
        return self.matcher.find_all(text, self.db.benign, exact_only=True)

    # ── 본체 ──────────────────────────────────────────────────────────────────

    def extract(self, text: str, index: int = 1) -> Evidence:
        """발화 하나에서 증거를 뽑는다. 누적은 `Evidence.merge`가 맡는다.

        :param index: 이 발화의 번호. 단계가 처음 등장한 시점을 기록하는 데 쓴다.
        """
        hits, matched, suppressed, specific = self.stage_hits(text)
        criticals, crit_skipped = self.find_criticals(text)
        if crit_skipped:
            suppressed = {**suppressed, "critical": crit_skipped}
        return Evidence(
            hits=hits,
            matched=matched,
            suppressed=suppressed,
            criticals=criticals,
            benign=self.find_benign(text),
            specific=specific,
            first_seen={s: index for s, v in hits.items() if v > 0},
            n_units=1,
        )

    def score(
        self,
        buffer_text: str,
        route: Route | None = None,
        deepvoice_score: float = 0.0,
        deepvoice_threshold: float = 0.7,
    ) -> ScoreResult:
        """전사 텍스트를 통째로 채점한다.

        통화 중에는 `CallState`가 발화 단위로 증거를 누적하므로 이 경로를 타지 않는다.
        단발 채점·테스트·오프라인 분석용 진입점이다.
        """
        return self.score_evidence(
            self.extract(buffer_text), route, deepvoice_score, deepvoice_threshold
        )

    def score_evidence(
        self,
        ev: Evidence,
        route: Route | None = None,
        deepvoice_score: float = 0.0,
        deepvoice_threshold: float = 0.7,
    ) -> ScoreResult:
        """누적된 증거로 위험도를 낸다."""
        hits, matched, suppressed = ev.hits, ev.matched, dict(ev.suppressed)
        route = route or route_of(hits)
        active = route.stages

        # 단계 가중 점수
        stage_score = sum(
            hits.get(s, 0.0) * self.db.stages[s].weight for s in active
        )

        # ② 커버리지 — 0보다 큰 단계만 계수한다. 조건이 없으면 항상 1.0이 된다.
        cov = sum(1 for s in active if hits.get(s, 0.0) > 0) / len(active)
        bonus = (cov ** 2) * COVERAGE_BONUS_SCALE

        benign_hits = ev.benign
        penalty = min(len(benign_hits) * BENIGN_PENALTY, BENIGN_PENALTY_MAX)

        # ③ 경로별 이론 최대로 정규화 — 8단계 전체로 나누면 가족사칭형이 불리해진다
        raw = (stage_score + bonus) / route.denominator - penalty
        score = min(1.0, max(0.0, raw))

        criticals = ev.criticals
        pairs = self.find_pairs(hits, ev.specific)

        # 하한은 감점을 받지 않는다 — 감점으로 무너뜨릴 수 있으면 하한이 아니다.
        #
        # 한때 `CRITICAL_FLOOR - penalty`로 두었다. 예방 교육 대화가 위험으로 뜨는 것을
        # 막으려던 것인데, 회피 평가에서 **전면 우회로**로 드러났다(F-E-02).
        # 사기범이 예방 문구 6개를 흘리자 감점 0.90이 하한 0.75를 지워
        # C1·C2·P1이 모두 발동한 통화가 0.000 "안전"이 됐다.
        #
        # 원래 막으려던 예방 교육 대화는 인용 판정이 이미 처리한다 —
        # 그런 대화에서는 위험 표현이 전부 인용문이라 critical이 **아예 발동하지 않는다.**
        # 같은 문제를 두 군데서 풀려다 구멍을 냈다.
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
            suppressed=suppressed,
            intervene=score >= THRESHOLD_ALERT,
        )


class CallState:
    """통화 하나의 누적 상태.

    모드 1이 stateful인 이유가 이 클래스에 있다.
    발화가 들어올 때마다 **새 발화만 채점해 증거에 누적**하고, 그 증거로 다시 판정한다.

    예전에는 발화마다 누적 버퍼 전체를 다시 채점했다. 통화가 길어질수록
    발화당 비용이 선형으로 늘어 30분 통화에서는 실시간 개입이 불가능했다.
    버퍼는 뒤에만 붙으므로 앞쪽 판정은 바뀌지 않는다 — 다시 잴 이유가 없다.
    """

    def __init__(self, scorer: Scorer) -> None:
        self.scorer = scorer
        self.utterances: list[str] = []
        self.evidence = Evidence()
        self.last: ScoreResult | None = None
        self.intervened = False

    @property
    def buffer_text(self) -> str:
        """줄바꿈으로 잇는다 — 발화 경계가 곧 문장 경계다.

        공백으로 이으면 앞 발화의 끝과 뒤 발화의 시작이 붙어 없던 표현이 생기고,
        문맥 판정도 발화를 넘어 잘못 걸린다. 분할기가 이미 의미 단위를 보장한다.
        """
        return "\n".join(self.utterances)

    def add_utterance(self, text: str, deepvoice_score: float = 0.0) -> ScoreResult:
        self.utterances.append(text)
        self.evidence.merge(self.scorer.extract(text, index=len(self.utterances)))
        self.last = self.scorer.score_evidence(
            self.evidence, deepvoice_score=deepvoice_score
        )
        return self.last

    def should_intervene(self) -> bool:
        """개입은 통화당 한 번만. 반복 재생은 오히려 각성을 방해한다."""
        if self.last and self.last.intervene and not self.intervened:
            self.intervened = True
            return True
        return False

    def reset(self) -> None:
        self.utterances.clear()
        self.evidence = Evidence()
        self.last = None
        self.intervened = False
