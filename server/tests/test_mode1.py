"""모드 1 단위 테스트 — D08 §07 표를 그대로 구현한다.

주의 — 여기 있는 것은 **알고리즘 구조 테스트**이지 탐지율 평가가 아니다.
탐지율은 DB 작성자와 분리된 사람이 만든 시나리오로 재야 한다(D08 §07 평가의 순환 문제).
이 파일과 pattern_db.json을 같은 사람이 썼다면, 여기서 나온 수치는 성능 근거가 될 수 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirinae.mode1 import Matcher, load_db                              # noqa: E402
from mirinae.mode1.router import ROUTES, FALLBACK, route_of             # noqa: E402
from mirinae.mode1.scorer import (                                      # noqa: E402
    CallState, Scorer, THRESHOLD_ALERT, THRESHOLD_WARN,
)
from mirinae.mode1.warning import build_warning, tts_bank_manifest      # noqa: E402


@pytest.fixture(scope="module")
def db():
    return load_db()


@pytest.fixture(scope="module")
def scorer(db):
    return Scorer(db)


# ── DB 정합성 ─────────────────────────────────────────────────────────────────

def test_db_matches_design_doc_counts(db):
    """D08 §02 — 기본 72개 · 변형 110개 · 합계 182개.

    문서와 DB가 어긋나면 심사에서 "182개라더니 몇 개냐"가 나온다.
    """
    assert db.n_base == 72, f"기본 표현 {db.n_base}개"
    assert db.n_variants == 110, f"변형 {db.n_variants}개"
    assert db.n_total == 182, f"합계 {db.n_total}개"


def test_stage_weights_match_doc(db):
    expected = {"S1": 0.80, "S2": 0.70, "S3": 0.85, "S4": 1.00,
                "S5": 0.90, "S6": 0.60, "S7": 0.85, "S8": 0.70}
    for sid, w in expected.items():
        assert db.stages[sid].weight == w, f"{sid} 가중치"


def test_route_denominators_match_doc():
    """D08 §04 — 분모 A 6.35 · B 5.70 · C 3.70."""
    assert ROUTES["A"].denominator == pytest.approx(6.35)
    assert ROUTES["B"].denominator == pytest.approx(5.70)
    assert ROUTES["C"].denominator == pytest.approx(3.70)


def test_db_has_all_criticals_and_pairs(db):
    assert [c.id for c in db.criticals] == ["C1", "C2", "C3", "C4", "C5"]
    assert [p.id for p in db.pairs] == ["P1", "P2", "P3"]


# ── D08 §07 테스트 표 ─────────────────────────────────────────────────────────

def test_no_match_no_crash(scorer):
    """매칭 0건 → 점수 0.0. 잡는 결함 — 빈 시퀀스 오류."""
    r = scorer.score("오늘 날씨가 참 좋네요. 점심 뭐 드셨어요?")
    assert r.score == 0.0
    assert r.level == "안전"


def test_coverage_not_always_one(scorer):
    """1단계만 매칭 → cov = 1/len(act). 잡는 결함 — 커버리지 상시 최대."""
    r = scorer.score("서울중앙지방검찰청 수사관입니다.")
    assert r.route.id == "A"
    assert r.coverage == pytest.approx(1 / len(r.route.stages))
    assert r.score < THRESHOLD_WARN, f"1단계만으로 {r.score:.3f} — 너무 높다"


def test_pair_family(scorer):
    """S7+S5 → ≥ 0.75. 딥보이스 시나리오 누락 회귀 방지.

    이 프로젝트가 방어하려는 바로 그 시나리오다.
    기존 방식(8단계 전체 분모)에서는 0.200 '낮음'이었다.
    """
    r = scorer.score("엄마 나야. 폰이 고장나서 친구 폰으로 전화했어. 급하게 돈 좀 송금해줘.")
    assert r.route.id == "B", f"경로 {r.route.id}"
    assert "P3" in r.pairs
    assert r.score >= THRESHOLD_ALERT, f"{r.score:.3f}"


def test_critical_floor(scorer):
    """'안전계좌' 단독 → ≥ 0.75. 잡는 결함 — C1~C5 미발동."""
    r = scorer.score("안전계좌로 옮기시면 됩니다.")
    assert "C1" in r.criticals
    assert r.score >= THRESHOLD_ALERT


def test_benign_suppress(scorer):
    """예방 교육 문맥 → < 0.45. 잡는 결함 — 오탐."""
    r = scorer.score(
        "보이스피싱 조심하셔야 해요. 사기 전화가 오면 절대 알려주지 마시고 "
        "112에 신고하세요. 꼭 가족과 상의하시고 천천히 생각하세요."
    )
    assert r.score < THRESHOLD_WARN, f"{r.score:.3f} — 정상 통화를 위험으로 봤다"


def test_stt_variants(scorer):
    """'안전 계좌'·'안전개좌' → 동일 판정. 잡는 결함 — STT 오차 취약성.

    정확 일치만 쓰면 21.9%만 잡힌다. 이 테스트가 그 회귀를 막는다.
    """
    for text in ("안전계좌로 이체하세요",
                 "안전 계좌로 이체하세요",
                 "안전개좌로 이체하세요"):
        r = scorer.score(text)
        assert r.score >= THRESHOLD_ALERT, f"“{text}” → {r.score:.3f}"


def test_buffer_accumulate(scorer):
    """발화 3개 순차 투입 → 커버리지가 늘어야 한다. 잡는 결함 — 상태 유지 실패.

    모드 1이 stateful인지 확인하는 테스트다.
    최신 발화만 채점하면 커버리지가 1/N에 고정된다.
    """
    state = CallState(scorer)
    covs = []
    for utt in ("서울중앙지방검찰청 수사관입니다.",
                "귀하 명의 계좌가 자금세탁에 이용됐습니다.",
                "체포영장이 발부된 상태입니다."):
        r = state.add_utterance(utt)
        covs.append(r.coverage)

    assert covs == sorted(covs), f"커버리지가 늘지 않았다: {covs}"
    assert covs[-1] > covs[0], "누적이 안 되고 있다"
    assert state.last.stage_hits["S1"] > 0, "첫 발화의 S1이 사라졌다"


def test_route_fallback(scorer):
    """유형 판정 실패 → 전 단계 활성. 잡는 결함 — 라우팅 실패 시 무판정."""
    r = scorer.score("지금 당장 이체하세요.")     # 진입 단계(S1/S7/S8) 없음
    assert r.route.id == FALLBACK.id
    assert len(r.route.stages) == 8


def test_route_of_picks_strongest():
    assert route_of({"S1": 1.0}).id == "A"
    assert route_of({"S7": 1.0}).id == "B"
    assert route_of({"S8": 1.0}).id == "C"
    assert route_of({}).id == FALLBACK.id


# ── 개입 ──────────────────────────────────────────────────────────────────────

def test_intervention_fires_at_s4(scorer):
    """S4 도달 시 개입 플래그. 자금 이동 전이면서 오탐률이 가장 낮은 구간이다."""
    state = CallState(scorer)
    state.add_utterance("서울중앙지방검찰청 수사관입니다.")
    assert not state.should_intervene()

    state.add_utterance("귀하 명의가 도용되어 대포통장에 이용됐습니다.")
    assert not state.should_intervene()

    state.add_utterance("수사 기밀이니 가족에게도 말하지 마십시오.")
    assert state.last.intervene, f"S4에서 개입 안 함 ({state.last.score:.3f})"
    assert state.should_intervene()


def test_intervention_fires_once(scorer):
    """개입은 통화당 한 번. 반복 재생은 오히려 각성을 방해한다."""
    state = CallState(scorer)
    state.add_utterance("안전계좌로 이체하세요.")
    assert state.should_intervene()
    state.add_utterance("지금 당장 해야 합니다.")
    assert not state.should_intervene()


def test_warning_quotes_the_phrase(scorer, db):
    """원칙1 — '의심됩니다'가 아니라 방금 들린 말을 그대로 인용해야 한다."""
    r = scorer.score("안전계좌로 즉시 이체하세요.")
    w = build_warning(r, db)

    assert "안전계좌" in w.quote, w.quote
    assert "존재하지 않습니다" in " ".join(w.counter)
    assert "끊으셔도" in w.control            # 원칙2 통제감
    assert w.cross_check                      # 원칙3 교차검증
    assert "112" in w.action                  # 원칙5 단일 행동


def test_warning_uses_scammer_wording(scorer, db):
    """용어는 사기범이 쓴 말 그대로. '가짜 계좌'로 번역하면 연결이 끊긴다."""
    r = scorer.score("안전계좌로 이체하세요.")
    w = build_warning(r, db)
    assert "가짜" not in w.quote


def test_warning_cross_check_differs_by_route(scorer, db):
    """유형마다 확인해야 할 상대가 다르다 — 가족사칭이면 자녀에게."""
    fam = build_warning(
        scorer.score("엄마 나야. 급하게 돈이 필요해. 송금 좀 해줘."), db)
    gov = build_warning(
        scorer.score("검찰청 수사관입니다. 안전계좌로 이체하세요."), db)
    assert fam.cross_check != gov.cross_check
    assert "자녀" in fam.cross_check


def test_tts_bank_covers_every_quotable_phrase(db):
    """인용될 수 있는 표현은 전부 사전 생성 목록에 있어야 한다.

    런타임에 없는 조각을 찾으면 그 자리에서 시연이 멈춘다.
    """
    manifest = tts_bank_manifest(db)
    for c in db.criticals:
        assert f"kw::{c.text}" in manifest
    for stage in db.stages.values():
        for kw in stage.keywords:
            assert f"kw::{kw.text}" in manifest


# ── 매처 ──────────────────────────────────────────────────────────────────────

def test_matcher_normalizes_spacing():
    m = Matcher()
    assert m.match("안전 계좌로 이체", "안전계좌")
    assert m.match("안전계좌로 이체", "안전 계좌")


def test_matcher_approx_catches_stt_error():
    m = Matcher()
    assert m.match("안전개좌로 이체하세요", "안전계좌")


def test_matcher_exact_only_rejects_approx():
    """benign 지시어에 쓰는 비대칭 설계의 한쪽."""
    m = Matcher()
    assert not m.match("안전개좌로 이체하세요", "안전계좌", exact_only=True)


def test_matcher_does_not_overmatch_short_words():
    """짧은 표현을 근사로 풀면 아무 데나 걸린다."""
    m = Matcher()
    assert not m.match("오늘 날씨가 좋네요", "이체")
    assert not m.match("점심 뭐 먹을까", "검사")


def test_matcher_does_not_match_ordinary_speech():
    """오탐 회귀 방지 — 실제로 잡혔던 사례들.

    "오늘 날씨"가 "오늘까지"(S6)에 매칭돼 정상 통화가 위험으로 잡혔다.
    비율만 쓰면 중간 길이 표현에서 허용치가 과하게 커지는 것이 원인이었다.
    """
    m = Matcher()
    assert not m.match("오늘 날씨가 참 좋네요", "오늘까지")
    assert not m.match("어제 친구를 만났어요", "오늘까지")
    assert not m.match("점심 뭐 드셨어요", "시간이 없")


def test_matcher_still_catches_intended_stt_errors():
    """오탐을 줄이면서 의도한 케이스는 유지되어야 한다."""
    m = Matcher()
    assert m.match("안전개좌로 이체", "안전계좌")
    assert m.match("가족에게도 말하지 마세요", "가족에게도 말하지")
