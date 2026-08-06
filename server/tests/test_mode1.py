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
    """DB 규모 — 문서 서술과 어긋나면 심사에서 "몇 개냐"가 나온다.

    D08 §02 원안은 기본 72 · 변형 110 · 합계 182였다.
    실제 공개 녹취로 교차검증(`eval/scenarios_real.json`)하면서 5항을 더했다 —
    S4 물리적 고립 유도 3항, S5 대면편취 2항. 출처는 각 항목의 `source` 필드에 있다.

    왜 더했나. **전체 보이스피싱의 66.9%가 대면편취형**(현금 인출 후 직접 전달)인데
    DB가 계좌이체 중심이라 실제 녹취 2건을 통째로 놓쳤다.
    """
    assert db.n_base == 78, f"기본 표현 {db.n_base}개"
    assert db.n_variants == 123, f"변형 {db.n_variants}개"
    assert db.n_total == 201, f"합계 {db.n_total}개"


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


def test_every_matchable_form_yields_an_available_fragment(scorer, db):
    """**변형으로 매칭돼도** 음성 조각이 반드시 있어야 한다.

    위 테스트는 기본 표현만 확인해서 이 결함을 놓쳤다.
    `build_warning`이 인용하는 것은 **실제로 매칭된 표현**이고 그건 변형일 수 있다.
    실측: "지금 바로 송금해 주세요" → `kw::송금` 요청 → 뱅크에 없음.
    5건 중 3건이 없는 조각을 요구했다("송금"·"현금으로 찾"·"팀뷰어").

    조용히 틀리는 종류다 — 서버는 정상 응답하고 화면에도 경고가 뜬다.
    **소리만 안 난다.** 통화 중에는 화면을 못 보므로 음성이 주 매체인데,
    하필 그 순간에 침묵한다.

    DB의 모든 표현(기본 + 변형)을 하나씩 실제로 채점해 확인한다.
    """
    manifest = set(tts_bank_manifest(db))
    bad: list[tuple[str, str]] = []

    for stage in db.stages.values():
        for kw in stage.keywords:
            for form in kw.all_forms():
                r = scorer.score(f"{form} 지금 처리하세요.")
                if not r.matched:
                    continue                    # 짧아서 안 잡히는 표현은 인용되지도 않는다
                w = build_warning(r, db)
                missing = [t for t in w.tts_tokens if t not in manifest]
                if missing:
                    bad.append((form, missing[0]))

    assert not bad, (
        f"음성 조각이 없는 표현 {len(bad)}건 — 이 표현이 잡히면 경고가 무음이 된다: "
        f"{bad[:5]}"
    )


def test_screen_keeps_the_exact_phrase_while_audio_uses_the_base_form(scorer, db):
    """화면은 들린 그대로, 음성은 기본 표현.

    원칙1(구체성)은 방금 들린 말을 그대로 보여주라고 한다. 그건 화면이 지킨다.
    음성까지 변형을 쓰면 뱅크에 없는 조각을 요청하게 되고,
    STT 오차형("안전개좌")을 그대로 읽어 잘못된 발음을 들려주게 된다.
    """
    r = scorer.score("지금 바로 송금해 주세요.")
    w = build_warning(r, db)

    assert "송금" in w.quote, f"화면에서 실제 표현이 사라졌다: {w.quote}"
    assert "kw::이체" in w.tts_tokens, f"음성이 기본 표현을 쓰지 않았다: {w.tts_tokens}"
    assert "kw::송금" not in w.tts_tokens


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


# ── 근사매칭 오탐 회귀 (eval_mode1.py가 찾아낸 것) ─────────────────────────────

@pytest.mark.parametrize("sentence,keyword", [
    ("여신상담 담당자입니다", "당장"),        # 음절 경계를 넘어 "담**당자**입" → 당장
    ("조금 전 해외 결제 승인", "송금"),
    ("등록금 내야 하는데", "송금"),
    ("궁금하신 점 있으면", "송금"),
    ("조심해야 돼", "실형"),
    ("방금 상대방이", "벌금"),
    ("지금 전화를 끊으셔도", "급전"),
    ("맞으면 그대로 두시면", "늦으면"),
    ("서부경찰서 수사과", "경찰청"),
    ("엄마 요즘 보이스피싱", "엄마 나"),
    ("오늘 날씨가 참 좋네요", "오늘까지"),
    # 아래 셋은 **실제 오디오를 통째로 돌려보고** 찾았다 (`e2e_test.py`).
    # 텍스트 시나리오에는 없던 형태라 그전까지 안 잡혔다.
    ("계좌에 있는 돈을 안전하게 관리하기 위해", "계좌이체"),
    ("기존 대출금을 현금으로 상환하여야 합니다", "현금으로 찾"),
    ("상환금을 지금 불러드리는 계좌로", "지금 바로"),
])
def test_matcher_no_phantom_match_in_ordinary_speech(sentence, keyword):
    """정상 발화에 없는 표현이 "있다"고 나오면 안 된다.

    전부 `eval_mode1.py`의 정상 통화 12건에서 **실제로 잡혔던** 사례다.
    이 오탐 하나하나가 위험 단계를 켜서 오탐률을 58.3%까지 올렸다.

    원인 세 가지를 고쳤다 — 음절 경계 정렬 · 4음절 미만 근사 차단 · 초성 선별.
    자세한 것은 `matcher.py` 모듈 설명에 있다.
    """
    assert not Matcher().match(sentence, keyword)


def test_matcher_respects_syllable_boundary():
    """자모 창이 음절 중간에서 시작하면 없던 표현이 생긴다.

    "담당자" + "입" 의 자모가 이어지면 "당장"(ㄷㅏㅇㅈㅏㅇ)이 **정확히** 나타난다.
    편집거리 0이라 어떤 임계값으로도 막을 수 없었다. 창을 음절에 정렬해야 사라진다.
    """
    m = Matcher()
    assert not m.match("담당자입니다", "당장")
    assert m.match("지금 당장 이체하세요", "당장")     # 진짜로 있으면 잡아야 한다


# ── 문맥 판정 — 인용은 사기가 아니다 ──────────────────────────────────────────

def test_quoted_risk_phrase_is_not_an_alert(scorer):
    """피해자가 사기범의 말을 옮기는 것은 사기가 아니다.

    1332 상담·경찰 신고에서 피해자는 반드시 사기범 발언을 인용한다.
    이걸 위험으로 잡으면 신고 통화마다 경고가 뜬다.
    """
    direct = scorer.score("안전계좌로 이체하세요.")
    quoted = scorer.score("안전계좌로 이체하라고 했어요.")
    assert direct.score >= THRESHOLD_ALERT, f"직접 지시 {direct.score:.3f}"
    assert quoted.score < THRESHOLD_WARN, f"인용 {quoted.score:.3f}"


def test_own_warning_text_does_not_retrigger(scorer):
    """미리내가 띄운 경고를 사용자가 읽으면 그 소리가 다시 마이크로 들어간다.

    자기 경고에 자기가 반응하면 개입이 무한히 반복된다.
    `warning.py`의 반박 문구가 그대로 위험 판정을 받던 실제 결함이다.
    """
    r = scorer.score("안전계좌라는 것은 존재하지 않습니다.")
    assert r.score < THRESHOLD_WARN, f"{r.score:.3f} — 자기 경고문에 반응했다"


def test_prevention_education_is_not_an_alert(scorer):
    """예방 교육 대화 — 위험 표현이 전부 나오지만 전부 인용·부정이다."""
    r = scorer.score(
        "엄마, 요즘 보이스피싱 진짜 조심해야 돼. "
        "검찰청이라고 하면서 전화 오면 그거 사기야. "
        "안전계좌로 이체하라는 말 나오면 바로 끊어. 그런 계좌는 존재하지 않아. "
        "수사 기밀이니까 가족에게도 말하지 말라고 하는 것도 다 수법이야."
    )
    assert r.score < THRESHOLD_WARN, f"{r.score:.3f} — 예방 교육을 사기로 봤다"


def test_scammer_direct_command_still_fires(scorer):
    """인용 억제가 사기범의 직접 지시까지 지우면 안 된다."""
    r = scorer.score(
        "서울중앙지방검찰청 수사관입니다.\n"
        "귀하 명의 계좌가 자금세탁에 연루되었습니다.\n"
        "수사 기밀이니 가족에게도 말하지 마십시오.\n"
        "국가안전계좌로 즉시 이체하십시오."
    )
    assert r.score >= THRESHOLD_ALERT, f"{r.score:.3f}"
    assert "C1" in r.criticals


# ── critical 하한과 정상 문맥 감점 ────────────────────────────────────────────

def test_benign_phrases_cannot_defeat_critical_floor(scorer):
    """감점으로 무너뜨릴 수 있으면 하한이 아니다.

    회피 평가 F-E-02가 찾아낸 구멍이다. 한때 하한을 `CRITICAL_FLOOR - penalty`로 두었다 —
    예방 교육 대화가 위험으로 뜨는 것을 막으려던 것인데, 사기범이 예방 문구를 여섯 개
    흘리자 감점 0.90이 하한 0.75를 지워 **C1·C2·P1이 모두 발동한 통화가 0.000 "안전"**이 됐다.

    SBS 공개 녹취에서 사기범이 실제로 예방 문구를 말한다("계좌번호 절대 말씀 마시고").
    가정이 아니라 이미 쓰이는 수법이다.

    원래 막으려던 예방 교육 대화는 인용 판정이 처리한다 —
    그런 대화에서는 위험 표현이 전부 인용문이라 critical이 아예 발동하지 않는다.
    """
    bare = scorer.score("안전계좌로 이체하세요.")
    padded = scorer.score(
        "보이스피싱 조심하셔야 합니다.\n"
        "저희는 절대 알려주지 마시라고 안내드립니다.\n"
        "112에 신고하셔도 되고 1332로 확인하셔도 됩니다.\n"
        "본인 확인 절차 진행하겠습니다. 천천히 생각하세요.\n"
        "안전계좌로 이체하세요."
    )
    assert bare.score >= THRESHOLD_ALERT
    assert padded.score >= THRESHOLD_ALERT, (
        f"{padded.score:.3f} — 예방 문구 도배로 하한이 무너졌다 "
        f"(benign {len(padded.benign_hits)}개)"
    )


def test_benign_penalty_is_capped(scorer):
    """감점 총량에 상한이 없으면 감점 자체가 회피 수단이 된다.

    benign 목록의 상당수는 사기범도 쓰는 말이다 — '본인 확인 절차'·'112에 신고'·'1332'.
    무제한 누적을 허용하면 그 말들을 늘어놓는 것만으로 점수를 0까지 끌어내릴 수 있다.
    """
    from mirinae.mode1.scorer import BENIGN_PENALTY, BENIGN_PENALTY_MAX
    assert BENIGN_PENALTY_MAX < len(scorer.db.benign) * BENIGN_PENALTY, (
        "상한이 benign 전체 합보다 크면 상한이 아니다"
    )

    many = scorer.score(
        "보이스피싱 조심. 사기 전화. 절대 알려주지 마. 112에 신고. 1332. "
        "본인 확인 절차. 영업점 방문. 가족과 상의. 천천히 생각."
    )
    assert len(many.benign_hits) >= 5, "감점 상한을 시험할 만큼 안 잡혔다"


def test_critical_signals_demand_stricter_evidence(scorer):
    """판정이 무거운 신호일수록 더 엄격한 증거를 요구한다.

    C1~C5는 하나만 걸려도 위험도를 0.75로 밀어올린다. 일반 단계 키워드와 같은
    허용치를 쓰면, STT가 망가뜨린 정상 통화가 그 하나에 걸려 즉시 위험이 된다.
    """
    from mirinae.mode1.matcher import Matcher
    from mirinae.mode1.scorer import CRITICAL_MAX_EDIT
    m = Matcher()

    # 자모 1개 오차는 여전히 흡수한다 — 근사매칭의 존재 이유다
    assert m.match("안전개좌로 이체", "안전계좌", max_distance=CRITICAL_MAX_EDIT)
    # 그보다 헐거운 매칭은 critical에는 허용하지 않는다
    assert m.match("계좌이채하세요", "계좌이체")
    assert not m.match("게자이채하세요", "계좌이체", max_distance=CRITICAL_MAX_EDIT)


# ── 증거 누적이 전체 재채점과 같은 결과를 내는가 ──────────────────────────────

def test_incremental_matches_full_rescore(scorer):
    """`CallState`는 새 발화만 채점해 누적한다. 전체를 다시 채점한 것과 같아야 한다.

    같지 않으면 성능 최적화가 판정을 바꾼 것이고, 그건 최적화가 아니라 버그다.
    """
    utterances = [
        "서울중앙지방검찰청 수사관입니다.",
        "귀하 명의 계좌가 자금세탁에 이용됐습니다.",
        "수사 기밀이니 가족에게도 말하지 마십시오.",
        "안전계좌로 즉시 이체하십시오.",
    ]
    state = CallState(scorer)
    for u in utterances:
        state.add_utterance(u)

    full = scorer.score("\n".join(utterances))
    assert state.last.score == pytest.approx(full.score)
    assert state.last.route.id == full.route.id
    assert sorted(state.last.criticals) == sorted(full.criticals)
    assert state.last.stage_hits == full.stage_hits


def test_evidence_records_first_appearance(scorer):
    """단계가 처음 등장한 발화 번호를 기록한다 — 전개 순서 분석의 재료."""
    state = CallState(scorer)
    state.add_utterance("서울중앙지방검찰청 수사관입니다.")          # S1
    state.add_utterance("오늘 날씨가 좋네요.")                      # 아무것도 없음
    state.add_utterance("수사 기밀이니 가족에게도 말하지 마십시오.")   # S4

    seen = state.evidence.first_seen
    assert seen["S1"] == 1
    assert seen["S4"] == 3
    assert seen["S1"] < seen["S4"], "처음 본 시점이 뒤바뀌었다"
