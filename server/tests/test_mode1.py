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

    이후 놓친 시나리오 4건(F-B-03·F-C-01·F-E-04·R-A-02)을 분석해 11항을 더했다.
    전부 **완곡·우회 표현**이었다 — "안전계좌" 대신 "제가 알려드리는 계좌",
    "이체" 대신 "옮겨 두시면", "오늘" 대신 "금일 중".

    검증셋 2차에서 3항을 더 더했다 — 메신저피싱의 신분증·카드 사진 요구(S5),
    릴레이 사기 1단계의 "끊지 마시고"(S4). 둘 다 실제 수법인데 비어 있었다.
    """
    assert db.n_base == 108, f"기본 표현 {db.n_base}개"
    assert db.n_variants == 245, f"변형 {db.n_variants}개"
    assert db.n_total == 353, f"합계 {db.n_total}개"


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
    assert [c.id for c in db.criticals] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    assert [p.id for p in db.pairs] == ["P1", "P2", "P3", "P4"]


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


# ── 어절 분리 매칭 (gapped) ───────────────────────────────────────────────────

def test_gapped_matching_survives_particle_insertion():
    """한국어는 어절 사이에 조사·부사가 자유롭게 끼어든다.

    아래 셋은 사기범이 노려서 쓴 회피가 아니라 **그냥 말하면 이렇게 된다.**
    연속 문자열 매칭은 구조적으로 못 잡았고, 자모 근사로도 풀 수 없다 —
    끼어든 "을 먼저 "는 자모 7개로 편집거리 예산(최대 2)을 훨씬 넘는다.
    """
    m = Matcher()
    assert m.match("기존 대출을 먼저 상환하셔야 신규 승인이 가능합니다", "기존 대출 상환")
    assert m.match("가족에게 이 사실을 알리지 마세요", "가족에게 알리지")
    assert m.match("국고 보호 예치 계좌로 자금을 이관하셔야", "국고 계좌")


def test_gapped_matching_respects_distance_limit():
    """멀리 떨어진 어절은 같은 표현으로 보지 않는다.

    상한을 두지 않으면 "가족"과 "말하지"가 문장 어디에 있든 걸린다.
    값(4음절)은 감이 아니라 스윕으로 골랐다 — 3 이하면 이득이 없고
    4~12는 결과가 같아서, 회피 여지가 가장 작은 4를 택했다.
    """
    m = Matcher()
    assert not m.match("가족들이랑 상의해보고 나중에 천천히 말하지 뭐", "가족에게 알리지")


def test_gapped_matching_is_fallback_only():
    """연속 매칭이 되는 경우의 판정을 바꾸지 않는다."""
    m = Matcher()
    span = m.find_span("기존 대출 상환이 필요합니다", "기존 대출 상환")
    assert span is not None and span.kind == "exact"


# ── generic 표현과 조합 신호 게이트 ───────────────────────────────────────────

def test_generic_transfer_alone_does_not_trigger_pair(scorer):
    """"이체"·"송금"만으로는 P4(고립+자금이동)가 발동하면 안 된다.

    검증셋에서 이 게이트 없이 오탐률이 42.9%였다. 그중 하나가
    **은행 직원이 보이스피싱을 말리는 통화**였다 — 막아야 할 쪽을 사기로 봤다.
    돈을 보낸다는 사실 자체는 사기의 증거가 아니다.
    """
    state = CallState(scorer)
    for line in [
        "고객님, 이체 금액이 커서 확인이 좀 필요합니다.",
        "여기 말고 조용한 곳으로 가서 잠깐 말씀 나눌 수 있을까요?",
        "가족분께 먼저 연락해 보시고 천천히 생각해 보세요.",
    ]:
        r = state.add_utterance(line)
    assert "P4" not in r.pairs
    assert r.level == "안전", r.why()


def test_specific_transfer_triggers_pair(scorer):
    """화자가 즉석에서 지정하는 계좌는 다르다. 이건 정상 통화에 없다."""
    state = CallState(scorer)
    for line in [
        "수사 기밀이니 주변에 알리지 마시고 조용한 공간으로 이동해 주세요.",
        "안전한 보관을 위해 제가 안내드리는 계좌로 예치하시면 됩니다.",
    ]:
        r = state.add_utterance(line)
    assert "P4" in r.pairs
    assert r.level == "위험", r.why()


def test_honorific_direction_separates_scam_from_normal(scorer):
    """한국어 경어법이 계좌 지정의 방향을 구분해 준다.

    사기  "제가 알려드리는 계좌로"   겸양 '-드리는' → 화자가 계좌를 준다
    정상  "계좌번호 알려주시면"      존경 '-주시면'  → 청자가 계좌를 준다

    한 발화만 보면 둘 다 S5 하나라 **점수는 같다.** 갈리는 곳은 그다음이다 —
    겸양 쪽만 generic이 아니므로 고립 신호와 만났을 때 P4를 성립시킨다.
    """
    scam = scorer.extract("제가 지금 알려드리는 계좌로 입금해 주세요")
    normal = scorer.extract("환급받으실 계좌번호를 알려주시면 입금해 드리겠습니다")
    assert scam.hits["S5"] > 0 and normal.hits["S5"] > 0
    assert scam.specific["S5"] is True
    assert normal.specific["S5"] is False


# ── 서술 맥락 (검증셋 3차) ────────────────────────────────────────────────────

def test_descriptive_context_spares_fraud_warning_broadcast(scorer):
    """은행 ARS의 보이스피싱 경고 안내를 사기로 보면 안 된다.

    수법을 3인칭으로 나열하므로 인용 어미가 없다. 그래서 인용 판정이
    걸러내지 못했고 C1·C5가 발동해 0.750 위험이 됐다(T-B-02).
    """
    state = CallState(scorer)
    for line in [
        "우리은행 안내입니다.",
        "검찰이나 금감원을 사칭해 안전계좌로 이체를 요구하는 전화가 있습니다.",
        "가족에게 말하지 말라거나 원격제어 앱 설치를 요구하면 사기입니다.",
        "의심되시면 112나 1332로 신고해 주세요.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "안전", r.why()


def test_descriptive_markers_are_narrow_enough_to_resist_evasion(scorer):
    """'사기'라는 단어를 흘리는 것만으로 억제가 켜지면 안 된다.

    "요즘 사기가 많으니 안전계좌로 옮기세요"가 통과되면 그 자체가 회피로다.
    그래서 서술 **서술어** 형태만 표지로 인정한다.
    """
    r = scorer.score("요즘 사기가 많으니 지금 바로 안전계좌로 옮기세요")
    assert "C1" in r.criticals, r.why()
    assert r.level == "위험", r.why()


def test_disclaimer_evasion_does_not_erase_the_call(scorer):
    """고지 문구를 붙여 자기 키워드를 지우려는 회피는 통하지 않는다.

    억제는 **문장 단위**다. 다른 문장의 지시는 그대로 남는다(T-F-01·02).
    """
    state = CallState(scorer)
    for line in [
        "금융감독원입니다. 저희는 전화로 계좌번호를 묻지 않습니다.",
        "국고 계좌로 자금을 이관하십시오. 저희는 절대 개인정보를 묻지 않습니다.",
        "수사 기밀이니 발설하지 마십시오. 저희는 비밀 유지를 요구하지 않습니다.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "위험", r.why()


# ── 검증셋 4차 ────────────────────────────────────────────────────────────────

def test_link_alone_is_not_a_critical_signal(scorer):
    """'문자로 보내드린 링크'만으로 단독 고위험이 되면 안 된다.

    정상 기업이 일상적으로 쓰는 말이다. 게다가 검증셋 4차에서 걸린 문장은
    링크를 **누르지 말라는 경고**였다(Q-B-06).
    위험한 것은 링크가 아니라 링크로 앱을 설치하는 것이다.
    """
    warn = scorer.score("문자로 보내드린 링크는 저희가 보내지 않으니 누르지 마세요")
    assert "C6" not in warn.criticals, warn.why()
    assert warn.level == "안전", warn.why()

    scam = scorer.score("문자로 보내드린 링크 눌러 설치하시고 결제해 주세요")
    assert "C6" in scam.criticals, scam.why()


def test_honorific_particle_isolation_is_caught(scorer):
    """'부모님께는 알리지 마세요' — 완전히 자연스러운 한국어인데 놓쳤었다.

    기존 S4 표현은 전부 '말하지'만 있고 '알리지'가 없었으며,
    높임 조사 '께'도 빠져 있었다(Q-F-08).
    지금은 열거가 아니라 형태 규칙(M-S4-함구)이 잡는다.
    """
    r = scorer.score("부모님께는 알리지 마세요. 본인 명의 건이라 그렇습니다.")
    assert r.stage_hits["S4"] > 0, r.why()


# ── 형태 규칙 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "사모님께 언급하지 마세요",
    "아드님한테도 밝히지 마십시오",
    "과장님보고 이야기하지 말고",
    "며느리에게만 전달하지 마세요",
    "옆집 아저씨에게는 알려주지 마",
    "이모한테 얘기하지 않으셔야 합니다",
])
def test_silence_rule_generalizes_to_unlisted_forms(scorer, text):
    """DB에 한 번도 적지 않은 표면형도 잡아야 한다.

    형태 규칙의 주장은 "표현을 나열하지 않아도 잡는다"이고,
    그 주장은 **적지 않은 형태**로만 검증된다.
    호칭은 무한하므로 열린 자리로 두고 조사·어간·어미로만 잡는다.
    """
    assert scorer.extract(text).hits["S4"] > 0, text


@pytest.mark.parametrize("text", [
    "친구한테는 말했지만 다른 사람에겐 안 했어",
    "동생에게 말하지 그랬어",
    "어머님께 이 소식을 알려 드리세요",
    "과장님께 보고하지 않으면 문제가 됩니다",
    "고객에게 안내하지 못한 부분은 죄송합니다",
])
def test_silence_rule_requires_actual_prohibition(scorer, text):
    """부정 어미가 금지가 아니면 걸리면 안 된다.

    열거본("한테는 말하지")은 근사매칭에 기대는 탓에
    "친구한테는 말했지만"을 자모 거리 2로 잡았다.
    형태 규칙은 '-지 마/말/않'을 명시적으로 요구해 이걸 가른다.
    """
    assert scorer.extract(text).hits["S4"] == 0, text


@pytest.mark.parametrize("text,expected", [
    ("제가 지금 알려드리는 계좌로 입금하세요", True),
    ("불러드리는 통장으로 보내십시오", True),
    ("안내드리는 학원 계좌로 넣어주세요", True),
    ("계좌번호를 알려주시면 입금해 드리겠습니다", False),
    ("환급받으실 계좌를 알려주십시오", False),
])
def test_humble_direction_rule(scorer, text, expected):
    """겸양 '-드리-'는 화자가 준다는 표지, 존경 '-주시-'는 청자가 준다는 표지다.

    존경 쪽도 S5 자체는 걸릴 수 있다 — "계좌번호 알려"는 generic 키워드다.
    갈리는 곳은 **specific 여부**다. 겸양 쪽만 조합 신호(P4)를 성립시킨다.
    한국어 경어법이 방향을 구분해 주므로 표현을 늘려도 오탐이 늘지 않는다.
    """
    ev = scorer.extract(text)
    assert ev.specific["S5"] is expected, text


# ── 검증셋 5차 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "이 건은 아무한테도 알리시면 곤란합니다",
    "가족에게 알리는 것은 삼가 주십시오",
    "은행 직원에게 언급하시는 것도 금물입니다",
])
def test_euphemistic_prohibition_is_caught(scorer, text):
    """완곡한 금지도 금지다. '-지 마' 계열만 보면 빠져나간다(P-F-03·10)."""
    assert scorer.extract(text).hits["S4"] > 0, text


@pytest.mark.parametrize("text", [
    "제가 문자로 적어드리는 계좌로 옮겨 두십시오",
    "제가 문자로 보내드리는 구좌로 자금을 옮기세요",
    "제가 알려드리는 구좌로 돈을 옮겨 두시면 됩니다",
])
def test_designation_variants_are_caught(scorer, text):
    """'구좌'는 고령층 대상에서 실제로 쓰이는 옛말이다(P-F-05·10)."""
    ev = scorer.extract(text)
    assert ev.hits["S5"] > 0 and ev.specific["S5"], text


def test_lawyer_confidentiality_duty_is_not_a_threat(scorer):
    """"저희 쪽에서 누설하는 일은 없습니다" — 화자가 자기 의무를 말하는 형태다.

    상대에게 함구를 요구하는 것과 정반대인데 위험으로 판정했었다(P-B-07).
    """
    state = CallState(scorer)
    for line in [
        "변호사 비밀유지 의무가 있으니 저희 쪽에서 누설하는 일은 없습니다.",
        "착수금은 제가 안내드리는 사무소 계좌로 입금해 주시면 됩니다.",
        "위임계약서 검토하시고 천천히 결정하세요.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "안전", r.why()


def test_mentioning_a_public_office_is_not_impersonating_one(scorer):
    """"가까운 주민센터 방문하셔도 됩니다"는 정상 안내다.

    지자체 사칭을 잡으려고 기관명을 S1에 넣었다가 이 문장이 걸렸다(H-B-04).
    기관을 **언급**하는 것과 **사칭**하는 것은 다르다.
    """
    state = CallState(scorer)
    for line in [
        "안녕하세요, 광주광역시 세무과입니다.",
        "환급받으실 계좌번호를 알려주시면 그쪽으로 입금해 드리겠습니다.",
        "본인 확인이 필요하니 가까운 주민센터 방문하셔도 됩니다.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "안전", r.why()


# ── 검증셋 6차 ────────────────────────────────────────────────────────────────

def test_conjunction_does_not_trigger_quote_suppression(scorer):
    """접속부사 "그래서"가 인용 억제를 켜면 안 된다.

    구어체 인용 어미를 넓히면서 '래서'를 넣었더니 "그래서"에 그대로 들어 있어,
    지시 뒤에 접속부사만 붙이면 억제가 켜졌다(X-F-01·02).

    억제 목록에 흔한 말이 섞이면 그 자체가 회피로다 — 세 번째 겪은 일이다.
    """
    state = CallState(scorer)
    for line in [
        "서울중앙지검 수사관입니다.",
        "안전계좌로 이체하셔야 합니다 그래서 지금 바로 진행하겠습니다.",
        "가족에게도 말하지 마십시오 그래서 조용한 곳으로 가 주세요.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "위험", r.why()


def test_long_call_does_not_dilute_evidence(scorer):
    """정상 대화 열다섯 마디 뒤에 본론이 와도 잡아야 한다(X-F-04).

    증거는 누적되므로 앞에 무엇이 오든 뒤의 신호가 지워지지 않는다.
    반대로 긴 통화 자체가 위험 신호가 되어서도 안 된다 — X-B-12가 그 대조군이다.
    """
    state = CallState(scorer)
    for _ in range(15):
        state.add_utterance("날씨가 많이 더워졌습니다. 건강은 어떠신지요.")
    for line in [
        "수사 기밀이라 주변에 알리지 마시고 조용한 곳으로 이동해 주세요.",
        "자산 검수를 위해 제가 안내드리는 계좌로 잔액을 옮겨 두십시오.",
    ]:
        r = state.add_utterance(line)
    assert r.level == "위험", r.why()


# ── 검증셋 7차 ────────────────────────────────────────────────────────────────

def test_guarded_quotative_separates_victim_narration(scorer):
    """"옮기래서"는 인용이고 "그래서"는 접속부사다. 앞 글자로 가른다.

    6차에서 '래서'를 통째로 뺐더니 7차에서 피해자 진술이 위험이 됐다(Y-B-06).
    넣으면 회피로가 되고 빼면 피해자를 잡는다 — 가드가 답이었다.
    """
    victim = CallState(scorer)
    for line in ["안전계좌로 옮기래서 은행까지 갔었어요.",
                 "가족한테 말하지 말래서 혼자 갔는데 창구 직원이 말려줬어요."]:
        r = victim.add_utterance(line)
    assert r.level == "안전", r.why()

    scam = CallState(scorer)
    for line in ["서울중앙지검 수사관입니다.",
                 "안전계좌로 이체하셔야 합니다 그래서 지금 바로 진행하겠습니다."]:
        r = scam.add_utterance(line)
    assert r.level == "위험", r.why()


def test_pair_needs_at_least_one_specific_stage(scorer):
    """정상 상거래 용어 두 개가 만나 하한 0.75를 강제하면 안 된다(Y-B-05).

    이 두 마디만으로는 여전히 "주의"가 나온다 — 대출 경로(C)는 분모가 작아
    S5·S8 두 단계만 걸려도 0.61이 된다. 여기서 막는 것은 **조합 신호가
    하한을 강제하는 것**이고, 그것만으로 실사용 오탐이 사라졌다.
    """
    state = CallState(scorer)
    for line in ["장기 렌트라 선납금 삼십만 원 받고 있습니다.",
                 "보증금 먼저 결제하시면 인수하실 때 정산해 드립니다."]:
        r = state.add_utterance(line)
    assert not r.pairs, r.why()
    assert r.level != "위험", r.why()


def test_blocking_third_party_contact_is_isolation(scorer):
    """"기존 보험사에는 문의하지 마세요" — 확인을 막는 것도 고립이다(Y-F-06)."""
    assert scorer.extract("기존 보험사에는 문의하지 마세요").hits["S4"] > 0


def test_short_agency_name_does_not_ghost_match(scorer):
    """공백을 지우면 "이상한 전화를"이 "한전"이 된다(Y-B-06)."""
    assert scorer.extract("어제 이상한 전화를 받았는데요").hits["S1"] == 0


# ── 검증셋 8차 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "안전계좌로 이체하십시오. 규정이 그렇다니까요.",
    "안전계좌 그렇다는 말씀입니다. 그쪽으로 옮기십시오.",
    "안전계좌로 옮기세요. 그러라고 있는 규정입니다.",
])
def test_conjunctions_do_not_hide_instructions(scorer, line):
    """접속·강조 표현이 인용 어미를 품고 있어도 억제되면 안 된다.

    '그렇다는'에 '다는'이, '그러라고'에 '라고'가 들어 있다(Z-F-02).
    어미마다 앞 글자 가드를 다는 방식은 한계에 닿았다 — '라고'·'다고'는
    가장 흔한 인용 어미라 가드를 잘못 달면 진짜 인용을 통째로 잃는다.
    그래서 어미가 아니라 **표현을 지운다.**
    """
    state = CallState(scorer)
    state.add_utterance("서울중앙지검 수사관입니다.")
    r = state.add_utterance(line)
    assert "C1" in r.criticals, r.why()


def test_real_quotation_still_suppressed_after_conjunction_stripping(scorer):
    """접속부사를 지워도 진짜 인용은 그대로 걸러져야 한다."""
    r = scorer.score("안전계좌로 이체하라고 했어요. 그래서 은행에 갔었죠.")
    assert "C1" not in r.criticals, r.why()


def test_bankbook_photo_request_is_critical(scorer):
    """정상 기관은 계좌번호를 묻지 통장 사진을 요구하지 않는다(Z-F-10)."""
    r = scorer.score("통장이랑 신분증 찍어서 보내주시면 등록해 드릴게요")
    assert "C7" in r.criticals, r.why()


# ── 검증셋 9차 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "안전계좌 그렇다는 말씀입니다. 그쪽으로 이체하십시오.",
    "안전계좌 이렇다는 말씀입니다. 그쪽으로 이체하십시오.",
    "안전계좌 저렇다는 겁니다. 잔액을 옮기십시오.",
    "안전계좌로 옮기세요. 어쩌라고 있는 규정입니다.",
])
def test_deictic_conjunctions_are_matched_by_construction(scorer, line):
    """접속 표현을 열거하지 않고 구성으로 잡는다.

    8차에서 9개짜리 열거로 막았더니 9차에서 지시대명사만 바꾼
    "저렇다는"이 그대로 통과했다(W-F-02). 1~4차에서 배운 것과 같다 —
    **열거는 끝이 없다.** 지시 어간 × 인용 어미로 곱한다.
    """
    state = CallState(scorer)
    state.add_utterance("서울중앙지검 수사관입니다.")
    r = state.add_utterance(line)
    assert "C1" in r.criticals, r.why()


@pytest.mark.parametrize("text,flagged", [
    ("통장이랑 도장, 신분증 가지고 오시면 됩니다", False),
    ("기초연금 신청하시려면 통장이랑 신분증 가지고 방문하시면 됩니다", False),
    ("어머니 통장이랑 도장 어디 두셨는지 알아?", False),
    ("통장이랑 신분증 찍어서 보내주시면 등록해 드릴게요", True),
    ("통장 사본 보내주세요", True),
])
def test_bankbook_discriminator_is_the_act_not_the_noun(scorer, text, flagged):
    """지참은 정상이고 전송이 위험하다.

    8차에서 '통장이랑 도장'을 명사쌍만으로 단독 고위험에 넣었더니
    진짜 은행 창구·주민센터 안내와 형제간 대화가 걸렸다(W-B-01·02·10).
    은행은 "가지고 오세요"라고 하고 사기범은 "찍어서 보내세요"라고 한다.
    """
    ev = scorer.extract(text)
    assert (bool(ev.criticals) or ev.specific.get("S5", False)) is flagged, text


# ── 검증셋 10차 ───────────────────────────────────────────────────────────────

def test_adnominal_past_fusion(scorer):
    """드리 + ㄴ = 드린. 음절 규칙에서 자모는 결합하지 못한다(V-S1)."""
    assert scorer.extract("제가 말씀드린 데로 옮기시면 됩니다").specific["S5"]
    assert scorer.extract("제가 말씀드리는 데로 옮기시면 됩니다").specific["S5"]


def test_blocking_official_payment_channel(scorer):
    """정상 기관은 자기가 발급한 납부 경로를 막지 않는다.

    최소대립쌍 3은 이 한 문장만 다르다.
    """
    scam = scorer.score("고지서 계좌는 마감돼서 제가 불러드리는 계좌로만 됩니다")
    assert "C8" in scam.criticals, scam.why()
    ok = scorer.score("고지서에 있는 가상계좌로 납부하시면 됩니다")
    assert not ok.criticals, ok.why()


def test_news_context_is_descriptive(scorer):
    """"안전계좌 그거 뉴스에서 봤어"는 잡담이다(V-B9)."""
    r = scorer.score("야 요즘 안전계좌 그거 뉴스에서 봤어")
    assert "C1" not in r.criticals, r.why()


# ── 행위 프레임 (10차 이후) ───────────────────────────────────────────────────

def test_frame_beats_noun_for_remote_access(scorer):
    """명사 '원격제어'가 아니라 **설치를 시키는가**가 판별자다.

    10차에서 정반대 결과가 나왔다 — 사기범은 '원격으로'라고만 말해 빠져나가고
    사내 IT 지원이 정확한 용어를 써서 걸렸다(V-S6 대 V-B10).
    """
    scam = scorer.score("지금 화면에 뜨는 거 설치하시면 제가 원격으로 봐드릴게요")
    assert "C5" in scam.criticals, scam.why()
    ok = scorer.score("제가 원격제어로 봐드릴게요 사내 프로그램이요")
    assert "C5" not in ok.criticals, ok.why()


def test_cash_handover_frame(scorer):
    """인출과 대면 전달이 이어지는 것이 대면편취형의 정의다(V-S2)."""
    r = scorer.score("일단 인출하신 다음에 저희 직원한테 건네주시면 보관증 드려요")
    assert "C9" in r.criticals, r.why()
    ok = scorer.score("지금 은행 가셔서 전액 인출하셔야 됩니다")
    assert "C9" not in ok.criticals, ok.why()


def test_install_warning_is_not_install_instruction(scorer):
    """"링크는 누르지 마세요"는 경고다. 프레임은 부정형을 배제한다."""
    r = scorer.score("문자로 보내드린 링크는 저희가 보내지 않으니 누르지 마세요")
    assert not r.criticals, r.why()


def test_minimal_pair_account_provenance(scorer):
    """계좌 출처가 화자 지정이냐 기존이냐로 갈린다(최소대립쌍 1)."""
    scam = CallState(scorer)
    for l in ["네 여보세요 어머니 저 준영이요",
              "제가 지금 알려드리는 계좌로 삼백만 원만 보내주실 수 있어요"]:
        r = scam.add_utterance(l)
    assert r.level == "위험", r.why()

    ok = CallState(scorer)
    for l in ["네 여보세요 어머니 저 준영이요",
              "저번에 쓰던 제 계좌로 삼백만 원만 보내주실 수 있어요"]:
        r = ok.add_utterance(l)
    assert r.level == "안전", r.why()
