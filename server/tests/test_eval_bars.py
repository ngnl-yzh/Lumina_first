"""평가 기준선 — 성능이 조용히 나빠지는 것을 막는다.

`tests/test_mode1.py`가 개별 동작을 보는 단위 테스트라면, 이 파일은 **전체 성능이
어떤 선 아래로 떨어지지 않는지**만 본다. 정확한 수치를 박아두지 않는 이유는
시나리오가 늘어나면 수치가 움직이기 때문이다. 움직여도 되는 것과 절대 나빠지면
안 되는 것을 구분한다.

특히 오탐률은 **0을 요구한다.** 실사용을 막는 것은 미탐이 아니라 오탐이다.
경고가 자꾸 잘못 뜨면 사용자는 앱을 끈다. 그러면 미탐률은 100%가 된다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_mode1 import Report, run_scenario                    # noqa: E402
from mirinae.mode1 import load_db                              # noqa: E402
from mirinae.mode1.scorer import Scorer                        # noqa: E402

SCENARIO_FILES = ["scenarios.json", "scenarios_evasion.json", "scenarios_real.json"]


@pytest.fixture(scope="module")
def report() -> Report:
    scenarios: list[dict] = []
    for name in SCENARIO_FILES:
        spec = json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))
        scenarios.extend(spec["scenarios"])
    scorer = Scorer(load_db())
    return Report([run_scenario(scorer, s) for s in scenarios])


def test_scenario_set_is_big_enough(report):
    """작은 세트로 낸 0%는 0%가 아니다.

    정상 시나리오가 사기보다 적으면 오탐률을 신뢰할 수 없다.
    """
    assert len(report.benign) >= 30, f"정상 시나리오 {len(report.benign)}건 — 너무 적다"
    assert len(report.benign) >= len(report.fraud), "정상이 사기보다 적다"


def test_family_finance_talk_is_never_flagged(report):
    """가족끼리 하는 돈 이야기를 잡으면 앱을 못 쓴다.

    "계좌로 보내줘"·"오늘까지 입금해야 해"·"급하게 필요해"는 사기범만 하는 말이 아니다.
    등록금·병원비·전세 계약금·경조사비 — 전부 정상 통화에서 같은 어휘가 나온다.
    여기서 경고가 뜨면 사용자는 앱을 끄고, 그 순간 미탐률은 100%가 된다.
    """
    flagged = [(r.id, r.title, r.final_score)
               for r in report.benign if r.tag == "family" and r.warned]
    assert not flagged, f"가족 금융 대화가 경고를 받았다: {flagged}"


def test_real_transcripts_are_detected(report):
    """공개 녹취 기반 시나리오 — 자체 작성분보다 이쪽이 실제 성능에 가깝다.

    R-A-02는 예외로 둔다. SBS가 공개한 **발췌**라 사기범이 소속을 밝히는 도입부가 빠져 있고,
    진입 단계(S1/S7/S8)가 없으면 경로 판정이 폴백으로 떨어져 크게 불리해진다.
    시스템 결함이라기보다 자료의 한계이지만, "통화 중간부터 듣기 시작하면 약하다"는
    운영상 위험 자체는 실재한다.
    """
    real = [r for r in report.fraud if r.tag == "real"]
    assert real, "공개 녹취 시나리오가 없다"
    missed = sorted(r.id for r in real if not r.detected)
    assert missed in ([], ["R-A-02"]), f"공개 녹취를 놓쳤다: {missed}"


def test_no_false_positive_on_clean_transcript(report):
    """정상 통화를 위험으로 보면 안 된다 — **주의 등급도 안 된다.**

    깨끗한 전사 기준이다. STT 오차가 끼면 오른다는 것은
    `--noise-curve`로 따로 측정하고 README에 적어 두었다.
    """
    bad = [(r.id, r.title, r.final_score) for r in report.benign if r.warned]
    assert not bad, f"정상 통화가 경고를 받았다: {bad}"


def test_detection_rate_holds(report):
    assert report.recall >= 0.75, f"탐지율 {report.recall:.1%}"


def test_precision_holds(report):
    assert report.precision >= 0.95, f"정밀도 {report.precision:.1%}"


def test_evasion_mostly_fails(report):
    """탐지 규칙을 아는 사기범에게 얼마나 버티는가.

    F-E-04(키워드를 우회 표현으로 교체)는 사전 기반 탐지의 원리적 한계라
    통과를 허용한다. 그 외에는 전부 잡아야 한다.
    """
    missed = [r.id for r in report.evasion if not r.detected]
    assert missed == ["F-E-04"] or not missed, f"새로운 회피가 뚫렸다: {missed}"


def test_benign_padding_cannot_hide_fraud(report):
    """F-E-02 회귀 — 예방 문구 도배로 하한을 무너뜨리던 구멍.

    한때 C1·C2·P1이 모두 발동한 통화가 0.000 "안전"으로 나왔다.
    """
    r = next(x for x in report.evasion if x.id == "F-E-02")
    assert r.detected, f"{r.final_score:.3f} — 예방 문구 도배에 다시 뚫렸다"


def test_route_classification_holds(report):
    assert report.route_acc >= 0.85, f"경로 판정 {report.route_acc:.1%}"
