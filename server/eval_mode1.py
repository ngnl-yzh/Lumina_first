"""모드 1 탐지 성능 평가 — 탐지율과 오탐률을 함께 잰다.

`tests/test_mode1.py`와 목적이 다르다. 저쪽은 **알고리즘 구조**가 설계대로인지 보는
단위 테스트이고, 이쪽은 **탐지 성능**을 수치로 내는 평가다. 둘을 섞으면
"테스트 47건 통과"가 곧 "잘 잡는다"로 오독된다.

재현율만 재면 안 된다. 무조건 "위험"이라고 답해도 재현율은 100%가 된다.
그래서 정상 통화 시나리오를 같은 수만큼 넣고 **오탐률을 함께** 낸다.
정상 시나리오가 이 평가의 절반이다.

사용:
    python eval_mode1.py                          # 평가
    python eval_mode1.py -v                       # 실패 케이스 근거까지
    python eval_mode1.py --save out/eval_a.json   # 스냅샷 저장
    python eval_mode1.py --compare out/eval_a.json  # 개선 전후 비교
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

import stt_noise                                                    # noqa: E402
from mirinae.mode1 import load_db                                   # noqa: E402
from mirinae.mode1.matcher import Matcher                           # noqa: E402
from mirinae.mode1.scorer import (                                  # noqa: E402
    THRESHOLD_ALERT, THRESHOLD_WARN, CallState, Scorer,
)

EVAL_DIR = Path(__file__).parent / "eval"
DEFAULT_SCENARIOS = [
    EVAL_DIR / "scenarios.json",           # 자체 작성 (사기 10 · 정상 24)
    EVAL_DIR / "scenarios_evasion.json",   # 회피 6 — 판정 규칙을 아는 사기범
    EVAL_DIR / "scenarios_real.json",      # 공개 녹취 5 · 가족 금융 대화 12
]


# ── 한 시나리오 실행 결과 ──────────────────────────────────────────────────────

@dataclass
class Run:
    id: str
    label: str                  # fraud | benign
    tag: str                    # "" | evasion — 회피 시나리오는 따로 집계한다
    title: str
    expect: str                 # 도달해야 할 최소 등급
    route_expected: str | None

    final_score: float
    final_level: str
    route_actual: str

    first_warn_idx: int | None  # 처음 '주의'에 도달한 발화 번호 (1-base)
    first_alert_idx: int | None
    alert_by: int | None        # 요구되는 조기성
    n_utterances: int

    trace: list[float] = field(default_factory=list)
    criticals: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    benign_hits: list[str] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)

    # ── 판정 ──────────────────────────────────────────────────────────────────

    @property
    def detected(self) -> bool:
        """위험 등급에 도달했는가."""
        return self.first_alert_idx is not None

    @property
    def warned(self) -> bool:
        return self.first_warn_idx is not None

    @property
    def correct(self) -> bool:
        if self.label == "fraud":
            return self.detected
        return not self.warned          # 정상은 '주의'조차 뜨면 안 된다

    @property
    def early_enough(self) -> bool:
        """자금 이동 발화보다 먼저 잡았는가."""
        if self.label != "fraud" or self.alert_by is None:
            return True
        return self.first_alert_idx is not None and self.first_alert_idx <= self.alert_by

    @property
    def route_ok(self) -> bool:
        if not self.route_expected:
            return True
        return self.route_actual == self.route_expected

    def verdict(self) -> str:
        if self.label == "fraud":
            if not self.detected:
                return "미탐"
            if not self.early_enough:
                return "지연탐지"
            return "정탐"
        return "정상" if not self.warned else ("오탐-위험" if self.detected else "오탐-주의")


# ── 평가 ──────────────────────────────────────────────────────────────────────

def run_scenario(sc: Scorer, spec: dict) -> Run:
    state = CallState(sc)
    dv = float(spec.get("deepvoice_score", 0.0))

    first_warn = first_alert = None
    trace: list[float] = []

    for i, utt in enumerate(spec["utterances"], start=1):
        r = state.add_utterance(utt["text"], deepvoice_score=dv)
        trace.append(round(r.score, 4))
        if first_warn is None and r.score >= THRESHOLD_WARN:
            first_warn = i
        if first_alert is None and r.score >= THRESHOLD_ALERT:
            first_alert = i

    last = state.last
    return Run(
        id=spec["id"],
        label=spec["label"],
        tag=spec.get("tag", ""),
        title=spec.get("title", ""),
        expect=spec.get("expect", "위험" if spec["label"] == "fraud" else "안전"),
        route_expected=spec.get("route_expected"),
        final_score=round(last.score, 4),
        final_level=last.level,
        route_actual=last.route.id,
        first_warn_idx=first_warn,
        first_alert_idx=first_alert,
        alert_by=spec.get("alert_by"),
        n_utterances=len(spec["utterances"]),
        trace=trace,
        criticals=list(last.criticals),
        pairs=list(last.pairs),
        benign_hits=list(last.benign_hits),
        matched={k: list(v) for k, v in last.matched.items()},
    )


@dataclass
class Report:
    runs: list[Run]

    @property
    def fraud(self) -> list[Run]:
        return [r for r in self.runs if r.label == "fraud"]

    @property
    def benign(self) -> list[Run]:
        return [r for r in self.runs if r.label == "benign"]

    @property
    def plain_fraud(self) -> list[Run]:
        """회피를 시도하지 않는 일반 사기."""
        return [r for r in self.fraud if r.tag != "evasion"]

    @property
    def evasion(self) -> list[Run]:
        return [r for r in self.runs if r.tag == "evasion"]

    @property
    def evasion_rate(self) -> float:
        """회피 성공률 — 낮을수록 좋다. 탐지 규칙을 아는 상대에게 얼마나 버티는가."""
        ev = self.evasion
        return sum(1 for r in ev if not r.detected) / len(ev) if ev else 0.0

    # 혼동행렬 — '위험' 등급 기준
    @property
    def tp(self) -> int:
        return sum(1 for r in self.fraud if r.detected)

    @property
    def fn(self) -> int:
        return len(self.fraud) - self.tp

    @property
    def fp(self) -> int:
        return sum(1 for r in self.benign if r.detected)

    @property
    def tn(self) -> int:
        return len(self.benign) - self.fp

    @property
    def recall(self) -> float:
        return self.tp / len(self.fraud) if self.fraud else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr_alert(self) -> float:
        return self.fp / len(self.benign) if self.benign else 0.0

    @property
    def fpr_warn(self) -> float:
        """'주의'까지 포함한 오탐률. 사용자 체감은 이쪽에 가깝다."""
        n = sum(1 for r in self.benign if r.warned)
        return n / len(self.benign) if self.benign else 0.0

    @property
    def early_rate(self) -> float:
        got = [r for r in self.fraud if r.detected]
        return sum(1 for r in got if r.early_enough) / len(got) if got else 0.0

    @property
    def route_acc(self) -> float:
        wanted = [r for r in self.fraud if r.route_expected]
        return sum(1 for r in wanted if r.route_ok) / len(wanted) if wanted else 0.0

    def summary(self) -> dict:
        return {
            "n_fraud": len(self.fraud),
            "n_benign": len(self.benign),
            "n_evasion": len(self.evasion),
            "evasion_rate": round(self.evasion_rate, 4),
            "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "fpr_alert": round(self.fpr_alert, 4),
            "fpr_warn": round(self.fpr_warn, 4),
            "early_rate": round(self.early_rate, 4),
            "route_acc": round(self.route_acc, 4),
        }


# ── 출력 ──────────────────────────────────────────────────────────────────────

MARK = {"정탐": "O", "미탐": "X", "지연탐지": "~", "정상": "O",
        "오탐-위험": "X", "오탐-주의": "~"}


def print_table(runs: list[Run], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'':2} {'ID':<8} {'점수':>6} {'등급':<4} {'경로':<3} {'탐지':>6}  제목")
    print("  " + "-" * 76)
    for r in runs:
        v = r.verdict()
        at = f"{r.first_alert_idx}/{r.n_utterances}" if r.first_alert_idx else "—"
        print(f"  {MARK[v]:2} {r.id:<8} {r.final_score:>6.3f} {r.final_level:<4} "
              f"{r.route_actual:<3} {at:>6}  {r.title}")


def print_failures(runs: list[Run]) -> None:
    bad = [r for r in runs if not r.correct or not r.early_enough]
    if not bad:
        print("\n실패 케이스 없음")
        return
    print(f"\n{'='*78}\n실패 케이스 근거 ({len(bad)}건)\n{'='*78}")
    for r in bad:
        print(f"\n[{r.id}] {r.title}  →  {r.verdict()}")
        print(f"  최종 {r.final_score:.3f} ({r.final_level}) · 경로 {r.route_actual}"
              + (f" (기대 {r.route_expected})" if not r.route_ok else ""))
        print(f"  점수 추이: {' → '.join(f'{s:.2f}' for s in r.trace)}")
        if r.criticals:
            print(f"  critical 발동: {', '.join(r.criticals)}   ← 하한 {THRESHOLD_ALERT} 강제")
        if r.pairs:
            print(f"  pair 발동: {', '.join(r.pairs)}")
        if r.benign_hits:
            print(f"  benign 감점: {', '.join(r.benign_hits)}")
        if r.matched:
            hits = "  ".join(f"{k}:{','.join(v[:2])}" for k, v in sorted(r.matched.items()))
            print(f"  단계 매칭: {hits}")


# ── STT 오차 견고성 ───────────────────────────────────────────────────────────

def averaged_at_noise(db, scenarios: list[dict], rate: float, repeats: int,
                      approx: bool = True, seed0: int = 0) -> dict:
    """오차율 `rate`에서 여러 번 돌려 평균 지표를 낸다.

    한 번만 재면 어떤 음절이 망가졌느냐에 따라 결과가 크게 흔들린다.
    난수 시드를 바꿔가며 평균을 내야 오차율의 효과를 볼 수 있다.
    """
    sums: dict[str, float] = defaultdict(float)
    for k in range(repeats):
        rng = random.Random(seed0 + k)
        scorer = Scorer(db, Matcher(approx=approx))
        runs = [
            run_scenario(scorer,
                         stt_noise.perturb_scenario(s, rate, rng) if rate else s)
            for s in scenarios
        ]
        for key, v in Report(runs).summary().items():
            sums[key] += v
    return {k: v / repeats for k, v in sums.items()}


def print_noise_curve(db, scenarios: list[dict], repeats: int) -> None:
    """전사 품질이 떨어질 때 탐지가 얼마나 버티는가.

    근사매칭을 켠 것과 끈 것을 나란히 놓는다. 근사매칭이 실제로 무엇을 사주는지,
    그리고 오탐을 줄이려고 조인 규칙이 견고성을 얼마나 깎았는지를 함께 본다.
    """
    rates = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
    print(f"\n{'='*78}")
    print(f"STT 오차 견고성 — 오차율별 지표 ({repeats}회 평균)")
    print(f"{'='*78}")
    print("  오차율은 **음절당** 확률이다. 0.10이면 8음절 표현이 하나 이상 틀릴 확률 57%.\n")
    print(f"  {'오차율':>6} │ {'근사매칭 켬':^26} │ {'근사매칭 끔 (정확 일치만)':^26}")
    print(f"  {'':>6} │ {'탐지율':>8}{'오탐률':>9}{'F1':>9} │ {'탐지율':>8}{'오탐률':>9}{'F1':>9}")
    print("  " + "-" * 74)

    rows = []
    for rate in rates:
        on = averaged_at_noise(db, scenarios, rate, repeats, approx=True)
        off = averaged_at_noise(db, scenarios, rate, repeats, approx=False)
        rows.append((rate, on, off))
        print(f"  {rate*100:>5.0f}% │ {on['recall']*100:>7.1f}%{on['fpr_alert']*100:>8.1f}%"
              f"{on['f1']*100:>8.1f}% │ {off['recall']*100:>7.1f}%"
              f"{off['fpr_alert']*100:>8.1f}%{off['f1']*100:>8.1f}%")

    base_on = rows[0][1]["recall"]
    worst = rows[-1]
    print()
    print(f"  근사매칭이 사주는 것 — 오차율 {worst[0]*100:.0f}%에서 탐지율 "
          f"{worst[1]['recall']*100:.1f}% 대 {worst[2]['recall']*100:.1f}% "
          f"(차이 {(worst[1]['recall']-worst[2]['recall'])*100:+.1f}%p)")
    print(f"  견고성 — 오차율 0%→{worst[0]*100:.0f}%에서 탐지율 "
          f"{base_on*100:.1f}% → {worst[1]['recall']*100:.1f}% "
          f"({(worst[1]['recall']-base_on)*100:+.1f}%p)")


def print_gaps(rep: Report, spec: dict) -> None:
    """미탐의 원인이 알고리즘인지 DB 공백인지 가른다.

    둘은 대응이 완전히 다르다. 알고리즘 결함은 코드로 고치지만,
    DB 공백은 **패턴 DB 작성자가** 금감원 공개 자료로 채워야 한다(D08 §07).
    평가 시나리오를 쓴 사람이 그걸 보고 키워드를 넣으면 그 순간 평가가 순환한다 —
    자기가 낸 문제를 자기가 채점하는 셈이 된다.

    그래서 여기서는 **어느 단계가 비었는지만** 알리고 표현을 제안하지 않는다.
    """
    by_id = {s["id"]: s for s in spec["scenarios"]}
    missed = [r for r in rep.fraud if not r.detected or not r.early_enough]
    if not missed:
        print("\n미탐·지연 없음 — DB 공백 리포트 생략")
        return

    print(f"\n{'='*78}\nDB 커버리지 공백 — 미탐·지연 {len(missed)}건의 원인\n{'='*78}")
    print("  (표현은 제안하지 않는다. 채우는 사람은 평가 시나리오를 보지 않아야 한다)\n")
    for r in missed:
        route = r.route_expected or r.route_actual
        from mirinae.mode1.router import FALLBACK, ROUTES
        stages = ROUTES[route].stages if route in ROUTES else FALLBACK.stages
        empty = [s for s in stages if s not in r.matched]
        print(f"  [{r.id}] {r.title}  → {r.verdict()} ({r.final_score:.3f})")
        print(f"     기대 경로 {route} · 잡힌 단계 {sorted(r.matched) or '없음'}")
        if empty:
            print(f"     **비어 있는 단계: {', '.join(empty)}**")
        if r.route_expected and not r.route_ok:
            print(f"     경로 판정 실패 — 진입 단계(S1/S7/S8)가 하나도 안 잡혔다")
        for i, u in enumerate(by_id[r.id]["utterances"], start=1):
            if i > r.n_utterances:
                break
            print(f"        {i}. {u['text']}")
        print()


def _ci(successes: int, n: int) -> str:
    """비율의 95% 신뢰구간 (Wilson).

    작은 세트로 낸 0%를 "오탐 없음"으로 읽으면 안 된다.
    24건에서 0건이면 실제 오탐률은 [0.0%, 13.8%] 어디든 될 수 있다.
    구간을 함께 찍으면 caveat 문장을 읽지 않아도 그 사실이 보인다.

    Wald(정규근사)는 0%·100%에서 폭이 0이 되어 못 쓴다. 그래서 Wilson을 쓴다.
    `mirinae.metrics.wilson_ci`를 그대로 재사용한다 — 두 군데서 다르게 계산하면
    보고서마다 숫자가 달라진다. torch를 끌어오므로 여기서만 늦게 import한다.
    """
    if n == 0:
        return ""
    from mirinae.metrics import wilson_ci
    lo, hi = wilson_ci(successes, n)
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


def print_summary(rep: Report) -> None:
    s = rep.summary()
    print(f"\n{'='*78}\n종합\n{'='*78}")
    print(f"  사기 {s['n_fraud']}건 · 정상 {s['n_benign']}건        (혼동행렬은 '위험' 등급 기준)")
    print()
    print(f"                 실제 사기   실제 정상")
    print(f"    위험 판정        {s['tp']:>3}         {s['fp']:>3}")
    print(f"    위험 아님        {s['fn']:>3}         {s['tn']:>3}")
    print()
    n_warn = sum(1 for r in rep.benign if r.warned)
    print(f"                        {'값':>6}   {'95% 신뢰구간':^14}")
    print(f"  탐지율 (recall)      {s['recall']*100:6.1f}%  {_ci(rep.tp, len(rep.fraud)):^14}"
          f"  사기를 실제로 잡은 비율")
    print(f"  정밀도 (precision)   {s['precision']*100:6.1f}%  "
          f"{_ci(rep.tp, rep.tp + rep.fp):^14}  위험 판정 중 진짜 사기 비율")
    print(f"  F1                   {s['f1']*100:6.1f}%")
    print(f"  오탐률 (위험)        {s['fpr_alert']*100:6.1f}%  "
          f"{_ci(rep.fp, len(rep.benign)):^14}  정상 통화를 위험으로 본 비율")
    print(f"  오탐률 (주의 포함)   {s['fpr_warn']*100:6.1f}%  "
          f"{_ci(n_warn, len(rep.benign)):^14}  사용자 체감에 가까운 값")
    print(f"  조기 탐지            {s['early_rate']*100:6.1f}%     자금 이동 발화 전에 잡은 비율")
    print(f"  경로 판정 정확도     {s['route_acc']*100:6.1f}%")
    if s["n_evasion"]:
        print(f"  **회피 성공률**      {s['evasion_rate']*100:6.1f}%     "
              f"탐지 규칙을 아는 사기범이 빠져나간 비율 ({s['n_evasion']}건 중)")


def print_compare(cur: Report, base: dict) -> None:
    print(f"\n{'='*78}\n이전 스냅샷 대비\n{'='*78}")
    now = cur.summary()
    rows = [
        ("탐지율", "recall", +1), ("정밀도", "precision", +1), ("F1", "f1", +1),
        ("오탐률(위험)", "fpr_alert", -1), ("오탐률(주의)", "fpr_warn", -1),
        ("조기 탐지", "early_rate", +1), ("경로 정확도", "route_acc", +1),
    ]
    print(f"  {'지표':<14} {'이전':>8} {'현재':>8} {'변화':>9}")
    print("  " + "-" * 45)
    for name, key, good in rows:
        b, n = base["summary"].get(key, 0.0), now[key]
        d = n - b
        arrow = "  ·" if abs(d) < 1e-9 else ("  ↑" if d * good > 0 else "  ↓")
        print(f"  {name:<14} {b*100:>7.1f}% {n*100:>7.1f}% {d*100:>+7.1f}%p{arrow}")

    # 개별 시나리오 판정이 뒤집힌 것
    prev = {r["id"]: r for r in base["runs"]}
    flips = []
    for r in cur.runs:
        p = prev.get(r.id)
        if p and p["verdict"] != r.verdict():
            flips.append((r.id, p["verdict"], r.verdict(), r.title))
    if flips:
        print(f"\n  판정이 바뀐 시나리오 {len(flips)}건")
        for sid, before, after, title in flips:
            print(f"    {sid:<8} {before:<10} → {after:<10}  {title}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="모드 1 탐지 성능 평가")
    ap.add_argument("--scenarios", type=Path, nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--db", type=Path, default=None, help="pattern_db.json 경로")
    ap.add_argument("-v", "--verbose", action="store_true", help="실패 케이스 근거 출력")
    ap.add_argument("--gaps", action="store_true",
                    help="미탐 원인을 DB 공백 관점에서 리포트")
    ap.add_argument("--stt-noise", type=float, default=0.0, metavar="RATE",
                    help="음절당 STT 오차 확률을 주입해 평가 (예: 0.10)")
    ap.add_argument("--noise-curve", action="store_true",
                    help="오차율 0~25% 곡선 + 근사매칭 on/off 비교")
    ap.add_argument("--repeat", type=int, default=20,
                    help="오차 주입 시 반복 횟수 (기본 20)")
    ap.add_argument("--save", type=Path, help="결과를 JSON으로 저장")
    ap.add_argument("--compare", type=Path, help="저장된 스냅샷과 비교")
    args = ap.parse_args()

    spec = {"meta": {"version": "+".join(str(p.stem) for p in args.scenarios)},
            "scenarios": []}
    for path in args.scenarios:
        part = json.loads(path.read_text(encoding="utf-8"))
        spec["scenarios"].extend(part["scenarios"])

    db = load_db(args.db)

    if args.noise_curve:
        print_noise_curve(db, spec["scenarios"], args.repeat)
        return 0

    if args.stt_noise > 0:
        avg = averaged_at_noise(db, spec["scenarios"], args.stt_noise, args.repeat)
        print(f"STT 오차 {args.stt_noise*100:.0f}% · {args.repeat}회 평균\n")
        for k in ("recall", "precision", "f1", "fpr_alert", "early_rate",
                  "evasion_rate"):
            print(f"  {k:<12} {avg[k]*100:6.1f}%")
        return 0

    scorer = Scorer(db)
    runs = [run_scenario(scorer, s) for s in spec["scenarios"]]
    rep = Report(runs)

    print("모드 1 탐지 성능 평가 — " + ", ".join(p.name for p in args.scenarios))
    print_table(rep.plain_fraud, "사기 시나리오 — 잡아야 한다")
    print_table(rep.benign, "정상 시나리오 — 잡으면 안 된다")
    if rep.evasion:
        print_table(rep.evasion,
                    "회피 시나리오 — 탐지 규칙을 아는 사기범. 전부 잡아야 한다")
    print_summary(rep)

    if args.verbose:
        print_failures(runs)

    if args.gaps:
        print_gaps(rep, spec)

    if args.compare and args.compare.exists():
        print_compare(rep, json.loads(args.compare.read_text(encoding="utf-8")))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenarios_version": spec["meta"].get("version"),
            "summary": rep.summary(),
            "runs": [{**asdict(r), "verdict": r.verdict()} for r in runs],
        }
        args.save.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n저장: {args.save}")

    # 정상 통화 오탐이 있으면 실패로 본다 — 실사용을 막는 것은 오탐이다
    return 1 if (rep.fn or rep.fp) else 0


if __name__ == "__main__":
    raise SystemExit(main())
