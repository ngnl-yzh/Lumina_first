"""패턴 DB 로더 — 8단계 구조체 + critical + pair + benign.

DB 내용 자체는 `pattern_db.json`에 있다. 코드와 분리한 이유가 두 가지다.
  ① 클라이언트(PWA)에서도 같은 JSON을 그대로 읽는다.
  ② **DB 작성자와 평가 시나리오 작성자를 분리**해야 하는데(D08 §07),
     내용이 코드에 섞여 있으면 그 분리가 흐려진다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# D08 §02 — 단계 가중치. S4가 최고인 이유는 §2.1에 있다.
STAGE_WEIGHTS: dict[str, float] = {
    "S1": 0.80,   # 권위 확립
    "S2": 0.70,   # 연루 통보
    "S3": 0.85,   # 공포 조성
    "S4": 1.00,   # 고립 유도 — 정상 통화에 나타날 이유가 없어 오탐률이 극히 낮다
    "S5": 0.90,   # 행동 지시
    "S6": 0.60,   # 시간 압박
    "S7": 0.85,   # 가족 사칭
    "S8": 0.70,   # 대출 사기
}

STAGE_LABELS: dict[str, str] = {
    "S1": "권위 확립", "S2": "연루 통보", "S3": "공포 조성", "S4": "고립 유도",
    "S5": "행동 지시", "S6": "시간 압박", "S7": "가족 사칭", "S8": "대출 사기",
}

# 커버리지 보너스 최대치. ROUTE_MAX 계산에 들어간다.
COVERAGE_BONUS_MAX = 1.5

DEFAULT_DB_PATH = Path(__file__).parent / "pattern_db.json"


@dataclass
class Keyword:
    text: str
    score: float = 1.0
    variants: list[str] = field(default_factory=list)
    generic: bool = False
    """정상 통화에도 흔히 나오는 표현인가.

    "이체"·"송금"은 금융 생활의 기본 동작이다. 돈을 보낸다는 **사실 자체는
    사기의 증거가 아니다.** 증거는 어디로·어떻게 보내느냐다 —
    화자가 즉석에서 지정한 계좌, 현금 인출 후 대면 전달, 인증정보 요구.

    단계 점수에는 그대로 반영한다. 다만 조합 신호(P4)처럼 하한 0.75를
    강제하는 무거운 판정은 이 표현만으로 발동하지 않게 막는다.
    """

    def all_forms(self) -> list[str]:
        return [self.text, *self.variants]


@dataclass
class Stage:
    id: str
    label: str
    weight: float
    keywords: list[Keyword]

    @property
    def n_base(self) -> int:
        return len(self.keywords)

    @property
    def n_variants(self) -> int:
        return sum(len(k.variants) for k in self.keywords)


@dataclass
class Critical:
    """단독으로 고위험인 신호. 하나만 나와도 위험도 하한 0.75가 걸린다."""

    id: str
    text: str
    rationale: str
    variants: list[str] = field(default_factory=list)

    def all_forms(self) -> list[str]:
        return [self.text, *self.variants]


@dataclass
class Pair:
    """조합 고위험 신호.

    금융감독원의 '정부기관 판별 3원칙'은 단일 표현이 아니라 조합 규칙이다.
    세 유형과 1:1로 대응한다.
    """

    id: str
    stages: tuple[str, str]
    principle: str
    fraud_type: str
    needs_specific: tuple[str, ...] = ()
    """이 단계들은 **generic이 아닌** 표현으로 걸려야 조합이 성립한다.

    검증셋에서 P4(고립+자금이동)의 오탐 3건이 전부 "이체"·"송금" 단독이었다.
    은행 직원이 보이스피싱을 말리는 통화까지 사기로 판정했다 — 최악의 오탐이다.
    같은 검증셋의 진짜 사기 3건은 전부 구체적 형태로 걸렸다
    ("알려드리는 계좌", "현금으로 찾"+"수사관에게 전달", "예치하시면").
    """


@dataclass
class PatternDB:
    stages: dict[str, Stage]
    criticals: list[Critical]
    pairs: list[Pair]
    benign: list[str]
    meta: dict = field(default_factory=dict)

    # ── 집계 ──────────────────────────────────────────────────────────────────

    @property
    def n_base(self) -> int:
        return sum(s.n_base for s in self.stages.values())

    @property
    def n_variants(self) -> int:
        return sum(s.n_variants for s in self.stages.values())

    @property
    def n_total(self) -> int:
        """문서 서술은 '8단계 72개 표현 · 변형 포함 182개 항목'으로 통일한다."""
        return self.n_base + self.n_variants

    def summary(self) -> str:
        parts = [
            f"기본 {self.n_base}개 · 변형 {self.n_variants}개 · 합계 {self.n_total}개",
            f"critical {len(self.criticals)}개 · pair {len(self.pairs)}개 "
            f"· benign {len(self.benign)}개",
        ]
        for sid in sorted(self.stages):
            s = self.stages[sid]
            parts.append(f"  {sid} {s.label:<8} w {s.weight:.2f} "
                         f"· {s.n_base} + 변형 {s.n_variants}")
        return "\n".join(parts)


def load_db(path: Path | str | None = None) -> PatternDB:
    raw = json.loads(Path(path or DEFAULT_DB_PATH).read_text(encoding="utf-8"))

    stages: dict[str, Stage] = {}
    for sid, body in raw["stages"].items():
        stages[sid] = Stage(
            id=sid,
            label=body.get("label", STAGE_LABELS.get(sid, sid)),
            weight=body.get("weight", STAGE_WEIGHTS[sid]),
            keywords=[
                Keyword(text=k["text"], score=k.get("score", 1.0),
                        variants=k.get("variants", []),
                        generic=k.get("generic", False))
                for k in body["keywords"]
            ],
        )

    return PatternDB(
        stages=stages,
        criticals=[
            Critical(id=c["id"], text=c["text"], rationale=c["rationale"],
                     variants=c.get("variants", []))
            for c in raw.get("critical", [])
        ],
        pairs=[
            Pair(id=p["id"], stages=tuple(p["stages"]), principle=p["principle"],
                 fraud_type=p["fraud_type"],
                 needs_specific=tuple(p.get("needs_specific", [])))
            for p in raw.get("pairs", [])
        ],
        benign=raw.get("benign", []),
        meta=raw.get("meta", {}),
    )
