"""패턴 매칭 — NFC 정규화 + 음절 정렬 자모 근사매칭.

STT 오차가 이 모드의 최대 취약점이다. D08 §07 측정 결과,
키워드 10개 × 흔한 변형 3종에서 **정확 일치는 21.9%만 잡았다.**
C1 "안전계좌"가 "안전 계좌"로 전사되면 최고위험 신호가 발동하지 않는다.
공백 정규화 + 자모 근사매칭으로 93.8%까지 오른다.

다만 근사매칭은 오탐도 함께 늘린다. 그래서 **비대칭 설계**를 쓴다.
  - 위험 키워드 → 근사매칭 허용 (놓치는 비용이 크다)
  - benign 지시어 → 정확 일치만 (잘못 발동하면 위험 신호를 깎아버린다)

## 근사매칭이 정상 통화를 잡던 문제 (실측 후 재설계)

`eval_mode1.py`로 정상 통화 12건을 돌렸더니 **근사매칭 오탐이 31건** 나왔다.
전부 원문에 없는 표현이 "있다"고 판정된 것이다.

| 잡힌 키워드 | 실제 원문 | 자모 편집거리 |
|---|---|---|
| 송금 | "조금 전", "등록금", "궁금하신" | 2 |
| 당장 | "담**당자**입니다" | **0** |
| 실형 | "조**심해**야 돼" | 2 |
| 벌금 | "**방금**" | 2 |
| 경찰청 | "경찰**서** 수사과" | 2 |
| 늦으면 | "**맞으면**" | 2 |

원인이 세 가지였고 셋 다 고쳤다.

**① 윈도우가 음절 경계를 넘나들었다.**
자모 문자열 위를 1칸씩 미끄러지므로 "담당자입니다"의 자모
`ㄷㅏㅁㄷㅏㅇㅈㅏㅇㅣㅂ…` 안에서 "당장"(`ㄷㅏㅇㅈㅏㅇ`)이 **정확히** 발견된다.
"자입"의 `ㅈㅏ` 뒤에 "입"의 초성 `ㅇ`이 붙어 만들어진 유령이다.
→ 윈도우 시작과 끝을 **음절 경계에만** 둔다.

**② 짧은 표현에 허용치가 과했다.**
2음절은 자모 6개다. 편집 2를 허용하면 자모의 3분의 1이 달라도 통과한다.
근사매칭이 흡수해야 할 STT 오차는 그런 크기가 아니다.
→ **4음절 미만은 근사매칭을 아예 끈다.** 정확 일치로만 잡는다.
   짧은 표현일수록 정확 일치가 잘 되므로 손실이 작다.

**③ 초성을 보지 않았다.**
한국어 STT 오차는 대부분 중성·종성에서 난다("계좌"→"개좌"는 초성이 같다).
반대로 초성이 다르면 대개 다른 단어다("송금" ㅅㄱ ↔ "조금" ㅈㄱ).
→ **초성 시퀀스 편집거리 1 이내**를 먼저 통과해야 자모 거리를 잰다.
   판별력이 높고, 비싼 편집거리 계산을 대부분 건너뛰게 해 **속도도 빨라진다.**

의도한 케이스는 그대로 남는다 — "안전계좌"↔"안전개좌"는 초성 `ㅇㅈㄱㅈ`가 같고
자모 거리 1이라 4음절 허용치 안에 들어온다.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# 한글 자모 분해 상수
HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3
CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 근사매칭 임계값 — 자모 편집거리를 길이로 정규화한 값이 이 아래면 같은 표현으로 본다.
# R04/D03이 지정한 값은 0.34다. 아래 음절 기준 상한과 함께 걸린다.
APPROX_THRESHOLD = 0.34

# 근사매칭을 켜는 최소 음절 수. 3음절 이하는 정확 일치로만 잡는다.
# 실측에서 오탐 31건 중 대부분이 2~3음절 표현이었다.
MIN_APPROX_SYLLABLES = 4

# 편집 허용치 상한 — 비율만 쓰면 긴 표현에서 과하게 커진다.
MAX_EDIT_BUDGET = 2

# 초성 시퀀스 편집 허용치. 0으로 두면 초성을 틀리는 전사 오차를 놓치고,
# 2 이상이면 판별력이 사라진다. 1이 "초성 하나까지는 봐준다"에 해당한다.
CHO_EDIT_BUDGET = 1

# 어절 사이에 끼어들 수 있는 음절 수의 상한. 아래 스윕 결과로 정했다.
GAP_MAX_SYLLABLES = 4

# 이 길이 미만의 어절은 gapped 매칭에서 정확 일치만 인정한다.
GAP_MIN_SEGMENT = 3

# 하위 호환 — 예전 이름을 참조하는 코드가 있을 수 있다.
MIN_APPROX_JAMO = MIN_APPROX_SYLLABLES * 2


def normalize(text: str) -> str:
    """NFC 정규화 + 공백·문장부호 제거.

    "안전 계좌" → "안전계좌". STT는 띄어쓰기를 거의 신뢰할 수 없다.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if not ch.isspace() and ch.isalnum())


def decompose(ch: str) -> str:
    """음절 하나를 자모로. 한글이 아니면 그대로 돌려준다."""
    code = ord(ch)
    if HANGUL_BASE <= code <= HANGUL_LAST:
        idx = code - HANGUL_BASE
        return CHO[idx // 588] + JUNG[(idx % 588) // 28] + JONG[idx % 28]
    return ch


def initial_of(ch: str) -> str:
    """음절의 초성. 한글이 아니면 그 문자 자체를 초성으로 본다."""
    code = ord(ch)
    if HANGUL_BASE <= code <= HANGUL_LAST:
        return CHO[(code - HANGUL_BASE) // 588]
    return ch


def to_jamo(text: str) -> str:
    """한글 음절을 자모 시퀀스로 분해한다.

    "개좌"와 "계좌"는 글자 단위로는 완전히 다르지만 자모로는 ㄱㅐ / ㄱㅖ로 한 칸 차이다.
    STT 오차가 대부분 이 수준이라 자모 단위로 재야 잡힌다.
    """
    return "".join(decompose(ch) for ch in text)


def edit_distance(a: str, b: str, cutoff: int | None = None) -> int:
    """Levenshtein 거리. 두 줄만 유지해 메모리를 아낀다.

    :param cutoff: 이 값을 넘는 것이 확정되면 즉시 중단하고 cutoff+1을 돌려준다.
        매칭 여부만 필요한 호출에서 계산을 크게 줄인다.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    if cutoff is not None and len(a) - len(b) > cutoff:
        return cutoff + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,                 # 삭제
                cur[j - 1] + 1,              # 삽입
                prev[j - 1] + (ca != cb),    # 치환
            ))
        if cutoff is not None and min(cur) > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


# ── 분석 결과 ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Analyzed:
    """정규화·자모·초성·음절경계를 한 번만 계산해 들고 다닌다.

    누적 버퍼는 발화가 쌓일 때마다 전체를 다시 채점하므로,
    이걸 캐시하지 않으면 같은 문자열을 수백 번 다시 분해하게 된다.
    """

    text: str                      # 정규화된 음절 시퀀스
    jamo: str
    cho: str                       # 음절별 초성 (len == len(text))
    starts: tuple[int, ...]        # 음절 i가 시작하는 jamo 인덱스. len == len(text) + 1

    @property
    def n_syllables(self) -> int:
        return len(self.text)


def analyze(raw: str) -> Analyzed:
    norm = normalize(raw)
    parts: list[str] = []
    cho: list[str] = []
    starts: list[int] = []
    pos = 0
    for ch in norm:
        starts.append(pos)
        j = decompose(ch)
        parts.append(j)
        cho.append(initial_of(ch))
        pos += len(j)
    starts.append(pos)
    return Analyzed(norm, "".join(parts), "".join(cho), tuple(starts))


@dataclass(frozen=True)
class MatchSpan:
    """어디가 왜 걸렸는지. 근거 패널과 문맥 판정(부정·인용)에 쓴다."""

    form: str
    kind: str          # "exact" | "approx"
    start: int         # 정규화 텍스트에서의 음절 인덱스
    end: int
    distance: int

    @property
    def approx(self) -> bool:
        return self.kind == "approx"


# 편집거리 2를 허용하려면 이 음절 수 이상이어야 한다.
#
# 4음절은 자모가 8개 안팎이라 거리 2면 **4분의 1이 달라도 통과**한다.
# 종단 실행(`e2e_test.py`)에서 실제로 걸렸다 —
# "계좌**에** 있는 돈을 안전하게 관리하기 위해"가 `계좌이체`로 매칭됐다(거리 2).
# "계좌에"는 완전히 정상적인 한국어이고, 텍스트 시나리오에는 없던 형태였다.
# 오디오를 통째로 돌려보지 않았으면 못 찾았을 오탐이다.
BUDGET2_MIN_SYLLABLES = 6


def approx_budget(n_syllables: int, n_jamo: int,
                  threshold: float = APPROX_THRESHOLD) -> int | None:
    """근사매칭 허용 편집거리. None이면 근사매칭을 쓰지 않는다.

    짧은 표현일수록 엄격하게 간다. 같은 편집거리라도 짧을수록 **비율이 크고**,
    비율이 크면 다른 단어가 걸린다.
    """
    if n_syllables < MIN_APPROX_SYLLABLES:
        return None
    cap = MAX_EDIT_BUDGET if n_syllables >= BUDGET2_MIN_SYLLABLES else 1
    return max(1, min(int(n_jamo * threshold), cap))


def _find_approx(hay: Analyzed, ndl: Analyzed, budget: int) -> MatchSpan | None:
    """음절 경계에 정렬된 창만 본다. 초성 시퀀스로 먼저 거른다."""
    ns, nh = ndl.n_syllables, hay.n_syllables
    if ns == 0 or nh == 0:
        return None

    best: MatchSpan | None = None
    widths = sorted({w for w in (ns - 1, ns, ns + 1) if 1 <= w <= nh})

    for w in widths:
        for i in range(0, nh - w + 1):
            # 초성 선별 — 대부분 여기서 걸러진다. 자모 편집거리보다 훨씬 싸다.
            if edit_distance(hay.cho[i:i + w], ndl.cho, cutoff=CHO_EDIT_BUDGET) \
                    > CHO_EDIT_BUDGET:
                continue
            a, b = hay.starts[i], hay.starts[i + w]
            d = edit_distance(hay.jamo[a:b], ndl.jamo, cutoff=budget)
            if d <= budget and (best is None or d < best.distance):
                best = MatchSpan(ndl.text, "approx", i, i + w, d)
                if d == 0:
                    return best
    return best


def approx_contains(haystack_jamo: str, needle_jamo: str,
                    threshold: float = APPROX_THRESHOLD) -> bool:
    """자모 문자열끼리 직접 비교하는 옛 진입점 — 하위 호환용.

    음절 경계 정보가 없으므로 **정렬 보정을 할 수 없다.**
    새 코드는 `Matcher.find_span`을 쓸 것.
    """
    n, m = len(haystack_jamo), len(needle_jamo)
    if m == 0 or n < m:
        return False
    budget = max(1, min(int(m * threshold), MAX_EDIT_BUDGET))
    for width in {max(1, m - budget), m, m + budget}:
        for start in range(0, n - width + 1):
            if edit_distance(haystack_jamo[start:start + width],
                             needle_jamo, cutoff=budget) <= budget:
                return True
    return False


class Matcher:
    """정규화·자모 변환 결과를 캐시한다.

    누적 전사 버퍼는 발화가 쌓일 때마다 매번 전체를 다시 채점하므로,
    캐시가 없으면 같은 문자열을 수백 번 다시 분해하게 된다.
    """

    def __init__(self, approx: bool = True, threshold: float = APPROX_THRESHOLD,
                 gapped: bool = True) -> None:
        self.approx = approx
        self.gapped = gapped
        self.threshold = threshold
        self._cache: dict[str, Analyzed] = {}
        self._segcache: dict[str, list[str]] = {}

    def _segments(self, needle: str) -> list[str]:
        """키워드의 어절 분해를 캐시한다. 같은 키워드를 발화마다 다시 쪼갤 이유가 없다."""
        got = self._segcache.get(needle)
        if got is None:
            got = [seg for seg in needle.split() if normalize(seg)]
            self._segcache[needle] = got
        return got

    def analyzed(self, text: str) -> Analyzed:
        got = self._cache.get(text)
        if got is None:
            got = analyze(text)
            self._cache[text] = got
        return got

    # ── 조회 ──────────────────────────────────────────────────────────────────

    def find_span(self, haystack: str, needle: str, exact_only: bool = False,
                  max_distance: int | None = None) -> MatchSpan | None:
        """매칭됐다면 어디가 왜 걸렸는지 함께 돌려준다.

        문맥 판정(부정·인용)이 매칭 위치를 알아야 하므로 이 진입점이 필요하다.

        :param max_distance: 허용 편집거리를 이 값 이하로 더 조인다.
            **판정 결과가 무거운 신호일수록 더 엄격한 증거를 요구한다.**
            critical 신호는 하나만 걸려도 위험도 하한 0.75를 강제하므로,
            전사가 망가진 상태에서 헐겁게 매칭되면 그 비용이 크다.
            실측: STT 오차 25%에서 이 제한이 오탐률을 크게 낮춘다.
        """
        hay = self.analyzed(haystack)
        ndl = self.analyzed(needle)
        if not ndl.text:
            return None

        at = hay.text.find(ndl.text)
        if at >= 0:
            return MatchSpan(ndl.text, "exact", at, at + ndl.n_syllables, 0)

        if exact_only or not self.approx:
            return None

        budget = approx_budget(ndl.n_syllables, len(ndl.jamo), self.threshold)
        if budget is None:
            return None            # 짧은 표현은 근사로 풀지 않는다
        if max_distance is not None:
            budget = min(budget, max_distance)
            if budget < 1:
                return None
        return _find_approx(hay, ndl, budget)

    def find_gapped(self, haystack: str, needle: str,
                    max_gap: int | None = None,
                    max_distance: int | None = None) -> MatchSpan | None:
        """어절 사이에 다른 성분이 끼어든 경우를 잡는다.

        한국어는 어절 사이에 조사·부사·수식어가 자유롭게 들어간다.
        연속 문자열 매칭은 이걸 **구조적으로** 못 잡는다. 실측:

            "가족에게 알리지"   ← "가족에게 이 사실을 알리지 마세요"      놓침
            "기존 대출 상환"    ← "기존 대출을 먼저 상환하셔야"          놓침
            "국고 계좌"        ← "국고 보호 예치 계좌로"                놓침

        셋 다 완전히 자연스러운 한국어 문장이다. 사기범이 노려서 쓴 회피가
        아니라 **그냥 말하면 이렇게 된다.** 놓치는 쪽이 기본값이었다는 뜻이다.

        자모 근사매칭으로는 풀 수 없다. 끼어든 어절이 편집거리 예산(최대 2)을
        훨씬 넘기 때문이다 — "을 먼저 "는 자모 7개다.

        그래서 **어절 단위로 쪼개 순서대로** 찾는다. 각 어절은 기존 매칭
        (정확 → 자모 근사)을 그대로 쓰고, 어절 사이에 `max_gap` 음절까지
        허용한다. 문장 단위로 호출되므로 문장을 넘어가지 않는다.

        오탐을 막는 제약 셋:
          - **연속 매칭이 실패했을 때만** 시도한다. 기존 판정을 바꾸지 않는다.
          - 어절이 `GAP_MIN_SEGMENT` 음절 미만이면 **정확 일치만** 인정한다.
            1~2음절 어절은 근사까지 허용하면 아무 데나 걸린다.
          - 어절이 하나뿐인 키워드는 대상이 아니다. 쪼갤 것이 없다.
        """
        if not self.gapped:
            return None
        if max_gap is None:
            max_gap = GAP_MAX_SYLLABLES
        segs = self._segments(needle)
        if len(segs) < 2:
            return None

        # 조기 종료 — 짧은 어절은 어차피 정확 일치만 인정하므로,
        # 문자열에 아예 없으면 순서 탐색을 시작할 이유가 없다.
        # 이 검사가 gapped 경로 비용의 절반을 걷어낸다.
        hay_text = self.analyzed(haystack).text
        for seg in segs:
            seg_norm = normalize(seg)
            if len(seg_norm) < GAP_MIN_SEGMENT and seg_norm not in hay_text:
                return None

        cursor = 0
        start = end = None
        total_distance = 0
        hay_norm = hay_text

        for seg in segs:
            seg_n = self.analyzed(seg)
            tail = hay_norm[cursor:]
            span = self.find_span(
                tail, seg,
                exact_only=seg_n.n_syllables < GAP_MIN_SEGMENT,
                max_distance=max_distance,
            )
            if span is None:
                return None
            abs_start, abs_end = cursor + span.start, cursor + span.end
            if start is None:
                start = abs_start
            elif abs_start - end > max_gap:
                return None            # 너무 멀다. 같은 표현으로 보기 어렵다
            end = abs_end
            total_distance += span.distance
            cursor = abs_end

        return MatchSpan(normalize(needle), "gapped", start, end, total_distance)

    def match(self, haystack: str, needle: str, exact_only: bool = False,
              max_distance: int | None = None) -> bool:
        """haystack 안에 needle이 있는가.

        :param exact_only: benign 지시어에 쓴다. 근사매칭을 끄는 비대칭 설계의 한쪽.
        """
        span = self.find_span(haystack, needle, exact_only, max_distance)
        if span is None and not exact_only:
            span = self.find_gapped(haystack, needle)
        return span is not None

    def find_all(self, haystack: str, needles: list[str],
                 exact_only: bool = False) -> list[str]:
        """매칭된 표현을 모두 돌려준다 — 근거 패널에 그대로 쓴다."""
        return [n for n in needles if self.match(haystack, n, exact_only)]

    def find_all_spans(self, haystack: str, needles: list[str],
                       exact_only: bool = False) -> list[MatchSpan]:
        out = []
        for n in needles:
            span = self.find_span(haystack, n, exact_only)
            if span is None and not exact_only:
                span = self.find_gapped(haystack, n)
            if span:
                out.append(span)
        return out
