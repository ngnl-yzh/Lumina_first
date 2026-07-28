"""패턴 매칭 — NFC 정규화 + 자모 근사매칭.

STT 오차가 이 모드의 최대 취약점이다. D08 §07 측정 결과,
키워드 10개 × 흔한 변형 3종에서 **정확 일치는 21.9%만 잡았다.**
C1 "안전계좌"가 "안전 계좌"로 전사되면 최고위험 신호가 발동하지 않는다.
공백 정규화 + 자모 근사매칭으로 93.8%까지 오른다.

다만 근사매칭은 오탐도 함께 늘린다. 그래서 **비대칭 설계**를 쓴다.
  - 위험 키워드 → 근사매칭 허용 (놓치는 비용이 크다)
  - benign 지시어 → 정확 일치만 (잘못 발동하면 위험 신호를 깎아버린다)
"""

from __future__ import annotations

import unicodedata

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
# R04/D03이 지정한 값은 0.34다.
APPROX_THRESHOLD = 0.34

# 근사매칭을 적용할 최소 길이. 짧은 표현은 근사로 풀면 아무 데나 걸린다.
MIN_APPROX_JAMO = 6

# 편집 허용치 상한 — 비율만 쓰면 중간 길이 표현에서 오탐이 난다.
#
# 실측 사례: "오늘 날씨가 참 좋네요"가 S6 키워드 "오늘까지"에 매칭됐다.
#   오늘까지 → ㅇㅗㄴㅡㄹㄲㅏㅈㅣ (자모 9개), 비율 0.34면 허용치 3
#   오늘날씨 → ㅇㅗㄴㅡㄹㄴㅏㄹㅆㅣ 와의 편집거리가 정확히 3
# 완전한 정상 발화가 위험 키워드로 잡힌다.
#
# 근사매칭의 목적은 **STT 전사 오차 흡수**이고, 그 오차는 대개 자모 1~2개다
# ("계좌"→"개좌"는 1개). 어형 변화는 DB의 variants가 명시적으로 담당한다.
# 따라서 허용치를 2로 묶는다. 의도한 케이스는 그대로 잡히고 위 오탐은 사라진다.
MAX_EDIT_BUDGET = 2


def normalize(text: str) -> str:
    """NFC 정규화 + 공백·문장부호 제거.

    "안전 계좌" → "안전계좌". STT는 띄어쓰기를 거의 신뢰할 수 없다.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if not ch.isspace() and ch.isalnum())


def to_jamo(text: str) -> str:
    """한글 음절을 자모 시퀀스로 분해한다.

    "개좌"와 "계좌"는 글자 단위로는 완전히 다르지만 자모로는 ㄱㅐ / ㄱㅖ로 한 칸 차이다.
    STT 오차가 대부분 이 수준이라 자모 단위로 재야 잡힌다.
    """
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if HANGUL_BASE <= code <= HANGUL_LAST:
            idx = code - HANGUL_BASE
            out.append(CHO[idx // 588])
            out.append(JUNG[(idx % 588) // 28])
            jong = JONG[idx % 28]
            if jong:
                out.append(jong)
        else:
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 거리. 두 줄만 유지해 메모리를 아낀다."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,          # 삭제
                cur[j - 1] + 1,       # 삽입
                prev[j - 1] + (ca != cb),   # 치환
            ))
        prev = cur
    return prev[-1]


def approx_contains(haystack_jamo: str, needle_jamo: str,
                    threshold: float = APPROX_THRESHOLD) -> bool:
    """needle과 충분히 비슷한 구간이 haystack 안에 있는지 본다.

    슬라이딩 윈도우로 needle 길이만큼씩 잘라 편집거리를 재고,
    길이로 정규화한 값이 임계값 아래면 매칭으로 판정한다.
    """
    n, m = len(haystack_jamo), len(needle_jamo)
    if m == 0 or n < m:
        return False

    budget = min(int(m * threshold), MAX_EDIT_BUDGET)
    if budget == 0:
        return needle_jamo in haystack_jamo

    # 창 길이를 needle 기준으로 ±budget 만큼 흔들어 삽입·삭제도 흡수한다
    for width in {max(1, m - budget), m, m + budget}:
        for start in range(0, n - width + 1):
            window = haystack_jamo[start:start + width]
            if edit_distance(window, needle_jamo) <= budget:
                return True
    return False


class Matcher:
    """정규화·자모 변환 결과를 캐시한다.

    누적 전사 버퍼는 발화가 쌓일 때마다 매번 전체를 다시 채점하므로,
    캐시가 없으면 같은 문자열을 수백 번 다시 분해하게 된다.
    """

    def __init__(self, approx: bool = True, threshold: float = APPROX_THRESHOLD) -> None:
        self.approx = approx
        self.threshold = threshold
        self._jamo_cache: dict[str, str] = {}

    def _jamo(self, text: str) -> str:
        cached = self._jamo_cache.get(text)
        if cached is None:
            cached = to_jamo(normalize(text))
            self._jamo_cache[text] = cached
        return cached

    def match(self, haystack: str, needle: str, exact_only: bool = False) -> bool:
        """haystack 안에 needle이 있는가.

        :param exact_only: benign 지시어에 쓴다. 근사매칭을 끄는 비대칭 설계의 한쪽.
        """
        h_norm = normalize(haystack)
        n_norm = normalize(needle)
        if not n_norm:
            return False
        if n_norm in h_norm:
            return True
        if exact_only or not self.approx:
            return False

        n_jamo = to_jamo(n_norm)
        if len(n_jamo) < MIN_APPROX_JAMO:
            return False        # 짧은 표현은 근사로 풀지 않는다
        return approx_contains(self._jamo(haystack), n_jamo, self.threshold)

    def find_all(self, haystack: str, needles: list[str],
                 exact_only: bool = False) -> list[str]:
        """매칭된 표현을 모두 돌려준다 — 근거 패널에 그대로 쓴다."""
        return [n for n in needles if self.match(haystack, n, exact_only)]
