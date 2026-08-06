"""한국어 STT 오차 주입 — 전사 품질이 떨어질 때 탐지가 얼마나 버티는가.

평가 시나리오는 전부 깨끗한 텍스트다. 그런데 **실제 입력은 Whisper 출력**이고,
D08 §07은 STT 오차를 이 모드의 최대 취약점으로 지목했다.
깨끗한 텍스트로 잰 탐지율은 상한이지 실제 성능이 아니다.

특히 이 측정이 필요한 이유가 하나 더 있다.
오탐을 줄이려고 근사매칭을 **더 엄격하게** 바꿨다 — 4음절 미만 차단, 초성 일치 요구.
근사매칭의 존재 이유가 STT 오차 흡수인데 그걸 조인 것이므로,
**정밀도를 얻는 대신 견고성을 잃었을 가능성**을 반드시 확인해야 한다.

## 오차 모델

Whisper 한국어에서 실제로 자주 나는 오차만 넣는다.
문법적으로 가능한 모든 변형을 넣으면 측정이 무의미해진다.

| 오차 | 예 | 비중 |
|---|---|---|
| 중성 혼동 | 계좌 → 게좌 · 됐 → 뒜 | 가장 흔하다. ㅐ/ㅔ, ㅚ/ㅙ/ㅞ는 현대 한국어에서 발음이 거의 같다 |
| 종성 탈락 | 검찰청 → 검차청 | 받침은 약하게 발음되고 자주 사라진다 |
| 종성 혼동 | 있 → 읻 · 감 → 강 | 평파열음화·비음 혼동 |
| 초성 혼동 | 계좌 → 께좌 | 경음·격음 구별 실패 |
| 띄어쓰기 | 안전계좌 → 안전 계좌 | STT는 띄어쓰기를 거의 신뢰할 수 없다 |

**정규화(`matcher.normalize`)가 공백을 지우므로 띄어쓰기 오류는 이미 무해하다.**
그래도 넣는 이유는, 무해하다는 것 자체가 측정으로 확인되어야 하기 때문이다.

## 쓰는 법

```bash
python eval_mode1.py --stt-noise 0.10 --repeat 20     # 음절당 10% 오차, 20회 평균
python eval_mode1.py --noise-curve                    # 0~25% 곡선 + 근사매칭 on/off 비교
```

난수 시드를 고정하므로 같은 명령은 같은 결과를 낸다.
"오차율 10%"는 **음절당** 확률이다. 8음절 표현이면 하나 이상 틀릴 확률이 57%다.
"""

from __future__ import annotations

import random

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"   # 인덱스 0은 받침 없음

# 서로 잘 혼동되는 묶음. 같은 묶음 안에서만 바꾼다.
JUNG_CONFUSIONS = [
    "ㅐㅔ",          # 발음이 사실상 같다 — 가장 흔한 오차
    "ㅒㅖ",
    "ㅘㅙㅚ",
    "ㅝㅞㅟ",
    "ㅓㅏ",
    "ㅗㅜ",
    "ㅕㅓ",
    "ㅡㅜ",
    "ㅢㅣ",
]
CHO_CONFUSIONS = [
    "ㄱㄲㅋ", "ㄷㄸㅌ", "ㅂㅃㅍ", "ㅅㅆ", "ㅈㅉㅊ",
]
JONG_CONFUSIONS = [
    "ㄱㄲㅋ", "ㄷㅅㅆㅈㅊㅌ", "ㅂㅍ", "ㄴㅇㅁ",
]

# 오차 종류별 상대 비중. 합이 1일 필요는 없다.
WEIGHTS = {
    "jung": 0.45,      # 중성 혼동
    "jong_drop": 0.20, # 종성 탈락
    "jong": 0.20,      # 종성 혼동
    "cho": 0.15,       # 초성 혼동
}


def _swap_in_group(ch: str, groups: list[str], rng: random.Random) -> str:
    for g in groups:
        if ch in g:
            alts = [c for c in g if c != ch]
            return rng.choice(alts) if alts else ch
    return ch


def perturb_syllable(ch: str, rng: random.Random) -> str:
    """한 음절에 오차 하나를 넣는다. 한글이 아니면 그대로."""
    code = ord(ch)
    if not (HANGUL_BASE <= code <= HANGUL_LAST):
        return ch

    idx = code - HANGUL_BASE
    cho, jung, jong = idx // 588, (idx % 588) // 28, idx % 28

    kinds = list(WEIGHTS)
    weights = [WEIGHTS[k] for k in kinds]
    # 받침이 없으면 종성 관련 오차는 낼 수 없다
    if jong == 0:
        weights = [0.0 if k.startswith("jong") else w for k, w in zip(kinds, weights)]
        if sum(weights) == 0:
            return ch
    kind = rng.choices(kinds, weights=weights, k=1)[0]

    if kind == "jung":
        new = _swap_in_group(JUNG[jung], JUNG_CONFUSIONS, rng)
        jung = JUNG.index(new)
    elif kind == "jong_drop":
        jong = 0
    elif kind == "jong":
        new = _swap_in_group(JONG[jong - 1], JONG_CONFUSIONS, rng)
        jong = JONG.index(new) + 1
    else:
        new = _swap_in_group(CHO[cho], CHO_CONFUSIONS, rng)
        cho = CHO.index(new)

    return chr(HANGUL_BASE + cho * 588 + jung * 28 + jong)


def perturb(text: str, rate: float, rng: random.Random,
            spacing: bool = True) -> str:
    """음절마다 `rate` 확률로 오차를 넣는다.

    :param rate: 음절당 오차 확률. 0.10이면 8음절 표현이 하나 이상 틀릴 확률 57%.
    :param spacing: 띄어쓰기 오류도 함께 넣을지.
    """
    out = [perturb_syllable(ch, rng) if rng.random() < rate else ch for ch in text]
    text = "".join(out)

    if spacing and rate > 0:
        chars: list[str] = []
        for ch in text:
            if ch == " " and rng.random() < rate * 2:
                continue                                  # 띄어쓰기 삭제
            chars.append(ch)
            if ch != " " and rng.random() < rate * 0.5:
                chars.append(" ")                         # 없던 띄어쓰기 삽입
        text = "".join(chars)
    return text


def perturb_scenario(spec: dict, rate: float, rng: random.Random) -> dict:
    """시나리오 하나의 모든 발화에 오차를 넣은 복사본을 만든다."""
    return {
        **spec,
        "utterances": [
            {**u, "text": perturb(u["text"], rate, rng)}
            for u in spec["utterances"]
        ],
    }


if __name__ == "__main__":
    rng = random.Random(0)
    samples = [
        "서울중앙지방검찰청 첨단범죄수사부 박정호 검사입니다",
        "안전계좌로 자금을 이체하셔야 합니다",
        "수사 기밀이니 가족에게도 말하지 마십시오",
    ]
    print("오차율별 전사 예시\n")
    for rate in (0.05, 0.10, 0.20):
        print(f"--- rate {rate:.2f} ---")
        for s in samples:
            print(f"  {perturb(s, rate, rng)}")
        print()
