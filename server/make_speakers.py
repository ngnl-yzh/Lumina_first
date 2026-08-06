"""C-D 대조군용 다화자 생성 — **.venv-xtts 전용 스크립트.**

## 왜 이게 필요한가

`controls.py`의 C-D는 "타 화자 음성 — '다른 사람' 판정 기준선. DSR 임계값의 근거"다.
지금까지 이 대조군이 **없었다.** 그래서 SRS 수치를 해석할 수가 없었다.

    보호본의 SRS가 0.46이다. → 그래서 이게 좋은 건가?

답할 수 없다. **서로 다른 두 사람의 SRS가 얼마인지 모르기 때문이다.**
그 값이 0.75라면 0.46은 "남남보다 더 멀어졌다"는 강한 주장이 되고,
0.20이라면 0.46은 "여전히 같은 사람으로 들린다"는 뜻이 된다.
정반대의 결론이 같은 숫자에서 나온다.

`pipeline.py`의 `PROVISIONAL_THRESHOLD = 0.75`도 이 대조군에서 나와야 할 값인데,
지금은 근거 없는 잠정값이라 주석에 "절대 판정으로 읽으면 안 된다"고 적혀 있다.

## 한계 — 먼저 밝힌다

여기서 만드는 것은 **XTTS 내장 화자의 합성음**이지 사람 목소리가 아니다.
합성 화자끼리의 거리가 실제 사람끼리의 거리와 같다는 보장이 없다.
따라서 여기서 나온 임계값도 **잠정값**이며, 팀원 녹음이 확보되면 다시 재야 한다.
그래도 근거 없는 0.75보다는 낫다 — 적어도 어디서 나온 숫자인지 말할 수 있다.

`make_reference.py`와 달리 모델을 **한 번만** 올린다.
화자마다 스크립트를 새로 띄우면 CPU에서 모델 로드가 생성보다 오래 걸린다.

사용:
    .venv-xtts\\Scripts\\python make_speakers.py -o out/speakers -n 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# 화자 임베딩이 안정되려면 6초 이상이어야 한다.
DEFAULT_TEXT = (
    "안녕하세요. 저는 오늘 날씨가 참 좋다고 생각합니다. "
    "점심에는 근처 식당에서 밥을 먹고, 오후에는 도서관에 갈 예정입니다."
)

# 성별·음색이 고루 섞이도록 고른다. 한쪽으로 몰리면 "다른 사람" 거리가
# 실제보다 좁게 나와 임계값이 낙관적으로 잡힌다.
PREFERRED = [
    "Nova Hogarth", "Maja Ruoho", "Alexandra Hisakawa",
    "Aaron Dreschner", "Kumar Dahl", "Luis Moray",
    "Barbora MacLean", "Damjan Chapman",
]


def main() -> int:
    p = argparse.ArgumentParser(description="C-D 대조군용 다화자 생성 (격리 venv 전용)")
    p.add_argument("-o", "--out", default="out/speakers")
    p.add_argument("-n", "--count", type=int, default=6)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--language", default="ko")
    args = p.parse_args()

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[xtts] 장치 {device} · 모델 로드 중...", flush=True)
    tts = TTS(MODEL).to(device)

    available = list(getattr(tts, "speakers", None) or [])
    if not available:
        print("내장 화자가 없다. 이 모델 버전은 참조 음성이 반드시 필요하다.")
        return 1

    chosen = [s for s in PREFERRED if s in available][: args.count]
    if len(chosen) < args.count:                       # 선호 목록이 모자라면 채운다
        chosen += [s for s in available if s not in chosen][: args.count - len(chosen)]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, speaker in enumerate(chosen, start=1):
        path = outdir / f"spk{i:02d}.wav"
        print(f"[xtts] {i}/{len(chosen)} {speaker} → {path}", flush=True)
        tts.tts_to_file(text=args.text, speaker=speaker,
                        language=args.language, file_path=str(path))

    (outdir / "speakers.txt").write_text(
        "\n".join(f"spk{i:02d}\t{s}" for i, s in enumerate(chosen, start=1)),
        encoding="utf-8",
    )
    print(f"\n생성 완료: {len(chosen)}명 → {outdir}")
    print("※ 합성 화자다. 여기서 나온 임계값은 잠정값이며 사람 목소리로 다시 잴 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
