"""테스트용 참조 음성 생성 — **.venv-xtts 전용 스크립트.**

팀원 녹음이 들어오기 전까지 파이프라인 전체를 돌려보기 위한 임시 수단이다.
XTTS-v2에 내장된 화자로 음성을 만들어 "보호할 대상"으로 쓴다.

**이것으로 얻은 수치는 결과가 아니다.** 두 가지 이유 때문이다.
  ① 대상 음성 자체가 XTTS가 만든 것이라 XTTS 입장에서 분포 안쪽이다 — 복제가 유난히 쉽다.
  ② 따라서 "복제 실패"가 나오면 의미가 있지만, "복제 성공"이 나와도
     사람 목소리에서 그렇다는 뜻이 아니다.
목적은 어디까지나 **코드 경로를 미리 뚫어 두는 것**이다.
팀원 녹음이 오면 같은 명령을 그대로 다시 돌린다.

사용:
    .venv-xtts\\Scripts\\python make_reference.py out/reference.wav
    .venv-xtts\\Scripts\\python make_reference.py out/ref.wav --list
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# 6초 이상이어야 화자 임베딩이 안정된다. 계획서 시연 대본과 같은 계열의 문장을 쓴다.
DEFAULT_TEXT = (
    "안녕하세요. 저는 오늘 날씨가 참 좋다고 생각합니다. "
    "점심에는 근처 식당에서 밥을 먹고, 오후에는 도서관에 갈 예정입니다."
)


def main() -> int:
    p = argparse.ArgumentParser(description="테스트용 참조 음성 생성 (격리 venv 전용)")
    p.add_argument("output", nargs="?", default="out/reference.wav")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--speaker", default=None, help="내장 화자 이름 (미지정 시 첫 번째)")
    p.add_argument("--language", default="ko")
    p.add_argument("--list", action="store_true", help="내장 화자 목록만 출력")
    args = p.parse_args()

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[xtts] 장치 {device} · 모델 로드 중...", flush=True)
    tts = TTS(MODEL).to(device)

    speakers = list(getattr(tts, "speakers", None) or [])
    if args.list:
        print(f"내장 화자 {len(speakers)}명:")
        for s in speakers:
            print(f"  {s}")
        return 0

    if not speakers:
        print("내장 화자가 없다. 이 모델 버전은 참조 음성이 반드시 필요하다.")
        return 1

    speaker = args.speaker or speakers[0]
    print(f"[xtts] 화자: {speaker}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=args.text,
        speaker=speaker,
        language=args.language,
        file_path=args.output,
    )
    print(f"[xtts] 생성 완료: {args.output}")
    print("※ 이 음성은 배관 확인용이다. 결과 수치는 사람 목소리로 다시 낼 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
