"""XTTS-v2 음성 복제 — **.venv-xtts 전용 스크립트.**

메인 venv에서 실행하지 말 것. coqui-tts의 의존성이 무거워
잘 돌고 있는 PGD 환경(torch/numpy 핀)을 깨뜨릴 수 있어 격리했다.
호출은 clone_test.py가 subprocess로 한다.

라이선스 — XTTS-v2는 CPML(Coqui Public Model License)로 **비상업 용도만** 허용된다.
연구·경진대회 시연은 범위 안이지만, 상용화 시에는 OpenVoice v2(MIT) 등으로 교체해야 한다.
계획서 R02의 "한국어 지원 유일 후보" 서술도 이때 함께 수정할 것.

사용:
    .venv-xtts\\Scripts\\python clone_xtts.py 참조.wav 출력.wav --text "엄마 나 사고 났어"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 모델 다운로드 시 라이선스 동의 프롬프트가 뜨면 프로세스가 멈춘다.
# 비대화 실행이므로 미리 동의를 표시한다. (위 라이선스 설명 참조)
os.environ.setdefault("COQUI_TOS_AGREED", "1")

DEFAULT_TEXT = "엄마, 나 사고 났어. 지금 급하게 돈이 필요해."
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def main() -> int:
    p = argparse.ArgumentParser(description="XTTS-v2 복제 (격리 venv 전용)")
    p.add_argument("reference", help="참조 음성 WAV — 이 목소리를 복제한다")
    p.add_argument("output", help="출력 WAV")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--language", default="ko")
    args = p.parse_args()

    ref = Path(args.reference)
    if not ref.exists():
        print(f"참조 파일 없음: {ref}", file=sys.stderr)
        return 1

    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[xtts] 장치 {device} · 모델 로드 중...", flush=True)

    tts = TTS(MODEL).to(device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=args.text,
        speaker_wav=str(ref),
        language=args.language,
        file_path=args.output,
    )
    print(f"[xtts] 생성 완료: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
