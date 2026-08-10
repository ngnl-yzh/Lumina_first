"""F5-TTS 음성 복제 — **.venv-f5 전용 스크립트.**

메인 venv에서 실행하지 말 것. `clone_xtts.py`와 같은 이유로 격리했다.
호출은 `clone_test.py`가 subprocess로 한다.

## 왜 F5-TTS를 추가하나

지금까지의 복제 검증은 **XTTS-v2 하나**였다. 거기서 "보호본도 복제된다"가 나왔는데,
그것만으로는 두 가지를 구별할 수 없다.

    ① XTTS 특유의 성질이라 다른 모델에는 통할지도 모른다
    ② 임베딩 표적 방어 자체의 구조적 한계다

**F5-TTS는 XTTS와 구조가 완전히 다르다.**
XTTS는 GPT 계열 자기회귀 + HiFi-GAN 보코더이고, F5-TTS는 flow matching 기반이다.
화자 조건을 넣는 경로도 다르다. 여기서도 같은 결과가 나오면 ②가 지지된다.

즉 이 스크립트는 **"모델을 하나 더 시험한다"가 아니라 결론의 일반성을 검증한다.**

## 설치 (RTX 장비 기준)

```bash
cd server
py -3.11 -m venv .venv-f5
.venv-f5\\Scripts\\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv-f5\\Scripts\\python -m pip install f5-tts
```

한국어는 F5-TTS 기본 체크포인트가 다국어 학습이 아니라 약할 수 있다.
**그래서 기준선(원본 복제) 유사도를 반드시 함께 본다** — 기준선이 낮으면
모델이 애초에 그 목소리를 복제하지 못한 것이라 아무것도 주장할 수 없다.
`clone_test.py`가 그 판정을 자동으로 한다.

사용:
    .venv-f5\\Scripts\\python clone_f5.py 참조.wav 출력.wav --text "엄마 나 사고 났어"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TEXT = "엄마, 나 사고 났어. 지금 급하게 돈이 필요해."

# 참조 음성의 전사. F5-TTS는 참조 오디오와 **그 오디오의 텍스트**를 함께 받는다.
# 비워 두면 내부 ASR로 뽑는데, 그건 느리고 오차가 섞인다.
# 우리 실험은 참조 문장을 알고 있으므로 직접 넘기는 편이 낫다.
DEFAULT_REF_TEXT = ""


def main() -> int:
    p = argparse.ArgumentParser(description="F5-TTS 복제 (격리 venv 전용)")
    p.add_argument("reference", help="참조 음성 WAV — 이 목소리를 복제한다")
    p.add_argument("output", help="출력 WAV")
    p.add_argument("--text", default=DEFAULT_TEXT, help="생성할 문장")
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT,
                   help="참조 음성의 전사. 비우면 F5가 ASR로 추정한다")
    p.add_argument("--model", default="F5TTS_v1_Base")
    args = p.parse_args()

    ref = Path(args.reference)
    if not ref.exists():
        print(f"참조 파일 없음: {ref}", file=sys.stderr)
        return 1

    import soundfile as sf
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[f5] 장치 {device} · 모델 로드 중...", flush=True)

    try:
        from f5_tts.api import F5TTS
    except ImportError:
        print("f5-tts가 설치되지 않았다. 이 스크립트 상단의 설치 절차를 볼 것.",
              file=sys.stderr)
        return 2

    tts = F5TTS(model=args.model, device=device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    wav, sr, _ = tts.infer(
        ref_file=str(ref),
        ref_text=args.ref_text,      # ""면 내부 ASR이 채운다
        gen_text=args.text,
        remove_silence=False,        # 무음 제거는 섭동 구간까지 건드릴 수 있다
    )
    sf.write(args.output, wav, sr)

    print(f"[f5] 생성 완료: {args.output}  ({len(wav) / sr:.1f}초 · {sr} Hz)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
