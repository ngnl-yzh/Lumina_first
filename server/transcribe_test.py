"""STT 설정 비교 — 어느 조합이 실제로 나은지 재본다.

"모델을 키우면 좋아진다"는 추측이고, 필요한 건 **이 목소리, 이 문장에서** 얼마나 나은가다.
같은 음성에 여러 설정을 돌려 나란히 놓는다. 지연도 함께 잰다 —
정확해도 3초 걸리면 개입 시점을 놓친다.

녹음 방법:
    1) 앱에서 모드 2로 녹음하고 "원본 저장"을 누르면 16 kHz WAV가 나온다
    2) 또는 휴대폰 녹음기로 찍어서 옮겨도 된다 (자동 리샘플)

사용:
    python transcribe_test.py 녹음.wav
    python transcribe_test.py 녹음.wav --models base,small,medium
    python transcribe_test.py 녹음.wav --expect "안전계좌로 이체하세요"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mirinae.mode1.stt import SpeechToText, build_initial_prompt


def cer(ref: str, hyp: str) -> float:
    """문자 오류율. 공백을 지우고 잰다 — 띄어쓰기는 어차피 정규화 단계에서 없앤다."""
    r = "".join(ref.split())
    h = "".join(hyp.split())
    if not r:
        return 0.0

    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def main() -> int:
    p = argparse.ArgumentParser(description="STT 설정 비교")
    p.add_argument("input")
    p.add_argument("--models", default="base,small",
                   help="쉼표 구분. medium은 CPU에서 상당히 느리다")
    p.add_argument("--beams", default="1,3")
    p.add_argument("--expect", default=None,
                   help="정답 문장. 주면 문자 오류율(CER)을 계산한다")
    p.add_argument("--no-prompt-compare", action="store_true",
                   help="initial_prompt 유무 비교를 생략한다")
    args = p.parse_args()

    from protect import load_wav

    wav = load_wav(Path(args.input)).numpy().astype(np.float32)
    dur = len(wav) / 16_000
    print(f"입력: {args.input} · {dur:.1f}초")
    if args.expect:
        print(f"정답: {args.expect}")

    prompt = build_initial_prompt()
    print(f"\ninitial_prompt ({len(prompt)}자):")
    print(f"  {prompt[:150]}{'...' if len(prompt) > 150 else ''}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    beams = [int(b) for b in args.beams.split(",") if b.strip()]
    prompts = [prompt] if args.no_prompt_compare else ["", prompt]

    print()
    header = f"{'모델':<8}{'beam':>5}{'프롬프트':>9}{'지연':>8}{'실시간배수':>11}"
    if args.expect:
        header += f"{'CER':>8}"
    print(header + "   전사")
    print("─" * (len(header) + 40))

    best = None
    for model_size in models:
        for beam in beams:
            for pr in prompts:
                stt = SpeechToText(model_size=model_size, beam_size=beam,
                                   initial_prompt=pr)
                stt.transcribe_text(np.zeros(16_000, dtype=np.float32))  # 예열

                t0 = time.perf_counter()
                text = stt.transcribe_text(wav)
                elapsed = time.perf_counter() - t0

                row = (f"{model_size:<8}{beam:>5}{'있음' if pr else '없음':>9}"
                       f"{elapsed:>7.2f}s{elapsed / dur:>10.2f}x")
                if args.expect:
                    e = cer(args.expect, text)
                    row += f"{e * 100:>7.1f}%"
                    if best is None or e < best[0]:
                        best = (e, model_size, beam, bool(pr), elapsed)
                print(f"{row}   {text}")

    print()
    if best:
        e, m, b, pr, el = best
        print(f"가장 정확: {m} · beam {b} · 프롬프트 {'있음' if pr else '없음'} "
              f"→ CER {e * 100:.1f}% · {el:.2f}초")
        print()
        print("서버에 반영하려면:")
        print(f"  python ws_server.py --whisper {m}")
        if el > 1.0:
            print(f"  ⚠ 전사에 {el:.1f}초가 걸린다. 개입 지연 목표(1.5초)를 넘길 수 있다.")
    else:
        print("--expect 로 정답 문장을 주면 오류율까지 비교한다.")

    print("\n※ 한 문장으로 판단하지 말 것. 서로 다른 발화 5~10개로 재야 의미가 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
