"""복제 실패 검증 — D1~D2 리스크 스파이크 게이트.

계획서(D07 §04)가 UI보다 먼저 하라고 못박은 검증이다.
**보호본을 XTTS-v2에 넣었는데 복제가 성공하면 시연 영상의 클라이맥스가 사라진다.**
XTTS는 자체 인코더를 쓰므로 Resemblyzer에서 만든 섭동이 전이되지 않을 수 있다.

측정 구조 — 같은 문장을 두 번 복제해 나란히 놓는다.

    원본  → XTTS → 복제A     원본과 얼마나 닮았나 (복제가 되는가)
    보호본 → XTTS → 복제B     원본과 얼마나 닮았나 (복제가 실패하는가)

복제A는 높고 복제B는 낮아야 방어가 작동한 것이다.
복제A까지 낮으면 그건 방어가 아니라 **XTTS가 애초에 이 목소리를 복제하지 못한 것**이므로
아무것도 주장할 수 없다. 그래서 두 값을 항상 함께 본다.

사용:
    python clone_test.py out/original.wav out/protected.wav
    python clone_test.py out/original.wav out/protected.wav --controls out/
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from mirinae.config import SAMPLE_RATE, default_device
from mirinae.encoder import SpeakerEncoder, cosine_similarity

XTTS_PYTHON = Path(__file__).parent / ".venv-xtts" / "Scripts" / "python.exe"
XTTS_SCRIPT = Path(__file__).parent / "clone_xtts.py"

# 복제 모델 — 각각 격리 venv에서 돈다.
#
# **모델을 하나만 쓰면 결론의 일반성을 알 수 없다.**
# XTTS에서 "보호본도 복제된다"가 나왔는데, 그것만으로는
#   ① XTTS 특유의 성질이라 다른 모델에는 통할지도 모른다
#   ② 임베딩 표적 방어 자체의 구조적 한계다
# 를 구별할 수 없다.
#
# F5-TTS는 XTTS와 구조가 완전히 다르다 — XTTS는 GPT 계열 자기회귀 + HiFi-GAN,
# F5-TTS는 flow matching이고 화자 조건 경로도 다르다.
# 둘에서 같은 결과가 나오면 ②가 지지된다.
CLONERS: dict[str, dict] = {
    "xtts": {
        "venv": Path(__file__).parent / ".venv-xtts" / "Scripts" / "python.exe",
        "script": Path(__file__).parent / "clone_xtts.py",
        "name": "XTTS-v2 (GPT + HiFi-GAN)",
        "setup": [
            "py -3.11 -m venv .venv-xtts",
            ".venv-xtts\\Scripts\\python -m pip install -r requirements-xtts.txt",
        ],
        "lang_flag": True,
    },
    "f5": {
        "venv": Path(__file__).parent / ".venv-f5" / "Scripts" / "python.exe",
        "script": Path(__file__).parent / "clone_f5.py",
        "name": "F5-TTS (flow matching)",
        "setup": [
            "py -3.11 -m venv .venv-f5",
            ".venv-f5\\Scripts\\python -m pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121",
            ".venv-f5\\Scripts\\python -m pip install f5-tts",
        ],
        "lang_flag": False,
    },
}


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """평균과 95% 신뢰구간 반폭.

    **왜 반복이 필요한가.**
    XTTS 생성은 확률적이다. 완전히 같은 오디오를 두 번 넣어도 복제 결과가 다르다.
    실측: 원본(0.9332)과 C-C 무섭동(0.9520)은 같은 파일인데 0.019 차이가 났다.
    그런데 보호에 의한 하락은 0.027~0.047 수준이다 — **효과가 노이즈와 같은 자릿수다.**
    n=1로 재고 "하락했다"고 말하면 그건 측정이 아니라 착시다.

    D09 §06이 "n · 신뢰구간 · 대조군 대비 셋이 항상 함께 나온다"고 못박은 이유가 이것이다.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = statistics.fmean(values)
    if n == 1:
        return (m, float("nan"), float("nan"))
    sd = statistics.stdev(values)
    half = z * sd / math.sqrt(n)
    return (m, sd, half)


def run_cloner(model: str, reference: Path, output: Path,
               text: str, language: str) -> bool:
    """격리 venv에서 복제 모델을 돌린다."""
    spec = CLONERS[model]
    venv: Path = spec["venv"]
    if not venv.exists():
        print(f"{spec['name']} venv 없음: {venv}", file=sys.stderr)
        for line in spec["setup"]:
            print(f"  {line}", file=sys.stderr)
        return False

    cmd = [str(venv), str(spec["script"]), str(reference), str(output), "--text", text]
    if spec["lang_flag"]:
        cmd += ["--language", language]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"{spec['name']} 실패 (exit {proc.returncode})", file=sys.stderr)
        print((proc.stdout or "")[-2000:], file=sys.stderr)
        print((proc.stderr or "")[-2000:], file=sys.stderr)
        return False
    return True


def run_xtts(reference: Path, output: Path, text: str, language: str) -> bool:
    """하위 호환 — 예전 호출부가 남아 있을 수 있다."""
    return run_cloner("xtts", reference, output, text, language)


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · 복제 실패 검증")
    p.add_argument("original", help="원본 WAV")
    p.add_argument("protected", help="보호본 WAV")
    p.add_argument("--controls", help="대조군 WAV가 있는 폴더 (control_*.wav)")
    p.add_argument("--model", default="xtts", choices=list(CLONERS),
                   help="복제 모델. 구조가 다른 모델에서도 같은 결과가 나와야 "
                        "'임베딩 표적 방어의 한계'를 주장할 수 있다")
    p.add_argument("--text", default="엄마, 나 사고 났어. 지금 급하게 돈이 필요해.")
    p.add_argument("--language", default="ko")
    p.add_argument("-o", "--out", default=None, help="복제음 출력 폴더")
    p.add_argument("--repeat", type=int, default=1,
                   help="조건마다 몇 번 복제할지. 생성이 확률적이라 "
                        "n=1로는 효과와 노이즈를 구별할 수 없다. 최소 5 권장")
    args = p.parse_args()

    model = args.model
    spec = CLONERS[model]
    # 모델마다 다른 폴더에 넣는다. 섞이면 어느 모델 결과인지 알 수 없게 된다.
    if args.out is None:
        args.out = f"out/clone_{model}"

    from protect import load_wav

    original = Path(args.original)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 복제 대상 목록 ────────────────────────────────────────────────────────
    targets: list[tuple[str, Path]] = [
        ("원본", original),
        ("보호본", Path(args.protected)),
    ]
    if args.controls:
        for wav in sorted(Path(args.controls).glob("control_*.wav")):
            targets.append((wav.stem.replace("control_", ""), wav))

    print(f"복제 모델: {spec['name']}")
    print(f"복제 문장: “{args.text}”")
    print(f"대상 {len(targets)}개 · 조건당 {args.repeat}회 복제\n")
    if args.repeat < 3:
        print("  ※ n<3이면 생성 편차와 방어 효과를 구별할 수 없다. --repeat 5 권장\n")

    clones: dict[str, list[Path]] = {}
    for label, ref in targets:
        paths = []
        for k in range(args.repeat):
            dst = out / f"clone_from_{ref.stem}_{k}.wav"
            print(f"[{label}] {ref.name} → {model} 복제 {k + 1}/{args.repeat}...",
                  flush=True)
            if not run_cloner(model, ref, dst, args.text, args.language):
                return 1
            paths.append(dst)
        clones[label] = paths

    # ── 측정 ──────────────────────────────────────────────────────────────────
    device = default_device()
    encoder = SpeakerEncoder(device=device)
    x_orig = load_wav(original).to(device)

    samples: dict[str, list[float]] = {}
    with torch.no_grad():
        ref_embed = encoder(x_orig)
        for label, paths in clones.items():
            vals = []
            for path in paths:
                wav = load_wav(path).to(device)
                vals.append(float(cosine_similarity(encoder(wav), ref_embed)))
            samples[label] = vals

    stats = {k: mean_ci(v) for k, v in samples.items()}
    results = {k: s[0] for k, s in stats.items()}

    base = results.get("원본", 0.0)
    prot = results.get("보호본", 0.0)

    print()
    print(f"{'복제 출처':<14}{'n':>3}{'평균 SRS':>11}{'±95% CI':>11}   의미")
    print("─" * 70)
    for label in samples:
        m, sd, half = stats[label]
        ci = "—" if math.isnan(half) else f"±{half:.4f}"
        if label == "원본":
            note = "복제 성공 기준선"
        elif label == "보호본":
            note = "복제가 실패하는가 ← 핵심"
        elif label == "C-C":
            note = "원본과 동일한 오디오 = 생성 편차"
        else:
            note = "대조군"
        print(f"{label:<14}{len(samples[label]):>3}{m:>11.4f}{ci:>11}   {note}")

    print()
    drop = base - prot

    # 생성 편차의 기준 — 원본과 C-C는 같은 오디오이므로 그 차이가 곧 노이즈 바닥이다
    noise_floor = abs(base - results["C-C"]) if "C-C" in results else float("nan")
    if not math.isnan(noise_floor):
        print(f"생성 편차 (원본 vs C-C, 같은 오디오): {noise_floor:.4f}")
    print(f"보호에 의한 하락: {drop:+.4f}")

    # 효과가 노이즈보다 큰지 — 이 판정 없이 하락폭만 보고하면 착시다
    if not math.isnan(noise_floor) and drop <= noise_floor * 1.5:
        print("  ⚠ 하락폭이 생성 편차 수준이다. 이 데이터로는 방어 효과를 주장할 수 없다.")

    # ── 게이트 판정 ───────────────────────────────────────────────────────────
    print()
    if not math.isnan(noise_floor) and drop <= noise_floor * 1.5:
        verdict = "판정 불가 · 효과가 노이즈 수준"
        detail = ("하락폭이 XTTS 생성 편차와 같은 자릿수다. 방어가 없다는 뜻이 아니라 "
                  "**이 표본 수로는 있는지 없는지 알 수 없다**는 뜻이다. "
                  "--repeat를 늘려 다시 잴 것.")
    elif base < 0.60:
        verdict = "판정 불가"
        detail = ("원본조차 복제 유사도가 낮다. XTTS가 이 음성을 제대로 복제하지 못한 것이므로 "
                  "방어 효과를 분리할 수 없다. **사람 목소리로 다시 할 것.**")
    elif drop >= 0.25 and prot < 0.60:
        verdict = "게이트 통과"
        detail = "보호본에서 복제가 실패했다. 시연의 클라이맥스가 성립한다."
    elif drop >= 0.10:
        verdict = "부분 통과"
        detail = ("하락은 있으나 결정적이지 않다. 마스킹 배율 상향 또는 앙상블 확대를 검토하고, "
                  "그래도 안 되면 주장을 '임베딩 교란 확인'으로 좁힌다. (D07 리스크표 SEV1 대응)")
    else:
        verdict = "게이트 실패"
        detail = ("보호본에서도 복제가 성공했다. Resemblyzer 섭동이 XTTS 인코더로 전이되지 않는다. "
                  "계획서 대응대로 마스킹 배율 상향 → 앙상블 확대 → 그래도 안 되면 주장 축소.")
    print(f"판정: {verdict}")
    print(f"  {detail}")

    (out / "clone_report.json").write_text(json.dumps({
        "model": model,
        "model_name": spec["name"],
        "text": args.text,
        "language": args.language,
        "repeat": args.repeat,
        "srs_samples": samples,
        "srs_mean": results,
        "srs_ci_half": {k: (None if math.isnan(s[2]) else s[2]) for k, s in stats.items()},
        "generation_noise_floor": None if math.isnan(noise_floor) else noise_floor,
        "drop": drop,
        "verdict": verdict,
        "note": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out / 'clone_report.json'}")
    print("\n※ 이 판정은 SRS(Resemblyzer 임베딩) 기준이다. 시연에서는 반드시 "
          "**귀로 들어서** 다른 사람 목소리인지 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
