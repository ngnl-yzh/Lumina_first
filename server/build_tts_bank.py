"""경고 음성 뱅크 생성 — 시연 안정성을 위한 오프라인 사전 생성.

## 왜 필요한가

앱은 지금 브라우저 `SpeechSynthesis`로 경고를 읽는다. 동작하지만 위험이 셋 있다.

**① 한국어 음성이 없는 기기에서 영어 음성이 한글을 읽는다.**
`tts.ts`의 폴백이 `voices.find(ko) ?? voices[0]`이라, ko 음성이 없으면 영어 엔진이
"안전계좌는 존재하지 않습니다"를 읽는다. 고령층에게는 무음보다 나쁘다 —
소리는 나는데 알아들을 수 없으니 "뭔가 잘못됐다"는 인상만 남는다.

**② 음성 목록 로딩이 비동기다.** Chrome은 `getVoices()`가 처음에 빈 배열을 준다.
경고가 뜨는 순간이 하필 그때면 음성이 안 나온다.

**③ 발음을 미리 검수할 수 없다.** "안전계좌"를 엔진이 어떻게 읽는지는 그 자리에서 알게 된다.

D08 §5.2가 사전 생성 뱅크를 지정한 이유가 이것이다.
런타임 지연이 0이고, 발음을 미리 들어보고 고칠 수 있다.

## 무엇을 만드는가

`warning.tts_bank_manifest()`가 낸 조각 목록을 그대로 합성한다.
조각은 **고정 프레임 + 키워드**로 나뉘어 있어서, 런타임에는 이어붙이기만 하면 된다.

생성물은 `tts_bank/`에 들어가고 **git에 넣지 않는다**(수백 개 WAV).
`manifest.json`만 커밋하므로 오디오 없이도 완결성을 검증할 수 있고,
시연 장비에서는 이 스크립트를 한 번 돌리면 된다 — 모델 캐시와 같은 취급이다(BUILD.md).

## 합성 엔진

Windows 내장 SAPI(`System.Speech`)를 쓴다. 네트워크도 추가 설치도 필요 없다.
한국어 음성(예: Microsoft Heami)이 설치돼 있어야 한다.

    python build_tts_bank.py --list-voices      # 설치된 음성 확인
    python build_tts_bank.py                    # 뱅크 생성
    python build_tts_bank.py --verify           # 생성 없이 완결성만 점검
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirinae.mode1 import load_db
from mirinae.mode1.warning import tts_bank_manifest

BANK_DIR = Path(__file__).parent / "tts_bank"
MANIFEST = BANK_DIR / "manifest.json"

# 경고 음성은 평상보다 느려야 한다 (D08 §5.2 — 인지 처리 시간 확보).
# SAPI Rate는 -10~10이고 0이 보통 속도다.
SPEECH_RATE = -2


def build_index(manifest: dict[str, str]) -> list[dict]:
    """토큰 → 파일명. 토큰에 `::`와 한글이 섞여 있어 파일명으로 못 쓴다.

    번호를 붙이고 매핑을 manifest.json에 남긴다. 파일명 인코딩 문제를 원천 차단한다.
    """
    return [
        {"token": tok, "text": text, "file": f"frag{i:04d}.wav"}
        for i, (tok, text) in enumerate(sorted(manifest.items()))
    ]


PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {rate}

$wanted = '{voice}'
if ($wanted -ne '') {{
    $synth.SelectVoice($wanted)
}} else {{
    $ko = $synth.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'ko*' }}
    if (-not $ko) {{ Write-Error 'ko-KR 음성이 설치돼 있지 않다'; exit 2 }}
    $synth.SelectVoice($ko[0].VoiceInfo.Name)
}}
Write-Output ("voice=" + $synth.Voice.Name)

$index = Get-Content -Raw -Encoding UTF8 '{index}' | ConvertFrom-Json
$n = 0
foreach ($row in $index) {{
    $path = Join-Path '{outdir}' $row.file
    $synth.SetOutputToWaveFile($path)
    $synth.Speak($row.text)
    $n++
}}
$synth.SetOutputToNull()
$synth.Dispose()
Write-Output ("done=" + $n)
"""


def synthesize(index: list[dict], outdir: Path, voice: str = "") -> bool:
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    script = PS_SCRIPT.format(
        rate=SPEECH_RATE, voice=voice,
        index=str(index_path).replace("\\", "\\\\"),
        outdir=str(outdir).replace("\\", "\\\\"),
    )
    script_path = outdir / "_build.ps1"
    script_path.write_text(script, encoding="utf-8")

    print(f"합성 시작 — 조각 {len(index)}개")
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(script_path)],
        capture_output=True, text=True,
    )
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            print(f"  {line.strip()}")
    if proc.returncode != 0:
        print(f"합성 실패 (exit {proc.returncode})", file=sys.stderr)
        print((proc.stderr or "")[:800], file=sys.stderr)
        return False

    script_path.unlink(missing_ok=True)
    index_path.unlink(missing_ok=True)
    return True


def verify(index: list[dict], outdir: Path) -> tuple[list[str], list[str]]:
    """빠진 조각과 빈 파일을 찾는다.

    **빈 파일이 더 위험하다.** 없으면 폴백이 돌지만, 0바이트 WAV는
    "있다"고 판정되고 재생만 안 된다 — 정작 경고가 필요한 순간에 조용해진다.
    """
    missing, empty = [], []
    for row in index:
        p = outdir / row["file"]
        if not p.exists():
            missing.append(row["token"])
        elif p.stat().st_size < 1024:          # WAV 헤더만 있는 수준
            empty.append(row["token"])
    return missing, empty


def main() -> int:
    ap = argparse.ArgumentParser(description="경고 음성 뱅크 생성")
    ap.add_argument("--out", type=Path, default=BANK_DIR)
    ap.add_argument("--voice", default="", help="SAPI 음성 이름 (미지정 시 첫 ko 음성)")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--verify", action="store_true", help="생성 없이 점검만")
    args = ap.parse_args()

    if args.list_voices:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
             ".GetInstalledVoices() | ForEach-Object { "
             "'{0} | {1}' -f $_.VoiceInfo.Name, $_.VoiceInfo.Culture }"],
            check=False)
        return 0

    db = load_db()
    manifest = tts_bank_manifest(db)
    index = build_index(manifest)
    print(f"매니페스트 조각 {len(index)}개 (DB 항목 {db.n_total}개 기준)")

    if not args.verify:
        if not synthesize(index, args.out, args.voice):
            return 1

    missing, empty = verify(index, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps({"rate": SPEECH_RATE, "fragments": index},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if missing or empty:
        print(f"\n미완성 — 없음 {len(missing)}개 · 빈 파일 {len(empty)}개")
        for t in (missing + empty)[:10]:
            print(f"    {t}")
        print("\n이 상태로 시연하면 해당 조각에서 소리가 안 난다.")
        return 1

    total_mb = sum((args.out / r["file"]).stat().st_size for r in index) / 1e6
    print(f"\n완성 — 조각 {len(index)}개 · {total_mb:.1f} MB · {args.out}")
    print("manifest.json은 커밋하고 WAV는 커밋하지 않는다 (BUILD.md의 모델 캐시와 같은 취급)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
