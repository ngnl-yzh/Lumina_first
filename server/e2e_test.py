"""종단 검증 — 오디오부터 개입까지 실제 경로를 통째로 돌린다.

## 왜 필요한가

`eval_mode1.py`가 재는 것은 **텍스트 단계**다. 시나리오 문장을 스코어러에 직접 넣는다.
그런데 실제 경로에는 그 앞에 두 단계가 더 있다.

    오디오 → [VAD 분할] → [Whisper 전사] → 채점 → 개입

탐지율 81%는 **전사가 완벽하다고 가정한** 값이다. 실제로는 Whisper가 무엇을 내놓느냐에
따라 달라지고, VAD가 발화를 어떻게 자르느냐에 따라 키워드가 경계에 걸릴 수 있다.
`--stt-noise`가 오차를 흉내내긴 하지만 그건 모델이지 실측이 아니다.

이 도구는 시나리오를 **소리로 만들어** WebSocket 서버에 흘려보내고,
서버가 실제로 무엇을 전사하고 어떻게 판정하는지 받아 적는다.

## 한계 — 먼저 밝힌다

합성 음성(Windows SAPI)이라 **실제 통화보다 깨끗하다.**
잡음도 없고 발음도 또박또박하다. 여기서 나온 전사 정확도는 **상한**이다.
전화 통화의 대역 제한·코덱·배경 잡음은 `--telephone`으로 근사할 수 있다.

목적은 "이만큼 잘 된다"가 아니라 **"전 경로가 실제로 이어져 있는가,
텍스트 단계와 결과가 얼마나 다른가"**를 보는 것이다.

사용:
    python e2e_test.py --scenario R-A-01          # 실제 공개 녹취 기반 사기
    python e2e_test.py --scenario FAM-03          # 가족 금융 대화 (잡히면 안 된다)
    python e2e_test.py --all --telephone          # 전체 · 통화 채널 통과
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirinae.config import SAMPLE_RATE

EVAL_DIR = Path(__file__).parent / "eval"
AUDIO_DIR = Path(__file__).parent / "out" / "e2e"

# 발화 사이 무음. `segmenter.SILENCE_MS`가 500 ms를 문장 경계로 보므로 넉넉히 준다.
GAP_SEC = 0.9

# 스트리밍 청크. 실제 AudioWorklet은 128 프레임 단위로 올라오지만
# 여기서는 전송 횟수를 줄이려고 0.1초씩 보낸다 — VAD는 어느 쪽이든 같게 동작한다.
CHUNK_SEC = 0.1


def load_scenarios() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(EVAL_DIR.glob("scenarios*.json")):
        for s in json.loads(f.read_text(encoding="utf-8"))["scenarios"]:
            out[s["id"]] = s
    return out


# ── 합성 ──────────────────────────────────────────────────────────────────────

PS_SPEAK = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ko = $synth.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'ko*' }}
if (-not $ko) {{ Write-Error 'ko-KR 음성이 없다'; exit 2 }}
$synth.SelectVoice($ko[0].VoiceInfo.Name)
$rows = Get-Content -Raw -Encoding UTF8 '{index}' | ConvertFrom-Json
foreach ($r in $rows) {{
    $synth.SetOutputToWaveFile($r.file)
    $synth.Speak($r.text)
}}
$synth.SetOutputToNull(); $synth.Dispose()
"""


def synthesize_call(scenario: dict, outdir: Path) -> Path:
    """시나리오 발화들을 소리로 만들어 한 통화로 잇는다."""
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{scenario['id']}.wav"
    if final.exists():
        return final

    parts = outdir / "_parts"
    parts.mkdir(exist_ok=True)
    rows = [{"file": str(parts / f"u{i:02d}.wav"), "text": u["text"]}
            for i, u in enumerate(scenario["utterances"])]
    idx = parts / "_idx.json"
    idx.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    script = parts / "_speak.ps1"
    script.write_text(PS_SPEAK.format(index=str(idx).replace("\\", "\\\\")),
                      encoding="utf-8")
    proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(script)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"합성 실패: {(proc.stderr or '')[:300]}")

    gap = np.zeros(int(SAMPLE_RATE * GAP_SEC), dtype=np.float32)
    chunks: list[np.ndarray] = [gap]
    for r in rows:
        data, sr = sf.read(r["file"], dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy import signal as sps
            data = sps.resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
        chunks += [data, gap]

    call = np.concatenate(chunks)
    peak = float(np.abs(call).max()) or 1.0
    sf.write(final, (call / peak * 0.7).astype(np.float32), SAMPLE_RATE)

    for r in rows:
        Path(r["file"]).unlink(missing_ok=True)
    idx.unlink(missing_ok=True)
    script.unlink(missing_ok=True)
    return final


# ── 스트리밍 ──────────────────────────────────────────────────────────────────

async def stream(path: Path, url: str, telephone: bool) -> dict:
    import websockets

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SAMPLE_RATE, f"{sr} Hz — 16 kHz가 아니다"

    if telephone:
        # 실제 통화에 가깝게 만든다. 합성 음성은 너무 깨끗하다.
        import torch
        from mirinae.codec import CHANNELS, telephone_channel
        audio = telephone_channel(torch.as_tensor(audio),
                                  CHANNELS["ulaw"]).numpy()

    got: dict = {"utterances": [], "warning": None}
    step = int(SAMPLE_RATE * CHUNK_SEC)

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"mode": "mode1"}))
        ready = json.loads(await ws.recv())
        assert ready.get("type") == "ready", ready

        async def receive():
            try:
                async for m in ws:
                    if not isinstance(m, str):
                        continue
                    msg = json.loads(m)
                    if msg.get("type") == "utterance":
                        got["utterances"].append(msg)
                        lvl = msg["level"]
                        print(f"  [{lvl:<2}] {msg['score']:.3f}  “{msg['text']}”")
                        if msg.get("matched"):
                            hits = " ".join(f"{k}:{','.join(v[:2])}"
                                            for k, v in sorted(msg["matched"].items()))
                            print(f"        매칭 {hits}")
                    elif msg.get("type") == "warning":
                        got["warning"] = msg
                        print(f"  ★ 개입 — {msg['quote']}")
            except Exception:
                pass

        task = asyncio.create_task(receive())
        for i in range(0, len(audio), step):
            await ws.send(audio[i:i + step].astype("<f4").tobytes())
            await asyncio.sleep(0)          # 이벤트 루프에 수신 기회를 준다
        # 전사·채점이 끝날 시간을 준다. STT가 CPU에서 발화당 수 초 걸린다.
        await asyncio.sleep(max(8.0, len(audio) / SAMPLE_RATE))
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(1.0)
        task.cancel()

    return got


async def run(ids: list[str], url: str, telephone: bool) -> int:
    scenarios = load_scenarios()
    rows = []
    for sid in ids:
        s = scenarios.get(sid)
        if not s:
            print(f"{sid} — 시나리오 없음")
            continue

        print(f"\n{'=' * 74}")
        print(f"[{sid}] {s.get('title', '')}   기대: "
              f"{'탐지' if s['label'] == 'fraud' else '무경고'}")
        print("=" * 74)
        wav = synthesize_call(s, AUDIO_DIR)
        dur = sf.info(wav).duration
        print(f"  오디오 {dur:.1f}초 · 발화 {len(s['utterances'])}개"
              + (" · 통화 채널 통과" if telephone else ""))

        got = await stream(wav, url, telephone)
        n = len(got["utterances"])
        best = max((u["score"] for u in got["utterances"]), default=0.0)
        fired = got["warning"] is not None
        ok = fired if s["label"] == "fraud" else not fired

        print(f"  → 전사 {n}개 · 최고 위험도 {best:.3f} · 개입 {'예' if fired else '아니오'}"
              f"   {'OK' if ok else '**실패**'}")
        rows.append({"id": sid, "label": s["label"], "n_utterances": n,
                     "expected": len(s["utterances"]), "max_score": best,
                     "intervened": fired, "correct": ok,
                     "transcripts": [u["text"] for u in got["utterances"]]})

    print(f"\n{'=' * 74}\n종단 결과\n{'=' * 74}")
    for r in rows:
        print(f"  {'OK' if r['correct'] else 'X '} {r['id']:<8} "
              f"전사 {r['n_utterances']}/{r['expected']}  "
              f"최고 {r['max_score']:.3f}  개입 {'예' if r['intervened'] else '—'}")
    bad = [r["id"] for r in rows if not r["correct"]]
    print(f"\n  {len(rows) - len(bad)}/{len(rows)} 통과" + (f" · 실패 {bad}" if bad else ""))

    out = Path("out/e2e_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {out}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="오디오→개입 종단 검증")
    ap.add_argument("--scenario", nargs="*", default=[])
    ap.add_argument("--all", action="store_true",
                    help="사기 2건 + 정상 2건 기본 묶음")
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--telephone", action="store_true",
                    help="통화 채널(300~3400 Hz · 8 kHz · G.711)을 통과시킨다")
    args = ap.parse_args()

    ids = args.scenario or (["R-A-01", "R-B-01", "FAM-03", "FAM-11"]
                            if args.all else ["R-A-01"])
    return asyncio.run(run(ids, args.url, args.telephone))


if __name__ == "__main__":
    raise SystemExit(main())
