"""STT 설정 튜닝 — 전사 품질과 지연을 함께 잰다.

## 왜 재야 하나

실사용 화면에서 오인식이 눈에 띄게 나왔다.

    "카드가 1시 정지되어"        ← 일시
    "해외 결치 시도"             ← 결제
    "광주분행국액센터입니다"       ← 지점 고객센터

전사 정확도가 이 모드 전체의 상한이다. 아무리 스코어러가 정교해도
"안전계좌"가 다르게 들리면 거기서 끝난다.

**추측으로 설정을 바꾸면 안 된다.** `initial_prompt`이 그 교훈이었다 —
이론상 맞아 보였는데 실측에서 지연만 2.4배가 되고 정확도는 그대로였고,
프롬프트 어휘가 없는데도 전사에 나타나는 환각까지 생겼다(그게 곧 오탐이다).

## 무엇을 재는가

`e2e_test.py`가 만든 통화 오디오에는 **정답 텍스트**가 있다(시나리오 원문).
서버와 같은 방식으로 VAD 분할 → 전사한 뒤 정답과 대조해 CER을 낸다.

CER은 자모가 아니라 **음절** 기준이다. 매처가 자모 근사매칭으로 일부를 흡수하므로
자모 CER은 실제 탐지 영향보다 낙관적으로 나온다.
함께 내는 **키워드 적중률**이 더 직접적인 지표다 — 정답에 있던 위험 표현이
전사에서도 매칭되는 비율이다. 결국 이게 탐지율을 정한다.

사용:
    python stt_tune.py --scenario R-A-01 FAM-03
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from mirinae.mode1 import load_db
from mirinae.mode1.matcher import Matcher, edit_distance, normalize
from mirinae.mode1.segmenter import StreamingVAD
from mirinae.mode1.stt import SpeechToText, build_initial_prompt

EVAL_DIR = Path(__file__).parent / "eval"
AUDIO_DIR = Path(__file__).parent / "out" / "e2e"


def load_scenarios() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(EVAL_DIR.glob("scenarios*.json")):
        for s in json.loads(f.read_text(encoding="utf-8"))["scenarios"]:
            out[s["id"]] = s
    return out


def keyword_forms(db) -> list[str]:
    forms: list[str] = []
    for stage in db.stages.values():
        for kw in stage.keywords:
            forms.extend(kw.all_forms())
    for c in db.criticals:
        forms.extend(c.all_forms())
    return forms


def segment(audio: np.ndarray) -> list[np.ndarray]:
    """서버와 같은 VAD로 자른다. 전사 품질은 조각 길이에 크게 좌우된다."""
    vad = StreamingVAD()
    out = [u.audio for u in vad.push(audio)]
    tail = vad.finish()
    if tail:
        out.append(tail.audio)
    return out


def measure(stt: SpeechToText, chunks: list[np.ndarray], truth: str,
            forms: list[str], matcher: Matcher) -> dict:
    t0 = time.perf_counter()
    parts = [stt.transcribe_text(c) for c in chunks]
    dt = time.perf_counter() - t0
    hyp = " ".join(p for p in parts if p)

    ref_n, hyp_n = normalize(truth), normalize(hyp)
    cer = edit_distance(ref_n, hyp_n) / max(len(ref_n), 1)

    # 정답에 실제로 있던 위험 표현이 전사에서도 잡히는가 — 탐지율을 직접 정하는 값
    in_truth = [f for f in forms if matcher.match(truth, f)]
    kept = [f for f in in_truth if matcher.match(hyp, f)]
    hit = len(kept) / len(in_truth) if in_truth else float("nan")

    # **환각 키워드** — 정답에 없는데 전사에 생긴 위험 표현.
    # 프롬프트를 켤 때 가장 무서운 부작용이다. 프롬프트 어휘가 곧 위험 키워드라서
    # 환각이 그대로 **오탐**이 된다. 정상 통화에서 이게 나오면 앱을 못 쓴다.
    ghosts = [f for f in forms if matcher.match(hyp, f) and not matcher.match(truth, f)]

    return {"cer": cer, "sec": dt, "n_chunks": len(chunks),
            "keywords_in_truth": len(in_truth), "keywords_kept": len(kept),
            "keyword_hit": hit, "ghost_keywords": sorted(set(ghosts)),
            "n_ghost": len(set(ghosts)), "hyp": hyp}


def main() -> int:
    ap = argparse.ArgumentParser(description="STT 설정 튜닝")
    ap.add_argument("--scenario", nargs="*", default=["R-A-01", "R-C-01", "FAM-03"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--telephone", action="store_true",
                    help="통화 채널을 통과시켜 열화된 조건에서 잰다")
    ap.add_argument("--noise-db", type=float, default=0.0,
                    help="배경 잡음 SNR(dB). 0이면 넣지 않는다. 실제 통화는 15~25 dB")
    ap.add_argument("-o", "--out", default="out/stt_tune.json")
    args = ap.parse_args()

    db = load_db()
    forms = keyword_forms(db)
    matcher = Matcher()
    scenarios = load_scenarios()

    # 오디오 준비 — e2e_test.py가 만든 것을 재사용한다
    cases = []
    for sid in args.scenario:
        wav = AUDIO_DIR / f"{sid}.wav"
        if not wav.exists():
            print(f"{wav} 없음 — 먼저 `python e2e_test.py --scenario {sid}` 를 돌릴 것")
            continue
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)

        # 합성 음성은 실제 마이크 입력보다 훨씬 깨끗하다. 그대로 재면 낙관적인 값이 나온다.
        if args.noise_db > 0:
            rng = np.random.default_rng(0)
            n = rng.standard_normal(len(audio)).astype(np.float32)
            n *= np.sqrt((audio ** 2).mean()) / (10 ** (args.noise_db / 20)) / \
                max(float(np.sqrt((n ** 2).mean())), 1e-12)
            audio = (audio + n).astype(np.float32)
        if args.telephone:
            import torch
            from mirinae.codec import CHANNELS, telephone_channel
            audio = telephone_channel(torch.as_tensor(audio),
                                      CHANNELS["ulaw"]).numpy()

        truth = " ".join(u["text"] for u in scenarios[sid]["utterances"])
        cases.append((sid, segment(audio), truth))

    if not cases:
        return 1

    prompt = build_initial_prompt(db)
    hot = ", ".join(f for f in (c.text for c in db.criticals))

    configs: list[tuple[str, dict]] = [
        ("기본 (현재)",            {}),
        ("beam 1",                {"beam_size": 1}),
        ("beam 5",                {"beam_size": 5}),
        ("문맥 유지",              {"condition_on_previous_text": True}),
        ("vad_filter",            {"vad_filter": True}),
        ("hotwords(critical)",    {"hotwords": hot}),
        ("initial_prompt",        {"initial_prompt": prompt}),
        ("타임스탬프 없음",         {"without_timestamps": True}),
    ]

    print(f"모델 {args.model} · 시나리오 {[c[0] for c in cases]}")
    print(f"조각 수 {[len(c[1]) for c in cases]}\n")
    print(f"{'설정':<22}{'CER':>8}{'키워드 적중':>12}{'환각 키워드':>12}{'초/통화':>10}")
    print("─" * 66)

    rows = []
    for label, over in configs:
        stt = SpeechToText(model_size=args.model)
        # transcribe()에 그대로 넘길 추가 인자
        base_transcribe = stt.transcribe

        def patched(wav, sample_rate=16_000, _o=over, _s=stt):
            segments, _ = _s.model.transcribe(
                wav.astype(np.float32),
                **{"language": _s.language, "beam_size": _s.beam_size,
                   "vad_filter": False, "condition_on_previous_text": False,
                   "initial_prompt": _s.initial_prompt or None, **_o},
            )
            from mirinae.mode1.stt import Transcript
            return [Transcript(text=s.text.strip(), start=s.start, end=s.end,
                               avg_logprob=getattr(s, "avg_logprob", 0.0),
                               no_speech_prob=getattr(s, "no_speech_prob", 0.0))
                    for s in segments]

        stt.transcribe = patched  # type: ignore[method-assign]

        per = [measure(stt, chunks, truth, forms, matcher) for _, chunks, truth in cases]
        cer = statistics.fmean(r["cer"] for r in per)
        hits = [r["keyword_hit"] for r in per if r["keyword_hit"] == r["keyword_hit"]]
        hit = statistics.fmean(hits) if hits else float("nan")
        sec = statistics.fmean(r["sec"] for r in per)
        ghosts = sum(r["n_ghost"] for r in per)
        print(f"{label:<22}{cer * 100:>7.1f}%{hit * 100:>11.0f}%"
              f"{ghosts:>11}개{sec:>10.1f}")
        rows.append({"config": label, "override": over, "cer": cer,
                     "keyword_hit": hit, "n_ghost": ghosts, "sec_per_call": sec,
                     "per_scenario": [{"id": c[0], **r} for c, r in zip(cases, per)]})
        del stt, base_transcribe

    best = min(rows, key=lambda r: (r["cer"], -r["keyword_hit"]))
    print(f"\nCER 최저: {best['config']} ({best['cer'] * 100:.1f}%)")
    bh = max(rows, key=lambda r: (r["keyword_hit"], -r["cer"]))
    print(f"키워드 적중 최고: {bh['config']} ({bh['keyword_hit'] * 100:.0f}%)")
    print("\n키워드 적중이 CER보다 중요하다 — 탐지율을 직접 정하는 것은 이쪽이다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
