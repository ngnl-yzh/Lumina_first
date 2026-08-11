"""오프라인 보호 파일 생성 — D1~D2 리스크 스파이크용 CLI.

계획서(D07 §04)가 못박은 순서를 지키기 위한 도구다.
  ① 오프라인으로 보호 파일 하나 만들어 넣어보고
  ② 복제 실패를 확인한 다음
  ③ 앱을 만든다.
이 스크립트가 ①이다. ②는 clone_test.py가 맡는다.

사용:
    python protect.py 입력.wav -o out/
    python protect.py 입력.wav -o out/ --steps 200 --ratio 3.0 --snr 20
    python protect.py --synth -o out/          # 음성 파일 없이 배관만 확인
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
from mirinae.controls import CONTROL_DESCRIPTIONS, make_controls
from mirinae.encoder import SpeakerEncoder, build_ensemble
from mirinae.pipeline import protect_utterance


def load_wav(path: Path) -> torch.Tensor:
    """16 kHz 모노로 읽어들인다."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"  리샘플 {sr} → {SAMPLE_RATE} Hz")
    peak = float(np.abs(wav).max())
    if peak > 1.0:
        wav = wav / peak
    return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))


def synth_wav(n_sec: float = 6.0) -> torch.Tensor:
    """실제 음성이 없을 때 배관을 확인하기 위한 합성 신호.

    이것으로 얻은 SRS·DSR은 **아무 의미가 없다.** 사람 목소리로 다시 재야 한다.
    """
    n = int(SAMPLE_RATE * n_sec)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    sig = torch.zeros(n)
    for k, amp in enumerate([1.0, 0.6, 0.35, 0.2, 0.1], start=1):
        sig = sig + amp * torch.sin(2 * math.pi * 130.0 * k * t)
    sig = sig * (0.5 + 0.5 * torch.sin(2 * math.pi * 3.0 * t))
    sig = sig / sig.abs().max() * 0.5
    sig[: n // 10] = 0.0
    return sig


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · 오프라인 보호 파일 생성")
    p.add_argument("input", nargs="?", help="입력 WAV 경로")
    p.add_argument("--synth", action="store_true", help="합성 신호로 배관만 확인")
    p.add_argument("--seconds", type=float, default=6.0,
                   help="처리 길이(초). 입력 파일도 앞에서 이만큼만 잘라 쓴다. "
                        "0이면 파일 전체")
    p.add_argument("-o", "--out", default="out", help="출력 폴더")
    p.add_argument("--steps", type=int, default=PGDConfig.steps)
    p.add_argument("--ratio", type=float, default=PGDConfig.masking_ratio,
                   help="마스킹 배율 — 청취 평가로 확정할 값")
    p.add_argument("--snr", type=float, default=PGDConfig.target_snr_db)
    p.add_argument("--enforce-masking", action="store_true",
                   help="불변식을 전역 축소로 강제한다 (안 들림 보장 · 방어 약화)")
    p.add_argument("--no-controls", action="store_true")
    args = p.parse_args()

    if not args.input and not args.synth:
        p.error("입력 WAV를 주거나 --synth를 쓸 것")

    device = default_device()
    print(f"장치: {device}")
    if device.type == "cpu":
        print("  ※ CUDA가 없어 CPU로 돈다. 200스텝×2초 청크가 수십 초 걸린다.")

    if args.synth:
        x = synth_wav(args.seconds)
        print(f"입력: 합성 신호 {args.seconds:.1f}초  ※ 여기서 나온 SRS는 참고값이 아니다")
    else:
        x = load_wav(Path(args.input))
        full = x.shape[-1] / SAMPLE_RATE
        if args.seconds and full > args.seconds:
            x = x[: int(SAMPLE_RATE * args.seconds)]
            print(f"입력: {args.input} · {full:.1f}초 → 앞 {args.seconds:.1f}초만 사용")
        else:
            print(f"입력: {args.input} · {full:.1f}초")
    x = x.to(device)

    cfg = PGDConfig(
        steps=args.steps,
        masking_ratio=args.ratio,
        target_snr_db=args.snr,
        enforce_masking=args.enforce_masking,
    )
    print(f"설정: {cfg.steps}스텝 · 마스킹 배율 {cfg.masking_ratio} · "
          f"목표 SNR {cfg.target_snr_db} dB · 불변식 강제 {cfg.enforce_masking}")

    # 앙상블이 기본이다. 단독 최적화는 다른 복제 모델로 전이되지 않는다 —
    # 실측에서 Resemblyzer만 보고 최적화한 보호본이 ECAPA-TDNN에는
    # 0.8265로 남았다(판정 임계값 0.7962). 앙상블은 0.5094까지 내린다.
    encoder = build_ensemble(device=device)
    print("처리 중...")
    result = protect_utterance(x, encoder, cfg, with_controls=not args.no_controls)

    print()
    print(result.report())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, wav: torch.Tensor) -> None:
        sf.write(str(out / name), wav.detach().cpu().numpy(), SAMPLE_RATE)

    write("original.wav", result.original)
    write("protected.wav", result.protected)
    write("delta.wav", result.delta)

    if not args.no_controls:
        for name, wav in make_controls(x, result.delta, cfg.target_snr_db).items():
            write(f"control_{name}.wav", wav)

    meta = {
        "config": {
            "steps": cfg.steps,
            "masking_ratio": cfg.masking_ratio,
            "target_snr_db": cfg.target_snr_db,
            "band_hz": [cfg.band_low_hz, cfg.band_high_hz],
            "enforce_masking": cfg.enforce_masking,
            "chunk_sec": 2.0,
            "hop_sec": 1.0,
        },
        "input": "synth" if args.synth else str(args.input),
        "duration_sec": x.shape[-1] / SAMPLE_RATE,
        "global_snr_db": result.global_snr_db,
        "out_of_band_db": result.out_of_band_db,
        "audibility": {
            "max_excess_db": result.audibility.max_excess_db,
            "violation_ratio": result.audibility.violation_ratio,
            "mean_excess_db": result.audibility.mean_excess_db,
        },
        "srs": {"protected": result.srs_protected, **result.srs_controls},
        "control_descriptions": CONTROL_DESCRIPTIONS,
        "chunks": [vars(c) for c in result.chunks],
        "total_sec": result.total_sec,
    }
    (out / "report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n출력: {out.resolve()}")
    print("  protected.wav · original.wav · delta.wav · control_*.wav · report.json")
    print("\n다음 단계 — 복제 실패 확인:")
    print(f"  python clone_test.py {out}/original.wav {out}/protected.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
