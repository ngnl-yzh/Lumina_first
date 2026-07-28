"""환경 점검 — 새 장비에 옮긴 뒤 가장 먼저 돌릴 것.

RTX 3060 장비로 옮기면 확인해야 할 것이 두 가지다.

  ① 설치가 제대로 됐는가 (CUDA 빌드 torch인가, 패키지가 다 있는가)
  ② **PGD가 실제로 얼마나 빠른가**

②가 중요하다. 계획서(D09)의 "2초 청크 200스텝 = 0.36초"는 실측이 아니라
FLOP 모델로 계산한 추정치다. 각주에도 "실측값이 다르면 청크 길이와 스텝 수를 재조정한다"고
적혀 있다. 이 스크립트가 그 실측을 대신 해준다.

기준은 단순하다 — **hop 1.0초마다 청크가 하나씩 생기므로,
청크 하나를 1.0초 안에 처리하지 못하면 큐가 무한히 쌓인다.**

사용:
    python check_env.py
    python check_env.py --steps 200 --chunk 2.0
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def ok(label: str, good: bool, detail: str = "") -> bool:
    mark = "✓" if good else "✕"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    return good


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · 환경 점검")
    p.add_argument("--steps", type=int, default=200, help="벤치마크 PGD 스텝 수")
    p.add_argument("--chunk", type=float, default=2.0, help="청크 길이(초)")
    p.add_argument("--hop", type=float, default=1.0, help="홉(초) — 이 안에 처리해야 한다")
    p.add_argument("--skip-bench", action="store_true")
    args = p.parse_args()

    print("=" * 62)
    print("미리내 · 환경 점검")
    print("=" * 62)
    print(f"  {platform.platform()}")
    print(f"  Python {sys.version.split()[0]}")

    # ── 1. 패키지 ─────────────────────────────────────────────────────────────
    print("\n[1] 패키지")
    try:
        import torch
    except ImportError:
        print("  ✕ torch 없음 — pip install -r requirements.txt")
        return 1

    print(f"  ✓ torch {torch.__version__}")
    for name in ("numpy", "scipy", "soundfile", "resemblyzer", "librosa"):
        try:
            __import__(name)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✕ {name} — {e}")
            return 1

    has_whisper = True
    try:
        import faster_whisper  # noqa: F401
        print("  ✓ faster-whisper (모드 1 STT)")
    except ImportError:
        has_whisper = False
        print("  ✕ faster-whisper — 모드 1이 안 된다. pip install faster-whisper")

    try:
        import websockets  # noqa: F401
        print("  ✓ websockets (서버)")
    except ImportError:
        print("  ✕ websockets — 서버가 안 뜬다. pip install websockets")

    # ── 2. GPU ────────────────────────────────────────────────────────────────
    print("\n[2] GPU")
    cuda = torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ CUDA 사용 가능 — {name} · {vram:.1f} GB")
        print(f"    torch CUDA {torch.version.cuda}")
    else:
        print("  ✕ CUDA 사용 불가")
        if "+cpu" in torch.__version__:
            print("    → torch가 CPU 빌드다. 이게 가장 흔한 원인이다.")
            print("      pip uninstall torch")
            print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("    → NVIDIA 드라이버를 확인할 것 (명령 프롬프트에서 nvidia-smi)")
        print("    모드 1은 CPU로도 동작한다. 모드 2 실시간은 불가하다.")

    # ── 3. 패턴 DB ────────────────────────────────────────────────────────────
    print("\n[3] 패턴 DB")
    try:
        from mirinae.mode1 import load_db

        db = load_db()
        ok("항목 수", db.n_total == 182,
           f"기본 {db.n_base} + 변형 {db.n_variants} = {db.n_total} (설계도 182)")
        ok("critical", len(db.criticals) == 5, f"{len(db.criticals)}개")
        ok("pair", len(db.pairs) == 3, f"{len(db.pairs)}개")
    except Exception as e:
        print(f"  ✕ 로드 실패 — {e}")
        return 1

    # ── 4. 인코더 ─────────────────────────────────────────────────────────────
    print("\n[4] 화자 인코더")
    from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
    from mirinae.encoder import SpeakerEncoder

    device = default_device()
    t0 = time.perf_counter()
    encoder = SpeakerEncoder(device=device)
    print(f"  ✓ Resemblyzer 로드 {time.perf_counter() - t0:.1f}초 · 장치 {device}")

    # ── 5. PGD 실측 ───────────────────────────────────────────────────────────
    if args.skip_bench:
        print("\n[5] PGD 벤치마크 — 건너뜀")
        return 0

    print(f"\n[5] PGD 실측 — {args.chunk}초 청크 · {args.steps}스텝")
    print("  측정 중...", flush=True)

    import math

    import torch as T

    from mirinae.perturbation import pgd_perturbation
    from mirinae.psychoacoustic import MaskingModel

    n = int(SAMPLE_RATE * args.chunk)
    t = T.arange(n, dtype=T.float32) / SAMPLE_RATE
    x = sum(T.sin(2 * math.pi * 130 * k * t) / k for k in range(1, 6))
    x = (x / x.abs().max() * 0.4).to(device)

    masking = MaskingModel(device=device)

    # 첫 실행은 커널 컴파일·메모리 할당이 섞여 느리다. 버린다.
    pgd_perturbation(x, encoder, PGDConfig(steps=3), masking_model=masking)
    if cuda:
        T.cuda.synchronize()

    t0 = time.perf_counter()
    pgd_perturbation(x, encoder, PGDConfig(steps=args.steps), masking_model=masking)
    if cuda:
        T.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ratio = elapsed / args.hop
    print(f"\n  실측: {elapsed:.2f}초 ({elapsed / args.steps * 1000:.1f} ms/스텝)")
    print(f"  홉 {args.hop}초 대비 {ratio:.2f}배")
    print(f"  계획서(D09) 상정치: 0.36초 · 여유율 64%")

    print()
    if ratio <= 0.7:
        print(f"  ✓ 실시간 가능. 여유율 {(1 - ratio) * 100:.0f}%")
        print("    모드 2를 그대로 돌려도 청크가 밀리지 않는다.")
    elif ratio <= 1.0:
        print(f"  ⚠ 실시간 가능하지만 여유가 없다 ({(1 - ratio) * 100:.0f}%).")
        print("    발화가 길어지거나 다른 부하가 겹치면 밀린다.")
        safe = int(args.steps * 0.7 / ratio)
        print(f"    스텝을 {safe} 정도로 낮추면 여유율 30%가 된다.")
    else:
        print(f"  ✕ 실시간 불가. 청크 하나 처리에 홉의 {ratio:.1f}배가 걸린다.")
        safe = max(20, int(args.steps * 0.7 / ratio))
        print(f"    → 스텝을 {args.steps} → {safe} 로 낮추거나,")
        print(f"    → 청크 길이를 늘려 홉을 키워야 한다.")
        print("    (서버의 worker.py가 적체 시 자동 감축하지만, 그건 방어가 약해진다는 뜻이다)")

    print("\n  ※ 스텝을 바꾸면 방어 강도가 달라진다. 절제 실험(sweep_params.py)으로")
    print("    이 장비에서의 스텝-강도 곡선을 다시 그릴 것.")

    if not has_whisper:
        print("\n  ※ faster-whisper가 없어 모드 1은 아직 못 돌린다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
