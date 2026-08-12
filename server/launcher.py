"""미리내 실행기 — EXE로 묶이는 진입점.

파이썬·노드·npm 설치 없이 더블클릭 한 번으로 돌아가게 하는 것이 목적이다.
한 프로세스에서 세 가지를 한다.

  ① 정적 파일 서버 (8080) — 빌드된 PC 앱을 내보낸다
  ② WebSocket 서버 (8765) — 전사·채점·섭동
  ③ 브라우저 열기

모델은 EXE에 넣지 않는다. Whisper small 460 MB + 딥보이스 380 MB를 넣으면
배포 파일이 감당이 안 되고, 어차피 처음 한 번만 받으면 캐시된다.
**첫 실행에는 인터넷이 필요하다.** 시연 전에 반드시 한 번 돌려볼 것.
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

APP_PORT = 8080
WS_PORT = 8765


def resource_dir() -> Path:
    """PyInstaller로 묶였는지에 따라 자원 위치가 다르다."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)      # type: ignore[attr-defined]
    return Path(__file__).parent


def find_web_root() -> Path | None:
    """빌드된 PC 앱(dist)을 찾는다."""
    candidates = [
        resource_dir() / "webapp",              # EXE에 함께 묶인 경우
        Path(__file__).parent.parent / "app-desktop" / "dist",   # 개발 중
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """접속 로그를 찍지 않는다. 콘솔에는 서버 상태만 보이는 편이 낫다."""

    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def end_headers(self):
        # 개발 중 고친 화면이 캐시 때문에 안 바뀌는 일을 막는다
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_static(root: Path, port: int) -> None:
    handler = lambda *a, **kw: QuietHandler(*a, directory=str(root), **kw)  # noqa: E731
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.allow_reuse_address = True
        httpd.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 실행기")
    p.add_argument("--app-port", type=int, default=APP_PORT)
    p.add_argument("--ws-port", type=int, default=WS_PORT)
    p.add_argument("--whisper", default=None, help="tiny/base/small/medium")
    p.add_argument("--steps", type=int, default=None, help="모드 2 PGD 스텝 수")
    p.add_argument("--deepvoice", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, str(resource_dir()))

    print("=" * 58)
    print("  미리내 — 보이스피싱 경고 · 딥보이스 학습 방지")
    print("  Team Lumina · 전남대학교 인공지능학부")
    print("=" * 58)

    web_root = find_web_root()
    if web_root is None:
        print("\n[오류] 앱 화면 파일을 찾지 못했다.")
        print("  개발 중이라면 먼저 빌드할 것:")
        print("    cd app-desktop && npm run build")
        input("\n엔터를 누르면 종료합니다...")
        return 1

    threading.Thread(target=serve_static, args=(web_root, args.app_port),
                     daemon=True).start()
    url = f"http://localhost:{args.app_port}"
    print(f"\n  화면  {url}")
    print(f"  서버  ws://localhost:{args.ws_port}")

    import torch

    from mirinae.config import PGDConfig
    from ws_server import amain

    if torch.cuda.is_available():
        print(f"  장치  {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        print("  장치  CPU")
        print("        ※ 모드 2 실시간 처리는 불가합니다. 모드 1은 정상 동작합니다.")

    print("\n  모델을 준비하는 중입니다. 처음 실행이면 다운로드가 있어 몇 분 걸립니다.")
    print("  이 창을 닫으면 프로그램이 종료됩니다.\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # ws_server.amain이 기대하는 형태로 인자를 맞춘다
    class Args:
        host = "127.0.0.1"
        port = args.ws_port
        whisper = args.whisper or _default_whisper()
        steps = args.steps if args.steps is not None else PGDConfig.steps
        ratio = PGDConfig.masking_ratio
        ssl_cert = None
        ssl_key = None
        no_warmup = False
        # 딥보이스 탐지는 모드 1에서 제거됐다. 플래그는 무시된다.
        deepvoice = args.deepvoice
        deepvoice_scoring = False
        # 모드 2 — EXE에는 복제 모델을 넣지 않는다(수 GB). 검증기 경로로 돈다.
        cloner = ""
        encoders = 2
        time_budget = 90.0

    try:
        asyncio.run(amain(Args()))
    except KeyboardInterrupt:
        print("\n종료합니다.")
    return 0


def _default_whisper() -> str:
    from mirinae.mode1.stt import DEFAULT_MODEL

    return DEFAULT_MODEL


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:  # EXE에서 창이 즉시 닫히면 원인을 볼 수 없다
        import traceback

        traceback.print_exc()
        input(f"\n오류: {e}\n엔터를 누르면 종료합니다...")
        raise
