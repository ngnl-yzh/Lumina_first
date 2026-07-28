"""서버 통합 테스트 — 실제 WebSocket으로 붙여 왕복을 확인한다.

단위 테스트가 다 통과해도 배선이 틀리면 시연 당일에 죽는다.
STT는 무거우므로 대본을 미리 넣은 NullSTT로 갈아끼워 **배선만** 검증한다.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ws_server                                              # noqa: E402
from mirinae.config import SAMPLE_RATE                        # noqa: E402
from mirinae.mode1.stt import NullSTT                         # noqa: E402

HOST, PORT = "127.0.0.1", 8799


def speech_like(n_sec: float, amp: float = 0.3) -> np.ndarray:
    """VAD가 발화로 인식할 만한 신호. 실제 말일 필요는 없다 — STT는 대본을 쓴다."""
    n = int(SAMPLE_RATE * n_sec)
    t = np.arange(n) / SAMPLE_RATE
    sig = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 6))
    return (sig / np.abs(sig).max() * amp).astype(np.float32)


def silence(n_sec: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * n_sec), dtype=np.float32)


class FakeServices(ws_server.Services):
    """무거운 모델을 올리지 않는 서비스. 배선만 본다."""

    def __init__(self, script: list[str]) -> None:
        from mirinae.mode1 import load_db
        from mirinae.mode1.scorer import Scorer

        self.device = torch.device("cpu")
        self.db = load_db()
        self.scorer = Scorer(self.db)
        self.stt = NullSTT(script)
        self._encoder = None
        self._masking = None
        # 딥보이스 탐지는 끈 상태로 배선만 확인한다.
        # 켜면 수백 MB 모델을 받게 되고, 어차피 재현율 18.8%라 테스트 가치가 없다.
        self._deepvoice = None
        self.deepvoice_scoring = False


@pytest.mark.asyncio
async def test_mode1_end_to_end():
    """오디오를 밀어넣으면 발화 판정과 경고가 돌아와야 한다.

    시나리오는 기관사칭 A 경로 — S1 → S2 → S4(고립 유도)에서 개입이 걸려야 한다.
    """
    script = [
        "서울중앙지방검찰청 수사관입니다.",
        "귀하 명의가 도용되어 대포통장에 이용됐습니다.",
        "수사 기밀이니 가족에게도 말하지 마십시오.",
    ]
    svc = FakeServices(script)
    cfg = ws_server.PGDConfig()

    async with websockets.serve(
        lambda ws: ws_server.handler(ws, svc, cfg), HOST, PORT, max_size=8 << 20
    ):
        async with websockets.connect(f"ws://{HOST}:{PORT}") as ws:
            await ws.send(json.dumps({"type": "start", "mode": "mode1"}))

            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert ready["type"] == "ready" and ready["mode"] == "mode1"

            # 발화 3개 — 사이에 침묵을 넣어 VAD가 경계를 잡게 한다
            for _ in script:
                await ws.send(speech_like(1.2).tobytes())
                await ws.send(silence(0.8).tobytes())

            utterances, warning = [], None
            try:
                while len(utterances) < len(script) or warning is None:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg["type"] == "utterance":
                        utterances.append(msg)
                    elif msg["type"] == "warning":
                        warning = msg
            except asyncio.TimeoutError:
                pass

            await ws.send(json.dumps({"type": "stop"}))

    assert len(utterances) == 3, f"발화 {len(utterances)}개만 돌아왔다"
    assert utterances[-1]["route"] == "A"
    assert utterances[-1]["score"] >= 0.75, utterances[-1]["score"]
    assert utterances[0]["score"] < utterances[-1]["score"], "누적이 안 되고 있다"

    assert warning is not None, "S4에서 경고가 안 나왔다"
    assert "가족" in warning["quote"] or "기밀" in warning["quote"], warning["quote"]
    assert warning["cross_check"]
    assert warning["tts_tokens"]

    # 딥보이스 탐지가 꺼져 있으면 **점수를 0으로 채우지 않고** 판정 안 함을 알려야 한다.
    # 모르는 것을 0으로 보내면 UI가 "정상 음성"이라고 표시해 거짓말이 된다.
    dv = utterances[-1].get("deepvoice")
    assert dv is not None, "deepvoice 필드가 없다"
    assert dv["enabled"] is False
    assert dv["fake_prob"] is None, "판정하지 않았는데 숫자가 들어 있다"


@pytest.mark.asyncio
async def test_mode2_returns_delta():
    """모드 2는 청크마다 헤더 + 바이너리 δ를 돌려줘야 한다.

    PGD가 실제로 돌아가므로 스텝을 최소로 낮춰 빠르게 확인한다.
    """
    from mirinae.encoder import SpeakerEncoder

    svc = FakeServices([])
    svc._encoder = SpeakerEncoder(device=torch.device("cpu"))
    cfg = ws_server.PGDConfig(steps=3)

    async with websockets.serve(
        lambda ws: ws_server.handler(ws, svc, cfg), HOST, PORT + 1, max_size=8 << 20
    ):
        async with websockets.connect(f"ws://{HOST}:{PORT + 1}") as ws:
            await ws.send(json.dumps({"type": "start", "mode": "mode2"}))
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert ready["type"] == "ready" and ready["mode"] == "mode2"

            chunk = speech_like(2.0)
            await ws.send(json.dumps({"type": "seq", "n": 0}))
            await ws.send(chunk.tobytes())

            header = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            assert header["type"] == "chunk"
            assert header["seq"] == 0
            assert header["snr_db"] > 0

            payload = await asyncio.wait_for(ws.recv(), timeout=30)
            delta = np.frombuffer(payload, dtype="<f4")

            await ws.send(json.dumps({"type": "stop"}))

    assert len(delta) == len(chunk), f"δ 길이 {len(delta)} ≠ 입력 {len(chunk)}"
    assert np.isfinite(delta).all(), "δ에 NaN/Inf"
    assert np.abs(delta).max() > 0, "δ가 전부 0이다"
