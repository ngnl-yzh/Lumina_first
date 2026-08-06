"""WebSocket 서버 — 폰과 서버를 잇는다.

프로토콜은 단순하다. 제어는 JSON 텍스트, 오디오는 바이너리로 보낸다.
바이너리 프레임 앞에 항상 JSON 헤더가 먼저 오므로 순번이 어긋나지 않는다.

  클라 → 서버
    {"type":"start","mode":"mode1"|"mode2"}
    <binary>  Float32 PCM 16 kHz 모노 (mode1: 연속 스트림 / mode2: 2초 청크)
    {"type":"seq","n":k}   mode2에서 다음 바이너리의 순번을 알린다
    {"type":"stop"}

  서버 → 클라
    {"type":"ready","mode":...}
    mode1: {"type":"utterance",...} · {"type":"warning",...}
    mode2: {"type":"chunk","seq":k,...} 뒤에 <binary δ>

실행:
    python ws_server.py --host 0.0.0.0 --port 8765
    python ws_server.py --ssl-cert cert.pem --ssl-key key.pem   # iOS는 HTTPS 필수
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
from pathlib import Path

import numpy as np
import torch
import websockets

sys.path.insert(0, str(Path(__file__).parent))

from mirinae.config import PGDConfig, SAMPLE_RATE, default_device
from mirinae.encoder import SpeakerEncoder
from mirinae.mode1 import load_db
from mirinae.mode1.scorer import CallState, Scorer
from mirinae.mode1.segmenter import StreamingVAD
from mirinae.mode1.stt import DEFAULT_MODEL as DEFAULT_WHISPER, SpeechToText
from mirinae.mode1.warning import build_warning
from mirinae.psychoacoustic import MaskingModel
from mirinae.worker import ChunkJob, ChunkWorker


class Services:
    """무거운 모델을 프로세스당 한 번만 올린다.

    연결마다 Whisper를 새로 올리면 첫 발화가 몇 초씩 밀린다.
    """

    def __init__(
        self,
        device: torch.device,
        whisper_size: str = "base",
        deepvoice: bool = False,
        deepvoice_scoring: bool = False,
    ) -> None:
        self.device = device
        self.db = load_db()
        self.scorer = Scorer(self.db)
        self.stt = SpeechToText(model_size=whisper_size)
        self._encoder: SpeakerEncoder | None = None
        self._masking: MaskingModel | None = None

        # 딥보이스 탐지 (D08 P1).
        #
        # 자체 측정에서 XTTS-v2 합성음 16개 중 3개만 잡았다(재현율 18.8%).
        # 게다가 완전히 같은 조건에서 만든 5개 중 3개는 1.0000, 2개는 0.0000으로
        # 판정이 갈렸다 — 신뢰할 수 없는 수준의 분산이다.
        # D08이 예고한 "학습에 없던 합성 방식에 일반화가 약하다"가 그대로 나온 것이다.
        #
        # 그래서 기본값이 둘 다 False다.
        #   deepvoice          — 탐지를 아예 돌릴지
        #   deepvoice_scoring  — 탐지 결과를 **위험도 점수에 반영**할지
        # 표시만 하고 점수는 건드리지 않는 것이 기본이다.
        # 못 믿을 신호가 위험도를 올리면 그건 곧 오탐이 된다.
        self.deepvoice_scoring = deepvoice_scoring
        self._deepvoice = None
        if deepvoice:
            from mirinae.mode1.deepvoice import DeepvoiceDetector

            self._deepvoice = DeepvoiceDetector(device="cuda" if device.type == "cuda" else "cpu")

    @property
    def encoder(self) -> SpeakerEncoder:
        if self._encoder is None:
            self._encoder = SpeakerEncoder(device=self.device)
        return self._encoder

    @property
    def masking(self) -> MaskingModel:
        if self._masking is None:
            self._masking = MaskingModel(device=self.device)
        return self._masking

    async def detect_deepvoice(self, audio: np.ndarray, loop) -> dict:
        """딥보이스 판정. 탐지기가 꺼져 있으면 '판정 안 함'을 그대로 돌려준다.

        점수를 0으로 채워 보내지 않는다. **모르는 것과 0은 다르다** —
        UI가 "정상 음성"이라고 표시해 버리면 그건 거짓말이 된다.
        """
        if self._deepvoice is None:
            return {"enabled": False, "usable": False, "fake_prob": None, "label": "미사용"}

        r = await loop.run_in_executor(None, self._deepvoice.score, audio)
        return {
            "enabled": True,
            "usable": r.usable,
            "fake_prob": r.fake_prob if r.usable else None,
            "label": r.label(),
            "scoring": self.deepvoice_scoring,
        }

    def warm_up(self) -> None:
        """첫 연결이 느리지 않도록 미리 올린다. 시연 안정성에 직접 영향을 준다."""
        print("  모델 예열 중...", flush=True)
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.stt.transcribe_text(silence)
        _ = self.encoder
        _ = self.masking
        if self._deepvoice is not None:
            self._deepvoice.score(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        print("  예열 완료", flush=True)


def decode_audio(payload: bytes) -> np.ndarray:
    """Float32 리틀엔디언 PCM. 폰의 AudioWorklet이 내보내는 그대로다."""
    return np.frombuffer(payload, dtype="<f4").astype(np.float32)


# ── 모드 1 ────────────────────────────────────────────────────────────────────

async def handle_mode1(ws, svc: Services) -> None:
    vad = StreamingVAD()
    state = CallState(svc.scorer)
    loop = asyncio.get_running_loop()

    await ws.send(json.dumps({"type": "ready", "mode": "mode1"}))

    async for message in ws:
        if isinstance(message, str):
            msg = json.loads(message)
            if msg.get("type") == "stop":
                break
            continue

        for utt in vad.push(decode_audio(message)):
            # STT와 스코어링은 CPU를 오래 잡는다. 이벤트 루프를 막으면
            # 그 사이 들어오는 오디오가 쌓여 지연이 눈덩이처럼 커진다.
            #
            # 어디서 시간을 쓰는지 나눠서 잰다. "느리다"만으로는 STT를 줄여야 할지
            # 스코어러를 고쳐야 할지 알 수 없다 — 실측상 거의 전부 STT다.
            t0 = time.perf_counter()
            text = await loop.run_in_executor(None, svc.stt.transcribe_text, utt.audio)
            stt_ms = (time.perf_counter() - t0) * 1000.0
            if not text:
                continue

            dv = await svc.detect_deepvoice(utt.audio, loop)
            # 탐지 결과를 점수에 반영할지는 설정에 달렸다. 기본은 반영하지 않는다.
            dv_for_score = (
                dv["fake_prob"] if (dv["usable"] and svc.deepvoice_scoring) else 0.0
            )

            t1 = time.perf_counter()
            result = state.add_utterance(text, deepvoice_score=dv_for_score)
            score_ms = (time.perf_counter() - t1) * 1000.0
            await ws.send(json.dumps({
                "type": "utterance",
                "text": text,
                "start": utt.start_sec,
                "end": utt.end_sec,
                "score": result.score,
                "level": result.level,
                "route": result.route.id,
                "route_name": result.route.name,
                "coverage": result.coverage,
                "stages": {k: v for k, v in result.stage_hits.items() if v > 0},
                "matched": result.matched,
                "criticals": result.criticals,
                "pairs": result.pairs,
                "benign": result.benign_hits,
                # 인용으로 판단해 점수에서 뺀 표현. 화면에 "왜 안 올렸는가"를 보이려면 필요하다 —
                # 근거 패널이 올린 이유만 설명하면 안 올린 판단은 검증할 수 없다.
                "suppressed": result.suppressed,
                # 처리 시간 분해 — 어디를 줄여야 하는지 화면에서 바로 보이게 한다
                "stt_ms": round(stt_ms, 1),
                "score_ms": round(score_ms, 1),
                "audio_sec": round(utt.duration, 2),
                "deepvoice": dv,
            }, ensure_ascii=False))

            if state.should_intervene():
                w = build_warning(result, svc.db)
                await ws.send(json.dumps({
                    "type": "warning",
                    "quote": w.quote,
                    "counter": w.counter,
                    "control": w.control,
                    "cross_check": w.cross_check,
                    "action": w.action,
                    "lines": w.screen_lines(),
                    "tts_tokens": w.tts_tokens,
                    "score": result.score,
                }, ensure_ascii=False))

    tail = vad.finish()
    if tail is not None:
        text = await loop.run_in_executor(None, svc.stt.transcribe_text, tail.audio)
        if text:
            state.add_utterance(text)

    await ws.send(json.dumps({
        "type": "done",
        "utterances": len(state.utterances),
        "final_score": state.last.score if state.last else 0.0,
    }))


# ── 모드 2 ────────────────────────────────────────────────────────────────────

async def handle_mode2(ws, svc: Services, cfg: PGDConfig) -> None:
    worker = ChunkWorker(svc.encoder, cfg, masking_model=svc.masking)
    loop = asyncio.get_running_loop()
    pending = 0
    next_seq = 0

    await ws.send(json.dumps({
        "type": "ready", "mode": "mode2",
        "chunk_sec": 2.0, "hop_sec": 1.0, "sample_rate": SAMPLE_RATE,
    }))

    async for message in ws:
        if isinstance(message, str):
            msg = json.loads(message)
            if msg.get("type") == "stop":
                break
            if msg.get("type") == "seq":
                next_seq = int(msg["n"])
            continue

        audio = torch.from_numpy(decode_audio(message)).to(svc.device)
        job = ChunkJob(seq=next_seq, audio=audio)
        next_seq += 1
        pending += 1

        out = await loop.run_in_executor(None, worker.process, job, pending)
        pending -= 1

        await ws.send(json.dumps({
            "type": "chunk",
            "seq": out.seq,
            "srs": out.srs,
            "snr_db": out.snr_db,
            "steps": out.steps_used,
            "degraded": out.degraded,      # 감축된 청크는 숨기지 않고 알린다
            "elapsed": out.elapsed_sec,
        }))
        await ws.send(out.delta.detach().cpu().numpy().astype("<f4").tobytes())

    await ws.send(json.dumps({
        "type": "done",
        "chunks": worker.stats.processed,
        "degraded": worker.stats.degraded,
        "max_queue": worker.stats.max_queue,
        "report": worker.stats.report(),
    }, ensure_ascii=False))


# ── 진입점 ────────────────────────────────────────────────────────────────────

async def handler(ws, svc: Services, cfg: PGDConfig) -> None:
    peer = getattr(ws, "remote_address", None)
    print(f"[접속] {peer}", flush=True)
    try:
        first = await ws.recv()
        msg = json.loads(first) if isinstance(first, str) else {}
        mode = msg.get("mode", "mode1")
        print(f"  모드: {mode}", flush=True)

        if mode == "mode2":
            await handle_mode2(ws, svc, cfg)
        else:
            await handle_mode1(ws, svc)
    except websockets.exceptions.ConnectionClosed:
        print(f"[끊김] {peer}", flush=True)
    except Exception as e:
        print(f"[오류] {peer} — {e}", flush=True)
        try:
            await ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        print(f"[종료] {peer}", flush=True)


async def amain(args) -> None:
    device = default_device()
    print(f"장치: {device}")
    if device.type == "cpu":
        print("  ※ CUDA 없음 — 모드 2 실시간 처리는 불가하다. 모드 1은 동작한다.")

    svc = Services(
        device,
        whisper_size=args.whisper,
        deepvoice=args.deepvoice,
        deepvoice_scoring=args.deepvoice_scoring,
    )
    if args.deepvoice:
        print("  딥보이스 탐지 켜짐 — 자체 측정 재현율 18.8%(XTTS). 참고용으로만 볼 것")
        if args.deepvoice_scoring:
            print("  ⚠ 탐지 결과를 위험도에 반영한다. 검증되지 않은 신호다")
    if not args.no_warmup:
        svc.warm_up()

    cfg = PGDConfig(steps=args.steps, masking_ratio=args.ratio)

    ssl_ctx = None
    if args.ssl_cert and args.ssl_key:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(args.ssl_cert, args.ssl_key)
        print("  TLS 활성 — iOS Safari는 HTTPS(WSS)가 아니면 마이크를 열지 않는다")

    scheme = "wss" if ssl_ctx else "ws"
    print(f"서버 시작: {scheme}://{args.host}:{args.port}")

    async with websockets.serve(
        lambda ws: handler(ws, svc, cfg),
        args.host, args.port, ssl=ssl_ctx,
        max_size=8 * 1024 * 1024,      # 2초 청크는 128 KB지만 여유를 둔다
        ping_interval=20, ping_timeout=20,
    ):
        await asyncio.Future()


def main() -> int:
    p = argparse.ArgumentParser(description="미리내 · WebSocket 서버")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--whisper", default=DEFAULT_WHISPER,
                   help="tiny/base/small/medium. 한국어는 small 이상을 권장한다. "
                        "GPU가 있으면 medium도 실시간을 유지한다")
    p.add_argument("--steps", type=int, default=PGDConfig.steps)
    p.add_argument("--ratio", type=float, default=PGDConfig.masking_ratio)
    p.add_argument("--ssl-cert")
    p.add_argument("--ssl-key")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--deepvoice", action="store_true",
                   help="딥보이스 탐지를 켠다 (표시만 · 자체 측정 재현율 18.8%%)")
    p.add_argument("--deepvoice-scoring", action="store_true",
                   help="탐지 결과를 위험도 점수에 반영한다. 검증 전에는 쓰지 말 것")
    args = p.parse_args()

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
