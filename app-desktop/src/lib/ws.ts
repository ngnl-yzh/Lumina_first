/**
 * WebSocket 클라이언트 — 청크 재전송·순번 관리.
 *
 * D07 리스크표가 SEV2로 잡아둔 "WebSocket 스트리밍 불안정"에 대응한다.
 * 최악의 경우 전체 녹음 후 일괄 업로드로 폴백한다 — 실시간성은 잃지만 동작은 한다.
 */

/**
 * 딥보이스 탐지 결과.
 *
 * `fake_prob`이 null일 수 있다는 점이 중요하다 — **모르는 것과 0은 다르다.**
 * 탐지기가 꺼져 있거나 발화가 너무 짧아 판정하지 못한 경우를 0으로 채우면
 * 화면에 "정상 음성"이라고 뜨는데, 그건 확인한 적 없는 주장이다.
 */
export interface DeepvoiceInfo {
  enabled: boolean;
  usable: boolean;
  fake_prob: number | null;
  label: string;
  scoring?: boolean;
}

export type ServerMessage =
  | { type: "ready"; mode: string; [k: string]: unknown }
  | { type: "utterance"; text: string; score: number; level: string; route: string;
      route_name: string; coverage: number; stages: Record<string, number>;
      matched: Record<string, string[]>; criticals: string[]; pairs: string[];
      benign: string[]; suppressed?: Record<string, string[]>;
      start: number; end: number;
      deepvoice?: DeepvoiceInfo }
  | { type: "warning"; quote: string; counter: string[]; control: string;
      cross_check: string; action: string; lines: string[]; tts_tokens: string[];
      score: number }
  | { type: "chunk"; seq: number; srs: number; snr_db: number; steps: number;
      degraded: boolean; elapsed: number }
  | { type: "done"; [k: string]: unknown }
  | { type: "error"; message: string };

export type ConnState = "disconnected" | "connecting" | "connected" | "error";

export interface ClientOptions {
  url: string;
  mode: "mode1" | "mode2";
  onMessage: (msg: ServerMessage) => void;
  /** 모드 2에서 서버가 돌려준 섭동 δ. 직전 chunk 헤더의 순번에 대응한다. */
  onBinary?: (seq: number, delta: Float32Array) => void;
  onState?: (state: ConnState) => void;
}

export class StreamClient {
  private ws: WebSocket | null = null;
  private lastSeq = -1;
  private queue: ArrayBuffer[] = [];
  private opts: ClientOptions;

  state: ConnState = "disconnected";

  constructor(opts: ClientOptions) {
    this.opts = opts;
  }

  private setState(s: ConnState) {
    this.state = s;
    this.opts.onState?.(s);
  }

  connect(): Promise<void> {
    this.setState("connecting");

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.opts.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      const timer = setTimeout(() => {
        ws.close();
        this.setState("error");
        reject(new Error("서버 연결 시간 초과"));
      }, 8000);

      ws.onopen = () => {
        clearTimeout(timer);
        ws.send(JSON.stringify({ type: "start", mode: this.opts.mode }));
        this.setState("connected");
        // 연결 전에 쌓인 오디오를 흘려보낸다
        for (const buf of this.queue) ws.send(buf);
        this.queue = [];
        resolve();
      };

      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          const msg = JSON.parse(e.data) as ServerMessage;
          // 바이너리는 항상 chunk 헤더 **뒤에** 온다. 순번을 여기서 기억한다.
          if (msg.type === "chunk") this.lastSeq = msg.seq;
          this.opts.onMessage(msg);
        } else {
          this.opts.onBinary?.(this.lastSeq, new Float32Array(e.data as ArrayBuffer));
        }
      };

      ws.onerror = () => {
        clearTimeout(timer);
        this.setState("error");
        reject(new Error("서버 연결 실패"));
      };

      ws.onclose = () => {
        if (this.state !== "error") this.setState("disconnected");
      };
    });
  }

  sendAudio(pcm: Float32Array, seq?: number): void {
    const buf = pcm.buffer.slice(
      pcm.byteOffset,
      pcm.byteOffset + pcm.byteLength,
    ) as ArrayBuffer;

    if (this.ws?.readyState === WebSocket.OPEN) {
      if (seq !== undefined) {
        this.ws.send(JSON.stringify({ type: "seq", n: seq }));
      }
      this.ws.send(buf);
    } else {
      // 아직 안 열렸으면 버린다 — 다만 무한정 쌓이지 않게 상한을 둔다
      if (this.queue.length < 64) this.queue.push(buf);
    }
  }

  stop(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  close(): void {
    this.stop();
    this.ws?.close();
    this.ws = null;
    this.setState("disconnected");
  }
}

/** Hann overlap-add 조립 — 서버가 돌려준 청크별 δ를 하나의 파형으로 합친다. */
export class OverlapAdder {
  private acc: Float32Array;
  private wsum: Float32Array;
  private win: Float32Array;
  private received = new Set<number>();

  constructor(
    private chunkSamples: number,
    private hopSamples: number,
    capacitySec = 300,
    sampleRate = 16000,
  ) {
    const n = capacitySec * sampleRate;
    this.acc = new Float32Array(n);
    this.wsum = new Float32Array(n);
    this.win = new Float32Array(chunkSamples);
    for (let i = 0; i < chunkSamples; i++) {
      this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / chunkSamples);
    }
  }

  add(seq: number, delta: Float32Array): void {
    if (this.received.has(seq)) return; // 재전송으로 중복 도착
    this.received.add(seq);

    const start = seq * this.hopSamples;
    const half = this.chunkSamples >> 1;
    for (let i = 0; i < delta.length && start + i < this.acc.length; i++) {
      // 첫 청크의 앞쪽 반은 겹칠 상대가 없다. 창을 1로 펴서 진폭을 보존한다.
      const w = seq === 0 && i < half ? 1 : this.win[i];
      this.acc[start + i] += delta[i] * w;
      this.wsum[start + i] += w;
    }
  }

  /** 원본과 합쳐 보호본을 만든다. */
  result(original: Float32Array): Float32Array {
    const out = new Float32Array(original.length);
    for (let i = 0; i < original.length; i++) {
      const w = this.wsum[i];
      out[i] = original[i] + (w > 1e-8 ? this.acc[i] / w : 0);
    }
    return out;
  }

  get count(): number {
    return this.received.size;
  }

  reset(): void {
    this.acc.fill(0);
    this.wsum.fill(0);
    this.received.clear();
  }
}

/** Float32 PCM → WAV Blob. 내보내기·공유 시트에 쓴다. */
export function encodeWav(pcm: Float32Array, sampleRate = 16000): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buf);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, pcm.length * 2, true);

  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
