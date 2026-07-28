/**
 * 마이크 캡처 — AudioWorklet으로 raw 16 kHz Float32 PCM을 받는다.
 *
 * 모바일 전용이라 걸리는 제약이 코드에 그대로 반영되어 있다.
 *  - iOS Safari는 **사용자 제스처** 안에서만 AudioContext가 살아난다.
 *    버튼 onClick 핸들러 밖에서 start()를 부르면 조용히 suspended 상태로 남는다.
 *  - secure context(HTTPS 또는 localhost)가 아니면 getUserMedia 자체가 없다.
 *  - 녹음 중 화면이 꺼지면 처리가 끊기므로 Wake Lock을 잡는다.
 */

export interface RecorderOptions {
  frameSize?: number;
  onFrame: (pcm: Float32Array) => void;
  onLevel?: (rms: number) => void;
  onError?: (message: string) => void;
}

export const TARGET_SAMPLE_RATE = 16000;

export function isSecureContextOk(): boolean {
  return (
    window.isSecureContext ||
    location.hostname === "localhost" ||
    location.hostname === "127.0.0.1"
  );
}

export function micSupportMessage(): string | null {
  if (!isSecureContextOk()) {
    return "HTTPS로 접속해야 마이크를 쓸 수 있습니다. (iOS는 예외 없음)";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "이 브라우저는 마이크 접근을 지원하지 않습니다.";
  }
  if (typeof AudioWorkletNode === "undefined") {
    return "이 브라우저는 AudioWorklet을 지원하지 않습니다.";
  }
  return null;
}

export class MicRecorder {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private wakeLock: WakeLockSentinel | null = null;
  private opts: RecorderOptions;

  recording = false;

  constructor(opts: RecorderOptions) {
    this.opts = opts;
  }

  /** 반드시 사용자 제스처(탭) 안에서 호출할 것. iOS는 그 밖에서는 열리지 않는다. */
  async start(): Promise<void> {
    const problem = micSupportMessage();
    if (problem) {
      this.opts.onError?.(problem);
      throw new Error(problem);
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false, // 섭동을 지울 수 있는 처리는 전부 끈다
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    this.ctx = new AudioContext();
    // iOS는 제스처 안에서도 suspended로 시작하는 경우가 있다
    if (this.ctx.state === "suspended") await this.ctx.resume();

    await this.ctx.audioWorklet.addModule("/pcm-worklet.js");

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-processor", {
      processorOptions: {
        targetRate: TARGET_SAMPLE_RATE,
        frameSize: this.opts.frameSize ?? 1024,
      },
    });

    this.node.port.onmessage = (e) => {
      const pcm = e.data as Float32Array;
      this.opts.onFrame(pcm);
      if (this.opts.onLevel) {
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
        this.opts.onLevel(Math.sqrt(sum / pcm.length));
      }
    };

    this.source.connect(this.node);
    // Worklet은 목적지에 연결되어야 돌아간다. 소리를 내면 안 되므로 무음 게인을 거친다.
    const silent = this.ctx.createGain();
    silent.gain.value = 0;
    this.node.connect(silent);
    silent.connect(this.ctx.destination);

    await this.acquireWakeLock();
    this.recording = true;
  }

  async stop(): Promise<void> {
    this.recording = false;
    this.node?.port.postMessage({ type: "stop" });
    this.node?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close().catch(() => undefined);
    await this.releaseWakeLock();

    this.node = null;
    this.source = null;
    this.stream = null;
    this.ctx = null;
  }

  /** 녹음 중 화면이 꺼지면 오디오 스레드가 멈춘다. 실패해도 녹음은 계속한다. */
  private async acquireWakeLock(): Promise<void> {
    try {
      const nav = navigator as Navigator & {
        wakeLock?: { request(type: "screen"): Promise<WakeLockSentinel> };
      };
      if (nav.wakeLock) this.wakeLock = await nav.wakeLock.request("screen");
    } catch {
      /* 지원 안 하는 브라우저가 있다. 치명적이지 않다. */
    }
  }

  private async releaseWakeLock(): Promise<void> {
    try {
      await this.wakeLock?.release();
    } catch {
      /* 무시 */
    }
    this.wakeLock = null;
  }
}

/** 2.0초 청크 · hop 1.0초로 잘라주는 링버퍼 (모드 2용). */
export class ChunkBuffer {
  private buf: Float32Array;
  private filled = 0;
  private seq = 0;

  constructor(
    private chunkSamples = TARGET_SAMPLE_RATE * 2,
    private hopSamples = TARGET_SAMPLE_RATE * 1,
  ) {
    this.buf = new Float32Array(this.chunkSamples);
  }

  /** 프레임을 넣고, 완성된 청크가 있으면 [순번, 오디오] 목록을 돌려준다. */
  push(frame: Float32Array): Array<{ seq: number; audio: Float32Array }> {
    const out: Array<{ seq: number; audio: Float32Array }> = [];
    let offset = 0;

    while (offset < frame.length) {
      const room = this.chunkSamples - this.filled;
      const take = Math.min(room, frame.length - offset);
      this.buf.set(frame.subarray(offset, offset + take), this.filled);
      this.filled += take;
      offset += take;

      if (this.filled >= this.chunkSamples) {
        out.push({ seq: this.seq++, audio: this.buf.slice(0) });
        // 50% 겹침 — 앞쪽 hop만큼 버리고 나머지를 당긴다
        this.buf.copyWithin(0, this.hopSamples);
        this.filled = this.chunkSamples - this.hopSamples;
      }
    }
    return out;
  }

  reset(): void {
    this.filled = 0;
    this.seq = 0;
    this.buf.fill(0);
  }
}
