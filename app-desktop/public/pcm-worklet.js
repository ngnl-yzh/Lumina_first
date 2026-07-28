/**
 * AudioWorklet — raw Float32 PCM 캡처.
 *
 * MediaRecorder를 쓰지 않는 이유가 두 가지다. 설계도(D08·D09)는 MediaRecorder를
 * 지정하지만 이 파이프라인에서는 쓸 수 없다.
 *
 *  ① MediaRecorder는 **압축된** 오디오를 낸다. iOS Safari는 audio/mp4(AAC),
 *     Chrome은 WebM/Opus다. 모드 2는 overlap-add로 섭동을 샘플 단위 정렬해 합치므로
 *     압축을 거치면 정렬이 깨진다.
 *
 *  ② 더 결정적으로, 손실 압축은 우리가 심어놓은 섭동을 **그 자리에서 지운다.**
 *     코덱은 정확히 "안 들리는 성분"을 버리도록 설계돼 있는데,
 *     우리 섭동이 바로 그 성분이다. 마스킹 임계값 아래 숨긴 신호와
 *     심리음향 코덱은 같은 원리로 만들어졌다.
 *
 * 그래서 Web Audio로 raw PCM을 직접 받는다.
 */

const TARGET_RATE = 16000;

class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || TARGET_RATE;
    // 브라우저 샘플레이트는 보통 48 kHz다. 폰마다 달라서 고정할 수 없다.
    this.ratio = sampleRate / this.targetRate;
    this.frameSize = opts.frameSize || 1024;

    this.buffer = new Float32Array(this.frameSize);
    this.filled = 0;
    this.pos = 0; // 리샘플 위치 (소수)
    this.prev = 0;
    this.running = true;

    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "stop") this.running = false;
    };
  }

  /**
   * 선형 보간 리샘플. 48000 → 16000처럼 정수배면 정확하고,
   * 44100 → 16000처럼 나누어떨어지지 않아도 안전하다.
   */
  process(inputs) {
    if (!this.running) return false;

    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;

    while (this.pos < ch.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = i === 0 ? this.prev : ch[i - 1];
      const b = ch[i];
      this.buffer[this.filled++] = a + (b - a) * frac;

      if (this.filled >= this.frameSize) {
        // 복사본을 넘긴다. transfer하면 다음 프레임에서 버퍼가 비어 있다.
        this.port.postMessage(this.buffer.slice(0));
        this.filled = 0;
      }
      this.pos += this.ratio;
    }

    this.pos -= ch.length;
    this.prev = ch[ch.length - 1];
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
