import { useCallback, useEffect, useRef, useState } from "react";
import { ChunkBuffer, MicRecorder, TARGET_SAMPLE_RATE, micSupportMessage } from "./lib/recorder";
import { OverlapAdder, StreamClient, encodeWav, type ServerMessage } from "./lib/ws";
import type { AppSettings, ConnState } from "./types";

type ChunkState = "pending" | "sending" | "done" | "degraded";

const CHUNK_COLOR: Record<ChunkState, string> = {
  pending: "#DDE4EC",
  sending: "#E8B84B",
  done: "#5E5A94",
  degraded: "#E07840",
};

const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * 2;
const HOP_SAMPLES = TARGET_SAMPLE_RATE * 1;

/**
 * C-B 대조군 — 섭동과 **같은 세기**의 통화대역 잡음을 섞은 음성.
 *
 * 복제 검증에서 이것 없이는 "그냥 잡음 아니냐"에 답할 수 없다.
 * 세기를 섭동과 정확히 맞추는 것이 핵심이다 — 잡음을 더 크게 넣고 이겼다고 하면
 * 아무 의미가 없다. 그래서 δ의 RMS를 그대로 목표로 삼는다.
 *
 * 대역 제한은 300~3400 Hz 1차 근사(고역·저역 통과 각 1단)로 한다.
 * 서버의 `controls.bandlimited_noise`만큼 가파르지는 않지만,
 * 이 파일의 용도는 **복제 서비스에 올릴 비교 대상**이므로 그 정도면 충분하다.
 * 정밀 측정은 서버가 만든 `control_C-B.wav`를 쓴다.
 */
function bandNoiseMix(original: Float32Array, protectedPcm: Float32Array): Float32Array {
  const n = Math.min(original.length, protectedPcm.length);
  let deltaSq = 0;
  for (let i = 0; i < n; i++) {
    const d = protectedPcm[i] - original[i];
    deltaSq += d * d;
  }
  const targetRms = Math.sqrt(deltaSq / Math.max(n, 1));

  // 백색잡음 → 대역 제한 (단순 1차 IIR 두 단)
  const noise = new Float32Array(n);
  for (let i = 0; i < n; i++) noise[i] = Math.random() * 2 - 1;

  const dt = 1 / TARGET_SAMPLE_RATE;
  const hpRc = 1 / (2 * Math.PI * 300);
  const hpA = hpRc / (hpRc + dt);
  let prevIn = 0, prevOut = 0;
  for (let i = 0; i < n; i++) {
    const out = hpA * (prevOut + noise[i] - prevIn);
    prevIn = noise[i];
    prevOut = out;
    noise[i] = out;
  }
  const lpRc = 1 / (2 * Math.PI * 3400);
  const lpA = dt / (lpRc + dt);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += lpA * (noise[i] - acc);
    noise[i] = acc;
  }

  let sq = 0;
  for (let i = 0; i < n; i++) sq += noise[i] * noise[i];
  const rms = Math.sqrt(sq / Math.max(n, 1)) || 1;
  const g = targetRms / rms;

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = original[i] + noise[i] * g;
  return out;
}

export default function Mode2({
  settings,
  onConn,
}: {
  settings: AppSettings;
  onConn: (s: ConnState) => void;
}) {
  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chunks, setChunks] = useState<ChunkState[]>([]);
  const [snr, setSnr] = useState<number | null>(null);
  const [srs, setSrs] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [steps, setSteps] = useState<number | null>(null);
  const [degraded, setDegraded] = useState(0);
  const [secs, setSecs] = useState(0);
  const [level, setLevel] = useState(0);
  const [urls, setUrls] = useState<
    { original: string; protected: string; control: string } | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const recRef = useRef<MicRecorder | null>(null);
  const wsRef = useRef<StreamClient | null>(null);
  const bufRef = useRef(new ChunkBuffer(CHUNK_SAMPLES, HOP_SAMPLES));
  const adderRef = useRef(new OverlapAdder(CHUNK_SAMPLES, HOP_SAMPLES));
  const rawRef = useRef<Float32Array[]>([]);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const support = micSupportMessage();
  const addLog = (s: string) =>
    setLog((prev) => [...prev.slice(-60), `${new Date().toLocaleTimeString()}  ${s}`]);

  const setChunk = (i: number, s: ChunkState) =>
    setChunks((prev) => {
      const next = [...prev];
      while (next.length <= i) next.push("pending");
      next[i] = s;
      return next;
    });

  const finalize = useCallback(() => {
    const total = rawRef.current.reduce((n, f) => n + f.length, 0);
    if (total === 0) return;
    const original = new Float32Array(total);
    let off = 0;
    for (const f of rawRef.current) {
      original.set(f, off);
      off += f.length;
    }
    const prot = adderRef.current.result(original);
    setUrls({
      original: URL.createObjectURL(encodeWav(original, TARGET_SAMPLE_RATE)),
      protected: URL.createObjectURL(encodeWav(prot, TARGET_SAMPLE_RATE)),
      control: URL.createObjectURL(
        encodeWav(bandNoiseMix(original, prot), TARGET_SAMPLE_RATE)),
    });
    setFinished(true);
    addLog(`조립 완료 · 청크 ${adderRef.current.count}개 · ${(total / TARGET_SAMPLE_RATE).toFixed(1)}초`);
  }, []);

  const stop = useCallback(async () => {
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = null;
    await recRef.current?.stop();
    recRef.current = null;
    wsRef.current?.stop();
    setRunning(false);
    // 마지막 청크의 δ가 돌아올 시간을 준다 — "발화 후 대기 0.5초"의 실체
    setTimeout(() => {
      finalize();
      wsRef.current?.close();
      wsRef.current = null;
      onConn("disconnected");
    }, 1200);
  }, [finalize, onConn]);

  const onMessage = useCallback((msg: ServerMessage) => {
    if (msg.type === "chunk") {
      setChunk(msg.seq, msg.degraded ? "degraded" : "done");
      setSnr(msg.snr_db);
      setSrs(msg.srs);
      setElapsed(msg.elapsed);
      setSteps(msg.steps);
      if (msg.degraded) setDegraded((d) => d + 1);
      addLog(
        `청크 ${msg.seq} · SRS ${msg.srs.toFixed(3)} · SNR ${msg.snr_db.toFixed(1)} dB · ` +
          `${msg.steps}스텝 · ${msg.elapsed.toFixed(2)}초${msg.degraded ? " [감축]" : ""}`,
      );
    } else if (msg.type === "error") {
      setError(msg.message);
      addLog(`오류: ${msg.message}`);
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setChunks([]);
    setSnr(null);
    setSrs(null);
    setElapsed(null);
    setSteps(null);
    setDegraded(0);
    setSecs(0);
    setFinished(false);
    setUrls(null);
    setLog([]);
    bufRef.current.reset();
    adderRef.current.reset();
    rawRef.current = [];

    try {
      addLog(`서버 연결 ${settings.serverUrl}`);
      const ws = new StreamClient({
        url: settings.serverUrl,
        mode: "mode2",
        onMessage,
        onBinary: (seq, delta) => adderRef.current.add(seq, delta),
        onState: onConn,
      });
      await ws.connect();
      wsRef.current = ws;
      addLog("연결됨 · 2.0초 청크 / 1.0초 홉");

      const rec = new MicRecorder({
        onLevel: setLevel,
        onError: setError,
        onFrame: (pcm) => {
          rawRef.current.push(pcm.slice(0));
          for (const { seq, audio } of bufRef.current.push(pcm)) {
            setChunk(seq, "sending");
            ws.sendAudio(audio, seq);
          }
        },
      });
      await rec.start();
      recRef.current = rec;
      addLog("마이크 시작");

      tickRef.current = setInterval(() => setSecs((s) => s + 1), 1000);
      setRunning(true);
    } catch (e) {
      const m = e instanceof Error ? e.message : String(e);
      setError(m);
      addLog(`실패: ${m}`);
      await stop();
    }
  }, [settings.serverUrl, onMessage, onConn, stop]);

  useEffect(() => () => void recRef.current?.stop(), []);

  const play = (url: string) => {
    audioRef.current?.pause();
    const a = new Audio(url);
    audioRef.current = a;
    void a.play();
  };

  const download = (url: string, name: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
  };

  const done = chunks.filter((c) => c === "done" || c === "degraded").length;

  return (
    <div className="split">
      <div className="pane">
        <div>
          <h1 className="page">딥보이스 학습 방지</h1>
          <p className="lede">
            녹음과 동시에 적대적 섭동을 주입합니다. 2초 청크를 서버로 보내고 δ를 받아
            50% 겹침으로 조립합니다.
          </p>
        </div>

        {support && <div className="banner banner-warn">{support}</div>}
        {error && <div className="banner banner-warn">{error}</div>}

        {!running ? (
          <button className="btn btn-mode2" onClick={() => void start()} disabled={!!support}>
            {finished ? "다시 녹음" : "녹음 시작"}
          </button>
        ) : (
          <button className="btn btn-dark" onClick={() => void stop()}>
            녹음 중지
          </button>
        )}

        <div className="card">
          <div className="card-title">지표</div>
          <div className="metrics">
            <div className="metric">
              <div className="k">SNR</div>
              <div className="v" style={{ color: (snr ?? 0) >= 20 ? "#5E5A94" : "#8794A0" }}>
                {snr ? snr.toFixed(1) : "—"}
              </div>
              <div className="small">목표 ≥20 dB</div>
            </div>
            <div className="metric">
              <div className="k">SRS</div>
              <div className="v" style={{ color: (srs ?? 1) < 0.5 ? "#5E5A94" : "#8794A0" }}>
                {srs !== null ? srs.toFixed(3) : "—"}
              </div>
              <div className="small">낮을수록 좋음</div>
            </div>
            <div className="metric">
              <div className="k">청크</div>
              <div className="v">{done}</div>
              <div className="small">{secs}초 녹음</div>
            </div>
          </div>
          {elapsed !== null && (
            <p className="small mono" style={{ marginTop: 10 }}>
              마지막 청크 {steps}스텝 · {elapsed.toFixed(2)}초 처리
              {elapsed > 1.0 && " ← 홉(1.0초)보다 느립니다. 적체가 쌓입니다"}
            </p>
          )}
          {degraded > 0 && (
            <div className="banner banner-note" style={{ marginTop: 10, marginBottom: 0 }}>
              서버가 밀려 {degraded}개 청크의 스텝을 줄였습니다. 이 구간은 방어가 약합니다.
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">입력 레벨</div>
          <div className="bar">
            <span style={{ width: `${Math.min(100, level * 300)}%`, background: "#5E5A94" }} />
          </div>
        </div>
      </div>

      <div className="pane">
        <div className="card">
          <div className="card-title">청크 진행 — 2.0초 단위 / 1.0초 홉</div>
          <div className="chunk-grid">
            {chunks.length === 0 ? (
              <p className="muted">녹음을 시작하면 표시됩니다</p>
            ) : (
              chunks.map((c, i) => (
                <div key={i} className="chunk" style={{ background: CHUNK_COLOR[c] }}>
                  {i}
                </div>
              ))
            )}
          </div>
          <div className="small" style={{ marginTop: 10, display: "flex", gap: 14 }}>
            {(
              [
                ["#DDE4EC", "대기"],
                ["#E8B84B", "전송"],
                ["#5E5A94", "완료"],
                ["#E07840", "스텝 감축"],
              ] as const
            ).map(([c, l]) => (
              <span key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <i style={{ width: 10, height: 10, borderRadius: 3, background: c, display: "inline-block" }} />
                {l}
              </span>
            ))}
          </div>
        </div>

        {finished && urls && (
          <div className="card">
            <div className="card-title">A/B 비교 · 내보내기</div>
            <p className="small" style={{ marginBottom: 12 }}>
              두 파일이 같게 들려야 정상입니다. 차이가 들리면 마스킹 배율이 너무 높은 것입니다.
            </p>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button className="btn btn-dark" onClick={() => play(urls.original)}>▶ 원본</button>
              <button className="btn btn-mode2" onClick={() => play(urls.protected)}>▶ 보호본</button>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button className="btn btn-ghost" onClick={() => download(urls.original, "원본.wav")}>
                원본 저장
              </button>
              <button
                className="btn btn-ghost"
                onClick={() => download(urls.protected, "미리내_보호본.wav")}
              >
                보호본 저장
              </button>
              <button
                className="btn btn-ghost"
                onClick={() => download(urls.control, "대조군_잡음.wav")}
              >
                대조군 저장
              </button>
            </div>
            <p className="small" style={{ marginTop: 10 }}>
              세 파일을 각각 복제 서비스에 올려 같은 문장을 생성한 뒤,
              <strong> 복제 검증</strong> 탭에 넣으면 유사도를 비교해 드립니다.
            </p>
            <p className="small" style={{ marginTop: 6 }}>
              대조군은 섭동과 <strong>같은 세기</strong>의 통화대역 잡음입니다.
              이게 없으면 “그냥 잡음 아니냐”에 답할 수 없습니다.
            </p>
          </div>
        )}

        <div className="card">
          <div className="card-title">처리 로그</div>
          <div className="log" style={{ maxHeight: 240 }}>
            {log.length ? log.join("\n") : "대기 중"}
          </div>
        </div>
      </div>
    </div>
  );
}
