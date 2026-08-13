import { useCallback, useEffect, useRef, useState } from "react";
import { MicRecorder, TARGET_SAMPLE_RATE, micSupportMessage } from "./lib/recorder";
import { StreamClient, encodeWav, type ServerMessage } from "./lib/ws";
import type { AppSettings, ConnState } from "./types";

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
  const [snr, setSnr] = useState<number | null>(null);
  const [srs, setSrs] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState<{ step: number; total: number }>(
    { step: 0, total: 0 });
  const [basis, setBasis] = useState<string>("");
  const [violation, setViolation] = useState<number | null>(null);
  const [passed, setPassed] = useState<boolean | null>(null);
  const [secs, setSecs] = useState(0);
  const [level, setLevel] = useState(0);
  const [urls, setUrls] = useState<
    { original: string; protected: string; control: string } | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const recRef = useRef<MicRecorder | null>(null);
  const wsRef = useRef<StreamClient | null>(null);
  const rawRef = useRef<Float32Array[]>([]);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const support = micSupportMessage();
  const addLog = (s: string) =>
    setLog((prev) => [...prev.slice(-60), `${new Date().toLocaleTimeString()}  ${s}`]);

  /**
   * 녹음을 멈추고 **발화 전체를 한 번에** 보호한다.
   *
   * 예전에는 2초 청크를 녹음 중에 실시간으로 보호했다. 그런데 그 방식은
   * 방어가 성립하지 않았다 — 청크마다 자기 임베딩에서 멀어지므로 미는 방향이
   * 제각각이고, 복제 모델이 파일 전체를 볼 때 서로 상쇄된다.
   *
   * 실사용 녹음 14초 실측 (60스텝 · SNR 20 dB · 통화채널 통과 후):
   *   청크 스트리밍  SRS 0.8045   ← 판정 임계값 0.7962 위. 실패
   *   전체 발화 일괄  SRS 0.3709   ← 크게 아래. 성공. 게다가 13배 빠르다
   *
   * 대가는 녹음이 끝나야 시작할 수 있다는 것이다.
   * 되지도 않는 실시간을 흉내내는 것보다 되는 방어를 주는 편이 낫다.
   */
  const protectAll = useCallback(async () => {
    const total = rawRef.current.reduce((n, f) => n + f.length, 0);
    if (total === 0) return;
    const original = new Float32Array(total);
    let off = 0;
    for (const f of rawRef.current) {
      original.set(f, off);
      off += f.length;
    }

    setProcessing(true);
    setProgress({ step: 0, total: 0 });
    addLog(`녹음 ${(total / TARGET_SAMPLE_RATE).toFixed(1)}초 · 전체 발화 보호 시작`);

    const ws = new StreamClient({
      url: settings.serverUrl,
      mode: "protect",
      onState: onConn,
      onMessage: (msg: ServerMessage) => {
        if (msg.type === "ready") {
          addLog(`서버 준비 · 최대 ${(msg as { steps?: number }).steps ?? "?"}스텝`);
        } else if (msg.type === "progress") {
          setProgress({ step: msg.step, total: msg.total });
        } else if (msg.type === "done") {
          const d = msg as unknown as {
            srs: number; srs_basis: string; snr_db: number;
            elapsed: number; audible_violation: number; below_threshold: boolean;
          };
          setSrs(d.srs);
          setSnr(d.snr_db);
          setElapsed(d.elapsed);
          setBasis(d.srs_basis);
          setViolation(d.audible_violation);
          setPassed(d.below_threshold);
          addLog(
            `완료 · SRS ${d.srs.toFixed(4)} (${d.srs_basis}) · ` +
              `SNR ${d.snr_db.toFixed(1)} dB · ${d.elapsed.toFixed(0)}초 · ` +
              (d.below_threshold ? "섭동 충분히 주입됨" : "섭동 부족 — 스텝을 늘릴 것"),
          );
        } else if (msg.type === "error") {
          setError(msg.message);
          addLog(`오류: ${msg.message}`);
        }
      },
      onBinary: (_seq, delta) => {
        const n = Math.min(original.length, delta.length);
        const prot = new Float32Array(original.length);
        prot.set(original);
        for (let i = 0; i < n; i++) prot[i] = original[i] + delta[i];
        setUrls({
          original: URL.createObjectURL(encodeWav(original, TARGET_SAMPLE_RATE)),
          protected: URL.createObjectURL(encodeWav(prot, TARGET_SAMPLE_RATE)),
          control: URL.createObjectURL(
            encodeWav(bandNoiseMix(original, prot), TARGET_SAMPLE_RATE)),
        });
        setFinished(true);
        setProcessing(false);
        ws.close();
        onConn("disconnected");
      },
    });

    try {
      await ws.connect();
      wsRef.current = ws;
      ws.sendLabeledAudio({ type: "audio", sample_rate: TARGET_SAMPLE_RATE }, original);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setProcessing(false);
      onConn("disconnected");
    }
  }, [settings.serverUrl, onConn]);

  const stop = useCallback(async () => {
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = null;
    await recRef.current?.stop();
    recRef.current = null;
    setRunning(false);
    void protectAll();
  }, [protectAll]);

  const start = useCallback(async () => {
    setError(null);
    setSnr(null);
    setSrs(null);
    setElapsed(null);
    setSecs(0);
    setFinished(false);
    setPassed(null);
    setViolation(null);
    setBasis("");
    setProgress({ step: 0, total: 0 });
    setUrls(null);
    setLog([]);
    rawRef.current = [];

    try {
      // 녹음 중에는 서버에 붙지 않는다. 보호는 정지 후 한 번에 한다.
      const rec = new MicRecorder({
        onLevel: setLevel,
        onError: setError,
        onFrame: (pcm) => rawRef.current.push(pcm.slice(0)),
      });
      await rec.start();
      recRef.current = rec;
      addLog("마이크 시작 · 녹음 중 (보호는 정지 후 일괄)");

      tickRef.current = setInterval(() => setSecs((s) => s + 1), 1000);
      setRunning(true);
    } catch (e) {
      const m = e instanceof Error ? e.message : String(e);
      setError(m);
      addLog(`실패: ${m}`);
      await recRef.current?.stop();
      recRef.current = null;
      setRunning(false);
    }
  }, []);

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

  return (
    <div className="split">
      <div className="pane">
        <div>
          <h1 className="page">딥보이스 학습 방지</h1>
          <p className="lede">
            녹음을 마치면 <strong>발화 전체를 한 번에</strong> 최적화해 적대적 섭동을 주입합니다.
            청크로 쪼개 실시간 처리하던 방식은 방어가 성립하지 않아 바꿨습니다.
          </p>
        </div>

        {support && <div className="banner banner-warn">{support}</div>}
        {error && <div className="banner banner-warn">{error}</div>}

        {!running ? (
          <button
            className="btn btn-mode2"
            onClick={() => void start()}
            disabled={!!support || processing}
          >
            {processing ? "보호 처리 중…" : finished ? "다시 녹음" : "녹음 시작"}
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
              <div className="k">복제기 조건</div>
              <div className="v" style={{ color: (srs ?? 1) < 0.5 ? "#5E5A94" : "#8794A0" }}>
                {srs !== null ? srs.toFixed(3) : "—"}
              </div>
              <div className="small">낮을수록 좋음</div>
            </div>
            <div className="metric">
              <div className="k">섭동</div>
              <div
                className="v"
                style={{
                  fontSize: 20,
                  color: passed === null ? "#8794A0" : passed ? "#2E7D52" : "#A62F5B",
                }}
              >
                {passed === null ? "—" : passed ? "충분" : "부족"}
              </div>
              <div className="small">{secs}초 녹음</div>
            </div>
          </div>
          {elapsed !== null && (
            <p className="small mono" style={{ marginTop: 10 }}>
              {elapsed.toFixed(0)}초 처리 · 기준 {basis}
            </p>
          )}
          {finished && basis !== "" && !basis.includes("복제기") && (
            <div className="banner banner-note" style={{ marginTop: 10, marginBottom: 0 }}>
              <strong>제한 모드입니다.</strong> 지금 서버는 복제 모델 없이
              화자 검증기만 보고 최적화했습니다. 이 경로는 실측에서
              <strong> 실제 복제를 막지 못했습니다(저지 0%)</strong> —
              화면의 숫자는 검증기 유사도일 뿐입니다.
              제대로 보호하려면 서버를 <code>--cloner xtts,gsv</code>로 띄우세요.
            </div>
          )}
          {violation !== null && violation >= 0.05 && (
            <div className="banner banner-note" style={{ marginTop: 10, marginBottom: 0 }}>
              마스킹 임계값을 넘는 구간이 {(violation * 100).toFixed(0)}%입니다.
              들릴 수 있으니 원본과 나란히 들어보고 판단하세요.
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
          <div className="card-title">보호 진행</div>
          {processing ? (
            <>
              <p className="small" style={{ marginBottom: 10 }}>
                발화 전체를 최적화하는 중입니다. 녹음 길이와 장비에 따라 수십 초 걸립니다.
              </p>
              <div className="bar">
                <span
                  style={{
                    width: progress.total
                      ? `${(progress.step / progress.total) * 100}%` : "0%",
                    background: "#5E5A94",
                  }}
                />
              </div>
              <p className="small mono" style={{ marginTop: 8 }}>
                {progress.total ? `${progress.step} / ${progress.total} 스텝` : "준비 중…"}
              </p>
            </>
          ) : finished ? (
            <p className="small">
              완료되었습니다. 아래에서 들어보고 파일을 저장하세요.
            </p>
          ) : (
            <p className="muted">녹음을 마치면 시작됩니다</p>
          )}
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
