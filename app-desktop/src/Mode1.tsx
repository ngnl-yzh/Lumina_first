import { useCallback, useEffect, useRef, useState } from "react";
import { MicRecorder, micSupportMessage } from "./lib/recorder";
import { StreamClient, type DeepvoiceInfo, type ServerMessage } from "./lib/ws";
import { speakWarning, stopSpeaking } from "./lib/tts";
import { ALL_STAGES, STAGE_COLOR, STAGE_LABEL, type AppSettings, type ConnState } from "./types";

interface Utterance {
  id: number;
  text: string;
  matched: string[];
  score: number;
  criticals: string[];
  pairs: string[];
  latencyMs: number;
}

interface Warning {
  quote: string;
  counter: string[];
  control: string;
  crossCheck: string;
  action: string;
}

function Highlighted({ text, keywords }: { text: string; keywords: string[] }) {
  if (!keywords.length) return <>{text}</>;
  const esc = keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${esc.join("|")})`, "g"));
  return (
    <>
      {parts.map((p, i) => (keywords.includes(p) ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}
    </>
  );
}

export default function Mode1({
  settings,
  onConn,
}: {
  settings: AppSettings;
  onConn: (s: ConnState) => void;
}) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [stages, setStages] = useState<Record<string, number>>({});
  const [score, setScore] = useState(0);
  const [level, setLevel] = useState("안전");
  const [route, setRoute] = useState("");
  const [coverage, setCoverage] = useState(0);
  const [warning, setWarning] = useState<Warning | null>(null);
  const [secs, setSecs] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const [deepvoice, setDeepvoice] = useState<DeepvoiceInfo | null>(null);

  const recRef = useRef<MicRecorder | null>(null);
  const wsRef = useRef<StreamClient | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(0);
  const lastAudioRef = useRef(0);

  const support = micSupportMessage();
  const addLog = (s: string) =>
    setLog((prev) => [...prev.slice(-60), `${new Date().toLocaleTimeString()}  ${s}`]);

  const stop = useCallback(async () => {
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = null;
    stopSpeaking();
    await recRef.current?.stop();
    recRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    setRunning(false);
    onConn("disconnected");
  }, [onConn]);

  const onMessage = useCallback(
    (msg: ServerMessage) => {
      if (msg.type === "utterance") {
        // 발화 종료부터 판정까지의 지연 — D08이 지표로 삼은 값이다(목표 ≤ 1.5초).
        // PC 버전에서는 이걸 화면에 띄워 실제로 몇 초인지 눈으로 확인할 수 있게 한다.
        const latency = lastAudioRef.current ? performance.now() - lastAudioRef.current : 0;
        const flat = Object.values(msg.matched ?? {}).flat();
        setUtterances((prev) => [
          ...prev,
          {
            id: idRef.current++,
            text: msg.text,
            matched: flat,
            score: msg.score,
            criticals: msg.criticals ?? [],
            pairs: msg.pairs ?? [],
            latencyMs: latency,
          },
        ]);
        setStages(msg.stages ?? {});
        setScore(msg.score);
        setLevel(msg.level);
        setRoute(`${msg.route} · ${msg.route_name ?? ""}`);
        setCoverage(msg.coverage ?? 0);
        if (msg.deepvoice) setDeepvoice(msg.deepvoice);
        addLog(`발화 → ${msg.level} ${(msg.score * 100).toFixed(0)}% (경로 ${msg.route})`);
      } else if (msg.type === "warning") {
        const w: Warning = {
          quote: msg.quote,
          counter: msg.counter,
          control: msg.control,
          crossCheck: msg.cross_check,
          action: msg.action,
        };
        setWarning(w);
        addLog("개입 발동");
        if (settings.enableVoiceWarning) {
          speakWarning([w.quote, ...w.counter, w.control, w.crossCheck, w.action]);
        }
      } else if (msg.type === "error") {
        setError(msg.message);
        addLog(`오류: ${msg.message}`);
      }
    },
    [settings.enableVoiceWarning],
  );

  const start = useCallback(async () => {
    setError(null);
    setUtterances([]);
    setStages({});
    setScore(0);
    setLevel("안전");
    setRoute("");
    setCoverage(0);
    setWarning(null);
    setSecs(0);
    setLog([]);

    try {
      addLog(`서버 연결 ${settings.serverUrl}`);
      const ws = new StreamClient({
        url: settings.serverUrl,
        mode: "mode1",
        onMessage,
        onState: onConn,
      });
      await ws.connect();
      wsRef.current = ws;
      addLog("연결됨");

      const rec = new MicRecorder({
        onFrame: (pcm) => {
          lastAudioRef.current = performance.now();
          ws.sendAudio(pcm);
        },
        onError: setError,
      });
      await rec.start();
      recRef.current = rec;
      addLog("마이크 시작 · 16 kHz raw PCM");

      tickRef.current = setInterval(() => setSecs((s) => s + 1), 1000);
      setRunning(true);
    } catch (e) {
      const m = e instanceof Error ? e.message : String(e);
      setError(m);
      addLog(`실패: ${m}`);
      await stop();
    }
  }, [settings.serverUrl, onMessage, onConn, stop]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [utterances]);
  useEffect(() => () => void stop(), [stop]);

  const pct = Math.round(score * 100);
  const color = score >= 0.75 ? "#A62F5B" : score >= 0.45 ? "#9A6410" : "#2E7D52";
  const mmss = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;

  return (
    <>
      {warning && (
        <div className="intervene">
          <div className="intervene-box">
            <h2>보이스피싱 위험</h2>
            <div className="quote-box">{warning.quote}</div>
            {warning.counter.map((c, i) => (
              <div key={i} className="action-line">
                <span>🏛</span>
                <span>{c}</span>
              </div>
            ))}
            <div className="action-line">
              <span>📵</span>
              <span>{warning.control}</span>
            </div>
            <div className="action-line">
              <span>📞</span>
              <span>{warning.crossCheck}</span>
            </div>
            <div className="action-line">
              <span>🚨</span>
              <span>{warning.action}</span>
            </div>
            <button
              className="btn"
              style={{ background: "#fff", color: "#A62F5B", marginTop: 14 }}
              onClick={() => {
                stopSpeaking();
                setWarning(null);
              }}
            >
              확인
            </button>
          </div>
        </div>
      )}

      <div className="split">
        {/* 좌 — 조작과 상태 */}
        <div className="pane">
          <div>
            <h1 className="page">보이스피싱 실시간 감지</h1>
            <p className="lede">
              마이크로 들어오는 발화를 서버가 전사·채점하고, 위험 단계에 도달하면 개입합니다.
            </p>
          </div>

          {support && <div className="banner banner-warn">{support}</div>}
          {error && <div className="banner banner-warn">{error}</div>}

          {!running ? (
            <button className="btn btn-mode1" onClick={() => void start()} disabled={!!support}>
              모니터링 시작
            </button>
          ) : (
            <button className="btn btn-dark" onClick={() => void stop()}>
              모니터링 중지
            </button>
          )}

          <div className="card">
            <div className="card-title">상태</div>
            <div className="row">
              <span className="small">경과</span>
              <span className="mono">{mmss}</span>
            </div>
            <div className="row">
              <span className="small">위험도</span>
              <span className="mono" style={{ color, fontWeight: 800 }}>
                {level} {pct}%
              </span>
            </div>
            <div className="row">
              <span className="small">유형 경로</span>
              <span className="mono small">{route || "—"}</span>
            </div>
            <div className="row">
              <span className="small">단계 커버리지</span>
              <span className="mono">{(coverage * 100).toFixed(0)}%</span>
            </div>
            <div style={{ marginTop: 10 }}>
              <div className="bar">
                <span style={{ width: `${pct}%`, background: color }} />
                <i className="tick" style={{ left: "45%" }} />
                <i className="tick" style={{ left: "75%" }} />
              </div>
              <div className="small mono" style={{ marginTop: 4 }}>
                주의 45% · 위험 75%
              </div>
            </div>
          </div>

          {/* 딥보이스 탐지 — P1 항목.
              자체 측정 재현율이 18.8%라 **표시만 하고 위험도는 바꾸지 않는다.**
              못 믿을 신호로 점수를 올리면 그게 곧 오탐이 된다. */}
          <div className="card">
            <div className="card-title">딥보이스 탐지 · P1</div>
            {!deepvoice || !deepvoice.enabled ? (
              <>
                <p className="small">
                  꺼져 있습니다. 켜려면 서버를 <span className="mono">--deepvoice</span>{" "}
                  옵션으로 실행하세요.
                </p>
                <p className="small" style={{ marginTop: 6 }}>
                  자체 측정에서 XTTS 합성음 16개 중 3개만 잡았습니다(재현율 18.8%).
                  기본값이 꺼짐인 이유입니다.
                </p>
              </>
            ) : !deepvoice.usable ? (
              <p className="small">판정 불가 — {deepvoice.label}</p>
            ) : (
              <>
                <div className="row">
                  <span className="small">합성 확률</span>
                  <span
                    className="mono"
                    style={{
                      fontWeight: 800,
                      color: (deepvoice.fake_prob ?? 0) >= 0.7 ? "#A62F5B" : "#56636E",
                    }}
                  >
                    {((deepvoice.fake_prob ?? 0) * 100).toFixed(1)}% · {deepvoice.label}
                  </span>
                </div>
                <div className="bar" style={{ marginTop: 8 }}>
                  <span
                    style={{
                      width: `${(deepvoice.fake_prob ?? 0) * 100}%`,
                      background: (deepvoice.fake_prob ?? 0) >= 0.7 ? "#A62F5B" : "#8794A0",
                    }}
                  />
                </div>
                <p className="small" style={{ marginTop: 8 }}>
                  {deepvoice.scoring
                    ? "⚠ 이 값이 위험도 점수에 반영되고 있습니다 (검증되지 않은 신호)"
                    : "참고 표시입니다. 위험도 점수에는 반영하지 않습니다."}
                </p>
              </>
            )}
          </div>

          <div className="card">
            <div className="card-title">8단계 패턴</div>
            <div className="stage-row">
              {ALL_STAGES.map((s) => {
                const hit = (stages[s] ?? 0) > 0;
                return (
                  <div
                    key={s}
                    className="stage"
                    style={{
                      background: hit ? STAGE_COLOR[s] : "#F1F5F8",
                      transform: hit ? "scale(1.04)" : "none",
                      outline: s === "S4" ? `2px solid ${hit ? STAGE_COLOR[s] : "#DDE5EB"}` : "none",
                      outlineOffset: -1,
                    }}
                  >
                    <span className="sid" style={{ color: hit ? "#fff" : STAGE_COLOR[s] }}>{s}</span>
                    <span
                      className="sname"
                      style={{ color: hit ? "rgba(255,255,255,.85)" : "#8794A0" }}
                    >
                      {STAGE_LABEL[s].slice(0, 2)}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="small" style={{ marginTop: 9 }}>
              S4(고립 유도)가 최고 가중치입니다. 정상 통화에 나올 이유가 없어 오탐률이 가장 낮습니다.
            </p>
          </div>

          <div className="card">
            <div className="card-title">서버 로그</div>
            <div className="log">{log.length ? log.join("\n") : "대기 중"}</div>
          </div>
        </div>

        {/* 우 — 실시간 전사 */}
        <div className="pane">
          <div className="card" style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
            <div className="card-title">실시간 전사 · 매칭 근거</div>
            <div className="transcript">
              {utterances.length === 0 ? (
                <p className="muted" style={{ padding: 20, textAlign: "center" }}>
                  {running ? "발화 대기 중 — 말을 하면 여기에 쌓입니다" : "모니터링을 시작하세요"}
                </p>
              ) : (
                utterances.map((u, i) => (
                  <div key={u.id} className="utt" data-hit={u.matched.length > 0}>
                    <div className="small mono" style={{ marginBottom: 4 }}>
                      발화 {i + 1} · 위험도 {(u.score * 100).toFixed(0)}%
                      {u.latencyMs > 0 && ` · 판정 지연 ${(u.latencyMs / 1000).toFixed(2)}초`}
                    </div>
                    <Highlighted text={u.text} keywords={u.matched} />
                    {(u.criticals.length > 0 || u.pairs.length > 0) && (
                      <div className="evidence">
                        발동 규칙: {[...u.criticals, ...u.pairs].join(", ")}
                      </div>
                    )}
                  </div>
                ))
              )}
              <div ref={endRef} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
