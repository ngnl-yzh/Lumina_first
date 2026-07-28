import { useCallback, useEffect, useRef, useState } from "react";
import { speakWarning, stopSpeaking } from "./lib/tts";
import { DEMO_ROUTE, DEMO_SCRIPT, DEMO_WARNING, playScript, type DemoLine } from "./lib/demo";
import { ALL_STAGES, STAGE_LABEL, type AppSettings } from "./types";

/**
 * 폰 버전 모드 1 — **UI 전용.**
 * 마이크도 서버도 쓰지 않고 정해진 대본을 재생한다.
 * 실제로 동작하는 것은 app-desktop이다.
 */

const STAGE_COLOR: Record<string, string> = {
  S1: "#9A6410", S2: "#B85C20", S3: "#CC4A20", S4: "#A62F5B",
  S5: "#CC4A20", S6: "#9A6410", S7: "#B85C20", S8: "#9A6410",
};

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

export default function Mode1({ settings }: { settings: AppSettings }) {
  const [running, setRunning] = useState(false);
  const [lines, setLines] = useState<DemoLine[]>([]);
  const [stages, setStages] = useState<Record<string, boolean>>({});
  const [score, setScore] = useState(0);
  const [level, setLevel] = useState("안전");
  const [warning, setWarning] = useState(false);
  const [secs, setSecs] = useState(0);

  const stopRef = useRef<(() => void) | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const warnRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const stop = useCallback(() => {
    stopRef.current?.();
    stopRef.current = null;
    if (tickRef.current) clearInterval(tickRef.current);
    if (warnRef.current) clearTimeout(warnRef.current);
    tickRef.current = null;
    warnRef.current = null;
    stopSpeaking();
    setRunning(false);
    setWarning(false);
    setLines([]);
    setStages({});
    setScore(0);
    setLevel("안전");
    setSecs(0);
  }, []);

  const start = useCallback(() => {
    setLines([]);
    setStages({});
    setScore(0);
    setLevel("안전");
    setWarning(false);
    setSecs(0);
    setRunning(true);

    tickRef.current = setInterval(() => setSecs((s) => s + 1), 1000);

    stopRef.current = playScript<DemoLine>(DEMO_SCRIPT, (line) => {
      setLines((prev) => [...prev, line]);
      setStages((prev) => ({ ...prev, [line.stage]: true }));
      setScore(line.score);
      setLevel(line.level);

      if (line.level === "위험") {
        // 발화 종료 후 개입까지의 지연 — 설계 목표가 1.5초 이내다
        warnRef.current = setTimeout(() => {
          setWarning(true);
          if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
          if (settings.enableVoiceWarning) {
            speakWarning([
              DEMO_WARNING.quote,
              ...DEMO_WARNING.counter,
              DEMO_WARNING.control,
              DEMO_WARNING.crossCheck,
            ]);
          }
        }, 800);
      }
    });
  }, [settings.enableVoiceWarning]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);
  useEffect(() => () => stop(), [stop]);

  const pct = Math.round(score * 100);
  const color = score >= 0.75 ? "#A62F5B" : score >= 0.45 ? "#9A6410" : "#2E7D52";
  const mmss = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;

  // ── 개입 ──────────────────────────────────────────────────────────────────
  if (warning) {
    return (
      <div className="intervene">
        <h1>보이스피싱 위험</h1>
        <div className="quote-box">
          <div style={{ fontSize: 11, opacity: 0.6, fontWeight: 800, letterSpacing: "0.1em" }}>
            탐지된 발화
          </div>
          <div style={{ fontSize: 17, fontWeight: 800, marginTop: 5, lineHeight: 1.4 }}>
            {DEMO_WARNING.quote}
          </div>
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          {DEMO_WARNING.counter.map((c, i) => (
            <div key={i} className="action-line">
              <span>🏛</span>
              <span>{c}</span>
            </div>
          ))}
          <div className="action-line">
            <span>📵</span>
            <span>{DEMO_WARNING.control}</span>
          </div>
          <div className="action-line">
            <span>📞</span>
            <span>{DEMO_WARNING.crossCheck}</span>
          </div>
        </div>

        <button className="hangup" onClick={stop}>
          지금 전화 끊기
        </button>
        <p style={{ textAlign: "center", fontSize: 11, opacity: 0.5, marginTop: 10 }}>
          UI 미리보기 — 실제 상황이 아닙니다
        </p>
      </div>
    );
  }

  // ── 모니터링 ──────────────────────────────────────────────────────────────
  if (running) {
    return (
      <div className="screen" style={{ paddingTop: 12 }}>
        <div className="card" style={{ background: "#14202B", color: "#fff" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: 700 }}>
              <span style={{ color: "#F87171" }}>●</span> 분석 중{" "}
              <span className="mono" style={{ color: "#8B97A8" }}>{mmss}</span>
            </span>
            <span className="mono" style={{ color, fontWeight: 900 }}>
              {level} {pct}%
            </span>
          </div>
          <div className="small" style={{ color: "#8B97A8", marginTop: 4 }}>
            유형: {DEMO_ROUTE}
          </div>
        </div>

        <div className="card">
          <div className="card-title">8단계 패턴</div>
          <div className="stage-row">
            {ALL_STAGES.map((s) => {
              const hit = !!stages[s];
              return (
                <div
                  key={s}
                  className="stage"
                  style={{
                    background: hit ? STAGE_COLOR[s] : "#F0F4F8",
                    transform: hit ? "scale(1.05)" : "none",
                    outline: s === "S4" ? `2px solid ${hit ? STAGE_COLOR[s] : "#E4EBF0"}` : "none",
                    outlineOffset: -1,
                  }}
                >
                  <span className="sid" style={{ color: hit ? "#fff" : STAGE_COLOR[s] }}>{s}</span>
                  <span className="sname" style={{ color: hit ? "rgba(255,255,255,.8)" : "#B0BAC5" }}>
                    {STAGE_LABEL[s]}
                  </span>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: 12 }}>
            <div className="bar">
              <span style={{ width: `${pct}%`, background: color }} />
            </div>
          </div>
        </div>

        {lines.length === 0 ? (
          <div className="card" style={{ textAlign: "center", color: "#B0BAC5", padding: 28 }}>
            <div style={{ fontSize: 30 }}>🎤</div>
            <div style={{ marginTop: 8 }}>상대방 발화 대기 중</div>
          </div>
        ) : (
          lines.map((u, i) => (
            <div key={i} className="utt">
              <div className="small" style={{ marginBottom: 3 }}>
                발화 {i + 1} · {u.stage} {STAGE_LABEL[u.stage]}
              </div>
              <Highlighted text={u.text} keywords={u.matched} />
            </div>
          ))
        )}
        <div ref={endRef} />

        <button className="btn btn-dark" style={{ marginTop: 12 }} onClick={stop}>
          중지
        </button>
      </div>
    );
  }

  // ── 대기 ──────────────────────────────────────────────────────────────────
  return (
    <div className="screen">
      <h2 style={{ fontSize: 21, fontWeight: 900, letterSpacing: "-0.03em" }}>
        보이스피싱 실시간 감지
      </h2>
      <p className="muted" style={{ marginTop: 4, marginBottom: 16 }}>
        통화 중 상대방 목소리를 분석해 위험한 말이 나오면 즉시 알립니다
      </p>

      <div className="banner banner-note">
        이 화면은 <strong>UI 미리보기</strong>입니다. 정해진 시나리오를 재생하며
        마이크와 서버를 쓰지 않습니다. 실제로 동작하는 것은 PC 버전입니다.
      </div>

      <div className="card">
        <div className="card-title">작동 방식</div>
        {[
          ["🔍", "182개 패턴 · 8단계 분류"],
          ["⚡", "말이 끝나고 1초 안에 경고"],
          ["🔊", "화면을 못 봐도 음성으로 알림"],
        ].map(([icon, text]) => (
          <div key={text} style={{ display: "flex", gap: 12, padding: "7px 0" }}>
            <span style={{ fontSize: 19 }}>{icon}</span>
            <span style={{ fontSize: 15 }}>{text}</span>
          </div>
        ))}
      </div>

      <button className="btn btn-mode1" onClick={start}>
        시연 재생
      </button>
      <p className="small" style={{ textAlign: "center", marginTop: 10 }}>
        기관사칭 시나리오 · 약 20초
      </p>
    </div>
  );
}
