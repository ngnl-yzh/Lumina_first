import { useCallback, useEffect, useRef, useState } from "react";
import { DEMO_M2 } from "./lib/demo";

/**
 * 폰 버전 모드 2 — **UI 전용.**
 * 청크가 전송→처리→완료로 바뀌는 흐름을 보여준다.
 * 실제 PGD 계산은 서버에서 도는 것이고, PC 버전이 그것을 진짜로 한다.
 */

type ChunkState = "pending" | "sending" | "done";

const CHUNK_COLOR: Record<ChunkState, string> = {
  pending: "#DDE4EC",
  sending: "#E8B84B",
  done: "#5E5A94",
};

/** 파형 모양 — 결정적 난수라 새로 고쳐도 같은 그림이 나온다. */
function genBars(n: number): number[] {
  const bars: number[] = [];
  let state = 0x12348765 >>> 0;
  for (let i = 0; i < n; i++) {
    state = ((state * 1664525 + 1013904223) | 0) >>> 0;
    const r = state / 0xffffffff;
    const env = Math.sin((i / n) * Math.PI) * 0.65 + 0.3;
    const burst = 0.58 + 0.42 * Math.sin(i * 0.55);
    bars.push(Math.min(0.96, Math.max(0.06, r * env * burst * 1.4)));
  }
  return bars;
}

const BARS = genBars(72);

export default function Mode2() {
  const [state, setState] = useState<"idle" | "recording" | "done">("idle");
  const [chunks, setChunks] = useState<ChunkState[]>([]);
  const [recFrac, setRecFrac] = useState(0);
  const [procFrac, setProcFrac] = useState(0);
  const [snr, setSnr] = useState<number | null>(null);
  const [srs, setSrs] = useState<number | null>(null);
  const [exported, setExported] = useState(false);
  const [playing, setPlaying] = useState<"original" | "protected" | null>(null);

  const timers = useRef<Array<ReturnType<typeof setTimeout>>>([]);
  const raf = useRef(0);
  const startAt = useRef(0);

  const clear = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    cancelAnimationFrame(raf.current);
  }, []);

  const reset = useCallback(() => {
    clear();
    setState("idle");
    setChunks([]);
    setRecFrac(0);
    setProcFrac(0);
    setSnr(null);
    setSrs(null);
    setExported(false);
  }, [clear]);

  const start = useCallback(() => {
    clear();
    setState("recording");
    setChunks(Array(DEMO_M2.chunkCount).fill("pending"));
    setRecFrac(0);
    setProcFrac(0);
    setSnr(null);
    setSrs(null);
    setExported(false);
    startAt.current = performance.now();

    const totalMs = DEMO_M2.durationSec * 1000;
    const animate = () => {
      const frac = Math.min(1, (performance.now() - startAt.current) / totalMs);
      setRecFrac(frac);
      if (frac < 1) raf.current = requestAnimationFrame(animate);
    };
    raf.current = requestAnimationFrame(animate);

    for (let i = 0; i < DEMO_M2.chunkCount; i++) {
      const at = (i + 1) * DEMO_M2.chunkMs;
      timers.current.push(
        setTimeout(() => {
          setChunks((p) => {
            const n = [...p];
            n[i] = "sending";
            return n;
          });
        }, at),
        setTimeout(() => {
          setChunks((p) => {
            const n = [...p];
            n[i] = "done";
            return n;
          });
          setProcFrac(Math.min(1, (i + 1) / DEMO_M2.chunkCount));
          setSnr(20.1 + (i % 3) * 0.3);
          setSrs(0.28 + (i % 4) * 0.02);
        }, at + DEMO_M2.processMs),
      );
    }

    timers.current.push(
      setTimeout(() => {
        cancelAnimationFrame(raf.current);
        setRecFrac(1);
        setProcFrac(1);
        setSnr(DEMO_M2.finalSnr);
        setSrs(DEMO_M2.finalSrs);
        setState("done");
      }, DEMO_M2.chunkCount * DEMO_M2.chunkMs + DEMO_M2.processMs + 400),
    );
  }, [clear]);

  useEffect(() => () => clear(), [clear]);

  const done = chunks.filter((c) => c === "done").length;

  const Waveform = () => (
    <svg viewBox="0 0 300 70" style={{ width: "100%", height: 70 }} preserveAspectRatio="none">
      <rect x={0} y={34} width={300} height={2} fill="#E4EBF0" rx={1} />
      {BARS.map((h, i) => {
        const x = i * (300 / BARS.length);
        const bh = Math.max(2, h * 60);
        const frac = i / BARS.length;
        const fill = frac < procFrac ? "#5E5A94" : frac < recFrac ? "#8B97A8" : "#D0D8E4";
        return (
          <rect
            key={i}
            x={x}
            y={35 - bh / 2}
            width={300 / BARS.length - 1.2}
            height={bh}
            fill={fill}
            opacity={frac < procFrac ? 1 : frac < recFrac ? 0.75 : 0.5}
            rx={0.8}
          />
        );
      })}
    </svg>
  );

  // ── 대기 ──────────────────────────────────────────────────────────────────
  if (state === "idle") {
    return (
      <div className="screen">
        <h2 style={{ fontSize: 21, fontWeight: 900, letterSpacing: "-0.03em" }}>
          딥보이스 학습 방지
        </h2>
        <p className="muted" style={{ marginTop: 4, marginBottom: 16 }}>
          내 목소리에 사람 귀로는 안 들리는 보호막을 씌웁니다. AI가 이 목소리를 따라
          만들지 못하게 됩니다.
        </p>

        <div className="banner banner-note">
          이 화면은 <strong>UI 미리보기</strong>입니다. 실제 섭동 계산은 서버에서 이뤄지며
          PC 버전이 그것을 수행합니다.
        </div>

        <div className="card">
          <div className="card-title">처리 방식</div>
          {[
            ["🎙", "2초 단위로 잘라 녹음과 동시에 처리"],
            ["🔄", "서버에서 적대적 섭동 계산"],
            ["🛡", "말이 끝나고 0.5초 뒤 보호 파일 완성"],
          ].map(([icon, text]) => (
            <div key={text} style={{ display: "flex", gap: 12, padding: "7px 0" }}>
              <span style={{ fontSize: 19 }}>{icon}</span>
              <span style={{ fontSize: 15 }}>{text}</span>
            </div>
          ))}
        </div>

        <button className="btn btn-mode2" onClick={start}>
          시연 재생
        </button>
      </div>
    );
  }

  // ── 녹음 / 완료 ───────────────────────────────────────────────────────────
  return (
    <div className="screen">
      <div
        className="card"
        style={{
          background:
            state === "done"
              ? "linear-gradient(135deg,#2E7D52,#1E5C3C)"
              : "linear-gradient(135deg,#5E5A94,#4A4674)",
          color: "#fff",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontWeight: 900, fontSize: 17 }}>
              {state === "done" ? "처리 완료" : "녹음 중"}
            </div>
            <div className="small" style={{ color: "rgba(255,255,255,.7)" }}>
              {state === "done"
                ? `청크 ${done}개 · SNR ${snr?.toFixed(1)} dB · SRS ${srs?.toFixed(2)}`
                : `${(recFrac * DEMO_M2.durationSec).toFixed(1)}초 · 청크 ${done}/${DEMO_M2.chunkCount}`}
            </div>
          </div>
          {state === "recording" ? (
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: "#fff",
                animation: "pulse 1s infinite",
              }}
            />
          ) : (
            <span style={{ fontSize: 20 }}>✓</span>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">파형 — 처리된 구간이 보라색</div>
        <Waveform />
        <div className="small" style={{ display: "flex", gap: 12, marginTop: 6 }}>
          {(
            [
              ["#D0D8E4", "미기록"],
              ["#8B97A8", "녹음됨"],
              ["#5E5A94", "보호 처리"],
            ] as const
          ).map(([c, l]) => (
            <span key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <i style={{ width: 9, height: 9, borderRadius: 2, background: c, display: "inline-block" }} />
              {l}
            </span>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">청크 진행 · 2초 단위 / 1초 홉</div>
        <div className="chunk-grid">
          {chunks.map((c, i) => (
            <div key={i} className="chunk" style={{ background: CHUNK_COLOR[c] }}>
              {i + 1}
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">지표</div>
        <div style={{ display: "flex", gap: 10 }}>
          {[
            { label: "SNR", value: snr ? `${snr.toFixed(1)} dB` : "—", ok: (snr ?? 0) >= 20 },
            { label: "SRS", value: srs !== null ? srs.toFixed(2) : "—", ok: (srs ?? 1) < 0.5 },
          ].map((m) => (
            <div
              key={m.label}
              style={{
                flex: 1,
                textAlign: "center",
                padding: 10,
                borderRadius: 12,
                background: m.ok ? "rgba(94,90,148,.09)" : "#F8FAFB",
              }}
            >
              <div className="small">{m.label}</div>
              <div
                className="mono"
                style={{ fontSize: 19, fontWeight: 900, color: m.ok ? "#5E5A94" : "#9BA8B5" }}
              >
                {m.value}
              </div>
            </div>
          ))}
        </div>
        <p className="small" style={{ marginTop: 8 }}>
          SNR이 높을수록 안 들리고, SRS가 낮을수록 복제가 어려워집니다.
        </p>
      </div>

      {state === "done" && (
        <>
          <div className="card">
            <div className="card-title">들어서 비교하기</div>
            <p className="small" style={{ marginBottom: 10 }}>
              두 파일이 같게 들려야 정상입니다.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              {(["original", "protected"] as const).map((m) => (
                <button
                  key={m}
                  className="btn"
                  style={{
                    background: playing === m ? (m === "protected" ? "#5E5A94" : "#14202B") : "#F0F4F8",
                    color: playing === m ? "#fff" : "#6B7A8D",
                    fontSize: 15,
                    padding: 13,
                  }}
                  onClick={() => setPlaying(m)}
                >
                  ▶ {m === "original" ? "원본" : "보호본"}
                </button>
              ))}
            </div>
            <p className="small" style={{ marginTop: 8, textAlign: "center" }}>
              UI 미리보기라 실제 소리는 나지 않습니다
            </p>
          </div>

          <button
            className="btn btn-mode2"
            onClick={() => setExported(true)}
            style={{ background: exported ? "linear-gradient(135deg,#2E7D52,#1E5C3C)" : undefined }}
          >
            {exported ? "✓ 내보내기 완료" : "보호본 내보내기"}
          </button>
        </>
      )}

      <button className="btn btn-dark" style={{ marginTop: 10 }} onClick={reset}>
        {state === "recording" ? "중지" : "새로 시작"}
      </button>
    </div>
  );
}
