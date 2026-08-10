/**
 * 복제 검증 — 방어가 실제로 통했는지 확인한다.
 *
 * **복제 모델을 이 프로그램에 넣지 않는다.** 사용자가 GPT-SoVITS든 F5-TTS든
 * ElevenLabs든 원하는 서비스에 원본과 보호본을 각각 올려 복제음을 만들고,
 * 그 결과물을 여기 넣으면 원래 목소리와의 유사도를 재 준다.
 *
 * 그렇게 하는 이유가 셋이다.
 *   ① 모델을 붙일 때마다 러너를 새로 쓰지 않아도 된다
 *   ② 상용 서비스까지 같은 방식으로 검증할 수 있다
 *   ③ 시연에서 "우리 도구로 우리가 검증했다"가 아니라
 *      "제3자 서비스에 실제로 넣어봤다"가 된다 — 반박하기 훨씬 어렵다
 *
 * 화면 설계에서 두 가지를 꼭 지킨다.
 *
 * **반복 측정.** 생성이 확률적이라 1회로는 착시가 난다 —
 * 실측에서 완전히 같은 파일을 두 번 복제했는데 유사도가 0.019 차이 났고,
 * 방어 효과는 0.027~0.047 수준이었다. 같은 자릿수다.
 * 그래서 조건마다 여러 파일을 받아 평균과 95% 신뢰구간을 낸다.
 *
 * **기준선 판정.** 원본 복제부터 유사도가 낮으면 그 모델이 애초에 그 목소리를
 * 복제하지 못한 것이라 방어 효과를 논할 수 없다. 그 경우를 먼저 알린다.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { StreamClient, type ServerMessage } from "./lib/ws";
import type { AppSettings, ConnState } from "./types";

const TARGET_SAMPLE_RATE = 16000;

/** 파일명으로 조건을 추정한다. 사용자가 일일이 라벨을 달지 않아도 되게. */
function guessLabel(name: string): string {
  const n = name.toLowerCase();
  if (/protect|보호/.test(n)) return "보호본";
  if (/control_c-?a|white|백색/.test(n)) return "C-A 백색잡음";
  if (/control_c-?b|band|대역|잡음|noise/.test(n)) return "C-B 통화대역잡음";
  if (/control_c-?e|shuffle|셔플/.test(n)) return "C-E 셔플";
  if (/control_c-?c/.test(n)) return "C-C 무섭동";
  if (/origin|원본|clean/.test(n)) return "원본";
  return "미분류";
}

const LABELS = ["원본", "보호본", "C-A 백색잡음", "C-B 통화대역잡음",
  "C-E 셔플", "C-C 무섭동", "미분류"];

interface Item {
  id: number;
  name: string;
  label: string;
  pcm: Float32Array;
  srs?: number;
}

interface Stat {
  label: string;
  n: number;
  mean: number;
  half: number;
  values: number[];
}

/** 평균과 95% 신뢰구간 반폭. n=1이면 구간을 낼 수 없다 — NaN으로 표시한다. */
function meanCi(values: number[]): { mean: number; half: number } {
  const n = values.length;
  if (n === 0) return { mean: 0, half: NaN };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n === 1) return { mean, half: NaN };
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  return { mean, half: (1.96 * sd) / Math.sqrt(n) };
}

/** WAV/오디오 파일 → 16 kHz 모노 Float32 */
async function decodeTo16k(file: File): Promise<Float32Array> {
  const buf = await file.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 1, TARGET_SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(buf);
  const off = new OfflineAudioContext(
    1,
    Math.ceil((decoded.duration * TARGET_SAMPLE_RATE) || 1),
    TARGET_SAMPLE_RATE,
  );
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice();
}

export default function Verify({
  settings, onConn,
}: { settings: AppSettings; onConn: (s: ConnState) => void }) {
  const [reference, setReference] = useState<{ name: string; pcm: Float32Array } | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);
  const [threshold, setThreshold] = useState(0.7962);
  const [error, setError] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const idRef = useRef(0);

  const addLog = (s: string) =>
    setLog((p) => [...p.slice(-40), `${new Date().toLocaleTimeString()}  ${s}`]);

  // ── 파일 받기 ─────────────────────────────────────────────────────────────

  const pickReference = useCallback(async (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    try {
      setReference({ name: f.name, pcm: await decodeTo16k(f) });
      setError("");
    } catch {
      setError(`${f.name} 을 읽지 못했습니다. WAV 파일인지 확인해 주세요.`);
    }
  }, []);

  const addClones = useCallback(async (files: FileList | null) => {
    if (!files?.length) return;
    const next: Item[] = [];
    for (const f of Array.from(files)) {
      try {
        next.push({
          id: idRef.current++, name: f.name,
          label: guessLabel(f.name), pcm: await decodeTo16k(f),
        });
      } catch {
        setError(`${f.name} 을 읽지 못했습니다.`);
      }
    }
    setItems((p) => [...p, ...next]);
  }, []);

  const setLabel = (id: number, label: string) =>
    setItems((p) => p.map((it) => (it.id === id ? { ...it, label } : it)));

  const removeItem = (id: number) => setItems((p) => p.filter((it) => it.id !== id));

  // ── 검증 실행 ─────────────────────────────────────────────────────────────

  const run = useCallback(async () => {
    if (!reference || items.length === 0) return;
    setBusy(true);
    setError("");
    setItems((p) => p.map((it) => ({ ...it, srs: undefined })));

    const pending = new Map<number, (srs: number) => void>();

    const ws = new StreamClient({
      url: settings.serverUrl,
      mode: "compare",
      onState: onConn,
      onMessage: (msg: ServerMessage) => {
        if (msg.type === "ready" && typeof msg.threshold === "number") {
          setThreshold(msg.threshold);
          addLog(`판정 임계값 ${msg.threshold}`);
        } else if (msg.type === "reference_ok") {
          addLog(`기준 음성 ${msg.duration}초 등록`);
        } else if (msg.type === "result") {
          pending.get(msg.id)?.(msg.srs);
          pending.delete(msg.id);
        } else if (msg.type === "error") {
          setError(msg.message);
        }
      },
    });

    try {
      await ws.connect();
      addLog("서버 연결됨");

      ws.sendLabeledAudio({ type: "reference" }, reference.pcm);
      await new Promise((r) => setTimeout(r, 200));

      for (const it of items) {
        const got = new Promise<number>((resolve) => pending.set(it.id, resolve));
        ws.sendLabeledAudio({ type: "sample", label: it.label, id: it.id }, it.pcm);
        const srs = await got;
        setItems((p) => p.map((x) => (x.id === it.id ? { ...x, srs } : x)));
        addLog(`${it.name} → ${srs.toFixed(4)}`);
      }
      ws.close();
      addLog("완료");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      onConn("disconnected");
    }
  }, [reference, items, settings.serverUrl, onConn]);

  // ── 집계 ──────────────────────────────────────────────────────────────────

  const stats: Stat[] = useMemo(() => {
    const by = new Map<string, number[]>();
    for (const it of items) {
      if (it.srs === undefined) continue;
      const arr = by.get(it.label) ?? [];
      arr.push(it.srs);
      by.set(it.label, arr);
    }
    return LABELS.filter((l) => by.has(l)).map((label) => {
      const values = by.get(label)!;
      return { label, n: values.length, values, ...meanCi(values) };
    });
  }, [items]);

  const baseline = stats.find((s) => s.label === "원본");
  const protectedStat = stats.find((s) => s.label === "보호본");
  const noiseStats = stats.filter((s) => s.label.includes("잡음"));
  const bestNoise = noiseStats.length
    ? Math.min(...noiseStats.map((s) => s.mean)) : null;

  const done = stats.length > 0;
  const baselineBroken = baseline ? baseline.mean < 0.85 : false;

  return (
    <div className="pane pane-wide">
      <h1 className="page">복제 검증</h1>
      <p className="lede">
        원본과 보호본을 각각 복제 서비스에 올려 만든 음성을 여기 넣으면,
        원래 목소리와 얼마나 닮았는지 재 드립니다.
      </p>

      {error && <div className="banner banner-warn">{error}</div>}

      <div className="card">
        <div className="card-title">① 기준 음성 — 원래 목소리</div>
        <p className="small" style={{ marginBottom: 10 }}>
          모드 2에서 저장한 <span className="mono">원본.wav</span>를 넣습니다.
          모든 복제음을 이 목소리와 비교합니다.
        </p>
        <input type="file" accept="audio/*" onChange={(e) => void pickReference(e.target.files)} />
        {reference && (
          <p className="small" style={{ marginTop: 8 }}>
            선택됨 — <span className="mono">{reference.name}</span>{" "}
            ({(reference.pcm.length / TARGET_SAMPLE_RATE).toFixed(1)}초)
          </p>
        )}
      </div>

      <div className="card">
        <div className="card-title">② 복제음 파일</div>
        <p className="small" style={{ marginBottom: 10 }}>
          복제 서비스가 만든 음성들을 한꺼번에 넣습니다.
          파일명으로 조건을 자동 분류하며, 틀리면 표에서 직접 바꿀 수 있습니다.
          <br />
          <strong>같은 조건을 3~5개씩</strong> 넣어 주세요 — 생성이 확률적이라
          1개로는 효과와 편차를 구별할 수 없습니다.
        </p>
        <input
          type="file" accept="audio/*" multiple
          onChange={(e) => void addClones(e.target.files)}
        />

        {items.length > 0 && (
          <table className="verify-table" style={{ marginTop: 12 }}>
            <thead>
              <tr><th>파일</th><th>조건</th><th>유사도</th><th /></tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td className="mono small">{it.name}</td>
                  <td>
                    <select value={it.label} onChange={(e) => setLabel(it.id, e.target.value)}>
                      {LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
                    </select>
                  </td>
                  <td className="mono">{it.srs === undefined ? "—" : it.srs.toFixed(4)}</td>
                  <td>
                    <button className="btn btn-ghost" onClick={() => removeItem(it.id)}>
                      제거
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={{ margin: "16px 0" }}>
        <button
          className="btn btn-mode2"
          disabled={!reference || items.length === 0 || busy}
          onClick={() => void run()}
        >
          {busy ? "검증 중…" : "검증 실행"}
        </button>
      </div>

      {done && (
        <div className="card">
          <div className="card-title">결과 · 판정 임계값 {threshold.toFixed(4)}</div>

          {baselineBroken && (
            <div className="banner banner-warn">
              <strong>기준선이 낮습니다 ({baseline!.mean.toFixed(3)}).</strong>{" "}
              그 복제 모델이 애초에 이 목소리를 제대로 복제하지 못했다는 뜻입니다.
              이 상태에서는 방어 효과를 논할 수 없습니다 —
              한국어에 더 강한 모델로 다시 시도해 주세요.
            </div>
          )}

          <table className="verify-table">
            <thead>
              <tr>
                <th>조건</th><th>n</th><th>유사도</th>
                <th>95% 신뢰구간</th><th>판정</th><th>기준선 대비</th>
              </tr>
            </thead>
            <tbody>
              {stats.map((s) => {
                const other = s.mean < threshold;
                return (
                  <tr key={s.label} data-key={s.label === "보호본" ? "1" : undefined}>
                    <td>{s.label}</td>
                    <td className="mono">{s.n}</td>
                    <td className="mono">{s.mean.toFixed(4)}</td>
                    <td className="mono small">
                      {Number.isNaN(s.half) ? "n=1 — 낼 수 없음" : `±${s.half.toFixed(4)}`}
                    </td>
                    <td style={{ color: other ? "#5E5A94" : "#9B2C4A" }}>
                      {other ? "다른 화자" : "같은 화자"}
                    </td>
                    <td className="mono">
                      {baseline && s.label !== "원본"
                        ? (s.mean - baseline.mean).toFixed(4) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ marginTop: 14 }}>
            {protectedStat && (
              protectedStat.mean < threshold ? (
                <p className="verdict verdict-ok">
                  보호본이 임계값 아래입니다 — 복제음이 다른 사람으로 판정됩니다.
                </p>
              ) : (
                <p className="verdict verdict-bad">
                  <strong>보호본이 임계값 위입니다 ({protectedStat.mean.toFixed(4)}).</strong>{" "}
                  복제를 막지 못했습니다.
                </p>
              )
            )}

            {protectedStat && bestNoise !== null && (
              protectedStat.mean < bestNoise ? (
                <p className="verdict verdict-ok">
                  잡음 대조군({bestNoise.toFixed(4)})보다 낮습니다 —
                  “그냥 잡음 아니냐”에 답할 수 있습니다.
                </p>
              ) : (
                <p className="verdict verdict-bad">
                  <strong>잡음 대조군({bestNoise.toFixed(4)})보다 높습니다.</strong>{" "}
                  같은 세기의 단순 잡음이 더 잘 막았다는 뜻이라,
                  적대적 최적화의 우위를 주장할 수 없습니다.
                </p>
              )
            )}

            {stats.some((s) => s.n < 3) && (
              <p className="verdict verdict-bad">
                조건당 표본이 3개 미만인 것이 있습니다. 생성 편차와 방어 효과를
                구별할 수 없으니 3~5개씩 채워 주세요.
              </p>
            )}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">쓰는 순서</div>
        <ol className="small" style={{ paddingLeft: 18, lineHeight: 1.9 }}>
          <li>모드 2에서 녹음하고 <span className="mono">원본</span>·
            <span className="mono">보호본</span>·<span className="mono">대조군</span>을 저장합니다.</li>
          <li>복제 서비스(GPT-SoVITS 등)에 <strong>각 파일을 따로</strong> 올려
            목소리를 학습시킵니다.</li>
          <li>세 경우 모두 <strong>같은 문장</strong>으로 음성을 생성합니다.
            조건당 3~5번 반복하세요.</li>
          <li>생성된 파일을 전부 위 ②에 넣고 검증을 실행합니다.</li>
        </ol>
        <p className="small" style={{ marginTop: 10 }}>
          원본까지 함께 복제해야 하는 이유는, 그 모델이 이 목소리를 복제할 수 있는지
          먼저 확인해야 하기 때문입니다. 기준선 없이는 아무것도 주장할 수 없습니다.
        </p>
      </div>

      <div className="card">
        <div className="card-title">로그</div>
        <div className="log">
          {log.length === 0 ? <p className="muted">검증을 실행하면 표시됩니다</p>
            : log.map((l, i) => <div key={i} className="mono small">{l}</div>)}
        </div>
      </div>
    </div>
  );
}
