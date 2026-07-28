import { useState } from "react";
import { StreamClient } from "./lib/ws";
import { speakWarning, ttsAvailable } from "./lib/tts";
import { isSecureContextOk, micSupportMessage } from "./lib/recorder";
import { DEFAULT_SETTINGS, defaultServerUrl, type AppSettings, type ConnState } from "./types";

export default function Settings({
  settings,
  conn,
  onSave,
  onConn,
}: {
  settings: AppSettings;
  conn: ConnState;
  onSave: (s: AppSettings) => void;
  onConn: (s: ConnState) => void;
}) {
  const [draft, setDraft] = useState(settings);
  const [saved, setSaved] = useState(false);
  const [testMsg, setTestMsg] = useState<string | null>(null);

  const micProblem = micSupportMessage();

  const test = async () => {
    setTestMsg("연결 시도 중...");
    const c = new StreamClient({
      url: draft.serverUrl,
      mode: "mode1",
      onMessage: () => undefined,
      onState: onConn,
    });
    try {
      await c.connect();
      setTestMsg("서버 연결 성공");
      c.close();
    } catch (e) {
      setTestMsg(e instanceof Error ? e.message : "연결 실패");
    }
  };

  const checks = [
    {
      label: "보안 컨텍스트",
      ok: isSecureContextOk(),
      hint: "localhost는 HTTPS 없이도 마이크가 열립니다",
    },
    { label: "마이크 · AudioWorklet", ok: !micProblem, hint: micProblem ?? "raw PCM 캡처 가능" },
    { label: "음성 안내 (TTS)", ok: ttsAvailable(), hint: "경고를 소리로 읽어줍니다" },
  ];

  return (
    <div className="split">
      <div className="pane">
        <div>
          <h1 className="page">설정</h1>
          <p className="lede">시연 전에 이 화면부터 확인하세요.</p>
        </div>

        <div className="card">
          <div className="card-title">실행 환경 점검</div>
          {checks.map((c) => (
            <div key={c.label} className="row">
              <div>
                <div style={{ fontWeight: 600 }}>{c.label}</div>
                <div className="small">{c.hint}</div>
              </div>
              <span style={{ fontSize: 19, color: c.ok ? "#2E7D52" : "#A62F5B" }}>
                {c.ok ? "✓" : "✕"}
              </span>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="card-title">서버</div>
          <input
            className="field"
            value={draft.serverUrl}
            onChange={(e) => {
              setDraft({ ...draft, serverUrl: e.target.value });
              setSaved(false);
            }}
            placeholder={defaultServerUrl()}
            spellCheck={false}
          />
          <p className="small" style={{ marginTop: 8 }}>
            서버 실행: <span className="mono">python ws_server.py --port 8765</span>
          </p>
          <button className="btn btn-ghost" style={{ marginTop: 10 }} onClick={() => void test()}>
            연결 테스트
          </button>
          {testMsg && (
            <p className="small mono" style={{ marginTop: 8 }}>
              {testMsg}
            </p>
          )}
          <div className="row" style={{ marginTop: 6 }}>
            <span className="small">현재 상태</span>
            <span className="mono small">{conn}</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">경고</div>
          <div className="row">
            <div>
              <div style={{ fontWeight: 600 }}>음성으로 알리기</div>
              <div className="small">개입 시 경고를 읽어줍니다</div>
            </div>
            <input
              type="checkbox"
              checked={draft.enableVoiceWarning}
              onChange={(e) => {
                setDraft({ ...draft, enableVoiceWarning: e.target.checked });
                setSaved(false);
              }}
              style={{ width: 18, height: 18 }}
            />
          </div>
          <button
            className="btn btn-ghost"
            style={{ marginTop: 8 }}
            onClick={() =>
              speakWarning([
                "지금 통화에서 안전계좌 라는 말이 나왔습니다.",
                "안전계좌는 존재하지 않습니다.",
              ])
            }
          >
            경고 음성 미리 듣기
          </button>
        </div>

        <button
          className="btn"
          style={{ background: saved ? "#2E7D52" : "#14202B" }}
          onClick={() => {
            onSave(draft);
            setSaved(true);
            setTimeout(() => setSaved(false), 1800);
          }}
        >
          {saved ? "✓ 저장했습니다" : "설정 저장"}
        </button>
        <button className="btn btn-ghost" onClick={() => setDraft(DEFAULT_SETTINGS)}>
          기본값으로
        </button>
      </div>

      <div className="pane">
        <div className="card">
          <div className="card-title">이 버전에 대하여</div>
          <p style={{ fontSize: 14, marginBottom: 12 }}>
            <strong>PC 버전은 실제로 동작합니다.</strong> 마이크에서 raw PCM을 받아
            WebSocket으로 서버에 보내고, 전사·채점·섭동 계산 결과를 그대로 받습니다.
          </p>
          <p className="small" style={{ marginBottom: 12 }}>
            폰 버전(<span className="mono">app-mobile</span>)은 최종 제품 형태를 보여주는
            UI 전용입니다. 서버 없이 단독으로 돌며 정해진 시나리오를 재생합니다.
          </p>
          <div className="row">
            <span className="small">버전</span>
            <span className="mono small">미리내 PC 0.1.0</span>
          </div>
          <div className="row">
            <span className="small">팀</span>
            <span className="small">Team Lumina · 전남대학교 인공지능학부</span>
          </div>
          <div className="row">
            <span className="small">패턴 DB</span>
            <span className="mono small">8단계 182항</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">알아두어야 할 한계</div>
          <ul style={{ paddingLeft: 18, fontSize: 13.5, lineHeight: 1.8, color: "#56636E" }}>
            <li>발화가 서버로 전송됩니다. 온디바이스 처리는 로드맵입니다.</li>
            <li>모드 2 실시간 처리는 CUDA GPU가 필요합니다. CPU에서는 청크가 밀립니다.</li>
            <li>
              XTTS-v2 복제 실패 검증이 아직 통과하지 않았습니다. 자세한 내용은
              <span className="mono"> server/README.md</span>를 보세요.
            </li>
            <li>패턴 DB는 초안이며 금융감독원 공개 자료로 교차 검증이 필요합니다.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
