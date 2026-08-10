import { useCallback, useState } from "react";
import Mode1 from "./Mode1";
import Mode2 from "./Mode2";
import Settings from "./Settings";
import Verify from "./Verify";
import { loadSettings, saveSettings, type AppSettings, type ConnState, type Tab } from "./types";

/**
 * 미리내 PC 버전 — **실제로 동작하는 쪽이다.**
 *
 * 폰 버전과 목적이 다르다.
 *   폰(app-mobile) — 최종 제품 형태를 보여주는 UI. 서버 없이 단독 실행.
 *   PC(이 앱)      — 마이크·서버·전사·채점·섭동이 전부 진짜로 도는 실행 환경.
 *
 * 그래서 화면도 다르다. PC는 넓은 화면을 좌우로 나눠
 * 왼쪽에 조작·상태·로그를, 오른쪽에 실시간 내용을 놓는다.
 * 시연 중에 "지금 무슨 일이 일어나고 있는지"가 한 화면에 보여야 하기 때문이다.
 */

const TAB_COLOR: Record<Tab, string> = {
  mode1: "#0D7A85",
  mode2: "#5E5A94",
  verify: "#4A5568",
  settings: "#14202B",
};

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "mode1", label: "보이스피싱 감지" },
  { id: "mode2", label: "딥보이스 방지" },
  { id: "verify", label: "복제 검증" },
  { id: "settings", label: "설정" },
];

const CONN_LABEL: Record<ConnState, string> = {
  disconnected: "서버 대기",
  connecting: "연결 중",
  connected: "서버 연결됨",
  error: "연결 오류",
};

const CONN_COLOR: Record<ConnState, string> = {
  disconnected: "rgba(255,255,255,.4)",
  connecting: "#FDE68A",
  connected: "#6EE7B7",
  error: "#FCA5A5",
};

export default function App() {
  const [tab, setTab] = useState<Tab>("mode1");
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [conn, setConn] = useState<ConnState>("disconnected");

  const onSave = useCallback((s: AppSettings) => {
    setSettings(s);
    saveSettings(s);
  }, []);

  return (
    <div className="app">
      <div className="topbar" style={{ background: TAB_COLOR[tab] }}>
        <span className="brand">미리내</span>
        <span
          style={{
            fontSize: 11,
            fontWeight: 800,
            letterSpacing: "0.1em",
            padding: "3px 8px",
            borderRadius: 6,
            background: "rgba(255,255,255,.2)",
          }}
        >
          PC
        </span>

        <nav className="nav">
          {TABS.map((t) => (
            <button key={t.id} data-active={tab === t.id} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>

        <div className="conn">
          <span className="dot" style={{ background: CONN_COLOR[conn] }} />
          <span style={{ color: CONN_COLOR[conn] }}>{CONN_LABEL[conn]}</span>
        </div>
      </div>

      {/* 쓰지 않는 모드는 언마운트한다. 숨기기만 하면 마이크와 소켓이 계속 살아 있다. */}
      {tab === "mode1" && <Mode1 settings={settings} onConn={setConn} />}
      {tab === "mode2" && <Mode2 settings={settings} onConn={setConn} />}
      {tab === "verify" && <Verify settings={settings} onConn={setConn} />}
      {tab === "settings" && (
        <Settings settings={settings} conn={conn} onSave={onSave} onConn={setConn} />
      )}
    </div>
  );
}
