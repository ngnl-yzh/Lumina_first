import { useCallback, useEffect, useState } from "react";
import Mode1 from "./Mode1";
import Mode2 from "./Mode2";
import Settings from "./Settings";
import { primeSpeech } from "./lib/tts";
import { loadSettings, saveSettings, type AppSettings, type Tab } from "./types";

/**
 * 미리내 폰 버전 — **UI 전용.**
 *
 * 최종 제품이 어떤 모습인지 보여주는 것이 목적이라 마이크도 서버도 쓰지 않는다.
 * 정해진 시나리오를 재생하므로 인터넷 없이 폰에서 단독으로 돌아간다.
 * 실제로 동작하는 것은 `app-desktop`이다.
 *
 * 그래도 **폰에서 제대로 보이는 것**은 진짜다 —
 * 데스크톱 목업(폰 베젤·가짜 상태바)이 아니라 실기기 전체 뷰포트를 쓰고
 * 노치·홈바는 safe-area-inset으로 피한다.
 */

const TAB_COLOR: Record<Tab, string> = {
  mode1: "#0D7A85",
  mode2: "#5E5A94",
  settings: "#14202B",
};

const TAB_META: Array<{ id: Tab; icon: string; label: string }> = [
  { id: "mode1", icon: "🛡", label: "보이스피싱" },
  { id: "mode2", icon: "🎙", label: "딥보이스" },
  { id: "settings", icon: "⚙", label: "설정" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("mode1");
  const [settings, setSettings] = useState<AppSettings>(loadSettings);

  const onSave = useCallback((s: AppSettings) => {
    setSettings(s);
    saveSettings(s);
  }, []);

  useEffect(() => {
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", TAB_COLOR[tab]);
  }, [tab]);

  useEffect(() => {
    document.documentElement.style.fontSize = settings.largeText ? "18px" : "16px";
  }, [settings.largeText]);

  const selectTab = (t: Tab) => {
    // 탭 전환은 사용자 제스처다. iOS는 제스처 밖 speak()를 무시하므로
    // 여기서 음성 권한을 미리 열어 둔다 — 경고 순간에 소리가 안 나는 사고를 막는다.
    primeSpeech();
    setTab(t);
  };

  return (
    <div className="app">
      <div className="header" style={{ background: TAB_COLOR[tab] }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span className="brand">미리내</span>
          <span className="badge">
            {tab === "mode1" ? "MODE 1" : tab === "mode2" ? "MODE 2" : "설정"}
          </span>
        </div>
        <div className="conn">
          <span style={{ color: "rgba(255,255,255,.55)", fontSize: 10 }}>UI 미리보기</span>
        </div>
      </div>

      <div className="tabs">
        {TAB_META.map(({ id, icon, label }) => (
          <button
            key={id}
            className="tab"
            onClick={() => selectTab(id)}
            style={{
              color: tab === id ? TAB_COLOR[id] : undefined,
              borderBottomColor: tab === id ? TAB_COLOR[id] : "transparent",
            }}
          >
            <span className="icon">{icon}</span>
            <span className="label">{label}</span>
          </button>
        ))}
      </div>

      {tab === "mode1" && <Mode1 settings={settings} />}
      {tab === "mode2" && <Mode2 />}
      {tab === "settings" && <Settings settings={settings} onSave={onSave} />}
    </div>
  );
}
