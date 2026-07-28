import { useState } from "react";
import { speakWarning, ttsAvailable } from "./lib/tts";
import { DEFAULT_SETTINGS, type AppSettings } from "./types";

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      aria-pressed={value}
      style={{
        width: 50,
        minHeight: 30,
        height: 30,
        borderRadius: 15,
        background: value ? "#0D7A85" : "#C8D2DA",
        position: "relative",
        transition: "background .2s",
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 3,
          left: value ? 23 : 3,
          width: 24,
          height: 24,
          borderRadius: "50%",
          background: "#fff",
          transition: "left .2s",
        }}
      />
    </button>
  );
}

export default function Settings({
  settings,
  onSave,
}: {
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
}) {
  const [draft, setDraft] = useState(settings);
  const [saved, setSaved] = useState(false);

  const set = <K extends keyof AppSettings>(k: K, v: AppSettings[K]) => {
    setDraft((d) => ({ ...d, [k]: v }));
    setSaved(false);
  };

  const standalone = window.matchMedia("(display-mode: standalone)").matches;

  return (
    <div className="screen">
      <div className="banner banner-note">
        이 앱은 <strong>UI 미리보기 전용</strong>입니다. 마이크와 서버를 쓰지 않고
        정해진 시나리오를 재생합니다. 실제 동작은 PC 버전에서 확인하세요.
      </div>

      <div className="card">
        <div className="card-title">표시</div>
        <div className="row">
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>음성으로 알리기</div>
            <div className="small">경고를 소리로 읽어줍니다</div>
          </div>
          <Toggle value={draft.enableVoiceWarning} onChange={(v) => set("enableVoiceWarning", v)} />
        </div>
        <div className="row">
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>큰 글씨</div>
            <div className="small">글자를 키웁니다</div>
          </div>
          <Toggle value={draft.largeText} onChange={(v) => set("largeText", v)} />
        </div>
        <button
          className="btn"
          style={{ background: "#F0F4F8", color: "#14202B", marginTop: 6, fontSize: 15 }}
          onClick={() =>
            speakWarning([
              "지금 통화에서 안전계좌 라는 말이 나왔습니다.",
              "안전계좌는 존재하지 않습니다.",
            ])
          }
          disabled={!ttsAvailable()}
        >
          경고 음성 미리 듣기
        </button>
      </div>

      <div className="card">
        <div className="card-title">설치 상태</div>
        <div className="row">
          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>홈 화면에 추가</div>
            <div className="small">
              {standalone
                ? "전체화면으로 실행 중입니다"
                : "공유 → 홈 화면에 추가하면 앱처럼 실행됩니다"}
            </div>
          </div>
          <span style={{ fontSize: 19, color: standalone ? "#2E7D52" : "#B0BAC5" }}>
            {standalone ? "✓" : "—"}
          </span>
        </div>
      </div>

      <div className="card">
        <div className="card-title">정보</div>
        <div className="row">
          <span style={{ fontSize: 15 }}>버전</span>
          <span className="mono small">미리내 모바일 0.1.0 (UI)</span>
        </div>
        <div className="row">
          <span style={{ fontSize: 15 }}>팀</span>
          <span className="small">Team Lumina · 전남대학교 인공지능학부</span>
        </div>
        <div className="row">
          <span style={{ fontSize: 15 }}>패턴 DB</span>
          <span className="mono small">8단계 182항</span>
        </div>
      </div>

      <button className="btn" style={{ background: saved ? "#2E7D52" : "#14202B" }} onClick={() => {
        onSave(draft);
        setSaved(true);
        setTimeout(() => setSaved(false), 1800);
      }}>
        {saved ? "✓ 저장했습니다" : "설정 저장"}
      </button>
      <button
        className="btn"
        style={{ background: "transparent", color: "#9BA8B5", fontSize: 14 }}
        onClick={() => setDraft(DEFAULT_SETTINGS)}
      >
        기본값으로
      </button>
    </div>
  );
}
