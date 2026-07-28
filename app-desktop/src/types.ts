export type Tab = "mode1" | "mode2" | "settings";
export type ConnState = "disconnected" | "connecting" | "connected" | "error";

export interface AppSettings {
  serverUrl: string;
  enableVoiceWarning: boolean;
}

/** PC 버전은 서버를 같은 기계에서 돌리는 것이 기본이다. */
export function defaultServerUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.hostname || "localhost"}:8765`;
}

export const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: defaultServerUrl(),
  enableVoiceWarning: true,
};

const KEY = "mirinae.desktop.settings";

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    return raw
      ? { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<AppSettings>) }
      : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: AppSettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* 무시 */
  }
}

export const STAGE_LABEL: Record<string, string> = {
  S1: "권위확립", S2: "연루통보", S3: "공포조성", S4: "고립유도",
  S5: "행동지시", S6: "시간압박", S7: "가족사칭", S8: "대출사기",
};

export const STAGE_COLOR: Record<string, string> = {
  S1: "#9A6410", S2: "#B85C20", S3: "#CC4A20", S4: "#A62F5B",
  S5: "#CC4A20", S6: "#9A6410", S7: "#B85C20", S8: "#9A6410",
};

export const ALL_STAGES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"];
