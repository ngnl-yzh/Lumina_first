export type Tab = "mode1" | "mode2" | "settings";
export type ConnState = "disconnected" | "connecting" | "connected" | "error";

export interface AppSettings {
  serverUrl: string;
  enableVoiceWarning: boolean;
  largeText: boolean;
}

/**
 * 서버 주소 기본값을 현재 접속 호스트에서 만든다.
 * 폰에서 노트북 IP로 접속하면 서버도 같은 IP일 가능성이 높다 —
 * 시연 현장에서 IP를 손으로 입력하는 단계를 없애는 것이 목적이다.
 */
export function defaultServerUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.hostname}:8765`;
}

export const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: defaultServerUrl(),
  enableVoiceWarning: true,
  largeText: false,
};

const KEY = "mirinae.settings";

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<AppSettings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: AppSettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* 사파리 프라이빗 모드에서는 실패한다. 치명적이지 않다. */
  }
}

export const STAGE_LABEL: Record<string, string> = {
  S1: "권위", S2: "연루", S3: "공포", S4: "고립",
  S5: "지시", S6: "압박", S7: "가족", S8: "대출",
};

export const ALL_STAGES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"];
