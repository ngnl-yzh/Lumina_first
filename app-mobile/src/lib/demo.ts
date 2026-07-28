/**
 * 시연용 시나리오 재생 — **폰 버전은 UI 전용이다.**
 *
 * 서버도 마이크도 쓰지 않는다. 최종 제품이 어떤 모습인지 보여주는 것이 목적이라
 * 정해진 대본을 시간에 맞춰 흘려보낸다. 서버가 없어도, 인터넷이 없어도 돌아간다.
 *
 * 실제로 동작하는 버전은 `app-desktop`이다. 그쪽은 마이크에서 raw PCM을 받아
 * WebSocket으로 서버에 보내고 전사·채점·섭동을 진짜로 계산한다.
 *
 * 대본은 D07 §04 시연 순서를 그대로 따른다 — 기관사칭 A 경로로 S1→S2→S3→S4 전개.
 * S4(고립 유도)에서 개입이 걸리는 것이 이 앱의 핵심 장면이다.
 */

export interface DemoLine {
  atMs: number;
  text: string;
  stage: string;
  matched: string[];
  score: number;
  level: "안전" | "주의" | "위험";
  criticals?: string[];
}

export const DEMO_ROUTE = "A · 정부기관 사칭";

/**
 * 점수는 서버 스코어러(mirinae.mode1.scorer)가 같은 문장에 대해 내는 값을 옮긴 것이다.
 * UI만 보여주더라도 숫자는 실제 알고리즘과 어긋나지 않아야 한다.
 */
export const DEMO_SCRIPT: DemoLine[] = [
  {
    atMs: 2600,
    text: "여보세요, 서울중앙지방검찰청 수사관 김민수입니다.",
    stage: "S1",
    matched: ["서울중앙지검", "수사관"],
    score: 0.19,
    level: "안전",
  },
  {
    atMs: 7400,
    text: "귀하 명의가 도용되어 대포통장에 이용된 것이 확인됐습니다.",
    stage: "S2",
    matched: ["명의가 도용", "대포통장"],
    score: 0.34,
    level: "안전",
  },
  {
    atMs: 12800,
    text: "사안이 심각해서 현재 체포영장이 발부된 상태입니다.",
    stage: "S3",
    matched: ["체포영장"],
    score: 0.52,
    level: "주의",
  },
  {
    atMs: 18600,
    text: "수사 기밀이니 가족에게도 말하지 마십시오.",
    stage: "S4",
    matched: ["수사 기밀", "가족에게도 말하지"],
    score: 0.75,
    level: "위험",
    criticals: ["C2", "C4"],
  },
];

export const DEMO_WARNING = {
  quote: "지금 통화에서 “가족에게도 말하지” 라는 말이 나왔습니다.",
  counter: [
    "수사기관은 가족에게 알리지 말라고 하지 않습니다.",
    "수사기관은 전화로 돈을 요구하지 않습니다.",
  ],
  control: "지금 전화를 끊으셔도 아무 일도 생기지 않습니다.",
  crossCheck: "끊고 나서 112로 직접 전화해 확인하세요.",
  action: "확인이 어려우면 112로 전화하세요.",
};

/** 모드 2 시연 — 12초 녹음에 11개 청크. */
export const DEMO_M2 = {
  durationSec: 12,
  chunkCount: 11,
  chunkMs: 1000, // 홉 1.0초마다 청크 하나
  processMs: 900, // 전송 → 처리 완료까지
  finalSnr: 20.8,
  finalSrs: 0.31,
};

/** 대본을 타이머로 흘려보낸다. 반환값을 호출하면 남은 타이머가 정리된다. */
export function playScript<T>(
  items: Array<T & { atMs: number }>,
  onItem: (item: T, index: number) => void,
  onEnd?: () => void,
  tailMs = 1200,
): () => void {
  const timers: Array<ReturnType<typeof setTimeout>> = [];
  items.forEach((item, i) => {
    timers.push(setTimeout(() => onItem(item, i), item.atMs));
  });
  if (onEnd) {
    const last = items.length ? items[items.length - 1].atMs : 0;
    timers.push(setTimeout(onEnd, last + tailMs));
  }
  return () => timers.forEach(clearTimeout);
}
