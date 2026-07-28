/**
 * 경고 음성 재생.
 *
 * 통화 중 화면 팝업은 귀에 대고 있으면 보이지 않는다. 그래서 **음성이 주 매체**다.
 * 고령층 전달 설계(D08 §5.2)를 그대로 옮겼다.
 *   - 발화 속도를 평상보다 느리게 (인지 처리 시간 확보)
 *   - 핵심 문장은 2회 반복 (1회는 놓칠 수 있음)
 *   - 용어는 사기범이 쓴 말 그대로 (번역하면 연결이 끊긴다 — 이건 서버가 보장한다)
 *
 * 설계도는 사전 생성 TTS 뱅크를 쓰라고 한다. 런타임 지연이 0이고 발음 품질도
 * 미리 검수할 수 있어서다. 현재는 브라우저 SpeechSynthesis로 동작하게 해 두었고,
 * 뱅크가 준비되면 `playBank()`로 교체하면 된다 — 인터페이스는 같다.
 */

const RATE = 0.85; // 평상 대비 느리게
const REPEAT_FIRST = 2; // 첫 문장(인용)은 두 번 읽는다

let voice: SpeechSynthesisVoice | null = null;

function pickKoreanVoice(): SpeechSynthesisVoice | null {
  if (voice) return voice;
  const voices = window.speechSynthesis?.getVoices?.() ?? [];
  voice = voices.find((v) => v.lang?.startsWith("ko")) ?? voices[0] ?? null;
  return voice;
}

export function ttsAvailable(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** 경고 문장들을 순서대로 읽는다. */
export function speakWarning(lines: string[]): void {
  if (!ttsAvailable()) return;

  const synth = window.speechSynthesis;
  synth.cancel(); // 이전 재생이 남아 있으면 겹친다

  const queue: string[] = [];
  lines.forEach((line, i) => {
    if (!line?.trim()) return;
    const times = i === 0 ? REPEAT_FIRST : 1;
    for (let k = 0; k < times; k++) queue.push(line);
  });

  for (const line of queue) {
    const u = new SpeechSynthesisUtterance(line);
    u.lang = "ko-KR";
    u.rate = RATE;
    u.pitch = 1.0;
    const v = pickKoreanVoice();
    if (v) u.voice = v;
    synth.speak(u);
  }
}

export function stopSpeaking(): void {
  if (ttsAvailable()) window.speechSynthesis.cancel();
}

/**
 * iOS는 사용자 제스처 없이 음성 합성이 시작되지 않는다.
 * 모니터링을 시작하는 탭에서 무음을 한 번 재생해 권한을 열어 둔다.
 * 이걸 안 하면 정작 경고가 필요한 순간에 소리가 안 난다.
 */
export function primeSpeech(): void {
  if (!ttsAvailable()) return;
  const u = new SpeechSynthesisUtterance(" ");
  u.volume = 0;
  window.speechSynthesis.speak(u);
}
