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

/**
 * 한국어 음성만 쓴다. 없으면 **아무것도 재생하지 않는다.**
 *
 * 예전에는 `voices.find(ko) ?? voices[0]`이었다. 한국어 음성이 없는 기기에서
 * 영어 엔진이 "안전계좌는 존재하지 않습니다"를 읽는다는 뜻이다.
 * 고령층에게는 무음보다 나쁘다 — 소리는 나는데 알아들을 수 없으니
 * "뭔가 잘못됐다"는 인상만 남고 정작 경고 내용은 전달되지 않는다.
 * 화면 경고는 그대로 뜨므로, 잘못 읽느니 화면에 맡기는 편이 낫다.
 */
function pickKoreanVoice(): SpeechSynthesisVoice | null {
  if (voice) return voice;
  const voices = window.speechSynthesis?.getVoices?.() ?? [];
  voice = voices.find((v) => v.lang?.toLowerCase().startsWith("ko")) ?? null;
  return voice;
}

/**
 * Chrome은 `getVoices()`가 처음 호출에서 빈 배열을 준다 — 목록이 비동기로 로드된다.
 * 경고가 뜨는 순간이 하필 그때면 음성이 안 나온다.
 * 앱 시작 시 한 번 걸어두고, 목록이 채워지면 다시 고른다.
 */
export function warmUpVoices(): void {
  if (!ttsAvailable()) return;
  window.speechSynthesis.getVoices();
  window.speechSynthesis.addEventListener?.("voiceschanged", () => {
    voice = null;
    pickKoreanVoice();
  });
}

export function ttsAvailable(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** 한국어 음성이 실제로 있는가. 없으면 화면 경고만으로 동작한다. */
export function koreanVoiceAvailable(): boolean {
  return ttsAvailable() && pickKoreanVoice() !== null;
}

/** 경고 문장들을 순서대로 읽는다. */
export function speakWarning(lines: string[]): void {
  if (!ttsAvailable()) return;
  // 한국어 음성이 없으면 재생하지 않는다 — 영어 엔진이 한글을 읽는 것보다 낫다.
  const v = pickKoreanVoice();
  if (!v) return;

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
    u.voice = v;
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
