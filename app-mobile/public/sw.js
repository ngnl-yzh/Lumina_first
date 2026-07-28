/**
 * 서비스 워커 — 앱 셸만 캐시한다.
 *
 * 오디오·WebSocket은 절대 캐시하지 않는다. 실시간 데이터를 캐시하면
 * 지난 통화의 결과가 다시 나올 수 있고, 그건 이 앱에서 가장 위험한 종류의 버그다.
 */

const CACHE = "mirinae-shell-v1";
const SHELL = ["/", "/index.html", "/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // 같은 출처의 GET만 다룬다. WebSocket과 외부 요청은 건드리지 않는다.
  if (e.request.method !== "GET" || url.origin !== location.origin) return;

  // 네트워크 우선 — 개발 중 코드를 고쳤는데 캐시된 옛 화면이 뜨면 디버깅이 지옥이 된다.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => undefined);
        return res;
      })
      .catch(() => caches.match(e.request).then((r) => r || caches.match("/index.html"))),
  );
});

