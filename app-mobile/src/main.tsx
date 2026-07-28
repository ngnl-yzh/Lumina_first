import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// 홈 화면에 추가했을 때 오프라인에서도 셸이 뜨게 한다.
// 서버가 없으면 분석은 못 하지만, 앱이 열리지 않는 것과는 다르다.
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      /* 자체 서명 인증서에서는 등록이 실패할 수 있다. 앱 동작에는 지장 없다. */
    });
  });
}
