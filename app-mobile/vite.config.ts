import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

/**
 * iOS Safari는 HTTPS가 아니면 마이크를 열지 않는다. localhost도 폰에서는 해당이 없다.
 * 그래서 개발 서버에 자체 서명 인증서를 붙인다.
 *   npm run dev:https  → https://<노트북IP>:5173 으로 폰에서 접속
 * 첫 접속에서 "안전하지 않음" 경고가 뜨는데 계속 진행하면 된다.
 * 시연 당일 실패 요인을 줄이려면 같은 WiFi에 서버를 두고 미리 한 번 수락해 둘 것.
 */
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  build: {
    target: "es2020",
  },
});
