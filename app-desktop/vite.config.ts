import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * PC 버전은 localhost에서 돌린다.
 * 브라우저는 localhost를 secure context로 취급하므로 마이크가 그냥 열린다.
 * 폰 버전에 필요했던 자체 서명 인증서 단계가 통째로 사라진다 —
 * 시연 당일 실패 요인이 하나 줄어든다는 뜻이다.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    open: true,
  },
  build: {
    target: "es2020",
  },
});
