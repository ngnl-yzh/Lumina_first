@echo off
chcp 65001 >nul
title 미리내 — 시연용 실행기

REM ============================================================
REM  미리내 시연 실행기
REM
REM  EXE는 모드 1(보이스피싱 탐지)만 담는다. 모드 2의 복제 모델
REM  (XTTS-v2 · GPT-SoVITS)은 수 GB이고 실행 시점에 내려받아야 해서
REM  EXE에 넣지 않았다. 그래서 모드 2는 이 배치가 파이썬으로 직접 띄운다.
REM ============================================================

cd /d "%~dp0server"

echo.
echo   미리내 — 무엇을 시연합니까?
echo.
echo     [1] 모드 1만 (보이스피싱 탐지)      기동 빠름 · 20초
echo     [2] 모드 1 + 모드 2 (딥보이스 방지)  기동 1~2분 · 보호 15초
echo.
set /p MODE="   번호 입력 (기본 2): "
if "%MODE%"=="" set MODE=2

echo.
if "%MODE%"=="1" (
    echo   [서버] 모드 1 전용으로 기동합니다...
    start "미리내 서버" cmd /k ".venv\Scripts\python.exe ws_server.py --port 8765"
) else (
    echo   [서버] GPT-SoVITS를 올립니다. 1~2분 걸립니다. 창을 닫지 마세요.
    start "미리내 서버" cmd /k ".venv-xtts\Scripts\python.exe ws_server.py --port 8765 --cloner gsv --steps 400 --time-budget 120"
)

echo   [앱] 화면을 띄웁니다...
cd /d "%~dp0app-desktop"
start "미리내 앱" cmd /k "npm run dev"

echo.
echo   ------------------------------------------------------------
echo    서버 창에 "서버 시작: ws://0.0.0.0:8765" 가 뜬 뒤에
echo    브라우저에서 http://localhost:5173 을 여세요.
echo.
echo    그 전에 앱에서 버튼을 누르면 "서버 대기"로만 보입니다.
echo   ------------------------------------------------------------
echo.
pause
