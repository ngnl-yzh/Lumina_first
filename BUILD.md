# 배포용 실행 파일 만들기

파이썬·노드 설치 없이 **더블클릭으로 도는** PC 실행본을 만든다.
시연 장비를 세팅하는 사람이 개발자가 아닐 수 있으므로 이 경로를 준비해 둔다.

## 만들기

```bash
# 1) 앱 화면 빌드 — EXE 안에 함께 들어간다
cd app-desktop
npm install
npm run build

# 2) EXE 빌드
cd ../server
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python -m PyInstaller mirinae.spec --noconfirm --clean
```

결과는 `server/dist/미리내/` 폴더다. 안에 `미리내.exe`가 있다.
**폴더째 옮겨야 한다.** exe만 빼내면 돌지 않는다.

## 왜 onedir인가

`onefile`로 묶으면 실행할 때마다 수 GB를 임시 폴더에 풀어서 시작에 수십 초가 걸린다.
시연 도구로는 못 쓴다. `onedir`은 첫 화면까지 몇 초다.

배포할 때는 폴더를 zip으로 압축해 넘긴다.

## 크기

| 항목 | 크기 |
|---|---|
| `미리내.exe` | 약 50 MB |
| 폴더 전체 | **약 850 MB** |

대부분이 PyTorch와 CTranslate2 바이너리다.
줄이려면 GPU 없는 장비 전용으로 torch를 더 가볍게 잡는 방법이 있지만,
정작 필요한 RTX 3060 장비에서 못 쓰게 되므로 하지 않았다.

## 모델은 들어 있지 않다

Whisper small(460 MB)과 딥보이스 탐지기(380 MB)는 EXE에 넣지 않는다.
첫 실행에 받아서 사용자 홈의 캐시에 저장되고, 두 번째부터는 오프라인으로 돈다.

**시연 전에 반드시 그 장비에서 한 번 실행해 캐시를 만들어 둘 것.**
현장에서 인터넷이 안 되면 모델을 못 받아 아무것도 안 된다.
이건 실제로 자주 나는 사고다.

## 경고 음성 뱅크도 같은 취급이다

```bash
cd server
.venv\Scripts\python build_tts_bank.py --list-voices   # ko-KR 음성 확인
.venv\Scripts\python build_tts_bank.py                 # 조각 91개 · 약 10 MB
```

WAV는 저장소에 넣지 않는다(재생성 가능하고 크다). `manifest.json`만 커밋한다.

**한국어 음성(예: Microsoft Heami)이 설치돼 있어야 한다.**
없으면 뱅크가 안 만들어지고, 앱은 화면 경고만으로 동작한다 —
한국어 음성이 없는 기기에서 영어 엔진이 한글을 읽는 것보다 그게 낫다고 판단해
그렇게 고쳐 두었다. 소리는 나는데 알아들을 수 없으면 고령층에게는 무음보다 나쁘다.

## 빌드하면서 걸린 것

### excludes를 추측으로 짜지 말 것

크기를 줄이겠다고 짐작으로 제외했다가 **두 번 깨졌다.** 빌드가 20분씩 걸리므로 비싼 실수다.

| 시도 | 결과 |
|---|---|
| `torch.testing`, `torch.distributions` 제외 | `No module named 'torch.testing'` — **torch 자체가 안 올라온다.** `torch/__init__.py`가 내부에서 import한다 |
| `tokenizers`, `transformers` 제외 | `No module named 'tokenizers'` — `faster_whisper.transcribe`가 둘 다 모듈 수준에서 import한다 |

두 번째가 특히 함정이었다. `transformers`는 딥보이스 탐지 전용이라고 생각했는데,
faster-whisper가 배치 추론 기능 때문에 직접 끌어온다.

**확인 방법** — launcher와 같은 경로를 import해보고 실제로 뭐가 올라오는지 센다.

```python
import sys
before = set(sys.modules)
# launcher가 하는 import를 그대로 따라한다 (STT는 모델까지 올려야 한다)
...
print(sorted({m.split(".")[0] for m in set(sys.modules) - before}))
```

135개가 나왔고, 그중 실제로 안 쓰이는 것만 `excludes`에 남겼다.

부수 효과로 `transformers`가 포함되므로 **EXE에서도 `--deepvoice`가 동작한다.**

### 그 밖에

**`typing` 백포트 패키지를 지워야 한다.**
PyInstaller가 거부한다. `pip uninstall typing`.
파이썬 3.5부터 표준 라이브러리에 들어간 것의 옛 백포트라 지워도 문제없다.

**빌드 전에 이전 프로세스를 죽이고 `dist`/`build`를 지운다.**
실행 중이던 EXE나 그 폴더에 들어가 있는 셸이 잠금을 걸면
`PermissionError: [WinError 32]`로 빌드가 통째로 실패한다.

**UPX 압축은 쓰지 않는다.** torch DLL이 깨지는 사례가 있다.

**콘솔 창을 남긴다.** 모델 다운로드 진행과 오류 메시지를 봐야 한다.
`--windowed`로 숨기면 실패했을 때 원인을 알 방법이 없다.
실제로 위 두 번의 실패도 콘솔 덕에 바로 원인을 알았다.

## 실행

```
미리내.exe                    # 브라우저가 자동으로 열린다
미리내.exe --whisper medium   # GPU 장비에서 전사 정확도를 올린다
미리내.exe --deepvoice        # 딥보이스 탐지 표시 (EXE에는 미포함, 소스 실행 필요)
미리내.exe --no-browser       # 브라우저를 열지 않는다
```

EXE는 두 포트를 쓴다.

- `localhost:8080` — 앱 화면
- `localhost:8765` — WebSocket 서버

## EXE에서 빠진 기능

| 기능 | 왜 |
|---|---|
| XTTS 복제 검증 | 별도 venv가 필요하고 모델이 1.8 GB다. 연구용이지 시연용이 아니다 |
| 실험 도구 | `sweep_params.py` · `benchmark_deepvoice.py` · `check_env.py` 등은 소스로 실행한다 |

딥보이스 탐지는 `transformers`가 어차피 포함되므로 **EXE에서도 `--deepvoice`로 켤 수 있다.**
다만 자체 측정 재현율이 18.8%라 기본값은 꺼짐이고, 켜도 위험도 점수는 바꾸지 않는다.

이 기능들이 필요하면 저장소를 받아 `server/README.md`대로 환경을 만든다.

## 폰 UI 배포

폰 버전은 정적 파일이라 EXE가 필요 없다.

```bash
cd app-mobile
npm run build       # dist/ 를 아무 정적 호스팅에나 올린다
```

GitHub Pages, Netlify, Vercel 어디든 된다. HTTPS만 되면 홈 화면에 추가해 쓸 수 있다.
