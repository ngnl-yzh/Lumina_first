# 다른 PC에서 실행하기 (RTX 장비 포함)

`BUILD.md`는 배포용 EXE를 만드는 문서다. 이 문서는 **소스로 돌리는** 절차다.

---

## 0. 받기

```bash
git clone https://github.com/ngnl-yzh/Lumina_first.git
cd Lumina_first
```

git이 없으면 GitHub 페이지 → **Code ▾ → Download ZIP**.

저장소에는 **코드만** 들어 있다. 아래 셋은 각 PC에서 만든다 —
크고 재생성 가능해서 일부러 뺐다.

| 빠진 것 | 만드는 법 |
|---|---|
| Python 가상환경 `.venv` | 1단계 |
| Whisper·XTTS 모델 캐시 | 첫 실행에 자동 내려받음 |
| 경고 음성 뱅크 `tts_bank/*.wav` | 4단계 |

---

## 1. 서버 환경

Python **3.11**이 필요하다. 3.12 이상은 일부 의존성이 아직 안 맞는다.

```bash
cd server
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

### RTX 장비라면 여기서 torch를 CUDA 빌드로 갈아끼운다

`requirements.txt`의 torch는 CPU 빌드다. **이 한 줄을 안 하면 GPU를 안 쓴다.**

```bash
.venv\Scripts\python -m pip uninstall -y torch
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

확인:

```bash
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

`True RTX ...`가 나와야 한다. `False`면 위 설치가 안 먹은 것이다.
코드는 device를 자동 판별하므로 이것만 맞으면 나머지는 그대로 돌아간다.

> **함정** — `pip install -r requirements.txt`를 나중에 다시 돌리면
> CPU 빌드로 되돌아간다. 순서를 지킬 것.

---

## 2. 앱 환경

Node 18 이상.

```bash
cd app-desktop
npm install
```

---

## 3. 실행 — 터미널 두 개

**터미널 1 · 서버**

```bash
cd server
.venv\Scripts\python ws_server.py --port 8765
```

`예열 완료`가 뜨면 준비된 것이다. 첫 실행은 모델을 받느라 몇 분 걸린다.

**터미널 2 · 앱**

```bash
cd app-desktop
npm run dev
```

브라우저에서 **http://localhost:5174**

---

## 4. 경고 음성 뱅크 (선택, 시연 전 권장)

```bash
cd server
.venv\Scripts\python build_tts_bank.py --list-voices   # ko-KR 음성이 있는지 확인
.venv\Scripts\python build_tts_bank.py
```

한국어 음성(예: Microsoft Heami)이 없으면 만들어지지 않는다.
그 경우 앱은 **화면 경고만** 띄운다 — 영어 엔진이 한글을 읽는 것보다 그게 낫다고 판단해
그렇게 만들어 두었다.

---

## 5. GPU에서 바뀌는 것

코드가 자동으로 감지하지만, **성능 설정은 손으로 올려야 값이 산다.**

### 모드 1 — STT 모델을 키운다

CPU에서는 `small`이 기본이다. GPU면 `medium`이 실시간을 유지하면서 전사가 눈에 띄게 좋아진다.

```bash
.venv\Scripts\python ws_server.py --port 8765 --whisper medium
```

실사용에서 나왔던 오인식("카드가 1시 정지되어", "해외 결치 시도", "자산"→"사산")이
줄어드는지 확인해 볼 것. CPU에서는 3배 느려져 못 썼던 선택지다.

### 모드 2 — PGD 스텝을 기본값으로 되돌린다

CPU에서는 실시간을 맞추려고 스텝을 깎아야 했다. 실측:

| 스텝 | SRS(낮을수록 강함) | CPU 초/청크 | 실시간(예산 1.0초) |
|---|---|---|---|
| 15 | 0.603 | 0.91 | 경계선 |
| 20 | 0.567 | 1.36 | 초과 |
| 200 | 0.438 | 19.1 | ×19 |

**GPU에서는 200스텝을 그대로 쓴다.** `worker.py`의 자동 감축 사다리가 있으므로
큐가 밀리지 않으면 알아서 200으로 돈다. 별도 설정이 필요 없다.

오프라인 보호도 훨씬 빨라진다.

```bash
.venv\Scripts\python protect.py 목소리.wav -o out/demo --steps 200 --seconds 0
```

### XTTS 복제 검증까지 하려면 (선택)

의존성이 충돌해서 환경을 나눠 놨다. 복제 검증을 안 하면 만들 필요 없다.

```bash
cd server
py -3.11 -m venv .venv-xtts
.venv-xtts\Scripts\python -m pip install -r requirements-xtts.txt
.venv-xtts\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> `coqui-tts`는 torch를 자동으로 끌어오지 않는다. 위처럼 따로 설치해야 한다.

---

## 6. 잘 도는지 확인

```bash
cd server
.venv\Scripts\python -m pytest tests -q          # 86건 통과해야 정상
.venv\Scripts\python eval_mode1.py               # 탐지율·오탐률
```

서버를 띄운 상태에서 **오디오부터 개입까지** 전 경로를 한 번에 돌려볼 수 있다.

```bash
.venv\Scripts\python e2e_test.py --all
```

`4/4 통과`가 나오면 전 경로가 정상이다.
(사기 2건은 개입, 가족 금융 대화 2건은 위험도 0.000)

---

## 자주 나는 문제

**`서버 연결 실패`**
→ 서버가 안 떠 있거나 `예열 완료` 전이다. 터미널 1을 확인하고 브라우저를 새로고침한다.

**`ModuleNotFoundError: pkg_resources`**
→ setuptools가 81 이상이다. `requirements.txt`에 `setuptools<81`로 핀이 걸려 있으니
가상환경을 다시 만들면 해결된다.

**마이크가 안 잡힘**
→ 브라우저 주소가 `localhost`인지 확인한다. `127.0.0.1`이나 IP로 접속하면
secure context가 아니라 마이크 권한이 막힌다.

**GPU를 안 씀 (`torch.cuda.is_available()` → False)**
→ 1단계의 CUDA torch 설치를 건너뛰었거나, 그 뒤에 `requirements.txt`를 다시 설치했다.

**첫 실행이 오래 걸림**
→ 모델을 받는 중이다. **시연 장비에서는 반드시 미리 한 번 돌려 캐시를 만들어 둘 것.**
현장에서 인터넷이 안 되면 아무것도 안 된다.
