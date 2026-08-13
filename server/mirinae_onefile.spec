# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 스펙 — 미리내 PC 실행 파일.

onedir로 묶는다. onefile은 실행할 때마다 수 GB를 임시 폴더에 풀어서
시작이 수십 초 걸린다. 시연 도구로는 못 쓴다.

모델 가중치는 넣지 않는다.
  Whisper small 460 MB + 딥보이스 380 MB를 넣으면 배포 파일이 감당이 안 되고,
  어차피 첫 실행에 받아서 캐시되므로 두 번째부터는 오프라인으로 돈다.
  대신 **시연 전에 반드시 한 번 실행해 캐시를 만들어 둘 것.**
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)
WEBAPP = ROOT.parent / "app-desktop" / "dist"

datas = [
    # 빌드된 PC 앱 화면
    (str(WEBAPP), "webapp"),
    # 패턴 DB — 코드가 아니라 데이터라 별도로 넣어야 한다
    (str(ROOT / "mirinae" / "mode1" / "pattern_db.json"), "mirinae/mode1"),
]

# Resemblyzer는 pretrained.pt를 패키지 안에 들고 있다. 이게 빠지면 인코더가 안 뜬다.
datas += collect_data_files("resemblyzer")

hiddenimports = [
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "scipy.special.cython_special",
    "webrtcvad",
    "av",              # faster-whisper의 오디오 디코딩
    "tokenizers",      # faster-whisper의 Whisper 토크나이저
    "transformers",    # faster_whisper.transcribe가 모듈 수준에서 import한다
    "huggingface_hub",  # 모델 다운로드
    "safetensors",
]
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")

# ── excludes 원칙: 추측으로 넣지 말 것 ────────────────────────────────────────
#
# 크기를 줄이겠다고 짐작으로 제외했다가 두 번 깨졌다.
#
#   ① torch.testing 제외 → `No module named 'torch.testing'`
#      torch/__init__.py가 내부에서 import한다. torch 자체가 안 올라온다.
#   ② tokenizers·transformers 제외 → `No module named 'tokenizers'`
#      faster_whisper.transcribe가 둘 다 모듈 수준에서 import한다.
#      transformers는 딥보이스 탐지 전용이라고 짐작했는데 틀렸다.
#
# 무엇이 실제로 필요한지는 launcher와 같은 경로를 import해보고 확인한다
# (scratchpad/deps.py 방식). 아래는 그렇게 확인해 **실제로 안 쓰이는 것**만 남긴 것이다.
#
# 부수 효과 — transformers가 들어가므로 EXE에서도 --deepvoice 가 동작한다.
excludes = [
    "matplotlib", "tkinter", "PyQt5", "PySide2", "PIL.ImageQt",
    "IPython", "jupyter", "notebook",
    "pytest", "_pytest", "pyinstaller",
]

a = Analysis(
    ["launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ── 단일 파일 빌드 ────────────────────────────────────────────────────────────
#
# 폴더가 아니라 **EXE 하나**로 묶는다. 파일 하나만 건네면 되지만 대가가 있다 —
# 실행할 때마다 1 GB 가까이를 임시 폴더에 풀어야 해서 **첫 화면까지 2~4분**
# 걸린다. 폴더 배포판(mirinae.spec)은 40초다.
#
# 공유가 목적이면 이쪽, 시연 당일 속도가 목적이면 그쪽을 쓴다.
exe = EXE(
    pyz,
    a.scripts,
    [],
    a.binaries,
    a.datas,
    name="미리내_단일파일",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX는 torch DLL을 망가뜨리는 사례가 있다
    console=True,       # 콘솔을 남긴다 — 모델 다운로드 진행과 오류를 봐야 한다
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# COLLECT 없음 — 단일 파일이라 풀어 놓을 폴더가 없다.
