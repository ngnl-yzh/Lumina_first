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
]
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")

# 쓰지 않는 무거운 것들을 뺀다. 넣으면 배포 크기가 배로 뛴다.
excludes = [
    "matplotlib", "tkinter", "PyQt5", "PySide2", "PIL.ImageQt",
    "IPython", "jupyter", "notebook", "pytest", "pyinstaller",
    # 딥보이스 탐지용. 재현율 18.8%라 기본값이 꺼짐이고,
    # transformers를 넣으면 1 GB 이상 늘어난다. 필요하면 소스로 실행할 것.
    "transformers", "tokenizers",
]

# torch 하위 모듈은 빼면 안 된다.
# torch/__init__.py가 torch.testing 등을 내부에서 import하므로
# 제외하면 `No module named 'torch.testing'`으로 torch 자체가 안 올라온다.
# 실제로 이 함정에 한 번 빠졌다. 크기를 줄이겠다고 건드릴 곳이 아니다.

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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="미리내",
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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="미리내",
)
