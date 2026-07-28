"""인코더 검증 — mel 일치와 gradient 도달.

D09 §5.1의 gradcheck.py 명세에 대응한다. 이 두 테스트가 통과해야
"우리가 최적화하는 대상이 실제 Resemblyzer다"라고 말할 수 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirinae.config import SAMPLE_RATE                                # noqa: E402
from mirinae.encoder import SpeakerEncoder, cosine_similarity         # noqa: E402
from mirinae.gradcheck import check_gradient                          # noqa: E402
from test_sanity import synth_speech                                  # noqa: E402


@pytest.fixture(scope="module")
def encoder() -> SpeakerEncoder:
    return SpeakerEncoder(device=torch.device("cpu"))


def test_mel_matches_resemblyzer(encoder):
    """torch mel이 librosa mel과 같은 값을 내야 한다.

    여기가 어긋나면 실제 인코더가 아닌 것을 공략하게 되고, 섭동은 어디에도 통하지 않는다.
    """
    wav = synth_speech(2.0).numpy()
    r = encoder.verify_against_resemblyzer(wav)

    assert r["n_frames_ref"] == r["n_frames_got"], "프레임 수 불일치 — center/pad 설정 확인"
    assert r["corr"] > 0.9999, f"상관 {r['corr']:.6f}"
    assert r["rel_err"] < 1e-4, f"상대오차 {r['rel_err']:.2e}"


def test_embedding_is_unit_norm(encoder):
    """임베딩은 L2 정규화되어 있어야 한다 — 코사인 유사도의 전제."""
    x = synth_speech(2.0)
    with torch.no_grad():
        e = encoder(x)
    assert e.shape == (256,)
    assert float(torch.norm(e)) == pytest.approx(1.0, abs=1e-5)


def test_self_similarity_is_one(encoder):
    """같은 파형끼리는 SRS = 1.0. C-C 무섭동 대조군이 확인하는 성질이다."""
    x = synth_speech(2.0)
    with torch.no_grad():
        s = float(cosine_similarity(encoder(x), encoder(x)))
    assert s == pytest.approx(1.0, abs=1e-5)


def test_gradient_reaches_waveform(encoder):
    """파형에 gradient가 실제로 닿아야 한다. 0이면 PGD가 아무것도 못 한다."""
    x = synth_speech(2.0)
    delta = torch.zeros_like(x, requires_grad=True)
    with torch.no_grad():
        ref = encoder(x).detach()

    loss = -(1.0 - cosine_similarity(encoder(x + delta), ref))
    (g,) = torch.autograd.grad(loss, delta)

    assert not bool(torch.isnan(g).any()), "gradient에 NaN"
    assert float(g.abs().max()) > 0, "gradient가 전부 0 — 경로가 끊겼다"


def test_gradient_matches_numeric(encoder):
    """수치미분 대조 — 상대오차 < 1e-3 · 상관 > 0.999 (D09 §6.2 test_gradient)."""
    x = synth_speech(2.0)
    report = check_gradient(encoder, x, n_probes=32)
    assert report.correlation > 0.999, str(report)
    assert report.rel_error < 1e-3, str(report)


def test_partial_slices_for_2s_chunk(encoder):
    """2.0초 청크 → 온전한 임베딩 1.25개. 청크 길이 확정의 근거다."""
    n_frames = encoder.mel_spectrogram(torch.zeros(int(SAMPLE_RATE * 2.0))).shape[0]
    slices = encoder._partial_slices(n_frames)
    assert len(slices) == 2, f"슬라이스 {len(slices)}개 (프레임 {n_frames})"
    assert slices[0] == (0, 160)


def test_short_chunk_does_not_crash(encoder):
    """1.6초 미만은 임베딩이 부정확해지지만 죽지는 않아야 한다."""
    with torch.no_grad():
        e = encoder(synth_speech(1.0))
    assert e.shape == (256,)
    assert not bool(torch.isnan(e).any())
