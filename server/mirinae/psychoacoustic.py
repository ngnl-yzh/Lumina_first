"""심리음향 마스킹 임계값 — Bark 25대역 (Zwicker & Fastl).

역할 분담이 중요하다. **손실이 섭동의 방향을 정하고, 마스킹이 크기를 정한다.**
큰 소리 옆의 작은 소리는 들리지 않는다는 성질을 이용해
"여기까지는 안 들린다"는 프레임별·주파수별 상한을 계산한다.

출력은 STFT 크기와 직접 비교 가능한 선형 magnitude 상한이다. 즉

    abs(stft(delta)) <= threshold * ratio

가 그대로 불변식이 된다. (D09 §6.2 test_masking_invariant)
"""

from __future__ import annotations

import numpy as np
import torch

from .config import EPS, HOP_LENGTH, N_FFT, SAMPLE_RATE

# Zwicker 임계대역 경계 (Hz). 25개 대역 → 경계 26개.
# 16 kHz 표본화에서는 Nyquist가 8 kHz이므로 상위 대역 일부는 비어 있다.
# 설계도가 "Bark 25대역"이라 표를 전부 싣고, 사용 가능한 대역만 자동으로 잡힌다.
BARK_EDGES_HZ = np.array(
    [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
     2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
     15500, 20500],
    dtype=np.float64,
)

# 전대역 신호를 96 dB SPL로 두는 관례(Qin et al. 2019). dB SPL ↔ STFT 크기 변환의 기준점.
FULL_SCALE_DB = 96.0

# 마스킹 오프셋 O(b) = a*(14.5+b) + (1-a)*5.5 의 톤성 계수.
# a=1이 순음 마스커, a=0이 잡음 마스커. 음성은 둘이 섞여 있어 중간값을 쓴다.
TONALITY_ALPHA = 0.3


def hz_to_bark(f_hz: np.ndarray) -> np.ndarray:
    """Zwicker & Terhardt 근사식."""
    f = np.asarray(f_hz, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def absolute_threshold_db(f_hz: np.ndarray) -> np.ndarray:
    """절대 가청 임계값 ATH (dB SPL).

    아무 마스커가 없어도 이 아래로는 들리지 않는다. 마스킹 임계값의 하한이 된다.
    """
    f = np.maximum(np.asarray(f_hz, dtype=np.float64), 20.0) / 1000.0
    ath = (
        3.64 * f ** -0.8
        - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2)
        + 1e-3 * f ** 4
    )
    # 초저역에서 발산하므로 상한을 둔다. 어차피 300 Hz 미만은 대역 제한으로 잘려 나간다.
    return np.clip(ath, -20.0, 80.0)


def _spreading_matrix(bark_centers: np.ndarray) -> np.ndarray:
    """대역 간 마스킹 확산 행렬 (선형 power 도메인).

    Schroeder 확산 함수. dz = 0에서 약 0 dB이므로 자기 대역은 그대로 통과한다.
    """
    dz = bark_centers[None, :] - bark_centers[:, None]   # [masker, maskee]
    sf_db = (
        15.81
        + 7.5 * (dz + 0.474)
        - 17.5 * np.sqrt(1.0 + (dz + 0.474) ** 2)
    )
    return 10.0 ** (sf_db / 10.0)


class MaskingModel:
    """대역 분할·확산 행렬처럼 신호와 무관한 것들을 미리 계산해 재사용한다.

    청크마다 새로 만들면 200스텝 × 청크 수만큼 낭비가 난다.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        device: torch.device | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device or torch.device("cpu")

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        self.n_bins = len(freqs)

        # 각 FFT bin이 속한 Bark 대역
        edges = BARK_EDGES_HZ[BARK_EDGES_HZ <= sample_rate / 2 + 1e-6]
        if edges[-1] < sample_rate / 2:
            edges = np.append(edges, sample_rate / 2)
        self.n_bands = len(edges) - 1
        band_idx = np.clip(np.digitize(freqs, edges) - 1, 0, self.n_bands - 1)

        # bin → band 집계 행렬 (n_bands, n_bins)
        agg = np.zeros((self.n_bands, self.n_bins), dtype=np.float64)
        agg[band_idx, np.arange(self.n_bins)] = 1.0
        bins_per_band = np.maximum(agg.sum(axis=1), 1.0)

        centers_hz = 0.5 * (edges[:-1] + edges[1:])
        spread = _spreading_matrix(hz_to_bark(centers_hz))
        offset_db = (
            TONALITY_ALPHA * (14.5 + np.arange(self.n_bands))
            + (1.0 - TONALITY_ALPHA) * 5.5
        )

        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        self.agg = t(agg)                      # (n_bands, n_bins)
        self.scatter = t(agg.T)                # (n_bins, n_bands)
        self.bins_per_band = t(bins_per_band)  # (n_bands,)
        self.spread = t(spread)                # (n_bands, n_bands)
        self.offset_db = t(offset_db)          # (n_bands,)
        self.ath_db = t(absolute_threshold_db(freqs))  # (n_bins,)
        self.window = torch.hann_window(n_fft, device=self.device)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window.to(x.dtype if x.is_floating_point() else torch.float32),
            center=True, pad_mode="constant", return_complex=True,
        )

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True, length=length,
        )

    def threshold(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """깨끗한 신호 x로부터 프레임별 마스킹 임계값을 계산한다.

        :return: (thr_mag, spl_offset_db)
            thr_mag — (n_bins, n_frames) STFT magnitude 상한. abs(stft(δ))와 직접 비교한다.
            spl_offset_db — dB SPL 환산에 쓴 오프셋. 지표 계산에서 재사용한다.
        """
        with torch.no_grad():
            mag = self.stft(x).abs()                                  # (bins, frames)

            # STFT 크기 → dB SPL. 최대값이 96 dB이 되도록 평행이동한다.
            peak = torch.clamp(mag.max(), min=EPS)
            spl_offset = FULL_SCALE_DB - 20.0 * torch.log10(peak)
            psd_db = 20.0 * torch.log10(torch.clamp(mag, min=EPS)) + spl_offset

            # 대역 power 합 → 확산 → 오프셋 차감
            band_pow = self.agg @ (10.0 ** (psd_db / 10.0))           # (bands, frames)
            spread_pow = self.spread @ band_pow
            band_thr_db = 10.0 * torch.log10(torch.clamp(spread_pow, min=EPS))
            band_thr_db = band_thr_db - self.offset_db[:, None]

            # 대역 임계값을 bin 하나당으로 나눠 편다
            band_thr_db = band_thr_db - 10.0 * torch.log10(self.bins_per_band)[:, None]
            bin_thr_db = self.scatter @ band_thr_db                   # (bins, frames)

            # ATH가 하한. 마스커가 없어도 이만큼은 안 들린다
            bin_thr_db = torch.maximum(bin_thr_db, self.ath_db[:, None])

            # 다시 STFT magnitude 단위로
            thr_mag = 10.0 ** ((bin_thr_db - spl_offset) / 20.0)
            return thr_mag, float(spl_offset)


def masking_threshold(
    x: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """일회성 호출용 편의 함수. 반복 호출에는 MaskingModel을 재사용할 것."""
    model = MaskingModel(sample_rate, n_fft, hop_length, device=x.device)
    thr, _ = model.threshold(x)
    return thr
