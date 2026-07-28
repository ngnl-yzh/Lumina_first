"""청크 분할과 Hann overlap-add 조립 (D09 §4.3).

청크 방식은 실시간을 위한 타협이 아니다. **발췌 공격에 더 강한 구조**다.
공격자가 20초 녹음에서 어느 6초를 참조로 쓸지 우리는 모른다.
발화 전체를 한 번에 최적화하면 *전체 평균 임베딩*이 흔들리도록 맞춰지므로
특정 구간만 잘라내면 보호가 약해질 수 있다.
청크별 최적화 + 50% 겹침은 **모든 2초 창이 각각 보호**된다.
"""

from __future__ import annotations

import torch

from .config import CHUNK_SEC, EPS, HOP_SEC, SAMPLE_RATE


def chunk_bounds(n_samples: int, chunk: int, hop: int) -> list[int]:
    """각 청크의 시작 위치. 마지막 청크는 0으로 채워 길이를 맞춘다."""
    if n_samples <= chunk:
        return [0]
    starts = list(range(0, n_samples - chunk + 1, hop))
    if starts[-1] + chunk < n_samples:
        starts.append(n_samples - chunk)      # 꼬리를 버리지 않는다
    return starts


def split(
    x: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: float = CHUNK_SEC,
    hop_sec: float = HOP_SEC,
) -> tuple[list[torch.Tensor], list[int], int]:
    """:return: (청크 리스트, 시작 위치 리스트, 청크 길이)"""
    chunk = int(sample_rate * chunk_sec)
    hop = int(sample_rate * hop_sec)

    if x.shape[-1] < chunk:                    # 너무 짧으면 0으로 채워 한 청크로
        x = torch.nn.functional.pad(x, (0, chunk - x.shape[-1]))

    starts = chunk_bounds(x.shape[-1], chunk, hop)
    return [x[s:s + chunk] for s in starts], starts, chunk


def overlap_add(
    pieces: list[torch.Tensor],
    starts: list[int],
    length: int,
    chunk: int,
) -> torch.Tensor:
    """Hann 창 overlap-add. 창 합으로 나눠 진폭을 보존한다.

    인접 청크의 섭동이 불연속이면 경계에서 클릭음이 난다. Hann 창이 그것을 없앤다.
    양 끝은 겹칠 상대가 없어 창 보정이 불완전한데,
    **창 합으로 나누는 방식**이 그 경우까지 한 번에 처리한다 —
    끝에서는 분모가 반쪽 창이 되어 자동으로 진폭이 복원된다.
    """
    device = pieces[0].device
    dtype = pieces[0].dtype
    win = torch.hann_window(chunk, periodic=True, device=device, dtype=dtype)
    half = chunk // 2

    acc = torch.zeros(length, device=device, dtype=dtype)
    wsum = torch.zeros(length, device=device, dtype=dtype)

    for i, (piece, s) in enumerate(zip(pieces, starts)):
        # 양 끝은 겹칠 상대가 없어 창 보정이 불완전하다.
        # Hann은 첫 샘플이 정확히 0이라 그대로 두면 신호 맨 앞이 사라진다.
        # 바깥쪽 반쪽을 1로 펴서(=창을 반쪽만 적용) 진폭을 보존한다.
        w = win
        if i == 0 or i == len(pieces) - 1:
            w = win.clone()
            if i == 0:
                w[:half] = 1.0
            if i == len(pieces) - 1:
                w[half:] = 1.0

        end = min(s + chunk, length)
        n = end - s
        acc[s:end] += piece[:n] * w[:n]
        wsum[s:end] += w[:n]

    return acc / torch.clamp(wsum, min=EPS)
