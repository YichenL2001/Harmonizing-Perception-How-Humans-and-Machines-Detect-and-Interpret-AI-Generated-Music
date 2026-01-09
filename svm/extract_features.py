import argparse
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as TAF
import torchaudio.transforms as TAT

FEATURE_COLS = (
    "M_mean",
    "M_var",
    "M_skew",
    "M_kurt",
    "P_mean",
    "P_var",
    "P_skew",
    "P_kurt",
    "mfcc_mean",
    "mfcc_var",
    "delta_mean",
    "delta_var",
    "delta2_mean",
    "delta2_var",
)


def load_audio(
    path: str,
    target_sr: Optional[int] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    if waveform.ndim != 2:
        raise RuntimeError("Expected waveform [channels, time], got {}".format(tuple(waveform.shape)))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if offset > 0.0 or (duration is not None and duration > 0.0):
        start = max(0, int(round(offset * sr)))
        if duration is not None and duration > 0.0:
            end = start + int(round(duration * sr))
        else:
            end = waveform.shape[1]
        waveform = waveform[:, start:end]

    if target_sr is not None and sr != target_sr:
        waveform = TAF.resample(waveform, sr, target_sr)
        sr = int(target_sr)

    if waveform.numel() == 0:
        raise RuntimeError("Empty waveform after trimming")

    return waveform.squeeze(0), int(sr)


def _torch_unwrap(ph: torch.Tensor, dim: int = -1) -> torch.Tensor:
    d = torch.diff(ph, dim=dim)
    tau = 2.0 * math.pi
    pi = math.pi
    d = (d + pi) % tau - pi
    first = ph.narrow(dim, 0, 1)
    csum = torch.cumsum(d, dim=dim)
    return torch.cat((first, first + csum), dim=dim)


def _torch_skew(z: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(z)
    std = torch.std(z, unbiased=False).clamp_min(1e-6)
    normalized = (z - mean) / std
    return torch.mean(normalized.pow(3))


def _torch_kurt(z: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(z)
    std = torch.std(z, unbiased=False).clamp_min(1e-6)
    normalized = (z - mean) / std
    return torch.mean(normalized.pow(4))


def _ensure_mfcc(sample_rate: int) -> TAT.MFCC:
    window_length = max(1, int(round(0.03 * sample_rate)))
    overlap_length = max(0, int(round(0.025 * sample_rate)))
    hop_length = max(1, window_length - overlap_length)
    melkwargs = {
        "n_fft": max(window_length, 1),
        "win_length": window_length,
        "hop_length": hop_length,
        "n_mels": 26,
        "f_min": 0.0,
        "f_max": sample_rate / 2.0,
        "center": False,
        "pad_mode": "constant",
        "power": 2.0,
        "norm": None,
        "mel_scale": "htk",
        "window_fn": lambda win_length, **_: torch.hamming_window(win_length, periodic=False),
    }
    return TAT.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        dct_type=2,
        norm="ortho",
        log_mels=True,
        melkwargs=melkwargs,
    )


def extract_features(
    path: str,
    offset: float = 0.0,
    duration: Optional[float] = None,
    target_sr: Optional[int] = None,
    k_blocks: int = 100,
) -> Dict[str, float]:
    segment, sr = load_audio(path, target_sr=target_sr, offset=offset, duration=duration)
    segment = segment.to(torch.float64)

    spectrum = torch.fft.fft(segment)
    total_len = spectrum.shape[0]
    if total_len < k_blocks:
        raise ValueError("Signal too short for K={} segmentation".format(k_blocks))
    m_size = total_len // k_blocks
    usable = m_size * k_blocks
    spectrum = spectrum[:usable]

    bm = torch.zeros((m_size, m_size), dtype=torch.float32)
    ba = torch.zeros((m_size, m_size), dtype=torch.float32)

    for blk in range(k_blocks):
        start = blk * m_size
        stop = start + m_size
        y = spectrum[start:stop]
        ld = y[:, None] + y[None, :]

        amp = torch.abs(y)
        apf = amp[:, None] * amp[None, :]
        asf = torch.abs(ld)
        bm.add_(torch.abs(apf * asf))

        phase_y = _torch_unwrap(torch.angle(y))
        af = phase_y[:, None] + phase_y[None, :]
        df = _torch_unwrap(torch.angle(ld), dim=0)
        ba.add_(af - df)

    bm.div_(k_blocks)
    ba.div_(k_blocks)

    bm = bm - torch.min(bm)
    max_bm = torch.max(bm)
    if max_bm > 0:
        bm = bm / max_bm
    else:
        bm.zero_()

    ba = ba - torch.min(ba)
    max_ba = torch.max(ba)
    if max_ba > 0:
        ba = ba / max_ba
    else:
        ba.zero_()

    col_bm = bm.reshape(-1).to(torch.float64)
    col_ba = ba.reshape(-1).to(torch.float64)

    m_mean = torch.mean(col_bm).item()
    m_var = torch.var(col_bm, unbiased=False).item()
    m_skew = _torch_skew(col_bm).item()
    m_kurt = _torch_kurt(col_bm).item()
    p_mean = torch.mean(col_ba).item()
    p_var = torch.var(col_ba, unbiased=False).item()
    p_skew = _torch_skew(col_ba).item()
    p_kurt = _torch_kurt(col_ba).item()

    mfcc_transform = _ensure_mfcc(sr)
    seg32 = segment.to(torch.float32)
    mfcc = mfcc_transform(seg32.unsqueeze(0))
    mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
    delta = TAF.compute_deltas(mfcc, win_length=5)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    delta2 = TAF.compute_deltas(delta, win_length=5)
    delta2 = torch.nan_to_num(delta2, nan=0.0, posinf=0.0, neginf=0.0)

    def _moments(t: torch.Tensor) -> Tuple[float, float]:
        flat = t.reshape(-1).to(torch.float64)
        return torch.mean(flat).item(), torch.var(flat, unbiased=False).item()

    mfcc_mean, mfcc_var = _moments(mfcc)
    delta_mean, delta_var = _moments(delta)
    delta2_mean, delta2_var = _moments(delta2)

    values = (
        m_mean,
        m_var,
        m_skew,
        m_kurt,
        p_mean,
        p_var,
        p_skew,
        p_kurt,
        mfcc_mean,
        mfcc_var,
        delta_mean,
        delta_var,
        delta2_mean,
        delta2_var,
    )

    return {name: float(val) for name, val in zip(FEATURE_COLS, values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SVM features from a single audio file.")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--offset", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate")
    parser.add_argument("--k", type=int, default=100, help="Number of FFT blocks")
    args = parser.parse_args()

    feats = extract_features(
        args.audio,
        offset=args.offset,
        duration=args.duration,
        target_sr=args.sr,
        k_blocks=args.k,
    )
    for name in FEATURE_COLS:
        print("{}\t{:.6f}".format(name, feats[name]))


if __name__ == "__main__":
    main()
