from pathlib import Path
from typing import Optional
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as TAF


def load_audio(
    path: str,
    target_sr: Optional[int] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> tuple[torch.Tensor, int]:
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

    return waveform.squeeze(0), int(sr)


def _resolve_target(value) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"real", "human"}:
            return 0.0
        if lowered in {"fake", "ai"}:
            return 1.0
    return float(value)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sample_rate: int,
        max_len: int,
        normalize: str = "std",
    ):
        if dataframe.empty:
            raise ValueError("Provided DataFrame is empty.")
        self.df = dataframe.reset_index(drop=True)
        self.sample_rate = int(sample_rate)
        self.max_len = int(max_len)
        self.normalize = normalize

        lower = {c.lower(): c for c in self.df.columns}
        if "filepath" not in lower or "target" not in lower:
            raise ValueError("CSV must include 'filepath' and 'target'.")
        self.col_path = lower["filepath"]
        self.col_target = lower["target"]
        self.col_seg_start = lower.get("seg_start")
        self.col_seg_end = lower.get("seg_end")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = str(row[self.col_path])
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)

        start_sec = float(row[self.col_seg_start]) if self.col_seg_start and pd.notna(row[self.col_seg_start]) else 0.0
        seg_duration = None
        if self.col_seg_end and pd.notna(row[self.col_seg_end]):
            seg_duration = max(0.0, float(row[self.col_seg_end]) - start_sec)

        waveform, _ = load_audio(
            file_path,
            target_sr=self.sample_rate,
            offset=start_sec,
            duration=seg_duration,
        )

        if waveform.numel() == 0:
            waveform = torch.zeros(1)

        if waveform.shape[0] < self.max_len:
            pad = self.max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]

        if self.normalize == "std":
            waveform = waveform / torch.clamp(waveform.std(), min=1e-6)
        elif self.normalize == "minmax":
            waveform = waveform - waveform.min()
            waveform = waveform / torch.clamp(waveform.max(), min=1e-6)

        target = torch.tensor(_resolve_target(row[self.col_target]), dtype=torch.float32)
        return waveform, target
