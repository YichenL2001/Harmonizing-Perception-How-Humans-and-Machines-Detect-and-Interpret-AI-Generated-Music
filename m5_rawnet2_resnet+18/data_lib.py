from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import torch
import torchaudio as ta
import torchaudio.functional as AF
import params
import utils

model_labels: Dict[str, int] = {"real": 0, "fake": 1}

_LABEL_ALIASES = {
    "real": 0,
    "human": 0,
    "fake": 1,
    "ai": 1,
}

def _resolve_target(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _LABEL_ALIASES:
            return _LABEL_ALIASES[key]
        try:
            value = int(float(value))
        except ValueError as exc:
            raise ValueError("Unsupported label value '{}'".format(value)) from exc
    return int(value)

def _load_via_torchaudio(source: str) -> tuple[torch.Tensor, int]:
    waveform, sr_local = ta.load(source, normalize=True)
    return waveform, sr_local


def load_audio_with_fallback(
    path: str,
    target_sr: Optional[int] = None,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> tuple[torch.Tensor, int]:
    try:
        waveform, sr_out = _load_via_torchaudio(path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load audio with torchaudio. Ensure ffmpeg backend is available."
        ) from exc

    if waveform.ndim != 2:
        raise RuntimeError("Expected waveform [channels, time], got {}".format(tuple(waveform.shape)))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if offset > 0.0 or (duration is not None and duration > 0.0):
        start = max(0, int(round(offset * sr_out)))
        if duration is not None and duration > 0.0:
            end = start + int(round(duration * sr_out))
        else:
            end = waveform.shape[1]
        waveform = waveform[:, start:end]

    if target_sr is not None and sr_out != target_sr:
        if waveform.shape[1] == 0:
            waveform = torch.zeros(1, 0)
        else:
            waveform = AF.resample(waveform, sr_out, target_sr)
        sr_out = target_sr

    return waveform, sr_out


class MusicDeepFakeDataset(torch.utils.data.Dataset):
    """
    Dataset backed by a CSV file.

    Required columns (case-insensitive):
        - filepath  : absolute path to the audio file
        - target    : integer label (0=real/human, 1=fake/ai) or compatible string
        - seg_start : start time for the segment (seconds)

    Optional columns:
        - seg_end   : end time (seconds)
        - bucket    : generator identifier (used for per-generator accuracy)
    """

    def __init__(
        self,
        dataframe_or_path: Union[pd.DataFrame, str, Path],
        AUDIO_LENGTH_SECONDS: float,
        sample_rate: Optional[int] = None,
        feat_type: str = "raw",
        random_crop: bool = False,
        return_offset: bool = False,
    ) -> None:
        if isinstance(dataframe_or_path, (str, Path)):
            df = pd.read_csv(dataframe_or_path)
        else:
            df = dataframe_or_path.copy()

        if df.empty:
            raise ValueError("Provided CSV/DataFrame is empty.")

        self.df = df.reset_index(drop=True)
        self.seconds = float(AUDIO_LENGTH_SECONDS)
        self.sample_rate = int(sample_rate or params.DESIRED_SR)
        self.target_length = int(round(self.seconds * self.sample_rate))
        self.feat_type = feat_type
        self.random_crop = bool(random_crop)
        self.return_offset = bool(return_offset)

        lower_map = {col.lower(): col for col in self.df.columns}
        required = {"filepath", "target", "seg_start"}
        missing = required - set(lower_map.keys())
        if missing:
            raise ValueError("Missing required CSV columns: {}".format(sorted(missing)))

        self.cols = {name: lower_map[name] for name in required}
        self.optional_cols = {
            "seg_end": lower_map.get("seg_end"),
            "bucket": lower_map.get("bucket"),
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = str(row[self.cols["filepath"]])
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)

        start_value = row[self.cols["seg_start"]]
        start_sec = float(start_value) if pd.notna(start_value) else 0.0
        seg_end_col = self.optional_cols.get("seg_end")
        seg_duration = None
        if seg_end_col:
            end_value = row[seg_end_col]
            if pd.notna(end_value):
                seg_duration = max(0.0, float(end_value) - start_sec)
        if seg_duration is None or seg_duration <= 0:
            seg_duration = self.seconds

        if self.random_crop:
            if seg_end_col is None or pd.isna(row.get(seg_end_col, None)):
                start_sec = 0.0
                seg_duration = None

        waveform, _ = load_audio_with_fallback(
            file_path,
            target_sr=self.sample_rate,
            offset=start_sec,
            duration=seg_duration,
        )

        crop_start_sec = start_sec
        if self.random_crop and waveform.shape[1] > self.target_length:
            max_start = waveform.shape[1] - self.target_length
            start_sample = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start_sample : start_sample + self.target_length]
            crop_start_sec = start_sec + start_sample / float(self.sample_rate)

        audio = waveform
        if audio.shape[1] < self.target_length:
            pad = self.target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        elif audio.shape[1] > self.target_length:
            audio = audio[:, : self.target_length]

        audio = audio.squeeze(0)
        if torch.sum(torch.abs(audio)) > 0:
            audio = utils.normalize_tensor(audio)
        audio = audio.unsqueeze(0)

        target_value = _resolve_target(row[self.cols["target"]])
        label = torch.tensor([target_value], dtype=torch.float32)

        if self.feat_type == "freq":
            log_stft = torch.log(
                torch.abs(
                    torch.stft(
                        audio,
                        n_fft=512,
                        hop_length=128,
                        window=torch.hann_window(window_length=512),
                        center=True,
                        return_complex=True,
                    )
                )
                + 1e-3
            )
            if torch.sum(torch.abs(audio)) > 0:
                log_stft = utils.normalize_tensor(log_stft)
            if self.return_offset:
                return log_stft, label, torch.tensor(float(crop_start_sec), dtype=torch.float32)
            return log_stft, label

        if self.return_offset:
            return audio, label, torch.tensor(float(crop_start_sec), dtype=torch.float32)
        return audio, label
