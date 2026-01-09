import csv
import json
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
    module="torchaudio._backend.utils",
)

TARGET_SAMPLE_RATE = 48000
MIN_FBANK_SAMPLES = 1200


def _ensure_mono_resample(waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
    if waveform.ndim != 2:
        raise RuntimeError(f"Expected waveform with shape [channels, time], got {tuple(waveform.shape)}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = TARGET_SAMPLE_RATE
    return waveform, sr


def load_audio_with_fallback(filename: str):
    try:
        return torchaudio.load(filename)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load {filename} with torchaudio. Ensure ffmpeg backend is available."
        ) from exc


def _normalize_label(value):
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_label(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        lowered = str(value).strip().lower()
        if lowered in {"fake", "ai"}:
            return 1
        if lowered in {"real", "human"}:
            return 0
        return None


def _build_segment_map(records):
    seg_map = {}
    for record in records:
        start = record.get("seg_start")
        end = record.get("seg_end")
        if start is not None and end is not None:
            seg_map[record["wav"]] = (float(start), float(end))
    return seg_map


class AudiosetDataset(Dataset):
    def __init__(self, dataset_path: str, audio_conf):
        self.datapath = dataset_path
        self.data = self._load_metadata(dataset_path)
        self.seg_map = _build_segment_map(self.data)

        self.audio_conf = audio_conf
        print("---------------the {} dataloader---------------".format(self.audio_conf.get("mode")))

        self.melbins = int(self.audio_conf.get("num_mel_bins"))
        self.freqm = int(self.audio_conf.get("freqm") or 0)
        self.timem = int(self.audio_conf.get("timem") or 0)
        print("now using following mask: {} freq, {} time".format(self.freqm, self.timem))

        self.mixup = float(self.audio_conf.get("mixup") or 0.0)
        print("now using mix-up with rate {}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        print("now process {}".format(self.dataset))

        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        self.skip_norm = bool(self.audio_conf.get("skip_norm") or False)

        if self.skip_norm:
            print("now skip normalization")
        else:
            print(
                "use dataset mean {:.3f} and std {:.3f} to normalize the input".format(
                    self.norm_mean, self.norm_std
                )
            )

        self.noise = bool(self.audio_conf.get("noise"))
        if self.noise:
            print("now use noise augmentation")

        self.label_num = int(self.audio_conf.get("label_num", 2))
        print("number of classes is {}".format(self.label_num))

    def _load_metadata(self, metadata_path: str):
        path = Path(metadata_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(metadata_path, "r") as fp:
                data_json = json.load(fp)
            data = data_json.get("data")
            if data is None:
                raise KeyError(f"'data' key not found in JSON metadata: {metadata_path}")
            for item in data:
                item["labels"] = _normalize_label(item.get("labels"))
            return data
        if suffix == ".csv":
            return self._load_csv(metadata_path)
        raise ValueError(f"Unsupported metadata format '{suffix}' for file: {metadata_path}")

    def _load_csv(self, csv_path: str):
        records = []
        skipped = 0

        with open(csv_path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                wav_path = row.get("filepath") or row.get("wav") or row.get("path")
                if not wav_path:
                    skipped += 1
                    continue
                label = _parse_label(row.get("target"))
                if label is None:
                    label = _parse_label(row.get("labels"))
                if label is None:
                    label = _parse_label(row.get("label"))
                if label is None:
                    skipped += 1
                    continue

                records.append(
                    {
                        "id": row.get("id"),
                        "wav": wav_path,
                        "labels": label,
                        "seg_start": _parse_float(row.get("seg_start")),
                        "seg_end": _parse_float(row.get("seg_end")),
                    }
                )

        if not records:
            raise ValueError(f"No usable rows found in CSV metadata: {csv_path}")
        if skipped:
            print("[AudiosetDataset] Skipped {} rows while parsing {}".format(skipped, csv_path))
        return records

    def _wav2fbank(self, filename: str, filename2: Optional[str] = None):
        if filename2 is None:
            waveform, sr = load_audio_with_fallback(filename)
            seg = self.seg_map.get(filename)
            if seg is not None:
                s_samp = max(0, min(int(seg[0] * sr), waveform.shape[1] - 1))
                e_samp = max(s_samp + 1, min(int(seg[1] * sr), waveform.shape[1]))
                waveform = waveform[:, s_samp:e_samp]
            waveform, sr = _ensure_mono_resample(waveform, sr)
            if waveform.shape[1] == 0:
                raise RuntimeError(f"Empty waveform for {filename} after segment trimming")
            if waveform.shape[1] < MIN_FBANK_SAMPLES:
                pad = MIN_FBANK_SAMPLES - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad))
            waveform = waveform - waveform.mean()
            mix_lambda = 0.0
        else:
            waveform1, sr1 = load_audio_with_fallback(filename)
            seg1 = self.seg_map.get(filename)
            if seg1 is not None:
                s1 = max(0, min(int(seg1[0] * sr1), waveform1.shape[1] - 1))
                e1 = max(s1 + 1, min(int(seg1[1] * sr1), waveform1.shape[1]))
                waveform1 = waveform1[:, s1:e1]
            waveform1, sr1 = _ensure_mono_resample(waveform1, sr1)
            if waveform1.shape[1] == 0:
                raise RuntimeError(f"Empty waveform for {filename} after segment trimming")
            if waveform1.shape[1] < MIN_FBANK_SAMPLES:
                pad = MIN_FBANK_SAMPLES - waveform1.shape[1]
                waveform1 = F.pad(waveform1, (0, pad))

            waveform2, sr2 = load_audio_with_fallback(filename2)
            seg2 = self.seg_map.get(filename2)
            if seg2 is not None:
                s2 = max(0, min(int(seg2[0] * sr2), waveform2.shape[1] - 1))
                e2 = max(s2 + 1, min(int(seg2[1] * sr2), waveform2.shape[1]))
                waveform2 = waveform2[:, s2:e2]
            waveform2, sr2 = _ensure_mono_resample(waveform2, sr2)
            if waveform2.shape[1] == 0:
                raise RuntimeError(f"Empty waveform for {filename2} after segment trimming")
            if waveform2.shape[1] < MIN_FBANK_SAMPLES:
                pad = MIN_FBANK_SAMPLES - waveform2.shape[1]
                waveform2 = F.pad(waveform2, (0, pad))
            if sr2 != sr1:
                raise RuntimeError(
                    "Resample mismatch between {} ({}) and {} ({})".format(
                        filename, sr1, filename2, sr2
                    )
                )

            if waveform1.shape[1] != waveform2.shape[1]:
                min_len = min(waveform1.shape[1], waveform2.shape[1])
                if min_len <= 0:
                    raise RuntimeError(
                        "Cannot mix waveforms with non-positive length "
                        "({} vs {}) from {} and {}".format(
                            waveform1.shape[1], waveform2.shape[1], filename, filename2
                        )
                    )
                waveform1 = waveform1[:, :min_len]
                waveform2 = waveform2[:, :min_len]

            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
            sr = sr1

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=10,
        )

        target_length = self.audio_conf.get("target_length")
        n_frames = fbank.shape[0]
        pad_frames = target_length - n_frames
        if pad_frames > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_frames))(fbank)
        elif pad_frames < 0:
            fbank = fbank[0:target_length, :]

        return fbank, mix_lambda

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]
            fbank, _ = self._wav2fbank(datum["wav"], mix_datum["wav"])
        else:
            datum = self.data[index]
            fbank, _ = self._wav2fbank(datum["wav"])

        label_indices = np.zeros(self.label_num)
        label = _normalize_label(datum.get("labels"))
        if label is None or label < 0:
            label_indices = torch.zeros(self.label_num)
        else:
            label_indices[int(label)] = 1.0
            label_indices = torch.FloatTensor(label_indices)

        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        fbank = torch.nan_to_num(fbank, nan=0.0, posinf=0.0, neginf=0.0)

        return fbank, label_indices

    def __len__(self):
        return len(self.data)
