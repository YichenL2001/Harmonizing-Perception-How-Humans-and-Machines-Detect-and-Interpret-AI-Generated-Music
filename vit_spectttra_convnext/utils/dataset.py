import os
os.environ.setdefault("TORCHAUDIO_USE_LIBTORCHCODEC", "0")

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchaudio


def load_audio_with_fallback(path, sr=None, offset=0.0, duration=None):
    """
    Load audio with torchaudio (ffmpeg backend required for mp3/aac).
    Returns: (audio_array, sample_rate)
    """
    try:
        waveform, sr_out = torchaudio.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {path} with torchaudio. Ensure ffmpeg backend is available."
        ) from exc

    if waveform.ndim != 2:
        raise RuntimeError(f"Expected waveform [channels, time], got {tuple(waveform.shape)} for {path}")

    # downmix to mono if needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if offset > 0.0 or (duration is not None and duration > 0.0):
        start = max(0, int(round(offset * sr_out)))
        if duration is not None and duration > 0.0:
            end = start + int(round(duration * sr_out))
        else:
            end = waveform.shape[1]
        waveform = waveform[:, start:end]

    target_sr = sr
    if target_sr is not None and sr_out != target_sr:
        if waveform.shape[1] == 0:
            return np.zeros(target_sr, dtype=np.float32), target_sr
        resampler = torchaudio.transforms.Resample(orig_freq=sr_out, new_freq=target_sr)
        waveform = resampler(waveform)
        sr_out = target_sr

    return waveform.squeeze(0).numpy(), sr_out


class AudioDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        skip_times=None,
        num_classes=1,
        normalize="std",
        max_len=32000,
        random_sampling=True,
        train=False,
        seg_starts=None,    
        seg_ends=None, 
        sample_rate=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.labels = labels
        self.skip_times = skip_times
        self.num_classes = num_classes
        self.random_sampling = random_sampling
        self.normalize = normalize
        self.max_len = max_len
        self.train = train
        self.seg_starts = seg_starts
        self.seg_ends = seg_ends
        self.sample_rate = sample_rate
        if not self.train:
            assert (
                not self.random_sampling
            ), "Ensure random_sampling is disabled for val"

    def __len__(self):
        return len(self.filepaths)

    def crop_or_pad(self, audio, max_len, random_sampling=True):
        audio_len = audio.shape[0]
        if random_sampling:
            diff_len = abs(max_len - audio_len)
            if audio_len < max_len:
                pad1 = np.random.randint(0, diff_len)
                pad2 = diff_len - pad1
                audio = np.pad(audio, (pad1, pad2), mode="constant")
            elif audio_len > max_len:
                idx = np.random.randint(0, diff_len)
                audio = audio[idx : (idx + max_len)]
        else:
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
            elif audio_len > max_len:
                # Crop from the beginning
                # audio = audio[:max_len]

                # Crop from 3/4 of the audio
                # eq: l = (3x + t + x) => idx = 3x = (l - t) / 4 * 3
                idx = int((audio_len - max_len) / 4 * 3)
                audio = audio[idx : (idx + max_len)]
        return audio

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        target = np.array([self.labels[idx]])

        # If seg_start/end are provided, load only that window (offset/duration in seconds).
        if self.seg_starts is not None and self.seg_ends is not None:
            seg_start = float(self.seg_starts[idx])
            seg_end = float(self.seg_ends[idx])
            seg_dur = max(0.0, seg_end - seg_start)
            audio, sr = load_audio_with_fallback(path, sr=self.sample_rate, offset=seg_start, duration=seg_dur)
        else:
            # Fallback: load full audio
            audio, sr = load_audio_with_fallback(path, sr=self.sample_rate)

            # Legacy: optional "skip_time" trimming if caller passed it
            if self.skip_times is not None:
                skip_time = float(self.skip_times[idx])
                audio = audio[int(skip_time * sr):]

        # Ensure fixed length to `max_len` samples (crop/pad)
        audio = self.crop_or_pad(audio, self.max_len, self.random_sampling)

        # Normalization
        if self.normalize == "std":
            audio /= np.maximum(np.std(audio), 1e-6)
        elif self.normalize == "minmax":
            audio -= np.min(audio)
            audio /= np.maximum(np.max(audio), 1e-6)

        audio = torch.from_numpy(audio).float()
        target = torch.from_numpy(target).float().squeeze()
        return {"audio": audio, "target": target}

def get_dataloader(
    filepaths,
    labels,
    skip_times=None,
    batch_size=8,
    num_classes=1,
    max_len=32000,
    random_sampling=True,
    normalize="std",
    train=False,
    # drop_last=False,
    pin_memory=True,
    worker_init_fn=None,
    collate_fn=None,
    num_workers=0,
    distributed=False,
    seg_starts=None,     
    seg_ends=None,
    sample_rate=None,
):
    dataset = AudioDataset(
        filepaths,
        labels,
        skip_times=skip_times,
        num_classes=num_classes,
        max_len=max_len,
        random_sampling=random_sampling,
        normalize=normalize,
        train=train,
        seg_starts=seg_starts,    # <— NEW
        seg_ends=seg_ends, 
        sample_rate=sample_rate,
    )

    if distributed:
        # drop_last is set to True to validate properly
        # Ref: https://discuss.pytorch.org/t/how-do-i-validate-with-pytorch-distributeddataparallel/172269/8
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=train, drop_last=not train
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and train,
        # drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dataloader
if hasattr(torchaudio, "set_audio_backend"):
    try:
        torchaudio.set_audio_backend("sox_io")
    except RuntimeError:
        pass
