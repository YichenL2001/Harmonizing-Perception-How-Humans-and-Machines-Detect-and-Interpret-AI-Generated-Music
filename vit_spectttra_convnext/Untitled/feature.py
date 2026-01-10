import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        power: float,
        top_db: float,
        norm: str,
    ):
        super().__init__()
        self.audio2melspec = MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=top_db)

        if norm == "mean_std":
            self.normalizer = MeanStdNorm()
        elif norm == "min_max":
            self.normalizer = MinMaxNorm()
        elif norm == "simple":
            self.normalizer = SimpleNorm()
        else:
            self.normalizer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        melspec = self.audio2melspec(x.float())
        melspec = self.amplitude_to_db(melspec)
        return self.normalizer(melspec)


class MinMaxNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        min_ = torch.amax(x, dim=(1, 2), keepdim=True)
        max_ = torch.amin(x, dim=(1, 2), keepdim=True)
        return (x - min_) / (max_ - min_ + self.eps)


class SimpleNorm(nn.Module):
    def forward(self, x):
        return (x - 40) / 80


class MeanStdNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean((1, 2), keepdim=True)
        std = x.reshape(x.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (x - mean) / (std + self.eps)
