import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

from dataset import AudioDataset
from models import AudioClassifier


MODEL_DEFAULTS: Dict[str, Dict] = {
    "ViT": {
        "patch_size": 16,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "patch_norm": True,
        "pe_learnable": True,
        "pos_drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "proj_drop_rate": 0.0,
        "mlp_ratio": 2.67,
    },
    "SpecTTTra": {
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "t_clip": 3,
        "f_clip": 1,
        "pre_norm": True,
        "pe_learnable": True,
        "pos_drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "proj_drop_rate": 0.0,
        "mlp_ratio": 2.67,
    },
    "ConvNeXt": {
        "timm_name": "convnext_tiny",
        "pretrained": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ViT/SpecTTTra/ConvNeXt on a CSV split.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=("ViT", "SpecTTTra", "ConvNeXt"), default="SpecTTTra")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--max_time", type=float, default=10.0)
    parser.add_argument("--input_shape", type=str, default="128,128")
    parser.add_argument("--normalize", choices=("std", "minmax", "none"), default="std")

    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--f_max", type=float, default=24000.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--top_db", type=float, default=80.0)
    parser.add_argument("--spec_norm", choices=("mean_std", "min_max", "simple", "none"), default="mean_std")

    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        choice = "cpu"
    return torch.device(choice)


def parse_shape(value: str) -> Tuple[int, int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--input_shape must be like 128,128")
    return int(parts[0]), int(parts[1])


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if path is None:
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def evaluate(model: torch.nn.Module, loader, device: torch.device, num_classes: int):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for audio, target in loader:
            audio = audio.to(device)
            target = target.to(device)
            logits = model(audio)
            if num_classes == 1:
                probs = torch.sigmoid(logits.view(-1))
                pred = (probs > 0.5).long()
            else:
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            targets.extend(target.long().cpu().tolist())
    return np.asarray(preds, dtype=int), np.asarray(targets, dtype=int)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError("CSV is empty.")

    max_len = int(round(args.sample_rate * args.max_time))
    dataset = AudioDataset(
        df,
        sample_rate=args.sample_rate,
        max_len=max_len,
        normalize=args.normalize,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    input_shape = parse_shape(args.input_shape)
    model_cfg = MODEL_DEFAULTS[args.model]

    melspec_cfg = {
        "sample_rate": args.sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "n_mels": args.n_mels,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "power": args.power,
        "top_db": args.top_db,
        "norm": args.spec_norm,
    }

    model = AudioClassifier(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=args.num_classes,
        melspec_cfg=melspec_cfg,
        model_cfg=model_cfg,
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)

    preds, targets = evaluate(model, loader, device, args.num_classes)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    bal_acc = balanced_accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print("samples", len(targets))
    print("balanced_accuracy", float(bal_acc))
    print("precision", float(precision))
    print("recall", float(recall))
    print("fpr", float(fpr))
    print("fnr", float(fnr))
    print("confusion_matrix", cm.tolist())


if __name__ == "__main__":
    main()
