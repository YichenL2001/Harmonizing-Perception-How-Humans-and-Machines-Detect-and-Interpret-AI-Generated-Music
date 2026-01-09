import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

import data_lib
import network_models_lib
import params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-set evaluation for M5/RawNet2/ResNet18.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model_name", type=str, default="SpecResNet", choices=("SpecResNet", "M5", "RawNet2"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained .pth checkpoint.")
    parser.add_argument("--test_csv", type=Path, required=True, help="CSV containing filepath/target/segment metadata.")
    parser.add_argument("--audio_duration", type=float, default=params.AUDIO_LENGTH_SECONDS)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=Path(params.RESULTS_DIR))
    parser.add_argument("--save_json", action="store_true", help="Save metrics as JSON inside output_dir.")
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="Randomly crop a segment of audio_duration seconds from each clip instead of using seg_start/seg_end.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print(">> CUDA requested but not available. Falling back to CPU.")
        choice = "cpu"
    return torch.device(choice)


def build_model(name: str, audio_seconds: float) -> Tuple[torch.nn.Module, str]:
    if name == "M5":
        model = network_models_lib.M5(n_input=1, n_output=len(data_lib.model_labels))
        feat_type = "raw"
    elif name == "RawNet2":
        d_args = {
            "nb_samp": int(audio_seconds * params.DESIRED_SR),
            "first_conv": 3,
            "in_channels": 1,
            "filts": [128, [128, 128], [128, 256], [256, 256]],
            "blocks": [2, 4],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 1,
            "nb_classes": len(data_lib.model_labels),
        }
        model = network_models_lib.RawNet2(d_args)
        feat_type = "raw"
    elif name == "SpecResNet":
        model = network_models_lib.ResNet(
            img_channels=1,
            num_layers=18,
            block=network_models_lib.BasicBlock,
            num_classes=len(data_lib.model_labels),
        )
        feat_type = "freq"
    else:
        raise ValueError("Unsupported model name {}".format(name))
    return model, feat_type


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[List[int], List[int], List[float], List[float]]:
    preds: List[int] = []
    targets: List[int] = []
    prob_real: List[float] = []
    prob_fake: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                batch_audio, batch_target, _ = batch
            else:
                batch_audio, batch_target = batch
            batch_audio = batch_audio.to(device)
            batch_target = batch_target.to(device).long().squeeze()
            if batch_target.dim() == 0:
                batch_target = batch_target.unsqueeze(0)

            output = model(batch_audio)
            if output.dim() == 3:
                output = output.squeeze(1)

            probs = output.exp()
            pred_idx = probs.argmax(dim=1)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            if probs.shape[1] == 1:
                real_batch = 1.0 - probs[:, 0]
                fake_batch = probs[:, 0]
            else:
                real_batch = probs[:, 0]
                fake_batch = probs[:, 1]

            preds.extend(pred_idx.cpu().tolist())
            targets.extend(batch_target.cpu().tolist())
            prob_real.extend(real_batch.detach().cpu().tolist())
            prob_fake.extend(fake_batch.detach().cpu().tolist())

    return preds, targets, prob_real, prob_fake


def per_generator_accuracy(df: pd.DataFrame, preds: List[int], targets: List[int]) -> Dict[str, Dict[str, float]]:
    bucket_col = None
    for col in df.columns:
        if col.lower() == "bucket":
            bucket_col = col
            break

    if bucket_col is None:
        return {"unknown": {"accuracy": float(np.mean(np.array(preds) == np.array(targets))), "total": len(preds)}}

    buckets = df[bucket_col].fillna("unknown").astype(str).tolist()
    stats: Dict[str, Dict[str, float]] = {}
    for bucket, pred, tgt in zip(buckets, preds, targets):
        bucket_stats = stats.setdefault(bucket, {"correct": 0.0, "total": 0.0})
        bucket_stats["total"] += 1
        if pred == tgt:
            bucket_stats["correct"] += 1

    for bucket, bucket_stats in stats.items():
        total = bucket_stats["total"]
        acc = bucket_stats["correct"] / total if total else 0.0
        bucket_stats["accuracy"] = acc
        del bucket_stats["correct"]

    return stats


def main() -> None:
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = resolve_device(args.device)
    print("> Using device:", device)

    df = pd.read_csv(args.test_csv)
    if df.empty:
        raise RuntimeError("No rows found in the test CSV.")

    model, feat_type = build_model(args.model_name, args.audio_duration)
    model.to(device)
    load_checkpoint(model, args.checkpoint, device)

    dataset = data_lib.MusicDeepFakeDataset(
        df,
        AUDIO_LENGTH_SECONDS=args.audio_duration,
        sample_rate=params.DESIRED_SR,
        feat_type=feat_type,
        random_crop=args.random_crop,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    preds, targets, prob_real, prob_fake = evaluate(model, loader, device)
    if not preds:
        raise RuntimeError("No samples were evaluated.")

    preds_arr = np.asarray(preds, dtype=int)
    targets_arr = np.asarray(targets, dtype=int)

    cm = confusion_matrix(targets_arr, preds_arr, labels=[0, 1])
    bal_acc = balanced_accuracy_score(targets_arr, preds_arr)
    precision = precision_score(targets_arr, preds_arr, zero_division=0)
    recall = recall_score(targets_arr, preds_arr, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    per_gen = per_generator_accuracy(df, preds, targets)

    metrics = {
        "samples": int(len(targets)),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm.tolist(),
        "per_generator_accuracy": per_gen,
    }

    print("\n=== Evaluation Metrics ===")
    print("Samples:", metrics["samples"])
    print("Balanced accuracy: {:.4f}".format(metrics["balanced_accuracy"]))
    print("Precision: {:.4f}".format(metrics["precision"]))
    print("Recall: {:.4f}".format(metrics["recall"]))
    print("FPR: {:.4f}".format(metrics["fpr"]))
    print("FNR: {:.4f}".format(metrics["fnr"]))
    print("Confusion matrix (rows=true [real,fake], cols=pred):")
    for row in metrics["confusion_matrix"]:
        print("  {}".format(row))

    print("\nPer-generator accuracy:")
    for bucket, bucket_stats in sorted(metrics["per_generator_accuracy"].items()):
        print("  {}: {:.4f} (n={})".format(bucket, bucket_stats["accuracy"], int(bucket_stats["total"])))

    if args.save_json:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "{}_metrics.json".format(args.model_name.lower())
        out_path.write_text(json.dumps(metrics, indent=2))
        print("\nMetrics saved to", out_path)


if __name__ == "__main__":
    main()
