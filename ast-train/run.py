import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import models
from dataloader import AudiosetDataset
from traintest import train, validate


AST_TRAIN_ROOT = Path(__file__).resolve().parent
AST_ROOT = AST_TRAIN_ROOT.parent
DEFAULT_DATA_ROOT = AST_ROOT / "data"
DEFAULT_EXP_ROOT = AST_TRAIN_ROOT / "experiments"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"yes", "true", "t", "y", "1"}:
            return True
        if lowered in {"no", "false", "f", "n", "0"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_leave_exp_dirs(exp_root: Path, leave_root: Path):
    exp_root.mkdir(parents=True, exist_ok=True)
    created = []
    if leave_root.exists():
        for model_dir in sorted([d for d in leave_root.iterdir() if d.is_dir()]):
            target = exp_root / model_dir.name
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
    return created


def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if isinstance(model, torch.nn.DataParallel):
        prefixed = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                prefixed[k] = v
            else:
                prefixed[f"module.{k}"] = v
        model.load_state_dict(prefixed, strict=False)
        return

    stripped = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    model.load_state_dict(stripped, strict=False)


def build_parser():
    parser = argparse.ArgumentParser(description="AST training and evaluation")

    parser.add_argument("--data_root", default=None)
    parser.add_argument("--data_train", default=None)
    parser.add_argument("--data_val", default=None)
    parser.add_argument("--data_eval", default=None)

    parser.add_argument("--exp_dir", default=None)
    parser.add_argument("--exp_root", default=str(DEFAULT_EXP_ROOT / "leave"))
    parser.add_argument("--leave_root", default=str(DEFAULT_DATA_ROOT / "leave"))
    parser.add_argument("--leave_model", default=None)
    parser.add_argument("--init_leave_dirs", action="store_true")
    parser.add_argument("--skip_training", action="store_true")

    parser.add_argument("--gpu", default="")
    parser.add_argument("--seed", type=int, default=52)

    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--dataset", default="audioset")
    parser.add_argument("--model_size", default="base384")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--n_print_steps", type=int, default=100)

    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=200)
    parser.add_argument("--mixup", type=float, default=0.5)
    parser.add_argument("--fstride", type=int, default=10)
    parser.add_argument("--tstride", type=int, default=10)
    parser.add_argument("--imagenet_pretrain", type=str2bool, default=True)
    parser.add_argument("--audioset_pretrain", type=str2bool, default=False)

    parser.add_argument("--dataset_mean", type=float, default=-2.8469415)
    parser.add_argument("--dataset_std", type=float, default=4.2768264)
    parser.add_argument("--audio_length", type=int, default=3000)
    parser.add_argument("--noise", type=str2bool, default=False)

    parser.add_argument("--loss", default="BCE")
    parser.add_argument("--warmup", type=str2bool, default=True)
    parser.add_argument("--lrscheduler_start", type=int, default=10)
    parser.add_argument("--lrscheduler_step", type=int, default=5)
    parser.add_argument("--lrscheduler_decay", type=float, default=0.5)

    return parser


def resolve_paths(args):
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT

    if args.data_train is None:
        args.data_train = str(data_root / "train_data.json")
    if args.data_val is None:
        args.data_val = str(data_root / "valid_data.json")
    if args.data_eval is None:
        args.data_eval = str(data_root / "test_data.json")

    if args.exp_dir is None:
        args.exp_dir = str(DEFAULT_EXP_ROOT / "default")

    leave_root = Path(args.leave_root)
    exp_root = Path(args.exp_root)

    if args.init_leave_dirs or leave_root.exists():
        created = ensure_leave_exp_dirs(exp_root, leave_root)
        if created:
            print("Ensured leave-one-model experiment directories:", [d.name for d in created])

    if args.leave_model:
        model_dir = leave_root / args.leave_model
        if not model_dir.exists():
            raise FileNotFoundError(f"Leave-one-model data directory not found: {model_dir}")
        expected_files = {
            "train": model_dir / "train_data.json",
            "valid": model_dir / "valid_data.json",
            "test": model_dir / "test_data.json",
        }
        missing = [str(p) for p in expected_files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing JSON files for leave-one-model dataset '{args.leave_model}': {missing}"
            )
        args.data_train = str(expected_files["train"])
        args.data_val = str(expected_files["valid"])
        args.data_eval = str(expected_files["test"])
        args.exp_dir = str(exp_root / args.leave_model)

    return data_root


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed_everything(args.seed)
    resolve_paths(args)

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)

    audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.audio_length,
        "freqm": args.freqm,
        "timem": args.timem,
        "mixup": args.mixup,
        "dataset": args.dataset,
        "mode": "train",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": args.noise,
        "label_num": args.n_class,
    }
    val_audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.audio_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": args.dataset,
        "mode": "evaluation",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": False,
        "label_num": args.n_class,
    }

    train_loader = DataLoader(
        AudiosetDataset(args.data_train, audio_conf=audio_conf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        AudiosetDataset(args.data_val, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        AudiosetDataset(args.data_eval, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    audio_model = models.ASTModel(
        label_dim=args.n_class,
        fstride=args.fstride,
        tstride=args.tstride,
        input_fdim=128,
        input_tdim=args.audio_length,
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain,
        model_size=args.model_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using GPU(s):", args.gpu or "default")
    gpu_count = torch.cuda.device_count()
    print("PyTorch sees", gpu_count, "GPUs:")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    audio_model.to(device)
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs with DataParallel")
        audio_model = torch.nn.DataParallel(audio_model)

    if args.skip_training:
        print("Skipping training (--skip_training set)")
    else:
        print(f"Now starting training for {args.n_epochs} epochs")
        train(audio_model, train_loader, val_loader, args)

    best_model_path = exp_dir / "models" / "best_audio_model.pth"
    if best_model_path.exists():
        print("\nLoading best model for evaluation...")
        state = torch.load(best_model_path, map_location=device)
        load_state_dict_flexible(audio_model, state)
        audio_model.to(device)
    else:
        print("\nWarning: Best model not found, using last trained model.")

    test_stats, _ = validate(audio_model, eval_loader, args)
    test_samples = len(eval_loader.dataset)

    test_entry = test_stats[0]
    print(
        "\nTest Set Results - Balanced Accuracy: {:.6f}, Precision: {:.6f}, "
        "Recall: {:.6f}, FPR: {:.6f}, FNR: {:.6f}, mAP: {:.6f}, Total Samples: {}".format(
            test_entry["balanced_acc"],
            test_entry["precision"],
            test_entry["recall"],
            test_entry["fpr"],
            test_entry["fnr"],
            test_entry["AP"],
            test_samples,
        )
    )
    print("Test Confusion Matrix:")
    print(test_entry["cm"])

    eval_results_path = exp_dir / "eval_result.csv"
    with open(eval_results_path, "w") as fp:
        fp.write("Dataset,Total Samples,Balanced Accuracy,Precision,Recall,FPR,FNR,mAP,Confusion Matrix\n")
        fp.write(
            "Test,{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                test_samples,
                test_entry["balanced_acc"],
                test_entry["precision"],
                test_entry["recall"],
                test_entry["fpr"],
                test_entry["fnr"],
                test_entry["AP"],
                test_entry["cm"],
            )
        )
    print("\nEvaluation results saved to", eval_results_path)

    test_dataset = eval_loader.dataset
    test_data = getattr(test_dataset, "data", None)
    if test_data is None:
        raise AttributeError("Evaluation dataset does not expose raw 'data' entries.")

    all_test_predictions = []
    audio_model.eval()
    with torch.no_grad():
        for data, _ in eval_loader:
            data = data.to(device)
            output = audio_model(data)
            output = torch.sigmoid(output)
            pred = output.argmax(dim=1)
            all_test_predictions.extend(pred.cpu().numpy().tolist())

    results_list = []
    for i, item in enumerate(test_data):
        true_label = item.get("labels")
        if isinstance(true_label, list) and true_label:
            true_label = true_label[0]
        if true_label is not None and not isinstance(true_label, (int, float)):
            try:
                true_label = int(float(true_label))
            except (ValueError, TypeError):
                pass
        results_list.append(
            {
                "wav": item.get("wav"),
                "true_label": true_label,
                "predicted_label": int(all_test_predictions[i]),
            }
        )

    df_results = pd.DataFrame(results_list)
    results_csv_path = exp_dir / "test_predictions.csv"
    df_results.to_csv(results_csv_path, index=False)

    df_roc = pd.DataFrame(
        {
            "fpr": test_entry["fpr_curve"],
            "tpr": test_entry["tpr_curve"],
            "threshold": test_entry["thresholds"],
        }
    )
    df_roc.to_csv(exp_dir / "roc_ast.csv", index=False)


if __name__ == "__main__":
    main()
