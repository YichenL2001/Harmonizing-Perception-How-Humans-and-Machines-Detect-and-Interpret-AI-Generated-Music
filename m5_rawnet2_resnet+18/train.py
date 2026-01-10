import argparse
import datetime
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data_lib
import network_models_lib
import params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M5/RawNet2/SpecResNet on CSV data.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--model_name", type=str, default="SpecResNet", choices=("SpecResNet", "M5", "RawNet2"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--audio_duration", type=float, default=10.0)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha; 0 disables mixup.")
    parser.add_argument("--random_crop", action="store_true", help="Randomly crop within each clip.")
    parser.add_argument("--output_dir", type=str, default="models")
    return parser.parse_args()


def build_model(name: str, audio_seconds: float):
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


def train_one_epoch(model, loader, optimizer, device, mixup_alpha):
    model.train()
    total_loss = 0.0
    seen_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).squeeze().to(torch.long)

        if mixup_alpha > 0 and x.size(0) > 1:
            mix = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            lam = float(mix.sample().item())
            index = torch.randperm(x.size(0), device=x.device)
            mixed_x = lam * x + (1.0 - lam) * x[index, ...]
            y_a = y
            y_b = y[index]
            out = model(mixed_x).squeeze(1)
            loss = lam * F.nll_loss(out, y_a) + (1.0 - lam) * F.nll_loss(out, y_b)
        else:
            out = model(x).squeeze(1)
            loss = F.nll_loss(out, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        seen_batches += 1

    return total_loss / max(1, seen_batches)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).squeeze().to(torch.long)
            out = model(x).squeeze(1)
            loss = F.nll_loss(out, y)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

    avg_loss = total_loss / max(1, len(loader))
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()

    gpu_id = args.gpu.split(",")[0].strip() if args.gpu else ""
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, feat_type = build_model(args.model_name, args.audio_duration)
    model.to(device)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_data = data_lib.MusicDeepFakeDataset(
        train_df,
        args.audio_duration,
        feat_type=feat_type,
        random_crop=args.random_crop,
    )
    val_data = data_lib.MusicDeepFakeDataset(
        val_df,
        args.audio_duration,
        feat_type=feat_type,
        random_crop=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    duration_value = float(args.audio_duration)
    duration_tag = f"{int(duration_value)}s" if duration_value.is_integer() else f"{duration_value:g}s"
    run_name = "{}_{}_{}".format(args.model_name, duration_tag, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Run:", run_name)

    best_val_loss = float("inf")
    early_stop_cnt = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.mixup_alpha)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(
            "Epoch {:03d} | train_loss {:.6f} | val_loss {:.6f} | val_acc {:.4f}".format(
                epoch, train_loss, val_loss, val_acc
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = "{}_duration_{}.pth".format(args.model_name, duration_tag)
            model_save_path = os.path.join(output_dir, model_filename)
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model to", model_save_path)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            print("PATIENCE {}/{}".format(early_stop_cnt, args.patience))

        if early_stop_cnt >= args.patience:
            print("Early stopping on val loss.")
            break


if __name__ == "__main__":
    main()
