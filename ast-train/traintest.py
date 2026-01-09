import datetime
import os
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utilities import AverageMeter, calculate_stats


def train(audio_model, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running on", device)
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()

    history = []

    patience = 10
    early_stop_cnt = 0
    best_val_loss = float("inf")

    global_step, epoch = 0, 1
    exp_dir = args.exp_dir

    audio_model = audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print(
        "Total parameter number is : {:.3f} million".format(
            sum(p.numel() for p in audio_model.parameters()) / 1e6
        )
    )
    print(
        "Total trainable parameter number is : {:.3f} million".format(
            sum(p.numel() for p in trainables) / 1e6
        )
    )

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    loss_fn = nn.BCEWithLogitsLoss() if args.loss.upper() == "BCE" else nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay,
    )

    print(
        "Training with dataset: {}, loss function: {}, LR scheduler: {}".format(
            args.dataset, loss_fn, scheduler
        )
    )
    print(
        "LR scheduler starts at epoch {}, decays by {:.3f} every {} epochs".format(
            args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step
        )
    )

    scaler = GradScaler()

    print("current #steps={}, #epochs={}".format(global_step, epoch))
    print("Start training...")
    audio_model.train()

    while epoch <= args.n_epochs:
        end_time = time.time()
        audio_model.train()
        print("---------------")
        print(datetime.datetime.now())
        print("Epoch: {}, Step: {}".format(epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):
            batch_size = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / batch_size)
            dnn_start_time = time.time()

            if global_step <= 1000 and global_step % 50 == 0 and args.warmup:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warm_lr
                print("Warm-up learning rate is {:.6f}".format(optimizer.param_groups[0]["lr"]))

            with autocast():
                audio_output = audio_model(audio_input)
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(audio_output, labels)

            if not torch.isfinite(loss):
                print("Encountered non-finite loss.")
                print("  Epoch {}, global_step {}, batch {}".format(epoch, global_step, i))
                print(
                    "  audio_output stats: min={}, max={}, has_nan={}, has_inf={}".format(
                        audio_output.min().item(),
                        audio_output.max().item(),
                        torch.isnan(audio_output).any().item(),
                        torch.isinf(audio_output).any().item(),
                    )
                )
                print(
                    "  labels stats: min={}, max={}".format(
                        labels.min().item(), labels.max().item()
                    )
                )
                print(
                    "  input stats: min={}, max={}, mean={}, std={}".format(
                        audio_input.min().item(),
                        audio_input.max().item(),
                        audio_input.mean().item(),
                        audio_input.std().item(),
                    )
                )
                raise ValueError("Stopping due to non-finite loss.")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), batch_size)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / batch_size)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / batch_size)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\tTrain Loss {loss_meter.avg:.4f}".format(
                        epoch, i, len(train_loader), loss_meter=loss_meter
                    )
                )
            end_time = time.time()
            global_step += 1

        stats, valid_loss = validate(audio_model, val_loader, args)
        if stats and isinstance(stats[0], dict):
            mAP = stats[0].get("AP", 0)
        else:
            mAP = 0

        print("Validation - mAP: {:.6f}, Loss: {:.4f}".format(mAP, valid_loss))

        history.append(
            {"epoch": epoch, "train_loss": loss_meter.avg, "val_loss": valid_loss, "mAP": mAP}
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(audio_model.state_dict(), os.path.join(exp_dir, "models", "best_audio_model.pth"))
            print("Best model updated based on val_loss")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            print("Early stop patience: {}/{}".format(early_stop_cnt, patience))

        if early_stop_cnt >= patience:
            print("Early stopping triggered after {} epochs without improvement".format(patience))
            break

        scheduler.step()
        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    history_path = os.path.join(exp_dir, "models", "training_history.pkl")
    with open(history_path, "wb") as fp:
        pickle.dump(history, fp)
    print("Training history saved to", history_path)


def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model.to(device)
    audio_model.eval()

    all_predictions = []
    all_targets = []
    all_losses = []

    with torch.no_grad():
        for audio_input, labels in val_loader:
            audio_input = audio_input.to(device)
            labels = labels.to(device)

            logits = audio_model(audio_input)
            loss = args.loss_fn(logits, labels)
            all_losses.append(loss.cpu().numpy())

            probs = torch.sigmoid(logits)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    audio_output = np.vstack(all_predictions)
    target = np.vstack(all_targets)

    if np.isnan(audio_output).any() or np.isinf(audio_output).any():
        print("Warning: Detected NaN or Inf in validation predictions; replacing with finite values.")
        audio_output = np.nan_to_num(audio_output, nan=0.5, posinf=1.0, neginf=0.0)

    true_labels = np.argmax(target, axis=1)
    pred_scores = audio_output[:, 1] if audio_output.shape[1] > 1 else audio_output[:, 0]
    pred_labels = np.argmax(audio_output, axis=1)

    bal_acc = balanced_accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average="binary")
    rec = recall_score(true_labels, pred_labels, average="binary")

    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)

    TN, FP, FN, TP = cm.ravel()
    total_neg = TN + FP
    total_pos = TP + FN
    fpr = FP / total_neg if total_neg > 0 else 0
    fnr = FN / total_pos if total_pos > 0 else 0

    stats = calculate_stats(audio_output, target)
    mean_AP = np.mean([s.get("AP", 0) for s in stats])

    fpr_curve, tpr_curve, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = roc_auc_score(true_labels, pred_scores)

    updated_stats = {
        "balanced_acc": bal_acc,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
        "cm": cm.tolist(),
        "AP": mean_AP,
        "roc_auc": roc_auc,
        "fpr_curve": fpr_curve.tolist(),
        "tpr_curve": tpr_curve.tolist(),
        "thresholds": thresholds.tolist(),
    }

    return [updated_stats], np.mean(all_losses)
