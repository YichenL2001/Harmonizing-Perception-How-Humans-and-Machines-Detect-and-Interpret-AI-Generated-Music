import argparse
import gc
import logging
import os
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

import torch

try:
    from torch.amp import GradScaler, autocast

    torch_amp_new = True
except:
    from torch.cuda.amp import GradScaler, autocast

    torch_amp_new = False
import torch.distributed as dist
import torch.multiprocessing as mp

from vit_spectttra_convnext.models.model import AudioClassifier
from vit_spectttra_convnext.utils.config import dict2cfg
from vit_spectttra_convnext.utils.dataset import get_dataloader
from vit_spectttra_convnext.utils.metrics import (
    AverageMeter,
    AccuracyMeter,
    F1Meter,
    SensitivityMeter,
    SpecificityMeter,
    get_part_result,
)
from vit_spectttra_convnext.utils.losses import BCEWithLogitsLoss, SigmoidFocalLoss
from vit_spectttra_convnext.utils.perf import profile_model

# from sonics.utils.scheduler import get_cosine_schedule_with_warmup, get_scheduler
from vit_spectttra_convnext.utils.seed import set_seed, worker_init_fn

from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("fvcore").setLevel(logging.ERROR)

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))


def train_loop(
    model, train_dataloader, criterion, optimizer, scaler, device, cfg, scheduler=None
):
    model.train()
    running_loss = AverageMeter()
    accuracy = AverageMeter()
    f1 = AverageMeter()
    sensitivity = AverageMeter()
    specificity = AverageMeter()
    # gpu = AverageMeter()
    progress_bar = tqdm(
        train_dataloader, desc="Train", ncols=150, bar_format="{l_bar}{bar:5}{r_bar}"
    )

    # Automatically set accumulation_steps based on the batch size
    # batch_size = cfg.training.batch_size
    # accumulation_steps = max(1, 32 // batch_size)

    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        x, y = batch["audio"], batch["target"]
        x, y = x.to(device), y.to(device)
        if cfg.environment.mixed_precision:
            with autocast("cuda") if torch_amp_new else autocast():
                preds, y = model(x, y)
                preds = preds.squeeze()
                loss = criterion(preds, y) / cfg.optimizer.grad_accum_steps
            scaler.scale(loss).backward()
        else:
            preds, y = model(x, y)
            preds = preds.squeeze()
            loss = criterion(preds, y) / cfg.optimizer.grad_accum_steps
            loss.backward()

        if (i + 1) % cfg.optimizer.grad_accum_steps == 0:
            """Ref: https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping"""
            clip_grad_norm = getattr(cfg.optimizer, "clip_grad_norm", None)
            if clip_grad_norm is not None:
                if cfg.environment.mixed_precision:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            if cfg.environment.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # if scheduler is not None:
            #     scheduler.step()

        running_loss.update(loss.item() * cfg.optimizer.grad_accum_steps, x.size(0))

        preds = torch.sigmoid(preds).cpu().detach().numpy()
        targets = (y.cpu().numpy() > 0.5).astype(int)
        pred_labels = (preds > 0.5).astype(int)
        accuracy.update(balanced_accuracy_score(targets, pred_labels), x.size(0))
        f1.update(f1_score(targets, pred_labels, average="binary"), x.size(0))
        sensitivity.update(
            recall_score(targets, pred_labels, pos_label=1, average="binary"), x.size(0)
        )
        specificity.update(
            recall_score(targets, pred_labels, pos_label=0, average="binary"), x.size(0)
        )
        lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            loss=running_loss.avg,
            acc=accuracy.avg,
            f1=f1.avg,
            sens=sensitivity.avg,
            spec=specificity.avg,
            lr=f"{lr:0.8f}",
        )
        torch.cuda.empty_cache()
        gc.collect()

    return (
        running_loss.avg,
        accuracy.avg,
        f1.avg,
        sensitivity.avg,
        specificity.avg,
        lr,
    )


def valid_loop(model, valid_dataloader, criterion, device, cfg, desc="Valid"):
    model.eval()
    running_loss = AverageMeter()
    accuracy = AccuracyMeter()
    f1 = F1Meter()
    sensitivity = SensitivityMeter()
    specificity = SpecificityMeter()
    progress_bar = tqdm(
        valid_dataloader, desc=desc, ncols=150, bar_format="{l_bar}{bar:5}{r_bar}"
    )

    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in progress_bar:
            x, y = batch["audio"], batch["target"]
            x, y = x.to(device), y.to(device)

            if cfg.environment.mixed_precision:
                with autocast("cuda") if torch_amp_new else autocast():
                    preds = model(x)
            else:
                preds = model(x)

            preds = preds.squeeze()
            loss = criterion(preds, y)
            running_loss.update(loss.item(), x.size(0))

            preds = torch.sigmoid(preds).cpu().numpy()
            y_pred_list.append(preds)

            targets = y.cpu().numpy().astype(int)
            y_true_list.append(targets)

            pred_labels = (preds > 0.5).astype(int)
            accuracy.update(targets, pred_labels)
            f1.update(targets, pred_labels)
            sensitivity.update(targets, pred_labels)  # TPR
            specificity.update(targets, pred_labels)  # TNR

            progress_bar.set_postfix(
                loss=running_loss.avg,
                acc=accuracy.avg,
                f1=f1.avg,
                sens=sensitivity.avg,
                spec=specificity.avg,
            )

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    pred_df = pd.DataFrame(
        {"y_true": np.concatenate(y_true_list), "y_pred": np.concatenate(y_pred_list)}
    )

    return (
        running_loss.avg,
        accuracy.avg,
        f1.avg,
        sensitivity.avg,
        specificity.avg,
        pred_df,
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def main():
    # Parse arguments
    args = arg_parser()
    dict_ = yaml.safe_load(open(args.config).read())
    cfg = dict2cfg(dict_)
    print(cfg)

    # Create output directory (allows override via OUTPUT_ROOT env var)
    output_root = os.getenv("OUTPUT_ROOT", "output")
    exp_dir = os.path.join(output_root, cfg.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    cfg.environment.exp_dir = exp_dir

    # Set seed
    set_seed(cfg.environment.seed)

    # Set up distributed training
    cfg.environment.world_size = torch.cuda.device_count()
    cfg.environment.distributed = cfg.environment.world_size > 1
    cfg.environment.dist_backend = "nccl" if cfg.environment.distributed else None

    # Start training
    if cfg.environment.distributed:
        mp.spawn(main_worker, nprocs=cfg.environment.world_size, args=(cfg,))
    else:
        main_worker(0, cfg)


def main_worker(gpu, cfg):
    # Initialize distributed training
    cfg.environment.gpu = gpu
    cfg.environment.rank = gpu
    exp_dir = getattr(cfg.environment, "exp_dir", None)
    if exp_dir is None:
        raise RuntimeError("Expected cfg.environment.exp_dir to be set before launching workers.")

    if cfg.environment.distributed:
        dist.init_process_group(
            backend=cfg.environment.dist_backend,
            init_method=f"tcp://localhost:12355",
            world_size=cfg.environment.world_size,
            rank=cfg.environment.rank,
        )

    if not torch.cuda.is_available():
        print("> Using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(cfg.environment.gpu)
        device = torch.device(f"cuda:{cfg.environment.gpu}")
        print(f"> Using GPU: {cfg.environment.gpu}")

    # Load metadata
    train_df = pd.read_csv(cfg.dataset.train_dataframe)
    valid_df = pd.read_csv(cfg.dataset.valid_dataframe)
    test_df = pd.read_csv(cfg.dataset.test_dataframe)

    # Shuffle data
    train_df = train_df.sample(frac=1.0, random_state=cfg.environment.seed).reset_index(
        drop=True
    )
    valid_df = valid_df.sample(frac=1.0, random_state=cfg.environment.seed).reset_index(
        drop=True
    )
    test_df = test_df.sample(frac=1.0, random_state=cfg.environment.seed).reset_index(
        drop=True
    )

    # Store data stats
    cfg.dataset.num_train = len(train_df)
    cfg.dataset.num_train_real = len(train_df.query("target == 0"))
    cfg.dataset.num_train_fake = len(train_df.query("target == 1"))

    cfg.dataset.num_valid = len(valid_df)
    cfg.dataset.num_valid_real = len(valid_df.query("target == 0"))
    cfg.dataset.num_valid_fake = len(valid_df.query("target == 1"))

    cfg.dataset.num_test = len(test_df)
    cfg.dataset.num_test_real = len(test_df.query("target == 0"))
    cfg.dataset.num_test_fake = len(test_df.query("target == 1"))

    # --- helpers right before the dataloaders (optional) ---
    def col_or_none(df, name):
        return df[name].tolist() if name in df.columns else None

    # --- Load dataloaders ---
    train_dataloader = get_dataloader(
        train_df.filepath.tolist(),
        train_df.target.tolist(),
        skip_times=train_df.skip_time.tolist() if getattr(cfg.audio, "skip_time", False) and "skip_time" in train_df.columns else None,
        max_len=cfg.audio.max_len,
        batch_size=cfg.training.batch_size,
        num_classes=cfg.num_classes,
        train=True,
        # If you want to use EXACTLY the preselected 10s, set False here:
        random_sampling=False,
        num_workers=cfg.environment.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=None,
        distributed=cfg.environment.distributed,
        # NEW: fixed 10s window from CSV
        seg_starts=col_or_none(train_df, "seg_start"),
        seg_ends=col_or_none(train_df, "seg_end"),
    )

    valid_dataloader = get_dataloader(
        valid_df.filepath.tolist(),
        valid_df.target.tolist(),
        skip_times=valid_df.skip_time.tolist() if getattr(cfg.audio, "skip_time", False) and "skip_time" in valid_df.columns else None,
        max_len=cfg.audio.max_len,
        batch_size=cfg.validation.batch_size,
        num_classes=cfg.num_classes,
        train=False,
        random_sampling=False,  # keep eval deterministic
        num_workers=cfg.environment.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=None,
        distributed=cfg.environment.distributed,
        seg_starts=col_or_none(valid_df, "seg_start"),
        seg_ends=col_or_none(valid_df, "seg_end"),
    )

    test_dataloader = get_dataloader(
        test_df.filepath.tolist(),
        test_df.target.tolist(),
        skip_times=test_df.skip_time.tolist() if getattr(cfg.audio, "skip_time", False) and "skip_time" in test_df.columns else None,
        max_len=cfg.audio.max_len,
        batch_size=cfg.validation.batch_size,
        num_classes=cfg.num_classes,
        train=False,
        random_sampling=False,  # keep eval deterministic
        num_workers=cfg.environment.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=None,
        distributed=cfg.environment.distributed,
        seg_starts=col_or_none(test_df, "seg_start"),
        seg_ends=col_or_none(test_df, "seg_end"),
    )

    # Load model
    model = AudioClassifier(cfg)
    model.to(device)

    # Profile model
    if cfg.environment.gpu == 0:
        print("\n> Model Profile:")
        input_tensor = torch.randn((cfg.training.batch_size, cfg.audio.max_len)).to(
            device
        )
        profile_df = profile_model(model, input_tensor, display=True)

    # Distributed Model
    if cfg.environment.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.environment.gpu]
        )

    # Learning Rate
    if not cfg.scheduler.lr:
        global_batch_size = (
            cfg.training.batch_size
            * cfg.environment.world_size
            * cfg.optimizer.grad_accum_steps
        )
        batch_ratio = global_batch_size / cfg.scheduler.lr_base_size
        if cfg.scheduler.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        cfg.scheduler.lr = cfg.scheduler.lr_base * batch_ratio

    # Optimizer
    opt_cfg = getattr(cfg, "optimizer")
    opt_cfg.lr = cfg.scheduler.lr
    optimizer = create_optimizer_v2(model.parameters(), **optimizer_kwargs(opt_cfg))

    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float("-inf")
    run_id = None
    if cfg.model.resume:
        if not os.path.exists(cfg.model.resume):
            raise FileNotFoundError(f"> Checkpoint file not found: {cfg.model.resume}")
        
        checkpoint = torch.load(cfg.model.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        if cfg.environment.gpu == 0:
            print(f"> Resuming training from epoch {start_epoch + 1}")

    # LR Scheduler
    sched_cfg = getattr(cfg, "scheduler")
    sched_cfg.epochs = cfg.training.epochs
    sched_cfg.start_epoch = start_epoch
    steps_per_epoch = len(train_dataloader)
    updates_per_epoch = (
        steps_per_epoch + cfg.optimizer.grad_accum_steps - 1
    ) // cfg.optimizer.grad_accum_steps
    scheduler, num_epochs = create_scheduler_v2(
        optimizer, **scheduler_kwargs(sched_cfg), updates_per_epoch=updates_per_epoch
    )
    if start_epoch > 0:
        scheduler.step(start_epoch)

    # Loss
    if cfg.loss.name == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss(label_smoothing=cfg.loss.label_smoothing)
    elif cfg.loss.name == "SigmoidFocalLoss":
        criterion = SigmoidFocalLoss(
            alpha=cfg.loss.alpha,
            gamma=cfg.loss.gamma,
            label_smoothing=cfg.loss.label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss function: {cfg.loss.name}")

    # Mixed Precision
    scaler = (
        (GradScaler("cuda") if torch_amp_new else GradScaler())
        if cfg.environment.mixed_precision
        else None
    )

    # Training and validation loop
    best_loss = np.inf
    best_acc = -1
    best_epoch = -1

    patience  = getattr(cfg.training, "early_stopping_patience", None)   # e.g., 5
    min_delta = float(getattr(cfg.training, "early_stopping_min_delta", 0.0))
    no_improve = 0

    if cfg.environment.gpu == 0:
        print("\n> Training:")

    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.environment.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        if cfg.environment.gpu == 0:
            print(f"EPOCH: {epoch+1}/{cfg.training.epochs}")

        (
            train_loss,
            train_acc,
            train_f1,
            train_sens,
            train_spec,
            lr,
        ) = train_loop(
            model,
            train_dataloader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg,
            scheduler,
        )
        (
            val_loss,
            val_acc,
            val_f1,
            val_sens,
            val_spec,
            valid_pred_df,
        ) = valid_loop(model, valid_dataloader, criterion, device, cfg)

        # Get the current metric value based on the primary_metric
        current_metric = locals()[f"val_{cfg.logger.primary_metric}"]

        # Save checkpoint for best result
        if cfg.environment.gpu == 0:
            metric_improved = current_metric > (best_metric + min_delta)
            # if improved:
            #     no_improve = 0
            # else:
            #     no_improve += 1
            loss_improved = val_loss < (best_loss - min_delta)
            if loss_improved:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if cfg.logger.primary_metric not in [
                "f1",
                "acc",
                "sens",
                "spec",
            ]:
                raise ValueError(
                    f"Invalid primary_metric: {cfg.logger.primary_metric}. Must be 'f1', 'acc', 'sens', or 'spec'."
                )

            is_best = metric_improved

            if is_best:
                print(
                    print(f"> {cfg.logger.primary_metric.upper()} improved from {best_metric:.4f} to {current_metric:.4f}")
                )
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_sens": val_sens,
                    "val_spec": val_spec,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "train_sens": train_sens,
                    "train_spec": train_spec,
                    "best_metric": current_metric,
                }
                torch.save(
                    checkpoint, os.path.join(exp_dir, "best_checkpoint.pth")
                )

                # Save validation predictions
                valid_df = valid_df[
                    : len(valid_pred_df)
                ]  # in case valid_df is longer than valid_pred_df
                valid_pred_df = pd.concat([valid_df, valid_pred_df], axis=1)

                print(f"> Saving validation predictions to {exp_dir}/valid_predictions.csv")
                valid_pred_df.to_csv(
                    os.path.join(exp_dir, "valid_predictions.csv"), index=False
                )

                best_metric = current_metric
                best_epoch = epoch + 1
                best_valid_result = {
                    "loss": val_loss,
                    "acc": val_acc,
                    "f1": val_f1,
                    "sens": val_sens,
                    "spec": val_spec,
                    "epoch": best_epoch,
                }

            # Save checkpoint for last successful epoch for resuming
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_sens": val_sens,
                "val_spec": val_spec,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_sens": train_sens,
                "train_spec": train_spec,
                "best_metric": best_metric,
            }
            torch.save(checkpoint, os.path.join(exp_dir, "last_checkpoint.pth"))

            # Update lr for next epoch
            if scheduler is not None:
                scheduler.step(epoch + 1)
        else:
            improved = False
        
        stop_now = False
        if cfg.environment.gpu == 0 and patience is not None and no_improve >= patience:
            print(f"> Early stopping at epoch {epoch+1}: no improvement in {patience} epochs (min_delta={min_delta}).")
            stop_now = True

        if cfg.environment.distributed:
            flag = torch.tensor([1 if stop_now else 0], device=device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())

        if stop_now:
            break
        print()
        

    # Display best result of valid and test in markdown
    if cfg.environment.gpu == 0:
        print("> Best Validation Result:")
        best_valid_result_df = pd.DataFrame([best_valid_result])
        print(best_valid_result_df.to_markdown(index=False, tablefmt="grid"))
        print()

        # Load best model for test inference
        print("> Loading best model")
        checkpoint = torch.load(
            os.path.join(exp_dir, "best_checkpoint.pth"), map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])

        # Test loop
        (
            test_loss,
            test_acc,
            test_f1,
            test_sens,
            test_spec,
            test_pred_df,
        ) = valid_loop(model, test_dataloader, criterion, device, cfg, desc="Test")
        best_test_result = {
            "loss": test_loss,
            "acc": test_acc,
            "f1": test_f1,
            "sens": test_sens,
            "spec": test_spec,
        }

        print("> Best Test Result:")
        best_test_result_df = pd.DataFrame([best_test_result])
        print(best_test_result_df.to_markdown(index=False, tablefmt="grid"))
        print()

        test_df = test_df[
            : len(test_pred_df)
        ]  # in case test_df is longer than test_pred_df
        test_pred_df = pd.concat([test_df, test_pred_df], axis=1)

        # Get partition results
        available = set(test_pred_df.columns)
        need_any = {"artist_overlap", "algorithm", "label", "duration"}
        if available & need_any:
            part_result_df, part_result_dict = get_part_result(test_pred_df)
            print("> Test Partition Results:")
            print(part_result_df.to_markdown(index=False))
            print()
        else:
            print("> Skipping partition results: no grouping metadata found.")
            part_result_df, part_result_dict = pd.DataFrame(), {}

        # Save test prediction
        print(f"> Saving test predictions to {exp_dir}/test_predictions.csv")
        test_pred_df.to_csv(
            os.path.join(exp_dir, "test_predictions.csv"), index=False
        )

    # Tear down the process group
    if cfg.environment.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
