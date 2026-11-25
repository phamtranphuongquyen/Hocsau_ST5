"""Huấn luyện chỉ Training; lựa chọn checkpoint chỉ theo PublicTest loss."""

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from data_loader import DATA_DIR, PROJECT_ROOT, create_loaders, seed_everything
from model import CHECKPOINT_PATH, EmotionCNN, save_checkpoint

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
PATIENCE = 8
SEED = 42


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Loss NaN/Inf: dừng, không lưu checkpoint lỗi."
                )
            if training:
                loss.backward()
                # Kiểm tra gradient, đồng thời hạn chế gradient quá lớn.
                nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    if count == 0:
        raise ValueError("DataLoader rỗng.")
    return total_loss / count, correct / count


def plot_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, history[f"train_{metric}"], label="Training")
        axis.plot(epochs, history[f"valid_{metric}"], label="PublicTest (validation)")
        axis.set(xlabel="Epoch", ylabel=metric.title())
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.patience < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("epochs, patience và learning-rate phải là số dương hữu hạn.")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = create_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        not args.no_augmentation,
        args.seed,
        splits=("train", "valid"),
    )
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {
        key: []
        for key in ("train_loss", "valid_loss", "train_accuracy", "valid_accuracy")
    }
    best_loss, stale_epochs = float("inf"), 0
    print(f"Device: {device}; augmentation: {not args.no_augmentation}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        valid_loss, valid_accuracy = run_epoch(model, valid_loader, criterion, device)
        for key, value in zip(
            history, (train_loss, valid_loss, train_accuracy, valid_accuracy)
        ):
            history[key].append(value)
        improved = valid_loss < best_loss
        if improved:
            best_loss, stale_epochs = valid_loss, 0
            save_checkpoint(
                model,
                args.checkpoint,
                epoch=epoch,
                validation_loss=valid_loss,
                validation_accuracy=valid_accuracy,
                history=history,
                config={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                train_size=len(train_loader.dataset),
                valid_size=len(valid_loader.dataset),
            )
        else:
            stale_epochs += 1
        plot_history(history, args.results_dir / "training_history.png")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
            f"valid loss={valid_loss:.4f} acc={valid_accuracy:.4f} | "
            f"{time.perf_counter() - start:.1f}s{' | saved best' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(
                f"EarlyStopping: {args.patience} epoch không cải thiện validation loss."
            )
            break
    print(f"Checkpoint tốt nhất: {args.checkpoint}; validation loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
