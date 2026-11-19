import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASSES, load_datasets
from models import EmotionCNN

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 7
SEED = 42


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images


def save_augmentation_examples(train_dataset, output_path):
    # Đọc cùng một ảnh với phép biến đổi của tập validation để lấy ảnh gốc.
    from dataset import eval_transform, FERDataset

    row = {
        "pixels": [train_dataset.pixels[0]],
        "emotion": [train_dataset.labels[0]],
    }
    original_dataset = FERDataset(pd.DataFrame(row), eval_transform)
    original_image, label = original_dataset[0]

    images = [original_image]
    for _ in range(4):
        augmented_image, _ = train_dataset[0]
        images.append(augmented_image)

    fig, axes = plt.subplots(1, 5, figsize=(11, 3))
    for index, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Ảnh gốc" if index == 0 else f"Biến đổi {index}")
        ax.axis("off")

    fig.suptitle(f"Data augmentation trên ảnh huấn luyện ({CLASSES[label]})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_learning_curves(history, output_path, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history["epoch"], history[f"train_{metric}"], "o-", label="Train")
        ax.plot(history["epoch"], history[f"val_{metric}"], "s-", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.25)

    if len(history) == 1:
        for ax in axes:
            ax.set_xticks([1])

    fig.suptitle(f"Kết quả huấn luyện {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs phải từ 1 trở lên")

    model_name = f"CNN-{args.model}"
    output_dir = RESULTS_DIR / f"cnn_{args.model}_e{args.epochs}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chuẩn bị dữ liệu cho {model_name} trên {device}.", flush=True)

    datasets = load_datasets()
    save_augmentation_examples(
        datasets["Training"], output_dir / "augmentation_examples.png"
    )

    # Đặt seed sau khi vẽ ảnh để việc tạo hình không làm đổi chuỗi ngẫu nhiên khi train.
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        datasets["Training"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        datasets["PublicTest"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EmotionCNN(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    records = []

    for epoch in range(1, args.epochs + 1):
        print(f"Đang chạy epoch {epoch}/{args.epochs}...", flush=True)
        start_time = time.perf_counter()

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        history = pd.DataFrame(records)
        history.to_csv(output_dir / "history.csv", index=False)

        elapsed = time.perf_counter() - start_time
        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | Acc: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f} | Acc: {val_accuracy:.2%} | "
            f"{elapsed:.1f} giây",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Dừng sớm sau {PATIENCE} epoch không giảm validation loss.")
                break

    save_learning_curves(history, output_dir / "learning_curves.png", model_name)
    print(f"Epoch tốt nhất: {best_epoch} | Validation loss: {best_val_loss:.4f}")
    print(f"Đã lưu kết quả tại: {output_dir}")


if __name__ == "__main__":
    main()