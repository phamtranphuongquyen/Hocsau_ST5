import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASSES, load_datasets
from models import EmotionCNN
from train import RESULTS_DIR

DEPTHS = (2, 3, 4)
MAX_EPOCHS = 40
LABELS = list(range(len(CLASSES)))


@torch.inference_mode()
def predict(model, loader, device):
    actual, predicted = [], []

    for images, labels in loader:
        logits = model(images.to(device))
        actual.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())

    return actual, predicted


def save_confusion_matrix(actual, predicted, model_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=LABELS,
        display_labels=CLASSES,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set(
        title=f"{model_name} trên PrivateTest",
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thật",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_last_conv_filters(model, model_name, output_path):
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    weights = conv_layers[-1].weight.detach().cpu().numpy()
    kernels = weights[:16, 0]
    limit = max(float(np.abs(kernels).max()), 1e-8)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6), layout="constrained")
    for number, (ax, kernel) in enumerate(zip(axes.flat, kernels), start=1):
        plotted = ax.imshow(kernel, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"Bộ lọc {number}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{model_name} — 16 bộ lọc lớp tích chập cuối, kênh đầu vào 1")
    fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=0.75, label="Trọng số")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_comparison_chart(summary):
    positions = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.5))

    accuracy_bars = ax.bar(
        positions - width / 2, summary["test_accuracy"] * 100, width,
        label="Accuracy",
    )
    f1_bars = ax.bar(
        positions + width / 2, summary["test_macro_f1"] * 100, width,
        label="Macro-F1",
    )
    ax.bar_label(accuracy_bars, fmt="%.1f", padding=3)
    ax.bar_label(f1_bars, fmt="%.1f", padding=3)
    ax.set_xticks(positions, summary["model"])
    ax.set(
        title="So sánh ba kiến trúc CNN trên PrivateTest",
        ylabel="Điểm (%)",
        ylim=(0, 110),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_comparison.png", dpi=200)
    plt.close(fig)


def main():
    run_dirs = {
        depth: RESULTS_DIR / f"cnn_{depth}_e{MAX_EPOCHS}"
        for depth in DEPTHS
    }

    for output_dir in run_dirs.values():
        for filename in ("best.pt", "history.csv", "learning_curves.png"):
            if not (output_dir / filename).is_file():
                raise SystemExit(f"Chưa thấy {filename} trong {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = load_datasets()["PrivateTest"]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Đánh giá {len(test_dataset):,} ảnh PrivateTest trên {device}.")

    records = []
    for depth, output_dir in run_dirs.items():
        model_name = f"CNN-{depth}"
        print(f"Đang đánh giá {model_name}...", flush=True)

        model = EmotionCNN(depth).to(device)
        weights = torch.load(
            output_dir / "best.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(weights)
        model.eval()

        actual, predicted = predict(model, test_loader, device)
        test_accuracy = accuracy_score(actual, predicted)
        test_macro_f1 = f1_score(
            actual, predicted, labels=LABELS, average="macro", zero_division=0
        )

        history = pd.read_csv(output_dir / "history.csv")
        best_row = history.loc[history["val_loss"].idxmin()]
        records.append({
            "model": model_name,
            "epochs_ran": len(history),
            "best_epoch": int(best_row["epoch"]),
            "val_loss": best_row["val_loss"],
            "val_accuracy": best_row["val_accuracy"],
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
        })

        report = classification_report(
            actual, predicted, labels=LABELS, target_names=CLASSES,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            output_dir / "classification_report.csv"
        )
        save_confusion_matrix(
            actual, predicted, model_name, output_dir / "confusion_matrix.png"
        )
        save_last_conv_filters(
            model, model_name, output_dir / "last_conv_filters.png"
        )

        print(f"{model_name} | Accuracy: {test_accuracy:.2%} | Macro-F1: {test_macro_f1:.2%}")

    summary = pd.DataFrame(records)
    summary.to_csv(RESULTS_DIR / "test_summary.csv", index=False)
    save_comparison_chart(summary)
    print(f"\nĐã lưu bảng và biểu đồ so sánh tại {RESULTS_DIR}")


if __name__ == "__main__":
    main()