"""Đánh giá cuối cùng trên PrivateTest, không cập nhật trọng số."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import CLASSES, DATA_DIR, PROJECT_ROOT, create_loaders
from model import CHECKPOINT_PATH, checkpoint_sha256, load_checkpoint


def evaluate(model, loader, device):
    if loader.dataset.split != "test" or loader.dataset.augment:
        raise ValueError("Đánh giá chỉ được dùng PrivateTest, không augmentation.")
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Logits chứa NaN/Inf.")
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not targets:
        raise ValueError("Tập PrivateTest rỗng.")
    report = classification_report(
        targets,
        predictions,
        labels=list(range(7)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(7)))
    return {
        "split": "PrivateTest",
        "num_samples": len(targets),
        "classes": CLASSES,
        "accuracy": accuracy_score(targets, predictions),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    (loader,) = create_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        augment=False,
        splits=("test",),
    )
    metrics, matrix = evaluate(model, loader, device)
    metrics.update(
        checkpoint_sha256=checkpoint_sha256(args.checkpoint),
        checkpoint_epoch=checkpoint.get("epoch"),
        evaluated_at=datetime.now(UTC).isoformat(),
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False
    )
    axis.set_title(f"PrivateTest (n={metrics['num_samples']})")
    figure.tight_layout()
    figure.savefig(args.results_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"{key}: {metrics[key]:.6f}")
    print(
        f"Đã đánh giá {metrics['num_samples']} ảnh PrivateTest. Kết quả: {args.results_dir}"
    )


if __name__ == "__main__":
    main()
