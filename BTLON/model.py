"""CNN duy nhất và các hàm checkpoint dùng chung."""

import argparse
import hashlib
import pickle
import tempfile
from pathlib import Path

import torch
from torch import nn

from data_loader import (
    CLASSES,
    DATA_DIR,
    PREPROCESSING,
    PROJECT_ROOT,
    create_loaders,
    seed_everything,
)

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def checkpoint_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(model, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **metadata,
        "model_state_dict": model.state_dict(),
        "classes": list(CLASSES),
        "preprocessing": PREPROCESSING,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có checkpoint: {path}. Hãy chạy python train.py."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Không đọc được checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("classes") != CLASSES:
        raise ValueError("Thứ tự lớp trong checkpoint không khớp CLASSES.")
    if checkpoint.get("preprocessing") != PREPROCESSING:
        raise ValueError(
            "Tiền xử lý trong checkpoint không khớp grayscale 48x48, [0, 1]."
        )
    model = EmotionCNN(len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Forward CNN bằng batch Training thật; không lưu checkpoint."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Kiểm tra backward và checkpoint tạm bằng batch Training thật.",
    )
    args = parser.parse_args()
    seed_everything()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    (loader,) = create_loaders(
        args.data_dir, batch_size=8, augment=False, splits=("train",)
    )
    images, _ = next(iter(loader))
    model = EmotionCNN().eval()
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (len(images), len(CLASSES)) and torch.isfinite(logits).all()
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input: {tuple(images.shape)}; logits: {tuple(logits.shape)}")
    if args.self_test:
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            seed_everything()
            model = EmotionCNN().to(device).train()
            images, labels = next(iter(loader))
            images, labels = images.to(device), labels.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            before = model.classifier[-1].weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss()(model(images), labels)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            optimizer.step()
            assert not torch.equal(before, model.classifier[-1].weight)
            model.eval()
            with torch.inference_mode():
                expected = model(images)
            # Checkpoint từ một bước học thật, chỉ tồn tại trong thư mục tạm.
            with tempfile.TemporaryDirectory(prefix="fer-checkpoint-") as directory:
                path = Path(directory) / "batch_check.pth"
                save_checkpoint(model, path, smoke_test=True)
                restored, metadata = load_checkpoint(path, device)
                with torch.inference_mode():
                    torch.testing.assert_close(
                        restored(images), expected, rtol=0, atol=0
                    )
                assert metadata["classes"] == CLASSES
            print(
                f"PASS {device}: forward, backward, optimizer step, checkpoint round-trip."
            )


if __name__ == "__main__":
    main()
