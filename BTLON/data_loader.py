"""Đọc JPG gốc; không sao chép, xuất ảnh hoặc chia dữ liệu ngẫu nhiên."""

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "BTLON02" / "DATA"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SPLITS = {
    "train": ("train", "Training_"),
    "valid": ("test", "PublicTest_"),
    "test": ("test", "PrivateTest_"),
}
PREPROCESSING = {"mode": "L", "size": [48, 48], "scale": "0..1"}


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


def build_transform(augment=False):
    steps = [transforms.Grayscale(1), transforms.Resize((48, 48))]
    if augment:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    return transforms.Compose(steps + [transforms.ToTensor()])


def preprocess_image(image):
    """Tiền xử lý xác định dùng cho validation, testing và upload."""
    return build_transform(False)(ImageOps.exif_transpose(image))


def scan_dataset(data_dir=DATA_DIR):
    """Kiểm tra toàn bộ tên/nhãn/đường dẫn; chưa giải mã pixel ảnh."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")
    samples = {split: [] for split in SPLITS}
    seen = set()
    for folder in ("train", "test"):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục: {directory}")
        actual = {p.name for p in directory.iterdir() if p.is_dir()}
        if actual != set(CLASSES):
            raise ValueError(
                f"{directory}: thiếu lớp {set(CLASSES) - actual}; lớp lạ {actual - set(CLASSES)}"
            )
        for path in sorted(directory.rglob("*")):
            # Windows shortcuts (for example ``DO_AN.lnk``) are not dataset
            # samples. Only JPG files are consumed by this project.
            if not path.is_file() or path.suffix.lower() != ".jpg":
                continue
            relative = path.relative_to(directory)
            if len(relative.parts) != 2 or path.parent.name not in CLASSES:
                raise ValueError(f"Ảnh không nằm trực tiếp trong thư mục lớp: {path}")
            if path.suffix.lower() != ".jpg":
                raise ValueError(f"Chỉ chấp nhận ảnh JPG gốc: {path}")
            matching = [
                s
                for s, (f, prefix) in SPLITS.items()
                if f == folder and path.name.startswith(prefix)
            ]
            if len(matching) != 1:
                raise ValueError(f"Sai tiền tố hoặc sai tập dữ liệu: {path}")
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Ảnh trỏ ra ngoài thư mục dữ liệu: {path}")
            if resolved in seen:
                raise ValueError(f"Đường dẫn ảnh trùng giữa/trong các tập: {path}")
            seen.add(resolved)
            samples[matching[0]].append((resolved, CLASSES.index(path.parent.name)))
    for split, items in samples.items():
        counts = Counter(label for _, label in items)
        missing = [name for label, name in enumerate(CLASSES) if not counts[label]]
        if missing:
            raise ValueError(f"Tập {split} thiếu ảnh của lớp: {missing}")
    return samples


def print_counts(samples):
    print(
        f"{'Split':<10}" + "".join(f"{name:>10}" for name in CLASSES) + f"{'Total':>10}"
    )
    for split, items in samples.items():
        counts = Counter(label for _, label in items)
        print(
            f"{split:<10}"
            + "".join(f"{counts[i]:>10}" for i in range(7))
            + f"{len(items):>10}"
        )


class FERDataset(Dataset):
    def __init__(self, samples, split, augment=False):
        if split not in SPLITS:
            raise ValueError(f"Tập không hợp lệ: {split}")
        self.samples = list(samples)
        self.split = split
        self.augment = augment and split == "train"
        self.transform = build_transform(True) if self.augment else None
        folder, prefix = SPLITS[split]
        for path, label in self.samples:
            if label not in range(len(CLASSES)):
                raise ValueError(f"Nhãn phải nằm trong 0..6: {path}, nhãn={label}")
            if (
                not path.name.startswith(prefix)
                or path.parent.parent.name != folder
                or path.parent.name != CLASSES[label]
            ):
                raise ValueError(f"Ảnh/nhãn không thuộc tập {split}: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image.load()
                tensor = (
                    self.transform(ImageOps.exif_transpose(image))
                    if self.augment
                    else preprocess_image(image)
                )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise RuntimeError(f"Không đọc được ảnh {path}: {exc}") from exc
        return tensor, label


def create_loaders(
    data_dir=DATA_DIR,
    batch_size=64,
    num_workers=0,
    augment=True,
    seed=42,
    splits=("train", "valid", "test"),
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size phải > 0; num_workers phải >= 0.")
    samples = scan_dataset(data_dir)
    print_counts({split: samples[split] for split in splits})
    loaders = []
    for split in splits:
        dataset = FERDataset(samples[split], split, augment)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
            )
        )
    return tuple(loaders)


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra toàn bộ JPG gốc qua Dataset/DataLoader."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    seed_everything()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    loaders = create_loaders(args.data_dir, augment=False)
    for loader in loaders:
        count = 0
        for images, labels in loader:
            assert images.shape[1:] == (1, 48, 48)
            assert images.dtype == torch.float32 and labels.dtype == torch.int64
            assert (
                torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
            )
            assert labels.min() >= 0 and labels.max() < len(CLASSES)
            count += len(labels)
        print(f"OK {loader.dataset.split}: {count} ảnh đọc được, shape=(N, 1, 48, 48).")


if __name__ == "__main__":
    main()
