from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CSV_PATH = Path(__file__).resolve().parent / "data" / "fer2013.csv"
SPLITS = ("Training", "PublicTest", "PrivateTest")
CLASSES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
])

eval_transform = transforms.ToTensor()


class FERDataset(Dataset):
    def __init__(self, rows, transform):
        self.pixels = rows["pixels"].tolist()
        self.labels = rows["emotion"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pixels = np.fromstring(self.pixels[index], sep=" ", dtype=np.uint8)
        image = Image.fromarray(pixels.reshape(48, 48))
        return self.transform(image), int(self.labels[index])


def load_datasets(csv_path=CSV_PATH):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    usage = data["Usage"].str.strip()

    datasets = {}
    for split in SPLITS:
        rows = data.loc[usage.eq(split)]
        if rows.empty:
            raise ValueError(f"Không tìm thấy ảnh thuộc tập {split}")

        transform = train_transform if split == "Training" else eval_transform
        datasets[split] = FERDataset(rows, transform)

    return datasets


if __name__ == "__main__":
    datasets = load_datasets()

    for split, dataset in datasets.items():
        image, label = dataset[0]
        print(f"{split}: {len(dataset)} ảnh")
        print(f"  Kích thước một ảnh: {tuple(image.shape)}")
        print(f"  Nhãn ảnh đầu tiên: {label} ({CLASSES[label]})")

    train_loader = DataLoader(datasets["Training"], batch_size=64, shuffle=True)
    images, labels = next(iter(train_loader))

    print(f"\nKích thước một batch ảnh: {tuple(images.shape)}")
    print(f"Kích thước một batch nhãn: {tuple(labels.shape)}")