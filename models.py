from pathlib import Path

import torch
from torch import nn

MODEL_CHANNELS = {
    2: (32, 64),
    3: (32, 64, 128),
    4: (32, 64, 128, 256),
}


class EmotionCNN(nn.Module):
    def __init__(self, depth):
        super().__init__()
        channels = MODEL_CHANNELS[depth]
        layers = []
        in_channels = 1

        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(channels[-1], 7),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def save_architecture_figure(rows, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    ax.set_title("So sánh kiến trúc CNN trên FER2013", fontsize=16, pad=18)

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Mô hình", "Số khối", "Số kênh mỗi khối",
            "Đặc trưng sau khối cuối", "Số tham số",
        ],
        colWidths=[0.14, 0.11, 0.25, 0.29, 0.21],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDEAF7")
            cell.set_text_props(weight="bold")

    fig.text(
        0.5, 0.12,
        "Đầu vào: ảnh xám 1 × 48 × 48. Đầu ra: 7 điểm cho 7 lớp cảm xúc.",
        ha="center",
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    sample_images = torch.zeros(2, 1, 48, 48)
    rows = []

    for depth, channels in MODEL_CHANNELS.items():
        model = EmotionCNN(depth)
        model.eval()

        with torch.inference_mode():
            features = model.features(sample_images)
            logits = model(sample_images)

        if logits.shape != (2, 7):
            raise RuntimeError(f"CNN-{depth} có đầu ra sai: {tuple(logits.shape)}")

        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        feature_shape = " × ".join(str(size) for size in features.shape[1:])

        print(f"CNN-{depth} | Đặc trưng: {feature_shape} | Đầu ra: {tuple(logits.shape)}")
        print(f"  Số tham số học được: {parameters:,}")

        rows.append([
            f"CNN-{depth}",
            depth,
            " → ".join(map(str, channels)),
            feature_shape,
            f"{parameters:,}".replace(",", "."),
        ])

    output_path = Path(__file__).resolve().parent / "results" / "cnn_architectures.png"
    save_architecture_figure(rows, output_path)
    print(f"\nĐã lưu hình tại: {output_path}")


if __name__ == "__main__":
    main()