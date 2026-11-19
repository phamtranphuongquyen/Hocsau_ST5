from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

project_dir = Path(__file__).resolve().parent
csv_path = project_dir / "data" / "fer2013.csv"
output_path = project_dir / "anh_mau_fer2013.png"

if not csv_path.is_file():
    raise SystemExit(f"Không tìm thấy file CSV tại: {csv_path}")

data = pd.read_csv(csv_path)
data.columns = data.columns.str.strip()

required_columns = {"emotion", "pixels", "Usage"}
missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise SystemExit(f"File CSV thiếu cột: {', '.join(sorted(missing_columns))}")

if data.empty:
    raise SystemExit("File CSV không có dữ liệu.")

print("Các cột:", data.columns.tolist())
print("Tổng số ảnh:", len(data))
print("\nSố ảnh theo từng tập:")
print(data["Usage"].str.strip().value_counts().to_string())

first_row = data.iloc[0]
pixels = np.fromstring(first_row["pixels"], sep=" ", dtype=np.uint8)

if pixels.size != 48 * 48:
    raise SystemExit(f"Ảnh đầu tiên có {pixels.size} pixel, cần đúng 2304 pixel.")

image = pixels.reshape(48, 48)

print("\nSố pixel của ảnh đầu tiên:", pixels.size)
print("Kích thước ảnh đầu tiên:", image.shape)
print("Nhãn của ảnh đầu tiên:", int(first_row["emotion"]))

Image.fromarray(image).save(output_path)
print("\nĐã lưu ảnh mẫu tại:", output_path)