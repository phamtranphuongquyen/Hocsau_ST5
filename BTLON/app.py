"""Demo khuôn mặt đã cắt; cùng checkpoint và preprocessing với evaluate.py."""

import json
import math

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from data_loader import CLASSES, PROJECT_ROOT, preprocess_image
from model import CHECKPOINT_PATH, checkpoint_sha256, load_checkpoint

EMOTIONS_VI = [
    "Tức giận",
    "Ghê tởm",
    "Sợ hãi",
    "Vui vẻ",
    "Trung tính",
    "Buồn",
    "Ngạc nhiên",
]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"


@st.cache_resource
def get_model(checkpoint_path, checkpoint_digest):
    # Digest là khóa cache: tự tải lại khi checkpoint đổi sau huấn luyện.
    model, _ = load_checkpoint(checkpoint_path, "cpu")
    return model


def predict(image, model):
    tensor = preprocess_image(image).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if not torch.isfinite(logits).all():
            raise ValueError("Mô hình trả về NaN/Inf.")
        return torch.softmax(logits, dim=1)[0].cpu()


def show_metrics(digest):
    st.subheader("Kết quả mô hình")
    if not METRICS_PATH.is_file():
        st.info("Chưa có kết quả đánh giá trên PrivateTest.")
        return
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if (
            metrics["classes"] != CLASSES
            or metrics["split"] != "PrivateTest"
            or metrics["checkpoint_sha256"] != digest
        ):
            st.info(
                "Kết quả đánh giá chưa khớp mô hình hiện tại. Cần đánh giá lại mô hình."
            )
            return
        accuracy, f1 = float(metrics["accuracy"]), float(metrics["f1_macro"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (accuracy, f1)):
            raise ValueError("Chỉ số không hợp lệ.")
        left, right = st.columns(2)
        left.metric("Accuracy · PrivateTest", f"{accuracy:.2%}")
        right.metric("Macro F1 · PrivateTest", f"{f1:.4f}")
        st.caption(
            f"Đánh giá trên {metrics['num_samples']:,} ảnh; checkpoint epoch {metrics['checkpoint_epoch']}."
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f"Không đọc được kết quả đánh giá: {exc}")


def main():
    st.set_page_config(
        page_title="FER2013 · Nhận diện cảm xúc", page_icon="🙂", layout="centered"
    )
    st.title("Nhận diện cảm xúc khuôn mặt bằng CNN")
    st.caption("Python · PyTorch · Streamlit | Dataset FER2013 | 7 lớp cảm xúc")
    st.write("Tải một ảnh đã cắt vùng khuôn mặt để xem cảm xúc mà mô hình dự đoán.")
    st.info("Hỗ trợ JPG, JPEG, PNG. Mỗi ảnh nên chứa một khuôn mặt nhìn rõ.")
    digest = checkpoint_sha256(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    uploaded = st.file_uploader("Chọn ảnh khuôn mặt", type=["jpg", "jpeg", "png"])
    if digest is None:
        st.warning(
            "Chưa có mô hình đã huấn luyện. Hãy chạy python train.py trước khi dự đoán."
        )
    if uploaded is not None:
        try:
            with Image.open(uploaded) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
            st.image(image, caption="Ảnh đã tải", width=300)
            if st.button("Dự đoán cảm xúc", type="primary", disabled=digest is None):
                with st.spinner("Đang phân tích khuôn mặt…"):
                    scores = predict(image, get_model(str(CHECKPOINT_PATH), digest))
                index = int(scores.argmax())
                st.success(f"Cảm xúc dự đoán: {EMOTIONS_VI[index]} ({CLASSES[index]})")
                st.metric("Điểm Softmax cao nhất", f"{scores[index].item():.2%}")
                st.caption(
                    "Điểm Softmax là điểm của mô hình, không phải độ chắc chắn đã được hiệu chuẩn."
                )
                figure, axis = plt.subplots(figsize=(8, 3.5))
                axis.barh(CLASSES, scores.numpy(), color="#2563eb")
                axis.set(xlim=(0, 1), xlabel="Softmax")
                axis.invert_yaxis()
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as exc:
            st.error(f"Không thể dự đoán: {exc}")
    st.divider()
    show_metrics(digest)
    st.caption(
        "Giới hạn: chỉ nhận diện biểu cảm trong ảnh; không xác định trạng thái tâm lý của người dùng."
    )


if __name__ == "__main__":
    main()
