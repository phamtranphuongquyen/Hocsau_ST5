# Nhận diện cảm xúc khuôn mặt bằng CNN — PyTorch / FER2013

Đồ án phân loại ảnh khuôn mặt đã cắt thành 7 cảm xúc bằng một CNN PyTorch, có giao diện Streamlit. Chỉ sử dụng JPG đã tồn tại trong `BTLON02/DATA/`. Không tải dataset, sinh ảnh/nhãn, xuất CSV, sao chép, đổi tên hoặc sửa ảnh gốc.

## Dữ liệu đã kiểm tra

FER2013 là bài toán phân loại biểu cảm khuôn mặt; các JPG trong workspace có kích thước 48×48, grayscale. Số lượng dưới đây được đếm trực tiếp, không lấy theo số lượng lý thuyết:

| Lớp (nhãn) | Training | PublicTest (validation) | PrivateTest (testing) |
|---|---:|---:|---:|
| angry (0) | 3.995 | 467 | 490 |
| disgust (1) | 436 | 56 | 55 |
| fear (2) | 4.097 | 496 | 528 |
| happy (3) | 7.215 | 895 | 879 |
| neutral (4) | 4.965 | 607 | 626 |
| sad (5) | 4.830 | 653 | 594 |
| surprise (6) | 3.171 | 415 | 416 |
| **Tổng** | **28.709** | **3.589** | **3.588** |

- Training: `BTLON02/DATA/train/<lớp>/Training_*.jpg`.
- Validation: `BTLON02/DATA/test/<lớp>/PublicTest_*.jpg`.
- Testing: `BTLON02/DATA/test/<lớp>/PrivateTest_*.jpg`.
- Có thêm `DATA/test/angry/` ngoài cùng với 2 ảnh. Thư mục này được giữ nguyên, **không đưa vào pipeline**.
- Không chia tập ngẫu nhiên. PrivateTest không tham gia cập nhật trọng số, chọn checkpoint hoặc chỉnh siêu tham số.
- Tất cả 35.886 ảnh thuộc dataset chính đọc được; không có lớp thiếu hoặc tên sai tiền tố tại thời điểm kiểm tra. Không giả định dataset trên máy khác có cùng số lượng: chương trình kiểm tra lại khi chạy.

## Cấu trúc và vai trò

```text
PROJECT/                        # thư mục chứa README này
├── BTLON02/DATA/{train,test}/   # JPG gốc, giữ nguyên
├── DATA/                       # dữ liệu phụ có sẵn, giữ nguyên và không dùng
├── BTLON02.ipynb                # gốc, hiện 0 byte
├── demo_code.html               # bản xuất mã Keras gốc, giữ nguyên
├── structure.txt                # giữ nguyên
├── data_loader.py               # nhãn, tiền xử lý, Dataset, DataLoader, kiểm tra dữ liệu
├── model.py                     # CNN và lưu/tải checkpoint chung
├── train.py                     # Training/PublicTest, EarlyStopping, biểu đồ
├── evaluate.py                  # đánh giá PrivateTest và lưu chỉ số
├── app.py                       # upload khuôn mặt và dự đoán
├── models/best_model.pth        # chỉ sinh sau epoch huấn luyện thành công
├── results/training_history.png # chỉ sinh sau epoch thành công
├── results/confusion_matrix.png # chỉ sinh sau đánh giá thành công
├── results/metrics.json         # chỉ sinh sau đánh giá thành công
├── requirements.txt
└── .gitignore
```

Đường dẫn mặc định tính theo vị trí file Python, không hardcode ổ đĩa. Có thể chỉ định `--data-dir` trỏ tới thư mục có `train/` và `test/`. Trường hợp ảnh hỏng, thiếu lớp, sai tên hoặc trùng đường dẫn, chương trình báo lỗi thay vì âm thầm bỏ ảnh. Việc kiểm tra đường dẫn không tương đương kiểm tra ảnh giống nhau về nội dung.

## Môi trường và cài đặt

Môi trường ban đầu: Python 3.13.2, Windows, NVIDIA Quadro T1000 (4 GB). Đã tạo `.venv` riêng, không đổi phiên bản Python và không sửa thư viện môi trường toàn cục.

Mở PowerShell tại thư mục dự án:

```powershell
# Chỉ tạo nếu chưa có .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell không cho chạy `Activate.ps1`, dùng trực tiếp `.\.venv\Scripts\python.exe` thay cho `python`, và `python -m pip` / `python -m streamlit` bằng interpreter đó, không cần đổi execution policy.

PyTorch tự chọn CUDA khi khả dụng, ngược lại dùng CPU. Bộ thư viện CUDA dùng cho GPU của workspace:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Chạy demo ngay:
```powershell
cd D:\DO_AN\BTLON02
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Huấn luyện và đánh giá lại:
```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
```
      
20:34 15 thg 11
Nguồn lệnh cài CUDA: [bản phát hành chính thức PyTorch](https://pytorch.org/get-started/previous-versions/). Máy không có NVIDIA có thể dùng bộ CPU tương ứng; không bắt buộc cài CUDA để demo.

## CNN và luồng dữ liệu

PyTorch cung cấp Tensor, `Dataset`/`DataLoader`, lớp `nn.Module`, tự động tính gradient và optimizer. Kiến trúc duy nhất:

```text
(N, 1, 48, 48)
→ Conv2d(1,32,3,padding=1) → BatchNorm → ReLU → MaxPool(2)
→ Conv2d(32,64,3,padding=1) → BatchNorm → ReLU → MaxPool(2)
→ Conv2d(64,128,3,padding=1) → BatchNorm → ReLU → MaxPool(2)
→ Flatten (4608) → Linear(4608,128) → ReLU → Dropout(0.3)
→ Linear(128,7) → 7 logits
```

JPG gốc → nhãn từ tên thư mục → chọn tập theo tiền tố → đọc bằng Pillow trong Dataset → grayscale → resize 48×48 → Tensor `float32 (1,48,48)` trong `[0,1]`. DataLoader ghép batch; nhãn là `int64 (N,)`, giá trị 0–6.

Training có lật ngang và xoay tối đa 10° trong bộ nhớ; không ghi ảnh biến đổi. Validation/testing/upload dùng chung `preprocess_image()` và hoàn toàn không augmentation. `--no-augmentation` tắt cả biến đổi training. Seed mặc định 42 cố định shuffle, khởi tạo trọng số và augmentation; vẫn có thể có sai khác số học giữa phần cứng/phiên bản thư viện.

`CrossEntropyLoss` nhận trực tiếp logits và nhãn số nguyên, không one-hot và không Softmax trong CNN. Chỉ demo dùng Softmax để hiển thị điểm.

## Chạy kiểm tra, huấn luyện và đánh giá

```powershell
python data_loader.py
python model.py
python model.py --self-test
python train.py
python evaluate.py
streamlit run app.py
```

`data_loader.py` duyệt toàn bộ ảnh thật trong ba tập qua DataLoader, kiểm tra shape, kiểu dữ liệu và khoảng pixel. `model.py` chạy forward trên batch Training thật; không tạo ảnh/Tensor ngẫu nhiên và không lưu model chưa huấn luyện.

`model.py --self-test` kiểm tra thêm backward, gradient hữu hạn, Adam thay đổi trọng số, lưu/tải checkpoint cho kết quả logits giống hệt trên CPU và CUDA (nếu có). Checkpoint kiểm thử được lưu trong thư mục tạm rồi dọn tự động; không thay `models/best_model.pth`.

Huấn luyện mặc định: batch 64, tối đa 50 epoch, Adam, learning rate 0.0001, patience 8. Mỗi epoch ghi loss/accuracy của Training và PublicTest. EarlyStopping dựa trên **PublicTest loss**; checkpoint tốt nhất chứa trọng số, thứ tự nhãn, quy tắc tiền xử lý, epoch, cấu hình và lịch sử đến epoch đó. Biểu đồ chứa mọi epoch đã chạy. Gradient được chặn norm 5; loss/gradient NaN/Inf làm dừng huấn luyện với lỗi rõ ràng.

```powershell
# Lượt kiểm tra ngắn, vẫn dùng toàn bộ Training và PublicTest
python train.py --epochs 1 --checkpoint models/smoke_model.pth --results-dir results/smoke

# Tắt augmentation nếu cần
python train.py --no-augmentation
```

Mỗi lệnh train bắt đầu một lượt mới, không tự resume. Chạy mặc định sẽ thay checkpoint mặc định khi tìm được epoch tốt hơn trong lượt mới. Sử dụng `--checkpoint` và `--results-dir` riêng nếu muốn giữ kết quả trước. `--num-workers 0` mặc định dễ chạy trên Windows; có thể thử 2 khi cần.

`evaluate.py` tải checkpoint bằng `weights_only=True`, kiểm tra nhãn và tiền xử lý, gọi `eval()` + `inference_mode()`, chỉ giải mã ảnh PrivateTest. Chỉ số gồm Accuracy, Precision macro, Recall macro, Macro F1, báo cáo từng lớp và confusion matrix 7×7 (hàng: nhãn thật, cột: dự đoán). Giá trị trong JSON ở thang 0–1. `zero_division=0` xử lý lớp không được dự đoán. Chỉ đánh giá cuối cùng sau khi đã chọn mô hình bằng PublicTest, không dùng kết quả test để chọn lại mô hình.

## Demo Streamlit

Mở URL local do Streamlit in ra, tải JPG/JPEG/PNG đã cắt một khuôn mặt, nhấn **Dự đoán cảm xúc**. Giao diện hiển thị ảnh, tên cảm xúc, điểm Softmax cao nhất và biểu đồ 7 lớp. Inference dùng CPU để demo nhẹ, cùng checkpoint và tiền xử lý như evaluate.

Chưa có checkpoint: hiển thị hướng dẫn huấn luyện, không đoán từ trọng số ngẫu nhiên. Chưa có metrics: thông báo chưa đánh giá. Metrics chứa SHA-256 checkpoint; nếu model thay đổi, ứng dụng không hiển thị chỉ số cũ như kết quả của model mới. Cache model cũng theo hash checkpoint.

## So với mã gốc và giới hạn

`BTLON02.ipynb` là file rỗng nên không có cell để phân tích. `demo_code.html` chứa mã Keras/Colab, các CNN SMALL/MEDIUM/MEDIUM1, quy trình CSV → PNG, generator và callback cũ. Bản refactor giữ tư tưởng chia ba tập, CNN, Adam, EarlyStopping; thay bằng đọc trực tiếp JPG, một CNN nhỏ có BatchNorm, Dropout, kiểm tra lỗi, checkpoint thống nhất, đánh giá riêng và giao diện demo. Không dùng hay chuyển đổi trọng số Keras, không khẳng định accuracy cao hơn khi chưa có thí nghiệm đối chứng.

- Dataset mất cân bằng (đặc biệt disgust); không oversampling, SMOTE hoặc tạo dữ liệu cân bằng.
- Ảnh 48×48 mất chi tiết; ánh sáng, góc mặt và dữ liệu ngoài FER2013 có thể làm giảm chất lượng.
- Không phát hiện/cắt khuôn mặt tự động, không webcam, không nhiều khuôn mặt.
- Điểm Softmax chưa hiệu chuẩn, không phải xác suất chính xác về cảm xúc thực của con người.
- Không có scheduler, resume optimizer, TensorBoard hoặc so sánh nhiều mô hình; giữ phạm vi đồ án đơn giản.
- Kết quả của HTML gốc không được sao chép làm kết quả của mô hình mới.

## Kiểm thử thực tế

Kết quả chạy thực tế sẽ được cập nhật tại đây sau khi hoàn tất kiểm thử và huấn luyện; không tạo checkpoint hay metrics để minh họa.
