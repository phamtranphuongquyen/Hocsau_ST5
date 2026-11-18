# PyTorch Examples

Các notebook trong thư mục này đã được chuyển từ Keras sang PyTorch.

## 1. Cài đặt môi trường

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\DO_AN\BTLON03
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio numpy matplotlib scipy scikit-learn scikit-image seaborn jupyter notebook
```

Nếu PowerShell chặn việc activate môi trường:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Kiểm tra PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. Chạy notebook

Khởi động Jupyter Notebook:

```powershell
jupyter notebook
```

Sau đó mở notebook cần chạy và chọn **Kernel > Restart Kernel and Run All**.

Các ví dụ chính:

- `1. MLP\MNIST_MLP.ipynb`: MLP nhận dạng chữ số MNIST.
- `1. MLP\Func_apxm_MLP.ipynb`: xấp xỉ hàm Gaussian bằng MLP.
- `2. CNN + MLP\1_MNIST_CNN.ipynb`: CNN nhận dạng MNIST và trích xuất feature.
- `3. CNN + LSTM\MNIST_CNN_LSTM.ipynb`: CNN kết hợp LSTM cho MNIST.
- `5. Others\1_segnet.ipynb`: phân đoạn ảnh bằng SegNet.
- `5. Others\2_obj_detection.ipynb`: nhận diện xe bằng heatmap.

Có thể chạy trực tiếp một notebook bằng lệnh:

```powershell
jupyter nbconvert --to notebook --execute ".\1. MLP\MNIST_MLP.ipynb" --output "MNIST_MLP.executed.ipynb"
```

## 3. Dữ liệu đầu vào

- Ví dụ MNIST sẽ tự tải dữ liệu qua `torchvision` ở lần chạy đầu tiên.
- Các notebook giới tính, cảm xúc, SegNet và object detection cần dữ liệu tương ứng trong thư mục `data` hoặc các file `.npy`, `.csv` được notebook tham chiếu.
- Nếu chỉ muốn kiểm tra phần mô hình, hãy chạy các ví dụ MNIST trước.

## 4. Chạy trên Google Colab

Mở `colab_getting_started.ipynb`, chọn **Runtime > Change runtime type > T4 GPU**, rồi chạy cell cài đặt:

```python
!pip install torch torchvision
```

Các notebook khác có thể được tải lên Colab và chạy tương tự. Nếu notebook dùng GPU, PyTorch sẽ tự chọn CUDA khi `torch.cuda.is_available()` trả về `True`.

## 5. Lưu ý

PyTorch sử dụng thứ tự chiều ảnh `(N, C, H, W)`, khác với Keras thường dùng `(N, H, W, C)`. Các notebook đã chuyển đổi phần này trong bước chuẩn bị dữ liệu.
