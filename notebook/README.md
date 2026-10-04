# Lab 03 - House Prices: Advanced Regression Techniques

## Cấu trúc thư mục khi nộp
```
lab03_house_price_hoten_masv/
├── data/            (train.csv, test.csv, sample_submission.csv)
├── notebooks/       (01, 02, 03, 04 .ipynb)
├── submissions/     (sinh ra khi chạy)
├── results/         (sinh ra khi chạy: bảng CV, biểu đồ)
└── report.docx / report.pdf
```
Các notebook dùng đường dẫn `../data/`, nên **chạy từ trong thư mục `notebooks/`**.

## Thứ tự chạy
1. `01_EDA_Preprocessing.ipynb`   -> tạo data/X_train_clean.csv, X_test_clean.csv, y_train_clean.csv
2. `04_feature_experiments.ipynb` -> thực nghiệm đặc trưng (A->D), tạo data/X_train_best.csv, X_test_best.csv
3. `02_sklearn_models.ipynb`      -> submissions/submission_sklearn.csv
4. `03_pytorch_mlp.ipynb`         -> submissions/submission_mlp.csv (+ submission_blend.csv)

Nếu 04 cho thấy bộ `best` tốt hơn `clean`, đổi `FEATURE_SET = "best"` ở đầu notebook 02 và 03 rồi chạy lại.
Nhớ **Run All** và lưu notebook (kèm output) trước khi nén.
