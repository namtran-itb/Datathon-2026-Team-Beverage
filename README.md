# Datathon 2026 — Team Beverage

> **Cuộc thi:** Datathon 2026 – The Gridbreakers – Vòng 1  
> **Đội:** Beverage  
> **Trường:** Đại học Công nghệ – Đại học Quốc gia Hà Nội  
> **Thành viên:** Trần Hoài Nam (Leader) · Trần Minh Khuê · Lâm Việt Phúc · Nguyễn Thanh Thủy

---

## Mục lục

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt thư viện](#cài-đặt-thư-viện)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
5. [Hướng dẫn chạy](#hướng-dẫn-chạy)
   - [Phần 1 – Trắc nghiệm](#phần-1--trắc-nghiệm-10-câu)
   - [Phần 2 – EDA & Visualization](#phần-2--eda--visualization)
   - [Phần 3 – Model dự đoán](#phần-3--model-dự-đoán)
6. [Giải thích output](#giải-thích-output)

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản khuyên dùng |
|---|---|
| Python | >= 3.10 |
| pip | >= 22.0 |

---

## Cài đặt thư viện

Cài qua pip:

```bash
pip install pandas numpy matplotlib xgboost lightgbm statsmodels
```

**Chi tiết phiên bản đã test:**

| Thư viện | Version | Dùng ở đâu |
|---|---|---|
| `pandas` | >= 2.0 | `merge_table.py`, `model.py` — đọc CSV, merge, xử lý chuỗi thời gian |
| `numpy` | >= 1.24 | `visualize_insights.py`, `model.py` — tính toán mảng, hàm toán học |
| `matplotlib` | >= 3.7 | `visualize_insights.py` — vẽ biểu đồ insight |
| `xgboost` | >= 2.0 | `model.py` — mô hình XGBoost dự đoán Revenue/COGS |
| `lightgbm` | >= 4.0 | `model.py` — mô hình LightGBM (thành phần ensemble) |
| `statsmodels` | >= 0.14 | `model.py` — phân rã chuỗi thời gian (STL) |

> **Lưu ý:** File `trac_nghiem.ipynb` chạy trên Jupyter Notebook / Google Colab, không cần thêm thư viện gì ngoài pandas.

---

## Chuẩn bị dữ liệu

Tải bộ dữ liệu từ đề thi, giải nén toàn bộ file CSV vào thư mục `data/` nằm cùng cấp với các file `.py`:

```
Datathon-2026-Team-Beverage/
├── data/
│   ├── customers.csv
│   ├── geography.csv
│   ├── inventory.csv
│   ├── order_items.csv
│   ├── orders.csv
│   ├── payments.csv
│   ├── products.csv
│   ├── promotions.csv
│   ├── returns.csv
│   ├── reviews.csv
│   ├── sales.csv
│   ├── sample_submission.csv
│   ├── shipments.csv
│   └── web_traffic.csv
```

Tổng cộng **14 file CSV** gốc.

> **Quan trọng:** File `.gitignore` đã ignore `*.csv` nên dữ liệu không được push lên GitHub. Người dùng phải tự copy data vào thư mục `data/`.

---

## Cấu trúc thư mục

```
Datathon-2026-Team-Beverage/
│
├── data/                        # 14 file CSV gốc từ đề thi (không push lên git)
│
├── merge_table/                 # Output của merge_table.py (tự tạo khi chạy)
│   ├── dim_products.csv         #   Bảng dimension sản phẩm
│   ├── dim_promotions.csv       #   Bảng dimension khuyến mãi
│   ├── dim_customers.csv        #   Bảng dimension khách hàng + vùng miền
│   ├── fact_transactions.csv    #   Bảng fact giao dịch (nặng nhất ~70 MB)
│   └── fact_operations.csv      #   Bảng fact vận hành (sales + traffic + inventory)
│
├── visualization/               # Output của visualize_insights.py (tự tạo khi chạy)
│   ├── insight1.png             #   Biểu đồ: Giảm giá ăn mòn biên lợi nhuận
│   ├── insight2.png             #   Biểu đồ: Vốn đông kết trong tồn kho
│   └── insight3.png             #   Biểu đồ: Hệ thống không bổ sung tồn kho
│
├── outputs/                      # Output của model.py (tự tạo khi chạy)
│   └── submission.csv            #   File nộp bài dự đoán Kaggle
│
├── trac_nghiem.ipynb            # Phần 1: Bài làm trắc nghiệm 10 câu
├── merge_table.py               # Phần 2a: Gộp 14 file CSV → 5 bảng phân tích
├── visualize_insights.py        # Phần 2b: Vẽ 3 biểu đồ insight cho báo cáo
├── model.py                     # Phần 3: Model dự đoán doanh thu & giá vốn
├── .gitignore
└── README.md                    # File này
```

---

## Hướng dẫn chạy

### Phần 1 — Trắc nghiệm 10 câu

**File:** `trac_nghiem.ipynb`

Mở bằng Jupyter Notebook hoặc upload lên Google Colab:

```bash
jupyter notebook trac_nghiem.ipynb
```

File này chứa bài làm 10 câu trắc nghiệm, mỗi câu có giải thích lý do chọn đáp án. Chạy tuần tự các cell từ trên xuống dưới.

---

### Phần 2 — EDA & Visualization

Phần này gồm 2 bước, chạy **theo thứ tự**:

#### Bước 1: Gộp dữ liệu — `merge_table.py`

```bash
python merge_table.py
```

Script này đọc 14 file CSV từ thư mục `data/`, join/aggregate thành 5 bảng và lưu vào thư mục `merge_table/`:

| Bảng output | Mô tả | Gộp từ |
|---|---|---|
| `dim_products.csv` | Thông tin sản phẩm | `products.csv` |
| `dim_promotions.csv` | Thông tin khuyến mãi | `promotions.csv` |
| `dim_customers.csv` | Khách hàng + vùng miền | `customers.csv` + `geography.csv` |
| `fact_transactions.csv` | Giao dịch chi tiết | `order_items` + `orders` + `payments` + `shipments` + `returns` + `reviews` |
| `fact_operations.csv` | Vận hành theo ngày | `sales` + `web_traffic` + `inventory` |

> **Thời gian chạy:** ~30-60 giây tùy máy (file `orders.csv` ~45 MB nên hơi lâu).

#### Bước 2: Vẽ biểu đồ — `visualize_insights.py`

```bash
python visualize_insights.py
```

Script này vẽ 3 biểu đồ insight phục vụ báo cáo PDF, lưu vào thư mục `visualization/`:

- `insight1.png` — Giảm giá phá hủy biên lợi nhuận nhưng không tăng sell-through
- `insight2.png` — Vốn bị đóng băng trong tồn kho do phân bổ sai
- `insight3.png` — Hệ thống không bổ sung tồn kho khi hết hàng

> **Lưu ý:** File này **không cần** chạy `merge_table.py` trước vì dữ liệu insight đã được tổng hợp sẵn trong code (lấy từ kết quả EDA).

---

### Phần 3 — Model dự đoán

**File:** `model.py`

```bash
python model.py
```

Script này dự đoán doanh thu (Revenue) và giá vốn (COGS) cho giai đoạn 2023-01-01 đến 2024-07-01, xuất file `outputs/submission.csv` theo đúng mẫu nộp bài.

#### Phương pháp

1. **Phân rã chuỗi thời gian** — tách doanh thu thành 3 thành phần:
   - **Xu hướng (trend):** tốc độ tăng trưởng CAGR riêng cho từng nửa năm (H1: tháng 1-6, H2: tháng 7-12), vì doanh thu nửa cuối năm thường tăng nhanh hơn.
   - **Mùa vụ (seasonal):** kết hợp 60% mùa vụ theo Tháng × Ngày trong tuần (ổn định, 84 cụm) + 40% mùa vụ theo ngày trong năm (làm mượt bằng rolling 7 ngày, giữ chi tiết từng ngày).
   - **Phần dư (residual):** log tỷ số giữa giá trị thực tế và giá trị kỳ vọng — đây là mục tiêu huấn luyện.

2. **Ensemble 3 mô hình** dự đoán phần dư Revenue:
   - XGBoost #1: `max_depth=3`, `learning_rate=0.025`, trọng số 0.40
   - LightGBM: `max_depth=4`, `num_leaves=31`, trọng số 0.35
   - XGBoost #2: `max_depth=4`, `learning_rate=0.02`, trọng số 0.25

3. **Dự đoán COGS** — kết hợp 2 nguồn:
   - Mô hình độc lập dự đoán phần dư COGS (tương tự Revenue nhưng giảm độ phức tạp).
   - Mô hình phụ dự đoán tỷ lệ COGS/Revenue, rồi nhân với Revenue dự đoán được.
   - Trọng số kết hợp: 70% độc lập + 30% từ tỷ lệ (`ratio_blend=0.3`).

4. **Ràng buộc biên lợi nhuận** — học phân vị 2% và 98% của tỷ lệ COGS/Revenue theo tháng từ tập huấn luyện, rồi áp lên tập test. Cho phép COGS > Revenue (khoảng 10% ngày trong thực tế).

#### Đặc trưng đầu vào

| Nhóm đặc trưng | Các cột | Giải thích |
|---|---|---|
| Thời gian | `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`, `is_weekend`, `is_payday` | Lịch, cuối tuần, ngày trả lương |
| Chu kỳ sin/cos | `sin_1`, `cos_1`, `sin_2`, `cos_2`, `sin_dow`, `cos_dow` | Mã hóa tuần hoàn năm và tuần |
| Hiệu ứng Tết | `is_tet`, `pre_tet` | Trong tuần Tết và 30 ngày trước Tết |
| Khuyến mãi | `promo`, `discount` | Số đợt sale và tổng giảm giá trong ngày |
| Ngày lễ | `is_holiday` | Các ngày lễ lớn Việt Nam |

#### Tham số tăng trưởng

Tham số CAGR được đặt trong hàm `main()`:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `cagr_h1_23` | 24% | Tốc độ tăng nửa đầu năm 2023 |
| `cagr_h2_23` | 32% | Tốc độ tăng nửa cuối năm 2023 |
| `cagr_h1_24` | 2% | Tốc độ tăng nửa đầu năm 2024 |
| `cagr_h2_24` | 5% | Tốc độ tăng nửa cuối năm 2024 |

> **Lưu ý:** Tham số CAGR được điều chỉnh thủ công dựa trên phân tích xu hướng dữ liệu lịch sử. Nếu dữ liệu thay đổi, cần tinh chỉnh lại các giá trị này.

> **Thời gian chạy:** ~1-3 phút tùy máy (huấn luyện 3 mô hình × 2 target + 1 mô hình tỷ lệ).

---

## Giải thích output

### Thư mục `merge_table/`

Dữ liệu được tổ chức theo mô hình **star schema** (dim + fact):

- **dim_** (dimension): Bảng mô tả — sản phẩm, khách hàng, khuyến mãi. Thay đổi ít, dùng để join.
- **fact_** (fact): Bảng ghi nhận sự kiện — giao dịch, vận hành. Dữ liệu lớn, chứa các metric cần phân tích.

### Thư mục `visualization/`

3 file ảnh PNG dùng để chèn trực tiếp vào báo cáo LaTeX (báo cáo nộp dạng PDF theo template NeurIPS 2025). Kích thước ảnh và font đã được tối ưu cho khổ giấy A4.

### Thư mục `outputs/`

Chứa file `submission.csv` — kết quả dự đoán nộp bài:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `Date` | `YYYY-MM-DD` | Ngày dự đoán (2023-01-01 đến 2024-07-01) |
| `Revenue` | số thực | Doanh thu dự đoán (đơn vị VND) |
| `COGS` | số thực | Giá vốn hàng bán dự đoán (đơn vị VND) |

File này khớp đúng mẫu `sample_submission.csv` từ đề thi, sẵn sàng nộp.