# Datathon 2026 — Team Beverage

> **Cuộc thi:** Datathon 2026 – The Gridbreakers – Vòng 1  
> **Đội:** Team Beverage  
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
| OS | Windows 10/11 (đã test), Linux/macOS cũng chạy được |
| RAM | >= 8 GB (file `fact_transactions.csv` khá nặng ~70 MB) |

---

## Cài đặt thư viện

Chỉ cần 3 thư viện chính, cài qua pip:

```bash
pip install pandas numpy matplotlib
```

**Chi tiết phiên bản đã test:**

| Thư viện | Version | Dùng ở đâu |
|---|---|---|
| `pandas` | >= 2.0 | `merge_table.py` — đọc CSV, merge, xuất bảng |
| `numpy` | >= 1.24 | `visualize_insights.py` — tính toán mảng cho biểu đồ |
| `matplotlib` | >= 3.7 | `visualize_insights.py` — vẽ biểu đồ insight |

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

Tổng cộng **14 file CSV** gốc. Nếu thiếu file nào thì script sẽ báo lỗi `FileNotFoundError`.

> **Quan trọng:** File `.gitignore` đã ignore `*.csv` nên dữ liệu không được push lên GitHub. Mỗi thành viên phải tự copy data vào thư mục `data/`.

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
├── trac_nghiem.ipynb            # Phần 1: Bài làm trắc nghiệm 10 câu
├── merge_table.py               # Phần 2a: Gộp 14 file CSV → 5 bảng phân tích
├── visualize_insights.py        # Phần 2b: Vẽ 3 biểu đồ insight cho báo cáo
├── model_v11.py                 # Phần 3: Model dự đoán (đang phát triển)
├── model_v10.py                 # Phần 3: Model dự đoán (bản cũ)
├── Đề thi Vòng 1.pdf            # Đề thi gốc
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

**File:** `model_v11.py` (phiên bản mới nhất), `model_v10.py` (bản cũ)

⚠️ **Phần này đang trong quá trình phát triển, chưa hoàn thiện.**

---

## Giải thích output

### Thư mục `merge_table/`

Dữ liệu được tổ chức theo mô hình **star schema** (dim + fact):

- **dim_** (dimension): Bảng mô tả — sản phẩm, khách hàng, khuyến mãi. Thay đổi ít, dùng để join.
- **fact_** (fact): Bảng ghi nhận sự kiện — giao dịch, vận hành. Dữ liệu lớn, chứa các metric cần phân tích.

### Thư mục `visualization/`

3 file ảnh PNG dùng để chèn trực tiếp vào báo cáo LaTeX (báo cáo nộp dạng PDF theo template NeurIPS 2025). Kích thước ảnh và font đã được tối ưu cho khổ giấy A4.