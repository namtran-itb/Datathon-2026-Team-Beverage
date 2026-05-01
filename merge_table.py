"""
Script gộp dữ liệu — Datathon 2026

Đọc 14 file CSV gốc, join và aggregate thành 5 bảng theo mô hình star schema:
  - dim_products.csv        : sản phẩm (giữ nguyên vì đã sạch)
  - dim_promotions.csv      : khuyến mãi (tách riêng, chỉ 30 dòng)
  - dim_customers.csv       : khách hàng gộp thêm region/district từ geography
  - fact_transactions.csv   : giao dịch = order_items nối payments, shipments, returns, reviews
  - fact_operations.csv     : vận hành = sales + web_traffic (daily) + inventory (monthly)
"""

import pandas as pd
import os

DATA_DIR = "data"                 # thư mục chứa 14 file CSV gốc
OUTPUT_DIR = "merge_table"        # thư mục xuất kết quả

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_merge_with_validation(left_df, right_df, on_cols, how='left', table_name=''):
    """Join 2 bảng, kiểm tra duplicate key và log kết quả để dễ debug."""
    print(f"    📊 Merging {table_name}...")
    
    # kiểm tra key có bị trùng không — trùng sẽ gây nhân bản dòng sau merge
    left_keys = left_df[on_cols].duplicated().sum()
    right_keys = right_df[on_cols].duplicated().sum()
    
    if left_keys > 0:
        print(f"      ⚠️ Bảng trái có {left_keys} key trùng")
    if right_keys > 0:
        print(f"      ⚠️ Bảng phải có {right_keys} key trùng")
    
    # đếm dòng trước merge để so sánh
    before_count = len(left_df)
    
    # Merge
    result = left_df.merge(right_df, on=on_cols, how=how, validate='many_to_one' if how == 'left' else None)
    
    # so sánh dòng trước/sau — nếu tăng bất thường là do key trùng
    after_count = len(result)
    match_rate = (after_count / before_count) * 100 if before_count > 0 else 0
    
    print(f"      ✓ Dòng: {before_count} → {after_count} ({match_rate:.1f}% match)")
    
    # cảnh báo nếu có dòng bên trái không khớp được với bảng phải
    if how == 'left':
        unmatched = before_count - result.dropna(subset=[c for c in right_df.columns if c not in on_cols]).shape[0]
        if unmatched > 0:
            print(f"      ⚠️ {unmatched} rows không match với {table_name}")
    
    return result

# ─────────────────────────────────────────────
# BẢNG 1: dim_products
# Giữ nguyên products.csv, tách promotions riêng vì quá nhỏ (30 dòng)
# ─────────────────────────────────────────────
print("📦 [1/4] Building dim_products ...")

products   = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
promotions = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"))

# loại bỏ product_id trùng nếu có
if products["product_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Phát hiện {products['product_id'].duplicated().sum()} product_id trùng")
    products = products.drop_duplicates(subset=["product_id"])

if promotions["promo_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Phát hiện {promotions['promo_id'].duplicated().sum()} promo_id trùng")
    promotions = promotions.drop_duplicates(subset=["promo_id"])

products.to_csv(os.path.join(OUTPUT_DIR, "dim_products.csv"), index=False)
promotions.to_csv(os.path.join(OUTPUT_DIR, "dim_promotions.csv"), index=False)

print(f"    dim_products  : {len(products):>7,} rows | {list(products.columns)}")
print(f"    dim_promotions: {len(promotions):>7,} rows | {list(promotions.columns)}")


# ─────────────────────────────────────────────
# BẢNG 2: dim_customers
# Join customers với geography qua cột zip → lấy thêm region, district
# ─────────────────────────────────────────────
print("\n👥 [2/4] Building dim_customers ...")

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
geography = pd.read_csv(os.path.join(DATA_DIR, "geography.csv"))

# chỉ lấy cột cần từ geography — city đã có sẵn trong customers rồi
geo_cols = geography[["zip", "region", "district"]].drop_duplicates(subset=["zip"])

# kiểm tra zip bị trùng — 1 zip chỉ nên thuộc 1 khu vực
zip_duplicates = geo_cols["zip"].duplicated().sum()
if zip_duplicates > 0:
    print(f"      ⚠️ Có {zip_duplicates} mã zip trùng trong geography")
    geo_cols = geo_cols.drop_duplicates(subset=["zip"], keep='first')

dim_customers = safe_merge_with_validation(
    customers, geo_cols, ["zip"], how='left', table_name='geography'
)

# đảm bảo 1 customer_id = 1 dòng duy nhất
if dim_customers["customer_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Có {dim_customers['customer_id'].duplicated().sum()} customer_id trùng sau merge")
    dim_customers = dim_customers.drop_duplicates(subset=["customer_id"])

dim_customers.to_csv(os.path.join(OUTPUT_DIR, "dim_customers.csv"), index=False)
print(f"    dim_customers : {len(dim_customers):>7,} rows | {list(dim_customers.columns)}")


# ─────────────────────────────────────────────
# BẢNG 3: fact_transactions
# Grain: 1 dòng = 1 sản phẩm trong 1 đơn hàng
# Gộp: order_items + orders + payments + shipments + returns + reviews
# ─────────────────────────────────────────────
print("\n🛒 [3/4] Building fact_transactions ...")

orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))

# xử lý trùng order_id — giữ dòng đầu tiên
order_duplicates = orders["order_id"].duplicated().sum()
if order_duplicates > 0:
    print(f"      ⚠️ Có {order_duplicates} order_id trùng trong orders")
    orders = orders.drop_duplicates(subset=["order_id"], keep='first')

print(f"      📋 Orders: {len(orders)} unique orders")

order_items = pd.read_csv(
    os.path.join(DATA_DIR, "order_items.csv"),
    dtype={"promo_id": str, "promo_id_2": str},
    low_memory=False
)
payments  = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
shipments = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"))
returns   = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
reviews   = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

# gom payments theo order — tránh nhân bản dòng khi 1 đơn có nhiều lần thanh toán
payments_agg = payments.groupby("order_id", as_index=False).agg(
    # nếu khách trả bằng 2 phương thức, nối lại thành chuỗi
    payment_method = ("payment_method", lambda x: ', '.join(str(v) for v in x.dropna().unique() if str(v) != 'nan')),
    total_payment  = ("payment_value", "sum")
)

# kiểm tra sau khi gom có bị trùng không
payment_duplicates = payments_agg["order_id"].duplicated().sum()
if payment_duplicates > 0:
    print(f"      ⚠️ Vẫn còn {payment_duplicates} order_id trùng sau gom payments")
    payments_agg = payments_agg.drop_duplicates(subset=["order_id"], keep='first')

# gom shipments — sort theo ngày mới nhất rồi lấy first
shipments_agg = (
    shipments
    .sort_values(["order_id", "ship_date"], ascending=[True, False])
    .groupby("order_id", as_index=False)
    .agg(
        ship_date      = ("ship_date", "first"),   # lấy lần ship gần nhất
        shipping_fee   = ("shipping_fee", "sum")    # cộng dồn phí ship
    )
)

shipment_duplicates = shipments_agg["order_id"].duplicated().sum()
if shipment_duplicates > 0:
    print(f"      ⚠️ Vẫn còn {shipment_duplicates} order_id trùng sau gom shipments")
    shipments_agg = shipments_agg.drop_duplicates(subset=["order_id"], keep='first')

# gom returns theo (order_id, product_id) — 1 sản phẩm có thể trả nhiều lần
returns_agg = (
    returns
    .groupby(["order_id", "product_id"], as_index=False)
    .agg(
        return_date     = ("return_date",     "first"),
        return_reason   = ("return_reason",   "first"),
        return_quantity = ("return_quantity",  "sum"),
        refund_amount   = ("refund_amount",    "sum"),
        is_returned     = ("return_id",        "count"),
    )
)
returns_agg["is_returned"] = (returns_agg["is_returned"] > 0).astype(int)

# gom reviews theo (order_id, product_id)
reviews_agg = (
    reviews
    .groupby(["order_id", "product_id"], as_index=False)
    .agg(
        review_date  = ("review_date",  "first"),
        rating       = ("rating",       "mean"),
        review_title = ("review_title", lambda x: ', '.join(str(v) for v in x.dropna().unique() if str(v) != 'nan')),
    )
)

# kiểm tra grain sau gom
review_duplicates = reviews_agg.duplicated(subset=["order_id", "product_id"]).sum()
if review_duplicates > 0:
    print(f"      ⚠️ Còn {review_duplicates} cặp (order_id, product_id) trùng trong reviews")
    reviews_agg = reviews_agg.drop_duplicates(subset=["order_id", "product_id"], keep='first')

# bắt đầu ghép — lấy order_items làm gốc, join lần lượt từng bảng
fact_transactions = order_items.copy()

# join thông tin đơn hàng (customer_id, order_date)
fact_transactions = safe_merge_with_validation(
    fact_transactions, orders[['order_id', 'customer_id', 'order_date']], 
    ["order_id"], how='left', table_name='orders'
)

fact_transactions = safe_merge_with_validation(
    fact_transactions, payments_agg, 
    ["order_id"], how='left', table_name='payments'
)

fact_transactions = safe_merge_with_validation(
    fact_transactions, shipments_agg, 
    ["order_id"], how='left', table_name='shipments'
)

fact_transactions = safe_merge_with_validation(
    fact_transactions, returns_agg, 
    ["order_id", "product_id"], how='left', table_name='returns'
)

fact_transactions = safe_merge_with_validation(
    fact_transactions, reviews_agg, 
    ["order_id", "product_id"], how='left', table_name='reviews'
)

fact_transactions["is_returned"] = fact_transactions["is_returned"].fillna(0).astype(int)

fact_transactions.to_csv(os.path.join(OUTPUT_DIR, "fact_transactions.csv"), index=False)
print(f"    fact_transactions: {len(fact_transactions):>7,} rows | {list(fact_transactions.columns)}")


# ─────────────────────────────────────────────
# BẢNG 4: fact_operations
# Gộp sales (daily) + web_traffic (daily) + inventory (monthly)
# Grain: 1 dòng = 1 ngày
# ─────────────────────────────────────────────
print("\n📈 [4/4] Building fact_operations ...")

sales       = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), parse_dates=["Date"])
web_traffic = pd.read_csv(os.path.join(DATA_DIR, "web_traffic.csv"), parse_dates=["date"])
inventory   = pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"))

# chuẩn hóa cột Date — bỏ phần giờ phút để merge chính xác
sales["Date"] = sales["Date"].dt.normalize()
web_traffic = web_traffic.rename(columns={"date": "Date"})
web_traffic["Date"] = web_traffic["Date"].dt.normalize()

# gom web_traffic theo ngày — phòng trường hợp 1 ngày có nhiều dòng (theo device/channel)
metrics_cols = [col for col in web_traffic.select_dtypes(include='number').columns if not col.endswith('_id')]
web_traffic_agg = web_traffic.groupby("Date", as_index=False)[metrics_cols].sum()

print(f"      📊 Web traffic: {len(web_traffic)} → {len(web_traffic_agg)} days after aggregation")

# gom inventory theo tháng — grain gốc là (product_id, year, month), 
# nhưng mình chỉ cần tổng hợp toàn bộ sản phẩm theo tháng
inv_monthly = (
    inventory
    .groupby(["year", "month"], as_index=False)
    .agg(
        total_stock_on_hand    = ("stock_on_hand",     "sum"),
        total_units_received   = ("units_received",    "sum"),
        total_units_sold       = ("units_sold",        "sum"),
        avg_fill_rate          = ("fill_rate",         "mean"),
        avg_days_of_supply     = ("days_of_supply",    "mean"),
        avg_sell_through_rate  = ("sell_through_rate", "mean"),
        stockout_product_count = ("stockout_flag",     "sum"),
        overstock_product_count= ("overstock_flag",    "sum"),
        reorder_product_count  = ("reorder_flag",      "sum"),
    )
)

# thêm cột year/month vào sales để join được với inventory
sales["year"]  = sales["Date"].dt.year
sales["month"] = sales["Date"].dt.month

# ghép sales + web_traffic theo ngày
fact_ops = sales.merge(web_traffic_agg, on="Date", how="left")

# ghép thêm inventory theo tháng
fact_ops = fact_ops.merge(inv_monthly, on=["year", "month"], how="left")

# dọn dẹp cột phụ
fact_ops = fact_ops.drop(columns=["year", "month"])

fact_ops.to_csv(os.path.join(OUTPUT_DIR, "fact_operations.csv"), index=False)
print(f"    fact_operations: {len(fact_ops):>7,} rows | {list(fact_ops.columns)}")


# ─────────────────────────────────────────────
# Tổng kết
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("✅ Xong! Các file đã xuất:")
output_files = [
    "dim_products.csv", "dim_promotions.csv",
    "dim_customers.csv", "fact_transactions.csv", "fact_operations.csv"
]
total_original = 14
for f in output_files:
    path = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    rows = sum(1 for _ in open(path)) - 1
    print(f"  {f:<30} {rows:>8,} rows  {size_mb:>6.1f} MB")

print(f"\n  Từ {total_original} file → {len(output_files)} bảng (gộp promotions vì nhỏ)")
print("="*60)