"""
Datathon 2026 — Data Consolidation Script
Gộp 14 file CSV thành 4 bảng phân tích

Output:
    dim_products.csv        — products (giữ nguyên, nhỏ + rõ ràng)
    dim_customers.csv       — customers + geography (join qua zip)
    fact_transactions.csv   — order_items + payments + shipments + returns + reviews
    fact_operations.csv     — sales + web_traffic + inventory (aggregated monthly)
"""

import pandas as pd
import os

DATA_DIR = "data"
OUTPUT_DIR = "merge_table"        # Thư mục lưu output

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_merge_with_validation(left_df, right_df, on_cols, how='left', table_name=''):
    """Merge với validation và logging"""
    print(f"    📊 Merging {table_name}...")
    
    # Check duplicates in key columns
    left_keys = left_df[on_cols].duplicated().sum()
    right_keys = right_df[on_cols].duplicated().sum()
    
    if left_keys > 0:
        print(f"      ⚠️ Left có {left_keys} duplicate keys")
    if right_keys > 0:
        print(f"      ⚠️ Right có {right_keys} duplicate keys")
    
    # Count before merge
    before_count = len(left_df)
    
    # Merge
    result = left_df.merge(right_df, on=on_cols, how=how, validate='many_to_one' if how == 'left' else None)
    
    # Count after merge and report
    after_count = len(result)
    match_rate = (after_count / before_count) * 100 if before_count > 0 else 0
    
    print(f"      ✓ Rows: {before_count} → {after_count} ({match_rate:.1f}% match)")
    
    # Check for unmatched rows in left join
    if how == 'left':
        unmatched = before_count - result.dropna(subset=[c for c in right_df.columns if c not in on_cols]).shape[0]
        if unmatched > 0:
            print(f"      ⚠️ {unmatched} rows không match với {table_name}")
    
    return result

# ─────────────────────────────────────────────
# TABLE 1: dim_products  (giữ nguyên products.csv)
# promotions gộp vào đây vì được reference từ transactions
# ─────────────────────────────────────────────
print("📦 [1/4] Building dim_products ...")

products   = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
promotions = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"))

# Validate unique keys
if products["product_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Found {products['product_id'].duplicated().sum()} duplicate product_ids")
    products = products.drop_duplicates(subset=["product_id"])

if promotions["promo_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Found {promotions['promo_id'].duplicated().sum()} duplicate promo_ids")
    promotions = promotions.drop_duplicates(subset=["promo_id"])

products.to_csv(os.path.join(OUTPUT_DIR, "dim_products.csv"), index=False)
promotions.to_csv(os.path.join(OUTPUT_DIR, "dim_promotions.csv"), index=False)

print(f"    dim_products  : {len(products):>7,} rows | {list(products.columns)}")
print(f"    dim_promotions: {len(promotions):>7,} rows | {list(promotions.columns)}")


# ─────────────────────────────────────────────
# TABLE 2: dim_customers  (customers + geography)
# Join qua cột zip để flatten region/district vào luôn
# ─────────────────────────────────────────────
print("\n👥 [2/4] Building dim_customers ...")

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
geography = pd.read_csv(os.path.join(DATA_DIR, "geography.csv"))

# Chỉ lấy các cột cần thiết từ geography (city đã có trong customers)
geo_cols = geography[["zip", "region", "district"]].drop_duplicates(subset=["zip"])

# Validate zip uniqueness
zip_duplicates = geo_cols["zip"].duplicated().sum()
if zip_duplicates > 0:
    print(f"      ⚠️ Found {zip_duplicates} duplicate zip codes in geography")
    geo_cols = geo_cols.drop_duplicates(subset=["zip"], keep='first')

dim_customers = safe_merge_with_validation(
    customers, geo_cols, ["zip"], how='left', table_name='geography'
)

# Validate customer uniqueness
if dim_customers["customer_id"].duplicated().sum() > 0:
    print(f"      ⚠️ Found {dim_customers['customer_id'].duplicated().sum()} duplicate customer_ids")
    dim_customers = dim_customers.drop_duplicates(subset=["customer_id"])

dim_customers.to_csv(os.path.join(OUTPUT_DIR, "dim_customers.csv"), index=False)
print(f"    dim_customers : {len(dim_customers):>7,} rows | {list(dim_customers.columns)}")


# ─────────────────────────────────────────────
# TABLE 3: fact_transactions
# order_items + payments + shipments + returns (agg) + reviews (agg)
# Grain: 1 dòng = 1 sản phẩm trong 1 đơn hàng
# ─────────────────────────────────────────────
print("\n🛒 [3/4] Building fact_transactions ...")

orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))

# Validate and fix order_id uniqueness
order_duplicates = orders["order_id"].duplicated().sum()
if order_duplicates > 0:
    print(f"      ⚠️ Found {order_duplicates} duplicate order_ids in orders")
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

# Gom nhóm payments tránh đẻ dòng
payments_agg = payments.groupby("order_id", as_index=False).agg(
    # Nếu khách quẹt 2 thẻ, nối tên lại thay vì lấy first
    payment_method = ("payment_method", lambda x: ', '.join(str(v) for v in x.dropna().unique() if str(v) != 'nan')),
    total_payment  = ("payment_value", "sum")
)

# Validate uniqueness after aggregation
payment_duplicates = payments_agg["order_id"].duplicated().sum()
if payment_duplicates > 0:
    print(f"      ⚠️ Still have {payment_duplicates} duplicate order_ids in payments_agg")
    payments_agg = payments_agg.drop_duplicates(subset=["order_id"], keep='first')

# Gom nhóm shipments tránh đẻ dòng
shipments_agg = (
    shipments
    .sort_values(["order_id", "ship_date"], ascending=[True, False]) # Sort để lấy ngày mới nhất
    .groupby("order_id", as_index=False)
    .agg(
        ship_date      = ("ship_date", "first"),
        shipping_fee   = ("shipping_fee", "sum")
    )
)

# Validate uniqueness after aggregation
shipment_duplicates = shipments_agg["order_id"].duplicated().sum()
if shipment_duplicates > 0:
    print(f"      ⚠️ Still have {shipment_duplicates} duplicate order_ids in shipments_agg")
    shipments_agg = shipments_agg.drop_duplicates(subset=["order_id"], keep='first')

# Aggregate returns → grain (order_id, product_id)
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

# Aggregate reviews → grain (order_id, product_id)
reviews_agg = (
    reviews
    .groupby(["order_id", "product_id"], as_index=False)
    .agg(
        review_date  = ("review_date",  "first"),
        rating       = ("rating",       "mean"),
        review_title = ("review_title", lambda x: ', '.join(str(v) for v in x.dropna().unique() if str(v) != 'nan')),
    )
)

# Validate grain uniqueness
review_duplicates = reviews_agg.duplicated(subset=["order_id", "product_id"]).sum()
if review_duplicates > 0:
    print(f"      ⚠️ Found {review_duplicates} duplicate (order_id, product_id) in reviews_agg")
    reviews_agg = reviews_agg.drop_duplicates(subset=["order_id", "product_id"], keep='first')

# Build fact_transactions
fact_transactions = order_items.copy()

# Merge với validation cho từng bước
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
# TABLE 4: fact_operations
# sales + web_traffic (daily) + inventory (aggregate → monthly)
# Grain: 1 dòng = 1 ngày (sales & web_traffic); inventory join theo year+month
# ─────────────────────────────────────────────
print("\n📈 [4/4] Building fact_operations ...")

sales       = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), parse_dates=["Date"])
web_traffic = pd.read_csv(os.path.join(DATA_DIR, "web_traffic.csv"), parse_dates=["date"])
inventory   = pd.read_csv(os.path.join(DATA_DIR, "inventory.csv"))

# Cắt bỏ phần giờ phút giây để đảm bảo merge khớp 100%
sales["Date"] = sales["Date"].dt.normalize()
web_traffic = web_traffic.rename(columns={"date": "Date"})
web_traffic["Date"] = web_traffic["Date"].dt.normalize()

# Aggregate web_traffic đề phòng 1 ngày có nhiều dòng (theo device/channel)
metrics_cols = [col for col in web_traffic.select_dtypes(include='number').columns if not col.endswith('_id')]
web_traffic_agg = web_traffic.groupby("Date", as_index=False)[metrics_cols].sum()

print(f"      📊 Web traffic: {len(web_traffic)} → {len(web_traffic_agg)} days after aggregation")

# Aggregate inventory → monthly grain
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

# Add year/month helper columns to sales for joining inventory
sales["year"]  = sales["Date"].dt.year
sales["month"] = sales["Date"].dt.month

# Merge sales + web_traffic (daily)
fact_ops = sales.merge(web_traffic_agg, on="Date", how="left")

# Merge inventory (monthly)
fact_ops = fact_ops.merge(inv_monthly, on=["year", "month"], how="left")

# Clean up helper columns
fact_ops = fact_ops.drop(columns=["year", "month"])

fact_ops.to_csv(os.path.join(OUTPUT_DIR, "fact_operations.csv"), index=False)
print(f"    fact_operations: {len(fact_ops):>7,} rows | {list(fact_ops.columns)}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("✅ Done! Output files:")
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