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

# ─────────────────────────────────────────────
# TABLE 1: dim_products  (giữ nguyên products.csv)
# promotions gộp vào đây vì được reference từ transactions
# ─────────────────────────────────────────────
print("📦 [1/4] Building dim_products ...")

products   = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
promotions = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"))

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
geo_cols = geography[["zip", "region", "district"]]

dim_customers = customers.merge(geo_cols, on="zip", how="left")

dim_customers.to_csv(os.path.join(OUTPUT_DIR, "dim_customers.csv"), index=False)
print(f"    dim_customers : {len(dim_customers):>7,} rows | {list(dim_customers.columns)}")


# ─────────────────────────────────────────────
# TABLE 3: fact_transactions
# order_items + payments + shipments + returns (agg) + reviews (agg)
# Grain: 1 dòng = 1 sản phẩm trong 1 đơn hàng
# ─────────────────────────────────────────────
print("\n🛒 [3/4] Building fact_transactions ...")

order_items = pd.read_csv(
    os.path.join(DATA_DIR, "order_items.csv"),
    dtype={"promo_id": str, "promo_id_2": str},
    low_memory=False
)
payments  = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
shipments = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"))
returns   = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
reviews   = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

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
        customer_id  = ("customer_id",  "first"),
        review_date  = ("review_date",  "first"),
        rating       = ("rating",       "mean"),
        review_title = ("review_title", "first"),
    )
)

# Build fact_transactions
fact_transactions = (
    order_items
    .merge(payments,   on="order_id",              how="left")
    .merge(shipments,  on="order_id",              how="left")
    .merge(returns_agg, on=["order_id", "product_id"], how="left")
    .merge(reviews_agg, on=["order_id", "product_id"], how="left")
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

web_traffic = web_traffic.rename(columns={"date": "Date"})

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
fact_ops = sales.merge(web_traffic, on="Date", how="left")

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