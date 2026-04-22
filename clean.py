import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data_cleaned_final')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def find_col(df, *possible_names):
    # Hệ thống dò tìm: Ưu tiên tìm đúng tên, nếu không có sẽ tìm không phân biệt Hoa/Thường
    # Tìm chính xác
    for name in possible_names:
        if name in df.columns:
            return name
    # Tìm không phân biệt hoa thường (ví dụ: 'Date' == 'date')
    lower_columns = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_columns:
            return lower_columns[name.lower()]
    return None  # Không tìm thấy thì báo None để bỏ qua an toàn



def clean_sales(df):
    date_col = find_col(df, 'Date', 'date', 'order_date')
    rev_col = find_col(df, 'Revenue', 'revenue')
    cogs_col = find_col(df, 'COGS', 'cogs', 'cost')

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col)
    if rev_col:
        df[rev_col] = df[rev_col].interpolate(method='linear').clip(lower=0)
        df[rev_col] = df[rev_col].clip(upper=df[rev_col].quantile(0.99))
    if cogs_col:
        df[cogs_col] = df[cogs_col].interpolate(method='linear').clip(lower=0)
    return df


def clean_promotions(df):
    start_col = find_col(df, 'start_date', 'StartDate')
    end_col = find_col(df, 'end_date', 'EndDate')
    disc_col = find_col(df, 'discount_value', 'discount')
    type_col = find_col(df, 'promo_type', 'type')

    if start_col:
        df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        df = df.dropna(subset=[start_col])
    if end_col:
        df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
    if disc_col:
        df[disc_col] = df[disc_col].fillna(0)
    if type_col:
        df[type_col] = df[type_col].fillna('Standard')
    return df


def clean_traffic(df):
    date_col = find_col(df, 'date', 'Date')
    sess_col = find_col(df, 'sessions', 'visits')

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
    if sess_col:
        df[sess_col] = df[sess_col].fillna(df[sess_col].median())
        df['sessions_smooth'] = df[sess_col].rolling(window=7, min_periods=1).mean()
    return df


def clean_customers(df):
    age_col = find_col(df, 'age', 'Age')
    gen_col = find_col(df, 'gender', 'Gender')
    join_col = find_col(df, 'join_date', 'JoinDate')

    if age_col:
        df[age_col] = df[age_col].fillna(df[age_col].median())
    if gen_col:
        df[gen_col] = df[gen_col].fillna('Other')
    if join_col:
        df[join_col] = pd.to_datetime(df[join_col], errors='coerce').ffill()
    return df


def clean_products(df):
    price_col = find_col(df, 'base_price', 'price', 'Price')
    cat_col = find_col(df, 'category', 'Category')
    name_col = find_col(df, 'product_name', 'name', 'ProductName')

    if price_col:
        df[price_col] = df[price_col].fillna(df[price_col].median())
    if cat_col:
        df[cat_col] = df[cat_col].fillna('General')
    if name_col:
        df[name_col] = df[name_col].fillna('Unnamed_Product')
    return df


def clean_inventory(df):
    date_col = find_col(df, 'date', 'Date', 'snapshot_date')
    pid_col = find_col(df, 'product_id', 'ProductID')
    stock_col = find_col(df, 'stock_level', 'stock', 'quantity')

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    if date_col and pid_col:
        df = df.sort_values([pid_col, date_col])
    elif date_col:
        df = df.sort_values(date_col)

    if stock_col:
        if pid_col:
            df[stock_col] = df.groupby(pid_col)[stock_col].ffill().fillna(0)
        else:
            df[stock_col] = df[stock_col].ffill().fillna(0)
    return df


def clean_orders(df):
    date_col = find_col(df, 'order_date', 'Date', 'date')
    status_col = find_col(df, 'status', 'Status')
    amt_col = find_col(df, 'total_amount', 'amount')
    oid_col = find_col(df, 'order_id', 'OrderID')

    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if status_col: df[status_col] = df[status_col].fillna('Completed')
    if amt_col: df[amt_col] = df[amt_col].clip(lower=0)
    if oid_col: df = df.dropna(subset=[oid_col])
    return df


def clean_payments(df):
    pay_val_col = find_col(df, 'payment_value')
    inst_col = find_col(df, 'installments', 'payment_installments')

    if pay_val_col: df[pay_val_col] = df[pay_val_col].fillna(0).clip(lower=0)
    if inst_col: df[inst_col] = df[inst_col].fillna(1)
    return df


def clean_order_items(df):
    qty_col = find_col(df, 'quantity', 'Qty')
    price_col = find_col(df, 'unit_price', 'price')

    if qty_col: df[qty_col] = df[qty_col].fillna(1).clip(lower=1)
    if price_col: df[price_col] = df[price_col].fillna(0)
    return df


def clean_geography(df):
    city_col = find_col(df, 'city', 'City')
    reg_col = find_col(df, 'region', 'Region')
    if city_col: df[city_col] = df[city_col].fillna('Unknown_City')
    if reg_col: df[reg_col] = df[reg_col].fillna('Unknown_Region')
    return df


def clean_reviews(df):
    rating_col = find_col(df, 'rating', 'score')
    text_col = find_col(df, 'review_text', 'text')
    if rating_col: df[rating_col] = df[rating_col].fillna(3.0).clip(1, 5)
    if text_col: df[text_col] = df[text_col].fillna('')
    return df


def clean_returns(df):
    date_col = find_col(df, 'return_date', 'Date')
    rsn_col = find_col(df, 'reason', 'Reason')
    oid_col = find_col(df, 'order_id', 'OrderID')
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if rsn_col: df[rsn_col] = df[rsn_col].fillna('Not_Specified')
    if oid_col: df = df.dropna(subset=[oid_col])
    return df


def clean_suppliers(df):
    name_col = find_col(df, 'supplier_name', 'name')
    lead_col = find_col(df, 'lead_time', 'time')
    if name_col: df[name_col] = df[name_col].fillna('General_Supplier')
    if lead_col: df[lead_col] = df[lead_col].fillna(df[lead_col].median())
    return df


def clean_marketing(df):
    date_col = find_col(df, 'date', 'Date')
    spend_col = find_col(df, 'spend', 'cost')
    chan_col = find_col(df, 'channel', 'source')
    if date_col: df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if spend_col: df[spend_col] = df[spend_col].fillna(0).clip(lower=0)
    if chan_col: df[chan_col] = df[chan_col].fillna('Organic')
    return df


def clean_loyalty(df):
    pts_col = find_col(df, 'points', 'score')
    lvl_col = find_col(df, 'membership_level', 'level')
    if pts_col: df[pts_col] = df[pts_col].fillna(0).clip(lower=0)
    if lvl_col: df[lvl_col] = df[lvl_col].fillna('Bronze')
    return df


def clean_submission(df):
    return df


CLEAN_MAP = {
    'sales.csv': clean_sales, 'promotions.csv': clean_promotions, 'web_traffic.csv': clean_traffic,
    'customers.csv': clean_customers, 'products.csv': clean_products, 'inventory.csv': clean_inventory,
    'orders.csv': clean_orders, 'order_items.csv': clean_order_items, 'geography.csv': clean_geography,
    'reviews.csv': clean_reviews, 'returns.csv': clean_returns, 'suppliers.csv': clean_suppliers,
    'marketing_spend.csv': clean_marketing, 'loyalty_program.csv': clean_loyalty,
    'sample_submission.csv': clean_submission,
    'payments.csv': clean_payments
}


def create_master_data():
    try:
        df_order_items = pd.read_csv(os.path.join(OUTPUT_DIR, 'order_items.csv'), low_memory=False)
        df_orders = pd.read_csv(os.path.join(OUTPUT_DIR, 'orders.csv'), low_memory=False)
        df_products = pd.read_csv(os.path.join(OUTPUT_DIR, 'products.csv'))
        df_customers = pd.read_csv(os.path.join(OUTPUT_DIR, 'customers.csv'))
        df_geo = pd.read_csv(os.path.join(OUTPUT_DIR, 'geography.csv'))
        df_payments = pd.read_csv(os.path.join(OUTPUT_DIR, 'payments.csv'))

        # Nối chi tiết đơn hàng (order_items) với đơn hàng tổng quát (orders)
        master = pd.merge(df_order_items, df_orders, on='order_id', how='left')

        # Nối với thông tin sản phẩm (products)
        master = pd.merge(master, df_products, on='product_id', how='left')

        # Nối với thông tin khách hàng (customers)
        master = pd.merge(master, df_customers.drop(columns=['zip', 'city'], errors='ignore'),on='customer_id', how='left')

        # Nối với thông tin địa lý (geography) dựa trên mã zip giao hàng
        master = pd.merge(master, df_geo, on='zip', how='left')

        # Nối với thông tin thanh toán (payments)
        master = pd.merge(master, df_payments.drop(columns=['payment_method'], errors='ignore'),on='order_id', how='left')

        # Tính toán các chỉ số dẫn xuất (Feature Engineering)
        master['net_revenue'] = master['quantity'] * master['unit_price']
        master['total_cogs'] = master['quantity'] * master['cogs']
        master['gross_profit'] = master['net_revenue'] - master['total_cogs']
        master['profit_margin'] = (master['gross_profit'] / master['net_revenue']).fillna(0)

        # Xuất file
        master_path = os.path.join(OUTPUT_DIR, 'master_data.csv')
        master.to_csv(master_path, index=False)

    except Exception as e:
        print(f"Lỗi khi gộp Master Data: {e}")


def main():
    for file_name, clean_func in CLEAN_MAP.items():
        file_path = os.path.join(INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df_clean = clean_func(df)
        df_clean.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False)
    create_master_data()

if __name__ == "__main__":
    main()