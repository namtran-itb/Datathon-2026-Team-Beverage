import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def run_analysis():
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {DATA_DIR}")
        return

    # Đọc các bảng dữ liệu
    df_products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    df_customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
    df_geography = pd.read_csv(os.path.join(DATA_DIR, 'geography.csv'))
    df_orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    df_order_items = pd.read_csv(os.path.join(DATA_DIR, 'order_items.csv'), low_memory=False)
    df_payments = pd.read_csv(os.path.join(DATA_DIR, 'payments.csv'))
    df_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'))
    df_sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
    df_web_traffic = pd.read_csv(os.path.join(DATA_DIR, 'web_traffic.csv'))

    print("-" * 30)

    # --- Câu 1: Khoảng cách mua hàng (Trung vị) ---
    df_order_complete = df_orders[df_orders['order_status'] == 'delivered'].copy()
    df_order_complete['order_date'] = pd.to_datetime(df_order_complete['order_date'])
    df_lich_su_mua = df_order_complete[['customer_id', 'order_date']].drop_duplicates().sort_values(by=['customer_id', 'order_date'])
    df_lich_su_mua['ngay_khoang_cach'] = df_lich_su_mua.groupby('customer_id')['order_date'].diff().dt.days
    print(f"Câu 1 (Trung vị khoảng cách): {df_lich_su_mua['ngay_khoang_cach'].median()} ngày")

    # --- Câu 2: Tỉ suất lợi nhuận (Segment) ---
    df_products['margin'] = (df_products['price'] - df_products['cogs']) / df_products['price']
    res2 = df_products.groupby('segment')['margin'].mean().idxmax()
    print(f"Câu 2 (Segment lợi nhuận nhất): {res2}")

    # --- Câu 3: Lý do trả hàng (Streetwear) ---
    df_streetwear = pd.merge(df_returns, df_products[df_products['category'] == 'Streetwear'], on='product_id')
    res3 = df_streetwear['return_reason'].value_counts().idxmax()
    print(f"Câu 3 (Lý do trả hàng Streetwear): {res3}")

    # --- Câu 4: Nguồn traffic (Bounce rate thấp nhất) ---
    df_traffic = pd.read_csv(os.path.join(DATA_DIR, 'web_traffic.csv'))
    res4 = df_traffic.groupby('traffic_source')['bounce_rate'].mean().idxmin()
    print(f"Câu 4 (Nguồn traffic ổn định nhất): {res4}")

    # --- Câu 5: Tỷ lệ đơn có khuyến mãi ---
    ti_le_promo = (df_order_items['promo_id'].notna().sum() / len(df_order_items)) * 100
    print(f"Câu 5 (Tỷ lệ khuyến mãi): {ti_le_promo:.0f}%")

    # --- Câu 6: Độ tuổi mua hàng nhiều nhất ---
    # Phân tích dựa trên số lượng đơn hàng trung bình trên mỗi khách hàng
    df_cust = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
    df_merge_6 = pd.merge(df_orders, df_cust, on='customer_id')
    res6 = (df_merge_6.groupby('age_group')['order_id'].count() / df_cust.groupby('age_group')['customer_id'].count()).idxmax()
    print(f"Câu 6 (Độ tuổi mua nhiều nhất): {res6}")

    # --- Câu 7: Doanh thu vùng cao nhất ---
    df_order_items['doanh_thu'] = df_order_items['quantity'] * df_order_items['unit_price']
    df_merge_7 = pd.merge(df_order_items, df_order_complete, on='order_id')
    df_merge_7 = pd.merge(df_merge_7, df_geography, on='zip')
    res7 = df_merge_7.groupby('region')['doanh_thu'].sum().idxmax()
    print(f"Câu 7 (Vùng doanh thu cao nhất): {res7}")

    # --- Câu 8: Thanh toán của đơn hủy ---
    res8 = df_orders[df_orders['order_status'] == 'cancelled']['payment_method'].value_counts().idxmax()
    print(f"Câu 8 (Thanh toán đơn hủy nhiều nhất): {res8}")

    # --- Câu 9: Tỷ lệ trả hàng theo Size ---
    df_item_size = pd.merge(df_order_items, df_products[['product_id', 'size']], on='product_id')
    df_ret_size = pd.merge(df_returns, df_products[['product_id', 'size']], on='product_id')
    res9 = (df_ret_size['size'].value_counts() / df_item_size['size'].value_counts()).idxmax()
    print(f"Câu 9 (Size có tỷ lệ trả hàng cao nhất): {res9}")

    # --- Câu 10: Giá trị trả góp trung bình ---
    df_q10 = df_payments.groupby('installments')['payment_value'].mean()
    res10 = df_q10.idxmax()
    print(f"Câu 10 (Gói trả góp cao nhất): {res10} kỳ")

if __name__ == "__main__":
    run_analysis()