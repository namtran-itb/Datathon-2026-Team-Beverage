"""
=============================================================
INSIGHT 2: NGHỊCH LÝ "FLEXING" - ĐAM MÊ HÀNG HIỆU, TRẢ GÓP NHIỀU
=============================================================
* HIỆN TRẠNG: Khách hàng trẻ (18-25 tuổi) mua đồ giá cao (Premium/Streetwear) rất nhiều, nhưng tỷ lệ sử dụng trả góp lại cao đột biến so với các nhóm tuổi khác.
* NGUYÊN NHÂN: Giới trẻ có tâm lý thích hàng hiệu để thể hiện bản thân ("flexing"), nhưng dòng tiền hàng tháng chưa đủ lớn để trả thẳng 1 lần.
* GIẢI PHÁP: Thiết kế gói "Mua trước trả sau". Tự động hiển thị to/rõ ràng nút "Trả góp 6 kỳ 0%" khi hệ thống nhận diện khách hàng dưới 25 tuổi đang xem sản phẩm đắt tiền.
=============================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình giao diện
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# Thư mục lưu biểu đồ
CHART_DIR = 'eda_charts'
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

print("--- ĐANG PHÂN TÍCH INSIGHT 2 ---")
# ==========================================
# INSIGHT 2: NGHỊCH LÝ FLEXING (Join trực tiếp trên RAM)
# ==========================================
print("Đang xử lý Insight 2 (Tự động gộp bảng)...")
# 1. Đọc các file lẻ từ thư mục gốc
df_items = pd.read_csv('data/order_items.csv', low_memory=False)
df_orders = pd.read_csv('data/orders.csv', low_memory=False)
df_products = pd.read_csv('data/products.csv')
df_customers = pd.read_csv('data/customers.csv')
df_payments = pd.read_csv('data/payments.csv')

# 2. Lấy xương sống là order_items, nhặt thêm các cột cần thiết vào
# Nhặt customer_id từ bảng orders
df_merge = pd.merge(df_items, df_orders[['order_id', 'customer_id']], on='order_id', how='left')

# Nhặt segment từ bảng products
df_merge = pd.merge(df_merge, df_products[['product_id', 'segment']], on='product_id', how='left')

# Nhặt age_group từ bảng customers
df_merge = pd.merge(df_merge, df_customers[['customer_id', 'age_group']], on='customer_id', how='left')

# Nhặt installments từ bảng payments
df_merge = pd.merge(df_merge, df_payments[['order_id', 'installments']], on='order_id', how='left')

# 3. Tính toán Insight
# Lọc ra những người mua hàng Premium/Streetwear
df_premium = df_merge[df_merge['segment'].isin(['Premium', 'Streetwear'])].copy()

# Tạo cột cờ: Có trả góp hay không (Kỳ hạn > 1)
df_premium['is_installment'] = df_premium['installments'].apply(lambda x: 1 if pd.notna(x) and x > 1 else 0)

# Tính tỷ lệ trả góp theo nhóm tuổi
age_installment = df_premium.groupby('age_group')['is_installment'].mean().reset_index()
age_installment['is_installment'] = age_installment['is_installment'] * 100

# 4. Vẽ biểu đồ
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=age_installment, x='age_group', y='is_installment', hue='age_group', palette='magma', legend=False)

# Vòng lặp để gắn số liệu lên từng cột (định dạng 1 chữ số thập phân + dấu %)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

plt.title('Tỷ lệ sử dụng Trả góp khi mua đồ giá cao theo Nhóm tuổi')
plt.xlabel('Nhóm tuổi (Age Group)')
plt.ylabel('Tỷ lệ đơn dùng Trả góp (%)')
plt.savefig(os.path.join(CHART_DIR, 'Insight_2_NghichLyFlexing.png'), dpi=300, bbox_inches='tight')
plt.close()
print("2. Đã vẽ xong biểu đồ Nghịch lý Flexing!")
print(f"✅ Ảnh đã lưu tại thư mục: {CHART_DIR}")