"""
=============================================================
INSIGHT 1: "BẪY TRAFFIC" - NGUỒN KHÁCH ĐÔNG NHƯNG KHÔNG TINH
=============================================================
* HIỆN TRẠNG: Kênh Paid Search mang về lượng truy cập khổng lồ (Top 2) nhưng thời gian khách ở lại trang rất ngắn. Ngược lại, Email Campaign ít traffic hơn nhưng thời gian onsite lại cao nhất.
* NGUYÊN NHÂN: Paid Search thường kéo nhầm tệp khách hàng "vãng lai" (tò mò click vào rồi thoát). Trong khi Email tiếp cận đúng tệp khách quen, có nhu cầu mua sắm thực sự.
* GIẢI PHÁP: Cắt giảm ngân sách Paid Search ở các từ khóa chung chung. Dồn tiền nâng cấp hệ thống Automation Email để tăng tỷ lệ chốt đơn với chi phí thấp hơn.
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

print("--- ĐANG PHÂN TÍCH INSIGHT 1 ---")

# ==========================================
# INSIGHT 1: BẪY TRAFFIC (Đọc từ web_traffic.csv)
# ==========================================
df_traffic = pd.read_csv('data/web_traffic.csv')

# Tính tổng truy cập và trung bình thời gian ở lại trang
traffic_insight = df_traffic.groupby('traffic_source').agg({
    'sessions': 'sum',
    'avg_session_duration_sec': 'mean'
}).reset_index().sort_values(by='sessions', ascending=False)

# Vẽ biểu đồ Insight 1: So sánh Sessions vs Duration
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

sns.barplot(data=traffic_insight, x='traffic_source', y='sessions', ax=ax1, color='lightblue', alpha=0.7)
sns.lineplot(data=traffic_insight, x='traffic_source', y='avg_session_duration_sec', ax=ax2, color='red', marker='o', linewidth=2)

ax1.set_ylabel('Tổng số lượt truy cập (Sessions)', color='blue')
ax2.set_ylabel('Thời gian trung bình (Giây)', color='red')
plt.title('Bẫy Traffic: Lượng truy cập (Cột) vs Độ gắn kết (Đường đỏ)')
plt.savefig(os.path.join(CHART_DIR, 'Insight_1_BayTraffic.png'), dpi=300, bbox_inches='tight')
plt.close()

print("1. Đã vẽ xong biểu đồ Bẫy Traffic!")
print(f"✅ Ảnh đã lưu tại thư mục: {CHART_DIR}")