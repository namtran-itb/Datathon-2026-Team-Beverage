"""
visualize_insights.py
─────────────────────────────────────────────────────────────────
Input  : merge_table/   (đặt 5 file CSV vào đây)
Output : visualization/ (tự tạo nếu chưa có, xuất 3 file PNG)
─────────────────────────────────────────────────────────────────
Cấu trúc thư mục khi chạy:
  project/
  ├── visualize_insights.py
  ├── merge_table/
  │   ├── dim_customers.csv
  │   ├── dim_products.csv
  │   ├── dim_promotions.csv
  │   ├── fact_operations.csv
  │   └── fact_transactions.csv
  └── visualization/          ← tự tạo
      ├── insight1_chat_luong_luu_luong.png
      ├── insight2_thiet_ke_khuyen_mai.png
      └── insight3_rui_ro_hoan_hang.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "merge_table"
OUTPUT_DIR = BASE_DIR / "visualization"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Bảng màu ─────────────────────────────────────────────────────────────────
C_PAID    = "#E05252"
C_ORGANIC = "#4A90D9"
C_WARN    = "#F5A623"
C_GOOD    = "#5BAD72"
C_NEUTRAL = "#9B9B9B"
C_BG      = "#F7F9FC"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
})

# ════════════════════════════════════════════════════════════════════════════
#  ĐỌC DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════
print("Đang đọc dữ liệu...")
df_t  = pd.read_csv(DATA_DIR / "fact_transactions.csv", low_memory=False)
df_c  = pd.read_csv(DATA_DIR / "dim_customers.csv")

order_counts = df_t.groupby("customer_id")["order_id"].nunique().reset_index(name="order_count")
df_c = df_c.merge(order_counts, on="customer_id", how="left")
df_c["order_count"] = df_c["order_count"].fillna(0)

df_p  = pd.read_csv(DATA_DIR / "dim_products.csv")
df_pr = pd.read_csv(DATA_DIR / "dim_promotions.csv")

# Guard NaN trước khi chia bằng np.where
df_p["price"] = df_p["price"].fillna(0)
df_p["cogs"]  = df_p["cogs"].fillna(0)
df_p["margin_pct"] = np.where(
    df_p["price"] > 0,
    ((df_p["price"] - df_p["cogs"]) / df_p["price"]) * 100,
    0.0
)

# Cảnh báo nếu giá bán còn rẻ hơn giá vốn
if (df_p["cogs"] > df_p["price"]).any():
    negative_count = (df_p["cogs"] > df_p["price"]).sum()
    print(f"⚠️ CẢNH BÁO: Dữ liệu có {negative_count} sản phẩm đang bán lỗ (COGS > Price)!")

if "start_date" in df_pr.columns:
    df_pr["year"] = pd.to_datetime(df_pr["start_date"]).dt.year
else:
    df_pr["year"] = 2026

df_p_unique = df_p.drop_duplicates(subset=["product_id"])
df_full = df_t.merge(df_p_unique, on="product_id", how="left")

print("✓ Đọc xong\n")

# ════════════════════════════════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ════════════════════════════════════════════════════════════════════════════
def add_banner(fig, title, subtitle, bg_color):
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=24, fontweight="bold",
             color="white", bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, linewidth=0))
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=12, color="#555555", style="italic")

def save_fig(fig, filename):
    out = OUTPUT_DIR / filename
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓ Saved → visualization/{filename}")


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHT 1 — Chất lượng lưu lượng truy cập (Traffic Quality)
# ════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 1...")
fig1 = plt.figure(figsize=(20, 9), facecolor=C_BG)
add_banner(fig1,
    "Chất Lượng Lưu Lượng Truy Cập",
    "",
    "#2C5F8A")
gs1 = gridspec.GridSpec(
    1, 3,
    figure=fig1,
    hspace=0.35,
    wspace=0.45,
    top=0.84,
    bottom=0.12,
    left=0.05,
    right=0.98
)

# 1A
ax1a = fig1.add_subplot(gs1[0, 0])
ch_map = {"organic_search": "Tìm kiếm tự nhiên", "social_media": "Mạng xã hội",
          "paid_search": "Tìm kiếm trả phí", "email_campaign": "Email",
          "referral": "Giới thiệu", "direct": "Trực tiếp"}
ch_counts = df_c["acquisition_channel"].value_counts().rename(ch_map)
clrs = [C_PAID if k in ("Mạng xã hội", "Tìm kiếm trả phí") else C_ORGANIC for k in ch_counts.index]
bars = ax1a.barh(ch_counts.index[::-1], ch_counts.values[::-1],
                 color=clrs[::-1], edgecolor="white", linewidth=0.6, height=0.65)
for bar, val in zip(bars, ch_counts.values[::-1]):
    ax1a.text(bar.get_width() + 250, bar.get_y() + bar.get_height() / 2,
              f"{val:,.0f}", va="center", fontsize=11)
ax1a.set_title("1A · Phân bổ kênh thu hút khách hàng", fontweight="bold", pad=10)
ax1a.set_xlabel("Số khách hàng")
ax1a.legend(handles=[mpatches.Patch(color=C_PAID, label="Kênh trả phí"),
                     mpatches.Patch(color=C_ORGANIC, label="Kênh miễn phí")],
            fontsize=11, loc="lower right")
ax1a.set_facecolor(C_BG)

# 1B
ax1b = fig1.add_subplot(gs1[0, 1])
paid_u = df_c[df_c["acquisition_channel"].isin(["social_media", "paid_search"])]
org_u  = df_c[df_c["acquisition_channel"] == "organic_search"]
age_order = sorted([str(x) for x in df_c["age_group"].dropna().unique()])
paid_ag = paid_u["age_group"].value_counts(normalize=True).reindex(age_order, fill_value=0) * 100
org_ag  = org_u["age_group"].value_counts(normalize=True).reindex(age_order, fill_value=0) * 100
x1b = np.arange(len(age_order)); w = 0.36
ax1b.bar(x1b - w/2, paid_ag, w, label="Trả phí",  color=C_PAID,    edgecolor="white")
ax1b.bar(x1b + w/2, org_ag,  w, label="Tự nhiên", color=C_ORGANIC, edgecolor="white")
ax1b.set_xticks(x1b); ax1b.set_xticklabels(age_order)
ax1b.set_ylabel("Tỷ lệ (%)"); ax1b.legend(fontsize=11)
ax1b.set_title("1B · Phân bổ nhóm tuổi: Trả phí và Tự nhiên", fontweight="bold", pad=10)
ax1b.set_facecolor(C_BG)

# 1C
ax1c = fig1.add_subplot(gs1[0, 2])
all_u = df_c.copy()


def calculate_returning_rate(df_sub):
    if df_sub is None or len(df_sub) == 0:
        return 0.0, 0.0
    total = len(df_sub)
    returning = (df_sub["order_count"] > 1).sum()
    ret_pct = (returning / total) * 100
    new_pct = ((total - returning) / total) * 100
    return new_pct, ret_pct

groups_data = {
    "Trả phí\n(social+paid)": calculate_returning_rate(paid_u),
    "Tự nhiên\n(organic)": calculate_returning_rate(org_u),
    "Toàn bộ\n(tổng)": calculate_returning_rate(all_u)
}

x1c = np.arange(len(groups_data))
new_vals = [v[0] for v in groups_data.values()]
ret_vals = [v[1] for v in groups_data.values()]

b_new = ax1c.bar(x1c, new_vals, label="Khách mới (1 đơn)", color=C_WARN, edgecolor="white")
b_ret = ax1c.bar(x1c, ret_vals, bottom=new_vals, label="Khách quay lại (≥2 đơn)", color=C_ORGANIC, edgecolor="white")

ax1c.set_xticks(x1c)
ax1c.set_xticklabels(groups_data.keys(), fontsize=10)
ax1c.set_ylabel("Tỷ lệ (%)")
ax1c.set_ylim(0, 110)
ax1c.set_title("1C · Tỷ lệ khách mới và khách quay lại theo kênh", fontweight="bold", pad=10)
ax1c.legend(loc="upper right")

for bar, val in zip(b_new, new_vals):
    if val > 5:
        ax1c.text(bar.get_x() + bar.get_width() / 2, val / 2, f"{val:.1f}%", ha="center", va="center", color="white",
                  fontweight="bold", fontsize=10)

for bar, nv, rv in zip(b_ret, new_vals, ret_vals):
    if rv > 5:
        ax1c.text(bar.get_x() + bar.get_width() / 2, nv + rv / 2, f"{rv:.1f}%", ha="center", va="center", color="white",
                  fontweight="bold", fontsize=10)

save_fig(fig1, "insight1_chat_luong_luu_luong.png")


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHT 2 — Thiết kế chương trình khuyến mãi (Promo Design)
# ════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 2...")
fig2 = plt.figure(figsize=(20, 9), facecolor=C_BG)
add_banner(fig2,
    "Thiết Kế Chương Trình Khuyến Mãi",
    "",
    "#7A3A0A")
gs2 = gridspec.GridSpec(
    1, 3,
    figure=fig2,
    hspace=0.35,
    wspace=0.45,
    top=0.84,
    bottom=0.12,
    left=0.05,
    right=0.98
)

# 2A
ax2a = fig2.add_subplot(gs2[0, 0])
seg_stats = df_p_unique.groupby("segment").agg(price=("price", "mean"), cogs=("cogs", "mean")).reset_index()

# Tự động lấy chiết khấu trung bình và biên an toàn từ Data
disc_val = df_pr["discount_value"].mean() if "discount_value" in df_pr.columns else 18.0
if pd.isna(disc_val): disc_val = 18.0
dynamic_discount = (disc_val / 100) if disc_val > 1 else disc_val
min_margin_thres = df_p_unique["margin_pct"].quantile(0.25) if len(df_p_unique) > 0 else 10.0

# Tính Margin Before & After
seg_stats["m_before"] = np.where(seg_stats["price"] > 0, (seg_stats["price"] - seg_stats["cogs"]) / seg_stats["price"] * 100, 0)
price_after = seg_stats["price"] * (1 - dynamic_discount)
seg_stats["m_after"] = np.where(price_after > 0, (price_after - seg_stats["cogs"]) / price_after * 100, 0)

seg_stats = seg_stats.sort_values("m_before")
seg_labels = seg_stats["segment"].tolist()
m_before = seg_stats["m_before"].tolist()
m_after = seg_stats["m_after"].tolist()

x2a = np.arange(len(seg_labels)); w2 = 0.36
ax2a.bar(x2a - w2/2, m_before, w2, label=f"Trước giảm {dynamic_discount*100:.0f}%", color=C_NEUTRAL, edgecolor="white")
# Đổi màu linh hoạt theo ngưỡng min_margin_thres
ax2a.bar(x2a + w2/2, m_after,  w2, label=f"Sau giảm {dynamic_discount*100:.0f}%",
         color=[C_PAID if v < min_margin_thres else C_WARN if v < min_margin_thres+3 else C_GOOD for v in m_after], edgecolor="white")

ax2a.axhline(min_margin_thres, color=C_WARN, linestyle="--", linewidth=1.5, label=f"Ngưỡng biên ({min_margin_thres:.1f}%)")
ax2a.set_xticks(x2a)
ax2a.set_xticklabels([s[:12] for s in seg_labels], rotation=25, ha="right")
ax2a.set_ylabel("Biên lợi nhuận (%)")
ax2a.set_title(f"2A · Biên lợi nhuận trước & sau giảm giá ({dynamic_discount*100:.0f}%)\ntheo phân khúc (Data thật)", fontweight="bold", pad=10)
ax2a.legend(loc="upper left")

# Cập nhật nhãn % trên cột
for i, v in enumerate(m_after):
    ax2a.text(x2a[i] + w2/2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10,
              color=C_PAID if v < min_margin_thres else C_WARN if v < min_margin_thres+3 else C_GOOD, fontweight="bold")

# 2B
ax2b = fig2.add_subplot(gs2[0, 1])
if "discount_value" in df_pr.columns:
    yr_disc = df_pr.groupby("year")["discount_value"].mean()

    if yr_disc.mean() <= 1.0:
        yr_disc = yr_disc * 100

    ax2b.bar(yr_disc.index, yr_disc.values, color=[C_PAID if v > 18 else C_GOOD for v in yr_disc.values], width=0.7)
    ax2b.set_xlabel("Năm"); ax2b.set_ylabel("Chiết khấu trung bình (%)")
    ax2b.set_title("2B · Chu kỳ chiết khấu theo năm", fontweight="bold", pad=10)
    for yr, val in zip(yr_disc.index, yr_disc.values):
        ax2b.text(yr, val + 0.35, f"{val:.0f}%", ha="center", color=C_PAID if val > 18 else C_GOOD, fontweight="bold")
else:
    ax2b.text(0.5, 0.5, "Không có cột discount_value", ha="center")


# 2C
ax2c = fig2.add_subplot(gs2[0, 2])
if all(col in df_pr.columns for col in ["stackable_flag", "min_order_value"]):
    high_risk = df_pr[(df_pr["stackable_flag"] == 1) & (df_pr["min_order_value"] == 0)]
    med_risk = df_pr[(df_pr["stackable_flag"] == 1) & (df_pr["min_order_value"] > 0)]
    low_risk = df_pr[df_pr["stackable_flag"] == 0]

    sizes = [len(high_risk), len(med_risk), len(low_risk)]
    if sum(sizes) == 0: sizes = [1, 1, 1]  # Dummy tránh crash nếu data trống

    labels_pie = [f"Rủi ro CAO\n(stackable, không min)\nn={len(high_risk)}",
                  f"Rủi ro TB\n(stackable, có min)\nn={len(med_risk)}",
                  f"Rủi ro THẤP\n(không stackable)\nn={len(low_risk)}"]

    wedges, texts, autotexts = ax2c.pie(
        sizes, labels=labels_pie, colors=[C_PAID, C_WARN, C_GOOD],
        autopct="%1.0f%%", startangle=90, labeldistance=1.1, pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=1.8))
    for at in autotexts:
        at.set_fontsize(13);
        at.set_fontweight("bold");
        at.set_color("white")
    ax2c.set_title(f"2C · Ma trận rủi ro khuyến mãi\n(Tổng n={sum(sizes)})", fontweight="bold", pad=10)
else:
    ax2c.text(0.5, 0.5, "Thiếu cột tính rủi ro", ha="center")

save_fig(fig2, "insight2_thiet_ke_khuyen_mai.png")


# ════════════════════════════════════════════════════════════════════════════
#  INSIGHT 3 — Rủi ro hoàn hàng (Return Risk)
# ════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 3...")
fig3 = plt.figure(figsize=(20, 9), facecolor=C_BG)
add_banner(fig3,
    "Rủi Ro Hoàn Hàng",
    "",
    "#5A1515")
gs3 = gridspec.GridSpec(
    1, 3,
    figure=fig3,
    hspace=0.35,
    wspace=0.45,
    top=0.84,
    bottom=0.12,
    left=0.05,
    right=0.98
)

# 3A
ax3a = fig3.add_subplot(gs3[0, 0])
cat_stats = df_p_unique.groupby("category").agg(count=("product_id", "count"), avg_margin=("margin_pct", "mean")).sort_values("count", ascending=False)
x3a = np.arange(len(cat_stats)); ax3a_tw = ax3a.twinx()

# Tự động tìm Top 1 Category
top_cat = cat_stats.index[0] if len(cat_stats) > 0 else ""
cat_colors = [C_PAID if cat == top_cat else C_NEUTRAL for cat in cat_stats.index]

ax3a.bar(x3a, cat_stats["count"], color=cat_colors, edgecolor="white", label="Số SKU")

ax3a_tw.plot(x3a, cat_stats["avg_margin"], "o--", color=C_WARN, linewidth=2.2, markersize=8, label="Biên TB (%)")
ax3a.set_xticks(x3a); ax3a.set_xticklabels(cat_stats.index)
ax3a.set_ylabel("Số lượng SKU")
ax3a_tw.set_ylabel("Biên LN trung bình (%)", color=C_WARN)
ax3a_tw.tick_params(axis="y", colors=C_WARN)
ax3a.set_title("3A · Số lượng SKU & Biên lợi nhuận\ntheo danh mục sản phẩm", fontweight="bold", pad=10)

l1, lb1 = ax3a.get_legend_handles_labels(); l2, lb2 = ax3a_tw.get_legend_handles_labels()
ax3a.legend(l1 + l2, lb1 + lb2, loc="upper right")

if len(cat_stats) > 0:
    top_pct = (cat_stats["count"].iloc[0] / cat_stats["count"].sum()) * 100
    ax3a.text(0, cat_stats["count"].iloc[0] * 1.01, f"{top_pct:.1f}%\nDanh mục", ha="center", fontsize=10, color=C_PAID, fontweight="bold")

# 3B
ax3b = fig3.add_subplot(gs3[0, 1])
seg_risk = df_p_unique.groupby("segment").agg(avg_price=("price", "mean"), avg_margin=("margin_pct", "mean"), count=("product_id", "count")).reset_index()

# Guard chống crash/mất biểu đồ khi median() trả về NaN
if len(seg_risk) > 0:
    p_med = seg_risk["avg_price"].median()
    m_med = seg_risk["avg_margin"].median()
    price_thres  = p_med if pd.notna(p_med) else 0.0
    margin_thres = m_med if pd.notna(m_med) else 0.0
else:
    price_thres, margin_thres = 0.0, 0.0

s_colors = [C_PAID if (r["avg_price"] > price_thres and r["avg_margin"] < margin_thres) else C_GOOD for _, r in seg_risk.iterrows()]
ax3b.scatter(seg_risk["avg_price"], seg_risk["avg_margin"], s=seg_risk["count"] / 2.8, c=s_colors, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)

for _, r in seg_risk.iterrows():
    ax3b.annotate(str(r["segment"])[:10], (r["avg_price"], r["avg_margin"]), textcoords="offset points", xytext=(6, 4), fontsize=10)

# Ngăn vẽ đường thẳng nếu thresholds rớt xuống 0 (trường hợp rỗng)
if margin_thres > 0:
    ax3b.axhline(margin_thres, color=C_WARN, linestyle="--", linewidth=1.2, label=f"Ngưỡng biên {margin_thres:.0f}%", zorder=2)
    ax3b.axhspan(ymin=0, ymax=margin_thres, xmin=0, color=C_PAID, alpha=0.04, zorder=1)
if price_thres > 0:
    ax3b.axvline(price_thres, color=C_WARN, linestyle=":", linewidth=1.2, label=f"Ngưỡng giá {price_thres:.0f} VND", zorder=2)
ax3b.set_xlabel("Giá trung bình (VND)"); ax3b.set_ylabel("Biên lợi nhuận trung bình (%)")
ax3b.set_title("3B · Bản đồ rủi ro phân khúc", fontweight="bold", pad=10)

# Chú thích tự động scale theo Data
if len(seg_risk) > 0:
    min_sz, max_sz = seg_risk["count"].min(), seg_risk["count"].max()
    mid_sz = (min_sz + max_sz) / 2
    for sz_val in [min_sz, mid_sz, max_sz]:
        ax3b.scatter([], [], s=sz_val/2.8, c=C_NEUTRAL, alpha=0.6, label=f"~{int(sz_val)} SKU")
ax3b.legend(loc="lower right")

# Bổ sung Guard cho legend
if margin_thres > 0 or price_thres > 0:
    ax3b.legend(loc="lower right")
else:
    ax3b.text(0.5, 0.5, "Không đủ data để tính ngưỡng",
              ha="center", transform=ax3b.transAxes,
              color="gray", fontsize=11)

# 3C
ax3c = fig3.add_subplot(gs3[0, 2])
if "is_returned" in df_full.columns:
    ret_by_cat = df_full.groupby("category").agg(
        total_refund=("refund_amount", "sum"),
        total_orders=("order_id", "count"),
        total_returns=("is_returned", "sum")
    ).assign(return_rate=lambda x: np.where(x["total_orders"] > 0, x["total_returns"] / x["total_orders"] * 100, 0))

    ret_by_cat = ret_by_cat.sort_values("total_refund", ascending=False)
    x3c = np.arange(len(ret_by_cat));
    ax3c_tw = ax3c.twinx()
    bar_c3 = []
    for i in range(len(ret_by_cat)):
        if i == 0:
            bar_c3.append(C_PAID)
        elif i == 1:
            bar_c3.append(C_WARN)
        else:
            bar_c3.append(C_NEUTRAL)

    ax3c.bar(x3c, ret_by_cat["total_refund"] / 1e6, color=bar_c3, edgecolor="white",
             label="Tổng giá trị hoàn (Triệu VND)")
    ax3c_tw.plot(x3c, ret_by_cat["return_rate"], "D--", color=C_ORGANIC, linewidth=2.2, markersize=8,
                 label="Tỷ lệ hoàn hàng (%)")

    ax3c.set_xticks(x3c);
    ax3c.set_xticklabels(ret_by_cat.index)
    ax3c.set_ylabel("Tổng giá trị hoàn (Triệu VND)")
    ax3c_tw.set_ylabel("Tỷ lệ hoàn hàng (%)", color=C_ORGANIC)
    ax3c.set_title("3C · Tỷ lệ hoàn hàng theo danh mục", fontweight="bold", pad=10)

    l1, lb1 = ax3c.get_legend_handles_labels();
    l2, lb2 = ax3c_tw.get_legend_handles_labels()
    ax3c.legend(l1 + l2, lb1 + lb2, loc="upper right")

    for bar, val in zip(ax3c.patches, ret_by_cat["total_refund"] / 1e6):
        ax3c.text(bar.get_x() + bar.get_width() / 2, val + (ret_by_cat["total_refund"].max() / 1e6) * 0.02,
                  f"{val:.0f}M", ha="center", fontweight="bold")
else:
    ax3c.text(0.5, 0.5, "Thiếu dữ liệu hoàn hàng (is_returned)", ha="center")

save_fig(fig3, "insight3_rui_ro_hoan_hang.png")


# ════════════════════════════════════════════════════════════════════════════
print("\n✅ Hoàn tất! Đã lưu 3 file PNG vào thư mục: visualization/")
print("   insight1_chat_luong_luu_luong.png")
print("   insight2_thiet_ke_khuyen_mai.png")
print("   insight3_rui_ro_hoan_hang.png")