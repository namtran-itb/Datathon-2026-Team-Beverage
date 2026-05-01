import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─── Đường dẫn ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "visualization"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Bảng màu ───────────────────────────────────────────────────────────────
RED   = "#E24B4A"
AMBER = "#EF9F27"
GRAY  = "#B4B2A9"
DARK  = "#2C2C2A"
LIGHT = "#F1EFE8"
GREEN = "#5BAD72"

# ─── Global style ───────────────────────────────────────────────────────────
# Đã TĂNG toàn bộ kích thước chữ mặc định
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6,
    "axes.titlesize": 34,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save_fig(fig, filename):
    out = OUTPUT_DIR / filename
    # GIẢM khoảng trắng phía trên: tăng rect top từ 0.90 lên 0.92
    fig.tight_layout(rect=[0, 0.02, 1, 0.92], h_pad=4, w_pad=6)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Đã lưu → visualization/{filename}")


# ═════════════════════════════════════════════════════════════════════════════
# INSIGHT 1 — Giảm giá ăn mòn biên lợi nhuận, không khuấy được hàng
# ═════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 1...")

channels     = ["Trực tuyến", "Mạng xã hội", "Tất cả kênh", "Tại cửa hàng", "Email"]
avg_discount = [28.3, 18.0, 15.8, 15.0, 10.6]
sell_through = [24.4, 24.4, 24.4, 24.4, 24.4]

segments       = ["Everyday", "Trendy", "Balanced", "Performance", "Activewear",
                  "All-weather", "Premium", "Standard"]
base_margins   = [23.6, 24.1, 25.8, 26.4, 26.6, 28.4, 28.5, 31.3]
disc_rate      = 20.0
# Công thức đúng: margin sau giảm giá = (margin_gốc - discount) / (100 - discount) * 100
# VD: Everyday 23.6% margin, giảm 20% → (23.6-20)/(100-20)*100 = 4.5%
effective_margins = [round((b - disc_rate) / (100 - disc_rate) * 100, 1) for b in base_margins]

fig1, axes1 = plt.subplots(1, 2, figsize=(28, 12))
fig1.subplots_adjust(wspace=0.5)
# GIẢM khoảng trắng: chỉnh y=0.96 và tăng fontsize=40
fig1.suptitle("GIẢM GIÁ ĂN MÒN BIÊN LỢI NHUẬN,\nKHÔNG KHUẤY ĐƯỢC HÀNG",
              fontsize=40, fontweight="bold", y=0.96)

# ── 1A: Giảm giá vs Tỷ lệ bán hàng theo kênh ───────────────────────────────
ax1a = axes1[0]
x     = np.arange(len(channels))
width = 0.35

bars1 = ax1a.bar(x - width / 2, avg_discount, width,
                 color=[RED if v > 20 else AMBER if v > 15 else GRAY for v in avg_discount],
                 label="Giảm giá trung bình (%)", zorder=3)
bars2 = ax1a.bar(x + width / 2, sell_through, width,
                 color=LIGHT, edgecolor=DARK, linewidth=0.8,
                 label="Tỷ lệ bán hàng (%)", zorder=3)

for bar, val in zip(bars1, avg_discount):
    ax1a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
              f"{val}%", ha="center", va="bottom", fontsize=22, fontweight="bold", color=DARK)
for bar, val in zip(bars2, sell_through):
    ax1a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
              f"{val}%", ha="center", va="bottom", fontsize=21, color=DARK)

ax1a.axhline(24.4, color=DARK, linestyle="--", linewidth=1, zorder=2)
ax1a.set_xticks(x)
ax1a.set_xticklabels(channels, fontsize=24, rotation=15, ha="right")
ax1a.set_ylabel("Tỷ lệ (%)", fontsize=26)
ax1a.set_ylim(0, 44)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax1a.set_title("1A · Giảm giá theo kênh —\ntỷ lệ bán hàng không tăng", fontsize=32, fontweight="bold", pad=10)
ax1a.legend(loc="upper right", fontsize=22)

# ── 1B: Biên lợi nhuận trước & sau giảm giá 20% ────────────────────────────
ax1b = axes1[1]
x1b  = np.arange(len(segments))
width1 = 0.38

colors_base = [RED if b < 25 else AMBER if b < 27 else GRAY for b in base_margins]
colors_eff  = [RED if e < 5 else AMBER if e < 10 else GREEN for e in effective_margins]

ax1b.bar(x1b - width1 / 2, base_margins, width1, color=colors_base,
         label="Biên lợi nhuận gốc (%)", zorder=3)
ax1b.bar(x1b + width1 / 2, effective_margins, width1, color=colors_eff,
         label="Biên lợi nhuận sau giảm 20% (%)", zorder=3, alpha=0.9)

for i, (b, e) in enumerate(zip(base_margins, effective_margins)):
    ax1b.text(i - width1 / 2, b + 0.7, f"{b}%", ha="center", fontsize=20, color=DARK)
    clr = RED if e < 5 else DARK
    ax1b.text(i + width1 / 2, max(e, 0) + 0.7, f"{e}%", ha="center", fontsize=20,
              color=clr, fontweight="bold" if e < 5 else "normal")

ax1b.axhline(5, color=RED, linestyle=":", linewidth=1.2, zorder=2)
ax1b.set_xticks(x1b)
ax1b.set_xticklabels(segments, rotation=35, ha="right", fontsize=24)
ax1b.set_ylabel("Biên lợi nhuận (%)", fontsize=26)
ax1b.set_ylim(-2, 46)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax1b.set_title("1B · Biên lợi nhuận trước & sau\ngiảm 20% theo phân khúc", fontsize=32, fontweight="bold", pad=10)
ax1b.legend(loc="upper left", fontsize=22)

save_fig(fig1, "insight1.png")


# ═════════════════════════════════════════════════════════════════════════════
# INSIGHT 2 — Vốn đông kết trong hàng tồn kho đóng băng
# ═════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 2...")

categories  = ["Streetwear", "Outdoor", "Casual", "GenZ"]
value_pct   = [75.1, 15.5, 6.6, 2.8]
colors_cat  = [RED, AMBER, GRAY, LIGHT]

categories2 = ["Streetwear", "Balanced", "Everyday", "Performance"]
total_val   = [75.1, 23.8, 25.7, 19.2]
sold_pct    = 0.244
sold_val    = [round(v * sold_pct, 1) for v in total_val]
locked_val  = [round(v * (1 - sold_pct), 1) for v in total_val]

regions       = ["Đông", "Trung", "Tây"]
customer_pct  = [47.7, 36.3, 16.0]
implied_inv   = [30.0, 35.0, 35.0]

fig2, axes2 = plt.subplots(1, 3, figsize=(36, 12))
fig2.subplots_adjust(wspace=0.55)
# GIẢM khoảng trắng: chỉnh y=0.96 và tăng fontsize=40
fig2.suptitle("VỐN ĐÔNG KẾT TRONG HÀNG\nTỒN KHO ĐÓNG BĂNG",
              fontsize=40, fontweight="bold", y=0.96)

# ── 2A: Phân phối giá trị danh mục ──────────────────────────────────────────
ax2a, ax2b, ax2c = axes2

# Donut
wedges, texts, autotexts = ax2a.pie(
    value_pct, labels=None, autopct="%1.1f%%",
    colors=colors_cat, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    pctdistance=0.75)
for at, val in zip(autotexts, value_pct):
    at.set_fontsize(22); at.set_fontweight("bold")
    at.set_color("white" if val > 10 else DARK)
ax2a.legend(wedges, [f"{c} ({v}%)" for c, v in zip(categories, value_pct)],
            loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=22, ncol=2)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax2a.set_title("2A · Phân phối giá trị\ndanh mục sản phẩm", fontsize=32, fontweight="bold", pad=10)
ax2a.text(0, 0, "75.1%\nStreet-\nwear", ha="center", va="center",
          fontsize=26, fontweight="bold", color=RED)

# Stacked bar: vốn đông kết vs đã bán
x2 = np.arange(len(categories2))
ax2b.bar(x2, locked_val, color=RED, label="Vốn đông kết (chưa bán)", zorder=3)
ax2b.bar(x2, sold_val, bottom=locked_val, color=GREEN,
         edgecolor=DARK, linewidth=0.5, label="Đã bán (24.4%)", zorder=3)
for i, (l, s) in enumerate(zip(locked_val, sold_val)):
    ax2b.text(i, l / 2, f"{l}%", ha="center", va="center", fontsize=22,
              color="white", fontweight="bold")
    # Đặt text phía trên thanh nếu segment sold quá mỏng để tránh tràn
    if s > 10:
        ax2b.text(i, l + s / 2, f"{s}%", ha="center", va="center", fontsize=20, color=DARK)
    else:
        ax2b.text(i, l + s + 0.8, f"{s}%", ha="center", va="bottom", fontsize=20, color=DARK)
ax2b.set_xticks(x2)
ax2b.set_xticklabels(categories2, rotation=25, ha="right", fontsize=24)
ax2b.set_ylabel("% Giá trị danh mục", fontsize=26)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax2b.set_title("2B · Vốn đông kết và đã bán\ntheo phân khúc", fontsize=32, fontweight="bold", pad=10)
ax2b.legend(fontsize=22, loc="upper right")

# ── 2C: Khách hàng theo vùng vs tồn kho ước tính ──────────────────────────
x3    = np.arange(len(regions))
width = 0.38

b1 = ax2c.bar(x3 - width/2, customer_pct, width, color=[RED, AMBER, GRAY],
              label="% Khách hàng thực tế", zorder=3)
b2 = ax2c.bar(x3 + width/2, implied_inv, width, color=LIGHT,
              edgecolor=DARK, linewidth=0.8, label="% Tồn kho (ước tính)", zorder=3)

for bar, val in zip(b1, customer_pct):
    ax2c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
              f"{val}%", ha="center", fontsize=22, fontweight="bold", color=DARK)
for bar, val in zip(b2, implied_inv):
    ax2c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
              f"{val}%", ha="center", fontsize=21, color=DARK)

ax2c.set_xticks(x3)
ax2c.set_xticklabels(regions, fontsize=24)
ax2c.set_ylabel("% Phân bổ", fontsize=26)
ax2c.set_ylim(0, 68)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax2c.set_title("2C · Khách hàng miền Đông cao —\ntồn kho phân bổ ngược chiều", fontsize=32, fontweight="bold", pad=10)
ax2c.legend(fontsize=22)

save_fig(fig2, "insight2.png")


# ═════════════════════════════════════════════════════════════════════════════
# INSIGHT 3 — 322 SKU hết hàng, hệ thống không bổ sung tồn kho
# ═════════════════════════════════════════════════════════════════════════════
print("Đang vẽ Insight 3...")

seg_names   = ["Balanced", "Everyday", "Performance", "All-weather",
               "Standard", "Activewear", "Premium", "Trendy"]
avg_prices2 = [9230, 7549, 6573, 3865, 2929, 2598, 2388, 2213]
margins2    = [25.8, 23.6, 26.4, 28.4, 31.3, 26.6, 28.5, 24.1]

rev_at_risk = [round(p * m / 100) for p, m in zip(avg_prices2, margins2)]
sort_idx    = np.argsort(rev_at_risk)[::-1]
seg_s       = [seg_names[i] for i in sort_idx]
rev_s       = [rev_at_risk[i] for i in sort_idx]
colors10    = [RED if v > 1800 else AMBER if v > 1000 else GRAY for v in rev_s]

channels2  = ["Tìm kiếm tự nhiên", "Mạng xã hội", "Tìm kiếm trả phí",
              "Email", "Giới thiệu", "Trực tiếp"]
cust_pct2  = [29.9, 20.1, 19.9, 12.0, 10.1, 8.0]

fig3, axes3 = plt.subplots(1, 2, figsize=(28, 12))
fig3.subplots_adjust(wspace=0.55)
# GIẢM khoảng trắng: chỉnh y=0.96 và tăng fontsize=40
fig3.suptitle("HỆ THỐNG KHÔNG BỔ SUNG TỒN KHO",
              fontsize=40, fontweight="bold", y=0.96)

# ── 3A: Giá trị biên LN mất trên mỗi SKU hết hàng ─────────────────────────
ax3a = axes3[0]
bars = ax3a.barh(seg_s[::-1], rev_s[::-1], color=colors10[::-1], zorder=3, height=0.6)
for bar, val in zip(bars, rev_s[::-1]):
    ax3a.text(val + 60, bar.get_y() + bar.get_height() / 2,
              f"{val:,} VND", va="center", fontsize=22, color=DARK)
ax3a.set_xlabel("Giá trị biên lợi nhuận mất trên mỗi SKU hết hàng (VND)", fontsize=26)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax3a.set_title("3A · Rủi ro doanh thu theo\nphân khúc khi hết hàng", fontsize=32, fontweight="bold", pad=10)
ax3a.set_xlim(0, 3600)
ax3a.axvline(1800, color=RED, linestyle=":", linewidth=1.5)
ax3a.tick_params(axis='y', labelsize=24)

# ── 3B: Khách hàng theo kênh thu hút vs phản hồi bổ sung = 0 ───────────────
ax3b = axes3[1]
x4    = np.arange(len(channels2))
width = 0.45

b_cust = ax3b.bar(x4, cust_pct2, width,
                   color=[RED if p > 20 else AMBER if p > 15 else GRAY for p in cust_pct2],
                   label="% Khách hàng theo kênh", zorder=3)

for bar, val in zip(b_cust, cust_pct2):
    ax3b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
              f"{val}%", ha="center", fontsize=22, fontweight="bold", color=DARK)

# Dòng phản hồi bổ sung = 0 — hiển thị ở đáy cột
ax3b.axhline(0, color=DARK, linestyle="-", linewidth=3, zorder=4, alpha=0.8,
             label="Phản hồi bổ sung tồn kho = 0")

ax3b.set_xticks(x4)
ax3b.set_xticklabels(channels2, rotation=30, ha="right", fontsize=24)
ax3b.set_ylabel("% Khách hàng", fontsize=26)
ax3b.set_ylim(0, 45)
# GIẢM khoảng trắng: chỉnh pad=10, tăng fontsize=32
ax3b.set_title("3B · Khách hàng theo kênh thu hút —\nkhông có phản hồi bổ sung", fontsize=32, fontweight="bold", pad=10)
ax3b.legend(fontsize=22, loc="upper right")

save_fig(fig3, "insight3.png")


print("\nHoàn tất! Đã lưu 3 ảnh vào thư mục visualization/:")
print("   insight1.png")
print("   insight2.png")
print("   insight3.png")