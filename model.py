"""
Script dự đoán doanh thu và giá vốn — Datathon 2026

Phương pháp chính:
  - Phân rã chuỗi thời gian theo Tháng × Ngày trong tuần (Month × DOW) để lấy mùa vụ ổn định.
  - Tính tốc độ tăng trưởng (CAGR) riêng cho từng nửa năm (H1: tháng 1-6, H2: tháng 7-12).
  - Dự đoán tỷ lệ COGS/Revenue bằng mô hình riêng, rồi kết hợp với dự đoán COGS độc lập.
  - Xuất file submission.csv theo đúng mẫu nộp bài.
"""

import os, sys, warnings
import numpy as np, pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ─────────────────────────────────────────────
# Cấu hình chung
# ─────────────────────────────────────────────
DATA_DIR  = 'data/'          # thư mục chứa dữ liệu gốc
OUTPUT_DIR = 'outputs/'      # thư mục xuất kết quả
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_END  = pd.Timestamp('2022-12-31')   # ngày cuối cùng của tập huấn luyện
TEST_START  = pd.Timestamp('2023-01-01')   # ngày bắt đầu dự đoán
TEST_END    = pd.Timestamp('2024-07-01')   # ngày kết thúc dự đoán
SEED = 42
np.random.seed(SEED)

# Ngày Tết Âm lịch từng năm — dùng để đánh dấu hiệu ứng Tết
TET = {2013:('2013-02-10','2013-02-17'), 2014:('2014-01-31','2014-02-07'),
       2015:('2015-02-19','2015-02-26'), 2016:('2016-02-08','2016-02-15'),
       2017:('2017-01-28','2017-02-04'), 2018:('2018-02-16','2018-02-23'),
       2019:('2019-02-05','2019-02-12'), 2020:('2020-01-25','2020-02-01'),
       2021:('2021-02-12','2021-02-19'), 2022:('2022-02-01','2022-02-08'),
       2023:('2023-01-22','2023-01-29'), 2024:('2024-02-10','2024-02-17')}

# Danh sách đặc trưng đưa vào mô hình
FEATS = [
    'month','day','dayofweek','dayofyear','weekofyear',
    'is_weekend','is_holiday','is_payday',
    'promo','discount',
    'sin_1','cos_1','sin_2','cos_2','sin_dow','cos_dow',
    'is_tet','pre_tet',
]


def load_data():
    """Đọc 3 file dữ liệu cần thiết: sales, promotions, sample_submission."""
    sales  = pd.read_csv(DATA_DIR+'sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    promos = pd.read_csv(DATA_DIR+'promotions.csv', parse_dates=['start_date','end_date'])
    sample = pd.read_csv(DATA_DIR+'sample_submission.csv', parse_dates=['Date'])
    return sales, promos, sample


def get_discount(ngay_check, df_promo):
    """
    Kiểm tra một ngày có rơi vào đợt khuyến mãi nào không.
    Trả về (số đợt khuyến mãi, tổng mức giảm giá).
    Xử lý luôn trường hợp đợt sale vắt qua năm mới (vd 25/12 → 05/01).
    """
    ngay_check = pd.to_datetime(ngay_check)
    curr = (ngay_check.month, ngay_check.day)
    is_odd_year = ngay_check.year % 2 == 1

    count_promo = 0
    sum_discount = 0

    for _, row in df_promo.iterrows():
        # bỏ qua đợt sale chỉ dành năm chẵn nếu đang xét năm lẻ, và ngược lại
        if row['odd'] == True and not is_odd_year: continue

        start = (row['start_date'].month, row['start_date'].day)
        end   = (row['end_date'].month, row['end_date'].day)

        is_in_range = False

        # trường hợp bình thường: ngày bắt đầu <= ngày kết thúc trong cùng năm
        if start <= end:
            if start <= curr <= end:
                is_in_range = True
        else:
            # trường hợp vắt năm: vd từ 25/12 đến 05/01 — ngày đó nằm trước hoặc sau đều tính
            if curr >= start or curr <= end:
                is_in_range = True

        if is_in_range:
            count_promo += 1
            sum_discount += row['discount_value']
    return count_promo, sum_discount


def build_features(df, promos):
    """
    Tạo toàn bộ đặc trưng (features) cho mô hình:
      - Thời gian: tháng, ngày, thứ trong tuần, quý, cuối tuần, ngày lương...
      - Lag & rolling: doanh thu/giá vốn các ngày trước đó
      - Mùa vụ dạng sin/cos: mã hóa chu kỳ năm và chu kỳ tuần
      - Hiệu ứng Tết: đánh dấu ngày Tết và giai đoạn trước Tết
      - Khuyến mãi: số đợt sale và tổng giảm giá trong ngày
    """
    # đặc trưng thời gian cơ bản
    df['year']      = df['Date'].dt.year
    df['month']     = df['Date'].dt.month
    df['day']       = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear']= df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend']= (df['dayofweek'] >= 5).astype(int)
    df['is_payday'] = df['day'].isin([1,2,5,15,25,30]).astype(int)

    # lag features — lấy giá trị Revenue/COGS của các ngày trước đó
    # lag 364 = cách đây 1 năm (364 = 52×7), giúp mô hình học mùa vụ năm
    for lag in [7, 14, 28, 364]:
        df[f'rev_lag_{lag}']  = df['Revenue'].shift(lag)
        df[f'cogs_lag_{lag}'] = df['COGS'].shift(lag)

    # rolling mean/std — trung bình và độ lệch doanh thu trong cửa sổ trượt
    # shift(1) để tránh rò rỉ dữ liệu tương lai vào huấn luyện
    for w in [7, 14, 30]:
        df[f'rev_roll_mean_{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).mean()
        df[f'rev_roll_std_{w}']  = df['Revenue'].shift(1).rolling(w, min_periods=1).std().fillna(0)

    # quý trong năm — dùng cho tính tăng trưởng theo nửa năm
    df['quarter'] = df['month'].map({1:1,2:1,3:1, 4:2,5:2,6:2,
                                    7:3,8:3,9:3, 10:4,11:4,12:4})

    # ngày lễ Việt Nam — các ngày nghỉ lớn ảnh hưởng đến doanh thu
    vn_hol = [(1,1),(4,30),(5,1),(9,2),(12,25),(12,31)]
    df['is_holiday'] = df.apply(lambda x: 1 if (int(x.month), int(x.day)) in vn_hol else 0, axis=1)

    # mã hóa chu kỳ thời gian bằng sin/cos
    # thay vì dùng dayofyear trực tiếp (1 và 365 xa nhau trên số nhưng gần nhau thực tế),
    # sin/cos giúp mô hình hiểu được tính tuần hoàn
    for k, p in enumerate([365.25, 182.625], 1):  # chu kỳ 1 năm và nửa năm
        df[f'sin_{k}'] = np.sin(2 * np.pi * df['dayofyear'] / p)
        df[f'cos_{k}'] = np.cos(2 * np.pi * df['dayofyear'] / p)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)   # chu kỳ tuần
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # hiệu ứng Tết — doanh thu thường tăng mạnh trước Tết rồi giảm trong ngày Tết
    # is_tet: đánh dấu ngày trong tuần Tết
    # pre_tet: hàm mũ giảm dần khi tiến gần đến Tết (30 ngày trước)
    df['is_tet'] = 0
    df['dt_tet'] = 999
    for yr, (s, e) in TET.items():
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        df.loc[(df['Date'] >= s_ts) & (df['Date'] <= e_ts), 'is_tet'] = 1
        pre = (df['Date'] >= s_ts - pd.Timedelta(days=30)) & (df['Date'] < s_ts)
        df.loc[pre, 'dt_tet'] = (s_ts - df.loc[pre, 'Date']).dt.days
    df['pre_tet'] = np.where(df['dt_tet'] < 30, np.exp(-df['dt_tet'] / 10), 0)

    # xử lý khuyến mãi — chuẩn hóa tên promo và lấy quy luật cố định từ lần sale gần nhất
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date']   = pd.to_datetime(promos['end_date'])

    # bỏ số trong tên promo để gom các đợt cùng loại (vd "Summer Sale 2022" → "Summer Sale")
    promos['promo_name'] = promos['promo_name'].str.replace(r'\d+', '', regex=True)
    # đánh dấu đợt sale chỉ diễn ra vào năm lẻ (Rural Special, Urban Blowout)
    promos['odd'] = promos['promo_name'].str.strip().isin(['Rural Special', 'Urban Blowout'])

    # mỗi loại promo chỉ giữ 1 dòng đại diện (lần xuất hiện gần nhất)
    # vì quy luật khuyến mãi lặp lại theo (tháng, ngày) chứ không theo năm
    templates = promos.sort_values('start_date').drop_duplicates('promo_name', keep='last').copy()
    templates.head(10)

    promo = templates[['promo_name', 'start_date', 'end_date', 'discount_value', 'odd']]
    df[['promo', 'discount']] = df['Date'].apply(lambda x: pd.Series(get_discount(x, promo)))

    return df


def build_decomposition(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24):
    """
    Phân rã chuỗi thời gian thành 3 thành phần: xu hướng (trend), mùa vụ (seasonal), phần dư (residual).

    Cách làm:
      1. Tính quy mô trung bình theo năm → hệ số điều chỉnh xu hướng (rts, cts).
      2. Tính tốc độ tăng trưởng CAGR riêng cho H1 và H2 — vì doanh thu nửa cuối năm
         thường tăng nhanh hơn nửa đầu, chia ra sẽ dự đoán chính xác hơn.
      3. Tính yếu tố mùa vụ theo Tháng × Ngày trong tuần (84 cụm) — ổn định hơn DOY
         vì DOY bị trượt do năm nhuận.
      4. Kết hợp 60% mùa vụ Month×DOW + 40% mùa vụ DOY (làm mượt bằng rolling 7 ngày)
         để vừa ổn định vừa giữ được chi tiết từng ngày.
      5. Tính giá trị kỳ vọng = mùa vụ × xu hướng, rồi lấy log tỷ số giữa thực tế và kỳ vọng
         làm mục tiêu huấn luyện — mô hình chỉ cần học phần dư (residual).
    """
    # trung bình doanh thu và giá vốn theo năm — dùng làm mốc quy mô
    yr_r = train_sub.groupby('year')['Revenue'].mean().to_dict()
    yr_c = train_sub.groupby('year')['COGS'].mean().to_dict()
    mx = max(yr_r.keys())         # năm gần nhất làm mốc chuẩn
    rb, cb = yr_r[mx], yr_c[mx]   # revenue base, cogs base

    # hệ số tăng trưởng theo nửa năm — áp cho các năm 2023 và 2024 (tập test)
    # H1 (tháng 1-6) và H2 (tháng 7-12) có tốc độ tăng khác nhau
    month_rev_scale = {}
    month_cogs_scale = {}

    # nửa năm đầu (tháng 1-6)
    for m in range(1, 7):
        month_rev_scale[(2023, m)] = (1 + cagr_h1_23)
        month_rev_scale[(2024, m)] = (1 + cagr_h1_23) * (1 + cagr_h1_24)

    # nửa năm sau (tháng 7-12)
    for m in range(7, 13):
        month_rev_scale[(2023, m)] = (1 + cagr_h2_23)
        month_rev_scale[(2024, m)] = (1 + cagr_h2_23) * (1 + cagr_h2_24)

    # COGS dùng cùng hệ số tăng với Revenue
    month_cogs_scale = {k: v for k, v in month_rev_scale.items()}

    # gán hệ số quy mô cho từng dòng dữ liệu
    # rts = revenue trend scale, cts = cogs trend scale
    df['rts'] = 1.0
    df['cts'] = 1.0
    for y in train_sub['year'].unique():
        mask = df['year'] == y
        df.loc[mask, 'rts'] = yr_r.get(y, rb) / rb
        df.loc[mask, 'cts'] = yr_c.get(y, cb) / cb
    for (y, m), scale in month_rev_scale.items():
        mask = (df['year'] == y) & (df['month'] == m)
        df.loc[mask, 'rts'] = scale
    for (y, m), scale in month_cogs_scale.items():
        mask = (df['year'] == y) & (df['month'] == m)
        df.loc[mask, 'cts'] = scale

    # tính yếu tố mùa vụ — trước hết phải bỏ xu hướng ra khỏi dữ liệu huấn luyện
    # nr = normalized revenue, nc = normalized cogs (đã chia cho hệ số quy mô năm)
    tr = train_sub.copy()
    tr['rts_tr'] = tr['year'].map(yr_r) / rb
    tr['cts_tr'] = tr['year'].map(yr_c) / cb
    tr['nr'] = tr['Revenue'] / tr['rts_tr']
    tr['nc'] = tr['COGS'] / tr['cts_tr']

    # mùa vụ theo Tháng × Ngày trong tuần — 12 tháng × 7 ngày = 84 cụm
    # ổn định hơn so với gom theo dayofyear vì không bị trượt do năm nhuận
    mdow_r = tr.groupby(['month', 'dayofweek'])['nr'].mean().reset_index()
    mdow_r.columns = ['month', 'dayofweek', 'seasonal_rev']
    mdow_c = tr.groupby(['month', 'dayofweek'])['nc'].mean().reset_index()
    mdow_c.columns = ['month', 'dayofweek', 'seasonal_cogs']

    df = df.merge(mdow_r, on=['month', 'dayofweek'], how='left')
    df = df.merge(mdow_c, on=['month', 'dayofweek'], how='left')
    df['seasonal_rev']  = df['seasonal_rev'].ffill().bfill()
    df['seasonal_cogs'] = df['seasonal_cogs'].ffill().bfill()

    # mùa vụ bổ sung theo dayofyear — làm mượt bằng rolling 7 ngày để giảm nhiễu
    # giữ lại chi tiết từng ngày mà Month×DOW không phản ánh được
    dr = tr.groupby('dayofyear')['nr'].mean().reset_index()
    dr['doy_rev'] = dr['nr'].rolling(7, min_periods=1, center=True).mean()
    dc = tr.groupby('dayofyear')['nc'].mean().reset_index()
    dc['doy_cogs'] = dc['nc'].rolling(7, min_periods=1, center=True).mean()

    df = df.merge(dr[['dayofyear', 'doy_rev']], on='dayofyear', how='left')
    df = df.merge(dc[['dayofyear', 'doy_cogs']], on='dayofyear', how='left')
    df['doy_rev']  = df['doy_rev'].ffill().bfill()
    df['doy_cogs'] = df['doy_cogs'].ffill().bfill()

    # kết hợp 2 nguồn mùa vụ: 60% Month×DOW (ổn định) + 40% DOY (chi tiết ngày)
    df['blended_rev']  = 0.60 * df['seasonal_rev'] + 0.40 * df['doy_rev']
    df['blended_cogs'] = 0.60 * df['seasonal_cogs'] + 0.40 * df['doy_cogs']

    # giá trị kỳ vọng = mùa vụ × xu hướng
    df['expected_revenue'] = df['blended_rev'] * df['rts']
    df['expected_cogs']    = df['blended_cogs'] * df['cts']

    # log tỷ số giữa thực tế và kỳ vọng — đây là mục tiêu (target) huấn luyện
    # mô hình chỉ cần học phần dư (residual), dễ hơn nhiều so với học nguyên chuỗi
    eps = 1e-6
    df['log_rev_ratio']  = np.log((df['Revenue'] + eps) / (df['expected_revenue'] + eps))
    df['log_cogs_ratio'] = np.log((df['COGS'] + eps) / (df['expected_cogs'] + eps))

    return df


def train_with_cogs_ratio(tr_df, te_df):
    """
    Huấn luyện ensemble 3 mô hình (2 XGBoost + 1 LightGBM) dự đoán phần dư Revenue và COGS.
    Đồng thời huấn luyện thêm mô hình dự đoán tỷ lệ COGS/Revenue để kết hợp kết quả.

    Trả về:
      - ens_r: dự đoán phần dư Revenue (log ratio)
      - ens_c: dự đoán phần dư COGS (log ratio)
      - pred_ratio: dự đoán tỷ lệ COGS/Revenue
    """
    X_tr, X_te = tr_df[FEATS], te_df[FEATS]

    # cấu hình 3 mô hình dự đoán Revenue — đa dạng để ensemble ổn định hơn
    configs_rev = [
        ('XGB1', xgb.XGBRegressor, dict(n_estimators=1000, learning_rate=0.025, max_depth=3, objective='reg:pseudohubererror',
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)),
        ('LGB', lgb.LGBMRegressor, dict(n_estimators=1000, learning_rate=0.025, max_depth=4, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_alpha=0.2, reg_lambda=1.0, random_state=42, verbose=-1)),
        ('XGB2', xgb.XGBRegressor, dict(n_estimators=1200, learning_rate=0.02, max_depth=4, objective='reg:pseudohubererror',
            subsample=0.75, colsample_bytree=0.75, min_child_weight=15,
            reg_alpha=0.2, reg_lambda=2.0, random_state=99, verbosity=0)),
    ]
    # trọng số ensemble — XGB1 nặng nhất vì max_depth=3 ít quá khớp nhất
    w = [0.40, 0.35, 0.25]

    # dự đoán phần dư Revenue — lấy trung bình có trọng số từ 3 mô hình
    pr = []
    for name, cls, params in configs_rev:
        mr = cls(**params)
        mr.fit(X_tr, tr_df['log_rev_ratio'])
        pr.append(mr.predict(X_te))
    ens_r = sum(wi * pi for wi, pi in zip(w, pr))

    # dự đoán phần dư COGS — dùng cùng cấu hình nhưng giảm độ phức tạp
    # (max_depth nhỏ hơn, num_leaves ít hơn) vì COGS biến động ít hơn Revenue
    pc = []
    for name, cls, params in configs_rev:
        cp = {**params, 'max_depth': min(params.get('max_depth', 3), 3)}
        if 'num_leaves' in params: cp['num_leaves'] = min(params['num_leaves'], 15)
        mc = cls(**cp)
        mc.fit(X_tr, tr_df['log_cogs_ratio'])
        pc.append(mc.predict(X_te))
    ens_c = sum(wi * pi for wi, pi in zip(w, pc))

    # mô hình phụ: dự đoán trực tiếp tỷ lệ COGS/Revenue
    # kết quả sẽ được kết hợp (blend) với dự đoán COGS độc lập để giảm sai số
    tr_ratio = tr_df['COGS'] / (tr_df['Revenue'] + 1e-6)
    ratio_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=0.2, reg_lambda=2.0, random_state=42, verbosity=0)
    ratio_model.fit(X_tr, tr_ratio)
    pred_ratio = ratio_model.predict(X_te)

    return ens_r, ens_c, pred_ratio


def make_submission(sales, promos, sample,
                    cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24,
                    use_cogs_ratio=False, ratio_blend=0.3):
    """
    Quy trình chính: từ dữ liệu gốc → tạo đặc trưng → phân rã → huấn luyện → xuất submission.

    Tham số:
      - cagr_h1_23, cagr_h2_23: tốc độ tăng trưởng H1/H2 năm 2023
      - cagr_h1_24, cagr_h2_24: tốc độ tăng trưởng H1/H2 năm 2024
      - use_cogs_ratio: có kết hợp mô hình tỷ lệ COGS/Revenue hay không
      - ratio_blend: trọng số kết hợp (0 = chỉ dùng COGS độc lập, 1 = chỉ dùng tỷ lệ)
    """

    # tạo khung dữ liệu gồm tất cả các ngày từ đầu đến cuối kỳ dự đoán
    all_dates = pd.DataFrame({'Date': pd.date_range(sales.Date.min(), TEST_END, freq='D')})
    df = all_dates.merge(sales[['Date', 'Revenue', 'COGS']], on='Date', how='left')

    # tạo đặc trưng và phân rã chuỗi thời gian
    df = build_features(df, promos)
    train_sub = df[df['Date'] <= TRAIN_END].dropna(subset=['Revenue']).copy()
    df = build_decomposition(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24)

    # chia tập huấn luyện và tập test
    tr_df = df[df['Date'] <= TRAIN_END].dropna(subset=['log_rev_ratio'])
    te_df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()

    # huấn luyện mô hình và dự đoán
    ens_r, ens_c, pred_ratio = train_with_cogs_ratio(tr_df, te_df)

    te_df = te_df.copy()

    # phục hồi giá trị Revenue thực từ phần dư: Revenue = kỳ vọng × exp(phần dư)
    te_df['Revenue'] = np.clip(te_df['expected_revenue'].values * np.exp(ens_r), 0, None)

    # phục hồi COGS — kết hợp 2 nguồn:
    #   cogs_independent: từ mô hình dự đoán COGS độc lập
    #   cogs_from_ratio: từ Revenue × tỷ lệ COGS/Revenue dự đoán
    cogs_independent = np.clip(te_df['expected_cogs'].values * np.exp(ens_c), 0, None)
    cogs_from_ratio  = te_df['Revenue'].values * np.clip(pred_ratio, 0.70, 0.95)

    if use_cogs_ratio:
        te_df['COGS'] = (1 - ratio_blend) * cogs_independent + ratio_blend * cogs_from_ratio
    else:
        te_df['COGS'] = cogs_independent

    # ── ràng buộc biên lợi nhuận ──
    # thực tế có khoảng 10% ngày COGS >= Revenue, không được hardcode capping cứng
    # mà phải học biên margin từ dữ liệu huấn luyện rồi áp lên tập test

    # bước 1: đo tỷ lệ COGS >= Revenue trong tập huấn luyện
    n_inverted = (train_sub['COGS'] >= train_sub['Revenue']).sum()
    pct_inverted = n_inverted / len(train_sub)
    print(f"Train: {pct_inverted:.1%} ngày COGS >= Revenue")

    # bước 2: học biên margin theo tháng — lấy phân vị 2% và 98% để cho phép giá trị cực
    margin_bounds = (
        train_sub.assign(margin = train_sub['COGS'] / (train_sub['Revenue'] + 1e-6))
        .groupby('month')['margin']
        .quantile([0.02, 0.98])
        .unstack()
    )

    # bước 3: áp ràng buộc — COGS phải nằm trong biên margin học được, không hardcode
    for m in range(1, 13):
        mask = te_df['month'] == m
        lo = margin_bounds.loc[m, 0.02] if m in margin_bounds.index else 0.55
        hi = margin_bounds.loc[m, 0.98] if m in margin_bounds.index else 1.10   # cho phép > 1
        te_df.loc[mask, 'COGS'] = te_df.loc[mask, 'COGS'].clip(
            te_df.loc[mask, 'Revenue'] * lo,
            te_df.loc[mask, 'Revenue'] * hi
        )

    # bước 4: kiểm tra — tỷ lệ COGS >= Revenue ở tập test phải gần tỷ lệ ở tập train
    n_test_inverted = (te_df['COGS'] >= te_df['Revenue']).sum()
    print(f"Test:  {n_test_inverted/len(te_df):.1%} ngày COGS >= Revenue")
    print(f"Chênh lệch so với train: {abs(n_test_inverted/len(te_df) - pct_inverted):.1%}")

    te_df['Revenue'] = te_df['Revenue'].clip(lower=0).round(2)
    te_df['COGS']    = te_df['COGS'].clip(lower=0).round(2)

    # lọc dữ liệu khớp với mẫu submission — chỉ giữ các ngày có trong sample_submission
    sub = te_df[['Date', 'Revenue', 'COGS']].copy()
    sub = sub[sub['Date'].isin(sample['Date'])].sort_values('Date').reset_index(drop=True)
    sub['Date'] = sub['Date'].dt.strftime('%Y-%m-%d')

    # ghi file submission.csv
    fn = 'submission.csv'
    sub.to_csv(os.path.join(OUTPUT_DIR, fn), index=False)

    # in thống kê nhanh để kiểm tra
    margin = ((sub.Revenue - sub.COGS) / sub.Revenue * 100)
    print(f"[{fn}]")
    print(f"  Rev={sub.Revenue.mean():,.0f} | COGS={sub.COGS.mean():,.0f} | Margin={margin.mean():.1f}%")
    print(f"  H1-23={cagr_h1_23:.0%} H2-23={cagr_h2_23:.0%} H1-24={cagr_h1_24:.0%} H2-24={cagr_h2_24:.0%}\n")


def main():
    print("=" * 60)
    print("  DỰ ĐOÁN DOANH THU & GIÁ VỐN — DATATHON 2026".center(60))
    print("=" * 60)

    sales, promos, sample = load_data()

    print("Khởi chạy model dự báo...\n")
    # dùng tốc độ tăng trưởng riêng cho H2 (cao hơn H1)
    # và kết hợp mô hình tỷ lệ COGS/Revenue để dự đoán COGS ổn định hơn
    make_submission(
        sales, promos, sample,
        cagr_h1_23=0.24, cagr_h2_23=0.32,
        cagr_h1_24=0.02, cagr_h2_24=0.05,
        use_cogs_ratio=True,
        ratio_blend=0.3
    )

    print("Hoàn tất! File đã được lưu tại thư mục outputs/.")

if __name__ == '__main__':
    main()
