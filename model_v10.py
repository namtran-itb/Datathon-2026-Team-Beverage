"""
Phiên bản tạo file submission_v10.csv
- Phân rã chuỗi thời gian bằng Tháng x Ngày trong tuần (Month x DOW) để giảm nhiễu.
- Tính tốc độ tăng trưởng (CAGR) riêng cho 2 nửa năm (H1 và H2).
- Sử dụng tỷ lệ COGS/Revenue để mô hình dự đoán ổn định hơn.
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

# Cấu hình thư mục và thời gian
DATA_DIR = 'data/'
OUTPUT_DIR = 'outputs_v10/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_END = pd.Timestamp('2022-12-31')
TEST_START = pd.Timestamp('2023-01-01')
TEST_END = pd.Timestamp('2024-07-01')
SEED = 42
np.random.seed(SEED)

# Dữ liệu ngày Tết qua các năm
TET = {2013:('2013-02-10','2013-02-17'), 2014:('2014-01-31','2014-02-07'),
       2015:('2015-02-19','2015-02-26'), 2016:('2016-02-08','2016-02-15'),
       2017:('2017-01-28','2017-02-04'), 2018:('2018-02-16','2018-02-23'),
       2019:('2019-02-05','2019-02-12'), 2020:('2020-01-25','2020-02-01'),
       2021:('2021-02-12','2021-02-19'), 2022:('2022-02-01','2022-02-08'),
       2023:('2023-01-22','2023-01-29'), 2024:('2024-02-10','2024-02-17')}

FEATS = [
    'month','day','dayofweek','dayofyear','weekofyear',
    'is_weekend','is_holiday','is_payday',
    'promo','discount',
    'sin_1','cos_1','sin_2','cos_2','sin_dow','cos_dow',
    'is_tet','pre_tet',
]


def load_data():
    sales = pd.read_csv(DATA_DIR+'sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    promos = pd.read_csv(DATA_DIR+'promotions.csv', parse_dates=['start_date','end_date'])
    sample = pd.read_csv(DATA_DIR+'sample_submission.csv', parse_dates=['Date'])
    return sales, promos, sample

def get_discount(ngay_check, df_promo):
    # Đưa về bộ (tháng, ngày)
    ngay_check = pd.to_datetime(ngay_check)
    curr = (ngay_check.month, ngay_check.day)
    is_odd_year = ngay_check.year % 2 == 1

    count_promo = 0
    sum_discount = 0
   
    for _, row in df_promo.iterrows():
        if row['odd'] == True and not is_odd_year: continue

        start = (row['start_date'].month, row['start_date'].day)
        end = (row['end_date'].month, row['end_date'].day)

        is_in_range = False
       
        # Xử lý trường hợp vắt qua năm mới (vd: 25/12 - 05/01)
        if start <= end:
            if start <= curr <= end:
                is_in_range = True
        else: # Trường hợp vắt năm
            if curr >= start or curr <= end:
                is_in_range = True

        if is_in_range:
            count_promo += 1
            sum_discount += row['discount_value']
    return count_promo, sum_discount


def build_features(df, promos):
    # Tạo các đặc trưng thời gian cơ bản
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_payday'] = df['day'].isin([1,2,5,15,25,30]).astype(int)

    # Lag features
    for lag in [7, 14, 28, 364]:
        df[f'rev_lag_{lag}']  = df['Revenue'].shift(lag)
        df[f'cogs_lag_{lag}'] = df['COGS'].shift(lag)

    for w in [7, 14, 30]:
        df[f'rev_roll_mean_{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).mean()
        df[f'rev_roll_std_{w}']  = df['Revenue'].shift(1).rolling(w, min_periods=1).std().fillna(0)

    df['quarter'] = df['month'].map({1:1,2:1,3:1, 4:2,5:2,6:2,
                                    7:3,8:3,9:3, 10:4,11:4,12:4})

    # Ngày lễ Việt Nam
    vn_hol = [(1,1),(4,30),(5,1),(9,2),(12,25),(12,31)]
    df['is_holiday'] = df.apply(lambda x: 1 if (int(x.month), int(x.day)) in vn_hol else 0, axis=1)

    # Mã hóa chu kỳ thời gian (sin/cos)
    for k, p in enumerate([365.25, 182.625], 1):
        df[f'sin_{k}'] = np.sin(2 * np.pi * df['dayofyear'] / p)
        df[f'cos_{k}'] = np.cos(2 * np.pi * df['dayofyear'] / p)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Xử lý hiệu ứng trước và trong Tết
    df['is_tet'] = 0
    df['dt_tet'] = 999
    for yr, (s, e) in TET.items():
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        df.loc[(df['Date'] >= s_ts) & (df['Date'] <= e_ts), 'is_tet'] = 1
        pre = (df['Date'] >= s_ts - pd.Timedelta(days=30)) & (df['Date'] < s_ts)
        df.loc[pre, 'dt_tet'] = (s_ts - df.loc[pre, 'Date']).dt.days
    df['pre_tet'] = np.where(df['dt_tet'] < 30, np.exp(-df['dt_tet'] / 10), 0)

    # Xử lý khuyến mãi
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date'] = pd.to_datetime(promos['end_date'])

    promos['promo_name'] = promos['promo_name'].str.replace(r'\d+', '', regex=True)
    promos['odd'] = promos['promo_name'].str.strip().isin(['Rural Special', 'Urban Blowout'])

    # 2. Lấy quy luật cố định (Ngày, Tháng) từ lần xuất hiện gần nhất của mỗi đợt sale
    # Ta dùng .drop_duplicates để mỗi loại promo chỉ xuất hiện 1 dòng đại diện
    templates = promos.sort_values('start_date').drop_duplicates('promo_name', keep='last').copy()
    templates.head(10)

    promo = templates[['promo_name', 'start_date', 'end_date', 'discount_value', 'odd']]
    df[['promo', 'discount']] = df['Date'].apply(lambda x: pd.Series(get_discount(x, promo)))
   
    return df


def build_decomposition_v10(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24):
    # Lấy trung bình Doanh thu và COGS theo năm
    yr_r = train_sub.groupby('year')['Revenue'].mean().to_dict()
    yr_c = train_sub.groupby('year')['COGS'].mean().to_dict()
    mx = max(yr_r.keys())
    rb, cb = yr_r[mx], yr_c[mx]

    # Cài đặt tốc độ tăng trưởng chia theo 2 nửa năm (H1 và H2)
    month_rev_scale = {}
    month_cogs_scale = {}

    # Nửa năm đầu (Tháng 1-6)
    for m in range(1, 7):
        month_rev_scale[(2023, m)] = (1 + cagr_h1_23)
        month_rev_scale[(2024, m)] = (1 + cagr_h1_23) * (1 + cagr_h1_24)

    # Nửa năm sau (Tháng 7-12)
    for m in range(7, 13):
        month_rev_scale[(2023, m)] = (1 + cagr_h2_23)
        month_rev_scale[(2024, m)] = (1 + cagr_h2_23) * (1 + cagr_h2_24)

    # Áp dụng cho COGS tương tự Revenue
    month_cogs_scale = {k: v for k, v in month_rev_scale.items()}

    # Gán hệ số quy mô cho từng dòng
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

    # Tính yếu tố mùa vụ theo Tháng x Ngày trong tuần (ổn định hơn tính theo DOY)
    tr = train_sub.copy()
    tr['rts_tr'] = tr['year'].map(yr_r) / rb
    tr['cts_tr'] = tr['year'].map(yr_c) / cb
    tr['nr'] = tr['Revenue'] / tr['rts_tr']
    tr['nc'] = tr['COGS'] / tr['cts_tr']

    # Trung bình theo Month x DOW (84 cụm)
    mdow_r = tr.groupby(['month', 'dayofweek'])['nr'].mean().reset_index()
    mdow_r.columns = ['month', 'dayofweek', 'seasonal_rev']
    mdow_c = tr.groupby(['month', 'dayofweek'])['nc'].mean().reset_index()
    mdow_c.columns = ['month', 'dayofweek', 'seasonal_cogs']

    df = df.merge(mdow_r, on=['month', 'dayofweek'], how='left')
    df = df.merge(mdow_c, on=['month', 'dayofweek'], how='left')
    df['seasonal_rev'] = df['seasonal_rev'].ffill().bfill()
    df['seasonal_cogs'] = df['seasonal_cogs'].ffill().bfill()

    # Tính yếu tố mùa vụ bổ sung theo DOY để giữ được chi tiết ngày
    dr = tr.groupby('dayofyear')['nr'].mean().reset_index()
    dr['doy_rev'] = dr['nr'].rolling(7, min_periods=1, center=True).mean()
    dc = tr.groupby('dayofyear')['nc'].mean().reset_index()
    dc['doy_cogs'] = dc['nc'].rolling(7, min_periods=1, center=True).mean()

    df = df.merge(dr[['dayofyear', 'doy_rev']], on='dayofyear', how='left')
    df = df.merge(dc[['dayofyear', 'doy_cogs']], on='dayofyear', how='left')
    df['doy_rev'] = df['doy_rev'].ffill().bfill()
    df['doy_cogs'] = df['doy_cogs'].ffill().bfill()

    # Kết hợp: 60% từ Month x DOW (ổn định) + 40% từ DOY (để giữ xu hướng ngày cụ thể)
    df['blended_rev'] = 0.60 * df['seasonal_rev'] + 0.40 * df['doy_rev']
    df['blended_cogs'] = 0.60 * df['seasonal_cogs'] + 0.40 * df['doy_cogs']

    df['expected_revenue'] = df['blended_rev'] * df['rts']
    df['expected_cogs'] = df['blended_cogs'] * df['cts']

    # Tính log tỷ số làm mục tiêu huấn luyện
    eps = 1e-6
    df['log_rev_ratio'] = np.log((df['Revenue'] + eps) / (df['expected_revenue'] + eps))
    df['log_cogs_ratio'] = np.log((df['COGS'] + eps) / (df['expected_cogs'] + eps))

    return df


def train_with_cogs_ratio(tr_df, te_df):
    X_tr, X_te = tr_df[FEATS], te_df[FEATS]

    # Cấu hình Ensemble dự đoán Revenue
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
    w = [0.40, 0.35, 0.25]

    # Dự đoán Revenue
    pr = []
    for name, cls, params in configs_rev:
        mr = cls(**params)
        mr.fit(X_tr, tr_df['log_rev_ratio'])
        pr.append(mr.predict(X_te))
    ens_r = sum(wi * pi for wi, pi in zip(w, pr))

    # Dự đoán COGS theo lô logic tương tự Revenue
    pc = []
    for name, cls, params in configs_rev:
        cp = {**params, 'max_depth': min(params.get('max_depth', 3), 3)}
        if 'num_leaves' in params: cp['num_leaves'] = min(params['num_leaves'], 15)
        mc = cls(**cp)
        mc.fit(X_tr, tr_df['log_cogs_ratio'])
        pc.append(mc.predict(X_te))
    ens_c = sum(wi * pi for wi, pi in zip(w, pc))

    # Huấn luyện thêm mô hình dự đoán tỷ lệ COGS / Revenue để blend
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

    # Tạo dataframe khung gồm tất cả các ngày
    all_dates = pd.DataFrame({'Date': pd.date_range(sales.Date.min(), TEST_END, freq='D')})
    df = all_dates.merge(sales[['Date', 'Revenue', 'COGS']], on='Date', how='left')

    df = build_features(df, promos)
    train_sub = df[df['Date'] <= TRAIN_END].dropna(subset=['Revenue']).copy()

    df = build_decomposition_v10(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24)

    tr_df = df[df['Date'] <= TRAIN_END].dropna(subset=['log_rev_ratio'])
    te_df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)].copy()

    ens_r, ens_c, pred_ratio = train_with_cogs_ratio(tr_df, te_df)

    te_df = te_df.copy()
    # Phục hồi giá trị Revenue thực
    te_df['Revenue'] = np.clip(te_df['expected_revenue'].values * np.exp(ens_r), 0, None)

    # Phục hồi giá trị COGS, kết hợp mô hình độc lập và mô hình tỷ lệ
    cogs_independent = np.clip(te_df['expected_cogs'].values * np.exp(ens_c), 0, None)
    cogs_from_ratio = te_df['Revenue'].values * np.clip(pred_ratio, 0.70, 0.95)

    if use_cogs_ratio:
        te_df['COGS'] = (1 - ratio_blend) * cogs_independent + ratio_blend * cogs_from_ratio
    else:
        te_df['COGS'] = cogs_independent

    # Bước 1: Đo tỷ lệ thực từ train
    n_inverted = (train_sub['COGS'] >= train_sub['Revenue']).sum()
    pct_inverted = n_inverted / len(train_sub)
    print(f"Train: {pct_inverted:.1%} ngày COGS >= Revenue")  # expect ~10%

    # Bước 2: Học margin band — cho phép vượt 1.0
    margin_bounds = (
        train_sub.assign(margin = train_sub['COGS'] / (train_sub['Revenue'] + 1e-6))
        .groupby('month')['margin']
        .quantile([0.02, 0.98])
        .unstack()
    )

    # Bước 3: Áp ràng buộc học từ data, không hardcode
    for m in range(1, 13):
        mask = te_df['month'] == m
        lo = margin_bounds.loc[m, 0.02] if m in margin_bounds.index else 0.55
        hi = margin_bounds.loc[m, 0.98] if m in margin_bounds.index else 1.10  # cho phép > 1
        te_df.loc[mask, 'COGS'] = te_df.loc[mask, 'COGS'].clip(
            te_df.loc[mask, 'Revenue'] * lo,
            te_df.loc[mask, 'Revenue'] * hi
        )

    # Bước 4: Kiểm tra kết quả — tỷ lệ phải gần ~10%
    n_test_inverted = (te_df['COGS'] >= te_df['Revenue']).sum()
    print(f"Test:  {n_test_inverted/len(te_df):.1%} ngày COGS >= Revenue")
    print(f"Chênh lệch so với train: {abs(n_test_inverted/len(te_df) - pct_inverted):.1%}")

    te_df['Revenue'] = te_df['Revenue'].clip(lower=0).round(2)
    te_df['COGS'] = te_df['COGS'].clip(lower=0).round(2)

    # Lọc dữ liệu khớp với mẫu submission
    sub = te_df[['Date', 'Revenue', 'COGS']].copy()
    sub = sub[sub['Date'].isin(sample['Date'])].sort_values('Date').reset_index(drop=True)
    sub['Date'] = sub['Date'].dt.strftime('%Y-%m-%d')

    # Ghi file với tên mới là submission_v10.csv
    fn = 'submission_k10.csv'
    sub.to_csv(os.path.join(OUTPUT_DIR, fn), index=False)

    margin = ((sub.Revenue - sub.COGS) / sub.Revenue * 100)
    print(f"[{fn}]")
    print(f"  Rev={sub.Revenue.mean():,.0f} | COGS={sub.COGS.mean():,.0f} | Margin={margin.mean():.1f}%")
    print(f"  H1-23={cagr_h1_23:.0%} H2-23={cagr_h2_23:.0%} H1-24={cagr_h1_24:.0%} H2-24={cagr_h2_24:.0%}\n")


def main():
    print("=" * 60)
    print("  TẠO FILE SUBMISSION V10".center(60))
    print("=" * 60)

    sales, promos, sample = load_data()

    print("Khởi chạy model dự báo...\n")
    # Tốc độ tăng trưởng trọng tâm H2 và dự báo COGS bằng tỷ lệ kết hợp
    make_submission(
        sales, promos, sample,
        cagr_h1_23=0.24, cagr_h2_23=0.32,
        cagr_h1_24=0.02, cagr_h2_24=0.05,
        use_cogs_ratio=True,
        ratio_blend=0.3
    )

    print("Hoàn tất! File đã được lưu tại thư mục outputs_v10/.")

if __name__ == '__main__':
    main()
