import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

DIR = 'data_cleaned_final/'


def build_master_pipeline():

    # 1. ĐỌC DỮ LIỆU
    sales = pd.read_csv(DIR + 'sales.csv', parse_dates=['Date'])
    promos = pd.read_csv(DIR + 'promotions.csv', parse_dates=['start_date', 'end_date'])
    traffic = pd.read_csv(DIR + 'web_traffic.csv', parse_dates=['date'])
    sub_df = pd.read_csv(DIR + 'sample_submission.csv', parse_dates=['Date'])

    # BẢO MẬT 1: Vứt bỏ các con số ảo của Ban tổ chức để tránh lỗi "0 dòng"
    sub_df = sub_df[['Date']]

    # 2. XỬ LÝ LƯU LƯỢNG WEB LỊCH SỬ (An toàn cho tương lai)
    traffic['dow'] = traffic['date'].dt.dayofweek
    hist_traffic = traffic.groupby('dow')['sessions'].mean().reset_index(name='avg_sessions')

    # Gộp khung thời gian chuẩn
    full_ts = pd.concat([sales[['Date', 'Revenue', 'COGS']], sub_df], axis=0).sort_values('Date').reset_index(drop=True)

    # 3. KỸ THUẬT ĐẶC TRƯNG TƯƠNG LAI (Chỉ dùng những gì biết trước)
    df = full_ts.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Chu kỳ xoay vòng
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lễ Tết & Ngày lương
    vn_holidays = [(1, 1), (4, 30), (5, 1), (9, 2), (12, 25), (12, 31)]
    df['is_holiday'] = df.apply(lambda x: 1 if (x['month'], x['day']) in vn_holidays else 0, axis=1)
    df['is_payday'] = df['day'].apply(lambda x: 1 if x in [1, 5, 15, 25, 30] else 0)

    # Ghép Khuyến mãi
    df['promo_active'] = 0
    df['max_discount'] = 0.0
    valid_promos = promos.dropna(subset=['start_date', 'end_date'])
    for _, row in valid_promos.iterrows():
        mask = (df['Date'] >= row['start_date']) & (df['Date'] <= row['end_date'])
        df.loc[mask, 'promo_active'] += 1
        if 'discount_value' in row and pd.notnull(row['discount_value']):
            df.loc[mask, 'max_discount'] = np.maximum(df.loc[mask, 'max_discount'], row['discount_value'])

    # Ghép Traffic trung bình
    df = df.merge(hist_traffic, left_on='dayofweek', right_on='dow', how='left').drop(columns=['dow'])

    # 4. CHIA TẬP TRAIN / TEST CHÍNH XÁC BẰNG THỜI GIAN
    train = df[df['Date'] < '2023-01-01'].copy()
    test = df[df['Date'] >= '2023-01-01'].copy()

    # 5. KÝ ỨC LỊCH SỬ (TARGET ENCODING) - Vũ khí chủ lực
    hist_mean = train.groupby(['month', 'dayofweek'])[['Revenue', 'COGS']].mean().reset_index()
    hist_mean.rename(columns={'Revenue': 'hist_rev', 'COGS': 'hist_cogs'}, inplace=True)

    train = train.merge(hist_mean, on=['month', 'dayofweek'], how='left')
    test = test.merge(hist_mean, on=['month', 'dayofweek'], how='left')

    features = [
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'month_sin', 'month_cos',
        'is_weekend', 'is_holiday', 'is_payday', 'promo_active', 'max_discount',
        'avg_sessions', 'hist_rev', 'hist_cogs'
    ]

    # 6. HUẤN LUYỆN XGBOOST
    params = {
        'n_estimators': 2500,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1
    }

    m_rev = xgb.XGBRegressor(**params)
    m_cogs = xgb.XGBRegressor(**params)

    print("--- Đang huấn luyện Cỗ máy Dự báo... ---")
    m_rev.fit(train[features], train['Revenue'])
    m_cogs.fit(train[features], train['COGS'])

    print(f"--- Đang dự báo cho {len(test)} ngày tương lai... ---")
    test['Revenue'] = m_rev.predict(test[features]).clip(0).round(2)
    test['COGS'] = m_cogs.predict(test[features]).clip(0).round(2)

    # 7. XUẤT FILE
    submission = test[['Date', 'Revenue', 'COGS']]
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
    submission.to_csv('submission_05.csv', index=False)

    print("\n THÀNH CÔNG! Đã tạo file: submission_05.csv")


if __name__ == "__main__":
    build_master_pipeline()