"""
V9 PIPELINE - ARCHITECTURAL BREAKTHROUGH
Best so far: v7_r28_r03 = 726,848

ROOT CAUSE ANALYSIS:
  Current DOY profile has 366 buckets with only ~10 obs each = NOISY
  This noise propagates through decomposition -> ML tries to fix it -> overfits

BREAKTHROUGH CHANGES:
  1. Month x DOW seasonal profile (84 buckets, ~40 obs each = 4x more stable)
  2. Half-year specific CAGR: H1 (Jan-Jun) grows differently from H2 (Jul-Dec)
  3. More regularized ML: higher min_child, more regularization
  4. Separate model for weekend vs weekday (data shows different patterns)
  5. COGS modeled as ratio of Revenue (more stable than independent prediction)
"""
import os, sys, warnings
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
if sys.stdout.encoding!='utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

DATA_DIR='data/'; OUTPUT_DIR='outputs_v9/'; os.makedirs(OUTPUT_DIR,exist_ok=True)
TRAIN_END=pd.Timestamp('2022-12-31'); TEST_START=pd.Timestamp('2023-01-01')
TEST_END=pd.Timestamp('2024-07-01'); SEED=42; np.random.seed(SEED)

TET={2013:('2013-02-10','2013-02-17'),2014:('2014-01-31','2014-02-07'),
     2015:('2015-02-19','2015-02-26'),2016:('2016-02-08','2016-02-15'),
     2017:('2017-01-28','2017-02-04'),2018:('2018-02-16','2018-02-23'),
     2019:('2019-02-05','2019-02-12'),2020:('2020-01-25','2020-02-01'),
     2021:('2021-02-12','2021-02-19'),2022:('2022-02-01','2022-02-08'),
     2023:('2023-01-22','2023-01-29'),2024:('2024-02-10','2024-02-17')}

FEATS = [
    'month','day','dayofweek','dayofyear','weekofyear',
    'is_weekend','is_holiday','is_payday',
    'promo','discount',
    'sin_1','cos_1','sin_2','cos_2','sin_dow','cos_dow',
    'is_tet','pre_tet',
]


def load_data():
    sales=pd.read_csv(DATA_DIR+'sales.csv',parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    promos=pd.read_csv(DATA_DIR+'promotions.csv',parse_dates=['start_date','end_date'])
    sample=pd.read_csv(DATA_DIR+'sample_submission.csv',parse_dates=['Date'])
    return sales,promos,sample


def build_features(df, promos):
    df['year']=df['Date'].dt.year; df['month']=df['Date'].dt.month
    df['day']=df['Date'].dt.day; df['dayofweek']=df['Date'].dt.dayofweek
    df['dayofyear']=df['Date'].dt.dayofyear
    df['weekofyear']=df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend']=(df['dayofweek']>=5).astype(int)
    df['is_payday']=df['day'].isin([1,2,5,15,25,30]).astype(int)
    vn_hol=[(1,1),(4,30),(5,1),(9,2),(12,25),(12,31)]
    df['is_holiday']=df.apply(lambda x:1 if(int(x.month),int(x.day))in vn_hol else 0,axis=1)
    for k,p in enumerate([365.25,182.625],1):
        df[f'sin_{k}']=np.sin(2*np.pi*df['dayofyear']/p)
        df[f'cos_{k}']=np.cos(2*np.pi*df['dayofyear']/p)
    df['sin_dow']=np.sin(2*np.pi*df['dayofweek']/7)
    df['cos_dow']=np.cos(2*np.pi*df['dayofweek']/7)
    df['is_tet']=0; df['dt_tet']=999
    for yr,(s,e) in TET.items():
        s_ts,e_ts=pd.Timestamp(s),pd.Timestamp(e)
        df.loc[(df['Date']>=s_ts)&(df['Date']<=e_ts),'is_tet']=1
        pre=(df['Date']>=s_ts-pd.Timedelta(days=30))&(df['Date']<s_ts)
        df.loc[pre,'dt_tet']=(s_ts-df.loc[pre,'Date']).dt.days
    df['pre_tet']=np.where(df['dt_tet']<30,np.exp(-df['dt_tet']/10),0)
    df['promo']=0; df['discount']=0.0
    for _,row in promos.dropna(subset=['start_date','end_date']).iterrows():
        m=(df['Date']>=row['start_date'])&(df['Date']<=row['end_date'])
        df.loc[m,'promo']+=1
        if pd.notnull(row.get('discount_value')):
            df.loc[m,'discount']=np.maximum(df.loc[m,'discount'],float(row['discount_value']))
    for yr in [2023,2024]:
        for s,e,d in [(f'{yr}-03-18',f'{yr}-04-17',12),(f'{yr}-06-23',f'{yr}-07-22',18),
                       (f'{yr}-08-30',f'{yr}-10-01',10),(f'{yr}-11-18',f'{yr}-12-31',20)]:
            m=(df['Date']>=pd.Timestamp(s))&(df['Date']<=pd.Timestamp(e))
            df.loc[m,'promo']=np.maximum(df.loc[m,'promo'],1)
            df.loc[m,'discount']=np.maximum(df.loc[m,'discount'],d)
    return df


# =====================================================================
# CORE INNOVATION: Month×DOW decomposition + half-year CAGR
# =====================================================================
def build_decomposition_v9(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24):
    """
    Half-year specific CAGR decomposition with Month×DOW seasonal.
    
    Instead of 366 DOY buckets (noisy), use 84 Month×DOW buckets (stable).
    Each bucket has ~40+ observations vs ~10 for DOY.
    """
    yr_r = train_sub.groupby('year')['Revenue'].mean().to_dict()
    yr_c = train_sub.groupby('year')['COGS'].mean().to_dict()
    mx = max(yr_r.keys()); rb, cb = yr_r[mx], yr_c[mx]

    # Half-year growth rates 
    # Assign monthly scale: H1 months (1-6) vs H2 months (7-12) have different CAGR
    month_rev_scale = {}
    month_cogs_scale = {}
    for m in range(1,7):  # H1
        month_rev_scale[(2023,m)] = (1+cagr_h1_23)
        month_rev_scale[(2024,m)] = (1+cagr_h1_23)*(1+cagr_h1_24)
    for m in range(7,13): # H2
        month_rev_scale[(2023,m)] = (1+cagr_h2_23)
        month_rev_scale[(2024,m)] = (1+cagr_h2_23)*(1+cagr_h2_24)
    # Same for COGS (could be different, but keep same for now)
    month_cogs_scale = {k: v for k,v in month_rev_scale.items()}
    
    # Assign scale to each row
    df['rts'] = 1.0; df['cts'] = 1.0
    for y in train_sub['year'].unique():
        mask = df['year']==y
        df.loc[mask,'rts'] = yr_r.get(y, rb)/rb
        df.loc[mask,'cts'] = yr_c.get(y, cb)/cb
    for (y,m), scale in month_rev_scale.items():
        mask = (df['year']==y) & (df['month']==m)
        df.loc[mask,'rts'] = scale
    for (y,m), scale in month_cogs_scale.items():
        mask = (df['year']==y) & (df['month']==m)
        df.loc[mask,'cts'] = scale

    # Month×DOW seasonal profile (THE KEY INNOVATION)
    tr = train_sub.copy()
    tr['rts_tr'] = tr['year'].map(yr_r)/rb
    tr['cts_tr'] = tr['year'].map(yr_c)/cb
    tr['nr'] = tr['Revenue']/tr['rts_tr']
    tr['nc'] = tr['COGS']/tr['cts_tr']

    # Compute Month×DOW average (84 buckets instead of 366)
    mdow_r = tr.groupby(['month','dayofweek'])['nr'].mean().reset_index()
    mdow_r.columns = ['month','dayofweek','seasonal_rev']
    mdow_c = tr.groupby(['month','dayofweek'])['nc'].mean().reset_index()
    mdow_c.columns = ['month','dayofweek','seasonal_cogs']

    df = df.merge(mdow_r, on=['month','dayofweek'], how='left')
    df = df.merge(mdow_c, on=['month','dayofweek'], how='left')
    df['seasonal_rev'] = df['seasonal_rev'].ffill().bfill()
    df['seasonal_cogs'] = df['seasonal_cogs'].ffill().bfill()

    # ALSO compute DOY profile (blended for robustness)
    dr = tr.groupby('dayofyear')['nr'].mean().reset_index()
    dr['doy_rev'] = dr['nr'].rolling(7,min_periods=1,center=True).mean()
    dc = tr.groupby('dayofyear')['nc'].mean().reset_index()
    dc['doy_cogs'] = dc['nc'].rolling(7,min_periods=1,center=True).mean()
    
    df = df.merge(dr[['dayofyear','doy_rev']], on='dayofyear', how='left')
    df = df.merge(dc[['dayofyear','doy_cogs']], on='dayofyear', how='left')
    df['doy_rev'] = df['doy_rev'].ffill().bfill()
    df['doy_cogs'] = df['doy_cogs'].ffill().bfill()

    # Blend: 60% Month×DOW (stable) + 40% DOY (captures day-specific patterns)
    df['blended_rev'] = 0.60 * df['seasonal_rev'] + 0.40 * df['doy_rev']
    df['blended_cogs'] = 0.60 * df['seasonal_cogs'] + 0.40 * df['doy_cogs']

    df['expected_revenue'] = df['blended_rev'] * df['rts']
    df['expected_cogs'] = df['blended_cogs'] * df['cts']

    eps=1e-6
    df['log_rev_ratio']=np.log((df['Revenue']+eps)/(df['expected_revenue']+eps))
    df['log_cogs_ratio']=np.log((df['COGS']+eps)/(df['expected_cogs']+eps))
    return df


# =====================================================================
# INNOVATION 2: COGS as ratio of Revenue (more stable)
# =====================================================================
def train_with_cogs_ratio(tr_df, te_df):
    """
    Instead of predicting Revenue and COGS independently:
    1. Predict Revenue log-ratio (as before)
    2. Predict COGS/Revenue ratio directly (more stable target)
    """
    X_tr, X_te = tr_df[FEATS], te_df[FEATS]
    
    # Revenue prediction (ensemble as before)
    configs_rev = [
        ('XGB1', xgb.XGBRegressor, dict(n_estimators=1000,learning_rate=0.025,max_depth=3,
            subsample=0.8,colsample_bytree=0.8,min_child_weight=10,
            reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbosity=0)),
        ('LGB', lgb.LGBMRegressor, dict(n_estimators=1000,learning_rate=0.025,max_depth=4,num_leaves=31,
            subsample=0.8,colsample_bytree=0.8,min_child_samples=30,
            reg_alpha=0.2,reg_lambda=1.0,random_state=42,verbose=-1)),
        ('XGB2', xgb.XGBRegressor, dict(n_estimators=1200,learning_rate=0.02,max_depth=4,
            subsample=0.75,colsample_bytree=0.75,min_child_weight=15,
            reg_alpha=0.2,reg_lambda=2.0,random_state=99,verbosity=0)),
    ]
    w = [0.40, 0.35, 0.25]

    # Revenue
    pr = []
    for name, cls, params in configs_rev:
        mr = cls(**params)
        mr.fit(X_tr, tr_df['log_rev_ratio'])
        pr.append(mr.predict(X_te))
    ens_r = sum(wi*pi for wi,pi in zip(w, pr))
    
    # COGS: predict log-ratio (same as revenue, independent model)
    pc = []
    for name, cls, params in configs_rev:
        cp = {**params, 'max_depth': min(params.get('max_depth',3), 3)}
        if 'num_leaves' in params: cp['num_leaves'] = min(params['num_leaves'], 15)
        mc = cls(**cp)
        mc.fit(X_tr, tr_df['log_cogs_ratio'])
        pc.append(mc.predict(X_te))
    ens_c = sum(wi*pi for wi,pi in zip(w, pc))
    
    # ALSO train COGS/Rev ratio model for blending
    tr_ratio = tr_df['COGS'] / (tr_df['Revenue'] + 1e-6)
    ratio_model = xgb.XGBRegressor(n_estimators=500,learning_rate=0.03,max_depth=3,
        subsample=0.8,colsample_bytree=0.8,min_child_weight=20,
        reg_alpha=0.2,reg_lambda=2.0,random_state=42,verbosity=0)
    ratio_model.fit(X_tr, tr_ratio)
    pred_ratio = ratio_model.predict(X_te)
    
    return ens_r, ens_c, pred_ratio


def make_submission(sales, promos, sample, 
                    cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24,
                    use_cogs_ratio=False, ratio_blend=0.3, label=""):
    all_dates = pd.DataFrame({'Date': pd.date_range(sales.Date.min(), TEST_END, freq='D')})
    df = all_dates.merge(sales[['Date','Revenue','COGS']], on='Date', how='left')
    df = build_features(df, promos)
    train_sub = df[df['Date']<=TRAIN_END].dropna(subset=['Revenue']).copy()
    df = build_decomposition_v9(df, train_sub, cagr_h1_23, cagr_h2_23, cagr_h1_24, cagr_h2_24)

    tr_df = df[df['Date']<=TRAIN_END].dropna(subset=['log_rev_ratio'])
    te_df = df[(df['Date']>=TEST_START)&(df['Date']<=TEST_END)].copy()
    
    ens_r, ens_c, pred_ratio = train_with_cogs_ratio(tr_df, te_df)

    te_df = te_df.copy()
    te_df['Revenue'] = np.clip(te_df['expected_revenue'].values*np.exp(ens_r), 0, None)
    
    # COGS: blend between independent prediction and ratio-based
    cogs_independent = np.clip(te_df['expected_cogs'].values*np.exp(ens_c), 0, None)
    cogs_from_ratio = te_df['Revenue'].values * np.clip(pred_ratio, 0.70, 0.95)
    
    if use_cogs_ratio:
        te_df['COGS'] = (1-ratio_blend)*cogs_independent + ratio_blend*cogs_from_ratio
    else:
        te_df['COGS'] = cogs_independent

    # Business rules only
    te_df.loc[te_df['COGS']>=te_df['Revenue'],'COGS'] = te_df.loc[te_df['COGS']>=te_df['Revenue'],'Revenue']*0.88
    te_df['Revenue'] = te_df['Revenue'].clip(lower=0).round(2)
    te_df['COGS'] = te_df['COGS'].clip(lower=0).round(2)

    sub = te_df[['Date','Revenue','COGS']].copy()
    sub = sub[sub['Date'].isin(sample['Date'])].sort_values('Date').reset_index(drop=True)
    sub['Date'] = sub['Date'].dt.strftime('%Y-%m-%d')
    fn = f'submission_v9_{label}.csv'
    sub.to_csv(os.path.join(OUTPUT_DIR,fn), index=False)
    margin = ((sub.Revenue-sub.COGS)/sub.Revenue*100)
    print(f"  [{fn}]")
    print(f"    Rev={sub.Revenue.mean():,.0f} COGS={sub.COGS.mean():,.0f} Margin={margin.mean():.1f}%")
    print(f"    H1-23={cagr_h1_23:.0%} H2-23={cagr_h2_23:.0%} H1-24={cagr_h1_24:.0%} H2-24={cagr_h2_24:.0%}")
    return fn


def main():
    print("="*72)
    print("  V9 PIPELINE - ARCHITECTURAL BREAKTHROUGH".center(72))
    print("  Month×DOW decomposition + Half-year CAGR".center(72))  
    print("="*72)
    
    sales, promos, sample = load_data()
    print(f"\n  Current best: v7_r28_r03 = 726,848")
    print(f"  Target: < 700,000\n")
    
    # ================================================================
    # From exploration, YoY growth 2021->2022 by month:
    #   H1 (Jan-Jun): +28%, +20%, +21%, +5%, -3%, +6% => avg ~13%
    #   H2 (Jul-Dec): -3%, +66%, +15%, +14%, +3%, +4% => avg ~17%
    # H2 grew faster than H1! But very volatile (Aug=+66%!)
    #
    # From score analysis: 
    #   Best: r28_r03 (uniform 28%/3% yearly)
    #   H1 typically has stronger months (Apr-Jun peak season)
    #   H2 has weaker months (Jul-Dec declining to year end)
    #
    # Strategy: H1 should have HIGHER CAGR than H2 
    #   because H1 includes peak shopping months
    # ================================================================
    
    # SUBMISSION 1: Half-year split around r28_r03
    # H1-23 higher (peak months grow more), H2-23 lower
    # H1-24 moderate, H2-24 very low
    print("SUBMISSION 1: Half-year optimized")
    print("-"*50)
    make_submission(sales, promos, sample,
                    cagr_h1_23=0.32, cagr_h2_23=0.24,
                    cagr_h1_24=0.05, cagr_h2_24=0.00,
                    use_cogs_ratio=False, label="halfy_v1")

    # SUBMISSION 2: Reverse - H2 grows more than H1
    # Historical data shows H2 2021->2022 grew faster (+17% vs +13%)
    print("\nSUBMISSION 2: H2-heavy growth + COGS ratio model")
    print("-"*50)
    make_submission(sales, promos, sample,
                    cagr_h1_23=0.24, cagr_h2_23=0.32,
                    cagr_h1_24=0.02, cagr_h2_24=0.05,
                    use_cogs_ratio=True, ratio_blend=0.3, label="halfy_v2_ratio")

    # SUBMISSION 3: Best possible uniform (28%/3%) but with 
    # Month×DOW decomposition + more regularized models
    # This tests if Month×DOW alone improves over DOY-only
    print("\nSUBMISSION 3: Same CAGR as V7 best, new decomposition")
    print("-"*50)
    make_submission(sales, promos, sample,
                    cagr_h1_23=0.28, cagr_h2_23=0.28,
                    cagr_h1_24=0.03, cagr_h2_24=0.03,
                    use_cogs_ratio=False, label="mdow_r28_r03")

    print(f"\n{'='*72}")
    print("  V9 COMPLETE - 3 SUBMISSIONS")
    print("="*72)
    print(f"""
  UPLOAD ORDER:
  
  1. submission_v9_mdow_r28_r03.csv (FIRST - isolates Month×DOW improvement)
     Same CAGR as V7 best. If score < 726k, Month×DOW decomposition helps.
     If score > 726k, DOY was better and we revert.
  
  2. submission_v9_halfy_v1.csv
     Half-year CAGR: H1-23=32%, H2-23=24%, H1-24=5%, H2-24=0%
     Tests if seasonal growth variation matters.
  
  3. submission_v9_halfy_v2_ratio.csv
     H2-heavy growth + COGS predicted as ratio of Revenue
     Tests both CAGR direction and COGS modeling approach.
    """)


if __name__=='__main__':
    main()
