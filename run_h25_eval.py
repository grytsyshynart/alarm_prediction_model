import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
import warnings
warnings.filterwarnings("ignore")

TZ = "Europe/Kyiv"
FEATS = [
    "r_lag1", "r_lag2", "r_lag3", "r_lag7", "r_lag14",
    "r_mean3", "r_mean7", "r_mean14", "r_max7", "log_base",
    "nat_r1", "nat_r7", "dow", "month", "region_id"
]
HGB_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.06,
    max_depth=3,
    max_iter=200,
    min_samples_leaf=100,
    random_state=0
)

ew = lambda x, h: x.ewm(halflife=h).mean().shift(1)
m28 = lambda x: x.rolling(28).mean().shift(1)

def load_panel_custom(csv_path, merge_mode="fixed", gap_min=30):
    v = pd.read_csv(csv_path, parse_dates=["started_at", "finished_at"])
    v = v[v.region != "Luhanska oblast"].sort_values(["region", "started_at"])
    
    prev_end = v.groupby("region").finished_at.transform(lambda s: s.cummax().shift())
    v["dur"] = (v.finished_at - v.started_at).dt.total_seconds() / 60.0
    dur = v["dur"]
    gap = (v.started_at - prev_end).dt.total_seconds() / 60.0
    
    if merge_mode == "fixed":
        thresh = gap_min
        mask = gap.isna() | (gap > thresh)
    elif merge_mode == "dynamic_duration":
        # If alert is short (< 30 min) or previous alert was short, merge up to 60 min gap, else 30 min
        prev_dur = v.groupby("region")["dur"].shift()
        short_alert = (dur < 30) | (prev_dur.fillna(0) < 30)
        thresh = np.where(short_alert, 60.0, 30.0)
        mask = gap.isna() | (gap > thresh)
    elif merge_mode == "dynamic_region":
        active_regions = {
            "Dnipropetrovska oblast", "Kharkivska oblast", "Donetska oblast", 
            "Zaporizka oblast", "Sumska oblast", "Khersonska oblast", "Mykolaivska oblast"
        }
        thresh = np.where(v.region.isin(active_regions), 60.0, 30.0)
        mask = gap.isna() | (gap > thresh)
    else:
        raise ValueError(f"Unknown merge_mode: {merge_mode}")
        
    v_clean = v[mask].copy()
    v_clean["day"] = v_clean.started_at.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    last = v_clean.day.max() - pd.Timedelta(days=1)
    v_clean = v_clean[v_clean.day <= last]
    
    W = v_clean.groupby(["region", "day"]).size().unstack(0).reindex(pd.date_range(v_clean.day.min(), last)).fillna(0)
    W.index.name = "day"
    return W, len(v_clean)

def make_long_features(W):
    N = W.sum(axis=1, min_count=1)
    B = ew(W, 14) + 0.1
    Bn = ew(N, 14) + 1.0
    nd, nr = W.shape
    rep = lambda s: np.repeat(np.asarray(s, float), nr)
    
    f = {}
    for k in (1, 2, 3, 7, 14):
        f[f"r_lag{k}"] = (W.shift(k) / B).values.ravel()
    for w in (3, 7, 14):
        f[f"r_mean{w}"] = (W.rolling(w).mean().shift(1) / B).values.ravel()
    f["r_max7"] = (W.rolling(7).max().shift(1) / B).values.ravel()
    f["log_base"] = np.log(B).values.ravel()
    f["nat_r1"] = rep(N.shift(1) / Bn)
    f["nat_r7"] = rep(N.rolling(7).mean().shift(1) / Bn)
    f["dow"] = rep(W.index.dayofweek)
    f["month"] = rep(W.index.month)
    f["region_id"] = np.tile(np.arange(nr), nd)
    
    idx = pd.MultiIndex.from_product([W.index, W.columns], names=["day", "region"])
    X = pd.DataFrame(f, index=idx)
    X["y"] = W.values.ravel()
    X["base"] = B.values.ravel()
    return X

def eval_config(name, mode, gap):
    W, n_alerts = load_panel_custom("volunteer_data_en.csv", merge_mode=mode, gap_min=gap)
    X = make_long_features(W)
    day = X.index.get_level_values("day")
    
    # Dev 2024 (Train 2023, Eval 2024)
    tr24 = X[(day >= pd.Timestamp("2023-01-01")) & (day <= pd.Timestamp("2023-12-31"))].dropna(subset=FEATS)
    m24 = HGB(**HGB_PARAMS).fit(tr24[FEATS], tr24.y / tr24.base, sample_weight=tr24.base)
    te24 = X[(day >= pd.Timestamp("2024-01-01")) & (day <= pd.Timestamp("2024-12-31"))].dropna(subset=FEATS)
    p24 = np.floor((m24.predict(te24[FEATS]) * te24.base.values + ew(W, 14).stack().reindex(te24.index).values) / 2.0 + 0.5)
    b24 = m28(W).stack().reindex(te24.index).values
    mae_d24 = np.mean(np.abs(te24.y.values - p24))
    mae_b24 = np.mean(np.abs(te24.y.values - b24))
    
    # Dev 2025 (Train 2023-2024, Eval 2025)
    tr25 = X[(day >= pd.Timestamp("2023-01-01")) & (day <= pd.Timestamp("2024-12-31"))].dropna(subset=FEATS)
    m25 = HGB(**HGB_PARAMS).fit(tr25[FEATS], tr25.y / tr25.base, sample_weight=tr25.base)
    te25 = X[(day >= pd.Timestamp("2025-01-01")) & (day <= pd.Timestamp("2025-12-31"))].dropna(subset=FEATS)
    p25 = np.floor((m25.predict(te25[FEATS]) * te25.base.values + ew(W, 14).stack().reindex(te25.index).values) / 2.0 + 0.5)
    b25 = m28(W).stack().reindex(te25.index).values
    mae_d25 = np.mean(np.abs(te25.y.values - p25))
    mae_b25 = np.mean(np.abs(te25.y.values - b25))
    
    dev_mae = (mae_d24 + mae_d25) / 2.0
    dev_base = (mae_b24 + mae_b25) / 2.0
    dev_gain = (dev_mae / dev_base - 1.0) * 100.0
    
    # Test 2026 (Train 2023-2025, Eval 2026)
    tr26 = X[(day >= pd.Timestamp("2023-01-01")) & (day <= pd.Timestamp("2025-12-31"))].dropna(subset=FEATS)
    m26 = HGB(**HGB_PARAMS).fit(tr26[FEATS], tr26.y / tr26.base, sample_weight=tr26.base)
    te26 = X[(day >= pd.Timestamp("2026-01-01")) & (day <= W.index.max())].dropna(subset=FEATS)
    p26 = np.floor((m26.predict(te26[FEATS]) * te26.base.values + ew(W, 14).stack().reindex(te26.index).values) / 2.0 + 0.5)
    b26 = m28(W).stack().reindex(te26.index).values
    mae_t26 = np.mean(np.abs(te26.y.values - p26))
    mae_b26 = np.mean(np.abs(te26.y.values - b26))
    test_gain = (mae_t26 / mae_b26 - 1.0) * 100.0
    
    mean_y_dev = float(np.mean(np.concatenate([te24.y.values, te25.y.values])))
    mean_y_test = float(np.mean(te26.y.values))
    
    # Normalized MAE
    nmae_dev = dev_mae / mean_y_dev
    nmae_test = mae_t26 / mean_y_test
    
    return {
        "Config": name,
        "Total_Alerts": n_alerts,
        "Dev_MAE": dev_mae,
        "Dev_Base": dev_base,
        "Dev_Gain%": dev_gain,
        "Dev_NMAE": nmae_dev,
        "Test26_MAE": mae_t26,
        "Test26_Base": mae_b26,
        "Test26_Gain%": test_gain,
        "Test26_NMAE": nmae_test,
    }

configs = [
    ("Merge 15m", "fixed", 15),
    ("Merge 30m (Champion Baseline)", "fixed", 30),
    ("Merge 45m", "fixed", 45),
    ("Merge 60m", "fixed", 60),
    ("Merge 90m", "fixed", 90),
    ("Dynamic Duration (30m/60m)", "dynamic_duration", 0),
    ("Dynamic Region (30m/60m)", "dynamic_region", 0),
]

results = []
for name, mode, gap in configs:
    print(f"Testing {name}...")
    res = eval_config(name, mode, gap)
    results.append(res)

df_all = pd.DataFrame(results)
print("\n=== HYPOTHESIS H25 COMPREHENSIVE 3R EVALUATION ===")
print(df_all.to_string(index=False))
df_all.to_csv("h25_full_evaluation.csv", index=False)
