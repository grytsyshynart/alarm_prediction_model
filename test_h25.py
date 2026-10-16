"""
Hypothesis H25: Alert Merge Gap Optimization
Evaluate merge gaps (0m, 15m, 30m, 45m, 60m, 90m) on 2026 test data.
Investigates how alert fragmentation in 2026 affects predictability, MAE, baseline MAE, and 3R relative gain.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

TZ = "Europe/Kyiv"
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

FEATS = [
    "r_lag1", "r_lag2", "r_lag3", "r_lag7", "r_lag14",
    "r_mean3", "r_mean7", "r_mean14", "r_max7", "log_base",
    "nat_r1", "nat_r7", "dow", "month", "region_id"
]

def load_panel_gap(csv_path, merge_gap):
    v = pd.read_csv(csv_path, parse_dates=["started_at", "finished_at"])
    v = v[v.region != "Luhanska oblast"].sort_values(["region", "started_at"])
    
    if merge_gap > 0:
        prev_end = v.groupby("region").finished_at.transform(lambda s: s.cummax().shift())
        gap = (v.started_at - prev_end).dt.total_seconds() / 60.0
        v = v[gap.isna() | (gap > merge_gap)]
        
    v["day"] = v.started_at.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    last = v.day.max() - pd.Timedelta(days=1)
    v = v[v.day <= last]
    
    W = v.groupby(["region", "day"]).size().unstack(0).reindex(pd.date_range(v.day.min(), last)).fillna(0)
    W.index.name = "day"
    return W

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

def fit_hgb(X, asof_date):
    day = X.index.get_level_values("day")
    tr = X[(day <= asof_date) & (day >= pd.Timestamp("2023-01-01"))].dropna(subset=FEATS)
    return HGB(**HGB_PARAMS).fit(tr[FEATS], tr.y / tr.base, sample_weight=tr.base)

def evaluate_gap_on_2026(csv_path="volunteer_data_en.csv", gaps=[0, 15, 30, 45, 60, 90]):
    print("="*80)
    print("EXPERIMENT 1: Full Pipeline under each Merge Gap on 2026 Test Period")
    print("Train: 2023-01-01 to 2025-12-31 | Test: 2026-01-01 to 2026-10-15")
    print("="*80)
    
    results = []
    
    for gap in gaps:
        W = load_panel_gap(csv_path, gap)
        X = make_long_features(W)
        
        # Train on pre-2026
        model = fit_hgb(X, pd.Timestamp("2025-12-31"))
        
        # Test on 2026
        te = X[X.index.get_level_values("day") >= "2026-01-01"].dropna(subset=FEATS)
        h = pd.Series(model.predict(te[FEATS]) * te.base.values, index=te.index)
        lvl = pd.Series(ew(W, 14).stack().reindex(te.index).values, index=te.index)
        pred = np.floor((h + lvl) / 2.0 + 0.5)
        
        # Baseline mean28
        base = m28(W).stack().reindex(te.index)
        # Also ew14 alone
        ew14_base = ew(W, 14).stack().reindex(te.index)
        
        y_true = te.y.values
        mae_model = float(np.mean(np.abs(y_true - pred.values)))
        mae_m28 = float(np.mean(np.abs(y_true - base.values)))
        mae_ew14 = float(np.mean(np.abs(y_true - ew14_base.values)))
        
        mean_y = float(np.mean(y_true))
        nmae_model = mae_model / mean_y
        gain_vs_m28 = (mae_model / mae_m28 - 1.0) * 100.0
        gain_vs_ew14 = (mae_model / mae_ew14 - 1.0) * 100.0
        
        # Last 90 days of 2026 backtest (recent regime)
        te_90 = X[X.index.get_level_values("day") >= W.index[-90]].dropna(subset=FEATS)
        h_90 = pd.Series(model.predict(te_90[FEATS]) * te_90.base.values, index=te_90.index)
        lvl_90 = pd.Series(ew(W, 14).stack().reindex(te_90.index).values, index=te_90.index)
        pred_90 = np.floor((h_90 + lvl_90) / 2.0 + 0.5)
        base_90 = m28(W).stack().reindex(te_90.index)
        mae_90_model = float(np.mean(np.abs(te_90.y.values - pred_90.values)))
        mae_90_base = float(np.mean(np.abs(te_90.y.values - base_90.values)))
        gain_90 = (mae_90_model / mae_90_base - 1.0) * 100.0
        
        results.append({
            "gap": gap,
            "mean_y_2026": mean_y,
            "mae_model_2026": mae_model,
            "mae_m28_2026": mae_m28,
            "nmae_model": nmae_model,
            "gain_vs_m28_%": gain_vs_m28,
            "gain_vs_ew14_%": gain_vs_ew14,
            "mae_90d_model": mae_90_model,
            "mae_90d_m28": mae_90_base,
            "gain_90d_%": gain_90
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df_res

if __name__ == "__main__":
    evaluate_gap_on_2026()
