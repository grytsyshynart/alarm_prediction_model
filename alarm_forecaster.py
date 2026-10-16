#!/usr/bin/env python3
"""
UKRAINIAN AIR RAID SIRENS FORECASTER (Tomorrow and Next 7 Days)
Final Champion Architecture from the 3R Protocol Tournament.

Implements:
  - Task 1 (Tomorrow by Oblast): Ensembled Ratio-HGB (loss='absolute_error') + ew14
  - Task 2 (Tomorrow National): 28-day rolling baseline
  - Task 3 (Next 7 Days by Oblast): 4-smoother ensemble (mean28 + ew7 + ew14 + ew28) * 7 / 4
  - Task 4 (Next 7 Days National): 4-smoother ensemble + Historical 10-90% empirical error intervals
"""
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

DEFAULT_CSV = "volunteer_data_en.csv"
TZ, MERGE_GAP = "Europe/Kyiv", 30
HGB_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.06,
    max_depth=3,
    max_iter=200,
    min_samples_leaf=100,
    random_state=0
)

# 1. DATA INGESTION
def load_panel(csv_path):
    v = pd.read_csv(csv_path, parse_dates=["started_at", "finished_at"])
    v = v[v.region != "Luhanska oblast"].sort_values(["region", "started_at"])
    
    # Merge consecutive alerts if gap <= 30 min (addresses reporting fragmentation)
    prev_end = v.groupby("region").finished_at.transform(lambda s: s.cummax().shift())
    gap = (v.started_at - prev_end).dt.total_seconds() / 60.0
    v = v[gap.isna() | (gap > MERGE_GAP)]
    
    v["day"] = v.started_at.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    last = v.day.max() - pd.Timedelta(days=1)
    v = v[v.day <= last]
    
    W = v.groupby(["region", "day"]).size().unstack(0).reindex(pd.date_range(v.day.min(), last)).fillna(0)
    W.index.name = "day"
    return W

# 2. FEATURE EXTRACTION
ew = lambda x, h: x.ewm(halflife=h).mean().shift(1)
m28 = lambda x: x.rolling(28).mean().shift(1)

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

FEATS = [
    "r_lag1", "r_lag2", "r_lag3", "r_lag7", "r_lag14",
    "r_mean3", "r_mean7", "r_mean14", "r_max7", "log_base",
    "nat_r1", "nat_r7", "dow", "month", "region_id"
]

# 3. FORECASTING ENGINES
def fit_hgb(X, asof_date):
    day = X.index.get_level_values("day")
    tr = X[(day <= asof_date) & (day >= pd.Timestamp("2023-01-01"))].dropna(subset=FEATS)
    return HGB(**HGB_PARAMS).fit(tr[FEATS], tr.y / tr.base, sample_weight=tr.base)

def tomorrow_by_region(W_hist):
    asof = W_hist.index.max()
    tmr = asof + pd.Timedelta(days=1)
    W = pd.concat([W_hist, pd.DataFrame(np.nan, index=[tmr], columns=W_hist.columns)])
    W.index.name = "day"
    
    X = make_long_features(W)
    model = fit_hgb(X, asof)
    row = X[X.index.get_level_values("day") == tmr]
    
    # Predict scale-free ratio, multiply by base
    h = pd.Series(model.predict(row[FEATS]) * row.base.values, index=row.index.get_level_values("region"))
    level = ew(W, 14).loc[tmr]
    
    # Champion ensemble for T1: 50% HGB + 50% ew14, rounded to integer
    pred = np.floor((h + level) / 2.0 + 0.5).astype(int).rename("завтра")
    return pred, tmr

def ensemble_level(x):
    return (m28(x) + ew(x, 7) + ew(x, 14) + ew(x, 28)) / 4.0

def next_week(W_hist):
    asof = W_hist.index.max()
    N = W_hist.sum(axis=1)
    
    ext = lambda x: pd.concat([x, x.iloc[[-1]].set_axis([asof + pd.Timedelta(days=1)])])
    Wn = ext(W_hist.astype(float)).copy()
    Wn.iloc[-1] = np.nan
    Nn = Wn.sum(axis=1, min_count=1)
    
    lvl_reg = ensemble_level(Wn).iloc[-1]
    lvl_nat = float(ensemble_level(Nn).iloc[-1])
    
    # Historical empirical deviation quantiles (104 past weeks on Mondays)
    Lh = ensemble_level(N)
    y7 = N[::-1].rolling(7).sum()[::-1]
    ok = (N.index.dayofweek == 0) & y7.notna() & Lh.notna()
    ratio = (y7[ok] / (7 * Lh[ok])).tail(104)
    lo, hi = ratio.quantile([0.1, 0.9])
    
    reg_pred = (7 * lvl_reg).round(0).astype(int).rename("тиждень")
    return reg_pred, 7 * lvl_nat, (7 * lvl_nat * lo, 7 * lvl_nat * hi)

# 4. BUILT-IN SELF TESTS
def leakage_test(W, seed=0):
    cut = W.index[-400]
    X1 = make_long_features(W)
    W2 = W.copy()
    W2.loc[W2.index >= cut] = np.random.default_rng(seed).integers(0, 40, W2.loc[W2.index >= cut].shape)
    X2 = make_long_features(W2)
    sel = X1.index.get_level_values("day") <= cut
    diff = np.nansum(np.abs(X1.loc[sel, FEATS].values - X2.loc[sel, FEATS].values))
    ctrl = np.nansum(np.abs(X1.loc[~sel, FEATS].values - X2.loc[~sel, FEATS].values))
    assert diff == 0 and ctrl > 0, "LEAKAGE DETECTED"
    print(f"[Self-Test] Leakage check passed: past diff = {diff}, future diff = {ctrl:.0f}")

def quick_backtest(W, days=90):
    start = W.index[-days]
    X = make_long_features(W)
    model = fit_hgb(X, start - pd.Timedelta(days=1))
    te = X[X.index.get_level_values("day") >= start].dropna(subset=FEATS)
    h = pd.Series(model.predict(te[FEATS]) * te.base.values, index=te.index)
    lvl = pd.Series(ew(W, 14).stack().reindex(te.index).values, index=te.index)
    pred = np.floor((h + lvl) / 2.0 + 0.5)
    base = m28(W).stack().reindex(te.index)
    mae = lambda p: float(np.mean(np.abs(te.y.values - np.asarray(p))))
    print(f"[Backtest last {days}d, T1 Oblasts] Model MAE = {mae(pred):.3f} | Baseline mean28 = {mae(base):.3f} | Gain = {(mae(pred)/mae(base)-1)*100:+.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to volunteer_data_en.csv")
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()
    
    try:
        W = load_panel(args.csv)
    except FileNotFoundError:
        print(f"Error: Dataset '{args.csv}' not found.")
        print("Please provide the path via --csv <path_to_volunteer_data_en.csv>")
        sys.exit(1)
        
    print(f"Dataset span: {W.index.min().date()} to {W.index.max().date()}, Oblasts: {W.shape[1]}")
    leakage_test(W)
    if not args.no_backtest:
        quick_backtest(W)
        
    tm, tmr = tomorrow_by_region(W)
    wk, nat_wk, (lo, hi) = next_week(W)
    nat_tm = float(m28(pd.concat([W.sum(axis=1), pd.Series([np.nan], index=[tmr])])).iloc[-1])
    
    out = pd.concat([tm, wk], axis=1)
    out.loc["УКРАЇНА (разом)"] = [round(nat_tm), round(nat_wk)]
    out.to_csv("forecast_latest.csv", encoding="utf-8")
    
    print(f"\n=== FORECAST: Tomorrow = {tmr.date()} | Next 7 Days = {tmr.date()} to {(tmr + pd.Timedelta(days=6)).date()} ===")
    print(out.sort_values("тиждень", ascending=False).to_string())
    print(f"\nNational 7-Day Forecast: {nat_wk:.0f} (80% CI: {lo:.0f} ... {hi:.0f}) | Past 7-Day Actual: {W.sum(axis=1).tail(7).sum():.0f}")
