#!/usr/bin/env python3
"""
HYPOTHESIS H28 TEST: Tactical Replenishment Pauses After Mass Strikes
Evaluates whether adding strictly past features capturing tactical pauses after
mass nationwide strikes improves forecasting MAE under the 3R Protocol.

Features tested:
  - nat_surge1: max(0, nat_r1 - 1.4) (intensity of surge yesterday)
  - nat_surge2: max(0, nat_r2 - 1.4) (intensity of surge 2 days ago)
  - is_post_mass1: indicator if nat_r1 > 1.4
  - is_post_mass2: indicator if nat_r2 > 1.4
  - days_since_mass: days since last day with nat_r > 1.4 (strictly shifted)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
import warnings
warnings.filterwarnings("ignore")

from features import load_daily

CSV = "volunteer_data_en.csv"
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

def load_panel(csv_path):
    v = pd.read_csv(csv_path, parse_dates=["started_at", "finished_at"])
    v = v[v.region != "Luhanska oblast"].sort_values(["region", "started_at"])
    prev_end = v.groupby("region").finished_at.transform(lambda s: s.cummax().shift())
    gap = (v.started_at - prev_end).dt.total_seconds() / 60.0
    v = v[gap.isna() | (gap > 30)]
    v["day"] = v.started_at.dt.tz_convert("Europe/Kyiv").dt.tz_localize(None).dt.normalize()
    last = v.day.max() - pd.Timedelta(days=1)
    v = v[v.day <= last]
    W = v.groupby(["region", "day"]).size().unstack(0).reindex(pd.date_range(v.day.min(), last)).fillna(0)
    W.index.name = "day"
    return W

def build_features(W, add_pause_features=False):
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
    
    # National level lags
    nat_r1 = N.shift(1) / Bn
    nat_r2 = N.shift(2) / Bn
    f["nat_r1"] = rep(nat_r1)
    f["nat_r7"] = rep(N.rolling(7).mean().shift(1) / Bn)
    f["dow"] = rep(W.index.dayofweek)
    f["month"] = rep(W.index.month)
    f["region_id"] = np.tile(np.arange(nr), nd)
    
    if add_pause_features:
        # Tactical pause features (strictly shifted <= D-1)
        f["nat_surge1"] = rep(np.maximum(0, nat_r1 - 1.4))
        f["nat_surge2"] = rep(np.maximum(0, nat_r2 - 1.4))
        f["is_post_mass1"] = rep((nat_r1 > 1.4).astype(float))
        f["is_post_mass2"] = rep((nat_r2 > 1.4).astype(float))
        
        # Days since last mass strike (strictly past)
        mass_mask = (N / (ew(N, 14) + 1.0)) > 1.4
        # Shift mask by 1 day so day D only knows if past days had mass strikes
        mass_past = mass_mask.shift(1).fillna(False)
        
        days_since = []
        c = 999
        for val in mass_past:
            if val:
                c = 1
            else:
                c = min(999, c + 1)
            days_since.append(min(c, 14))
        f["days_since_mass"] = rep(pd.Series(days_since, index=W.index))
        
    idx = pd.MultiIndex.from_product([W.index, W.columns], names=["day", "region"])
    X = pd.DataFrame(f, index=idx)
    X["y"] = W.values.ravel()
    X["base"] = B.values.ravel()
    return X, list(f.keys())

def evaluate_model(X, feats, W, test_start="2026-01-01"):
    day = X.index.get_level_values("day")
    
    # Dev: 2023-01-01 to 2025-12-31
    tr_mask = (day >= "2023-01-01") & (day < test_start)
    tr = X[tr_mask].dropna(subset=feats)
    
    model = HGB(**HGB_PARAMS).fit(tr[feats], tr.y / tr.base, sample_weight=tr.base)
    
    # Evaluate on Dev
    pred_ratio_dev = model.predict(tr[feats])
    h_dev = pd.Series(pred_ratio_dev * tr.base.values, index=tr.index)
    lvl_dev = pd.Series(ew(W, 14).stack().reindex(tr.index).values, index=tr.index)
    pred_dev = np.floor((h_dev + lvl_dev) / 2.0 + 0.5)
    base_dev = m28(W).stack().reindex(tr.index)
    
    mae_dev_model = float(np.mean(np.abs(tr.y.values - pred_dev.values)))
    mae_dev_base = float(np.mean(np.abs(tr.y.values - base_dev.values)))
    gain_dev = (mae_dev_model / mae_dev_base - 1.0) * 100
    
    # Evaluate on Test (2026)
    te_mask = (day >= test_start)
    te = X[te_mask].dropna(subset=feats)
    pred_ratio_te = model.predict(te[feats])
    h_te = pd.Series(pred_ratio_te * te.base.values, index=te.index)
    lvl_te = pd.Series(ew(W, 14).stack().reindex(te.index).values, index=te.index)
    pred_te = np.floor((h_te + lvl_te) / 2.0 + 0.5)
    base_te = m28(W).stack().reindex(te.index)
    
    mae_te_model = float(np.mean(np.abs(te.y.values - pred_te.values)))
    mae_te_base = float(np.mean(np.abs(te.y.values - base_te.values)))
    gain_te = (mae_te_model / mae_te_base - 1.0) * 100
    
    # Evaluate on last 90 days
    last_90_start = W.index[-90]
    last90_mask = te_mask & (day >= last_90_start)
    te90 = X[last90_mask].dropna(subset=feats)
    p90 = pred_te.reindex(te90.index)
    b90 = base_te.reindex(te90.index)
    mae_90_model = float(np.mean(np.abs(te90.y.values - p90.values)))
    mae_90_base = float(np.mean(np.abs(te90.y.values - b90.values)))
    gain_90 = (mae_90_model / mae_90_base - 1.0) * 100
    
    return {
        "dev_mae": mae_dev_model,
        "dev_base": mae_dev_base,
        "dev_gain": gain_dev,
        "test_mae": mae_te_model,
        "test_base": mae_te_base,
        "test_gain": gain_te,
        "90d_mae": mae_90_model,
        "90d_base": mae_90_base,
        "90d_gain": gain_90
    }

def main():
    W = load_panel(CSV)
    print(f"Loaded {CSV}, date range: {W.index.min().date()} to {W.index.max().date()}")
    
    # Baseline Champion (without pause features)
    X_champ, feats_champ = build_features(W, add_pause_features=False)
    res_champ = evaluate_model(X_champ, feats_champ, W)
    
    # Variant 1: All pause features
    X_p1, feats_p1 = build_features(W, add_pause_features=True)
    res_p1 = evaluate_model(X_p1, feats_p1, W)
    
    # Variant 2: Only nat_surge1 & is_post_mass1 (1-day immediate pause)
    feats_p2 = feats_champ + ["nat_surge1", "is_post_mass1"]
    res_p2 = evaluate_model(X_p1, feats_p2, W)
    
    # Variant 3: Continuous days_since_mass only
    feats_p3 = feats_champ + ["days_since_mass"]
    res_p3 = evaluate_model(X_p1, feats_p3, W)
    
    print("\n" + "="*80)
    print("HYPOTHESIS H28: TACTICAL PAUSE AFTER MASS STRIKES - 3R RESULTS")
    print("="*80)
    df_res = pd.DataFrame([
        {"Variant": "Champion (Baseline)", "Features": len(feats_champ), **res_champ},
        {"Variant": "H28.A: All Pause Features (Surge 1-2, Ind 1-2, DaysSince)", "Features": len(feats_p1), **res_p1},
        {"Variant": "H28.B: Immediate 1-Day Pause (Surge1 + PostMass1)", "Features": len(feats_p2), **res_p2},
        {"Variant": "H28.C: Decay Only (DaysSinceMass)", "Features": len(feats_p3), **res_p3},
    ])
    
    print(df_res[["Variant", "dev_mae", "dev_gain", "test_mae", "test_gain", "90d_mae", "90d_gain"]].to_string(index=False))
    
    # 3R Rule Check
    gain_vs_champ_dev = (res_p1["dev_mae"] / res_champ["dev_mae"] - 1.0) * 100
    gain_vs_champ_te = (res_p1["test_mae"] / res_champ["test_mae"] - 1.0) * 100
    print(f"\n3R Delta vs Champion on Dev (H28.A): {gain_vs_champ_dev:+.3f}%")
    print(f"3R Delta vs Champion on Test 2026 (H28.A): {gain_vs_champ_te:+.3f}%")
    
    if gain_vs_champ_dev <= -1.0:
        verdict = "CONFIRMED & ADOPTED (>= 1% improvement on Dev)"
    else:
        verdict = "REJECTED under 3R protocol (< 1% threshold gain on Dev)"
    print(f"\n3R Verdict: {verdict}")
    
    df_res.to_csv("h28_results.csv", index=False)
    print("\nSaved detailed results to h28_results.csv")

if __name__ == "__main__":
    main()
