"""
Hypothesis H25 Extension: Multi-gap features for canonical 30m target.
Test if adding features from other merge gaps (e.g. gap 15 or gap 60 counts/ratios)
improves the forecasting of the canonical 30m merged siren target.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from test_h25 import load_panel_gap, make_long_features, fit_hgb, FEATS, HGB_PARAMS, ew, m28

def test_multigap_features(csv_path="volunteer_data_en.csv"):
    W30 = load_panel_gap(csv_path, 30)
    W0 = load_panel_gap(csv_path, 0)
    W60 = load_panel_gap(csv_path, 60)
    
    # Baseline champion data
    X30 = make_long_features(W30)
    
    # Add multi-gap ratio features (strictly shifted past)
    # Ratio of raw (gap 0) to 30m: reflects degree of fragmentation yesterday and past 7 days
    frag_lag1 = (W0.shift(1) / (W30.shift(1) + 0.1)).values.ravel()
    frag_mean7 = (W0.rolling(7).mean().shift(1) / (W30.rolling(7).mean().shift(1) + 0.1)).values.ravel()
    
    # Ratio of 60m to 30m: reflects clustering into long continuous events
    clust_lag1 = (W60.shift(1) / (W30.shift(1) + 0.1)).values.ravel()
    clust_mean7 = (W60.rolling(7).mean().shift(1) / (W30.rolling(7).mean().shift(1) + 0.1)).values.ravel()
    
    X_ext = X30.copy()
    X_ext["frag_lag1"] = frag_lag1
    X_ext["frag_mean7"] = frag_mean7
    X_ext["clust_lag1"] = clust_lag1
    X_ext["clust_mean7"] = clust_mean7
    
    ext_feats = FEATS + ["frag_lag1", "frag_mean7", "clust_lag1", "clust_mean7"]
    
    # Train champion on pre-2026
    day = X30.index.get_level_values("day")
    tr = X30[(day <= "2025-12-31") & (day >= "2023-01-01")].dropna(subset=FEATS)
    champ_model = HGB(**HGB_PARAMS).fit(tr[FEATS], tr.y / tr.base, sample_weight=tr.base)
    
    # Train extended on pre-2026
    tr_ext = X_ext[(day <= "2025-12-31") & (day >= "2023-01-01")].dropna(subset=ext_feats)
    ext_model = HGB(**HGB_PARAMS).fit(tr_ext[ext_feats], tr_ext.y / tr_ext.base, sample_weight=tr_ext.base)
    
    # Test on 2026
    te = X30[day >= "2026-01-01"].dropna(subset=FEATS)
    te_ext = X_ext[day >= "2026-01-01"].dropna(subset=ext_feats)
    
    # Champion predictions
    h_champ = pd.Series(champ_model.predict(te[FEATS]) * te.base.values, index=te.index)
    lvl = pd.Series(ew(W30, 14).stack().reindex(te.index).values, index=te.index)
    pred_champ = np.floor((h_champ + lvl) / 2.0 + 0.5)
    
    # Extended predictions
    h_ext = pd.Series(ext_model.predict(te_ext[ext_feats]) * te_ext.base.values, index=te_ext.index)
    pred_ext = np.floor((h_ext + lvl) / 2.0 + 0.5)
    
    y_true = te.y.values
    mae_champ = np.mean(np.abs(y_true - pred_champ.values))
    mae_ext = np.mean(np.abs(y_true - pred_ext.values))
    
    gain = (mae_ext / mae_champ - 1.0) * 100.0
    print(f"2026 Test Period (288 days x 24 oblasts):")
    print(f"Champion MAE: {mae_champ:.4f}")
    print(f"Multi-gap Extended MAE: {mae_ext:.4f}")
    print(f"Gain: {gain:+.2f}%")
    
    # Check last 90 days
    te_90 = X30[day >= W30.index[-90]].dropna(subset=FEATS)
    te_ext_90 = X_ext[day >= W30.index[-90]].dropna(subset=ext_feats)
    h_champ_90 = pd.Series(champ_model.predict(te_90[FEATS]) * te_90.base.values, index=te_90.index)
    h_ext_90 = pd.Series(ext_model.predict(te_ext_90[ext_feats]) * te_ext_90.base.values, index=te_ext_90.index)
    lvl_90 = pd.Series(ew(W30, 14).stack().reindex(te_90.index).values, index=te_90.index)
    
    pred_champ_90 = np.floor((h_champ_90 + lvl_90) / 2.0 + 0.5)
    pred_ext_90 = np.floor((h_ext_90 + lvl_90) / 2.0 + 0.5)
    
    mae_champ_90 = np.mean(np.abs(te_90.y.values - pred_champ_90.values))
    mae_ext_90 = np.mean(np.abs(te_ext_90.y.values - pred_ext_90.values))
    gain_90 = (mae_ext_90 / mae_champ_90 - 1.0) * 100.0
    print(f"\nLast 90 Days Backtest:")
    print(f"Champion MAE: {mae_champ_90:.4f}")
    print(f"Multi-gap Extended MAE: {mae_ext_90:.4f}")
    print(f"Gain: {gain_90:+.2f}%")

if __name__ == "__main__":
    test_multigap_features()
