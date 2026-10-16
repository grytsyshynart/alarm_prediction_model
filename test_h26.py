"""
Hypothesis H26: Regional Clustering Evaluation under the 3R Protocol
Evaluates whether segmenting Ukrainian oblasts into quiet/rear vs frontline/active
improves forecasting MAE over the Global Champion architecture.

Tested Variants:
  - Baseline Champion: Single Global HGB model + ew14 ensemble
  - H26.A: 2 Clusters (Historical mean split < 1.5)
  - H26.B: 2 Clusters (Historical mean split < 2.5)
  - H26.C: 3 Clusters (Quiet < 1.5, Transit 1.5-3.5, Frontline >= 3.5)
  - H26.D: Unsupervised K-Means Clustering (k=2 on historical summary stats)
  - H26.E: Global HGB with 'cluster_id' feature
  - H26.F: Hybrid Model (Quiet = ew14, Active = HGB ensemble)
  - H26.G: 2 Clusters (Historical median split == 0)
  - H26.H: Specialized Quiet Hyperparameters (depth=2, lr=0.03, unweighted)
  - H26.I: Poisson Loss on raw counts for quiet cluster

STRICT ZERO-LEAKAGE:
All cluster definitions and models are trained strictly on pre-eval historical data.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.cluster import KMeans

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

from alarm_forecaster import load_panel, make_long_features, FEATS, HGB_PARAMS, ew, m28

CSV_PATH = "volunteer_data_en.csv"

def run_all_h26_experiments():
    print(f"Loading data from {CSV_PATH}...")
    W = load_panel(CSV_PATH)
    X = make_long_features(W)
    day = X.index.get_level_values("day")
    reg = X.index.get_level_values("region")
    
    # Pre-2026 Dev data for cluster definition (STRICT ZERO LEAKAGE)
    W_dev = W.loc["2023-01-01":"2025-12-31"]
    means = W_dev.mean().sort_values()
    medians = W_dev.median()
    
    print("\n--- Pre-2026 Historical Alert Frequency Summary ---")
    print(f"Total Regions: {len(means)}")
    print(f"Lowest: {means.index[0]} ({means.iloc[0]:.2f}/day) | Highest: {means.index[-1]} ({means.iloc[-1]:.2f}/day)")
    
    # Cluster definitions
    # Mean < 1.5 (11 quiet, 13 active)
    q_mean15 = list(means[means < 1.5].index)
    act_mean15 = [r for r in W.columns if r not in q_mean15]
    
    # Mean < 2.5 (15 quiet/transit, 9 frontline)
    q_mean25 = list(means[means < 2.5].index)
    act_mean25 = [r for r in W.columns if r not in q_mean25]
    
    # 3 Clusters: Quiet (< 1.5), Transit (1.5 - 3.5), Frontline (>= 3.5)
    c_quiet = list(means[means < 1.5].index)
    c_front = list(means[means >= 3.5].index)
    c_trans = [r for r in W.columns if r not in c_quiet and r not in c_front]
    
    # KMeans k=2
    km_feats = pd.DataFrame({
        "mean": means,
        "std": W_dev.std(),
        "p90": W_dev.quantile(0.9)
    })
    km = KMeans(n_clusters=2, random_state=42).fit(km_feats)
    km_q = list(means.index[km.labels_ == 0])
    km_act = [r for r in W.columns if r not in km_q]
    
    # Median == 0 (7 quiet, 17 active)
    q_med0 = list(medians[medians == 0].index)
    act_med0 = [r for r in W.columns if r not in q_med0]
    
    # Split windows
    tr_mask_26 = (day >= "2023-01-01") & (day <= "2025-12-31")
    te_mask_26 = (day >= "2026-01-01") & (day <= W.index.max())
    te_mask_90 = (day >= W.index[-90]) & (day <= W.index.max())
    
    tr_26 = X[tr_mask_26].dropna(subset=FEATS)
    te_26 = X[te_mask_26].dropna(subset=FEATS)
    te_90 = X[te_mask_90].dropna(subset=FEATS)
    
    lvl_26 = pd.Series(ew(W, 14).stack().reindex(te_26.index).values, index=te_26.index)
    lvl_90 = pd.Series(ew(W, 14).stack().reindex(te_90.index).values, index=te_90.index)
    
    base_m28_26 = m28(W).stack().reindex(te_26.index).values
    base_m28_90 = m28(W).stack().reindex(te_90.index).values
    
    # Baseline Champion
    m_champ = HGB(**HGB_PARAMS).fit(tr_26[FEATS], tr_26.y / tr_26.base, sample_weight=tr_26.base)
    h_champ_26 = pd.Series(m_champ.predict(te_26[FEATS]) * te_26.base.values, index=te_26.index)
    h_champ_90 = pd.Series(m_champ.predict(te_90[FEATS]) * te_90.base.values, index=te_90.index)
    
    p_champ_26 = np.floor((h_champ_26 + lvl_26) / 2.0 + 0.5)
    p_champ_90 = np.floor((h_champ_90 + lvl_90) / 2.0 + 0.5)
    
    mae_champ_26 = float(np.mean(np.abs(te_26.y.values - p_champ_26.values)))
    mae_champ_90 = float(np.mean(np.abs(te_90.y.values - p_champ_90.values)))
    
    results = []
    
    def log_variant(name, p_26, p_90, details=""):
        mae_26 = float(np.mean(np.abs(te_26.y.values - p_26.values)))
        mae_90 = float(np.mean(np.abs(te_90.y.values - p_90.values)))
        gain_26 = (mae_26 / mae_champ_26 - 1.0) * 100.0
        gain_90 = (mae_90 / mae_champ_90 - 1.0) * 100.0
        
        # Breakdown by quiet vs active (using mean < 1.5)
        q_mask_26 = te_26.index.get_level_values("region").isin(q_mean15)
        mae_q26 = float(np.mean(np.abs(te_26.y[q_mask_26].values - p_26[q_mask_26].values)))
        mae_act26 = float(np.mean(np.abs(te_26.y[~q_mask_26].values - p_26[~q_mask_26].values)))
        
        # 3R Decision
        passed = (gain_26 <= -1.0) and (gain_90 <= -1.0)
        decision = "ACCEPT (NEW CHAMPION)" if passed else "REJECT (Gain < 1% / Degradation)"
        
        results.append({
            "Variant": name,
            "2026_MAE": mae_26,
            "Gain_vs_Champ_2026_%": gain_26,
            "Last_90d_MAE": mae_90,
            "Gain_vs_Champ_90d_%": gain_90,
            "Quiet_2026_MAE": mae_q26,
            "Active_2026_MAE": mae_act26,
            "3R_Decision": decision,
            "Notes": details
        })
        print(f"{name: <45} | 2026: {mae_26:.4f} ({gain_26:+.2f}%) | 90d: {mae_90:.4f} ({gain_90:+.2f}%) | {decision}")

    print("\n" + "="*95)
    print("HYPOTHESIS H26 TOURNAMENT EXECUTION (3R PROTOCOL)")
    print("="*95)
    
    # 0. Champion
    log_variant("Champion (Global Single HGB)", p_champ_26, p_champ_90, "Baseline champion")
    
    # Helper to eval separate models by cluster list
    def eval_clusters(clusters):
        h26_list, h90_list = [], []
        for c in clusters:
            tr_c = tr_26[tr_26.index.get_level_values("region").isin(c)]
            te26_c = te_26[te_26.index.get_level_values("region").isin(c)]
            te90_c = te_90[te_90.index.get_level_values("region").isin(c)]
            mc = HGB(**HGB_PARAMS).fit(tr_c[FEATS], tr_c.y / tr_c.base, sample_weight=tr_c.base)
            h26_list.append(pd.Series(mc.predict(te26_c[FEATS]) * te26_c.base.values, index=te26_c.index))
            h90_list.append(pd.Series(mc.predict(te90_c[FEATS]) * te90_c.base.values, index=te90_c.index))
        h26 = pd.concat(h26_list).reindex(te_26.index)
        h90 = pd.concat(h90_list).reindex(te_90.index)
        return np.floor((h26 + lvl_26) / 2.0 + 0.5), np.floor((h90 + lvl_90) / 2.0 + 0.5)

    # H26.A
    p26_a, p90_a = eval_clusters([q_mean15, act_mean15])
    log_variant("H26.A: 2 Clusters (Mean Split < 1.5)", p26_a, p90_a, "11 Quiet vs 13 Active")

    # H26.B
    p26_b, p90_b = eval_clusters([q_mean25, act_mean25])
    log_variant("H26.B: 2 Clusters (Mean Split < 2.5)", p26_b, p90_b, "15 Quiet/Transit vs 9 Frontline")

    # H26.C
    p26_c, p90_c = eval_clusters([c_quiet, c_trans, c_front])
    log_variant("H26.C: 3 Clusters (Quiet/Transit/Frontline)", p26_c, p90_c, "11 Quiet, 6 Transit, 7 Frontline")

    # H26.D
    p26_d, p90_d = eval_clusters([km_q, km_act])
    log_variant("H26.D: K-Means Clustering (k=2)", p26_d, p90_d, "K-Means on mean, std, p90")

    # H26.E: Global HGB + cluster_id Feature
    cmap = {r: (0 if r in q_mean15 else 1) for r in W.columns}
    X_e = X.copy()
    X_e["cluster_id"] = [cmap[r] for r in X_e.index.get_level_values("region")]
    fe_feats = FEATS + ["cluster_id"]
    tr_e = X_e[tr_mask_26].dropna(subset=fe_feats)
    te26_e = X_e[te_mask_26].dropna(subset=fe_feats)
    te90_e = X_e[te_mask_90].dropna(subset=fe_feats)
    m_e = HGB(**HGB_PARAMS).fit(tr_e[fe_feats], tr_e.y / tr_e.base, sample_weight=tr_e.base)
    h26_e = pd.Series(m_e.predict(te26_e[fe_feats]) * te26_e.base.values, index=te26_e.index)
    h90_e = pd.Series(m_e.predict(te90_e[fe_feats]) * te90_e.base.values, index=te90_e.index)
    p26_e = np.floor((h26_e + lvl_26) / 2.0 + 0.5)
    p90_e = np.floor((h90_e + lvl_90) / 2.0 + 0.5)
    log_variant("H26.E: Global HGB + cluster_id Feature", p26_e, p90_e, "Adds cluster_id to FEATS")

    # H26.F: Hybrid Model (Quiet=ew14, Active=HGB)
    p26_f = p_champ_26.copy()
    p90_f = p_champ_90.copy()
    qm26 = te_26.index.get_level_values("region").isin(q_mean15)
    qm90 = te_90.index.get_level_values("region").isin(q_mean15)
    p26_f[qm26] = np.floor(lvl_26[qm26] + 0.5)
    p90_f[qm90] = np.floor(lvl_90[qm90] + 0.5)
    log_variant("H26.F: Hybrid (Quiet=ew14, Active=HGB)", p26_f, p90_f, "Direct smoother for quiet oblasts")

    # H26.G: 2 Clusters (Median Split == 0)
    p26_g, p90_g = eval_clusters([q_med0, act_med0])
    log_variant("H26.G: 2 Clusters (Median Split == 0)", p26_g, p90_g, "7 zero-median vs 17 active")

    # H26.H: Specialized Quiet Hyperparameters
    tr_q = tr_26[tr_26.index.get_level_values("region").isin(q_med0)]
    te26_q = te_26[te_26.index.get_level_values("region").isin(q_med0)]
    te90_q = te_90[te_90.index.get_level_values("region").isin(q_med0)]
    hp_q = dict(loss="absolute_error", learning_rate=0.03, max_depth=2, max_iter=150, min_samples_leaf=50, random_state=0)
    mq_h = HGB(**hp_q).fit(tr_q[FEATS], tr_q.y / tr_q.base)
    h26_qh = pd.Series(mq_h.predict(te26_q[FEATS]) * te26_q.base.values, index=te26_q.index)
    h90_qh = pd.Series(mq_h.predict(te90_q[FEATS]) * te90_q.base.values, index=te90_q.index)
    
    tr_act = tr_26[~tr_26.index.get_level_values("region").isin(q_med0)]
    te26_act = te_26[~te_26.index.get_level_values("region").isin(q_med0)]
    te90_act = te_90[~te_90.index.get_level_values("region").isin(q_med0)]
    m_act = HGB(**HGB_PARAMS).fit(tr_act[FEATS], tr_act.y / tr_act.base, sample_weight=tr_act.base)
    h26_act = pd.Series(m_act.predict(te26_act[FEATS]) * te26_act.base.values, index=te26_act.index)
    h90_act = pd.Series(m_act.predict(te90_act[FEATS]) * te90_act.base.values, index=te90_act.index)
    
    h26_h = pd.concat([h26_qh, h26_act]).reindex(te_26.index)
    h90_h = pd.concat([h90_qh, h90_act]).reindex(te_90.index)
    p26_h = (h26_h + lvl_26) / 2.0
    p26_h.loc[h26_qh.index] = np.where(p26_h.loc[h26_qh.index] < 0.4, 0.0, p26_h.loc[h26_qh.index])
    p26_h = np.floor(p26_h + 0.5)
    p90_h = (h90_h + lvl_90) / 2.0
    p90_h.loc[h90_qh.index] = np.where(p90_h.loc[h90_qh.index] < 0.4, 0.0, p90_h.loc[h90_qh.index])
    p90_h = np.floor(p90_h + 0.5)
    log_variant("H26.H: Specialized Quiet Hyperparameters", p26_h, p90_h, "depth=2, lr=0.03, zero-guard")

    # H26.I: Poisson Loss on counts for quiet cluster
    hp_p = dict(loss="poisson", learning_rate=0.04, max_depth=2, max_iter=150, min_samples_leaf=50, random_state=0)
    mq_p = HGB(**hp_p).fit(tr_q[FEATS], tr_q.y)
    h26_qp = pd.Series(mq_p.predict(te26_q[FEATS]), index=te26_q.index)
    h90_qp = pd.Series(mq_p.predict(te90_q[FEATS]), index=te90_q.index)
    p26_i = np.floor((h26_act + lvl_26.reindex(te26_act.index)) / 2.0 + 0.5)
    p26_i = pd.concat([p26_i, np.round(h26_qp)]).reindex(te_26.index)
    p90_i = np.floor((h90_act + lvl_90.reindex(te90_act.index)) / 2.0 + 0.5)
    p90_i = pd.concat([p90_i, np.round(h90_qp)]).reindex(te_90.index)
    log_variant("H26.I: Poisson Loss for Quiet Cluster", p26_i, p90_i, "Poisson on counts for quiet cluster")

    df_out = pd.DataFrame(results)
    df_out.to_csv("h26_results.csv", index=False)
    print("\nSaved full results to h26_results.csv")
    return df_out

if __name__ == "__main__":
    run_all_h26_experiments()
