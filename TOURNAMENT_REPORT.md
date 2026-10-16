# Tournament Report: Hypotheses H25 & H26 (3R Evaluation Protocol)

## Protocol Summary
- **Protocol**: 3R (Repetitive, Rule-based with >=1% gain required to beat champion, Return)
- **Target Variable**: Daily count of air raid siren episodes per oblast (with canonical 30-min merge gap)
- **Evaluation Period (2026 Out-of-Sample)**: 2026-01-01 to 2026-10-15 (288 days × 24 oblasts = 6,912 region-days)
- **Recent Regime Backtest**: Last 90 days of dataset (2026-07-18 to 2026-10-15)
- **Champion Architecture**:
  - Task 1 (Tomorrow by Oblast): Ensembled Ratio-HGB (loss='absolute_error', lr=0.06, max_depth=3, leaf=100) + ew14
  - Task 2 (Tomorrow National): 28-day rolling baseline (mean28)
  - Task 3 (Next 7 Days by Oblast): 4-smoother ensemble (mean28 + ew7 + ew14 + ew28) * 7 / 4
  - Task 4 (Next 7 Days National): 4-smoother ensemble + Historical 10-90% empirical error intervals

---

## Hypothesis H25: Alert Merge Gap Optimization

### Hypothesis Statement
In 2026, air defense sirens exhibited severe fragmentation (e.g. repeated on/off alarms within minutes). Modifying the merge gap parameter (e.g., 15m, 30m, 45m, 60m, 90m) may reduce noise and improve model MAE and predictability.

### Methodology & Results
1. **Raw Alert Counts & Fragmentation in 2026**:
   - Total alerts in 2026: 32,660
   - Gaps ≤ 15 min: 3,015 (9.2%)
   - Gaps ≤ 30 min: 6,018 (18.4%)
   - Gaps ≤ 45 min: 8,742 (26.8%)
   - Gaps ≤ 60 min: 11,129 (34.1%)
   - Gaps ≤ 90 min: 15,066 (46.1%)

2. **Full Pipeline Evaluation on 2026 Test Period (2026-01-01 to 2026-10-15)**:
| Merge Gap | Mean Alarms/Day (2026) | Model MAE (2026) | Baseline mean28 MAE | Normalized MAE (MAE/mean) | Gain vs mean28 (%) | Last 90d Model MAE | Last 90d Baseline MAE | Last 90d Gain (%) |
|---|---|---|---|---|---|---|---|---|
| **0 min (raw)** | 4.7196 | 1.5906 | 1.6548 | 0.3370 | -3.88% | 1.7463 | 1.7661 | -1.12% |
| **15 min** | 4.2839 | 1.4100 | 1.4712 | 0.3291 | -4.16% | 1.5181 | 1.5308 | -0.83% |
| **30 min (Champ)**| **3.8498** | **1.2619** | **1.3156** | **0.3278** | **-4.09%** | **1.3398** | **1.3669** | **-1.98%** |
| **45 min** | 3.4557 | 1.1399 | 1.1908 | 0.3299 | -4.27% | 1.1801 | 1.2080 | -2.31% |
| **60 min** | 3.1110 | 1.0294 | 1.0733 | 0.3309 | -4.09% | 1.0366 | 1.0660 | -2.76% |
| **90 min** | 2.5425 | 0.8853 | 0.9332 | 0.3482 | -5.14% | 0.8838 | 0.9117 | -3.06% |

3. **Multi-Gap Feature Extension (Target = 30m)**:
   - Added strictly shifted past fragmentation ratios (`gap0 / gap30` lag1 and rolling 7-day mean) and clustering ratios (`gap60 / gap30` lag1 and rolling 7-day mean).
   - Champion 30m MAE: 1.2619
   - Multi-gap Extended MAE: 1.2541
   - Gain: -0.62% (below 1.0% threshold)
   - Last 90d Gain: -0.66% (below 1.0% threshold)

### Decision & Conclusion: RETURN TO CHAMPION (MERGE_GAP = 30 min)
- **Statistical Finding**: Raw MAE naturally falls as merge gap widens simply because fewer alert events exist. However, **Normalized MAE (NMAE) achieves its global optimum at 30 minutes (0.3278)**.
- Gaps < 30m suffer from reporting noise/fragmentation, while gaps > 30m over-merge distinct combat sorties, increasing normalized relative error.
- Multi-gap feature extension achieved only -0.62% gain, failing the 3R ≥ 1.0% threshold.

---

## Hypothesis H26: Regional Clustering

### Hypothesis Statement
Splitting oblasts into quiet/rear (low alert frequency) and active/frontline clusters allows models to specialize on distinct regional siren dynamics, improving overall forecasting MAE.

### Methodology & Results
Clustering was derived strictly from pre-test historical data (2023-01-01 to 2025-12-31) to prevent data leakage:
- **Quiet / Rear Regions (mean ≤ 1.21 alerts/day)**: Zakarpatska (0.44), Ivano-Frankivska (0.49), Chernivetska (0.51), Ternopilska (0.54), Rivnenska (0.60), Lvivska (0.64), Volynska (0.75), Khmelnytska (0.76), Zhytomyrska (1.01), Vinnytska (1.02), Kyiv City (1.21).
- **Transit / Active / Frontline (mean ≥ 1.83 alerts/day)**: Odeska (1.83), Cherkaska (1.98), Kyivska (2.08), Chernihivska (2.45), Poltavska (3.24), Kirovohradska (3.47), Khersonska (3.71), Mykolaivska (3.73), Sumska (4.35), Zaporizka (4.69), Donetska (4.76), Kharkivska (5.60), Dnipropetrovska (6.36).

#### Tournament Results (2026 Test Period & 90-day Backtest):
| Variant | 2026 Test MAE | Gain vs Champion (%) | Last 90d MAE | Last 90d Gain (%) | Quiet 2026 MAE | Active 2026 MAE | 3R Decision |
|---|---|---|---|---|---|---|---|
| **Champion (Global Single HGB)** | **1.2619** | **0.00%** | **1.3398** | **0.00%** | **0.8542** | **1.6068** | **CHAMPION RETAINED** |
| H26.A: 2 Clusters (Mean Split < 1.5) | 1.2633 | +0.11% | 1.3458 | +0.45% | 0.8598 | 1.6047 | REJECT (Degradation) |
| H26.B: 2 Clusters (Mean Split < 2.5) | 1.2708 | +0.71% | 1.3449 | +0.38% | 0.8706 | 1.6095 | REJECT (Degradation) |
| H26.C: 3 Clusters (Quiet, Transit, Frontline) | 1.2601 | -0.14% | 1.3403 | +0.03% | 0.8598 | 1.5988 | REJECT (Gain < 1%) |
| H26.D: K-Means Clustering (k=2) | 1.2642 | +0.18% | 1.3458 | +0.45% | 0.8545 | 1.6108 | REJECT (Degradation) |
| H26.E: Global HGB + `cluster_id` Feature | 1.2588 | -0.24% | 1.3431 | +0.24% | 0.8557 | 1.5999 | REJECT (Gain < 1%) |
| H26.F: Hybrid (Quiet=ew14, Active=HGB) | 1.2713 | +0.75% | 1.3352 | -0.35% | 0.8747 | 1.6068 | REJECT (Degradation) |
| H26.G: 2 Clusters (Median Split == 0) | 1.2597 | -0.17% | 1.3361 | -0.28% | 0.8482 | 1.6079 | REJECT (Gain < 1%) |
| H26.H: Specialized Quiet Hyperparameters | 1.2948 | +2.61% | 1.4106 | +5.29% | 0.9249 | 1.6079 | REJECT (Degradation) |
| H26.I: Poisson Loss for Quiet Cluster | 1.2911 | +2.32% | 1.3741 | +2.56% | 0.9167 | 1.6079 | REJECT (Degradation) |

### Decision & Conclusion: RETURN TO CHAMPION (Single Global Model)
- **Why clustering failed**:
  1. In the champion architecture, target normalization (`y / base`, where `base = ew14 + 0.1`) already maps all regions—quiet and active alike—into a scale-free ratio distribution centered around 1.0.
  2. Sample weighting by `base` already allows frontline regions to anchor gradient updates appropriately while quiet regions contribute proportional regularizing signal.
  3. Splitting into separate models halves or thirds the available sample size per tree and destroys cross-regional correlations (e.g., nationwide missile/drone raid synchronization captured by `nat_r1` and `nat_r7`).
  4. Specialized quiet hyperparameters and count-based Poisson regression performed substantially worse (+2.3% to +5.3% degradation), demonstrating that quiet regions benefit from global scale-free ratio representations rather than isolated low-count fitting.
- All clustering variants failed the 3R rule (gain ranged from +5.29% degradation to at best -0.24%, strictly below the required -1.0% threshold).
- Champion retained without modification.

---

## Hypothesis H28: Tactical Replenishment Pauses After Mass Strikes

### Hypothesis Statement
Following a major nationwide missile/drone raid (captured when `nat_r1 > 1.4`), enemy aviation and drone launch units often require a replenishment/preparation window, leading to a temporary lull or drop below the recent rolling trend in subsequent days.

### Methodology & Results
Features evaluated under strict pre-2026 dev isolation:
- `nat_surge1`: $\max(0, \text{nat\_r1} - 1.4)$ (magnitude of yesterday's surge)
- `nat_surge2`: $\max(0, \text{nat\_r2} - 1.4)$ (magnitude of 2-days-ago surge)
- `is_post_mass1`, `is_post_mass2`: binary indicators for post-surge days
- `days_since_mass`: discrete count of days since last mass strike ($\le D-1$)

| Variant | Features | Dev MAE | Dev Gain (%) | 2026 Test MAE | Test Gain (%) | 90d MAE | 90d Gain (%) | 3R Decision |
|---|---|---|---|---|---|---|---|---|
| **Champion (Baseline)** | 15 | **0.9248** | **-6.38%** | **1.2619** | **-4.09%** | **1.3398** | **-1.98%** | **CHAMPION RETAINED** |
| H28.A: All Pause Features | 20 | 0.9220 | -6.67% | 1.2649 | -3.86% | 1.3366 | -2.22% | REJECT (Dev Gain < 1%, Test Degradation) |
| H28.B: Immediate 1-Day Pause | 17 | 0.9248 | -6.38% | 1.2619 | -4.09% | 1.3398 | -1.98% | REJECT (No delta) |
| H28.C: Decay Only (`days_since_mass`) | 16 | 0.9203 | -6.83% | 1.2642 | -3.91% | 1.3426 | -1.78% | REJECT (Dev Gain < 1%, Test Degradation) |

### Decision & Conclusion: RETURN TO CHAMPION
- On the Dev set, adding pause features reduced MAE by only -0.31% to -0.49%, failing the 3R rule threshold ($\ge 1.0\%$).
- On the 2026 Test set, pause features slightly degraded generalizability (+0.24% higher MAE).
- **Physical Reason**: In 2026, routine Shahed drone swarms and frontline tactical guided bomb (KAB) strikes occur nearly every single night, overriding classical post-missile lull periods at the siren frequency level.
- Champion model retained.

---

## Production Forecast Output
Generated and verified in `forecast_latest.csv`:
- **Date for Tomorrow**: 2026-10-16
- **7-Day Horizon**: 2026-10-16 to 2026-10-22
- **National Forecast Tomorrow**: 129 alert episodes
- **National Forecast 7-Day**: 881 alert episodes (80% Empirical CI: [788, 1027])
