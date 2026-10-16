"""
Adversarial Zero-Leakage Test:
Corrupts the future with random noise and asserts that historical feature vectors
remain completely identical (L1 diff == 0).
"""
import sys
import numpy as np
import pandas as pd
from features import load_daily, add_features, FEATURES

def run_leak_test(csv_path):
    print(f"Loading data from {csv_path}...")
    base = load_daily(csv_path)
    cut = base.day.max() - pd.Timedelta(days=400)
    
    # Baseline run
    f1 = add_features(base.copy())
    
    # Corrupted future run
    corrupt = base.copy()
    mask = corrupt.day >= cut
    corrupt.loc[mask, "y"] = np.random.default_rng(42).integers(0, 50, mask.sum())
    f2 = add_features(corrupt)
    
    # Check historical features (day <= cut)
    past_mask = f1.day <= cut
    diff = (f1.loc[past_mask, FEATURES] - f2.loc[past_mask, FEATURES]).abs().sum().sum()
    future_diff = (f1.loc[~past_mask, FEATURES] - f2.loc[~past_mask, FEATURES]).abs().sum().sum()
    
    assert diff == 0, f"DATA LEAKAGE DETECTED! Historical feature diff: {diff}"
    assert future_diff > 0, "Adversarial control failed to alter future."
    print(f"Zero-leakage test PASSED: past diff = {diff}, future diff = {future_diff:.0f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leak_test.py <path_to_volunteer_data_en.csv>")
        sys.exit(1)
    run_leak_test(sys.argv[1])
