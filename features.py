"""
Data preprocessing: raw siren logs -> daily panel (oblast x day) -> zero-leakage features.
STRICT LEAKAGE RULE: features for day D use ONLY days < D via shift(1).
"""
import pandas as pd
import numpy as np

TZ = "Europe/Kyiv"
MERGE_GAP = 30  # minutes

def load_daily(csv_path, merge_gap=30):
    v = pd.read_csv(csv_path, parse_dates=["started_at", "finished_at"])
    # Luhanska oblast is under continuous occupation/alert and uninformative
    v = v[v.region != "Luhanska oblast"].sort_values(["region", "started_at"])
    
    # Merge alert episodes if separated by <= 30 minutes
    if merge_gap > 0:
        prev_end = v.groupby("region").finished_at.transform(lambda s: s.cummax().shift())
        gap = (v.started_at - prev_end).dt.total_seconds() / 60.0
        v = v[gap.isna() | (gap > merge_gap)]
        
    v["day"] = v.started_at.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    last_full_day = v.day.max() - pd.Timedelta(days=1)
    v = v[v.day <= last_full_day]
    
    cnt = v.groupby(["region", "day"]).size().rename("y")
    regions = sorted(v.region.unique())
    days = pd.date_range(v.day.min(), last_full_day)
    idx = pd.MultiIndex.from_product([regions, days], names=["region", "day"])
    return cnt.reindex(idx, fill_value=0).reset_index()

def add_features(df):
    df = df.sort_values(["region", "day"]).reset_index(drop=True)
    g = df.groupby("region")["y"]
    
    # Strictly shifted past features
    for k in (1, 2, 3, 7, 14):
        df[f"lag{k}"] = g.shift(k)
        
    past = g.shift(1)
    for w in (3, 7, 14, 28):
        df[f"mean{w}"] = past.groupby(df["region"]).transform(lambda s: s.rolling(w).mean())
    df["max7"] = past.groupby(df["region"]).transform(lambda s: s.rolling(7).max())
    
    # National level aggregates
    nat = df.groupby("day")["y"].sum().rename("nat").to_frame()
    nat["nat_lag1"] = nat["nat"].shift(1)
    nat["nat_mean7"] = nat["nat"].shift(1).rolling(7).mean()
    nat["nat_mean28"] = nat["nat"].shift(1).rolling(28).mean()
    
    df = df.merge(nat[["nat_lag1", "nat_mean7", "nat_mean28"]], left_on="day", right_index=True)
    df["dow"] = df.day.dt.dayofweek
    df["month"] = df.day.dt.month
    df["region_id"] = df.region.astype("category").cat.codes
    return df

FEATURES = [
    "lag1", "lag2", "lag3", "lag7", "lag14", "mean3", "mean7", "mean14", "mean28",
    "max7", "nat_lag1", "nat_mean7", "nat_mean28", "dow", "month", "region_id"
]
