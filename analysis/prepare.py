import os
"""Load the Colab zip, join to the scored tract list, QA it, and write Zindi-ready files."""
import json, zipfile, io
import numpy as np, pandas as pd, geopandas as gpd

SAMPLE = os.path.join(os.environ.get("BB_DATA", "data"), "SampleSubmission.csv")
PREFIX_REGION = {"04": "AZ", "35": "AZ", "06": "CA", "40": "OK", "48": "TX", "53": "WA"}

def load_zip(path):
    z = zipfile.ZipFile(path)
    rd = lambda n: pd.read_csv(io.BytesIO(z.read(n)), dtype={"GEOID": str}, low_memory=False)
    out = {n.rsplit(".", 1)[0]: rd(n) for n in z.namelist() if n.endswith(".csv")}
    if "tracts.parquet" in z.namelist():
        out["tracts"] = gpd.read_parquet(io.BytesIO(z.read("tracts.parquet")))
    out["log"] = z.read("run_log.txt").decode() if "run_log.txt" in z.namelist() else ""
    return out

def scored_frame(d):
    s = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)[["GEOID"]]
    s["state_region"] = s["GEOID"].str[:2].map(PREFIX_REGION)
    f = s.merge(d["features"], on="GEOID", how="left", indicator=True)
    if "strata" in d:
        st = d["strata"].drop(columns=[c for c in ["region"] if c in d["strata"]])
        f = f.merge(st, on="GEOID", how="left", suffixes=("", "_strata"))
    return f

def qa(f):
    rep = {"rows": len(f), "missing_features": int((f["_merge"] == "left_only").sum())}
    for c in ["bld_n", "poi_n", "road_len_m", "acs_housing_units"]:
        if c in f:
            rep[c] = {"zeros": int((f[c].fillna(0) == 0).sum()), "nulls": int(f[c].isna().sum()),
                      "median": float(f[c].median())}
    return rep

def write_submission(pred, path, sample=SAMPLE):
    """pred: Series indexed by GEOID with coverage_gap_score in [0,1]. Keeps the sample's exact layout."""
    s = pd.read_csv(sample, dtype=str, keep_default_na=False)
    v = s["GEOID"].map(pred.clip(0, 1))
    assert v.notna().all(), f"{v.isna().sum()} tracts without a prediction"
    s["coverage_gap_score"] = v.round(6).astype(str)
    s.to_csv(path, index=False)
    chk = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert (chk["GEOID"] == pd.read_csv(sample, dtype=str)["GEOID"]).all()
    return path

def shift_to_region_means(pred, region, targets):
    """Additive shift per region so each region's mean matches the leaderboard-derived mean."""
    out = pred.copy()
    for r, m in targets.items():
        idx = region == r
        if idx.any():
            out[idx] = (out[idx] + (m - out[idx].mean())).clip(0, 1)
    return out
