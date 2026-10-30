# Lifelines analysis: which registry places are missing from Overture, and where.
# Usage: python3 analyze.py <lifelines_registry_matches.csv> <out_dir>
import sys, os, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
import statsmodels.formula.api as smf
import statsmodels.api as sm

STRATA = os.environ.get("BB_STRATA", "data/strata.csv")
COLONIAS = os.environ.get("BB_COLONIAS", "data/hud_colonias_communities.geojson")
TO_M = Transformer.from_crs(4326, 5070, always_xy=True)
RNG = np.random.default_rng(2026)
SCOLS = ["GEOID", "region", "STATEFP", "COUNTYFP", "ALAND", "pop_total", "pct_urban", "ur_class", "svi_overall",
         "svi_socioeconomic", "svi_minority", "cvi_overall", "ruca_primary", "tribal_any", "tribal_pct", "aiannh_name",
         "usfs_WHP_mean", "epht_heat_days_summer", "usfs_HUCount_sum"]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def load(path):
    M = pd.read_csv(path, dtype={"GEOID": str, "ref_id": str, "zip5": str, "hnum": str}, low_memory=False)
    S = pd.read_csv(STRATA, dtype={"GEOID": str, "STATEFP": str, "COUNTYFP": str}, usecols=SCOLS, low_memory=False)
    M = M.merge(S.drop(columns=["region"]), on="GEOID", how="left")
    M["county"] = M["GEOID"].str[:5]
    for c in ["s_150", "s_500", "s_any", "d_150", "d_500", "d_any", "addr_ss", "addr_s", "kind_50", "n_150", "n_500", "d_near"]:
        if c in M.columns:
            M[c] = num(M[c])
    M["chain"] = M["chain"].astype(str).str.lower().isin(["true", "1"])
    M["addr_kind"] = M.get("addr_kind", False)
    M["addr_kind"] = M["addr_kind"].astype(str).str.lower().isin(["true", "1"])
    # verdicts
    near_name = M["s_150"].fillna(0) >= 80
    mid_name = (M["s_500"].fillna(0) >= 90) & (~M["chain"] | (M["d_500"].fillna(1e9) <= 250))
    far_name = (~M["chain"]) & (M["s_any"].fillna(0) >= 95)
    addr = (M["addr_ss"].fillna(0) >= 85) & ((M["addr_s"].fillna(0) >= 60) | M["addr_kind"])
    M["found"] = near_name | mid_name | far_name | addr
    M["found_strict"] = (M["s_150"].fillna(0) >= 90) | ((M["addr_ss"].fillna(0) >= 90) & (M["addr_s"].fillna(0) >= 80))
    M["found_loose"] = M["found"] | (M["kind_50"].fillna(0) > 0)
    M["missing"] = ~M["found"]
    # strata
    M["rural"] = (num(M["ruca_primary"]) >= 4) if "ruca_primary" in M else (M["ur_class"] == "rural")
    M["tribal"] = (num(M["tribal_pct"]).fillna(0) > 0) | (num(M["tribal_any"]).fillna(0) > 0)
    M["svi"] = num(M["svi_overall"])
    M["svi_q"] = pd.cut(M["svi"], [-0.01, 0.25, 0.5, 0.75, 1.0], labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    M["dens"] = num(M["pop_total"]) / (num(M["ALAND"]) / 1e6).clip(lower=0.01)
    M["log_dens"] = np.log1p(M["dens"].fillna(0))
    # HUD colonias: distance to the nearest colonia point
    try:
        import geopandas as gpd
        C = gpd.read_file(COLONIAS).to_crs(4326)
        C = C[C.geometry.notna() & ~C.geometry.is_empty]
        cx, cy = TO_M.transform(C.geometry.x.values, C.geometry.y.values)
        ok = np.isfinite(cx) & np.isfinite(cy)
        tree = cKDTree(np.column_stack([cx[ok], cy[ok]]))
        names = C["NAME"].values[ok]
        x, y = TO_M.transform(num(M["lon"]).values, num(M["lat"]).values)
        good = np.isfinite(x) & np.isfinite(y)
        d = np.full(len(M), np.nan); nm = np.full(len(M), "", dtype=object)
        dd, ii = tree.query(np.column_stack([x[good], y[good]]))
        d[good], nm[good] = dd, names[ii]
        M["d_colonia_km"] = d / 1000
        M["colonia_name"] = nm
        M["colonia_1km"] = M["d_colonia_km"] <= 1.0
    except Exception as e:
        print("colonias:", e)
        M["d_colonia_km"], M["colonia_1km"] = np.nan, False
    # SNAP store types
    if "rtype_hist" in M.columns:
        t = M["rtype_hist"].fillna("").astype(str)
        t = t.where(t != "", M["rtype"].fillna("").astype(str))
    else:
        t = M["rtype"].fillna("").astype(str)
    tl = t.str.lower()
    M["stype"] = np.select(
        [tl.str.contains("super store|supermarket|large grocery"), tl.str.contains("medium grocery|small grocery|^grocery"),
         tl.str.contains("convenience"), tl.str.contains("combination"), tl.str.contains("farmer"),
         tl.str.contains("specialty|meat|bakery|seafood|fruit|vegetable")],
        ["supermarket or large grocery", "small or medium grocery", "convenience store", "combination (dollar, pharmacy, other)",
         "farmers market", "specialty (meat, bakery, produce)"], default="other")
    return M

def rate_ci(g, col="missing", reps=1000):
    """Share with a tract-cluster bootstrap 95% interval."""
    g = g[["GEOID", col]].dropna()
    if len(g) == 0:
        return np.nan, np.nan, np.nan, 0
    by = g.groupby("GEOID")[col].agg(["sum", "count"])
    s, n = by["sum"].values, by["count"].values
    est = s.sum() / n.sum()
    k = len(by)
    idx = RNG.integers(0, k, (reps, k))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return est, np.quantile(boots, 0.025), np.quantile(boots, 0.975), int(n.sum())

def table(M, by, col="missing"):
    rows = []
    for key, g in M.groupby(by, observed=True, dropna=False):
        est, lo, hi, n = rate_ci(g, col)
        rows.append({**({by: key} if isinstance(by, str) else dict(zip(by, key))), "n": n,
                     "share": round(est, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    return pd.DataFrame(rows)

def logit(M, formula):
    d = M.dropna(subset=["svi", "log_dens"]).copy()
    d["missing_i"] = d["missing"].astype(int)
    for c in ("rural", "tribal", "colonia_1km", "chain"):
        if c in d:
            d[c] = d[c].astype(int)
    lhs, rhs = formula.split("~")
    terms = [t.strip() for t in rhs.split("+")]
    keep = []
    for t in terms:
        v = t[2:-1] if t.startswith("C(") else t
        if v in d.columns and d[v].nunique(dropna=True) > 1:
            keep.append(t)
    formula = lhs + "~ " + " + ".join(keep)
    try:
        f = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["county"])[0]})
        out = pd.DataFrame({"coef": f.params, "se": f.bse, "p": f.pvalues, "odds_ratio": np.exp(f.params)})
        return out.round(4), int(f.nobs)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), 0

def add_context(M, ctx_dir):
    """ACS tract context and county business counts, when the run fetched them."""
    a = os.path.join(ctx_dir, "acs_tract_context.csv")
    if os.path.exists(a):
        A = pd.read_csv(a, dtype={"GEOID": str})
        A["snap_hh_share"] = num(A["B22010_002E"]) / num(A["B22010_001E"]).replace(0, np.nan)
        A["hisp_share"] = num(A["B03003_003E"]) / num(A["B03003_001E"]).replace(0, np.nan)
        inc = num(A["B19013_001E"])
        A["median_income"] = inc.where(inc > 0)
        M = M.merge(A[["GEOID", "snap_hh_share", "hisp_share", "median_income", "B22010_002E"]].rename(
            columns={"B22010_002E": "snap_households"}), on="GEOID", how="left")
        M["snap_hh_bin"] = pd.cut(M["snap_hh_share"], [-0.01, 0.10, 0.20, 0.30, 1.0],
                                  labels=["under 10%", "10 to 20%", "20 to 30%", "30% or more"])
    b = os.path.join(ctx_dir, "county_business_counts.csv")
    if os.path.exists(b):
        B = pd.read_csv(b, dtype={"state": str, "county": str})
        B["county5"] = B["state"].str.zfill(2) + B["county"].str.zfill(3)
        B["estab"] = num(B["estab"])
        P = B.pivot_table(index=["county5", "NAME"], columns=["dataset", "sector"], values="estab", aggfunc="first")
        P.columns = [f"{d}_{s}" for d, s in P.columns]
        P = P.reset_index()
        if "nonemp_44-45" in P and "cbp_44-45" in P:
            P["retail_nonemp_per_employer"] = P["nonemp_44-45"] / P["cbp_44-45"].replace(0, np.nan)
        if "nonemp_00" in P and "cbp_00" in P:
            P["all_nonemp_per_employer"] = P["nonemp_00"] / P["cbp_00"].replace(0, np.nan)
        M = M.merge(P.rename(columns={"county5": "county", "NAME": "county_name"}), on="county", how="left")
    return M

def main(path, out, ctx_dir=None):
    os.makedirs(out, exist_ok=True)
    M = load(path)
    if ctx_dir:
        M = add_context(M, ctx_dir)
    R = {}
    R["overall"] = table(M, ["ref_set", "region"]).to_dict("records")
    R["overall_strict_loose"] = (M.groupby("ref_set")[["found", "found_strict", "found_loose"]].mean().round(4)
                                 .reset_index().to_dict("records"))
    S = M[M["ref_set"] == "SNAP"].copy()
    if len(S):
        if "auth_date" in S.columns:
            ad = pd.to_datetime(S["auth_date"], errors="coerce")
            S["stable"] = ad.notna() & (ad <= "2025-06-30")
            R["snap_auth_known_share"] = round(float(ad.notna().mean()), 4)
            if S["stable"].sum() > 0.3 * len(S):
                S = S[S["stable"]]
        R["snap_types_all"] = S["stype"].value_counts().to_dict()
        R["snap_rtype_raw"] = S["rtype"].fillna("").value_counts().head(20).to_dict()
        S = S[~S["stype"].isin(["farmers market", "other"])]
        R["snap_n"] = int(len(S))
        R["snap_chain_share"] = round(float(S["chain"].mean()), 4)
        S["group"] = np.where(S["chain"], "chain", "independent")
        bys = ["group", "stype", "region", "rural", "tribal", "svi_q", "colonia_1km"]
        if "snap_hh_bin" in S.columns:
            bys.append("snap_hh_bin")
        for by in bys:
            R[f"snap_by_{by}"] = table(S, by).to_dict("records")
            if by != "group":
                R[f"snap_by_{by}_x_group"] = table(S, [by, "group"]).to_dict("records")
        I = S[~S["chain"]]
        lt, n = logit(I, "missing_i ~ C(stype) + rural + svi + tribal + colonia_1km + log_dens + C(region)")
        R["snap_logit_independent"] = {"n": n, "table": lt.reset_index().to_dict("records")}
        lt2, n2 = logit(S, "missing_i ~ chain + C(stype) + rural + svi + tribal + colonia_1km + log_dens + C(region)")
        R["snap_logit_all"] = {"n": n2, "table": lt2.reset_index().to_dict("records")}
        # composition: share of independents among SNAP stores by stratum
        R["snap_indep_share_by_svi_q"] = S.groupby("svi_q", observed=True)["chain"].apply(lambda c: round(1 - c.mean(), 4)).to_dict()
        R["snap_indep_share_by_colonia"] = S.groupby("colonia_1km")["chain"].apply(lambda c: round(1 - c.mean(), 4)).to_dict()
        # double blindness: where the business reference misses owner-run firms, is the map also missing stores?
        if "retail_nonemp_per_employer" in I.columns:
            cc = I.groupby("county").agg(n=("missing", "size"), miss=("missing", "mean"),
                                         ratio=("retail_nonemp_per_employer", "first"), name=("county_name", "first"))
            cc = cc[(cc["n"] >= 10) & cc["ratio"].notna()]
            from scipy.stats import spearmanr
            rho, p = spearmanr(cc["ratio"], cc["miss"])
            R["county_nonemp_vs_missing"] = {"counties": int(len(cc)), "spearman_rho": round(float(rho), 3), "p": float(p),
                                             "top_ratio": cc.sort_values("ratio", ascending=False).head(12).round(3).reset_index().to_dict("records")}
        if "snap_hh_share" in S.columns:
            R["snap_stores_in_tracts_30pct_snap"] = table(I[I["snap_hh_share"] >= 0.30], "group").to_dict("records")
        # counties with the most missing independent stores
        cty = I.groupby("county").agg(n=("missing", "size"), missing=("missing", "sum"),
                                      county_name=("county_name" if "county_name" in I else "city", "first"))
        cty["share"] = (cty["missing"] / cty["n"]).round(3)
        R["snap_counties"] = cty[cty["n"] >= 15].sort_values("share", ascending=False).head(25).reset_index().to_dict("records")
        ex = I[I["missing"]].copy()
        ex["prio"] = ex["colonia_1km"].astype(int) * 2 + ex["tribal"].astype(int) * 2 + (ex["svi"].fillna(0) >= 0.9).astype(int)
        ex = ex.sort_values(["prio", "svi"], ascending=False)
        keep = ["ref_id", "name", "rtype", "hnum", "street", "city", "zip5", "lat", "lon", "GEOID", "region", "svi", "tribal",
                "aiannh_name", "colonia_1km", "colonia_name", "d_colonia_km", "near_name", "near_cat", "d_near", "s_any",
                "many_name", "d_any", "kind_name", "kind_cat"]
        ex[[c for c in keep if c in ex.columns]].head(400).to_csv(os.path.join(out, "snap_missing_examples.csv"), index=False)
    N = M[M["ref_set"] == "NH"].copy()
    if len(N):
        for by in ["region", "rural", "tribal", "svi_q"]:
            R[f"nh_by_{by}"] = table(N, by).to_dict("records")
        N["beds"], N["residents"] = num(N.get("beds")), num(N.get("residents"))
        R["nh_beds_missing"] = {"facilities": int(N["missing"].sum()), "beds": float(N.loc[N["missing"], "beds"].sum()),
                                "residents": float(N.loc[N["missing"], "residents"].sum()), "of_facilities": int(len(N))}
        N["whp_q4"] = num(N["usfs_WHP_mean"]) >= num(N["usfs_WHP_mean"]).quantile(0.75)
        R["nh_by_wildfire_q4"] = table(N, "whp_q4").to_dict("records")
        lt, n = logit(N, "missing_i ~ rural + svi + tribal + log_dens + C(region)")
        R["nh_logit"] = {"n": n, "table": lt.reset_index().to_dict("records")}
        N[N["missing"]].to_csv(os.path.join(out, "nh_missing.csv"), index=False)
    H = M[M["ref_set"] == "HC"].copy()
    if len(H):
        for by in ["region", "rural", "tribal", "svi_q", "setting"] if "setting" in H else ["region", "rural", "tribal", "svi_q"]:
            R[f"hc_by_{by}"] = table(H, by).to_dict("records")
        H[H["missing"]].to_csv(os.path.join(out, "hc_missing.csv"), index=False)
    M.to_csv(os.path.join(out, "matches_scored.csv"), index=False)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    return M, R

if __name__ == "__main__":
    M, R = main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    for k, v in R.items():
        if isinstance(v, list) and v and len(v) <= 40:
            print(f"\n== {k}"); print(pd.DataFrame(v).to_string(index=False))
        elif isinstance(v, dict) and "table" in v:
            print(f"\n== {k} (n={v['n']})"); print(pd.DataFrame(v["table"]).to_string(index=False))
        else:
            print(f"\n== {k}: {v if not isinstance(v, list) else str(v)[:300]}")
