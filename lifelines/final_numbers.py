import os
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as az
RNG = np.random.default_rng(2026)
M = az.load(os.environ.get("BB_MATCHES", "results/lifelines_registry_matches.csv"))
S = M[M["ref_set"] == "SNAP"].copy()
ad = pd.to_datetime(S["auth_date"], errors="coerce")
S = S[ad.notna() & (ad <= "2025-06-30") & ~S["stype"].isin(["farmers market", "other"])].copy()
S["group"] = np.where(S["chain"], "chain", "independent")
out = {"n_stores": int(len(S)), "n_missing": int(S.missing.sum()), "share_missing": round(float(S.missing.mean()), 4),
       "n_tracts_with_stores": int(S.GEOID.nunique())}
def ci(g, col="missing"):
    e, lo, hi, n = az.rate_ci(g, col, reps=2000)
    return {"n": n, "share": round(e, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
out["by_group"] = {k: ci(g) for k, g in S.groupby("group")}
out["by_svi"] = {str(k): ci(g) for k, g in S.groupby("svi_q", observed=True)}
out["by_svi_group"] = {f"{k[0]}|{k[1]}": ci(g) for k, g in S.groupby(["svi_q", "group"], observed=True)}
out["indep_share_by_svi"] = {str(k): round(float(1 - g.chain.mean()), 4) for k, g in S.groupby("svi_q", observed=True)}
# ratio Q4/Q1 with a tract bootstrap
q1, q4 = S[S.svi_q == "Q1 (least)"], S[S.svi_q == "Q4 (most)"]
def boot_ratio(a, b, reps=2000):
    A = a.groupby("GEOID")["missing"].agg(["sum", "count"]); B = b.groupby("GEOID")["missing"].agg(["sum", "count"])
    ra = RNG.integers(0, len(A), (reps, len(A))); rb = RNG.integers(0, len(B), (reps, len(B)))
    pa_ = A["sum"].values[ra].sum(1) / A["count"].values[ra].sum(1); pb_ = B["sum"].values[rb].sum(1) / B["count"].values[rb].sum(1)
    r = pb_ / pa_
    return round(float(b.missing.mean() / a.missing.mean()), 3), round(float(np.quantile(r, 0.025)), 3), round(float(np.quantile(r, 0.975)), 3)
out["ratio_q4_q1_all"] = boot_ratio(q1, q4)
out["ratio_q4_q1_indep"] = boot_ratio(q1[~q1.chain], q4[~q4.chain])
out["ratio_q4_q1_chain"] = boot_ratio(q1[q1.chain], q4[q4.chain])
# rules
rules = {}
for col in ["found_strict", "found", "found_loose"]:
    d = S.assign(m=~S[col])
    rules[col] = {f"{q}|{g}": round(float(x.m.mean()), 4) for (q, g), x in d.groupby(["svi_q", "group"], observed=True)}
    rules[col].update({f"{q}|all": round(float(x.m.mean()), 4) for q, x in d.groupby("svi_q", observed=True)})
out["rules"] = rules
# urban only
U = S[~S.rural]
out["urban_by_svi_group"] = {f"{k[0]}|{k[1]}": round(float(g.missing.mean()), 4) for k, g in U.groupby(["svi_q", "group"], observed=True)}
out["urban_n"] = int(len(U))
# within region
I = S[~S.chain]
out["indep_region_svi"] = {f"{k[0]}|{k[1]}": [int(len(g)), round(float(g.missing.mean()), 4)] for k, g in I.groupby(["region", "svi_q"], observed=True)}
# long-standing
L = S[pd.to_datetime(S.auth_date, errors="coerce") <= "2021-06-30"]
out["pre2021_by_svi_group"] = {f"{k[0]}|{k[1]}": round(float(g.missing.mean()), 4) for k, g in L.groupby(["svi_q", "group"], observed=True)}
# types
out["by_type"] = {k: ci(g) for k, g in S.groupby("stype")}
out["by_type_group"] = {f"{k[0]}|{k[1]}": ci(g) for k, g in S.groupby(["stype", "group"])}
# logit
lt, n = az.logit(I, "missing_i ~ C(stype) + rural + svi + tribal + colonia_1km + log_dens + C(region)")
out["logit_indep"] = {"n": n, "svi": lt.loc["svi"].to_dict(), "rural": lt.loc["rural"].to_dict(), "tribal": lt.loc["tribal"].to_dict(),
                      "colonia_1km": lt.loc["colonia_1km"].to_dict(), "log_dens": lt.loc["log_dens"].to_dict()}
lt2, n2 = az.logit(S, "missing_i ~ chain + C(stype) + rural + svi + tribal + colonia_1km + log_dens + C(region)")
out["logit_all"] = {"n": n2, "chain": lt2.loc["chain"].to_dict(), "svi": lt2.loc["svi"].to_dict()}
# people
T = S.groupby("GEOID").agg(n=("missing", "size"), miss=("missing", "sum"), pop=("pop_total", "first"), svi_q=("svi_q", "first"))
out["tracts_any_missing"] = int((T.miss > 0).sum()); out["pop_any_missing"] = float(T.loc[T.miss > 0, "pop"].sum())
out["tracts_all_missing"] = int((T.miss == T.n).sum()); out["pop_all_missing"] = float(T.loc[T.miss == T.n, "pop"].sum())
out["tracts_all_missing_by_svi"] = {str(k): int(v) for k, v in T[T.miss == T.n].groupby("svi_q", observed=True).size().items()}
out["missing_by_svi_count"] = {str(k): int(v) for k, v in S.groupby("svi_q", observed=True)["missing"].sum().items()}
F = pd.read_csv(os.path.join(os.environ.get("BB_DATA", "data"), "features.csv"), dtype={"GEOID": str}, usecols=["GEOID", "poi_n"]).set_index("GEOID")
T = T.join(F)
T = T[T["pop"] > 0].assign(per1k=lambda d: 1000 * d.poi_n / d["pop"])
out["places_per1k_tracts_with_missing"] = round(float(T[T.miss > 0].per1k.median()), 1)
out["places_per1k_tracts_without_missing"] = round(float(T[T.miss == 0].per1k.median()), 1)
json.dump(out, open(os.environ.get("BB_OUT", "results/final_numbers.json"), "w"), indent=1, default=str)
for k, v in out.items():
    print(k, ":", v if not isinstance(v, dict) or len(str(v)) < 400 else json.dumps(v)[:900])
