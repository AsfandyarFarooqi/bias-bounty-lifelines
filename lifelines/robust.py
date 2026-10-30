import os
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as az
M = az.load(os.environ.get("BB_MATCHES", "results/lifelines_registry_matches.csv"))
S = M[M["ref_set"] == "SNAP"].copy()
ad = pd.to_datetime(S["auth_date"], errors="coerce")
S["stable"] = ad.notna() & (ad <= "2025-06-30")
S = S[S["stable"] & ~S["stype"].isin(["farmers market", "other"])]
S["group"] = np.where(S["chain"], "chain", "independent")
print("n", len(S))
for col in ["found_strict", "found", "found_loose"]:
    S["miss_" + col] = ~S[col]
    t = S.groupby(["svi_q", "group"], observed=True)["miss_" + col].mean().unstack().round(4)
    t["all"] = S.groupby("svi_q", observed=True)["miss_" + col].mean().round(4)
    print("\n== missing under", col); print(t.to_string())
# long-standing stores only
ad2 = pd.to_datetime(S["auth_date"], errors="coerce")
L = S[ad2 <= "2021-06-30"]
print("\n== authorized before mid-2021 (n=%d)" % len(L))
print(L.groupby(["svi_q", "group"], observed=True)["missing"].mean().unstack().round(4).to_string())
# same-spot share among misses by SVI
mm = S[S["missing"]]
print("\n== among misses: share with a food place of another name within 50 m, by SVI")
print(mm.groupby("svi_q", observed=True)["kind_50"].apply(lambda k: (k > 0).mean()).round(3).to_string())
print("\n== among misses: nearest Overture place distance (median m), by SVI")
print(mm.groupby("svi_q", observed=True)["d_near"].median().round(1).to_string())
print("\n== n Overture places within 150 m of the store (median), by SVI x group")
print(S.groupby(["svi_q", "group"], observed=True)["n_150"].median().unstack().to_string())
# chain detail: which chains miss most
c = S[S["chain"]].groupby("nname").agg(n=("missing", "size"), miss=("missing", "mean")).sort_values("n", ascending=False)
print("\n== top chains"); print(c.head(25).round(3).to_string())
