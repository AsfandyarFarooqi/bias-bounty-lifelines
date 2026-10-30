# Manual check sheet: a random sample of "missing" and "found" registry entries, with the Overture
# candidates around each one and links a person can open to verify it by eye.
# Usage: python3 audit_review.py <matches_scored.csv> <lifelines_audit_candidates.csv> <out_csv> [n_missing] [n_found]
import sys, urllib.parse
import numpy as np, pandas as pd

def links(r):
    q = urllib.parse.quote_plus(f"{r['name']} {r.get('hnum', '') or ''} {r.get('street', '') or ''} {r.get('city', '') or ''} {r.get('zip5', '') or ''}")
    lat, lon = r["lat"], r["lon"]
    return (f"https://www.google.com/maps/search/?api=1&query={q}",
            f"https://explore.overturemaps.org/#17/{lat:.5f}/{lon:.5f}",
            f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=18/{lat:.5f}/{lon:.5f}")

def main(scored, audit, out, n_miss=60, n_found=20, seed=31):
    M = pd.read_csv(scored, dtype={"GEOID": str, "ref_id": str, "zip5": str, "hnum": str}, low_memory=False)
    A = pd.read_csv(audit, dtype={"ref_id": str}, low_memory=False) if audit else pd.DataFrame()
    rng = np.random.default_rng(seed)
    have = set(A["ri"]) if len(A) else set()
    rows = []
    for flag, n in ((True, n_miss), (False, n_found)):
        pool = M[(M["missing"] == flag) & M["ri"].isin(have)] if have else M[M["missing"] == flag]
        if "ref_set" in pool:
            pool = pool[pool["ref_set"] == "SNAP"]
        take = pool.sample(min(n, len(pool)), random_state=seed)
        for _, r in take.iterrows():
            g, o, s = links(r)
            near = named = ""
            if len(A):
                cands = A[A["ri"] == r["ri"]].sort_values(["kind", "d"])
                near = "; ".join(f"{c.place} ({c.cat}, {c.d:.0f} m, s={c.s:.0f})" for c in cands[cands["kind"] == "near"].head(5).itertuples())
                named = "; ".join(f"{c.place} ({c.cat}, {c.d:.0f} m, s={c.s:.0f})" for c in cands[cands["kind"] == "name"].head(3).itertuples())
            rows.append({"verdict_by_matcher": "missing" if flag else "found", "ref_id": r["ref_id"], "store": r["name"],
                         "type": r.get("rtype", ""), "chain": r.get("chain", ""),
                         "address": f"{r.get('hnum', '') or ''} {r.get('street', '') or ''}, {r.get('city', '') or ''} {r.get('zip5', '') or ''}".strip(),
                         "tract": r["GEOID"], "region": r["region"], "nearest_overture_places": near, "best_name_candidates": named,
                         "google_maps": g, "overture_explorer": o, "openstreetmap": s,
                         "store_exists_on_google (yes/no/unsure)": "", "store_visible_in_overture_explorer (yes/no)": "", "note": ""})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2] if len(a) > 2 and a[2] != "-" else None, a[3], int(a[4]) if len(a) > 4 else 60, int(a[5]) if len(a) > 5 else 20)
