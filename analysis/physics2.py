"""Expected coverage-gap components per tract (v2, real-data calibrated, permitted data only).

Score (organisers' README): y = mean of the defined components among
  T  transport  1 - min(1, Overture motorway..secondary km / TIGER S1100+S1200 km), both cut to the tract;
                undefined where TIGER has no highway in the tract
  B  building   1 - min(1, Overture footprints / Microsoft footprints), counted by bbox centre;
                undefined where Microsoft has no footprint
  P  places     mean of the defined halves: H (mean over defined facility types of
                1 - min(1, Overture / reference), fire / EMS / schools) and C (all places vs CBP)
Every reference count is replaced by an expectation built from Overture and the strata tables.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()
SCHOOL_CATS = ["elementary_school", "middle_school", "high_school", "school", "private_school", "public_school"]

P = dict(
    # transport: TIGER-like length = core routes + a_sub * state sub-systems + border share
    a_sub={"AZ": 1.0, "CA": 1.0, "OK": 1.0, "TX": 0.45, "WA": 1.0},
    border_w=0.8, sigma_T=0.35, def_scale=150.0,
    # building: Microsoft count ~ kappa * USFS building count
    kappa_B=1.03, sigma_B=0.12,
    # facilities
    fire_pop=11500.0, fire_area=390.0, fire_miss=0.45, fire_fp=0.12,
    ems_def_seen=0.7, ems_def_unseen=0.03,
    school_pop=2700.0, school_area=4000.0, school_miss=0.15, school_fp=0.30,
    # establishments (CBP half)
    kappa_C=0.55, sigma_C=0.6, w_pop=0.5,
)

def num(f, c):
    return pd.to_numeric(f[c], errors="coerce").fillna(0).values if c in f else np.zeros(len(f))

def exp_gap(o, m, sigma):
    """E[1 - min(1, o / (m * eps))], eps ~ LogNormal(0, sigma); 0 where m <= 0."""
    o = np.asarray(o, float)[:, None]
    m = np.asarray(m, float)[:, None]
    ref = m * np.exp(sigma * GH_X[None, :])
    g = np.where(ref > 0, 1 - np.minimum(1, o / np.maximum(ref, 1e-12)), 0.0)
    return (g * GH_W[None, :]).sum(1)

def gap_given_present(lam, o, hmax=15):
    lam = np.maximum(np.asarray(lam, float), 1e-6)
    o = np.asarray(o, float)
    hs = np.arange(1, hmax + 1)
    ph = poisson.pmf(hs[None, :], lam[:, None])
    ph = ph / np.maximum(ph.sum(1, keepdims=True), 1e-12)
    return (ph * (1 - np.minimum(1, o[:, None] / hs[None, :]))).sum(1)

def presence(prior, o, miss, fp):
    has = np.asarray(o) >= 1
    p1 = (1 - miss) * prior / ((1 - miss) * prior + fp * (1 - prior) + 1e-12)
    p0 = miss * prior / (miss * prior + (1 - fp) * (1 - prior) + 1e-12)
    return np.where(has, p1, p0)

def components(f, p=P):
    reg = f["state_region"].values
    out = pd.DataFrame(index=f.index)
    pop, aland = num(f, "pop_total"), num(f, "ALAND") / 1e6
    # ---------------- transport
    a_sub = pd.Series(reg).map(p["a_sub"]).fillna(1.0).values
    core = num(f, "cut_rt_interstate_m") + num(f, "cut_rt_us_m") + num(f, "cut_rt_state_m") + a_sub * num(f, "cut_rt_state_sub_m")
    bcore = num(f, "buf_rt_interstate_m") + num(f, "buf_rt_us_m") + num(f, "buf_rt_state_m") + a_sub * num(f, "buf_rt_state_sub_m")
    border = np.maximum(0, bcore - core)
    tiger = core + p["border_w"] * border
    out["tiger_hat_m"] = tiger
    out["p_T"] = 1 - np.exp(-(core + 0.5 * border) / p["def_scale"])
    out["T"] = exp_gap(num(f, "cut_major_m"), tiger, p["sigma_T"])
    # ---------------- building
    ms = p["kappa_B"] * num(f, "usfs_BuildingCount_sum")
    out["p_B"] = 1 - np.exp(-ms)            # Microsoft has at least one footprint
    out["B"] = exp_gap(num(f, "bld_n"), ms, p["sigma_B"])
    # ---------------- facilities (fire and EMS correlated, schools independent)
    o_f = num(f, "poi_cat_fire_department")
    o_e = num(f, "poi_cat_ambulance_and_ems_services")
    o_s = sum(num(f, f"poi_cat_{c}") for c in SCHOOL_CATS)
    lam_f = pop / p["fire_pop"] + aland / p["fire_area"]
    lam_s = pop / p["school_pop"] + aland / p["school_area"]
    pf = presence(1 - np.exp(-lam_f), o_f, p["fire_miss"], p["fire_fp"])
    pe = np.where(o_e >= 1, p["ems_def_seen"], p["ems_def_unseen"])
    ps = presence(1 - np.exp(-lam_s), o_s, p["school_miss"], p["school_fp"])
    G = {"fire": np.where(o_f >= 1, gap_given_present(lam_f, o_f), 1.0),
         "ems": np.where(o_e >= 1, gap_given_present(np.full(len(f), 0.3), o_e), 1.0),
         "school": np.where(o_s >= 1, gap_given_present(lam_s, o_s), 1.0)}
    probs = {"fire": pf, "ems": pe, "school": ps}
    contrib = {k: np.zeros(len(f)) for k in G}
    p_none = np.ones(len(f))
    for F in (0, 1):
        for E in (0, 1):
            for S in (0, 1):
                pr = (pf if F else 1 - pf) * (pe if E else 1 - pe) * (ps if S else 1 - ps)
                n = F + E + S
                if n == 0:
                    p_none = pr
                    continue
                for k, on in (("fire", F), ("ems", E), ("school", S)):
                    if on:
                        contrib[k] = contrib[k] + pr * G[k] / n
    for k in G:
        out[f"H_{k}"] = contrib[k]
    out["p_H"] = 1 - p_none
    out["p_fire"], out["p_school"], out["p_ems"] = pf, ps, pe
    # ---------------- establishments (CBP half): county establishments ~ kappa * county Overture places,
    # spread within the county half by population, half by large (non-residential-size) buildings
    cty = f["GEOID"].str[:5].values
    o_all = num(f, "poi_n")
    big = num(f, "bld_n_500_2000") + 3 * num(f, "bld_n_ge2000")
    df = pd.DataFrame({"cty": cty, "o": o_all, "pop": pop, "big": big})
    tot = df.groupby("cty")[["o", "pop", "big"]].transform("sum")
    share = p["w_pop"] * df["pop"] / tot["pop"].replace(0, np.nan) + (1 - p["w_pop"]) * df["big"] / tot["big"].replace(0, np.nan)
    est = (p["kappa_C"] * tot["o"] * share.fillna(0)).values
    out["p_C"] = (est >= 0.5).astype(float)
    out["C"] = exp_gap(o_all, est, p["sigma_C"])
    return out

def pieces(comp):
    """Additive contributions to E[y]; B and P taken as (nearly) always defined, T with probability p_T."""
    pT, pB, pH, pC = comp["p_T"].values, comp["p_B"].values, comp["p_H"].values, comp["p_C"].values
    n3, n2 = 1 / 3, 1 / 2
    inv_n = pT * n3 + (1 - pT) * n2
    s = pd.DataFrame(index=comp.index)
    s["s_T"] = pT * comp["T"].values * n3
    s["s_B"] = pB * comp["B"].values * inv_n
    w_h = pC / 2 + (1 - pC)          # H's weight inside P
    s["s_fire"] = (comp["H_fire"].values + comp["H_ems"].values) * w_h * inv_n
    s["s_school"] = comp["H_school"].values * w_h * inv_n
    s["s_C"] = comp["C"].values * pC * (1 - pH / 2) * inv_n
    s["inv_n"] = inv_n
    return s
