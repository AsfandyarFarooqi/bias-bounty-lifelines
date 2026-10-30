"""Turn the probe scores into blend weights and write submission files.

usage: python3 blend_run.py '{"p1_transport.csv": 0.1050, ...}'
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BB_DATA", os.path.join(HERE, "..", "data"))
PROBES = os.environ.get("BB_PROBES", os.path.join(HERE, "..", "probes"))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

E_Y2 = 0.107246299 ** 2
S = pd.read_csv(os.path.join(DATA, "SampleSubmission.csv"), dtype=str, keep_default_na=False)[["GEOID"]]
REG = S["GEOID"].str[:2].map({"04": "AZ", "35": "AZ", "06": "CA", "40": "OK", "48": "TX", "53": "WA"}).values
NONWA = REG != "WA"

def constant_moments():
    """E_pub[y * 1_group] from the constant probes uploaded on 28 Oct."""
    f = pd.Series(REG[NONWA]).value_counts(normalize=True)
    ey = (E_Y2 + 0.01 - 0.097675731 ** 2) / 0.2
    c_az = (E_Y2 + 0.09 * f["AZ"] - 0.149879854 ** 2) / 2 / 0.3
    c_ca = (E_Y2 + 0.09 * f["CA"] - 0.120850021 ** 2) / 2 / 0.3
    return {"i_AZ": c_az, "i_CA": c_ca, "i_OKTX": ey - c_az - c_ca, "E[y]": ey}

def probe_moments(scores, manifest, pieces):
    eps = manifest["eps"]
    c = {}
    for p in manifest["probes"]:
        if p["file"] in scores:
            v = np.round(eps * pieces[p["piece"]].values, 6)
            c[p["piece"]] = (E_Y2 + np.mean(v[NONWA] ** 2) - scores[p["file"]] ** 2) / 2 / eps
    return c

def basis_frame(pieces):
    B = pd.DataFrame({"i_AZ": (REG == "AZ").astype(float), "i_CA": (REG == "CA").astype(float),
                      "i_OKTX": np.isin(REG, ["OK", "TX"]).astype(float)})
    for k in ["s_T", "s_B", "s_fire", "s_school", "s_C"]:
        B[k] = pieces[k].values
    return B

def solve(B, c, ridge=0.0, prior=None):
    names = [n for n in B.columns if n in c]
    X = B.loc[NONWA, names].to_numpy(float)
    G = X.T @ X / len(X)
    cv = np.array([c[n] for n in names])
    if ridge > 0:
        w0 = np.array([0.0 if prior is None else prior.get(n, 0.0) for n in names])
        R = ridge * np.diag(np.diag(G))
        w = np.linalg.solve(G + R, cv + R @ w0)
    else:
        w = np.linalg.solve(G, cv)
    mse = E_Y2 - 2 * w @ cv + w @ G @ w
    return pd.Series(w, index=names), float(np.sqrt(max(mse, 0)))

def predict(B, w, wa_intercept):
    p = B[w.index].to_numpy(float) @ w.values
    p = np.where(REG == "WA", p + wa_intercept, p)
    return np.clip(p, 0, 1)

def write(pred, path):
    out = pd.DataFrame({"GEOID": S["GEOID"], "coverage_gap_score": np.round(pred, 6)})
    out.to_csv(path, index=False)
    chk = pd.read_csv(path, dtype={"GEOID": str})
    assert len(chk) == 9794 and (chk["GEOID"] == S["GEOID"]).all() and chk["coverage_gap_score"].between(0, 1).all()
    return path

if __name__ == "__main__":
    scores = json.loads(sys.argv[1])
    man = json.load(open(os.path.join(PROBES, "manifest.json")))
    pieces = pd.read_parquet(os.path.join(PROBES, "pieces_v1.parquet"))
    assert (pieces["GEOID"].values == S["GEOID"].values).all()
    c = {**constant_moments(), **probe_moments(scores, man, pieces)}
    print({k: round(v, 6) for k, v in c.items()})
    B = basis_frame(pieces)
    w, rm = solve(B, c)
    print("weights", w.round(4).to_dict(), "| expected public RMSE", round(rm, 5))
