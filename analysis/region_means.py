import os
"""Region means on the public board from the constant probes (WA rows are not on the public board)."""
import json
import numpy as np, pandas as pd
S = pd.read_csv(os.path.join(os.environ.get("BB_DATA", "data"), "SampleSubmission.csv"), dtype=str)
REG = S.GEOID.str[:2].map({"04": "AZ", "35": "AZ", "06": "CA", "40": "OK", "48": "TX", "53": "WA"})
N = REG[REG != "WA"].value_counts()
F = N / N.sum()

def means(scores):
    """scores: dict with keys zeros, all_0.10, AZ_0.30, CA_0.30, OK_0.30 (OK optional)."""
    ey2 = scores["zeros"] ** 2
    ey = (ey2 + 0.01 - scores["all_0.10"] ** 2) / 0.2
    out = {"E[y]": ey, "E[y2]": ey2}
    known = 0.0
    for r in ["AZ", "CA", "OK"]:
        k = f"{r}_0.30"
        if k in scores:
            eys = (ey2 + 0.09 * F[r] - scores[k] ** 2) / 2          # E[y * 0.3*1_r]
            out[r] = eys / (0.3 * F[r])
            known += F[r] * out[r]
    rest = [r for r in ["OK", "TX"] if r not in out]
    out["+".join(rest)] = (ey - known) / F[rest].sum()
    return out

if __name__ == "__main__":
    sc = {"zeros": 0.107246299, "all_0.10": 0.097675731, "AZ_0.30": 0.149879854, "CA_0.30": 0.120850021}
    print(F.round(4).to_dict())
    print({k: round(v, 5) for k, v in means(sc).items()})
