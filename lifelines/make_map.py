# Map of SNAP stores Overture is missing, for one focus area, as a PNG for the write-up.
# Usage: python3 make_map.py <matches_scored.csv> <out.png> <xmin> <ymin> <xmax> <ymax> "<title>" "<subtitle>"
import sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
TRACTS = os.environ.get("BB_TRACTS", "data/tracts.parquet")
COLONIAS = os.environ.get("BB_COLONIAS", "data/hud_colonias_communities.geojson")
GREY, LIGHT, ACCENT, INK, MUTED = "#9a9a9a", "#e4e2dc", "#2f6fdf", "#1f1f1f", "#6b6b6b"

def draw(scored, out, bbox, title, subtitle, only_independent=True):
    M = pd.read_csv(scored, dtype={"GEOID": str}, low_memory=False)
    M = M[M["ref_set"] == "SNAP"]
    if only_independent and "chain" in M:
        M = M[~M["chain"].astype(str).str.lower().isin(["true", "1"])]
    x0, y0, x1, y1 = bbox
    M = M[(M["lon"].between(x0, x1)) & (M["lat"].between(y0, y1))]
    T = gpd.read_parquet(TRACTS).cx[x0:x1, y0:y1]
    C = gpd.read_file(COLONIAS).to_crs(4326).cx[x0:x1, y0:y1]
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=200)
    T.boundary.plot(ax=ax, color=LIGHT, linewidth=0.4, zorder=1)
    if len(C):
        ax.scatter(C.geometry.x, C.geometry.y, s=9, marker="^", facecolor="none", edgecolor="#b08d57", linewidth=0.6,
                   zorder=2, label=f"HUD colonia ({len(C):,})")
    f, m = M[~M["missing"].astype(bool)], M[M["missing"].astype(bool)]
    ax.scatter(f["lon"], f["lat"], s=6, color=GREY, alpha=0.7, linewidth=0, zorder=3,
               label=f"On the map ({len(f):,})")
    ax.scatter(m["lon"], m["lat"], s=10, color=ACCENT, linewidth=0, zorder=4,
               label=f"Missing from Overture ({len(m):,})")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)))
    ax.axis("off")
    fig.text(0.02, 0.965, title, fontsize=12.5, color=INK, ha="left", va="top", fontweight="bold")
    fig.text(0.02, 0.925, subtitle, fontsize=9, color=MUTED, ha="left", va="top")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK, handletextpad=0.3)
    fig.text(0.02, 0.015, "Independent SNAP-authorized stores (USDA). Overture places release 2026-08-19.0. "
             "Colonias: HUD Colonias Communities. Tract lines: challenge census tracts.", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05)
    fig.savefig(out, dpi=200)
    print("wrote", out, len(f), len(m), len(C))

if __name__ == "__main__":
    a = sys.argv
    draw(a[1], a[2], tuple(float(v) for v in a[3:7]), a[7], a[8])
