# The corner store gap: Bias Bounty, Mapping Equity Challenge (Zindi, 2026)

Code and results for our entry to the Humane Intelligence Bias Bounty on Zindi:

1. **Best Bias Discovery.** Stores that accept SNAP food benefits are missing from Overture more often
   where social vulnerability is highest. Of 25,221 SNAP storefronts in the five study regions, 9.2% have
   no matching Overture place: 5.9% in the least vulnerable quarter of tracts (CDC SVI) and 10.5% in the
   most vulnerable. Chain stores, the control, stay at 4 to 5% missing everywhere; independent stores are
   missing 15.3% of the time and make up half of SNAP stores in the most vulnerable tracts.
2. **Best Documentation.** A label-free estimate of the coverage gap score from permitted data, calibrated
   with three component probes (public RMSE 0.0739).

Zindi user: AsfandyarFarooqi. Scored submissions: 6AMWPqPS (blend A, public RMSE 0.073886708) and tDuujvN9
(blend B, public RMSE 0.074494132). Write-up: [docs/Corner_Store_Gap_Bias_Bounty_Writeup.pdf](docs/Corner_Store_Gap_Bias_Bounty_Writeup.pdf).

## What is here

| Path | What it does | Runs where |
| --- | --- | --- |
| `notebooks/bias_bounty_features_v2.ipynb` | Per-tract Overture measurements for the five regions (buildings by source and size, places by category, roads by class, name and route network, cut to tract polygons in EPSG:5070) | Google Colab, about 25 min |
| `notebooks/bias_bounty_lifelines.ipynb` | The registry audit: downloads the USDA SNAP retailer list (current service and 2005 to 2025 history) and matches every store against Overture places by name, street address and distance | Google Colab, about 5 min |
| `src/bb_core.py` | Readers and per-tract aggregation used by both notebooks (embedded in them) | |
| `src/audit_core.py` | Registry download, normalisation and matching (embedded in the lifelines notebook) | |
| `analysis/physics2.py` | Expected-value model of each score component from permitted data | local |
| `analysis/blend_run.py` | Leaderboard calibration: probe scores to blend weights to submission files | local |
| `lifelines/analyze.py` | Rates, tract-bootstrap intervals and logistic regressions for the audit | local |
| `lifelines/final_numbers.py` | Every number quoted in the write-up, written to `results/final_numbers.json` | local |
| `lifelines/robust.py` | The finding under strict, main and lenient matching rules | local |
| `lifelines/audit_review.py` | Hand-check sheet with Google Maps, Overture Explorer and OpenStreetMap links | local |
| `lifelines/make_map.py` | Map of missing stores for one area | local |
| `results/` | `final_numbers.json`, the SNAP match table (`snap_registry_matches.parquet`), the audit run log and `lifelines_sources.json` (URL, retrieval time and SHA-256 of each download), and `blend_record.json` (probe scores, weights, SHA-256 of both scored files) | |
| `probes/` | The component pieces and probe manifest used by `blend_run.py` | |

## Reproduce

1. Open each notebook in Google Colab and choose Runtime, then Run all. Both read the public challenge
   bucket on Source Cooperative anonymously.
2. Put the notebook outputs, the challenge `SampleSubmission.csv`, `strata.csv`, `features.csv`,
   `tracts.parquet` and the HUD Colonias Communities GeoJSON in `data/`.
3. `BB_MATCHES=<lifelines_registry_matches.csv> python3 lifelines/final_numbers.py`
4. `python3 analysis/blend_run.py '{"p1_transport.csv": 0.093959899, "p3_fire_ems.csv": 0.1009168, "p5_business.csv": 0.103158904}'`

Paths can be changed with the environment variables `BB_DATA`, `BB_PROBES`, `BB_STRATA`, `BB_TRACTS`,
`BB_COLONIAS`, `BB_MATCHES` and `BB_OUT`. The only randomness is a seeded tract bootstrap.

## Matching rule (main)

A SNAP store is on the map when an Overture place carries its name within 150 m (name score 80 or more)
or 500 m (90 or more; chains within 250 m), when a distinctive non-chain name appears within 3 km (95 or
more), or when an Overture place sits at the same house number, street and ZIP code with a similar name
or a food category. Names are normalised (accents, store numbers, chain spellings) and scored with
rapidfuzz; a match on generic words only (market, food, store) does not count. The strict and lenient
variants, and the results under each, are in the write-up.

## Compliance

The scored file uses only the challenge package and aggregate statistics of our own public scores. The
withdrawn reference layers (TIGER roads, Microsoft footprints, HIFLD, CBP) are not used to build, fit or
calibrate it. The bias audit uses outside public data, as the rules allow for the Bias Discovery prize:
USDA SNAP retailer data and HUD Colonias Communities, each listed with its URL and retrieval date.

## License

MIT for the code. USDA and HUD data are U.S. federal public data.
