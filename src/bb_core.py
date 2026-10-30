# Bias Bounty feature extraction, version 2 (29 Oct 2026).
# Runs in Google Colab against the public Source Cooperative bucket (no login), and locally
# against synthetic test files. Every layer is read in batches so memory stays low.
#
# Per census tract it measures what Overture has: buildings by source and size, places by
# category (incl. the exact categories the score uses), and roads by class, name and route
# network. Major roads and numbered routes are CUT to the tract polygon (exact lengths), plus
# a second cut against the tract grown by 30 m, so roads running along tract borders are visible.
import os, re, gc, json, time, zipfile, traceback
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pyproj import Transformer

ROOT = "humane-intelligence/bias-bounty-mapping-equity-challenge"
BASE = f"us-west-2.opendata.source.coop/{ROOT}"
REGIONS = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx", "eastern-wa"]
BATCH = 250_000
CRS_M = 5070          # CONUS Albers, metres
BUF_M = 30.0          # border band for the second road cut
NEAR_M = 300.0        # how far to look for the nearest neighbouring tract of a key place
LOG = []
TO_M = Transformer.from_crs(4326, CRS_M, always_xy=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)

def ref_path(base, region, layer):
    return f"{base}/reference/{region}/{region}-{layer}.parquet"

def strata_path(base, region, name):
    return f"{base}/strata/{region}/{region}-{name}.parquet"

def open_pf(fs, path):
    f = fs.open_input_file(path)
    try:
        return pq.ParquetFile(f, pre_buffer=True)
    except TypeError:
        return pq.ParquetFile(f)

def project(geoms):
    """lon/lat shapely array to EPSG:5070 metres."""
    def f(c):
        x, y = TO_M.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geoms, f)

def bc(idx, n, w=None):
    """Sum per tract (fast np.add.at)."""
    if w is None:
        return np.bincount(idx, minlength=n).astype(float)
    return np.bincount(idx, weights=w, minlength=n)

# ---------------------------------------------------------------- geometry decoding (WKB or GeoArrow)
def geo_info(pf):
    md = pf.schema_arrow.metadata or {}
    try:
        geo = json.loads(md.get(b"geo", b"{}").decode())
    except Exception:
        geo = {}
    col = geo.get("primary_column", "geometry")
    enc = str((geo.get("columns", {}).get(col, {}) or {}).get("encoding", "WKB")).lower()
    return col, enc

def _coords(vals):
    t = vals.type
    if pa.types.is_struct(t):
        names = [t.field(i).name for i in range(t.num_fields)]
        ch = vals.flatten()
        x = np.asarray(ch[names.index("x")].to_numpy(zero_copy_only=False), dtype="float64")
        y = np.asarray(ch[names.index("y")].to_numpy(zero_copy_only=False), dtype="float64")
        return np.column_stack([x, y])
    if pa.types.is_fixed_size_list(t):
        d = t.list_size
        flat = np.asarray(vals.flatten().to_numpy(zero_copy_only=False), dtype="float64")
        return flat.reshape(-1, d)[:, :2]
    raise TypeError(f"unknown coordinate type {t}")

def _rebase(listarr):
    offs = np.asarray(listarr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    vals = listarr.values.slice(int(offs[0]), int(offs[-1] - offs[0]))
    return offs - offs[0], vals

def decode_geoms(arr, enc):
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if isinstance(arr, pa.ExtensionArray):
        arr = arr.storage
    t = arr.type
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return shapely.from_wkb(arr.to_numpy(zero_copy_only=False))
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return shapely.from_wkt(arr.to_numpy(zero_copy_only=False))
    GT = shapely.GeometryType
    if enc == "point":
        return shapely.points(_coords(arr))
    if enc == "linestring":
        o1, v = _rebase(arr)
        return shapely.from_ragged_array(GT.LINESTRING, _coords(v), (o1,))
    if enc == "multilinestring":
        o1, v = _rebase(arr); o2, vv = _rebase(v)
        return shapely.from_ragged_array(GT.MULTILINESTRING, _coords(vv), (o2, o1))
    if enc == "polygon":
        o1, v = _rebase(arr); o2, vv = _rebase(v)
        return shapely.from_ragged_array(GT.POLYGON, _coords(vv), (o2, o1))
    if enc == "multipolygon":
        o1, v = _rebase(arr); o2, vv = _rebase(v); o3, vvv = _rebase(vv)
        return shapely.from_ragged_array(GT.MULTIPOLYGON, _coords(vvv), (o3, o2, o1))
    raise TypeError(f"cannot decode geometry encoding {enc!r} with arrow type {t}")

# ---------------------------------------------------------------- nested-column helpers
def first_of_list_field(list_arr, field):
    """First element's `field` for each row of a list<struct> column; None where the list is null or empty."""
    if isinstance(list_arr, pa.ChunkedArray):
        list_arr = list_arr.combine_chunks()
    n = len(list_arr)
    out = np.full(n, None, dtype=object)
    flat = pc.list_flatten(list_arr)
    if len(flat) == 0:
        return out
    parents = np.asarray(pc.list_parent_indices(list_arr).to_numpy(zero_copy_only=False))
    vals = pc.struct_field(flat, field).to_numpy(zero_copy_only=False)
    uniq, first = np.unique(parents, return_index=True)
    out[uniq] = vals[first]
    return out

def list_nonempty(list_arr):
    if isinstance(list_arr, pa.ChunkedArray):
        list_arr = list_arr.combine_chunks()
    return np.asarray(pc.fill_null(pc.greater(pc.list_value_length(list_arr), 0), False).to_numpy(zero_copy_only=False))

def struct_str(arr, field):
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    v = pc.struct_field(arr, field)
    return np.asarray(pc.fill_null(v, "").to_numpy(zero_copy_only=False)).astype(str)

def first_source_dataset(batch):
    if "sources" not in batch.schema.names:
        return None
    ds = first_of_list_field(batch.column("sources"), "dataset")
    return np.array([("none" if v is None else str(v).lower()) for v in ds], dtype=object)

SRC_KEYS = {"osm": "openstreetmap", "msft": "microsoft", "google": "google", "esri": "esri"}

def src_bucket(ds):
    out = np.full(len(ds), "other", dtype=object)
    s = ds.astype(str)
    for k, pat in SRC_KEYS.items():
        out[np.char.find(s, pat) >= 0] = k
    return out

# ---------------------------------------------------------------- tracts
def load_tracts(fs, base, region):
    gdf = gpd.read_parquet(strata_path(base, region, "census-tracts"), filesystem=fs)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    gdf = gdf.reset_index(drop=True)
    g_ll = np.asarray(gdf.geometry.values, dtype=object)
    bad = ~shapely.is_valid(g_ll)
    if bad.any():
        g_ll[bad] = shapely.make_valid(g_ll[bad])
        log(f"{region}: repaired {int(bad.sum())} invalid tract polygons")
    g_m = project(g_ll)
    bad = ~shapely.is_valid(g_m)
    if bad.any():
        g_m[bad] = shapely.make_valid(g_m[bad])
    gdf["tract_area_km2_calc"] = shapely.area(g_m) / 1e6
    gdf["tract_perimeter_km"] = shapely.length(shapely.boundary(g_m)) / 1e3
    rp = shapely.point_on_surface(g_ll)
    gdf["cx"], gdf["cy"] = shapely.get_x(rp), shapely.get_y(rp)
    buf = shapely.buffer(g_m, BUF_M, quad_segs=2)
    bnd = shapely.boundary(g_m)
    for g in (g_m, buf):
        shapely.prepare(g)
    return {"gdf": gdf, "n": len(gdf), "tree": shapely.STRtree(g_ll), "g_m": g_m,
            "tree_m": shapely.STRtree(g_m), "buf": buf, "tree_buf": shapely.STRtree(buf), "bnd": bnd}

def assign(tree, xs, ys):
    """Index of the tract containing each point, -1 if none (first match on shared borders)."""
    pts = shapely.points(xs, ys)
    ip, it = tree.query(pts, predicate="intersects")
    out = np.full(len(xs), -1, dtype=np.int64)
    order = np.lexsort((it, ip))
    ip, it = ip[order], it[order]
    first = np.ones(len(ip), dtype=bool)
    first[1:] = ip[1:] != ip[:-1]
    out[ip[first]] = it[first]
    return out

def bbox_arrays(batch, gcol=None, enc="wkb"):
    """xmin, xmax, ymin, ymax per row: from the Overture bbox column, or from the geometry if it is missing."""
    if "bbox" in batch.schema.names:
        b = batch.column("bbox")
        get = lambda f: np.asarray(pc.struct_field(b, f).to_numpy(zero_copy_only=False), dtype="float64")
        return get("xmin"), get("xmax"), get("ymin"), get("ymax")
    bb = shapely.bounds(decode_geoms(batch.column(gcol), enc))
    return bb[:, 0], bb[:, 2], bb[:, 1], bb[:, 3]

def point_cols(pf, wanted):
    """Columns to read for a point-like layer; adds the geometry column when there is no bbox."""
    names = pf.schema_arrow.names
    gcol, enc = geo_info(pf)
    cols = [c for c in wanted if c in names]
    if "bbox" not in names:
        cols.append(gcol)
        log(f"{pf.schema_arrow.names[:3]}...: no bbox column, using geometry bounds")
    return cols, gcol, enc

def schema_note(fs, path):
    try:
        pf = open_pf(fs, path)
        return f"== {path}\nrows={pf.metadata.num_rows:,} row_groups={pf.metadata.num_row_groups}\n{pf.schema_arrow}\n"
    except Exception as e:
        return f"== {path}\nUNREADABLE: {type(e).__name__}: {e}\n"

# ---------------------------------------------------------------- buildings
SIZE_BINS = [("lt40", 0, 40), ("40_100", 40, 100), ("100_500", 100, 500), ("500_2000", 500, 2000), ("ge2000", 2000, 1e18)]
CLASS_GROUPS = {"residential": ["residential", "house", "apartments", "detached", "semidetached_house",
                                "terrace", "bungalow", "dormitory", "static_caravan"],
                "outbuilding": ["shed", "garage", "garages", "barn", "carport", "farm_auxiliary",
                                "outbuilding", "hut", "cabin", "greenhouse", "storage_tank"],
                "noclass": ["none"]}

def agg_buildings(fs, base, region, T):
    pf = open_pf(fs, ref_path(base, region, "overture-buildings"))
    cols, gcol, enc = point_cols(pf, ["bbox", "sources", "subtype", "class", "height", "num_floors"])
    n = T["n"]
    acc = {}
    def add(k, ii, w=None):
        acc[k] = acc.get(k, 0) + bc(ii, n, w)
    totals, rows = {}, 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        xmin, xmax, ymin, ymax = bbox_arrays(batch, gcol, enc)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        idx = assign(T["tree"], cx, cy)
        ok = idx >= 0
        ii = idx[ok]
        a = ((xmax - xmin) * 111320.0 * np.cos(np.radians(cy))) * ((ymax - ymin) * 110540.0)
        add("bld_n", ii); add("bld_bbox_area_m2", ii, a[ok])
        for nm, lo, hi in SIZE_BINS:
            m = ok & (a >= lo) & (a < hi)
            add(f"bld_n_{nm}", idx[m])
        ds = first_source_dataset(batch)
        if ds is not None:
            sb = src_bucket(ds)
            for k in list(SRC_KEYS) + ["other"]:
                m = ok & (sb == k)
                add(f"bld_n_src_{k}", idx[m]); add(f"bld_area_src_{k}", idx[m], a[m])
                m2 = m & (a >= 500)
                add(f"bld_n_src_{k}_ge500", idx[m2])
            for v, c in zip(*np.unique(ds.astype(str), return_counts=True)):
                totals[v] = totals.get(v, 0) + int(c)
        if "class" in batch.schema.names:
            cl = np.asarray(pc.fill_null(batch.column("class"), "none").to_numpy(zero_copy_only=False)).astype(str)
            for key, vals in CLASS_GROUPS.items():
                m = ok & np.isin(cl, vals)
                add(f"bld_n_{key}", idx[m])
        hcols = [c for c in ["height", "num_floors"] if c in batch.schema.names]
        if hcols:
            has = np.zeros(batch.num_rows, dtype=bool)
            for c in hcols:
                has |= np.asarray(pc.is_valid(batch.column(c)).to_numpy(zero_copy_only=False))
            add("bld_n_has_height", idx[ok & has])
        rows += batch.num_rows
    df = pd.DataFrame({k: (v if isinstance(v, np.ndarray) else np.zeros(n)) for k, v in acc.items()})
    df.insert(0, "GEOID", T["gdf"]["GEOID"].values)
    log(f"{region}: buildings read {rows:,}, assigned {int(df['bld_n'].sum()):,}")
    src = pd.DataFrame({"region": region, "dataset": list(totals), "n": list(totals.values())})
    return df, src

# ---------------------------------------------------------------- places (POIs)
FIRE = r"fire_department|fire_station|fire_and_rescue|fire_protection_service|firefighter"
EMS = r"ambulance|emergency_medical|paramedic|emergency_service|emergency_room|urgent_care"
K12 = r"^(school|elementary_school|middle_school|high_school|public_school|private_school|charter_school|primary_school|secondary_school|montessori_school|boarding_school|religious_school|kindergarten|preschool|school_district_office|school_district_offices)$"
EXACT_CATS = ["fire_department", "ambulance_and_ems_services", "police_department", "hospital",
    "emergency_room", "urgent_care_clinic", "pharmacy", "drugstore", "grocery_store", "supermarket",
    "convenience_store", "gas_station", "community_center", "homeless_shelter", "church", "mosque",
    "place_of_worship", "town_hall", "post_office", "library", "school", "elementary_school",
    "middle_school", "high_school", "preschool", "private_school", "public_school", "charter_school",
    "montessori_school", "religious_school", "kindergarten", "specialty_school",
    "vocational_and_technical_school", "college_university", "school_district_offices"]
ALT_CATS = ["fire_department", "ambulance_and_ems_services", "school", "elementary_school", "middle_school",
            "high_school", "private_school", "public_school", "charter_school", "preschool"]
# places kept one-by-one in key_pois.csv (emergency, schools, relief and gathering places)
SIDE = (r"fire|ambulance|emergency|paramedic|rescue|police|sheriff|hospital|urgent_care|school|kindergarten|"
        r"preschool|shelter|pharmacy|drugstore|grocery|supermarket|gas_station|community_center|town_hall|"
        r"city_hall|post_office|library|food_bank|red_cross|disaster|evacuation|civil_defense|national_guard")

def agg_pois(fs, base, region, T):
    pf = open_pf(fs, ref_path(base, region, "overture-pois"))
    cols, gcol, enc = point_cols(pf, ["bbox", "categories", "confidence", "sources", "names", "operating_status",
                                      "basic_category"])
    n = T["n"]
    acc = {}
    def add(k, ii, w=None):
        acc[k] = acc.get(k, 0) + bc(ii, n, w)
    for k in (["poi_n", "poi_n_conf50", "poi_n_conf80", "poi_n_nocat", "poi_n_noname", "poi_n_not_open"]
              + [f"poi_cat_{c}" for c in EXACT_CATS] + [f"poi_alt_{c}" for c in ALT_CATS]):
        acc[k] = np.zeros(n)
    cat_tot, side, rows = {}, [], 0
    gid = T["gdf"]["GEOID"].values
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        xmin, xmax, ymin, ymax = bbox_arrays(batch, gcol, enc)
        px, py = (xmin + xmax) / 2, (ymin + ymax) / 2
        idx = assign(T["tree"], px, py)
        ok = idx >= 0
        add("poi_n", idx[ok])
        nb = batch.num_rows
        conf = np.zeros(nb)
        if "confidence" in batch.schema.names:
            conf = np.asarray(pc.fill_null(batch.column("confidence"), 0.0).to_numpy(zero_copy_only=False), dtype=float)
            add("poi_n_conf50", idx[ok & (conf >= 0.5)]); add("poi_n_conf80", idx[ok & (conf >= 0.8)])
        primn = np.full(nb, "", dtype=object)
        alts = np.full(nb, "", dtype=object)
        if "categories" in batch.schema.names:
            cats = batch.column("categories")
            prim = pc.fill_null(pc.struct_field(cats, "primary"), "")
            primn = np.asarray(prim.to_numpy(zero_copy_only=False)).astype(str)
            for c in EXACT_CATS:
                m = ok & (primn == c)
                if m.any():
                    add(f"poi_cat_{c}", idx[m])
            add("poi_n_nocat", idx[ok & (primn == "")])
            vc = pc.value_counts(prim)
            for v, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                cat_tot[v] = cat_tot.get(v, 0) + c
            try:
                alt = pc.fill_null(pc.binary_join(pc.struct_field(cats, "alternate"), "|"), "")
                alts = np.asarray(alt.to_numpy(zero_copy_only=False)).astype(str)
                for c in ALT_CATS:
                    m = np.asarray(pc.match_substring_regex(alt, f"(^|\\|){c}(\\||$)").to_numpy(zero_copy_only=False))
                    m = ok & m & (primn != c)
                    if m.any():
                        add(f"poi_alt_{c}", idx[m])
            except Exception as e:
                if rows == 0:
                    log(f"{region}: alternate categories unreadable: {type(e).__name__}: {e}")
        nm = np.full(nb, "", dtype=object)
        if "names" in batch.schema.names:
            nm = struct_str(batch.column("names"), "primary")
            add("poi_n_noname", idx[ok & (nm == "")])
        st = np.full(nb, "", dtype=object)
        if "operating_status" in batch.schema.names:
            st = np.asarray(pc.fill_null(batch.column("operating_status"), "").to_numpy(zero_copy_only=False)).astype(str)
            add("poi_n_not_open", idx[ok & (st != "") & (st != "open")])
        ds = first_source_dataset(batch)
        if ds is None:
            ds = np.full(nb, "none", dtype=object)
        for key, pat in {"meta": "meta", "msft": "microsoft"}.items():
            add(f"poi_n_src_{key}", idx[ok & (np.char.find(ds.astype(str), pat) >= 0)])
        keep = ok & (pd.Series(primn).str.contains(SIDE, regex=True).values |
                     pd.Series(alts).str.contains(SIDE, regex=True).values)
        if keep.any():
            side.append(pd.DataFrame({"GEOID": gid[idx[keep]], "lon": np.round(px[keep], 6), "lat": np.round(py[keep], 6),
                                      "cat": primn[keep], "alt": alts[keep], "conf": np.round(conf[keep], 3),
                                      "name": pd.Series(np.asarray(nm)[keep]).str.slice(0, 60).values,
                                      "ds": ds[keep], "status": np.asarray(st)[keep]}))
        rows += nb
    df = pd.DataFrame(acc)
    df.insert(0, "GEOID", gid)
    log(f"{region}: places read {rows:,}, assigned {int(df['poi_n'].sum()):,}")
    cats = pd.DataFrame({"region": region, "category": list(cat_tot), "n": list(cat_tot.values())})
    side = pd.concat(side, ignore_index=True) if side else pd.DataFrame(
        columns=["GEOID", "lon", "lat", "cat", "alt", "conf", "name", "ds", "status"])
    side.insert(1, "region", region)
    try:
        side = near_tracts(side, T)
    except Exception as e:
        log(f"{region}: nearest-tract step FAILED: {type(e).__name__}: {e}")
    log(f"{region}: key places kept {len(side):,}")
    return df, (cats, side)

def near_tracts(side, T):
    """Distance of each key place to its own tract border, and the nearest other tract within NEAR_M."""
    if len(side) == 0:
        return side
    pts = project(shapely.points(side["lon"].values, side["lat"].values))
    pos = pd.Series(np.arange(T["n"]), index=T["gdf"]["GEOID"].values)
    own = pos.reindex(side["GEOID"].values).values.astype(np.int64)
    side["d_border_m"] = np.round(shapely.distance(pts, T["bnd"][own]), 1)
    pi, tj = T["tree_m"].query(pts, predicate="dwithin", distance=NEAR_M)
    m = tj != own[pi]
    pi, tj = pi[m], tj[m]
    side["nbr_GEOID"] = ""
    side["nbr_d_m"] = np.nan
    if len(pi):
        d = shapely.distance(pts[pi], T["g_m"][tj])
        o = pd.DataFrame({"pi": pi, "tj": tj, "d": d}).sort_values(["pi", "d"]).drop_duplicates("pi")
        side.loc[o["pi"].values, "nbr_GEOID"] = T["gdf"]["GEOID"].values[o["tj"].values]
        side.loc[o["pi"].values, "nbr_d_m"] = np.round(o["d"].values, 1)
    return side

# ---------------------------------------------------------------- roads
ROAD_GROUPS = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential",
               "unclassified", "service", "living_street", "track", "other"]
MAJOR = ["motorway", "trunk", "primary", "secondary"]
STATE_SUB = {"FM", "RM", "LOOP", "SPUR", "BUSINESS", "PR", "TOLL", "TURNPIKE", "TRUCK", "BYPASS", "ALTERNATE",
             "ALT", "BELTWAY", "CONNECTOR", "PARK", "RE", "SR", "SH", "HIGHWAY", "SCENIC", "FUTURE", "TEMPORARY"}
ROUTE_RANK = {"none": 0, "other": 1, "county": 2, "state_sub": 3, "state": 4, "us": 5, "interstate": 6}
RANK_NAME = {v: k for k, v in ROUTE_RANK.items()}

def network_kind(s):
    s = (s or "").strip()
    if not s:
        return "none"
    p = s.split(":")
    if len(p) >= 2 and p[0] == "US":
        if p[1] == "I":
            return "interstate"
        if p[1] == "US":
            return "us"
        if len(p[1]) == 2 and p[1].isalpha() and p[1].isupper():
            if len(p) == 2:
                return "state"
            return "state_sub" if p[2].upper() in STATE_SUB else "county"
    return "other"

def best_route(routes_arr):
    """Highest-ranked route network per segment (interstate > us > state > state_sub > county > other)."""
    if isinstance(routes_arr, pa.ChunkedArray):
        routes_arr = routes_arr.combine_chunks()
    n = len(routes_arr)
    best = np.zeros(n, dtype=np.int64)
    flat = pc.list_flatten(routes_arr)
    if len(flat) == 0:
        return best, []
    parents = np.asarray(pc.list_parent_indices(routes_arr).to_numpy(zero_copy_only=False))
    nets = pc.fill_null(pc.struct_field(flat, "network"), "").to_pylist()
    cache = {}
    rank = np.array([cache.setdefault(s, ROUTE_RANK[network_kind(s)]) for s in nets], dtype=np.int64)
    np.maximum.at(best, parents, rank)
    return best, nets

def safe_intersection_length(a, b):
    try:
        return shapely.length(shapely.intersection(a, b))
    except Exception:
        out = np.zeros(len(a))
        for k in range(len(a)):
            try:
                out[k] = shapely.length(shapely.intersection(a[k], b[k]))
            except Exception:
                try:
                    out[k] = shapely.length(shapely.intersection(a[k], shapely.make_valid(b[k])))
                except Exception:
                    out[k] = 0.0
        return out

def clip_pairs(seg_m, tree, polys):
    """(segment index, tract index, length of the segment inside that polygon) for all touching pairs."""
    si, ti = tree.query(seg_m, predicate="intersects")
    L = np.zeros(len(si))
    if len(si) == 0:
        return si, ti, L
    inside = shapely.contains_properly(polys[ti], seg_m[si])
    L[inside] = shapely.length(seg_m[si[inside]])
    cr = ~inside
    if cr.any():
        L[cr] = safe_intersection_length(seg_m[si[cr]], polys[ti[cr]])
    return si, ti, L

def agg_roads(fs, base, region, T, layer="overture-roads"):
    path = ref_path(base, region, layer)
    pf = open_pf(fs, path)
    names = pf.schema_arrow.names
    gcol, enc = geo_info(pf)
    base_cols = [c for c in ["class", "subclass", "subtype"] if c in names]
    attr_cols = [c for c in ["names", "routes", "road_surface", "speed_limits"] if c in names]
    n = T["n"]
    acc = {}
    def add(k, ii, w=None):
        acc[k] = acc.get(k, 0) + bc(ii, n, w)
    for k in ["road_len_m", "road_n"] + [f"road_len_{g}_m" for g in ROAD_GROUPS]:
        acc[k] = np.zeros(n)
    class_tot, net_tot = {}, {}
    rows, attr_ok = 0, set()
    for batch in pf.iter_batches(batch_size=BATCH, columns=base_cols + attr_cols + [gcol]):
        nb = batch.num_rows
        g = decode_geoms(batch.column(gcol), enc)
        gm = project(g)
        del g
        lens = np.nan_to_num(shapely.length(gm))
        cen = shapely.centroid(gm)
        idx = assign(T["tree_m"], shapely.get_x(cen), shapely.get_y(cen))
        ok = idx >= 0
        cl = (np.asarray(pc.fill_null(batch.column("class"), "other").to_numpy(zero_copy_only=False)).astype(str)
              if "class" in batch.schema.names else np.full(nb, "other"))
        sub = (np.asarray(pc.fill_null(batch.column("subclass"), "").to_numpy(zero_copy_only=False)).astype(str)
               if "subclass" in batch.schema.names else np.full(nb, ""))
        kc =pd.DataFrame({"c": cl, "s": sub, "L": lens}).groupby(["c", "s"])["L"].agg(["size", "sum"])
        for (c, s), r in kc.iterrows():
            key = (c, s)
            a, b = class_tot.get(key, (0, 0.0))
            class_tot[key] = (a + int(r["size"]), b + float(r["sum"]))
        grp = np.where(np.isin(cl, ROAD_GROUPS), cl, "other")
        add("road_len_m", idx[ok], lens[ok]); add("road_n", idx[ok])
        for gname in ROAD_GROUPS:
            m = ok & (grp == gname)
            add(f"road_len_{gname}_m", idx[m], lens[m])
        major = np.isin(cl, MAJOR)
        link = sub == "link"
        # ---- attributes (each one optional; a failure only drops that feature)
        rk = np.zeros(nb, dtype=np.int64)
        if "routes" in batch.schema.names:
            try:
                rk, nets = best_route(batch.column("routes"))
                for s in nets:
                    net_tot[s] = net_tot.get(s, 0) + 1
                attr_ok.add("routes")
            except Exception as e:
                if "routes!" not in attr_ok:
                    log(f"{region}: routes unreadable: {type(e).__name__}: {e}"); attr_ok.add("routes!")
        if "names" in batch.schema.names:
            try:
                unnamed = struct_str(batch.column("names"), "primary") == ""
                for gname in ROAD_GROUPS:
                    m = ok & (grp == gname) & unnamed
                    add(f"road_len_{gname}_unnamed_m", idx[m], lens[m])
                attr_ok.add("names")
            except Exception as e:
                if "names!" not in attr_ok:
                    log(f"{region}: road names unreadable: {type(e).__name__}: {e}"); attr_ok.add("names!")
        if "road_surface" in batch.schema.names:
            try:
                surf = first_of_list_field(batch.column("road_surface"), "value")
                surf = np.array(["" if v is None else str(v) for v in surf], dtype=object)
                m = ok & (surf != "") & (surf != "unknown")
                add("road_len_surface_known_m", idx[m], lens[m])
                m = ok & np.isin(surf, ["unpaved", "gravel", "dirt", "ground", "sand", "grass", "compacted"])
                add("road_len_unpaved_m", idx[m], lens[m])
                attr_ok.add("road_surface")
            except Exception as e:
                if "road_surface!" not in attr_ok:
                    log(f"{region}: road_surface unreadable: {type(e).__name__}: {e}"); attr_ok.add("road_surface!")
        if "speed_limits" in batch.schema.names:
            try:
                m = ok & list_nonempty(batch.column("speed_limits"))
                add("road_len_has_speed_m", idx[m], lens[m])
                attr_ok.add("speed_limits")
            except Exception as e:
                if "speed_limits!" not in attr_ok:
                    log(f"{region}: speed_limits unreadable: {type(e).__name__}: {e}"); attr_ok.add("speed_limits!")
        # ---- exact cut of major roads, tertiary roads and numbered routes against tracts
        route = rk >= ROUTE_RANK["county"]
        hwy = rk >= ROUTE_RANK["state_sub"]
        key = major | (cl == "tertiary") | (rk > 0)
        if key.any():
            kidx = np.flatnonzero(key)
            seg = gm[kidx]
            groups = {
                "motorway": cl[kidx] == "motorway", "trunk": cl[kidx] == "trunk",
                "primary": cl[kidx] == "primary", "secondary": cl[kidx] == "secondary",
                "tertiary": cl[kidx] == "tertiary",
                "major": major[kidx], "major_nolink": major[kidx] & ~link[kidx],
                "route_any": rk[kidx] > 0, "route": route[kidx],
                "route_major": route[kidx] & major[kidx], "route_minor": route[kidx] & ~major[kidx],
                "hwy": hwy[kidx], "hwy_major": hwy[kidx] & major[kidx], "hwy_minor": hwy[kidx] & ~major[kidx],
            }
            for kname in ["interstate", "us", "state", "state_sub", "county", "other"]:
                groups[f"rt_{kname}"] = rk[kidx] == ROUTE_RANK[kname]
            for pre, tree, polys in [("cut_", T["tree_m"], T["g_m"]), ("buf_", T["tree_buf"], T["buf"])]:
                si, ti, L = clip_pairs(seg, tree, polys)
                for gname, m in groups.items():
                    sel = m[si]
                    add(f"{pre}{gname}_m", ti[sel], L[sel])
        rows += nb
        del gm, cen
    gc.collect()
    df = pd.DataFrame({k: (v if isinstance(v, np.ndarray) else np.zeros(n)) for k, v in acc.items()})
    df.insert(0, "GEOID", T["gdf"]["GEOID"].values)
    classes = pd.DataFrame([{"region": region, "class": c, "subclass": s, "n": a, "len_km": round(b / 1e3, 3)}
                            for (c, s), (a, b) in class_tot.items()])
    nets = pd.DataFrame({"region": region, "network": list(net_tot), "n": list(net_tot.values())})
    if len(nets):
        nets["kind"] = [network_kind(s) for s in nets["network"]]
    log(f"{region}: roads read {rows:,}, attributes used: {sorted(a for a in attr_ok if not a.endswith('!'))}")
    return df, (classes, nets)

# ---------------------------------------------------------------- tabular layers
def load_table(fs, path, drop_geometry=True):
    pf = open_pf(fs, path)
    gcol, _ = geo_info(pf)
    cols = [c for c in pf.schema_arrow.names if not (drop_geometry and c in ("geometry", gcol))]
    return pf.read(columns=cols).to_pandas()

# ---------------------------------------------------------------- one region
def run_region(fs, base, region):
    t0 = time.time()
    res = {"schema": ""}
    for layer in ["overture-buildings", "overture-roads", "overture-pois", "census-acs-housing"]:
        res["schema"] += schema_note(fs, ref_path(base, region, layer))
    T = load_tracts(fs, base, region)
    log(f"{region}: {T['n']:,} tracts")
    keep = [c for c in T["gdf"].columns if c != "geometry"]
    feats = T["gdf"][keep].copy()
    feats.insert(1, "region", region)
    simp = T["gdf"][["GEOID", "geometry"]].copy()
    simp["geometry"] = simp.geometry.simplify(0.0002, preserve_topology=True)
    simp.insert(1, "region", region)
    res["tracts"] = simp
    for name, fn in [("buildings", agg_buildings), ("pois", agg_pois), ("roads", agg_roads)]:
        for attempt in (1, 2):
            t1 = time.time()
            try:
                df, side = fn(fs, base, region, T)
                feats = feats.merge(df, on="GEOID", how="left")
                res[name] = side
                log(f"{region}: {name} took {time.time() - t1:,.0f} s")
                break
            except Exception as e:
                log(f"{region}: {name} FAILED (attempt {attempt}): {type(e).__name__}: {e}")
                LOG.append(traceback.format_exc())
                if not isinstance(e, OSError):
                    break
            finally:
                gc.collect()
    try:
        acs = load_table(fs, ref_path(base, region, "census-acs-housing"))
        acs["GEOID"] = acs["GEOID"].astype(str)
        acs = acs.rename(columns={c: f"acs_{c}" for c in acs.columns if c != "GEOID"})
        feats = feats.merge(acs.drop_duplicates("GEOID"), on="GEOID", how="left")
    except Exception as e:
        log(f"{region}: acs FAILED: {type(e).__name__}: {e}")
    try:
        strata = load_table(fs, strata_path(base, region, "strata-tract-table"))
        strata["GEOID"] = strata["GEOID"].astype(str)
        strata.insert(1, "region", region)
        res["strata"] = strata
    except Exception as e:
        log(f"{region}: strata FAILED: {type(e).__name__}: {e}")
    num = feats.select_dtypes("number").columns.difference(["cx", "cy"])
    feats[num] = feats[num].round(2)
    feats[["cx", "cy"]] = feats[["cx", "cy"]].round(6)
    res["features"] = feats
    log(f"{region}: done in {time.time() - t0:,.0f} s")
    return res

def run_all(fs, base, out_dir, regions=REGIONS):
    os.makedirs(out_dir, exist_ok=True)
    P = {k: [] for k in ["features", "strata", "tracts", "bsrc", "pcat", "keyp", "rcls", "rnet", "schema"]}
    for r in regions:
        try:
            res = run_region(fs, base, r)
        except Exception as e:
            log(f"{r}: REGION FAILED: {type(e).__name__}: {e}")
            LOG.append(traceback.format_exc())
            continue
        P["schema"].append(res.get("schema", ""))
        for k in ["features", "strata", "tracts"]:
            if k in res:
                P[k].append(res[k])
        if "buildings" in res:
            P["bsrc"].append(res["buildings"])
        if "pois" in res:
            P["pcat"].append(res["pois"][0]); P["keyp"].append(res["pois"][1])
        if "roads" in res:
            P["rcls"].append(res["roads"][0]); P["rnet"].append(res["roads"][1])
        gc.collect()
    files = []
    def save(parts, name):
        parts = [p for p in parts if p is not None and len(p)]
        if parts:
            p = os.path.join(out_dir, name)
            pd.concat(parts, ignore_index=True).to_csv(p, index=False)
            files.append(p)
    save(P["features"], "features.csv")
    save(P["strata"], "strata.csv")
    save(P["bsrc"], "building_sources.csv")
    save(P["pcat"], "poi_categories.csv")
    save(P["keyp"], "key_pois.csv")
    save(P["rcls"], "road_classes.csv")
    save(P["rnet"], "road_networks.csv")
    if P["tracts"]:
        p = os.path.join(out_dir, "tracts.parquet")
        gpd.GeoDataFrame(pd.concat(P["tracts"], ignore_index=True), crs=P["tracts"][0].crs).to_parquet(p)
        files.append(p)
    p = os.path.join(out_dir, "schema.txt")
    open(p, "w").write("\n".join(P["schema"]))
    files.append(p)
    p = os.path.join(out_dir, "run_log.txt")
    open(p, "w").write("\n".join(LOG))
    files.append(p)
    zpath = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    mb = os.path.getsize(zpath) / 1e6
    log(f"zip ready: {zpath} ({mb:,.1f} MB)")
    if mb > 25:
        lite = out_dir.rstrip("/") + "_lite.zip"
        with zipfile.ZipFile(lite, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                if not f.endswith(".parquet"):
                    z.write(f, os.path.basename(f))
        log(f"also made {lite} ({os.path.getsize(lite)/1e6:,.1f} MB) without tract shapes")
    return zpath


def summary(out_dir):
    """Short check of what came out, printed at the end of the run."""
    f = pd.read_csv(os.path.join(out_dir, "features.csv"), dtype={"GEOID": str}, low_memory=False)
    g = f.groupby("region")
    rows = []
    for r, d in g:
        rows.append({"region": r, "tracts": len(d),
                     "buildings": int(d.get("bld_n", pd.Series([0])).sum()),
                     "places": int(d.get("poi_n", pd.Series([0])).sum()),
                     "road_km": round(d.get("road_len_m", pd.Series([0])).sum() / 1e3),
                     "fire_dept_places": int(d.get("poi_cat_fire_department", pd.Series([0])).sum()),
                     "tracts_with_numbered_hwy": int((d.get("cut_hwy_m", pd.Series([0])) > 0).sum()),
                     "unnamed_residential_km": round(d.get("road_len_residential_unnamed_m", pd.Series([0])).sum() / 1e3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    bad = [l for l in LOG if "FAILED" in l]
    print("\nProblems:" if bad else "\nNo failed steps.")
    for l in bad:
        print("  ", l)
    return out
