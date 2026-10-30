
# ================================================================ Lifelines audit (29 Oct 2026)
# Are the places that low-income and vulnerable people depend on actually on the map?
# Three public federal registries are matched, point by point, against the Overture places the
# challenge provides (release 2026-08-19.0):
#   SNAP  USDA SNAP retailer data: every store authorized to accept SNAP benefits (food stamps)
#   NH    CMS nursing home Provider Information: Medicare/Medicaid certified nursing homes
#   HC    HRSA health center service delivery sites: community health centers (safety-net clinics)
# A registry entry counts as found when an Overture place nearby carries its name, or sits at its
# street address. Every distance, name score and address score is written out, so the thresholds
# can be checked and changed afterwards. Nothing here uses TIGER, Microsoft footprints or HIFLD, and
# nothing here feeds a leaderboard file. census_extras() adds public Census context tables (ACS, and
# county counts from County Business Patterns and Nonemployer Statistics) for the written finding only.
import io, csv, hashlib, unicodedata, datetime
from scipy.spatial import cKDTree
from rapidfuzz import fuzz, process

STATE_OF = {"maricopa-az": ("AZ", "NM"), "northern-ca": ("CA",), "eastern-ok": ("OK",), "south-central-tx": ("TX",),
            "eastern-wa": ("WA",)}
STATES = sorted({s for v in STATE_OF.values() for s in v})
# study-region bounding boxes (lon/lat, from the provided tract polygons plus 0.05 degree)
BBOX = {"eastern-ok": (-100.05, 33.57, -94.38, 37.05), "eastern-wa": (-122.33, 45.56, -116.87, 49.05),
        "maricopa-az": (-114.61, 31.28, -108.16, 36.46), "northern-ca": (-124.28, 38.26, -119.95, 42.06),
        "south-central-tx": (-103.30, 25.79, -94.60, 34.29)}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36", "Accept": "*/*"}
SNAP_LIVE = "https://services1.arcgis.com/RLQu0rK7h4kbsBq5/arcgis/rest/services/snap_retailer_location_data/FeatureServer/0/query"
SNAP_HIST = ["https://www.fna.usda.gov/sites/default/files/resource-files/snap-retailer-locator-data2005-2025.zip",
             "https://www.fns.usda.gov/sites/default/files/resource-files/snap-retailer-locator-data2005-2025.zip",
             "https://fns-prod.azureedge.us/sites/default/files/resource-files/snap-retailer-locator-data2005-2025.zip"]
CMS_NH_META = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/4pq5-n9py"
HRSA_SITES = ["https://data.hrsa.gov/DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.csv",
              "https://data.hrsa.gov//DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.csv"]
HRSA_XLSX = ["https://data.hrsa.gov/DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.xlsx"]
CENSUS_BATCH = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
SOURCES = []          # provenance: url, file, sha256, bytes, retrieved (UTC)

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fetch(urls, dest, min_bytes=20000, timeout=300, local=None):
    """Download the first URL that works into dest. A file the user uploaded by hand (same name, or
    matching the `local` pattern in /content or the work folder) is used as it is."""
    import requests, glob, shutil
    if local and not (os.path.exists(dest) and os.path.getsize(dest) >= min_bytes):
        for d in (os.path.dirname(dest), "/content", "."):
            hits = [h for h in glob.glob(os.path.join(d, local)) if os.path.getsize(h) >= min_bytes]
            if hits:
                shutil.copy(sorted(hits)[-1], dest)
                break
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        log(f"using existing file {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        SOURCES.append({"url": "uploaded by hand", "file": os.path.basename(dest), "sha256": sha256(dest),
                        "bytes": os.path.getsize(dest), "retrieved": utcnow()})
        return dest
    errs = []
    for u in urls:
        try:
            with requests.get(u, headers=UA, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if os.path.getsize(dest + ".part") >= min_bytes:
                os.replace(dest + ".part", dest)
                SOURCES.append({"url": u, "file": os.path.basename(dest), "sha256": sha256(dest),
                                "bytes": os.path.getsize(dest), "retrieved": utcnow()})
                log(f"downloaded {u} ({os.path.getsize(dest)/1e6:.1f} MB)")
                return dest
            errs.append(f"{u}: only {os.path.getsize(dest + '.part')} bytes")
        except Exception as e:
            errs.append(f"{u}: {type(e).__name__}: {str(e)[:150]}")
    log("download failed: " + " | ".join(errs))
    return None

def _key(c):
    return re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()

def pick(cols, *must, exclude=(), exact=None):
    """Column chosen by its simplified name: `exact` = the whole name; otherwise every word in `must`
    has to start one of the column's words, and no `exclude` text may appear in it."""
    keys = {c: _key(c) for c in cols}
    if exact is not None:
        for c, k in keys.items():
            if k == exact:
                return c
        if not must:
            return None
    if not must:
        return None
    for c, k in keys.items():
        words = k.split()
        if all(any(w.startswith(m) for w in words) for m in must) and not any(x in k for x in exclude):
            return c
    return None

def col(df, c, default=""):
    if c is None or c not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=object)
    return df[c].fillna(default).astype(str).str.strip()

def to_num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

def zip5(s):
    z = pd.Series(s).fillna("").astype(str).str.extract(r"(\d{3,5})")[0].fillna("")
    return z.where(z == "", z.str.zfill(5))

def norm_date(s):
    return pd.to_datetime(pd.Series(s).replace("", np.nan), errors="coerce", format="mixed")

# ---------------------------------------------------------------- text normalisation
def strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

_CHAIN_FIX = [(re.compile(p), r) for p, r in [
    (r"\b7\s*-?\s*(eleven|11)\b", "seveneleven"), (r"\bseven\s*-?\s*eleven\b", "seveneleven"),
    (r"\bh\s*-?\s*e\s*-?\s*b\b", "heb"), (r"\bwal\s*-?\s*mart\b", "walmart"), (r"\bk\s*-?\s*mart\b", "kmart"),
    (r"\bcircle\s*-?\s*k\b", "circlek"), (r"\b99\s*cents?\s*only\b", "ninetyninecentsonly"),
    (r"\bdollar\s*general\b", "dollargeneral"), (r"\bfamily\s*dollar\b", "familydollar"),
    (r"\bdollar\s*tree\b", "dollartree"), (r"\bsam\s*s\s*club\b", "samsclub"), (r"\bquik\s*trip\b", "quiktrip"),
    (r"\bfood\s*(4|for)\s*less\b", "foodforless"), (r"\bfiesta\s*mart\b", "fiestamart"), (r"\bfry\s*s\b", "frys"),
    (r"\bsprouts\s*farmers\s*market\b", "sprouts"), (r"\btrader\s*joe\s*s\b", "traderjoes"),
    (r"\bwhole\s*foods\b", "wholefoods"), (r"\bon\s*the\s*run\b", "ontherun"), (r"\bdollar\s*plus\b", "dollarplus"),
    (r"\bspeedy\s*stop\b", "speedystop"), (r"\btoot\s*n\s*totum\b", "tootntotum"), (r"\bcasey\s*s\b", "caseys"),
    (r"\blove\s*s\b", "loves"), (r"\bmurphy\s*(usa|express)\b", "murphyusa"), (r"\bcircle\s*k\b", "circlek")]]
_DROP = {"inc", "llc", "lc", "co", "corp", "corporation", "ltd", "company", "dba", "the", "l", "p", "lp", "pllc"}
GENERIC = set("""and of a an at de del la el los las y en mi su
store stores shop shoppe market markets mart supermarket supermercado mercado super mini minimart grocery groceries
abarrotes tienda tiendita food foods meat meats carniceria fresh quick stop express corner convenience discount center
centre centro gas station fuel fuels oil petroleum travel plaza truck n go enterprises enterprise group international
intl usa us american drive drug drugs pharmacy farmacia nursing rehabilitation rehab health healthcare care living
senior seniors home homes community manor village skilled post acute transitional medical clinic clinica clinics family
services service hospital memory assisted residence residential lodge north south east west n s e w new old city
county texas oklahoma arizona california washington tx ok az ca wa center centers rehabilitative facility""".split())

def norm_name(s):
    if s is None:
        return ""
    s = str(s)
    if s in ("", "nan", "None"):
        return ""
    s = strip_accents(s).lower().replace("&", " and ").replace("'", "").replace("’", "")
    for pat, rep in _CHAIN_FIX:
        s = pat.sub(rep, s)
    s = re.sub(r"#\s*\w+", " ", s)
    s = re.sub(r"\b(store|unit|no|number|num|ste|suite|loc|location|site)\s*\d+\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(t for t in s.split() if not t.isdigit() and t not in _DROP)

_STREET = {"street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "highway": "hwy", "hiway": "hwy",
           "hway": "hwy", "drive": "dr", "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl",
           "parkway": "pkwy", "pkway": "pkwy", "freeway": "fwy", "expressway": "expy", "circle": "cir", "trail": "trl",
           "route": "rt", "rte": "rt", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
           "northwest": "nw", "southeast": "se", "southwest": "sw", "square": "sq", "terrace": "ter", "plaza": "plz",
           "loop": "lp", "center": "ctr", "centre": "ctr", "mount": "mt", "saint": "st", "fort": "ft", "first": "1st",
           "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th"}

def norm_street(s):
    if s is None:
        return ""
    s = strip_accents(str(s)).lower()
    if s in ("", "nan", "none"):
        return ""
    s = re.sub(r"\bfarm\s*(to|-)?\s*market(\s*(road|rd))?\b", "fm", s)
    s = re.sub(r"\branch\s*(to|-)?\s*market(\s*(road|rd))?\b", "rm", s)
    s = re.sub(r"\b(fm|rm)\s*(road|rd)\b", r"\1", s)
    s = re.sub(r"\bcounty\s*(road|rd)\b", "cr", s)
    s = re.sub(r"\bstate\s*(highway|hwy|route|rt)\b", "sh", s)
    s = re.sub(r"\bu\s*\.?\s*s\s*\.?\s*(highway|hwy|route|rt)\b", "us", s)
    s = re.sub(r"\binterstate\b", "i", s)
    s = re.sub(r"\b(suite|ste|unit|apt|bldg|building|lot|space|spc|room|rm\s+\d)\s*[\w-]*", " ", s)
    s = re.sub(r"#\s*[\w-]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(_STREET.get(t, t) for t in s.split())

_ADDR = re.compile(r"^\s*(\d+)[a-z]?(?:[-/]\w+)?\s+(.+)$", re.I)

def split_addr(s):
    """House number and normalised street: '1234-B N Main St, Suite 5' gives ('1234', 'n main st')."""
    if s is None:
        return "", ""
    s = str(s).split(",")[0]
    m = _ADDR.match(s)
    if not m:
        return "", norm_street(s)
    return str(int(m.group(1))), norm_street(m.group(2))

# ---------------------------------------------------------------- registry loaders
def _frame(ref_set, ids, names, rtype, addr_num, addr_street, city, state, zipc, lat, lon, **extra):
    df = pd.DataFrame({"ref_set": ref_set, "ref_id": ids, "name": names, "rtype": rtype, "hnum": addr_num,
                       "street": addr_street, "city": city, "state": state, "zip5": zip5(zipc).values,
                       "lat": to_num(lat).values, "lon": to_num(lon).values})
    for k, v in extra.items():
        df[k] = v if np.ndim(v) == 0 else pd.Series(v).values
    return df

def snap_live(chunk=400):
    """Current SNAP retailers from USDA's own ArcGIS service: object ids inside each study-region box,
    then the records in chunks of object ids (no reliance on paging or on field names). Gives up fast
    when the service cannot be reached, so the history file can take over."""
    import requests
    def call(method, **kw):
        for attempt in range(3):
            try:
                r = (requests.get if method == "get" else requests.post)(SNAP_LIVE, headers=UA, timeout=120, **kw)
                return r.json()
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:150]}"
                time.sleep(2 * (attempt + 1))
        log(f"SNAP live: no answer ({last})")
        return None
    ids = set()
    for n, (region, (x0, y0, x1, y1)) in enumerate(BBOX.items()):
        r = call("get", params={"where": "1=1", "geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope",
                                "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "returnIdsOnly": "true", "f": "json"})
        if r is None and n == 0:
            return None
        if not r or "objectIds" not in r:
            log(f"SNAP live ids {region}: none ({str(r)[:300]})")
            continue
        ids.update(r["objectIds"] or [])
        log(f"SNAP live ids {region}: {len(r['objectIds'] or []):,}")
    if not ids:
        return None
    ids = sorted(ids)
    rows, fails = [], 0
    for i in range(0, len(ids), chunk):
        r = call("post", data={"objectIds": ",".join(map(str, ids[i:i + chunk])), "outFields": "*",
                               "returnGeometry": "true", "outSR": 4326, "f": "json"})
        if not r or "features" not in r:
            fails += 1
            log(f"SNAP live chunk {i}: failed ({str(r)[:200]})")
            if fails >= 3:
                log("SNAP live: too many failed chunks, using what arrived")
                break
            continue
        for f in r["features"]:
            a = dict(f.get("attributes", {}))
            g = f.get("geometry") or {}
            a["_x"], a["_y"] = g.get("x"), g.get("y")
            rows.append(a)
    log(f"SNAP live: {len(rows):,} records for {len(ids):,} ids")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    SOURCES.append({"url": SNAP_LIVE + " (all records inside the five study-region boxes)", "file": "snap_live",
                    "sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(), "bytes": None,
                    "retrieved": utcnow(), "rows": len(df), "columns": list(df.columns)})
    return df

def snap_frame(df, src):
    cols = list(df.columns)
    c = {"id": pick(cols, "record", "id") or pick(cols, exact="objectid"),
         "name": pick(cols, "store", "name") or pick(cols, "retailer", "name"),
         "type": pick(cols, "store", "type") or pick(cols, "retailer", "type"),
         "num": pick(cols, "street", "number"), "sname": pick(cols, "street", "name"),
         "full": pick(cols, "street", "address") or pick(cols, exact="address"),
         "city": pick(cols, exact="city") or pick(cols, "city"), "state": pick(cols, exact="state") or pick(cols, "state"),
         "zip": pick(cols, exact="zip code") or pick(cols, exact="zip5") or pick(cols, "zip", exclude=("4",)),
         "lat": pick(cols, exact="y") or pick(cols, "latitude"), "lon": pick(cols, exact="x") or pick(cols, "longitude"),
         "auth": pick(cols, "authoriz"), "end": pick(cols, "end", "date"), "county": pick(cols, "county")}
    log(f"SNAP {src} columns used: {c}")
    if c["num"] is not None:
        hnum = col(df, c["num"]).str.extract(r"(\d+)")[0].fillna("")
        hnum = hnum.where(hnum == "", hnum.str.lstrip("0"))
        street = col(df, c["sname"]).map(norm_street)
    else:
        sp = col(df, c["full"]).map(split_addr)
        hnum, street = sp.map(lambda t: t[0]), sp.map(lambda t: t[1])
    out = _frame("SNAP", col(df, c["id"]).values, col(df, c["name"]).values, col(df, c["type"]).values,
                 hnum.values, street.values, col(df, c["city"]).values, col(df, c["state"]).str.upper().values,
                 col(df, c["zip"]).values, col(df, c["lat"]).values, col(df, c["lon"]).values,
                 county=col(df, c["county"]).values,
                 auth_date=norm_date(col(df, c["auth"])).values if c["auth"] else pd.NaT,
                 end_date=norm_date(col(df, c["end"])).values if c["end"] else pd.NaT, snap_src=src)
    return out

def snap_hist(workdir):
    p = fetch(SNAP_HIST, os.path.join(workdir, "snap-retailer-locator-data2005-2025.zip"), min_bytes=1_000_000,
              local="*[Ss][Nn][Aa][Pp]*.zip")
    if p is None:
        return None
    z = zipfile.ZipFile(p)
    names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
    with z.open(names[0]) as f:
        df = pd.read_csv(f, dtype=str, encoding_errors="replace", low_memory=False)
    log(f"SNAP history: {len(df):,} rows, columns {list(df.columns)}")
    return df

SNAP_HUB = "https://usda-snap-retailers-usda-fns.hub.arcgis.com/datasets/8b260f9a10b0459aa441ad8588c2251c/explore"

def snap_local_csv(workdir):
    """A current-retailer CSV downloaded by hand from USDA's ArcGIS Hub page, if one was uploaded."""
    import glob
    for d in (workdir, "/content", "."):
        for h in sorted(glob.glob(os.path.join(d, "*[Ss][Nn][Aa][Pp]*.csv"))):
            if os.path.getsize(h) > 100_000:
                df = pd.read_csv(h, dtype=str, encoding_errors="replace", low_memory=False)
                SOURCES.append({"url": SNAP_HUB + " (CSV downloaded by hand)", "file": os.path.basename(h),
                                "sha256": sha256(h), "bytes": os.path.getsize(h), "retrieved": utcnow(), "rows": len(df)})
                log(f"SNAP current retailers from uploaded file {h}: {len(df):,} rows")
                return df
    return None

def load_snap(workdir):
    live = snap_local_csv(workdir)
    try:
        if live is None:
            live = snap_live()
    except Exception as e:
        log(f"SNAP live failed: {type(e).__name__}: {e}")
    hist = None
    try:
        hist = snap_hist(workdir)
    except Exception as e:
        log(f"SNAP history failed: {type(e).__name__}: {e}")
    H = snap_frame(hist, "history") if hist is not None else None
    if H is not None:
        H = H[H["state"].isin(STATES)].copy()
        H["active_2025"] = H["end_date"].isna()
    L = snap_frame(live, "live") if live is not None else None
    if L is not None:
        L = L[L["state"].isin(STATES) | (L["state"] == "")].copy()
        L["auth_date"], L["rtype_hist"], L["in_history_active"] = pd.NaT, "", False
        if H is not None:
            h = H.sort_values("auth_date").drop_duplicates("ref_id", keep="last").set_index("ref_id")
            L["auth_date"] = L["ref_id"].map(h["auth_date"])
            L["rtype_hist"] = L["ref_id"].map(h["rtype"])
            miss = L["auth_date"].isna()
            if miss.any():
                k = lambda d: d["name"].map(norm_name) + "|" + d["hnum"] + "|" + d["zip5"]
                hk = H.assign(k=k(H)).sort_values("auth_date").drop_duplicates("k", keep="last").set_index("k")
                kk = k(L[miss])
                L.loc[miss, "auth_date"] = kk.map(hk["auth_date"]).values
                L.loc[miss, "rtype_hist"] = kk.map(hk["rtype"]).values
            act = set(H.loc[H["active_2025"], "ref_id"])
            L["in_history_active"] = L["ref_id"].isin(act)
            log(f"SNAP: {int(L['auth_date'].notna().sum()):,} of {len(L):,} live stores found in the history file")
        return L, H
    if H is not None:
        A = H[H["active_2025"]].copy()
        A["rtype_hist"] = A["rtype"]
        return A, H
    return None, None

def load_nh(workdir):
    import requests
    try:
        meta = requests.get(CMS_NH_META, headers=UA, timeout=120).json()
        dist = meta.get("distribution") or meta.get("data", {}).get("distribution") or []
        urls = [d.get("downloadURL") or d.get("data", {}).get("downloadURL") for d in dist]
        urls = [u for u in urls if u]
        log(f"CMS nursing homes: {urls}")
    except Exception as e:
        log(f"CMS metastore failed: {type(e).__name__}: {e}")
        urls = []
    p = fetch(urls, os.path.join(workdir, "NH_ProviderInfo.csv"), min_bytes=500_000, local="NH_ProviderInfo*.csv")
    if p is None and os.path.exists(os.path.join(workdir, "NH_ProviderInfo.csv")):
        p = os.path.join(workdir, "NH_ProviderInfo.csv")
    if p is None:
        return None
    df = pd.read_csv(p, dtype=str, encoding_errors="replace", low_memory=False)
    cols = list(df.columns)
    c = {"id": pick(cols, "certification") or pick(cols, "ccn"), "name": pick(cols, "provider", "name"),
         "addr": pick(cols, "provider", "address") or pick(cols, "address"), "city": pick(cols, "city"),
         "state": pick(cols, exact="state") or pick(cols, "state"), "zip": pick(cols, "zip"),
         "lat": pick(cols, "latitude"), "lon": pick(cols, "longitude"), "beds": pick(cols, "certified", "beds"),
         "res": pick(cols, "average", "residents"), "own": pick(cols, "ownership"),
         "ptype": pick(cols, "provider", "type"), "county": pick(cols, "county", exclude=("code",)),
         "rating": pick(cols, "overall", "rating", exclude=("footnote",)), "legal": pick(cols, "legal", "business", "name")}
    log(f"CMS columns used: {c}")
    sp = col(df, c["addr"]).map(split_addr)
    out = _frame("NH", col(df, c["id"]).values, col(df, c["name"]).values, col(df, c["ptype"]).values,
                 sp.map(lambda t: t[0]).values, sp.map(lambda t: t[1]).values, col(df, c["city"]).values,
                 col(df, c["state"]).str.upper().values, col(df, c["zip"]).values,
                 col(df, c["lat"]).values, col(df, c["lon"]).values, county=col(df, c["county"]).values,
                 beds=to_num(col(df, c["beds"])).values, residents=to_num(col(df, c["res"])).values,
                 ownership=col(df, c["own"]).values, rating=col(df, c["rating"]).values,
                 full_address=col(df, c["addr"]).values, alt_names=col(df, c["legal"]).values)
    out = out[out["state"].isin(STATES)].copy()
    if out["lat"].isna().mean() > 0.5:
        log("CMS file has no usable coordinates: geocoding addresses with the Census geocoder")
        out = census_geocode(out)
    return out

def load_hc(workdir):
    p = fetch(HRSA_SITES, os.path.join(workdir, "Health_Center_Service_Delivery_and_LookAlike_Sites.csv"), min_bytes=500_000,
              local="Health_Center*.csv")
    if p is not None:
        df = pd.read_csv(p, dtype=str, encoding_errors="replace", low_memory=False)
    else:
        p = fetch(HRSA_XLSX, os.path.join(workdir, "Health_Center_Service_Delivery_and_LookAlike_Sites.xlsx"),
                  min_bytes=500_000, local="Health_Center*.xlsx")
        if p is None:
            return None
        df = pd.read_excel(p, dtype=str)
    cols = list(df.columns)
    c = {"id": pick(cols, "bphc", "assigned", "number") or pick(cols, "site", "number") or pick(cols, "site", "id"),
         "name": pick(cols, "site", "name"), "addr": pick(cols, "site", "address"), "city": pick(cols, "site", "city"),
         "state": pick(cols, "site", "state", "abbreviation") or pick(cols, "state", "abbreviation"),
         "zip": pick(cols, "site", "postal") or pick(cols, "site", "zip"),
         "lon": pick(cols, "x", "coordinate"), "lat": pick(cols, "y", "coordinate"),
         "status": pick(cols, "site", "status"), "setting": pick(cols, "location", "setting"),
         "ltype": pick(cols, "location", "type"),
         "hctype": pick(cols, exact="health center type description") or pick(cols, "health", "center", "type", "description", exclude=("location",)),
         "org": pick(cols, "health", "center", "name"), "county": pick(cols, "county", "name")}
    log(f"HRSA columns used: {c}")
    sp = col(df, c["addr"]).map(split_addr)
    out = _frame("HC", col(df, c["id"]).values, col(df, c["name"]).values, col(df, c["hctype"]).values,
                 sp.map(lambda t: t[0]).values, sp.map(lambda t: t[1]).values, col(df, c["city"]).values,
                 col(df, c["state"]).str.upper().values, col(df, c["zip"]).values,
                 col(df, c["lat"]).values, col(df, c["lon"]).values, county=col(df, c["county"]).values,
                 status=col(df, c["status"]).values, setting=col(df, c["setting"]).values,
                 loc_type=col(df, c["ltype"]).values, org=col(df, c["org"]).values,
                 full_address=col(df, c["addr"]).values,
                 alt_names=(col(df, c["org"]) + "|" + col(df, c["name"]).str.split(r"\s+-\s+").str[0]).values)
    out = out[out["state"].isin(STATES)].copy()
    if c["status"] is not None:
        s = out["status"].str.lower()
        out = out[s.str.contains("active") & ~s.str.contains("inactive")]
    if out["lat"].isna().mean() > 0.5:
        out = census_geocode(out)
    return out

def census_geocode(df):
    """Fallback for a registry without coordinates: US Census batch geocoder (free, no key)."""
    import requests
    df = df.copy()
    lines = df.apply(lambda r: [r.name, (r["hnum"] + " " + r["street"]).strip(), r["city"], r["state"], r["zip5"]], axis=1)
    got = {}
    for i in range(0, len(lines), 9000):
        buf = io.StringIO()
        csv.writer(buf).writerows(lines.iloc[i:i + 9000].tolist())
        try:
            r = requests.post(CENSUS_BATCH, files={"addressFile": ("a.csv", buf.getvalue())},
                              data={"benchmark": "Public_AR_Current"}, timeout=900)
            for row in csv.reader(io.StringIO(r.text)):
                if len(row) >= 6 and row[2] == "Match" and "," in row[5]:
                    x, y = row[5].split(",")[:2]
                    got[int(row[0])] = (float(x), float(y))
        except Exception as e:
            log(f"census geocoder failed: {type(e).__name__}: {e}")
    df["lon"] = [got.get(i, (np.nan, np.nan))[0] for i in df.index]
    df["lat"] = [got.get(i, (np.nan, np.nan))[1] for i in df.index]
    df["geocoded_by"] = "census_batch"
    log(f"census geocoder: {len(got):,} of {len(df):,} matched")
    SOURCES.append({"url": CENSUS_BATCH, "file": "(geocoding)", "retrieved": utcnow(), "rows": len(got)})
    return df

# ---------------------------------------------------------------- Overture places
FOOD_CATS = {"grocery_store", "convenience_store", "supermarket", "discount_store", "gas_station", "truck_gas_station",
             "butcher_shop", "specialty_foods", "fruits_and_vegetables", "health_food_store", "bakery", "farmers_market",
             "department_store", "superstore", "wholesale_store", "drugstore", "pharmacy", "liquor_store",
             "tobacco_shop", "international_grocery_store", "organic_grocery_store", "asian_grocery_store",
             "mexican_grocery_store", "indian_grocery_store", "seafood_market", "health_market", "delicatessen",
             "wholesale_grocer", "imported_food", "frozen_foods", "public_market", "beverage_store", "food",
             "meat_shop", "dollar_store", "shopping", "retail", "candy_store", "flea_market"}
CARE_CATS = {"skilled_nursing", "retirement_home", "assisted_living_facility", "hospice", "memory_care",
             "senior_citizen_services", "home_health_care", "rehabilitation_center", "nursing_home", "elder_care_planning",
             "health_and_medical", "hospital", "medical_center", "disability_services_and_support_organization"}
CLINIC_CATS = {"community_health_center", "public_health_clinic", "health_and_medical", "medical_center", "family_practice",
               "pediatrician", "dentist", "general_dentistry", "womens_health_clinic", "walk_in_clinic",
               "urgent_care_clinic", "counseling_and_mental_health", "hospital", "doctor", "internal_medicine",
               "physician_assistant", "health_department", "eye_care_clinic", "pediatric_dentist", "obstetrician_and_gynecologist",
               "medical_service_organizations", "mental_health", "pharmacy"}
KIND_CATS = {"SNAP": FOOD_CATS, "NH": CARE_CATS, "HC": CLINIC_CATS}
PCOLS = ["id", "names", "categories", "confidence", "addresses", "sources", "operating_status", "brand", "bbox"]

def read_places(fs, base, region):
    pf = open_pf(fs, ref_path(base, region, "overture-pois"))
    cols, gcol, enc = point_cols(pf, PCOLS)
    parts = []
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        sn = batch.schema.names
        xmin, xmax, ymin, ymax = bbox_arrays(batch, gcol, enc)
        d = {"lon": (xmin + xmax) / 2, "lat": (ymin + ymax) / 2}
        d["pid"] = np.asarray(batch.column("id").to_numpy(zero_copy_only=False)).astype(str) if "id" in sn else ""
        d["pname"] = struct_str(batch.column("names"), "primary") if "names" in sn else ""
        if "categories" in sn:
            cats = batch.column("categories")
            d["cat"] = struct_str(cats, "primary")
            try:
                alt = pc.fill_null(pc.binary_join(pc.struct_field(cats, "alternate"), "|"), "")
                d["alt"] = np.asarray(alt.to_numpy(zero_copy_only=False)).astype(str)
            except Exception:
                d["alt"] = ""
        else:
            d["cat"], d["alt"] = "", ""
        d["conf"] = (np.asarray(pc.fill_null(batch.column("confidence"), np.nan).to_numpy(zero_copy_only=False), dtype=float)
                     if "confidence" in sn else np.nan)
        if "addresses" in sn:
            a = batch.column("addresses")
            d["afree"] = first_of_list_field(a, "freeform")
            d["apost"] = first_of_list_field(a, "postcode")
        else:
            d["afree"], d["apost"] = None, None
        d["ds"] = first_source_dataset(batch) if "sources" in sn else "none"
        d["status"] = (np.asarray(pc.fill_null(batch.column("operating_status"), "").to_numpy(zero_copy_only=False)).astype(str)
                       if "operating_status" in sn else "")
        if "brand" in sn:
            try:
                bn = pc.struct_field(pc.struct_field(batch.column("brand"), "names"), "primary")
                d["brand"] = np.asarray(pc.fill_null(bn, "").to_numpy(zero_copy_only=False)).astype(str)
            except Exception:
                d["brand"] = ""
        else:
            d["brand"] = ""
        parts.append(pd.DataFrame(d))
    P = pd.concat(parts, ignore_index=True)
    P["x"], P["y"] = TO_M.transform(P["lon"].values, P["lat"].values)
    P["nname"] = [norm_name(s) for s in P["pname"].values]
    sp = [split_addr(s) for s in P["afree"].values]
    P["hnum"] = [t[0] for t in sp]
    P["street"] = [t[1] for t in sp]
    P["zip5"] = zip5(P["apost"]).values
    alt = P["alt"].astype(str)
    for k, cats in KIND_CATS.items():
        pat = r"(?:^|\|)(?:" + "|".join(sorted(cats)) + r")(?:\||$)"
        P[f"is_{k}"] = P["cat"].isin(cats).values | alt.str.contains(pat, regex=True).values
    log(f"{region}: {len(P):,} Overture places read")
    return P

# ---------------------------------------------------------------- matching
def strict_score(q, qset, c):
    """Name score that does not reward sharing only generic words ('market', 'food', 'nursing')."""
    if not q or not c:
        return 0.0
    cset = set(c.split())
    if (qset & cset) - GENERIC:
        return float(fuzz.token_set_ratio(q, c))
    return float(max(fuzz.token_sort_ratio(q, c), fuzz.ratio(q.replace(" ", ""), c.replace(" ", ""))))

def addr_matches(R, P, max_d=5000.0):
    """Registry entries whose house number + ZIP + street equal an Overture place's address."""
    r = R[(R["hnum"] != "") & (R["zip5"] != "")][["ri", "hnum", "zip5", "street", "x", "y", "nname"]]
    p = P[(P["hnum"] != "") & (P["zip5"] != "")][["hnum", "zip5", "street", "x", "y", "nname"]].reset_index()
    if r.empty or p.empty:
        return pd.DataFrame(columns=["ri", "pi", "addr_ss", "addr_d", "addr_s"])
    m = r.merge(p, on=["hnum", "zip5"], suffixes=("", "_p"))
    if m.empty:
        return pd.DataFrame(columns=["ri", "pi", "addr_ss", "addr_d", "addr_s"])
    m["addr_d"] = np.hypot(m["x"] - m["x_p"], m["y"] - m["y_p"])
    m = m[m["addr_d"] <= max_d]
    m["addr_ss"] = [float(fuzz.token_set_ratio(a, b)) for a, b in zip(m["street"], m["street_p"])]
    m = m[m["addr_ss"] >= 80]
    m["addr_s"] = [strict_score(a, set(a.split()), b) for a, b in zip(m["nname"], m["nname_p"])]
    m = m.sort_values(["ri", "addr_s", "addr_ss", "addr_d"], ascending=[True, False, False, True])
    m = m.drop_duplicates("ri").rename(columns={"index": "pi"})
    return m[["ri", "pi", "addr_ss", "addr_d", "addr_s"]]

def _pfields(P, i, prefix):
    if i is None or i < 0:
        return {f"{prefix}_name": "", f"{prefix}_cat": "", f"{prefix}_conf": np.nan, f"{prefix}_ds": "",
                f"{prefix}_status": "", f"{prefix}_id": ""}
    return {f"{prefix}_name": P["pname"].values[i], f"{prefix}_cat": P["cat"].values[i],
            f"{prefix}_conf": P["conf"].values[i], f"{prefix}_ds": P["ds"].values[i],
            f"{prefix}_status": P["status"].values[i], f"{prefix}_id": P["pid"].values[i]}

def match_region(R, P, k_max=3000, r_max=3000.0, audit_ids=()):
    """For every registry entry in R, the Overture evidence around it."""
    out, audit = [], []
    if len(P) == 0 or len(R) == 0:
        return pd.DataFrame(out), pd.DataFrame(audit)
    tree = cKDTree(np.column_stack([P["x"].values, P["y"].values]))
    K = int(min(k_max, len(P)))
    names = P["nname"].values
    kinds = {k: P[f"is_{k}"].values for k in KIND_CATS}
    A = addr_matches(R, P).set_index("ri")
    audit_ids = set(audit_ids)
    for i0 in range(0, len(R), 500):
        Rc = R.iloc[i0:i0 + 500]
        dd, ii = tree.query(np.column_stack([Rc["x"].values, Rc["y"].values]), k=K, distance_upper_bound=r_max)
        if K == 1:
            dd, ii = dd[:, None], ii[:, None]
        for j in range(len(Rc)):
            r = Rc.iloc[j]
            ok = np.isfinite(dd[j])
            d, idx = dd[j][ok], ii[j][ok]
            rec = {"ri": r["ri"], "n_3km": len(idx), "n_50": int((d <= 50).sum()), "n_150": int((d <= 150).sum()),
                   "n_500": int((d <= 500).sum()), "d_near": d[0] if len(d) else np.nan}
            rec.update(_pfields(P, idx[0] if len(idx) else None, "near"))
            kind = kinds[r["ref_set"]][idx] if len(idx) else np.zeros(0, bool)
            near_kind = np.nonzero(kind & (d <= 100))[0]
            rec["kind_50"] = int((kind & (d <= 50)).sum())
            rec["kind_100"] = int(len(near_kind))
            rec["d_kind"] = d[near_kind[0]] if len(near_kind) else np.nan
            rec.update(_pfields(P, idx[near_kind[0]] if len(near_kind) else None, "kind"))
            s = np.zeros(len(idx))
            queries = [q for q in [r["nname"]] + str(r.get("nalts", "") or "").split("|") if q]
            if queries and len(idx):
                cands = names[idx].tolist()
                for q in queries:
                    ts = process.cdist([q], cands, scorer=fuzz.token_set_ratio, workers=1)[0]
                    hi = np.nonzero(ts >= 60)[0]
                    if len(hi) > 300:
                        hi = hi[np.argsort(-ts[hi], kind="stable")[:300]]
                    qset = set(q.split())
                    for h in hi:
                        s[h] = max(s[h], strict_score(q, qset, cands[h]))
            for lab, lim in (("any", r_max), ("500", 500.0), ("150", 150.0)):
                m = np.nonzero((d <= lim) & (s > 0))[0]
                if len(m):
                    best = m[np.lexsort((d[m], -s[m]))[0]]
                    rec[f"s_{lab}"], rec[f"d_{lab}"] = s[best], d[best]
                    rec.update(_pfields(P, idx[best], f"m{lab}"))
                else:
                    rec[f"s_{lab}"], rec[f"d_{lab}"] = 0.0, np.nan
                    rec.update(_pfields(P, None, f"m{lab}"))
            if r["ri"] in A.index:
                a = A.loc[r["ri"]]
                rec.update({"addr_ss": a["addr_ss"], "addr_d": a["addr_d"], "addr_s": a["addr_s"],
                            "addr_kind": bool(kinds[r["ref_set"]][int(a["pi"])])})
                rec.update(_pfields(P, int(a["pi"]), "addr"))
            else:
                rec.update({"addr_ss": np.nan, "addr_d": np.nan, "addr_s": np.nan, "addr_kind": False})
                rec.update(_pfields(P, None, "addr"))
            out.append(rec)
            if r["ri"] in audit_ids and len(idx):
                top = np.argsort(-s, kind="stable")[:5]
                for rank, h in enumerate(list(range(min(8, len(idx)))) + [t for t in top if s[t] > 0]):
                    audit.append({"ri": r["ri"], "kind": "near" if rank < min(8, len(idx)) else "name",
                                  "d": round(float(d[h]), 1), "s": round(float(s[h]), 1), "place": P["pname"].values[idx[h]],
                                  "cat": P["cat"].values[idx[h]], "addr": P["afree"].values[idx[h]],
                                  "conf": P["conf"].values[idx[h]], "ds": P["ds"].values[idx[h]]})
    return pd.DataFrame(out), pd.DataFrame(audit)

def chain_flags(R):
    """A store name used by 10 or more registry entries in the study states is a chain."""
    key = R["nname"]
    cnt = key.map(key.value_counts())
    return (cnt >= 10) & (key != ""), key

# ---------------------------------------------------------------- the run
def run_lifelines(fs, base, out_zip, workdir="/content/lifelines", regions=REGIONS, loaders=None, seed=7):
    t0 = time.time()
    os.makedirs(workdir, exist_ok=True)
    LOG.clear(); SOURCES.clear()
    log(f"lifelines audit started {utcnow()}")
    loaders = loaders or {"SNAP": lambda: load_snap(workdir)[0], "NH": lambda: load_nh(workdir), "HC": lambda: load_hc(workdir)}
    regs = []
    for k, f in loaders.items():
        try:
            d = f()
            if d is None or len(d) == 0:
                log(f"{k}: NOT AVAILABLE")
                continue
            d = d[d["lat"].notna() & d["lon"].notna()].copy()
            regs.append(d)
            log(f"{k}: {len(d):,} entries with coordinates in {sorted(d['state'].unique())}")
        except Exception as e:
            log(f"{k}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    if not regs:
        raise RuntimeError("no registry could be loaded; see the log")
    R = pd.concat(regs, ignore_index=True)
    has_id = R["ref_id"].fillna("").astype(str) != ""
    R = pd.concat([R[has_id].drop_duplicates(["ref_set", "ref_id"]), R[~has_id]], ignore_index=True)
    R["nname"] = [norm_name(s) for s in R["name"].values]
    if "alt_names" not in R.columns:
        R["alt_names"] = ""
    R["nalts"] = ["|".join(sorted({norm_name(a) for a in str(v).split("|")} - {"", n}))
                  for v, n in zip(R["alt_names"].fillna("").values, R["nname"].values)]
    R["chain"] = False
    for k in R["ref_set"].unique():
        m = R["ref_set"] == k
        R.loc[m, "chain"] = chain_flags(R[m])[0].values
    R["x"], R["y"] = TO_M.transform(R["lon"].values, R["lat"].values)
    R["GEOID"], R["region"] = "", ""
    R = R.reset_index(drop=True)
    R["ri"] = np.arange(len(R))
    res, aud, typed = [], [], []
    rng = np.random.default_rng(seed)
    for region in regions:
        try:
            tr = time.time()
            st = STATE_OF.get(region, ())
            T = load_tracts(fs, base, region)
            sel = np.nonzero((R["state"].isin(st) | (R["state"] == "")).values & (R["region"] == "").values)[0]
            if len(sel) == 0:
                log(f"{region}: no registry entries in {st}")
                continue
            ti = assign(T["tree"], R["lon"].values[sel], R["lat"].values[sel])
            inside = sel[ti >= 0]
            R.loc[inside, "GEOID"] = T["gdf"]["GEOID"].values[ti[ti >= 0]]
            R.loc[inside, "region"] = region
            Rr = R.loc[inside]
            log(f"{region}: {len(Rr):,} registry entries inside the study tracts "
                f"({Rr['ref_set'].value_counts().to_dict()})")
            P = read_places(fs, base, region)
            # audit sample: up to 150 random entries per registry plus up to 400 weak-evidence ones (chosen later)
            samp = []
            for k, g in Rr.groupby("ref_set"):
                samp += list(rng.choice(g["ri"].values, size=min(150, len(g)), replace=False))
            m, a = match_region(Rr, P, audit_ids=samp)
            # second pass: full audit lines for entries with weak evidence, so misses can be checked by eye
            if len(m):
                weak = m[(m["s_500"] < 80) & (m["addr_ss"].isna())]["ri"].values
                weak = [w for w in weak if w not in set(samp)]
                if len(weak):
                    pick_w = list(rng.choice(weak, size=min(400, len(weak)), replace=False))
                    _, a2 = match_region(R.loc[R["ri"].isin(pick_w)], P, audit_ids=pick_w)
                    a = pd.concat([a, a2], ignore_index=True)
            res.append(m); aud.append(a)
            tp = P[P[[f"is_{k}" for k in KIND_CATS]].any(axis=1)].copy()
            tp["GEOID"] = ""
            ti2 = assign(T["tree"], tp["lon"].values, tp["lat"].values)
            tp.loc[ti2 >= 0, "GEOID"] = T["gdf"]["GEOID"].values[ti2[ti2 >= 0]]
            tp["region"] = region
            typed.append(tp[["pid", "pname", "cat", "alt", "conf", "ds", "status", "brand", "lon", "lat", "GEOID", "region"]
                            + [f"is_{k}" for k in KIND_CATS]])
            # counts of every Overture place per tract, for densities
            log(f"{region}: matched in {time.time() - tr:.0f} s")
            del P, T; gc.collect()
        except Exception as e:
            log(f"{region}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
    M = pd.concat(res, ignore_index=True) if res else pd.DataFrame({"ri": []})
    out = R[R["region"] != ""].drop(columns=["x", "y"]).merge(M, on="ri", how="left")
    # default verdicts (the analysis re-derives them; kept here so the zip is readable on its own)
    nm150 = out["s_150"].fillna(0) >= 80
    nm500 = out["s_500"].fillna(0) >= 90
    nm_far = (~out["chain"]) & (out["s_any"].fillna(0) >= 95)
    ad = out["addr_ss"].notna() & ((out["addr_s"].fillna(0) >= 60) | out["addr_kind"].fillna(False).astype(bool))
    out["found_name"] = nm150 | nm500 | nm_far
    out["found_addr"] = ad
    out["found"] = out["found_name"] | out["found_addr"]
    out["same_spot_kind"] = (~out["found"]) & (out["kind_50"].fillna(0) > 0)
    os.makedirs(os.path.join(workdir, "out"), exist_ok=True)
    od = os.path.join(workdir, "out")
    out.to_csv(os.path.join(od, "lifelines_registry_matches.csv"), index=False)
    A = pd.concat(aud, ignore_index=True) if aud else pd.DataFrame()
    if len(A):
        A = A.merge(R[["ri", "ref_set", "ref_id", "name", "rtype", "hnum", "street", "city", "zip5", "region", "GEOID"]],
                    on="ri", how="left")
    A.to_csv(os.path.join(od, "lifelines_audit_candidates.csv"), index=False)
    TP = pd.concat(typed, ignore_index=True) if typed else pd.DataFrame()
    TP.to_csv(os.path.join(od, "lifelines_overture_typed_places.csv"), index=False)
    allR = R.drop(columns=["x", "y"])
    allR.to_csv(os.path.join(od, "lifelines_registry_all_states.csv"), index=False)
    summ = out.groupby(["ref_set", "region"]).agg(n=("ri", "size"), found=("found", "mean"),
                                                  found_name=("found_name", "mean"), found_addr=("found_addr", "mean"),
                                                  same_spot=("same_spot_kind", "mean")).round(3)
    log("summary (share found):\n" + summ.to_string())
    log(f"total {time.time() - t0:.0f} s")
    with open(os.path.join(od, "sources.json"), "w") as f:
        json.dump(SOURCES, f, indent=1, default=str)
    with open(os.path.join(od, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG))
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(od)):
            z.write(os.path.join(od, fn), fn)
    log(f"wrote {out_zip} ({os.path.getsize(out_zip)/1e6:.1f} MB)")
    return out_zip

# ---------------------------------------------------------------- Census context tables (optional, for the write-up)
CENSUS_API = "https://api.census.gov/data"
STATE_FIPS = {"AZ": "04", "CA": "06", "NM": "35", "OK": "40", "TX": "48", "WA": "53"}

def _census_get(url, params):
    import requests
    try:
        r = requests.get(url, params=params, headers=UA, timeout=120)
        if r.status_code != 200:
            return None
        j = r.json()
        return pd.DataFrame(j[1:], columns=j[0]) if isinstance(j, list) and len(j) > 1 else None
    except Exception:
        return None

def census_extras(out_zip, workdir="/content/lifelines"):
    """ACS tract context (SNAP households, Hispanic share, income) and county counts of employer
    (CBP) and owner-operated (Nonemployer Statistics) businesses. Added to the zip; never used for
    the leaderboard."""
    od = os.path.join(workdir, "out")
    os.makedirs(od, exist_ok=True)
    acs_vars = "B22010_001E,B22010_002E,B03003_001E,B03003_003E,B19013_001E"
    parts = []
    for st, fips in STATE_FIPS.items():
        got = None
        for year in (2024, 2023, 2022):
            for params in ({"get": acs_vars, "for": "tract:*", "in": f"state:{fips}"},
                           {"get": acs_vars, "for": "tract:*", "in": [f"state:{fips}", "county:*"]}):
                got = _census_get(f"{CENSUS_API}/{year}/acs/acs5", params)
                if got is not None:
                    got["acs_year"] = year
                    break
            if got is not None:
                break
        log(f"ACS {st}: {'none' if got is None else f'{len(got):,} tracts ({got.acs_year.iloc[0]})'}")
        if got is not None:
            parts.append(got)
    files = []
    if parts:
        A = pd.concat(parts, ignore_index=True)
        A["GEOID"] = A["state"] + A["county"] + A["tract"]
        A.to_csv(os.path.join(od, "acs_tract_context.csv"), index=False)
        files.append("acs_tract_context.csv")
        SOURCES.append({"url": f"{CENSUS_API}/<year>/acs/acs5 ({acs_vars})", "retrieved": utcnow(), "rows": len(A)})
    rows = []
    for st, fips in STATE_FIPS.items():
        for ds, var, years in (("cbp", "ESTAB", (2023, 2022, 2021)), ("nonemp", "NESTAB", (2023, 2022, 2021))):
            for sector in ("00", "44-45"):
                got = None
                for year in years:
                    for nv in ("NAICS2022", "NAICS2017"):
                        got = _census_get(f"{CENSUS_API}/{year}/{ds}", {"get": f"{var},NAME", "for": "county:*",
                                                                        "in": f"state:{fips}", nv: sector})
                        if got is not None:
                            got = got.rename(columns={var: "estab"})
                            got["dataset"], got["year"], got["sector"] = ds, year, sector
                            break
                    if got is not None:
                        break
                if got is not None:
                    rows.append(got[["state", "county", "NAME", "estab", "dataset", "year", "sector"]])
        log(f"CBP and nonemployer counts {st}: {sum(len(r) for r in rows if r['state'].iloc[0] == fips):,} rows")
    if rows:
        B = pd.concat(rows, ignore_index=True)
        B.to_csv(os.path.join(od, "county_business_counts.csv"), index=False)
        files.append("county_business_counts.csv")
        SOURCES.append({"url": f"{CENSUS_API}/<year>/cbp and /nonemp (ESTAB, NESTAB by county; all sectors and retail 44-45)",
                        "retrieved": utcnow(), "rows": len(B)})
    with open(os.path.join(od, "sources.json"), "w") as f:
        json.dump(SOURCES, f, indent=1, default=str)
    with open(os.path.join(od, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG))
    with zipfile.ZipFile(out_zip, "a", zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for fn in files:
            if fn not in have:
                z.write(os.path.join(od, fn), fn)
    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()
    # rewrite sources.json and run_log.txt inside the zip with the new lines
    tmp = out_zip + ".tmp"
    with zipfile.ZipFile(out_zip, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in ("sources.json", "run_log.txt"):
                continue
            zout.writestr(item, zin.read(item.filename))
        zout.write(os.path.join(od, "sources.json"), "sources.json")
        zout.write(os.path.join(od, "run_log.txt"), "run_log.txt")
    os.replace(tmp, out_zip)
    log(f"census context added: {files}")
    return out_zip
