"""
HiRISE crawler (HiWish API + observation page scrape)

Pipeline per landing site:
1) /hiwish/locate/?q=...  -> "Name,lat,lon"
2) (lon,lat) -> (x,y) using Mars Mercator-like projection
3) bbox: x0=x-dx, x1=x+dx, y0=y-dy, y1=y+dy  (dx/dy from radius_km and aspect ratio)
4) /hiwish/footprints/?x0..&type=HiRDR&json -> GeoJSON-like features
5) Extract Observation IDs (ESP_/PSP_/TRA_...) from properties
6) For each obs_id: fetch https://www.uahirise.org/{obs_id} and parse required fields
7) Save one CSV per landing site

Dependencies:
  pip install requests beautifulsoup4
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

# ----------------------------- Config -----------------------------

RADIUS = 1000

SAVE_PATH = r'D:\PHD_learning\paper_writting\landing_site_statistics\{}km'.format(RADIUS)

BASE = "https://www.uahirise.org"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 "
    "HiriseFootprintCrawler/0.1"
)

# Mars radius used by the HiWish map projection (meters).
# This value reproduces the example bbox center very closely.
MARS_R_M = 3_394_500.0

# How wide the bbox is relative to its height (x half-width : y half-height).
# Your example had dx ≈ 330 km, dy ≈ 165 km => ratio ≈ 2.0
BBOX_ASPECT_X_OVER_Y = 1.0

# Politeness
SLEEP_S_LIST = 0.5       # locate + footprints requests
SLEEP_S_OBS = 0.7        # observation page requests
TIMEOUT_S = 30
RETRIES = 3


OBS_ID_RE = re.compile(r"\b(?:ESP|PSP|TRA)_[0-9]{6}_[0-9]{4}\b")


# Your required fields -> normalized label key (lowercase, no trailing colon)
FIELD_MAP: Dict[str, str] = {
    "Acquisition date": "acquisition date",
    "Local Mars time": "local mars time",
    "Latitude (centered)": "latitude (centered)",
    "Longitude (East)": "longitude (east)",
    "Spacecraft altitude": "spacecraft altitude",
    "Original image scale range": "original image scale range",
    "Map projected scale": "map projected scale",
    "Map projection": "map projection",
    "Emission angle": "emission angle",
    "Phase angle": "phase angle",
    "Solar incidence angle": "solar incidence angle",
    "Solar longitude": "solar longitude",
    "North azimuth": "north azimuth",
    "Sub-solar azimuth": "sub-solar azimuth",
}


# ----------------------------- Helpers -----------------------------

@dataclass
class LocateResult:
    name: str
    lat_deg: float
    lon_deg: float  # East-positive


def _norm_label(s: str) -> str:
    s = s.strip().replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(":")
    return s.lower()


def _fetch_text(session: requests.Session, url: str, *, params=None, sleep_s=0.5, debug=False) -> str:
    last_err = None

    for i in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT_S)
            status = r.status_code
            text = (r.text or "").strip()

            if debug:
                print(f"[GET] {r.url}")
                print(f"      status={status}, len={len(text)}, head={text[:100]!r}")

            if "<html" in text.lower():
                raise RuntimeError(f"HTML response (blocked/error). status={status}")

            if not text:
                raise RuntimeError(f"empty response. status={status}")

            time.sleep(sleep_s)
            return text

        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed after {RETRIES} retries: {url} ({last_err})")


def locate(session: requests.Session, query: str, debug: bool = True) -> LocateResult:
    url = f"{BASE}/hiwish/locate/?q={quote_plus(query)}"
    txt = _fetch_text(session, url, sleep_s=SLEEP_S_LIST, debug=debug)

    parts = [p.strip() for p in txt.split(",")]
    if debug:
        print("[locate] parts =", parts)

    if len(parts) < 3:
        raise ValueError(f"Unexpected locate response: {txt!r}")

    return LocateResult(name=parts[0], lat_deg=float(parts[1]), lon_deg=float(parts[2]))



def lonlat_to_xy_m(lon_deg: float, lat_deg: float, R: float = MARS_R_M) -> Tuple[float, float]:
    """
    Mars Mercator-like projection used in HiWish:
      x = R * lon_rad
      y = R * ln(tan(pi/4 + lat_rad/2))
    """
    # clamp lat a bit to avoid infinity at poles
    lat_deg = max(-89.9999, min(89.9999, lat_deg))
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


def footprints(session: requests.Session, x0: float, x1: float, y0: float, y1: float, ftype: str = "HiRDR") -> dict:
    """
    https://www.uahirise.org/hiwish/footprints/?x0=...&x1=...&y0=...&y1=...&type=HiRDR&json
    """
    url = f"{BASE}/hiwish/footprints/"
    params = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "type": ftype, "json": ""}
    txt = _fetch_text(session, url, params=params, sleep_s=SLEEP_S_LIST)
    return json.loads(txt)


def extract_obs_ids_from_footprints(payload: dict) -> List[str]:
    """
    Accepts either:
    A) dict keyed by obs_id:
       { "ESP_012345_6789": {...}, "PSP_...": {...} }
    B) GeoJSON FeatureCollection:
       { "type":"FeatureCollection", "features":[{"properties":...}, ...] }

    Returns sorted unique obs_ids.
    """
    obs_ids = set()

    if not isinstance(payload, dict):
        return []

    # Case A: keys are obs_ids
    # quick check: do we have at least one key matching the pattern?
    keys = list(payload.keys())
    if any(OBS_ID_RE.fullmatch(str(k)) for k in keys):
        for k in keys:
            k = str(k)
            if OBS_ID_RE.fullmatch(k):
                obs_ids.add(k)
        return sorted(obs_ids)

    # Case B: GeoJSON FeatureCollection
    feats = payload.get("features", None)
    if isinstance(feats, list):
        for f in feats:
            # search everywhere in each feature dict (robust to property name changes)
            s = json.dumps(f, ensure_ascii=False)
            for m in OBS_ID_RE.findall(s):
                obs_ids.add(m)
        return sorted(obs_ids)

    # Fallback: search whole payload
    s = json.dumps(payload, ensure_ascii=False)
    for m in OBS_ID_RE.findall(s):
        obs_ids.add(m)
    return sorted(obs_ids)


# ----------------------------- Observation page scrape -----------------------------

def _extract_kv_from_obs_page(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Build a normalized {label->value} dictionary from <dt>/<dd>, tables, and
    minimal regex fallbacks for azimuth lines.
    """
    data: Dict[str, str] = {}

    # 1) <dt>Label</dt><dd>Value</dd>
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        label = _norm_label(dt.get_text(" ", strip=True))
        value = re.sub(r"\s+", " ", dd.get_text(" ", strip=True)).strip()
        if label and value:
            data[label] = value

    # 2) tables (<th>/<td> or <td>/<td>)
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = _norm_label(cells[0].get_text(" ", strip=True))
        value = re.sub(r"\s+", " ", cells[1].get_text(" ", strip=True)).strip()
        if label and value and label not in data:
            data[label] = value

    # 3) minimal regex fallback for azimuth lines
    text = soup.get_text("\n")
    m = re.search(r"North azimuth:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m and "north azimuth" not in data:
        data["north azimuth"] = re.sub(r"\s+", " ", m.group(1)).strip()

    m = re.search(r"Sub-solar azimuth:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m and "sub-solar azimuth" not in data:
        data["sub-solar azimuth"] = re.sub(r"\s+", " ", m.group(1)).strip()

    return data

PAGE_FIELDS = [
    ("Acquisition date", "Acquisition date"),
    ("Local Mars time", "Local Mars time"),
    ("Latitude (centered)", "Latitude (centered)"),
    ("Longitude (East)", "Longitude (East)"),
    ("Spacecraft altitude", "Spacecraft altitude"),
    ("Original image scale range", "Original image scale range"),
    ("Map projected scale", "Map projected scale"),
    ("Map projection", "Map projection"),
    ("Emission angle", "Emission angle"),
    ("Phase angle", "Phase angle"),
    ("Solar incidence angle", "Solar incidence angle"),
    ("Solar longitude", "Solar longitude"),
]

def scrape_observation_page(obs_id: str, session: requests.Session) -> Optional[Dict[str, str]]:
    url = f"https://www.uahirise.org/{obs_id}"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    out: Dict[str, str] = {"Observation ID": obs_id, "URL": url, "Source": "Observation page"}

    # Basic key->next line parsing
    idx_map = {ln: i for i, ln in enumerate(lines)}
    for label, col in PAGE_FIELDS:
        if label in idx_map and idx_map[label] + 1 < len(lines):
            out[col] = lines[idx_map[label] + 1]

    # Non-map projected azimuth lines are formatted like:
    # "North azimuth:  94°"
    # "Sub-solar azimuth:  332.0°"
    for ln in lines:
        if ln.startswith("North azimuth"):
            # keep only numeric part with degree symbol
            m = re.search(r"North azimuth:\s*([0-9.]+°)", ln)
            if m:
                out["For non-map projected images: North azimuth"] = m.group(1)
        if ln.startswith("Sub-solar azimuth"):
            m = re.search(r"Sub-solar azimuth:\s*([0-9.]+°)", ln)
            if m:
                out["For non-map projected images: Sub-solar azimuth"] = m.group(1)

    # If the page didn't contain "Acquisition date", treat as not a caption page
    if "Acquisition date" not in out:
        return None
    return out


def parse_observation(session: requests.Session, obs_id: str) -> Dict[str, Optional[str]]:
    # url = f"{BASE}/{obs_id}"
    result = scrape_observation_page(obs_id, session)

    return result


# ----------------------------- Main per landing site -----------------------------

def bbox_from_center_xy(
    x: float,
    y: float,
    radius_km_y: float,
    aspect_x_over_y: float = BBOX_ASPECT_X_OVER_Y,
) -> Tuple[float, float, float, float]:
    """
    Create bbox from center (x,y).
    Use dy = radius_km_y, dx = dy * aspect ratio.

    - If you want a true "square" in projected meters, set aspect_x_over_y = 1.0
    - If you want to mimic your example view (wide), keep 2.0
    """
    dy = radius_km_y * 1000.0
    dx = dy * aspect_x_over_y
    return (x - dx, x + dx, y - dy, y + dy)


def crawl_landing_site(
    session: requests.Session,
    site_query: str,
    radius_km_y: float = 165.0,
    footprint_type: str = "HiRDR",
) -> Tuple[LocateResult, List[str], List[Dict[str, Optional[str]]]]:
    """
    radius_km_y controls y half-height in km (dy). x half-width will be dy*aspect.

    Your example bbox corresponds roughly to:
      dy ≈ 165 km, dx ≈ 330 km (aspect 2.0)
    """
    loc = locate(session, site_query)
    x, y = lonlat_to_xy_m(loc.lon_deg, loc.lat_deg)
    x0, x1, y0, y1 = bbox_from_center_xy(x, y, radius_km_y, BBOX_ASPECT_X_OVER_Y)

    fp = footprints(session, x0, x1, y0, y1, ftype=footprint_type)
    obs_ids = extract_obs_ids_from_footprints(fp)

    rows: List[Dict[str, Optional[str]]] = []

    fields = ["Observation ID", "URL", "Acquisition date", "Local Mars time", "Latitude (centered)", "Longitude (East)", "Spacecraft altitude",
              "Original image scale range", "Map projected scale", "Map projection", "Emission angle", "Emission angle", "Phase angle", "Solar incidence angle", "Solar longitude"]
    csv_path = os.path.join(SAVE_PATH, '{}.csv'.format(site_query))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, oid in enumerate(obs_ids, 1):

            try:
                results = parse_observation(session, oid)
                print(results)
                w.writerow({k: results.get(k) for k in fields})

                # w.writerow(
                #     results["Acquisition date"],
                #     results["Local Mars time"],
                #     results["Latitude (centered)"],
                #     results["Longitude (East)"],
                #     results["Spacecraft altitude"],
                #     results["Original image scale range"],
                #     results["Map projected scale"],
                #     results["Map projection"],
                #     results["Emission angle"],
                #     results["Phase angle"],
                #     results["Solar incidence angle"],
                #     results["Solar longitude"]
                # )

            except Exception as e:
                # Keep going; record error
                rows.append({"obs_id": oid, "url": f"{BASE}/{oid}", "error": str(e)})
            if i % 50 == 0:
                print(f"  {site_query}: parsed {i}/{len(obs_ids)}")

    return loc, obs_ids, rows


def save_csv(rows: List[Dict[str, Optional[str]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["obs_id", "url"] + list(FIELD_MAP.keys())
    # allow optional error column
    if any("error" in r for r in rows):
        fieldnames.append("error")

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # ensure all keys exist
            for k in fieldnames:
                r.setdefault(k, None)
            w.writerow({k: r.get(k) for k in fieldnames})


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]+", "", s)


def main():
    landing_sites = [
        "Hellespontus Montes",
        "Terra Sirenum",
        "Margaritifer Terra",
        "Chryse Planitia",
        "Utopia Planitia",
        "Planum Australe",
        "Ares Vallis",
        # "Near MPL",
        "Gusev",
        "Meridiani Planum",
        "Vastitas Borealis",
        "Gale",
        "Elysium Planitia",
        "Jezero",
        "Isidis Planitia",
        "Meridiani Planum"]

    out_dir = Path(SAVE_PATH)

    os.makedirs(out_dir, exist_ok=True)

    # This matches your example view size roughly (dy=165km, dx=330km).
    radius_km_y = RADIUS

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    for site in landing_sites:
        print(f"[{site}] locate -> footprints -> obs pages ...")
        loc, obs_ids, rows = crawl_landing_site(
            session,
            site_query=site,
            radius_km_y=radius_km_y,
            footprint_type="HiRDR",
        )

        # file name includes resolved name + center coords

if __name__ == '__main__':
    main()
