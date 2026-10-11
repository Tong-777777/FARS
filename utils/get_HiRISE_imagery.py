from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ----------------------------- Config -----------------------------

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 "
    "HiRISE-PDS-JP2-Downloader/1.0"
)

TIMEOUT_S = 120
HEAD_TIMEOUT_S = 30
RETRIES = 5
SLEEP_S = 0.4  # polite delay between obs

# obs_id like ESP_013277_2155
OBS_ID_RE = re.compile(r"^(ESP|PSP|TRA)_(\d{6})_(\d{4})$")


# ----------------------------- URL builder -----------------------------

def orbit_range_folder(orbit6: str) -> str:
    """013277 -> ORB_013200_013299 (bucket by 100)"""
    orbit = int(orbit6)
    start = (orbit // 100) * 100
    end = start + 99
    return f"ORB_{start:06d}_{end:06d}"


def build_red_nomap_url(obs_id: str) -> str:
    """
    Build unprojected JP2 (RED.NOMAP) URL on hirise-pds.

    Example:
    https://hirise-pds.lpl.arizona.edu/download/PDS/EXTRAS/RDR/ESP/ORB_013200_013299/ESP_013277_2155/ESP_013277_2155_RED.NOMAP.JP2
    """
    m = OBS_ID_RE.match(obs_id)
    if not m:
        raise ValueError(f"Bad obs_id: {obs_id}")

    phase = m.group(1)      # ESP/PSP/TRA
    orbit6 = m.group(2)     # 6-digit orbit number
    orb = orbit_range_folder(orbit6)

    return (
        "https://hirise-pds.lpl.arizona.edu/download/PDS/EXTRAS/RDR/"
        f"{phase}/{orb}/{obs_id}/{obs_id}_RED.NOMAP.JP2"
    )


# ----------------------------- IO helpers -----------------------------

def read_obs_ids_from_csv(csv_path: Path) -> List[str]:
    """
    Reads obs_id from either 'Observation ID' or 'obs_id' column.
    Deduplicates while preserving order.
    """
    obs_ids: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = (row.get("Observation ID") or row.get("obs_id") or "").strip()
            if oid and OBS_ID_RE.match(oid):
                obs_ids.append(oid.upper())

    # de-dup keep order
    seen = set()
    out = []
    for x in obs_ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def append_failed(log_csv: Path, row: Dict[str, str]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_csv.exists()
    with log_csv.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["obs_id", "url", "reason"])
        if new_file:
            w.writeheader()
        w.writerow(row)


# ----------------------------- Network helpers -----------------------------

def head_ok(session: requests.Session, url: str) -> bool:
    try:
        r = session.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT_S)
        return r.status_code == 200
    except Exception:
        return False


def download_with_resume(session: requests.Session, url: str, out_path: Path) -> None:
    """
    Download with resume (Range). If file exists, continue downloading.
    Retries on transient failures.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # determine resume position
    have = out_path.stat().st_size if out_path.exists() else 0
    headers = {}
    mode = "wb"
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"

    last_err: Optional[Exception] = None
    for i in range(RETRIES):
        try:
            with session.get(url, stream=True, headers=headers, timeout=TIMEOUT_S) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")

                with out_path.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))

    raise RuntimeError(f"Download failed after {RETRIES} retries: {last_err}")


# ----------------------------- Main -----------------------------

def main():

    CSV_PATH = Path(r"D:\PHD_learning\paper_writting\landing_site_statistics\1000km\Terra Sirenum.csv")
    OUT_DIR = Path(r"F:\mars_rock\Terra Sirenum\imagery")
    FAILED_LOG = OUT_DIR / "_failed_downloads.csv"

    obs_ids = read_obs_ids_from_csv(CSV_PATH)
    print(f"Loaded obs_ids: {len(obs_ids)} from {CSV_PATH}")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    for i, obs_id in enumerate(obs_ids, 1):
        url = build_red_nomap_url(obs_id)
        out_path = OUT_DIR / obs_id / f"{obs_id}_RED.NOMAP.JP2"

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i}/{len(obs_ids)}] {obs_id}  skip (exists)")
            continue

        print(f"[{i}/{len(obs_ids)}] {obs_id}")

        # quick existence check (avoid long GET on 404)
        if not head_ok(session, url):
            reason = "HEAD!=200 (file not found or blocked)"
            print(f"  FAIL: {reason}")
            append_failed(FAILED_LOG, {"obs_id": obs_id, "url": url, "reason": reason})
            time.sleep(SLEEP_S)
            continue

        try:
            print(f"  url: {url}")
            download_with_resume(session, url, out_path)
            mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  saved: {out_path}  ({mb:.1f} MB)")
        except Exception as e:
            reason = str(e)
            print(f"  FAIL: {reason}")
            append_failed(FAILED_LOG, {"obs_id": obs_id, "url": url, "reason": reason})

        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()