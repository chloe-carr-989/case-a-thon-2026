"""Fetch hourly 10 m wind (ERA5 reanalysis via the Open-Meteo archive API, no key) for the held-out period.

Points: centres of 0.5 deg cells that contain learned lane traffic (results/lanes_cells.parquet), so the
weather layer covers where vessels are, not open land. Output: results/weather_hourly.parquet with
columns t (UTC epoch hour), lat, lon, wind_kn, wind_dir_deg (direction the wind comes FROM), gust_kn, source.
Run once with network access; the app then works offline. Only coordinates and dates are sent.

    .venv\\Scripts\\python scripts\\fetch_weather.py                 # held-out period from results/manifest.json
    .venv\\Scripts\\python scripts\\fetch_weather.py --start 2026-01-04 --end 2026-01-04
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
API = "https://archive-api.open-meteo.com/v1/archive"


def grid(cell: float = 0.5) -> pd.DataFrame:
    c = pd.read_parquet(RES / "lanes_cells.parquet")
    c = c[(c.res == "coarse") & (c.g == "commercial") & c.lane]
    g = pd.DataFrame({"lat": (np.floor(c.lat / cell) + 0.5) * cell, "lon": (np.floor(c.lon / cell) + 0.5) * cell})
    return g.drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)


def fetch(pts: pd.DataFrame, start: str, end: str, batch: int = 50) -> pd.DataFrame:
    out = []
    for s in range(0, len(pts), batch):
        p = pts.iloc[s:s + batch]
        q = {"latitude": ",".join(f"{x:.2f}" for x in p.lat), "longitude": ",".join(f"{x:.2f}" for x in p.lon),
             "start_date": start, "end_date": end, "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
             "wind_speed_unit": "kn", "timezone": "GMT", "models": "era5"}
        for attempt in range(6):          # the free API limits location-requests per minute: wait and retry
            try:
                with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                print("  rate limited; waiting 65 s")
                time.sleep(65)
        for (_, pt), d in zip(p.iterrows(), data if isinstance(data, list) else [data]):
            h = d["hourly"]
            out.append(pd.DataFrame({"t": (pd.to_datetime(h["time"]) - pd.Timestamp(0)) // pd.Timedelta("1s"), "lat": pt.lat,
                                     "lon": pt.lon, "wind_kn": h["wind_speed_10m"],
                                     "wind_dir_deg": h["wind_direction_10m"], "gust_kn": h["wind_gusts_10m"]}))
        print(f"  {min(s + batch, len(pts))}/{len(pts)} points")
        time.sleep(8.0)
    w = pd.concat(out, ignore_index=True).dropna(subset=["wind_kn", "wind_dir_deg"])
    w["source"] = "ERA5 reanalysis via Open-Meteo archive API"
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start"), ap.add_argument("--end")
    a = ap.parse_args()
    m = json.loads((RES / "manifest.json").read_text())
    start = a.start or m["split"]["split_t_utc"][:10]
    end = a.end or m["dataset"]["profile_observed"]["time_max"][:10]
    pts = grid()
    print(f"fetching {start}..{end} for {len(pts)} points")
    w = fetch(pts, start, end)
    w.to_parquet(RES / "weather_hourly.parquet", index=False)
    print(f"wrote {len(w):,} rows to {RES / 'weather_hourly.parquet'}")


if __name__ == "__main__":
    main()
