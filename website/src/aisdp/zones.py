"""Dwell detection and train-period-derived study zones (shared by destination and port occupancy).

No port geometry ships with the data, so zones are *inferred arrival / dwell zones*: grid cells where
several distinct vessels sustained low speed during the training period. They are labelled by their
coordinates, never by a port name, and are not verified terminal or berth boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import R_EARTH
from .ingest import parquet_glob

VESSEL_WHERE = ("mmsi BETWEEN 201000000 AND 775999999 AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180 "
                "AND NOT (lat = 0 AND lon = 0)")


def clean_sog_sql(col: str = "sog") -> str:
    return f"CASE WHEN {col} >= 102.2 OR {col} < 0 THEN NULL ELSE {col} END"


def build_vessel_buckets(con, info: dict, bucket_s: int, table: str = "vb") -> int:
    """One row per (vessel, time bucket): mean position, max/mean SOG, report count, vessel type.

    Means are rounded to 1e-6 deg so parallel float summation order cannot move a centroid across a cell edge."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT mmsi, floor(epoch(ts) / {bucket_s})::BIGINT AS b, round(avg(lat), 6) AS lat, round(avg(lon), 6) AS lon,
               max({clean_sog_sql()}) AS max_sog, round(avg({clean_sog_sql()}), 4) AS sog, count(*) AS n,
               min(vessel_type) AS vessel_type
        FROM read_parquet('{parquet_glob(info)}', hive_partitioning=true)
        WHERE {VESSEL_WHERE}
        GROUP BY 1, 2
    """)
    return con.execute(f"SELECT count(*) FROM {table}").fetchall()[0][0]


def dwell_events(con, dc, table: str = "vb", out: str = "dwell") -> pd.DataFrame:
    """Runs of >= dwell_min_buckets consecutive buckets with max SOG < dwell_max_sog_kn.

    `arrival` marks dwells preceded by an underway bucket (mean SOG >= underway_min_sog_kn) within
    arrival_lookback_buckets: a vessel that was moving and then stopped. `confirm_t` is the earliest
    time the dwell is known to be sustained, i.e. when the label could first be observed.
    """
    B = dc.bucket_s
    con.execute(f"""
        CREATE OR REPLACE TABLE {out} AS
        WITH s AS (SELECT *, b - row_number() OVER (PARTITION BY mmsi ORDER BY b) AS grp
                   FROM {table} WHERE max_sog IS NOT NULL AND max_sog < {dc.dwell_max_sog_kn}),
        runs AS (SELECT mmsi, min(b) AS b0, max(b) AS b1, count(*) AS n_buckets, round(avg(lat), 6) AS lat, round(avg(lon), 6) AS lon
                 FROM s GROUP BY mmsi, grp HAVING count(*) >= {dc.dwell_min_buckets})
        SELECT r.*, (r.b0 * {B})::DOUBLE AS start_t, ((r.b1 + 1) * {B})::DOUBLE AS end_t,
               ((r.b0 + {dc.dwell_min_buckets}) * {B})::DOUBLE AS confirm_t,
               EXISTS (SELECT 1 FROM {table} u WHERE u.mmsi = r.mmsi AND u.sog >= {dc.underway_min_sog_kn}
                       AND u.b BETWEEN r.b0 - {dc.arrival_lookback_buckets} AND r.b0 - 1) AS arrival
        FROM runs r
    """)
    return con.execute(f"SELECT * FROM {out} ORDER BY mmsi, b0").df()


def cell_ij(lat, lon, deg: float):
    return np.floor(np.asarray(lat, float) / deg).astype(np.int64), np.floor(np.asarray(lon, float) / deg).astype(np.int64)


def cell_area_km2(i, deg: float):
    """Spherical area of a deg x deg cell whose southern edge is i*deg (cells shrink toward the poles)."""
    lat0 = np.radians(np.asarray(i, float) * deg)
    lat1 = np.radians((np.asarray(i, float) + 1) * deg)
    return (R_EARTH / 1000) ** 2 * np.radians(deg) * np.abs(np.sin(lat1) - np.sin(lat0))


def fit_zones(dw: pd.DataFrame, split_t: float, deg: float, min_vessels: int, max_zones: int | None = None,
              kind: str = "inferred arrival zone") -> pd.DataFrame:
    """Zones from dwell events that END before split_t (training period only; no test-period labels)."""
    d = dw[dw.end_t <= split_t]
    if d.empty:
        return pd.DataFrame(columns=["zone_id", "zi", "zj", "lat", "lon", "lat0", "lat1", "lon0", "lon1",
                                     "train_vessels", "train_arrivals", "area_km2", "kind"])
    zi, zj = cell_ij(d.lat, d.lon, deg)
    d = d.assign(zi=zi, zj=zj)
    z = d.groupby(["zi", "zj"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), train_vessels=("mmsi", "nunique"),
                                    train_arrivals=("arrival", "sum")).reset_index()
    z = z[z.train_vessels >= min_vessels].sort_values(["train_vessels", "train_arrivals", "zi", "zj"],
                                                      ascending=[False, False, True, True], kind="stable")
    if max_zones:
        z = z.head(max_zones)
    z = z.reset_index(drop=True)
    z["lat0"], z["lat1"] = z.zi * deg, (z.zi + 1) * deg
    z["lon0"], z["lon1"] = z.zj * deg, (z.zj + 1) * deg
    z["area_km2"] = cell_area_km2(z.zi, deg)
    z["zone_id"] = [f"Z{deg:g}_{a:+.3f}_{o:+.3f}" for a, o in zip(z.lat0 + deg / 2, z.lon0 + deg / 2)]
    z["kind"] = kind
    return z[["zone_id", "zi", "zj", "lat", "lon", "lat0", "lat1", "lon0", "lon1", "train_vessels",
              "train_arrivals", "area_km2", "kind"]]
