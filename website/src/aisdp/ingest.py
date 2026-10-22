"""Chunk-free, streaming ingest: raw AIS CSV (or .csv.zst) -> hour-partitioned Parquet via DuckDB.

DuckDB streams the CSV; the full file is never materialised in Python memory. Output is
cached by source fingerprint + region + schema version, so re-runs skip ingest and new
files only add new partitions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb

from .config import CACHE_DIR, REGIONS, SCHEMA_VERSION

CSV_TYPES = {
    "mmsi": "BIGINT", "base_date_time": "TIMESTAMP", "longitude": "DOUBLE", "latitude": "DOUBLE",
    "sog": "DOUBLE", "cog": "DOUBLE", "heading": "DOUBLE", "vessel_name": "VARCHAR", "imo": "VARCHAR",
    "call_sign": "VARCHAR", "vessel_type": "INTEGER", "status": "INTEGER", "length": "DOUBLE",
    "width": "DOUBLE", "draft": "DOUBLE", "cargo": "INTEGER", "transceiver": "VARCHAR",
}


def connect(memory_limit: str = "4GB", threads: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    return con


def fingerprint(path: Path, full_hash_limit: int = 1_000_000_000) -> dict:
    """sha256 of the whole file when <= full_hash_limit bytes, else of first+last 64 MiB."""
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if st.st_size <= full_hash_limit:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
            mode = "sha256_full"
        else:
            h.update(f.read(1 << 26))
            f.seek(max(0, st.st_size - (1 << 26)))
            h.update(f.read())
            mode = "sha256_head_tail_64MiB"
    return {"path": str(path), "bytes": st.st_size, "mtime": st.st_mtime, "hash": h.hexdigest(), "hash_mode": mode}


def _fp_cache(path: Path) -> dict:
    cache = CACHE_DIR / "fingerprints.json"
    db = json.loads(cache.read_text()) if cache.exists() else {}
    st = path.stat()
    key = f"{path}|{st.st_size}|{st.st_mtime}"
    if key not in db:
        db[key] = fingerprint(path)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(db, indent=1))
    return db[key]


def region_sql(region: str) -> str:
    bbox = REGIONS[region]
    if bbox is None:
        return "TRUE"
    x0, y0, x1, y1 = bbox
    return f"longitude BETWEEN {x0} AND {x1} AND latitude BETWEEN {y0} AND {y1}"


def csv_source(paths: list[Path]) -> str:
    comp = "zstd" if str(paths[0]).endswith(".zst") else "auto"
    types = ", ".join(f"'{k}': '{v}'" for k, v in CSV_TYPES.items())
    files = ", ".join(f"'{p.as_posix()}'" for p in paths)
    return f"read_csv([{files}], header=true, compression='{comp}', types={{{types}}})"


def ingest(con, src: Path | list[Path], name: str, region: str = "all", start: str | None = None,
           end: str | None = None, max_rows: int | None = None) -> dict:
    """Write Parquet partitions for one source file (or several daily files). Returns ingest info (cached)."""
    paths = list(src) if isinstance(src, (list, tuple)) else [src]
    fps = [_fp_cache(p) for p in paths]
    fp = fps[0] if len(fps) == 1 else {"path": [f["path"] for f in fps], "bytes": sum(f["bytes"] for f in fps),
                                       "hash": [f["hash"] for f in fps], "files": fps}
    key = hashlib.sha256(json.dumps([fp["hash"], region, start, end, max_rows, SCHEMA_VERSION]).encode()).hexdigest()[:12]
    out = CACHE_DIR / "parquet" / f"{name}_{region}_{key}"
    info_path = out / "_ingest.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["cache_hit"] = True
        return info
    t0 = time.perf_counter()
    where = [region_sql(region)]
    if start:
        where.append(f"base_date_time >= TIMESTAMP '{start}'")
    if end:
        where.append(f"base_date_time < TIMESTAMP '{end}'")
    limit = f"LIMIT {int(max_rows)}" if max_rows else ""
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
          SELECT mmsi, base_date_time AS ts, CAST(base_date_time AS DATE) AS day,
                 hour(base_date_time) AS hr, longitude AS lon, latitude AS lat,
                 sog, cog, heading, vessel_type, status, length, transceiver
          FROM {csv_source(paths)}
          WHERE {' AND '.join(where)}
          {limit}
        ) TO '{out.as_posix()}' (FORMAT parquet, PARTITION_BY (day, hr), OVERWRITE_OR_IGNORE true)
    """)
    info = {"source": fp, "parquet_dir": str(out), "region": region, "start": start, "end": end,
            "max_rows": max_rows, "ingest_seconds": round(time.perf_counter() - t0, 2), "cache_hit": False}
    info_path.write_text(json.dumps(info, indent=1))
    return info


def parquet_glob(info: dict) -> str:
    return (Path(info["parquet_dir"]) / "**" / "*.parquet").as_posix()


def profile(con, info: dict) -> dict:
    """Dataset characteristics of the ingested (raw-valued) partitions. Observed, not injected, defects."""
    g = parquet_glob(info)
    row = con.execute(f"""
        SELECT count(*) n, count(DISTINCT mmsi) vessels, min(ts) t0, max(ts) t1,
               min(lat), max(lat), min(lon), max(lon),
               avg((sog IS NULL)::INT), avg((cog IS NULL)::INT), avg((heading IS NULL)::INT),
               avg((vessel_type IS NULL)::INT),
               sum((sog >= 102.2)::INT), sum((cog >= 360)::INT), sum((heading >= 511)::INT),
               sum((lat > 90 OR lat < -90 OR lon > 180 OR lon < -180)::INT),
               sum((lat = 0 AND lon = 0)::INT),
               sum((sog > 40 AND sog < 102.2)::INT)
        FROM read_parquet('{g}', hive_partitioning=true)
    """).fetchall()[0]
    dups = con.execute(f"""
        SELECT count(*) - count(DISTINCT (mmsi, ts)) FROM read_parquet('{g}', hive_partitioning=true)
    """).fetchall()[0][0]
    tx = con.execute(f"""
        SELECT transceiver, count(*) FROM read_parquet('{g}', hive_partitioning=true) GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    keys = ["rows", "vessels", "time_min", "time_max", "lat_min", "lat_max", "lon_min", "lon_max",
            "null_rate_sog", "null_rate_cog", "null_rate_heading", "null_rate_vessel_type",
            "sentinel_sog_102_3", "sentinel_cog_ge_360", "sentinel_heading_511",
            "invalid_lat_lon", "null_island_0_0", "sog_over_40kn_not_sentinel"]
    out = {k: (str(v) if k.startswith("time") else v) for k, v in zip(keys, row)}
    out["duplicate_mmsi_timestamp_rows"] = dups
    out["transceiver_class_counts"] = {str(k): v for k, v in tx}
    return out
