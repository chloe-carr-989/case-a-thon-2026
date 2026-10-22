"""AIS validation, decision/target pairing, as-of state lookup and feature construction.

All joins are per-vessel ASOF joins in DuckDB (no all-pairs joins). A state looked up for
query time q is always the latest report with observation time <= q, so no future record
can enter the inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .geo import enu_displacement, sog_cog_to_ve_vn
from .ingest import parquet_glob

FEATURES = ["sog", "cog_sin", "cog_cos", "ve_rep", "vn_rep", "hdg_sin", "hdg_cos",
            "ve_obs", "vn_obs", "dt_prev", "prev_sog", "dcog", "vessel_type", "hour"]


def vessel_key_sql(col: str = "mmsi") -> str:
    return f"((({col} % 1000003) * 7919) % 1000003)"


def vessel_key(mmsi) -> np.ndarray:
    m = np.asarray(mmsi, dtype=np.int64)
    return ((m % 1000003) * 7919) % 1000003


def is_test_vessel(mmsi, cfg: ExperimentConfig) -> np.ndarray:
    return (vessel_key(mmsi) % cfg.test_vessel_mod) < cfg.test_vessel_cut


def load_reports(con, info: dict, cfg: ExperimentConfig) -> dict:
    """Clean AIS sentinels and build table `r` for a deterministic vessel sample."""
    g = parquet_glob(info)
    con.execute(f"""
        CREATE OR REPLACE TABLE vessels AS
        SELECT mmsi FROM (SELECT DISTINCT mmsi FROM read_parquet('{g}', hive_partitioning=true)
                          WHERE mmsi BETWEEN 100000000 AND 999999999)
        ORDER BY {vessel_key_sql()}, mmsi LIMIT {int(cfg.sample_vessels)}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE r AS
        SELECT p.mmsi, epoch(p.ts)::DOUBLE AS t, p.lat, p.lon,
               CASE WHEN p.sog >= 102.2 OR p.sog < 0 THEN NULL ELSE p.sog END AS sog,
               CASE WHEN p.cog >= 360 OR p.cog < 0 THEN NULL ELSE p.cog END AS cog,
               CASE WHEN p.heading >= 360 OR p.heading < 0 THEN NULL ELSE p.heading END AS heading,
               p.vessel_type
        FROM read_parquet('{g}', hive_partitioning=true) p SEMI JOIN vessels v ON p.mmsi = v.mmsi
        WHERE p.lat BETWEEN -90 AND 90 AND p.lon BETWEEN -180 AND 180 AND NOT (p.lat = 0 AND p.lon = 0)
        QUALIFY row_number() OVER (PARTITION BY p.mmsi, p.ts ORDER BY p.lat, p.lon) = 1
        ORDER BY p.mmsi, t
    """)
    n, nv, t0, t1 = con.execute("SELECT count(*), count(DISTINCT mmsi), min(t), max(t) FROM r").fetchall()[0]
    return {"report_rows": n, "sampled_vessels": nv, "t0": t0, "t1": t1}


def build_cases(con, cfg: ExperimentConfig) -> pd.DataFrame:
    """Decision anchors with an observed AIS position ~horizon later (the noisy evaluation proxy)."""
    H, tol = cfg.horizon_s, cfg.tol_s
    df = con.execute(f"""
        WITH anchors AS (
            SELECT mmsi, min(t) AS decision_t FROM r GROUP BY mmsi, floor(t / {cfg.anchor_spacing_s})
        ), firsts AS (SELECT mmsi, min(t) AS first_t FROM r GROUP BY mmsi),
        a AS (SELECT a.*, f.first_t, a.decision_t + {H - tol} AS tq FROM anchors a JOIN firsts f USING (mmsi))
        SELECT a.mmsi, a.decision_t, a.first_t, tr.t AS target_t, tr.lat AS target_lat, tr.lon AS target_lon
        FROM a ASOF JOIN r tr ON a.mmsi = tr.mmsi AND tr.t >= a.tq
        WHERE tr.t <= a.decision_t + {H + tol}
        ORDER BY a.decision_t, a.mmsi
    """).df()
    if df.empty:
        return df
    t0, t1 = con.execute("SELECT min(t), max(t) FROM r").fetchall()[0]
    split_t = t0 + cfg.train_time_frac * (t1 - t0)
    test_v = is_test_vessel(df["mmsi"].to_numpy(), cfg)
    train = (~test_v) & (df["target_t"].to_numpy() < split_t)
    test = test_v & (df["decision_t"].to_numpy() >= split_t + H) & \
        ((df["decision_t"] - df["first_t"]).to_numpy() >= cfg.min_history_s)
    df["split"] = np.where(train, "train", np.where(test, "test", "unused"))
    df = df[df["split"] != "unused"].reset_index(drop=True)
    df["case_id"] = np.arange(len(df))
    df.attrs["split_t"] = split_t
    te = df["split"] == "test"
    df["window"] = -1
    if te.any():
        lo, hi = df.loc[te, "decision_t"].min(), df.loc[te, "decision_t"].max() + 1e-6
        edges = np.linspace(lo, hi, cfg.n_windows + 1)
        df.loc[te, "window"] = np.clip(np.searchsorted(edges, df.loc[te, "decision_t"], side="right") - 1,
                                       0, cfg.n_windows - 1)
        df.attrs["window_edges"] = edges.tolist()
    return df


def build_states(con, q: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """For each (case_id, mmsi, query_t), the latest report with t <= query_t plus its previous fix."""
    con.register("q_df", q[["case_id", "mmsi", "query_t"]])
    df = con.execute(f"""
        WITH s AS (
            SELECT q.case_id, q.mmsi, q.query_t, r.t AS state_t, r.lat, r.lon, r.sog, r.cog, r.heading,
                   r.vessel_type, r.t - {cfg.prev_gap_min_s} AS pq
            FROM q_df q ASOF JOIN r ON q.mmsi = r.mmsi AND r.t <= q.query_t
        )
        SELECT s.* EXCLUDE (pq), p.t AS prev_t, p.lat AS prev_lat, p.lon AS prev_lon,
               p.sog AS prev_sog, p.cog AS prev_cog
        FROM s ASOF LEFT JOIN r p ON s.mmsi = p.mmsi AND p.t <= s.pq
        ORDER BY s.case_id
    """).df()
    con.unregister("q_df")
    far = (df["state_t"] - df["prev_t"]) > cfg.prev_gap_max_s
    df.loc[far, ["prev_t", "prev_lat", "prev_lon", "prev_sog", "prev_cog"]] = np.nan
    return df


def make_features(s: pd.DataFrame, decision_t: np.ndarray) -> pd.DataFrame:
    """Model features from a (possibly stale/corrupted) delivered state. Never uses targets."""
    sog = s["sog"].to_numpy(float)
    cog = s["cog"].to_numpy(float)
    hdg = s["heading"].to_numpy(float)
    ve_rep, vn_rep = sog_cog_to_ve_vn(sog, cog)
    dt_prev = (s["state_t"] - s["prev_t"]).to_numpy(float)
    de, dn = enu_displacement(s["prev_lat"].to_numpy(float), s["prev_lon"].to_numpy(float),
                              s["lat"].to_numpy(float), s["lon"].to_numpy(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        ve_obs, vn_obs = de / dt_prev, dn / dt_prev
    dcog = (cog - s["prev_cog"].to_numpy(float) + 180) % 360 - 180
    return pd.DataFrame({
        "sog": sog, "cog_sin": np.sin(np.radians(cog)), "cog_cos": np.cos(np.radians(cog)),
        "ve_rep": ve_rep, "vn_rep": vn_rep,
        "hdg_sin": np.sin(np.radians(hdg)), "hdg_cos": np.cos(np.radians(hdg)),
        "ve_obs": ve_obs, "vn_obs": vn_obs, "dt_prev": dt_prev,
        "prev_sog": s["prev_sog"].to_numpy(float), "dcog": dcog,
        "vessel_type": s["vessel_type"].to_numpy(float),
        "hour": ((np.asarray(decision_t, float) % 86400) / 3600.0),
    })[FEATURES]
