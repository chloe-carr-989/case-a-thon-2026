"""Shared AIS integrity checks and observation/reporting gap detection.

Produces an interpretable list of flag codes per record (never one opaque score) and a table of
reporting gaps per vessel. Raw records are never modified; derived outputs are written separately.

Gaps are *observation/reporting gaps*: the source has no report from that MMSI for longer than
expected. They are not proven radio failures; reception coverage, the product's own downsampling
and vessels leaving the study area produce the same signature.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import QualityConfig
from .geo import haversine_m
from .ingest import parquet_glob

FLAG_CODES = {
    "mmsi_malformed": "MMSI missing or not 9 digits",
    "mmsi_non_vessel_mid": "MMSI country prefix (MID) outside 201-775: base station, SAR, AtoN or test ID",
    "ts_missing": "timestamp missing or unparseable",
    "coord_missing": "latitude or longitude missing",
    "coord_out_of_range": "latitude outside +/-90 or longitude outside +/-180 (91/181 = 'not available')",
    "null_island": "position exactly 0,0",
    "sog_sentinel": "SOG = 102.3 kn, the AIS 'not available' code",
    "sog_invalid": "SOG negative or above 102.3 kn",
    "sog_implausible": "SOG validly encoded but above the implausible-speed threshold",
    "cog_sentinel": "COG = 360, the AIS 'not available' code",
    "cog_invalid": "COG negative or above 360",
    "heading_sentinel": "heading = 511, the AIS 'not available' code",
    "heading_invalid": "heading outside 0-359 and not 511",
    "exact_duplicate": "byte-identical repeat of another record (same MMSI, time and fields)",
    "ts_conflict": "same MMSI and timestamp as another record but different position/kinematics",
    "high_rate": "more reports from this MMSI in one minute than the threshold",
    "implausible_jump": "implied speed from the previous fix exceeds the threshold over > 1 km",
}
RESPONSE = {  # recommended response per flag; not every flag removes the record
    "mmsi_malformed": "drop", "mmsi_non_vessel_mid": "exclude from vessel counts",
    "ts_missing": "drop", "coord_missing": "drop", "coord_out_of_range": "drop", "null_island": "drop",
    "sog_sentinel": "treat SOG as missing", "sog_invalid": "treat SOG as missing",
    "sog_implausible": "keep; cross-check against track", "cog_sentinel": "treat COG as missing",
    "cog_invalid": "treat COG as missing", "heading_sentinel": "treat heading as missing",
    "heading_invalid": "treat heading as missing", "exact_duplicate": "keep first copy only",
    "ts_conflict": "keep, flag vessel for review", "high_rate": "keep; do not let it inflate counts",
    "implausible_jump": "keep raw; down-weight or fall back to last credible fix",
}


def _src(info: dict) -> str:
    return f"read_parquet('{parquet_glob(info)}', hive_partitioning=true)"


def flag_records(con, info: dict, qc: QualityConfig, table: str = "qflags") -> dict:
    """Create table `table` with one row per raw record and its flag codes. Returns counts per code."""
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH raw AS (
            SELECT mmsi, ts, lat, lon, sog, cog, heading, vessel_type, status,
                   row_number() OVER (PARTITION BY mmsi, ts, lat, lon, sog, cog, heading, status) AS dup_rank,
                   count(*) OVER (PARTITION BY mmsi, ts) AS n_same_ts,
                   count(DISTINCT (lat, lon, sog, cog)) OVER (PARTITION BY mmsi, ts) AS n_distinct_same_ts,
                   count(*) OVER (PARTITION BY mmsi, date_trunc('minute', ts)) AS n_minute
            FROM {_src(info)}
        ), ok AS (
            SELECT *, (dup_rank = 1 AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
                       AND NOT (lat = 0 AND lon = 0)) AS pos_ok
            FROM raw
        ), seq AS (
            SELECT *, lag(CASE WHEN pos_ok THEN lat END IGNORE NULLS) OVER w AS plat,
                      lag(CASE WHEN pos_ok THEN lon END IGNORE NULLS) OVER w AS plon,
                      epoch(ts) - epoch(lag(CASE WHEN pos_ok THEN ts END IGNORE NULLS) OVER w) AS dt
            FROM ok WINDOW w AS (PARTITION BY mmsi ORDER BY ts, dup_rank)
        ), r AS (
            SELECT *, CASE WHEN pos_ok AND plat IS NOT NULL THEN 2 * 6371008.8 * asin(sqrt(least(1,
                         pow(sin(radians(lat - plat) / 2), 2) +
                         cos(radians(plat)) * cos(radians(lat)) * pow(sin(radians(lon - plon) / 2), 2)))) END AS jump_m
            FROM seq
        )
        SELECT r.mmsi, r.ts, r.lat, r.lon, r.sog, r.cog, r.heading,
               list_filter([
                 CASE WHEN r.mmsi IS NULL OR r.mmsi NOT BETWEEN 100000000 AND 999999999 THEN 'mmsi_malformed' END,
                 CASE WHEN r.mmsi BETWEEN 100000000 AND 999999999 AND r.mmsi // 1000000 NOT BETWEEN 201 AND 775
                      THEN 'mmsi_non_vessel_mid' END,
                 CASE WHEN r.ts IS NULL THEN 'ts_missing' END,
                 CASE WHEN r.lat IS NULL OR r.lon IS NULL THEN 'coord_missing' END,
                 CASE WHEN r.lat NOT BETWEEN -90 AND 90 OR r.lon NOT BETWEEN -180 AND 180 THEN 'coord_out_of_range' END,
                 CASE WHEN r.lat = 0 AND r.lon = 0 THEN 'null_island' END,
                 CASE WHEN r.sog BETWEEN 102.25 AND 102.35 THEN 'sog_sentinel' END,
                 CASE WHEN r.sog < 0 OR r.sog > 102.35 THEN 'sog_invalid' END,
                 CASE WHEN r.sog > {qc.sog_implausible_kn} AND r.sog < 102.25 THEN 'sog_implausible' END,
                 CASE WHEN r.cog = 360 THEN 'cog_sentinel' END,
                 CASE WHEN r.cog < 0 OR r.cog > 360 THEN 'cog_invalid' END,
                 CASE WHEN r.heading = 511 THEN 'heading_sentinel' END,
                 CASE WHEN r.heading <> 511 AND (r.heading < 0 OR r.heading >= 360) THEN 'heading_invalid' END,
                 CASE WHEN r.dup_rank > 1 THEN 'exact_duplicate' END,
                 CASE WHEN r.dup_rank = 1 AND r.n_same_ts > 1 AND r.n_distinct_same_ts > 1 THEN 'ts_conflict' END,
                 CASE WHEN r.n_minute > {qc.high_rate_per_min} THEN 'high_rate' END,
                 CASE WHEN r.jump_m > {qc.implied_speed_min_dist_m}
                       AND (r.dt = 0 OR r.jump_m / r.dt / 0.514444 > {qc.implied_speed_max_kn})
                      THEN 'implausible_jump' END
               ], lambda x: x IS NOT NULL) AS flags
        FROM r
    """)
    n = con.execute(f"SELECT count(*), sum((len(flags) > 0)::INT) FROM {table}").fetchall()[0]
    by_code = dict(con.execute(f"""
        SELECT f, count(*) FROM (SELECT unnest(flags) f FROM {table}) GROUP BY 1 ORDER BY 2 DESC
    """).fetchall())
    return {"rows": int(n[0]), "rows_flagged": int(n[1] or 0),
            "by_code": {k: int(by_code.get(k, 0)) for k in FLAG_CODES},
            "seconds": round(time.perf_counter() - t0, 2)}


def flag_summary_frame(stats: dict, subset: str) -> pd.DataFrame:
    rows = [{"subset": subset, "flag_code": k, "description": FLAG_CODES[k], "records": v,
             "share_of_records": v / max(stats["rows"], 1), "recommended_response": RESPONSE[k],
             "origin": "observed (raw source)"} for k, v in stats["by_code"].items()]
    return pd.DataFrame(rows)


def vessel_flag_summary(con, table: str = "qflags") -> pd.DataFrame:
    """Per-vessel counts by flag code (compact; used by the app and for audits)."""
    df = con.execute(f"""
        SELECT mmsi, count(*) AS n_records, sum((len(flags) > 0)::INT) AS n_flagged,
               list_distinct(flatten(list(flags))) AS codes,
               min(ts) AS first_ts, max(ts) AS last_ts
        FROM {table} WHERE mmsi IS NOT NULL GROUP BY mmsi
    """).df()
    df["codes"] = df["codes"].map(lambda c: ";".join(sorted(c)) if c is not None else "")
    return df


# ------------------------------------------------------------------------------------------- gaps
def detect_gaps_sql(source: str, qc: QualityConfig, t_col: str = "t") -> str:
    """SQL over a relation with (mmsi, t, lat, lon) producing one row per observation/reporting gap."""
    return f"""
        WITH x AS (SELECT DISTINCT mmsi, {t_col} AS t, lat, lon FROM {source}),
        y AS (SELECT mmsi, t, lat, lon, lead(t) OVER w AS t_next, lead(lat) OVER w AS lat_next,
                     lead(lon) OVER w AS lon_next
              FROM x WINDOW w AS (PARTITION BY mmsi ORDER BY t)),
        cad AS (SELECT mmsi, median(t_next - t) AS cadence_s, count(*) AS n_reports
                FROM y WHERE t_next IS NOT NULL GROUP BY mmsi)
        SELECT y.mmsi, y.t AS gap_start_t, y.t_next AS gap_end_t, y.t_next - y.t AS gap_s,
               cad.cadence_s, greatest({qc.gap_min_s}, {qc.gap_cadence_mult} * cad.cadence_s) AS threshold_s,
               greatest(0, round((y.t_next - y.t) / greatest(cad.cadence_s, 1)) - 1) AS expected_missing_reports,
               y.lat AS start_lat, y.lon AS start_lon, y.lat_next AS end_lat, y.lon_next AS end_lon
        FROM y JOIN cad USING (mmsi)
        WHERE y.t_next - y.t > greatest({qc.gap_min_s}, {qc.gap_cadence_mult} * cad.cadence_s)
    """


def detect_gaps(con, info: dict, qc: QualityConfig) -> tuple[pd.DataFrame, dict]:
    """Reporting gaps on real records (valid coordinates, vessel MMSIs). Adds coverage-exit heuristics."""
    t0 = time.perf_counter()
    src = f"""(SELECT mmsi, epoch(ts)::DOUBLE AS t, lat, lon FROM {_src(info)}
               WHERE mmsi BETWEEN 100000000 AND 999999999 AND lat BETWEEN -90 AND 90
                     AND lon BETWEEN -180 AND 180 AND NOT (lat = 0 AND lon = 0))"""
    con.execute(f"CREATE OR REPLACE TABLE gaps AS {detect_gaps_sql(src, qc)}")
    d = qc.sparse_cell_deg
    con.execute(f"""
        CREATE OR REPLACE TABLE cov AS
        SELECT floor(lat / {d})::INT AS ci, floor(lon / {d})::INT AS cj, count(DISTINCT mmsi) AS v
        FROM {src} GROUP BY 1, 2
    """)
    g = con.execute(f"""
        SELECT g.*, coalesce(c1.v, 0) AS start_cell_vessels, coalesce(c2.v, 0) AS end_cell_vessels
        FROM gaps g
        LEFT JOIN cov c1 ON c1.ci = floor(g.start_lat / {d})::INT AND c1.cj = floor(g.start_lon / {d})::INT
        LEFT JOIN cov c2 ON c2.ci = floor(g.end_lat / {d})::INT AND c2.cj = floor(g.end_lon / {d})::INT
        ORDER BY g.gap_s DESC
    """).df()
    t_first, t_last = con.execute(f"SELECT min(t), max(t) FROM {src}").fetchall()[0]
    if len(g):
        g["moved_km"] = haversine_m(g.start_lat, g.start_lon, g.end_lat, g.end_lon) / 1000
        g["implied_speed_kn"] = g.moved_km * 1000 / g.gap_s / 0.514444
        sparse = (g.start_cell_vessels <= qc.sparse_cell_max_vessels) | \
            (g.end_cell_vessels <= qc.sparse_cell_max_vessels)
        g["possible_coverage_exit"] = sparse
        g["gap_class"] = np.where(sparse, "possible_coverage_exit",
                                  np.where(g.moved_km < 0.5, "stationary_reporting_gap", "underway_reporting_gap"))
    stats = {"n_gaps": int(len(g)), "n_vessels_with_gaps": int(g.mmsi.nunique()) if len(g) else 0,
             "median_gap_min": float(g.gap_s.median() / 60) if len(g) else None,
             "p90_gap_min": float(g.gap_s.quantile(0.9) / 60) if len(g) else None,
             "by_class": g.gap_class.value_counts().to_dict() if len(g) else {},
             "t_first": t_first, "t_last": t_last, "seconds": round(time.perf_counter() - t0, 2),
             "config": asdict(qc)}
    return g, stats


def gap_open_at(last_report_t, decision_t, cadence_s, qc: QualityConfig):
    """Is a reporting gap open at the decision time? (age beyond the vessel's expected cadence)."""
    age = np.asarray(decision_t, float) - np.asarray(last_report_t, float)
    thr = np.maximum(qc.gap_min_s, qc.gap_cadence_mult * np.nan_to_num(np.asarray(cadence_s, float), nan=60.0))
    return age > thr


def freshness(age_s, horizon_s: int, max_age_s: int = 1200) -> np.ndarray:
    """Report age at decision time -> fresh / stale / too_old / future_timestamp."""
    a = np.asarray(age_s, float)
    return np.select([~np.isfinite(a), a < 0, a <= 120, a <= max_age_s],
                     ["no_report", "future_timestamp", "fresh", "stale"], "too_old")


# ----------------------------------------------------------------- synthetic withheld-record test
def synthetic_gap_test(con, info: dict, qc: QualityConfig) -> pd.DataFrame:
    """Withhold or delay known runs of real records and check that the detector finds them.

    For each sampled clean track (no detected gap, cadence <= 2 min), one run of records is either
    withheld (never delivered, label `missing`) or delayed (delivered late, label `delayed`) with a
    duration drawn from 5-50 min. At the decision time just after the run both look the same: an open
    gap. Retrospectively the delayed records arrive and the gap closes; the withheld ones never do.
    Returns one row per injected event with what the detector concluded.
    """
    rng = np.random.default_rng(qc.synth_seed)
    tracks = con.execute(f"""
        WITH c AS (SELECT mmsi, count(*) n, max(t) - min(t) span FROM
                   (SELECT DISTINCT mmsi, epoch(ts)::DOUBLE t FROM {_src(info)}
                    WHERE mmsi BETWEEN 100000000 AND 999999999) GROUP BY mmsi)
        SELECT mmsi FROM c WHERE span >= 5400 AND n >= 45
          AND mmsi NOT IN (SELECT mmsi FROM gaps)
        ORDER BY ((mmsi % 1000003) * 7919) % 1000003 LIMIT {qc.synth_vessels}
    """).df().mmsi.to_numpy()
    if len(tracks) == 0:
        return pd.DataFrame()
    con.register("synth_v", pd.DataFrame({"mmsi": tracks}))
    r = con.execute(f"""SELECT DISTINCT mmsi, epoch(ts)::DOUBLE t, lat, lon FROM {_src(info)}
                        SEMI JOIN synth_v USING (mmsi) ORDER BY mmsi, t""").df()
    con.unregister("synth_v")
    rows, kept = [], []
    for k, (mmsi, g) in enumerate(r.groupby("mmsi", sort=False)):
        t = g.t.to_numpy()
        dur = 300 + rng.random() * 2700
        start = t[0] + 900 + rng.random() * max(1.0, t[-1] - t[0] - 900 - dur - 900)
        hide = (t > start) & (t < start + dur)
        kind = "missing" if k % 2 == 0 else "delayed"
        avail = t.copy()
        if kind == "delayed":
            avail[hide] = start + dur + 1200            # delivered 20 min after the run ends
        else:
            avail[hide] = np.inf                         # never delivered
        decision = start + dur                           # analyst looks right after the run
        before = t[(avail <= decision) & (t <= decision)]
        last = before.max() if len(before) else np.nan
        cad = np.median(np.diff(t))
        open_now = bool(gap_open_at(last, decision, cad, qc))
        delivered = t if kind == "delayed" else t[~hide]
        after_t = delivered[delivered > start]
        retro_gap = (after_t.min() - delivered[delivered <= start].max()) if len(after_t) else np.nan
        retro_detected = bool(np.isfinite(retro_gap) and retro_gap > max(qc.gap_min_s, qc.gap_cadence_mult * cad))
        true_gap = (t[~hide & (t > start)].min() - t[~hide & (t <= start)].max()) if hide.any() else 0.0
        should_detect = true_gap > max(qc.gap_min_s, qc.gap_cadence_mult * cad)
        rows.append({"mmsi": int(mmsi), "kind": kind, "injected_s": float(true_gap), "n_records_affected": int(hide.sum()),
                     "cadence_s": float(cad), "should_detect": bool(should_detect),
                     "open_gap_at_decision": open_now, "retrospective_gap": retro_detected,
                     "retro_label": ("observation_gap" if retro_detected else
                                     ("late_backfilled_delay" if kind == "delayed" and open_now else "no_gap")),
                     "origin": "SYNTHETIC withheld/delayed real records"})
    return pd.DataFrame(rows)


def synthetic_gap_metrics(s: pd.DataFrame) -> dict:
    """Real-time recall (open gap seen at the decision) and what the retrospective view concludes."""
    if s.empty:
        return {}
    out = {}
    for kind, g in s.groupby("kind"):
        pos = g[g.should_detect]
        out[kind] = {"n_injected": int(len(g)), "n_should_detect": int(len(pos)),
                     "realtime_recall": float(pos.open_gap_at_decision.mean()) if len(pos) else None,
                     "false_alarm_short_events": int((~g.should_detect & g.open_gap_at_decision).sum())}
        if kind == "missing":
            out[kind]["retrospective_gap_recall"] = float(pos.retrospective_gap.mean()) if len(pos) else None
        else:   # a delayed run is backfilled later: correct outcome is NO retrospective gap
            out[kind]["retrospective_relabelled_as_delay"] = float((~pos.retrospective_gap).mean()) if len(pos) else None
    return out
