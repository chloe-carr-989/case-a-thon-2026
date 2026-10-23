"""Runners for the expansion tasks: each runs one task on one subset, writes its artifacts into the run
directory and returns (metrics frame or None, metadata dict). A task that the data cannot support
returns status `insufficient_data` with the counts that blocked it instead of a pretend result."""
from __future__ import annotations

import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import counts as K
from . import destination as Dn
from . import quality as Q
from .config import (AnomalyConfig, CACHE_DIR, CountConfig, DestinationConfig, ExperimentConfig, LaneConfig,
                     QualityConfig)


def _save_model(model, subset: str, task: str) -> str:
    d = CACHE_DIR / "models" / subset
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task}.pkl"
    p.write_bytes(pickle.dumps(model))
    return str(p)


def run_quality(con, info: dict, rd: Path, subset: str, qc: QualityConfig, log=print):
    t0 = time.perf_counter()
    stats = Q.flag_records(con, info, qc)
    flags = Q.flag_summary_frame(stats, subset)
    flags.to_csv(rd / "quality_flags.csv", index=False)
    vs = Q.vessel_flag_summary(con)
    vs.to_parquet(rd / "quality_vessels.parquet", index=False)
    gaps, gstats = Q.detect_gaps(con, info, qc)
    gaps.head(qc.max_gap_rows_saved).to_parquet(rd / "gaps.parquet", index=False)
    syn = Q.synthetic_gap_test(con, info, qc)
    syn.to_csv(rd / "gap_synthetic_test.csv", index=False)
    smet = Q.synthetic_gap_metrics(syn)
    log(f"quality: {stats['rows_flagged']}/{stats['rows']} records flagged, {gstats['n_gaps']} reporting gaps, "
        f"synthetic test {smet}")
    rows = [{"task": "data_quality", "condition": "observed", "mitigation": "n/a", "predictor": "flag_rules",
             "subgroup": r.flag_code, "window": "all", "seed": np.nan, "n_cases": stats["rows"],
             "n_flagged": r.records, "flag_share": r.share_of_records} for r in flags.itertuples()]
    for kind, m in smet.items():
        rows.append({"task": "reporting_gaps", "condition": f"synthetic_{kind}", "mitigation": "n/a",
                     "predictor": "gap_detector", "subgroup": "all", "window": "all", "seed": qc.synth_seed,
                     "n_cases": m["n_injected"], "n_should_detect": m["n_should_detect"],
                     "realtime_recall": m["realtime_recall"], "false_alarms": m["false_alarm_short_events"],
                     "retrospective_gap_recall": m.get("retrospective_gap_recall"),
                     "retrospective_relabelled_as_delay": m.get("retrospective_relabelled_as_delay")})
    rows.append({"task": "reporting_gaps", "condition": "observed", "mitigation": "n/a", "predictor": "gap_detector",
                 "subgroup": "all", "window": "all", "seed": np.nan, "n_cases": gstats["n_gaps"],
                 "n_vessels_with_gaps": gstats["n_vessels_with_gaps"], "median_gap_min": gstats["median_gap_min"],
                 "p90_gap_min": gstats["p90_gap_min"]})
    meta = {"status": "completed", "flags": {k: v for k, v in stats.items()}, "gaps": {k: v for k, v in gstats.items()},
            "synthetic_gap_test": smet, "seconds": round(time.perf_counter() - t0, 2)}
    return pd.DataFrame(rows), meta


def run_counts(con, info: dict, rd: Path, subset: str, task: str, cc: CountConfig, zones=None, log=print):
    t0 = time.perf_counter()
    try:
        res = K.run(con, info, task, cc, zones=zones, log=log)
    except K.InsufficientCounts as e:
        return None, {"status": "insufficient_data", "reason": str(e), "seconds": round(time.perf_counter() - t0, 2)}
    name = "density" if task == "traffic_density" else "port"
    c = res["cases"]
    c0 = c[c.seed == cc.seeds[0]].drop(columns=["t_un_s", "t_mi_s"])
    for col in ["pred_un", "persistence", "rolling_mean", "hist_mean", "pred_mi"]:
        c0[col] = c0[col].astype("float32")
    c0.to_parquet(rd / f"{name}_cases_seed0.parquet", index=False)
    res["units"].to_csv(rd / f"{name}_units.csv", index=False)
    meta = {k: res[k] for k in ["defer_penalty", "n_units", "n_train_origins", "n_test_origins", "n_train_rows",
                                "train_seconds", "split_t", "test_window_edges_t", "n_dense_units", "seconds"]}
    meta.update(status="completed", corruption_logs=res["corruption_logs"], features=K.FEATURES,
                model_path=_save_model(res["model"], subset, task), config=asdict(cc))
    return res["metrics"], meta


def run_destination(con, info: dict, rd: Path, subset: str, dc: DestinationConfig, ecfg: ExperimentConfig,
                    cc: CountConfig, log=print, lanes=None):
    t0 = time.perf_counter()
    pz = {"deg": cc.port_zone_deg, "min_vessels": cc.port_min_vessels, "max_zones": cc.port_max_zones,
          "kind": "port/anchorage study zone (inferred from dwells)"}
    try:
        res = Dn.run(con, info, dc, ecfg, log=log, port_zone_args=pz, lanes=lanes)
    except Dn.InsufficientDestination as e:
        info_ = e.info
        meta = {"status": "insufficient_data", "reason": str(e), "seconds": round(time.perf_counter() - t0, 2),
                **{k: v for k, v in info_.items() if k not in ("zones", "port_zones")}}
        return None, meta, info_.get("port_zones")
    res["zones"].to_csv(rd / "destination_zones.csv", index=False)
    if res["port_zones"] is not None:
        res["port_zones"].to_csv(rd / "port_zones.csv", index=False)
    c = res["cases"]
    c[c.seed == dc.seeds[0]].to_parquet(rd / "destination_cases_seed0.parquet", index=False)
    res["confusion"].to_csv(rd / "destination_confusion.csv", index=False)
    keys = ["n_vessel_buckets", "n_dwell_events", "n_arrival_dwells", "n_zones", "split_t", "data_t", "status_counts",
            "n_train_labeled", "n_test_labeled", "n_test_scored", "n_train_classes", "n_test_classes", "n_train_candidate_rows",
            "train_seconds", "n_test_vessels", "n_train_vessels", "test_window_edges_t", "seconds"]
    meta = {k: res[k] for k in keys}
    meta.update(status="completed", features=Dn.CAND_FEATURES, config=asdict(dc),
                model_path=_save_model(res["model"], subset, "destination"))
    metrics = res["metrics"]
    if "eta_models" in res:
        em = res["eta_models"]
        meta["eta"] = {"status": "completed", **em["info"], "defer_penalty_h": res["eta_defer_penalty_h"],
                       "model_path": _save_model({k: v for k, v in em.items() if k != "info"}, subset, "eta")}
        metrics = pd.concat([metrics, res["eta_metrics"]], ignore_index=True)
    elif lanes is None:
        meta["eta"] = {"status": "skipped", "reason": "sea lanes unavailable for this subset"}
    return metrics, meta, res["port_zones"]


def vessel_registry(con, srcs: list[Path]) -> pd.DataFrame:
    """Self-reported identity per MMSI (latest non-null value). AIS identity fields are not verified."""
    from .ingest import csv_source
    return con.execute(f"""
        SELECT mmsi, arg_max(vessel_name, base_date_time) FILTER (vessel_name IS NOT NULL) AS name,
               arg_max(imo, base_date_time) FILTER (imo IS NOT NULL) AS imo,
               arg_max(call_sign, base_date_time) FILTER (call_sign IS NOT NULL) AS call_sign,
               mode(vessel_type) AS vessel_type, max(length) AS length, max(draft) AS draft
        FROM {csv_source(srcs)} WHERE mmsi BETWEEN 100000000 AND 999999999 GROUP BY mmsi
    """).df()


def run_lanes(con, info: dict, rd: Path, srcs: list[Path], split_t: float, ecfg: ExperimentConfig,
              lc: LaneConfig, log=print):
    from . import lanes as L
    t0 = time.perf_counter()
    n_vm = L.build_vessel_minutes(con, info)
    try:
        res = L.learn(con, info, split_t, ecfg, lc, log=log)
    except L.InsufficientLanes as e:
        return None, {"status": "insufficient_data", "reason": str(e), "seconds": round(time.perf_counter() - t0, 2)}
    L.save(res, rd)
    vessel_registry(con, srcs).to_parquet(rd / "vessels.parquet", index=False)
    c = res["cells"]
    meta = {"status": "completed", "n_vessel_minutes": int(n_vm), "split_t": split_t,
            "n_train_vessel_minutes": res["n_train_vessel_minutes"], "n_train_vessels": res["n_train_vessels"],
            "lane_cells": {g: int(c[(c.res == "fine") & (c.g == g) & c.lane].shape[0]) for g in c.g.unique()},
            "graph_nodes": int(len(res["nodes"])), "graph_edges": int(len(res["edges"])), "config": asdict(lc),
            "seconds": round(time.perf_counter() - t0, 2)}
    return L.LaneModel.from_result(res, lc), meta


WINDOW_COLS = {"float32": ["lat", "lon", "sog", "cog", "score", "p_off_lane", "p_wrong_way", "p_speed", "p_stop",
                           "off_lane", "wrong_way", "speed", "stop", "age_s", "max_gap_s", "uncovered"],
               "int16": ["n_pts", "n_moving", "n_spike", "n_invalid", "window"]}


def run_anomaly(con, rd: Path, lanes, split_t: float, ecfg: ExperimentConfig, lc: LaneConfig, ac: AnomalyConfig,
                log=print):
    from . import anomaly as A
    t0 = time.perf_counter()
    try:
        res = A.run(con, lanes, split_t, ecfg, lc, ac, log=log)
    except A.InsufficientAnomaly as e:
        return None, {"status": "insufficient_data", "reason": str(e), "seconds": round(time.perf_counter() - t0, 2)}
    W = res["windows"]
    keep = ["condition", "mitigation", "mmsi", "T", "category", "label", "reasons", "is_stale", "is_lowq", "last_t",
            *WINDOW_COLS["float32"], *WINDOW_COLS["int16"]]
    W = W[keep].copy()
    for dt, cols in WINDOW_COLS.items():
        W[cols] = W[cols].fillna(-1 if dt == "int16" else np.nan).astype(dt)
    for c in ["condition", "mitigation", "category", "label"]:
        W[c] = W[c].astype("category")
    W.to_parquet(rd / "anomaly_windows.parquet", index=False)
    res["synthetic"].to_parquet(rd / "anomaly_synthetic.parquet", index=False)
    meta = {"status": "completed", **{k: res[k] for k in ["n_test_windows", "n_test_vessels", "n_calibration_windows",
                                                         "window_edges_t", "calibration", "seconds"]},
            "config": asdict(ac)}
    return res["metrics"], meta
