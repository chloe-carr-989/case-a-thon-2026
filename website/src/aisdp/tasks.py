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
from .config import CACHE_DIR, CountConfig, DestinationConfig, ExperimentConfig, QualityConfig


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
                    cc: CountConfig, log=print):
    t0 = time.perf_counter()
    pz = {"deg": cc.port_zone_deg, "min_vessels": cc.port_min_vessels, "max_zones": cc.port_max_zones,
          "kind": "port/anchorage study zone (inferred from dwells)"}
    try:
        res = Dn.run(con, info, dc, ecfg, log=log, port_zone_args=pz)
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
    return res["metrics"], meta, res["port_zones"]
