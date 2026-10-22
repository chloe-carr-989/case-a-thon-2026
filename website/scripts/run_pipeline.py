"""Reproducible pipeline + scale benchmark.

Examples (PowerShell, from the website/ folder):
    .venv\\Scripts\\python scripts\\run_pipeline.py                       # subsets 4 -> 3 -> 2 -> 1
    .venv\\Scripts\\python scripts\\run_pipeline.py --subsets subset2 --seeds 0
    .venv\\Scripts\\python scripts\\run_pipeline.py --input ..\\data\\dataset3\\ais-2026-01-01.csv.zst --subsets custom
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aisdp import corruption as C  # noqa: E402
from aisdp.config import CODE_VERSION, REPO_ROOT, RESULTS_DIR, SUBSETS, ExperimentConfig  # noqa: E402
from aisdp.experiment import InsufficientData, run  # noqa: E402
from aisdp.ingest import connect, ingest, profile  # noqa: E402


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def pick_examples(cases: pd.DataFrame) -> pd.DataFrame:
    c0 = cases[cases["seed"] == cases["seed"].min()]
    nom = c0[c0["condition"] == "nominal"].set_index("case_id")
    picks = []

    def take(df, why, sort_col, asc=False):
        if len(df):
            r = df.sort_values(sort_col, ascending=asc).iloc[0].copy()
            r["why_selected"] = why
            picks.append(r)

    cb = c0[c0["condition"] == "combined"]
    take(cb[cb["is_stale"] & cb["is_lowq"] & ~cb["answered"]],
         "Combined stale+corrupted input: large unmitigated error, countermeasure defers to reviewer", "err_un_km")
    lq = c0[c0["condition"] == "low_quality"]
    take(lq[(lq["action"] == "fallback_last_credible") & lq["defects"].str.contains("position_outlier")],
         "Position outlier from synthetic source: fallback to last credible fix", "err_un_km")
    st = c0[c0["condition"] == "stale"]
    take(st[(st["action"] == "age_compensated") & (st["err_mi_km"] < st["err_un_km"])],
         "Stale report: age compensation using preserved timestamp helps", "err_un_km")
    take(st[(st["action"] == "age_compensated") & (st["err_mi_km"] > st["err_un_km"] + 1)],
         "FAILURE: age compensation over a long gap extrapolates a turn/stop wrongly", "err_mi_km")
    n0 = c0[c0["condition"] == "nominal"]
    take(n0[(n0["nominal_sog"] >= 20) & ((n0["flags"].str.len() > 0) | ~n0["answered"]) & (n0["err_un_km"] < 1)],
         "TRADEOFF: unmodified fast vessel with a good (<1 km) unmitigated estimate flagged by countermeasure",
         "nominal_sog")
    take(n0[(n0["flags"].str.len() > 0) & (n0["err_un_km"] > 50)],
         "OBSERVED source defect (no injection): countermeasure flags a real AIS glitch", "err_un_km")
    take(n0[~n0["answered"]], "TRADEOFF: unmodified input deferred by countermeasure", "err_un_km", asc=True)
    take(lq[lq["answered"] & (lq["err_mi_km"] > lq["err_un_km"] + 0.2)],
         "FAILURE: countermeasure answer worse than unmitigated", "err_mi_km")
    if not picks:
        return pd.DataFrame()
    ex = pd.DataFrame(picks).reset_index(drop=True)
    ex["raw_lat"] = ex["case_id"].map(nom["in_lat"])
    ex["raw_lon"] = ex["case_id"].map(nom["in_lon"])
    ex["raw_sog"] = ex["case_id"].map(nom["in_sog"])
    ex["raw_cog"] = ex["case_id"].map(nom["in_cog"])
    ex["raw_obs_time_utc"] = pd.to_datetime(ex["case_id"].map(nom["state_t"]), unit="s").astype(str)
    ex["delivered_obs_time_utc"] = pd.to_datetime(ex["state_t"], unit="s").astype(str)
    ex["decision_time_utc"] = pd.to_datetime(ex["decision_t"], unit="s").astype(str)
    ex["target_time_utc"] = pd.to_datetime(ex["target_t"], unit="s").astype(str)
    return ex


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", default="subset4,subset3,subset2,subset1",
                    help="comma list from " + ",".join(SUBSETS) + " or 'custom' with --input")
    ap.add_argument("--input", help="CSV or .csv.zst path (used with --subsets custom)")
    ap.add_argument("--region", default="all")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--sample-vessels", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget-s", type=float, default=300, help="stop larger subsets after one exceeds this")
    ap.add_argument("--output", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    out = Path(args.output)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(sample_vessels=args.sample_vessels, region=args.region,
                           seeds=tuple(int(s) for s in args.seeds.split(",")))
    con = connect()
    bench_path = out / "benchmarks.csv"
    bench = pd.read_csv(bench_path).to_dict("records") if bench_path.exists() else []
    stop_reason = None
    completed = []
    for name in args.subsets.split(","):
        src = Path(args.input) if name == "custom" else SUBSETS[name]
        row = {"subset": name, "file": rel(src), "bytes": src.stat().st_size,
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "region": args.region, "sample_vessels": args.sample_vessels, "seeds": args.seeds}
        if stop_reason:
            row.update(status="skipped", reason=stop_reason)
            bench.append(row)
            print(f"[{name}] skipped: {stop_reason}")
            continue
        t0 = time.perf_counter()
        try:
            info = ingest(con, src, name, args.region, args.start, args.end, args.max_rows)
            prof = profile(con, info)
            row.update(ingest_s=info["ingest_seconds"], ingest_cache_hit=info["cache_hit"],
                       rows=prof["rows"], vessels=prof["vessels"], time_min=prof["time_min"],
                       time_max=prof["time_max"])
            print(f"[{name}] ingested {prof['rows']} rows, {prof['vessels']} vessels "
                  f"({prof['time_min']} .. {prof['time_max']})")
            res = run(con, info, cfg, log=lambda m: print(f"[{name}] {m}"))
        except InsufficientData as e:
            row.update(status="insufficient_data", reason=str(e), total_s=round(time.perf_counter() - t0, 2))
            bench.append(row)
            print(f"[{name}] {e}")
            continue
        total = time.perf_counter() - t0
        rd = out / "runs" / name
        rd.mkdir(parents=True, exist_ok=True)
        m = res["metrics"].assign(subset=name)
        m.to_csv(rd / "metrics.csv", index=False)
        cases = res["cases"]
        c0 = cases[cases["seed"] == cfg.seeds[0]]
        c0.to_parquet(rd / "cases_seed0.parquet", index=False)
        ex = pick_examples(cases)
        ex.to_csv(rd / "example_cases.csv", index=False)
        meta = {k: res[k] for k in ["model_info", "reports", "defer_penalty_km", "n_test_cases",
                                    "n_test_vessels", "n_train_cases", "split_t", "window_edges",
                                    "volumes", "seconds"]}
        meta.update(subset=name, profile=prof, ingest=info, config=cfg.to_dict())
        (rd / "run.json").write_text(json.dumps(meta, indent=1, default=str))
        row.update(status="completed", total_s=round(total, 2), experiment_s=res["seconds"],
                   test_cases=res["n_test_cases"], train_cases=res["n_train_cases"],
                   test_vessels=res["n_test_vessels"], reason="")
        bench.append(row)
        completed.append((name, rd, meta, prof, info))
        print(f"[{name}] completed in {total:.1f}s: {res['n_test_cases']} test cases")
        if total > args.budget_s:
            stop_reason = f"previous subset {name} took {total:.0f}s > budget {args.budget_s:.0f}s"

    pd.DataFrame(bench).to_csv(bench_path, index=False)
    if completed:
        write_outputs(out, completed[-1], cfg, args)


def write_outputs(out: Path, last, cfg: ExperimentConfig, args):
    name, rd, meta, prof, info = last
    import shutil
    for f in ["metrics.csv", "example_cases.csv"]:
        shutil.copy(rd / f, out / f)
    shutil.copy(rd / "cases_seed0.parquet", out / "cases_default.parquet")
    pkgs = {p: version(p) for p in ["duckdb", "pandas", "numpy", "scikit-learn", "pyarrow", "streamlit", "plotly"]}
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "default_demo_subset": name,
        "dataset": {
            "name": "MarineCadastre / NOAA AIS vessel traffic, broadcast-point CSV, 2026-01-01",
            "source_hub_url": "https://hub.marinecadastre.gov/pages/vesseltraffic (per case prompt; not re-verified offline)",
            "product": "daily CSV (compressed .csv.zst present locally); not the GeoParquet product",
            "retrieval_date": "unknown (local file mtime of .zst: 2026-10-18; organizer-provided subsets)",
            "local_files": {k: {"path": rel(v), "bytes": v.stat().st_size} for k, v in SUBSETS.items() if v.exists()},
            "files_used": [rel(info["source"]["path"])],
            "fingerprint": info["source"],
            "subset_rules": "Organizer subsets are byte-identical head-prefixes of ais-2026-01-01.csv "
                            "(verified with cmp); never combined. Region filter: " + args.region +
                            f"; deterministic vessel sample of {cfg.sample_vessels} MMSIs by hash key.",
            "profile_observed": prof,
            "units": {"lat/lon": "WGS84 degrees", "sog": "knots", "cog/heading": "degrees true",
                      "base_date_time": "UTC", "errors": "km (haversine, spherical earth R=6371.0088 km)"},
            "missing_value_policy": "SOG>=102.2 (102.3 sentinel) -> null; COG>=360 -> null; heading>=360 (511) -> null; "
                                    "lat/lon out of range or (0,0) dropped; duplicate (mmsi,timestamp) -> first kept. "
                                    "These are OBSERVED source problems, handled before any injected defects.",
            "rights_statement": "NOAA/BOEM MarineCadastre data are U.S. Government public data; check the product "
                                "metadata/terms before redistribution. Raw data are not redistributed in this repo "
                                "(data/ is git-ignored).",
        },
        "task": {"horizon_s": cfg.horizon_s, "target_tolerance_s": cfg.tol_s,
                 "evaluation_proxy": "first later AIS report of the same MMSI within horizon +/- tolerance; "
                                     "a noisy proxy, not verified truth",
                 "decision_spacing_s": cfg.anchor_spacing_s},
        "split": {"rule": "vessel-disjoint (hash(mmsi)%10<3 -> test) AND chronological: train targets end before "
                          "split_t; test decisions start at split_t + horizon and need >=45 min of vessel history",
                  "split_t_utc": str(pd.to_datetime(meta["split_t"], unit="s")),
                  "window_edges_utc": [str(pd.to_datetime(e, unit="s")) for e in meta["window_edges"]],
                  "holdout": "window 4 (latest period) plus the 'holdout_unseen_bias' defect type not used when "
                             "designing the countermeasure",
                  "n_train_cases": meta["n_train_cases"], "n_test_cases": meta["n_test_cases"],
                  "n_test_vessels": meta["n_test_vessels"]},
        "model": {"type": "2x HistGradientBoostingRegressor (absolute_error) predicting mean east/north velocity",
                  "retraining": "trained once on nominal data; NOT retrained during the degradation progression",
                  **meta["model_info"]},
        "seeds": list(cfg.seeds),
        "defect_configuration": {
            "failure_stage": "inference-time AIS feed degradation",
            "progression": "SIMULATED degradation progression over 5 real chronological test windows",
            "stale_schedule": list(cfg.stale_schedule),
            "stale_delay_s": [cfg.stale_delay_min_s, cfg.stale_delay_max_s],
            "lowq_schedule (share of latest records from synthetic source)": list(cfg.lowq_schedule),
            "synthetic_source_label": C.SYN_SOURCE,
            "lowq_defects": "jitter N(0,200 m) on state and previous fix; 25% gross outlier 3-15 km; SOG scaled "
                            "x0.3-2.5 (25%), missing (15%), raw 102.3 sentinel (10%); COG random (30%), missing (15%), "
                            "else +N(0,5 deg); heading dropped",
            "holdout_unseen_bias": "constant 1.5 km offset at bearing 60 deg on the delivered fix",
            "severity": cfg.severity,
            "synthetic_msgs_simulated": "reliable_msgs * share/(1-share); a simulated volume, not measured",
        },
        "modification_log": "Raw CSVs unchanged. Ingest writes cache/parquet (raw values). Cleaning and all injected "
                            "defects occur in memory; injected defects are labelled per case (columns source, defects, "
                            "is_stale, is_lowq) and kept separate from observed source problems.",
        "countermeasure": {"age_comp_s": cfg.age_comp_s, "max_age_defer_s": cfg.max_age_defer_s,
                           "max_valid_sog_kn": cfg.max_valid_sog_kn, "defer_quality": cfg.defer_quality,
                           "defer_penalty_km": meta["defer_penalty_km"],
                           "defer_penalty_rule": "p90 of nominal unmitigated error, charged per deferred case in "
                                                 "policy_cost_km so that abstaining on everything cannot win"},
        "config": cfg.to_dict(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cpu_count": __import__("os").cpu_count(), "packages": pkgs, "code_version": CODE_VERSION},
        "ai_disclosure": "Code, documentation and experiment design were generated by Claude (Anthropic, model "
                         "claude-opus-5-5) via Claude Code following info/CLAUDE_INSTRUCTIONS.md; the AI component "
                         "under test is the sklearn model, not the LLM.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"wrote {out/'metrics.csv'}, {out/'manifest.json'}, {out/'example_cases.csv'} (default subset {name})")


if __name__ == "__main__":
    main()
