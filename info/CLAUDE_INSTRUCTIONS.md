# Claude implementation instructions: Case-a-thon 2026

## Mission and time limit

Build the complete, demonstrable MVP for the William & Mary AI Club Case-a-thon 2026, **Unintentional Data Poisoning**, using **Dataset 3: MarineCadastre AIS vessel traffic** already present under `data/`. You have **at most 60 minutes** for an MVP. Work autonomously: inspect the repository, implement, run, test, fix, and leave usable deliverables. Do not spend the hour only planning. If any feature threatens the deadline, prioritize the experiment and working demonstration over polish.

Read `Case-a-thon Prompt.pdf` if available. Its binding requirements are summarized here. Do not fabricate experimental findings, dataset characteristics, timings, or successful runs.

## First action: inspect the actual files

1. List the repository and `data/` recursively with file sizes, file formats, and candidate subsets. Locate dataset 3 and files named or corresponding to `subset4`, `subset3`, `subset2`, `subset1` (including alternate naming, extensions, and nested directories). Do not assume file names or that all subsets exist.
2. Inspect only a few rows and metadata initially. Confirm the AIS schema, timestamp format, vessel identifier, latitude, longitude, speed/course fields, duplicate frequency, and geographic/time coverage. Likely fields include `mmsi`, `base_date_time`, `longitude`, `latitude`, `sog`, `cog`, `heading`, `vessel_type`, but map actual columns rather than assuming.
3. Determine whether subsets are nested, independently sampled, or different date ranges. Avoid double-counting. Preserve original raw data unchanged.
4. If subset files are missing, locate the available dataset and create deterministic, documented chronological or vessel-preserving development subsets, ordered smallest to largest. Do not silently claim these are organizer-provided subsets.
5. Record initial file sizes, row counts (where feasible), time range, coordinate bounds, null rates, unique vessels, and actual subset selection rules. Identify the hardware and available Python/Node dependencies before committing to a stack.

## Core problem and product

**Application:** Maritime AIS decision support. **User:** harbor traffic/logistics analyst. **Decision:** estimate where a vessel will be at a short future horizon (prefer 10 to 30 minutes if the observed sampling cadence supports it), and decide whether to trust the estimate or request review. **AI component:** a trained lightweight regression model predicting future position or displacement from *information available at prediction time*. **Consequences:** poor situational awareness, misleading estimated movement, unnecessary review, or missed unusual movements. Do not imply collision avoidance certification, verified port arrival, cargo delivery, or autonomous-vessel deployment.

Use future AIS observations as a **noisy evaluation proxy**, not perfect truth. Report error in **kilometers or meters** using haversine/geodesic distance, plus median and 90th percentile errors. Measure coverage, deferrals, and human-review workload separately. A trivial constant-velocity extrapolation is a useful non-ML benchmark but **does not replace the trained AI component**. Train a fast model such as HistGradientBoostingRegressor, RandomForestRegressor with bounded complexity, or LightGBM if already installed; predict future east/north displacement or future latitude/longitude with physically sensible scaling. Choose the simplest approach that runs promptly.

## Experimental design: mandatory

Implement a reproducible, chronological experiment with **four conditions** and **two treatments each**, always evaluated against the same held-out unmodified target positions:

| Condition | No mitigation | With mitigation |
|---|---|---|
| Nominal/reference | required | required |
| Stale only | required | required |
| Low quality only | required | required |
| Combined stale + low quality | required | required |

Select a precise failure stage. Default: **inference-time AIS feed degradation**, with a fixed trained model, so the experiment isolates bad input data rather than mixing in retraining effects. Explicitly state that the model is not retrained during scenario progression. Use the same base cases, model, target horizon, evaluation timestamps, and targets in all eight comparisons.

**Staleness:** simulate a growing fraction of reports that are delayed or frozen, making the most recent available vessel state older than the decision time. Preserve the actual observation timestamp and an arrival/availability timestamp so the system can calculate report age; never use a future observation. If simulating a stale state, select the most recent record that would have been available at that decision time, not a future row relabeled as old.

**Low quality:** independently inject plausible position jitter, speed/course corruption, missing values, and/or source-specific bias into input records, using configurable severity and seeded randomness. Keep defects separate from the original dataset's observed problems. Include a high-volume unreliable-source scenario or a controlled increase in the proportion of low-quality records over simulated windows, with source labels explicitly **synthetic** if the data lack source identifiers. Track source volume and mix. Do not simply add duplicates to evaluation cases and claim more independent observations.

**Combined:** apply both mechanisms together, including at least one interaction where delayed reports and corrupted reports coexist.

**Countermeasure:** quality-aware inference pipeline: age gating, physically plausible speed/position checks, robust handling of missing/outlier fields, confidence/quality scoring, and fallback to last credible trajectory or explicit abstention/human review. Compare against the unmitigated model on identical cases. Avoid scoring deferred cases as zero error: report **both error on answered cases and coverage**, plus a policy-level metric or transparent penalty so abstaining on everything cannot win. Track how many valid rare/fast movements were incorrectly flagged.

**Progression:** at least 5 time windows (real chronological windows if enough data; otherwise clearly label a controlled *simulated degradation progression*). Progressively increase staleness rate, corruption rate, and unreliable-source share. Show age, quality, source volume, model error, and coverage by window. Run a nominal window and a previously unseen later time window or different severity/type as holdout. Use vessel-group-aware, time-respecting splits: no future targets in training features, no randomly mixed overlapping trajectories between train/test, no same-vessel leakage where it can be avoided without making the experiment impossible. Explain the compromise if a strict vessel-disjoint chronological split is infeasible.

**Mechanism evidence:** provide a case-level walkthrough from raw AIS observation and its timestamp -> altered observation -> model prediction -> actual later AIS proxy -> error -> quality flags -> mitigated prediction/deferral. Include a concrete countermeasure failure or tradeoff, for example a real high-speed maneuver flagged as implausible, lower coverage, or increased latency. Repeat injected experiments for multiple fixed seeds if time allows; otherwise state uncertainty limits explicitly.

## Data engineering for tens of gigabytes

Design now for **10 to 100+ GB of raw AIS CSV**, even though the MVP runs on subsets. Use **DuckDB + partitioned Parquet** (and optionally Polars/PyArrow) to avoid loading whole CSVs into pandas memory. Support chunked ingestion, projection/filter pushdown, selective time and geographic queries, bounded training samples, incremental append, and precomputed/downsampled map snapshots. Do not create a giant JSON payload or send all AIS points to the browser. Keep intermediate artifacts under `artifacts/` or `cache/`, ignored by git. Separate ingest, validation, feature construction, experiments, model training, API, and UI.

Partition by observation date or month and optionally spatial region, chosen based on actual dataset distribution; avoid millions of tiny files. Persist sorted/filtered features and experiment results, and cache by source fingerprint + config + code/schema version. Ingest only new files/partitions when possible. Use streaming scans and explicit memory/thread limits. Make a reproducible CLI with `--input`, `--subset`, `--max-rows` or `--sample-vessels`, `--region`, `--start`, `--end`, `--seed`, `--output`, and sensible defaults where relevant. Don't assume every CLI option must exist on every command. Document the difference between architecture capable of scaling and a benchmark **actually verified** at tens-of-GB scale. Do not claim untested throughput.

Feature design: time-of-day, report age, prior positions/velocity from historical observations, speed over ground, course over ground, heading if valid, vessel type if usable, recent track quality, and position. Handle AIS sentinel/invalid values and longitude wrapping; compute geographic displacement appropriately. Choose a prediction horizon that yields sufficient observed pairs. Construct target pairs using per-vessel as-of/nearest-future matching with an explicit tolerance, never cross vessel IDs, and ensure no future leakage. Avoid expensive all-pairs joins. Filter impossible jumps carefully without deleting legitimate rare events from the evaluation set.

## Web application

Build a **minimal, responsive, professional dashboard**. Prefer Streamlit + Plotly/PyDeck for speed if appropriate; FastAPI + lightweight frontend is acceptable only if it can be completed faster with existing project dependencies. No animations. No unnecessary accounts, cloud services, API keys, or external map dependencies that prevent offline demo. Prioritize a real working UI over mockups.

Required views:

1. **Overview:** dataset/subset name, file size, vessel count, time coverage, prediction horizon, model, baseline error, experiment status.
2. **Experiment controls:** condition (nominal, stale, low quality, combined), mitigation toggle, severity/progression window, vessel/region/time filters, and rerun or load cached results. Defaults must show an actual computed example.
3. **Map:** bounded number of vessel points or trajectories with actual location, predicted future position, and target position when available; color or symbol distinguishes quality flags, and selected vessel shows timestamp, age, error, and correction status. Plot a geographic subset and decimate points. No requirement for an animated timeline; a discrete time slider is sufficient.
4. **Results:** eight-way comparison, errors by time window, data age, quality, source mix/volume, coverage and deferrals, plus one mechanism case study.
5. **Human oversight:** show why a prediction is unreliable, supporting AIS evidence, age and quality score, operator options to accept, correct, or defer; log actions locally with timestamp and reason. Include a sample walkthrough. Distinguish measured human effort from estimated/demo-only effort.

Use accessible labels, meaningful empty/error states, responsive layout, and local data. UI controls must be functional. Never display synthetic results as if measured. If training/experiments are slow, run offline and let the UI load cached machine-readable results immediately.

## Output files and reproducibility

Create and populate at least:

- `README.md`: installation, Windows PowerShell commands, exact one-command or two-command demo launch, data location, dependency versions, troubleshooting, architecture, scale strategy, and limitations.
- `requirements.txt` or `pyproject.toml` with pinned or bounded versions.
- `src/` or equivalent modules for ingest, AIS validation, features/targets, modeling, corruption, mitigation, evaluation, and dashboard.
- `scripts/` for reproducible pipeline execution and benchmark runs.
- `results/metrics.csv` or `.parquet` containing condition, mitigation, window, seed, sample count, error metrics, coverage, deferral, data-age and corruption measures, and runtime.
- `results/manifest.json`: original source/URL if verifiable, file fingerprints, retrieval date if known (otherwise unknown), actual file paths, sizes, row counts, independent vessels/trajectories, time range, units, subset/filter rules, missing-value policy, split definitions, random seeds, defect configuration and modification log, dataset rights statement/status, model/dependency versions, and AI-tool/prompt disclosure. Do not invent licenses or source dates.
- `results/example_cases.csv` or `.json` for explainable case-level examples.
- `docs/DEMO.md`: timed 10-minute presentation/demo narrative covering problem, architecture, four conditions, countermeasure, result, failure/tradeoff, human oversight, and next step. Include a recorded fallback plan. If video recording cannot be created automatically, provide a runnable screen-recording checklist, not a fictitious video.
- `docs/DECISION.md`: who reviews flagged cases, who owns data quality/model performance, logging and approval requirements, stop/no-go criteria, bounded pilot proposal, named **roles** (not invented people), and decision checkpoint.
- At least a few automated tests covering leakage, reproducibility, corruption isolation, matching of evaluation cases, and deferral accounting.

Preserve raw data and do not redistribute it. Keep generated artifacts compact. The prompt requires a modest web app, results/data manifest, and 10-minute presentation; if time permits create presentation slides, but a correct app and experiment take priority.

## Hard 60-minute execution plan

**0-5 minutes: inspect and decide.** Examine `data/`, detect subset naming, inspect schema, choose actual region, horizon, and minimum viable architecture. Write a short plan to `docs/PLAN.md`, then implement immediately.

**5-18 minutes: pipeline.** Build chunked/DuckDB ingest, AIS cleaning, chronological future-target construction, model training, and cached reference metrics using **subset 4**. Verify actual output rows and nonempty targets.

**18-32 minutes: controlled experiments.** Implement the four conditions, two treatments, seeded progression, metrics, and case-level explanation. Run subset 4 end-to-end; validate all eight comparisons. Save artifacts.

**32-45 minutes: dashboard.** Build and run the UI with a map, comparison plot/table, time/progression controls, and human review panel using computed results.

**45-53 minutes: scale tests.** If subset 4 works comfortably, run **subset 3**, then **subset 2**, then **subset 1**, stopping when runtime or memory risk threatens completion. Set a per-subset execution budget, for example ~2 minutes each including training and experiment; stop immediately if exceeded. Keep the largest successfully completed subset as the default demo, and log skipped subsets with reasons. Prefer bounded model training and streamed features rather than letting model training scale linearly with raw volume.

**53-60 minutes: verify and document.** Launch the app, run smoke tests, verify plots reflect computed results, write manifest and demo narrative, capture benchmark timings and caveats, and fix critical failures. Leave an exact working launch command. If you cannot finish everything, state precisely what works and what is missing; never claim completion without verification.

## Quality gates and acceptance criteria

- An actual trained predictor runs and is evaluated against held-out future AIS observations.
- All **eight** experiment cells have real computed results on identical target cases, or any incomplete cells are clearly identified as unfinished.
- A multi-window time/progression chart shows data quality/age/volume alongside error and coverage.
- There is at least one meaningful mitigation and one documented failure/tradeoff.
- Human reviewer can inspect a flagged case and record an accept/correct/defer action.
- The app opens locally and uses genuine persisted results.
- Subset 4 is run first, then 3, 2, 1 only as time permits; every attempted subset has timing, row count, status, and reason for stopping.
- Large-data processing does not require reading the full source into RAM. Tens-of-GB support is an architectural target, not an unverified performance claim.
- No leaked future data, no fabricated metrics, no implied real-world deployment validation.

## Working style

Execute commands, inspect errors, iterate, and finish with a concise status report: implemented features, exact run command, dataset/subset used, number of evaluated cases, model and horizon, eight-way results, measured timings, tests, and remaining issues. Ask questions only if an essential missing detail blocks progress and cannot be discovered from the repository. Do not change unrelated project files. Keep code complete and readable; do not leave placeholders that make the app fail.
