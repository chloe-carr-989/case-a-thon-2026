# AIS Maritime Intelligence (Case-a-thon 2026: Unintentional Data Poisoning)

One map, two workflows, both built on a model of **normal traffic learned from historical AIS**:

* **Commercial: when will this vessel arrive?** Select a cargo ship or tanker to see its learned lane, its expected
  route (dashed, along observed traffic), the ETA with an 80% window, and the other vessels inbound to the same place.
* **Government: which vessels move unusually?** Vessels are scored every 30 min against their own category's lanes.
  Selecting one shows the score, the reasons in plain language, AIS data quality and freshness, the ERA5 wind as
  context, and a human-review log (reviewed / false positive / note). Labels describe movement relative to history.
  They are not evidence of identity, intent or threat.
* **Shared model** (`src/aisdp/lanes.py`): per-category sea lanes (0.01° cells: traffic, course histogram, speed
  quartiles, stop share) and a routing graph built from observed vessel transitions, learned only from training-period
  reports of vessels that are never tested. Pleasure craft are excluded throughout.
* **Data poisoning story:** every model runs on the same held-out cases under a clean feed and three SIMULATED
  degradations (stale reports, an unreliable high-volume source, both), with and without a quality-aware countermeasure.
  The compact **Model evaluation** section in each mode shows the effect; the original pages are under **Evaluation**.

Measured on the website default `jan01_04` (train Jan 1-3, test the unseen Jan 4; `results/metrics.csv`):

| | clean feed | unreliable source, naive | unreliable source + countermeasure |
|---|---|---|---|
| ETA mean error, model / straight-line baseline | 1 h 03 min / 1 h 22 min | 1 h 05 min | 1 h 05 min (98% answered) |
| anomaly alerts caused by bad data, per 1,000 vessel-windows | 0 | 46 | 7 |
| injected 2-8 km route drifts caught (known labels) | 48% | 52% | 49% |

* ETA arrival proxy = start of the next >= 30-min stop in an inferred arrival zone (not a berth call, not unloading).
  The 80% window is split-conformal calibrated: it holds 81% of the time when the predicted zone is right (about half
  of cases) and 53% when it is wrong, so destination prediction is the limiting factor.
* Anomaly thresholds: 97.5th / 99.5th percentile of the same category's calibration windows (the 48 h before the split,
  vessels the lanes never saw), so about 2.5% of normal windows are "unusual" by construction. No verified anomaly
  labels exist, so real-world detection accuracy is not claimed. Component weights were chosen on a vessel-disjoint
  split of the calibration period: off-lane distance alone caught 62% of injected drifts there, an equal-weight sum of
  all four components 21%, an Isolation Forest 19%. The injected test only contains off-lane drifts.
* Weather is real ERA5 reanalysis wind (Open-Meteo archive, fetched once by `scripts/fetch_weather.py`). It is shown
  as context and never changes a label.

```powershell
cd website
.venv\Scripts\python scripts\run_pipeline.py --subsets jan01_04 --train-frac 0.75 --budget-s 6000   # ~16 min, all tasks
.venv\Scripts\python scripts\run_pipeline.py --subsets jan01_04 --train-frac 0.75 --tasks sea_lanes,anomaly  # some tasks, merged into results/
.venv\Scripts\python scripts\fetch_weather.py      # optional, needs network once
.venv\Scripts\streamlit run app\dashboard.py --server.address localhost
```

The sections below describe the original feed-degradation lab. Their numbers come from the earlier `jan01_02` and
single-day runs (kept in `results/runs/`).

---

## Original lab: AIS Feed-Degradation Lab

**Question.** When stale reports and a high-volume, low-quality source enter a vessel-tracking feed, how
much worse do 15-minute AI position estimates get, and does a quality-aware inference pipeline restore
useful performance without just refusing to answer?

* **User / decision:** harbor-traffic or logistics analyst. Where will this vessel be in 15 min,
  and should I trust the estimate or send it for review?
* **AI component:** two `HistGradientBoostingRegressor`s predict the vessel's mean east/north velocity
  over the horizon. They learn a residual on top of the reported SOG/COG (or, when that is missing, the observed track).
  Inputs: latest AIS report plus the previous fix. The model is **trained once on nominal data and frozen.**
  Constant-velocity dead reckoning is reported as a non-ML benchmark.
* **Failure stage:** inference-time AIS feed degradation (stale, low quality, and both together).
* **Four questions** (one app page each, plus a shared data-integrity page; see `docs/CASE_DEMO.md`):
  1. *Where is this vessel going?* Next **inferred arrival zone** within 6 h, from a trained candidate-zone ranker.
  2. *Where will this vessel be in 15 minutes?* The original task, unchanged; its results are bit-identical.
  3. *How congested will this port be?* An AIS **occupancy proxy** (vessels dwelling in a study zone in the next
     hour), not congestion or waiting time.
  4. *How busy will this area be?* Unique vessels per 0.25° cell in the next hour.
* **Shared checks:** record-level integrity flags and observation/reporting-gap detection
  (`src/aisdp/quality.py`), tested on real records and on synthetic withheld or delayed runs.
* **Data:** MarineCadastre AIS broadcast points for 2026-01-01 and 2026-01-02 (`../data/dataset3/marine_2026-01/`; subsets in
  `../data/dataset3/marine_subsets/`; not redistributed). The website default was the two-day run `jan01_02`; it is now `jan01_04` (see top).
  Everything for this project lives in `website/`; run all commands from that folder.

## Quick start (Windows PowerShell)

```powershell
cd website
# one-time setup (Python 3.14 was used; 3.11+ should work)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1) pipeline: all tasks on subsets 4 -> 3 -> 2 -> 1, then the full day (about 4 min measured, warm Parquet cache)
.venv\Scripts\python scripts\run_pipeline.py --subsets subset4,subset3,subset2,subset1,full
#    only some tasks:  --tasks quality,position_15m,traffic_density,destination,port_occupancy
#    earlier two-day default (train on Jan 1, test on Jan 2; ~5 min):
.venv\Scripts\python scripts\run_pipeline.py --subsets jan01_02 --train-frac 0.5 --budget-s 3000

# 2) dashboard (opens http://localhost:8501)
.venv\Scripts\streamlit run app\dashboard.py
```

The repository already contains computed `results/`, so step 2 works without step 1 except for the
past-hour track of the selected vessel on the map, which reads the git-ignored `cache/parquet`. Tests:
`.venv\Scripts\python -m pytest -q tests`.

CLI options: `--tasks`, `--subsets`, `--input <csv|csv.zst>` with `--subsets custom`, `--region all|us_east|gulf|us_west`,
`--start/--end 'YYYY-MM-DD HH:MM:SS'`, `--max-rows`, `--sample-vessels` (default 4000), `--seeds 0,1,2`,
`--budget-s` (larger subsets are skipped after one exceeds it), `--output`.

## Data and subsets (measured)

The organizer subsets are **byte-identical head-prefixes** of `ais-2026-01-01.csv` (checked with `cmp`), so
they are nested and never combined.

| subset | bytes | rows | vessels | span (UTC) | status | total s |
|---|---|---|---|---|---|---|
| subset4 | 881 | 7 | 7 | 00:00:01–00:00:04 | insufficient target pairs (no 15-min futures) | 0.08 |
| subset3 | 21,440 | 200 | 200 | 00:00:00–00:00:28 | insufficient target pairs | 0.11 |
| subset2 | 60.5 MB | 589,669 | 14,165 | 00:00–01:57 | completed, 2,648 test cases | 5.9 |
| subset1 | 433 MB | 4,229,555 | 15,605 | 00:00–13:46 | completed, 29,776 test cases | 13.5 |
| full day | 869 MB | 8,497,825 | 17,123 | 00:00–23:59 | completed, 57,160 test cases (**default demo**) | 27.4 |

With all tasks (latest rows of `results/benchmarks.csv`, warm ingest cache): subset2 7.1 s. There, integrity and
position are completed, but counts and destination are `insufficient_data`: the 3-h lookback plus 1-h target does
not fit in 2 h, and there are only 31 / 5 labelled arrivals. Subset1 takes 69 s and the full day 172 s, with all
tasks completed. Full-day task times: integrity 25 s, density 54 s, destination 54 s, port 11 s, position 27 s.
Warm full-day runs measured 145–172 s on this machine. A cold start (full-day Parquet cache deleted first, with code
before the final determinism fix) took 159.9 s, including a 4.7 s ingest.

Timings are from `results/benchmarks.csv` on a 12-thread machine with 19 GB RAM. All runs include a cold Parquet ingest
(full day: 4.7 s of the 27.4 s).

## Two-day results (earlier default `jan01_02`, train on Jan 1, test on Jan 2)

Jan 1 and Jan 2 ingested together: 16,158,949 reports, 19,231 vessels, 313 s for all tasks (warm cache for Jan 1).
`--train-frac 0.5` puts the chronological split at Jan 2 00:00 UTC for every task, so all models are trained on Jan 1
and tested on a day they never saw. Position: 217,560 train cases (capped at 200k rows), 93,015 test cases from 992
test vessels. Means over seeds 0/1/2. Single-day numbers (next section) are kept in `results/runs/full/`.

| metric (clean feed) | Jan 1 only | Jan 1 → Jan 2 |
|---|---|---|
| position mean / p90 km, all vessels | 0.182 / 0.385 | 0.169 / 0.361 |
| position mean km, underway (constant velocity) | 0.916 (0.952) | 0.899 (0.941) |
| density MAE (persistence) | 0.409 (0.458) | 0.325 (0.368) |
| port proxy MAE (persistence) | 2.62 (**2.26**) | 2.19 (**2.08**) |
| destination top-1 / top-3 (baseline top-1) | 0.502 / 0.773 (0.363) | 0.499 / 0.756 (0.326) |

| position, all vessels | unmitigated mean km | quality-aware mean km (coverage) | policy cost |
|---|---|---|---|
| nominal | 0.169 | 0.170 (0.997) | 0.170 |
| stale | 0.454 | 0.208 (0.771) | 0.243 |
| low quality | 3.926 | 0.383 (0.967) | 0.383 |
| combined | 4.249 | 0.418 (0.735) | 0.403 |
| holdout unseen bias | 1.135 | 0.312 (0.935) | 0.315 |

What changed with the second day:

* Clean-feed errors are slightly lower on Jan 2 for position, density and port, but the comparisons with the baselines
  look the same as before. The port-proxy model still does **not** beat persistence.
* Destination top-1 accuracy holds (0.50). Top-3 drops a little (0.77 → 0.76). The model's lead over the heading
  baseline grows (+0.14 → +0.17).
* The countermeasure results look like the single-day run (combined policy cost 4.25 → 0.40 km at 74% coverage).
  For underway vessels the unseen-bias holdout is now clearly **worse** with the countermeasure: mean
  1.277 → 1.536 km, p90 2.56 → 3.61 km. On Jan 1 it was 1.295 → 1.305 km.
* The model is still trained once and frozen. Note that "more data" here means more training *time span*. The
  vessel sample (4,000) and the row cap (200k) are unchanged, so the position model saw 200k rows, up from 110k.

## Single-day results (Jan 1 only, 3 seeds, 57,160 held-out cases from 1,060 test vessels)

Errors are haversine km between the predicted position and the later AIS report, measured **on answered cases**.
Coverage is the share of cases answered. *Policy cost* charges each deferral 0.385 km (the nominal unmitigated p90), so
deferring everything cannot win. Values are means over seeds 0/1/2. The seed-to-seed std of mean error is ≤0.05 km for all vessels and ≤0.25 km for underway vessels.

**All vessels**

| condition | mitigation | coverage | median | p90 | mean | policy cost |
|---|---|---|---|---|---|---|
| nominal | none | 1.000 | 0.003 | 0.385 | 0.182 | 0.182 |
| nominal | quality-aware | 0.997 | 0.003 | 0.385 | 0.176 | 0.176 |
| stale | none | 1.000 | 0.003 | 0.879 | 0.478 | 0.478 |
| stale | quality-aware | 0.774 | 0.003 | 0.447 | 0.214 | 0.252 |
| low quality | none | 1.000 | 0.013 | 6.046 | 4.017 | 4.017 |
| low quality | quality-aware | 0.965 | 0.010 | 1.141 | 0.399 | 0.399 |
| combined | none | 1.000 | 0.016 | 7.797 | 4.311 | 4.311 |
| combined | quality-aware | 0.737 | 0.008 | 1.208 | 0.434 | 0.421 |
| holdout unseen bias | none | 1.000 | 0.013 | 2.055 | 1.183 | 1.183 |
| holdout unseen bias | quality-aware | 0.932 | 0.004 | 0.588 | 0.287 | 0.294 |

**Underway vessels (SOG ≥ 2 kn at decision, 8,994 cases):** nominal 0.916 km mean (constant velocity: 0.952);
combined 6.84 km unmitigated, 1.66 km mitigated at 76% coverage; **holdout bias 1.295 km unmitigated vs 1.305 km mitigated
(no improvement in mean, worse p90 2.52 → 2.86 km): the countermeasure does not generalize to the unseen defect for moving vessels.**

**Key findings**
1. *More data made it worse.* As the synthetic high-rate source rose from 10% to 60% of the latest records
   (windows 0→4, combined condition), unmitigated p90 error rose from 0.98 km to 18.9 km.
2. *The trained model is more fragile than physics.* On nominal data the model beats constant velocity only
   slightly (p90 0.385 vs 0.397 km). Under low quality, its mean error is about 2× worse (4.0 vs 2.1 km): one
   corrupted fix becomes a huge observed-track velocity feature, which the model extrapolates (see case 2 in
   `results/example_cases.csv`, a 249 km error).
3. *The countermeasure recovers most of the loss* (combined policy cost 4.31 → 0.42 km) but costs coverage (74%).
   It also flagged 12 of 116 fast (≥20 kn) unmodified cases. Some were real AIS glitches (a 393 km jump),
   others were good estimates.
4. *Failure / tradeoff:* age compensation over a 17-minute gap made one estimate worse (16.9 → 24.7 km, the vessel turned or stopped).
   The unseen-bias holdout is not fixed for underway vessels. Deferral rate reaches 47% in the worst window.

Model compute latency is about 0.005 ms per case in batch for both pipelines. Human review time is measured separately in the UI log and is demo-only.

## Expansion tasks (full day, 3 seeds; definitions in `results/manifest.json` → `tasks`)

| task | trained model | comparator(s) | clean-feed result | evaluated on |
|---|---|---|---|---|
| destination | HGB candidate-zone ranker | heading-aligned nearest zone | top-1 0.502 / top-3 0.773 vs 0.363 / 0.633 | 3,594 decisions, 647 unseen vessels, 324 zones |
| traffic density | HGB on change from persistence | persistence, 3-h mean, train mean | MAE 0.409 vs 0.458 / 0.582 / 0.865 vessels | 3,225 cells × 45 hourly-ahead origins |
| port occupancy proxy | same as density | same | MAE 2.62 vs **2.26 (persistence wins)** / 2.78 / 4.95 | 60 study zones × 45 origins |

Matched stress tests (the same four conditions × countermeasure off/on, identical cases and references):

| metric | clean off / on | stale off / on | low quality off / on | combined off / on |
|---|---|---|---|---|
| density MAE | 0.409 / 0.425 | 0.931 / 0.453 (85% answered) | 4.279 / 0.457 | 3.538 / 0.496 (85%) |
| port proxy MAE | 2.62 / 2.93 | 12.13 / 2.86 (85%) | 63.4 / 3.62 | 51.4 / 3.27 (85%) |
| destination top-1 on answered (over all cases) | 0.502 / 0.502 | 0.476 / 0.488 (0.474 / **0.389**) | 0.487 / 0.494 | 0.464 / 0.480 (0.461 / **0.382**) |

Read the honest parts too:

* The port-proxy model does **not** beat persistence.
* The count countermeasure slightly hurts clean-feed accuracy, because it drops real one-report vessels.
* Persistent synthetic ghost IDs pass the filters.
* For destination, reusing the 15-minute age gate lowers accuracy over all cases. It defers stale but still useful
  inputs (306 of 733 deferred stale cases had been answered correctly).

**Data integrity (full day, observed):**

* 7,375 of 8,497,825 records are flagged: 4,007 non-vessel MIDs, 2,456 implausible jumps, 1,059 exact duplicates
  and 43 implausible SOG.
* 31,216 reporting gaps from 7,445 vessels (median 21.7 min). Of these, 1,983 are possible coverage exits.
* On synthetic withheld/delayed runs, real-time recall is 96–97% with no false alarms on shorter events.
  At decision time a delay is indistinguishable from a loss; it is only relabelled once the late records arrive.

## Architecture

```
raw CSV / .csv.zst --(DuckDB streaming read, region/time pushdown)--> cache/parquet/<name>_<region>_<hash>/day=/hr=
   ingest.py        cached by sha256 fingerprint + region + schema version; partitions by date and hour
features.py   sentinel cleaning (observed defects) -> deterministic vessel sample -> per-vessel ASOF joins:
              decision anchors (1 per vessel per 10 min) -> target = first report in [t+13, t+17] min
              state(q) = latest report <= q, prev fix 1-15 min earlier  (no future rows possible)
model.py      residual HGB on mean velocity; constant-velocity benchmark
corruption.py seeded stale delays / SYN-FASTFEED defects / unseen bias (labelled per case)
mitigation.py age gate + compensation, validity + track-consistency checks, imputation, fallback, quality score, deferral
              (+ sanitize_state: the same checks for the destination task)
quality.py    record flag codes (DuckDB window pass), reporting gaps + coverage-exit heuristic, synthetic gap test
zones.py      10-min vessel buckets -> dwell events -> train-period inferred arrival / port study zones
counts.py     (unit, 15-min bucket, mmsi) presence -> sort-once interval sweep for unique counts per window,
              residual HGB, SYN-FASTFEED ghosts/duplicates/omissions, late-feed detection, transient filter
destination.py arrival labels with cutoffs, candidate-zone features, HGB ranker, baseline, abstention, metrics
tasks.py      per-task runners writing results/runs/<subset>/*;  contracts.py  thin PredictionResult adapters
experiment.py 5 conditions x 2 treatments x 5 windows x 3 seeds on identical cases -> metrics
scripts/run_pipeline.py   CLI, subset benchmark, results/*, manifest
app/dashboard.py          Streamlit router (st.navigation) -> q_position.py (original page), q_destination.py,
                          q_port.py / q_density.py (shared count_page.py), q_integrity.py; common.py helpers
```

**Scale strategy.** DuckDB streams CSV into hour-partitioned Parquet (24 files per day, so no tiny-file explosion).
Memory is capped at 4 GB with 8 threads. Joins are per-vessel ASOF joins, not all-pairs. Training is bounded
(`--sample-vessels`, `max_train_rows=200k`). The UI reads compact results (≈19 MB of per-case rows for the
default run) and decimated map points. New days add new `day=` partitions.
**Verified scale:** one 869 MB, 8.5M-row day in 21–28 s end to end (27.4 s in the final cold run). **Not verified:** 10–100 GB multi-day runs.
The design targets them, but no throughput claim is made beyond what is measured above.

## Limitations

* **New tasks:**
  * Zones are inferred grid cells, not ports. Destination accuracy is conditional on an observed stop in a known zone.
  * The port count is an occupancy proxy, so there are no delay claims.
  * Count origins overlap (60-min windows every 15 min) and the same cells are used for training and test.
  * The new-task timings were measured with a warm Parquet ingest cache. The cold ingest timings are the earlier rows of
    `results/benchmarks.csv`.

* One day of data. The windows are real chronological periods, but the degradation progression is **simulated**.
* The later AIS report is a noisy proxy for position: it is not ground truth, and it can itself be glitched.
* Most vessels are moored, so all-vessel medians are a few meters. Read the *underway* subgroup for movement skill.
* The synthetic source label (`SYN-FASTFEED`) and its message volumes are simulated. AIS here has no source IDs.
* Countermeasure thresholds were set a priori, without tuning on the test windows. They were not optimized either.
* Vessel-disjoint split by MMSI hash. Vessels sharing routes or ports can still make the test easier than truly new traffic.
* Human-review effort is the measured wall time in a demo UI, not an operator study.

See `docs/CASE_DEMO.md` (all four questions), `docs/DEMO.md`, `docs/DECISION.md`, `docs/PLAN.md`, and `results/manifest.json`.

## Troubleshooting

* `ModuleNotFoundError`: use the venv interpreter (`.venv\Scripts\python`).
* The dashboard says "No results found": run step 1.
* The selected vessel has no blue track line: `cache/` was deleted, so rerun step 1.
* DuckDB memory errors on bigger inputs: pass `--region` or `--start/--end`, or lower `--sample-vessels`.
