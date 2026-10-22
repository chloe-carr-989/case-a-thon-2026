# AIS Feed-Degradation Lab (Case-a-thon 2026: Unintentional Data Poisoning)

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
* **Data:** MarineCadastre AIS broadcast points for 2026-01-01 (`../data/dataset3/` at the repository root, not redistributed).
  Everything for this project lives in `website/`; run all commands from that folder.

## Quick start (Windows PowerShell)

```powershell
cd website
# one-time setup (Python 3.14 was used; 3.11+ should work)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1) pipeline: subsets 4 -> 3 -> 2 -> 1, then the full day (~47 s total measured here)
.venv\Scripts\python scripts\run_pipeline.py --subsets subset4,subset3,subset2,subset1,full

# 2) dashboard (opens http://localhost:8501)
.venv\Scripts\streamlit run app\dashboard.py
```

The repository already contains computed `results/`, so step 2 works without step 1 except for the
past-hour track of the selected vessel on the map, which reads the git-ignored `cache/parquet`. Tests:
`.venv\Scripts\python -m pytest -q tests`.

CLI options: `--subsets`, `--input <csv|csv.zst>` with `--subsets custom`, `--region all|us_east|gulf|us_west`,
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

Timings are from `results/benchmarks.csv` on a 12-thread machine with 19 GB RAM. All runs include a cold Parquet ingest
(full day: 4.7 s of the 27.4 s).

## Results (default demo = full day, 3 seeds, 57,160 held-out cases from 1,060 test vessels)

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
experiment.py 5 conditions x 2 treatments x 5 windows x 3 seeds on identical cases -> metrics
scripts/run_pipeline.py   CLI, subset benchmark, results/*, manifest
app/dashboard.py          Streamlit + Plotly (offline Scattergeo, no tile server or API keys)
```

**Scale strategy.** DuckDB streams CSV into hour-partitioned Parquet (24 files per day, so no tiny-file explosion).
Memory is capped at 4 GB with 8 threads. Joins are per-vessel ASOF joins, not all-pairs. Training is bounded
(`--sample-vessels`, `max_train_rows=200k`). The UI reads compact results (≈19 MB of per-case rows for the
default run) and decimated map points. New days add new `day=` partitions.
**Verified scale:** one 869 MB, 8.5M-row day in 21–28 s end to end (27.4 s in the final cold run). **Not verified:** 10–100 GB multi-day runs.
The design targets them, but no throughput claim is made beyond what is measured above.

## Limitations

* One day of data. The windows are real chronological periods, but the degradation progression is **simulated**.
* The later AIS report is a noisy proxy for position: it is not ground truth, and it can itself be glitched.
* Most vessels are moored, so all-vessel medians are a few meters. Read the *underway* subgroup for movement skill.
* The synthetic source label (`SYN-FASTFEED`) and its message volumes are simulated. AIS here has no source IDs.
* Countermeasure thresholds were set a priori, without tuning on the test windows. They were not optimized either.
* Vessel-disjoint split by MMSI hash. Vessels sharing routes or ports can still make the test easier than truly new traffic.
* Human-review effort is the measured wall time in a demo UI, not an operator study.

See `docs/DEMO.md`, `docs/DECISION.md`, `docs/PLAN.md`, and `results/manifest.json`.

## Troubleshooting

* `ModuleNotFoundError`: use the venv interpreter (`.venv\Scripts\python`).
* The dashboard says "No results found": run step 1.
* The selected vessel has no blue track line: `cache/` was deleted, so rerun step 1.
* DuckDB memory errors on bigger inputs: pass `--region` or `--start/--end`, or lower `--sample-vessels`.
