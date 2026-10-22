# Case demo: four maritime questions under unintentional data poisoning

All numbers below come from `results/metrics.csv` and `results/manifest.json`. The run was the full day
(2026-01-01, 8.50 M reports, 17,123 vessels) with 3 seeds; values are seed means. Every degradation is
**simulated** on real data, and every reference outcome is a later AIS observation, which is a noisy proxy rather than truth.

Launch: `.venv\Scripts\streamlit run app\dashboard.py` (from `website/`), then open http://localhost:8501. The sidebar
has one page per question plus **Data integrity and reporting gaps**.

## The four questions: who decides what, and what an error costs

| question | user / decision | output | cost of an error |
|---|---|---|---|
| Where will this vessel be in 15 min? | Harbor-traffic analyst: trust the estimate or send it for review | lat/lon, or deferred | Wrong situational picture, unnecessary or missed review. **Not** for navigation or collision avoidance. |
| Where is this vessel going? | Logistics/port planner: which arrival area to expect a vessel at in the next 6 h | top-3 inferred arrival zones, or "unknown" | Resources planned for the wrong area. A zone is a 0.05° cell where vessels stop, not a named or verified berth. |
| How congested will this port be? | Port operations: expect a busy hour in a port/anchorage study zone | vessels dwelling (<1 kn) in the zone in the next 60 min | Misjudged load. This is an **occupancy proxy**: no berth capacity, queue or waiting-time data exist, so it cannot measure delays. |
| How busy will this area be? | VTS / waterway planner: expected traffic per 0.25° cell in the next 60 min | unique vessels, plus per 1000 km² | Under-staffed watch or misallocated patrols. Counts only vessels the AIS feed observed. |

## Trained components and their non-ML comparators (clean feed)

| task | model | comparator | held-out result |
|---|---|---|---|
| position 15 min | HGB residual on SOG/COG velocity | constant velocity | mean 0.182 km (all vessels); underway 0.916 vs 0.952 km |
| destination | HGB candidate-zone ranker | heading-aligned nearest zone | top-1 **50.2%** vs 36.3%, top-3 **77.3%** vs 63.3% (3,594 decisions from 647 vessels not seen in training, 324 distinct true zones) |
| traffic density | HGB on change from persistence | persistence / 3-h mean / train mean | MAE **0.409** vs 0.458 / 0.582 / 0.865 vessels per cell-hour (3,225 cells × 45 origins) |
| port occupancy proxy | same as density | same | MAE 2.62 vs **persistence 2.26** / 2.78 / 4.95: **the trained model does not beat persistence here** (only 1,920 training rows from 60 zones) |

## The 15-minute predictor: four feed conditions × countermeasure

This is the primary experiment. All eight cells use the same 57,160 held-out decisions and unmodified targets, a
frozen model and the same seeds. Values are for all vessels. Policy cost charges each deferral 0.385 km.

| condition | countermeasure | answered | mean km | p90 km | policy cost km |
|---|---|---|---|---|---|
| clean | off / on | 100% / 99.7% | 0.182 / 0.176 | 0.385 / 0.385 | 0.182 / 0.176 |
| stale | off / on | 100% / 77.4% | 0.478 / 0.214 | 0.879 / 0.447 | 0.478 / 0.252 |
| low quality | off / on | 100% / 96.5% | 4.017 / 0.399 | 6.046 / 1.141 | 4.017 / 0.399 |
| combined | off / on | 100% / 73.7% | 4.311 / 0.434 | 7.797 / 1.208 | 4.311 / 0.421 |

The same matched design was extended to the other three questions. It uses the same held-out cases and references
and the same seeds, and labels the defects SYNTHETIC.

| task (metric) | clean off / on | stale off / on (answered) | low quality off / on | combined off / on (answered) |
|---|---|---|---|---|
| density MAE, vessels | 0.409 / 0.425 | 0.931 / 0.453 (85%) | 4.279 / 0.457 | 3.538 / 0.496 (85%) |
| port proxy MAE, vessels | 2.62 / 2.93 | 12.13 / 2.86 (85%) | 63.4 / 3.62 | 51.4 / 3.27 (85%) |
| destination top-1 (answered) | 0.502 / 0.502 | 0.476 / 0.488 (80%) | 0.487 / 0.494 | 0.464 / 0.480 (80%) |
| destination top-1 over **all** cases | 0.502 / 0.500 | 0.474 / 0.389 | 0.485 / 0.491 | 0.461 / 0.382 |

## Why added observations can make outcomes worse

* **Position:** when the synthetic high-rate source supplied 10% → 60% of the latest records (windows 0 → 4), the
  unmitigated p90 error rose from 0.98 km to 18.9 km. More messages arrived, but the newest ones were wrong, and the
  model trusts the newest record.
* **Counts:** ghost IDs, duplicates and omissions from SYN-FASTFEED raise apparent traffic. Unmitigated density MAE
  grows across the combined windows: 1.47 → 2.57 → 3.51 → 4.87 → 5.27 vessels. Port-proxy MAE grows 17 → 81. Any
  forecast that reads recent inputs inherits the inflation: under a degraded feed the input-free training-period
  mean (0.865) beats every input-driven model. Counting unique MMSIs makes the duplicates harmless to the counts
  themselves; they only inflate message-volume features.
* **Late feeds** are the opposite failure. The newest buckets are missing, so naive counts read as a sudden drop
  (stale bias −0.76 vessels per cell, −10.8 per port zone).

## Countermeasures, and what they cost

* **Position:** age gate and compensation, plausibility and consistency checks, fallback to the last credible fix,
  and deferral. Details in README.
* **Counts:** drop invalid-MID IDs and one-message transients (a causal trailing-hour rule), detect a late feed from
  the delivered message volume, fall back to the last complete hour, and defer at 45+ min late.
* **Destination:** the same input checks as position (`mitigation.sanitize_state`), plus "unknown" when the top
  normalised score is below 0.15.

Failures and tradeoffs, as measured:

1. **The count filter costs accuracy on a clean feed.** Real vessels that report once in an hour are removed.
   Density MAE goes 0.409 → 0.425 and bias −0.17 → −0.21. Port proxy MAE goes 2.62 → 2.93.
2. **Persistent ghosts get through.** 15% of synthetic ghosts reuse a valid-looking ID within the hour. The filters
   cannot see them, so mitigated low-quality port MAE (3.62) stays above the clean level (2.62–2.93).
3. **Omissions cannot be repaired.** Lost reports stay lost, and the reference count is itself an undercount.
4. **For destination, the age gate is the wrong policy.** It defers 21% of stale cases. Top-1 accuracy on answered
   cases rises only 1–2 points, while top-1 over all cases falls from 0.474 to 0.389. Of the 733 deferred stale cases
   (seed 0), 306 had been answered correctly by the unmitigated ranker. Destinations are hours away, so a 20-minute-old
   report is still useful. The 15-minute gate should not be reused unchanged here.
5. **Position:** the unseen 1.5 km bias is not fixed for moving vessels; 12 of 116 valid fast vessels are flagged;
   and age compensation over a 17-minute gap made one estimate worse (16.9 → 24.7 km).
6. **Coverage:** in the worst window, the counts answer 67% of origins and the position task defers 47%.

## Case-level causal chain (destination, from `results/destination_cases_seed0.parquet`)

1. **Original report:** MMSI 368067890 at 15:40 UTC (input cutoff) is at 27.837 N, 97.098 W, 9.4 kn, course 84°.
2. **Injected defect** (low-quality condition, seed 0, SYN-FASTFEED): jitter and a position outlier. The delivered
   fix is 27.771 N, 97.030 W, about 9.9 km away.
3. **Prediction change:** the clean input gives top-1 zone `Z0.05_+27.825_-97.075`. The corrupted input gives
   `Z0.05_+27.825_-97.175`.
4. **Held-out reference:** the vessel stopped for 30+ min in `Z0.05_+27.825_-97.075` from 16:10 UTC, 30 min later.
   The unmitigated answer is wrong.
5. **Flags:** `implausible_jump` (the jump from the previous fix implies an impossible speed) and `stale_report`.
6. **Corrective action:** the countermeasure falls back to the previous credible fix and answers
   `Z0.05_+27.825_-97.075`, which is correct. In the app, a reviewer can record accept / correct / defer for this
   case on the *Where is this vessel going?* page.

The position page has the equivalent chain in `results/example_cases.csv` (e.g. a 249 km unmitigated miss caused
by one corrupted fix).

## One human correction / defer walkthrough (about 1 min)

1. Open **How busy will this area be?**, select *Stale + unreliable*, and turn the countermeasure on. Choose the
   latest forecast time.
2. If the feed is 45+ min late, the forecasts are withheld (hollow circles). If it is 15–30 min late, the side panel
   shows "fell back to the last complete hour".
3. In step 4, pick **Defer: wait for the late feed**, type a reason, and click **Record decision**.
4. The row goes to `results/review_log.csv` with the task, unit, origin, system action, reason and the
   **measured** seconds since the case opened. That is demo wall time, not an operator study. Model response time
   is logged separately (`latency_ms_per_case` in metrics.csv, well below 1 ms per case in batch).

## Who owns what

See `docs/DECISION.md`. In short:

* The **data-quality owner** owns the flag rules, gap thresholds, source onboarding and the defect log.
* The **model owner** owns all four models, their comparators and the metrics files.
* The **duty reviewer** works the deferral queue across tasks.
* The **operations supervisor** approves threshold changes and pilot go/no-go.

## Pilot and no-go criteria (additions for the new tasks)

* **Go only for a task whose trained model beats its simple comparator on the pilot replay.** Today that holds for
  destination and density, but **not for the port proxy**: ship persistence there, or keep it shadow-only.
* **No-go if mitigated policy-level accuracy is worse than unmitigated** on the pilot week. Destination currently
  fails this under stale input, so its age gate must be redesigned (e.g. widen it to the 6 h horizon) before a pilot.
* **No-go if the port proxy is presented as congestion or delay.** It is an occupancy proxy.
* **Deferral budget:** at most 30% of forecasts per day across all tasks, the same as the position rule.

## What is not done or not claimed

No named ports, verified arrivals, cargo, berth occupancy, waiting time, collision risk or autonomy claims. The data
cover one day, so there is no seasonality. Subsets 4 and 3 (seconds of data) and subset 2 (2 h) cannot support the
counts or destination, and the benchmark records them as `insufficient_data`. The counts use overlapping windows, so
their case counts overstate the independent evidence. No video was recorded.
