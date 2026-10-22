# Claude implementation specification: expand the AIS case-a-thon MVP

## Assignment

Update the **existing, working Streamlit application** (currently answering **“Where will this vessel be in 15 minutes?”** at `http://localhost:8501/`) so it can answer four maritime prediction questions using Dataset 3, MarineCadastre AIS, located in the repository's `data/` directory:

1. **Where is this vessel going?** (destination / next port or arrival zone)
2. **Where will this vessel be in 15 minutes?** (future position, already implemented)
3. **How congested will this port be?** (port congestion proxy forecast)
4. **How busy will this area be?** (vessel traffic density forecast)

Add **AIS data integrity checks** and **missing-transmission detection** as shared supporting capabilities. These are essential to the competition's theme of unintentional data poisoning. If time remains, add a limited anomalous-movement flag built from the same checks. Do not expand into ship activity classification, vessel type classification, or collision-risk forecasting for this iteration.

This document specifies **functionality, modeling, data contracts, experiments, evaluation, and implementation priorities only**. It deliberately contains no UI or visual design guidance.

## Non-negotiable constraints

- **Modify the current repository, do not rewrite it.** Inspect its code, models, data pipeline, caches, pages, tests and dependencies first. Keep the existing 15-minute prediction working, with backward-compatible functions/interfaces where feasible.
- Use actual available data. Before coding, inspect `data/` and existing processed artifacts; detect actual filenames, schema, temporal coverage, vessel types, subsets, and geographic extent. Do not assume an `arrival_port` column, destination labels, real port-capacity measurements, AIS reception time, AIS message source ID, or validated ship activity labels exist.
- Build for **tens of gigabytes of AIS** via streaming / DuckDB / partitioned Parquet / bounded model-training data. Never load the entire raw dataset into pandas or browser state. Reuse the existing scalable ingest pipeline when present.
- Use **subset 4** first; run **subset 3**, then **subset 2**, then **subset 1** only after prior runs complete and the added processing fits the remaining runtime. Log what was attempted, runtime, scale, and skips. Subsets may overlap; verify their relationship and do not treat overlapping subsets as independent evidence.
- Preserve chronological order; do not use future locations, future port arrivals, labels, or simulated defect metadata as prediction-time features. Keep the evaluation reference unchanged across matched comparisons.
- The case requires a **trained or otherwise evaluated AI component**, not just map calculations and hand-written threshold rules. Maintain a meaningful ML prediction task and quantitative comparator for each implemented forecast.
- No invented results, “true destinations,” port congestion levels, sensor sources, confidence scores, performance improvements, or claims of operational safety. Be explicit about proxies, synthetic contamination, abstentions, and unavailable features.
- No unnecessary unrelated changes, external paid services, credentials, or large downloads. Check data redistribution and licensing restrictions before bundling any new external source.

## Scope and definitions

### A. Future vessel location (P0: protect the existing functionality)

**Question:** Where will vessel `mmsi` be at time `t + 15 minutes`?

**Inputs available at decision time:** vessel's timestamped AIS track through `t`, position, speed over ground (SOG), course over ground (COG), heading where valid, vessel type where present, and quality/freshness indicators.

**Output:** predicted latitude/longitude at the 15-minute horizon, prediction timestamp and horizon, quality/abstention status, and provenance/model version. Expose actual future AIS position only for *historical evaluation*, never as a live feature.

**Model:** preserve the current trained model if working. Retain or add a constant-velocity/geodesic extrapolation benchmark. Do not silently change the current target definition or prediction horizon.

**Evaluation:** haversine distance error (median, mean, p90 in km), valid prediction coverage, and deferrals. Match observed future positions within an explicit target tolerance; if no valid target exists, exclude the case from positional-error denominators and report the exclusion count. Evaluate a held-out later period and disclose label noise.

**Acceptance:** current working vessel/time example still runs and produces the same output contract; quality degradation and mitigation can be evaluated on identical reference cases.

### B. Vessel destination (P1: new prediction)

**Question:** What is the vessel's likely next destination, using only observations through `t`?

**Important data limit:** AIS trajectories usually do not supply trustworthy destination labels or confirmed terminal visits. If a real port geometry or approved port reference is present, use it. Otherwise derive **clearly labeled inferred arrival zones** (not named ports) from historical sustained low-speed/dwell clusters. Do not invent port names or claim verified cargo delivery.

**Define a reproducible arrival label:**

- Segment histories by vessel, time gaps, and geographic jumps.
- Identify a later arrival event from the **future portion of historical tracks** (e.g., sustained presence inside a known port/anchorage polygon or derived zone over a specified dwell duration). A low-speed point alone is not a confirmed arrival.
- Set the training label as the next qualifying arrival zone within a defined prediction window; exclude or censor trajectories where follow-up is insufficient. Record the share with no verifiable label.
- Prevent model leakage: zone definitions, preprocessing and label mapping must be fit/defined from training-period data or an external preexisting geography. A later arrival label may be used as `y` in training/evaluation, but never as an input feature at prediction time.

**Model:** start with a fast multiclass model (or candidate-zone ranking model) trained on trajectory-derived features at `t`. Compare to a simple nearest/heading-aligned zone baseline. Do not create an artificial model with only one destination class; if subset 4 lacks enough distinct completed arrival events, report insufficient data and continue to larger subsets rather than fabricate training cases.

**Outputs:** likely destination/arrival zone, predicted class probabilities if calibrated or explicitly called model scores, and an unknown/abstain outcome for insufficient history or low reliability. A prediction may remain “unknown” rather than choose a spurious port.

**Evaluation:** top-1 accuracy, top-3 accuracy where ≥3 classes exist, coverage/unknown rate, class support, and performance on a later holdout period. Split by complete trip/vessel episode and time to avoid cross-split duplicates. Report rare-zone performance, confusion matrix, and potential geographic bias when support permits.

**Acceptance:** at least one genuine held-out labeled arrival event is demonstrated with input cutoff, prediction, reference event, and outcome; if not possible on current data, record the blocker and leave this capability explicitly unavailable rather than faking it.

### C. Port congestion forecast (P1/P2: new prediction with honest proxy)

**Question:** How many vessels will be present in a selected port/anchorage zone during the next forecast window?

**Do not assert actual congestion or wait time** from AIS presence alone. Without berth capacity, berth occupancy, queues, and terminal processing records, AIS can support a **congestion proxy**, not verified congestion. Define the target precisely as, for example, `unique vessels present in the port/anchorage polygon in the next 60-minute interval`, with separate inbound arrivals or dwell duration only if defensible. Name the metric **port traffic/occupancy proxy**, and only refer to “congestion” with that qualification.

**Geography:** use actual polygons/zones found in the repository, an openly licensed external geometry already accessible to the project, or reproducibly defined regional polygons with documented coordinates. If polygons are constructed by the team, call them study zones rather than verified port facility boundaries. Avoid creating port zones from future test labels.

**Features:** prior counts per consistent time bucket, rolling counts and moving averages, observed arrival/departure trends, hour/day, and feed-quality indicators, all calculated strictly from reports available by the forecast origin.

**Target:** unique MMSIs meeting the presence rule in a future fixed bucket. Use one count per MMSI per bucket and explicit gap treatment to avoid AIS reporting frequency inflating vessel counts. Missing reports may make the reference count an underestimate; document it.

**Model:** lightweight count regressor; compare with persistence and seasonal/rolling-average baselines. Train only if multiple chronological buckets and enough history exist.

**Evaluation:** MAE in vessels, RMSE, bias, prediction coverage, and error by zone/time window. Report reference count completeness and separately measure the effect of lost/duplicated AIS inputs.

**Acceptance:** for a data-supported zone, forecast a future count and compare against the held-out observed AIS-derived count. If no viable polygon/history exists, ship the clearly labeled AIS occupancy proxy without claiming it measures delays.

### D. Traffic density forecast (P1: new prediction)

**Question:** How many unique vessels will be active in a specified spatial cell or maritime study area in the next 60 minutes?

**Definition:** use geographic grid cells (e.g., fixed-resolution H3 if already a dependency, otherwise consistent bounded geographic bins) or user-provided polygons. Define traffic density as `unique MMSIs observed in the cell over the time bucket / cell area` when computing a per-area rate; also expose raw count. Avoid latitude/longitude rectangle areas treated as equal when comparing far-apart latitudes. A cell's “observed traffic” is not total unobserved vessel traffic.

**Features/target:** past hourly (or dataset-appropriate) unique-vessel counts, rolling volume, hour/day, geographic area, and feed health. Target the next time bucket, computed only from held-out subsequent AIS reports. Do not use the future bucket in any rolling input feature.

**Model:** a light forecasting regressor against persistence/rolling-average baselines. Reuse general time-series feature generation and count forecasting modules shared with port occupancy, but keep their targets and metrics distinct.

**Evaluation:** MAE, RMSE, bias, coverage, and error over multiple chronological windows. Evaluate dense and sparse cells separately. Compare performance under delayed/duplicated/missing reports, particularly where high-volume low-quality AIS overwhelms representative reports.

**Acceptance:** train/evaluate on at least one nonempty region with multiple time buckets, producing numeric out-of-sample error. Empty cells remain zeros only if confirmed included in the cell/time index; do not treat missing source data as observed zero traffic.

## Shared supporting capabilities

### E. Data-quality checks (P0)

Build or extend a reusable AIS validation layer that produces an **interpretable flag list**, not one unexplained magic quality score:

- Invalid/missing coordinates, out-of-bounds coordinates, malformed dates/identifiers.
- Duplicated positions/reports and unusually high report frequency; differentiate repeated valid broadcasts from exact duplicate records.
- Invalid AIS values and documented sentinels for SOG/COG/heading, with units checked against the actual schema.
- Implausible position/time transitions (distance/time-based inferred speed), impossible timestamp order, and anomalous jumps. Do not automatically erase every fast or unusual legitimate movement.
- Report age at the decision time and whether the last record is sufficiently fresh for the requested forecast.
- Optional movement-anomaly flags using transparent trajectory rules; do not present these as confirmed misconduct or safety incidents.

Outputs per record/track/window: validity/flag codes, actual/synthetic origin, age, counts by failure type, and enough provenance to support audit. Keep original raw records unchanged and write derived validation outputs separately.

### F. Missing-transmission detection (P0/P1)

Detect gaps between consecutive observed reports per vessel with a configurable threshold plus a vessel/region-specific cadence baseline if support permits. Record gap start/end, gap duration, expected-versus-observed cadence estimate, and whether it overlaps the decision time. **Call them observation/reporting gaps**, not proven radio transmission failures: AIS reception coverage, filtering, and ships leaving the study region can produce apparent gaps.

Distinguish a genuine missing observation from a deliberately simulated delay. Where feasible, exclude out-of-region exits from false-positive reporting and test on synthetic withheld records whose ground-truth labels are known.

### G. Human review and abstention (P1, integrate into current workflow)

A flagged prediction must include age, specific quality issues, consequence for each relevant prediction, and an accept/correct/defer action with audit entry. Capture an action timestamp and reviewer-entered correction/reason, and distinguish **model response time** from **measured or explicitly estimated human handling time**. One complete, reproducible flagged-case walkthrough meets the MVP scope; a standalone administrative system is unnecessary.

## Central case-a-thon experiment (P0, applies to all enabled prediction tasks)

**The scientific claim is not that more data is always worse.** Test when late, stale or low-quality inputs alter outputs and whether quality-aware processing improves decisions, including failure/coverage tradeoffs.

Select and document the affected stage. Preferred baseline: **inference-time feed degradation with a frozen trained model**. Do not mix model retraining into the same comparison without a separately identified experiment.

Required experiment matrix for the primary 15-minute position task:

| Condition | Mitigation disabled | Mitigation enabled |
|---|---|---|
| Nominal reference | required | required |
| Stale only | required | required |
| Low quality only | required | required |
| Combined stale + low quality | required | required |

- Run each cell on the **same held-out original cases and targets**, with controlled seeds, model version, and time origin. Never corrupt evaluation reference outcomes.
- **Stale:** replace timely input with the latest genuinely available older report or simulate delay/freeze, retaining original event time and simulated availability time distinctly. Ensure no later row is accessed at the decision time.
- **Low quality:** controlled position jitter, course/speed defects, omissions or duplicates, and a synthetic high-volume low-reliability input segment when source IDs are absent. Log severity, rates and synthetic provenance.
- **Combined:** apply both mechanisms at the same decision points; measure interaction effects.
- **Mitigation:** quality checks, freshness-aware feature gating or weighting, plausible-motion checks, fallback model where justified, or abstention. The mitigated pipeline must not see uncorrupted originals unavailable to the unmitigated one.
- Run at least five real time windows, or explicitly label a five-step simulated progression. Track performance, age, defect prevalence, input volume/source mix, decision coverage, deferral rates, and workload by window.
- Show one concrete case-level causal chain: original report -> injected defect -> prediction change -> held-out target/error -> flags -> corrective action. Document at least one countermeasure failure or tradeoff, such as false flagging a legitimate fast vessel or reducing coverage.
- Prefer 3 fixed random seeds when feasible; otherwise log the tested seed(s) and uncertainty limitation.
- **For regression tasks:** error on answered cases must always be accompanied by percentage answered and deferred; report a coverage-aware comparison so withholding all answers cannot masquerade as improved accuracy.
- **For classification:** class counts, confusion data, abstentions, and metrics on answered cases with overall coverage.

Make the shared experiment framework support each task's own input corruption and target metric. **Do not block release on a full eight-cell grid for every newly added task if the data do not support credible evaluation**; finish the full eight-cell primary position task first, then extend matched comparisons to density, port proxy, and destination as supported. Explicitly mark unsupported cells.

## Data contracts and separation of concerns

Use or adapt existing repository conventions. Keep domain implementations independently testable:

- `AISRecord`: vessel ID, observed event time, availability time if known/simulated, lat/lon, SOG/COG/heading if present, source/provenance identifiers only if truly present, validation flags.
- `PredictionRequest`: task enum (`position_15m`, `destination`, `port_occupancy`, `traffic_density`), decision time, vessel ID or region/zone, horizon, scenario, mitigation config.
- `PredictionResult`: task, prediction or abstention, units, horizon, reference/forecast origin, model version, available input cutoff, validation/age metadata, and measured runtime; task-specific fields are explicit rather than overloaded.
- `EvaluationRecord`: task, subset, independent group/case ID, target time/window, scenario, mitigation, seed, severity, expected/reference value, result, metric values, coverage flag, and status.
- `QualityEvent`: flag code, offending input key/time, observed or synthetic defect, severity where meaningful, and recommended response; not all flags require removing a record.

Do not rewrite the existing functional interfaces solely to match these suggested names. Prefer thin adapters if the repository already defines equivalent structures. Keep predictions, label generation, experiments, storage, and Streamlit entrypoint separate enough to unit-test without starting the server.

For multi-GB support: only inspect and ingest columns needed for each task; materialize date/region partitions in Parquet where worthwhile; aggregate per-vessel/per-zone/per-cell time buckets with DuckDB; persist derived tables and trained artifacts; use data/version/config hashes for invalidation; bound query and model-training memory. Do not create one row per every possible vessel-by-cell-by-minute combination or join every vessel to every other vessel. Avoid recomputing global aggregations on every interactive request.

## Chronological evaluation and data eligibility

- Inspect report cadence and horizon feasibility before choosing nearest-future matching tolerance or hour bucket. Record selected tolerance and target coverage.
- Use training/development/held-out evaluation periods in chronological order. Keep complete voyage episodes or overlapping forecast windows together where possible, and never share identical target events across evaluation and training due to near-duplicate observations.
- For future position, hold out later target reports. For destination, derive arrival outcomes only after the prediction cutoff. For area counts, compute target windows only from later AIS broadcasts. A target can be computed retrospectively, but may not enter the input feature table for its own origin.
- If raw data lack source-received times, `availability_time` is unknown in nominal data; simulate known arrival delays separately and do not present nominal observed event time as verified ingestion time.
- An inferred arrival or congestion proxy remains an inferred outcome. Document the uncertainty and inability to independently validate actual berthing/delivery/delays.
- Report sample sizes, number of independent vessels/voyages/zones/time buckets and time ranges. Bigger row counts are not more independent examples.

## Implementation order

Do not spend time redesigning the existing interface. Keep it functional while extending the underlying capabilities.

1. **Repository and dataset audit.** Locate current 15-minute prediction pipeline and its existing tests; confirm datasets/subsets, performance, and available port geography. Record the exact present behavior before edits.
2. **Shared quality + gap detection.** Implement/test record flags, freshness, report gaps, and provenance. Preserve original data and feed availability modeling.
3. **Regression framework for regions.** Implement traffic density from AIS-derived unique-vessel time buckets, persistence benchmark, and short-horizon learned forecast. Reuse that pipeline for a clearly named port occupancy proxy when valid geographic regions are available.
4. **Destination inference.** Add train-period-defined arrival zones/qualified visit labels and a model only when sufficient distinct arrival outcomes and future follow-up exist. Otherwise produce a documented `insufficient_data` outcome with counts and reasons, not an unreliable pretend classifier.
5. **Integrate with existing 15-minute forecast.** Make new prediction tasks available through the existing Streamlit application without changing unrelated behavior or requiring a new tech stack.
6. **Experiment extension.** Preserve the primary eight-condition position experiment and add meaningful matched stress tests to new tasks in order of evidence/feasibility. Include contamination progression, source volume, coverage and at least one failure analysis.
7. **Reproducibility and scale.** Verify subset 4, then 3, 2, 1 as runtime permits. Measure actual performance and peak memory where feasible. Validate a cold-start run and warm/cached run separately.

If a new feature threatens the existing functional MVP, **defer that feature and leave the present application operational**. Work in incremental commits or logical patches where possible.

## Required deliverables

Update or create the following only where equivalent files do not already exist:

- The functioning Streamlit app at its current launch command/port, with real (or explicitly unavailable) outputs for the four questions.
- Reusable data validation/gap detection, destination, count-forecast, model evaluation, and degradation/mitigation modules.
- `results/metrics.csv` or Parquet: task, subset, scenario, mitigation, evaluation period/window, seed, counts, performance, coverage/deferral, and measured runtime.
- `results/manifest.json`: source/version/rights status, selected files, sizes/row counts, independent case counts, timestamp bounds, missing values and units, subset rules, temporal splits, target labels and proxy definitions, model versions, parameters, random seeds, injected defect log, limitations, and AI/prompt disclosure.
- Reproducible scripts/commands for preparing subsets, training, evaluating, stress testing, and opening the existing app. Update dependency lock/requirements and `README.md` as needed.
- Tests for time leakage, counting unique MMSIs, missing-time-bucket handling, same-case eight-way comparison, controlled defect reproducibility, arrival label cutoffs, abstention accounting, and backward compatibility of 15-minute prediction.
- A brief `docs/CASE_DEMO.md` narrative: each question's user/decision and error consequence; the 15-minute predictor's four-by-two experiments; why added observations can worsen outcomes; mitigation and its failure/coverage tradeoff; one human correction/defer walkthrough; who owns data quality and model performance; pilot/no-go decision criteria. Keep setup and results inspectable for a 10-minute demo.

No fabricated screenshot, video, benchmark, accuracy, destination, or port name. Document any incomplete features plainly.

## Done criteria

1. **Regression preserved:** “Where will this vessel be in 15 minutes?” works with a trained model and a held-out measurable reference.
2. **New predictions:** destination and both count-based forecasting questions either return reproducible, evaluated results **or** explicit evidence-backed unavailability states; do not call an unevaluated heuristic a trained forecast.
3. **Data integrity:** flags and report gaps work on real records and seeded simulated defects; outputs indicate their limits.
4. **Required science:** primary task's four scenarios × mitigation on/off are evaluated using matched cases; temporal progression, quality/age/volume metrics, uncertainty, coverage, and a mitigation tradeoff are reported.
5. **Scope:** no verified-cargo, true-port-delay, collision-risk, or autonomous-navigation claims. Proxy outcomes and synthetic conditions are labeled.
6. **Scale:** bounded memory and disk-backed operations are implemented and documented; scaling to tens of GB is a design objective unless separately benchmarked.
7. **Verification:** run tests and the app; provide exact commands, successful subset(s), row/case counts, recorded metrics, and a precise list of remaining gaps.

## Final response expected from Claude

Report succinctly: existing code preserved, capabilities implemented versus deferred, commands to run, actual subset used, training/evaluation data coverage, model metrics per supported task, main eight-cell experiment results, tests run, measured runtime, and blockers/limits. **Do not claim a feature complete until its pipeline has produced and validated an actual result.**
