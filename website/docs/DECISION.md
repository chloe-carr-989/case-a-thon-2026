# Decision, governance and pilot proposal

## Roles (roles only, no named people)

| role | responsibility |
|---|---|
| Harbor-traffic analyst (user) | Uses 15-min position estimates for situational awareness. Does not use them for navigation or collision avoidance. |
| Duty data-quality reviewer | Works the review queue (deferred and `implausible_jump` / `too_old` cases). Accepts, corrects, or escalates. |
| Data-quality owner (AIS feed steward) | Owns source onboarding, source labels, feed latency monitoring, and the defect log. |
| Model owner (ML engineer) | Owns model performance, countermeasure thresholds, retraining, and the metrics in `results/`. |
| Operations supervisor (approver) | Approves pilot start, threshold changes, and go/no-go at checkpoints. |

## When the system must defer

The system defers when any of these hold: the report is older than 20 min; there is an implausible jump with no credible fallback fix; SOG/COG are missing and there is no recent track; or the quality score is below 0.35.
A deferred case shows its evidence in the review tab (age, flags, real track) and is never scored as zero error.

## Logging and approvals

* Every operator action goes to `results/review_log.csv`. It records the UTC time, case, system action, quality, operator action, any correction, the reason, the reviewer role, and the measured review seconds.
  In a pilot, this would be append-only storage with access control.
* Threshold or model changes need a versioned config (`results/manifest.json` → `config`). They also need a rerun of all eight
  cells plus the holdout on the frozen evaluation set, and sign-off from the operations supervisor.
* Feed incidents (latency spikes, a new source, a schema change) are logged by the data-quality owner separately from injected test defects.

## Stop / no-go criteria

* Combined-condition policy cost of the mitigated pipeline is not at least 50% below the unmitigated one on the pilot week.
* Deferral rate is above 30% for two consecutive days, which would overload the review queue.
* More than 10% of fast (≥20 kn) unmodified cases are flagged, so valid rare movements get suppressed. The current value is 12/116 ≈ 10%, which is at the limit.
* Any evidence that operators use estimates for navigation or collision decisions.
* The holdout result shows the countermeasure can make an unseen defect worse for moving vessels. A new defect type found in the pilot
  triggers the pause-and-review checkpoint.

## Bounded pilot proposal

* **Scope:** shadow mode only (estimates shown beside the existing display, with no automated action). One port region
  (e.g. `--region us_east` restricted to one harbor), 4 weeks, historical replay first, then live read-only feed.
* **Measure:** the same metrics schema (`results/metrics.csv`) per day, the review-queue size, and median review seconds.
  Also record how often operators disagree with the system action.
* **Decision checkpoint (end of week 2):** the model owner and the data-quality owner present the metrics to the operations supervisor.
  Options are stop, iterate (adjust thresholds or add source IDs), or continue to week 4. At the end of week 4 the options are a go/no-go for a larger
  multi-region evaluation, or stop.

## What was learned

A frozen ML model can be *more* fragile than a physics baseline when its inputs degrade. Track-derived features
amplify single bad fixes. Gating on report age and track consistency recovers most accuracy, at a visible
coverage cost. It does not cover every new defect, so human review and monitoring remain necessary.
This is not a deployment authorization. NIST AI RMF was used only as an informal organizing reference.
