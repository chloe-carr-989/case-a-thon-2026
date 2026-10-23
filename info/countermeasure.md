# The countermeasure (rubric C4)

Numbers below come from the default run `jan01_04`: trained on Jan 1–3 and tested on Jan 4. The degradation is simulated, seeded and labelled. These are the same figures as the Model Performance → Countermeasure tab.

## What it's fighting

The feed is "poisoned" by accident, in two ways:

| Failure | What happens | Simulated how |
|---|---|---|
| **Stale** | The feed goes quiet or arrives late, so the system "forgets" the latest real movement and trusts an old position as if it were current | The newest 5–45 min of a vessel's reports are not delivered yet |
| **Unreliable** | A second, low-quality source (`SYN-FASTFEED`) mixes junk into the real reports | One extra report per two real ones: position jitter, 25% gross outliers 3–15 km off, wrong or missing speed and course |
| **Both** | The two at once | — |

The share of degraded vessels rises through the test day, from 10% in window 1 to 60% in window 5.

Without the fix, the system takes every report at face value and assumes the newest one is current.

## What the countermeasure does

It's a set of data-quality checks between the feed and the model. **Every threshold was set before testing and never tuned on the test day.**

1. **Drop impossible data.**
   - Speeds over 40 kn, and speed or course values that are missing or sentinels, are thrown out.
   - Position spikes are dropped. A spike is a report that implies more than 60 kn to get there and more than 60 kn to leave, and is over 1 km away.
   - For the ETA, a jump is impossible if the implied speed is more than twice the reported speed + 2.5 m/s. Then the system falls back to the last credible position.
2. **Repair what it can.** If the reported speed or course disagrees with how the vessel is actually moving across its last two positions, the system uses the observed track instead.
3. **Move old positions forward.** If the newest report is 2–20 min old, the position is projected forward along its course to "now" instead of being treated as current.
4. **Admit it doesn't know.** If the newest report is **over 20 min old**, the system won't guess. It **withholds the answer and sends it to a human**:
   - Government mode shows "no verdict" and reports it as a data problem.
   - Commercial mode holds the ETA.
5. **Use robust statistics (Government).** The lane-deviation score uses medians instead of means, so a few junk points can't drag a vessel "off its lane".

All of it lives in `website/src/aisdp/mitigation.py` (`sanitize_state`, `mitigated_predict`) and `anomaly.py` (`aggregate`, `label`). The thresholds are in `config.py`.

## Results

### Government: lane anomaly alerts

| Feed | False alerts per 1000, off → on | Real alerts missed, off → on | Vessels given a verdict, off → on |
|---|---|---|---|
| Clean | 0 → 0 | 0% → 0% | 98% → 95% |
| Stale | 0.05 → 0.04 | **33% → 34%** | 98% → **75%** |
| Unreliable | **46.1 → 7.4** | 0.4% → 0.9% | 98% → 95% |
| Both | **31.2 → 4.7** | 33% → 34% | 98% → 75% |

### Commercial: ETA

| Feed | Mean error on ETAs given, off → on | ETAs held for review | Error with held ETAs charged as misses, off → on |
|---|---|---|---|
| Clean | 64 → 63 min | 2% | 64 → 65 min |
| Stale | 66 → 65 min | **23%** | 67 → **89 min** |
| Unreliable | 66 → 65 min | 2% | 66 → 67 min |
| Both | 68 → 66 min | 23% | 68 → 90 min |

## What it fixes

- **It removes about 84–85% of the false alerts from unreliable data.** Junk positions no longer make vessels look off-lane.
- **It holds up on an unseen condition.** Window 5, the latest and most degraded period, was held out when the thresholds were set. False alerts there still drop from 80.5 to 12.1 per 1000 checks (−85%).
- **It costs almost nothing on clean data.** The alert rate is unchanged (2.8% → 2.7%). About 3% of vessels lose their verdict, because their own reports really are over 20 min apart.

## Where it fails (we say this openly)

- **It can't fix staleness.** Missed real alerts stay at about 34%. You can't invent reports that never arrived. What it *does* do is replace a falsely clean map with "no verdict", so the officer knows the picture is incomplete. By window 5 of "both", only 57% of vessels get a verdict.
- **For the ETA, the tradeoff goes the wrong way.** It holds 23% of ETAs on a stale feed, but most of them were still usable: a 20-min-old report barely matters for a 0–6 h arrival estimate. Counting held ETAs as misses, error gets worse (67 → 89 min). **Fix for next time:** scale the age limit with the forecast horizon.
- **It suppresses some real deviations.** Injected route drifts that we know are real were caught 53% of the time before the fix and 49% after on the unreliable feed (49% → 48% on the clean feed).
- **Rejecting everything would also give zero false alerts.** That's why coverage, meaning how many vessels get an answer, is always reported next to error.

## One-line pitch

> The countermeasure catches the junk from an unreliable source: about 85% fewer false alerts, including on a held-out window it was never tuned on. It can't recover data that never arrived, but it stops pretending: stale vessels get "no verdict" and go to a human, instead of looking safely normal.
