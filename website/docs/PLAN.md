# MVP plan (written at minute ~6)

**Data found.** Dataset 3 = MarineCadastre AIS CSV for a single day, 2026-01-01, at
`data/dataset3/ais-2026-01-01.csv/`. The organizer subsets are **nested head-prefixes**
of the full file (verified byte-for-byte with `cmp`):

| file | bytes | rows | time span |
|---|---|---|---|
| jan_01_subset_4.csv | 881 | 7 | 00:00:01–00:00:04 |
| jan_01_subset_3.csv | 21,440 | 200 | seconds |
| jan_01_subset_2.csv | 60,462,921 | 589,669 | 00:00–01:57 |
| jan_01_subset_1.csv | 433,045,688 | ~4.23 M | 00:00–~13:43 |
| ais-2026-01-01.csv | 868,700,202 | ~8.4 M | full day |

Because they are prefixes, subsets are never combined (no double counting).
Subsets 4 and 3 cannot yield 15-minute target pairs; they are attempted and logged as
"insufficient target pairs".

**Decision.** Harbor/logistics analyst; predict vessel position 15 min ahead (±2 min
target tolerance) and decide trust vs. review.

**AI component.** HistGradientBoostingRegressor predicting mean east/north velocity
(m/s) over the horizon, from features available at decision time. Constant-velocity
dead reckoning is a non-ML benchmark only.

**Failure stage.** Inference-time feed degradation; model trained once on nominal data
and frozen (never retrained during the progression).

**Split.** Vessel-disjoint (MMSI hash) and chronological (train targets end before the
split time; test decisions start one horizon after it).

**Stack.** DuckDB (CSV -> hour-partitioned Parquet, ASOF joins) + pandas/sklearn +
Streamlit/Plotly (offline `scatter_geo`, no tile server).
