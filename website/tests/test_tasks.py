"""Expansion tasks: unique-MMSI counting, missing buckets, time leakage, defect reproducibility,
arrival-label cutoffs, abstention accounting, integrity flags/gaps and 15-minute backward compatibility."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisdp import counts as K  # noqa: E402
from aisdp import destination as Dn  # noqa: E402
from aisdp import quality as Q  # noqa: E402
from aisdp.config import CountConfig, DestinationConfig, QualityConfig  # noqa: E402

RES = ROOT / "results"
CC = CountConfig()


def presence(rows):
    return pd.DataFrame(rows, columns=["unit", "b", "mmsi", "n_msgs"])


# ------------------------------------------------------------------------------------- counts
def test_unique_mmsi_counted_once_per_window():
    # vessel 1 reports in all 4 buckets of the hour (many messages), vessel 2 once: 2 vessels, not 5 rows / 42 msgs
    ub = presence([(0, 10, 1, 10), (0, 11, 1, 10), (0, 12, 1, 10), (0, 13, 1, 10), (0, 12, 2, 2)])
    f = K.window_features(K.Presence(ub, 1), [14], [0], CC)
    assert f.h0.iloc[0] == 2 and f.msgs_h0.iloc[0] == 42 and f.q0.iloc[0] == 1
    # a vessel in two units counts once in each unit
    ub2 = presence([(0, 10, 1, 1), (1, 11, 1, 1)])
    f2 = K.window_features(K.Presence(ub2, 2), [14], [0], CC)
    assert f2.h0.tolist() == [1, 1]


def test_window_counts_match_bruteforce():
    rng = np.random.default_rng(0)
    ub = presence([(u, b, m, 1) for u in range(3) for b in range(0, 30) for m in range(8) if rng.random() < 0.3])
    origins = np.arange(12, 26)
    f = K.window_features(K.Presence(ub, 3), origins, np.zeros(len(origins), int), CC)
    for _, r in f.iterrows():
        w = ub[(ub.unit == r.unit) & (ub.b >= r.o - 4) & (ub.b < r.o)]
        h3 = ub[(ub.unit == r.unit) & (ub.b >= r.o - 12) & (ub.b < r.o)]
        assert r.h0 == w.mmsi.nunique() and r.h3 == h3.mmsi.nunique()


def test_missing_buckets_are_zero_rows_and_targets_never_leak_into_features():
    ub = presence([(0, 5, 1, 1), (0, 20, 2, 1), (0, 21, 3, 1)])
    P = K.Presence(ub, 2)                       # unit 1 never reports -> confirmed zero, still in the index
    f = K.window_features(P, [20], [0], CC)
    assert len(f) == 2 and f.set_index("unit").h0.to_dict() == {0: 0, 1: 0}   # buckets 20, 21 are in the future
    y = K.targets(P, [20], CC)
    assert y.set_index("unit").y.to_dict() == {0: 2, 1: 0}
    # a late feed hides the most recent buckets from the features
    ub2 = presence([(0, 18, 1, 1), (0, 19, 2, 1)])
    late = K.window_features(K.Presence(ub2, 1), [20], [2], CC)
    assert late.h0.iloc[0] == 0 and late.h0_shift.iloc[0] == 0
    timely = K.window_features(K.Presence(ub2, 1), [20], [0], CC)
    assert timely.h0.iloc[0] == 2


def test_count_corruption_reproducible_and_isolated():
    rng = np.random.default_rng(1)
    ub = presence([(u, b, 300000000 + m, 3) for u in range(4) for b in range(40) for m in range(5) if rng.random() < .5])
    snapshot = ub.copy()
    a, la = K.corrupt_presence(ub, 20, np.linspace(24, 40, 6), seed=3, cc=CC)
    b, lb = K.corrupt_presence(ub, 20, np.linspace(24, 40, 6), seed=3, cc=CC)
    pd.testing.assert_frame_equal(a, b)
    assert la == lb
    pd.testing.assert_frame_equal(ub, snapshot)             # original input untouched
    before = a[a.b < 20].drop(columns=["source", "ghost_kind"]).sort_values(["unit", "b", "mmsi"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(before, ub[ub.b < 20].sort_values(["unit", "b", "mmsi"]).reset_index(drop=True),
                                  check_dtype=False)        # training-period inputs are never corrupted
    assert (a[a.source == K.SYN_SOURCE].b >= 20).all() and la["ghost_rows"] > 0


def test_count_mitigation_filters_invalid_and_transients_causally():
    ub = presence([(0, 10, 150000000, 5),                  # invalid MID
                   (0, 10, 300000001, 1),                  # one-message transient
                   (0, 10, 300000002, 1), (0, 11, 300000002, 1),   # second message arrives in bucket 11
                   (0, 10, 300000003, 4)])
    m = K.mitigate_presence(ub, CC)
    kept = set(zip(m.mmsi, m.b))
    assert (150000000, 10) not in kept and (300000001, 10) not in kept
    assert (300000002, 10) not in kept, "a later message must not rescue an earlier bucket (causality)"
    assert (300000002, 11) in kept and (300000003, 10) in kept


def test_count_deferral_accounting():
    r = K._metrics(y=[1, 2, 3, 4], p=[1, 2, np.nan, np.nan], ans=[True, True, False, False], pen=5.0)
    assert r["coverage"] == 0.5 and r["mae"] == 0 and r["policy_mae"] == (0 + 0 + 5 + 5) / 4


def test_delay_detection():
    vol = pd.Series({b: 100 for b in range(0, 30)})
    d = K.detect_delay(vol, [20, 20, 20], [0, 2, 3], CC)
    assert d.tolist() == [0, 2, 3]


# --------------------------------------------------------------------------------- destination
def test_arrival_labels_respect_cutoffs():
    dc = DestinationConfig()
    zones = pd.DataFrame({"zone_id": ["ZA"], "zi": [600], "zj": [-1800]})          # cell of lat 30.0x, lon -90.0x
    dw = pd.DataFrame({"mmsi": [1, 1, 2, 3], "start_t": [500., 2000., 3000., 900.],
                       "confirm_t": [2300., 3800., 4800., 2700.], "lat": [30.01] * 4, "lon": [-89.99] * 4})
    dec = pd.DataFrame({"mmsi": [1, 2, 3, 4], "decision_t": [1000., 1000., 1000., 1000.]})
    lab = Dn.label_decisions(dec, dw, zones, dc, data_end=4000.).set_index("mmsi")
    assert lab.at[1, "status"] == "arrival_zone" and lab.at[1, "arrival_t"] == 2000.   # dwell before decision ignored
    assert lab.at[2, "status"] == "censored"     # its dwell is only confirmed after data_end
    assert lab.at[3, "status"] == "censored"     # dwell started before the decision: not a future arrival
    assert lab.at[4, "status"] == "censored" and pd.isna(lab.at[4, "true_zone"])
    long = Dn.label_decisions(dec, dw, zones, dc, data_end=1e6).set_index("mmsi")
    assert long.at[4, "status"] == "no_arrival_in_window"
    assert (lab.arrival_t.dropna() >= lab.loc[lab.arrival_t.notna(), "decision_t"]).all()


def test_destination_accuracy_counts_abstentions():
    g = pd.DataFrame({"true_zone": ["A", "B", "C", "D"], "x_pred1": ["A", "X", None, "D"], "x_pred2": [None, "B", None, None],
                      "x_pred3": [None] * 4, "ans": [True, True, False, False]})
    r = Dn._acc(g, "x", "ans")
    assert r["coverage"] == 0.5 and r["top1_acc"] == 0.5 and r["top3_acc"] == 1.0
    assert r["top1_over_all_cases"] == 0.25      # an abstention is never counted as a hit


def test_zones_fit_only_on_training_period():
    from aisdp.zones import fit_zones
    dw = pd.DataFrame({"mmsi": [1, 2, 3, 4, 5, 6], "end_t": [10, 10, 10, 99, 99, 99], "lat": [30.01] * 3 + [40.01] * 3,
                       "lon": [-90.01] * 3 + [-70.01] * 3, "arrival": [True] * 6})
    z = fit_zones(dw, split_t=50, deg=0.05, min_vessels=3)
    assert len(z) == 1 and z.lat.iloc[0] < 31          # the zone that only appears after split_t is not created


# ----------------------------------------------------------------------------------- quality
def test_flags_and_gaps_on_synthetic_records(tmp_path):
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    rows = [(366000001, t0 + pd.Timedelta(minutes=k), 30.0 + k * 1e-3, -90.0, 10.0, 90.0, 90.0) for k in range(20)]
    rows += [(366000001, t0 + pd.Timedelta(minutes=60), 30.1, -90.0, 10.0, 90.0, 90.0)]       # 41-min gap
    rows += [rows[3]]                                                                        # exact duplicate
    rows += [(366000002, t0, 31.0, -91.0, 102.3, 360.0, 511.0),                              # sentinels
             (366000002, t0 + pd.Timedelta(minutes=1), 32.0, -91.0, 5.0, 10.0, 10.0),        # 111 km in 1 min
             (970000001, t0, 31.0, -91.0, 0.0, 0.0, 0.0)]                                    # AIS-SART MID
    df = pd.DataFrame(rows, columns=["mmsi", "ts", "lat", "lon", "sog", "cog", "heading"])
    df["vessel_type"], df["status"], df["day"], df["hr"] = 70, 0, "2026-01-01", 0
    d = tmp_path / "pq"
    d.mkdir()
    df.to_parquet(d / "x.parquet")
    con = duckdb.connect()
    info = {"parquet_dir": str(d)}
    qc = QualityConfig()
    st = Q.flag_records(con, info, qc)
    bc = st["by_code"]
    assert bc["exact_duplicate"] == 1 and bc["sog_sentinel"] == 1 and bc["cog_sentinel"] == 1
    assert bc["heading_sentinel"] == 1 and bc["implausible_jump"] == 1 and bc["mmsi_non_vessel_mid"] == 1
    assert st["rows"] == len(df)                                  # raw records are flagged, never dropped
    gaps, gs = Q.detect_gaps(con, info, qc)
    g = gaps[gaps.mmsi == 366000001]
    assert len(g) == 1 and abs(g.gap_s.iloc[0] - 41 * 60) < 1
    assert Q.freshness([60, 600, 5000, -5], 900).tolist() == ["fresh", "stale", "too_old", "future_timestamp"]


# ------------------------------------------------------------------------- backward compatibility
@pytest.mark.skipif(not (RES / "cases_default.parquet").exists(), reason="run the pipeline first")
def test_position_output_contract_unchanged():
    c = pd.read_parquet(RES / "cases_default.parquet")
    required = ["condition", "seed", "case_id", "mmsi", "window", "decision_t", "target_t", "target_lat", "target_lon",
                "state_t", "age_s", "in_lat", "in_lon", "un_lat", "un_lon", "err_un_km", "err_cv_km", "mi_lat", "mi_lon",
                "err_mi_km", "answered", "action", "quality", "flags"]
    assert set(required) <= set(c.columns)
    m = pd.read_csv(RES / "metrics.csv", low_memory=False)
    pos = m[m.task.fillna("position_15m") == "position_15m"] if "task" in m else m
    assert {"nominal", "stale", "low_quality", "combined"} <= set(pos.condition)
    assert {"mean_km", "median_km", "p90_km", "coverage", "policy_cost_km"} <= set(pos.columns)


@pytest.mark.skipif(not (RES / "destination_cases_seed0.parquet").exists(), reason="run the pipeline first")
def test_contract_adapters_on_stored_results():
    from aisdp import contracts as K2
    c = pd.read_parquet(RES / "cases_default.parquet").iloc[0]
    r = K2.position_result(c, mitigated=False)
    assert r.task == K2.Task.POSITION_15M and r.horizon_s == 900 and {"lat", "lon"} <= set(r.value)
    assert r.input_cutoff_t <= r.decision_t
    d = pd.read_parquet(RES / "destination_cases_seed0.parquet")
    deferred = d[~d.mi_answered].iloc[0]
    rd = K2.destination_result(deferred, mitigated=True)
    assert not rd.answered and rd.value == {"zones": ["unknown"]}
    k = pd.read_parquet(RES / "density_cases_seed0.parquet").iloc[0]
    rk = K2.count_result(k, K2.Task.TRAFFIC_DENSITY, "C0", 700.0, mitigated=False)
    assert rk.to_dict()["task"] == "traffic_density" and rk.horizon_s == 3600
