"""Leakage, reproducibility, corruption isolation, case matching and deferral accounting."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisdp import corruption as C  # noqa: E402
from aisdp.config import ExperimentConfig  # noqa: E402
from aisdp.experiment import _summ  # noqa: E402
from aisdp.features import build_cases, build_states, is_test_vessel  # noqa: E402
from aisdp.geo import enu_displacement, haversine_m, offset  # noqa: E402

RES = ROOT / "results"


def synthetic_reports(n_vessels=40, hours=3, step=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vessels):
        mmsi = 366000000 + v * 7919
        lat, lon = 30 + rng.random(), -80 + rng.random()
        for k in range(int(hours * 3600 / step)):
            t = 1767225600 + k * step + rng.integers(0, 5)
            rows.append((mmsi, float(t), lat + k * 1e-4, lon, 5.0, 0.0, 0.0, 70))
    return pd.DataFrame(rows, columns=["mmsi", "t", "lat", "lon", "sog", "cog", "heading", "vessel_type"])


@pytest.fixture
def con():
    c = duckdb.connect()
    r = synthetic_reports()
    c.register("rdf", r)
    c.execute("CREATE TABLE r AS SELECT * FROM rdf ORDER BY mmsi, t")
    return c


def test_geo_roundtrip():
    lat2, lon2 = offset(40.0, -70.0, 1000.0, -500.0)
    de, dn = enu_displacement(40.0, -70.0, lat2, lon2)
    assert abs(de - 1000) < 1 and abs(dn + 500) < 1
    assert abs(haversine_m(0, 179.99, 0, -179.99) - 2224) < 5   # antimeridian wrap


def test_split_has_no_leakage(con):
    cfg = ExperimentConfig(n_windows=3, min_history_s=1800)
    cases = build_cases(con, cfg)
    tr, te = cases[cases.split == "train"], cases[cases.split == "test"]
    assert len(tr) and len(te)
    assert set(tr.mmsi).isdisjoint(set(te.mmsi)), "train/test vessels overlap"
    assert tr.target_t.max() < cases.attrs["split_t"] <= te.decision_t.min() - cfg.horizon_s
    assert (cases.target_t > cases.decision_t).all()
    assert (abs(cases.target_t - cases.decision_t - cfg.horizon_s) <= cfg.tol_s).all()
    assert is_test_vessel(te.mmsi.to_numpy(), cfg).all()


def test_states_never_use_future_records(con):
    cfg = ExperimentConfig(n_windows=3, min_history_s=1800)
    te = build_cases(con, cfg).query("split == 'test'")
    d = C.draw(len(te), 0)
    q = C.stale_query_t(te.decision_t.to_numpy(), np.ones(len(te), bool), d, cfg)
    s = build_states(con, te.assign(query_t=q), cfg)
    assert (s.state_t <= q + 1e-9).all()
    assert (s.state_t <= te.decision_t.to_numpy() - cfg.stale_delay_min_s).all()
    assert (s.prev_t.dropna() < s.state_t[s.prev_t.notna()]).all()
    assert (s.mmsi.to_numpy() == te.mmsi.to_numpy()).all()


def test_corruption_isolated_and_reproducible(con):
    cfg = ExperimentConfig(n_windows=3, min_history_s=1800)
    te = build_cases(con, cfg).query("split == 'test'")
    s = build_states(con, te.assign(query_t=te.decision_t), cfg)
    mask = np.arange(len(s)) % 2 == 0
    a = C.corrupt(s, mask, C.draw(len(s), 7))
    b = C.corrupt(s, mask, C.draw(len(s), 7))
    pd.testing.assert_frame_equal(a, b)
    untouched = ~mask
    for col in ["lat", "lon", "sog", "cog", "state_t"]:
        np.testing.assert_array_equal(a.loc[untouched, col].to_numpy(), s.loc[untouched, col].to_numpy())
    assert (a.loc[mask, "source"] == C.SYN_SOURCE).all() and (a.loc[untouched, "source"] == "AIS-original").all()
    assert (s.columns.difference(a.columns)).empty
    assert not s.isin([C.SYN_SOURCE]).any().any(), "input frame mutated"


def test_deferral_accounting():
    g = pd.DataFrame({"e": [1.0, 2.0, 10.0, 20.0]})
    ans = np.array([True, True, False, False])
    r = _summ(g, "e", ans, defer_pen=3.0)
    assert r["coverage"] == 0.5 and r["n_answered"] == 2
    assert r["mean_km"] == 1.5                     # error on answered cases only
    assert r["policy_cost_km"] == (1 + 2 + 3 + 3) / 4  # deferrals are charged, not scored as zero
    all_defer = _summ(g, "e", np.zeros(4, bool), defer_pen=3.0)
    assert np.isnan(all_defer["mean_km"]) and all_defer["policy_cost_km"] == 3.0


@pytest.mark.skipif(not (RES / "cases_default.parquet").exists(), reason="run the pipeline first")
def test_results_use_identical_cases_and_targets():
    c = pd.read_parquet(RES / "cases_default.parquet")
    base = c[c.condition == "nominal"].set_index("case_id")[["mmsi", "target_t", "target_lat", "target_lon"]]
    for cond, g in c.groupby("condition"):
        g = g.set_index("case_id")[base.columns]
        assert g.index.sort_values().equals(base.index.sort_values()), cond
        pd.testing.assert_frame_equal(g.loc[base.index], base)
    m = pd.read_csv(RES / "metrics.csv")
    cells = m[(m.window == "all") & (m.subgroup == "all") & (m.predictor == "hgb")]
    req = {(c_, mi) for c_ in ["nominal", "stale", "low_quality", "combined"] for mi in ["none", "quality_aware"]}
    assert req <= set(zip(cells.condition, cells.mitigation))
    assert (cells.groupby(["condition", "mitigation", "seed"]).n_cases.first().nunique()) == 1
