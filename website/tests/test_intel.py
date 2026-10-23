"""Sea lanes, anomaly windows and ETA: causality, routing along observed traffic, data-quality handling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aisdp import anomaly as A  # noqa: E402
from aisdp.config import AnomalyConfig, LaneConfig  # noqa: E402
from aisdp.geo import haversine_m  # noqa: E402
from aisdp.lanes import LaneModel, cell_key, direction_support  # noqa: E402


def l_shaped_lanes() -> LaneModel:
    """Pooled lane cells along an L (east, then north): the only observed way between its two ends."""
    deg = 0.01
    ij = [(0, j) for j in range(0, 40)] + [(i, 39) for i in range(1, 40)]
    cells = pd.DataFrame({"g": "commercial", "res": "fine", "i": [a for a, _ in ij], "j": [b for _, b in ij]})
    cells["lat"], cells["lon"] = (cells.i + 0.5) * deg, (cells.j + 0.5) * deg
    for c in ["n_vessels", "n_vessels_any", "n_block_vessels"]:
        cells[c] = 5
    cells[["n_all", "n_moving", "stop_share", "sog_med", "sog_q25", "sog_q75"]] = [50, 40, 0.0, 10.0, 8.0, 12.0]
    for k in range(8):
        cells[f"s{k}"] = 0
    cells["s2"] = 40                      # all training traffic heads east (90 deg)
    cells["lane"] = True
    nodes = cells[["i", "j", "lat", "lon", "n_vessels"]].reset_index(drop=True)
    nk = pd.Series(np.arange(len(nodes)), index=cell_key(nodes.i, nodes.j))
    a, b = [], []
    for n, (i, j) in enumerate(zip(nodes.i, nodes.j)):
        for di, dj in [(0, 1), (1, 0)]:
            m = nk.get(cell_key([i + di], [j + dj])[0])
            if m is not None:
                a.append(n), b.append(int(m))
    a, b = np.array(a), np.array(b)
    edges = pd.DataFrame({"a": a, "b": b, "km": haversine_m(nodes.lat[a].to_numpy(), nodes.lon[a].to_numpy(),
                                                            nodes.lat[b].to_numpy(), nodes.lon[b].to_numpy()) / 1000})
    return LaneModel(cells, nodes, edges, LaneConfig())


def test_route_follows_lanes_not_straight_line():
    M = l_shaped_lanes()
    start, end = (0.005, 0.005), (0.395, 0.395)
    route = M.route_km([start[0]], [start[1]], [end[0]], [end[1]])[0]
    straight = haversine_m(*start, *end) / 1000
    assert route > 1.3 * straight                     # along both legs of the L (~1.41x the diagonal)
    p = M.path(*start, *end)
    on_l = (np.abs(p.lat - 0.005) < 0.011) | (np.abs(p.lon - 0.395) < 0.011)
    assert len(p) > 40 and on_l.all()                 # every point of the expected route lies on the L


def test_off_lane_distance_and_direction():
    M = l_shaped_lanes()
    d = M.lane_distance_km([0.005, 0.105], [0.2, 0.2], np.array(["commercial", "commercial"], dtype=object))
    assert d[0] < 1 and 9 < d[1] < 12
    st = M.cell_stats([0.005, 0.005], [0.2, 0.2], np.array(["commercial", "commercial"], dtype=object))
    sup = direction_support(st, np.array([90.0, 270.0]))
    assert sup[0] > 0.9 and sup[1] == 0            # westbound on an eastbound-only lane


def test_windows_never_contain_future_fixes():
    ac = AnomalyConfig()
    t = np.arange(0, 6 * 3600, 60, dtype=float) + 7
    idx, T = A.expand(t, ac, 3600, 6 * 3600)
    assert (t[idx] <= T).all() and (t[idx] > T - ac.window_s).all()
    assert set(np.unique(T)) <= set(np.arange(3600, 6 * 3600 + 1, ac.step_s, dtype=float))


def test_spike_flags_only_the_outlier():
    ac = AnomalyConfig()
    key = np.zeros(5, np.int64)
    t = np.array([0, 60, 120, 180, 240], float)
    lat = np.array([0.0, 0.002, 0.10, 0.006, 0.008])        # third fix jumps ~10 km and back
    lon = np.zeros(5)
    s = A._spikes(key, t, lat, lon, ac)
    assert s.tolist() == [False, False, True, False, False]


def test_stored_results_contracts():
    res = ROOT / "results"
    if not (res / "anomaly_windows.parquet").exists():
        return
    W = pd.read_parquet(res / "anomaly_windows.parquet", filters=[("condition", "==", "nominal")])
    assert set(W.label.astype(str)) <= {"normal", "unusual", "high", "no_verdict", "no_data", "no_history"}
    assert (W.last_t.dropna() <= W["T"][W.last_t.notna()]).all()     # score at T uses only fixes up to T
    naive = W[W.mitigation == "none"]
    assert not (naive.label == "no_verdict").any()                   # only the countermeasure withholds verdicts
    d = pd.read_parquet(res / "destination_cases_seed0.parquet")
    if "un_eta_h" in d:
        a = d[d.mi_eta_answered]
        assert (a.mi_eta_lo_h <= a.mi_eta_h + 1e-6).all() and (a.mi_eta_h <= a.mi_eta_hi_h + 1e-6).all()
        assert not d[~d.mi_answered].mi_eta_answered.any()            # deferred destination -> no ETA
