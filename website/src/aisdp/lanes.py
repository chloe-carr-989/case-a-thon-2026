"""Learned sea lanes: where vessels of each category normally travel, in which direction and how fast.

Model (lightweight spatial density estimation on trajectories, no route assumed):
* One fix per vessel per minute (`vm`), pleasure craft excluded.
* Training data only: reports before split_t from vessels outside the test-vessel hash, so lanes never
  see a test vessel or a test-period report.
* Per category group and 0.01 deg cell: distinct moving vessels, vessel-minutes, course histogram
  (8 x 45 deg sectors), speed quartiles of moving traffic, share of stopped reports. A *lane cell* has
  >= lane_min_vessels distinct moving vessels. Cargo, tanker, passenger and tug/service vessels get their
  own lanes; all other non-pleasure categories (and fishing) are compared with the pooled commercial lanes.
* Routing graph: pooled commercial lane cells are nodes; edges join 8-neighbour cells and any two lane
  cells that one vessel actually moved between within 15 min (<= 15 km). Shortest paths therefore follow
  observed traffic, not straight lines, and an expected route only exists where traffic was observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig, LaneConfig
from .features import vessel_key_sql
from .geo import R_EARTH, haversine_m
from .ingest import parquet_glob
from .zones import VESSEL_WHERE, clean_sog_sql

CATEGORY_SQL = """CASE
  WHEN {c} BETWEEN 70 AND 79 THEN 'cargo'
  WHEN {c} BETWEEN 80 AND 89 THEN 'tanker'
  WHEN {c} BETWEEN 60 AND 69 THEN 'passenger'
  WHEN {c} IN (35, 55) THEN 'government'
  WHEN {c} IN (31, 32, 33, 34, 50, 51, 52, 53, 54, 56, 57, 58, 59) THEN 'tug_service'
  WHEN {c} = 30 THEN 'fishing'
  WHEN {c} IN (36, 37) THEN 'pleasure'
  WHEN {c} IS NULL OR {c} = 0 THEN 'unclassified'
  ELSE 'other' END"""
CATEGORY_LABEL = {"cargo": "Cargo", "tanker": "Tanker", "passenger": "Passenger", "tug_service": "Tug / service",
                  "government": "Government / law enf.", "fishing": "Fishing", "unclassified": "Unclassified",
                  "other": "Other", "pleasure": "Pleasure"}
POOLED = "commercial"
SECTORS = [f"s{k}" for k in range(8)]


class InsufficientLanes(RuntimeError):
    pass


def cell_key(i, j) -> np.ndarray:
    return (np.asarray(i, np.int64) + 20_000) * 100_000 + (np.asarray(j, np.int64) + 50_000)


def lane_group(category) -> np.ndarray:
    c = np.asarray(category, dtype=object)
    return np.where(np.isin(c, list(LaneConfig().groups)), c, POOLED)


def build_vessel_minutes(con, info: dict, table: str = "vm") -> int:
    """First fix per vessel per UTC minute plus a per-vessel category table `vcat` (modal AIS type)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE vcat AS
        SELECT mmsi, mode(vessel_type) AS vessel_type, {CATEGORY_SQL.format(c='mode(vessel_type)')} AS category
        FROM read_parquet('{parquet_glob(info)}', hive_partitioning=true) WHERE {VESSEL_WHERE} GROUP BY mmsi
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT p.mmsi, floor(epoch(p.ts) / 60)::BIGINT AS m, min(epoch(p.ts))::DOUBLE AS t,
               arg_min(p.lat, p.ts) AS lat, arg_min(p.lon, p.ts) AS lon,
               arg_min({clean_sog_sql('p.sog')}, p.ts) AS sog,
               arg_min(CASE WHEN p.cog >= 360 OR p.cog < 0 THEN NULL ELSE p.cog END, p.ts) AS cog,
               any_value(c.category) AS category
        FROM read_parquet('{parquet_glob(info)}', hive_partitioning=true) p JOIN vcat c USING (mmsi)
        WHERE {VESSEL_WHERE}
          AND c.category != 'pleasure'
        GROUP BY p.mmsi, 2
    """)
    return con.execute(f"SELECT count(*) FROM {table}").fetchall()[0][0]


def _cell_stats(con, src: str, deg: float, lc: LaneConfig) -> pd.DataFrame:
    mv = f"sog >= {lc.moving_min_sog_kn} AND cog IS NOT NULL"
    sec = ", ".join(f"count(*) FILTER ({mv} AND floor(cog / 45) = {k}) AS s{k}" for k in range(8))
    return con.execute(f"""
        SELECT g, floor(lat / {deg})::BIGINT AS i, floor(lon / {deg})::BIGINT AS j,
               avg(lat) AS lat, avg(lon) AS lon,
               count(DISTINCT mmsi) AS n_vessels_any, count(DISTINCT mmsi) FILTER ({mv}) AS n_vessels,
               count(*) AS n_all, count(*) FILTER ({mv}) AS n_moving,
               avg((sog < {lc.stop_max_sog_kn})::INT) AS stop_share,
               quantile_cont(sog, 0.5) FILTER ({mv}) AS sog_med,
               quantile_cont(sog, 0.25) FILTER ({mv}) AS sog_q25,
               quantile_cont(sog, 0.75) FILTER ({mv}) AS sog_q75, {sec}
        FROM {src} GROUP BY 1, 2, 3
    """).df()


def learn(con, info: dict, split_t: float, ecfg: ExperimentConfig, lc: LaneConfig, log=print) -> dict:
    """Fit lanes on training-period, non-test-vessel minutes. Requires `vm` (build_vessel_minutes)."""
    test_sql = f"({vessel_key_sql('mmsi')} % {ecfg.test_vessel_mod}) < {ecfg.test_vessel_cut}"
    groups = ", ".join(f"'{g}'" for g in lc.groups)
    con.execute(f"""
        CREATE OR REPLACE TABLE lane_pts AS
        SELECT * FROM vm WHERE t < {split_t} AND NOT {test_sql}
    """)
    n_pts, n_v = con.execute("SELECT count(*), count(DISTINCT mmsi) FROM lane_pts").fetchall()[0]
    if n_pts < 5000 or n_v < 30:
        raise InsufficientLanes(f"only {n_pts} training vessel-minutes from {n_v} vessels; need >= 5000 / >= 30")
    src = f"""(SELECT *, category AS g FROM lane_pts WHERE category IN ({groups})
              UNION ALL SELECT *, '{POOLED}' AS g FROM lane_pts WHERE category != 'fishing')"""
    fine = _cell_stats(con, src, lc.cell_deg, lc).assign(res="fine")
    coarse = _cell_stats(con, src, lc.coarse_deg, lc).assign(res="coarse")
    mv = f"sog >= {lc.moving_min_sog_kn} AND cog IS NOT NULL"
    blk = con.execute(f"""
        SELECT g, floor(lat / {lc.block_deg})::BIGINT AS bi, floor(lon / {lc.block_deg})::BIGINT AS bj,
               count(DISTINCT mmsi) AS n_block FROM {src} WHERE {mv} GROUP BY 1, 2, 3""").df()
    bi = np.floor((fine.i.to_numpy() + 0.5) * lc.cell_deg / lc.block_deg).astype(np.int64)
    bj = np.floor((fine.j.to_numpy() + 0.5) * lc.cell_deg / lc.block_deg).astype(np.int64)
    bk = pd.Series(blk.n_block.to_numpy(), index=pd.MultiIndex.from_arrays([blk.g, cell_key(blk.bi, blk.bj)]))
    fine["n_block_vessels"] = bk.reindex(pd.MultiIndex.from_arrays([fine.g, cell_key(bi, bj)])).fillna(0).to_numpy()
    coarse["n_block_vessels"] = coarse.n_vessels
    cells = pd.concat([fine, coarse], ignore_index=True)
    cells["lane"] = (cells.n_vessels >= 1) & (cells.n_block_vessels >= lc.lane_min_vessels)
    tot = cells[SECTORS].sum(axis=1).to_numpy(float)
    axial = cells[SECTORS[:4]].to_numpy(float) + cells[SECTORS[4:]].to_numpy(float)
    cells["axis_deg"] = np.argmax(axial, axis=1) * 45.0 + 22.5           # dominant travel axis (0-180)
    cells["axis_share"] = np.where(tot > 0, axial.max(axis=1) / np.maximum(tot, 1), np.nan)
    for c in SECTORS + ["n_vessels_any", "n_vessels", "n_all", "n_moving", "n_block_vessels"]:
        cells[c] = cells[c].astype(np.int32)
    # routing graph over pooled fine lane cells
    nodes = cells[(cells.res == "fine") & (cells.g == POOLED) & cells.lane].reset_index(drop=True)
    if len(nodes) < 20:
        raise InsufficientLanes(f"only {len(nodes)} pooled lane cells; need >= 20")
    trans = con.execute(f"""
        WITH s AS (SELECT mmsi, t, floor(lat / {lc.cell_deg})::BIGINT AS i, floor(lon / {lc.cell_deg})::BIGINT AS j,
                          lag(t) OVER w AS pt, lag(floor(lat / {lc.cell_deg})::BIGINT) OVER w AS pi,
                          lag(floor(lon / {lc.cell_deg})::BIGINT) OVER w AS pj
                   FROM lane_pts WHERE category != 'fishing' WINDOW w AS (PARTITION BY mmsi ORDER BY t))
        SELECT DISTINCT pi, pj, i, j FROM s
        WHERE pi IS NOT NULL AND (pi != i OR pj != j) AND t - pt <= {lc.edge_max_gap_s}
    """).df()
    nk = pd.Series(np.arange(len(nodes)), index=cell_key(nodes.i, nodes.j))
    a = nk.reindex(cell_key(trans.pi, trans.pj)).to_numpy()
    b = nk.reindex(cell_key(trans.i, trans.j)).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    ea, eb = a[ok].astype(np.int64), b[ok].astype(np.int64)
    ni, nj = nodes.i.to_numpy(), nodes.j.to_numpy()
    for di in (-1, 0, 1):                                   # 8-neighbour adjacency inside lanes
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            nb = nk.reindex(cell_key(ni + di, nj + dj)).to_numpy()
            m = np.isfinite(nb)
            ea = np.concatenate([ea, np.flatnonzero(m)])
            eb = np.concatenate([eb, nb[m].astype(np.int64)])
    la, lo = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    w = haversine_m(la[ea], lo[ea], la[eb], lo[eb]) / 1000
    keep = (w <= lc.edge_max_km) & (ea != eb)
    lo_, hi_ = np.minimum(ea, eb)[keep], np.maximum(ea, eb)[keep]          # undirected: one row per pair
    edges = pd.DataFrame({"a": lo_, "b": hi_, "km": w[keep]}).drop_duplicates(["a", "b"]).reset_index(drop=True)
    log(f"sea lanes: {n_pts:,} training vessel-minutes from {n_v:,} vessels; "
        f"{int(cells[(cells.res == 'fine') & cells.lane].shape[0]):,} fine lane cells; "
        f"graph {len(nodes):,} nodes / {len(edges):,} edges")
    return {"cells": cells, "nodes": nodes[["i", "j", "lat", "lon", "n_vessels"]], "edges": edges,
            "n_train_vessel_minutes": int(n_pts), "n_train_vessels": int(n_v), "split_t": split_t}


def save(res: dict, rd: Path):
    res["cells"].to_parquet(rd / "lanes_cells.parquet", index=False)
    res["nodes"].to_parquet(rd / "lanes_nodes.parquet", index=False)
    res["edges"].to_parquet(rd / "lanes_edges.parquet", index=False)


def _unit(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


@dataclass
class LaneModel:
    """Query side of the learned lanes (used by the pipeline and, read-only, by the app)."""
    cells: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    lc: LaneConfig

    @classmethod
    def from_result(cls, res: dict, lc: LaneConfig | None = None):
        return cls(res["cells"], res["nodes"], res["edges"], lc or LaneConfig())

    @classmethod
    def load(cls, d: Path, lc: LaneConfig | None = None):
        return cls(pd.read_parquet(d / "lanes_cells.parquet"), pd.read_parquet(d / "lanes_nodes.parquet"),
                   pd.read_parquet(d / "lanes_edges.parquet"), lc or LaneConfig())

    def __post_init__(self):
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree
        n = len(self.nodes)
        e = self.edges
        self.graph = coo_matrix((e.km.to_numpy(), (e.a.to_numpy(), e.b.to_numpy())), shape=(n, n)).tocsr()
        self._node_tree = cKDTree(_unit(self.nodes.lat, self.nodes.lon))
        fine = self.cells[(self.cells.res == "fine")]
        self._fine = {g_: d.set_index(cell_key(d.i, d.j)) for g_, d in fine.groupby("g")}
        self._lane_trees = {g_: (cKDTree(_unit(d[d.lane].lat, d[d.lane].lon)) if d.lane.any() else None)
                            for g_, d in self._fine.items()}

    # ---------------------------------------------------------------- per-point lookups
    def lane_distance_km(self, lat, lon, group) -> np.ndarray:
        """Distance to the nearest lane cell of the point's own lane group (km, capped at 50)."""
        lat, lon, group = np.asarray(lat, float), np.asarray(lon, float), np.asarray(group, dtype=object)
        out = np.full(len(lat), np.nan)
        for g_, tree in self._lane_trees.items():
            m = (group == g_) & np.isfinite(lat) & np.isfinite(lon)
            if tree is None or not m.any():
                continue
            chord, _ = tree.query(_unit(lat[m], lon[m]), distance_upper_bound=50 / 6371.0)
            out[m] = np.minimum(2 * np.arcsin(np.minimum(chord / 2, 1)) * R_EARTH / 1000, 50.0)
        return out

    def cell_stats(self, lat, lon, group) -> pd.DataFrame:
        """Training statistics of the fine cell under each point, for the point's lane group (NaN if none)."""
        lat, lon, group = np.asarray(lat, float), np.asarray(lon, float), np.asarray(group, dtype=object)
        cols = ["n_moving", "n_all", "stop_share", "sog_med", "sog_q25", "sog_q75"] + SECTORS
        out = pd.DataFrame(np.nan, index=np.arange(len(lat)), columns=cols)
        k = cell_key(np.floor(np.nan_to_num(lat) / self.lc.cell_deg), np.floor(np.nan_to_num(lon) / self.lc.cell_deg))
        for g_, d in self._fine.items():
            m = group == g_
            if m.any():
                out.loc[m, cols] = d[cols].reindex(k[m]).to_numpy(float)
        return out

    def snap(self, lat, lon):
        chord, idx = self._node_tree.query(_unit(lat, lon))
        km = 2 * np.arcsin(np.minimum(np.asarray(chord) / 2, 1)) * R_EARTH / 1000
        return np.where(km <= self.lc.snap_max_km, idx, -1), km

    # ---------------------------------------------------------------- routing
    def route_km(self, lat0, lon0, lat1, lon1, chunk: int = 32) -> np.ndarray:
        """Distance along learned lanes between point pairs (snap + graph + snap); NaN if not connected."""
        from scipy.sparse.csgraph import dijkstra
        a, da = self.snap(lat0, lon0)
        b, db = self.snap(lat1, lon1)
        out = np.full(len(a), np.nan)
        ok = (a >= 0) & (b >= 0)
        ub = np.unique(b[ok])
        for s in range(0, len(ub), chunk):
            src = ub[s:s + chunk]
            D = dijkstra(self.graph, directed=False, indices=src, limit=self.lc.route_limit_km)
            pos = {n: r for r, n in enumerate(src)}
            m = ok & np.isin(b, src)
            rows = np.array([pos[x] for x in b[m]])
            out[m] = D[rows, a[m]]
        out = out + da + db
        return np.where(np.isfinite(out), out, np.nan)

    def path(self, lat0, lon0, lat1, lon1) -> pd.DataFrame:
        """Expected route between two positions along learned lanes (empty if either end is off-lane)."""
        from scipy.sparse.csgraph import dijkstra
        (a,), _ = self.snap([lat0], [lon0])
        (b,), _ = self.snap([lat1], [lon1])
        if a < 0 or b < 0:
            return pd.DataFrame(columns=["lat", "lon"])
        D, P = dijkstra(self.graph, directed=False, indices=b, limit=self.lc.route_limit_km, return_predecessors=True)
        if not np.isfinite(D[a]):
            return pd.DataFrame(columns=["lat", "lon"])
        seq, n = [a], a
        while n != b and n >= 0:
            n = P[n]
            seq.append(n)
        nd = self.nodes.iloc[[x for x in seq if x >= 0]]
        return pd.DataFrame({"lat": np.r_[lat0, nd.lat, lat1], "lon": np.r_[lon0, nd.lon, lon1]})


def direction_support(stats: pd.DataFrame, cog) -> np.ndarray:
    """Share of the cell's moving training traffic within about +/-45 deg of this course."""
    s = stats[SECTORS].to_numpy(float)
    tot = s.sum(axis=1)
    k = np.floor(np.nan_to_num(np.asarray(cog, float)) / 45).astype(int) % 8
    r = np.arange(len(k))
    sup = s[r, k] + 0.5 * s[r, (k - 1) % 8] + 0.5 * s[r, (k + 1) % 8]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, sup / tot, np.nan)
