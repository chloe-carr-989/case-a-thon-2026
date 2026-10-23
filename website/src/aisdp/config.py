"""Paths, subset registry and experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # website/ (project root)
REPO_ROOT = ROOT.parent                          # repository root, which holds data/
DATA_DIR = REPO_ROOT / "data" / "dataset3"
DAYS_DIR = DATA_DIR / "marine_2026-01"
SUBSETS_DIR = DATA_DIR / "marine_subsets"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

# Organizer-provided files. The subsets are nested head-prefixes of the full day file.
# A value may be a list of daily files; they are ingested together into one day-partitioned dataset.
SUBSETS = {
    "subset4": SUBSETS_DIR / "jan_01_subset_4.csv",
    "subset3": SUBSETS_DIR / "jan_01_subset_3.csv",
    "subset2": SUBSETS_DIR / "jan_01_subset_2.csv",
    "subset1": SUBSETS_DIR / "jan_01_subset_1.csv",
    "full": DAYS_DIR / "ais-2026-01-01.csv",
    "jan01_02": [DAYS_DIR / "ais-2026-01-01.csv", DAYS_DIR / "ais-2026-01-02.csv"],
    "jan01_04": [DAYS_DIR / f"ais-2026-01-0{d}.csv" for d in range(1, 5)],
}

# (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    "all": None,
    "us_east": (-82.0, 24.0, -65.0, 46.0),
    "gulf": (-98.0, 24.0, -80.0, 31.0),
    "us_west": (-130.0, 30.0, -115.0, 50.0),
}

CODE_VERSION = "0.1.0"
SCHEMA_VERSION = 1


@dataclass
class ExperimentConfig:
    horizon_s: int = 900            # 15-minute prediction horizon
    tol_s: int = 120                # target must be observed within +/- 2 min of horizon
    anchor_spacing_s: int = 600     # at most one decision per vessel per 10 minutes
    min_history_s: int = 2700       # test vessel must have >= 45 min of history (stale state exists)
    prev_gap_min_s: int = 60        # previous fix is >= 60 s before the state fix
    prev_gap_max_s: int = 900       # ... and <= 15 min before it
    sample_vessels: int = 4000      # bounded vessel sample (deterministic hash)
    train_time_frac: float = 0.45   # chronological split point
    test_vessel_mod: int = 10       # mmsi hash % mod < test_vessel_cut -> test vessel
    test_vessel_cut: int = 3
    max_train_rows: int = 200_000
    n_windows: int = 5
    seeds: tuple = (0, 1, 2)
    # Simulated degradation progression (per window). Labelled SIMULATED.
    stale_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)
    stale_delay_min_s: int = 300
    stale_delay_max_s: int = 2700
    lowq_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)
    severity: float = 1.0
    # Countermeasure thresholds: fixed a priori, never tuned on test windows.
    age_comp_s: int = 120
    max_age_defer_s: int = 1200
    max_valid_sog_kn: float = 40.0
    defer_quality: float = 0.35
    region: str = "all"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["stale_schedule"] = list(self.stale_schedule)
        d["lowq_schedule"] = list(self.lowq_schedule)
        return d


# ---------------------------------------------------------------- expansion tasks (see info/CLAUDE_MVP_EXPANSION_SPEC.md)
TASKS = ["position_15m", "destination", "port_occupancy", "traffic_density"]


@dataclass
class QualityConfig:
    """Thresholds for the record-level integrity flags and observation/reporting gap detection."""
    high_rate_per_min: int = 4          # > this many reports from one MMSI in one UTC minute
    implied_speed_max_kn: float = 60.0  # between consecutive fixes ...
    implied_speed_min_dist_m: float = 1000.0  # ... and only if they are > 1 km apart (GPS jitter at short dt)
    sog_implausible_kn: float = 50.0    # informational: valid encoding but implausible for most vessels
    gap_min_s: int = 600                # a gap is > max(gap_min_s, gap_cadence_mult * vessel median cadence)
    gap_cadence_mult: float = 5.0
    sparse_cell_deg: float = 0.5        # reception-coverage proxy for "possible coverage exit"
    sparse_cell_max_vessels: int = 5
    max_gap_rows_saved: int = 200_000
    synth_vessels: int = 400            # synthetic withheld-record test
    synth_seed: int = 0


@dataclass
class CountConfig:
    """Shared count forecasting (traffic density on grid cells, port occupancy proxy on study zones)."""
    bucket_s: int = 900                 # 15-minute buckets
    window_buckets: int = 4             # target = unique MMSIs in the next 60 minutes
    lookback_buckets: int = 12          # features use at most the previous 3 hours
    cell_deg: float = 0.25              # density grid resolution (area computed on the sphere)
    dense_min_mean: float = 5.0         # dense cell = train-period mean count >= 5
    train_time_frac: float = 0.5        # origins whose target window ends before split_t train the model
    n_windows: int = 5
    seeds: tuple = (0, 1, 2)
    # simulated feed degradation (labelled SYNTHETIC), per chronological test window
    delay_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)   # share of origins whose latest buckets are late
    delay_max_buckets: int = 4
    ghost_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)   # synthetic ghost vessel-buckets per real one
    omission_rate: float = 0.10         # share of real vessel-buckets lost, scaled by the window's ghost share / 0.6
    duplicate_rate: float = 0.30        # share of real vessel-buckets whose messages are tripled
    # countermeasure (fixed a priori)
    feed_complete_ratio: float = 0.5    # latest bucket volume / trailing median below this = late feed
    max_delay_defer_buckets: int = 3    # >= 45 min of missing recent feed -> defer
    min_msgs_per_vessel: int = 2        # in the lookback window; single-message transients are dropped
    port_zone_deg: float = 0.1
    port_min_vessels: int = 10          # train-period distinct dwelling vessels to define a port/anchorage study zone
    port_max_zones: int = 60
    max_train_rows: int = 300_000


@dataclass
class LaneConfig:
    """Learned sea lanes: per-category traffic grid + a routing graph over the pooled commercial lanes."""
    cell_deg: float = 0.01              # ~1.1 km cells for lanes, scoring and routing
    coarse_deg: float = 0.1             # display aggregate for whole-country views
    moving_min_sog_kn: float = 2.0
    stop_max_sog_kn: float = 0.5
    lane_min_vessels: int = 3           # a lane cell = moving traffic in the cell and >= 3 distinct moving
    block_deg: float = 0.03             # vessels in its ~3 km block during the training period
    min_cell_moving: int = 10           # direction/speed statistics need >= 10 moving vessel-minutes in the cell
    groups: tuple = ("cargo", "tanker", "passenger", "tug_service")   # own lanes; everything else -> pooled
    edge_max_km: float = 15.0           # trajectory transitions between lane cells become graph edges ...
    edge_max_gap_s: int = 900           # ... when consecutive fixes are <= 15 min apart
    own_lane_max_km: float = 20.0       # anomaly scoring falls back to pooled lanes beyond this
    snap_max_km: float = 10.0           # a position further than this from any lane has no expected route
    route_limit_km: float = 800.0


@dataclass
class AnomalyConfig:
    """Movement anomaly scoring relative to learned lanes (Government mode). Thresholds fixed a priori."""
    calib_s: int = 2 * 86400            # calibration windows: the last 48 h before the split
    step_s: int = 1800                  # one scored vessel-window every 30 min ...
    window_s: int = 3600                # ... over the previous 60 min of reports
    recent_s: int = 600                 # off-lane distance uses the newest 10 min of the window
    lane_far_km: float = 2.0
    coverage_km: float = 20.0           # off-lane distance only counts within 20 km of any learned traffic
    wrongway_support: float = 0.10      # < 10% of the cell's traffic moves within +/-45 deg of this course
    unusual_stop_share: float = 0.05    # stopped where < 5% of the cell's training reports are stops
    # off-lane carries the score; the noisier components add supporting evidence. Chosen on a vessel-disjoint
    # split of the calibration period (sum of all four at weight 1: 21% injected-drift recall; these: 63%).
    weights: tuple = (("off_lane", 1.0), ("wrong_way", 0.25), ("speed", 0.25), ("stop", 0.25))
    unusual_pct: float = 97.5           # score percentile vs the same category's calibration windows
    high_pct: float = 99.5
    min_calib_windows: int = 500        # smaller categories are calibrated against the pooled set
    spike_speed_kn: float = 60.0        # countermeasure: a fix implying > 60 kn both in and out is a spike
    spike_min_km: float = 1.0
    max_valid_sog_kn: float = 40.0
    stale_hold_s: int = 1200            # countermeasure: newest report older than 20 min -> no movement verdict
    stale_flag_s: int = 600
    gap_flag_s: int = 900
    # SIMULATED degradation, per chronological test window (same schedule as the other tasks)
    stale_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)
    stale_delay_min_s: int = 300
    stale_delay_max_s: int = 2700
    lowq_schedule: tuple = (0.10, 0.20, 0.30, 0.45, 0.60)
    lowq_point_share: float = 0.5       # in an affected window the faulty source adds one record per two real ones
    synth_windows: int = 3000           # injected route deviations with known labels
    synth_offset_km: tuple = (2.0, 8.0)
    n_windows: int = 5
    seed: int = 0


@dataclass
class DestinationConfig:
    bucket_s: int = 600                 # 10-minute buckets for dwell detection
    dwell_max_sog_kn: float = 1.0
    dwell_min_buckets: int = 3          # >= 30 min of sustained low speed = a dwell
    underway_min_sog_kn: float = 3.0
    arrival_lookback_buckets: int = 6   # dwell counts as an arrival if underway within the previous 60 min
    zone_deg: float = 0.05              # inferred arrival zone = 0.05 deg grid cell
    zone_min_vessels: int = 3           # distinct vessels dwelling there in the training period
    window_s: int = 6 * 3600            # next arrival must start within 6 h of the decision
    anchor_spacing_s: int = 1800        # at most one decision per vessel per 30 minutes
    train_time_frac: float = 0.5
    max_candidates: int = 25
    min_radius_km: float = 20.0
    max_radius_km: float = 150.0
    abstain_score: float = 0.15         # top normalised model score below this -> "unknown"
    max_age_defer_s: int = 1200
    n_windows: int = 5
    seeds: tuple = (0, 1, 2)
