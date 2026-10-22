"""Paths, subset registry and experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # website/ (project root)
REPO_ROOT = ROOT.parent                          # repository root, which holds data/
DATA_DIR = REPO_ROOT / "data" / "dataset3" / "ais-2026-01-01.csv"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

# Organizer-provided files. They are nested head-prefixes of the full day file.
SUBSETS = {
    "subset4": DATA_DIR / "jan_01_subset_4.csv",
    "subset3": DATA_DIR / "jan_01_subset_3.csv",
    "subset2": DATA_DIR / "jan_01_subset_2.csv",
    "subset1": DATA_DIR / "jan_01_subset_1.csv",
    "full": DATA_DIR / "ais-2026-01-01.csv",
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
