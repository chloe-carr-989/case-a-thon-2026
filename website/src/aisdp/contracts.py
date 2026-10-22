"""Thin data contracts shared by the four prediction tasks.

The existing modules keep their own frames; these dataclasses are adapters for audit, the app and tests.
Task-specific fields are explicit (`value` holds the task's own prediction) rather than overloaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np


class Task(str, Enum):
    POSITION_15M = "position_15m"
    DESTINATION = "destination"
    PORT_OCCUPANCY = "port_occupancy"
    TRAFFIC_DENSITY = "traffic_density"


@dataclass
class AISRecord:
    mmsi: int
    event_t: float                      # observed AIS timestamp (UTC epoch s)
    lat: float
    lon: float
    sog: float | None = None
    cog: float | None = None
    heading: float | None = None
    availability_t: float | None = None  # unknown in nominal data; only set when a delay is SIMULATED
    source: str = "AIS-original"         # no real source IDs exist in this data; SYN-* labels are synthetic
    flags: list[str] = field(default_factory=list)


@dataclass
class QualityEvent:
    flag_code: str
    key: str                            # offending input (mmsi / unit) ...
    t: float | None                     # ... and time
    origin: str                         # "observed" or "synthetic"
    recommended_response: str
    severity: float | None = None


@dataclass
class PredictionRequest:
    task: Task
    decision_t: float
    target: str                         # vessel MMSI, grid cell id or zone id
    horizon_s: int
    scenario: str = "nominal"
    mitigation: bool = False


@dataclass
class PredictionResult:
    task: Task
    target: str
    decision_t: float
    horizon_s: int
    answered: bool
    value: dict                          # task-specific: {"lat","lon"} | {"zones","scores"} | {"count","per_1000km2"}
    units: str
    model_version: str
    input_cutoff_t: float
    age_s: float | None = None
    flags: list[str] = field(default_factory=list)
    action: str = "model"
    runtime_ms: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task.value
        return d


@dataclass
class EvaluationRecord:
    task: Task
    subset: str
    case_id: str
    target_window: str
    scenario: str
    mitigation: bool
    seed: int
    reference: object
    result: PredictionResult
    metric: dict
    covered: bool
    status: str = "evaluated"


def _flags(s) -> list[str]:
    return [f for f in str(s).split(";") if f and f != "nan"]


def position_result(row, mitigated: bool, model_version: str = "hgb-position-0.1.0") -> PredictionResult:
    """From a row of results/cases_default.parquet (backward-compatible 15-minute output)."""
    ans = bool(row["answered"]) if mitigated else True
    lat, lon = (row["mi_lat"], row["mi_lon"]) if mitigated else (row["un_lat"], row["un_lon"])
    return PredictionResult(Task.POSITION_15M, str(int(row["mmsi"])), float(row["decision_t"]), 900, ans,
                            {"lat": float(lat), "lon": float(lon)} if ans else {}, "deg WGS84", model_version,
                            float(row["state_t"]), float(row["age_s"]), _flags(row["flags"]) if mitigated else [],
                            str(row["action"]) if mitigated else "model")


def destination_result(row, mitigated: bool, model_version: str = "hgb-ranker-0.1.0") -> PredictionResult:
    t = "mi" if mitigated else "un"
    ans = bool(row[f"{t}_answered"])
    zones = [row[f"{t}_pred{k}"] for k in (1, 2, 3) if isinstance(row[f"{t}_pred{k}"], str)]
    return PredictionResult(Task.DESTINATION, str(int(row["mmsi"])), float(row["decision_t"]), 6 * 3600, ans,
                            {"zones": zones, "top_score": float(row[f"{t}_top_score"])} if ans else {"zones": ["unknown"]},
                            "inferred arrival zone id; normalised model score", model_version,
                            float(row["decision_t"]) - float(row["age_s"]), float(row["age_s"]),
                            _flags(row["mi_flags"]) if mitigated else [], str(row["mi_action"]) if mitigated else "model")


def count_result(row, task: Task, unit_id: str, area_km2: float, mitigated: bool, bucket_s: int = 900,
                 model_version: str = "hgb-count-0.1.0") -> PredictionResult:
    ans = bool(row["answered"]) if mitigated else True
    v = float(row["pred_mi"] if mitigated else row["pred_un"])
    o = float(row["o"]) * bucket_s
    return PredictionResult(task, unit_id, o, 3600, ans,
                            {"count": v, "per_1000km2": 1000 * v / area_km2} if ans and np.isfinite(v) else {},
                            "unique vessels (MMSIs) in the next 60 min", model_version,
                            o - float(row["delay_buckets"]) * bucket_s, float(row["delay_buckets"]) * bucket_s,
                            ["late_feed"] if row["delay_detected"] else [], str(row["action"]) if mitigated else "model")
