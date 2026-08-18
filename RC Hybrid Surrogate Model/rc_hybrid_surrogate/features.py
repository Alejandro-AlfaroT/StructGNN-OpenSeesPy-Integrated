"""Derived ground-motion, modal, and structure-motion interaction features."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .data import SampleRecord


GRAVITY_IN_PER_SEC2 = 386.0886
FEATURE_CACHE_VERSION = 2
SPECTRUM_PERIODS_SEC = np.geomspace(0.05, 5.0, 40)
REPORTED_SPECTRUM_PERIODS_SEC = np.asarray(
    [0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00],
    dtype=np.float64,
)
FREQUENCY_BANDS_HZ = (
    (0.0, 1.0, "0_1"),
    (1.0, 2.0, "1_2"),
    (2.0, 5.0, "2_5"),
    (5.0, 10.0, "5_10"),
    (10.0, math.inf, "10_plus"),
)


def _period_label(period: float) -> str:
    return f"{period:g}".replace(".", "p")


MOTION_FEATURE_NAMES_PER_DIRECTION = [
    "pgv_in_per_sec",
    "arias_intensity_in_per_sec",
    "cav_in_per_sec",
    "significant_duration_5_95_sec",
    "predominant_frequency_hz",
    "mean_frequency_hz",
    *[f"fft_energy_ratio_{label}_hz" for _, _, label in FREQUENCY_BANDS_HZ],
    *[
        f"psa_{_period_label(period)}_sec_g"
        for period in REPORTED_SPECTRUM_PERIODS_SEC
    ],
]
MOTION_FEATURE_NAMES = [
    f"{direction}_{name}"
    for direction in ("x", "y")
    for name in MOTION_FEATURE_NAMES_PER_DIRECTION
]
MODAL_FEATURE_NAMES = [
    "period_x_mode_1_sec",
    "period_y_mode_1_sec",
    "roof_eigenvector_x_mode_1_abs",
    "roof_eigenvector_y_mode_1_abs",
]
INTERACTION_FEATURE_NAMES = [
    "psa_x_at_period_x_g",
    "psa_y_at_period_y_g",
    "psa_x_at_period_y_g",
    "psa_y_at_period_x_g",
    "psa_x_at_period_x_over_pga_x",
    "psa_y_at_period_y_over_pga_y",
]
ENGINEERED_FEATURE_NAMES = (
    MOTION_FEATURE_NAMES + MODAL_FEATURE_NAMES + INTERACTION_FEATURE_NAMES
)
FEATURE_GROUPS = {
    "motion": MOTION_FEATURE_NAMES,
    "modal": MODAL_FEATURE_NAMES,
    "interaction": INTERACTION_FEATURE_NAMES,
}


@dataclass(frozen=True)
class EngineeredFeatureCache:
    feature_names: list[str]
    values: dict[str, np.ndarray]

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def vector(self, sample_id: str) -> np.ndarray:
        try:
            return self.values[sample_id]
        except KeyError as error:
            raise KeyError(
                f"Engineered features are missing for sample {sample_id!r}. "
                "Rebuild the feature cache after importing new cases."
            ) from error

    def select_groups(self, group_names: Sequence[str]) -> "EngineeredFeatureCache":
        unknown = [name for name in group_names if name not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(f"Unknown engineered feature groups: {unknown}")
        selected_names = [
            feature_name
            for group_name in group_names
            for feature_name in FEATURE_GROUPS[group_name]
        ]
        indices = [self.feature_names.index(name) for name in selected_names]
        return EngineeredFeatureCache(
            feature_names=selected_names,
            values={
                sample_id: vector[indices].astype(np.float32, copy=False)
                for sample_id, vector in self.values.items()
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "EngineeredFeatureCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != FEATURE_CACHE_VERSION:
            raise ValueError("Unsupported engineered-feature cache version.")
        feature_names = list(payload["feature_names"])
        if feature_names != ENGINEERED_FEATURE_NAMES:
            raise ValueError("Engineered-feature names do not match the current code.")
        values = {
            sample_id: np.asarray(vector, dtype=np.float32)
            for sample_id, vector in payload["samples"].items()
        }
        expected = len(feature_names)
        invalid = [key for key, value in values.items() if value.shape != (expected,)]
        if invalid:
            raise ValueError(f"Invalid engineered vectors: {invalid[:5]}")
        return cls(feature_names=feature_names, values=values)


GROUP_INTENSITY_FEATURE_NAMES = (
    "x_arias_intensity_in_per_sec",
    "y_arias_intensity_in_per_sec",
    "x_psa_1_sec_g",
    "y_psa_1_sec_g",
    "x_pgv_in_per_sec",
    "y_pgv_in_per_sec",
)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    if values.size > 1:
        ranks /= values.size - 1
    return ranks


def load_group_intensity_scores(path: str | Path) -> dict[str, float]:
    """Composite ground-motion intensity score per record-group, in [0, 1].

    Used to stratify train/validation/test splits so a small pool of
    distinct ground-motion pairs doesn't hand one split a systematically
    more (or less) severe subset by chance. Combines several features of
    different units (Arias intensity, PSa at 1 sec, PGV, each X and Y) as
    an average percentile rank rather than a raw weighted sum, so no single
    feature's scale dominates the score.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    motion_groups = payload.get("motion_groups")
    if not motion_groups:
        raise ValueError(
            f"{path} has no 'motion_groups' entries; rebuild it with "
            "build_engineered_features.py before stratifying a split."
        )
    indices = [
        MOTION_FEATURE_NAMES.index(name) for name in GROUP_INTENSITY_FEATURE_NAMES
    ]
    group_names = sorted(motion_groups)
    raw = np.asarray(
        [
            [motion_groups[name]["motion_features"][index] for index in indices]
            for name in group_names
        ],
        dtype=np.float64,
    )
    rank_columns = np.stack(
        [_percentile_ranks(raw[:, column]) for column in range(raw.shape[1])],
        axis=1,
    )
    composite = rank_columns.mean(axis=1)
    return {name: float(score) for name, score in zip(group_names, composite)}


def pseudo_spectral_acceleration(
    acceleration_in_per_sec2: np.ndarray,
    dt_sec: float,
    periods_sec: np.ndarray = SPECTRUM_PERIODS_SEC,
    damping_ratio: float = 0.05,
) -> np.ndarray:
    """Compute 5%-damped PSA with vectorized Newmark average acceleration."""
    acceleration = np.asarray(acceleration_in_per_sec2, dtype=np.float64)
    periods = np.asarray(periods_sec, dtype=np.float64)
    if acceleration.ndim != 1 or acceleration.size < 2:
        raise ValueError("Acceleration must be a one-dimensional history.")
    if dt_sec <= 0.0 or np.any(periods <= 0.0):
        raise ValueError("Time step and oscillator periods must be positive.")

    ground = acceleration - acceleration.mean()
    omega = 2.0 * np.pi / periods
    stiffness = np.square(omega)
    damping = 2.0 * damping_ratio * omega
    beta = 0.25
    gamma = 0.5
    coefficient_0 = 1.0 / (beta * dt_sec * dt_sec)
    coefficient_1 = gamma / (beta * dt_sec)
    coefficient_2 = 1.0 / (beta * dt_sec)
    coefficient_3 = 1.0 / (2.0 * beta) - 1.0
    coefficient_4 = gamma / beta - 1.0
    coefficient_5 = dt_sec * (gamma / (2.0 * beta) - 1.0)
    effective_stiffness = stiffness + coefficient_0 + damping * coefficient_1

    displacement = np.zeros_like(periods)
    velocity = np.zeros_like(periods)
    relative_acceleration = np.full_like(periods, -ground[0])
    maximum_displacement = np.zeros_like(periods)
    for ground_value in ground[1:]:
        effective_force = (
            -ground_value
            + coefficient_0 * displacement
            + coefficient_2 * velocity
            + coefficient_3 * relative_acceleration
            + damping
            * (
                coefficient_1 * displacement
                + coefficient_4 * velocity
                + coefficient_5 * relative_acceleration
            )
        )
        new_displacement = effective_force / effective_stiffness
        new_acceleration = (
            coefficient_0 * (new_displacement - displacement)
            - coefficient_2 * velocity
            - coefficient_3 * relative_acceleration
        )
        new_velocity = velocity + dt_sec * (
            (1.0 - gamma) * relative_acceleration + gamma * new_acceleration
        )
        displacement = new_displacement
        velocity = new_velocity
        relative_acceleration = new_acceleration
        maximum_displacement = np.maximum(maximum_displacement, np.abs(displacement))
    return stiffness * maximum_displacement / GRAVITY_IN_PER_SEC2


def _direction_motion_features(
    acceleration: np.ndarray, dt_sec: float
) -> tuple[list[float], np.ndarray]:
    acceleration = np.asarray(acceleration, dtype=np.float64)
    centered = acceleration - acceleration.mean()
    velocity = np.zeros_like(centered)
    velocity[1:] = np.cumsum(
        0.5 * (centered[1:] + centered[:-1]) * dt_sec
    )
    pgv = float(np.max(np.abs(velocity)))

    squared = np.square(acceleration)
    arias_increment = 0.5 * (squared[1:] + squared[:-1]) * dt_sec
    cumulative_arias_raw = np.concatenate(([0.0], np.cumsum(arias_increment)))
    arias = float(
        np.pi * cumulative_arias_raw[-1] / (2.0 * GRAVITY_IN_PER_SEC2)
    )
    cav = float(
        np.sum(0.5 * (np.abs(acceleration[1:]) + np.abs(acceleration[:-1])) * dt_sec)
    )
    if cumulative_arias_raw[-1] > 0.0:
        normalized_arias = cumulative_arias_raw / cumulative_arias_raw[-1]
        time_axis = np.arange(acceleration.size, dtype=np.float64) * dt_sec
        duration_5_95 = float(
            np.interp(0.95, normalized_arias, time_axis)
            - np.interp(0.05, normalized_arias, time_axis)
        )
    else:
        duration_5_95 = 0.0

    frequencies = np.fft.rfftfreq(centered.size, d=dt_sec)
    power = np.square(np.abs(np.fft.rfft(centered)))
    if power.size:
        power[0] = 0.0
    power_sum = float(power.sum())
    if power_sum > 0.0:
        predominant_frequency = float(frequencies[int(np.argmax(power))])
        mean_frequency = float(np.sum(frequencies * power) / power_sum)
        band_ratios = []
        for lower, upper, _ in FREQUENCY_BANDS_HZ:
            if math.isinf(upper):
                selected = frequencies >= lower
            else:
                selected = (frequencies >= lower) & (frequencies < upper)
            band_ratios.append(float(power[selected].sum() / power_sum))
    else:
        predominant_frequency = 0.0
        mean_frequency = 0.0
        band_ratios = [0.0] * len(FREQUENCY_BANDS_HZ)

    spectrum = pseudo_spectral_acceleration(centered, dt_sec)
    reported_spectrum = np.interp(
        REPORTED_SPECTRUM_PERIODS_SEC, SPECTRUM_PERIODS_SEC, spectrum
    )
    features = [
        pgv,
        arias,
        cav,
        duration_5_95,
        predominant_frequency,
        mean_frequency,
        *band_ratios,
        *reported_spectrum.tolist(),
    ]
    return features, spectrum


def _directional_modes(modal_payload: Sequence[dict], direction: int) -> list[dict]:
    valid_modes = [
        mode
        for mode in modal_payload
        if mode.get("valid") and len(mode.get("roof_eigenvector") or []) >= 2
    ]
    if not valid_modes:
        return []
    maximum_amplitude = max(
        abs(float(mode["roof_eigenvector"][direction])) for mode in valid_modes
    )
    minimum_amplitude = max(maximum_amplitude * 1.0e-3, 1.0e-12)
    eligible = []
    for mode in valid_modes:
        roof = mode.get("roof_eigenvector") or []
        amplitudes = np.abs(np.asarray(roof[:2], dtype=np.float64))
        if amplitudes[direction] < minimum_amplitude:
            continue
        eligible.append(mode)
    candidates = [
        mode
        for mode in eligible
        if abs(float(mode["roof_eigenvector"][direction]))
        >= abs(float(mode["roof_eigenvector"][1 - direction]))
    ]
    candidates.sort(key=lambda mode: int(mode["mode"]))
    if len(candidates) < 2:
        existing = {int(mode["mode"]) for mode in candidates}
        candidates.extend(
            mode
            for mode in sorted(eligible, key=lambda item: int(item["mode"]))
            if int(mode["mode"]) not in existing
        )
    return candidates[:2]


def _modal_features(modal_path: Path) -> tuple[list[float], float, float]:
    modal_payload = json.loads(modal_path.read_text(encoding="utf-8"))
    x_modes = _directional_modes(modal_payload, 0)
    y_modes = _directional_modes(modal_payload, 1)
    if not x_modes or not y_modes:
        raise ValueError(f"Could not identify first X/Y modes in {modal_path}")
    period_x_1 = float(x_modes[0]["period"])
    period_y_1 = float(y_modes[0]["period"])
    values = [
        period_x_1,
        period_y_1,
        abs(float(x_modes[0]["roof_eigenvector"][0])),
        abs(float(y_modes[0]["roof_eigenvector"][1])),
    ]
    return values, period_x_1, period_y_1


def _atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    delay = 0.05
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.5)


def build_engineered_feature_cache(
    records: Sequence[SampleRecord], output_path: str | Path
) -> dict:
    """Build or incrementally refresh a cache without modifying source samples."""
    output_path = Path(output_path)
    existing_motion_groups: dict[str, dict] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            int(existing.get("version", -1)) == FEATURE_CACHE_VERSION
            and existing.get("feature_names") == ENGINEERED_FEATURE_NAMES
            and existing.get("spectrum_periods_sec")
            == SPECTRUM_PERIODS_SEC.tolist()
        ):
            existing_motion_groups = dict(existing.get("motion_groups", {}))

    records_by_group: dict[str, list[SampleRecord]] = {}
    for record in records:
        records_by_group.setdefault(record.record_group, []).append(record)

    motion_groups: dict[str, dict] = {}
    reused = 0
    for group_name, group_records in records_by_group.items():
        if group_name in existing_motion_groups:
            motion_groups[group_name] = existing_motion_groups[group_name]
            reused += 1
            continue
        representative = group_records[0]
        with np.load(representative.path, allow_pickle=False) as sample:
            ground_motion = np.asarray(sample["ground_motion"], dtype=np.float64)
            time_seconds = np.asarray(sample["time_seconds"], dtype=np.float64)
        dt_sec = float(np.median(np.diff(time_seconds)))
        x_features, spectrum_x = _direction_motion_features(ground_motion[:, 0], dt_sec)
        y_features, spectrum_y = _direction_motion_features(ground_motion[:, 1], dt_sec)
        motion_groups[group_name] = {
            "motion_features": x_features + y_features,
            "spectrum_x_g": spectrum_x.tolist(),
            "spectrum_y_g": spectrum_y.tolist(),
        }

    sample_vectors: dict[str, list[float]] = {}
    for record in records:
        group = motion_groups[record.record_group]
        case_directory = record.path.parents[2]
        modal_path = (
            case_directory
            / "ntha"
            / record.run_name
            / "modal_results_post_gravity_tangent.json"
        )
        modal_values, period_x, period_y = _modal_features(modal_path)
        spectrum_x = np.asarray(group["spectrum_x_g"], dtype=np.float64)
        spectrum_y = np.asarray(group["spectrum_y_g"], dtype=np.float64)
        psa_x_at_x = float(np.interp(period_x, SPECTRUM_PERIODS_SEC, spectrum_x))
        psa_y_at_y = float(np.interp(period_y, SPECTRUM_PERIODS_SEC, spectrum_y))
        psa_x_at_y = float(np.interp(period_y, SPECTRUM_PERIODS_SEC, spectrum_x))
        psa_y_at_x = float(np.interp(period_x, SPECTRUM_PERIODS_SEC, spectrum_y))
        with np.load(record.path, allow_pickle=False) as sample:
            record_features = np.asarray(sample["record_features"], dtype=np.float64)
        pga_x = max(abs(float(record_features[0, 4])), 1.0e-8)
        pga_y = max(abs(float(record_features[1, 4])), 1.0e-8)
        interactions = [
            psa_x_at_x,
            psa_y_at_y,
            psa_x_at_y,
            psa_y_at_x,
            psa_x_at_x / pga_x,
            psa_y_at_y / pga_y,
        ]
        vector = list(group["motion_features"]) + modal_values + interactions
        if len(vector) != len(ENGINEERED_FEATURE_NAMES) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid engineered feature vector for {record.sample_id}")
        sample_vectors[record.sample_id] = [float(value) for value in vector]

    payload = {
        "version": FEATURE_CACHE_VERSION,
        "feature_names": ENGINEERED_FEATURE_NAMES,
        "sample_count": len(sample_vectors),
        "record_group_count": len(motion_groups),
        "spectrum_periods_sec": SPECTRUM_PERIODS_SEC.tolist(),
        "motion_groups": motion_groups,
        "samples": sample_vectors,
    }
    _atomic_json_write(payload, output_path)
    return {
        "sample_count": len(sample_vectors),
        "record_group_count": len(motion_groups),
        "feature_count": len(ENGINEERED_FEATURE_NAMES),
        "reused_motion_groups": reused,
        "computed_motion_groups": len(motion_groups) - reused,
        "output_path": str(output_path),
    }
