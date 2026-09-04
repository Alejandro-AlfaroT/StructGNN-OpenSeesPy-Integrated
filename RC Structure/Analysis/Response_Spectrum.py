"""
Analysis/Response_Spectrum.py
=============================

Elastic response spectra for ground-motion records.

This file existed as an empty placeholder. The only implementation lived in
the surrogate package, where the generator could not reach it, so intensity
scaling had no way to ask how hard a record shakes a structure at its own
period. That is the question the whole calibration turns on: a record's PGA
says little about what it does to an eight-story frame.

Unit system: acceleration in in/sec^2, periods in seconds, spectral
acceleration returned in g.
"""

from __future__ import annotations

import numpy as np


GRAVITY_IN_PER_SEC2 = 386.4

# Log-spaced period grid covering the range the generated frames occupy,
# with margin on both sides for interpolation near the ends.
SPECTRUM_PERIODS_SEC = np.logspace(np.log10(0.03), np.log10(5.0), 120)


def pseudo_spectral_acceleration(
    acceleration_in_per_sec2,
    dt_sec,
    periods_sec=SPECTRUM_PERIODS_SEC,
    damping_ratio=0.05,
):
    """5%-damped pseudo-spectral acceleration, vectorized over periods.

    Newmark average-acceleration integration of a single-degree-of-freedom
    oscillator at every period at once. The mean is removed first so a record
    with a small baseline offset does not bias the low-period end.

    Returns an array of PSA in g, one value per requested period.
    """
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

    beta, gamma = 0.25, 0.5
    c0 = 1.0 / (beta * dt_sec * dt_sec)
    c1 = gamma / (beta * dt_sec)
    c2 = 1.0 / (beta * dt_sec)
    c3 = 1.0 / (2.0 * beta) - 1.0
    c4 = gamma / beta - 1.0
    c5 = dt_sec * (gamma / (2.0 * beta) - 1.0)
    effective_stiffness = stiffness + c0 + damping * c1

    displacement = np.zeros_like(periods)
    velocity = np.zeros_like(periods)
    relative_acceleration = np.full_like(periods, -ground[0])
    maximum_displacement = np.zeros_like(periods)

    for ground_value in ground[1:]:
        effective_force = (
            -ground_value
            + c0 * displacement
            + c2 * velocity
            + c3 * relative_acceleration
            + damping * (c1 * displacement + c4 * velocity + c5 * relative_acceleration)
        )
        new_displacement = effective_force / effective_stiffness
        new_acceleration = (
            c0 * (new_displacement - displacement)
            - c2 * velocity
            - c3 * relative_acceleration
        )
        velocity = velocity + dt_sec * (
            (1.0 - gamma) * relative_acceleration + gamma * new_acceleration
        )
        displacement = new_displacement
        relative_acceleration = new_acceleration
        maximum_displacement = np.maximum(maximum_displacement, np.abs(displacement))

    return stiffness * maximum_displacement / GRAVITY_IN_PER_SEC2


def spectral_acceleration_at(acceleration_in_per_sec2, dt_sec, period_sec,
                             damping_ratio=0.05):
    """PSA at one period, in g."""
    spectrum = pseudo_spectral_acceleration(
        acceleration_in_per_sec2, dt_sec, damping_ratio=damping_ratio
    )
    return float(np.interp(period_sec, SPECTRUM_PERIODS_SEC, spectrum))


def geometric_mean_sa(spectrum_x, spectrum_y, period_x, period_y):
    """Geometric-mean spectral acceleration for a bidirectional record pair.

    Each component is read at the period of the direction it excites, then
    combined geometrically. This is the standard bidirectional intensity
    measure and is less sensitive to which component happens to be stronger
    than taking the maximum would be.
    """
    sa_x = float(np.interp(period_x, SPECTRUM_PERIODS_SEC, spectrum_x))
    sa_y = float(np.interp(period_y, SPECTRUM_PERIODS_SEC, spectrum_y))
    return float(np.sqrt(max(sa_x, 1.0e-12) * max(sa_y, 1.0e-12)))
