"""Gaussian heat pulse deposited on a strand through the solver's
user-defined heat mode (``heat_flux_mode: -2``): the solver calls
``strand.user_heat_function(conductor)`` inside ``get_heat`` at every step
with ``conductor.cond_time[-1]`` already advanced to t_{n+1}, and reads the
linear power density from ``strand.node_fields.EXTFLX[:, 0]`` (W/m).

    q(x, t) = Q_peak exp(-(x - x0)^2 / (2 sigma_x^2)) exp(-(t - t0)^2 / (2 sigma_t^2))

The pulse time is measured from a settable origin (the end of the settling
phase, see recorder.PhaseController), so the same object serves runs that
settle first. The peak amplitude is a fixed constant of the suite
(calibrated once with ``driver.py --calibrate-pulse`` for a peak strand
temperature of ~18 K on channel 2, then frozen here).
"""

from __future__ import annotations

import numpy as np

# Frozen after calibration (driver.py --calibrate-pulse, 2026-09-08, coarsest
# Tier-3a mesh, velocity formulation, target peak strand temperature 18 K:
# 2.0e4 W/m -> 43.6 K, 6.9e3 -> 29.5 K, 3.7e3 -> 24.3 K, 2.54e3 -> 19.6 K,
# extrapolated 2.27e3 W/m; see results/pulse_calibration.json).
PEAK_LINEAR_POWER = 2.274e3  # W/m
SIGMA_T = 0.020  # s
PULSE_CENTRE_TIME = 0.050  # s (relative to the pulse origin)
SIGMA_X_FRACTION = 0.05  # of the channel length


class GaussianPulse:
    def __init__(self, centre_x: float, sigma_x: float, centre_t: float = PULSE_CENTRE_TIME,
                 sigma_t: float = SIGMA_T, peak: float = PEAK_LINEAR_POWER):
        self.centre_x = centre_x
        self.sigma_x = sigma_x
        self.centre_t = centre_t
        self.sigma_t = sigma_t
        self.peak = peak
        self.time_origin = 0.0  # absolute time of the pulse origin
        self.active = False  # off until the phase controller enables it

    def linear_power(self, x, t_relative):
        shape_x = np.exp(-0.5 * ((x - self.centre_x) / self.sigma_x) ** 2)
        shape_t = np.exp(-0.5 * ((t_relative - self.centre_t) / self.sigma_t) ** 2)
        return self.peak * shape_x * shape_t

    def total_energy(self, length_scale_check=None) -> float:
        """Deposited energy integral (Gaussian in x and t): Q 2 pi sigma_x sigma_t."""
        return self.peak * 2.0 * np.pi * self.sigma_x * self.sigma_t

    def install(self, strand) -> None:
        """Attach as the strand's user heat function."""
        pulse = self

        def user_heat_function(conductor):
            fields = strand.node_fields
            if "EXTFLX" not in fields:
                fields.EXTFLX = np.zeros((conductor.mesh.number_of_nodes, 2))
            if not pulse.active:
                fields.EXTFLX[:, 0] = 0.0
                return
            t_relative = conductor.cond_time[-1] - pulse.time_origin
            fields.EXTFLX[:, 0] = pulse.linear_power(
                conductor.mesh.node_coordinates, t_relative
            )

        strand.user_heat_function = user_heat_function
