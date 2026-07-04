"""
This module provides an interface to the CoolProp module.
"""

import os

import CoolProp
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI
import numpy as np
from scipy.interpolate import RectBivariateSpline

from hydraulics.hydraulic_flags import FluidType

# Map from the CoolProp high-level property alias (as used by PropsSI) to the
# corresponding low-level AbstractState getter name.
_ABSTRACT_STATE_GETTER_NAMES = {
    "isobaric_expansion_coefficient": "isobaric_expansion_coefficient",
    "isothermal_compressibility": "isothermal_compressibility",
    "Prandtl": "Prandtl",
    "Dmass": "rhomass",
    "viscosity": "viscosity",
    "Hmass": "hmass",
    "Cpmass": "cpmass",
    "Cvmass": "cvmass",
    "speed_of_sound": "speed_sound",
    "conductivity": "conductivity",
}

# One reusable low-level state object per fluid (creation is expensive).
_abstract_state_cache: dict = {}


def _get_abstract_state(fluid_type: FluidType) -> AbstractState:
    if fluid_type not in _abstract_state_cache:
        _abstract_state_cache[fluid_type] = AbstractState("HEOS", fluid_type.value)
    return _abstract_state_cache[fluid_type]


# --- Tabular property evaluation -------------------------------------------
#
# The full Helmholtz equation of state costs ~7 us per point in a Python
# loop and dominates the thermal-hydraulic step. With this flag enabled the
# fast-path properties are instead read from a bicubic spline fit on a
# log(T) x log(p) grid, built once per fluid from the full EOS (~1 s) and
# evaluated vectorized. Agreement with the EOS is ~1e-6 relative at the
# 99th percentile; the worst case is ~1e-2 directly on the pseudocritical
# specific-heat peak. Queries outside the table range are clamped to its
# boundary. Isolated EOS failures on the grid (CoolProp's helium
# conductivity model returns NaN near the critical point) are filled from
# finite neighbours, which also makes the table more robust than the
# direct EOS calls it replaces.
USE_TABULAR_PROPERTIES = True
_TABLE_TEMPERATURE_RANGE = (2.4, 350.0)  # K
_TABLE_PRESSURE_RANGE = (1.0e5, 2.5e7)  # Pa
_TABLE_POINTS = (400, 300)  # (temperature, pressure) grid points

_property_table_cache: dict = {}


class _FluidPropertyTable:
    """Bicubic spline fit of all fast-path properties for one fluid."""

    def __init__(self, fluid_type: FluidType):
        temperature_grid = np.geomspace(*_TABLE_TEMPERATURE_RANGE, _TABLE_POINTS[0])
        pressure_grid = np.geomspace(*_TABLE_PRESSURE_RANGE, _TABLE_POINTS[1])
        state = AbstractState("HEOS", fluid_type.value)
        getters = [
            getattr(state, getter_name)
            for getter_name in _ABSTRACT_STATE_GETTER_NAMES.values()
        ]
        values = np.empty((temperature_grid.size, pressure_grid.size, len(getters)))
        for i, temperature in enumerate(temperature_grid):
            for j, pressure in enumerate(pressure_grid):
                try:
                    state.update(CoolProp.PT_INPUTS, pressure, temperature)
                    for k, getter in enumerate(getters):
                        values[i, j, k] = getter()
                except ValueError:
                    values[i, j, :] = np.nan
        for k in range(len(getters)):
            values[:, :, k] = _fill_from_finite_neighbours(values[:, :, k])

        self._log_temperature_bounds = np.log(temperature_grid[[0, -1]])
        self._log_pressure_bounds = np.log(pressure_grid[[0, -1]])
        log_temperature = np.log(temperature_grid)
        log_pressure = np.log(pressure_grid)
        self._splines = {
            alias: RectBivariateSpline(
                log_temperature, log_pressure, values[:, :, k], kx=3, ky=3
            )
            for k, alias in enumerate(_ABSTRACT_STATE_GETTER_NAMES)
        }

    def evaluate(self, alias: str, temperature: np.ndarray,
                 pressure: np.ndarray) -> np.ndarray:
        log_temperature = np.clip(np.log(temperature),
                                  *self._log_temperature_bounds)
        log_pressure = np.clip(np.log(pressure), *self._log_pressure_bounds)
        return self._splines[alias].ev(log_temperature, log_pressure)


def _fill_from_finite_neighbours(grid: np.ndarray) -> np.ndarray:
    """Replace NaN grid entries by the mean of their finite neighbours,
    iterating until none remain (isolated holes converge immediately)."""
    grid = grid.copy()
    while True:
        invalid = ~np.isfinite(grid)
        if not invalid.any():
            return grid
        padded = np.pad(grid, 1, constant_values=np.nan)
        neighbours = np.stack(
            [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
        )
        with np.errstate(invalid="ignore"):
            neighbour_mean = np.nanmean(neighbours, axis=0)
        fillable = invalid & np.isfinite(neighbour_mean)
        if not fillable.any():
            raise ValueError("Property grid contains an unfillable NaN region.")
        grid[fillable] = neighbour_mean[fillable]


def _get_property_table(fluid_type: FluidType) -> _FluidPropertyTable:
    if fluid_type not in _property_table_cache:
        _property_table_cache[fluid_type] = _FluidPropertyTable(fluid_type)
    return _property_table_cache[fluid_type]


# CoolProp's PT flash fails (with a misleading "p is not a valid number")
# when the temperature lands within float noise (~1e-8 K) of the critical
# temperature, e.g. helium compression-heated smoothly through 5.1953 K.
# Shift such points by a physically invisible 1e-7 K.
_CRITICAL_TEMPERATURE_WINDOW = 1e-7  # K
_critical_temperature_cache: dict = {}


def _shift_temperature_off_critical(fluid_type: FluidType,
                                    temperature: np.ndarray) -> np.ndarray:
    if fluid_type not in _critical_temperature_cache:
        _critical_temperature_cache[fluid_type] = PropsSI(
            "Tcrit", fluid_type.value
        )
    critical_temperature = _critical_temperature_cache[fluid_type]
    near_critical = (
        np.abs(temperature - critical_temperature)
        < _CRITICAL_TEMPERATURE_WINDOW
    )
    if np.any(near_critical):
        temperature = np.where(
            near_critical,
            critical_temperature + _CRITICAL_TEMPERATURE_WINDOW,
            temperature,
        )
    return temperature


def compute_isobaric_expansion_coefficient(fluid_type: FluidType, 
                                           temperature: np.ndarray, 
                                           pressure: np.ndarray) -> np.ndarray:
    """
    Computes the isobaric expansion coefficient.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            isobaric expansion coefficient in 1/K
    """
    return PropsSI("isobaric_expansion_coefficient", "T", temperature, 
                   "P", pressure, fluid_type.value)


def compute_isobaric_specific_heat(fluid_type: FluidType, 
                                   temperature: np.ndarray, 
                                   pressure: np.ndarray) -> np.ndarray:
    """
    Computes the isobaric specific heat (mass specific constrant pressure
    specific heat).
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            isobaric specific heat in J/(kg K)
    """
    return PropsSI("Cpmass", "T", temperature, "P", pressure, fluid_type.value)


def compute_isochoric_specific_heat(fluid_type: FluidType, 
                                    temperature: np.ndarray, 
                                    pressure: np.ndarray) -> np.ndarray:
    """
    Computes the isochoric specific heat (mass specific constant volume 
    specific heat).
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            isochoric specific heat in J/(kg K)
    """
    return PropsSI("Cvmass", "T", temperature, "P", pressure, fluid_type.value)


def compute_isothermal_compressibility(fluid_type: FluidType, 
                                       temperature: np.ndarray, 
                                       pressure: np.ndarray) -> np.ndarray:
    """
    Computes the isothermal compressibility.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            isothermal compressibility in 1/Pa
    """
    return PropsSI("isothermal_compressibility", "T", temperature, 
                   "P", pressure, fluid_type.value)


def compute_mass_density(fluid_type: FluidType, 
                         temperature: np.ndarray, 
                         pressure: np.ndarray) -> np.ndarray:
    """
    Computes mass density.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            mass density in kg/m³
    """
    scalar_inputs = np.ndim(temperature) == 0 and np.ndim(pressure) == 0
    temperature = _shift_temperature_off_critical(
        fluid_type, np.asarray(temperature, dtype=float)
    )
    if USE_TABULAR_PROPERTIES:
        values = _get_property_table(fluid_type).evaluate(
            "Dmass", temperature, np.asarray(pressure, dtype=float)
        )
        # Keep the PropsSI return contract: a float for scalar inputs, a 0-d
        # array for length-1 inputs.
        if scalar_inputs:
            return float(values)
        if np.size(values) == 1:
            return np.asarray(values).reshape(())
        return values
    return PropsSI("Dmass", "T", temperature, "P", pressure, fluid_type.value)


def compute_mass_specific_enthalpy(fluid_type: FluidType, 
                                   temperature: np.ndarray, 
                                   pressure: np.ndarray) -> np.ndarray:
    """
    Computes the mass specific enthalpy.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            mass specifc enthalpy in J/kg
    """
    return PropsSI("Hmass", "T", temperature, "P", pressure, fluid_type.value)


def compute_prandtl_number(fluid_type: FluidType, 
                           temperature: np.ndarray, 
                           pressure: np.ndarray) -> np.ndarray:
    """
    Computes the Prandtl number.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            Prandtl number (dimensionless)
    """
    return PropsSI("Prandtl", "T", temperature, "P", pressure, fluid_type.value)


def compute_speed_of_sound(fluid_type: FluidType, 
                           temperature: np.ndarray, 
                           pressure: np.ndarray) -> np.ndarray:
    """
    Computes the speed of sound.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            speed of sound in m/s
    """
    return PropsSI("speed_of_sound", "T", temperature, "P", pressure, fluid_type.value)


def compute_thermal_conductivity(fluid_type: FluidType, 
                                 temperature: np.ndarray, 
                                 pressure: np.ndarray) -> np.ndarray:
    """
    Computes the thermal conductivity.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            thermal conductivity in W/(m K)
    """
    return PropsSI("conductivity", "T", temperature, "P", pressure, fluid_type.value)


def compute_viscosity(fluid_type: FluidType, 
                      temperature: np.ndarray, 
                      pressure: np.ndarray) -> np.ndarray:
    """
    Computes viscosity.
    
    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        temperature : np.ndarray 
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray 
            viscosity in Pa s
    """
    return PropsSI("viscosity", "T", temperature, "P", pressure, fluid_type.value)

def compute_property(fluid_type: FluidType,
                     property_alias: str,
                     temperature: np.ndarray,
                     pressure: np.ndarray) -> np.ndarray:
    """
    Computes a generic fluid property from its CoolProp alias.

    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        property_alias: str
            CoolProp name of the property to evaluate (e.g. "Dmass",
            "viscosity", "Hmass")
        temperature : np.ndarray
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        np.ndarray
            the requested property in its CoolProp unit
    """
    return PropsSI(property_alias, "T", temperature, "P", pressure, fluid_type.value)


def compute_properties(fluid_type: FluidType,
                       property_aliases: dict,
                       temperature: np.ndarray,
                       pressure: np.ndarray) -> dict:
    """
    Computes several fluid properties with a single equation-of-state flash
    per point.

    The high-level ``PropsSI`` interface repeats the full (T, P) flash for
    every requested property; the low-level ``AbstractState`` interface
    updates the thermodynamic state once per point and then reads all the
    requested properties from it, which is roughly ``len(property_aliases)``
    times faster. Values are identical to ``PropsSI`` since both use the
    same HEOS backend.

    Parameters
    ----------
        fluid_type: FluidType
            the coolant fluid, used as CoolProp fluid name
        property_aliases: dict
            mapping ``{result_name: CoolProp property alias}``; any alias
            without a known low-level getter falls back to ``PropsSI``
        temperature : np.ndarray
            nodal temperature values in K
        pressure: np.ndarray
            nodal pressure values in Pa

    Returns
    -------
        dict
            mapping ``{result_name: np.ndarray}`` with the requested
            properties in their CoolProp units
    """
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))
    temperature, pressure = np.broadcast_arrays(temperature, pressure)
    temperature = _shift_temperature_off_critical(fluid_type, temperature)

    bad = ~(np.isfinite(temperature) & np.isfinite(pressure))
    if bad.any():
        bad_points = np.nonzero(bad)[0]
        dump_path = os.path.join(os.getcwd(), "nonfinite_coolprop_inputs.npz")
        np.savez(
            dump_path,
            temperature=temperature,
            pressure=pressure,
            bad_points=bad_points,
        )
        raise RuntimeError(
            f"Non-finite CoolProp inputs at {bad_points.size} of "
            f"{temperature.size} points (first at index {bad_points[0]}: "
            f"T = {temperature[bad_points[0]]}, p = {pressure[bad_points[0]]}). "
            f"Arrays dumped to {dump_path}."
        )

    state = _get_abstract_state(fluid_type)
    table = _get_property_table(fluid_type) if USE_TABULAR_PROPERTIES else None

    fast_names = []
    getters = []
    results = {}
    for result_name, alias in property_aliases.items():
        getter_name = _ABSTRACT_STATE_GETTER_NAMES.get(alias)
        if getter_name is None:
            # Unknown alias: keep the robust high-level path for it.
            results[result_name] = compute_property(
                fluid_type, alias, temperature, pressure
            )
        elif table is not None:
            results[result_name] = table.evaluate(alias, temperature, pressure)
        else:
            fast_names.append(result_name)
            getters.append(getattr(state, getter_name))

    if not getters:
        return results

    values = np.empty((temperature.size, len(getters)))
    for point, (pressure_value, temperature_value) in enumerate(
        zip(pressure, temperature)
    ):
        try:
            state.update(CoolProp.PT_INPUTS, pressure_value, temperature_value)
        except ValueError as err:
            dump_path = os.path.join(os.getcwd(), "failed_coolprop_update.npz")
            np.savez(
                dump_path,
                temperature=temperature,
                pressure=pressure,
                failed_point=point,
            )
            raise RuntimeError(
                f"CoolProp state update failed at point {point} of "
                f"{temperature.size}: T = {temperature_value!r} K, "
                f"p = {pressure_value!r} Pa. Arrays dumped to {dump_path}."
            ) from err
        for column, getter in enumerate(getters):
            values[point, column] = getter()

    for column, result_name in enumerate(fast_names):
        results[result_name] = values[:, column]

    return results
