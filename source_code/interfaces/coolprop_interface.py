"""
This module provides an interface to the CoolProp module.
"""

import CoolProp
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI
import numpy as np

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

    state = _get_abstract_state(fluid_type)

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
        else:
            fast_names.append(result_name)
            getters.append(getattr(state, getter_name))

    values = np.empty((temperature.size, len(getters)))
    for point, (pressure_value, temperature_value) in enumerate(
        zip(pressure, temperature)
    ):
        state.update(CoolProp.PT_INPUTS, pressure_value, temperature_value)
        for column, getter in enumerate(getters):
            values[point, column] = getter()

    for column, result_name in enumerate(fast_names):
        results[result_name] = values[:, column]

    return results
