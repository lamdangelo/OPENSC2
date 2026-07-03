"""
This module provides an interface to the CoolProp module.
"""

from CoolProp.CoolProp import PropsSI
import numpy as np 

from hydraulics.hydraulic_flags import FluidType


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
