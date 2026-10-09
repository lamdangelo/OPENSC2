"""
This module validates the read input data.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from opensc2.conductor.conductor_inputs import (
    ConductorInputs,
    ConductorCoupling, 
    ConductorOperations
)

def check_coupling_sheet_names(coupling_sheet_names: List[str]):
    """
    Checks sheet names in input file conductor_coupling.xlsx.

    Args:
        coupling_sheet_names: list of sheet names

    Raises:
        KeyError: any of the sheet names in file conductor_coupling.xlsx is not consistent with the reference ones.
    """

    ref_sheet_names = {
        "contact_perimeter_flag",
        "contact_perimeter",
        "HTC_choice",
        "contact_HTC",
        "thermal_contact_resistance",
        "HTC_multiplier",
        "electric_conductance_mode",
        "electric_conductance",
        "open_perimeter_fract",
        "interf_thickness",
        "trans_transp_multiplier",
        "view_factors",
        }
    
    wrong_sheet_names = list()

    # Loop to check sheet names in input file conductor_coupling.xlsx.
    for sheet_name in coupling_sheet_names:
        if sheet_name not in ref_sheet_names:
            wrong_sheet_names.append(sheet_name)

    return len(wrong_sheet_names) == 0


def check_for_unphysical_coupling_data(coupling_data: ConductorCoupling) -> bool:

    """Checks whether there are negative values where there should not be due to physics.
    Args:
        coupling_data: ConductorCoupling object
    
    Raises:
        ValueError: if unphysical values are provided by the user.
    """
    valid = coupling_data.is_physically_valid()
    if not valid:
        # negative_idx is not empty: raise ValueError.
        raise ValueError(f"Unphysical data encountered that should be >= 0.0. Please check coupling data.")
    return valid 



def check_equipotential_surface_coordinate(operations: ConductorOperations, z_length: float):
    """Make consistency checks on input values EQUIPOTENTIAL_SURFACE_COORDINATE and EQUIPOTENTIAL_SURFACE_NUMBER.

    Args:
        operations : ConductorOperations
        z_length : length of the conductor

    Raises:
        ValueError: raise error if the number of assinged coordinates is different from the number of declared equipotential surfaces.
        ValueError: raises value error if equipotential coordinate is exceeded conductor length.
        ValueError: raises value error if equipotential coordinate is negative.
    """

    if len(operations.equipotential_surface_coordinates) != operations.number_of_equipotential_surfaces:
        raise ValueError(
            f"The number of the equipotential surfaces coordinates must be equal to the number of declared equipotential surfaces!"
        )
    if np.max(operations.equipotential_surface_coordinates) > z_length:
        raise ValueError(
            f"Equipotential surface coordinate cannot exceed conductor length!"
        )
    if np.min(operations.equipotential_surface_coordinates) < 0.0:
        raise ValueError(
            f"Equipotential surface coordinate must be positive!"
        )


def check_consistency_regarding_external_contact_perimeter(file_paths: ConductorInputs, 
                                                           coupling: ConductorCoupling) -> bool:
    # Check if user declared variable contact perimeter for some of the 
    # conductor components.
    is_consistent = (
        file_paths.external_contact_perimeter_exists() 
        and coupling.interface_thickness.check_for_negative_data()
    )
    if not is_consistent:
        raise ValueError("User provides a file for variable contact perimeter but in sheet contact_perimeter_flag none of the interfaces has the valid flag for the variable contact perimeter (-1).")
    else:
        return is_consistent 


def check_coarse_region_quality(
    last_pitch: float,
    n_elem: int,
    region_length: float,
    is_left: bool = True,
):
    """Check that the uniform-region pitch is not finer than the last coarsening pitch.

    Args:
        last_pitch: last element size used during coarsening.
        n_elem: number of available elements in the uniform region.
        region_length: length of the uniform region.
        is_left: True when checking the region left of the refined zone.

    Raises:
        ValueError: if the uniform pitch is finer than the coarsening pitch.
    """
    uniform_pitch = region_length / n_elem
    if uniform_pitch < last_pitch:
        if is_left:
            side = "left"
        else:
            side = "right"
        raise ValueError(
            f"Bad spatial discretization.\n"
            f"Discretization pitch in the uniform region to the {side} of the "
            f"refined region is lower than the last discretization pitch used for "
            f"coarsening. Please consider using a larger DXINCRE_{side.upper()} "
            f"or a different number of total elements and elements used in the "
            f"refined region or a combination of both."
        )