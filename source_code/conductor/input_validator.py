"""
This module validates the read input data.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from openpyxl import load_workbook
import pandas as pd

from conductor.conductor_inputs import (
    ConductorInputs,
    ConductorCoupling, 
    ConductorOperations
)

class ConductorInputValidator:
    """Validate conductor definition, grid, diagnostic, and coupling input files."""

    @staticmethod
    def validate_component_workbooks(
        simulation: object,
        input_path: Path,
        operation_path: Path,
    ) -> tuple[Any, Any, List[str]]:
        """Validate conductor component input and operation workbooks."""
        wb_input = load_workbook(input_path, data_only=True)
        wb_operations = load_workbook(operation_path, data_only=True)

        list_of_sheets = wb_input.sheetnames
        for sheet_name in list_of_sheets:
            sheet = wb_input[sheet_name]
            sheet_operation = wb_operations[sheet_name]
            check_object_number(
                simulation,
                str(input_path),
                str(operation_path),
                sheet,
                sheet_operation,
            )
            check_repeated_headings(input_path, sheet)
            check_repeated_headings(operation_path, sheet_operation)
            check_headers(
                simulation,
                str(input_path),
                str(operation_path),
                sheet,
                sheet_operation,
            )

        return wb_input, wb_operations, list_of_sheets


    @staticmethod
    def validate_conductor_definition_files(
        simulation: object,
        conductor_definition_path: Path,
        grid_path: Path,
        diagnostic_path: Path,
    ) -> Tuple[Any, Any, Any, List[Any]]:
        """Validate the conductor definition, grid, and diagnostic input files.

        This method checks repeated headings and object counts across the
        conductor_definition, grid, and diagnostic Excel files.
        """
        conductor_definition_workbook = load_workbook(
            conductor_definition_path, data_only=True
        )
        grid_workbook = load_workbook(grid_path, data_only=True)
        diagnostic_workbook = load_workbook(diagnostic_path, data_only=True)

        list_conductor_sheets = [
            conductor_definition_workbook["CONDUCTOR_files"],
            conductor_definition_workbook["CONDUCTOR_input"],
            conductor_definition_workbook["CONDUCTOR_operation"],
        ]

        for sheet in list_conductor_sheets:
            check_repeated_headings(conductor_definition_path, sheet)

        check_repeated_headings(grid_path, grid_workbook["GRID"])

        for sheet in diagnostic_workbook:
            check_repeated_headings(diagnostic_path, sheet)

        for cond_sheet in list_conductor_sheets:
            check_object_number(
                simulation,
                str(conductor_definition_path),
                str(grid_path),
                cond_sheet,
                grid_workbook["GRID"],
            )
            check_object_number(
                simulation,
                str(conductor_definition_path),
                str(diagnostic_path),
                cond_sheet,
                diagnostic_workbook["Spatial_distribution"],
            )
            check_object_number(
                simulation,
                str(conductor_definition_path),
                str(diagnostic_path),
                cond_sheet,
                diagnostic_workbook["Time_evolution"],
            )

        return (
            conductor_definition_workbook,
            grid_workbook,
            diagnostic_workbook,
            list_conductor_sheets,
        )


def check_headers(cond, path_input, path_operation, sheet_input, sheet_operation):
    """[summary]

    Args:
        cond ([type]): [description]
        path_input ([type]): [description]
        path_operation ([type]): [description]
        sheet_input ([type]): [description]
        sheet_operation ([type]): [description]

    Raises:
        SyntaxError: [description]
    """
    header_input = list(
        pd.read_excel(
            path_input, sheet_name=sheet_input.title, skiprows=2, header=0, index_col=0
        ).columns
    )
    header_operation = list(
        pd.read_excel(
            path_operation,
            sheet_name=sheet_operation.title,
            skiprows=2,
            header=0,
            index_col=0,
        ).columns
    )

    for ii in range(len(header_input)):
        if header_input[ii] != header_operation[ii]:
            raise SyntaxError(
                f"ERROR in method {cond.__init__.__name__} of class {cond.__class__.__name__}: object headers in {sheet_input} of file {cond.file_paths.structure_elements_path} and in {sheet_operation} of file {cond.file_paths.operation_path} should be the same!\nCompare the third row, column {ii+1} of those sheets.\n"
            )
        


def check_object_number(object, path_1, path_2, sheet_1, sheet_2):
    """[summary]

    Raises:
        ValueError: [description]
    """
    dict_method = dict(
        Simulation="conductor_instance", Conductor="conductor_components_instance"
    )
    if int(sheet_1.cell(row=1, column=2).value) != int(
        sheet_2.cell(row=1, column=2).value
    ):
        raise ValueError(
            f"ERROR in class {object.__class__.__name__} method {dict_method[object.__class__.__name__]}: number of objects defined in file {path_1} sheet {sheet_1.title} and in file {path_2} sheet {sheet_2.title} must be the same.\nPlease compare cell B1 of those sheets.\n"
        )


def check_repeated_headings(input_file, sheet):
    """[summary]

    Args:
        input_file ([type]): [description]
        sheet ([type]): [description]

    Raises:
        ValueError: [description]
    """

    # Get the columns names that user can define (except the first four ones that are fixed). The variable columns is a tuple.
    columns = list(
        sheet.iter_rows(
            min_row=3,
            max_row=3,
            min_col=sheet.min_column,
            max_col=sheet.max_column,
            values_only=True,
        )
    )[0][4:]
    # Buil dictionay exploiting dict comprehension: each key as the numer of repetitions of the column as the corresponding value.
    dict_colum = {column: columns.count(column) for column in columns}
    # Raise error message
    if max(list(dict_colum.values())) > 1:
        raise ValueError(
            f"ERROR! Different objects of the same kind ({sheet['A1'].value}) can not have the same identifier.\nUser defines the following:\n{dict_colum.items()}.\nPlease check the headers in sheet {sheet.title} of file {input_file}"
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