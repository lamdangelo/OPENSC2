"""
This module contains the component factory class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from components.component import Component
from components.component_flags import ComponentType, get_component_type
from components.fluid.fluid_component import FluidComponent
from components.fluid.fluid_component_inputs import FluidComponentInputLoader
from components.jacket.jacket_component import JacketComponent
from components.jacket.jacket_component_inputs import JacketComponentInputLoader
from components.solid.stack_component import StackComponent
from components.solid.stack_component_inputs import StackComponentInputLoader
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_mixed_component_inputs import StrandMixedInputLoader
from components.solid.strand_stabilizer_component import StrandStabilizerComponent
from components.solid.strand_stabilizer_component_inputs import StrandStabilizerInputLoader

from components.component_collection import ComponentInventory 

if TYPE_CHECKING:
    from simulation import Simulation
    from conductor.conductor import Conductor


@dataclass(slots=True)
class ComponentBuildContext:
    simulation: Simulation
    conductor: Conductor


class ComponentFactory:

    def __init__(self, context: ComponentBuildContext):
        self.context = context


    def create(self, sheet, num_components: int, dict_file_path: dict) -> list[Component]:
        """Read inputs from Excel via the appropriate loader and return a list of
        fully constructed component instances — one per component column in the sheet.

        Args:
            sheet: openpyxl Worksheet; provides component kind (A1), count (B1),
                   and identifier values (row 3, columns 5+).
            num_components: number of component instances to create.
            dict_file_path: dict with keys "input" and "operation" (Path objects).

        Returns:
            list of Component instances.
        """
        kind_obj: str = sheet.cell(row=1, column=1).value
        component_type = get_component_type(kind_obj)
        sheet_name: str = sheet.title
        input_path = dict_file_path["input"]
        ops_path = dict_file_path["operation"]
        sim = self.context.simulation
        cond = self.context.conductor

        components = ComponentInventory.empty()

        for ii in range(1, num_components + 1):
            identifier: str = sheet.cell(row=3, column=4 + ii).value

            if component_type is ComponentType.FLUID:
                loader = FluidComponentInputLoader(input_path, ops_path, identifier)
                components.add(FluidComponent(
                    identifier,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                ))

            elif component_type is ComponentType.STACK:
                loader = StackComponentInputLoader(input_path, ops_path, identifier, sheet_name)
                components.add(StackComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.STRAND_MIXED:
                loader = StrandMixedInputLoader(input_path, ops_path, identifier, sheet_name)
                components.add(StrandMixedComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.STRAND_STABILIZER:
                loader = StrandStabilizerInputLoader(input_path, ops_path, identifier, sheet_name)
                components.add(StrandStabilizerComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.JACKET:
                loader = JacketComponentInputLoader(input_path, ops_path, identifier, sheet_name)
                components.add(JacketComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

        return components.all_components
