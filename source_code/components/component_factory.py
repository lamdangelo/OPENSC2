"""
This module contains the component factory class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from components.component import Component
from components.component_flags import ComponentType, get_component_type
from components.fluid.fluid_component import FluidComponent
from components.jacket.jacket_component import JacketComponent
from components.solid.stack_component import StackComponent
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_stabilizer_component import StrandStabilizerComponent

from components.component_collection import ComponentInventory
from adapters.input.component_loaders import (
    FluidComponentInputLoader,
    JacketComponentInputLoader,
    StackComponentInputLoader,
    StrandMixedInputLoader,
    StrandStabilizerInputLoader,
)

from ports.input_source import InputSource

if TYPE_CHECKING:
    from simulation import Simulation
    from conductor.conductor import Conductor


@dataclass(slots=True)
class ComponentBuildContext:
    simulation: Simulation
    conductor: Conductor
    input_source: InputSource


class ComponentFactory:

    def __init__(self, context: ComponentBuildContext):
        self.context = context


    def create_components(
        self,
        kind_obj: str,
        sheet_name: str,
        identifiers: list[str],
    ) -> list[Component]:
        """Construct component instances from their kind, legacy sheet name
        and identifiers; the raw inputs come from the context's InputSource."""
        component_type = get_component_type(kind_obj)
        source = self.context.input_source
        sim = self.context.simulation
        cond = self.context.conductor
        counter = cond.counter

        components = ComponentInventory.empty()

        for identifier in identifiers:

            if component_type is ComponentType.FLUID:
                loader = FluidComponentInputLoader(source, counter, sheet_name, identifier)
                components.add(FluidComponent(
                    identifier,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                ))

            elif component_type is ComponentType.STACK:
                loader = StackComponentInputLoader(source, counter, identifier, sheet_name)
                components.add(StackComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.STRAND_MIXED:
                loader = StrandMixedInputLoader(source, counter, identifier, sheet_name)
                components.add(StrandMixedComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.STRAND_STABILIZER:
                loader = StrandStabilizerInputLoader(source, counter, identifier, sheet_name)
                components.add(StrandStabilizerComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

            elif component_type is ComponentType.JACKET:
                loader = JacketComponentInputLoader(source, counter, identifier, sheet_name)
                components.add(JacketComponent(
                    sim, identifier, kind_obj,
                    loader.load_input_file(),
                    loader.load_operations_file(),
                    cond,
                ))

        return components.all_components
