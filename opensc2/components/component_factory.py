"""Builds component objects from their typed definitions.

Pure domain code: no file reading. The definitions (dataclasses of inputs
and operations) come from adapters.input.conductor_loader.
"""

from __future__ import annotations

from opensc2.components.component import Component
from opensc2.components.component_build import ComponentBuildSettings
from opensc2.components.component_flags import ComponentType
from opensc2.components.fluid.fluid_component import FluidComponent
from opensc2.components.jacket.jacket_component import JacketComponent
from opensc2.components.solid.stack_component import StackComponent
from opensc2.components.solid.strand_mixed_component import StrandMixedComponent
from opensc2.components.solid.strand_stabilizer_component import StrandStabilizerComponent
from opensc2.conductor.conductor_definition import ComponentDefinition

_SOLID_CLASSES = {
    ComponentType.STACK: StackComponent,
    ComponentType.STRAND_MIXED: StrandMixedComponent,
    ComponentType.STRAND_STABILIZER: StrandStabilizerComponent,
    ComponentType.JACKET: JacketComponent,
}


def build_component(
    definition: ComponentDefinition, build: ComponentBuildSettings
) -> Component:
    if definition.component_type is ComponentType.FLUID:
        return FluidComponent(
            definition.identifier, definition.inputs, definition.operations,
            external_inputs=build.external_inputs,
        )
    try:
        component_class = _SOLID_CLASSES[definition.component_type]
    except KeyError:
        raise ValueError(f"Unknown component type {definition.component_type!r}.")
    return component_class(
        definition.identifier,
        definition.kind,
        definition.inputs,
        definition.operations,
        build,
    )
