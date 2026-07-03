"""
This module contains the ComponentCollection and ComponentInventory classes.
"""

from dataclasses import dataclass 
from typing import Generic, TypeVar

from components.component import Component 

from components.fluid.fluid_component import FluidComponent
from components.jacket.jacket_component import JacketComponent
from components.solid.stack_component import StackComponent
from components.solid.strand_component import StrandComponent
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_stabilizer_component import StrandStabilizerComponent
from components.solid.solid_component import SolidComponent  # TODO: put component class files in subfolder components

T = TypeVar("T")


class ComponentCollection(Generic[T]):
    """Class that allows to organize the components that build the conductor in collection(s).
    A collection is characterized by:
        1) a list of objects of the same class T;
        2) the number of objects that make up the collection;
        3) the collection type T
    """

    def __init__(self, dtype: type[T] = None):
        """Make an instance of class ComponentCollection of type T.

        Args:
            dtype: type of the collection (optional for backward-compat callers
                   that pass a string kind tag; the string is stored as-is).
        """
        self.collection: list[T] = []
        self.component_type: type[T] = dtype
        self.number: int = 0  # kept for the dict-based inventory in conductor.py


    def __getitem__(self, index: int) -> T:
        return self.collection[index]
    

    def __iter__(self):
        return iter(self.collection)
    

    def __len__(self):
        return len(self.collection)


    def add(self, new_component: T) -> None:
        self.collection.append(new_component)
        self.number += 1


@dataclass(slots=True)
class ComponentInventory:
    """
    Class that stores multiple component collections.
    """
    fluids: ComponentCollection[FluidComponent]
    mixed_strands: ComponentCollection[StrandMixedComponent]
    stabilizer_strands: ComponentCollection[StrandStabilizerComponent]
    stacks: ComponentCollection[StackComponent]
    jackets: ComponentCollection[JacketComponent]
    strands: ComponentCollection[StrandComponent]
    solids: ComponentCollection[SolidComponent]
    all_components: ComponentCollection[Component]


    @classmethod 
    def empty(cls):
        """Creates a component inventory instance with empty component collections."""
        return cls(
            fluids=ComponentCollection(FluidComponent),
            mixed_strands=ComponentCollection(StrandMixedComponent),
            stabilizer_strands=ComponentCollection(StrandStabilizerComponent),
            stacks=ComponentCollection(StackComponent),
            jackets=ComponentCollection(JacketComponent),
            strands=ComponentCollection(StrandComponent),  # contains mixed and stabilizer strand and stack components
            solids=ComponentCollection(SolidComponent),  # contains every non-fluid component
            all_components=ComponentCollection(Component)  # contains every component 
        )


    def add(self, component: Component) -> None:
        """
        Add the component to the list of all components and to the corresponding 
        component-specific list.
        """
        self.all_components.add(component)

        if isinstance(component, FluidComponent):
            self.fluids.add(component)

        elif isinstance(component, SolidComponent):
            self.solids.add(component)

            if isinstance(component, StrandComponent):
                self.strands.add(component)
                if isinstance(component, StrandMixedComponent):
                    self.mixed_strands.add(component)
                elif isinstance(component, StrandStabilizerComponent):
                    self.stabilizer_strands.add(component)
                else:  # stack component 
                    self.stacks.add(component)

            elif isinstance(component, JacketComponent):
                self.jackets.add(component)
            elif isinstance(component, SolidComponent):
                self.solids.add(component)

        else:
            ValueError(f"Unknown component type {type(component).__name__}.")