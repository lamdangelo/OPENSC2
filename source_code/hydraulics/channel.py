"""
This module contains the Channel class. This class stores the geometric information
of the fluid component.
"""

import numpy as np

from hydraulics.hydraulic_flags import FlowDirection
from components.fluid.fluid_component_inputs import (
    FluidComponentInputs,
    FluidComponentOperations
)

from hydraulics.friction_factor_factory import FrictionFactorFactory
from hydraulics.friction_factor_models import FrictionFactors

from physical_fields.physical_field import TimeEvolution
from thermal.heat_transfer_factory import HeatTransferFactory


class LocationCache:
    """Stores one value per evaluation location, addressed with the flag_nodal
    convention used across the code base: True for the mesh nodes, False for
    the Gauss points, None for the flow-initialization scratch value.
    """

    __slots__ = ("nodal", "gauss", "initialization")

    def __init__(self):
        self.nodal = None
        self.gauss = None
        self.initialization = None

    @staticmethod
    def _attribute_name(flag_nodal) -> str:
        if flag_nodal is True:
            return "nodal"
        if flag_nodal is False:
            return "gauss"
        if flag_nodal is None:
            return "initialization"
        raise KeyError(f"Invalid location flag: {flag_nodal!r}")

    def __getitem__(self, flag_nodal):
        return getattr(self, self._attribute_name(flag_nodal))

    def __setitem__(self, flag_nodal, value) -> None:
        setattr(self, self._attribute_name(flag_nodal), value)


class Channel:
    """
    Responsible for:
        * Geometric information
        * Hydraulic properties
        * Reynolds number calculation
        * Friction factor correlations
        * Heat transfer models (Nusselt correlations)
    """

    def __init__(self, identifier: str, fluid_inputs: FluidComponentInputs,
                 fluid_operations: FluidComponentOperations):

        self.identifier = identifier
        self.type = fluid_inputs.channel_type

        # Geometric information
        self.inputs = fluid_inputs

        self.flow_sign = self._initialize_flow_direction(fluid_operations.flow_direction)
        self.friction_models = FrictionFactorFactory.create(fluid_inputs)
        self.heat_transfer_model = HeatTransferFactory.create(fluid_inputs)

        # Caches of the evaluated friction factors (FrictionFactors instances)
        # and steady-state heat transfer coefficients per evaluation location.
        self.friction_factors = LocationCache()
        self.steady_state_htc = LocationCache()

        # Record of the total friction factor time evolution at user-selected
        # spatial coordinates (the friction factor is not a PhysicalField, so
        # the channel owns its recorder directly).
        self.friction_factor_time_evolution = TimeEvolution()


    def __repr__(self):
        return f"{self.__class__.__name__}, type: {self.type}, identifier: {self.identifier})"


    def _initialize_flow_direction(self, flow_direction: FlowDirection) -> int:
        """Returns -1 or 1 depending on the flow direction."""
        if flow_direction is FlowDirection.FORWARD:
            return 1
        elif flow_direction is FlowDirection.BACKWARD:
            return -1
        else:
            raise ValueError(f"Invalid flow direction {flow_direction}.")


    def evaluate_friction_factors(self, reynolds: np.ndarray) -> FrictionFactors:
        """
        Evaluates the friction factors.
        """
        reynolds = np.abs(reynolds)

        laminar = self.friction_models.laminar(reynolds)
        turbulent = self.friction_models.turbulent(reynolds)
        total = self.friction_models.total(reynolds, laminar, turbulent)
        multiplier = self.inputs.friction_multiplier
        total *= multiplier
        return FrictionFactors(laminar=laminar, turbulent=turbulent, total=total)


    def eval_friction_factor(self, reynolds: np.ndarray, nodal=True) -> None:
        """Evaluate the friction factors and cache them at the given location
        (flag_nodal convention: True nodal, False Gauss, None initialization).
        """
        self.friction_factors[nodal] = self.evaluate_friction_factors(reynolds)


    def eval_steady_state_htc(self, fields, nodal=True) -> None:
        """Evaluate the steady-state heat transfer coefficient Nu * k / D_h
        with the channel's Nusselt correlation and cache it at the given
        location (flag_nodal convention: True nodal, False Gauss).

        Args:
            fields (FieldContainer): coolant fields providing Reynolds, Prandtl
                and total_thermal_conductivity.
            nodal (bool, optional): evaluation location flag. Defaults to True.
        """
        nusselt = self.heat_transfer_model(fields.Reynolds, fields.Prandtl)
        self.steady_state_htc[nodal] = (
            nusselt
            * fields.total_thermal_conductivity
            / self.inputs.hydraulic_diameter
        )
