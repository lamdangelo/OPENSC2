"""
This module contains the flow channel model.
"""

import numpy as np

from components.component_flags import FlowDirection
from components.fluid_component_inputs import FluidComponentInputs

from channel.friction_factor_factory import FrictionFactorFactory
from channel.friction_factor_models import FrictionFactors 

from channel.heat_transfer_factory import HeatTransferFactory


class Channel:
    """
    Flow channel model. 

    Responsible for:
        * Hydraulic properties
        * Reynolds number calculation 
        * Friction factor correlations
        * Heat transfer models (Nusselt correlations)
    """

    def __init__(self, identifier: str, fluid_inputs: FluidComponentInputs, 
                 flow_direction: FlowDirection):

        self.identifier = identifier 
        self.channel_type = fluid_inputs.channel_type
        self.flow_sign = self._initialize_flow_direction(flow_direction)
        self.friction_models = FrictionFactorFactory.create(fluid_inputs)
        self.heat_transfer_model = HeatTransferFactory.create(fluid_inputs)


    def __repr__(self):
        return f"{self.__class__.__name__}, type: {self.channel_type}, identifier: {self.identifier})"


    def _initialize_flow_direction(self, flow_direction: FlowDirection) -> int:
        """Returns -1 or 1 depending on the flow direction."""
        if flow_direction is FlowDirection.FORWARD:
            return 1
        elif flow_direction is FlowDirection.BACKWARD:
            return -1 
        else:
            raise ValueError(f"Invalid flow direction {flow_direction}.")
        

    def reynolds_number(self, velocity: np.ndarray, 
                        density: np.ndarray, 
                        viscosity: np.ndarray) -> np.ndarray:
        """
        Returns the Reynolds number: Re = rho * v * Dh / mu 
        """
        return (
            density 
            * np.abs(velocity) 
            * self.inputs.hydraulic_diameter
            / viscosity 
        )
    

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


### Not yet refactored code ###

    def eval_steady_state_htc(self, dict_prop, nn=0.3, nodal=True):
        """[summary]

        Args:
            dict_prop ([type]): [description]
            nn (float, optional): [description]. Defaults to 0.3.
            nodal (bool, optional): [description]. Defaults to True.
        """
        # Initiaize useful quantities.
        self._initialize_nusselt_and_htc(dict_prop["Reynolds"], nodal)
        # Evaluate Nusselt with the sutable correlation.
        self.nusselt_correlation(dict_prop, nn, nodal)
        # Evaluate steady state heat transfer coefficient: Nu*ther_cond/L
        self.dict_htc_steady[nodal] = (
            self.dict_nusselt[nodal]
            * dict_prop["total_thermal_conductivity"]
            / self.inputs["HYDIAMETER"]
        )
