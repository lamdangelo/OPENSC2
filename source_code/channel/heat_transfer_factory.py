"""
This module contains the Heat Transfer Factory class.
"""

from components.fluid_component_inputs import (
    FluidComponentInputs
)

from channel.heat_transfer_models import HeatTransferModelType
import channel.nusselt as nus


class HeatTransferFactory:

    @staticmethod 
    def create(inputs: FluidComponentInputs) -> nus.NusseltCorrelation:
        model = inputs.heat_transfer_model 

        if model is HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_1:
            return nus.LowerBoundNusselt(lower_limit=8.235)
        
        elif model is HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_119:
            return nus.LowerBoundNusselt_119(
                hydraulic_diameter=inputs.hydraulic_diameter
            )
        
        elif model is HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_120:
            return nus.LowerBoundNusselt(lower_limit=2.181)
        
        elif model is HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_121:
            return nus.LowerBoundNusselt(lower_limit=4.01)
        
        elif model is HeatTransferModelType.CORRELATION_211:
            return nus.Correlation_211()
        
        elif model is HeatTransferModelType.RECTANGULAR_DUCT_ENEA_HTS_CICC:
            return nus.RectangularDuctEneaHtsCicc()
        
        else:
            raise NotImplementedError(
                    f"Heat transfer model {model.name} "
                    "has not been implemented."
                )