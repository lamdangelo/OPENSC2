from __future__ import annotations

from components.fluid.fluid_component_inputs import (
    FluidComponentInputs
)

from hydraulics.friction_factor_models import (
    FrictionFactorModelType,
    FrictionFactors
)

import hydraulics.laminar_friction as lamf
import hydraulics.turbulent_friction as turf
import hydraulics.total_friction as totf


class FrictionFactorFactory:

    @staticmethod
    def create(
        inputs: FluidComponentInputs
    ) -> FrictionFactors:

        laminar = FrictionFactorFactory._create_laminar(
            inputs
        )

        turbulent = FrictionFactorFactory._create_turbulent(
            inputs
        )

        total = FrictionFactorFactory._create_total(
            inputs.friction_factor_model
        )

        return FrictionFactors(
            laminar=laminar,
            turbulent=turbulent,
            total=total,
        )
    

    @staticmethod
    def _create_laminar(
        inputs: FluidComponentInputs,
    ):
        model = inputs.friction_factor_model

        if model is FrictionFactorModelType.USER_DEFINED:
            return lamf.UserDefinedLaminar()

        if model is FrictionFactorModelType.RECTANGULAR_DUCT_MEMO:
            return lamf.RectangularDuctLaminar(
                side1=inputs.width,
                side2=inputs.height,
            )

        if model is FrictionFactorModelType.DUCT_DEMO_COMMON_RECTANGULAR:
            return lamf.DuctDemoCommonRectangular()
        
        if model is FrictionFactorModelType.DUCT_DEMO_COMMON_TRIANGULAR:
            return lamf.DuctDemoCommonTriangular()
        
        if model is FrictionFactorModelType.INCROPERA_RECTANGULAR_TF:
            return lamf.IncroperaRectangularLaminarTFHole()
        
        if model is FrictionFactorModelType.INCROPERA_RECTANGULAR_CS:
            return lamf.IncroperaRectangularLaminarCSHole(
                is_rectangular=inputs.is_rectangular,
                width=inputs.width,
                height=inputs.height,
                hydraulic_diameter=inputs.hydraulic_diameter
            )

        return lamf.SmoothTubeLaminar()
    

    @staticmethod
    def _create_turbulent(
        inputs: FluidComponentInputs,
    ):
        model = inputs.friction_factor_model

        match model:

            case FrictionFactorModelType.ITER_CONDUCTOR_79:
                return turf.IterConductor79(
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.ITER_CONDUCTOR_810:
                return turf.IterConductor810(
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V1:
                return turf.NewtonHole(
                    number=1, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V2:
                return turf.NewtonHole(
                    number=2, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V3:
                return turf.NewtonHole(
                    number=3, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V4:
                return turf.NewtonHole(
                    number=4, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V5:
                return turf.NewtonHole(
                    number=5, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V6:
                return turf.NewtonHole(
                    number=6, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V7:
                return turf.NewtonHole(
                    number=7, hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V8:
                return turf.NewtonHole(
                    number=8, hydraulic_diameter=inputs.hydraulic_diameter
                )

            case FrictionFactorModelType.BLASIUS:
                return turf.Blasius()

            case FrictionFactorModelType.HAALAND:
                return turf.Haaland(
                    roughness=inputs.roughness,
                    hydraulic_diameter=inputs.hydraulic_diameter,
                )

            case FrictionFactorModelType.COLEBROOK:
                return turf.Colebrook(
                    roughness=inputs.roughness,
                    hydraulic_diameter=inputs.hydraulic_diameter,
                )

            case FrictionFactorModelType.PETUKHOV:
                return turf.Petukhov()

            case FrictionFactorModelType.LHC_MAGNET:
                return turf.LHCMagnetRecipe()
            
            case FrictionFactorModelType.BESSETTE_ITER_CS_SPIRAL:
                return turf.BessetteIterCS2015(
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.CSI_LS_BEST_FIT:
                return turf.CSILSBestFit()
            
            case FrictionFactorModelType.TRONZA_ITER_TF_HOLE:
                return turf.TronzaIterTFHole(
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.INCROPERA_RECTANGULAR_TF:
                return turf.IncroperaRectangularTurbulentTFHole()
            
            case FrictionFactorModelType.INCROPERA_RECTANGULAR_CS:
                return turf.IncroperaRectangularTurbulentCSHole(
                    is_rectangular=inputs.is_rectangular,
                    width=inputs.width,
                    height=inputs.height,
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.FLAT_SPIRAL:
                return turf.FlatSpiral()
            
            case FrictionFactorModelType.DUCT_DEMO_COMMON_RECTANGULAR:
                return turf.DuctDemoCommon()
            
            case FrictionFactorModelType.DUCT_DEMO_COMMON_TRIANGULAR:
                return turf.DuctDemoCommon()
            
            case FrictionFactorModelType.ENEA_HTS_CICC:
                return turf.EneaHtsCicc()
            
            case FrictionFactorModelType.TRONZA_ITER_TF_BUNDLE:
                return turf.TronzaIterTFBundle(
                    cross_section=inputs.cross_section,
                    hydraulic_diameter=inputs.hydraulic_diameter
                )

            case FrictionFactorModelType.WANNER:
                return turf.Wanner()

            case FrictionFactorModelType.KSTAR:
                return turf.KSTAR()
            
            case FrictionFactorModelType.KATHEDER_EAST:
                return turf.KathederEast(void_fraction=inputs.void_fraction)
            
            case FrictionFactorModelType.KATHEDER_PURE:
                return turf.KathederPure(void_fraction=inputs.void_fraction)
            
            case FrictionFactorModelType.DARCY_FORCHHEIMER_POROUS_MEDIUM:
                return turf.DarcyForchheimerPorousMedium(
                    void_fraction=inputs.void_fraction, 
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.DTT:
                return turf.DttBundle(
                    void_fraction=inputs.void_fraction, 
                    hydraulic_diameter=inputs.hydraulic_diameter
                )
            
            case FrictionFactorModelType.LHC_PONCET:
                return turf.LhcPoncetBundle()
            
            case FrictionFactorModelType.DEMO_TF_HTS:
                return turf.DemoHtsBundle()
            
            case FrictionFactorModelType.HTS_CL:
                return turf.HtsClBundle()

            case FrictionFactorModelType.USER_DEFINED:
                return turf.UserDefinedTurbulent()

            case _:
                raise NotImplementedError(
                    f"Turbulent model {model.name} "
                    "has not been implemented."
                )


    @staticmethod
    def _create_total(
        model: FrictionFactorModelType,
    ):

        transitional_models = {
            FrictionFactorModelType.INCROPERA_RECTANGULAR_TF,
            FrictionFactorModelType.INCROPERA_RECTANGULAR_CS,
            FrictionFactorModelType.DUCT_DEMO_COMMON_RECTANGULAR,
            FrictionFactorModelType.DUCT_DEMO_COMMON_TRIANGULAR,
        }

        if model is FrictionFactorModelType.USER_DEFINED:
            return totf.UserDefinedTotal()

        if model in transitional_models:
            return totf.TransitionalInterpolation()

        return totf.MaximumModel()