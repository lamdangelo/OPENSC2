from components.solid.solid_component import SolidComponent
import numpy as np
from physical_fields.physical_field import FieldContainer, GridLocation

from components.jacket.jacket_component_inputs import JacketComponentInputs
from components.solid.solid_component_inputs import SolidComponentOperations
from conductor.conductor_flags import MethodFlag
from electromagnetics.electromagnetic_flags import BFieldDefinitionType


# Stainless steel properties
from properties_of_materials.stainless_steel import (
    thermal_conductivity_ss,
    isobaric_specific_heat_ss,
    density_ss,
    electrical_resistivity_ss,
)

# Glass-epoxy properties
from properties_of_materials.glass_epoxy import (
    thermal_conductivity_ge,
    isobaric_specific_heat_ge,
    density_ge,
    electrical_resistivity_ge,
)

DENSITY_FUNC = dict(ge=density_ge, ss=density_ss)
ISOBARIC_SPECIFIC_HEAT_FUNC = dict(
    ge=isobaric_specific_heat_ge, ss=isobaric_specific_heat_ss
)
THERMAL_CONDUCTIVITY_FUNC = dict(ge=thermal_conductivity_ge, ss=thermal_conductivity_ss)
ELECTRICAL_RESISTIVITY_FUNC = dict(
    ge=electrical_resistivity_ge, ss=electrical_resistivity_ss
)


class JacketComponent(SolidComponent):

    # Class for jacket objects

    ### INPUT PARAMETERS
    # some are inherited from class SolidComponent

    ### THERMOPHYSICAL PROPERTIES
    # inherited from class SolidComponent

    ### OPERATIONAL PARAMETERS
    # inherited from class SolidComponent

    ### COMPUTED IN INITIALIZATION
    # inherited from class SolidComponent

    ### COMPUTED VECTOR FOR MAGNETIC FIELD
    # inherited from class SolidComponent

    KIND = "JacketComponent"

    # Names of the nodal and Gauss fields whose time evolution at
    # user-selected spatial coordinates is recorded (in the fields'
    # PhysicalField.time_evolution).
    TIME_EVOLUTION_FIELDS = ("temperature",)
    TIME_EVOLUTION_GAUSS_FIELDS = (
        "current_along",
        "delta_voltage_along",
        "linear_power_el_resistance",
    )

    def __init__(
        self,
        simulation,
        identifier: str,
        name: str,
        inputs: JacketComponentInputs,
        operations: SolidComponentOperations,
        conductor,
    ):
        super().__init__(inputs, operations)
        self.name = name
        self.identifier = identifier

        self.node_fields = FieldContainer(GridLocation.NODE)
        self.gauss_fields = FieldContainer(GridLocation.GAUSS)
        self.dict_num_step = dict()
        self.radiative_heat_env = ""
        self.radiative_heat_inn = dict()
        self.coordinate = dict()

        if self.operations.magnetic_field_bc_mode is not BFieldDefinitionType.FROM_FILE:
            self.operations.magnetic_field_units = None

        self.deal_with_flag_IOP_MODE()

        self.__reorganize_input()
        self.__check_consistency(conductor)

        self.__jacket_density_flag = False

        self.initialize_heat_flux_schedule(simulation)

    def __repr__(self):
        return f"{self.__class__.__name__}(Type: {self.name}, identifier: {self.identifier})"

    def __str__(self):
        pass

    def _radiative_source_therm_env(self, conductor, environment):
        """Method that evaluates the heat transferred by radiation with the external environment.

        Args:
            conductor ([type]): [description]
            environment ([type]): [description]
        """
        key = f"{environment.KIND}_{self.identifier}"
        if conductor.inputs.thermohydraulic_method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
            # Backward Euler or Crank-Nicolson.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.radiative_heat_env = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
            elif conductor.cond_time[-1] > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, \
                    # since after that the whole SYSLOD array is saved and there is no \
                    # need to compute twice the same values.
                    self.radiative_heat_env[:, 1] = self.radiative_heat_env[:, 0].copy()
                # Update value at the current time step.
                self.radiative_heat_env[:, 0] = (
                    conductor.dict_interf_peri["env_sol"]["nodal"][key]
                    * conductor.node_fields.HTC["env_sol"][key]["rad"]
                    * (
                        environment.inputs["Temperature"]
                        - self.node_fields.temperature
                    )
                )
            # end if conductor.cond_time[-1].
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.radiative_heat_env = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0:
                self.radiative_heat_env[:, 1:4] = self.radiative_heat_env[:, 0:3].copy()
                # Update value at the current time step.
                self.radiative_heat_env[:, 0] = (
                    conductor.dict_interf_peri["env_sol"]["nodal"][key]
                    * conductor.node_fields.HTC["env_sol"][key]["rad"]
                    * (
                        environment.inputs["Temperature"]
                        - self.node_fields.temperature
                    )
                )
            # end if conductor.cond_time[-1].
        # end if conductor.inputs["METHOD"].

    # End method _radiative_source_therm.

    def _radiative_heat_exc_inner(self, conductor, jk_inner):
        """Method that evaluates the heat transferred by radiation with the inner surface of the enclosure and the inner jackets.

        Args:
            conductor ([type]): [description]
            environment ([type]): [description]
        """
        if self.identifier < jk_inner.identifier:
            key = f"{self.identifier}_{jk_inner.identifier}"
        else:
            key = f"{jk_inner.identifier}_{self.identifier}"
        # End if self.identifier.
        if conductor.inputs.thermohydraulic_method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
            # Backward Euler or Crank-Nicolson.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.radiative_heat_inn[key] = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
            elif conductor.cond_time[-1] > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, \
                    # since after that the whole SYSLOD array is saved and there is no \
                    # need to compute twice the same values.
                    self.radiative_heat_inn[key][:, 1] = self.radiative_heat_inn[key][
                        :, 0
                    ].copy()
                # Update value at the current time step.
                self.radiative_heat_inn[key][:, 0] = (
                    conductor.dict_interf_peri["sol_sol"]["nodal"][key]
                    * conductor.node_fields.HTC["sol_sol"]["rad"][key]
                    * (
                        jk_inner.node_fields.temperature
                        - self.node_fields.temperature
                    )
                )
            # end if conductor.cond_time[-1].
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.radiative_heat_inn[key] = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0:
                self.radiative_heat_inn[key][:, 1:4] = self.radiative_heat_inn[key][
                    :, 0:3
                ].copy()
                # Update value at the current time step.
                self.radiative_heat_inn[key][:, 0] = (
                    conductor.dict_interf_peri["sol_sol"]["nodal"][key]
                    * conductor.node_fields.HTC["sol_sol"]["rad"][key]
                    * (
                        jk_inner.node_fields.temperature
                        - self.node_fields.temperature
                    )
                )
            # end if conductor.cond_time[-1].
        # end if conductor.inputs["METHOD"].

    # End method _radiative_source_therm.

    def __reorganize_input(self):
        """Private method that reorganizes input data to simplify properties homogenization."""

        self.materials = np.array(
            [self.inputs.jacket_material, self.inputs.insulation_material],
            dtype=str,
        )

        self.__index_material_none = np.nonzero(self.materials == "none")[0]
        self.materials = self.materials[np.nonzero(self.materials != "none")[0]]

        self.cross_sections = np.array(
            [self.inputs.jacket_cross_section, self.inputs.insulation_cross_section],
            dtype=float,
        )

        # Get the indexes corresponding to 0 used for consistency check.
        self.__index_cross_section_0 = np.nonzero(self.cross_sections == 0)[0]
        self.cross_sections = self.cross_sections[np.nonzero(self.cross_sections)[0]]

        # Total value of homogenization coefficients.
        self.__cross_section = self.cross_sections.sum()

        # Create numpy array with density functions according to jacket
        # materials; order is consistent with values in self.materials.
        self.density_function = np.array([DENSITY_FUNC[key] for key in self.materials])

        # Create numpy array with electrical resistivity functions according to
        # jacket materials; order is consistent with values in
        # self.materials.
        self.electrical_resistivity_function = np.array(
            [ELECTRICAL_RESISTIVITY_FUNC[key] for key in self.materials]
        )

        # Create numpy array with isobaric specific heat functions according to
        # jacket materials; order is consistent with values in
        # self.materials.
        self.isobaric_specific_heat_function = np.array(
            [ISOBARIC_SPECIFIC_HEAT_FUNC[key] for key in self.materials]
        )

        # Create numpy array with thermal conductivity functions according to
        # jacket material; order is consistent with values in
        # self.materials.
        self.thermal_conductivity_function = np.array(
            [THERMAL_CONDUCTIVITY_FUNC[key] for key in self.materials]
        )

    def __check_consistency(self, conductor):
        """Private method that checks consistency of jacket user definition.

        Args:
            conductor (Conductor): instance of class Conductor.

        Raises:
            ValueError: if number of jacket materials given in input is not consistent with user declared materials.
            ValueError: if number of jacket materials given in input is not consistent with not zero user defined material thicknes.
            ValueError: if the indexes of "none" material are not equal to the indexes of thickness equal to 0.
            ValueError: if jacket cross section given in input is not consistent with the evaluated one.
        """
        # Check that number of jacket materials given in input is consistent
        # with user declared materials.
        if self.materials.size != self.inputs.num_material_types:
            raise ValueError(
                f"{conductor.identifier = } -> {self.identifier = }\nThe number of material constituting the jacket ({self.inputs.num_material_types = }) is inconsistent with the number of defined materials ({self.materials.size = }).\nPlease check..."
            )

        if self.cross_sections.size != self.inputs.num_material_types:
            raise ValueError(
                f"{conductor.identifier = } -> {self.identifier = }\nThe number of material constituting the jacket ({self.inputs.num_material_types = }) is inconsistent with the number of defined cross sections ({self.cross_sections.size = }).\nPlease check..."
            )

        if any(self.__index_material_none != self.__index_cross_section_0):
            raise ValueError(
                f"{conductor.identifier = } -> {self.identifier = }\nDefined materials and defined cross sections must be consistent.\nPlease check..."
            )

        tol = 1e-3
        if (
            abs(self.__cross_section - self.inputs.cross_section)
            / self.inputs.cross_section
            > tol
        ):
            raise ValueError(
                f"{conductor.identifier = } -> {self.identifier = }\nInconsistent cross section value: user defines {self.inputs.cross_section = } while computed one is {self.__cross_section = }.\nPlease check..."
            )

        # Delete no longer useful attributes.
        del (
            self.__index_material_none,
            self.__index_cross_section_0,
            self.__cross_section,
        )

    def jacket_density(self, property: dict) -> np.ndarray:
        """Method that evaluates the homogenized denstiy of the jacket, in the case it is made by at most by two materials (jacket and insulation). Homogenization is based on material cross sections.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with homogenized density of jacket in kg/m^3.
        """
        # Set fleag to true to allow evaluation of homogenized isobaric
        # specific heat.
        self.__jacket_density_flag = True
        density = np.array(
            [func(property.temperature) for func in self.density_function]
        )
        if self.inputs.num_material_types > 1:
            # Evaluate homogenized density of the jacket:
            # rho_eq = (A_jk*rho_jk + A_in*rho_in)/(A_jk + A_in)
            self.__density_numerator = density.T * self.cross_sections
            self.__density_numerator_sum = self.__density_numerator.sum(axis=1)
            return self.__density_numerator_sum / self.inputs.cross_section
        elif self.inputs.num_material_types == 1:
            return density[0]

    def jacket_isobaric_specific_heat(self, property: dict) -> np.ndarray:
        """Method that evaluates homogenized isobaric specific heat of the jacket, in the case it is made at most by two materials (jacket and insulation). Homogenization is based on material mass.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with homogenized isobaric specific heat of the jacket in J/kg/K.
        """
        # Check on homogenized density evaluation before homogenized isobaric
        # specific heat, since some therms are in common and are not evaluated
        # twices.
        if self.__jacket_density_flag == False:
            raise ValueError(
                f"Call method {self.jacket_density.__name__} before evaluation of homogenized jacket isobaric specific heat.\n"
            )

        # Set flag to false to trigger error in the next homogenized isobaric
        # specific heat evaluation if not done properly.
        self.__jacket_density_flag = False
        isobaric_specific_heat = np.array(
            [
                func(property.temperature)
                for func in self.isobaric_specific_heat_function
            ]
        )
        if self.inputs.num_material_types > 1:
            # Evaluate homogenized isobaric specific heat of the jacket:
            # cp_eq = (cp_jk*A_jk*rho_jk + cp_in*A_in*rho_in)/(A_jk*rho_jk +
            # A_in*rho_in)
            return (isobaric_specific_heat.T * self.__density_numerator).sum(
                axis=1
            ) / self.__density_numerator_sum
        elif self.inputs.num_material_types == 1:
            return isobaric_specific_heat[0]

    def jacket_thermal_conductivity(self, property: dict) -> np.ndarray:
        """Method that evaluates the homogenized thermal conductivity of the jacket, in the case it is made by at most by two materials (jacket and insulation). Homogenization is based on material cross sections.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with homogenized thermal conductivity of the jacket in W/m/K.
        """
        thermal_conductivity = np.array(
            [
                func(property.temperature)
                for func in self.thermal_conductivity_function
            ]
        )
        if self.inputs.num_material_types > 1:
            # Evaluate homogenized thermal conductivity of the jacket:
            # k_eq = (A_jk*k_jk + A_in*k_in)/(A_jk + A_in)
            return (thermal_conductivity.T * self.cross_sections).sum(
                axis=1
            ) / self.inputs.cross_section
        elif self.inputs.num_material_types == 1:
            return thermal_conductivity[0]

    def jacket_electrical_resistivity(self, property: dict) -> np.ndarray:
        """Method that evaluates the homogenized electrical resistivity otf the jacket, in the case it is made by at most by two materials (jacket and insulation). Homogenization is based on material cross sections.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with homogenized electrical resistivity of the jacket in Ohm*m.
        """

        electrical_resistivity = np.array(
            [
                func(property.temperature)
                for func in self.electrical_resistivity_function
            ]
        )
        if self.inputs.num_material_types > 1:
            # Evaluate homogenized electrical resistivity of the jacket:
            # rho_el_eq = (A_jk + A_in) * ((A_jk/rho_el_jk + A_in/rho_el_in))^-1
            return self.inputs.cross_section * np.reciprocal(
                (self.cross_sections / electrical_resistivity.T).sum(axis=1)
            )
        elif self.inputs.num_material_types == 1:
            return electrical_resistivity[0]