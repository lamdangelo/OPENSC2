import numpy as np
from electromagnetics.electromagnetic_flags import BFieldDefinitionType
from physical_fields.physical_field import FieldContainer, GridLocation
from components.solid.strand_component import StrandComponent

from components.solid.strand_stabilizer_component_inputs import StrandStabilizerComponentInputs
from components.solid.solid_component_inputs import StrandComponentOperations

# Aluminium properties
from properties_of_materials.aluminium import (
    thermal_conductivity_al,
    isobaric_specific_heat_al,
    density_al,
    electrical_resistivity_al,
)

# Cu properties
from properties_of_materials.copper import (
    thermal_conductivity_cu_nist,
    isobaric_specific_heat_cu_nist,
    density_cu,
    electrical_resistivity_cu_nist,
)

DENSITY_FUNC = dict(
    al=density_al,
    cu=density_cu,
)

THERMAL_CONDUCTIVITY_FUNC = dict(
    al=thermal_conductivity_al,
    cu=thermal_conductivity_cu_nist,
)

ISOBARIC_SPECIFIC_HEAT_FUNC = dict(
    al=isobaric_specific_heat_al,
    cu=isobaric_specific_heat_cu_nist,
)

ELECTRICAL_RESISTIVITY_FUNC = dict(
    al=electrical_resistivity_al,
    cu=electrical_resistivity_cu_nist,
)


class StrandStabilizerComponent(StrandComponent):

    # Class for copper strands objects

    ### INPUT PARAMETERS
    # some are inherited form the parent classes StrandComponent and SolidComponent

    ### THERMOPHYSICAL PROPERTIES
    # inherited from class SolidComponent

    ##### OPERATIONAL PARAMETERS
    # inherited from parents class SolidComponent and StrandComponent

    ### COMPUTED IN INITIALIZATION
    # inherited from class SolidComponent

    ### COMPUTED VECTOR FOR MAGNETIC FIELD
    # inherited from class SolidComponent

    KIND = "StrandStabilizerComponent"

    # Names of the nodal and Gauss fields whose time evolution at
    # user-selected spatial coordinates is recorded (in the fields'
    # PhysicalField.time_evolution).
    TIME_EVOLUTION_FIELDS = ("temperature", "B_field")
    TIME_EVOLUTION_GAUSS_FIELDS = (
        "current_along",
        "delta_voltage_along",
        "linear_power_el_resistance",
        "delta_voltage_along_sum",
    )

    def __init__(
        self,
        simulation: object,
        identifier: str,
        name: str,
        inputs: StrandStabilizerComponentInputs,
        operations: StrandComponentOperations,
        conductor: object,
    ):
        """Method that makes instance of class StrandStabilizerComponent.

        Args:
            simulation (object): simulation object.
            identifier (str): component identifier.
            name (str): component kind string (e.g. "STR_STAB").
            inputs (StrandStabilizerComponentInputs): pre-parsed input parameters.
            operations (StrandComponentOperations): pre-parsed operations parameters.
            conductor (object): instance of class Conductor.
        """

        super().__init__(inputs, operations)
        self.name = name
        self.identifier = identifier

        self.node_fields = FieldContainer(GridLocation.NODE)
        self.gauss_fields = FieldContainer(GridLocation.GAUSS)
        self.dict_num_step = dict()
        self.coordinate = dict()
        self.dict_scaling_input = dict()

        self.initialize_heat_flux_schedule(simulation)
        if self.operations.magnetic_field_bc_mode is not BFieldDefinitionType.FROM_FILE:
            self.operations.magnetic_field_units = None

        self.deal_with_flag_IOP_MODE()

        self.radius = np.sqrt(self.inputs.cross_section / np.pi)

        self.deal_with_fixed_potential(conductor.inputs.zlength)

    def __repr__(self):
        return f"{self.__class__.__name__}(Type: {self.name}, identifier: {self.identifier})"

    def __str__(self):
        pass

    def strand_density(self, property: dict) -> np.ndarray:
        """Method that evaluates density of the stabilizer.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with density of the stabilizer in kg/m^3.
        """
        return DENSITY_FUNC[self.inputs.stabilizer_material](property.temperature)

    def strand_isobaric_specific_heat(self, property: dict) -> np.ndarray:
        """Method that evaluates isobaric specific heat of the stabilizer.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with isobaric specific heat of the stabilizer in kg/m^3.
        """
        return ISOBARIC_SPECIFIC_HEAT_FUNC[self.inputs.stabilizer_material](
            property.temperature
        )

    def strand_thermal_conductivity(self, property: dict) -> np.ndarray:
        """Method that evaluates thermal conductivity of the stabilizer.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with thermal conductivity of the stabilizer in W/m/K.
        """
        if self.inputs.stabilizer_material == "cu":
            return THERMAL_CONDUCTIVITY_FUNC[self.inputs.stabilizer_material](
                property.temperature,
                property.B_field,
                self.inputs.residual_resistivity_ratio,
            )
        else:
            return THERMAL_CONDUCTIVITY_FUNC[self.inputs.stabilizer_material](
                property.temperature
            )

    def strand_electrical_resistivity(self, property: dict) -> np.ndarray:
        """Method that evaluates electrical resistivity of the stabilizer.

        Args:
            property (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            np.ndarray: array with electrical resistivity of the stabilizer in Ohm*m.
        """
        if self.inputs.stabilizer_material == "cu":
            return ELECTRICAL_RESISTIVITY_FUNC[self.inputs.stabilizer_material](
                property.temperature,
                property.B_field,
                self.inputs.residual_resistivity_ratio,
            )
        else:
            return ELECTRICAL_RESISTIVITY_FUNC[self.inputs.stabilizer_material](
                property.temperature
            )

    def get_electric_resistance(self, conductor: object) -> np.ndarray:
        f"""Method that evaluate the electric resistance in Gauss node only, used to build the electric_resistance_matrix.

        Args:
            conductor (object): class Conductor object from which node distance is stored to do the calculation.

        Returns:
            np.ndarray: array of electrical resistance in Ohm of length {conductor.mesh.number_of_elements = }.
        """
        self.gauss_fields.electric_resistance = (self.gauss_fields.electrical_resistivity_stabilizer
            * conductor.node_distance[("StrandComponent", self.identifier)]
            / self.inputs.cross_section)
        return self.gauss_fields.electric_resistance