import numpy as np

from components.fluid.fluid_component_inputs import (
    FluidComponentOperations,
    FluidComponentInputs
)

from physical_fields.physical_field import (
    PhysicalField,
    FieldContainer,
    GridLocation,
)

import hydraulics.hydraulics as hydraulics
import interfaces.coolprop_interface as cpi


class Coolant():
    """
    Responsible for:
        * Storage of physical states (fields) in space and time
    """

    # Names of the nodal fields whose time evolution at user-selected spatial
    # coordinates is recorded (in the fields' PhysicalField.time_evolution).
    TIME_EVOLUTION_FIELDS = ("velocity", "pressure", "temperature", "total_density")

    def __init__(self, identifier: str,
                 fluid_inputs: FluidComponentInputs,
                 fluid_operations: FluidComponentOperations):

        self.identifier = identifier
        self.fluid_type = fluid_inputs.fluid_type
        self.inputs = fluid_inputs
        self.operations = fluid_operations

        # Initialize physical fields
        self.pressure = PhysicalField("pressure", "Pa")
        self.temperature = PhysicalField("temperature", "K")
        self.velocity = PhysicalField("velocity", "m/s")
        self.density = PhysicalField("density", "kg/m^3")

        # Containers of the spatial distributions of the fluid state and
        # transport properties in the mesh nodes and in the Gauss points.
        self.node_fields = FieldContainer(GridLocation.NODE)
        self.gauss_fields = FieldContainer(GridLocation.GAUSS)

        headers_inl_out = [
            "time (s)",
            "velocity_inl (m/s)",
            "pressure_inl (Pa)",
            "temperature_inl (K)",
            "total_density_inl (kg/m^3)",
            "mass_flow_rate_inl (kg/s)",
            "velocity_out (m/s)",
            "pressure_out (Pa)",
            "temperature_out (K)",
            "total_density_out (kg/m^3)",
            "mass_flow_rate_out (kg/s)",
        ]
        # Empty dictionary of list to save variable time evolutions at inlet and outlet spatial coordinates.
        self.time_evol_io = {key: list() for key in headers_inl_out}


    def __repr__(self):
        return f"{self.__class__.__name__}(Type: {self.fluid_type}, identifier: {self.identifier})"


    def eval_coolant_density_din_viscosity_gen_flow(self, pressure, temperature):
        """Compute mass density and dynamic viscosity at the given pressure and
        temperature (needed to perform the flow initialization)."""
        density = cpi.compute_mass_density(self.fluid_type, temperature, pressure)
        viscosity = cpi.compute_viscosity(self.fluid_type, temperature, pressure)
        return density, viscosity


    def eval_reynolds_from_mass_flow_rate(self, mass_flow_rate, din_viscosity):
        """Evaluate the Reynolds number from the mass flow rate."""
        return hydraulics.compute_reynolds_from_mass_flow_rate(
            self.inputs.hydraulic_diameter,
            self.inputs.cross_section,
            mass_flow_rate,
            din_viscosity,
        )


    def compute_velocity_gen_flow(
        self,
        length,
        channel,
        max_iterations,
        pressure_drop,
        density,
        viscosity,
        tolerance,
        velocity_guess=0.0,
        friction_guess=0.01,
    ):
        """Iteratively solve for the steady-state velocity given a prescribed
        pressure drop (inverted Darcy-Weisbach equation)."""
        return hydraulics.solve_darcy_weisbach_velocity(
            length,
            self.inputs.cos_theta,
            self.inputs.hydraulic_diameter,
            pressure_drop,
            density,
            viscosity,
            max_iterations,
            tolerance,
            total_friction_factor=(
                lambda reynolds: channel.evaluate_friction_factors(reynolds).total
            ),
            velocity_guess=velocity_guess,
            friction_guess=friction_guess,
            identifier=self.identifier,
        )


    def compute_mass_flow_with_direction(self, flow_sign, density, velocity):
        """Compute the signed mass flow rate from the flow direction sign."""
        return hydraulics.compute_mass_flow_rate_with_direction(
            flow_sign, self.inputs.cross_section, density, velocity
        )


    def _eval_nodal_pressure_temperature_velocity_initialization(self, conductor):
        """Initialize pressure, temperature, velocity and density spatial
        distributions in the mesh nodes from the boundary condition values."""
        from hydraulics.hydraulic_flags import HydraulicBC

        if self.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY:
            mfr = self.operations.outlet_mass_rate
        else:
            mfr = self.operations.inlet_mass_rate
        # Compute pressure from inlet and outlet values by linear interpolation.
        if mfr >= 0.0:
            # Flow direction from x = 0 to x = L.
            self.node_fields.pressure = np.interp(
                conductor.mesh.node_coordinates,
                [0.0, conductor.inputs.zlength],
                [self.operations.inlet_pressure, self.operations.outlet_pressure],
            )
        else:
            # Flow direction from x = L to x = 0.
            self.node_fields.pressure = np.interp(
                conductor.mesh.node_coordinates,
                [0.0, conductor.inputs.zlength],
                [self.operations.outlet_pressure, self.operations.inlet_pressure],
            )
        # Compute temperature from inlet and outlet values by linear interpolation.
        self.node_fields.temperature = np.interp(
            conductor.mesh.node_coordinates,
            [0.0, conductor.inputs.zlength],
            [self.operations.inlet_temperature, self.operations.outlet_temperature],
        )
        # Compute density (needed to compute the velocity from mass flow rate).
        self.node_fields.total_density = cpi.compute_mass_density(
            self.fluid_type,
            self.node_fields.temperature,
            self.node_fields.pressure,
        )
        # Compute velocity from mass flow rate; the sign is determined by the
        # mass flow rate.
        self.node_fields.velocity = mfr / (
            np.maximum(self.inputs.cross_section, 1e-7)
            * self.node_fields.total_density
        )


    def eval_dimensionless_numbers(self, fields: FieldContainer) -> FieldContainer:
        """Compute the Reynolds and Gruneisen dimensionless numbers."""
        fields.Reynolds = hydraulics.compute_reynolds_from_velocity(
            self.inputs.hydraulic_diameter,
            fields.velocity,
            fields.total_density,
            fields.total_dynamic_viscosity,
        )
        fields.Gruneisen = hydraulics.compute_gruneisen(
            fields.isobaric_expansion_coefficient,
            fields.isothermal_compressibility,
            fields.total_isochoric_specific_heat,
            fields.total_density,
        )
        return fields


    def _eval_properties(self, fields: FieldContainer, aliases: dict) -> FieldContainer:
        """Evaluate the coolant transport properties at the pressure and
        temperature stored in fields, exploiting the CoolProp library.

        All properties are evaluated with a single equation-of-state flash
        per point (see coolprop_interface.compute_properties)."""
        properties = cpi.compute_properties(
            self.fluid_type, aliases, fields.temperature, fields.pressure
        )
        for prop_name, values in properties.items():
            setattr(fields, prop_name, values)
        # Compute Reynolds and Gruneisen dimensionless numbers.
        return self.eval_dimensionless_numbers(fields)


    def _compute_density_and_mass_flow_rates(self, fields: FieldContainer) -> FieldContainer:
        """Evaluate mass density and mass flow rate spatial distributions."""
        fields.total_density = cpi.compute_mass_density(
            self.fluid_type, fields.temperature, fields.pressure
        )
        fields.mass_flow_rate = hydraulics.compute_mass_flow_rate(
            self.inputs.cross_section, fields.velocity, fields.total_density
        )
        return fields


    def _eval_gauss_pressure_temperature_velocity(self, conductor):
        """[summary]

        Args:
            conductor ([type]): [description]

        Raises:
            ValueError: [description]
        """
        if bool(self.node_fields):
            # Pressure, temperature and velocity directly evaluated from the value in nodal point, averaging on two consecutives nodes.
            list_average_prop = ["temperature", "pressure", "velocity"]
            # Compurte pressure, temperature and velocity in Gauss points exploiting dictionary comprehension. Remember that old keys in self.gauss_fields will be deleted and the new self.gauss_fields have only the keys in list_average_prop (i.e self.gauss_fields is cleaned and constructed from scratch).
            # N.B. valutare se posso usare np.interp per calcolare pressione e temperatura nel Gauss. Per la velocità capire se posso passare per la densità nel Gauss invertendo la formula della portata come fatto per l'inizializzazione della velocità nei nodi. In questo caso userei come portata il valore medio tra i primi due nodi. Occhio che questa funzione viene utilizzata at ogni timestep non solo all'inizializzazione.
            self.gauss_fields = FieldContainer.from_mapping(
                {
                    key: (
                        getattr(self.node_fields, key)[: conductor.mesh.number_of_nodes - 1]
                        + getattr(self.node_fields, key)[1:]
                    )
                    / 2.0
                    for key in list_average_prop
                },
                GridLocation.GAUSS,
            )
        else:
            # Empty dictionary self.node_fields: remember that nodal properties must be evaluated before the Gauss one.
            raise ValueError(
                f"ERROR! dictionary node_fields is empty. User must first evaluate properties in nodes and than in Gauss points.\n"
            )
        # End if bool(self.node_fields)

    # End method _eval_gauss_pressure_temperature_velocity


    def _eval_properties_nodal_gauss(self, conductor, aliases, nodal=True):
        """
        Method that evaluate density, dynamic viscosity, enthalpy, ... of Coolant
        class objects in both nodal points and Gauss points according to nodal
        input parameter exploiting the CoolProp library.
        """
        # Properties evaluation in each nodal point
        if nodal:
            self.node_fields = self._eval_properties(self.node_fields, aliases)
        # Properties evaluation in each Gauss point
        else:
            # Evaluate pressure temperature and velocity in Gauss point (still to be decided where to really put this line of code)
            self._eval_gauss_pressure_temperature_velocity(conductor)
            self.gauss_fields = self._eval_properties(self.gauss_fields, aliases)
        # End if nodal

    # End method _eval_properties_nodal_gauss

    def _compute_density_and_mass_flow_rates_nodal_gauss(self, conductor, nodal=True):
        """
        Method that evaluates channel density and mass flow rate in nodal points or in Gauss point according to options value, after that solution is evaluated with function STEP, to make plots of spatial distribution and time evolution. This function is also invoked in method Conductor.Initialization. (cdp, 10/2020)
        """
        if nodal:
            self.node_fields = self._compute_density_and_mass_flow_rates(
                self.node_fields
            )
        else:
            # Velocity directly evaluated from the value in nodal point, averaging on two consecutives nodes.
            # Velocity in Gauss points
            self.gauss_fields.velocity = (
                self.node_fields.velocity[: conductor.mesh.number_of_nodes - 1]
                + self.node_fields.velocity[1:]
            ) / 2.0
            self.gauss_fields = self._compute_density_and_mass_flow_rates(
                self.gauss_fields
            )

    # End method _compute_density_and_mass_flow_rates_nodal_gauss

