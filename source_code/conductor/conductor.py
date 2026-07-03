# Import packages
from collections import namedtuple
from pathlib import Path
import logging
import os
import warnings
from physical_fields.physical_field import FieldContainer, GridLocation
import thermal.heat_sources as thermal_heat_sources
import thermal.radiation as thermal_radiation
import thermal.htc_evaluation as htc_evaluation
import thermal.energy_balance as energy_balance
import conductor.conductor_topology as conductor_topology

import numpy as np
import pandas as pd
from scipy import constants, integrate
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, diags
from typing_extensions import Self

# import classes
from components.component_collection import ComponentInventory
from electromagnetics.electromagnetic_flags import CurrentMode
from conductor.conductor_mesh import MeshType
from conductor.conductor_flags import MethodFlag
import conductor.conductor_mesh as conductor_mesh
from components.fluid.fluid_component import FluidComponent
from components.component_factory import ComponentFactory, ComponentBuildContext
from components.jacket.jacket_component import JacketComponent
from components.solid.stack_component import StackComponent
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_stabilizer_component import StrandStabilizerComponent

# import electromagnetics module functions
from electromagnetics.circuit_topology import (
    build_connectivity_current_carriers,
    build_incidence_matrix,
    detect_contacts,
    build_contact_incidence_matrix,
)
from electromagnetics.resistance import build_resistance_matrix
from electromagnetics.conductance import build_conductance_matrix
from electromagnetics.system_assembly import (
    build_stiffness_matrix,
    build_mass_matrix,
    build_known_term_vector,
    build_right_hand_side,
)
from electromagnetics.boundary_conditions import (
    assign_equipotential_surfaces,
    assign_fixed_potential,
    build_reduction_operator,
    reduce_system,
)
from electromagnetics.electric_solver import (
    solve_steady_state,
    solve_transient,
    reorganize_solution,
    evaluate_joule_power_conductance,
    compute_voltage_sum,
    ELECTRIC_TIME_STEP_NUMBER,
)
from electromagnetics.operating_conditions import (
    update_em_operating_conditions,
    evaluate_total_operating_current as _evaluate_total_operating_current,
    user_defined_current,
)

# import functions
from utility_functions.auxiliary_functions import (
    set_diagnostic,
)
from conductor.input_validator import ConductorInputValidator
from utility_functions.gen_flow import gen_flow
from utility_functions.output import (
    save_properties,
    save_convergence_data,
    save_geometry_discretization,
)
from utility_functions.plots import update_real_time_plots, create_legend_rtp
from thermal.temperature_field import (
    solid_components_temperature_initialization,
)
from conductor.input_loader import ConductorInputLoader
import conductor.input_validator as validator

# Get the logger specified in the file
conductorlogger = logging.getLogger("opensc2Logger.conductor")


class Conductor:

    KIND = "Conductor"
    CHUNCK_SIZE = 100
    # Minimum pressure drop between channels in hydraulic parallel, used to
    # regularize the transport coefficients of open interfaces.
    Delta_p_min = 1e-4  # Pa
    # Localized pressure drop coefficient between two channels in hydraulic
    # parallel.
    k_loc = 1.0  # ~
    # Lambda velocity parameter (--> 0 if porous wall, --> 1 if helicoidal wall).
    lambda_v = 1.0  # ~

    def __init__(self: Self, base_path: str, ICOND: int, magnet_file_name: str):
        """Makes an instance of class conductor.

        Args:
            self (Self): conductor object.
            base_path (str): base path to input files.
            sheetConductorsList (list): list of sheets available in input file conductor_definition.
            ICOND (int): conductor counter.
            magnet_file_name (str): name of the magnet definition file relative to base_path.
        """

        self.base_path = Path(base_path)
        self.counter = ICOND

        loader = ConductorInputLoader(self.base_path, self.counter)
        self.file_paths = loader.load_input_files()
        self.inputs = loader.load_conductor_inputs()
        self.operations = loader.load_conductor_operations()
        self.mesh = loader.load_grid_input(self.inputs.zlength)
        self.coupling = loader.load_coupling_data()

        validator.check_equipotential_surface_coordinate(self.operations, 
                                                         self.inputs.zlength)
        validator.check_for_unphysical_coupling_data(self.coupling) 

        self.name = loader.conductor_name
        self.identifier = loader.conductor_identifier
        self.number = int(self.identifier.split("_")[1])

        self.workbook_name = self.base_path / magnet_file_name
        self.workbook_sheet_name = loader.conductor_sheet_names

        # TODO: loading and checking external files if any
        
        self.inventory: ComponentInventory = ComponentInventory.empty()


    # end method __init__ (cdp, 11/2020)

    def initialize_with_simulation(self: Self, simulation: object):
        """Initialize conductor with simulation object. Deferred from constructor.
        
        This method should be called after the conductor is created to complete
        initialization steps that require the simulation object.

        Args:
            self (Self): conductor object.
            simulation (object): simulation object.
        """
        # call method conductor_components_instance to make instance of conductor components (cdp, 11/2020)
        self.conductor_components_instance(simulation)

        # Total strand/superconductor cross sections can only be evaluated
        # once the components exist in the inventory.
        self.__get_total_cross_section()

        # Call function evaluate_component_coordinates to build grid coordinates.
        conductor_mesh.evaluate_component_coordinates(self, simulation)

        # Call private method __initialize_attributes to initialize all the other useful and necessary attributes of class Conductor.
        self.__initialize_attributes(simulation)


    def __repr__(self):
        return f"{self.__class__.__name__}(Type: {self.KIND}, identifier: {self.identifier})"
    

    def conductor_components_instance(self, simulation):
        """Makes instances of conductor components defined in the input Excel workbooks."""
        dict_file_path = dict(
            input=self.file_paths.structure_elements_path,
            operation=self.file_paths.operation_path,
        )
        wb_input, _wb_operations, listOfComponents = (
            ConductorInputValidator.validate_component_workbooks(
                self,
                dict_file_path["input"],
                dict_file_path["operation"],
            )
        )

        context = ComponentBuildContext(simulation=simulation, conductor=self)
        factory = ComponentFactory(context)

        for sheetID in listOfComponents:
            sheet = wb_input[sheetID]
            numObj = int(sheet.cell(row=1, column=2).value)
            for comp in factory.create(sheet, numObj, dict_file_path):
                self.inventory.add(comp)

    def __get_total_cross_section(self):
        """Private method that evaluates: 1) the total cross section of strands and stacks object of the conductor; 2) the total cross section of superconducting materials of the conductor."""

        self.total_so_cross_section = np.array(
            [
                obj.inputs.cross_section
                for obj in self.inventory.strands.collection
            ]
        ).sum()
        self.total_sc_cross_section = np.array(
            [
                obj.cross_section["sc"]
                for obj in self.inventory.strands.collection
                if isinstance(obj, (StackComponent, StrandMixedComponent))
            ]
        ).sum()

    def __build_equation_idx(self):
        """Private method that evaluates the index of the velocity, pressure and temperature equation of the FluidComponent objects, collecting them in a dictionary of NamedTuple, together with the index of the temperature equation of the SolidComponent objects stored as integer in the same dictionary.
        """
        
        # Constructor of the namedtuple to store the index of the equations for 
        # FluidComponent objects.
        Fluid_eq_idx = namedtuple(
            "Fluid_eq_idx",
            ("velocity","pressure","temperature")
        )

        # self.equation_index -> dict: collection of NamedTuple with the index
        # of velocity, pressure and temperaure equation for FluidComponent
        # objects and of integer for the index of the temperature equation of
        # SolidComponent.
        
        # Build dictionary of NamedTuple with the index of the equations for 
        # FluidComponent objects exploiting dictionary comprehension.
        self.equation_index = {
            fcomp.identifier:Fluid_eq_idx(
                # velocity equation index
                velocity=fcomp_idx,
                # pressure equation index
                pressure=fcomp_idx + self.inventory.fluids.number,
                # temperature equation index
                # Exploit left binary shift, equivalent to:
                # fcomp_idx + 2 * conductor.inventory.fluids.number
                temperature=(
                    fcomp_idx
                    + (self.inventory.fluids.number << 1)
                )
            )
            for fcomp_idx,fcomp in enumerate(
                self.inventory.fluids.collection
            )
        }
        
        # Update dictionary equation_index with integer corresponding to the 
        # index of the equations for SolidComponent objects exploiting 
        # dictionary comprehension and dictionary method update.
        self.equation_index.update(
            {
                scomp.identifier: scomp_idx + self.dict_N_equation[
                    "FluidComponent"
                ]
                for scomp_idx,scomp in enumerate(
                    self.inventory.solids.collection
                )
            }
        )

    def __initialize_attributes(self: Self, simulation: object):
        """Private method that initializes usefull attributes of conductor object.

        Args:
            self (Self):

        Args:
            self (Self): conductor object
            simulation (object): simulation object.

        Raises:
            ValueError: raise error if spatial coordinates in sheet Time_evolutions of file conductor_diagnostic are larger than the conductor length.
            ValueError: raise error if time values in sheet Spatial_distribution of file conductor diagnostic are larger than the end time of the simulation.
        """

        self.dict_topology = dict()  # dictionary declaration (cdp, 09/2020)
        self.dict_interf_peri = dict()  # dictionary declaration (cdp, 07/2020)
        # Call method Get_conductor_topology to evaluate conductor topology: \
        # interfaces between channels, channels and solid components and between \
        # solid components(cdp, 09/2020)
        conductor_topology.get_conductor_topology(self, simulation.environment)

        # Call function get_conductor_interfaces to get the interfaces
        # between conductor components. This function could replace
        # get_conductor_topology but this change must carefully discusse with
        # both prof Savoldi and prof Savino.
        conductor_topology.get_conductor_interfaces(self, simulation.environment)

        self.node_fields = FieldContainer(GridLocation.NODE)
        self.gauss_fields = FieldContainer(GridLocation.GAUSS)

        self.heat_rad_jk = dict()
        self.heat_exchange_jk_env = dict()

        # **NUMERICS**
        # evaluate value of theta_method according to flag METHOD (cdo, 08/2020)
        # Adams Moulton value is temporary and maybe non correct
        _ = {
            MethodFlag.BACKWARD_EULER: 1.0,
            MethodFlag.CRANK_NICOLSON: 0.5,
            MethodFlag.ADAMS_MOULTON_4TH_ORDER: 1.0 / 24.0,
        }
        self.theta_method = _[self.inputs.electric_method]
        self.electric_theta = _[self.inputs.electric_method]
        conductorlogger.debug(f"Defined electric_theta\n")
        ## Evaluate parameters useful in function \
        # Transient_solution_functions.py\STEP (cdp, 07/2020)
        # dict_N_equation keys meaning:
        # ["FluidComponent"]: total number of equations for FluidComponent \
        # objects (cdp, 07/2020);
        # ["StrandComponent"]: total number of equations for StrandComponent objects (cdp, 07/2020);
        # ["JacketComponent"]: total number of equations for JacketComponent objects (cdp, 07/2020);
        # ["SolidComponent"]: total number of equations for SolidComponent \
        # objects (cdp, 07/2020);
        # ["NODOFS"]: total number of equations for for each node, i.e. Number Of \
        # Degrees Of Freedom, given by: \
        # 3*(number of channels) + (number of strands) + (number of jackets) \
        # (cdp, 07/2020);
        self.dict_N_equation = dict(
            FluidComponent=3 * self.inventory.fluids.number,
            StrandComponent=self.inventory.strands.number,
            JacketComponent=self.inventory.jackets.number,
            SolidComponent=self.inventory.solids.number,
        )
        # necessary since it is not allowed to use the value of a dictionary key \
        # before that the dictionary is fully defined (cdp, 09/2020)
        self.dict_N_equation.update(
            NODOFS=self.dict_N_equation["FluidComponent"]
            + self.dict_N_equation["SolidComponent"]
        )
        # Exploit left binary shift, equivalent to:
        # self.dict_N_equation["NODOFS2"] = 2 * self.dict_N_equation["NODOFS"]
        self.dict_N_equation["NODOFS2"] = self.dict_N_equation["NODOFS"] << 1
        # dict_band keys meaning:
        # ["Half"]: half band width, including main diagonal (IEDOFS) (cdp, 09/2020)
        # ["Main_diag"]: main diagonal index within the band (IHBAND) (cdp, 09/2020)
        # ["Full"]: full band width, including main diagonal (IBWIDT) (cdp, 09/2020)
        self.dict_band = dict(
            Half=2 * self.dict_N_equation["NODOFS"],
            Main_diag=2 * self.dict_N_equation["NODOFS"] - 1,
            Full=4 * self.dict_N_equation["NODOFS"] - 1,
        )
        # self.MAXDOF = self.dict_N_equation["NODOFS"]*MAXNOD
        self.EQTEIG = np.zeros(self.dict_N_equation["NODOFS"])
        # dict_norm keys meaning:
        # ["Solution"]: norm of the solution (cdp, 09/2020)
        # ["Change"]: norm of the solution variation wrt the previous time step \
        # (cdp, 09/2020)
        self.dict_norm = dict(
            Solution=np.zeros(self.dict_N_equation["NODOFS"]),
            Change=np.zeros(self.dict_N_equation["NODOFS"]),
        )
        
        # Call method __build_equation_idx to build attribute equation_index;
        # self.equation_index -> dict: collection of NamedTuple with the index
        # of velocity, pressure and temperaure equation for FluidComponent
        # objects and of integer for the index of the temperature equation of
        # SolidComponent. This is used in funcion step to solve the thermal 
        # hydraulic problem.
        self.__build_equation_idx()

        # evaluate attribute EIGTIM exploiting function
        # evaluate_time_accuracy_eigenvalue (cdp, 08/2020)
        # Deferred import: utility_functions.transient_solution_functions
        # imports Conductor from this module, so importing it at module
        # level here would create a circular import.
        from utility_functions.transient_solution_functions import (
            evaluate_time_accuracy_eigenvalue,
        )
        evaluate_time_accuracy_eigenvalue(self)
        path_diagnostic = self.file_paths.diagnostics_path
        # Load the content of column self.ID of sheet Space in file conductors_disgnostic.xlsx as a series and convert to numpy array of float.
        df = pd.read_excel(
            path_diagnostic,
            sheet_name="Spatial_distribution",
            skiprows=2,
            header=0,
            usecols=[self.identifier],
        )

        self.Space_save = (
            df.iloc[:, 0]
            .dropna()
            .to_numpy()
            .astype(float)
        )
        # Adjust the user defined diagnostic.
        self.Space_save = set_diagnostic(
            self.Space_save, lb=0.0, ub=simulation.transient_input["TEND"]
        )
        # Check on spatial distribution diagnostic.
        if self.Space_save.max() > simulation.transient_input["TEND"]:
            raise ValueError(
                f"File {self.file_paths.diagnostics_path}, sheet Space, conductor {self.identifier}: impossible to save spatial distributions at time {self.Space_save.max()} s since it is larger than the end time of the simulation {simulation.transient_input['TEND']} s.\n"
            )
        # End if self.Space_save.max() > simulation.transient_input["TEND"]
        # index pointer to save solution spatial distribution (cdp, 12/2020)
        self.i_save = 0
        # list of number of time steps at wich save the spatial discretization
        self.num_step_save = np.zeros(self.Space_save.shape, dtype=int)
        # Load the content of column self.identifier of sheet Time in file conductors_disgnostic.xlsx as a series and convert to numpy array of float.
        df = pd.read_excel(
            path_diagnostic,
            sheet_name="Time_evolution",
            skiprows=2,
            header=0,
            usecols=[self.identifier],
        )

        self.Time_save = (
            df.iloc[:, 0]
            .dropna()
            .to_numpy()
            .astype(float)
        )

        # Adjust the user defined diagnostic.
        self.Time_save = set_diagnostic(
            self.Time_save, lb=0.0, ub=self.inputs.zlength
        )
        # Check on time evolution diagnostic.
        if self.Time_save.max() > self.inputs.zlength:
            raise ValueError(
                f"File {self.file_paths.diagnostics_path}, sheet Time, conductor {self.identifier}: impossible to save time evolutions at axial coordinate {self.Time_save.max()} s since it is ouside the computational domain of the simulation [0, {self.inputs.zlength}] m.\n"
            )
        # End if self.Time_save.max() > self.inputs.zlength

        # declare dictionaries to store Figure and axes objects to constructi real \
        # time figures (cdp, 10/2020)
        self.dict_Figure_animation = dict(T_max=dict(), mfr=dict())
        self.dict_axes_animation = dict(T_max=dict(), mfr=dict())
        self.dict_canvas = dict(T_max=dict(), mfr=dict())
        self.color = ["r*", "bo"]

        # Introduced for the electric module.
        self.total_elements = (
            self.mesh.number_of_elements * self.inventory.all_components.number
        )
        self.total_nodes = (
            self.mesh.number_of_elements + 1
        ) * self.inventory.all_components.number

        self.total_elements_current_carriers = (
            self.mesh.number_of_elements * self.inventory.strands.number
        )
        self.total_nodes_current_carriers = (
            self.mesh.number_of_elements + 1
        ) * self.inventory.strands.number

        # Extract the current-carrier (strand) blocks of the electric
        # conductance matrices.
        # +1 keeps into account the Environment object in the
        # conductor_coupling workbook.
        strand_block = slice(
            self.inventory.fluids.number + 1,
            self.inventory.fluids.number + self.inventory.strands.number + 1,
        )
        self.electric_conductance = (
            self.coupling.electric_conductance.matrix[strand_block, strand_block]
        )
        self.electric_conductance_mode = (
            self.coupling.electric_conductance_mode.matrix[strand_block, strand_block]
        )

        # Initialize resistance matrix to a dummy value (sparse matrix)
        self.electric_resistance_matrix = diags(
            10.0 * np.ones(self.total_elements_current_carriers),
            offsets=0,
            shape=(
                self.total_elements_current_carriers,
                self.total_elements_current_carriers,
            ),
            format="csr",
            dtype=float,
        )

        self.inductance_matrix = np.zeros(
            (
                self.total_elements_current_carriers,
                self.total_elements_current_carriers,
            )
        )

        self.electric_conductance_matrix = csr_matrix(
            (self.total_nodes_current_carriers, self.total_nodes_current_carriers),
            dtype=float,
        )

        self.build_electric_topology_flag = True
        self.electric_mass_matrix = lil_matrix(
            (
                self.total_elements_current_carriers
                + self.total_nodes_current_carriers,
                self.total_elements_current_carriers
                + self.total_nodes_current_carriers,
            ),
            dtype=float,
        )

        self.equipotential_node_index = np.zeros(
            (
                self.operations.number_of_equipotential_surfaces
                if self.operations.do_equipotential_surfaces_exist
                else 0,
                self.inventory.strands.number,
            ),
            dtype=int,
        )

        nn = 0
        for obj in self.inventory.strands.collection:
            nn += obj.operations.fix_potential_number

        self.fixed_potential_index = np.zeros(nn, dtype=int)
        self.fixed_potential_value = np.zeros(nn)

        # Initialization moved in method build_electric_known_term_vector.
        # self.node_fields.op_current = np.zeros(self.total_nodes_current_carriers)

        self.electric_known_term_vector = np.zeros(
            self.total_elements_current_carriers + self.total_nodes_current_carriers
        )

        self.electric_right_hand_side = np.zeros(
            self.total_elements_current_carriers + self.total_nodes_current_carriers
        )

        # Electric time initialization, to be understood where to actually do
        # this
        self.electric_time = 0.0  # s
        # Initialize the number of electric time steps to 0. This attribute 
        # will be updated at each electric time step; for each thermal time 
        # step it will start from 1. Value 0 is assumed only at initialization.
        self.cond_el_num_step = 0

        conductor_mesh.initialize_mesh_dataframes(self)

    ##############################################################################

    def initialization(self, simulation):

        time_simulation = simulation.simulation_time[-1]
        sim_name = simulation.transient_input["SIMULATION"]
        # Total number of equations for each conductor (cdp, 09/2020)
        self.dict_N_equation["Total"] = (
            self.dict_N_equation["NODOFS"] * self.mesh.number_of_nodes
        )
        # initialize conductor time values, it can be different for different \
        # conductors since the conductor time step can be different (cdp, 10/202)
        self.cond_time = [time_simulation]
        # Initialize conductor time step counter
        self.cond_num_step = 0

        """
    if (imsourcefun.eq.0) then
      premed(icond)=dble((preinl(icond)+preout(icond))/2.)
      temmed(icond)=dble((teminl(icond)+temout(icond))/2.)
    endif
      CALL GETsource  (NNODES,ncond) ### si trova in 
    ENTALINLET = hhe(premed(icond),temmed(icond))
    Questo per il momento non lo considero!!!!! (cdp)
    """

        gen_flow(self)
        # C*SET THE INITIAL VALUE OF THE FLOW VARIABLES (LINEAR P AND T)

        temp_ave = np.zeros(self.mesh.number_of_nodes)
        self.enthalpy_balance = 0.0
        self.enthalpy_out = 0.0
        self.enthalpy_inl = 0.0
        # For each channel evaluate following fluid properties (array):velocity, \
        # pressure, temperature, density, Reynolds, Prandtl (cdp, 06/2020)
        # N.B. questo loop si potrebbe fare usando map.
        for fluid_comp in self.inventory.fluids.collection:
            # Compute pressure, temperature and velocity in nodal points according to the initial conditions
            fluid_comp.coolant._eval_nodal_pressure_temperature_velocity_initialization(
                self
            )
            # Build namedtuples fluid_comp.inl_idx and fluid_comp.inl_idx with 
            # the index used to assign the inlet and outlet BC.
            fluid_comp.build_th_bc_index(self)

        # Call function SolidComponents_T_initialization to initialize \
        # SolidComponent temperature spatial distribution from FluidComponent \
        # temperature or from input values according to flag INTIAL (cdp, 12/2020)
        solid_components_temperature_initialization(self)

        # Nested loop jacket - jacket.
        for rr, jacket_r in enumerate(self.inventory.jackets.collection):
            # np array of shape (Node, 1) to avoid broadcasting error.
            jacket_r.radiative_heat_env = np.zeros(
                (jacket_r.node_fields.temperature.size, 1)
            )
            for _, jacket_c in enumerate(
                self.inventory.jackets.collection[rr + 1 :]
            ):
                jacket_r.radiative_heat_inn[
                    f"{jacket_r.identifier}_{jacket_c.identifier}"
                ] = np.zeros((jacket_r.node_fields.temperature.size, 1))
                jacket_c.radiative_heat_inn[
                    f"{jacket_r.identifier}_{jacket_c.identifier}"
                ] = np.zeros((jacket_c.node_fields.temperature.size, 1))
            # End for cc.
        # End for rr.

        # IOP=IOP0(icond)
        for obj in self.inventory.solids.collection:
            # Compute fractions of the total current that flows in
            # superconductor cross section of each strand or stack object if in
            # superconducting regime and fractions of the total current that
            # flows in the total cross section of each strand or stack object
            # if in current sharing regime.
            obj.get_current_fractions(
                self.total_sc_cross_section, self.total_so_cross_section, self.inventory
            )

        # Initialize electromagnetic quantities in both nodal and Gauss 
        # points.
        self.operating_conditions_em()
        # Initialize thermal hydraulic quantities in both nodal and Gauss 
        # points.
        self.operating_conditions_th_initialization(simulation)

        # # Loop on SolidComponent (cdp, 01/2021)
        # # N.B. questo loop si potrebbe fare usando map.
        # for s_comp in self.inventory.solids.collection:
        #     # compute, average density, thermal conductivity, specifi heat at \
        #     # constant pressure and electrical resistivity at initial \
        #     # SolidComponent temperature in nodal points (cdp, 01/2021)
        #     s_comp.eval_sol_comp_properties(self.inventory)
        # # end for s_comp.

        # Loop to initialize electric related quantities for each
        # SolidComponent object.
        # N.B. remember that JacketComponent objects do not carry current for
        # the time being so these quantities will remain 0.
        for obj in self.inventory.solids.collection:
            obj.initialize_electric_quantities(self)

        # Initialize to zeros all quantities related to heat source in nodal
        # points.
        thermal_heat_sources._build_heat_source_nodal_pt(self, simulation)

        # ENERGY BALANCE FLUID COMPONENTS
        for fluid_comp in self.inventory.fluids.collection:
            # Evaluate the density (if necessary) and the mass flow rate in 
            # points (nodal = True by default)
            fluid_comp.coolant._compute_density_and_mass_flow_rates_nodal_gauss(self)
            temp_ave = (
                temp_ave
                + fluid_comp.coolant.node_fields.temperature
                / self.inventory.fluids.number
            )
            # Enthalpy balance: dt*sum((mdot*w)_out - (mdot*w)_inl), used to 
            # check the imposition of SolidComponent temperature initial 
            # spatial distribution.
            # N.B. queste istruzioni posso inserirle in un metodo della classe.
            self.enthalpy_balance = self.enthalpy_balance + simulation.transient_input[
                "STPMIN"
            ] * (
                fluid_comp.coolant.node_fields.mass_flow_rate[-1]
                * fluid_comp.coolant.node_fields.total_enthalpy[-1]
                - fluid_comp.coolant.node_fields.mass_flow_rate[0]
                * fluid_comp.coolant.node_fields.total_enthalpy[0]
            )
            self.enthalpy_out = (
                self.enthalpy_out
                + simulation.transient_input["STPMIN"]
                * fluid_comp.coolant.node_fields.mass_flow_rate[-1]
                * fluid_comp.coolant.node_fields.total_enthalpy[-1]
            )
            self.enthalpy_inl = (
                self.enthalpy_inl
                + simulation.transient_input["STPMIN"]
                * fluid_comp.coolant.node_fields.mass_flow_rate[0]
                * fluid_comp.coolant.node_fields.total_enthalpy[0]
            )

        # Initialize the Energy of the SolidComponent (cdp, 12/2020)
        self.E_sol_ini = 0.0
        self.E_sol_fin = 0.0
        self.E_str_ini = 0.0
        self.E_str_fin = 0.0
        self.E_jk_ini = 0.0
        self.E_jk_fin = 0.0
        # Loop on SolidComponent to evaluate the total initial energy of \
        # SolidComponent, used to check the imposition of SolidComponent \
        # temperature initial spatial distribution (cdp, 12/2020)
        # N.B. questo loop si potrebbe fare usando map.
        for s_comp in self.inventory.solids.collection:
            # N.B. queste istruzioni posso inserirle in un metodo della classe.
            self.E_sol_ini = self.E_sol_ini + s_comp.inputs.cross_section * np.sum(
                (
                    self.mesh.node_coordinates[1 : self.mesh.number_of_nodes]
                    - self.mesh.node_coordinates[0:-1]
                )
                * s_comp.gauss_fields.total_density
                * s_comp.gauss_fields.total_isobaric_specific_heat
                * s_comp.gauss_fields.temperature
            )
            if s_comp.name != "Z_JACKET":
                self.E_str_ini = self.E_str_ini + s_comp.inputs.cross_section * np.sum(
                    (
                        self.mesh.node_coordinates[1 : self.mesh.number_of_nodes]
                        - self.mesh.node_coordinates[0:-1]
                    )
                    * s_comp.gauss_fields.total_density
                    * s_comp.gauss_fields.total_isobaric_specific_heat
                    * s_comp.gauss_fields.temperature
                )
            else:
                self.E_jk_ini = self.E_jk_ini + s_comp.inputs.cross_section * np.sum(
                    (
                        self.mesh.node_coordinates[1 : self.mesh.number_of_nodes]
                        - self.mesh.node_coordinates[0:-1]
                    )
                    * s_comp.gauss_fields.total_density
                    * s_comp.gauss_fields.total_isobaric_specific_heat
                    * s_comp.gauss_fields.temperature
                )
        # end for s_comp (cdp, 12/2020)

        # Construct and initialize dictionary dict_Step to correctly apply the \
        # method that solves the transient (cdp, 10/2020)
        if self.inputs.thermohydraulic_method in (
            MethodFlag.BACKWARD_EULER,
            MethodFlag.CRANK_NICOLSON,
        ):
            # Backward Euler or Crank-Nicolson (cdp, 10/2020)
            self.dict_Step = dict(
                SYSLOD=np.zeros((self.dict_N_equation["Total"], 2)),
                SYSVAR=np.zeros((self.dict_N_equation["Total"], 1)),
            )
        elif self.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton order 4 (cdp, 10/2020)
            self.dict_Step = dict(
                SYSLOD=np.zeros((self.dict_N_equation["Total"], 4)),
                SYSVAR=np.zeros((self.dict_N_equation["Total"], 3)),
                # AM4_AA: four matrices of size Full * Total
                AM4_AA=np.zeros(
                    (4,self.dict_band["Full"],self.dict_N_equation["Total"])
                ),
            )
        # end if self.inputs

        # Assign initial values to key SYSVAR (cdp, 10/2020)
        for jj, fluid_comp in enumerate(self.inventory.fluids.collection):
            # velocity (cdp, 10/2020)
            self.dict_Step["SYSVAR"][
                jj : self.dict_N_equation["Total"] : self.dict_N_equation["NODOFS"], 0
            ] = fluid_comp.coolant.node_fields.velocity
            # pressure (cdp, 10/2020)
            self.dict_Step["SYSVAR"][
                jj
                + self.inventory.fluids.number : self.dict_N_equation[
                    "Total"
                ] : self.dict_N_equation["NODOFS"],
                0,
            ] = fluid_comp.coolant.node_fields.pressure
            # temperature (cdp, 10/2020)
            self.dict_Step["SYSVAR"][
                jj
                + 2
                * self.inventory.fluids.number : self.dict_N_equation[
                    "Total"
                ] : self.dict_N_equation["NODOFS"],
                0,
            ] = fluid_comp.coolant.node_fields.temperature
        # end for jj (cdp, 10/2020)
        for ll, comp in enumerate(self.inventory.solids.collection):
            # solid components temperature (cdp, 10/2020)
            self.dict_Step["SYSVAR"][
                ll
                + self.dict_N_equation["FluidComponent"] : self.dict_N_equation[
                    "Total"
                ] : self.dict_N_equation["NODOFS"],
                0,
            ] = comp.node_fields.temperature
        # end for ll (cdp, 10/2020)
        if self.dict_Step["SYSVAR"].shape[-1] > 1:
            # if this is true, it means that an higher order method than \
            # Crank-Nicolson is applied to solve the transient (cdp, 10/2020)
            for cc in range(1, self.dict_Step["SYSVAR"].shape[-1]):
                # Copy the values of the first colum in all the other columns, like \
                # they are the results of a dummy initial steady state (cdp, 10/2020)
                self.dict_Step["SYSVAR"][:, cc] = self.dict_Step["SYSVAR"][:, 0].copy()
            # end for cc (cdp, 10/2020)
        # end if self.dict_Step["SYSVAR"].shape[-1] (cdp, 10/2020)

        conductorlogger.debug(
            f"Before call function {save_geometry_discretization.__name__}.\n"
        )
        save_geometry_discretization(
            self.inventory.all_components.collection,
            simulation.dict_path[f"Output_Initialization_{self.identifier}_dir"],
        )
        conductorlogger.debug(
            f"After call function {save_geometry_discretization.__name__}.\n"
        )

        conductorlogger.debug(f"Before call function {save_properties.__name__}.\n")
        # Call function Save_properties to save conductor inizialization
        save_properties(
            self, simulation.dict_path[f"Output_Initialization_{self.identifier}_dir"]
        )
        conductorlogger.debug(f"After call function {save_properties.__name__}.\n")

        # Call function update_real_time_plot
        update_real_time_plots(self)
        create_legend_rtp(self)

    # end method initialization

    ############################################################################

    ##### ELECTRIC PREPROCESSING ############

    def __build_connectivity_current_carriers(self):
        build_connectivity_current_carriers(self)


    def __build_incidence_matrix(self):
        build_incidence_matrix(self)

    def __build_electric_resistance_matrix(self):
        build_resistance_matrix(self)

    def __contact_current_carriers(self):
        detect_contacts(self)

    def __build_contact_incidence_matrix(self):
        build_contact_incidence_matrix(self)

    def __build_electric_conductance_matrix(self):
        build_conductance_matrix(self)

    def electric_preprocessing(self):
        """Method that allows to evaluate most of the quatities and data structures needed for the electric calculation.

        The expensive topology structures (nodal coordinates, connectivity,
        incidence and contact matrices, conductances, inductances and the
        boundary-condition reduction operator) depend only on the mesh
        geometry, so they are built once and reused until the mesh changes.
        Only the temperature- and current-dependent resistance and stiffness
        matrices are rebuilt at every electric time step.
        """
        if self.build_electric_topology_flag:
            self.__build_electric_topology()
            if self.mesh.mesh_type not in {MeshType.ADAPTED, MeshType.FROM_FILE}:
                # Discretization grid does not change at each time step, so
                # all topology structures stay valid for the whole transient.
                self.build_electric_topology_flag = False

        # Build electric resistance matrix: changes with temperature (thermal
        # time step) and with current (electric time step, since the
        # superconductor resistivity is current dependent).
        self.__build_electric_resistance_matrix()

        # Build electric stiffness matrix from the fresh resistance matrix
        # and the cached incidence and conductance blocks.
        self.__build_electric_stiffness_matrix()

    def __build_electric_topology(self):
        """Private method that builds all the geometry/topology dependent electric structures."""

        nn = 0

        for collection in (
            self.inventory.fluids,
            self.inventory.strands,
            self.inventory.jackets,
        ):
            # Build nodal coordinates
            conductor_mesh.build_nodal_coordinates(self, nn, collection)

            # Build connectivity matrix
            conductor_mesh.build_connectivity(self, nn, collection)
            nn += collection.number
        # End for collection

        # Build the connectivity matrix for the reduced system of components:
        # keeps into account only the StrandComponent ones.
        # conductorlogger.debug(
        #     f"Before call method {self.__build_connectivity_current_carriers.__name__}, operates on StrandComponents only.\n"
        # )
        self.__build_connectivity_current_carriers()
        # conductorlogger.debug(
        #     f"After call method {self.__build_connectivity_current_carriers.__name__}, operates on StrandComponents only.\n"
        # )

        # Convert index to categorical
        # conductorlogger.debug(
        #     f"Before convert index of dataframe self.connectivity_matrix to categorical.\n"
        # )
        self.connectivity_matrix.loc[:, "identifiers"] = self.connectivity_matrix.loc[
            :, "identifiers"
        ].astype("category")
        self.connectivity_matrix_current_carriers.loc[
            :, "identifiers"
        ] = self.connectivity_matrix_current_carriers.loc[:, "identifiers"].astype(
            "category"
        )
        # conductorlogger.debug(
        #     f"After convert index of dataframe self.connectivity_matrix to categorical.\n"
        # )

        # Compute node distance
        conductor_mesh.compute_node_distance(self)

        # Compute gauss node distance
        conductor_mesh.compute_gauss_node_distance(self)
        # conductorlogger.debug(
        #     f"After call method {self.__compute_gauss_node_distance.__name__}.\n"
        # )

        # conductorlogger.debug(
        #     f"Before call method {self.__build_incidence_matrix.__name__}.\n"
        # )
        # Build incidence matrix only for StrandComponent
        self.__build_incidence_matrix()
        # conductorlogger.debug(
        #     f"After call method {self.__build_incidence_matrix.__name__}.\n"
        # )

        if self.inventory.strands.number > 1:
            # There are more than 1 StrandComponent objects, therefore there
            # are contacts between StrandComponent objects and matrices
            # contact_incidence_matrix and electric_conductance_matix can be
            # built. If there is only one StrandComponent object
            # contact_incidence_matrix can not be defined while
            # electric_conductance_matix is full of 0 from initialization.

            # Find contacts between StrandComponent objects.
            # conductorlogger.debug(
            #     f"Before call method {self.__contact_current_carriers.__name__}.\n"
            # )
            self.__contact_current_carriers()
            # conductorlogger.debug(
            #     f"After call method {self.__contact_current_carriers.__name__}.\n"
            # )

            # Build contact incidence matrix
            # conductorlogger.debug(
            #     f"Before call method {self.__build_contact_incidence_matrix.__name__}.\n"
            # )
            # this method builds the contact incidence matrix for current carriers
            # only
            self.__build_contact_incidence_matrix()
            # conductorlogger.debug(
            #     f"After call method {self.__build_contact_incidence_matrix.__name__}.\n"
            # )

            # Build electric conductance matrix
            # conductorlogger.debug(
            #     f"Before call method {self.__build_electric_conductance_matrix.__name__}.\n"
            # )
            self.__build_electric_conductance_matrix()
            # conductorlogger.debug(
            #     f"After call method {self.__build_electric_conductance_matrix.__name__}.\n"
            # )

        # Build electric mass matrix (inductances only depend on the
        # geometry, so they belong to the topology structures).
        self.__build_electric_mass_matrix()

        # Assign equivalue surfaces
        self.__assign_equivalue_surfaces()

        # Assign fixed potential
        self.__assign_fix_potential()

        # Build the boundary-condition reduction operator used by
        # reduce_system at every electric time step.
        build_reduction_operator(self)

    def __build_electric_stiffness_matrix(self):
        build_stiffness_matrix(self)

    def __assign_equivalue_surfaces(self):
        assign_equipotential_surfaces(self)

    def __assign_fix_potential(self):
        assign_fixed_potential(self)

    def eval_total_operating_current(self):
        """Evaluate the total transport current boundary condition vector."""
        _evaluate_total_operating_current(self)

    def build_electric_known_term_vector(self):
        """Build the known-term vector for the electric module."""
        build_known_term_vector(self)

    def __build_electric_mass_matrix(self):
        build_mass_matrix(self)

    def __get_electric_time_step(self):
        """Private method that evaluates the electric time step according to user definition.

        A user-defined electric time step larger than the thermal time step
        is clamped to the thermal time step (a single electric sub-step per
        thermal step).

        Raises:
            ValueError: if electric time step is negative.
        """

        electric_step = self.inputs.electric_time_step
        if electric_step is None or (
            isinstance(electric_step, float) and np.isnan(electric_step)
        ):
            self.electric_time_step = self.time_step / ELECTRIC_TIME_STEP_NUMBER
        else:
            if electric_step <= 0.0:
                raise ValueError(
                    f"Electric time step must be > 0.0 s; current value is: {electric_step=} s\n"
                )
            self.electric_time_step = min(electric_step, self.time_step)

        self.electric_time_end = self.time_step

    def build_right_hand_side(self, foo: np.ndarray, bar: np.ndarray, idx: np.ndarray):
        """Build the RHS vector for the transient electric time step."""
        build_right_hand_side(self, foo, bar, idx)

    def __electric_solution_reorganization(self):
        reorganize_solution(self)

    def __get_total_joule_power_electric_conductance(self):
        evaluate_joule_power_conductance(self)

    def __compute_voltage_sum(self):
        compute_voltage_sum(self)

    def electric_method(self):
        """Solve the electric problem and post-process the solution."""
        if self.cond_num_step == 0:
            solve_steady_state(self)
        else:
            self.__get_electric_time_step()
            solve_transient(self)

        self.__electric_solution_reorganization()
        self.__compute_voltage_sum()
        self.__get_total_joule_power_electric_conductance()

    def build_heat_source(self, simulation):
        """Method that builds heat source therms in nodal and Gauss points for
        strand and jacket objects.

        Args:
            simulation (object): object with all information about the simulation.
        """
        thermal_heat_sources.build_heat_source(self, simulation)


    def operating_conditions_th_initialization(self,simulation):
        """Method that evaluates thermal hydraulic (th) operating conditions in both nodal and in Gauss points.
        To be called at initialization only since it avoids a second call to mesh.compute_derived_features, which is already called in __init__.
        """

        self.get_transport_coefficients(simulation)
        self.__eval_gauss_point_th(simulation)

    def operating_conditions_th(self,simulation):
        """Method that evaluates thermal hydraulic (th) operating conditions also in Gauss points."""

        self.mesh.compute_derived_features()
        self.get_transport_coefficients(simulation)

        self.__eval_gauss_point_th(simulation)

    def operating_conditions_em(self):
        """Update electromagnetic operating conditions at nodes and Gauss points."""
        update_em_operating_conditions(self)

    def __eval_gauss_point_th(self, simulation):
        """
        Method that evaluates transport coefficients at the Gauss point, i.e at the centre of the element.
        N.B. Fluid component properties are evaluated calling method get_transport_coefficients.
        """

        # call method get_transport_coefficients to evaluate transport
        # properties (heat transfer coefficient and friction factor) in each
        # Gauss point
        self.get_transport_coefficients(simulation, flag_nodal=False)

    def __eval_gauss_point_em(self):
        from electromagnetics.operating_conditions import evaluate_em_gauss_points
        evaluate_em_gauss_points(self)

    def post_processing(self, simulation):

        # bozza della funzione Post_processing

        # Evaluate the total final energy of SolidComponent, used to check
        # the imposition of SolidComponent temperature initial spatial
        # distribution (cdp, 12/2020)
        energy_balance.evaluate_final_energy(self)

        # call function evaluate_mass_energy_balance to get data for the
        # space convergence (cdp, 09/2020)
        energy_balance.evaluate_mass_energy_balance(self, simulation)
        # call function Save_convergence_data to save solution spatial \
        # distribution at TEND, together with the mass and energy balance results, \
        # to make the space convergence analisys (cdp, 12/2020)
        save_convergence_data(self, simulation.dict_path["Space_conv_output_dir"])
        # call function Save_convergence_data to save solution spatial \
        # distribution at TEND, together with the mass and energy balance results \
        # to make the time convergence analisys (cdp, 12/2020)
        save_convergence_data(
            self,
            simulation.dict_path["Time_conv_output_dir"],
            abs(simulation.n_digit_time),
            space_conv=False,
        )

    # end Post_processing

    def get_transport_coefficients(self, simulation, flag_nodal=True):  # (cpd 06/2020)

        """
    Method that compute channels friction factor and heat transfer coefficient \
    between channels, channel and solid component, solid components, both in \
    nodal and Gauss points. (cpd 06/2020)
    """
        # Properties evaluation in each nodal point (cdp, 07/2020)
        if flag_nodal:
            self.node_fields = self.evaluate_transport_coefficients(simulation, self.node_fields)
        # Properties evaluation in each Gauss point (cdp, 07/2020)
        elif flag_nodal == False:
            self.gauss_fields = self.evaluate_transport_coefficients(
                simulation, self.gauss_fields, flag_nodal=False
            )

    # end method get_transport_coefficients (cdp, 09/2020)

    def evaluate_transport_coefficients(self, simulation, dict_dummy, flag_nodal=True):

        """
        Method that actually computes channels friction factor and heat
        transfer coefficient between channels, channel and solid component,
        solid components, both in nodal and Gauss points.
        """

        # Alias
        interf_flag = self.coupling.contact_perimeter_flag

        # Evaluate coolant properties, steady-state HTC and friction factor
        # for each fluid component; must run before any of the per-interface
        # HTC evaluations below.
        htc_evaluation.evaluate_channel_transport_properties(
            self, simulation, flag_nodal
        )

        # htc dummy dictionary and its sub-dictionary declaration (cpd 09/2020)
        dict_dummy.HTC = dict()
        dict_dummy.HTC["ch_ch"] = dict()
        dict_dummy.HTC["ch_ch"]["Open"] = dict()
        dict_dummy.HTC["ch_ch"]["Close"] = dict()
        dict_dummy.HTC["ch_sol"] = dict()
        dict_dummy.HTC["sol_sol"] = dict()
        dict_dummy.HTC["sol_sol"]["cond"] = dict()
        dict_dummy.HTC["sol_sol"]["rad"] = dict()
        dict_dummy.HTC["env_sol"] = dict()

        # Counters to check the number of the different possible kinds of
        # interfaces (cdp, 09/2020). Evaluate each interface kind in turn;
        # order matters since downstream code (e.g. mass_energy_balance)
        # reads HTC dict keys written here.
        htc_len = 0
        htc_len += htc_evaluation.evaluate_channel_solid_htc(
            self, simulation, dict_dummy, flag_nodal
        )
        htc_len += htc_evaluation.evaluate_channel_channel_htc(
            self, simulation, dict_dummy, flag_nodal
        )
        htc_len += htc_evaluation.evaluate_solid_solid_htc(
            self, simulation, dict_dummy, flag_nodal
        )
        htc_len += htc_evaluation.evaluate_environment_solid_htc(
            self, simulation, dict_dummy, flag_nodal
        )

        # Check number of evaluated interface htc (cdp, 06/2020)

        # Evaluate total number of user defined interfaces.
        total_interf_number = np.abs(interf_flag.matrix).sum()
        if htc_len != total_interf_number:
            raise ValueError(
                f"ERROR!!! Number of interface and number of evaluated interface htc mismatch: {total_interf_number} != {htc_len}"
            )

        return dict_dummy

    # end method evaluate_transport_coefficients (cdp, 09/2020)

    def compute_radiative_heat_exhange_jk(self):
        """Method that evaluates the radiative heat exchanged by radiation between jackets."""
        thermal_radiation.compute_radiative_heat_exchange_jk(self)

    def compute_heat_exchange_jk_env(self, environment):
        """Method that computes the heat exchange between the outer surface of the conductor and the environment by convection and or radiation.

        Args:
            environment (object): environment object.
        """
        thermal_radiation.compute_heat_exchange_jk_env(self, environment)

    def load_user_defined_quantity(self, simulation, key, sheet_name):
        """[summary]

        Args:
            conductor ([type]): [description]

        Returns:
            [type]: [description]
        """
        fname = getattr(self.file_paths, key.lower())
        file_extension = str(fname).split(".")[1]
        # Call function used to open the file
        return simulation.func_open_aux[file_extension](
            fname, sheet_name, simulation.default_vals[file_extension]
        )

        # End method load_user_defined_quantity.
