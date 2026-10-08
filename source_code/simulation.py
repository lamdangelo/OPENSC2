from decimal import Decimal
import adapters.input.yaml_input_registry as yaml_input_registry
from adapters.input import open_input_source
from adapters.input.conductor_loader import load_conductor_definition
import numpy as np
import pandas as pd
import os
from typing import Union

from conductor.conductor import Conductor
from electromagnetics.electromagnetic_flags import CurrentMode
from adapters.input.environment_loader import load_environment
from hydraulics.network.coupling import build_coupled_network
from numerics.coupled_step import solve_coupled_conductors_step
from hydraulics.formulation import resolve_hydraulic_formulation
from hydraulics.hydraulic_flags import (
    HydraulicFormulation,
    MASS_FLOW_FORMULATIONS,
)
from numerics.transient_solution_functions import get_time_step, step
import adapters.output.simulation_paths as simulation_paths
from adapters.output.checkpoint import restore_from_checkpoint
from adapters.output.observers import ConsoleProgressObserver, OutputObserver
from adapters.plotting.observers import FigureObserver, RealTimePlotObserver
from application.observers import CompositeObserver
from adapters.output.diagnostics_npz import NpzDiagnosticsSink
from adapters.fluid_properties import coolprop_provider
from ports.fluid_properties import COOLANT_PROPERTY_ALIASES, register_provider_factory
from adapters.output.simulation_log import SimulationSummary


class Simulation:

    def __init__(self, base_path, step_callback=None):

        # Optional per-step hook, called at the top of every transient
        # iteration with the simulation as argument (drivers use it for
        # time-dependent boundary schedules, event detection and source
        # switching). A truthy return value stops the transient.
        self.step_callback = step_callback
        # Composition root: the domain resolves fluid properties through the
        # port; the CoolProp adapter provides them.
        register_provider_factory(coolprop_provider.provider_for)
        # Where the solvers drop state dumps when a step turns non-finite
        # (ports.diagnostics): npz files in the working directory, as before.
        self.diagnostics = NpzDiagnosticsSink()
        # Current working directory: SCMagnetCode (cdp, 10/2020)
        # self.cwd = os.getcwd()
        # Ask User the name of the cable. (cdp, 10/2020)
        # Default results root, used only when no target directory is given
        # to simulation_folders_manager (GUI flow); headless runs override it
        # with the model directory. Created on demand, never eagerly.
        # Working directory at construction (not at import) time.
        self.cwd = os.getcwd()
        self.dict_path = dict(
            Current_work_dir=self.cwd,
            Results_dir=os.path.join(self.cwd, "..", "Simulations_results"),
        )
        self.basePath = base_path
        # loop inside self.basePath (cdp, 10/2020)
        input_files = os.listdir(self.basePath)
        for f_name in input_files:
            if "transitory_input" in f_name:
                self.starter_file = f_name
        # Input source (ports.input_source): YAML when the directory holds
        # simulation.yaml, otherwise the legacy Excel workbooks.
        self.input_source = open_input_source(self.basePath)
        self.yaml_registry = yaml_input_registry.get_registry(self.basePath)
        if self.yaml_registry is not None:
            self.starter_file = yaml_input_registry.SIMULATION_FILE_NAME
        self.transient_input = self.input_source.transient_settings()
        self.flag_start = False
        # Optional lumped hydraulic network coupled to conductor channel
        # ends; built in conductor_initialization from the YAML input
        # (see hydraulics/network).
        self.hydraulic_network = None
        # get the order of maginitude of the minimum time step to make proper 
        # rounds to when saving data and figures of solution spatial 
        # distribution at default or User defined times.
        self.__count_sigfigs(self.transient_input["STPMIN"])
        self.list_of_Conductors = list()
        # Define the environment object as an attribute of the simulation
        self.environment = load_environment(
            os.path.join(self.basePath, self.transient_input["ENVIRONMENT"])
        )
        # Property aliases of the coolant (kept as attribute for callers;
        # the domain reads ports.fluid_properties.COOLANT_PROPERTY_ALIASES).
        self.fluid_prop_aliases = COOLANT_PROPERTY_ALIASES

    
    def run(self):
        """Run the simulation workflow."""
        self.simulation_summary = SimulationSummary()  # runtime timers start here
        # Everything observable (files, plots, log, checkpoints, console)
        # happens in observers (ports.observer); the model never writes.
        self.observers = CompositeObserver([
            OutputObserver(self.simulation_summary),
            RealTimePlotObserver(),
            FigureObserver(),
            ConsoleProgressObserver(),
        ])
        self.conductor_instance()  # read input files
        self.simulation_folders_manager(target_directory=self.basePath)  # create folders
        self.save_input_files()  # create metadata
        self.conductor_initialization()  # initialize conductors
        if self.transient_input["RESTART"]:
            # Resume from the newest autosaved checkpoint: the deterministic
            # initialization above rebuilt every object, now overwrite the
            # evolving state and truncate the outputs to the checkpoint time
            # (see utility_functions/checkpoint.py).
            restore_from_checkpoint(self)
            # Re-seed the summary extrema from the restored state; the
            # checkpoint-time hot-spot row is already in the truncated file.
            for cond in self.list_of_Conductors:
                self.simulation_summary.update(self, cond, record_hotspot=False)
        self.conductor_solution()  # solve the numerical problem
        self.conductor_post_processing()  # do post-processing
        self.observers.on_finished(self)  # simulation.log + hot-spot flush


    def __count_sigfigs(self,num:Union[int,float]):
        """Private method that counts the number of significant digits.
        N.B: quite general but does not work for all cases; see 
        https://stackoverflow.com/questions/8101353/counting-significant-figures-in-python

        Args:
            num Union[int,float]: number of which find the significant digits
        """

        numstr = str(num)
        decimal_norm = Decimal(numstr).normalize()
        if str(decimal_norm) == "0":
            # Deal with cases "0", "00", "00...0"
            self.n_digit_time = 0
        else:
            if 0.0 < num < 1.0:
                # Not the correct number of significant digits but rather the 
                # number of digits to be displayed.
                # E.g. 0.01 gives n_digit_time = 2 rather than n_digit_time = 1 
                # (which is the correct one).
                self.n_digit_time = abs(decimal_norm.as_tuple().exponent)
            else:
                # Deal with cases specified in https://stackoverflow.com/questions/8101353/counting-significant-figures-in-python
                self.n_digit_time = len(Decimal(numstr).as_tuple().digits)

    def conductor_instance(self):
        # Load spread-sheet conductor_definition.xlsx
        conductor_definition_path = os.path.join(self.basePath, self.transient_input["MAGNET"])
        self.numObj = self.input_source.conductor_count

        # LOAD MAIN CONDUCTORS PARAMETERS
        for ii in range(1, 1 + self.numObj):

            conductor = Conductor(
                load_conductor_definition(
                    self.basePath, ii, self.input_source, self.transient_input["MAGNET"]
                )
            )
            conductor.initialize_with_simulation(self)
            self.list_of_Conductors.append(conductor)
        # end for ii (cdp, 12/2020)
        self.contactBetweenConductors = self.input_source.conductor_coupling_dataframe()

        for conductor in self.list_of_Conductors:
            self.observers.on_conductor_created(self, conductor)

    # end method Conductor_instance

    def conductor_initialization(self):
        # Resolve the declared hydraulic formulation of every conductor
        # before any initialization work: the solution seeding inside
        # cond.initialization already depends on it. A conductor counts as
        # network-coupled when the (YAML-only) hydraulic_network section
        # declares a port on it; the network itself is built further below,
        # after the conductor loop.
        ported_conductor_identifiers = set()
        network_mapping = self.input_source.hydraulic_network()
        if network_mapping is not None:
            ported_conductor_identifiers = {
                port.get("conductor")
                for port in network_mapping.get("ports", [])
            }
        for cond in self.list_of_Conductors:
            coupling_enabled = cond.identifier in ported_conductor_identifiers
            cond.hydraulic_formulation = resolve_hydraulic_formulation(
                cond.inputs.hydraulic_formulation,
                coupling_enabled,
                cond.inputs.explicit_mass_flow_formulation,
            )
            for fluid_comp in cond.inventory.fluids.collection:
                fluid_comp.coolant.hydraulic_formulation = (
                    cond.hydraulic_formulation
                )
            declared = cond.inputs.hydraulic_formulation
            reason = (
                f"auto: network coupling {'declared' if coupling_enabled else 'absent'}"
                if declared is HydraulicFormulation.AUTO
                else "explicit input"
            )
            print(
                f"Conductor {cond.identifier}: hydraulic formulation "
                f"'{cond.hydraulic_formulation.value}' ({reason}).\n"
            )
        for cond in self.list_of_Conductors:
            # ** INITIALIZATION **
            # s time @ which simulation is started (cdp, 07/2020)
            self.simulation_time = [0.0]
            self.num_step = 0
            cond.initialization(self)
            self.observers.on_conductor_initialized(self, cond)
            # Use electric method only if needed, i.e., user specifies a 
            # current.
            if cond.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
                # Call to electric_electric method allows to define, 
                # initialize, solve and reorganize the electric problem.
                cond.eval_total_operating_current()
                cond.electric_method()
            else:
                # Quick fix to the following silent bug: solid components 
                # thermophysical, electromagnetic and critical properties 
                # do not update at each time step if flag 
                # conductor.inputs ["I0_OP_MODE"] == IOP_NOT_DEFINED. 
                # This will not cause the simulation to stop, but will give 
                # reasonable but wrong output. Since all solid component 
                # properties are updated in method operating_conditions_em of 
                # class Conductor, a quick fix is to explicitly call this 
                # method when flag 
                # conductor.inputs["I0_OP_MODE"] == IOP_NOT_DEFINED and let 
                # method electric_method of class Conductor call it in all the 
                # other cases.
                # A better fix involves a complete refactoring of at least 
                # methods operating_conditions_th and operating_conditions_em 
                # of class Conductor and should be done later.
                cond.operating_conditions_em()

            self.observers.on_initial_state(self, cond)
            # ** END INITIALIZATION **
        # end for cond (cdp, 12/202)
        # dictionary declaration (cdp,07/2020)
        self.dict_qsource = dict()
        if self.numObj == 1:
            # There is only 1 Conductor object, exploit cond instantiated above.
            # The number of columns is equal to the number of SolidComponent 
            # equations to exploit the new function build_svec in 
            # utility_functions/step_matrix_construction.py. Read the docstring 
            # for further details.
            self.dict_qsource[cond.identifier] = np.zeros(
                (
                    cond.mesh.number_of_nodes,
                    cond.equation_counts.solid_equations,
                )
            )
        else:
            # more than one Conductor object (cdp,07/2020)
            for rr in range(self.numObj):
                cond_r = self.list_of_Conductors[rr]
                if all(self.contactBetweenConductors.iloc[rr, :]) == 0:
                    # There is not contact between cond_r and all the others.
                    # Consider all the columns in order to not miss the info on 
                    # last raw (otherwise the last conductor will not be added 
                    # as key of the dictionary).
                    # The number of columns is equal to the number of 
                    # SolidComponent equations to exploit the new function 
                    # build_svec in 
                    # utility_functions/step_matrix_construction.py. Read the 
                    # docstring for further details.
                    self.dict_qsource[cond_r.identifier] = np.zeros(
                        (
                            cond_r.mesh.number_of_nodes,
                            cond_r.equation_counts.solid_equations,
                        )
                    )
                else:
                    for cc in range(rr + 1, self.numObj):
                        cond_c = self.list_of_Conductors[cc]
                        if self.contactBetweenConductors.iat[rr, cc] == 1:
                            # there is contact between cond_r and cond_c (cdp,07/2020)
                            # la colonna non nulla è solo quella del jacket esterno \
                            # eventualmente in contatto con un altro jacket esterno di un \
                            # altro conduttore. Nel caso il conduttore sia costituito da un \
                            # solo jacket la matrice degenera in un vettore colonna. \
                            # (cdp, 08/2020)
                            # N.B. controllare anche nella chiamata a step come passare self.dict_qsource.
                            raise ValueError(
                                "Multi conductors with thermal contact: not yet managed the situation, need to build dict_qsource properly!!"
                            )
                        else:
                            # There is not contact between cond_r and cond_c (cdp,07/2020)
                            # Proposta di soluzione ma va studiata decisamente meglio!
                            # N.B. controllare anche nella chiamata a step come passare self.dict_qsource.
                            # The number of columns is equal to the number of 
                            # SolidComponent equations to exploit the new 
                            # function build_svec in 
                            # utility_functions/step_matrix_construction.py. 
                            # Read the docstring for further details.
                            self.dict_qsource[
                                f"{cond_r.identifier}_{cond_c.identifier}"
                            ] = np.zeros(
                                (
                                    cond_r.mesh.number_of_nodes,
                                    cond_r.equation_counts.solid_equations,
                                )
                            )
                            raise ValueError(
                                "Multi conductors with thermal contact: not yet managed the situation, need to build dict_qsource properly!!"
                            )
                        # End if self.contactBetweenConductors[rr, cc].
                    # End for cc.
                # End if all(self.contactBetweenConductors[rr, rr + 1:])
                # End for rr.
        # end if numObj

        # Build, resolve and initialize the optional hydraulic network
        # coupled to conductor channel ends. Must run after the conductor
        # loop above: the port resolution reuses the boundary-condition
        # equation indices and the initial flow built there.
        network_mapping = self.input_source.hydraulic_network()
        if network_mapping is not None:
            self.hydraulic_network = build_coupled_network(
                self, network_mapping
            )
            self.observers.on_network_built(self)

    # end method Conductor_initialization

    def conductor_solution(self):
        # ** TRANSIENT SOLUTION **
        stoptime = 0  # flag to stop simulation if some problems (like quench) \
        # arise (cdp, 07/2020)
        # Time step initialization (cdp, 08/2020)
        # time_step = transient_input["STPMIN"]
        # Search if User asks to save the solution spatial distribution at TEND \
        # (cdp, 10/2020)
        # loop on conductors (cdp, 10/2020)
        for conductor in self.list_of_Conductors:
            # Compute radiative heat exchanged between jackets.
            conductor.compute_radiative_heat_exhange_jk()
            # Compute radiative heat exchanged outer jacket and environment.
            conductor.compute_heat_exchange_jk_env(self.environment)
            # get the times at which users saves the solution spatial distribution \
            # (cdp, 10/2020)
            # list_values = list(conductor.dict_Space_save.values())
            # Save of the solution spatial distribution at 0.0 s (cdp, 12/2020)
            self.observers.on_transient_start(self, conductor)
        # end for ii (cdp, 10/2020)
        # Any transverse insulation coupling declared on any solid forces
        # all conductors into lockstep time stepping (see the time-step
        # synchronization inside the loop).
        self.transverse_coupling_declared = any(
            getattr(solid.operations, "transverse_coupling_file", "")
            for conductor in self.list_of_Conductors
            for solid in conductor.inventory.solids.collection
        )
        # while loop to solve transient at each timestep (cdp, 07/2020)
        while (
            self.simulation_time[-1]
            < self.transient_input["TEND"] - 1e-5 * self.transient_input["STPMIN"]
            and stoptime == 0
        ):
            if self.step_callback is not None and self.step_callback(self):
                # The driver requested the end of the transient.
                stoptime = 1
                break
            self.num_step = self.num_step + 1
            time_step = np.zeros(self.numObj)
            # List of the conductors coupled to the hydraulic network: they
            # advance together with it (one shared linear solve per step).
            coupled_conductors = [
                conductor
                for conductor in self.list_of_Conductors
                if conductor.network_ports
            ]
            for conductor in self.list_of_Conductors:
                # Call function Get_time_step to select new time step (cdp, 08/2020)
                get_time_step(conductor, self.transient_input, self.num_step)
            # Conductors coupled to the hydraulic network share the network
            # unknowns and must advance with a common time step: force the
            # smallest proposed one on all of them. Conductors exchanging
            # heat through transverse coupling patches read each other's
            # state every step, so they must advance in lockstep as well
            # (otherwise a fast-stepping quenched conductor lags the
            # others in its own clock while the simulation clock follows
            # the largest step): with any transverse coupling declared,
            # every conductor of the simulation shares the smallest step.
            if self.transverse_coupling_declared:
                lockstep_conductors = list(self.list_of_Conductors)
            else:
                lockstep_conductors = coupled_conductors
            if lockstep_conductors:
                shared_time_step = min(
                    conductor.time_step for conductor in lockstep_conductors
                )
                for conductor in lockstep_conductors:
                    conductor.time_step = shared_time_step
            for ii, conductor in enumerate(self.list_of_Conductors):
                time_step[ii] = conductor.time_step
                # Increase time (cdp, 08/2020)
                conductor.cond_time.append(
                    conductor.cond_time[-1] + conductor.time_step
                )
                # update time step (cdp, 08/2020)
                conductor.cond_num_step = conductor.cond_num_step + 1
                # before calling Conductor method initialization adapt mesh if \
                # necessary as foreseen by ITYMSH. To do later (cdp, 07/2020)
                # se ho nuova griglia calcolare coefficenti, temperature, pressioni, \
                # parametri adimensionati sfruttando np.interp (in fortrand è adaptm)
            # End for conductor (cdp, 08/2020)
            # Evaluate simulation time and simulation time step (cdp, 08/2020)
            self.simulation_time_step = np.max(time_step)
            self.simulation_time.append(
                self.simulation_time[-1] + self.simulation_time_step
            )
            self.observers.on_step_started(self)
            for conductor in self.list_of_Conductors:
                
                # Use electric method only if needed, i.e., user specifies a 
                # current.
                if conductor.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
                    # Call to electric_electric method allows to define, 
                    # initialize, solve and reorganize the electric problem.
                    conductor.electric_method()
                else:
                    # Quick fix to the following silent bug: solid components 
                    # thermophysical, electromagnetic and critical properties 
                    # do not update at each time step if flag 
                    # conductor.inputs ["I0_OP_MODE"] == IOP_NOT_DEFINED. 
                    # This will not cause the simulation to stop, but will give 
                    # reasonable but wrong output. Since all solid component 
                    # properties are updated in method operating_conditions_em 
                    # of class Conductor, a quick fix is to explicitly call 
                    # this method when flag 
                    # conductor.inputs["I0_OP_MODE"] == IOP_NOT_DEFINED and let 
                    # method electric_method of class Conductor call it in all 
                    # the other cases.
                    # A better fix involves a complete refactoring of at least 
                    # methods operating_conditions_th and 
                    # operating_conditions_em of class Conductor and should be 
                    # done later.
                    conductor.operating_conditions_em()
                    
                # Evaluate thermal hydraulic properties and quantities in Gauss 
                # points, method __eval_Gauss_point_th is invoked inside method 
                # operating_conditions_th. Method operating_conditions_th is 
                # called at each time step before function step because the 
                # method for the integration in time is implicit.
                conductor.operating_conditions_th(self)
                
                conductor.build_heat_source(self)
                # call step to solve the problem @ new timestep; conductors
                # coupled to the hydraulic network are solved jointly with
                # the network below, once every coupled conductor has been
                # prepared (cdp, 07/2020)
                if not conductor.network_ports:
                    step(
                        conductor,
                        self.environment,
                        self.dict_qsource[conductor.identifier],
                        self.num_step,
                        self.diagnostics,
                    )
            # End for conductor: preparation and uncoupled solves.

            if coupled_conductors:
                # One monolithic solve of all network-coupled conductors
                # and the network; advances the network state exactly once.
                solve_coupled_conductors_step(
                    coupled_conductors,
                    self.hydraulic_network,
                    self.dict_qsource,
                    self.diagnostics,
                )

            for conductor in self.list_of_Conductors:
                # Loop on FluidComponent (cdp, 10/2020)
                for fluid_comp in conductor.inventory.fluids.collection:
                    # compute density and mass flow rate in nodal points with the
                    # updated FluidComponent temperature and velocity (nodal = True by default)
                    if conductor.hydraulic_formulation in MASS_FLOW_FORMULATIONS:
                        # Mass-flow formulation: the mass flow rate is the
                        # native unknown; density from the new (p, T) and
                        # the definitive velocity v = mdot / (rho A).
                        fluid_comp.coolant._refresh_density_and_velocity_from_mass_flow()
                    else:
                        fluid_comp.coolant._compute_density_and_mass_flow_rates_nodal_gauss(
                            conductor
                        )
                    # Enthalpy balance: sum((mdot*w)_out - (mdot*w)_inl), used to check \
                    # the imposition of SolidComponent temperature initial spatial \
                    # distribution (cdp, 12/2020)
                    conductor.enthalpy_balance = (
                        conductor.enthalpy_balance
                        + conductor.time_step
                        * (
                            fluid_comp.coolant.node_fields.mass_flow_rate[-1]
                            * fluid_comp.coolant.node_fields.total_enthalpy[-1]
                            - fluid_comp.coolant.node_fields.mass_flow_rate[0]
                            * fluid_comp.coolant.node_fields.total_enthalpy[0]
                        )
                    )
                    conductor.enthalpy_out = (
                        conductor.enthalpy_out
                        + conductor.time_step
                        * fluid_comp.coolant.node_fields.mass_flow_rate[-1]
                        * fluid_comp.coolant.node_fields.total_enthalpy[-1]
                    )
                    conductor.enthalpy_inl = (
                        conductor.enthalpy_inl
                        + conductor.time_step
                        * fluid_comp.coolant.node_fields.mass_flow_rate[0]
                        * fluid_comp.coolant.node_fields.total_enthalpy[0]
                    )
                # end for fluid_comp (cdp, 10/2020)

                # Compute radiative heat exchanged between jackets.
                conductor.compute_radiative_heat_exhange_jk()
                # Compute radiative heat exchanged outer jacket and environment.
                conductor.compute_heat_exchange_jk_env(self.environment)

                self.observers.after_conductor_step(self, conductor)
                # call sensor to plot results at any time the user asks (cdp, 07/2020)
            # End for conductor (cdp, 07/2020)

            autosave_interval = self.transient_input["AUTOSAVE_INTERVAL"]
            if (
                autosave_interval is not None
                and self.num_step % autosave_interval == 0
            ):
                self.observers.on_autosave(self)
        # end while (cdp, 07/2020)
        self.observers.on_transient_end(self)

    # end method Conductor_solution (cdp, 09/2020)

    def conductor_post_processing(self):
        for cond in self.list_of_Conductors:
            # Final energies and the mass/energy balances (domain).
            cond.post_processing(self)
            # Final solution, convergence data, spatial distribution files
            # and figures (observers).
            self.observers.on_conductor_post_processed(self, cond)
        self.observers.on_post_processing_done(self)

    # end method Conductor_post_processing (cdp, 09/2020)

    def simulation_folders_manager(self, target_directory: str = None):
        """Build the whole output-folder tree for this simulation run."""
        simulation_paths.manage_simulation_folders(self, target_directory)

    def save_input_files(self):
        """Save the input file of the simulation as .xlsx files in read only mode. These files are metadata for the simulation output."""
        simulation_paths.save_input_files(self)
