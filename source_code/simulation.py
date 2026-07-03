from decimal import Decimal
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import os
from typing import Union

from conductor.conductor import Conductor
from conductor.input_loader import ConductorInputLoader
from conductor.input_validator import ConductorInputValidator
from electromagnetics.electromagnetic_flags import CurrentMode
from environment import Environment
from utility_functions.auxiliary_functions import (
    with_read_csv,
    with_read_excel,
)
from utility_functions.transient_solution_functions import get_time_step, step
import utility_functions.simulation_paths as simulation_paths
from utility_functions.output import (
    save_simulation_space,
    reorganize_spatial_distribution,
    save_simulation_time,
    save_properties,
)
from utility_functions.plots import (
    plot_properties,
    make_plots,
    create_real_time_plots,
    update_real_time_plots,
)


class Simulation:

    # Current working directory
    CWD = os.getcwd()

    def __init__(self, base_path):

        # Current working directory: SCMagnetCode (cdp, 10/2020)
        # self.cwd = os.getcwd()
        # Ask User the name of the cable. (cdp, 10/2020)
        self.dict_path = dict(
            Current_work_dir=self.CWD,
            Results_dir=os.path.join(self.CWD, "..", "Simulations_results"),
        )
        # Create directory Simulations_results if it does not exist yet
        os.makedirs(self.dict_path["Results_dir"], exist_ok=True)
        self.basePath = base_path
        # loop inside self.basePath (cdp, 10/2020)
        input_files = os.listdir(self.basePath)
        for f_name in input_files:
            if "transitory_input" in f_name:
                self.starter_file = f_name
        # Load input file transitory_input.xlsx and convert to a dictionary.
        self.transient_input = pd.read_excel(
            os.path.join(self.basePath, self.starter_file),
            sheet_name="TRANSIENT",
            skiprows=1,
            header=0,
            index_col=0,
            usecols=["Variable name", "Value"],
        )["Value"].to_dict()
        self.flag_start = False
        # get the order of maginitude of the minimum time step to make proper 
        # rounds to when saving data and figures of solution spatial 
        # distribution at default or User defined times.
        self.__count_sigfigs(self.transient_input["STPMIN"])
        self.list_of_Conductors = list()
        # Define the environment object as an attribute of the simulation
        self.environment = Environment(
            os.path.join(self.basePath, self.transient_input["ENVIRONMENT"])
        )
        # Define dictionary with default values for functions with_read_csv and with_read_excel.
        self.default_vals = dict(dat="\t", csv=";", tsv="\t", xlsx=0)
        # Define dictionaty with the pointers to the function to be called according to the file extension if some quantities should be loaded from an auxiliary input file.
        self.func_open_aux = dict(
            dat=with_read_csv,
            csv=with_read_csv,
            tsv=with_read_csv,
            xlsx=with_read_excel,
        )

        self.fluid_prop_aliases = dict(
            isobaric_expansion_coefficient="isobaric_expansion_coefficient",
            isothermal_compressibility="isothermal_compressibility",  # 1/Pa
            Prandtl="Prandtl",
            total_density="Dmass",
            total_dynamic_viscosity="viscosity",
            total_enthalpy="Hmass",
            total_isobaric_specific_heat="Cpmass",
            total_isochoric_specific_heat="Cvmass",
            total_speed_of_sound="speed_of_sound",
            total_thermal_conductivity="conductivity",
        )

    # end method __init__ (cdp, 06/2020)

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
        list_conductor_sheet = ConductorInputLoader.load_conductor_definition_sheets(
            conductor_definition_path)
        
        self.numObj = int(list_conductor_sheet[0].cell(row=1, column=2).value)

        # LOAD MAIN CONDUCTORS PARAMETERS
        for ii in range(1, 1 + self.numObj):

            conductor = Conductor(
                self.basePath,
                ii,
                self.transient_input["MAGNET"]
            )
            conductor.initialize_with_simulation(self)
            self.list_of_Conductors.append(conductor)
        # end for ii (cdp, 12/2020)
        self.contactBetweenConductors = pd.read_excel(
            conductor_definition_path,
            sheet_name="CONDUCTOR_coupling",
            header=0,
            index_col=0,
        )

        # Loop to create the attributes required to make the real time plots (shortly rtp).
        for conductor in self.list_of_Conductors:
            create_real_time_plots(self, conductor)
        # End for conductor.

    # end method Conductor_instance

    def conductor_initialization(self):
        for cond in self.list_of_Conductors:
            # ** INITIALIZATION **
            # s time @ which simulation is started (cdp, 07/2020)
            self.simulation_time = [0.0]
            self.num_step = 0
            cond.initialization(self)
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

            # plot conductor initialization spatial distribution (cdp, 12/2020)
            plot_properties(self, cond)
            save_simulation_time(self, cond)
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

    # end method Conductor_initialization

    def conductor_solution(self):
        # ** TRANSIENT SOLUTION **
        num_step_store = 100
        count_store = 1
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
            save_simulation_space(
                conductor,
                self.dict_path[
                    f"Output_Spatial_distribution_{conductor.identifier}_dir"
                ],
                abs(self.n_digit_time),
            )
        # end for ii (cdp, 10/2020)
        # while loop to solve transient at each timestep (cdp, 07/2020)
        while (
            self.simulation_time[-1]
            < self.transient_input["TEND"] - 1e-5 * self.transient_input["STPMIN"]
            and stoptime == 0
        ):
            self.num_step = self.num_step + 1
            time_step = np.zeros(self.numObj)
            for ii, conductor in enumerate(self.list_of_Conductors):
                # Call function Get_time_step to select new time step (cdp, 08/2020)
                get_time_step(conductor, self.transient_input, self.num_step)
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
            print(
                f"Simulation time: {self.simulation_time[-1]:.{self.n_digit_time}f} s; {self.simulation_time[-1]/self.transient_input['TEND']*100:5.2f} %"
            )
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
                # call step to solve the problem @ new timestep (cdp, 07/2020)
                step(
                    conductor,
                    self.environment,
                    self.dict_qsource[conductor.identifier],
                    self.num_step,
                )
                # Loop on FluidComponent (cdp, 10/2020)
                for fluid_comp in conductor.inventory.fluids.collection:
                    # compute density and mass flow rate in nodal points with the
                    # updated FluidComponent temperature and velocity (nodal = True by default)
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

                update_real_time_plots(conductor)

                if self.num_step == num_step_store * count_store:
                    # Update counter to store the state of the simulation, still to come \
                    # (cdp, 08/2020)
                    count_store = count_store + 1
                if (
                    conductor.i_save < len(conductor.Space_save) - 1
                    and abs(
                        conductor.cond_time[-1] - conductor.Space_save[conductor.i_save]
                    )
                    < conductor.time_step / 2.0
                ):
                    # save simulation spatial distribution at user defined time steps \
                    # (cdp, 08/2020)
                    save_simulation_space(
                        conductor,
                        self.dict_path[
                            f"Output_Spatial_distribution_{conductor.identifier}_dir"
                        ],
                        abs(self.n_digit_time),
                    )
                # end if isave
                # Save variables time evolution at given spatial coordinates \
                # (cdp, 08/2020)
                save_simulation_time(self, conductor)
                # call sensor to plot results at any time the user asks (cdp, 07/2020)
            # End for conductor (cdp, 07/2020)
        # end while (cdp, 07/2020)
        print("End simulation called " + self.transient_input["SIMULATION"] + "\n")

    # end method Conductor_solution (cdp, 09/2020)

    def conductor_post_processing(self):
        # loop to save the norm of the solution at the end of the transient for 
        # each conductor, usefull to make space convergence
        # t_end = np.array([self.transient_input["TEND"]])
        for cond in self.list_of_Conductors:
            cond.post_processing(self)

            # save simulation spatial distribution at TEND.
            save_simulation_space(
                cond,
                self.dict_path[f"Output_Spatial_distribution_{cond.identifier}_dir"],
                abs(self.n_digit_time),
            )
            # Call function Save_properties to save the conductor final 
            # solution.
            save_properties(
                cond, self.dict_path[f"Output_Solution_{cond.identifier}_dir"]
            )
            print("Saved final solution\n")

            reorganize_spatial_distribution(
                cond,
                self.dict_path[f"Output_Spatial_distribution_{cond.identifier}_dir"],
                self.n_digit_time,
            )
            # Plot conductor solution spatial distribution (cdp, 12/2020)
            plot_properties(self, cond, what="solution")
        # end for cond (cdp, 12/2020)
        # Call function Make_plots to make plot of spatial distribution and time \
        # evolution (cdp, 11/2020)
        make_plots(self, kind="Space_distr")
        make_plots(self, kind="Time_evol")

    # end method Conductor_post_processing (cdp, 09/2020)

    def simulation_folders_manager(self, target_directory: str = None):
        """Build the whole output-folder tree for this simulation run."""
        simulation_paths.manage_simulation_folders(self, target_directory)

    def save_input_files(self):
        """Save the input file of the simulation as .xlsx files in read only mode. These files are metadata for the simulation output."""
        simulation_paths.save_input_files(self)
