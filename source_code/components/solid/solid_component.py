import numpy as np
import warnings

from utility_functions.auxiliary_functions import (
    load_auxiliary_files,
    build_interpolator,
    do_interpolation,
)

from components.component_flags import ComponentType
from components.solid.solid_component_inputs import (
    SolidComponentInputs,
    SolidComponentOperations
)
from electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode
from thermal.thermal_flags import HeatExcitation
from conductor.conductor_flags import MethodFlag, ONE_STEP_METHODS

_STRAND_MIXED_NAME = ComponentType.STRAND_MIXED.value          # "STR_MIX"
_STRAND_STABILIZER_NAME = ComponentType.STRAND_STABILIZER.value  # "STR_STAB"
_STACK_NAME = ComponentType.STACK.value                          # "STACK"
_JACKET_NAME = ComponentType.JACKET.value                        # "Z_JACKET"

class SolidComponent:
    def __init__(self, inputs: SolidComponentInputs,
                 operations: SolidComponentOperations):
        """
        Constructor method of class SolidComponent (cdp, 11/2020)
        """
        self.inputs = inputs
        self.operations = operations

    # end method __init__ (cdp, 11/2020)

    def initialize_heat_flux_schedule(self, simulation):
        """Initialize the imposed-heat-flux on/off time-step schedule from the
        component's operations, if a square-wave heat excitation is defined.
        """
        # Questa è una bozza, quando e se si dovranno considerare altri flag come \
        # IBFUN o IALPHAB, valutare se è il caso di sfruttare un metodo per \
        # evitare di scrivere lo stesso codice più volte (cdp, 11/2020)
        if self.operations.heat_flux_mode == HeatExcitation.SQUARE_WAVE_IN_TIME_AND_SPACE:
            # External heating parameters given in file operation.xlsx \
            # (cdp, 11/2020)
            # The heating will be on at some times (cdp, 01/2021)
            self.flag_heating = "On"
            if simulation.transient_input["IADAPTIME"] == 0:
                # Time adaptivity off and (cdp, 11/2020)
                self.dict_num_step["IQFUN"] = dict(
                    ON=int(
                        self.operations.heat_flux_time_start
                        / simulation.transient_input["STPMIN"]
                    ),
                    OFF=int(
                        self.operations.heat_flux_time_end
                        / simulation.transient_input["STPMIN"]
                    ),
                )
            else:
                # Time adaptivity on
                print("Still to be decided what to do there (cdp, 11/2020)\n")
        # End self.operations.heat_flux_mode.

    # end method initialize_heat_flux_schedule

    def eval_sol_comp_properties(self, inventory, nodal=True):

        """
        Method that evaluate total_density, specific_heat and thermal conductivity of SolidComponent class objects in both nodal points and Gauss points according to **options input parameter (cdp, 07/2020)
        """

        # Properties evaluation in each nodal point (cdp, 07/2020)
        if nodal:
            dict_dummy = self.node_fields
            self.node_fields = self.eval_properties(dict_dummy, inventory)
        # Properties evaluation in each Gauss point (cdp, 07/2020)
        elif nodal == False:
            dict_dummy = self.gauss_fields
            self.gauss_fields = self.eval_properties(dict_dummy, inventory)

    def eval_properties(self, dict_dummy: dict, inventory: dict) -> dict:
        """Method that actually evaluate total_density, specific_heat and thermal conductivity of SolidComponent class objects regardless of the location (nodal or Gauss points).

        Args:
            dict_dummy (dict): dictionary with material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.

        Returns:
            dict: dictionary with updated material properties in nodal points or Gauss points according to the value of flag nodal in method eval_sol_comp_properties of class SolidComponent.
        """

        if self.name in (_STRAND_MIXED_NAME, _STRAND_STABILIZER_NAME):
            dict_dummy.update(total_density=self.strand_density(dict_dummy))
            dict_dummy.update(
                total_isobaric_specific_heat=self.strand_isobaric_specific_heat(
                    dict_dummy
                )
            )
            dict_dummy.update(
                total_thermal_conductivity=self.strand_thermal_conductivity(dict_dummy)
            )
            if self.name == _STRAND_MIXED_NAME:
                dict_dummy.update(
                    electrical_resistivity_stabilizer=self.strand_electrical_resistivity_not_sc(
                        dict_dummy
                    )
                )
            elif self.name == _STRAND_STABILIZER_NAME:
                dict_dummy.update(
                    electrical_resistivity_stabilizer=self.strand_electrical_resistivity(
                        dict_dummy
                    )
                )
        elif self.name == _STACK_NAME:
            dict_dummy.update(total_density=self.stack_density(dict_dummy))
            dict_dummy.update(
                total_isobaric_specific_heat=self.stack_isobaric_specific_heat(
                    dict_dummy
                )
            )
            dict_dummy.update(
                total_thermal_conductivity=self.stack_thermal_conductivity(dict_dummy)
            )
            dict_dummy.update(
                electrical_resistivity_stabilizer=self.stack_electrical_resistivity_not_sc(
                    dict_dummy
                )
            )
        else:  # JacketComponent
            dict_dummy.update(total_density=self.jacket_density(dict_dummy))
            dict_dummy.update(
                total_isobaric_specific_heat=self.jacket_isobaric_specific_heat(
                    dict_dummy
                )
            )
            dict_dummy.update(
                total_thermal_conductivity=self.jacket_thermal_conductivity(dict_dummy)
            )
            dict_dummy.update(
                total_electrical_resistivity=self.jacket_electrical_resistivity(
                    dict_dummy
                )
            )

        return dict_dummy

    def get_current_fractions(
        self, total_sc_area: float, total_so_area: float, inventory: dict
    ):
        """Method that evaluates: 1) fraction of the total current that flows in superconductor cross section of each strand or stack object if in superconducting regime (op_current_fraction_sc); 2) fraction of the total current that flows in the total cross section (superconductor and not superconductor or stabilizer materials) of each strand or stack object if in current sharing regime (op_current_fraction_sh). Both are methods of the generic object.

        Note: for the time being both fractions are set to 0.0 for objects of kind JacketComponent.

        Args:
            total_sc_area (float): total superconductor cross section of the conductor.
            total_so_area (float): total cross section of strands and or stacks of the conductor.
            inventory (dict): dictionary with all the conductor components.
        """

        # Fraction of the total current that goes in the superconductor cross
        # section in superconducting regime.
        if self.name in (_STRAND_MIXED_NAME, _STACK_NAME):
            self.op_current_fraction_sc = self.cross_section["sc"] / total_sc_area
        elif self.name in (_JACKET_NAME, _STRAND_STABILIZER_NAME):
            self.op_current_fraction_sc = 0.0

        # Fraction of the total current that goes in the strand or tape cross
        # section in current sharing regime.
        if self.name in (_STACK_NAME, _STRAND_MIXED_NAME, _STRAND_STABILIZER_NAME):
            self.op_current_fraction_sh = self.inputs.cross_section / total_so_area
        else:  # jacket object.
            self.op_current_fraction_sh = 0.0

    def __check_current_mode(self, conductor: object):
        """Private method that checks consistency between flags conductor.inputs['I0_OP_MODE'] and self.operations['IOP_MODE'] to deal with current definition.

        Args:
            conductor (object): ConductorComponent object with all informations to make the check.

        Raises:
            ValueError: self.operations["IOP_MODE"] != None and conductor.inputs["I0_OP_MODE"] == -1 and self.operations["IOP_MODE"] != -1.
            ValueError: self.operations["IOP_MODE"] != None and conductor.inputs["I0_OP_MODE"] == 0 and self.operations["IOP_MODE"] != 0.
        """

        # Initialize dictionary with error message to be printed.
        message_switch = {
            CurrentMode.CURRENT_IS_FROM_FILE: f"{conductor.inputs.current_mode=} implies that current carried by object {self.identifier = } should be read from file. Flag self.operations.operating_current_mode should be set to -1; current value is {self.operations.operating_current_mode=}. Please check sheet {self.identifier} of input file conductor_operation.xlsx.\n",
            CurrentMode.CURRENT_IS_CONSTANT: f"{conductor.inputs.current_mode=} implies that current carried by object {self.identifier = } is evaluated from the code since the total current carried by the conductor is assigned. Flag self.operations.operating_current_mode should be set to 0; current value is {self.operations.operating_current_mode=}. Please check sheet {self.identifier} of input file conductor_operation.xlsx.\n",
        }

        # Check consistency between flags conductor.inputs['I0_OP_MODE'] and
        # self.operations.operating_current_mode.
        if self.operations.operating_current_mode != None:
            if (
                conductor.inputs.current_mode == CurrentMode.CURRENT_IS_FROM_FILE
                and self.operations.operating_current_mode != CurrentMode.CURRENT_IS_FROM_FILE
            ):
                raise ValueError(message_switch[conductor.inputs.current_mode])
            elif (
                conductor.inputs.current_mode == CurrentMode.CURRENT_IS_CONSTANT
                and self.operations.operating_current_mode != CurrentMode.CURRENT_IS_CONSTANT
            ):
                raise ValueError(message_switch[conductor.inputs.current_mode])


    def get_current(self, conductor: object):
        """Method that evaluates the current source therm at each thermal time step according to flag I0_OP_MODE set as input by the user.

        Args:
            conductor (object): ConductorComponent object with all informations to make the calculation.

        Raises:
            ValueError: if I0_OP_MODE is -2 (still to be implemented).
            ValueError: if a not valid value is given to flag I0_OP_MODE.
        """

        # Check consistency between flags conductor.inputs['I0_OP_MODE']
        # and self.operations['IOP_MODE'] only the first time.
        if conductor.cond_time[-1] == 0:
            self.__check_current_mode(conductor)

        # Get current.
        if self.operations.operating_current_mode != None:
            # The object carryes a current and its value is defied as below.
            if conductor.inputs.current_mode == CurrentMode.CURRENT_IS_FROM_FILE:

                if conductor.cond_time[-1] == 0:
                    # Build file path.
                    file_path = conductor.file_paths.external_current
                    # Load auxiliary input file.
                    current_df, self.flagSpecfield_current = load_auxiliary_files(
                        file_path, sheetname=self.identifier
                    )
                    # Build interpolator and get the interpolaion flag (space_only,time_only or space_and_time).
                    (
                        self.current_interpolator,
                        self.current_interp_flag,
                    ) = build_interpolator(
                        current_df, self.operations.operating_current_interpolation
                    )

                # Evaluate current of generic solid component object by
                # interpolation.
                self.node_fields.op_current = do_interpolation(
                    self.current_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.electric_time,
                    self.current_interp_flag,
                )

                if self.current_interp_flag == "time_only":
                    # Convert to array
                    self.node_fields.op_current = self.node_fields.op_current * np.ones(conductor.mesh.number_of_nodes)
                # Evaluate current in the Gauss nodal points.
                self.gauss_fields.op_current = (
                    self.node_fields.op_current[:-1]
                    + self.node_fields.op_current[1:]
                ) / 2.0
                # This is exploited in the electric resistance evaluation.
                if (
                    self.name in (_STACK_NAME, _STRAND_MIXED_NAME)
                ):
                    # Build an alias for convenience when dealing with electric
                    # resistance evaluation.
                    self.node_fields.op_current_sc = self.node_fields.op_current
                    self.gauss_fields.op_current_sc = self.gauss_fields.op_current

                if self.flagSpecfield_current == 2:
                    # Add also a logger
                    warnings.warn("Still to be decided what to do here\n")
            elif conductor.inputs.current_mode == CurrentMode.CURRENT_IS_CONSTANT:
                # Evaluate both attributes self.node_fields.op_current and
                # self.node_fields.op_current_sc for convenience in the evaluation of
                # electrical resistivity.
                self.node_fields.op_current = (
                    conductor.inputs.initial_current
                    * self.op_current_fraction_sh
                    * np.ones(conductor.mesh.number_of_nodes)
                )
                self.gauss_fields.op_current = (
                    self.node_fields.op_current[:-1]
                    + self.node_fields.op_current[1:]
                ) / 2.0
                if (
                    self.name in (_STACK_NAME, _STRAND_MIXED_NAME)
                ):
                    self.node_fields.op_current_sc = (
                        conductor.inputs.initial_current
                        * self.op_current_fraction_sc
                        * np.ones(conductor.mesh.number_of_nodes)
                    )
                    self.gauss_fields.op_current_sc = (
                        self.node_fields.op_current_sc[:-1]
                        + self.node_fields.op_current_sc[1:]
                    ) / 2.0

            elif conductor.inputs.current_mode == CurrentMode.CURRENT_NOT_DEFINED:
                # User does not specify a current: set current carrient
                # operating current to zero only the first time in both nodal
                # and gauss points.
                if conductor.cond_num_step == 0:
                    self.node_fields.op_current = np.zeros(conductor.mesh.number_of_nodes)
                    self.gauss_fields.op_current = np.zeros(conductor.mesh.number_of_elements)

            elif conductor.inputs.current_mode == CurrentMode.CURRENT_IS_FUNCTION:
                raise ValueError(
                    f"Current from external function to be implemented!"
                )
            else:
                raise ValueError(
                    f"Not defined value for flag I0_OP_MODE: {conductor.inputs.current_mode=}.\n"
                )
        else:
            # The object does not carry a current; arrays are initialized to 0.
            # Initialize array op_current to 0 in dictionary node_fields to
            # avoid error.
            self.node_fields.op_current = np.zeros(conductor.mesh.number_of_nodes)
            # Initialize array op_current to 0 in dictionary gauss_fields to
            # avoid error.
            self.gauss_fields.op_current = np.zeros(conductor.mesh.number_of_elements)
            # This is exploited in the electric resistance evaluation.
            if (
                self.name in (_STACK_NAME, _STRAND_MIXED_NAME)
            ):
                # Build an alias for convenience when dealing with electric
                # resistance evaluation.
                self.node_fields.op_current_sc = self.node_fields.op_current
                self.gauss_fields.op_current_sc = self.gauss_fields.op_current

    # end Get_I

    def _conductor_current_ratio(self, conductor) -> float:
        """Ratio I(t)/I0 of the conductor transport current to its initial value.

        Used by the proportional magnetic-field model
        (``BFieldDefinitionType.LINEAR_WITH_TRANSIENT``). The strand
        operating currents at the current electric time are already loaded
        by ``get_current``, which runs before ``get_magnetic_field`` in
        ``update_em_operating_conditions``.
        """
        if (
            conductor.inputs.current_mode == CurrentMode.CURRENT_IS_CONSTANT
            or conductor.inputs.initial_current == 0.0
        ):
            return 1.0

        if conductor.inputs.current_mode == CurrentMode.CURRENT_IS_FUNCTION:
            from electromagnetics.operating_conditions import (
                user_defined_current,
            )

            return (
                user_defined_current(conductor.electric_time)
                / conductor.inputs.initial_current
            )

        total_current = sum(
            strand.node_fields.op_current[0]
            for strand in conductor.inventory.strands.collection
        )
        return total_current / conductor.inputs.initial_current

    def get_magnetic_field(self, conductor, nodal=True):
        if nodal:
            # compute B_field in each node (cdp, 07/2020)
            if self.operations.magnetic_field_bc_mode is BFieldDefinitionType.FROM_FILE:
                # cza to enable other negative \
                # (read from file) flags -> ibifun.eq.-3, see below (August 29, 2018)

                if conductor.cond_time[-1] == 0:
                    # Build file path.
                    file_path = conductor.file_paths.external_bfield
                    # Load auxiliary input file.
                    bfield_df, _ = load_auxiliary_files(
                        file_path, sheetname=self.identifier
                    )
                    # Build interpolator and get the interpolaion flag (space_only,time_only or space_and_time).
                    (
                        self.bfield_interpolator,
                        self.bfield_interp_flag,
                    ) = build_interpolator(
                        bfield_df, self.operations.magnetic_field_interpolation
                    )

                # call load_user_defined_quantity on the component.
                self.node_fields.B_field = do_interpolation(
                    self.bfield_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.electric_time,
                    self.bfield_interp_flag,
                )
                if self.operations.magnetic_field_units == "T/A":
                    # BFIELD is per unit of current
                    self.node_fields.B_field = (
                        self.node_fields.B_field * conductor.inputs.initial_current
                    )
                if (
                    conductor.inputs.current_mode != CurrentMode.CURRENT_IS_CONSTANT
                    and conductor.inputs.initial_current > 0
                ):
                    #### bfield e' un self e' un vettore
                    self.node_fields.B_field = (
                        self.node_fields.B_field
                        * conductor.inputs.initial_current
                        / conductor.inputs.initial_current
                    )
            elif self.operations.magnetic_field_bc_mode is BFieldDefinitionType.CONSTANT_OR_LINEAR:
                self.node_fields.B_field = np.linspace(
                    self.operations.magnetic_field_inlet_initial,
                    self.operations.magnetic_field_outlet_initial,
                    conductor.mesh.number_of_nodes,
                )
            elif self.operations.magnetic_field_bc_mode is BFieldDefinitionType.LINEAR_WITH_TRANSIENT:
                # Proportional field model (THEA MagneticFieldModel
                # "proportional"): the transient part scales with the actual
                # conductor transport current, so the field collapses
                # together with the current during a dump.
                self.node_fields.B_field = np.linspace(
                    self.operations.magnetic_field_inlet_initial,
                    self.operations.magnetic_field_outlet_initial,
                    conductor.mesh.number_of_nodes,
                ) + self._conductor_current_ratio(conductor) * np.linspace(
                    self.operations.magnetic_field_inlet_transient,
                    self.operations.magnetic_field_outlet_transient,
                    conductor.mesh.number_of_nodes,
                )
        elif nodal == False:
            # compute B_field in each Gauss point (cdp, 07/2020)
            self.gauss_fields.B_field = (
                np.abs(
                    self.node_fields.B_field[:-1] + self.node_fields.B_field[1:]
                )
                / 2.0
            )

    # end Get_B_field

    # HERE STARTS THE DEFINITION OF MODULES USEFUL TO INITIALIZE THE DRIVERS FOR \
    # THE EXTERNAL HEATING. D. Placido (06/2020)

    def get_heat(self, conductor):

        """
        Method that evaluates the external heating according to the value of flag IQUFN, thaing unto account the chosen solution method (cdp, 11/2020)
        """

        # START INITIALIZATION (cdp, 10/2020)
        if conductor.cond_num_step == 0:
            if self.operations.heat_flux_mode == HeatExcitation.NO_HEATING:
                # Initialization is done always in the same way ragardless of the \
                # solution method: a column vector to exploit the array smart notation \
                # in Conductor class method Eval_Gauss_point. It is the only times at \
                # which this method is invoked (cdp, 11/2020)
                self.node_fields.EXTFLX = np.zeros(
                    (conductor.mesh.number_of_nodes, 1)
                )
            else:
                # Initialization is done always in the same way ragardless of the \
                # value of the flag IQFUN and coherently with the chosen solution \
                # algorithm (cdp, 10/2020)
                if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
                    # Theta family (backward Euler, Crank-Nicolson, Galerkin)
                    # or BDF2: two-time-level array layout (cdp, 10/2020)
                    self.node_fields.EXTFLX = np.zeros(
                        (conductor.mesh.number_of_nodes, 2)
                    )
                elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
                    # Adams-Moulton 4 (cdp, 10/2020)
                    self.node_fields.EXTFLX = np.zeros(
                        (conductor.mesh.number_of_nodes, 4)
                    )
            # end if self.operations.heat_flux_mode (cdp, 11/2020)
        # end if conductor.cond_num_step (cdp, 10/2020)
        # END INITIALIZATION (cdp, 10/2020)
        # criteria to decide how to evaluate the external heating (cdp, 10/2020)
        if self.operations.heat_flux_mode == HeatExcitation.SQUARE_WAVE_IN_TIME_AND_SPACE:
            # invoke method Q0_where to evaluate the external heating according to \
            # the function corresponding to the value of flag IQFUN. It is not \
            # necessary to distinguish according to "Method" options at this point \
            # (cdp, 10/2020)
            if (
                conductor.cond_time[-1] >= self.operations.heat_flux_time_start
                and conductor.cond_time[-1] <= self.operations.heat_flux_time_end
            ):
                self.heat0_where(conductor)
            elif (
                conductor.cond_time[-1] > self.operations.heat_flux_time_end
                and self.flag_heating == "On"
            ):
                self.node_fields.EXTFLX[:, 0] = 0.0
                self.flag_heating = "Off"
            # end if (cdp, 10/2020)
        elif self.operations.heat_flux_mode == HeatExcitation.FROM_FILE:
            if conductor.cond_time[-1] == 0:
                # Build file path.
                file_path = conductor.file_paths.external_heat
                # Load auxiliary input file.
                heat_df, _ = load_auxiliary_files(file_path, sheetname=self.identifier)
                # Build interpolator and get the interpolaion flag (space_only,time_only or space_and_time).
                self.heat_interpolator, self.heat_interp_flag = build_interpolator(
                    heat_df, self.operations.heat_flux_interpolation
                )

                # compute external heating at conductor initialization calling function do_interpolation.

                self.node_fields.EXTFLX[:, 0] = do_interpolation(
                    self.heat_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.cond_time[-1],
                    self.heat_interp_flag,
                )
            elif conductor.cond_num_step > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, since after that the whole system load vector is saved and there is no need to compute twice the same values.
                    self.node_fields.EXTFLX[:, 1] = self.node_fields.EXTFLX[
                        :, 0
                    ].copy()
                # end if conductor.cond_num_step (cdp, 10/2020)
                # call method load_user_defined_quantity to compute heat and overwrite the previous values.
                self.node_fields.EXTFLX[:, 0] = do_interpolation(
                    self.heat_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.cond_time[-1],
                    self.heat_interp_flag,
                )
            # end if conductor.cond_num_step (cdp, 10/2020)
        elif self.operations.heat_flux_mode == HeatExcitation.FROM_USER_FUNCTION:
            # AM2 part to be implemented
            if conductor.cond_time[-1] == 0:
                self.user_heat_function(conductor)
            elif conductor.cond_num_step > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, since after that the whole system load vector is saved and there is no need to compute twice the same values.
                    self.node_fields.EXTFLX[:, 1] = self.node_fields.EXTFLX[
                        :, 0
                    ].copy()
                # end if conductor.cond_num_step (cdp, 10/2020)
                # call method load_user_defined_quantity to compute heat and overwrite the previous values.
                self.user_heat_function(conductor)
            # end if conductor.cond_num_step (cdp, 10/2020)
        # end self.operations["IQFUN"] (cdp, 10/2020)

    # end Get_Q

    def heat0_where(self, conductor):

        # Compute at each time step since the mesh can change
        lower_bound = np.min(
            np.nonzero(conductor.mesh.node_coordinates >= self.operations.heat_flux_position_start)
        )
        upper_bound = np.max(
            np.nonzero(conductor.mesh.node_coordinates <= self.operations.heat_flux_position_end)
        )
        if self.operations.heat_flux_mode == HeatExcitation.SQUARE_WAVE_IN_TIME_AND_SPACE:
            # Square wave in time and space (cdp, 11/2020)
            if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
                # Backward Euler or Crank-Nicolson (cdp, 10/2020)
                if conductor.cond_num_step == 0:
                    # Initialization to Q0 value: this occurs when TQBEG = 0.0 s, i.e. \
                    # heating starts at the beginning of the simulation (cdp, 10/2020)
                    self.node_fields.EXTFLX[
                        lower_bound : upper_bound + 1, 0
                    ] = self.operations.heat_flux_amplitude
                elif conductor.cond_num_step > 0:
                    if conductor.cond_num_step == 1:
                        # Store the old values only immediately after the initializzation, \
                        # since after that the whole system load vector is saved and there is no \
                        # need to compute twice the same values (cdp, 10/2020)
                        self.node_fields.EXTFLX[:, 1] = self.node_fields.EXTFLX[
                            :, 0
                        ].copy()
                    self.node_fields.EXTFLX[
                        lower_bound : upper_bound + 1, 0
                    ] = self.operations.heat_flux_amplitude
                # end if (cdp, 10/2020)
            elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
                # Adams-Moulton 4 (cdp, 10/2020)
                if conductor.cond_num_step == 0:
                    self.node_fields.EXTFLX[
                        lower_bound : upper_bound + 1, 0
                    ] = self.operations.heat_flux_amplitude
                    for cc in range(1, 4):
                        # initialize the other columns to the same array: dummy steady \
                        # state (cdp, 10/2020)
                        self.node_fields.EXTFLX[:, cc] = self.node_fields.EXTFLX[:, 0].copy()
                    # end for cc (cdp, 10/2020)
                elif conductor.cond_num_step > 0:
                    self.node_fields.EXTFLX[:, 1:4] = self.node_fields.EXTFLX[
                        :, 0:3
                    ].copy()
                    self.node_fields.EXTFLX[
                        lower_bound : upper_bound + 1, 0
                    ] = self.operations.heat_flux_amplitude
                # end if (cdp, 10/2020)
            # end if conductor.inputs["METHOD"] (cdp, 10/2020)
        # end if self.operations["IQFUN"] (cdp, 11/2020)

    # end Q0_where

    def user_heat_function(self, arg):
        # Method that allows user to define an arbitrary function for heat
        # load.
        # To be edited.
        pass

    def jhtflx_new_0(self, conductor):  # tesded: ok (cdp, 06/2020)

        """
        ############################################################################
        #              JHTFLX_new_0(self, conductor)
        ############################################################################
        #
        # Method that initialize to zero the Joule heating flux in python objects
        # of class SolidComponent.
        #
        ############################################################################
        # VARIABLE    I/O    TYPE              DESCRIPTION                      UNIT
        # --------------------------------------------------------------------------
        # self        I      object            python object of
        #                                      class SolidComponent           -
        # zcoord*     I      np array float    conductor spatial
        #                                      discretization                  m
        # JHTFLX      O      np array float    Joule heating flux
        #                                      vector                          W/m^2
        #############################################################################
        # Invoched functions/methods: none
        #
        ############################################################################
        # * zcoord is given by conductor.zcoord
        # N.B. JHTFLX is a SolidComponent attribute so its value can be assigned
        # directly, it has the same of shape of zcoord and it is a np array.
        ############################################################################
        #
        # Author D. Placido Polito 06/2020
        #
        ############################################################################
        """

        # Method JHTFLX_new_0 starts here. (cdp, 06/2020)
        if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
            # Backward Euler or Crank-Nicolson (cdp, 10/2020)
            if conductor.cond_time[-1] == 0:
                # Initialization (cdp, 10/2020)
                self.node_fields.JHTFLX = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
            elif conductor.cond_time[-1] > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, \
                    # since after that the whole system load vector is saved and there is no \
                    # need to compute twice the same values (cdp, 10/2020)
                    self.node_fields.JHTFLX[:, 1] = self.node_fields.JHTFLX[
                        :, 0
                    ].copy()
                # Update value at the current time step (cdp, 10/2020)
                self.node_fields.JHTFLX[:, 0] = 0.0
            # end if conductor.cond_time[-1] (cdp, 10/2020)
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4 (cdp, 10/2020)
            if conductor.cond_time[-1] == 0:
                # Initialization (cdp, 10/2020)
                self.node_fields.JHTFLX = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0:
                self.node_fields.JHTFLX[:, 1:4] = self.node_fields.JHTFLX[
                    :, 0:3
                ].copy()
                # Update value at the current time step (cdp, 10/2020)
                self.node_fields.JHTFLX[:, 0] = 0.0
                # end if conductor.cond_time[-1] (cdp, 10/2020)
        # end if conductor.inputs["METHOD"] (cdp, 10/2020)

    # end JHTFLX_new_0

    def initialize_electric_quantities(self, conductor):
        """Method that initializes to zero some arrays that are an outcome of the electric method for each SolidComponent object:

        * self.gauss_fields.current_along;
        * self.gauss_fields.delta_voltage_along;
        * self.gauss_fields.delta_voltage_along_sum;
        * self.node_fields.total_power_el_cond.
        """

        self.gauss_fields.current_along = np.zeros(conductor.mesh.number_of_elements)
        self.gauss_fields.delta_voltage_along = np.zeros(
            conductor.mesh.number_of_elements
        )
        self.gauss_fields.delta_voltage_along_sum = np.zeros(
            conductor.mesh.number_of_elements
        )
        self.node_fields.total_power_el_cond = np.zeros(
            conductor.mesh.number_of_nodes
        )

    def get_joule_power_along(self, conductor: object):
        """Method that evaluate the contribution to the total power in the element of Joule power (in W/m) due to the electic resistances along the SolidComponent objects.

        Args:
            conductor (object): ConductorComponent object with all informations to make the calculation.
        """

        if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
            # Backward Euler or Crank-Nicolson.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.gauss_fields.linear_power_el_resistance = np.zeros(
                    (conductor.mesh.number_of_elements, 2)
                )
            elif conductor.cond_time[-1] > 0 and conductor.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the
                    # initializzation, since after that the whole system load vector
                    # is saved and there is no need to compute twice the same
                    # values.
                    self.gauss_fields.linear_power_el_resistance[
                        :, 1
                    ] = self.gauss_fields.linear_power_el_resistance[:, 0].copy()
                if self.name != "Z_JACKET":
                    # Evaluate Joule linear power along the strand in W/m, due
                    # to electric resistances only for current carriers:
                    # P_along = R_along * I_along ^2 / (Delta_z * costheta)
                    self.gauss_fields.linear_power_el_resistance[:, 0] = (
                        self.gauss_fields.current_along ** 2
                        * self.gauss_fields.electric_resistance
                        / (conductor.mesh.element_lengths * self.inputs.cos_theta)
                    )
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.gauss_fields.linear_power_el_resistance = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0 and conductor.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
                self.gauss_fields.linear_power_el_resistance[
                    :, 1:4
                ] = self.gauss_fields.linear_power_el_resistance[:, 0:3].copy()
                if self.name != "Z_JACKET":
                    # Evaluate Joule linear power along the strand in W/m, due
                    # to electric resistances only for current carriers:
                    # P_along = R_along * I_along ^2 / (Delta_z * costheta)
                    self.gauss_fields.linear_power_el_resistance[:, 0] = (
                        self.gauss_fields.current_along ** 2
                        * self.gauss_fields.electric_resistance
                        / (conductor.mesh.element_lengths * self.inputs.cos_theta)
                    )

    def get_joule_power_across(self, conductor: object):
        """Method that evaluates the contribution to the total power in the nodes of Joule power (in W/m) due to the electic conductance across the SolidComponent objects.

        Args:
            conductor (object): ConductorComponent object with all informations to make the calculation.
        """

        if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
            # Backward Euler or Crank-Nicolson.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.node_fields.total_linear_power_el_cond = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
            elif conductor.cond_time[-1] > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the
                    # initializzation, since after that the whole system load vector
                    # is saved and there is no need to compute twice the same
                    # values.
                    self.node_fields.total_linear_power_el_cond[
                        :, 1
                    ] = self.node_fields.total_linear_power_el_cond[:, 0].copy()
                if self.name != "Z_JACKET":
                    # Evaluate total Joule linear power across the strand in
                    # W/m, due to electric conductance only for current
                    # carriers:
                    # P_l_t = P_t / Delta_z_tilde
                    self.node_fields.total_linear_power_el_cond[:, 0] = (
                        self.node_fields.total_power_el_cond
                        / conductor.mesh.dual_element_lengths
                    )
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4.
            if conductor.cond_time[-1] == 0:
                # Initialization.
                self.node_fields.total_linear_power_el_cond = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0:
                self.node_fields.total_linear_power_el_cond[
                    :, 1:4
                ] = self.node_fields.total_linear_power_el_cond[:, 0:3].copy()
                if self.name != "Z_JACKET":
                    # Evaluate total Joule linear power across the strand in
                    # W/m, due to electric conductance only for current
                    # carriers:
                    # P_l_t = P_t / Delta_z_tilde
                    self.node_fields.total_linear_power_el_cond[:, 0] = (
                        self.node_fields.total_power_el_cond
                        / conductor.mesh.dual_element_lengths
                    )

    def set_energy_counters(self, conductor):
        # tesded: ok (cdp, 06/2020)

        """
        ############################################################################
        #              Set_energy_counters(self, conductor)
        ############################################################################
        #
        # Method that initialize to zero the external energy and Joule heating in
        # python objects of class SolidComponent.
        #
        ############################################################################
        # VARIABLE    I/O    TYPE              DESCRIPTION                      UNIT
        # --------------------------------------------------------------------------
        # self        I      object            python object of
        #                                      class SolidComponent           -
        # zcoord*     I      np array float    conductor spatial
        #                                      discretization                  m
        # EEXT        O      np array float    external heating vector         MJ
        # EJHT        O      np array float    external Joule heating
        #                                      vector                          MJ
        #############################################################################
        # Invoched functions/methods: none
        #
        ############################################################################
        # * zcoord is given by conductor.zcoord
        # N.B. EEXT and EJHT are SolidComponent attributes so therir value can be
        # assigned directly they have the same of shape of zcoord and they are np
        # arrays.
        ############################################################################
        #
        # Author D. Placido Polito 06/2020
        #
        ############################################################################
        """

        # Method Set_energy_counters starts here. (cdp, 06/2020)

        if conductor.inputs.thermohydraulic_method in ONE_STEP_METHODS:
            # Backward Euler or Crank-Nicolson (cdp, 10/2020)
            if conductor.cond_time[-1] == 0:
                # Initialization (cdp, 10/2020)
                self.node_fields.EEXT = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
                self.node_fields.EJHT = np.zeros(
                    (conductor.mesh.number_of_nodes, 2)
                )
            elif conductor.cond_time[-1] > 0:
                if conductor.cond_num_step == 1:
                    # Store the old values only immediately after the initializzation, \
                    # since after that the whole system load vector is saved and there is no \
                    # need to compute twice the same values (cdp, 10/2020)
                    self.node_fields.EEXT[:, 1] = self.node_fields.EEXT[
                        :, 0
                    ].copy()
                self.node_fields.EJHT[:, 1] = self.node_fields.EJHT[:, 0].copy()
                # Update value at the current time step (cdp, 10/2020)
                self.node_fields.EEXT[:, 0] = 0.0
                self.node_fields.EJHT[:, 0] = 0.0
            # end if conductor.cond_time[-1] (cdp, 10/2020)
        elif conductor.inputs.thermohydraulic_method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton 4 (cdp, 10/2020)
            if conductor.cond_time[-1] == 0:
                # Initialization (cdp, 10/2020)
                self.node_fields.EEXT = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
                self.node_fields.EJHT = np.zeros(
                    (conductor.mesh.number_of_nodes, 4)
                )
            elif conductor.cond_time[-1] > 0:
                self.node_fields.EEXT[:, 1:4] = self.node_fields.EEXT[
                    :, 0:3
                ].copy()
                self.node_fields.EJHT[:, 1:4] = self.node_fields.EJHT[
                    :, 0:3
                ].copy()
                # Update value at the current time step (cdp, 10/2020)
                self.node_fields.EEXT[:, 0] = 0.0
                self.node_fields.EJHT[:, 0] = 0.0
            # end if conductor.cond_time[-1] (cdp, 10/2020)
        # end if conductor.inputs["METHOD"] (cdp, 10/2020)

    # end Set_energy_counters

    def deal_with_flag_IOP_MODE(self):
        """Method that checks and converts values assigned to flag operating_current_mode.

        Raises:
            ValueError: if self.operations.operating_current_mode is a string different from 'none'.
        """
        if type(self.operations.operating_current_mode) == str:
            self.operations.operating_current_mode = self.operations.operating_current_mode.lower()
            if self.operations.operating_current_mode == "none":
                self.operations.operating_current_mode = None
            else:
                raise ValueError(
                    f"Not valid value to flag self.operations.operating_current_mode. Possible values are -1, 0 or 'none', current value is {self.operations.operating_current_mode=}. Check sheet {self.identifier} of input file conuctor_operation.xlsx.\n"
                )
        else:
            # Temporary solution to manage input file loading, strange
            # behavior: 1 are converted to True but 0 not converted to False.
            if self.operations.operating_current_mode == True:
                self.operations.operating_current_mode = 1
            elif self.operations.operating_current_mode == False:
                self.operations.operating_current_mode = 0