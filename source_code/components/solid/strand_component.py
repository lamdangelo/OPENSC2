import warnings
from components.solid.solid_component import SolidComponent
from electromagnetics.electromagnetic_flags import CurrentMode
import numpy as np
from utility_functions.auxiliary_functions import (
    load_auxiliary_files,
    build_interpolator,
    do_interpolation,
)

# from utility_functions.InitializationFunctions import Read_input_file
# NbTi properties
from properties_of_materials.niobium_titanium import (
    critical_temperature_nbti,
    critical_current_density_nbti,
    current_sharing_temperature_nbti,
)

# NbTi properties, W7-X parameterization (CryoSoft/THEA)
from properties_of_materials.niobium_titanium_w7x import (
    critical_temperature_nbti_w7x,
    critical_current_density_nbti_w7x,
    current_sharing_temperature_nbti_w7x,
)

# Nb3Sn properties
from properties_of_materials.niobium3_tin import (
    critical_temperature_nb3sn,
    critical_current_density_nb3sn,
    current_sharing_temperature_nb3sn,
)

# RE123 properties
from properties_of_materials.rare_earth_123 import (
    critical_current_density_re123,
    current_sharing_temperature_re123,
)

# MgB2 properties
from properties_of_materials.magnesium_diboride import (
    critical_magnetic_field_mgb2,
    critical_current_density_mgb2,
    current_sharing_temperature_mgb2,
)


class StrandComponent(SolidComponent):

    ### INPUT PARAMETERS

    ### OPERATIONAL PARAMETERS
    # inherited from class SolidComponent

    ### COMPUTED IN INITIALIZATION
    # inherited from class SolidComponent

    KIND = "Strand"

    def get_magnetic_field_gradient(self, conductor, nodal=True):

        """
        ############################################################################
        #              Get_alphaB(self, comp)
        ############################################################################
        #
        # Method that initialize magnetic field gradient in python objects of class
        # Strands.
        #
        ############################################################################
        # VARIABLE              I/O    TYPE              DESCRIPTION            UNIT
        # --------------------------------------------------------------------------
        # comp                  I      object            python object of
        #                                                class Strands             -
        # zcoord*               I      np array float    conductor spatial
        #                                                discretization            m
        # I0_OP_MODE*               I      scalar integer    flag to decide how to
        #                                                evaluate current:
        #                                                == 0 -> constant;
        #                                                == 1 -> exponential decay;
        #                                                == -1 -> read from
        #                                                I_file_dummy.xlsx         -
        # IOP_TOT*              I      scalar float      total operation
        #                                                current                   A
        # I0_OP_TOT*             I      scalar float      total operation current
        #                                                @ time = 0                A
        # IALPHAB§              I      scalar integer    flag to define the
        #                                                magnetic field gradient
        #                                                along the strand:
        #                                                == 0 -> no gradient;
        #                                                == -1 -> read from
        #                                                alphab.xlsx               -
        # BASE_PATH*             I      string           path of folder Description
        #                                               of Components              -
        # External_current_path* I     string           name of the external
        #                                               file from which
        #                                               interpolate magnetic
        #                                               field gradient value       -
        # alphaB§               O      np array float   magnetic field
        #                                               gradient                 T/m
        #############################################################################
        # Invoched functions/methods: Get_from_xlsx
        #
        ############################################################################
        # * zcoord, I0_OP_MODE, IOP_TOT, I0_OP_TOT, BASE_PATH and External_current_path
        # are given by conductor.zcoord, conductor.inputs["I0_OP_MODE"], conductor.IOP_TOT, conductor.inputs["I0_OP_TOT"],
        # conductor.BASE_PATH and conductor.file_input["EXTERNAL_CURRENT"].
        # § IALPHAB and alphaB are component attributes: self.IALPHAB, self.node_fields.alpha_B.
        # N.B. alphaB is a Strands attribute so its value can be assigned
        # directly, it has the same of shape of zcoord and it is a np array.
        ############################################################################
        #
        # Translated and optimized by D. Placido Polito 06/2020
        #
        ############################################################################
        """

        if nodal:
            # compute alpha_B in each node (cdp, 07/2020)
            if self.operations.alpha_b_mode <= -1:  # read from file
                if conductor.cond_time[-1] == 0:
                    # Build file path.
                    file_path = conductor.file_paths.external_alphab
                    # Load auxiliary input file.
                    alphab_df, self.flagSpecfield_alpha_b = load_auxiliary_files(
                        file_path, sheetname=self.identifier
                    )
                    # Build interpolator and get the interpolaion flag (space_only,time_only or space_and_time).
                    (
                        self.alphab_interpolator,
                        self.alphab_interp_flag,
                    ) = build_interpolator(
                        alphab_df, self.operations.alpha_b_interpolation
                    )

                # call load_user_defined_quantity on the component.
                self.node_fields.alpha_B = do_interpolation(
                    self.alphab_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.electric_time,
                    self.alphab_interp_flag,
                )

                # leggi un file come del campo magnetico
                # controlla se e' per unita' di corrente
                # in caso affermatico moltiplica per IOP_TOT
                if self.flagSpecfield_alpha_b == 2:  # alphaB is per unit of current
                    self.node_fields.alpha_B = (
                        self.node_fields.alpha_B * conductor.inputs.initial_current
                    )
                if conductor.inputs.current_mode in (
                    CurrentMode.CURRENT_IS_FROM_FILE,
                    CurrentMode.CURRENT_IS_FUNCTION,
                ):
                    self.node_fields.alpha_B = (
                        self.node_fields.alpha_B
                        * conductor.inputs.initial_current
                        / conductor.inputs.initial_current
                    )
            elif self.operations.alpha_b_mode == 0:
                self.node_fields.alpha_B = np.zeros(
                    conductor.mesh.number_of_nodes
                )
        elif nodal == False:
            # compute alpha_B in each Gauss point (cdp, 07/2020)
            self.gauss_fields.alpha_B = (
                np.abs(
                    self.node_fields.alpha_B[
                        0 : conductor.mesh.number_of_nodes - 1
                    ]
                    + self.node_fields.alpha_B[
                        1 : conductor.mesh.number_of_nodes + 1
                    ]
                )
                / 2.0
            )

    def get_superconductor_critical_prop(self, conductor, nodal=True):

        """
        Calcola i margini
        usa temperatura strand
        Usa campo magnetico BFIELD
        usa EPSI
        usa alphaB
        :return: Jcritck Tcritchk TCSHRE TCSHREmin
        """

        # Properties evaluation in each nodal point (cdp, 07/2020)
        # Variable where is necessary to correctly evaluate EPSILON calling method \
        # Get_EPS (cdp, 07/2020)
        if nodal:
            self.node_fields = self.eval_critical_properties(self.node_fields)
        # Properties evaluation in each Gauss point (cdp, 07/2020)
        elif nodal == False:
            self.gauss_fields = self.eval_critical_properties(self.gauss_fields)

    # End of Method get_superconductor_critical_prop

    def eval_critical_properties(self, dict_dummy):

        if self.inputs.superconducting_material == "nbti":
            dict_dummy.T_critical = critical_temperature_nbti(
                dict_dummy.B_field, self.inputs.upper_critical_field_at_0K, self.inputs.critical_temperature_at_0T
            )
            dict_dummy.J_critical = critical_current_density_nbti(
                dict_dummy.temperature,
                dict_dummy.B_field,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
        elif self.inputs.superconducting_material == "nbti-w7x":
            dict_dummy.T_critical = critical_temperature_nbti_w7x(
                dict_dummy.B_field,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_temperature_at_0T,
            )
            dict_dummy.J_critical = critical_current_density_nbti_w7x(
                dict_dummy.temperature,
                dict_dummy.B_field,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
        elif self.inputs.superconducting_material == "nb3sn":
            dict_dummy.T_critical = critical_temperature_nb3sn(
                dict_dummy.B_field,
                dict_dummy.Epsilon,
                self.inputs.critical_temperature_at_0T,
                self.inputs.upper_critical_field_at_0K,
            )
            dict_dummy.J_critical = critical_current_density_nb3sn(
                dict_dummy.temperature,
                dict_dummy.B_field,
                dict_dummy.Epsilon,
                self.inputs.critical_temperature_at_0T,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
            )
        elif self.inputs.superconducting_material == "ybco":
            dict_dummy.T_critical = self.inputs.critical_temperature_at_0T * np.ones(
                dict_dummy.temperature.shape
            )
            dict_dummy.J_critical = critical_current_density_re123(
                dict_dummy.temperature,
                dict_dummy.B_field,
                self.inputs.critical_temperature_at_0T,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
            )
        elif self.inputs.superconducting_material == "mgb2":
            dict_dummy.T_critical = self.inputs.critical_temperature_at_0T * np.ones(
                dict_dummy.temperature.shape
            )
            dict_dummy.J_critical = critical_current_density_mgb2(
                dict_dummy.temperature,
                dict_dummy.B_field,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
        elif self.inputs.superconducting_material == "scaling.dat":
            # Get user defined scaling invoking method User_scaling_margin \
            # (cdp, 10/2020)
            self.user_scaling_margin()

        return dict_dummy

    # end method Eval_critical_properties (cdp, 10/2020)

    def get_tcs(self, nodal=True):
        """Method that allows the evaluation of the current sharing temperature in nodal points or in Gauss points, according to flag nodal.

        Args:
            nodal (bool, optional): Flag to evaluate the current sharing temperature in proper location. If True evaluation is on the nodal points, if False evaluation is on the Gauss points. Defaults to True.
        """

        # Current sharing temperature evaluation in each nodal point
        if nodal:
            self.node_fields = self.eval_tcs(self.node_fields)
        # Current sharing temperature evaluation in each Gauss point
        elif nodal == False:
            self.gauss_fields = self.eval_tcs(self.gauss_fields)

    # End method get_tcs

    def eval_tcs(self, dict_dummy):

        # The scaling parameter c0 is converted to the physic definition if the 
        # ingegneristic definition is given as input. Therefore, for the 
        # evaluation of the current densities the total superconducting 
        # perpendicular cross section is always used.
        jop = (
            np.abs(dict_dummy.op_current)
            / (self.cross_section["sc"])
        )

        bmax = dict_dummy.B_field * (1 + dict_dummy.alpha_B)
        if self.inputs.superconducting_material == "nbti":
            dict_dummy.T_cur_sharing = current_sharing_temperature_nbti(
                dict_dummy.B_field,
                jop,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
            # dict_dummy.T_cur_sharing_min = current_sharing_temperature_nbti(
            #     bmax,
            #     jop,
            #     self.inputs.upper_critical_field_at_0K,
            #     self.inputs.critical_current_scaling_constant,
            #     self.inputs.critical_temperature_at_0T,
            # )
            dict_dummy.T_cur_sharing_min = dict_dummy.T_cur_sharing
        elif self.inputs.superconducting_material == "nbti-w7x":
            dict_dummy.T_cur_sharing = current_sharing_temperature_nbti_w7x(
                dict_dummy.B_field,
                jop,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
            dict_dummy.T_cur_sharing_min = dict_dummy.T_cur_sharing
        elif self.inputs.superconducting_material == "nb3sn":
            dict_dummy.T_cur_sharing = current_sharing_temperature_nb3sn(
                dict_dummy.B_field,
                dict_dummy.Epsilon,
                jop,
                self.inputs.critical_temperature_at_0T,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
            )
            # dict_dummy.T_cur_sharing_min = current_sharing_temperature_nb3sn(
            #     bmax,
            #     dict_dummy.Epsilon,
            #     jop,
            #     self.inputs.critical_temperature_at_0T,
            #     self.inputs.upper_critical_field_at_0K,
            #     self.inputs.critical_current_scaling_constant,
            # )
            dict_dummy.T_cur_sharing_min = dict_dummy.T_cur_sharing
        elif self.inputs.superconducting_material == "ybco":
            dict_dummy.T_cur_sharing = current_sharing_temperature_re123(
                dict_dummy.B_field,
                jop,
                self.inputs.critical_temperature_at_0T,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
            )
            # dict_dummy.T_cur_sharing_min = current_sharing_temperature_re123(
            #     bmax,
            #     jop,
            #     self.inputs.critical_temperature_at_0T,
            #     self.inputs.upper_critical_field_at_0K,
            #     self.inputs.critical_current_scaling_constant,
            # )
            dict_dummy.T_cur_sharing_min = dict_dummy.T_cur_sharing
        elif self.inputs.superconducting_material == "mgb2":
            dict_dummy.T_cur_sharing = current_sharing_temperature_mgb2(
                dict_dummy.B_field,
                jop,
                self.inputs.upper_critical_field_at_0K,
                self.inputs.critical_current_scaling_constant,
                self.inputs.critical_temperature_at_0T,
            )
            # dict_dummy.T_cur_sharing_min = 
            # current_sharing_temperature_mgb2(
            #     bmax,
            #     jop,
            #     self.inputs.upper_critical_field_at_0K,
            #     self.inputs.critical_current_scaling_constant,
            #     self.inputs.critical_temperature_at_0T,
            # )
            dict_dummy.T_cur_sharing_min = dict_dummy.T_cur_sharing
        elif self.inputs.superconducting_material == "scaling.dat":

            warnings.warn("Still to be understood what to do here!!")

        return dict_dummy

    # End method eval_tcs

    def get_eps(self, conductor, nodal=True):
        # For each strand of type StrandMixedComponent or StackComponent (cdp, 06/2020)
        if nodal:
            # compute Epsilon in each node (cdp, 07/2020)
            if self.operations.strain_mode < 0:  # strain from file strain.dat

                if conductor.cond_time[-1] == 0:
                    # Build file path.
                    file_path = conductor.file_paths.external_strain
                    # Load auxiliary input file.
                    eps_df, self.flagSpecfield_eps = load_auxiliary_files(
                        file_path, sheetname=self.identifier
                    )
                    # Build interpolator and get the interpolaion flag (space_only,time_only or space_and_time).
                    self.eps_interpolator, self.eps_interp_flag = build_interpolator(
                        eps_df, self.operations.operating_current_interpolation
                    )

                # call load_user_defined_quantity on the component.
                self.node_fields.Epsilon = do_interpolation(
                    self.eps_interpolator,
                    conductor.mesh.node_coordinates,
                    conductor.electric_time,
                    self.eps_interp_flag,
                )

                if self.flagSpecfield_eps == 1:
                    # Add also a logger
                    warnings.warn("Still to be decided what to do here\n")
            elif self.operations.strain_mode == 0:  # no strain (cdp, 06/2020)
                self.node_fields.Epsilon = np.zeros(
                    conductor.mesh.number_of_nodes
                )
            elif self.operations.strain_mode == 1:
                # constant strain to the value in input file \
                # conductor_i_operation.xlsx (cdp, 06/2020)
                self.node_fields.Epsilon = self.operations.strain_value * np.ones(
                    conductor.mesh.number_of_nodes
                )
        elif nodal == False:
            # compute Epsilon in each Gauss point (cdp, 07/2020)
            self.gauss_fields.Epsilon = (
                self.node_fields.Epsilon[0 : conductor.mesh.number_of_nodes - 1]
                + self.node_fields.Epsilon[1 : conductor.mesh.number_of_nodes + 1]
            ) / 2.0

    # end method Get_EPS (cdp, 10/2020)

    def user_scaling_margin(self):

        """
        Method that read file scaling_input.dat to get the user scaling for SuperConductors strands and convert it into an attribute dictionary (cdp, 10/2020)
        """

        # declare dictionary (cdp, 10/2020)
        # construct list of integer values (cdp, 10/2020)
        list_integer = ["emode", "ieavside"]
        # Loop to read file scaling_input.dat by lines and construct a \
        # dictionary (cdp, 10/2020)
        with open("scaling_input.dat", "r") as scaling:
            # Read lines (cdp, 10/2020)
            for line in scaling:
                if line[0] == "#" or line[0] == "\n":
                    # escape comments and void lines (cdp, 10/2020)
                    pass
                else:
                    # split the list into two fields, fields[0] is dictionary key,
                    # fields[1] is the corresponding value. The final \n is ignored \
                    # considering only the first [:-1] characters in string fields[1] \
                    # (cdp, 10/2020)
                    fields = line.split(" = ")
                    if fields[0] in list_integer:
                        # convert to integer (cdp, 10/2020)
                        self.dict_scaling_input[fields[0]] = int(fields[1][:-1])
                    elif fields[0] == "superconducting_material":
                        # flag to
                        self.dict_scaling_input[fields[0]] = str(fields[1][:-1])
                    else:
                        # convert to float (cdp, 10/2020)
                        self.dict_scaling_input[fields[0]] = float(fields[1][:-1])
                    # end if fields[0] (cdp, 10/2020)
                # end if line[0] (cdp, 10/2020)
            # end for line (cdp, 10/2020)
        # end with (cdp, 10/2020)

    # end method User_scaling_margin (cdp, 10/2020)

    def electric_resistance(
        self, conductor: object, electrical_resistivity_key: str, cross_section_key: str, ind: np.ndarray
    ) -> np.ndarray:
        f"""Method that evaluate electric resistance of a single material, such as superconductor, stabilizer, stainless steel and other electric conducting material.

        Args:
            conductor (object): class Conductor object in which distance between consecutive nodes is stored to do the calculation.
            electrical_resistivity_key (str): dictionary key for the electrical resistivity of the material.
            cross_section_key (str): dictionary key for the perpendicular cross section of the material.
            ind (np.ndarray): array with the index of the location in wich electric resistance should be evaluated with this method.

        Returns:
            np.ndarray: array of the electric resistance in Ohm of shape {ind.shape = }. The maximum lenght of the outcome is {conductor.mesh.number_of_elements = }.
        """
        node_dist = conductor.node_distance[("StrandComponent", self.identifier)].iloc[ind]

        return (
            getattr(self.gauss_fields, electrical_resistivity_key)[ind]
            * node_dist
            / self.cross_section[cross_section_key]
        )

    def parallel_electric_resistance(
        self, conductor: object, electrical_resistivity_keys: list, cross_section_keys: list, ind: np.ndarray
    ) -> np.ndarray:
        f"""Method that evaluate electric resistance in the case of a parallel of two electric conducting materials, as is the case for StackComponent and StrandMixedComponent in current sharing regime.

        Args:
            conductor (object): class Conductor object in which distance between consecutive nodes is stored to do the calculation.
            electrical_resistivity_key (list): list of dictionary key for the electrical resistivity of the materials (typical values for the application of this software are electrical_resistivity_superconductor and electrical_resistivity_stabilizer).
            cross_section_keys (list): list of dictionary key for the cross section of the materials (typical values for the application of this software are sc and stab).
            ind (np.ndarray): array with the index of the location in wich electric resistance should be evaluated with this method.

        Raises:
            ValueError: if list electrical_resistivity_keys does not have exactly two items.
            ValueError: if items in list electrical_resistivity_keys are not of type string.

        Returns:
            np.ndarray: array of the electric resistance in Ohm of shape {ind.shape = }. The maximum lenght of the outcome is {conductor.mesh.number_of_elements = }.
        """
        if len(electrical_resistivity_keys) != 2:
            # Check list lenght.
            raise ValueError(
                f"List electrical_resistivity_keys must have 2 items; {len(electrical_resistivity_keys) = }.\n"
            )

        if not all(isinstance(item, str) for item in electrical_resistivity_keys):
            # Check that all items in list are string.
            raise ValueError(
                f"All items in list electrical_resistivity_keys must be of type string. {electrical_resistivity_keys = }.\n"
            )
        
        if len(cross_section_keys) != 2:
            # Check list lenght.
            raise ValueError(
                f"List cross_section_keys must have 2 items; {cross_section_keys = }.\n"
            )

        if not all(isinstance(item, str) for item in cross_section_keys):
            # Check that all items in list are string.
            raise ValueError(
                f"All items in list cross_section_keys must be of type string. {cross_section_keys = }.\n"
            )

        # Electric resistance matrix initialization.
        electric_resistances = np.zeros((ind.size, 2))
        # Evaluate electri resistances
        for ii, item in enumerate(electrical_resistivity_keys):
            electric_resistances[:, ii] = self.electric_resistance(conductor, item, cross_section_keys[ii], ind)

        # Evaluate parallel electric resistance:
        # R_eq = R1*R2/(R1+R2)
        return (
            electric_resistances[:, 0]
            * electric_resistances[:, 1]
            / (electric_resistances.sum(axis=1))
        )

    def __manage_fixed_potental(self, length: float):
        """Method that deals with fixed potentials: converts fixed potential values to array if they are integers or strings and checks the coordinate where fixed potentials are assigned.

        Args:
            length (float): conductor length.
        """
        self.__convert_fixed_potential_to_array()
        self.__checks_fix_potential_coordinate(length)

    def __convert_fixed_potential_to_array(self):
        """Private method that allows to convert fix_potential_coordinate and fix_potential_value to numpy array."""
        private_methods = {
            int: self.__int_or_float_to_array,
            float: self.__int_or_float_to_array,
            str: self.__str_to_array,
        }
        for attr in ["fix_potential_coordinate", "fix_potential_value"]:
            private_methods[type(getattr(self.operations, attr))](attr)

    def __int_or_float_to_array(self, attr: str):
        """Private method that converts fix_potential_coordinate or fix_potential_value from int/float to ndarray of float.

        Args:
            attr (str): attribute name on self.operations.
        """
        setattr(self.operations, attr, np.array([getattr(self.operations, attr)], dtype=float))

    def __str_to_array(self, attr: str):
        """Private method that converts fix_potential_coordinate or fix_potential_value from str to ndarray of float.

        Args:
            attr (str): attribute name on self.operations.
        """
        setattr(self.operations, attr, np.array(getattr(self.operations, attr).split(","), dtype=float))

    def __checks_fix_potential_coordinate(self, length: float):
        """Private method that checks consistency between input values FIX_POTENTIAL_NUMBER, FIX_POTENTIAL_COORDINATE and FIX_POTENTIAL_VALUE after conversion of FIX_POTENTIAL_COORDINATE and FIX_POTENTIAL_VALUE to ndarray.

        Args:
            length (float): length of the conductor.

        Raises:
            ValueError: if length of ndarray FIX_POTENTIAL_COORDINATE or FIX_POTENTIAL_VALUE is not equal to FIX_POTENTIAL_NUMBER.
            ValueError: if maximum value in FIX_POTENTIAL_COORDINATE is larger than the length of the conductor.
        """

        # Build dictionary with error messages.
        message = dict(
            fix_potential_coordinate=f"The number of the coordinates of fixed potential must be equal to the number of declared fixed potential:\n{self.operations.fix_potential_coordinate=};\n{self.operations.fix_potential_number=}\n",
            fix_potential_value=f"The number of the values of fixed potential must be equal to the number of declared fixed potential:\n{self.operations.fix_potential_value=};\n{self.operations.fix_potential_number=}\n",
        )

        # Check ndarray length consistency.
        for attr in message.keys():
            if len(getattr(self.operations, attr)) != self.operations.fix_potential_number:
                raise ValueError(message[attr])

        # Check fix_potential_coordinate consistency with conductor length.
        if np.max(self.operations.fix_potential_coordinate) > length:
            raise ValueError(
                f"Fixed potential coordinate cannot exceed conductor length:\n{np.max(self.operations.fix_potential_coordinate)=}m;\n{length=}m"
            )

    def __delete_fixed_potential_inputs(self, _: float):
        """Method that forces self.operations["FIX_POTENTIAL_NUMBER"] to be zero and deletes no longer useful keys FIX_POTENTIAL_COORDINATE and FIX_POTENTIAL_VALUE from dictionary self.operations.

        Args:
            _ (float): any float varyables, not used.
        """

        self.operations.fix_potential_number = 0
        self.operations.fix_potential_coordinate = None
        self.operations.fix_potential_value = None

    def deal_with_fixed_potential(self, length: float):
        """Method that deals with fixed potential input. It converts True value to 1 (odd beaviour) if necessary; defines dictionary with private methods self.__manage_fixed_potental and self.__delete_fixed_potential_inputs, and calls them according to input value FIX_POTENTIAL_FLAG.

        Args:
            length (float): conductor length.
        """

        # Temporary solution to manage input file loading, strange behavior: 1
        # are converted to True but 0 not converted to False.
        for attr in ["fix_potential_number", "fix_potential_coordinate", "fix_potential_value"]:
            if getattr(self.operations, attr) == True:
                setattr(self.operations, attr, 1)

        methods = {
            True: self.__manage_fixed_potental,
            False: self.__delete_fixed_potential_inputs,
        }

        methods[self.operations.fix_potential_flag](length)