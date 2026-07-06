"""This module contains the FluidComponent class.
properties:
-number: Number of fluid channels (M)
-type: Type of fluid	To be chosen among He or N2 – thermophysical properties to be evaluated, according to the choice
-crossSection: Cross section of any of the M channels	Typically in [mm^2]
-costheta: Cos(teta) of any of the M channels	Angle of inclination wrt the jacket axis
-contactPerimeterSC: Contact perimeter (equivalent to area per unit length) with any of the Z SC strands and K stabilizer strands, or N SC + stabilizer strands	Typically in [mm]
-htc: Heat transfer coefficient (equivalent to area per unit length) with any of the Z SC strands and K stabilizer strands, or N SC + stabilizer strands	Typically in [W/m2K] NON SOLO NUMERI QUI, MA ANCHE FUNZIONI
-contactPerimeterJ: Contact perimeter (equivalent to area per unit length) with the jacket	Typically in [mm]
-htcJ: Heat transfer coefficient (equivalent to area per unit length) with the jacket	Typically in [W/m2K] NON SOLO NUMERI QUI, MA ANCHE FUNZIONI
-perimeterTransfer: Perimeter (equivalent to area per unit length) available for the mass transfer with the (M-1) fluid components	Typically in [mm]
-perimeterConductiveFluid: Perimeter (equivalent to area per unit length) available for the pure conductive heat transfer with the (M-1) fluid components	Typically in [mm]
-htc: Heat transfer coefficient (equivalent to area per unit length) with the (M-1) fluid components	Typically in [W/m2K] NON SOLO NUMERI QUI, MA ANCHE FUNZIONI
"""

import numpy as np
from collections import namedtuple
from utility_functions.auxiliary_functions import get_from_xlsx

from components.component import Component 
from components.fluid.fluid_component_inputs import (
    FluidComponentInputs,
    FluidComponentOperations
)
from hydraulics.hydraulic_flags import (
    HydraulicBC,
    FlowDirection,
)

from hydraulics.channel import Channel
from hydraulics.coolant import Coolant

import interfaces.coolprop_interface as cpi


class FluidComponent(Component):

    def __init__(self, identifier: str, inputs: FluidComponentInputs, 
                 operations: FluidComponentOperations):
        self.identifier = identifier
        self.channel = Channel(identifier, inputs, operations)
        self.coolant = Coolant(identifier, inputs, operations)
        self.coordinate = dict()

        # This is a switch to apply BC for the thermal hydraulic problem 
        # according to the absolute value of flat INTIAL.
        self.apply_th_bc = {
            1:self.impose_pressure_drop,  # abs(INTIAL) = 1
            2:self.impose_inl_p_out_v,  # abs(INTIAL) = 2
            3:self.impose_inl_v_out_p,  # abs(INTIAL) = 2
        }

    # End method __init__.

    def __repr__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return f"{self.__class__.__name__}(Type: {self.name}, identifier: {self.identifier})"


    def initialize_physical_fields(self, mesh_nodes: np.ndarray) -> None: 
        """
        Initializes all physical fields according to the boundary conditions.
        """ 
        initial_pressure = np.interp(
            mesh_nodes,
            [0, mesh_nodes[-1]],
            [
                self.coolant.operations.inlet_pressure, 
                self.coolant.operations.outlet_pressure
            ]
        )
        self.coolant.pressure.initialize(initial_pressure)

        initial_temperature = np.interp(
            mesh_nodes,
            [0, mesh_nodes[-1]],
            self.coolant.operations.initial_temperature_profile_bounds(),
        )
        self.coolant.temperature.initialize(initial_temperature)

        initial_density = cpi.compute_mass_density(self.coolant.fluid_type,
                                                   initial_temperature,
                                                   initial_pressure)
        self.coolant.density.initialize(initial_density)

        if self.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY:
            mass_flow_rate = self.coolant.operations.outlet_mass_rate
        else:
            mass_flow_rate = self.coolant.operations.inlet_mass_rate

        initial_velocity = mass_flow_rate / (self.channel.inputs.cross_section * initial_density)
        self.coolant.velocity.initialize(initial_velocity)


    def build_th_bc_index(
        self,
        conductor:object,
    ):
        """Method that builds and initialize the data structure to store the values of the index used to assign the boundary conditions at inlet and outlet for the thermal hydraulic problem. The data structure is a namedtuple with three keys (velocity pressure and temperature), each key has a namedtuple with two keys (forward and backward) to properly assing the boundary conditions in case of coolants in counter current flow.

        Args:
            conductor (Conductor): object with all the information of the conductor.
        """

        # ALIAS
        ndf = conductor.equation_counts.degrees_of_freedom_per_node
        # eq_idx (NamedTuple): collection of fluid equation index (velocity, 
        # pressure and temperaure equations).
        eq_idx = conductor.equation_index[self.identifier]

        # Namedtuple constuctor
        BC_idx = namedtuple("BC_idx",("velocity","pressure","temperature"))
        Flow_dir = namedtuple("Flow_dir",("forward","backward"))
        
        # Build namedtuple with the index used to assign the inlet BC.
        self.inl_idx = BC_idx(
            # Inlet velocity index (forward and backward flow).
            velocity=Flow_dir(
                forward=eq_idx.velocity, # first node
                backward=- ndf + eq_idx.velocity, # last node
            ),
            # Inlet pressure index (forward and backward flow).
            pressure=Flow_dir(
                forward=eq_idx.pressure, # first node
                backward=- ndf + eq_idx.pressure, # last node
            ),
            # Inlet temperature index (forward and backward flow).
            temperature=Flow_dir(
                forward=eq_idx.temperature, # first node
                backward=- ndf + eq_idx.temperature, # last node
            ),
        )

        # Build namedtuple with the index used to assign the outlet BC.
        self.out_idx = BC_idx(
            # Outlet velocity index (forward and backward flow).
            velocity=Flow_dir(
                forward=- ndf + eq_idx.velocity, # last node
                backward=eq_idx.velocity, # first node
            ),
            # Outlet pressure index (forward and backward flow).
            pressure=Flow_dir(
                forward=- ndf + eq_idx.pressure, # last node
                backward=eq_idx.pressure, # first node
            ),
            # Outlet temperature index (forward and backward flow).
            temperature=Flow_dir(
                forward=- ndf + eq_idx.temperature, # last node
                backward=eq_idx.temperature, # first node
            ),
        )

    def __impose_temperature_fw(
        self,
        sysmat:np.ndarray,
        known:np.ndarray,
        T_inl:float,
        T_out:float,
        main_d_idx:int,
        )->tuple:
        """Private method that imposes temperature boundary conditon in the case of forward flow according to the sign of the velocity.

        Args:
            sysmat (np.ndarray): system matrix (SYSMAT) on wich impose the boundary conditions.
            known (np.ndarray): known therm vector (Known) on wich impose the boundary conditions.
            T_inl (float): inlet temperature boundary condition.
            T_out (float): outlet temperature boundary condition.
            main_d_idx (int): index of the main diagonal in sysmat matrix.

        Returns:
            tuple: collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray with imposed temperature as boundary conditions.
        """
        
        # Alias
        inl_t_idx = self.inl_idx.temperature.forward
        out_t_idx = self.out_idx.temperature.forward
        velocity = self.coolant.node_fields.velocity
        
        # Assing inlet temperature (T_inl).
        if velocity[0] > 0:
            sysmat[:,inl_t_idx] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_t_idx] = 1.0
            known[inl_t_idx] = T_inl
        # Assing outlet temperature (T_out).
        if velocity[-1] < 0:
            sysmat[:,out_t_idx] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_t_idx] = 1.0
            known[out_t_idx] = T_out

        return known,sysmat

    def __impose_temperature_bw(self, sysmat:np.ndarray,
        known:np.ndarray,
        T_inl:float,
        T_out:float,
        main_d_idx:int,
        )->tuple:
        """Private method that imposes temperature boundary conditon in the case of backward flow according to the sign of the velocity.

        Args:
            sysmat (np.ndarray): system matrix (SYSMAT) on wich impose the boundary conditions.
            known (np.ndarray): known therm vector (Known) on wich impose the boundary conditions.
            T_inl (float): inlet temperature boundary condition.
            T_out (float): outlet temperature boundary condition.
            main_d_idx (int): index of the main diagonal in sysmat matrix.

        Returns:
            tuple: collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray with imposed temperature as boundary conditions.
        """
        
        # Alias
        inl_t_idx = self.inl_idx.temperature.backward
        out_t_idx = self.out_idx.temperature.backward
        velocity = self.coolant.node_fields.velocity

        # Assing inlet temperature (T_inl).
        if velocity[-1] < 0:
            sysmat[:,inl_t_idx] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_t_idx] = 1.0
            known[inl_t_idx] = T_inl
        # Assing outlet temperature (T_out).
        if velocity[0] > 0:
            sysmat[:,out_t_idx] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_t_idx] = 1.0
            known[out_t_idx] = T_out
        
        return known,sysmat

    def impose_pressure_drop(
        self,
        ndarrays:tuple,
        conductor:object,
        path:str,
        )->tuple:
        
        """Method that imposes a pressure drop as boundary conditions for the thermal hydraulic problem. Imposed quantities are:
            * inlet pressure
            * inlet temperature (or outlet temperature if back flow)
            * outlet pressure
        The method suitably accounts for forward (left to right)/backward (right to left) flow as well as for back flow (part of the fluid moves towards the inlet due to a pressure build up).

        Args:
            ndarrays (tuple): collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray on wich impose the boundary conditions.
            conductor (Conductor): object with all the information of the conductor.
            path (str): path of the auxiliary input file with the values for the boundary conditions (if INTIAL = -1).

        Returns:
            tuple: collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray with imposed boundary conditions.
        """
        
        known,sysmat = ndarrays
        # Get boundary condition values.
        if (self.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP and not self.coolant.operations.bc_values_from_file):
            # inlet pressure.
            p_inl = self.coolant.operations.inlet_pressure
            # inlet temperature.
            T_inl = self.coolant.operations.inlet_temperature
            # outlet pressure.
            p_out = self.coolant.operations.outlet_pressure
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = self.coolant.operations.outlet_temperature
        else:
            # get from file with interpolation in time.
            [flow_par, flagSpecfield] = get_from_xlsx(
                conductor,
                path,
                self,
                "INTIAL",
                -int(self.coolant.operations.hydraulic_bc_type)  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"""flagSpecfield == {flagSpecfield}: still to be decided if 
            it is useful and if yes still to be defined\n"""
            )
            # inlet pressure.
            p_inl = flow_par[2]
            # inlet temperature.
            T_inl = flow_par[0]
            # outlet pressure.
            p_out = flow_par[3]
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = flow_par[1]
        
        # ALIAS
        inl_p_idx = self.inl_idx.pressure
        out_p_idx = self.out_idx.pressure
        main_d_idx = conductor.band.number_of_subdiagonals
        flow_dir = self.coolant.operations.flow_direction
        
        # Assign BC
        if flow_dir is FlowDirection.FORWARD:
            # p_inl
            sysmat[:,inl_p_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_p_idx.forward] = 1.0
            known[inl_p_idx.forward] = p_inl
            # p_out
            sysmat[:,out_p_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_p_idx.forward] = 1.0
            known[out_p_idx.forward] = p_out
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_fw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )
        elif flow_dir is FlowDirection.BACKWARD:
            # p_inl
            sysmat[:,inl_p_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_p_idx.backward] = 1.0
            known[inl_p_idx.backward] = p_inl
            # p_out
            sysmat[:,out_p_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_p_idx.backward] = 1.0
            known[out_p_idx.backward] = p_out
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_bw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )

        return known,sysmat

    def impose_inl_p_out_v(
        self,
        ndarrays:tuple,
        conductor:object,
        path:str,
        )->tuple:
        
        """Method that imposes a inlet pressure and outlet velocity as boundary conditions for the thermal hydraulic problem. Imposed quantities are:
            * inlet pressure
            * inlet temperature (or outlet temperature if backflow)
            * outlet velocity
        The method suitably accounts for forward (left to right)/backward (right to left) flow as well as for back flow (part of the fluid moves towards the inlet due to a pressure build up).

        Args:
            ndarrays (tuple): collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray on wich impose the boundary conditions.
            conductor (Conductor): object with all the information of the conductor.
            path (str): path of the auxiliary input file with the values for the boundary conditions (if INTIAL = -2).

        Returns:
            tuple: collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray with imposed boundary conditions.
        """

        # INLET AND OUTLET RESERVOIRS, INLET CONDITIONS AND FLOW SPECIFIED
        known,sysmat = ndarrays
        # Get boundary condition values.
        if (self.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY and not self.coolant.operations.bc_values_from_file):
            # outlet mass flow rate.
            mfr_out = self.coolant.operations.outlet_mass_rate
            # inlet pressure.
            p_inl = self.coolant.operations.inlet_pressure
            # inlet temperature.
            T_inl = self.coolant.operations.inlet_temperature
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = self.coolant.operations.outlet_temperature
        else:  # N.B va aggiustato per renderlo conforme caso positivo!
            # all values from flow_dummy.xlsx: call get_from_xlsx.
            [flow_par, flagSpecfield] = get_from_xlsx(
                conductor,
                path,
                self,
                "INTIAL",
                -int(self.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"""flagSpecfield == {flagSpecfield}: still to be decided if it 
            useful and if yes still to be defined\n"""
            )
            mfr_out = flow_par[3]  # inlet mass flow rate.
            p_inl = flow_par[2]  # inlet pressure.
            T_inl = flow_par[0]  # inlet temperature.
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = flow_par[1]

        # ALIAS
        inl_p_idx = self.inl_idx.pressure
        out_v_idx = self.out_idx.velocity
        main_d_idx = conductor.band.number_of_subdiagonals
        flow_dir = self.coolant.operations.flow_direction
        density = self.coolant.node_fields.total_density
        cross_section = self.channel.inputs.cross_section
        
        # Assign BC
        if flow_dir is FlowDirection.FORWARD:
            # Flow direction from x = 0 to x = L.
            # v_out
            sysmat[:,out_v_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_v_idx.forward] = 1.0
            known[out_v_idx.forward] = (mfr_out / density[-1] / cross_section)
            # p_inl
            sysmat[:,inl_p_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx, inl_p_idx.forward] = 1.0
            known[inl_p_idx.forward] = p_inl
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_fw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )
        elif flow_dir is FlowDirection.BACKWARD:
            # Flow direction from x = L to x = 0.
            # v_out
            sysmat[:,out_v_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,out_v_idx.backward] = 1.0
            known[out_v_idx.backward] = (mfr_out / density[0] / cross_section)
            # p_inl
            sysmat[:,inl_p_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx, inl_p_idx.backward] = 1.0
            known[inl_p_idx.backward] = p_inl
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_bw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )
        
        return known,sysmat

    def impose_inl_v_out_p(
        self,
        ndarrays:tuple,
        conductor:object,
        path:str,
        )->tuple:
        
        """Method that imposes a inlet velocity and outlet pressure as boundary conditions for the thermal hydraulic problem. Imposed quantities are:
            * inlet velocity
            * inlet temperature (or outlet temperature if backflow)
            * outlet pressure
        The method suitably accounts for forward (left to right)/backward (right to left) flow as well as for back flow (part of the fluid moves towards the inlet due to a pressure build up).

        Args:
            ndarrays (tuple): collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray on wich impose the boundary conditions.
            conductor (Conductor): object with all the information of the conductor.
            path (str): path of the auxiliary input file with the values for the boundary conditions (if INTIAL = -3).

        Returns:
            tuple: collection of known term vector (Known) and system matrix (SYSMAT) np.ndarray with imposed boundary conditions.
        """

        known,sysmat = ndarrays
        # Get boundary condition values.
        if (self.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE and not self.coolant.operations.bc_values_from_file):
            # outlet mass flow rate.
            mfr_inl = self.coolant.operations.inlet_mass_rate
            # outlet pressure.
            p_out = self.coolant.operations.outlet_pressure
            # inlet temperature.
            T_inl = self.coolant.operations.inlet_temperature
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = self.coolant.operations.outlet_temperature
        else:  # N.B va aggiustato per renderlo conforme caso positivo!
            # all values from flow_dummy.xlsx: call get_from_xlsx.
            [flow_par, flagSpecfield] = get_from_xlsx(
                conductor,
                path,
                self,
                "INTIAL",
                -int(self.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"""flagSpecfield == {flagSpecfield}: still to be decided if it 
            useful and if yes still to be defined\n"""
            )
            mfr_inl = flow_par[3]  # inlet mass flow rate.
            p_out = flow_par[2]  # outlet pressure.
            T_inl = flow_par[0]  # inlet temperature.
            # outlet temperature: to be assigned if outlet velocity is negative.
            T_out = flow_par[1]

        # ALIAS
        inl_v_idx = self.inl_idx.velocity
        out_p_idx = self.out_idx.pressure
        main_d_idx = conductor.band.number_of_subdiagonals
        flow_dir = self.coolant.operations.flow_direction
        density = self.coolant.node_fields.total_density
        cross_section = self.channel.inputs.cross_section
        
        # Assign BC
        if flow_dir is FlowDirection.FORWARD:
            # Flow direction from x = 0 to x = L.
            # v_inl
            sysmat[:,inl_v_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_v_idx.forward] = 1.0
            known[inl_v_idx.forward] = (mfr_inl / density[0] / cross_section)
            # p_out
            sysmat[:,out_p_idx.forward] = 0.0
            # main diagonal.
            sysmat[main_d_idx, out_p_idx.forward] = 1.0
            known[out_p_idx.forward] = p_out
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_fw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )
        elif flow_dir is FlowDirection.BACKWARD:
            # Flow direction from x = L to x = 0.
            # v_inl
            sysmat[:,inl_v_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx,inl_v_idx.backward] = 1.0
            known[inl_v_idx.backward] = (mfr_inl / density[-1] / cross_section)
            # p_out
            sysmat[:,out_p_idx.backward] = 0.0
            # main diagonal.
            sysmat[main_d_idx, out_p_idx.backward] = 1.0
            known[out_p_idx.backward] = p_out
            # Impose temperature bondary condition.
            known,sysmat = self.__impose_temperature_bw(
                sysmat,
                known,
                T_inl,
                T_out,
                main_d_idx,
            )
        
        return known,sysmat

# End class FluidComponent.
