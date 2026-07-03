import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from typing import Iterator, Union, NamedTuple

from components.fluid.fluid_component import FluidComponent
from conductor.conductor import Conductor
from conductor.conductor_flags import MethodFlag


@dataclass
class SystemMatrices:
    """Typed bundle of the four assembled (banded) system matrices that used
    to be passed around as a MASMAT/FLXMAT/DIFMAT/SORMAT dict. Iterating
    yields the matrices themselves (not copies), so in-place mutation by
    callers such as :func:`assemble_system_matrices` is preserved.
    """
    mass_capacity: np.ndarray   # was MASMAT
    flux_jacobian: np.ndarray   # was FLXMAT (convection)
    diffusion: np.ndarray       # was DIFMAT
    source_jacobian: np.ndarray # was SORMAT

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(
            (
                self.mass_capacity,
                self.flux_jacobian,
                self.diffusion,
                self.source_jacobian,
            )
        )


@dataclass
class GaussPointMatrices:
    """Typed bundle of the Gauss-point matrices of all elements (was the
    MMAT/AMAT/KMAT/SMAT dict); each array has shape
    (number_of_elements, degrees_of_freedom_per_node, degrees_of_freedom_per_node).
    """
    mass_capacity: np.ndarray   # was MMAT
    flux_jacobian: np.ndarray   # was AMAT
    diffusion: np.ndarray       # was KMAT
    source_jacobian: np.ndarray # was SMAT


@dataclass
class ElementMatrices:
    """Typed bundle of the element matrices of all elements (was the
    ELMMAT/ELAMAT/ELKMAT/ELSMAT dict); each array has shape
    (number_of_elements, degrees_of_freedom_per_element, degrees_of_freedom_per_element).
    """
    mass_capacity: np.ndarray   # was ELMMAT
    flux_jacobian: np.ndarray   # was ELAMAT
    diffusion: np.ndarray       # was ELKMAT
    source_jacobian: np.ndarray # was ELSMAT

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(
            (
                self.mass_capacity,
                self.flux_jacobian,
                self.diffusion,
                self.source_jacobian,
            )
        )

def array_initialization(shape:tuple, num_step:int)-> Union[NamedTuple,np.ndarray]:
    """Wrapper of function np.zeros that initializes an array of the given shape according to the time step number.
    N.B. the application of the theta method should be completely rivisited in the whole code.

    Args:
        shape (tuple): shape of the array to be initialized.
        num_step (int): time step number.

    Returns:
        Union[NamedTuple,np.ndarray]: namedtuple with arrays (previous, present) if num_step is 1; np.ndarray in all other cases.
    """

    if num_step == 1:
        Array = namedtuple("Array",("previous","present"))
        # To correctly apply the theta method (to be rivisited in the whole
        # code!).
        # previous is for the initialization (time step number is 0);
        # present is for the first time step after the initialization
        return Array(
            previous=np.zeros(shape),
            present=np.zeros(shape),
        )
    else:
        return np.zeros(shape)

def build_amat(
    matrix:np.ndarray,
    f_comp:FluidComponent,
    eq_idx:NamedTuple,
    )->np.ndarray:
    """Function that builds the A matrix (AMAT) at the Gauss point (flux Jacobian).

    Args:
        matrix (np.ndarray): initialized A matrix (np.zeros)
        f_comp (FluidComponent): fluid component object from which get all info to buld the coefficients.
        eq_idx (NamedTuple): collection of fluid equation index (velocity, pressure and temperaure equations).

    Returns:
        np.ndarray: matrix with updated elements.
    """
    density = f_comp.coolant.gauss_fields.total_density

    # Build array to assign diagonal coefficients.
    diag_idx = np.array(eq_idx)

    # Set diagonal elements at every Gauss point (exploit broadcasting).
    matrix[:, diag_idx, diag_idx] = f_comp.coolant.gauss_fields.velocity[
        :, None
    ]

    # Set off diagonal coefficients.
    # from velocity equation.
    matrix[:, eq_idx.velocity, eq_idx.pressure] = 1. / density
    # from pressure equation.
    matrix[:, eq_idx.pressure, eq_idx.velocity] = (
        density
        * f_comp.coolant.gauss_fields.total_speed_of_sound ** 2.
    )
    # from temperature equation.
    matrix[:, eq_idx.temperature, eq_idx.velocity] = (
        f_comp.coolant.gauss_fields.Gruneisen
        * f_comp.coolant.gauss_fields.temperature
    )

    return matrix

def build_kmat_fluid(
    matrix:np.ndarray,
    upweqt:np.ndarray,
    f_comp:FluidComponent,
    conductor:Conductor,
    )->np.ndarray:

    """Function that builds the K matrix (KMAT) at the Gauss point, UPWIND is included.
    UPWIND: differencing contribution a' la finite-difference, necessary to guarantee numarical stability that does not came from KMAT algebraic construction.

    Args:
        matrix (np.ndarray): initialized K matrix (np.zeros)
        upweqt (np.ndarray): array with the upwind numerical scheme.
        f_comp (FluidComponent): fluid component object from which get all info to buld the coefficients.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Fluid velocity at every Gauss point.
    velocity = np.abs(f_comp.coolant.gauss_fields.velocity)
    # Length of every element of the spatial discretization.
    delta_z = conductor.mesh.element_lengths
    # Collection of fluid equation index (velocity, pressure and temperaure
    # equations).
    eq_idx = conductor.equation_index[f_comp.identifier]

    # Build array to assign diagonal coefficients.
    diag_idx = np.array(eq_idx)

    # For the fluid equation this matrix has only the diagonal elements in the
    # velocity, pressure and temperature equations.
    # Diagonal therms definition: dz * upweqt * v / 2
    # Set diagonal elements at every Gauss point (exploit broadcasting).
    matrix[:, diag_idx, diag_idx] = (
        (delta_z * velocity / 2.0)[:, None] * upweqt[diag_idx][None, :]
    )

    return matrix

def build_elmmat(
    matrix:np.ndarray,
    mmat:np.ndarray,
    conductor:Conductor,
    alpha:float=0,
    )->np.ndarray:
    """Function that builds the mass and capacity matrix (ELMMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELM matrix
        mmat (np.ndarray): mass and capacity matrix MMAT after call to function build_mmat_solid.
        conductor (Conductor): object with all the information of the conductor.
        alpha (float, optional): Lumped/consistent mass parameter. Defaults to 0.

    Returns:
        np.ndarray: matrix with updated elements.
    """
    
    # Alias
    # Length of every element of the spatial discretization, broadcastable
    # against the (element, row, column) batch layout.
    dz = conductor.mesh.element_lengths[:, None, None]
    # Number of degrees of freedom, used to slice ELMMAT matrix.
    ndf = conductor.equation_counts.degrees_of_freedom_per_node
    # Twice the number of degrees of freedom, used to slice ELMMAT matrix.
    ndf2 = conductor.equation_counts.degrees_of_freedom_per_element

    # Build diagonal block of the matrix.
    diag_block = dz * (1. / 3. + alpha) * mmat
    # Build off diagonal block of the matrix.
    off_diag_block = dz * (1. / 6. - alpha) * mmat

    # COMPUTE THE MASS AND CAPACITY MATRIX
    # array smart
    matrix[:, :ndf, :ndf] = diag_block
    matrix[:, :ndf, ndf:ndf2] = off_diag_block
    matrix[:, ndf:ndf2, :ndf] = off_diag_block
    matrix[:, ndf:ndf2, ndf:ndf2] = diag_block

    return matrix

def build_elamat(
    matrix:np.ndarray,
    amat:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:
    """Function that builds the convection matrix (ELAMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELA matrix.
        amat (np.ndarray): matrix AMAT after call to function build_amat.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Number of degrees of freedom, used to slice ELAMAT matrix.
    ndf = conductor.equation_counts.degrees_of_freedom_per_node
    # Twice the number of degrees of freedom, used to slice ELAMAT matrix.
    ndf2 = conductor.equation_counts.degrees_of_freedom_per_element

    block = amat / 2.

    matrix[:, :ndf, :ndf] = -block
    matrix[:, :ndf, ndf:ndf2] = block
    matrix[:, ndf:ndf2, :ndf] = -block
    matrix[:, ndf:ndf2, ndf:ndf2] = block

    return matrix

def build_elkmat(
    matrix:np.ndarray,
    kmat:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:
    """
    Function that builds the diffusion matrix (ELKMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELK matrix.
        kmat (np.ndarray): mass and capacity matrix KMAT after call to function build_kmat_solid.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Length of every element of the spatial discretization, broadcastable
    # against the (element, row, column) batch layout.
    dz = conductor.mesh.element_lengths[:, None, None]
    # Number of degrees of freedom, used to slice ELKMAT matrix.
    ndf = conductor.equation_counts.degrees_of_freedom_per_node
    # Twice the number of degrees of freedom, used to slice ELKMAT matrix.
    ndf2 = conductor.equation_counts.degrees_of_freedom_per_element

    block = kmat / dz

    # COMPUTE THE DIFFUSION MATRIX
    # array smart
    matrix[:, :ndf, :ndf] = block
    matrix[:, :ndf, ndf:ndf2] = - block
    matrix[:, ndf:ndf2, :ndf] = - block
    matrix[:, ndf:ndf2, ndf:ndf2] = block

    return matrix

def build_elsmat(
    matrix:np.ndarray,
    smat:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:
    """
    Function that builds the source matrix (ELSMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELS matrix.
        smat (np.ndarray): source matrix SMAT after call to function build_smat_env_solid_interface.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Length of every element of the spatial discretization, broadcastable
    # against the (element, row, column) batch layout.
    dz = conductor.mesh.element_lengths[:, None, None]
    # Number of degrees of freedom, used to slice ELSMAT matrix.
    ndf = conductor.equation_counts.degrees_of_freedom_per_node
    # Twice the number of degrees of freedom, used to slice ELSMAT matrix.
    ndf2 = conductor.equation_counts.degrees_of_freedom_per_element

    diag_block = smat * dz / 3.
    off_diag_block = diag_block / 2.

    # COMPUTE THE SOURCE MATRIX
    # array smart
    matrix[:, :ndf, :ndf] = diag_block
    matrix[:, :ndf, ndf:ndf2] = off_diag_block
    matrix[:, ndf:ndf2, :ndf] = off_diag_block
    matrix[:, ndf:ndf2, ndf:ndf2] = diag_block

    return matrix

def build_elslod(array:np.ndarray,
    svec:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:
    """
    Function that builds the source matrix (ELSLOD) at the Gauss point exploiting slicing.

    Args:
        array (np.ndarray): Initialized ELSLOD array.
        svec (np.ndarray): source array SVEC after call to function build_svec_env_jacket_interface.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: array with updated elements.
    """

    # Alias
    # Length of every element of the spatial discretization, broadcastable
    # against the (element, degree-of-freedom) batch layout.
    dz_6 = conductor.mesh.element_lengths[:, None] / 6.
    # Number of degrees of freedom, used to slice ELSLOD array.
    ndf = conductor.equation_counts.degrees_of_freedom_per_node
    # Twice the number of degrees of freedom, used to slice ELSLOD array.
    ndf2 = conductor.equation_counts.degrees_of_freedom_per_element

    # COMPUTE THE SOURCE VECTOR (ANALYTIC INTEGRATION)
    # array smart
    # This is independent from the solution method thanks to the escamotage of
    # the dummy steady state corresponding to the initialization.
    if conductor.cond_num_step == 1:
        # To correctly apply the theta method (to be rivisited in the whole
        # code!).
        # Current time step
        array.present[:, :ndf] = 2. * svec.present[:,:,0] + svec.present[:,:,1]
        array.present[:, ndf:ndf2] = svec.present[:,:,0] + 2. * svec.present[:,:,1]
        # Not use simply array.present: raises AttributeError
        array.present[:] *= dz_6
        # Previous time step
        array.previous[:, :ndf] = 2. * svec.previous[:,:,0] + svec.previous[:,:,1]
        array.previous[:, ndf:ndf2] = svec.previous[:,:,0] + 2. * svec.previous[:,:,1]
        # Not use simply array.previous: raises AttributeError
        array.previous[:] *= dz_6
    else:
        # Compute only at the current time step
        array[:, :ndf] = 2. * svec[:,:,0] + svec[:,:,1]
        array[:, ndf:ndf2] = svec[:,:,0] + 2. * svec[:,:,1]
        array *= dz_6

    return array

def assemble_system_matrices(
    fin_mat:SystemMatrices,
    element_matrices:"ElementMatrices",
    conductor:Conductor,
    )->SystemMatrices:
    """Function that assembles matrices MASMAT, FLXMAT, DIFMAT and SORMAT from the per-element matrices ELMMAT, ELAMAT, ELKMAT and ELSMAT of all elements at once. The matrices match is as follows:
        * ELMMAT builds MASMAT
        * ELAMAT builds FLXMAT
        * ELKMAT builds DIFMAT
        * ELSMAT builds SORMAT

    Element entry (local_row, local_col) of element e goes to band storage
    entry (half - 1 - local_row + local_col, NODOFS*e + local_row). For a
    fixed pair of local indices the destination columns are distinct across
    elements, so the addition is collision free and can be done for all
    elements with a single vectorized statement per local index pair (the
    overlap between consecutive elements only mixes different local pairs,
    which are separate statements).

    Args:
        fin_mat (SystemMatrices): collection of matrices MASMAT, FLXMAT, DIFMAT and SORMAT.
        element_matrices (ElementMatrices): per-element matrices of all elements.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        SystemMatrices: collection of matrices MASMAT, FLXMAT, DIFMAT and SORMAT with updated elements.
    """

    # Alias
    half = conductor.band.half_bandwidth

    first_columns = conductor.equation_counts.degrees_of_freedom_per_node * np.arange(
        conductor.mesh.number_of_elements
    )

    for fmat, batch in zip(fin_mat, element_matrices):
        for local_row in range(half):
            columns = first_columns + local_row
            for local_col in range(half):
                fmat[half - 1 - local_row + local_col, columns] += batch[
                    :, local_row, local_col
                ]

    return fin_mat

def assemble_syslod(
    array:np.ndarray,
    conductor:Conductor,
)->np.ndarray:
    """Function that assembles the source term vector load_vector from the per-element load vectors of all elements at once.

    Args:
        array (np.ndarray): per-element load vectors of shape (number_of_elements, NODOFS2) after call to function build_elslod (namedtuple of two such arrays at the first time step).
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: the updated load_vector array.
    """

    # Alias
    method = conductor.inputs.thermohydraulic_method
    num_step = conductor.cond_num_step
    half = conductor.band.half_bandwidth
    load_vector = conductor.time_integration.load_vector

    # Scatter the per-element load vectors of all elements at once: entry
    # local_dof of element e goes to system row NODOFS*e + local_dof. For a
    # fixed local_dof the destination rows are distinct across elements, so
    # the addition is collision free (the overlap between consecutive
    # elements only mixes different local dofs, which are separate
    # statements).
    first_rows = conductor.equation_counts.degrees_of_freedom_per_node * np.arange(
        conductor.mesh.number_of_elements
    )

    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson
        if num_step == 1:
            # Construct the load vector columns
            for local_dof in range(half):
                rows = first_rows + local_dof
                # Current time step
                load_vector[rows, 0] += array.present[:, local_dof]
                # Previous time step
                load_vector[rows, 1] += array.previous[:, local_dof]
        else:
            # Update only the first column, that correspond to the current time
            # step
            for local_dof in range(half):
                load_vector[first_rows + local_dof, 0] += array[:, local_dof]
    elif method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Adams-Moulton order 4
        # The implementation of higher order numerical schemes for time
        # integration should be completely reviewed!
        raise NotImplementedError(
            "The Adams-Moulton 4 load-vector assembly must be reviewed "
            "before it can be vectorized: the legacy per-element "
            "implementation shifted the columns of overlapping element "
            "slices more than once."
        )
    # end conductor.inputs["METHOD"]

    return load_vector

def eval_system_matrix(
    matrix:np.ndarray,
    aux_matrices:SystemMatrices,
    conductor:Conductor,
    )->np.ndarray:
    """Function that evaluates the system matrix using the values of the matrix MASMAT, FLXMAT, DIFMAT and SORMAT, according to the selected method for time integration.

    Args:
        matrix (np.ndarray): initialized SYSMAT matrix
        aux_matrices (SystemMatrices): collection of matrix MASMAT, FLXMAT, DIFMAT and SORMAT after call to function assemble_system_matrices.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    method = conductor.inputs.thermohydraulic_method
    # Unpack auxiliary matrices (mass capacity, flux Jacobian, diffusion,
    # source Jacobian).
    mass_capacity, flux_jacobian, diffusion, source_jacobian = aux_matrices
    # ** COMPUTE SYSTEM MATRIX **
    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson
        matrix = (
            mass_capacity / conductor.time_step
            + conductor.theta_method * (flux_jacobian + diffusion + source_jacobian)
        )
        
    elif method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Adams-Moulton order 4
        # Alias
        am4_aa = conductor.time_integration.adams_moulton_matrices # shallow copy
        if conductor.cond_num_step == 1:
            # This is due to the dummy initial steady state
            for cc in range(am4_aa.shape[2]):
                am4_aa[cc,:,:] = flux_jacobian + diffusion + source_jacobian
        else:
            # Shift the matrices by one towards right and compute the new first
            # matrix at the current time step.
            am4_aa[1:4,:,:] = am4_aa[0:3,:,:]
            am4_aa[0,:,:] = flux_jacobian + diffusion + source_jacobian
        # compute SYSMAT
        matrix = mass_capacity / conductor.time_step + 9. / 24. * am4_aa[0,:,:]
    
    return matrix
