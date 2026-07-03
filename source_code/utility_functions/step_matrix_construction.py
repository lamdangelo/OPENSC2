import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from typing import Iterator, Union, NamedTuple

from components.fluid.fluid_component import FluidComponent
from conductor.conductor import Conductor
from conductor.conductor_flags import MethodFlag


@dataclass
class SystemMatrices:
    """Typed bundle of the four assembled system matrices (mass/capacity,
    convection/flux, diffusion, source) that used to be passed around as a
    MASMAT/FLXMAT/DIFMAT/SORMAT dict. Iterating yields the matrices
    themselves (not copies), so in-place mutation by callers such as
    :func:`assemble_matrix` is preserved.
    """
    masmat: np.ndarray
    flxmat: np.ndarray
    difmat: np.ndarray
    sormat: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.masmat, self.flxmat, self.difmat, self.sormat))

def matrix_initialization(row:int,col:int,matrix_names:tuple)->dict:
    """Wrapper of function np.zeros that inizializes five identical rectangular matrices and collects them in a dictionary.

    Args:
        row (int): number of rows of the matrix.
        col (int): number of columns of the matrix.
        matrix_names (tuple): collection of valid keywords to build the dictionary of initialized matrices.

    Returns:
        dict: collection of initialized matrices.
    """
    
    return {name:np.zeros((row,col)) for name in matrix_names}

def array_initialization(dimension:int, num_step:int, col:int=0)-> Union[NamedTuple,np.ndarray]:
    """Wrapper of function np.zeros that initializes array of shape (dimension, col) according to the time step number.
    N.B. the application of the theta method should be completely rivisited in the whole code.

    Args:
        dimension (int): number of elements (rows) of the array to be initialized.
        num_step (int): time step number.
        col (int, optional): number of columns to be assigned to the array. If col is 0, the array shape is (dimension,), else array shape is (dimension,col). Defaults to 0.

    Returns:
        Union[NamedTuple,np.ndarray]: namedtuple with array if num_step is 1; np.ndarray in all other cases.
    """

    if num_step == 1:
        Array = namedtuple("Array",("previous","present"))
        # To correctly apply the theta method (to be rivisited in the whole 
        # code!).
        # previous is for the initialization (time step number is 0);
        # present is for the first time step after the initialization
        
        # Check on col to assign the correct shape to the array.
        if col: # used to define SVEC.
            return Array(
                previous=np.zeros((dimension, col)),
                present=np.zeros((dimension, col)),
            )
        else: # used to define ELSLOD.
            return Array(
                previous=np.zeros(dimension),
                present=np.zeros(dimension),
            )
    else:
        # Check on col to assign the correct shape to the array.
        if col: # used to define SVEC.
            return np.zeros((dimension, col))
        else: # used to define ELSLOD.
            return np.zeros(dimension)

def build_amat(
    matrix:np.ndarray,
    f_comp:FluidComponent,
    elem_idx:int,
    eq_idx:NamedTuple,
    )->np.ndarray:
    """Function that builds the A matrix (AMAT) at the Gauss point (flux Jacobian).

    Args:
        matrix (np.ndarray): initialized A matrix (np.zeros)
        f_comp (FluidComponent): fluid component object from which get all info to buld the coefficients.
        elem_idx (int): index of the i-th element of the spatial discretization.
        eq_idx (NamedTuple): collection of fluid equation index (velocity, pressure and temperaure equations).

    Returns:
        np.ndarray: matrix with updated elements.
    """
    density = f_comp.coolant.gauss_fields.total_density[elem_idx]
    
    # Build array to assign diagonal coefficients.
    diag_idx = np.array(eq_idx)
    
    # Set diagonal elements (exploit broadcasting).
    matrix[diag_idx,diag_idx] = f_comp.coolant.gauss_fields.velocity[elem_idx]

    # Set off diagonal coefficients.
    # from velocity equation.
    matrix[eq_idx.velocity, eq_idx.pressure] = 1. / density
    # from pressure equation.
    matrix[eq_idx.pressure, eq_idx.velocity] = (
        density
        * f_comp.coolant.gauss_fields.total_speed_of_sound[elem_idx] ** 2.
    )
    # from temperature equation.
    matrix[eq_idx.temperature, eq_idx.velocity] = (
        f_comp.coolant.gauss_fields.Gruneisen[elem_idx]
        * f_comp.coolant.gauss_fields.temperature[elem_idx]
    )

    return matrix

def build_kmat_fluid(
    matrix:np.ndarray,
    upweqt:np.ndarray,
    f_comp:FluidComponent,
    conductor:Conductor,
    elem_idx:int,
    )->np.ndarray:

    """Function that builds the K matrix (KMAT) at the Gauss point, UPWIND is included.
    UPWIND: differencing contribution a' la finite-difference, necessary to guarantee numarical stability that does not came from KMAT algebraic construction.

    Args:
        matrix (np.ndarray): initialized K matrix (np.zeros)
        upweqt (np.ndarray): array with the upwind numerical scheme.
        f_comp (FluidComponent): fluid component object from which get all info to buld the coefficients.
        conductor (Conductor): object with all the information of the conductor.
        elem_idx (int): index of the i-th element of the spatial discretization.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Fluid velocity at present gauss point index.
    velocity = np.abs(f_comp.coolant.gauss_fields.velocity[elem_idx])
    # Length of the interval that includes the present gauss points.
    delta_z = conductor.mesh.element_lengths[elem_idx]
    # Collection of fluid equation index (velocity, pressure and temperaure 
    # equations).
    eq_idx = conductor.equation_index[f_comp.identifier]

    # Build array to assign diagonal coefficients.
    diag_idx = np.array(eq_idx)

    # For the fluid equation this matrix has only the diagonal elements in the 
    # velocity, pressure and temperature equations.
    # Diagonal therms definition: dz * upweqt * v / 2
    # Set diagonal elements (exploit broadcasting).
    matrix[diag_idx,diag_idx] = delta_z * upweqt[diag_idx] * velocity / 2.0

    return matrix

def build_elmmat(
    matrix:np.ndarray,
    mmat:np.ndarray,
    conductor:Conductor,
    elem_idx:int,
    alpha:float=0,
    )->np.ndarray:
    """Function that builds the mass and capacity matrix (ELMMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELM matrix
        mmat (np.ndarray): mass and capacity matrix MMAT after call to function build_mmat_solid.
        conductor (Conductor): object with all the information of the conductor.
        elem_idx (int): index of the i-th element of the spatial discretization.
        alpha (float, optional): Lumped/consistent mass parameter. Defaults to 0.

    Returns:
        np.ndarray: matrix with updated elements.
    """
    
    # Alias
    # Length of the present element of the spatial discretization.
    dz = conductor.mesh.element_lengths[elem_idx]
    # Number of degrees of freedom, used to slice ELMMAT matrix.
    ndf = conductor.dict_N_equation["NODOFS"]
    # Twice the number of degrees of freedom, used to slice ELMMAT matrix.
    ndf2 = conductor.dict_N_equation["NODOFS2"]

    # Build diagonal block of the matrix.
    diag_block = dz * (1. / 3. + alpha) * mmat
    # Build off diagonal block of the matrix.
    off_diag_block = dz * (1. / 6. - alpha) * mmat
    
    # COMPUTE THE MASS AND CAPACITY MATRIX
    # array smart
    matrix[:ndf,:ndf] = diag_block
    matrix[:ndf,ndf:ndf2] = off_diag_block
    matrix[ndf:ndf2,:ndf] = off_diag_block
    matrix[ndf:ndf2,ndf:ndf2] = diag_block

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
    ndf = conductor.dict_N_equation["NODOFS"]
    # Twice the number of degrees of freedom, used to slice ELAMAT matrix.
    ndf2 = conductor.dict_N_equation["NODOFS2"]

    block = amat / 2.

    matrix[:ndf,:ndf] = -block
    matrix[:ndf,ndf:ndf2] = block
    matrix[ndf:ndf2,:ndf] = -block
    matrix[ndf:ndf2,ndf:ndf2] = block

    return matrix

def build_elkmat(
    matrix:np.ndarray,
    kmat:np.ndarray,
    conductor:Conductor,
    elem_idx:int,
    )->np.ndarray:
    """
    Function that builds the diffusion matrix (ELKMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELK matrix.
        kmat (np.ndarray): mass and capacity matrix KMAT after call to function build_kmat_solid.
        conductor (Conductor): object with all the information of the conductor.
        elem_idx (int): index of the i-th element of the spatial discretization.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Length of the present element of the spatial discretization.
    dz = conductor.mesh.element_lengths[elem_idx]
    # Number of degrees of freedom, used to slice ELKMAT matrix.
    ndf = conductor.dict_N_equation["NODOFS"]
    # Twice the number of degrees of freedom, used to slice ELKMAT matrix.
    ndf2 = conductor.dict_N_equation["NODOFS2"]

    block = kmat / dz
    
    # COMPUTE THE DIFFUSION MATRIX
    # array smart
    matrix[:ndf,:ndf] = block
    matrix[:ndf,ndf:ndf2] = - block
    matrix[ndf:ndf2,:ndf] = - block
    matrix[ndf:ndf2,ndf:ndf2] = block

    return matrix

def build_elsmat(
    matrix:np.ndarray,
    smat:np.ndarray,
    conductor:Conductor,
    elem_idx:int,
    )->np.ndarray:
    """
    Function that builds the source matrix (ELSMAT) at the Gauss point exploiting slicing.

    Args:
        matrix (np.ndarray): Initialized ELS matrix.
        smat (np.ndarray): source matrix SMAT after call to function build_smat_env_solid_interface.
        conductor (Conductor): object with all the information of the conductor.
        elem_idx (int): index of the i-th element of the spatial discretization.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    # Length of the present element of the spatial discretization.
    dz = conductor.mesh.element_lengths[elem_idx]
    # Number of degrees of freedom, used to slice ELSMAT matrix.
    ndf = conductor.dict_N_equation["NODOFS"]
    # Twice the number of degrees of freedom, used to slice ELSMAT matrix.
    ndf2 = conductor.dict_N_equation["NODOFS2"]

    diag_block = smat * dz / 3.
    off_diag_block = diag_block / 2.

    # COMPUTE THE SOURCE MATRIX
    # array smart
    matrix[:ndf,:ndf] = diag_block
    matrix[:ndf,ndf:ndf2] = off_diag_block
    matrix[ndf:ndf2,:ndf] = off_diag_block
    matrix[ndf:ndf2,ndf:ndf2] = diag_block

    return matrix

def build_elslod(array:np.ndarray,
    svec:np.ndarray,
    conductor:Conductor,
    elem_idx:int,
    )->np.ndarray:
    """
    Function that builds the source matrix (ELSLOD) at the Gauss point exploiting slicing.

    Args:
        array (np.ndarray): Initialized ELSLOD array.
        svec (np.ndarray): source array SVEC after call to function build_svec_env_jacket_interface.
        conductor (Conductor): object with all the information of the conductor.
        elem_idx (int): index of the i-th element of the spatial discretization.

    Returns:
        np.ndarray: array with updated elements.
    """

    # Alias
    # Length of the present element of the spatial discretization.
    dz = conductor.mesh.element_lengths[elem_idx]
    # Number of degrees of freedom, used to slice ELSMAT matrix.
    ndf = conductor.dict_N_equation["NODOFS"]
    # Twice the number of degrees of freedom, used to slice ELSMAT matrix.
    ndf2 = conductor.dict_N_equation["NODOFS2"]

    dz_6 = dz / 6.

    # COMPUTE THE SOURCE VECTOR (ANALYTIC INTEGRATION)
    # array smart
    # This is independent from the solution method thanks to the escamotage of
    # the dummy steady state corresponding to the initialization.
    if conductor.cond_num_step == 1:
        # To correctly apply the theta method (to be rivisited in the whole 
        # code!).
        # Current time step
        array.present[:ndf] = 2. * svec.present[:,0] + svec.present[:,1]
        array.present[ndf:ndf2] = svec.present[:,0] + 2. * svec.present[:,1]
        # Not use simply array.present: raises AttributeError
        array.present[:] *= dz_6
        # Previous time step
        array.previous[:ndf] = 2. * svec.previous[:,0] + svec.previous[:,1]
        array.previous[ndf:ndf2] = svec.previous[:,0] + 2. * svec.previous[:,1]
        # Not use simply array.previous: raises AttributeError
        array.previous[:] *= dz_6
    else:
        # Compute only at the current time step
        array[:ndf] = 2. * svec[:,0] + svec[:,1]
        array[ndf:ndf2] = svec[:,0] + 2. * svec[:,1]
        array *= dz_6
    
    return array

def assemble_matrix(
    fin_mat:SystemMatrices,
    el_mat:dict,
    conductor:Conductor,
    jump_idx:int,
    )->SystemMatrices:
    """Function that assembles matrices MASMAT, FLXMAT, DIFMAT and SORMAT starting from the values of matrices ELMMAT, ELAMAT, ELKMAT and ELSMAT respectively. The matrices match is as follows:
        * ELMMAT builds MATMAT
        * ELAMAT builds FLXMAT
        * ELKMAT builds DIFMAT
        * ELSMAT builds SORMAT

    Args:
        fin_mat (SystemMatrices): collection of matrices MASMAT, FLXMAT, DIFMAT and SORMAT.
        el_mat (dict): collection of matrices ELMMAT, ELAMAT, ELKMAT and ELSMAT.
        conductor (Conductor): object with all the information of the conductor.
        jump_idx (int): index to jump over NODOFS * elem_idx position in the big matrices.

    Returns:
        SystemMatrices: collection of matrices MASMAT, FLXMAT, DIFMAT and SORMAT with updated elements.
    """
    
    # Alias
    half = conductor.dict_band["Half"]
    full = conductor.dict_band["Full"]

    # ASSEMBLE THE MATRICES AND THE LOAD VECTOR
    # array smart
    for hbw_idx in range(half):
        # hbw_idx: half band widht index.
        # Lower bound for row slicing.
        row_lb = half - hbw_idx - 1
        # Upper bound for row slicing.
        row_ub = full - hbw_idx
        # Column index.
        col = jump_idx + hbw_idx
        # Loop to update matrices masmat, flxmat, difmat and sormat stored in 
        # dict fin_mat exploiting matrices elmmat, elamat, elkmat and 
        # elsmat respectively, stored in dict el_mat.
        for fmat, elmat in zip(fin_mat,el_mat.values()):
            # Sum elements from row_lb to row_ub of column col in big_matrix 
            # with all the elements in the row hbw_idx of small_matrix.
            fmat[row_lb:row_ub,col] += elmat[hbw_idx,:]
    
    return fin_mat

def assemble_syslod(
    array:np.ndarray,
    conductor:Conductor,
    jump_idx:int,
)->np.ndarray:
    """Function that assembles the source term vector syslod exploiting the information stored in array ELSLOD.

    Args:
        array (np.ndarray): ELSLOD array after call to funciton build_elslod.
        conductor (Conductor): object with all the information of the conductor.
        jump_idx (int): index to jump over NODOFS * elem_idx position in syslod array, used to slice syslod.

    Returns:
        np.ndarray: array with updated elements (the sliced updated part f syslod).
    """

    # Alias
    method = conductor.inputs.thermohydraulic_method
    num_step = conductor.cond_num_step
    half = conductor.dict_band["Half"]
    syslod = conductor.dict_Step["SYSLOD"][jump_idx:jump_idx + half,:]
    
    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson
        if num_step == 1:
            # Construct key SYSLOD of dictionary dict_Step
            # Current time step
            syslod[:,0] += array.present
            # Previous time step
            syslod[:,1] += array.previous
        else:
            # Update only the first column, that correspond to the current time
            # step
            syslod[:,0] += array
    elif method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Adams-Moulton order 4
        # The implementation of higher order numerical schemes for time 
        # integration should be completely reviewed!
        if num_step == 1:
            # Construct key SYSLOD of dictionary dict_Step
            # Current time step
            syslod[:,0] += array.present
            # This loop should be checked carefully!
            for cc in range(syslod.shape[1]):
                # Dummy initial steady state
                syslod[:,cc] += array.previous
        else:
            # Shift the colums by one towards right and compute the new first 
            # column at the current time step
            syslod[:,1:] = syslod[:,:3].copy()
            syslod[:,0] += array
        # end if conductor.cond_num_step
    # end conductor.inputs["METHOD"]

    return syslod

def eval_system_matrix(
    matrix:np.ndarray,
    aux_matrices:SystemMatrices,
    conductor:Conductor,
    )->np.ndarray:
    """Function that evaluates the system matrix using the values of the matrix MASMAT, FLXMAT, DIFMAT and SORMAT, according to the selected method for time integration.

    Args:
        matrix (np.ndarray): initialized SYSMAT matrix
        aux_matrices (SystemMatrices): collection of matrix MASMAT, FLXMAT, DIFMAT and SORMAT after call to function assemble_matrix.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias
    method = conductor.inputs.thermohydraulic_method
    # Unpack auxiliary matrices (MASMAT,FLXMAT,DIFMAT,SORMAT)
    masmat,flxmat,difmat,sormat = aux_matrices
    # ** COMPUTE SYSTEM MATRIX **
    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson
        matrix = (
            masmat / conductor.time_step
            + conductor.theta_method * (flxmat + difmat + sormat)
        )
        
    elif method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Adams-Moulton order 4
        # Alias
        am4_aa = conductor.dict_Step["AM4_AA"] # shallow copy
        if conductor.cond_num_step == 1:
            # This is due to the dummy initial steady state
            for cc in range(am4_aa.shape[2]):
                am4_aa[cc,:,:] = flxmat + difmat + sormat
        else:
            # Shift the matrices by one towards right and compute the new first
            # matrix at the current time step.
            am4_aa[1:4,:,:] = am4_aa[0:3,:,:]
            am4_aa[0,:,:] = flxmat + difmat + sormat
        # compute SYSMAT
        matrix = masmat / conductor.time_step + 9. / 24. * am4_aa[0,:,:]
    
    return matrix
