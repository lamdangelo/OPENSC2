"""
This module contains the ConductorMesh class, which is responsible for storing the mesh
data for the conductor simulation.
"""

import logging
from decimal import Decimal
from enum import Enum, auto
import numpy as np
import pandas as pd

import conductor.input_validator as validator
from components.solid.strand_component import StrandComponent

logger_discretization = logging.getLogger("opensc2Logger.discretization")


class MeshType(Enum):
    """
    An enumeration class to define the type of mesh used for the conductor simulation.
    """
    UNIFORM = auto()  # fixed and uniform
    REFINED = auto()  # fixed and refined zone
    ADAPTED = auto()  # fixed and adapted with initial refinement
    FROM_FILE = auto()  # read from file


    @staticmethod
    def get_mesh_type(mesh_type_flag: int):
        """Convert mesh type flag from the input file to MeshType enum."""
        if mesh_type_flag == 0:
            return MeshType.UNIFORM
        elif mesh_type_flag == 1:
            return MeshType.REFINED
        elif mesh_type_flag == 3:
            return MeshType.ADAPTED
        elif mesh_type_flag == -1:
            return MeshType.FROM_FILE
        else:
            raise ValueError(f"Invalid mesh type flag: {mesh_type_flag}")


class ConductorMesh:
    
    def __init__(self, conductor_length: float, grid_input: dict):
        # --- raw input parameters ---
        self.conductor_length = conductor_length
        self.number_of_elements = grid_input["NELEMS"]
        self.mesh_type = MeshType.get_mesh_type(grid_input["ITYMSH"])
        self.number_of_refined_elements = grid_input["NELREF"]
        self.start_refined_zone = grid_input["XBREFI"]
        self.end_refined_zone = grid_input["XEREFI"]
        self.minimum_element_size = grid_input["SIZMIN"]
        self.maximum_element_size = grid_input["SIZMAX"]
        self.increase_ratio_left = grid_input["DXINCRE_LEFT"]
        self.increase_ratio_right = grid_input["DXINCRE_RIGHT"]
        self.maximum_number_of_nodes = grid_input["MAXNOD"]

        # --- computed features (populated after spatial discretization) ---
        self.number_of_nodes = self.number_of_elements + 1
        self.node_coordinates = self._initialize_node_coordinates()  # z-coordinate of each node
        self.compute_derived_features()


    def compute_derived_features(self) -> None:
        """(Re-)compute the mesh features derived from the node coordinates.

        Called at construction and again whenever the node coordinates change
        (e.g. a user-defined grid loaded after construction, or an adapted
        mesh).
        """
        self.element_lengths = self._compute_element_lengths()  # length of each element (delta_z)
        self.max_element_length = max(self.element_lengths)
        self.min_element_length = min(self.element_lengths)
        self.gauss_point_coordinates = self._compute_gauss_points()  # z-coordinate of each Gauss points
        self.dual_element_lengths = self._compute_dual_element_lengths()  # dual-mesh lengths centred on nodes (delta_z_tilde)
        self.position_precision = self._compute_position_precision(str(self.min_element_length))  # significant figures in min_element_length, used for coordinate rounding


    def _initialize_node_coordinates(self) -> np.ndarray:
        if self.mesh_type is not MeshType.FROM_FILE:
            return self.build_node_coordinates()
        else:
            return self.set_user_defined_grid() # TODO
        

    # -------------------------------------------------------------------------
    # Derived feature computation
    # -------------------------------------------------------------------------

    def _compute_position_precision(self, numstr: str):
        """Compute the number of significant figures of a number string and store
        the result in position_precision.

        N.B: quite general but does not work for all cases; see
        https://stackoverflow.com/questions/8101353/counting-significant-figures-in-python
        """
        decimal_norm = Decimal(numstr).normalize()
        if str(decimal_norm) == "0":
            return 0
        else:
            return len(Decimal(numstr).as_tuple().digits)


    def _compute_element_lengths(self) -> np.ndarray:
        """
        Compute the element lengths of the mesh.
        """
        return self.node_coordinates[1:] - self.node_coordinates[:-1]


    def _compute_gauss_points(self) -> np.ndarray:
        """
        Compute Gauss points coordinates.
        """
        return (self.node_coordinates[:-1] + self.node_coordinates[1:]) / 2


    def _compute_dual_element_lengths(self) -> np.ndarray:
        """
        Computes the lengths of the dual elements, i.e., the elements whose 
        nodes lie in the center of the (primal) mesh elements.
        """
        dual_element_lengths = np.zeros(self.number_of_nodes)
        dual_element_lengths[0] = (
            self.node_coordinates[1] - self.node_coordinates[0]
        ) / 2
        dual_element_lengths[1:-1] = (
            self.node_coordinates[2:] - self.node_coordinates[:-2]
        ) / 2
        dual_element_lengths[-1] = (
            self.node_coordinates[-1] - self.node_coordinates[-2]
        ) / 2
        return dual_element_lengths 

    # -------------------------------------------------------------------------
    # z-coordinate generation
    # -------------------------------------------------------------------------

    def _build_uniform_mesh(self, length: float) -> np.ndarray:
        """Return a uniform z-coordinate array of length self.number_of_nodes.

        Args:
            length: total conductor length (z goes from 0 to length).

        Returns:
            1-D array of evenly spaced coordinates from 0 to length.
        """
        return np.linspace(0.0, length, self.number_of_nodes)


    def _build_refined_mesh(self, length: float) -> np.ndarray:
        """Return a z-coordinate array with a locally refined zone.

        The refined zone spans [self.start_refined_zone, self.end_refined_zone].
        Outside that zone the mesh coarsens gradually according to
        self.increase_ratio_left / self.increase_ratio_right, then becomes uniform.

        Assumption: the boundary nodes of the refined zone belong to that zone,
        so the coarse regions contain exactly n_elem_left and n_elem_right elements.

        Args:
            length: total conductor length.

        Raises:
            ValueError: if the pitch in the refined zone is outside [SIZMIN, SIZMAX].
            ValueError: if the coarsening cannot fit within the available elements.

        Returns:
            1-D array of node coordinates of length self.number_of_nodes.
        """
        node_coordinates = np.zeros(self.number_of_nodes)

        n_elem_coarse = self.number_of_elements - self.number_of_refined_elements
        n_elem_left = round(
            (self.start_refined_zone - 0.0)
            / (length - (self.end_refined_zone - self.start_refined_zone))
            * n_elem_coarse
        )
        n_elem_right = n_elem_coarse - n_elem_left

        n_nodes_refined_zone = self.number_of_refined_elements + 1

        refined_zone_pitch = (
            (self.end_refined_zone - self.start_refined_zone)
            / self.number_of_refined_elements
        )

        if refined_zone_pitch < self.minimum_element_size:
            raise ValueError(
                f"ERROR: {refined_zone_pitch=} m in refined zone "
                f"< {self.minimum_element_size=} m!!!\n"
            )
        if refined_zone_pitch > self.maximum_element_size:
            raise ValueError(
                f"ERROR: {refined_zone_pitch=} m in refined zone "
                f"> {self.maximum_element_size} m!!!\n"
            )

        node_coordinates[n_elem_left : n_elem_left + n_nodes_refined_zone] = np.linspace(
            self.start_refined_zone, self.end_refined_zone, n_nodes_refined_zone
        )

        # --- Left coarsening region ---
        coarsening_steps, last_pitch = self._coarsen_from_boundary(
            node_coordinates,
            boundary_idx=n_elem_left,
            boundary_position=self.start_refined_zone,
            n_elem=n_elem_left,
            region_length=self.start_refined_zone,
            starting_pitch=refined_zone_pitch,
            increase_ratio=self.increase_ratio_left,
            forward=False,
        )
        if n_elem_left > 0:
            if coarsening_steps < n_elem_left - 1:
                node_coordinates[: n_elem_left + 1 - coarsening_steps] = np.linspace(
                    0.0,
                    node_coordinates[n_elem_left - coarsening_steps],
                    n_elem_left + 1 - coarsening_steps,
                )
                validator.check_coarse_region_quality(
                    last_pitch=last_pitch,
                    n_elem=n_elem_left - coarsening_steps,
                    region_length=node_coordinates[n_elem_left - coarsening_steps],
                )
            else:
                if node_coordinates[1] > last_pitch * self.increase_ratio_left:
                    raise ValueError(
                        "Bad spatial discretization.\n"
                        "Element length between the second and first node is larger "
                        "than expected. Please consider using a larger DXINCRE_LEFT "
                        "or a different number of total elements and elements in the "
                        "refined region or a combination of both."
                    )

        # --- Right coarsening region ---
        right_boundary_idx = n_elem_left + self.number_of_refined_elements
        coarsening_steps, last_pitch = self._coarsen_from_boundary(
            node_coordinates,
            boundary_idx=right_boundary_idx,
            boundary_position=self.end_refined_zone,
            n_elem=n_elem_right,
            region_length=length - self.end_refined_zone,
            starting_pitch=refined_zone_pitch,
            increase_ratio=self.increase_ratio_right,
            forward=True,
        )
        if n_elem_right > 0:
            if coarsening_steps < n_elem_right - 1:
                node_coordinates[right_boundary_idx + coarsening_steps :] = np.linspace(
                    node_coordinates[right_boundary_idx + coarsening_steps],
                    length,
                    n_elem_right + 1 - coarsening_steps,
                )
                validator.check_coarse_region_quality(
                    last_pitch=last_pitch,
                    n_elem=n_elem_right - coarsening_steps,
                    region_length=length - node_coordinates[right_boundary_idx + coarsening_steps],
                    is_left=False,
                )
            else:
                node_coordinates[-1] = length
                if node_coordinates[-1] > node_coordinates[-2] + last_pitch * self.increase_ratio_right:
                    raise ValueError(
                        "Bad spatial discretization.\n"
                        "Element length between the last-but-one and last node is "
                        "larger than expected. Please consider using a larger "
                        "DXINCRE_RIGHT or a different number of total elements and "
                        "elements in the refined region or a combination of both."
                    )

        return node_coordinates

    # -------------------------------------------------------------------------
    # Angular (helicoidal) discretization
    # -------------------------------------------------------------------------

    def build_uniform_angular_mesh(self, winding_number: float) -> np.ndarray:
        """Return a uniform angular coordinate array for a helicoidal component.

        Args:
            winding_number: total number of windings of the helix.

        Returns:
            1-D array of angles from 0 to winding_number * 2π.
        """
        return np.linspace(0.0, winding_number * 2 * np.pi, self.number_of_nodes)


    def build_refined_angular_mesh(
        self,
        winding_number: float,
        reduced_pitch: float,
    ) -> np.ndarray:
        """Return an angular coordinate array with a locally refined zone, for a helicoidal component.

        Mirrors the logic of _build_refined_mesh but works in angle space.

        Args:
            winding_number: total number of windings of the helix.
            reduced_pitch: reduced pitch of the cylindrical helix (p / (2π)).

        Raises:
            ValueError: if winding counts are inconsistent with the total.
            ValueError: if the angular pitch in the refined zone is outside the allowed range.
            ValueError: if the coarsening cannot fit within the available elements.

        Returns:
            1-D array of angular node coordinates of length number_of_nodes.
        """
        tau = np.zeros(self.number_of_nodes)

        n_elem_coarse = self.number_of_elements - self.number_of_refined_elements
        n_elem_left = round(
            (self.start_refined_zone - 0.0)
            / (self.conductor_length - (self.end_refined_zone - self.start_refined_zone))
            * n_elem_coarse
        )
        n_elem_right = n_elem_coarse - n_elem_left

        n_nodes_refined_zone = self.number_of_refined_elements + 1

        n_windings_left  = self.start_refined_zone / (2 * np.pi * reduced_pitch)
        n_windings_right = (self.conductor_length - self.end_refined_zone) / (2 * np.pi * reduced_pitch)
        n_windings_ref   = (self.end_refined_zone - self.start_refined_zone) / (2 * np.pi * reduced_pitch)
        n_windings_total = n_windings_left + n_windings_ref + n_windings_right

        if not np.isclose(winding_number, n_windings_total):
            raise ValueError(
                f"Inconsistent winding counts: {winding_number = } != "
                f"{n_windings_total = }"
            )

        final_angle = 2 * np.pi * winding_number

        refined_angular_pitch = (
            2 * np.pi * n_windings_ref / self.number_of_refined_elements
        )

        # Angular-pitch bounds derived from SIZMIN / SIZMAX.
        n_elem_for_min_size = np.round(
            (self.end_refined_zone - self.start_refined_zone) / self.minimum_element_size
        )
        n_elem_for_max_size = np.round(
            (self.end_refined_zone - self.start_refined_zone) / self.maximum_element_size
        )
        min_angular_pitch = 2 * np.pi * n_windings_ref / n_elem_for_min_size
        max_angular_pitch = 2 * np.pi * n_windings_ref / n_elem_for_max_size

        if refined_angular_pitch < min_angular_pitch:
            raise ValueError(
                f"ERROR: {refined_angular_pitch=} rad in refined zone "
                f"< {min_angular_pitch=} rad!!!\n"
            )
        if refined_angular_pitch > max_angular_pitch:
            raise ValueError(
                f"ERROR: {refined_angular_pitch=} rad in refined zone "
                f"> {max_angular_pitch} rad!!!\n"
            )

        refined_zone_start_angle = 2 * np.pi * n_windings_left + refined_angular_pitch
        refined_zone_end_angle   = 2 * np.pi * (n_windings_left + n_windings_ref)
        tau[n_elem_left : n_elem_left + n_nodes_refined_zone] = np.linspace(
            refined_zone_start_angle, refined_zone_end_angle, n_nodes_refined_zone
        )

        # --- Left coarsening region ---
        coarsening_steps, last_pitch = self._coarsen_from_boundary(
            tau,
            boundary_idx=n_elem_left,
            boundary_position=refined_zone_start_angle,
            n_elem=n_elem_left,
            region_length=2 * np.pi * n_windings_left,
            starting_pitch=refined_angular_pitch,
            increase_ratio=self.increase_ratio_left,
            forward=False,
        )
        if n_elem_left > 0:
            if coarsening_steps < n_elem_left - 1:
                tau[: n_elem_left + 1 - coarsening_steps] = np.linspace(
                    0.0,
                    tau[n_elem_left - coarsening_steps],
                    n_elem_left + 1 - coarsening_steps,
                )
                validator.check_coarse_region_quality(
                    last_pitch=last_pitch,
                    n_elem=n_elem_left - coarsening_steps,
                    region_length=tau[n_elem_left - coarsening_steps],
                )
            else:
                if tau[1] > last_pitch * self.increase_ratio_left:
                    raise ValueError(
                        "Bad spatial discretization.\n"
                        "Element length between the second and first node is larger "
                        "than expected. Please consider using a larger DXINCRE_LEFT "
                        "or a different number of total elements and elements in the "
                        "refined region or a combination of both."
                    )

        # --- Right coarsening region ---
        right_boundary_idx = n_elem_left + self.number_of_refined_elements
        coarsening_steps, last_pitch = self._coarsen_from_boundary(
            tau,
            boundary_idx=right_boundary_idx,
            boundary_position=refined_zone_end_angle,
            n_elem=n_elem_right,
            region_length=2 * np.pi * n_windings_right,
            starting_pitch=refined_angular_pitch,
            increase_ratio=self.increase_ratio_right,
            forward=True,
        )
        if n_elem_right > 0:
            if coarsening_steps < n_elem_right - 1:
                tau[right_boundary_idx + coarsening_steps :] = np.linspace(
                    tau[right_boundary_idx + coarsening_steps],
                    final_angle,
                    n_elem_right + 1 - coarsening_steps,
                )
                validator.check_coarse_region_quality(
                    last_pitch=last_pitch,
                    n_elem=n_elem_right - coarsening_steps,
                    region_length=final_angle - tau[right_boundary_idx + coarsening_steps],
                    is_left=False,
                )
            else:
                tau[-1] = final_angle
                if tau[-1] > tau[-2] + last_pitch * self.increase_ratio_right:
                    raise ValueError(
                        "Bad spatial discretization.\n"
                        "Element length between the last-but-one and last node is "
                        "larger than expected. Please consider using a larger "
                        "DXINCRE_RIGHT or a different number of total elements and "
                        "elements in the refined region or a combination of both."
                    )

        return tau

    # -------------------------------------------------------------------------
    # User-defined grid validation
    # -------------------------------------------------------------------------

    def validate_user_defined_grid(
        self,
        dfs: dict,
        n_components: int,
        reference_component_id: str,
        file_path: str,
    ) -> "tuple[int, np.ndarray]":
        """Check consistency of a user-defined spatial discretization across all components.

        Args:
            dfs: dict of DataFrames keyed by component identifier; each DataFrame
                has at least columns "x [m]", "y [m]", "z [m]".
            n_components: total number of conductor components.
            reference_component_id: identifier of the FluidComponent used as the
                reference for z-coordinate comparison.
            file_path: path to the user-defined grid file (used in error messages).

        Raises:
            ValueError: if the number of sheets != n_components.
            ValueError: if fewer than 3 coordinate columns are present in any sheet.
            ValueError: if the number of nodes differs between sheets.
            ValueError: if the z-coordinates of any component differ from the reference.

        Returns:
            (number_of_nodes, reference_z_coordinates): total node count and the
            z-coordinate array from the reference component.
        """
        if len(dfs) != n_components:
            raise ValueError(
                f"The number of sheets in file {file_path} must equal the number of "
                f"defined conductor components.\n{len(dfs)} != {n_components}.\n"
            )

        reference_node_count = self.check_node_count(
            dfs[reference_component_id].shape[0], file_path, reference_component_id
        )
        reference_z_coordinates = dfs[reference_component_id]["z [m]"].to_numpy()

        for comp_id, df in dfs.items():
            if comp_id == reference_component_id:
                continue
            if df.shape[1] < 3:
                raise ValueError(
                    f"User must provide three coordinates in sheet {comp_id} of "
                    f"input file {file_path}.\n"
                )
            if df.shape[0] != reference_node_count:
                raise ValueError(
                    f"Inconsistent number of user-defined nodes. The number of nodes "
                    f"in sheet {comp_id} of file {file_path} must equal the one "
                    f"defined in sheet {reference_component_id} of the same file."
                )
            if not np.allclose(df["z [m]"].to_numpy(), reference_z_coordinates, equal_nan=True):
                raise ValueError(
                    f"User must provide the same z component of the coordinate for "
                    f"all components. Please check column 'z [m]' in sheet {comp_id} "
                    f"of file {file_path}."
                )

        return reference_node_count, reference_z_coordinates


    def build_node_coordinates(self, identifier: str = "conductor") -> np.ndarray:
        """Compute and store node coordinates for non-file mesh types.

        Dispatches to the appropriate builder based on self.mesh_type and sets
        self.node_coordinates. self.number_of_nodes must be set before calling this.
        Must not be called when mesh_type is FROM_FILE.

        Args:
            identifier: conductor identifier used in boundary-check error messages.

        Raises:
            ValueError: if mesh_type is FROM_FILE.
        """
        refined_zone_width = self.end_refined_zone - self.start_refined_zone

        if (
            self.mesh_type in (MeshType.UNIFORM, MeshType.ADAPTED)
            or abs(refined_zone_width) <= 1e-3
        ):
            node_coordinates = self._build_uniform_mesh(self.conductor_length)
        elif self.mesh_type == MeshType.REFINED:
            node_coordinates = self._build_refined_mesh(self.conductor_length)
        else:
            raise ValueError(
                f"build_node_coordinates cannot be used with {self.mesh_type}. "
                f"Use validate_user_defined_grid for FROM_FILE meshes."
            )

        self.check_boundary_coordinates(
            node_coordinates[0],
            node_coordinates[-1],
            identifier,
        )
        return node_coordinates


    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def check_boundary_coordinates(
        self, 
        z0: float,
        z1: float,
        identifier: str,
    ):
        """Check that the first and last node coordinates match the expected boundaries.

        Args:
            conductor_length: straight length of the cable.
            z0: first node z-coordinate.
            z1: last node z-coordinate.
            identifier: component identifier, used in error messages.

        Raises:
            ValueError: if z0 != 0 or z1 != conductor_length (within tolerance).
        """
        coordinate_tolerance = 1e-6
        if abs(z0 - 0.0) > coordinate_tolerance:
            message = (
                f"{identifier = }: z0 must be 0.0; {z0 = } (m) > {0.0} (m)."
            )
            logger_discretization.error(message)
            raise ValueError(message)

        if abs(z1 - self.conductor_length) > coordinate_tolerance:
            message = (
                f"{identifier = }: z1 does not equal conductor_length!\n"
                f"{z1 = } m; {self.conductor_length = } m"
            )
            logger_discretization.warning(message)
            raise ValueError(message)


    def check_number_of_nodes(self, number_of_nodes: int) -> None:
        """Check that the number of nodes does not exceed the maximum allowed value.

        Args:
            number_of_nodes: proposed number of nodes.

        Raises:
            ValueError: if number_of_nodes exceeds self.maximum_number_of_nodes.

        Returns:
            number_of_nodes if the check passes.
        """
        if number_of_nodes > self.maximum_number_of_nodes:
            message = (
                f"The number of nodes should not exceed the maximum value. "
                f"{number_of_nodes = } > {self.maximum_number_of_nodes = }.\n"
                f"Please check the grid input file.\n"
            )
            raise ValueError(message)
        


    # -------------------------------------------------------------------------
    # Coarsening helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _coarsen_from_boundary(
        coordinates: np.ndarray,
        boundary_idx: int,
        boundary_position: float,
        n_elem: int,
        region_length: float,
        starting_pitch: float,
        increase_ratio: float,
        forward: bool,
    ) -> "tuple[int, float]":
        """Fill a coarsening transition region using a geometric series of element sizes.

        Computes the minimum number of geometrically growing elements needed to bridge
        from the refined zone boundary to a uniform coarse region, then fills those
        coordinates with a single vectorized assignment.

        Both the left (backward) and right (forward) sides of the refined zone use
        this method — the only difference is the fill direction.

        Args:
            coordinates: node-coordinate array, modified in-place.
            boundary_idx: index of the already-set refined-zone boundary node.
            boundary_position: coordinate value at boundary_idx.
            n_elem: number of elements available in this coarsening region.
            region_length: total length of this region (determines tentative pitch).
            starting_pitch: refined-zone element size; first coarsening pitch is
                starting_pitch * increase_ratio.
            increase_ratio: growth factor per step (increase_ratio_left or _right).
            forward: False → fill backward toward index 0 (left region);
                     True  → fill forward toward the last index (right region).

        Returns:
            (coarsening_steps, last_pitch): number of geometric elements placed and
            the size of the final one. The caller uses these to fill the remaining
            uniform sub-region and run the quality check.
        """
        # n_elem == 1 means no coarsening elements can be placed (loop would never run).
        if n_elem <= 1:
            return 0, starting_pitch

        # Geometric pitch sequence: starting_pitch*r, starting_pitch*r², …
        k_range = np.arange(1, n_elem)
        pitches_k = starting_pitch * np.power(increase_ratio, k_range)
        cumulative_k = np.cumsum(pitches_k)

        # remaining_elements is always >= 1, so division is safe.
        remaining_lengths = region_length - cumulative_k
        remaining_elements = n_elem - k_range  # [n_elem-1, n_elem-2, …, 1]
        tentative_pitches = remaining_lengths / remaining_elements

        # Stop as soon as the tentative pitch no longer exceeds the next coarsening pitch.
        stop_condition = tentative_pitches <= increase_ratio * pitches_k
        stop_idx = int(np.argmax(stop_condition))
        if not stop_condition[stop_idx]:
            stop_idx = len(stop_condition) - 1  # condition never met → coarsen maximally

        coarsening_steps = stop_idx + 1          # convert 0-indexed to count
        last_pitch = pitches_k[stop_idx]

        if forward:
            coordinates[boundary_idx + 1 : boundary_idx + coarsening_steps + 1] = (
                boundary_position + cumulative_k[:coarsening_steps]
            )
        else:
            coordinates[boundary_idx - coarsening_steps : boundary_idx] = (
                boundary_position - cumulative_k[:coarsening_steps][::-1]
            )

        return coarsening_steps, last_pitch


# --------------------------------------------------------------------------
# The functions below assemble per-component coordinate/connectivity/distance
# data for a whole Conductor (spatial-discretization bookkeeping that
# consumes the conductor's ConductorMesh, rather than being part of the mesh
# itself). Relocated from conductor/conductor.py as part of splitting the
# file for single responsibility.
# --------------------------------------------------------------------------


def evaluate_component_coordinates(conductor: object, simulation: object) -> None:
    """Evaluate the grid coordinates and assign them to the conductor objects and its components according to the value of flag mesh_type."""
    # Deferred import: utility_functions.initialization_functions imports
    # ConductorMesh from this module, so importing it at module level here
    # would create a circular import.
    from utility_functions.initialization_functions import (
        build_coordinates_of_barycenter,
        user_defined_grid,
    )

    if conductor.mesh.mesh_type != MeshType.FROM_FILE:
        for comp in conductor.inventory.all_components.collection:
            build_coordinates_of_barycenter(simulation, conductor, comp) # TODO
    else:
        # Call function user_defined_grid: makes checks on the user defined
        # grid and then assigns the coordinates to the conductor components.
        user_defined_grid(conductor)


def build_multi_index(conductor: object) -> pd.MultiIndex:
    """Build the multindex used in pandas dataframes used to store the nodal coordinates and the connectivity (matrix) of each conductor component.

    Returns:
        pd.MultiIndex: pandas multindex with 'Kind' (parent class) and 'Identifier' (the component identifier).
    """
    identifiers = [
        obj.identifier for obj in conductor.inventory.all_components.collection
    ]
    kinds = list()
    for obj in conductor.inventory.all_components.collection:
        if isinstance(obj, StrandComponent):
            kinds.append(StrandComponent.__name__)
        else:
            kinds.append(obj.__class__.__name__)

    cat_kind = pd.CategoricalIndex(
        np.tile(kinds, conductor.mesh.number_of_elements + 1),
        dtype="category",
        ordered=True,
        categories=["FluidComponent", "StrandComponent", "JacketComponent"],
    )
    cat_ids = pd.CategoricalIndex(
        np.tile(identifiers, conductor.mesh.number_of_elements + 1),
        dtype="category",
        ordered=True,
        categories=identifiers,
    )

    return pd.MultiIndex.from_arrays(
        [cat_kind, cat_ids], names=["Kind", "Identifier"]
    )


def build_multi_index_current_carriers(conductor: object) -> pd.MultiIndex:
    """Build the multindex used in pandas dataframes used to store the nodal coordinates and the connectivity (matrix) only for conductor components of kind strand (StrandMixedComponent, StrandStabilizerComponent and StrandSuperconductroComponent).

    Returns:
        pd.MultiIndex:  pandas multindex with 'Kind' (parent class) and 'Identifier' (the component identifier).
    """
    identifiers = [
        obj.identifier for obj in conductor.inventory.strands.collection
    ]
    kinds = [
        obj.__class__.__name__
        for obj in conductor.inventory.strands.collection
    ]

    cat_kind = pd.CategoricalIndex(
        np.tile(kinds, conductor.mesh.number_of_elements + 1),
        dtype="category",
        ordered=True,
        categories=[
            "StrandMixedComponent",
            "StrandStabilizerComponent",
            "StackComponent",
        ],
    )
    cat_ids = pd.CategoricalIndex(
        np.tile(identifiers, conductor.mesh.number_of_elements + 1),
        dtype="category",
        ordered=True,
        categories=identifiers,
    )

    return pd.MultiIndex.from_arrays(
        [cat_kind, cat_ids], names=["Kind", "Identifier"]
    )


def initialize_mesh_dataframes(conductor: object) -> None:
    """Initialize pandas dataframes used to store nodal coordinates and connectivity (matrix)."""

    multi_index = build_multi_index(conductor)
    multi_index_current_carriers = build_multi_index_current_carriers(conductor)

    conductor.nodal_coordinates = pd.DataFrame(
        dict(
            x=np.zeros(conductor.total_nodes),
            y=np.zeros(conductor.total_nodes),
            z=np.zeros(conductor.total_nodes),
        ),
        index=multi_index,
    )

    conductor.connectivity_matrix = pd.DataFrame(
        dict(
            start=np.zeros(conductor.total_elements, dtype=int),
            end=np.zeros(conductor.total_elements, dtype=int),
            identifiers=pd.Series(np.zeros(conductor.total_elements), dtype=str),
        ),
        index=multi_index[: -conductor.inventory.all_components.number],
    )

    conductor.connectivity_matrix_current_carriers = pd.DataFrame(
        dict(
            start=np.zeros(conductor.total_elements_current_carriers, dtype=int),
            end=np.zeros(conductor.total_elements_current_carriers, dtype=int),
            identifiers=pd.Series(
                np.zeros(conductor.total_elements_current_carriers), dtype=str
            ),
        ),
        index=multi_index_current_carriers[
            : -conductor.inventory.strands.number
        ],
    )


def build_nodal_coordinates(conductor: object, nn: int, collection: object) -> None:
    """Build the dataframe with the nodal coordinates of all conductor components.

    Args:
        conductor (object): conductor object.
        nn (int): starting value of the index
        collection (ComponentCollection): one of conductor.inventory's collections (fluids, strands or jackets).
    """

    for ii, obj in enumerate(collection.collection, nn):
        for coord in ["x", "y", "z"]:
            conductor.nodal_coordinates.iloc[
                ii :: conductor.inventory.all_components.number,
                conductor.nodal_coordinates.columns.get_loc(coord),
            ] = obj.coordinate[coord]


def build_connectivity(conductor: object, nn: int, collection: object) -> None:
    """Build the dataframe with the connections (start and end node of each elements) of all conductor components.

    Args:
        conductor (object): conductor object.
        nn (int): starting value of the index
        collection (ComponentCollection): one of conductor.inventory's collections (fluids, strands or jackets).
    """
    for ii, obj in enumerate(collection.collection, nn):
        nodes = np.linspace(
            ii, ii + conductor.total_elements, conductor.mesh.number_of_elements + 1, dtype=int
        )
        conductor.connectivity_matrix.iloc[
            ii :: conductor.inventory.all_components.number,
            conductor.connectivity_matrix.columns.get_loc("start"),
        ] = nodes[:-1]
        conductor.connectivity_matrix.iloc[
            ii :: conductor.inventory.all_components.number,
            conductor.connectivity_matrix.columns.get_loc("end"),
        ] = nodes[1:]
        conductor.connectivity_matrix.iloc[
            ii :: conductor.inventory.all_components.number,
            conductor.connectivity_matrix.columns.get_loc("identifiers"),
        ] = obj.identifier
    # End for


def compute_node_distance(conductor: object) -> None:
    """Compute the distance between nodes taking into account all the coordinates (x,y,z). Values are stored in attribute node_distance."""
    conductor.node_distance = (
        (
            (
                conductor.nodal_coordinates.iloc[conductor.connectivity_matrix["end"], :]
                - conductor.nodal_coordinates.iloc[conductor.connectivity_matrix["start"], :]
            )
            ** 2
        )
        .sum(axis=1)
        .apply(np.sqrt)
    )

    # noda_distance correction.
    if conductor.inventory.strands.number == 1:
        # In this case attribute node_distance does not account for the
        # twist pich of the strand, i.e. the strand is straight. Apply the
        # costheta correction to account for the real distance between
        # nodal points. This correction has an impact only in the
        # evaluation of the quantity used in the electromagnetic module, as
        # for instance electric resistance, electric conductance and
        # inductance.
        # This is not necessary if there are more than one
        # strand since in this case the helicoidal coordinates are used.
        conductor.node_distance = conductor.node_distance / conductor.inventory.strands.collection[0].inputs.cos_theta


def compute_gauss_node_distance(conductor: object) -> None:
    """Evaluate the distance between consecutive gauss node (the mid point of the element), taking into account all the coordinates (x,y,z). Values are stored in attribute gauss_node_distance."""
    conductor.gauss_node_distance = np.zeros(conductor.total_nodes)
    # On the first cross section there is only the contribution from the
    # firts element
    conductor.gauss_node_distance[: conductor.inventory.all_components.number] = (
        conductor.node_distance[: conductor.inventory.all_components.number] / 2
    )

    # All the 'inner' distances are evaluated as
    # (l_k + l_(k+1))/2, for k in [0,total_nodes]
    conductor.gauss_node_distance[
        conductor.inventory.all_components
        .number : -conductor.inventory.all_components
        .number
    ] = (
        conductor.node_distance[: -conductor.inventory.all_components.number]
        + conductor.node_distance[conductor.inventory.all_components.number :]
    ) / 2

    # On the last cross section there is only the contribution from the
    # last element
    conductor.gauss_node_distance[-conductor.inventory.all_components.number :] = (
        conductor.node_distance[-conductor.inventory.all_components.number :] / 2
    )