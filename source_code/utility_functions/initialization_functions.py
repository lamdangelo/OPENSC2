import logging
import numpy as np
import os
import pandas as pd
import warnings
from typing import Union

from components.fluid.fluid_component import FluidComponent
from conductor.conductor_mesh import MeshType
from components.jacket.jacket_component import JacketComponent

# from conductor import Conductor
from components.solid.stack_component import StackComponent
from components.solid.strand_component import StrandComponent
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_stabilizer_component import StrandStabilizerComponent
from cylindrical_helix import CylindricalHelix

logger_discretization = logging.getLogger("opensc2Logger.discretization")


def user_defined_grid(conductor: object):
    """Load user-defined spatial discretization and assign it to each conductor component.

    Args:
        conductor (Conductor): conductor object with grid input and component inventory.

    Notes: does not allow interpolation in space or time; only fixed spatial
    discretization is available.
    """
    file_path = conductor.file_paths.external_grid
    coord_dfs = pd.read_excel(file_path, sheet_name=None)

    reference_component = conductor.inventory.fluids.collection[0]
    (
        conductor.mesh.number_of_nodes,
        conductor.mesh.node_coordinates,
    ) = conductor.mesh.validate_user_defined_grid(
        coord_dfs,
        conductor.inventory.all_components.number,
        reference_component.identifier,
        file_path,
    )
    conductor.mesh.number_of_elements = conductor.mesh.number_of_nodes - 1

    for comp in conductor.inventory.all_components.collection:
        assign_user_defined_spatial_discretization(comp, coord_dfs[comp.identifier])
        conductor.mesh.check_boundary_coordinates(
            comp.coordinate["z"][0],
            comp.coordinate["z"][-1],
            comp.identifier,
        )


def assign_user_defined_spatial_discretization(
    comp: Union[
        FluidComponent,
        JacketComponent,
        StrandMixedComponent,
        StrandStabilizerComponent,
        StackComponent,
    ],
    df: pd.DataFrame,
):
    """Assign the user-defined coordinates from a DataFrame to a conductor component.

    Args:
        comp: generic conductor component object.
        df: DataFrame with columns "x [m]", "y [m]", "z [m]".
    """
    comp.coordinate["x"] = df["x [m]"].to_numpy()
    comp.coordinate["y"] = df["y [m]"].to_numpy()
    comp.coordinate["z"] = df["z [m]"].to_numpy()


def straight_coordinates(
    sim: object,
    cond: object,
    comp: Union[
        FluidComponent,
        JacketComponent,
        StrandMixedComponent,
        StrandStabilizerComponent,
        StackComponent,
    ],
    xb: float,
    yb: float,
):
    """Build straight barycenter coordinates for a conductor component.

    Args:
        sim (object): simulation object, used when loading a user-defined mesh.
        cond (object): conductor object with mesh and grid input.
        comp: generic conductor component.
        xb (float): x coordinate of the component barycenter.
        yb (float): y coordinate of the component barycenter.
    """
    if cond.mesh.mesh_type is not MeshType.FROM_FILE:
        comp.coordinate["x"] = xb * np.ones(cond.mesh.number_of_nodes)
        comp.coordinate["y"] = yb * np.ones(cond.mesh.number_of_nodes)
        refined_zone_width = abs(
            cond.mesh.end_refined_zone - cond.mesh.start_refined_zone
        )
        if (
            cond.mesh.mesh_type is MeshType.UNIFORM
            or refined_zone_width <= 1e-3
        ):
            comp.coordinate["z"] = cond.mesh._build_uniform_mesh(
                cond.inputs.zlength
            )
        elif cond.mesh.mesh_type in (MeshType.REFINED, MeshType.ADAPTED):
            comp.coordinate["z"] = cond.mesh._build_refined_mesh(
                cond.inputs.zlength
            )
    else:
        coordinate, _ = cond.load_user_defined_quantity(
            sim, "EXTERNAL_GRID", cond.identifier
        )
        comp.coordinate["x"] = coordinate[:, 0]
        comp.coordinate["y"] = coordinate[:, 1]
        comp.coordinate["z"] = coordinate[:, 2]


def helicoidal_coordinates(
    sim: object,
    cond: object,
    comp: Union[
        StrandMixedComponent,
        StrandStabilizerComponent,
        StackComponent,
    ],
):
    """Build helicoidal barycenter coordinates for a strand or stack component.

    Args:
        sim (object): simulation object, used when loading a user-defined mesh.
        cond (object): conductor object with mesh and grid input.
        comp: strand or stack component with helix geometry.
    """
    comp.cyl_helix = CylindricalHelix(
        comp.inputs.x_barycenter,
        comp.inputs.y_barycenter,
        cond.inputs.zlength,
        comp.inputs.cos_theta,
    )
    if cond.mesh.mesh_type is not MeshType.FROM_FILE:
        if np.isclose(comp.inputs.x_barycenter, 0.0) and np.isclose(
            comp.inputs.y_barycenter, 0.0
        ):
            straight_coordinates(
                sim, cond, comp,
                comp.inputs.x_barycenter,
                comp.inputs.y_barycenter,
            )
            return

        refined_zone_width = abs(
            cond.mesh.end_refined_zone - cond.mesh.start_refined_zone
        )
        if (
            cond.mesh.mesh_type is MeshType.UNIFORM
            or refined_zone_width <= 1e-3
        ):
            tau = cond.mesh.build_uniform_angular_mesh(
                comp.cyl_helix.winding_number,
            )
        elif cond.mesh.mesh_type in (MeshType.REFINED, MeshType.ADAPTED):
            tau = cond.mesh.build_refined_angular_mesh(
                comp.cyl_helix.winding_number,
                comp.cyl_helix.reduced_pitch,
            )

        (
            comp.coordinate["x"],
            comp.coordinate["y"],
            comp.coordinate["z"],
        ) = comp.cyl_helix.helix_parametrization(tau)
    else:
        coordinate, _ = cond.load_user_defined_quantity(
            sim, "EXTERNAL_GRID", cond.identifier
        )
        comp.coordinate["x"] = coordinate[:, 0]
        comp.coordinate["y"] = coordinate[:, 1]
        comp.coordinate["z"] = coordinate[:, 2]


def build_coordinates_of_barycenter(
    sim: object,
    cond: object,
    comp: Union[
        FluidComponent,
        JacketComponent,
        StrandMixedComponent,
        StrandStabilizerComponent,
        StackComponent,
    ],
):
    """Build and assign barycenter coordinates for a conductor component.

    Args:
        sim (object): simulation object, used when loading a user-defined mesh.
        cond (object): conductor object with mesh and grid input.
        comp: generic conductor component.

    Raises:
        ValueError: if costheta != 1 for FluidComponent or JacketComponent.

    Note:
        If only one StrandComponent is defined, straight coordinates are used
        regardless of costheta, since inductance calculations are not needed.
    """
    if isinstance(comp, FluidComponent):
        costheta = comp.channel.inputs.cos_theta
        xb = comp.channel.inputs.x_barycenter
        yb = comp.channel.inputs.y_barycenter
    else:
        costheta = comp.inputs.cos_theta
        xb = comp.inputs.x_barycenter
        yb = comp.inputs.y_barycenter

    if costheta == 1:
        straight_coordinates(sim, cond, comp, xb, yb)
    else:
        if isinstance(comp, StrandComponent):
            if cond.inventory.strands.number > 1:
                helicoidal_coordinates(sim, cond, comp)
            elif cond.inventory.strands.number == 1:
                warnings.warn(
                    "User defined only one strand component, it is considered "
                    "straight since there is no need to evaluate the inductances."
                )
                straight_coordinates(sim, cond, comp, xb, yb)
        else:
            raise ValueError(
                r"$Cos(\theta)$"
                + f"for {comp.__class__.__name__} must be 1.0; "
                + f"current value {costheta = }\n"
            )

    cond.mesh.check_boundary_coordinates(
        comp.coordinate["z"][0],
        comp.coordinate["z"][-1],
        comp.identifier,
    )
