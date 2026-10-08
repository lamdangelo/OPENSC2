"""Electrical conductivity dispatcher for the eddy-current heat sources.

Maps a material name to its electrical conductivity ``sigma = 1 / rho``
[S/m], wrapping the existing per-material resistivity correlations. Used by
the eddy-current loss terms (aluminium jacket, copper strand matrix); the
superconductor interfilament/interstrand coupling loss keeps its own
effective-time-constant model and does not go through here.

Kept deliberately small and side-effect free so it can be unit tested in
isolation. The component classes already expose their own resistivity
fields (e.g. ``jacket_electrical_resistivity``); this helper is the
material-agnostic entry point when only a name and a thermodynamic state
are available.
"""

from __future__ import annotations

import numpy as np

from properties_of_materials.material import material_for


def electrical_resistivity_of(
    material: str,
    temperature: np.ndarray,
    magnetic_field: np.ndarray | None = None,
    residual_resistivity_ratio: float | None = None,
) -> np.ndarray:
    """Electrical resistivity [Ohm m] of ``material`` at the given state.

    Args:
        material: material name (case-insensitive; see ``_ALIASES``).
        temperature: temperature [K].
        magnetic_field: magnetic field [T]; required for field-dependent
            materials (copper), ignored otherwise.
        residual_resistivity_ratio: RRR; required for copper.

    Raises:
        KeyError: if no correlation is registered for ``material``.
        ValueError: if a required argument for the material is missing.
    """
    try:
        material_object = material_for(material)
    except KeyError:
        raise KeyError(
            f"no electrical resistivity correlation registered for material "
            f"'{material}'"
        ) from None
    if material_object.name == "cu" and (
        magnetic_field is None or residual_resistivity_ratio is None
    ):
        raise ValueError(
            "copper resistivity needs magnetic_field and "
            "residual_resistivity_ratio"
        )
    return np.asarray(
        material_object.electrical_resistivity(
            temperature,
            magnetic_field=magnetic_field,
            residual_resistivity_ratio=residual_resistivity_ratio,
        )
    )


def electrical_conductivity_of(
    material: str,
    temperature: np.ndarray,
    magnetic_field: np.ndarray | None = None,
    residual_resistivity_ratio: float | None = None,
) -> np.ndarray:
    """Electrical conductivity ``sigma = 1 / rho`` [S/m] of ``material``.

    Same signature as :func:`electrical_resistivity_of`.
    """
    return 1.0 / electrical_resistivity_of(
        material,
        temperature,
        magnetic_field=magnetic_field,
        residual_resistivity_ratio=residual_resistivity_ratio,
    )
