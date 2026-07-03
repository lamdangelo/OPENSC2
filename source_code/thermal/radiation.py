"""
This module owns radiative heat-exchange calculations for the thermal problem:

* the weighted external radiative heat transfer coefficient between a solid
  component and the environment (:func:`eval_weighted_radiative_htc`);
* the radiative heat transfer coefficient between two concentric solid
  (jacket) surfaces (:func:`inner_radiative_htc`);
* the radiative power exchanged between jackets
  (:func:`compute_radiative_heat_exchange_jk`);
* the convective/radiative power exchanged between the outermost jacket and
  the environment (:func:`compute_heat_exchange_jk_env`).

Relocated from ``conductor/conductor.py`` as part of consolidating the
thermal calculations into the ``thermal`` package.
"""

import numpy as np
from scipy import constants

from conductor.conductor_flags import HTC_Choice

_RADIATIVE_HTC_CHOICES = (
    HTC_Choice.RADIATIVE_HTC_COMPUTED,
    HTC_Choice.RADIATIVE_HTC_READ_FROM_FILE,
)


def eval_weighted_radiative_htc(conductor: object, simulation: object, s_comp: object, temperature_jk: np.ndarray) -> np.ndarray:
    """Evaluate the weighted external radiative heat transfer coefficient: htc_rad*Phi_rad."""
    if s_comp.inputs.emissivity <= 0.0:
        # Set the radiative heat transfer coefficient to zero: no radiative heat transfer.
        return np.zeros(temperature_jk.shape)  # W/m^2/K
    else:
        return (
            constants.Stefan_Boltzmann
            * s_comp.inputs.emissivity
            * (
                temperature_jk ** 2
                + simulation.environment.inputs["Temperature"] ** 2
            )
            * (temperature_jk + simulation.environment.inputs["Temperature"])
            * conductor.inputs.phi_radiative
        )  # W/m^2/K
    # End if s_comp_r.inputs.emissivity <= 0.


def inner_radiative_htc(jk_i: object, jk_j: object, temp_i: np.ndarray, temp_j: np.ndarray, view_factor_rec: np.ndarray) -> np.ndarray:
    """Radiative heat transfer coefficient between two concentric jacket surfaces."""
    return (
        constants.Stefan_Boltzmann
        * (temp_i ** 2 + temp_j ** 2)
        * (temp_i + temp_j)
        / (
            (1.0 - jk_i.inputs.emissivity) / jk_i.inputs.emissivity
            + view_factor_rec
            + (1.0 - jk_j.inputs.emissivity)
            / jk_j.inputs.emissivity
            * (jk_i.inputs.outer_perimeter / jk_j.inputs.inner_perimeter)
        )
    )  # W/m^2/K


def compute_radiative_heat_exchange_jk(conductor: object) -> None:
    """Evaluate the radiative heat exchanged by radiation between jackets."""
    # Nested loop on jackets.
    for rr, jk_r in enumerate(conductor.inventory.jackets.collection):
        for _, jk_c in enumerate(
            conductor.inventory.jackets.collection[rr + 1 :]
        ):
            if (
                HTC_Choice.get_htc_choice_flag(
                    conductor.coupling.htc_choice[
                        jk_r.identifier, jk_c.identifier
                    ]
                )
                in _RADIATIVE_HTC_CHOICES
            ):
                conductor.heat_rad_jk[f"{jk_r.identifier}_{jk_c.identifier}"] = (
                    conductor.coupling.contact_perimeter[
                        jk_r.identifier, jk_c.identifier
                    ]
                    * conductor.mesh.element_lengths
                    * conductor.gauss_fields.HTC["sol_sol"]["rad"][
                        conductor.dict_topology["sol_sol"][jk_r.identifier][
                            jk_c.identifier
                        ]
                    ]
                    * (
                        jk_r.gauss_fields.temperature
                        - jk_c.gauss_fields.temperature
                    )
                )  # W
            # End if abs.
        # End for cc.
    # End for rr.


def compute_heat_exchange_jk_env(conductor: object, environment: object) -> None:
    """Compute the heat exchange between the outer surface of the conductor and the environment by convection and or radiation.

    Args:
        conductor (object): ConductorComponent object with all information needed for the computation.
        environment (object): environment object.
    """

    # Alias
    interf_flag = conductor.coupling.contact_perimeter_flag

    for jk in conductor.inventory.jackets.collection:
        if (
            abs(interf_flag[
                    environment.KIND, jk.identifier
                ]
            ) == 1
        ):
            key = f"{environment.KIND}_{jk.identifier}"
            conductor.heat_exchange_jk_env[key] = (
                conductor.coupling.contact_perimeter[
                    environment.KIND, jk.identifier
                ]
                * conductor.mesh.element_lengths
                * (
                    conductor.gauss_fields.HTC["env_sol"][key]["conv"]
                    + conductor.gauss_fields.HTC["env_sol"][key]["rad"]
                )
                * (
                    environment.inputs["Temperature"]
                    - jk.gauss_fields.temperature
                )
            )  # W
        # End if interf_flag.
    # End for jk.
