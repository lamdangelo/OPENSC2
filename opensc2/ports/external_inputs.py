"""External (auxiliary) input port.

Components and the conductor need tabulated time/space profiles (external
current, magnetic field, heat, alpha_B, strain), the superconductor scaling
parameters, the flow boundary values of a channel and user-defined grids.
They obtain them through this Protocol; the file formats live in
adapters.input.external_inputs. The lazy, first-step evaluation order of the
physics is unchanged: the reader is called where the files used to be read.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Protocol


class TransverseCouplingRow(NamedTuple):
    """One patch of a transverse insulation-coupling file: local nodes with
    x in [x_low, x_high] exchange heat with the partner component between
    partner_x_start and partner_x_end (see
    components.solid.transverse_coupling). A partner conductor named
    ``BOUNDARY`` is a fixed-temperature reservoir whose temperature (K) is
    held in ``partner_component``."""

    x_low: float
    x_high: float
    partner_conductor: str
    partner_component: str
    partner_x_start: float
    partner_x_end: float
    conductance: float  # W/(m K)
    temperature_exponent: float = 0.0
    reference_temperature: float = 1.0
    saturation_temperature: float = math.inf

    @property
    def is_boundary(self) -> bool:
        return self.partner_conductor == "BOUNDARY"

    @property
    def boundary_temperature(self) -> float:
        return float(self.partner_component)


class TabulatedSeries(NamedTuple):
    interpolator: Any  # scipy interpolator built by numerics.interpolation
    interpolation_flag: str  # "space_only" | "time_only" | "space_and_time"
    unit_flag: Any  # flag stored in the first cell of the table (units)


class ExternalInputReader(Protocol):
    def tabulated_series(
        self, quantity: str, identifier: str, interpolation_kind
    ) -> TabulatedSeries:
        """``quantity`` in {"current", "bfield", "heat", "alphab", "strain"};
        ``identifier`` the component the table block belongs to."""
        ...

    def scaling_input(self) -> dict:
        """Superconductor scaling parameters (legacy scaling_input.dat)."""
        ...

    def read_flow_parameters(self, conductor, path, component, flag_name, *intial):
        """Time-interpolated flow boundary values (legacy get_from_xlsx
        signature, kept for numerics.initial_flow and the channel BCs)."""
        ...

    def user_defined_quantity(self, quantity: str, sheet_name: str):
        """Raw table of an external file (e.g. "EXTERNAL_GRID")."""
        ...

    def user_defined_grid_tables(self) -> dict:
        """{component identifier: {column name: 1-D numpy array}} of the
        user-defined spatial discretization file; the columns include
        "x [m]", "y [m]", "z [m]"."""
        ...

    def transverse_coupling_rows(self, file_name: str) -> tuple:
        """The patches of a transverse insulation-coupling file as
        ``TransverseCouplingRow`` tuples, in file order. ``file_name`` is
        the value of a solid component's ``transverse_coupling_file``
        operation (absolute, or relative to the conductor definition)."""
        ...
