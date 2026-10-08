"""External (auxiliary) input port.

Components and the conductor need tabulated time/space profiles (external
current, magnetic field, heat, alpha_B, strain), the superconductor scaling
parameters, the flow boundary values of a channel and user-defined grids.
They obtain them through this Protocol; the file formats live in
adapters.input.external_inputs. The lazy, first-step evaluation order of the
physics is unchanged: the reader is called where the files used to be read.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol


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
