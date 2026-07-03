"""
This module contains the PhysicalField class, representing a single physical
field in space and time, and the FieldContainer class, a typed container of
such fields defined over one grid location (nodal or Gauss points).

FieldContainer replaces the legacy string-keyed ``dict_node_pt`` /
``dict_Gauss_pt`` dictionaries that used to be scattered across the coolant,
solid-component, conductor and electromagnetic code. Fields are accessed as
attributes:

    fields = FieldContainer(GridLocation.NODE)
    fields.temperature = np.zeros(n_nodes)   # create / assign
    fields.temperature                       # -> np.ndarray
    fields.field("temperature")              # -> PhysicalField (name, unit, values)
"""

from __future__ import annotations

from enum import Enum
from typing import Iterator, Mapping, Optional

import numpy as np


class GridLocation(Enum):
    """Spatial location on which a physical field is sampled."""
    NODE = "node"
    GAUSS = "gauss"


# Registry of the physical unit associated with each known field name. Unknown
# names are accepted (with an empty unit) so the container stays a safe drop-in
# for the legacy dictionaries, but keeping the registry populated documents the
# fields and lets tooling reason about them.
FIELD_UNITS: dict[str, str] = {
    # Thermal-hydraulic state
    "temperature": "K",
    "temperature_change": "K",
    "pressure": "Pa",
    "velocity": "m/s",
    "mass_flow_rate": "kg/s",
    "total_density": "kg/m^3",
    "total_enthalpy": "J/kg",
    "total_isobaric_specific_heat": "J/kg/K",
    "total_isochoric_specific_heat": "J/kg/K",
    "total_speed_of_sound": "m/s",
    "total_thermal_conductivity": "W/m/K",
    "Gruneisen": "-",
    # Heat exchange / sources
    "HTC": "W/m^2/K",
    "EXTFLX": "W/m",
    "JHTFLX": "W/m",
    "EEXT": "J",
    "EJHT": "J",
    "K1": "-",
    "K2": "-",
    "K3": "-",
    "Q1": "-",
    "Q2": "-",
    "T_cur_sharing": "K",
    # Electromagnetic state
    "B_field": "T",
    "J_critical": "A/m^2",
    "op_current": "A",
    "op_current_sc": "A",
    "current_along": "A",
    "electric_resistance": "ohm",
    "electrical_resistivity_stabilizer": "ohm*m",
    "electrical_resistivity_superconductor": "ohm*m",
    "total_electrical_resistivity": "ohm*m",
    "alpha_B": "-",
    "Epsilon": "-",
    "linear_power_el_resistance": "W/m",
    "delta_voltage_along": "V",
    "delta_voltage_along_sum": "V",
    "delta_voltag_along_R": "V",
    "total_linear_power_el_cond": "W/m",
    "total_power_el_cond": "W",
}


class TimeEvolution:
    """Chunked record of a field's values at user-selected spatial coordinates
    over time.

    The record holds one column of time stamps plus one column per saved
    coordinate (labelled e.g. ``"zcoord = 0.5 (m)"``). The output functions
    flush the record to file every ``Conductor.CHUNCK_SIZE`` recorded times
    and re-initialize it.
    """

    __slots__ = ("times", "samples")

    TIME_LABEL = "time (s)"

    def __init__(self):
        self.times: list = []
        self.samples: dict[str, list] = {}

    def initialize(self, coordinate_labels) -> None:
        """(Re-)initialize the record with one empty column per coordinate label."""
        self.times = []
        self.samples = {label: [] for label in coordinate_labels}

    def record(self, time: float, values: np.ndarray, coordinate_indices: Mapping[str, int]) -> None:
        """Append the sample of values at the given time, one entry per saved
        coordinate (coordinate_indices maps column label to spatial index)."""
        self.times.append(time)
        for label, index in coordinate_indices.items():
            self.samples[label].append(values[index])

    def __len__(self) -> int:
        return len(self.times)

    def columns(self) -> dict:
        """Return the record as a column-label -> list mapping (time first),
        ready for DataFrame construction."""
        record = {self.TIME_LABEL: self.times}
        record.update(self.samples)
        return record


class PhysicalField:
    """A single named physical field (its metadata, its sampled values and
    the record of its time evolution at user-selected coordinates)."""

    __slots__ = ("name", "unit", "values", "time_evolution")

    def __init__(self, name: str, unit: str, values: Optional[np.ndarray] = None):
        self.name = name
        self.unit = unit
        self.values = values
        self.time_evolution = TimeEvolution()

    def initialize(self, initial_values: np.ndarray) -> None:
        self.values = initial_values

    def __repr__(self) -> str:
        shape = None if self.values is None else np.shape(self.values)
        return f"PhysicalField(name={self.name!r}, unit={self.unit!r}, shape={shape})"


# Instance attribute names that belong to the container itself rather than to a
# stored physical field. Kept in a set for fast membership tests in __setattr__.
_RESERVED = {"location", "_fields"}


class FieldContainer:
    """Typed, attribute-accessed container of :class:`PhysicalField` instances.

    All fields in a container share the same :class:`GridLocation`. Assigning to
    an unknown attribute transparently creates the corresponding field; reading
    it returns the stored ``np.ndarray`` (or whatever value was stored).
    """

    def __init__(self, location: GridLocation = GridLocation.NODE):
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "_fields", {})

    # -- attribute-style access -------------------------------------------------
    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so reserved
        # attributes (set via object.__setattr__) never reach here.
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name].values
        raise AttributeError(
            f"{type(self).__name__} has no field {name!r} "
            f"(location={object.__getattribute__(self, 'location')})."
        )

    def __setattr__(self, name: str, value) -> None:
        if name in _RESERVED:
            object.__setattr__(self, name, value)
            return
        fields = self._fields
        if name in fields:
            fields[name].values = value
        else:
            fields[name] = PhysicalField(name, FIELD_UNITS.get(name, ""), value)

    def __delattr__(self, name: str) -> None:
        if name in self._fields:
            del self._fields[name]
        else:
            object.__delattr__(self, name)

    # -- typed / metadata access ------------------------------------------------
    def field(self, name: str) -> PhysicalField:
        """Return the :class:`PhysicalField` object (metadata + values)."""
        return self._fields[name]

    def initialize(self, name: str, values: np.ndarray, unit: Optional[str] = None) -> None:
        """Create or overwrite a field, optionally overriding the default unit."""
        self._fields[name] = PhysicalField(
            name, unit if unit is not None else FIELD_UNITS.get(name, ""), values
        )

    def has(self, name: str) -> bool:
        """True if the field exists and has been given values."""
        f = self._fields.get(name)
        return f is not None and f.values is not None

    def ensure_field(self, name: str) -> PhysicalField:
        """Return the :class:`PhysicalField` object, creating an empty one if
        the field does not exist yet (e.g. to initialize its time-evolution
        record before the first values are assigned)."""
        if name not in self._fields:
            self._fields[name] = PhysicalField(name, FIELD_UNITS.get(name, ""))
        return self._fields[name]

    # -- mapping-like helpers (ergonomics for bulk operations) ------------------
    def names(self) -> Iterator[str]:
        return iter(self._fields)

    def __getitem__(self, name: str):
        """Subscript access to a field's values, equivalent to attribute access."""
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name].values
        raise KeyError(
            f"{type(self).__name__} has no field {name!r} "
            f"(location={object.__getattribute__(self, 'location')})."
        )

    def __setitem__(self, name: str, value) -> None:
        """Subscript assignment to a field, equivalent to attribute assignment."""
        setattr(self, name, value)

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __bool__(self) -> bool:
        return bool(self._fields)

    def update_from(self, mapping: Mapping[str, np.ndarray]) -> None:
        """Bulk-assign fields from a name -> values mapping."""
        for name, value in mapping.items():
            setattr(self, name, value)

    def update(self, mapping: Optional[Mapping[str, np.ndarray]] = None, **kwargs) -> None:
        """dict-like bulk assignment via a mapping and/or keyword arguments.

        Keyword form (``fields.update(total_density=arr)``) keeps the call site
        typed (identifiers, not string subscripts).
        """
        if mapping is not None:
            self.update_from(mapping)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def get(self, name: str, default=None):
        """Return the field's values, or ``default`` if the field is absent."""
        f = self._fields.get(name)
        return default if f is None else f.values

    def keys(self) -> Iterator[str]:
        return iter(self._fields)

    def values(self) -> Iterator:
        """Iterate over the stored values (arrays) of every field."""
        return (f.values for f in self._fields.values())

    def items(self) -> Iterator:
        """Iterate over (name, values) pairs of every field."""
        return ((name, f.values) for name, f in self._fields.items())

    def copy(self) -> "FieldContainer":
        """Shallow copy: a new container sharing the same value arrays."""
        return FieldContainer.from_mapping(dict(self.items()), self.location)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, np.ndarray], location: GridLocation
    ) -> "FieldContainer":
        """Build a container from a name -> values mapping (migration helper)."""
        container = cls(location)
        container.update_from(mapping)
        return container

    def __repr__(self) -> str:
        return (
            f"FieldContainer(location={self.location.value}, "
            f"fields={sorted(self._fields)})"
        )
