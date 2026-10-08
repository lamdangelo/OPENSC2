"""Transitional re-export shim: utility_functions.auxiliary_functions was split
into numerics.interpolation (pure interpolation helpers),
components.component_ordering (identifier helpers, cos(theta) check) and
adapters.input.auxiliary_files (file readers). Removed in Stage 9."""

import warnings

from numerics.interpolation import (  # noqa: F401
    build_interpolator,
    do_interpolation,
    set_diagnostic,
)
from components.component_ordering import (  # noqa: F401
    check_costheta,
    filter_component,
    natural_sort,
)
from adapters.input.auxiliary_files import (  # noqa: F401
    get_from_xlsx,
    interpolation,
    load_auxiliary_files,
    read_interp_file,
    with_read_csv,
    with_read_excel,
)

warnings.warn(
    "utility_functions.auxiliary_functions was split; import from "
    "numerics.interpolation, components.component_ordering or "
    "adapters.input.auxiliary_files",
    DeprecationWarning,
    stacklevel=2,
)
