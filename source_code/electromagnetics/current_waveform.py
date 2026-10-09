"""User-customisable transport-current waveform.

This is a leaf module (it depends on numpy only) so that both the strand
components and the conductor can import :func:`user_defined_current` at
module level without creating an import cycle through
``electromagnetics.operating_conditions``.
"""

from typing import Union

import numpy as np


def user_defined_current(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the transport current at the given time(s).

    **Customise this function** to define an arbitrary current waveform.
    It is called when ``CurrentMode.CURRENT_IS_FUNCTION`` is active.

    Args:
        time: Scalar or array of time values in seconds.

    Returns:
        Current value(s) in amperes at the requested time(s).
    """
    CURRENT_AMPLITUDE = 1.0  # [A]
    FREQUENCY = 50.0         # [Hz]
    return CURRENT_AMPLITUDE * np.cos(2 * np.pi * FREQUENCY * time)
