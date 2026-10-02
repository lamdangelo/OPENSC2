"""Event-time conventions of the AAB51 quench test (CEA Saclay, 2007-03-06).

Measured time is counted from the quench timestamp of the fast data
acquisition; the simulations of the winding-pack models start 1100 s
before it, on the 6.18 K plateau of the inlet-temperature staircase.
"""

from datetime import datetime

# Timestamp of the fast acquisition trigger (t = 0 of the event files).
QUENCH_TIMESTAMP = datetime(2007, 3, 6, 14, 51, 53, 573894)

# Measured time at which the simulations start (s).
SIMULATION_START_MEASURED_TIME = -1100.0

# Detection of the real quench: DUP11-3 crosses 0.5 V at +0.461 s, the
# current decay starts at +0.645 s.
MEASURED_DETECTION_TIME = 0.461  # s
DETECTION_THRESHOLD = 0.5  # V on any double layer
DETECTION_TO_DISCHARGE_DELAY = 0.184  # s

# Pre-detection clamp of the measured schedules: the supply reservoir
# follows the measurement only up to this measured time while the model
# has not detected yet (so a late model quench never sees the real
# event's response), then the event window is replayed anchored to the
# model detection.
EVENT_WINDOW_START = -2.0  # s

# Simulated time after the start of the current dump (s).
POST_DISCHARGE_DURATION = 30.0

BAR = 1.0e5  # Pa
