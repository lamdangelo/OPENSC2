"""Pure interpolation helpers for externally tabulated quantities (time
and/or space profiles) and the save-time/position schedule helper.

Numerics layer: numpy/scipy only. The file readers that produce the tables
live in adapters.input.auxiliary_files.
"""

import numpy as np
from scipy import interpolate

from conductor.conductor_flags import InterpolationType


def set_diagnostic(vv, **kwargs):
    """[summary]

    Args:
        vv ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Keep only the positive values, negative values means that user do not want to save spatial distribution (or time evolutions) of variables at that times (or spatial coordinate). Positive values are sorted in ascendent order.
    vv = np.sort(vv[vv >= 0])
    # add to array vv the lower boundary (lb) and the upper boundary (ub) to save data by default also at these times (or spatial coordinates)
    vv = np.concatenate((vv, np.array([kwargs["lb"], kwargs["ub"]], dtype=float)))
    return np.unique(vv)


# End function set_diagnostic.



def build_interpolator(df, interpolation_kind="linear"):
    """Function that builds the interpolator for the data loaded from auxiliary input files. If data are costant in time or in space 1D interpolate.interp1d is used, if values describe a dependence in both time and space interpolate.interp1d is used.

    Args:
        df (_type_): _description_
        interpolation_kind (str, optional): _description_. Defaults to "linear".

    Returns:
        _type_: _description_
    """
    # Normalize the enum to the string expected by scipy's interp1d (an
    # InterpolationType would silently be read as a spline order otherwise).
    if isinstance(interpolation_kind, InterpolationType):
        interpolation_kind = interpolation_kind.name.lower()
    # The first column of the dataframe stores the space points used in the
    # interpolation function starting from the second row.
    strand_space_points = df.iloc[1:,0].to_numpy(dtype=float)

    # The first row of the dataframe stores the time points used in the 
    # interpolation function starting from the second column.
    strand_time_points = df.iloc[0,1:].to_numpy(dtype=float)

    # Get the known values to be used in the interpolation, stored from the 
    # second row second column of the data frame.
    known_val = df.iloc[1:,1:].to_numpy(dtype=float)
    
    # Remark: value in position df.iloc[0,0] is not used; it is a kind of flag not well understood up to now.

    if strand_time_points.size == 1 and strand_space_points.size > 1:
        # Values are constant in time but not in space.
        known_val = known_val.reshape(known_val.shape[0])
        return (
            interpolate.interp1d(
                strand_space_points,
                known_val,
                bounds_error=False,
                fill_value=known_val[-1],
                kind=interpolation_kind,
            ),
            "space_only",
        )

    elif strand_time_points.size > 1 and strand_space_points.size == 1:
        # Values are constant in space but not in time.
        known_val = known_val.reshape(known_val.shape[1])
        return (
            interpolate.interp1d(
                strand_time_points,
                known_val,
                bounds_error=False,
                fill_value=known_val[-1],
                kind=interpolation_kind,
            ),
            "time_only",
        )

    elif strand_time_points.size > 1 and strand_space_points.size > 1:
        return (
            interpolate.interp2d(
                strand_time_points,
                strand_space_points,
                known_val,
                kind=interpolation_kind,
            ),
            "space_and_time",
        )


# End function build_interpolator



def do_interpolation(interpolator, zcoord, time_step, kind):
    """Functin that makes the interpolation of the data loaded from auxiliary input files accorting to the kind flag.

    Args:
        interpolator (_type_): _description_
        zcoord (_type_): _description_
        time_step (_type_): _description_
        kind (_type_): _description_

    Returns:
        _type_: _description_
    """

    if kind == "space_only":
        # Values are constant in time but not in space.
        return interpolator(zcoord)

    elif kind == "time_only":
        # Values are constant in space but not in time.
        return interpolator(time_step)

    elif kind == "space_and_time":
        return interpolator(time_step, zcoord).reshape(zcoord.shape)


# End function do_interpolation


