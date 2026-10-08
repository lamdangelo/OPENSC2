"""Component identifier helpers (natural ordering of interface names,
filtering of component collections) and the cos(theta) input check."""

import re
from collections import namedtuple
from typing import Union

import numpy as np


def check_costheta(cos_theta: float, input_file_path: str, sheet_name: str):
    """Checks that cos(theta) is in the range (0, 1] for a strand component.

    Args:
        cos_theta (float): the cos(theta) value to validate.
        input_file_path (str): path to the input file, used in the error message.
        sheet_name (str): sheet name in the input file, used in the error message.

    Raises:
        ValueError: if cos_theta is 0.0 or |cos_theta| > 1.0
    """
    if np.isclose(cos_theta, 0.0) or abs(cos_theta) > 1.0:
        raise ValueError(
            f"ERROR: cos(theta) must be in the range (0,1]. Current value is {cos_theta}.\nPlease, check sheet {sheet_name} in file {input_file_path}.\n"
        )


def filter_component(iterable:Union[list,tuple],item:object)->tuple:
    """Function that returns a new collection of tuple (index,object) with all the objects except the one in item; index are the locations of the objects in iterable. The returned collection (tuple) preserves the order of the items in iterable.

    Args:
        iterable (Union[list,tuple]): collection of object to be filtered.
        item (object): element to be removed from iterable.

    Raises:
        TypeError: if iterable is not a list or a tuple.

    Returns:
        tuple: collection of tuple (index,object). All objects from iterable exctep the element passed as item are collected. Index stores the original location in iterable; items order is preserved.
    """

    # Namedtuple constructor.
    Obj_info = namedtuple("Obj_info",["idx","obj"])
    # Sanity check.
    if isinstance(iterable, (list,tuple)):
        # Return a new collection of (index,obj) without item preserving the 
        # order.
        return tuple(Obj_info(index,obj) for index,obj in enumerate(iterable) if obj != item)
    else:
        raise TypeError(f"Iterable should be of type list or tuple; current type is {type(iterable)}")


def natural_sort(comp_a, comp_b):
    # Use the regexes to sort naturally (human like) the IDs of the components to be able to deal with all the interfaces in a general way.
    match_a = re.search(
        r"(?P<Fluid_component>CHAN)?(?P<Stack>STACK)?(?P<Mixed_sc_stab>STR_MIX)??(?P<StrandStabilizerComponent>STR_STAB)?(?P<JacketComponent>Z_JACKET)?_(\d+)",
        comp_a.identifier,
    )
    # r'((CHAN)?(STACK)?(STR_MIX)?(STR_STAB)?(Z_JACKET)?)_(\d)+'
    match_b = re.search(
        r"(?P<Fluid_component>CHAN)?(?P<Stack>STACK)?(?P<Mixed_sc_stab>STR_MIX)?(?P<StrandStabilizerComponent>STR_STAB)?(?P<JacketComponent>Z_JACKET)?_(\d+)",
        comp_b.identifier,
    )

    if match_a.group(comp_a.KIND) < match_b.group(comp_b.KIND):
        return f"{comp_a.identifier}_{comp_b.identifier}"
    elif match_a.group(comp_a.KIND) > match_b.group(comp_b.KIND):
        return f"{comp_b.identifier}_{comp_a.identifier}"
    else:
        # Equal string part, sort by number
        if int(match_a.group(6)) < int(match_b.group(6)):
            return f"{comp_a.identifier}_{comp_b.identifier}"
        else:
            return f"{comp_b.identifier}_{comp_a.identifier}"
        # end if
    # end if