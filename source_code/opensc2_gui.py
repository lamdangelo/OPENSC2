"""Compatibility shim: the GUI moved to drivers.gui.opensc2_gui."""

import sys

import drivers.gui.opensc2_gui as _target

sys.modules[__name__] = _target
