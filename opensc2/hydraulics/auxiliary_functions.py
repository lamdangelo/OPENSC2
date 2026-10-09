"""
This module contains auxiliary functions for the hydraulics package.
"""

def evaluate_correction_diameter_hole(hydraulic_diameter: float, tthick=1e-3):
    outer_diameter = hydraulic_diameter + 2 * tthick
    corrected_diameter = outer_diameter / hydraulic_diameter
    return outer_diameter, corrected_diameter 