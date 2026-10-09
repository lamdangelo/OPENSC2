"""
This module contains classes storing the data from the optional external input files.
"""

import numpy as np 


class ExternalContactPerimeter():
    def __init__(self, z_coordinates: np.ndarray, component_names: list[str], 
                 contact_perimeters: np.ndarray):
        self.z_coordinates = z_coordinates
        self.component_names = component_names 
        self.contact_perimeters = contact_perimeters 