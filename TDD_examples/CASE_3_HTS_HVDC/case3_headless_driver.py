"""
A headless (no-GUI) driver to run the simulation of the HTS HVDC cable model.

Author: Laura D'Angelo
"""

from simulation import Simulation


# Setup
input_directory_path = './TDD_examples/CASE_3_HTS_HVDC/'
simulation = Simulation(input_directory_path)

# Run simulation workflow
simulation.run()
