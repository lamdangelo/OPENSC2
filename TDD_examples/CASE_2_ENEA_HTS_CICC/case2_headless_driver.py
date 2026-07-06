"""
A headless (no-GUI) driver to run the simulation of the ENEA HTS CICC model.

Author: Laura D'Angelo
"""

from simulation import Simulation


# Setup
input_directory_path = './TDD_examples/CASE_2_ENEA_HTS_CICC/'
simulation = Simulation(input_directory_path)


# Run simulation workflow
simulation.run()
