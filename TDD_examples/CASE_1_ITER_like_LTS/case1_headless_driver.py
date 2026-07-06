"""
A minimal headless (no-GUI) driver to run the simulation of the ITER-like LTS.

Author: Laura D'Angelo
"""

from simulation import Simulation

input_directory_path = './TDD_examples/CASE_1_ITER_like_LTS/'
simulation = Simulation(input_directory_path)
simulation.run()