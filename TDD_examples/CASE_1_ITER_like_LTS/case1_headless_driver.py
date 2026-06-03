"""
A minimal headless (no-GUI) driver to run the simulation of the ITER-like LTS.

Author: Laura D'Angelo
"""

from simulation import Simulation


# Setup 
input_directory_path = './TDD_examples/CASE_1_ITER_like_LTS/'
simulation = Simulation(input_directory_path)


# Run simulation workflow 
simulation.conductor_instance()  # read input files 
simulation.simulation_folders_manager(target_directory=input_directory_path)  # create folders 
simulation.save_input_files()  # create metadata
simulation.conductor_initialization()  # initialize conductors
simulation.conductor_solution()  # solve the numerical problem
simulation.conductor_post_processing()  # do post-processing