"""
This module owns simulation output-folder bookkeeping: building the
directory tree for space/time convergence analysis, per-conductor
Initialization/Spatial_distribution/Time_evolution/Solution subfolders and
their Benchmark counterparts, and saving a read-only copy of the input files
as run metadata.

Relocated from ``simulation.py`` as part of splitting the file for single
responsibility: this is pure I/O bookkeeping, analogous to the existing
``utility_functions/output.py`` / ``utility_functions/plots.py`` split, not
simulation orchestration.
"""

import os
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import warnings

import pandas as pd

from conductor.conductor_flags import MethodFlag


def make_directories(simulation: object, list_key_val, exist_ok: bool = False) -> None:
    """Create the folders named by the values in ``list_key_val``.

    Args:
        simulation (object): simulation object.
        list_key_val: list of (dict_path key, path) tuples.
        exist_ok (bool, optional): forwarded to os.makedirs. Defaults to False.
    """
    # Loop to create the folders.
    for ii in range(len(list_key_val)):
        # Create the folders in list_key_val[ii][0].
        os.makedirs(simulation.dict_path[list_key_val[ii][0]], exist_ok=exist_ok)
    # End for ii.

# End function make_directories.

def build_space_convergence_paths(simulation: object, list_folder) -> None:
    """Build the paths for the space convergence analysis."""
    # list_folder = ["output", "figures"]
    str_dir = (
        "TEND_"
        + f"{simulation.transient_input['TEND']}_"
        + "STPMIN_"
        + f"{simulation.transient_input['STPMIN']}"
    )
    # Build list_key_val exploiting list comprehension. List of tuples: index [0] is the key of the dictionary, index [1] is the corresponding value that is the path to Output or Figures sub directories.
    list_key_val = [
        (
            f"Space_conv_{folder}_dir",
            os.path.join(
                simulation.dict_path["Sub_dir"],
                "Space_convergence",
                str_dir,
                folder.capitalize(),
            ),
        )
        for folder in list_folder
    ]
    # Update the dictionary simulation.dict_path with keys and values from the dictionary comprehension.
    simulation.dict_path.update(
        {
            list_key_val[ii][0]: list_key_val[ii][1]
            for ii in range(len(list_key_val))
        }
    )
    # Make the directories invoking function make_directories.
    make_directories(simulation, list_key_val, exist_ok=True)

# End function build_space_convergence_paths.

def build_time_convergence_paths(simulation: object, list_folder) -> None:
    """Build the paths for the time convergence analysis. The value of NELEMS is the one of the first defined conductor."""
    # list_folder = ["output", "figures"]
    str_dir = (
        "TEND_"
        + f"{simulation.transient_input['TEND']}_"
        + "NELEMS_"
        + f"{simulation.list_of_Conductors[0].mesh.number_of_elements}"
    )
    # Build list_key_val exploiting list comprehension. List of tuples: index [0] is the key of the dictionary, index [1] is the corresponding value that is the path to Output or Figures sub directories.
    list_key_val = [
        (
            f"Time_conv_{folder}_dir",
            os.path.join(
                simulation.dict_path["Sub_dir"],
                "Time_convergence",
                str_dir,
                folder.capitalize(),
            ),
        )
        for folder in list_folder
    ]
    # Update the dictionary simulation.dict_path with keys and values from the dictionary comprehension.
    simulation.dict_path.update(
        {
            list_key_val[ii][0]: list_key_val[ii][1]
            for ii in range(len(list_key_val))
        }
    )
    # Make the directories invoking function make_directories.
    make_directories(simulation, list_key_val, exist_ok=True)

    # Path of the directory to save the comparison of the time convergence \
    # with the same method, comparison is by NELEMS (cdp, 11/2020) [seems to not be used, I do not remember the aim of this!! 08/07/2020]
    # simulation.dict_path["Time_conv_comp_METHOD"] = os.path.join(\
    # 																		simulation.dict_path["Main_dir"], "METHOD")
    # Path of the directory to save the comparison of the time convergence \
    # with the same NELEMS, comparison is by METHODS (cdp, 11/2020) [seems to not be used, I do not remember the aim of this!! 08/07/2020]
    # simulation.dict_path["Time_conv_comp_NELEMS"] = os.path.join(\
    # 																		simulation.dict_path["Main_dir"], "NELEMS")

# End function build_time_convergence_paths.

def build_subfolders_paths(simulation: object, list_folder, list_f_names, dict_make, dict_benchmark) -> None:
    """Create sub folders Initialization, Spatial_distribution, Time_evolution and Benchmark in Output and Figures directories; each folder in f_names_list, will contain folder conductor.identifier."""
    # Loop to create sub folders initialization, Spatial_distribution, Time_evolution and Benchmark in Output and Figures directories; each folder in f_names_list, will contain folder conductor.identifier.
    for f_name in list_f_names:
        for conductor in simulation.list_of_Conductors:
            # Build list_key_val exploiting list comprehension. List of tuples: index [0] is the key of the dictionary, index [1] is the corresponding value that is the path to Output or Figures sub directories.
            list_key_val = [
                (
                    f"{folder.capitalize()}_{f_name}_{conductor.identifier}_dir",
                    os.path.join(
                        simulation.dict_path["Sub_dir"],
                        simulation.transient_input["SIMULATION"],
                        folder.capitalize(),
                        f_name,
                        conductor.identifier,
                    ),
                )
                for folder in list_folder
            ]
            # Update the dictionary simulation.dict_path with keys and values from dictionary comprehension, which may be either a mapping or an iterable of key/value pairs. The values of dictionary comprehension take priority when simulation.dict_path0 and other share keys.
            simulation.dict_path.update(
                {
                    list_key_val[ii][0]: list_key_val[ii][1]
                    for ii in range(len(list_key_val))
                }
            )
            # Invocke function warn_if_directories_exist if path exists, make_directories if path does not exist.
            dict_make[os.path.exists(simulation.dict_path[list_key_val[0][0]])](
                simulation, list_key_val
            )
            # Create benchmark directory if f_name is "Spatial_distribution" or "Time_evolution"
            dict_benchmark[f_name](simulation, conductor, list_folder, f_name)
        # End for conductor.
    # End for f_name.

# End function build_subfolders_paths.

def warn_if_directories_exist(simulation: object, list_key_val) -> None:
    """Warn the user that the given output directories already exist."""
    # Da sistemare nella GUI!
    warnings.warn(
        f"Directories\n{simulation.dict_path[list_key_val[0][0]]}\n{simulation.dict_path[list_key_val[1][0]]} already exist.\nProbably you have already performed a simulation with the same input data.\nPlease check and confirm if you want to continue with the simulation or not."
    )

# End function warn_if_directories_exist.

def build_benchmark_paths(simulation: object, conductor, list_folder, f_name) -> None:
    """Create the Benchmark subfolder paths for a given conductor and folder name."""
    # Build list_key_val exploiting list comprehension. List of tuples: index [0] is the key of the dictionary, index [1] is the corresponding value that is the path to Output or Figures sub directories.
    list_key_val = [
        (
            f"{folder.capitalize()}_{f_name}_benchmark_{conductor.identifier}_dir",
            os.path.join(
                simulation.dict_path["Sub_dir"],
                simulation.transient_input["SIMULATION"],
                folder.capitalize(),
                f_name,
                "Benchmark",
                conductor.identifier,
            ),
        )
        for folder in list_folder
    ]
    # Update the dictionary simulation.dict_path with keys and values from dictionary comprehension.
    simulation.dict_path.update(
        {
            list_key_val[ii][0]: list_key_val[ii][1]
            for ii in range(len(list_key_val))
        }
    )
    # Make the directories invoking function make_directories.
    make_directories(simulation, list_key_val, exist_ok=True)

# End function build_benchmark_paths.

def do_nothing(simulation: object, conductor, list_folder, f_name) -> None:
    # Do nothing function, indroduced to use the dictionary dict_benchmark when creating folders.
    pass

def manage_simulation_folders(simulation: object, target_directory: str = None) -> None:
    """Build the whole output-folder tree for a simulation run (space/time
    convergence directories, per-conductor subfolders and their Benchmark
    counterparts, and the read-only input-file save directory)."""
    # Map the integration method to the name of its output subfolder.
    dict_int_method = {
        MethodFlag.BACKWARD_EULER: "BE",
        MethodFlag.CRANK_NICOLSON: "CN",
        MethodFlag.ADAMS_MOULTON_4TH_ORDER: "AM4",
    }
    # Update dictionary simulation.dict_path
    if target_directory is not None:
        simulation.dict_path["Results_dir"] = target_directory  # user-defined target directory for headless mode
    simulation.dict_path["Main_dir"] = simulation.dict_path["Results_dir"]  # Main_Dir never defined, so just set to Results_dir
    simulation.dict_path["Sub_dir"] = os.path.join(
        simulation.dict_path["Main_dir"],
        dict_int_method[simulation.list_of_Conductors[0].inputs.thermohydraulic_method],
    )
    list_folder = ["output", "figures"]
    # Create paths and folders with function build_space_convergence_paths
    build_space_convergence_paths(simulation, list_folder)
    # Create paths and folders with function build_time_convergence_paths
    build_time_convergence_paths(simulation, list_folder)
    # Print a warning if os.path.exists() returns True, build the directories if returns False.
    dict_make = {True: warn_if_directories_exist, False: make_directories}
    list_f_names = [
        "Initialization",
        "Spatial_distribution",
        "Time_evolution",
        "Solution",
    ]
    dict_benchmark = dict(
        Initialization=do_nothing,
        Spatial_distribution=build_benchmark_paths,
        Time_evolution=build_benchmark_paths,
        Solution=do_nothing,
    )
    # Create subfolders path invocking function build_subfolders_paths
    build_subfolders_paths(simulation, list_folder, list_f_names, dict_make, dict_benchmark)

    # Path to save the input files of the simulation in read olny mode as
    # metadata for the simulation itself.
    simulation.dict_path["Save_input"] = os.path.join(
        simulation.dict_path["Sub_dir"],
        simulation.transient_input["SIMULATION"],
        simulation.basePath.split("/")[-1],
    )
    os.makedirs(simulation.dict_path["Save_input"], exist_ok=True)

# End function manage_simulation_folders.

def resolve_file_name_capitalization(base_path: str, file_name: str) -> str:
    """Return the on-disk spelling of file_name inside base_path.

    Input workbooks may spell file names with a different capitalization than
    the file system; if the exact name does not exist but exactly one entry of
    the directory matches case-insensitively, that entry is returned.
    """
    if os.path.exists(os.path.join(base_path, file_name)):
        return file_name
    matches = [
        entry
        for entry in os.listdir(base_path)
        if entry.lower() == file_name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    # Fall through: let the caller raise the informative file-not-found error.
    return file_name


def save_input_files(simulation: object) -> None:
    """Save the input file of the simulation as .xlsx files in read only mode. These files are metadata for the simulation output."""
    load_paths = list()
    save_paths = list()
    filenames = list()

    # Load and save paths for transitory_input file.
    load_transient_input = os.path.join(simulation.basePath, simulation.starter_file)
    load_paths.append(os.path.join(simulation.basePath, simulation.starter_file))
    filenames.append(simulation.starter_file)

    # Load file transitory_input.
    transient_input = pd.read_excel(
        load_transient_input,
        sheet_name="TRANSIENT",
        skiprows=1,
        header=0,
        index_col=0,
    )

    # Load paths of environment_input and conductor_definition files.
    for index in ["ENVIRONMENT", "MAGNET"]:
        load_paths.append(
            os.path.join(simulation.basePath, transient_input.loc[index, "Value"])
        )
        filenames.append(transient_input.loc[index, "Value"])

    # Load file conductor_definition
    conductors = pd.read_excel(
        load_paths[-1],
        sheet_name=None,
        header=0,
        index_col=0,
        skiprows=2,
    )

    # Get all the other input file names.
    # -3 and not -4 since one column of the input file becomes the index of
    # the data frame and should not be considered.
    n_cond = conductors["CONDUCTOR_files"].shape[1] - 3
    # Empty set to store default (necessary or primary) input file names.
    default_files = set()
    # Empty set to store auxiliary input file names.
    aux_files = set()

    files_index_name = conductors["CONDUCTOR_files"].index.to_list()
    for ii in range(n_cond):
        for idx, fname in enumerate(conductors["CONDUCTOR_files"].iloc[:, ii + 3]):
            # Check if file is classified as default or auxiliary (keyword
            # EXTERNAL in variable name).
            if "EXTERNAL" in files_index_name[idx]:
                # Auxiliary input file.
                aux_files.add(fname)
            else:
                # Default input file.
                default_files.add(fname)

    aux_files.discard("none")
    default_files.discard("none")

    del conductors, transient_input

    # Complete load_paths list
    for fname in default_files.union(aux_files):
        # The workbook may spell file names with a different capitalization
        # than the file system (e.g. "EXTERNAL_current.xlsx" vs
        # "external_current.xlsx"); resolve against the directory listing.
        fname = resolve_file_name_capitalization(simulation.basePath, fname)
        load_paths.append(os.path.join(simulation.basePath, fname))
        filenames.append(fname)

    for ii, fname in enumerate(filenames):
        # Build save_paths from load_paths.
        save_paths.append(os.path.join(simulation.dict_path["Save_input"], fname))
        if os.path.exists(save_paths[-1]):
            # Makes the file read/write for the owner if the file
            # already exists
            os.chmod(save_paths[-1], S_IWUSR | S_IREAD)

        if (
            "coupling" in fname
            or "environment_input" in fname
            or "transitory_input" in fname
        ):
            skip_rows = 1
        elif fname in aux_files:
            skip_rows = 0
        else:
            skip_rows = 2

        # Load input file
        dff = pd.read_excel(
            load_paths[ii],
            sheet_name=None,
            header=0,
            index_col=0,
            skiprows=skip_rows,
        )
        # Save input file
        with pd.ExcelWriter(save_paths[-1]) as writer:
            for key, df in dff.items():
                df.to_excel(writer, sheet_name=key)

    # Convert saved files to read only mode.
    for path in save_paths:
        os.chmod(path, S_IREAD | S_IRGRP | S_IROTH)

# End function save_input_files
