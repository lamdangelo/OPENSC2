"""Simulation facade and composition root.

``Simulation(base_path, step_callback=None).run()`` is the public entry point
used by the GUI, the CLI and the model drivers. The facade reads the deck
through the input adapters, builds the :class:`runtime.context.SimulationContext`
that the domain and the application use cases operate on, registers the
adapters behind the ports (fluid properties, diagnostics, observers) and
delegates the phases of a run to ``application``:

    instantiate_conductors -> folders/metadata -> initialize_conductors ->
    [restore_from_checkpoint] -> run_transient -> post_process -> on_finished

Legacy attribute names (``transient_input``, ``list_of_Conductors``,
``dict_path``, ``dict_qsource``, ``numObj``, ...) proxy to the context so that
drivers and tests written against the old monolithic class keep working;
drivers may still write to ``transient_input`` from their step callback.
"""

from decimal import Decimal
import os
from typing import Union

import opensc2.adapters.input.yaml_input_registry as yaml_input_registry
import opensc2.adapters.output.simulation_paths as simulation_paths
from opensc2.adapters.fluid_properties import coolprop_provider
from opensc2.adapters.input import open_input_source
from opensc2.adapters.input.conductor_loader import load_conductor_definition
from opensc2.adapters.input.environment_loader import load_environment
from opensc2.adapters.output.checkpoint import restore_from_checkpoint
from opensc2.adapters.output.diagnostics_npz import NpzDiagnosticsSink
from opensc2.adapters.output.observers import ConsoleProgressObserver, OutputObserver
from opensc2.adapters.output.simulation_log import SimulationSummary
from opensc2.adapters.plotting.observers import FigureObserver, RealTimePlotObserver
from opensc2.application.build_simulation import initialize_conductors, instantiate_conductors
from opensc2.application.observers import CompositeObserver
from opensc2.application.post_processing import post_process
from opensc2.application.transient_loop import run_transient
from opensc2.ports.fluid_properties import COOLANT_PROPERTY_ALIASES, register_provider_factory
from opensc2.runtime.context import SimulationContext


def significant_time_digits(num: Union[int, float]) -> int:
    """Number of digits used to label save times (derived from STPMIN).
    N.B: quite general but does not work for all cases; see
    https://stackoverflow.com/questions/8101353/counting-significant-figures-in-python
    """
    numstr = str(num)
    decimal_norm = Decimal(numstr).normalize()
    if str(decimal_norm) == "0":
        # Deal with cases "0", "00", "00...0"
        return 0
    if 0.0 < num < 1.0:
        # Not the correct number of significant digits but rather the number
        # of digits to be displayed (0.01 gives 2 rather than 1).
        return abs(decimal_norm.as_tuple().exponent)
    return len(Decimal(numstr).as_tuple().digits)


def _context_property(name, readonly=False):
    def getter(self):
        return getattr(self.context, name)

    def setter(self, value):
        setattr(self.context, name, value)

    return property(getter, None if readonly else setter)


class Simulation:

    def __init__(self, base_path, step_callback=None):
        # Optional per-step hook, called at the top of every transient
        # iteration with the simulation as argument (drivers use it for
        # time-dependent boundary schedules, event detection and source
        # switching). A truthy return value stops the transient.
        self.step_callback = step_callback
        # Composition root: the domain resolves fluid properties through the
        # port; the CoolProp adapter provides them.
        register_provider_factory(coolprop_provider.provider_for)
        # Working directory at construction (not at import) time.
        self.cwd = os.getcwd()
        self.basePath = base_path
        self.starter_file = None
        for f_name in os.listdir(self.basePath):
            if "transitory_input" in f_name:
                self.starter_file = f_name
        # Input source (ports.input_source): YAML when the directory holds
        # simulation.yaml, otherwise the legacy Excel workbooks.
        self.input_source = open_input_source(self.basePath)
        self.yaml_registry = yaml_input_registry.get_registry(self.basePath)
        if self.yaml_registry is not None:
            self.starter_file = yaml_input_registry.SIMULATION_FILE_NAME
        transient_settings = self.input_source.transient_settings()
        self.flag_start = False
        environment = load_environment(
            os.path.join(self.basePath, transient_settings["ENVIRONMENT"])
        )
        # The running simulation as the domain sees it.
        self.context = SimulationContext(
            transient_settings=transient_settings,
            environment=environment,
            base_path=base_path,
            # Default results root, used only when no target directory is
            # given to simulation_folders_manager (GUI flow); headless runs
            # override it with the model directory.
            output_paths=dict(
                Current_work_dir=self.cwd,
                Results_dir=os.path.join(self.cwd, "..", "Simulations_results"),
            ),
            # Where the solvers drop state dumps when a step turns non-finite
            # (ports.diagnostics): npz files in the working directory.
            diagnostics=NpzDiagnosticsSink(),
            n_digit_time=significant_time_digits(transient_settings["STPMIN"]),
        )
        # Property aliases of the coolant (kept as attribute for callers;
        # the domain reads ports.fluid_properties.COOLANT_PROPERTY_ALIASES).
        self.fluid_prop_aliases = COOLANT_PROPERTY_ALIASES
        # Observers and summary are created by run() (the summary's timers
        # start there); phase methods called directly see no observers.
        self.simulation_summary = None
        self.observers = CompositeObserver()

    # -- context proxies (legacy names) ----------------------------------
    transient_input = _context_property("transient_settings")
    transient_settings = _context_property("transient_settings")
    environment = _context_property("environment")
    list_of_Conductors = _context_property("conductors")
    conductors = _context_property("conductors")
    simulation_time = _context_property("simulation_time")
    simulation_time_step = _context_property("simulation_time_step")
    num_step = _context_property("num_step")
    n_digit_time = _context_property("n_digit_time")
    hydraulic_network = _context_property("hydraulic_network")
    dict_qsource = _context_property("qsources")
    dict_path = _context_property("output_paths")
    diagnostics = _context_property("diagnostics")
    contactBetweenConductors = _context_property("conductor_contacts")
    numObj = _context_property("numObj", readonly=True)

    # -- workflow --------------------------------------------------------
    def run(self):
        """Run the simulation workflow."""
        self.simulation_summary = SimulationSummary()  # runtime timers start here
        # Everything observable (files, plots, log, checkpoints, console)
        # happens in observers (ports.observer); the model never writes.
        self.observers = CompositeObserver([
            OutputObserver(self.simulation_summary),
            RealTimePlotObserver(),
            FigureObserver(),
            ConsoleProgressObserver(),
        ])
        self.conductor_instance()  # read input files
        self.simulation_folders_manager(target_directory=self.basePath)  # create folders
        self.save_input_files()  # create metadata
        self.conductor_initialization()  # initialize conductors
        if self.transient_input["RESTART"]:
            # Resume from the newest autosaved checkpoint: the deterministic
            # initialization above rebuilt every object, now overwrite the
            # evolving state and truncate the outputs to the checkpoint time
            # (see adapters/output/checkpoint.py).
            restore_from_checkpoint(self)
            # Re-seed the summary extrema from the restored state; the
            # checkpoint-time hot-spot row is already in the truncated file.
            for cond in self.list_of_Conductors:
                self.simulation_summary.update(self, cond, record_hotspot=False)
        self.conductor_solution()  # solve the numerical problem
        self.conductor_post_processing()  # do post-processing
        self.observers.on_finished(self)  # simulation.log + hot-spot flush

    def conductor_instance(self):
        """Build the conductors from the input deck (application use case)."""
        instantiate_conductors(
            self.context,
            self.input_source.conductor_count,
            lambda counter: load_conductor_definition(
                self.basePath, counter, self.input_source, self.transient_input["MAGNET"]
            ),
            self.input_source.conductor_coupling_dataframe(),
            self.observers,
            observer_target=self,
        )

    def conductor_initialization(self):
        """Initial state of the conductors and the optional network."""
        initialize_conductors(
            self.context,
            self.input_source.hydraulic_network(),
            self.observers,
            observer_target=self,
        )

    def conductor_solution(self):
        """The transient (application use case)."""
        run_transient(
            self.context,
            self.observers,
            observer_target=self,
            step_callback=self.step_callback,
            callback_target=self,
        )

    def conductor_post_processing(self):
        post_process(self.context, self.observers, observer_target=self)

    def simulation_folders_manager(self, target_directory: str = None):
        """Build the whole output-folder tree for this simulation run."""
        simulation_paths.manage_simulation_folders(self, target_directory)

    def save_input_files(self):
        """Save the input files in read-only mode as metadata of the run."""
        simulation_paths.save_input_files(self)
