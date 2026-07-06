# OPENSC² 2.0

## OPENSC² in a nutshell

Object-oriented software for multiphysics simulations of Superconducting cables.

### Features

OPENSC² is a software for the multi-physical analysis of thermal-hydraulic and electro-dynamic transients in Superconducting Cable-in-Conduit Conductors (CICC) for fusion magnets and power transmission.

Currently it is developed mainly in [Python](https://www.python.org/) but future versions will possibly take advantage of other programming languages such as [TypeScript](https://www.typescriptlang.org/) and [Rust](https://www.rust-lang.org/) as well as the [OpenModelica](https://www.openmodelica.org/) environment.

The software is built based on well-established numerical models and assumptions, re-arranged in an object-oriented framework to be user-friendly and easily manageable through a GUI. The input set can be prescribed either through self-explanatory Excel files or, alternatively, through human-readable YAML files (see [Refactoring and modernization](#refactoring-and-modernization) below).
The **original developing team** includes Prof. L. Savoldi[^1], Prof. F. Freschi, D. Placido[^2], S. Viarengo[^2] @ Dipartimento Energia “Galileo Ferraris” @ [Politecnico di Torino](https://www.polito.it/). Please, contact us at:

* laura.savoldi@polito.it
* fabio.freschi@polito.it
* daniele.placido@polito.it
* sofia.viarengo@polito.it.

[^1]: Head of the [**MAHTEP** research group](http://www.mahtep.polito.it/).
[^2]: PhD students @ the [**MAHTEP** research group](http://www.mahtep.polito.it/).

The **refactoring developing team** includes Dr. Laura D'Angelo and Prof. Felix Warmer @ the Stellarator Reactor Studies research group[^3] at Max Planck Institute for Plasma Physics in Greifswald (Germany). More information about our refactoring and modernization work is found in [Refactoring and modernization](#refactoring-and-modernization). We can be contacted at:

* laura-anna-maria.dangelo@ipp.mpg.de
* felix.warmer@ipp.mpg.de

[^3]: [**Stellarator Reactor Studies (SRS)** research group](https://www.ipp.mpg.de/stellarator-reactor-studies) at the Max Planck Institute for Plasma Physics in Greifswald (Germany).


### Goals

The software is useful for steady state and transient analyses of CICC in operating conditions. It can deal with cables assemled with Low Temperatures (LTS) strands (both Nb3Sn and NbTi), and High Temperature Superconductors (HTS) tapes of different materials. Different coolants can be selected, together with very different cooling configurations. The software is useful to study the steady state operating conditions under environmental parasitic load, as well as transient operation such as: current variation in time, coolant flow variation in time, AC losses, quench, fast discharges, fault currents. The software is useful to assist the research for optimal configurations, subject to a set of constraints, and allows evaluating the temperature margin to current sharing along cables in any pre-defined operating scenarios.

A detailed description of the physics and of the first tests carried out for the initial phase of verification and validation of the software is available [here](https://doi.org/10.1016/j.cryogenics.2022.103457).

## Refactoring and modernization

Starting from the `develop` branch, OPENSC² underwent a substantial refactoring, modernization and performance-optimization effort, carried out by the **SRS group**[^3] at the Max Planck Institute for Plasma Physics in Greifswald (Germany) with the support of [Claude](https://claude.com/product/claude-code) (Anthropic). The full rationale and technical details are documented in [docs/OPENSC2_refactoring_summary.tex](docs/OPENSC2_refactoring_summary.tex); in short, the work covered three threads:

* **Modularization** — the formerly monolithic source tree was split into physics-oriented packages with single-responsibility modules (`components`, `conductor`, `electromagnetics`, `hydraulics`, `thermal`, `physical_fields`, `interfaces`, `utility_functions`, ...), replacing multi-thousand-line files that mixed unrelated concerns.
* **Modernization** — string-keyed dictionaries and Fortran-heritage identifiers were progressively replaced with typed data classes, enumerations and descriptive naming; a new **YAML input format** (schema documented in [docs/yaml_input_key_reference.tex](docs/yaml_input_key_reference.tex)) was introduced as a fully interchangeable, human-readable alternative to the Excel input files, with automatic conversion tooling and bit-identical backward compatibility.
* **Computational efficiency** — sparse-matrix assembly, vectorized finite-element construction, LAPACK-based linear algebra, tabulated fluid properties, and a new adaptive time-integration layer (Galerkin and second-order backward-differentiation time steppers with error-controlled adaptive time stepping) were introduced. On the W7-X quench benchmark (17,000 linear elements) these changes reduced the wall time for a full 25 s transient from several hours to about 19 minutes, while every optimization step was numerically verified against a frozen solver state.

This repository is currently a **fork** of the original MAHTEP OPENSC² project, hosting the above refactoring work. It may be merged back into the original project's branch at some point in the future.


## Get started

Users can benefit from several test cases to check the software functionalities:

1. Heat slug propagation in an ITER TF-like CICC
2. Heat slug propagation in a stacked-HTS slotted-core CICC for fusion applications
3. Steady state operation for a double-cryostat HVDC cable for power transmission

To run a simulation with one of the above test cases, download the repository and install the requirements (more informations in section [Install requirements](user-content-intall-requirements)). After that, you have two possibilities to run the software: 

1. **GUI mode**: You can run _simulation_starter.py_ and from the GUI you can navigate through the folder three until you enter directory _TDD_examples_ and then select one of the three folders contained with pre-compiled inpuput files. In the GUI window select **Add solution path** to select where to save the results (by default they are all collected in the directory _Simulation_results_, that is automatically created if does not already exist). User can create a new folder in this directory or open an existing one: the output (both .tsv files and .eps figures) will be saved in this folder.
2. **Headless mode**: You can run the headless driver Python scripts in the respective folders of the test examples, or in general run these few line of codes:

```
    from simulation import Simulation

    input_directory = './TDD_examples/CASE_1_ITER_like_LTS/'  # adapt path for other examples
    simulation = Simulation(input_directory)
    simulation.run()
```


### Install requirements

The selected Python version is now [3.11](https://www.python.org/downloads/release/python-3110/) (raised from the previous 3.8.10 target as part of the [refactoring](#refactoring-and-modernization)). Dependencies are declared in `pyproject.toml`, at the repository root. To install them, create a virtual environment (suggested name _OPENSC2_) and activate it, then from the repository root run:

    python -m pip install --upgrade pip   # update pip to the latest version
    python -m pip install -e .            # install OPENSC² and its dependencies

The editable install (`-e`) is recommended for running the test cases and developing against the `source_code` tree. If you also want the development tooling (formatting, linting, type checking, profiling), install the `dev` extra instead:

    python -m pip install -e ".[dev]"

Among the dependencies there is [CoolProp](http://www.coolprop.org/) that, according to the operative system you use, may require some other dependences and/or packages. To deal with this, please follow the [documentation](http://www.coolprop.org/coolprop/wrappers/Python/index.html) and [prerequisites](http://www.coolprop.org/coolprop/wrappers/index.html#wrapper-common-prereqs).

The GUI (`OPENSC2_gui.py`) is built with `tkinter`, part of the Python standard library; on some Linux distributions it requires an OS-level package (e.g. `python3-tk`) installed separately, since it is not distributed via pip.

As an alternative, a **headless simulation** is possible, which allows for a pure scripted / command-line execution of OPENSC² without the GUI with a couple of lines of code (see the `headless_driver.py` files in the TDD examples).

## Tests

A regression test suite lives in [tests/](tests/): it runs the TDD examples headless and compares the produced solution and time-evolution outputs against committed reference data within a tight numerical tolerance. TDD example 1 (CASE_1_ITER_like_LTS) runs its full transient in both the YAML and the Excel input format; TDD examples 2 (CASE_2_ENEA_HTS_CICC) and 3 (CASE_3_HTS_HVDC) run shortened variants of their scenarios (the full transients are too long for a CI pipeline). The suite runs automatically on GitHub via [.github/workflows/tests.yml](.github/workflows/tests.yml) for every push and pull request; locally, execute it from the repository root with:

    python -m pytest tests -v


## Help

Software documentation is under development, being the project at its initial stages. Detailed documentation will be provided as soon as an established version of the software is available.
For the time being feel free to send an e-mail to daniele.placido@polito.it if you need any help with your simulations.
Being currently an embryonic software, some of the possibilities provided in the input files may not yet be fully implemented or tested and you may get incorrect results or unexpected errors. A (not exhaustive) list of known issues is available in the [Issue](https://github.com/MAHTEP/OPENSC²/issues) section. To open a new issue, please [follow the procedure](https://github.com/MAHTEP/OPENSC²/blob/main/CONTRIBUTION.md).  
The development team apologizes for the inconvenience and is committed to fixing them as soon as possible.

## Contribution

The developing team wish to receive help form the users in the definition and test of new test cases, in the benchmark against other established software, in the inclusion of other functionalities.
To contribute please refer to [contribution](CONTRIBUTION.md).

## Code of Conduct

The developing team agreed to embrace the [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) **Code of Conduct**.

## License

OPENSC² is licensed under [![AGPL](https://www.gnu.org/graphics/agplv3-with-text-100x42.png)](LICENSE) or any other version of it.
