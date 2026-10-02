# THEA reference data

Results of the CryoSoft THEA code used as reference for the OPENSC2 models of
this package. Files are plain CSV with a header row; times in s, temperatures
in K, pressures in Pa, mass flows in kg/s, currents in A.

| file | content | used by |
|---|---|---|
| `high_fidelity_data_thermal.csv` | strand, jacket and epoxy temperature at the hot spot of the W7-X heat-impulse case (4.1 K, 15320 A, 2 J at 80.0-80.1 m, prescribed current dump), 0-25 s | `adiabatic_model/make_figures.py` |
| `high_fidelity_data_hydraulic.csv` | helium temperature, mass flow, pressure and velocity at the hot spot of the same run | for reference |
| `high_fidelity_data_electric.csv` | coil current, resistance and voltage of the same run | for reference |
| `cea_data_thermal.csv` | strand, jacket and epoxy temperature of the 1D THEA model of the AAB51 quench test (single double layer, initial temperature 6.295 K), on the measured event-time axis | WP figure scripts ("adiabatic" outlet-1 trace) |
| `cea_data_hydraulic_01m.csv` | helium state at the 0.1 m station of the same run | for reference |


