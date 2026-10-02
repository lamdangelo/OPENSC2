# adiabatic_model

Single-conductor (1D) W7-X model of the THEA benchmark case: one double
layer (STR_MIX_1 NbTi/Cu strand bundle, Al6063 jacket, insulation) at 4.1 K
and 15320 A, helium at 3.9 ... 3.94 K, 17000 elements with 14450 of them in
the refined zone 60-100 m, a 2 J heat impulse (200 W/m over 0.1 m for 0.1 s)
at 80.0-80.1 m and the W7-X current dump prescribed in `external_current.csv`
(detection delay 0.895 s). The magnetic field scales with the current. No
hydraulic network, no transverse conduction, no induced losses; the deck
runs to 10 s with the BDF2 stepper and error-controlled time steps.

```
.venv/bin/python models/w7x/adiabatic_model/run_model.py     # ~20 min
.venv/bin/python models/w7x/adiabatic_model/make_figures.py
```

`make_figures.py` plots the strand temperature at the heated zone (probe at
80.24 m) against the THEA hot spot of
`../validation_data/thea/high_fidelity_data_thermal.csv` and their relative
difference (`figures/hot_spot_comparison_thea_opensc.eps`,
`figures/relative_difference_thea_opensc.csv`).
