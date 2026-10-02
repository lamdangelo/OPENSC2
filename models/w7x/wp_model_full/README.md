# wp_model_full

Full-feature winding-pack model of the AAB51 quench test: six double-layer
conductors (14000 elements each) with the CEA Saclay hydraulic circuit,
transverse heat conduction through the insulation (`transverse_CONDUCTOR_*.csv`),
induced losses during the dump (coupling n·tau = 58 ms, jacket and strand
eddy currents, NbTi hysteresis), critical-current scale 0.975 and the measured
supply schedules.

```
.venv/bin/python models/w7x/wp_model_full/run_model.py      # ~3 h, 6600 steps
.venv/bin/python models/w7x/wp_model_full/make_figures.py
```

The deck starts on the 6.18 K plateau 1100 s before the measured quench; the
measured step to 6.28 K arrives at t = 1100 + 631 s and DL1 quenches at
t ≈ 2400 s (detection 0.5 V on DL1, dump 0.184 s later, 30 s simulated after
the dump; spatial snapshots at 300 ... 2730 s). Outputs:
`simulation_results/W7X_QT/`, `final_run_log.tsv`, `hotspot_log.tsv`.
