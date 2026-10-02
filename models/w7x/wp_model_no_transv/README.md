# wp_model_no_transv

The full-feature winding-pack model (see `../wp_model_full/README.md`)
without transverse heat conduction: the `transverse_coupling_file` entries
of the six conductors are empty and no `transverse_CONDUCTOR_*.csv` tables
are present. Network, induced losses (n·tau = 58 ms, eddy, hysteresis),
schedules and mesh are identical to the full-feature model. Without the
transverse leak of the staircase front the quench is detected earlier
(≈ 178 s after the 6.28 K step instead of ≈ 1770 s), hence the spatial
snapshot times of this deck (300, 500, 700 s and 811.5 ... 839.5 s around
the dump).

```
.venv/bin/python models/w7x/wp_model_no_transv/run_model.py   # ~1.5 h, 3400 steps
.venv/bin/python models/w7x/wp_model_no_transv/make_figures.py
```
