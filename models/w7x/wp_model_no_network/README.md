# wp_model_no_network

The winding-pack model without hydraulic network: the same six double-layer
conductors with transverse conduction and induced losses (n·tau = 58 ms,
eddy, hysteresis), but every channel sees the measured supply and return
pressures (5.9503 -> 5.3319 bar) as fixed boundary conditions and 6.1 K
everywhere. No measured schedules are applied; the quench is started by a
2 J heat impulse (200 W/m over 0.1 m for 0.1 s) on the strand of DL1 at
x = 66.7 m. Detection (0.5 V on DL1 at ≈ 0.17 s) and the measured current
dump follow as in the network runs; 15 s are simulated (deck end time 15 s,
tolerance 3e-3, spatial snapshots 0.5 ... 15 s).

```
.venv/bin/python models/w7x/wp_model_no_network/run_model.py   # ~1.3 h, 2400 steps
.venv/bin/python models/w7x/wp_model_no_network/make_figures.py
```

Without network there are no outlet-group temperatures; the figure set
replaces them by the channel outlet temperatures of the six double layers.
