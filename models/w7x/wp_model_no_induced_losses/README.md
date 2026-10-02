# wp_model_no_induced_losses

The full-feature winding-pack model (see `../wp_model_full/README.md`) with
every induced-loss source switched off: `coupling_loss_time_constant`,
`eddy_loss_geometry_constant` and `filament_diameter` are 0 on all strands,
`eddy_loss_geometry_constant` is 0 on all jackets, and the driver runs with
`--no-induced-losses`. Network, transverse conduction, schedules and mesh are
identical to the full-feature model. The losses act only during the dump, so
the pre-quench phase and the detection time coincide with the full-feature
run; the differences appear in the hot spots of the non-quenched double
layers and in the expelled helium.

```
.venv/bin/python models/w7x/wp_model_no_induced_losses/run_model.py   # ~3 h
.venv/bin/python models/w7x/wp_model_no_induced_losses/make_figures.py
```
