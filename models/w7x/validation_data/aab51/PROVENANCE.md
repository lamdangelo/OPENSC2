# AAB51 measurement subset

Quench test of W7-X coil AAB51 at the CEA Saclay test facility, event of
2007-03-06 14:51:53.57 (fast-acquisition trigger = measured time 0).
Built from the measurement archive `W7-X/quench_test_validation/measurement_data` with
`common/tools/trim_measurement_data.py`.

* `Quench/`: fast event files, window [-1.43, +8.57] s around the trigger,
  numeric TIME column; uppercase `Q` files 0.2 ms, lowercase `q` 10 ms;
  Latin-1 encoded, copied unchanged.
* `Slow/`: slow control archive ('dd/mm/yyyy hh:mm:ss.f value' rows),
  restricted to measured time [-4500, 300] s; header lines
  and the kept rows are byte-identical to the archive.
* `readme.txt`: the sensor list of the archive (covers more sensors than
  packaged here).

| file | role | content |
|---|---|---|
| `Quench/Q070306-14_51_C1_DUP11-3_ai.txt` | compensated detection voltage (1,2)-(3,4), figures | complete |
| `Quench/Q070306-14_51_C1_IAL-C1_ai.txt` | coil current, dump waveform of the controller | complete |
| `Quench/q070306-14_51_C1_TT2100A_ai.txt` | outlet temperature double layer (1,2), figures | complete |
| `Quench/q070306-14_51_C1_TT2101A_ai.txt` | outlet temperature double layers (3,4)+(5,6), figures | complete |
| `Quench/q070306-14_51_C1_TT2102A_ai.txt` | outlet temperature double layers (7,8)+(9,10), figures | complete |
| `Quench/q070306-14_51_C1_TT2103A_ai.txt` | outlet temperature double layer (11,12), figures | complete |
| `Quench/q070306-14_51_C1_PT2111_ai.txt` | supply pressure, drives the supply reservoir | complete |
| `Quench/q070306-14_51_C1_TT2113C_ai.txt` | supply temperature, drives the supply reservoir | complete |
| `Slow/DUP11-3_ai.txt` | slow archive of DUP11-3 | 244 of 163916 rows |
| `Slow/TT2100A_ai.txt` | slow archive of TT2100A | 4355 of 95653 rows |
| `Slow/TT2101A_ai.txt` | slow archive of TT2101A | 4429 of 96522 rows |
| `Slow/TT2102A_ai.txt` | slow archive of TT2102A | 4245 of 93286 rows |
| `Slow/TT2103A_ai.txt` | slow archive of TT2103A | 4358 of 95959 rows |
| `Slow/PT2111_ai.txt` | slow archive of PT2111 | 3478 of 81591 rows |
| `Slow/TT2113C_ai.txt` | slow archive of TT2113C | 4444 of 97206 rows |

Only these sensors are used: PT2111 / TT2113C drive the supply reservoir of
the winding-pack runs, IAL-C1 provides the dump waveform, DUP11-3 and
TT2100A-TT2103A are compared with the simulations in the figures.
