# Case files

A study is described by JSON files: a **system file** (the network and its components), one or more **scenario
files** (what to run on it), and for the full EMT model a **full-model file** (the physical options). The system and
scenario files are validated against the JSON Schemas in [cases/schema/](../cases/schema) when they are loaded.

Any numeric value may also be written as an arithmetic expression over numbers and the constants `M_PI` / `pi`,
for example `"2.0 * M_PI * 60.0"`; it is read by a small expression reader, never evaluated as code.

## System file

```json
{
  "description": "...",
  "config":      {"mva_base": 100.0, "fn": 60.0, "omega_b": "2.0 * M_PI * 60.0"},
  "Bus":         [{"idx": 1, "name": "BUS1", "Vn": 345.0, "v0": 1.047, "a0": -0.147}],
  "PQ":          [{"idx": "PQ_1", "bus": 3, "p0": 3.22, "q0": 0.024, "Vn": 345.0}],
  "PV":          [{"idx": 30, "bus": 30, "v0": 1.0475, "p0": 2.5, "Sn": 1000.0, "Vn": 16.5,
                   "pmax": 10.0, "pmin": 0.0, "qmax": 6.0, "qmin": -4.0}],
  "Slack":       [{"idx": 31, "bus": 31, "v0": 0.982, "a0": 0.0, "Sn": 700.0, "Vn": 16.5, ...}],
  "Line":        [{"idx": "Line_1", "bus1": 1, "bus2": 2, "r": 0.0035, "x": 0.0411, "b": 0.6987, "tap": 1.0}],
  "Shunt":       [],
  "components":  {"GENROU_1": {"type": "GENROU_PHTRUE", "params": {"bus": 39, "Sn": 10000.0, "xd": 2.0, ...}}, ...},
  "connections": [{"from": "BUS_3.Vd", "to": "CLOAD_3.Vd"}, ...]
}
```

| Table | Fields | Meaning |
|---|---|---|
| `config` | `mva_base`, `fn`, `omega_b`; optional `skip_pf_solve`, `kpf`, `kqf` | system base (MVA), nominal frequency (Hz), base angular frequency; use the stored bus voltages instead of solving the power flow; global frequency dependence of loads |
| `Bus` | `idx`, `name`, `Vn`, `v0`, `a0` | bus id, name, base voltage (kV), stored voltage magnitude (pu) and angle (rad) |
| `PQ` | `idx`, `bus`, `p0`, `q0` | constant-power consumption (pu on `mva_base`, positive drawn from the network) |
| `PV`, `Slack` | `idx`, `bus`, `p0`, `q0`, `v0`, `a0` (slack), `pmax`, `pmin`, `qmax`, `qmin`, `Sn`, `Vn` | generator set-points and limits for the power flow |
| `Line` | `idx`, `bus1`, `bus2`, `r`, `x`, `b`, `tap`, `phi` | series impedance and total charging (pu), off-nominal ratio and phase shift (rad); a branch with `tap != 1` is a transformer |
| `Shunt` | `idx`, `bus`, `g`, `b` | fixed shunt admittance (pu) |
| `components` | name → `{type, params}` | every dynamic model: machines, exciters, governors, loads, converters. `type` is a model type (`GENROU_PHTRUE`, `IEEET1_PHTRUE`, `IEEEG1_PHTRUE`, `IEEEG3_PHTRUE`, `GENSAL_PHTRUE`, `COMPLEXLOAD`, `GFL_PHTRUE`, `GFM_VSM_PHTRUE`, `GFM_DROOP_PHTRUE`, `GFM_VOC_PHTRUE`); `params` holds its parameters. A controller names its machine in `syn` |
| `connections` | `from`, `to` | signal wiring between ports: bus voltages to components, exciter `Efd` to machine, governor `Tm` to machine, and so on |

Keys starting with `_` (for example `"_pf": "G 01"`, the name of the unit in a source model) are comments and are
ignored. The parameters of each model type are listed in its docstring (`?GENROU_PHTRUE`) and in its source file
under `src/components/`.

## Scenario file

```json
{
  "description": "3ph fault at bus 16, cleared after 150 ms",
  "system": "system_phtrue.json",
  "solver": {"method": "ida", "dt": 0.0005, "duration": 6.0, "log_dt": 0.001},
  "events": [{"type": "BusFault", "bus": 16, "r": 0.0001, "x": 0.0001, "t_start": 1.0, "t_duration": 0.15}],
  "output": {"directory": "outputs/IEEE39Bus_PF_base_bus16_150ms"}
}
```

| Key | Meaning |
|---|---|
| `system` | the system file, relative to the scenario file |
| `solver` | `method` (`ida`, `bdf1` or `rk4`), step `dt` and `duration` (s); optional `log_dt`, `rtol`, `atol` |
| `events` | `BusFault`: `bus`, `r`, `x` (pu), `t_start` and either `t_duration` or `t_end` (s; 0.1 s if neither); `LineFault`: `line_idx`, `distance` (fraction from `bus1`, default 0.5), `r`, `x`, and the same times |
| `output` | `directory` for the results |
| `plots` | optional plot requests |

In the EMT simulator, a scenario's bus faults become `PHFault`s (three-phase, cleared at once); unbalanced faults
and current-zero clearing are given as `PHFault` arguments ([EMT simulation](emt_simulation.md#faults)).

## Full-model file

`cases/IEEE39Bus_FullModel/full_model.json` tells `phs_full_model` which physical models to build and with what
values:

| Key | Meaning |
|---|---|
| `system`, `winding_circuits` | the system file, and the machine winding data (per generator bus: `Ll, Lad, Laq, Lfd, L1d, L1q, L2q, Ra, Rfd, R1d, R1q, R2q, H`, system base) |
| `network.loads` | the load form: `parallel`, `series`, `series_all` or `composite` |
| `exciters.dc_shunt` | `n`, `field_ratio`: a DC shunt exciter on every IEEET1 with `KE < 0` |
| `exciters.brushless` | `KC`, `KD`, `sat`, `field_share`: a brushless exciter on every IEEET1 with `KE > 0` |
| `steam` | `paths` (IEEEG1 name → plant path `A` or `C`), `r_v`, `rh_dT`, `tau_n`, `lowflow`, `cold_reheat`, `volumes`, `valve`: physical steam units |
| `hydro` | `Tw` (a number or `"match"`), `Te`, `N`, `fp`: the physical hydro unit and its elastic penstock |

The meaning of every value is in the docstring of `phs_controlled` ([API reference](api/phsim.md#phs_controlled)).

## The cases in this repository

See [cases/README.md](../cases/README.md).
