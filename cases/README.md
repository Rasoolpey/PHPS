# Cases

Case (system) and scenario JSON. A scenario names its system file, the solver settings, the events and the output.

| Directory | What |
|---|---|
| `IEEE39Bus_FullModel/` | The full physical, three-phase EMT model of the IEEE 39-bus system (`README.md` inside) |
| `IEEE39Bus_PF/` | IEEE 39-bus matched to the PowerFactory project: synchronous machines with IEEE controllers; fault and CCT scenarios |
| `IEEE39Bus_PF_gfl*/` | The same with grid-following converters (current source, and with interface impedance) |
| `IEEE39Bus_PF_gfm-vsm*/`, `IEEE39Bus_PF_gfm-droop*/`, `IEEE39Bus_PF_gfm-voc*/` | The same with grid-forming converters; the `-pf` variants follow PowerFactory's converter placement |
| `Bench3_RMS/` | A three-machine benchmark |

`schema/` holds the JSON Schemas (`system.schema.json`, `scenario.schema.json`) that `load_case` and `load_scenario`
validate against. The format is the one of the Python PHPS (versions 1 and 2), so its cases load unchanged.

A case variant is a new file; existing files are not edited by hand.
