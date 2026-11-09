# Accuracy: PHPS against PowerFactory EMT

These figures compare PHPS's EMT simulation with **DIgSILENT PowerFactory 2022 SP1 in EMT mode** (instantaneous
values), an established commercial simulator, on the same system and the same event. In every figure PowerFactory is
the **black** line, PHPS the **orange dashed** line, and the fault is shaded.

## The test

| | |
|---|---|
| System | IEEE 39-bus (New England): 39 buses, 10 synchronous machines (G 01 is the equivalent of the rest of the interconnection), 46 lines and transformers, 19 loads |
| Event | three-phase fault at bus 16 through 0.001 pu, applied at 0.1 s and removed at 0.2 s; simulated to 1 s |
| Controllers | IEEE Type-1 exciters (IEEET1) and steam and hydro governors (IEEEG1, IEEEG3) in service in both tools, with the same parameters; power-system stabilisers out of service in both |
| PowerFactory | EMT, fixed step 2.5 µs, output every 0.1 ms |
| PHPS | port-Hamiltonian EMT model (`src/phsim/`), implicit midpoint rule at a quarter of its default step (about 1 µs, 0.3 µs during the fault), read at PowerFactory's sample times |

## What was matched

So that the comparison tests the models and not differences in data, both tools simulate the same thing:

- **Machine data.** PowerFactory derives each machine's equivalent circuit (field and damper windings) from its
  standard data; PHPS uses those derived winding values, changed only in base
  (`cases/IEEE39Bus_FullModel/winding_circuits.json`).
- **Loads.** In EMT, PowerFactory models an inductive load as a resistor and an inductor in series; PHPS uses the same
  series R-L form for every load (`loads = :series_all`).
- **Shaft input.** Both tools turn the turbine power the governor sets into shaft torque as power / speed.
- **Fault removal.** PowerFactory's own clear event opens each phase at its next current zero. For this benchmark the
  fault is removed on all three phases at once in both tools (an ideal simultaneous interruption).
- **Signals that need no frame alignment.** Rotor angles are taken relative to G 01; speeds, torques, field currents,
  stator fluxes and voltage magnitudes are compared directly, with only base conversions.

Before the fault the two tools sit at the same operating point: largest difference over all machines and buses
3.5e-5° in rotor angle, 1.2e-6 pu in voltage.

## The figures

| File | What it shows |
|---|---|
| [accuracy_rotor_angle.png](accuracy_rotor_angle.png) | rotor angle of every machine relative to G 01, over the whole run |
| [accuracy_speed.png](accuracy_speed.png) | speed of every machine |
| [accuracy_bus_voltage.png](accuracy_bus_voltage.png) | voltage magnitude at the faulted bus and eleven others, including the electromagnetic ringing at fault inception and removal |
| [accuracy_fault_clearing.png](accuracy_fault_clearing.png) | a 35 ms window around fault removal at 0.1 ms resolution: electrical torques and stator fluxes of three machines, voltages at buses 16 and 17 |

## The differences

Largest difference over all machines or buses:

| Signal | Window | Largest difference | Share of the largest excursion in PowerFactory |
|---|---|---|---|
| Rotor angle (relative to G 01) | after the fault | 0.010° | 0.02 % |
| Speed | after the fault | 1.1e-6 pu | 0.014 % |
| Electrical torque | during the fault (largest / rms) | 0.0040 / 0.0010 pu | |
| Electrical torque | after the fault (largest / rms) | 0.0136 / 0.0008 pu | 0.1 % / 0.006 % |
| Field current | after the fault | 0.0019 pu | |
| Stator flux, d axis | after the fault | 0.0006 pu | |
| Excitation voltage | after the fault | 0.0010 pu | |
| Turbine power | after the fault | 1.6e-5 pu | |
| Bus voltage magnitude | during the fault (largest / rms) | 0.0048 / 0.0006 pu | |
| Bus voltage magnitude | after the fault (largest / rms) | 0.033 / 0.0011 pu | 1.1 % / 0.04 % |

The largest voltage difference sits in the first milliseconds after the fault is removed, where the voltage swings
by several per unit; the peak at the faulted bus is 4.190 pu in PowerFactory and 4.194 pu in PHPS. With the
controllers out of service in both tools, the rotor angles agree within 0.0015°.

## What this does not show

This is one balanced fault on one system. It says nothing here about unbalanced faults, about a real breaker's
current-zero interruption, or about how well either tool represents a real power system.
