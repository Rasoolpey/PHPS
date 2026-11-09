# EMT simulator

The port-Hamiltonian EMT model: system assembly, machines, exciters, steam and hydro units, converters, lines, transformers, motors, the zero-sequence network, fault clearing, the implicit midpoint integrator and the energy account. Source: [src/phsim/](../../src/phsim).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`phs_resolve_clearing`](#phs_resolve_clearing) | `clearing.jl` |
| [`PHControlled`](#phcontrolled) | `control.jl` |
| [`phs_control_inputs`](#phs_control_inputs) | `control.jl` |
| [`phs_modes`](#phs_modes) | `control.jl` |
| [`phs_control_signals`](#phs_control_signals) | `control.jl` |
| [`phs_fixed_units`](#phs_fixed_units) | `control.jl` |
| [`phs_controlled`](#phs_controlled) | `control.jl` |
| [`PHConverterDC`](#phconverterdc) | `converter.jl` |
| [`PHVsm`](#phvsm) | `converter.jl` |
| [`vsm_divider`](#vsm_divider) | `converter.jl` |
| [`phs_converter_signals`](#phs_converter_signals) | `converter.jl` |
| [`converter_dc_powers`](#converter_dc_powers) | `converter.jl` |
| [`PHDroop`](#phdroop) | `converter.jl` |
| [`droop_output`](#droop_output) | `converter.jl` |
| [`PHVoc`](#phvoc) | `converter.jl` |
| [`voc_rates`](#voc_rates) | `converter.jl` |
| [`voc_sample!`](#voc_sample) | `converter.jl` |
| [`PHGfl`](#phgfl) | `converter.jl` |
| [`gfl_call`](#gfl_call) | `converter.jl` |
| [`gfl_sample!`](#gfl_sample) | `converter.jl` |
| [`gfl_emf`](#gfl_emf) | `converter.jl` |
| [`gfl_rates!`](#gfl_rates) | `converter.jl` |
| [`gfl_initial`](#gfl_initial) | `converter.jl` |
| [`PHShuntExciter`](#phshuntexciter) | `exciter.jl` |
| [`PHBrushlessExciter`](#phbrushlessexciter) | `exciter.jl` |
| [`phs_full_model`](#phs_full_model) | `fullmodel.jl` |
| [`PHHydroUnit`](#phhydrounit) | `hydro.jl` |
| [`hydro_Tw_match`](#hydro_tw_match) | `hydro.jl` |
| [`hydro_initialise`](#hydro_initialise) | `hydro.jl` |
| [`PHFault`](#phfault) | `midpoint.jl` |
| [`phs_scales`](#phs_scales) | `midpoint.jl` |
| [`phs_step_rule`](#phs_step_rule) | `midpoint.jl` |
| [`PHRun`](#phrun) | `midpoint.jl` |
| [`phs_simulate`](#phs_simulate) | `midpoint.jl` |
| [`phs_simulate_bdf`](#phs_simulate_bdf) | `midpoint.jl` |
| [`phs_write_run`](#phs_write_run) | `midpoint.jl` |
| [`phs_read_run`](#phs_read_run) | `midpoint.jl` |
| [`PCCSplit`](#pccsplit) | `pccmanifold.jl` |
| [`pcc_Q`](#pcc_q) | `pccmanifold.jl` |
| [`pcc_rot`](#pcc_rot) | `pccmanifold.jl` |
| [`pcc_split3`](#pcc_split3) | `pccmanifold.jl` |
| [`pcc_qappend`](#pcc_qappend) | `pccmanifold.jl` |
| [`pcc_protected`](#pcc_protected) | `pccmanifold.jl` |
| [`pcc_pod`](#pcc_pod) | `pccmanifold.jl` |
| [`PCCReduced`](#pccreduced) | `pccmanifold.jl` |
| [`pcc_reduced`](#pcc_reduced) | `pccmanifold.jl` |
| [`pcc_phi`](#pcc_phi) | `pccmanifold.jl` |
| [`pcc_project`](#pcc_project) | `pccmanifold.jl` |
| [`pcc_rhs`](#pcc_rhs) | `pccmanifold.jl` |
| [`pcc_power`](#pcc_power) | `pccmanifold.jl` |
| [`steam_volumes_from_time_constants`](#steam_volumes_from_time_constants) | `steam.jl` |
| [`PHSteamUnit`](#phsteamunit) | `steam.jl` |
| [`steam_initialise`](#steam_initialise) | `steam.jl` |
| [`SteamTables`](#steamtables) | `steamprops.jl` |
| [`steam_tables`](#steam_tables) | `steamprops.jl` |
| [`steam_sat`](#steam_sat) | `steamprops.jl` |
| [`steam_state`](#steam_state) | `steamprops.jl` |
| [`steam_h_isentropic`](#steam_h_isentropic) | `steamprops.jl` |
| [`PHMachine`](#phmachine) | `system.jl` |
| [`PHCut`](#phcut) | `system.jl` |
| [`PHConverter`](#phconverter) | `system.jl` |
| [`PHSystem`](#phsystem) | `system.jl` |
| [`PHInput`](#phinput) | `system.jl` |
| [`ph_system`](#ph_system) | `system.jl` |
| [`phs_state_names`](#phs_state_names) | `system.jl` |
| [`phs_machine_signals`](#phs_machine_signals) | `system.jl` |
| [`phs_flows`](#phs_flows) | `system.jl` |
| [`phs_rhs!`](#phs_rhs) | `system.jl` |
| [`phs_rhs`](#phs_rhs) | `system.jl` |
| [`phs_energy`](#phs_energy) | `system.jl` |
| [`phs_gradient`](#phs_gradient) | `system.jl` |
| [`phs_matrices`](#phs_matrices) | `system.jl` |
| [`phs_ledger`](#phs_ledger) | `system.jl` |
| [`phs_network_mask`](#phs_network_mask) | `system.jl` |
| [`phs_terminal_voltage`](#phs_terminal_voltage) | `system.jl` |
| [`phs_frame`](#phs_frame) | `system.jl` |
| [`phs_rotate_network`](#phs_rotate_network) | `system.jl` |
| [`phs_cut_voltage`](#phs_cut_voltage) | `system.jl` |
| [`phs_phases`](#phs_phases) | `system.jl` |
| [`phs_equilibrium`](#phs_equilibrium) | `system.jl` |
| [`phs_rest_residual`](#phs_rest_residual) | `system.jl` |
| [`phs_core_signals`](#phs_core_signals) | `system.jl` |
| [`PHCoreCurve`](#phcorecurve) | `transformer.jl` |
| [`core_curve_gmd`](#core_curve_gmd) | `transformer.jl` |
| [`PHCore`](#phcore) | `transformer.jl` |
| [`core_solve`](#core_solve) | `transformer.jl` |
| [`core_energy`](#core_energy) | `transformer.jl` |
| [`core_dgrad`](#core_dgrad) | `transformer.jl` |

---

### phs_resolve_clearing

*[src/phsim/clearing.jl:34](../../src/phsim/clearing.jl#L34)*

    phs_resolve_clearing(s, x0, u; faults, tstop, kw...) -> (faults, report)

Locate the opening of every phase of the `clearing = :current_zero` faults (reviews 82-82d), in time order over all
faults: a detection run (the unresolved phases closed) stops at the first step end where a phase current changes sign
after max(t_off, the last located opening); every phase crossing in that step is refined by a safeguarded regula falsi
(Illinois) on runs to tstop = t, to |i| <= 1e-9 + 1e-7 x the phase's peak; the earliest is opened, and every other
phase of that fault whose current after the opening is within its tolerance opens with it. On an event grid (travelling-
wave lines, sampled controllers, `grid`) the phase opens at the first grid point at or after the detection step's end
(status :grid, its chopped left-limit current and an estimate Delta x the largest step secant slope within continuous
pieces reported, the jumps at fault switchings listed separately; no estimate with sampled controllers, whose inputs
jump at every sample). `cache = false` re-runs every evaluation (a check that the cache never changes the result). `kw` are
`phs_simulate`'s stepping keywords (`stops, hscale, h, grid, tol, per_period, maxiter`). The report has one entry per
phase opening.

---

### PHControlled

*[src/phsim/control.jl:63](../../src/phsim/control.jl#L63)*

    PHControlled

A `PHSystem` with exciters and governors. Per machine: `exc` (the unchanged IEEET1 component as
a control law, a physical `PHShuntExciter` or `PHBrushlessExciter`, or `nothing`) and `gov` (the unchanged RMS
component as a control law, a physical `PHSteamUnit`, or `nothing`), their first state indices `xoff`, `goff` (0 if none),
their set-points `Vref`, `Pref`, the factor `efd_gain = Rfd/Lad` from `Efd` to the field voltage
and the number of identical units the governor's output is multiplied by. `u0` holds the fixed
inputs (turbine power, field voltage) used where a machine has no controller. `conv`, `coff`: the
converters' controllers (`PHVsm`), which set the converters' EMFs, and their first state indices.

---

### phs_control_inputs

*[src/phsim/control.jl:171](../../src/phsim/control.jl#L171)*

    phs_control_inputs(cs, x, gf; cf = nothing) -> PHInput

The inputs of the physical model at the state `x`: shaft power and field voltage per machine
(the fixed `cs.u0` values where there is no controller), with the fault conductances `gf` and
the constant-power shaft law. The shaft power is the turbine power, less what a physical
exciter takes from the shaft; a physical exciter's field voltage is its generated voltage. `cf` is the frame phasor
of the stationary frame (`PHInput.cf`), passed on unchanged; ignored in the synchronous frame.

---

### phs_modes

*[src/phsim/control.jl:346](../../src/phsim/control.jl#L346)*

    phs_modes(cs, x, u) -> Matrix{Vector{Bool}}

Every limiter decision of the controllers' step kernels at the state `x`: entry `[j, 1]` for
machine `j`'s exciter, `[j, 2]` for its governor (empty where there is none), in the order the
kernel takes them.

---

### phs_control_signals

*[src/phsim/control.jl:493](../../src/phsim/control.jl#L493)*

    phs_control_signals(cs, x) -> (; Efd, Pm, vf, Pexc)

Per machine at the state `x`: the exciter output `Efd` (per unit, `Lad i_fd` at rest), the
turbine power `Pm` (system base, all units), the field voltage `vf` and the power a physical
exciter takes from the shaft, `Pexc` (zero otherwise). `Efd` is `vf Lad/Rfd` for a machine
without an exciter.

---

### phs_fixed_units

*[src/phsim/control.jl:522](../../src/phsim/control.jl#L522)*

    phs_fixed_units(cs::PHControlled) -> (; steam, hydro_Tw)

The turbine hardware of a built model, for `phs_controlled(...; fixed_units)`: every physical steam
unit by its IEEEG1's name, and the water time of its physical hydro unit (the entry omitted if there is
none; an error if there is more than one).

---

### phs_controlled

*[src/phsim/control.jl:537](../../src/phsim/control.jl#L537)*

    phs_controlled(eq::EquilibriumResult; loads = :parallel, circuits = nothing,
                   dc_exciter = nothing, brushless = nothing) -> (cs, x, u, report)

The model of `phs_equilibrium` (with the constant-power shaft law) and the case's exciters and
governors attached as control laws, at its equilibrium. Each controller is the case's own
component, found through its `syn` parameter and initialised by `init_from_targets` at its
machine's steady state (`Efd = Lad i_fd`, turbine power, generator-bus voltage); that also gives
its set-point. Identical units on one bus share one exciter and one governor, the governor's
output multiplied by the number of units; there must be exactly one controller per unit and
their parameters must be equal (the units are already required to be identical, they see the
same signals, and `Efd = Lad i_fd` does not depend on how many units are combined). `u`
carries the fault conductances for `phs_simulate`. `report` adds to `phs_equilibrium`'s the
residual of the whole vector field, each machine's controller names and the names of the
physical exciters.

With `dc_exciter = (; n, field_ratio)` every machine whose IEEET1 has a negative `KE` (a
self-excited shunt field, by the 1968 committee report) gets a physical `PHShuntExciter` in
place of the control law. Both numbers are ones no source gives and have no default: `n`, the
base of the regulator output over the base of the exciter voltage (a number, or a dictionary by
IEEET1 name), and `field_ratio`, the shunt field circuit's resistance over the machine's field
resistance. The operating point of the physical model is unchanged: the exciter starts at the
flux that gives the same field voltage, and the turbine power is raised by what the exciter
takes from the shaft.

With `brushless = (; KC, KD, sat, field_share)` every machine whose IEEET1 has a positive `KE`
gets a physical `PHBrushlessExciter` (an alternator with a rotating rectifier, its regulator
fed from the shaft) in place of the control law; the keywords are those of its constructor and
none has a default. Its operating point is in the rectifier's first mode, which needs
`KC / (1 + KC/sqrt(3)) <= sqrt(3)/4`.

With `steam = (; paths, r_v, rh_dT, tau_n, lowflow, cold_reheat, volumes, tables)` every machine
whose IEEEG1 is named in `paths` (a dictionary from the IEEEG1's name to `:A` or `:C`) gets a
physical `PHSteamUnit` in place of the IEEEG1 law (docs/active/PH_STEAM_UNIT_MODEL.md): IEEEG1's
signal law and servo, and the published steam unit of that path scaled to the IEEEG1's `PMAX`. The
other keywords are the constructor's, none with a default (`cold_reheat` is needed if any unit is on
path A, `volumes` if any is on path C; `tables` defaults to `steam_tables()`). `volumes` is
`:case` (every vessel from its IEEEG1's time constants, record 16.6), `:fit` (fitted to the case's
IEEEG1 turbine, record 16.7), or `(; A = ..., C = ...)` per
path (each `nothing`, `:case` or the constructor's named tuple), or, as before, path C's named tuple
alone (path A then keeps Vahidi's volumes). `valve` (`:characteristic` by default, or `:opening`) is
the constructor's. The unit is initialised at the machine's turbine power (per identical unit):
storages and reheater wall temperatures in steady state; the servo state is the flow demand (the
opening with `valve = :opening`), `Pref` from it.

With `hydro = (; Tw)` every machine with an IEEEG3 gets a physical `PHHydroUnit` in its place
(docs/active/PH_HYDRO_UNIT_MODEL.md): IEEEG3's governor rows, a penstock with the water starting time `Tw`
(s, at full gate) and the ideal turbine; `Tw = :match` takes the value with which the unit's
linearisation at its operating point equals the case's IEEEG3 (`hydro_Tw_match`).

With `fixed_units = (; steam = Dict(IEEEG1 name => PHSteamUnit), hydro_Tw = value)` (either entry may be
left out; `phs_fixed_units` collects both from a built model) the turbines' hardware is taken as given
instead of being built at this operating point: each named steam unit is a copy of the given unit
(its volumes and valve characteristic kept, only its operating state solved at the new power:
`steam_initialise(...; rebuild = false)`), and the hydro unit uses the given water time. Without it,
`volumes = :fit`, the valve characteristic and `Tw = :match` are all set at this operating point, so a
model built at another operating point is a different plant.

---

### PHConverterDC

*[src/phsim/converter.jl:37](../../src/phsim/converter.jl#L37)*

    PHConverterDC

The physical DC side and bridge of a converter (record section 12; S8.3), system base: the battery `Eb` behind `Rb`, the
DC-link capacitor `c` (`(c/omega_b) v_dc' = i_b - G v_dc - Re(m conj(i))`, `i` the injected current), the averaged lossless
bridge `u_x = m v_dc` with the radial limit `|m| <= m_max` (SVPWM `2/sqrt(3)` on the DC base `2 sqrt(2/3) V_LL`), and how
the modulation follows the controller: `:held` (sampled with the controller, `m = clip(u_c / v_dc)` at a call, held in
the alpha-beta frame) or `:continuous` (the command `u_c` held, `m = clip(u_c / v_dc)` at every instant: while unclipped
the AC side is the ideal source's). `Sr` (the rating, system per unit), `H` (s), `v_dc0` are the selected data
it was made from.

The battery may carry RC branches in series behind `Rb` (record section 32, part A: the measured BESS of Sossan et al. 2016,
`E` behind `R_s` and three RC branches): `R` and `cb` (each branch's resistance and its capacitance in the DC link's
convention, `(cb_k/omega_b) v_k' = i_b - v_k/R_k`), so `i_b = (Eb - sum v_k - v_dc)/Rb`. With none it is `Eb` behind `Rb`.

---

### PHVsm

*[src/phsim/converter.jl:78](../../src/phsim/converter.jl#L78)*

    PHVsm

The controller of one converter (all its identical units, system base): `Ta`, `Dp`, `omega_c`,
`omega_n`, `Tv`, the series reactance `Zs`, the virtual impedance's `i_lim`, `r_vi`, `x_vi`, `kpr`,
`kpx`, and the set-points `p_set`, `u_set`. States `theta, omega, omega_f, u_mag`; sampled (`Ts > 0`) also
the held output (the command, or with a physical DC side `dc` in mode `:held` the modulation) and with `dc` the DC-link
voltage.

---

### vsm_divider

*[src/phsim/converter.jl:150](../../src/phsim/converter.jl#L150)*

    vsm_divider(g, E, alpha, w) -> (; uc, ip, on)

The predictive virtual-impedance divider with the terminal voltage `v_k = alpha u_c + w`
eliminated (file header): the EMF `uc`, the predicted current `ip` and whether the virtual
impedance is engaged (`|ip| > i_lim`).

---

### phs_converter_signals

*[src/phsim/converter.jl:186](../../src/phsim/converter.jl#L186)*

    phs_converter_signals(cs, x, u) -> (; omega, theta, p, q, i, v, uc, ip, on, icr, ici, ...)

Per converter at the state `x` (inputs `u` for the fault conductances), system base, all units:
speed, angle (rad, simulator frame), active and reactive power `v_k conj(i_c)`, the injected current `icr + j ici`
(network frame), current and
terminal-voltage magnitudes, the EMF's magnitude, the predicted current's magnitude and whether
the virtual impedance is engaged. For a sampled converter (`Ts > 0`) the EMF is the held output, and the
predicted current and engagement are the divider at the present state, not the call that made the held
output; its `theta` is the angle prepared for the next call (C3, findings 6, 10). With a physical DC side also
`vdc`, the modulation's magnitude `m` and the battery current `ib` (NaN otherwise).

---

### converter_dc_powers

*[src/phsim/converter.jl:315](../../src/phsim/converter.jl#L315)*

    converter_dc_powers(g, xc, ic, omega_b) -> (; i_b, P_x, energy, loss, port, tport)

The physical DC side's account at its states `xc` and injected AC current `ic` (system base): the battery current,
the bridge power `P_x = v_dc Re(m conj(ic)) = Re(u_x conj(ic))`, the stored energy `(c v_dc^2 + sum cb_k v_k^2) / (2 omega_b)`
(the DC link and the battery's branches), the losses `Rb i_b^2 + sum v_k^2 / R_k + G v_dc^2` and the net port `Eb i_b - P_x`.

---

### PHDroop

*[src/phsim/converter.jl:387](../../src/phsim/converter.jl#L387)*

    PHDroop

The droop controller of one converter (all its identical units, system base). States `theta, p_f, q_f, u_mag, vm_d,
vm_q, im_d, im_q` (the filter states in the controller's frame), sampled (`Ts > 0`) also the held output `uc_Re, uc_Im`.

---

### droop_output

*[src/phsim/converter.jl:476](../../src/phsim/converter.jl#L476)*

    droop_output(g, xc) -> (; uc, vmeas, imeas, boost, stage1, stage2)

The droop controller's output (the DLL's `Model_Outputs`) from its states alone.

---

### PHVoc

*[src/phsim/converter.jl:565](../../src/phsim/converter.jl#L565)*

    PHVoc

The virtual-oscillator controller of one converter (all its identical units, system base), sampled (`Ts > 0`). States
`v_Re, v_Im` (the oscillator's voltage, simulator frame), `uc_Re, uc_Im` (the held output, turning at `-omega_b`; with a
physical DC side in mode `:held` the held modulation `m_Re, m_Im`), with `dc_reject` the filter state `dcrV_d, dcrV_q` and
`vmag_prev` (|v| at the previous call), and with a physical DC side `dc` (`PHConverterDC`, record section 32 part A) the DC-link
voltage and the battery's branch voltages. Between calls `v` holds the value for the
next call (GPT V3, items 4-5): the speed and the divider's predicted current recorded between calls are taken at that state, not at
the call that made the held output (as for the sampled VSM, record section 10).

---

### voc_rates

*[src/phsim/converter.jl:632](../../src/phsim/converter.jl#L632)*

    voc_rates(g, xc, vk, ic) -> (; dv, iref, ifilt, Z)

The oscillator's derivative (without the rotation at `omega_n`, applied in the step) at the states `xc`, the measured
terminal voltage `vk` and injected current `ic`; the current reference, the current the oscillator reads, and with
`dc_reject` the filter's input `Z` (the current in the frame of `v`).

---

### voc_sample!

*[src/phsim/converter.jl:653](../../src/phsim/converter.jl#L653)*

    voc_sample!(xc, g, vk, ic, omega_b)

One call of the DLL: the held output from the divider on the oscillator's voltage before the call and the measured `vk`
(with a physical DC side in mode `:held`, the modulation: the command over the DC voltage sampled at the call, clipped); then
the oscillator's Euler step with the DLL's rotation, and the filter's update.

---

### PHGfl

*[src/phsim/converter.jl:703](../../src/phsim/converter.jl#L703)*

    PHGfl

The grid-following controller of one converter (all its identical units, system base), sampled (`Ts > 0`). States: the DLL's
`theta` (the PLL angle in the simulator's frame), `phi`, `v_f`, `f_f`, `p_r`, `q_r`, `gamma_P`, `gamma_Q`, `id_l`, `iq_l`; the held
frame `ch_Re, ch_Im` and reference `ir_d, ir_q`; the static generator's PI integrators `z_d, z_q` (record section 38); the DLL's
monitor `omega` (per unit of `omega_n`) from the last call, held as the DLL holds its outputs (record section 40: recomputed from
the state between calls it reads the voltage after the call's frame reset, 9e-4 pu off the PLL's frequency).

---

### gfl_call

*[src/phsim/converter.jl:787](../../src/phsim/converter.jl#L787)*

    gfl_call(g, xc, vk) -> (; d, vd, vq, omega, sat, gate, idc, iqc)

The DLL's `Model_Outputs` at the states `xc` and the terminal voltage `vk` (simulator frame): the rates `d` of the ten DLL
states (`d[1]` is the PLL's `omega` in rad/s; the angle's step in the simulator's frame subtracts `omega_b`) and the monitors.

---

### gfl_sample!

*[src/phsim/converter.jl:821](../../src/phsim/converter.jl#L821)*

    gfl_sample!(xc, g, vk, omega_b)

One call of the DLL: the held frame (with `g.advance > 0`, its angle advanced by `advance * omega * Ts`, record section 42) and
reference latched from the states before the call, then the Euler step (`phi`
clamped to `+-phi_max`, the angle in the simulator's frame). The static generator's integrators are not touched.

---

### gfl_emf

*[src/phsim/converter.jl:841](../../src/phsim/converter.jl#L841)*

    gfl_emf(g, xc, ic) -> u_c

The static generator's EMF (TechRef 7.1, eqs. 81-85) at the states `xc` and the injected current `ic`.

---

### gfl_rates!

*[src/phsim/converter.jl:853](../../src/phsim/converter.jl#L853)*

    gfl_rates!(dx, g, xc, ic, omega_b)

The rows between calls: the DLL's states and the held reference constant, the held frame turning at `-omega_b`, the static
generator's PI integrators on the error in the held frame.

---

### gfl_initial

*[src/phsim/converter.jl:882](../../src/phsim/converter.jl#L882)*

    gfl_initial(g, vk, ic, E0) -> x

The DLL's `Model_Initialize` at the terminal voltage `vk` (simulator frame, t = 0): `theta = arg vk`, `phi = 0`, `v_f = |vk|`,
`f_f = 1`, the ramps at the set-points, `id0 = P*/|vk|`, `iq0 = -Q*/|vk|` (limited), the outer PI pre-charged, the lag at the gated
command; the held frame and reference as the first call would latch them; the static generator's integrators such that
`u_c = E0` at `ic` (G2 item 7: the DLL's start, not the sampled system's settled orbit).

---

### PHShuntExciter

*[src/phsim/exciter.jl:53](../../src/phsim/exciter.jl#L53)*

    PHShuntExciter

A self-excited DC exciter with an ideal armature on its machine's shaft (file header). `KA`,
`TA`, `KE`, `TE`, `KF`, `TF`, `VRMAX`, `VRMIN` and the saturation constants `A`, `B` are the
case's IEEET1 constants; `n` and `rF` are the two numbers no source gives; `c` and `b` are
derived (`x_E = c phi_f`, amplifier volts `b VR`). Local states: `phi_f, xa, xf`.

    PHShuntExciter(e::IEEET1_PHTRUE, Rfd, Lad; n, rF)

The physical exciter with the constants of the case's IEEET1 `e`, on a machine with field
resistance `Rfd` and mutual inductance `Lad`. Refused: an input filter (`TR > 0`), a time
constant or gain that is not positive, data that are not finite, a negative saturation
constant, and a field characteristic that is not increasing for every flux
(`1 + n (KE + A) <= 0`: the odd curve's slope is least at zero flux; the refusal asks for a
positive slope everywhere). For a machine that
stands for several identical units, `Rfd` and `Lad` are the combined machine's: `c` and `b` do
not change, and the field current, energy and loss are those of all the exciters together.

---

### PHBrushlessExciter

*[src/phsim/exciter.jl:212](../../src/phsim/exciter.jl#L212)*

    PHBrushlessExciter

A brushless exciter on its machine's shaft (comment above). `KA`, `TA`, `KE`, `TE`, `KF`, `TF`,
`VRMAX`, `VRMIN` are the case's IEEET1 constants; `KC`, `KD` the rectifier's loading factor and
the demagnetising factor; `A`, `B` the alternator's no-load saturation; `g` the field circuit's
power per unit of `VR^2`; `Lad` and `gain = Rfd/Lad` the machine's. Local states:
`x_E, xa, xf`.

    PHBrushlessExciter(e::IEEET1_PHTRUE, Rfd, Lad; KC, KD, sat, field_share)

The physical exciter with the regulator, stabiliser and time constant of the case's IEEET1 `e`,
on a machine with field resistance `Rfd` and mutual inductance `Lad`. `sat = (A, B)` is the
alternator's no-load saturation function `A exp(B x)` (`(0, 0)` for none; the case's own
saturation constants are a Type 1 curve and are not used). `field_share` is the exciter's field
loss over the power delivered to the machine's field, both at one per unit, at rated speed,
unsaturated and without load: `g = field_share Rfd / (Lad KE)^2`. Refused: an input filter,
`KE` not positive, a gain or time constant that is not positive, data that are not finite or
have the wrong sign, and a field characteristic that is not increasing.

---

### phs_full_model

*[src/phsim/fullmodel.jl:5](../../src/phsim/fullmodel.jl#L5)*

    phs_full_model(dir; scenario = "fault_bus16_3ph.json") -> (; cs, x0, u, eq, sc, case, faults, spec)

The full model of the case folder `dir`: `phs_controlled` with the physical exciters (DC shunt where the IEEET1 has
`KE < 0`, brushless where `KE > 0`), the physical steam units and the physical hydro unit as `full_model.json`
declares them, the loads as declared, PowerFactory's machine winding data from `winding_circuits.json`. `x0` is the
operating point, `faults` the scenario's bus faults as `PHFault`s.

---

### PHHydroUnit

*[src/phsim/hydro.jl:41](../../src/phsim/hydro.jl#L41)*

    PHHydroUnit

A physical hydro unit in place of the case's IEEEG3 (file header): the IEEEG3's parameters (its
governor rows are kept), the full gate `GMAX` on the system base and the water starting time `Tw` at
full gate. Local states `xp, xr, at` (IEEEG3's pilot valve, droop reset and gate) and `q` (the penstock
flow, per unit of full-gate flow).

    PHHydroUnit(gov::IEEEG3_PHTRUE; Tw)

The physical hydro unit with the governor of the case's IEEEG3 `gov` and water starting time `Tw` (s,
at full gate). Refused: a non-positive or non-finite `Tw`, a turbine rating that is not positive, time
constants that are not positive, and coefficients other than the ideal turbine's (`a11 = 0.5`,
`a13 = 1`, `a21 = 1.5`, `a23 = 1`), which are the ones the physical model reproduces (record section 2).

---

### hydro_Tw_match

*[src/phsim/hydro.jl:92](../../src/phsim/hydro.jl#L92)*

    hydro_Tw_match(gov::IEEEG3_PHTRUE, Pm) -> Tw

The water starting time (full gate) with which the physical unit, linearised at turbine power `Pm`
(system base), has the case's IEEEG3 transfer function from gate to power at rated speed: the 1992
report's eq. (7) gives `(1 - G0 Tw s) / (1 + G0 Tw s / 2)` at gate `G0 = Pm / GMAX`, the case's IEEEG3
`(1 - Tw_case s) / (1 + Tw_case s / 2)` (its full-gate coefficients), hence `Tw = Tw_case / G0`.

---

### hydro_initialise

*[src/phsim/hydro.jl:201](../../src/phsim/hydro.jl#L201)*

    hydro_initialise(g, Pm) -> (x, Pref)

The unit at rest at turbine power `Pm` (system base) and rated speed: head 1, the gate `at = Pm` (the
ideal turbine at the static head gives `P = GMAX G`), flow `q = Pm / GMAX`, the pilot valve at zero, the
droop reset at the gate; `Pref` from the governor's summing junction, `Pref = Sigma at`.

---

### PHFault

*[src/phsim/midpoint.jl:33](../../src/phsim/midpoint.jl#L33)*

    PHFault(bus, R, t_on, t_off; kind = :abc, Rg = R, clearing = :instant, topen = nothing)

A fault of resistance `R` (pu) at bus `bus` from `t_on`. `clearing = :instant`: every phase opens at `t_off` (whatever its
current). `clearing = :current_zero` (review 82): the fault is a star of per-phase switches (`fault_star`), and each phase
opens at its own first current zero after `t_off` (the arc kept at R until then; extinction of the shunt fault with
successful recovery, no restrike). `phs_simulate` locates the zeros (`phs_resolve_clearing`) unless `topen` (each phase's
opening time) is given; NaN in `topen` = not yet located (closed).

---

### phs_scales

*[src/phsim/midpoint.jl:65](../../src/phsim/midpoint.jl#L65)*

    phs_scales(s, x) -> Vector{Float64}

A magnitude for every state, used to scale residuals: the largest absolute value within its
group (node charges, branch fluxes, shunt fluxes, each machine's fluxes), the momentum itself,
and 1 for an angle.

---

### phs_step_rule

*[src/phsim/midpoint.jl:179](../../src/phsim/midpoint.jl#L179)*

    phs_step_rule(s, x, u; per_period = 20) -> (; h, w_max, r_max)

The step the midpoint rule needs at `x`: `2 pi / (per_period w_max)` to resolve the fastest
oscillation (`w_max`, the largest imaginary part of an eigenvalue of the linearisation) and
`1 / r_max` so that the fastest decay (`r_max`, the largest magnitude of a real part) is damped
monotonically; the smaller of the two.

---

### PHRun

*[src/phsim/midpoint.jl:205](../../src/phsim/midpoint.jl#L205)*

    PHRun

A trajectory and its energy account. Saved every `save_dt` and at every segment end: `t`, states
`X`, bus voltages `V` (rows in network order, generator buses recovered), component `energy`,
`loss` and `port` power, boundary `supply`, total `H`. Accumulated over the run, per component
(`names`, `kind`) and for the system (`total`): `scheme` (the discrete balance, see the file
header) with `scheme_max` the largest magnitude its running sum reached, `closure` (summed
increments against the energy function), `simpson` (the accumulated defect with Simpson's rule
on the powers at step ends and midpoints, one-sided at events), `throughput` (integral of the
sum of the absolute port powers) and `moved` (variation of the stored energy). `segments` lists each segment's `(t0, t1, h, steps, w_max, r_max)`.
`records` holds the quantities declared with `phs_simulate(...; record)`, by name: a vector,
or a matrix with one column per saved time. `phs_write_run` writes all of it to a file.

---

### phs_simulate

*[src/phsim/midpoint.jl:325](../../src/phsim/midpoint.jl#L325)*

    phs_simulate(s, x0, u; tstop, faults = PHFault[], stops = (), hscale = 1, h = nothing,
                 save_dt = 1e-3, grid = nothing, sample_at = nothing, monitor = nothing,
                 record = (), tol = 1e-13, per_period = 20) -> PHRun

Integrate the model (a `PHSystem`, or a `PHControlled` with its exciters and governors) from
`x0` with the implicit midpoint rule. For a controlled model the energy account is the physical
model's, with the inputs the controllers set at each evaluated state, and one more component
per physical exciter. A physical exciter's field energy is not quadratic (it saturates), so
for that component `closure` is not zero to rounding: it is the midpoint rule's energy error,
of second order in the step. Each segment between fault
switching times uses a fixed step: `h` if given, otherwise `hscale` times `phs_step_rule` at the
segment's start, shortened so that a whole number of steps fits. Nothing changes a state at an
event; the fault conductance changes. `stops` are further times at which a segment ends, so
that the trajectory is saved exactly there (a fixed step reaches no other prescribed time).
With `grid` given (seconds), every segment's step is shortened to divide `grid` and the
trajectory is saved exactly at the multiples of `grid`, for comparison with another tool's
samples; segment ends must then be multiples of `grid`. With `sample_at` given (increasing
times), the trajectory is saved at exactly those times and no others: within the step that
contains a time, the state is the cubic Hermite interpolant of the step's two ends and the
vector field there (the step is a few microseconds, so this adds nothing visible), and the
energy account is evaluated at that state. The accumulated accounts are unaffected.
`monitor(t, x, u)`, if given, is called after every step with the time, the state and the inputs
in force, for diagnostics that must see every step and not only the saved ones. `record` declares
quantities to keep while the run goes, as pairs `name => f` with `f(x, u)` a number or a vector
computed from the state and the inputs in force; each is evaluated at every saved time (once per
saved time, not per step) and kept in `r.records[name]`. The state itself is always kept, so a
quantity not declared can still be computed afterwards from `r.X`. The Newton iteration of a step stops when the residual,
scaled by `phs_scales`, is below `tol` (at `1e-12` the discrete balance of some rotors misses
`1e-10` of their throughput on IEEE-39; at `1e-13` every component meets it).

---

### phs_simulate_bdf

*[src/phsim/midpoint.jl:702](../../src/phsim/midpoint.jl#L702)*

    phs_simulate_bdf(s, x0, u; tstop, faults = PHFault[], saveat, reltol = 1e-10, abstol = 1e-12)
        -> (t, X)

The same model integrated by SUNDIALS CVODE (variable-order BDF), restarted at every fault
switching time: an independent trajectory for cross-checking `phs_simulate`. In the stationary frame every
right-hand side sees the frame phasor of its own time; a caller that evaluates ledgers or voltages from the returned
states must set the frame of each saved time itself (`_set_frame!`). `abstol` multiplies
`phs_scales`. BDF damps energy numerically; no energy claim is made for this solution.

---

### phs_write_run

*[src/phsim/midpoint.jl:745](../../src/phsim/midpoint.jl#L745)*

    phs_write_run(path, r::PHRun; model = nothing, meta = Dict(), states = true) -> path

Everything a run produced, to a JLD2 file, as plain arrays and dictionaries so that it can be
read without this code (`phs_read_run`, or any JLD2/HDF5 reader): the saved times `t`; the state
`X` (one column per saved time) with the state names of `model`; the bus voltages `V` with the
bus ids; the energy account per component (`energy`, `loss`, `port`, with `names`, `kind`) and for
the system (`supply`, `H`); the accumulated balances (`scheme`, `scheme_max`, `closure`,
`simpson`, `throughput`, `moved`, `total`); the segments and the Newton statistics; every record
declared with `phs_simulate(...; record)`; the git commit of the code that ran it, with a flag
if the working tree had changes; and `meta`, whatever the caller needs to say what was run.
Anything not recorded can be computed afterwards from `X` with the same model, without
simulating again. With `states = false` the state is left out (a smaller file; then only the
records, the voltages and the account remain).

---

### phs_read_run

*[src/phsim/midpoint.jl:779](../../src/phsim/midpoint.jl#L779)*

    phs_read_run(path) -> Dict{String,Any}: what `phs_write_run` wrote, by name.

---

### PCCSplit

*[src/phsim/pccmanifold.jl:10](../../src/phsim/pccmanifold.jl#L10)*

    PCCSplit

Index sets of a `PHSystem` seen from the machine `machine` (its index in `s.machines`): `G` that
machine's states, `V2` its attachment bus's voltage pair, `D` the other machines' angles, `Y` every
other state (all energy-bearing). Over `Y`: `pos` maps a state index to its position, `isnet` marks
network phasor pairs (rotated by the common rotation), `netpairs` their first positions, `P` the
positions of the other machines' momenta (in `D` order) and `H` their inertia constants.

---

### pcc_Q

*[src/phsim/pccmanifold.jl:61](../../src/phsim/pccmanifold.jl#L61)*

    pcc_Q(s) -> Q with H(x) = x'Qx/2 exactly (the model's energy is quadratic without cores).

---

### pcc_rot

*[src/phsim/pccmanifold.jl:64](../../src/phsim/pccmanifold.jl#L64)*

    pcc_rot(sp, v, g) -> v over `Y` with every network pair rotated by `g` (rotor-frame entries fixed).

---

### pcc_split3

*[src/phsim/pccmanifold.jl:74](../../src/phsim/pccmanifold.jl#L74)*

    pcc_split3(sp, v) -> [rotor-frame part, network part, j * network part] of `v` over `Y`.

---

### pcc_qappend

*[src/phsim/pccmanifold.jl:78](../../src/phsim/pccmanifold.jl#L78)*

    pcc_qappend(QY, Phi, cols; tol) -> (Phi, added)

Appends each column of `cols` orthonormalised in the physical energy `QY` against `Phi`, unless its
remainder is below `tol` of its own energy norm (then skipped). `added` counts the new columns.

---

### pcc_protected

*[src/phsim/pccmanifold.jl:102](../../src/phsim/pccmanifold.jl#L102)*

    pcc_protected(sp, QY, x0) -> the operating point's rotation-closed span (three columns over `Y`).

---

### pcc_pod

*[src/phsim/pccmanifold.jl:105](../../src/phsim/pccmanifold.jl#L105)*

    pcc_pod(sp, QY, Phi0, snaps) -> (modes, energies)

Proposal directions from snapshot columns `snaps` (over `Y`): each projected onto the physical-energy
complement of `Phi0`, the state covariance rotation-averaged (rotor-frame block kept, network block
`(C + L C L')/2`, cross blocks zero), then the eigenvectors in the physical energy:
`(QY^1/2 C QY^1/2) u = lambda u`, `phi = QY^-1/2 u`. Sorted by `lambda`, largest first. Network modes
come in pairs (`v`, `j v`).

---

### PCCReduced

*[src/phsim/pccmanifold.jl:134](../../src/phsim/pccmanifold.jl#L134)*

    PCCReduced

A reduced model: the split `sp`, the energy-bearing basis `Phi` (over `Y`, orthonormal in the physical
energy), the angle basis `Vd` and its left inverse `Wd`, the full trial and test matrices `V`, `W`
(`W'V = I`) and the operating angles `dstar`. Coordinates `xi = (x_G, v2, theta, a)`.

---

### pcc_reduced

*[src/phsim/pccmanifold.jl:151](../../src/phsim/pccmanifold.jl#L151)*

    pcc_reduced(s, sp, Q, Phi, x0; Vd_extra) -> PCCReduced

Builds the map from `Phi`. The angle basis is the kinematic image of the kept momenta plus the
common direction (derivation 3.7), with any extra angle columns `Vd_extra`; orthonormal, so
`Wd = Vd`.

---

### pcc_phi

*[src/phsim/pccmanifold.jl:183](../../src/phsim/pccmanifold.jl#L183)*

    pcc_phi(R, xi) -> the reconstructed full state.

---

### pcc_project

*[src/phsim/pccmanifold.jl:190](../../src/phsim/pccmanifold.jl#L190)*

    pcc_project(R, x) -> xi (the projection orthogonal in the physical energy; angles by Wd).

---

### pcc_rhs

*[src/phsim/pccmanifold.jl:197](../../src/phsim/pccmanifold.jl#L197)*

    pcc_rhs(R, s, xi, u) -> W' f(phi(xi)), the reduced vector field.

---

### pcc_power

*[src/phsim/pccmanifold.jl:200](../../src/phsim/pccmanifold.jl#L200)*

    pcc_power(s, sp, x) -> the power the PCC machine delivers into its bus (per unit), from the
    stator-bus coupling of `J` and the efforts (the pairing the reduction keeps exactly).

---

### steam_volumes_from_time_constants

*[src/phsim/steam.jl:124](../../src/phsim/steam.jl#L124)*

    steam_volumes_from_time_constants(; tau, path = :C) -> volumes

The volume (m3) a given isothermal time constant `V (drho/dp)_T p0 / q0` implies for each vessel of
the published unit of `path` at its design state (record 14.5, 16.6): for path C `tau = (chest,
rh1, rh2, crossover)`, for path A `tau = (chest, reheater, crossover)`, in seconds. `q0` is the
vessel's outflow at the design point. The chest is taken at the throttle's state, or with `r_v` at the
unit's design chest state (the throttle's enthalpy at `r_v` times its pressure), as `PHSteamUnit`
does for `volumes = :case`; `tables` defaults to `steam_tables()`. A reading, not a source; with the case's IEEEG1 time
constants (`volumes = :case`, record 16.6) the vessels are sized to the case's data.

---

### PHSteamUnit

*[src/phsim/steam.jl:217](../../src/phsim/steam.jl#L217)*

    PHSteamUnit

A physical steam unit on its machine's shaft (file header). IEEEG1's signal law and servo
constants `K, T1, T2, T3, UO, UC, PMAX, PMIN, R_base`; the steam network, already scaled by `n`:
storage `names`, volumes `V`, extraction fractions `beta`, heat conductances `kh` and wall
temperatures `Tw` (set at initialisation); the valve constant `cv` (kg/s at full stroke and
choked flow), the throttle state; sections `(up, down, K, eta, delta)`, resistances
`(up, down, K, delta)`, the valve's `delta`; the condenser pressure; the steam tables. Local states
`x0, x1`, then `M, U` of every storage. With `valve = :characteristic` the servo state `x1` is a
flow demand in power units, as in IEEEG1, and the opening is `vc(x1)`, the inverse of the unit's own
steady characteristic (the turbine control logic's function generator, IEEE working group 1991;
record 16.6); with `valve = :opening` it is `x1 / PMAX`.

    PHSteamUnit(gov::IEEEG1_PHTRUE; path, r_v, rh_dT, tau_n, lowflow, cold_reheat = nothing,
                volumes = nothing, tables = steam_tables())

The physical steam unit in place of the case's IEEEG1 `gov`: its signal law and servo constants,
and the published unit of `path` (`:A` Vahidi's 200 MW single-reheat unit, `:C` Zhao and Si's
1000 MW double-reheat unit) scaled to `gov`'s `PMAX` (record sections 6, 13.9, 14.8). Numbers no
source gives, all required: `r_v` the valve's pressure ratio at rated flow and full stroke;
`rh_dT` the reheater wall's temperature excess over the steam at the design point (K); `tau_n`
the extraction nodes' residence time at the design point (s); `lowflow` the low-flow regime's
width as a fraction of each square root's nominal argument; for path A `cold_reheat` (26.2 or
28.22 kgf/cm2); for path C `volumes = (; chest, rh1, rh2, crossover)` in m3 of the published unit.

---

### steam_initialise

*[src/phsim/steam.jl:586](../../src/phsim/steam.jl#L586)*

    steam_initialise(g, Pm; rebuild = true) -> (x, Pref)

The unit's steady state at turbine power `Pm` (system base, per case unit): the stroke, every
storage's mass and energy, and each reheater's wall temperature (written into `g.Tw`) such that
the reheat temperatures are their design values. Newton from the design point scaled by the load;
an error if the stroke would leave `[PMIN, PMAX]` or the solve fails.

With the valve characteristic (`valve = :characteristic`) and `rebuild = true`, the characteristic
is built at this power and, for `volumes = :fit`, the vessels' volumes are fitted at it: the unit's
hardware then depends on `Pm`. With `rebuild = false` the unit's hardware is kept as it is (volumes
and valve characteristic, e.g. from a build at the rated operating point) and only its operating
state is solved at `Pm`: the stroke, the storages, and the wall temperatures that hold the design
reheat temperatures (the boiler's firing, an operating condition). The servo demand is then the
inverse of the existing characteristic at the solved opening; an error if that opening lies outside
the characteristic's range over `[PMIN, PMAX]`.

---

### SteamTables

*[src/phsim/steamprops.jl:166](../../src/phsim/steamprops.jl#L166)*

Tabulated IAPWS-95 water and steam properties (saturation and gas branches) used by the physical steam units.

---

### steam_tables

*[src/phsim/steamprops.jl:187](../../src/phsim/steamprops.jl#L187)*

    steam_tables(; rebuild = false) -> SteamTables

The tables of this file, built from IAPWS-95 on first use and kept for the session; with `cache`
they are also written to `outputs/cache/` and read from there by later sessions (`rebuild`
forces a new build). The grid sizes are keywords: `nl` saturation points in log pressure, `nX` by
`nxi` for the vapour map, `nl`-spaced pressures by `neta` for the isentrope.

---

### steam_sat

*[src/phsim/steamprops.jl:415](../../src/phsim/steamprops.jl#L415)*

    steam_sat(tb, p) -> (; T, v_f, v_g, u_f, u_g, h_f, h_g, s_f, s_g)

Saturation properties at pressure `p` (Pa), from the table.

---

### steam_state

*[src/phsim/steamprops.jl:426](../../src/phsim/steamprops.jl#L426)*

    steam_state(tb, rho, u) -> (; T, p, h, s, x)

The equilibrium state at density `rho` (kg/m3) and specific internal energy `u` (J/kg):
superheated vapour from the vapour table, or inside the dome the wet mixture (quality `x`); `x`
is 1 for vapour. Only the vapour side of the dome is covered (densities up to 160 kg/m3).

---

### steam_h_isentropic

*[src/phsim/steamprops.jl:546](../../src/phsim/steamprops.jl#L546)*

    steam_h_isentropic(tb, p, s) -> (; h, T, x)

The enthalpy at pressure `p` and entropy `s`: from the isentrope table above the dome, the
equilibrium mixture `h_f + T_sat (s - s_f)` inside it. `x` is the quality (1 for vapour).

---

### PHMachine

*[src/phsim/system.jl:79](../../src/phsim/system.jl#L79)*

    PHMachine

One machine of a `PHSystem`: winding data `w` (stator leakage and resistance include the merged
transformer `trf`, `x_ext`, `r_ext`), terminal bus row `node` and ideal ratio `a` (the stator sees
`v_node / a`), original generator bus id `bus`, number of identical units represented, and the
first state index `offset`. Local states: `lam_d, lam_q, lam_fd, lam_1d, lam_1q[, lam_2q], p,
delta`. `Kd`, `Kq` are the inverse inductance matrices its currents are computed with: `w`'s,
or, at a reduced node (`cut`, the index in the system's `cuts`; 0 otherwise), those extended by
the branch winding, whose flux pair is the state `tidx`.

---

### PHCut

*[src/phsim/system.jl:111](../../src/phsim/system.jl#L111)*

    PHCut

A node where only inductors meet, reduced as the file header describes: bus row `node`, the
series load `RL`, `LL` (the shunt inductor `shunt` of the network, which has no state), the
branches at the node (`pos`, positions in the system's `branches`; `gamma`, `c` the coefficients
of the node's and the far bus's voltage in each branch's row; `far` the far bus rows), and
`machine` (index in the system's `machines`, 0 if none). Without a machine `M`, `Rm` are the
inductance and resistance matrices over the branches and `Minv = inv(M)`; with one they are the
2 by 2 matrices over (stator, branch) that the reduction adds: `M = [L_L/a^2 b L_L/a; b L_L/a
L_T + b^2 L_L]`, `Rm` as in the file header.

---

### PHConverter

*[src/phsim/system.jl:138](../../src/phsim/system.jl#L138)*

    PHConverter

One converter of a `PHSystem` (file header): its bus id `bus` and bus row `k` (removed if
`merged`), its source row `row`, the position `pos` of its branch in the system's `branches`, the
series inductance `Lc`, the constant `alpha` of its terminal voltage `v_k = alpha u_c + w`, the
number of identical units it stands for, the case's component `comp` (one unit) and the index `core` of its
transformer's saturating core in the system's `cores` (0 if none; then `alpha` is NaN: the terminal voltage is no longer
affine in `u_c` with a constant, record section 21, T1 row 4).

---

### PHSystem

*[src/phsim/system.jl:164](../../src/phsim/system.jl#L164)*

    PHSystem

A case as one port-Hamiltonian system. `net` is the full element list; `dyn` and `alg` are the
bus rows with a voltage state and with an eliminated voltage; `merged` the bus rows removed by
the series merge; `qidx[k]` the first state of bus row `k` (0 if it has none); `branches` the
kept branches with `bidx` their first states and `charge` their total charging; `sidx` the first
states of the shunt inductors (0 for the load of a reduced node) and `Rsh` their series
resistances (zero unless the load is a series R-L branch); `Cown` the node capacitance not owned
by a line (loads, shunts); `cut` the bus rows of the reduced inductor-only nodes, `cuts` their
data, `cutof[k]` the index in `cuts` of bus row `k` and `bcut[j]` that of branch position `j`
(0 if none); `cnames`, `ckind` the components of the energy account (`phs_ledger`), in its
order; `conv` the converters, whose source rows follow the bus rows; `cores` the saturating transformer cores
(`transformer.jl`), whose source loop fluxes follow the machines' states, and `iclk` the frame clock's state after them
(0 without cores); `step` holds the step's discrete-gradient currents while `phs_simulate` evaluates a step (`nothing`
otherwise; `:short` evaluates every core short-circuited, for the step rule's bound). `stationary` selects the frame
of the network states: `false` (the default) the synchronous frame, `true` the stationary frame
(`docs/realtime/docs/STEP0_CONTRACTS.md` A3-A4; made by `phs_frame`), with `theta0` the frame angle at t = 0.

---

### PHInput

*[src/phsim/system.jl:221](../../src/phsim/system.jl#L221)*

    PHInput

The boundary inputs: shaft torque `Tm` and field voltage `vf` per machine (generator torque
positive, reciprocal per unit `vf = Rfd i_fd` at rest), and the fault conductance `gf` per bus
row (0 without a fault). With `const_power`, `Tm` is the turbine power and the torque is
`Tm / omega` (what PowerFactory does with a turbine power input and no governor). `cf[1]` is the frame phasor
e^{j theta_f}: read only in the stationary frame, where the simulators set it to e^{j (omega_b t + theta0)} for the
time of each evaluation (`_set_frame!`); 1 otherwise and unused.

---

### ph_system

*[src/phsim/system.jl:273](../../src/phsim/system.jl#L273)*

    ph_system(sys::DAESystem; v0 = nothing, loads = :parallel, circuits = nothing,
              stray_capacitance = 0.0) -> PHSystem

Assemble the port-Hamiltonian model of the case behind `sys`: its `physical_network` (loads as
constant-impedance elements at the voltages `v0`, in network order) and one full-winding machine
(`winding_machine`, the Sauer-Pai map) per GENROU or GENSAL record, with the reductions listed in
the file header. `loads = :series`, `loads = :series_all` and `circuits` (a `Dict` from generator
bus id to winding data on the system base, all units of the bus together: `Ll, Lad, Laq, Lfd,
L1d, L1q, L2q, Ra, Rfd, R1d, R1q, R2q, H`, with `L2q = nothing` for one q-axis rotor winding)
are the comparison choices of the file header. `stray_capacitance > 0` (with `:series_all`)
puts that capacitance at every capacitance-free bus with an inductive load in place of reducing
the node: a different circuit, kept only as a reference the reduced model must be the limit of.
Throws when a reduction's condition does not hold (different machines on one bus, a generator
bus with more than its transformer, a machine with damping `D != 0`, an inductor-only node with
a phase-shifting ratio, with two machines, or whose branches do not end at buses with
capacitance).

`cores` (a dictionary from a converter's bus id to its step-up transformer's total rating in MVA, all parallel units)
gives those transformers a saturating core (`transformer.jl`; record sections 20-23): the leakage `R_T + j X_T` split
0.5/0.5 around the core, the GMD curve (`core_curve_gmd`) on the rating. Only a converter merged with its transformer.

---

### phs_state_names

*[src/phsim/system.jl:738](../../src/phsim/system.jl#L738)*

    phs_state_names(s) -> Vector{String}

    phs_state_names(cs) -> the physical state names, then `<controller>.<state>`.

---

### phs_machine_signals

*[src/phsim/system.jl:880](../../src/phsim/system.jl#L880)*

    phs_machine_signals(s, x) -> Vector of (; speed, delta, torque, ifd, id, iq, psid, psiq)

Per machine at the state `x`: speed, rotor angle, electromagnetic torque (motor convention,
system base), field current, stator currents, and the machine's own stator flux linkages (pu;
without the merged transformer's leakage and without what a reduced node adds).

---

### phs_flows

*[src/phsim/system.jl:927](../../src/phsim/system.jl#L927)*

    phs_flows(s, x, u) -> (; v, inj, mc, ib, ish, iload)

Bus voltages `v` (network frame; `NaN` at a merged bus, see `phs_terminal_voltage`, and at a
reduced node, see `phs_cut_voltage`), the current `inj` each bus receives from its branches,
shunt inductors and machines (and, at a reduced node, gives to its load, so that it is zero
there), the machine quantities `mc`, the branch currents `ib` (network frame, by position in
`s.branches`), the shunt inductor currents `ish` and the load current `iload` of each reduced
node.

---

### phs_rhs!

*[src/phsim/system.jl:1206](../../src/phsim/system.jl#L1206)*

    phs_rhs!(dx, s, x, u) -> dx

The vector field of the model (file header). Generic in the element type of `x`.

---

### phs_rhs

*[src/phsim/system.jl:1308](../../src/phsim/system.jl#L1308)*

    phs_rhs(s, x, u) -> dx

The vector field of the EMT model `s` at state `x` and input `u` (allocating form of `phs_rhs!`).

---

### phs_energy

*[src/phsim/system.jl:1326](../../src/phsim/system.jl#L1326)*

    phs_energy(s, x) -> the total stored energy `H(x)` (pu*s).

---

### phs_gradient

*[src/phsim/system.jl:1361](../../src/phsim/system.jl#L1361)*

    phs_gradient(s, x) -> grad H(x)

The efforts: bus voltages, branch and shunt currents, machine currents, speeds, and zero on every
angle.

---

### phs_matrices

*[src/phsim/system.jl:1418](../../src/phsim/system.jl#L1418)*

    phs_matrices(s, x, u) -> (; J, R, g, z)

The structure of the model at `x`, built independently of `phs_rhs!`: skew `J(x)`, symmetric
positive semidefinite `R(x)`, the input vector `g` (shaft torques, which depend on the speed
when the shaft input is a power, field voltages and the frame clock `-omega_b` on every angle) and the gradient `z`, with `x' = (J - R) z + g`. `R` holds the
element resistances and conductances and, for the eliminated nodes, `N diag(1/G) N'` with `N`
the incidence of the elements at such a node. At a reduced inductor-only node `J` holds the
frame rotation `-j M` of its coupled branches (or the speed-voltage pair of a machine's branch
winding) and `R` the matrix `Rm`.

---

### phs_ledger

*[src/phsim/system.jl:1522](../../src/phsim/system.jl#L1522)*

    phs_ledger(s, x, u; dx = nothing) -> NamedTuple

The energy account at `x`, by component. `names`, `kind` (`:line`, `:transformer`, `:load`,
`:machine`, `:rotor`, and one `:fault` entry for all fault conductances) and, per component: stored `energy`, `loss`, the power `port` it
receives through its ports (network ports; for a machine also the field, less the air-gap power
it passes to its rotor; for a rotor the shaft and the air-gap power), and the `rate` of its
energy from the vector field. Each satisfies `rate = port - loss`. `tport` is the sum of the
absolute values of its individual port powers (the integrand of its throughput). With `dx` given,
`rate` is `grad H_c(x)' dx` for that `dx` in place of the vector field; ports and losses are
unchanged. Every component energy is a quadratic form, so with `x` a step's midpoint and
`dx = (x1 - x0)/h` this is the exact energy increment of the step over `h`, free of the
cancellation in `H_c(x1) - H_c(x0)`.

Which balances are independent evidence: the series branches, shunt inductors, rotors and the
windings together with their merged transformer. The merged transformer's port power is defined
as its rate plus its loss, a fault's and a conductance's as its loss, and the capacitors at a
node share that node's balance in proportion to their capacitance. At a reduced inductor-only
node the voltage is recovered from the load's equation, so the load's balance holds by
construction, and those of the node's branches and of its machine's windings are evidence.
`supply` is the boundary power (all shaft and field power), `interconnection` the sum of the
internal port powers (zero), `H` the total energy and `dH` its rate.

Ownership: a line owns its series inductor and half its charging at each end; a load owns its
conductance, its shunt inductor and its capacitor; a merged transformer owns `x_ext`, `r_ext` of
the stator they share; a machine owns its windings; a rotor owns the kinetic energy and converts
between the air-gap power `Tem omega` and the shaft. At a reduced node each element still owns
its own inductor and resistor: their energies and losses are computed from the currents.

    phs_ledger(cs, x, u; dx = nothing)

The energy account at `x`: the physical model's, with the inputs the controllers set there,
and after its components one per physical exciter (`<name>.exciter`): the stored energy of its
field, the loss in its field circuit, the net power it receives (from the shaft and, for a DC
exciter, from its amplifier, less what it delivers to the machine's field) and the rate of its
energy along `dx` (the vector field if `dx` is not given). `supply` is then the turbine powers,
the field powers of machines without a physical exciter and the DC exciters' amplifier powers.

---

### phs_network_mask

*[src/phsim/system.jl:1749](../../src/phsim/system.jl#L1749)*

    phs_network_mask(s, L) -> BitVector

The components of the ledger `L` (`phs_ledger` of `s`, a `PHSystem` or `PHControlled`) on the network side of the
machines' ports, where a machine's power is `-v_node conj(i_net) / a` at its terminal bus row (the high-voltage bus
after the series merge): lines, network transformers, loads and the fault entry. A machine's merged step-up
transformer is not among them: it lies inside the port, between that bus and the windings (record 30). So at any
state the machines' port active powers sum to the masked components' `rate + loss`. Throws if a merged transformer
is not exactly one `:transformer` entry of `L`, or with converters or saturating cores (not covered).

---

### phs_terminal_voltage

*[src/phsim/system.jl:1802](../../src/phsim/system.jl#L1802)*

    phs_terminal_voltage(s, x, u) -> Dict(bus id => complex voltage)

The voltage of every generator bus removed by the series merge, recovered from the transformer
it was merged with: `v_g = v/a - (r_ext + j x_ext) i - (x_ext/omega_b) di/dt`, with `di/dt` from
the vector field. This is the eliminated variable of the unreduced model.

---

### phs_frame

*[src/phsim/system.jl:1812](../../src/phsim/system.jl#L1812)*

    phs_frame(s, frame; theta0 = 0.0) -> PHSystem

The same model with its network states in `frame`: `:synchronous` (rotating at omega_b; today's model) or
`:stationary` (`docs/realtime/docs/STEP0_CONTRACTS.md` A3-A4): every frame term `-j omega_b x` of a network-frame
state is removed, and each machine's terminal rotation becomes e^{j (delta + theta_f)} with
theta_f = omega_b t + theta0. Rotor-frame states (machine windings, a reduced node's machine winding) and the angles
delta keep their meaning. Converters and saturating cores are not available in the stationary frame yet (their
controllers and the cores' frame clock need their own derivation): the model is refused. Initial states:
`phs_rotate_network`.

    phs_frame(cs::PHControlled, frame; theta0 = 0.0) -> PHControlled

The controlled model with its physical model in `frame` (`phs_frame(::PHSystem, ...)`); the controllers' states are
rotor or scalar quantities and are unchanged.

---

### phs_rotate_network

*[src/phsim/system.jl:1853](../../src/phsim/system.jl#L1853)*

    phs_rotate_network(s, x, theta) -> state

`x` with every network-frame pair multiplied by e^{j theta} (rotor-frame states and angles unchanged): the state of
the stationary-frame model at time t from the synchronous one, with theta = omega_b t + theta0; `-theta` goes back.
Only the first `s.n` entries (the physical model) are touched; controller states follow unchanged.

    phs_rotate_network(cs::PHControlled, x, theta): the physical model's network pairs rotated; controller states unchanged.

---

### phs_cut_voltage

*[src/phsim/system.jl:1876](../../src/phsim/system.jl#L1876)*

    phs_cut_voltage(s, x, u) -> Dict(bus id => complex voltage)

The voltage of every reduced inductor-only node, recovered from its load's own equation,
`v = (L_L/omega_b) di_L/dt + (R_L + j L_L) i_L`, with `di_L/dt` from the vector field. This is
the eliminated multiplier of the unreduced model.

---

### phs_phases

*[src/phsim/system.jl:1888](../../src/phsim/system.jl#L1888)*

    phs_phases(z, theta) -> (a, b, c)

The three phase values of the network-frame phasor `z` at the frame angle
`theta = omega_b t + theta_0`: `Re(e^{j theta} z)` and the same a third of a turn behind and
ahead (the inverse dq0 transformation of the specification, section 2, in per unit of the peak
phase value). The balanced model has no zero-axis component, so nothing is added.

---

### phs_equilibrium

*[src/phsim/system.jl:1898](../../src/phsim/system.jl#L1898)*

    phs_equilibrium(eq::EquilibriumResult; loads = :parallel, circuits = nothing,
                    const_power = false, stray_capacitance = 0.0) -> (s, x, u, report)

The model of `eq`'s case and its balanced equilibrium. Start: the bus voltages of the RMS
equilibrium, with the loads as constant impedances at those voltages, so that the network's
quasi-static limit is the case's admittance matrix; branch and shunt currents from them; each
machine's angle, fluxes and field voltage by the steady-state construction of `emdq_pilot`.
Then, with every rotor angle, speed (1) and field voltage held, the remaining equations are
linear in the remaining states and are solved exactly, and each shaft torque is set to balance
its air-gap torque (with `const_power`, the turbine power, the same number at speed 1). The
keywords are those of `ph_system` and `PHInput`. `report` holds the full residual of the vector field, the largest change of
a bus voltage from the RMS equilibrium (reduced nodes included), and the recovered
generator-bus voltages.

With `cores` (`ph_system`) each cored branch starts from the linear T-circuit at the operating point (the core's first
segment), the frame clock at 0 and held; the clock's row (its rate is `omega_b`) is not part of the residual. If a core's
phase fluxes there exceed the curve's first knot the operating point is not a fixed point of the model (its steady state
is periodic: record section 21, T1 row 7); the rows are then solved at the clock's initial value only (an instantaneous
rest, not a periodic operating solution: T3 2d) and `report.core_flux` says how far the fluxes are.

---

### phs_rest_residual

*[src/phsim/system.jl:2159](../../src/phsim/system.jl#L2159)*

    phs_rest_residual(s, dx) -> the largest entry of the vector field `dx` at rest, the frame clock's row left out.

---

### phs_core_signals

*[src/phsim/system.jl:2166](../../src/phsim/system.jl#L2166)*

    phs_core_signals(s, x) -> Vector of (; bus, lam, phase, im, iH, iS, segment, E_Lc, E_leak, E_core)

Per saturating core at the state `x`: the core flux (complex, network frame, flux units), the three phase fluxes, the
magnetising current (`iH - iS`), the HV and source-side branch currents, the highest segment of the curve a phase is
on, and the stored energy split by owner (pu s): the converter's `Lc`, the two leakage halves and the core. The pieces
are energies at the state only; their step increments have no balance of their own (record section 23).

---

### PHCoreCurve

*[src/phsim/transformer.jl:25](../../src/phsim/transformer.jl#L25)*

    PHCoreCurve

The per-phase current-flux curve of a core on the system base: flux knots `lam` (peak, pu of the winding's rated flux,
starting at 0), current knots `cur` (peak, system base), the slope `di/dlambda` of each segment (`sl`, the last extended
beyond the last knot), the energy `wk` at each knot, `Lm` the linear inductance of the first segment and `wnk` the
non-linear energy (`w - lambda^2/(2 Lm)`) at each knot from the second on. `source` says where the data come from.

    PHCoreCurve(lam, cur, slope_end, scale; source) -> PHCoreCurve

From knots on the transformer's own base (`lam` and `cur` from zero, increasing, positive slopes; `slope_end` the
`di/dlambda` beyond the last knot) to the system base: currents times `scale = S_T / S_b` (the flux base is unchanged:
the winding's rated voltage is the bus's base voltage). Odd extension below zero.

---

### core_curve_gmd

*[src/phsim/transformer.jl:72](../../src/phsim/transformer.jl#L72)*

    core_curve_gmd(scale) -> PHCoreCurve

The measured curve of the GMD test case (Haddadi et al., IEEE Trans. EMC 2020, doi 10.1109/TEMC.2019.2920271, Table IV,
p. 958; a 300 MVA single-phase shell-form unit's field test, used there for every transformer): current 0.001, 0.01,
0.025, 0.05, 0.1, 0.5, 1, 2 pu at flux 1, 1.075, 1.15, 1.2, 1.23, 1.37, 1.55, 1.86 pu, the fully saturated slope 0.31 pu
(`dlambda/di`) beyond. Taken as peak flux against peak current (the table does not say: an assumption, record section
20), on the transformer's rating, scaled to the system base by `scale = S_T / S_b`.

---

### PHCore

*[src/phsim/transformer.jl:118](../../src/phsim/transformer.jl#L118)*

    PHCore

A saturating core on a converter's merged branch (file header): the branch name, the converter's bus id, the branch's
position `pos` in the system's `branches` (its state pair is the HV loop flux `x_H`), the converter's index `conv`, the
first index `xs` of the source loop flux `x_S`, the T-circuit `L1, R1, L2, R2` (system base, `L1` holding the converter's
`Lc`), the transformer's total rating `S` (MVA, all parallel units) and the curve.

---

### core_solve

*[src/phsim/transformer.jl:177](../../src/phsim/transformer.jl#L177)*

    core_solve(q, psH, psS, th) -> (; lam, iH, iS, laml)

The core flux minimising the branch's energy at the loop fluxes `psH`, `psS` (flux units, `psi = omega_b x`) and the
clock `th`, with the branch currents and the linear minimiser `laml`. In the curve's linear part (every phase flux of the
linear minimiser within the first knot) the linear minimiser itself, exactly. Otherwise a damped semismooth Newton on the
strictly convex objective (T1 row 6), exact once the segments are found; with dual numbers the derivative comes from
one Newton correction at the converged primal solution (the implicit function: K dlam = -dr). That gives first
derivatives; through nested dual numbers the mixed second derivatives are not those of the implicit solve (no supported
path takes them: T3 1b).

---

### core_energy

*[src/phsim/transformer.jl:228](../../src/phsim/transformer.jl#L228)*

    core_energy(q, psH, psS, th) -> h (flux units; the stored energy is h / omega_b).

---

### core_dgrad

*[src/phsim/transformer.jl:243](../../src/phsim/transformer.jl#L243)*

    core_dgrad(q, H0, S0, H1, S1, thm) -> (iH, iS)

Gonzalez's discrete gradient of `h(., thm)` between the (rotated) end points `(H0, S0)` and `(H1, S1)` (flux units),
returned as the branch currents it stands for (`(iH, -iS)` is the gradient): the gradient at the midpoint plus the
correction along the increment, its numerator on the non-linear part only (exactly zero in the linear part). With no
measurable increment (below 1e-13 of the ends' size) the midpoint gradient: there the discrete balance holds to that
tolerance instead of exactly (T3 1d).

