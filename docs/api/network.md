# Network

Bus ordering, Y-bus matrices, Norton stamps, load admittances and fault shunts. Source: [src/network/](../../src/network).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`FaultShunt`](#faultshunt) | `events.jl` |
| [`fault_admittance`](#fault_admittance) | `events.jl` |
| [`fault_shunts`](#fault_shunts) | `events.jl` |
| [`with_fault`](#with_fault) | `events.jl` |
| [`split_line_for_fault`](#split_line_for_fault) | `events.jl` |
| [`Network`](#network) | `ybus.jl` |
| [`nbus`](#nbus) | `ybus.jl` |
| [`bus_index`](#bus_index) | `ybus.jl` |
| [`NortonStamp`](#nortonstamp) | `ybus.jl` |
| [`norton_stamps`](#norton_stamps) | `ybus.jl` |
| [`ybus`](#ybus) | `ybus.jl` |
| [`ybus_pf`](#ybus_pf) | `ybus.jl` |
| [`ybus_dae`](#ybus_dae) | `ybus.jl` |
| [`LoadAdmittances`](#loadadmittances) | `ybus.jl` |
| [`load_admittances`](#load_admittances) | `ybus.jl` |

---

### FaultShunt

*[src/network/events.jl:3](../../src/network/events.jl#L3)*

    FaultShunt

A fault admittance `g + jb` at bus `bus` (row `index` of the network) on `[t_start, t_end)`.

---

### fault_admittance

*[src/network/events.jl:17](../../src/network/events.jl#L17)*

    fault_admittance(r, x) -> (g, b)

`1/(r + jx) = (r - jx)/(r^2 + x^2)`, computed as Python PHPS's DAE compiler does (`z^2` clipped at
`1e-20`).

---

### fault_shunts

*[src/network/events.jl:29](../../src/network/events.jl#L29)*

    fault_shunts(net, events) -> Vector{FaultShunt}

The bus-fault shunts of a scenario, in event order. Faults at buses outside the network are
skipped, as in Python PHPS.

---

### with_fault

*[src/network/events.jl:46](../../src/network/events.jl#L46)*

    with_fault(Y, shunts) -> SparseMatrixCSC{ComplexF64}

`Y` with the fault shunts added on the diagonal (the fault-on network).

---

### split_line_for_fault

*[src/network/events.jl:59](../../src/network/events.jl#L59)*

    split_line_for_fault(case, lf::LineFault) -> (lines, fault_bus, bus_fault)

Topology for a fault part-way along a line, as Python PHPS `SimulationRunner` builds it: the line
is replaced by two sections through a new bus (id 99, or one above the largest bus id if
99 is taken), whose series and shunt parameters are split by `distance`, and a
[`BusFault`](@ref) is placed at the new bus. The new bus starts at `v0 = 1`, `a0 = 0`.

---

### Network

*[src/network/ybus.jl:15](../../src/network/ybus.jl#L15)*

    Network

Bus ordering shared by every network matrix: bus ids sorted ascending (as Python PHPS does), and
the map from bus id to row.

---

### nbus

*[src/network/ybus.jl:33](../../src/network/ybus.jl#L33)*

Number of buses in the network.

---

### bus_index

*[src/network/ybus.jl:35](../../src/network/ybus.jl#L35)*

Position of bus id `bus` in the network's bus order.

---

### NortonStamp

*[src/network/ybus.jl:155](../../src/network/ybus.jl#L155)*

    NortonStamp

A generator's Norton admittance `1/(ra + j xd'')` at its bus.

---

### norton_stamps

*[src/network/ybus.jl:171](../../src/network/ybus.jl#L171)*

    norton_stamps(case) -> Vector{NortonStamp}

The Norton admittances Python PHPS adds to the DAE Y-bus, in component order: every generator
except the current-source converters (`GFL_PHTRUE`, `GFL_ZIF_PHTRUE`). Throws
[`UnsupportedModelError`](@ref) if the case uses a model type PHPS does not implement. `xd''` falls back to
`xd1`, then 0.2, as in Python PHPS. (At P4 this moves into each model's `injection`.)

---

### ybus

*[src/network/ybus.jl:204](../../src/network/ybus.jl#L204)*

    ybus(case; net = Network(case), lines = case.lines, loads = false, norton = ())
        -> SparseMatrixCSC{ComplexF64}

Bus admittance matrix in the bus order of `net`:

- `ybus(case)`: lines and shunts, the power-flow matrix (Python PHPS `build(include_loads=False)`);
- `loads = true`: adds each PQ load as the constant impedance `(P - jQ)/v0^2`, with `v0`
  the case `Bus` voltages or the override `v0` (magnitudes in network order);
- `norton`: generator Norton admittances to add (see [`norton_stamps`](@ref)).

See [`ybus_dae`](@ref) for the matrix the DAE uses.

---

### ybus_pf

*[src/network/ybus.jl:230](../../src/network/ybus.jl#L230)*

    ybus_pf(case) -> SparseMatrixCSC{ComplexF64}

The power-flow Y-bus: lines and shunts only; loads enter the power flow as P/Q
specifications.

---

### ybus_dae

*[src/network/ybus.jl:238](../../src/network/ybus.jl#L238)*

    ybus_dae(case) -> SparseMatrixCSC{ComplexF64}

The DAE network matrix of Python PHPS (`DiracCompiler.Y_full`): lines, shunts, constant-impedance
PQ loads, and the generator Norton admittances. No Kron reduction; every bus stays an
algebraic variable. The loads use the case `v0` unless `v0` is given; the matrix Python PHPS
simulates with uses the solved power-flow voltages (`v0 = solve_powerflow(case).V`).

---

### LoadAdmittances

*[src/network/ybus.jl:249](../../src/network/ybus.jl#L249)*

    LoadAdmittances

Per-bus load data the DAE residual uses (bus order of the network): `G = P0/V0^2`,
`B = -Q0/V0^2`, the scheduled `P`, `Q`, and the frequency coefficients `kpf`, `kqf`.
`COMPLEXLOAD` components override `G`, `B`, `kpf`, `kqf` at their bus.

---

### load_admittances

*[src/network/ybus.jl:266](../../src/network/ybus.jl#L266)*

    load_admittances(case; net = Network(case)) -> LoadAdmittances

Port of the load block of Python PHPS `_build_full_ybus` and the kpf/kqf fallback of
`DiracCompiler.build`. `v0` overrides the bus voltages (network order) and `load_params`
(component name => parameter dictionary) the COMPLEXLOAD parameters.

