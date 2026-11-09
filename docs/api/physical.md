# Physical steam and shaft models

IAPWS-95 water and steam properties, the steam vessel, valve, turbine and rotor, and their composition with the machine and the grid. Source: [src/physical/](../../src/physical).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`case_without_component`](#case_without_component) | `steam_grid_unit.jl` |
| [`SteamGridUnit`](#steamgridunit) | `steam_grid_unit.jl` |
| [`steam_grid_sizes`](#steam_grid_sizes) | `steam_grid_unit.jl` |
| [`steam_grid_algebraic`](#steam_grid_algebraic) | `steam_grid_unit.jl` |
| [`steam_grid_dae`](#steam_grid_dae) | `steam_grid_unit.jl` |
| [`steam_grid_jacobians`](#steam_grid_jacobians) | `steam_grid_unit.jl` |
| [`grid_rotation_direction`](#grid_rotation_direction) | `steam_grid_unit.jl` |
| [`SteamMachineShaft`](#steammachineshaft) | `steam_machine_shaft.jl` |
| [`steam_machine_energy`](#steam_machine_energy) | `steam_machine_shaft.jl` |
| [`steam_machine_balance`](#steam_machine_balance) | `steam_machine_shaft.jl` |
| [`steam_machine_rhs!`](#steam_machine_rhs) | `steam_machine_shaft.jl` |
| [`steam_machine_shift`](#steam_machine_shift) | `steam_machine_shaft.jl` |

---

### case_without_component

*[src/physical/steam_grid_unit.jl:10](../../src/physical/steam_grid_unit.jl#L10)*

    case_without_component(path, name) -> Case

Load the case at `path` with component `name` and every wire to or from it removed. A
machine input that loses its wire becomes an undriven constant at initialisation.

---

### SteamGridUnit

*[src/physical/steam_grid_unit.jl:27](../../src/physical/steam_grid_unit.jl#L27)*

    SteamGridUnit(sys, machine_name, vessel, flow_law; exhaust_pressure, ambient_temperature,
                  efficiency, mechanical_loss_fraction, minimum_speed_pu, friction_pu=0.0,
                  pole_pairs=1)

The steam vessel, valve and turbine on the rotor of machine `machine_name` of the
assembled system `sys`. The machine must be a `GENROU_PHTRUE` whose `Tm` input is a
constant source (assemble a case without its governor). The power base is the system
base of the case; the shared rotor has `J*Omega_b^2 = 2*H*S_b` with the kernel's
system-base `H`.

Composite differential state `X = [x; S; M]` with `x` the states of `sys`; composite
algebraic unknowns `(V, y)` with `V` the bus voltages of `sys` and `y` the nine steam
unknowns of `STEAM_ALGEBRAIC_NAMES`.

---

### steam_grid_sizes

*[src/physical/steam_grid_unit.jl:68](../../src/physical/steam_grid_unit.jl#L68)*

Number of composite differential states and of composite algebraic unknowns.

---

### steam_grid_algebraic

*[src/physical/steam_grid_unit.jl:79](../../src/physical/steam_grid_unit.jl#L79)*

The nine steam unknowns from the implemented steam path at composite state `X`.

---

### steam_grid_dae

*[src/physical/steam_grid_unit.jl:83](../../src/physical/steam_grid_unit.jl#L83)*

    steam_grid_dae(u, X, V, y; admission, heat and feed keywords...) -> (F, G, report)

The composite semi-explicit DAE `X' = F(X,V,y)`, `0 = G(X,V,y)`, root-free in all three
arguments. `F` is the grid right-hand side with the machine's speed row corrected by
`(Tm_steam - c0)/(2*H*omega)`, followed by the vessel entropy and mass rates. `G` is the
grid's KCL residual followed by the nine steam constraints. `Tm_steam` is the net shaft
power (after rotor friction) divided by the power base.

---

### steam_grid_jacobians

*[src/physical/steam_grid_unit.jl:120](../../src/physical/steam_grid_unit.jl#L120)*

    steam_grid_jacobians(u, X, V, y; kwargs...)

Jacobian blocks of the composite DAE by ForwardDiff, and the reduced linearisation
`A = F_X - F_z*G_z^{-1}*G_X` with `z = (V, y)`. AD results through the steam kernel must
be cross-checked (docs/active/STEAM_MACHINE_SHAFT_COUPLING.md, section 8); the audit
compares every block with central differences.

---

### grid_rotation_direction

*[src/physical/steam_grid_unit.jl:140](../../src/physical/steam_grid_unit.jl#L140)*

    grid_rotation_direction(sys, x, V) -> (dx, dV)

Generator of a common rotation of every rotor angle and every bus voltage phasor: one
radian on the angle state of each swing source, and `(-Vq, Vd)` at each bus.

---

### SteamMachineShaft

*[src/physical/steam_machine_shaft.jl:11](../../src/physical/steam_machine_shaft.jl#L11)*

    SteamMachineShaft(vessel, flow_law, machine; power_base, pole_pairs=1, exhaust_pressure,
                      ambient_temperature, efficiency, mechanical_loss_fraction,
                      minimum_speed_pu, friction_pu=0.0)

Steam vessel, valve and turbine on one rigid shaft with a `GENROU_PHTRUE` machine. The
state is `[S, M, delta, omega, Eq', psi_d, Ed', psi_q]`: vessel entropy (J/K) and mass (kg)
followed by the machine's own states. `power_base` is in W. `friction_pu` is the rotor
friction power at rated speed divided by `power_base`; the friction power is
`friction_pu*power_base*omega^2`.

The pole-pair number sets the mechanical base speed `omega_b/pole_pairs` and the SI
inertia, but it cancels from the dynamics and from every reported power.

---

### steam_machine_energy

*[src/physical/steam_machine_shaft.jl:58](../../src/physical/steam_machine_shaft.jl#L58)*

Thermal and mechanical energy `U(S,M) + S_b*H*omega^2` in J. No electromagnetic storage.

---

### steam_machine_balance

*[src/physical/steam_machine_shaft.jl:62](../../src/physical/steam_machine_shaft.jl#L62)*

    steam_machine_balance(c, z; admission, terminal_voltage, field_voltage, kwargs...)

Coupled rates and the thermal-mechanical power balance. `terminal_voltage` is the
network-frame pair `(V_Re, V_Im)` in per unit.

The state derivative `dz` uses the unchanged machine kernel with
`Tm = (shaft power - friction power)/power_base` (formulation B). `omega_rate_rotor` is the
same speed derivative obtained from the physical rotor with load torque
`power_base*(Te + D*(omega-1))/Omega_b` (formulation A); the two are equal identically.

`power.electrical_out` is the kernel-defined terminal power `power_base*(vd*id+vq*iq)`
withdrawn from the shaft. `power.damping_exchange` is `power_base*D*omega*(omega-1)`: a
signed exchange, not a dissipation about rest.

---

### steam_machine_rhs!

*[src/physical/steam_machine_shaft.jl:118](../../src/physical/steam_machine_shaft.jl#L118)*

Coupled right-hand side; the caller owns solver events and guard enforcement.

---

### steam_machine_shift

*[src/physical/steam_machine_shaft.jl:125](../../src/physical/steam_machine_shaft.jl#L125)*

    steam_machine_shift(c, z, zstar)

Bregman shift of the thermal-mechanical energy about `zstar` in the physical storage
coordinates `(S, M, l)`, `l = J*Omega_b*omega`:

    W = E(z) - E(zstar) - grad E(zstar) . (x - xstar),   grad E = (T, mu, Omega).

Its rotor part is `power_base*H*(omega - omegastar)^2`. `W` contains neither the rotor
angle nor any electromagnetic storage, so it is not a storage function of the eight-state
system. Returns `W` (J), the effort differences and the storage-coordinate difference.

