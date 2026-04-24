# PHPS — Port-Hamiltonian Power System Simulator

A compiled power system transient stability simulator that generates native C++ solvers from JSON system descriptions. The framework encodes network topology, component physics, and signal wiring into a single mathematical structure — a system of differential-algebraic equations (DAE) — and solves it with implicit BDF/IDA methods at native speed.

Every synchronous machine, exciter, governor, stabilizer, and renewable inverter is a self-contained component that declares its own states, ports, parameters, and dynamics. The framework provides six integration backends spanning four execution tiers: interpreted Python, SciPy, Numba JIT-compiled, and fully compiled C++. The compiled C++/IDA backend achieves a 378× speedup over the pure-Python baseline on the IEEE 14-bus system.

---

## How It Works

The simulation pipeline has four stages:

```
 JSON system file
       │
       ▼
 ┌─────────────┐    builds     ┌──────────────┐
 │  SystemGraph │ ──────────►  │  PowerFlow    │
 │  (topology,  │              │  (Newton-     │
 │   wires,     │              │   Raphson)    │
 │   components)│              └──────┬───────┘
 └──────┬──────┘                      │ V, θ per bus
        │                             ▼
        │                    ┌─────────────────┐
        │                    │  Initializer     │
        │                    │  (6-pass state   │
        │                    │   equilibrium)   │
        │                    └────────┬────────┘
        │                             │ x₀
        ▼                             ▼
 ┌──────────────────────────────────────────┐
 │  DiracCompiler                           │
 │  Generates a self-contained C++ file:    │
 │  • Component structs with parameters     │
 │  • dx/dt = f(x, V) per component         │
 │  • DAE residual: F(t, y, ẏ) = 0          │
 │  • Full Y-bus (no Kron reduction)         │
 │  • SUNDIALS IDA solver integration        │
 │  • Fault event injection / topology swap  │
 │  • CSV recording of all observables       │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
              g++ -O3 → binary → simulation_results.csv
```

### From JSON to Equations

A JSON system file defines three things:

1. **Network topology** — buses, transmission lines (π-model), loads, generators
2. **Components** — each with a type, parameters, and a bus assignment
3. **Connections** — explicit wires that route signals between components

```json
{
  "components": {
    "GEN_1":   { "type": "GENROU_PHS", "bus": 1, "params": { "H": 6.5, "D": 0, ... } },
    "AVR_1":   { "type": "ESST3A_PHS", "bus": 1, "params": { "TR": 0.01, ... } },
    "GOV_1":   { "type": "TGOV1_PHS",  "bus": 1, "params": { "R": 0.05, ... } },
    "PSS_1":   { "type": "ST2CUT_PHS", "bus": 1, "params": { "K1": 10, ... } }
  },
  "connections": [
    { "src": "GEN_1.omega",  "dst": "GOV_1.omega" },
    { "src": "GEN_1.omega",  "dst": "PSS_1.omega" },
    { "src": "GEN_1.Pe",     "dst": "PSS_1.Pe" },
    { "src": "GOV_1.Tm",     "dst": "GEN_1.Tm" },
    { "src": "AVR_1.Efd",    "dst": "GEN_1.Efd" },
    { "src": "PSS_1.Vs",     "dst": "AVR_1.Vs" },
    { "src": "BUS_1.Vterm",  "dst": "AVR_1.Vterm" }
  ]
}
```

The `SystemGraph` turns this into a directed wiring graph. Every connection becomes a compile-time variable assignment in the generated C++. There are no runtime lookups, no string dispatching, no interpreter overhead — just flat arrays and direct variable references.

### The DAE Structure

The compiled system is a single DAE of the form:

$$
F(t,\; y,\; \dot{y}) = 0
$$

where the state vector $y$ contains both differential states (rotor angles, flux linkages, controller integrators) and algebraic states (bus voltages $V_d$, $V_q$ at every bus):

$$
y = \begin{bmatrix} x_{\text{diff}} \\ V_{d,1} \\ V_{q,1} \\ \vdots \\ V_{d,n} \\ V_{q,n} \end{bmatrix}
$$

The residual has two parts:

- **Differential equations** (one per component state): $\text{res}_i = \dot{y}_i - f_i(x, V)$
- **Algebraic equations** (KCL at every bus): $\text{res}_{\text{bus}} = \sum I_{\text{injected}} - Y_{\text{bus}} \cdot V = 0$

This formulation preserves the full network — no Kron reduction, no reduced-order approximation. Bus voltages are solved implicitly alongside the component dynamics at every time step. SUNDIALS IDA (variable-order BDF with adaptive step control) handles the coupled system.

### Why This Matters

Traditional power system simulators treat the network algebraic and the component differential equations separately: solve the network, update components, repeat. This partitioned approach introduces artificial interface errors at every step.

The DAE formulation solves everything simultaneously. The Newton iteration at each time step sees the full Jacobian — component dynamics coupled to network KCL — and converges to a consistent solution. Faults, topology changes, and nonlinear saturation are all handled within the same implicit solve.

---

## Running a Simulation

### DAE Simulation (Primary)

```bash
# C++ compiled solver (default — requires C++ toolchain + SUNDIALS)
python3 tools/run_dae_simulation.py cases/IEEE14Bus/bus_fault_phs.json --no-plot

# Numba JIT solver (no C++ needed, ~8× faster than pure Python)
python3 tools/run_dae_simulation.py cases/IEEE14Bus/bus_fault_phs_jit.json --no-plot

# SciPy Radau solver (no C++ needed, adaptive step)
python3 tools/run_dae_simulation.py cases/IEEE14Bus/bus_fault_phs_scipy.json --no-plot
```

The C++ path uses `DiracRunner` → `DiracCompiler` to generate a DAE C++ kernel with full Y-bus and SUNDIALS IDA solver. The Python-based solvers (`jit`, `scipy`) skip C++ generation and solve a network-reduced ODE system directly, requiring only Python dependencies.

Verified no-fault PHS DAE cases on the current generic initialization path:

- `cases/IEEE14Bus/no_fault_phs.json`
- `cases/SMIB/no_fault_phs.json`
- `cases/Kundur/no_fault_phs.json`
- `cases/IEEE39Bus/no_fault_phs.json`

### ODE Simulation (Secondary)

```bash
python3 tools/run_simulation.py cases/IEEE14Bus/no_fault_phs.json --no-plot
```

An alternative pipeline using Kron-reduced network + explicit RK4 (or SDIRK-2). Useful for cross-validation and faster exploratory runs.

### Scenario JSON Format

Both runners use the same scenario JSON:

```json
{
  "description": "IEEE 14-Bus PHS — bus fault at bus 7",
  "system": "system_phs.json",
  "solver": {
    "method": "ida",
    "dt": 0.0005,
    "duration": 15.0
  },
  "events": [
    { "type": "BusFault", "bus": 7, "t_start": 1.0, "t_end": 1.1,
      "fault_impedance": [0.0, 0.001] }
  ],
  "output": { "directory": "outputs/IEEE14Bus_phs_fault" },
  "plots": [
    { "title": "Rotor Angles", "y_label": "δ [rad]",
      "signals": [{ "pattern": "GENROU*delta" }] },
    { "title": "Terminal Voltages", "y_label": "|V| [pu]",
      "signals": [{ "pattern": "Vterm_Bus*" }] }
  ]
}
```

**Supported events:** `BusFault` (three-phase fault with impedance), `Toggler` (line trip / reconnection).

**Supported DAE solvers:** `ida` (SUNDIALS IDA, variable-order BDF — compiled C++), `bdf1` (backward Euler + Newton — compiled C++), `midpoint` (implicit midpoint, structure-preserving — compiled C++), `jit` (Numba JIT BDF-1 — network-reduced ODE, no C++ compilation needed), `scipy` (SciPy Radau IIA — network-reduced ODE, adaptive step).

**Supported ODE solvers** (secondary pipeline): `rk4` (explicit Runge-Kutta 4th order), `sdirk2` (singly-diagonal implicit RK).

---

## The Component Protocol

Every component in the framework is a Python class that inherits from `PowerComponent` and declares a fixed interface. The framework never inspects component internals — it only reads the interface.

```python
class MyExciter(PowerComponent):

    @property
    def component_role(self) -> str:
        return 'exciter'              # 'generator' | 'exciter' | 'governor' | 'pss' | 'passive'

    @property
    def state_schema(self) -> list:
        return ['Vm', 'Vr', 'Efd']    # state variable names → C++ x[0], x[1], x[2]

    @property
    def port_schema(self) -> dict:
        return {
            'in':  [('Vterm', 'signal', 'pu'), ('Vref', 'signal', 'pu'), ('Vs', 'signal', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def param_schema(self) -> dict:
        return {'TR': 'Measurement lag [s]', 'KA': 'Regulator gain', ...}

    def get_cpp_step_code(self) -> str:
        return "dxdt[0] = (Vterm - x[0]) / TR; ..."

    def get_cpp_compute_outputs_code(self) -> str:
        return "outputs[0] = x[2];"

    def get_symbolic_phs(self) -> SymbolicPHS:
      sphs = SymbolicPHS(...)
      sphs.set_init_spec(...)
      return sphs
```

The compiler reads `state_schema` to allocate memory, `port_schema` to resolve wires, `param_schema` to emit C++ parameter structs, and the `get_cpp_*` methods to emit the actual dynamics. The component is fully self-describing.

For controller-side PHS components, `init_from_targets()` no longer needs to be hand-written when `get_symbolic_phs()` declares an `InitSpec` via `sphs.set_init_spec(...)`. The base class dispatches to `solve_equilibrium()` automatically.

Current controller coverage on the generic initialization interface:

- Symbolic solve: `IEEEG1_PHS`, `TGOV1_PHS`, `EXDC2_PHS`, `IEEEX1_PHS`
- Callback-backed steady-state solve on the same interface: `EXST1_PHS`, `ESST3A_PHS`, `ST2CUT_PHS`, `IEEEST_PHS`

Current runtime auto-generation paths:

- Symbolic PHS runtime generation: `IEEEG1_PHS`, `TGOV1_PHS`, `EXDC2_PHS`, `IEEEX1_PHS`
- Signal-flow runtime generation: `ST2CUT_PHS`, `IEEEST_PHS`

`GENROU_PHS` uses `init_from_phasor()` for the initial phasor-based flux state computation. In the DAE pipeline no Kron reduction is ever performed — the DAE-consistent voltage solve (`Y_full · V = I_norton`) replaces the ODE pipeline's Kron equilibrium refinement entirely.

### Adding a New Component

1. Create a Python class inheriting `PowerComponent`
2. Implement the five schema properties + two code-generation methods
3. Register it in `src/json_compat.py` (two lines: import + dict entry)
4. Reference it in a JSON system file

No framework code changes. No if/else chains. The compiler handles any component that implements the protocol.

---

## Port-Hamiltonian Structure

### What Is a Port-Hamiltonian System?

A port-Hamiltonian system describes a physical component through three objects:

- **$H(x)$** — the Hamiltonian (stored energy as a function of states)
- **$J$** — skew-symmetric interconnection matrix (lossless energy routing)
- **$R$** — positive semi-definite dissipation matrix (energy lost to heat/friction)
- **$g$** — port coupling matrix (how external inputs enter the system)

The dynamics are:

$$
\dot{x} = (J - R)\,\nabla H(x) + g \cdot u
$$

The energy balance is guaranteed by construction:

$$
\dot{H} = -\nabla H^\top R\,\nabla H + \nabla H^\top g\,u \;\leq\; y^\top u
$$

The first term ($-\nabla H^\top R\,\nabla H$) is always non-positive — energy dissipation. The second term is the port power exchange with the environment. This means: **a PHS component can never generate energy internally.** Stability is structural.

### Symbolic PHS Layer

Each PHS component implements `get_symbolic_phs()` returning a `SymbolicPHS` object with SymPy matrices for $(J, R, g, Q, H)$. This symbolic definition is the **single source of truth** for:

| Derived artefact | How |
|---|---|
| C++ dynamics `dxdt[i] = ...` | `generate_phs_cpp_dynamics()` via SymPy → C99 code printer |
| Python callable $H(x) \to \mathbb{R}$ | `make_hamiltonian_func()` via SymPy lambdify |
| Python callable $\nabla H(x)$ | `make_grad_hamiltonian_func()` |
| Numerical matrices $\{J, R, g, Q\}$ | `evaluate_phs_matrices()` |
| C++ expression for $H(x)$ | `generate_hamiltonian_cpp_expr()` |
| Generic equilibrium init | `solve_equilibrium()` + `set_init_spec()` |
| LaTeX documentation | `phs_to_latex()` → standalone `.tex` |
| Structural validation | `validate_phs_structure()` — 9 machine-verifiable checks |

**Auto-generation of C++ dynamics:** When a component provides `get_symbolic_phs()` or `get_signal_flow_graph()`, the base class `get_cpp_step_code()` automatically generates C++ step code. Signal-flow graphs are preferred for controller block-diagram chains (PSS), while symbolic PHS remains the source for energy-structured components.

**Generic equilibrium initialization:** The same symbolic layer now drives controller initialization. `PowerComponent.init_from_targets()` calls `solve_equilibrium()` when the component's `SymbolicPHS` declares an `InitSpec`. Linear lag-chain controllers solve directly from `dynamics_expr = 0`; nonlinear rectifier or signal-flow models can register a callback while still using the same framework path.

Example — the IEEEG1 governor defines its entire dynamics in ~30 lines of symbolic math:

```python
def get_symbolic_phs(self):
    x1, x2 = sp.symbols('x_1 x_2')
    omega, Pref, u_agc = sp.symbols('omega P_ref u_agc')
    K, T1, T3 = sp.Symbol('K'), sp.Symbol('T_1'), sp.Symbol('T_3')

    H = sp.Rational(1, 2) * (x1**2 + x2**2)
    R = sp.diag(1/T1, 1/T3)
    J = sp.Matrix([[0, -1/(T1*T3)], [1/(T1*T3), 0]])
    g = sp.Matrix([[K/T1, K/T1, -K/T1], [0, 0, 0]])

    # Explicit dynamics matching the IEEE standard cascaded-lag model
    err = Pref + u_agc - omega
    dynamics_expr = sp.Matrix([(K*err - x1)/T1, (x1 - x2)/T3])

    return SymbolicPHS(name='IEEEG1', states=[x1, x2], inputs=[omega, Pref, u_agc],
                       params={'K': K, 'T1': T1, 'T3': T3},
                       J=J, R=R, g=g, H=H, dynamics_expr=dynamics_expr)
```

The framework generates this C++:

```cpp
// Auto-generated from SymbolicPHS 'IEEEG1_PHS'
double x1 = x[0];
double x2 = x[1];
double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];
dxdt[0] = K*Pref/T1 - K*omega/T1 + K*u_agc/T1 - x1/T1;
dxdt[1] = x1/T3 - x2/T3;
```

No hand-written C++. The symbolic PHS is the code.

### Compile-Time Validation

When the compiler builds the system, it runs `validate_phs_structure()` on every PHS component before generating any C++. This catches structural errors — a non-skew-symmetric $J$, a non-positive-semi-definite $R$ — at build time, not at runtime.

---

## Initialization Pipeline

Power system initialization is not trivial: the steady-state operating point of 50+ coupled states must be found before simulation starts. The two pipelines share the first five steps, then diverge for network consistency.

### Shared steps (both pipelines)

1. **Power flow** — Newton-Raphson AC power flow for bus voltages $(V, \theta)$
2. **Generator initialization** — Park transform → stator algebraic → initial flux states from $(V, I)$ phasors
3. **Exciter initialization** — Generic `solve_equilibrium()` / `InitSpec` path for controller-side PHS models, producing the required $E_{fd}$ at steady state
4. **PSS initialization** — Generic `InitSpec` steady-state solve (callback-backed for ST2CUT/IEEEST), with runtime equations auto-generated from `SignalFlowGraph`
5. **Governor initialization** — Generic `solve_equilibrium()` back-solve from mechanical torque $T_m = T_e$ at equilibrium

### DAE pipeline (step 6) — full Y-matrix, no Kron reduction

6. **DAE-consistent voltage solve** — Iteratively solve $V = Y_{\text{bus}}^{-1} \cdot I_{\text{norton}}$ against the **full** Y-bus (all buses present). Generator Eq′ states are adjusted until PV-bus voltage setpoints are met, then $T_m$, governor, and exciter states are rebalanced to the DAE operating point. No buses are ever eliminated.

### ODE pipeline (step 6) — Kron-reduced Z-bus

6. **Kron equilibrium convergence** — Generator bus voltages are computed from the **Kron-reduced** Z-bus (load and passive buses eliminated). Component states are iteratively refined against the reduced network until self-consistent.

Generators still implement `init_from_phasor()`. Controller-side PHS components now declare an `InitSpec` on their `SymbolicPHS`, and the base class routes `init_from_targets()` through the generic equilibrium interface automatically.

For DAE builds, the Python pre-solve now enforces the same static-slack bus constraints as the generated IDA algebraic model, so the compiled DAE no longer needs to shift voltages at startup to reconcile the network equations.

The current verified DAE no-fault cases on this path are IEEE14Bus, SMIB, Kundur, and IEEE39Bus. IEEE14Bus and IEEE39Bus now run with ST2CUT/IEEEST runtime code generated from `SignalFlowGraph`.

---

## Co-Simulation

The framework supports a two-rate co-simulation mode where the compiled C++ plant runs at high frequency (physics time step) and Python controllers run at a lower communication rate.

```python
from cosim import CosimOrchestrator, PlantInterface, CosimConfig

plant = PlantInterface("outputs/plant.so", config)
orchestrator = CosimOrchestrator(plant, config, dt_phy=0.001, dt_ctrl=0.1)

def my_agc_controller(bundle):
    freq_error = bundle.meas['omega_COI'] - 1.0
    return np.array([−10.0 * freq_error])

orchestrator.register_controller(my_agc_controller)
orchestrator.run(duration=60.0)
```

The C++ plant exposes a C ABI (`plant.so`) with functions for `init`, `step_rk4`, `set_inputs`, `get_outputs`, `get_state`, and `swap_topology`. The Python side wraps this via `ctypes`. Power-port physics stays inside C++ — only control signals cross the boundary.

This enables:
- **AGC/secondary frequency control** studies at system level
- **Reinforcement learning** agents controlling grid assets
- **Hardware-in-the-loop** prototyping with Python as the control layer
- **Topology switching** (line trips, breaker operations) at runtime

---

## Component Library

### Synchronous Generators

| Model | States | Description |
|---|---|---|
| `GENROU` / `GENROU_PHS` | 6 | Round-rotor: $\delta, \omega, E'_q, E'_d, \psi''_d, \psi''_q$ |
| `GENSAL` | 5 | Salient-pole: $\delta, \omega, E'_q, \psi''_d, \psi''_q$ |
| `GENTPF` | 6 | Round-rotor with multiplicative saturation |
| `GENTPJ` | 6 | GENTPF + Kis armature leakage saturation |
| `GENCLS` | 2 | Classical: $\delta, \omega$ |

### Exciters

| Model | States | Standard |
|---|---|---|
| `ESST3A` / `ESST3A_PHS` | 5 | IEEE ST3A compound-source rectifier |
| `EXST1` / `EXST1_PHS` | 4 | IEEE ST1A static exciter |
| `EXDC2` / `EXDC2_PHS` | 4 | IEEE DC2A DC commutator |
| `IEEEX1` / `IEEEX1_PHS` | 5 | IEEE Type 1 DC exciter |

### Governors

| Model | States | Standard |
|---|---|---|
| `TGOV1` / `TGOV1_PHS` | 3 | IEEE steam turbine governor |
| `IEEEG1` / `IEEEG1_PHS` | 2 | IEEE Type G1 multi-stage steam turbine |

### Power System Stabilizers

| Model | States | Standard |
|---|---|---|
| `ST2CUT` / `ST2CUT_PHS` | 6 | Dual-input PSS (speed + power) |
| `IEEEST` / `IEEEST_PHS` | 7 | IEEE standard single-input PSS |

### Renewable / IBR Models

| Model | States | Description |
|---|---|---|
| `REGCA1` | 3 | WECC renewable generator/converter |
| `REECA1` | 4 | WECC renewable electrical controller |
| `REPCA1` | 5 | WECC renewable plant controller |
| `PVD1` | 0 | WECC distributed PV (stateless) |
| `DGPRCT1` | 1 | DG protection relay |
| `VOC_INVERTER` | 2 | Virtual Oscillator Control grid-forming inverter |

### DFIG Wind Turbine (Full Model)

| Model | States | Description |
|---|---|---|
| `DFIG` | 5 | Doubly-fed induction generator (stator flux PCH) |
| `DFIG_RSC` | 4 | Rotor-side converter (cascaded PI, SFO) |
| `DFIG_GSC` | 2 | Grid-side converter controller |
| `DFIG_DCLINK` | 1 | DC-link capacitor |
| `DFIG_DRIVETRAIN` | 2 | Two-mass flexible shaft (PCH) |
| `WIND_AERO` | 0 | Aerodynamics (Cp lookup tables) |

### Other

| Model | States | Description |
|---|---|---|
| `AGC` | 1 | Automatic Generation Control (area-based) |
| `BUSFREQ` | 2 | Bus frequency estimator |
| `PMU` | 0 | Phasor measurement unit |
| `PI_LINE` | variable | Pi-section transmission line (dynamic PH model) |
| `TRANSFORMER_2W` | 0 | Two-winding transformer |

All `_PHS` variants implement `get_symbolic_phs()` with SymPy matrices for $(J, R, g, Q, H)$, enabling auto-generated C++ dynamics, symbolic validation, and LaTeX export.

---

## Test Cases

| System | Description | Key scenarios |
|---|---|---|
| `SMIB` | Single machine infinite bus | Bus fault, line trip, co-simulation, DAE/IDA |
| `IEEE14Bus` | IEEE 14-bus (5 generators) | Bus fault, line trip, mid-line fault, PHS variants |
| `IEEE39Bus` | IEEE 39-bus New England (10 gen) | Bus fault, mid-line fault, PHS variants |
| `Kundur` | Kundur two-area (4 generators) | Inter-area oscillations, bus fault, PHS |
| `IEEE14Bus_DFIG` | IEEE 14-bus + DFIG wind turbine | Fault ride-through |
| `IEEE14Bus_Solar` | IEEE 14-bus + REGCA1/REECA1/REPCA1 | Solar + grid-following inverter |
| `IEEE14Bus_PVD1` | IEEE 14-bus + distributed PV | PVD1 integration |
| `IEEE14Bus_VOC` | IEEE 14-bus + VOC inverter | Grid-forming inverter dynamics |
| `DFIG_Full_WT` | Full wind turbine chain | DFIG + RSC + GSC + DC-link + drivetrain + aero + wind file |

---

## Project Structure

```
PHPS/
├── src/                              # Core framework
│   ├── core.py                       # PowerComponent base class + auto-generation
│   ├── system_graph.py               # SystemGraph: topology + wiring
│   ├── compiler.py                   # SystemCompiler: ODE C++ kernel generation
│   ├── initialization.py             # Multi-pass state initializer
│   ├── runner.py                     # SimulationRunner (ODE pipeline)
│   ├── powerflow.py                  # Newton-Raphson AC power flow
│   ├── ybus.py                       # Y-bus assembly; Z-bus / Kron reduction (ODE pipeline only)
│   ├── json_compat.py                # JSON format upgrade + component registry
│   ├── errors.py                     # Structured error types
│   │
│   ├── dirac/                        # DAE pipeline (primary)
│   │   ├── incidence.py              # Network incidence matrix B
│   │   ├── dirac_structure.py        # Dirac subspace: power conservation verification
│   │   ├── hamiltonian.py            # Total Hamiltonian assembler
│   │   ├── dae_compiler.py           # DiracCompiler: full Y-bus → C++ DAE + IDA
│   │   ├── dae_runner.py             # DiracRunner: end-to-end build/compile/run
│   │   ├── jit_solver.py             # Numba JIT BDF-1 solver (network-reduced ODE)
│   │   └── py_solver.py              # Python/SciPy Radau solver (network-reduced ODE)
│   │
│   ├── symbolic/                     # Symbolic PHS layer (SymPy)
│   │   ├── core.py                   # SymbolicPHS: (J, R, Q, g, H) representation
│   │   ├── codegen.py                # SymPy → C++ code generation
│   │   ├── validation.py             # Structural PHS validation (9 checks)
│   │   └── latex_export.py           # Publication-ready LaTeX generation
│   │
│   └── components/                   # Component library (~40 models)
│       ├── generators/               # GENROU, GENCLS, GENSAL, GENTPF, GENTPJ
│       ├── exciters/                 # ESST3A, EXST1, EXDC2, IEEEX1, AVR
│       ├── governors/                # TGOV1, IEEEG1
│       ├── pss/                      # ST2CUT, IEEEST
│       ├── renewables/               # DFIG chain, PVD1, REGCA1/REECA1/REPCA1, VOC
│       ├── control/                  # AGC
│       └── measurements/             # BusFreq, PMU
│
├── tools/                            # Command-line entry points
│   ├── run_dae_simulation.py         # Primary: DAE simulation runner
│   ├── run_simulation.py             # Secondary: ODE simulation runner
│   └── plot_results.py               # Plot results from CSV
│
├── cosim/                            # Co-simulation layer
│   ├── orchestrator.py               # Two-rate simulation orchestrator
│   ├── plant_interface.py            # ctypes FFI to compiled plant.so
│   ├── signals.py                    # CosimConfig, CosimBundle, port definitions
│   ├── zone_manager.py              # Zone-based signal routing + breakers
│   └── logger.py                     # Port logging
│
├── cases/                            # Test systems and scenario files
│   ├── SMIB/                         # Single machine infinite bus
│   ├── IEEE14Bus/                    # IEEE 14-bus system
│   ├── IEEE39Bus/                    # IEEE 39-bus New England
│   ├── Kundur/                       # Kundur two-area system
│   ├── IEEE14Bus_DFIG/               # DFIG integration
│   ├── IEEE14Bus_Solar/              # Solar (REGCA1 chain)
│   ├── IEEE14Bus_PVD1/               # Distributed PV
│   ├── IEEE14Bus_VOC/                # VOC grid-forming
│   └── DFIG_Full_WT/                 # Full wind turbine model
│
├── doc/                              # LaTeX manual
└── outputs/                          # Simulation results (CSV + plots)
```

---

## Installation

### Requirements

- Python 3.10+
- g++ (C++17 compatible)
- SUNDIALS library (for IDA solver)

### Python dependencies

```bash
pip install numpy pandas matplotlib scipy sympy networkx
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# No-fault equilibrium check (60 seconds)
python3 tools/run_dae_simulation.py cases/IEEE14Bus/no_fault_phs.json

# Bus fault transient (15 seconds, fault at t=1.0s cleared at t=1.1s)
python3 tools/run_dae_simulation.py cases/IEEE14Bus/bus_fault_phs.json

# Skip plotting
python3 tools/run_dae_simulation.py cases/IEEE14Bus/no_fault_phs.json --no-plot

# Additional verified no-fault PHS cases
python3 tools/run_dae_simulation.py cases/SMIB/no_fault_phs.json --no-plot
python3 tools/run_dae_simulation.py cases/Kundur/no_fault_phs.json --no-plot
python3 tools/run_dae_simulation.py cases/IEEE39Bus/no_fault_phs.json --no-plot
```

---

## Design Principles

### Everything Is a Connection

There is no hardcoded knowledge of "which exciter goes with which generator" in the framework core. All relationships are explicit wires in the JSON:

```json
{ "src": "GEN_1.omega", "dst": "GOV_1.omega" }
```

The compiler resolves every wire to a direct C++ variable reference at code-generation time. This means:
- Any component can be wired to any other component
- New component types need zero framework changes
- The wiring graph is the system's mathematical structure
- Incorrect wiring is caught at validation time with actionable error messages

### Compile Once, Run Fast

The Python layer runs once (power flow, initialization, code generation, compilation). The resulting C++ binary runs the entire time-domain simulation with no Python overhead. The C++/IDA backend achieves a 378× speedup over pure Python on the IEEE 14-bus benchmark. For rapid prototyping without a C++ compilation step, the Numba JIT backend provides an 8.1× speedup while keeping all component models in readable Python.

### Two Pipelines, One System File

The same JSON system file works with both pipelines:

| | DAE Pipeline (`run_dae_simulation.py`) | ODE Pipeline (`run_simulation.py`) |
|---|---|---|
| **Network model** | **Full Y-bus** — all buses present, no elimination | **Kron-reduced Z-bus** — load/passive buses eliminated |
| Solver | C++: IDA / BDF-1 / Midpoint; Python: JIT / SciPy Radau | RK4 / SDIRK-2 (explicit/implicit) |
| Bus voltages | Part of the DAE state vector — solved implicitly at every step | Solved in an explicit algebraic loop via LU at each step |
| Init network step | Full Y-bus: $V = Y^{-1} I_{\text{norton}}$ — no Kron reduction ever | Kron Z-bus: iterative refinement on reduced network |
| Primary use | Production simulation (C++/IDA) and prototyping (JIT/SciPy) | Cross-validation, exploration |

### Generic at the Protocol Level

The framework operates on the `PowerComponent` protocol — not on specific component names or types. The compiler, initializer, and runner contain no `isinstance()` checks. They ask each component: "what are your states?", "what are your ports?", "what is your C++?" — and compose the system from the answers.

This means the framework handles synchronous generators, DFIGs, solar inverters, VOC grid-forming converters, and any future component type through the same pipeline, without modification.

---

## Symbolic PHS Analysis Tools

Beyond simulation, the symbolic layer supports standalone analysis:

```python
from src.components.governors.ieeeg1_phs import Ieeeg1PHS

comp = Ieeeg1PHS("GOV_1", params)
sphs = comp.get_symbolic_phs()

# Structural validation
from src.symbolic.validation import validate_phs_structure
report = validate_phs_structure(sphs)
print(report)  # 9 checks: J skew-symmetry, R PSD, ...

# LaTeX export
from src.symbolic.latex_export import phs_to_latex
tex = phs_to_latex(sphs)        # single component
# or full document:
from src.symbolic.latex_export import phs_collection_to_tex_document
phs_collection_to_tex_document([sphs1, sphs2, ...], "output.tex")

# Symbolic dynamics
print(sphs.dynamics)             # SymPy column vector: dx/dt
print(sphs.power_balance)        # dH/dt = dissipation + supply
print(sphs.dissipation_rate)     # -∇H^T R ∇H ≤ 0

# Numerical evaluation
H_val = comp.hamiltonian(x)      # H(x) → float
grad = comp.grad_hamiltonian(x)  # ∇H(x) → ndarray
mats = comp.get_phs_matrices(x)  # {J, R, g, Q} → ndarrays
```

---

## License

See LICENSE file.
