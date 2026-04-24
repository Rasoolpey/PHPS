# PH Power-System Framework ÔÇö Toolbox

A curated set of analysis tools built on top of the **Port-Hamiltonian (PH)**
power-system simulation framework.  Each tool uses the framework's compiler and
initializer pipeline in `src/` to extract physically meaningful information from
a power-system model ÔÇö **no prior simulation run is required**.

---

## Quick-start

```bash
# A-1 Equilibrium Checker
python3 tools/check_equilibrium.py                                    # IEEE14Bus (default)
python3 tools/check_equilibrium.py --case cases/Kundur/system.json    # Kundur
python3 tools/check_equilibrium.py --case cases/IEEE39Bus/system.json  # IEEE39Bus
python3 tools/check_equilibrium.py --case cases/Kundur/system.json --save-report  # save full table

# A-3 Eigenvalue Analyzer
python3 tools/compute_eigenvalues.py                                   # IEEE14Bus (default)
python3 tools/compute_eigenvalues.py --case cases/Kundur/system.json   # Kundur
python3 tools/compute_eigenvalues.py --case cases/Kundur/system.json --plot            # save figure
python3 tools/compute_eigenvalues.py --case cases/Kundur/system.json --plot --component EXDC2_2

# A-4 Participation Factor Analyzer
python3 tools/compute_participation.py                                  # IEEE14Bus (default)
python3 tools/compute_participation.py --case cases/Kundur/system.json --plot          # heatmap
python3 tools/compute_participation.py --case cases/Kundur/system.json --save-report   # text report

# A-5 Inter-Area Oscillation Detector
python3 tools/detect_oscillations.py                                    # IEEE14Bus
python3 tools/detect_oscillations.py --case cases/Kundur/system.json --plot
python3 tools/detect_oscillations.py --case cases/Kundur/system.json --save-report

# A-10 Component I/O Inspector
python3 tools/inspect_equilibrium.py                                         # IEEE14Bus (default)
python3 tools/inspect_equilibrium.py --case cases/SMIB_IdaPBC/system.json    # SMIB with IDA-PBC
python3 tools/inspect_equilibrium.py --case cases/SMIB_IdaPBC/system.json --component IDAPBC_1
python3 tools/inspect_equilibrium.py --case cases/Kundur/system.json --save-report
```

All tools accept `--help` for a full list of options.

---

## Output Convention

Every toolbox script writes its output files (figures, CSVs, reports) to:

```
outputs/<CaseName>/tools/
```

The case name is derived automatically from the JSON path ÔÇö no manual path
needed.  Example directory tree after running A-1 and A-3 on three cases:

```
outputs/
  Kundur/
    tools/
      equilibrium_report.txt      ÔåÉ A-1 --save-report
      eigenvalues.png             ÔåÉ A-3 --plot
      eigenvalues_EXDC2_2.png     ÔåÉ A-3 --plot --component EXDC2_2
      jacobian.csv                ÔåÉ A-3 --save-csv
  IEEE14Bus/
    tools/
      equilibrium_report.txt
      eigenvalues.png
  IEEE39Bus/
    tools/
      equilibrium_report.txt
      eigenvalues.png
```

All output paths are created automatically (`mkdir -p`).  You can override any
path by supplying an explicit file argument to the relevant flag.

---

## Relation to the Port-Hamiltonian Format

The framework models each dynamic component (generator, exciter, governor, PSS)
as a PH subsystem with state vector **x**, interconnection matrix **J**, damping
matrix **R**, and input/output ports.  The assembled system takes the form:

$$\dot{x} = (J - R)\,\nabla H(x) + G\,u$$

where *H* is the Hamiltonian (stored energy).  Each toolbox script drives the
framework's `SystemCompiler` and `Initializer` directly, runs the full power-flow
and state-refinement pipeline, then generates the C++ kernel via
`compiler.generate_cpp()` ÔÇö all in memory and for any case JSON supplied.

The toolbox tools exploit this for three classes of analysis:

| Class | Tools | What they evaluate |
|---|---|---|
| **Equilibrium** | A-1, A-10 | Is **f(xÔéÇ) Ôëê 0**?  Component I/O and wiring consistency at xÔéÇ |
| **Small-signal** | A-3, A-4, A-5 | Eigenvalues of **Ôêéf/Ôêéx\|_{xÔéÇ}** (Jacobian) |
| **Network** | A-6 | Graph Laplacian of the admittance network |
| **Transient** | A-9 | Critical Clearing Time via bisection |

---

## Tools

### A-1 ÔÇö Equilibrium Checker (`check_equilibrium.py`)

#### What it does

Verifies that **ß║ï = f(xÔéÇ) Ôëê 0** at the operating point produced by the
initialization pipeline ÔÇö the fundamental PH equilibrium condition.

The tool is self-contained: given only a `system.json` it:

1. Runs the full initialization pipeline internally (power flow ÔåÆ Kron-reduced
   network correction ÔåÆ exciter / d-axis / governor refinement ÔÇö the same
   sequence used by `SimulationRunner.build()`).
2. Calls `compiler.generate_cpp()` to produce the C++ kernel in memory.
3. Injects a minimal `main()` that evaluates `system_step(xÔéÇ, fÔéÇ, ...)` once
   and writes `fÔéÇ` to a temporary CSV.
4. Compiles with `g++ -O2`, runs it, reads the residuals, and maps every index
   to a human-readable `COMPONENT.state_name` label.

This is a **design-time** check: run it on a new model before committing to a
full transient simulation to catch initialization problems early.

A perfect equilibrium has **f(xÔéÇ) = 0** for every state.  Non-zero residuals reveal:
- Initialization inconsistency (power-flow solution not fully compatible with the dynamic model)
- Parameter error that prevents a valid steady state (e.g. wrong saturation curve, wrong Vref)
- Numerical ill-conditioning in a specific component

#### How to run

```bash
# Default: IEEE14Bus, tol = 1e-4
python3 tools/check_equilibrium.py

# Other cases
python3 tools/check_equilibrium.py --case cases/Kundur/system.json
python3 tools/check_equilibrium.py --case cases/IEEE39Bus/system.json

# Tighter tolerance, quiet mode
python3 tools/check_equilibrium.py --case cases/Kundur/system.json --tol 1e-5 --quiet

# Save violations report to outputs/Kundur/tools/equilibrium_report.txt
python3 tools/check_equilibrium.py --case cases/Kundur/system.json --save-report
```

#### Example outputs

**Kundur (4-machine, 2-area) ÔÇö PASS:**
```
=================================================================
  PH Equilibrium Checker ÔÇö A-1
=================================================================
  Case : cases/Kundur/system.json
  Tol  : 1.0e-04

  Running power flow ...
  Correcting states to Kron-reduced network ...
  Refining exciter voltages ...
  Refining d-axis / Eq_p coupled states ...
  Refining governors / renewable controllers ...
  Generating C++ kernel ... 53 states
  Compiling ... OK
  Evaluating f(x0) ... OK

  States   : 53
  Max |ß║ïßÁó| : 4.571e-05

  All 53 derivatives satisfy |ß║ïßÁó| < 1.0e-04

  Ô£ô  EQUILIBRIUM VERIFIED
=================================================================
```

**IEEE39Bus (10-machine New England) ÔÇö residuals found:**
```
  States   : 211
  Max |ß║ïßÁó| : 1.053e-03

  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
     i  Component.State                                   dxdt
  ÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    75  IEEEX1_4.Vm                               1.053e-03
    90  IEEEX1_7.Vm                              -1.022e-03
    85  IEEEX1_6.Vm                              -5.996e-04
    ...

  Ô£ù  EQUILIBRIUM NOT VERIFIED  (tol = 1.0e-04)
```

The `Vm` residuals on the IEEEX1 exciters are voltage transducer states whose
initial value has a small lag behind the network solution ÔÇö a known convergence
artefact that can be tightened by adjusting exciter `TR` or running more
refinement iterations.

#### Relation to PH format

In the PH framework the equilibrium condition at steady state (constant input
`u = ┼½`) is:

$$( J(x^*) - R(x^*) )\,\nabla H(x^*) + G\,\bar{u} = 0$$

Any non-zero entry in `f(xÔéÇ)` means the corresponding subsystem's stored-energy
gradient is not at rest.  Because the tool evaluates the **full assembled C++
kernel** ÔÇö including all port interconnections, the Kron-reduced network, and the
network Norton current injection ÔÇö the residual is physically meaningful across
the entire multi-machine system.

#### Use in control design and parameter tuning

| Workflow | How this tool helps |
|---|---|
| **New model validation** | Verify the operating point before running any simulation |
| **Parameter tuning** | Tighten exciter `TR`, governor `T1`, or PSS gains and re-run until `max\|ß║ïßÁó\|` drops below tolerance |
| **Saturation fitting** | Non-zero `psi_d` / `psi_q` residuals point directly to incorrect saturation coefficients |
| **Multi-case sweep** | Run with `--quiet` across all cases in a CI pipeline; exit code 0 = pass, 1 = fail |
| **Post-fault steady state** | Re-initialize with a modified topology (line trip, load change) and check the new equilibrium |

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--tol` | `1e-4` | Residual threshold |
| `--save-report` | off | Save violations table to `outputs/<CaseName>/tools/equilibrium_report.txt` |
| `--quiet` | off | Print only the final PASS/FAIL line (for scripting) |

---

### A-3 ÔÇö Eigenvalue Analyzer (`compute_eigenvalues.py`)

#### What it does

Computes the linearisation Jacobian **A = Ôêéf/Ôêéx|_{xÔéÇ}** via forward finite
differences entirely in compiled C++ and analyzes its eigenspectrum for:

- **Small-signal stability** ÔÇö all Re(╬╗ßÁó) Ôëñ 0 required (VSJ14 ┬º7.1, Thm. 7.1)
- **Damping ratios** ÔÇö ╬ÂßÁó = ÔêÆRe(╬╗ßÁó)/|╬╗ßÁó|, flagged if ╬Â < 0.03 (WECC threshold)
- **Oscillation frequencies** ÔÇö fßÁó = Im(╬╗ßÁó)/(2¤Ç) [Hz]
- **PH structure checks**:
  - Purely imaginary eigenvalues ÔåÆ **lossless (Casimir) modes**: ker(R) Ôê® im(J)
  - Zero eigenvalues ÔåÆ **structural zeros**: conserved quantities / constraints

Like A-1, the tool is **self-contained** ÔÇö no prior simulation or `jacobian.csv`
needed. It runs the full init pipeline, generates the C++ kernel in memory, and
computes the A matrix in a single compiled binary.

#### How to run

```bash
# Default: IEEE14Bus
python3 tools/compute_eigenvalues.py

# Specific case
python3 tools/compute_eigenvalues.py --case cases/Kundur/system.json

# Show top 40 modes, save figure + Jacobian CSV
python3 tools/compute_eigenvalues.py --n-modes 40 --plot --save-csv

# Component sub-block analysis with figure
python3 tools/compute_eigenvalues.py --case cases/Kundur/system.json --component EXDC2_2 --plot

# CI-pipeline one-liner
python3 tools/compute_eigenvalues.py --quiet
```

#### Example output ÔÇö Kundur 4-machine

```
  EIGENVALUE ANALYSIS  (top 30 modes)
  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
     #         Re(╬╗)         Im(╬╗)   Freq(Hz)        ╬Â  Status        Top state
  ÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
     1      -0.17727       4.08683     0.6504   0.0433                EXDC2_2.Vr
     2      -0.72319       7.05690     1.1231   0.1019                EXDC2_2.Vr
     3      -0.76599       7.26805     1.1567   0.1048                EXDC2_3.Vr
  ...
  Summary
  Total modes          : 53
  Unique modes shown   : 37
  UNSTABLE  Re(╬╗)>0   : 0
  WECC_WARN ╬Â<0.03    : 0
  LOSSLESS  |Re(╬╗)|Ôëê0 : 0  [Casimir-dominated]
  ZERO      |╬╗|Ôëê0     : 6  [conserved quantities]
  Max Re(╬╗)            : +0.0000e+00
  Min damping ╬Â        : +0.0433

  PH stability check (Re(╬╗ßÁó) Ôëñ 0 ÔêÇi  [VSJ14 ┬º7.1, Thm. 7.1]):
  Ô£ô  PASSED ÔÇö all eigenvalues in the closed left half-plane
```

#### Verified results (25 Feb 2026)

| Case | States | Min ╬Â | Max Re(╬╗) | WECC warns | Result |
|---|---|---|---|---|---|
| Kundur 4-machine | 53 | 0.0433 | Ôëê 0 | 0 | **STABLE** |
| IEEE 14-bus | 81 | 0.2028 | Ôëê 0 | 0 | **STABLE** |
| IEEE 39-bus | 211 | 0.0703 | Ôëê 0 | 0 | **STABLE** |

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--eps` | `1e-6` | Finite-difference step size |
| `--n-modes` | `30` | Rows in the printed mode table |
| `--plot` | off | Save eigenvalue figure to `outputs/<CaseName>/tools/eigenvalues.png`; supply a path to override |
| `--save-csv` | off | Save Jacobian matrix to `outputs/<CaseName>/tools/jacobian.csv`; supply a path to override |
| `--component` | off | Also analyze the sub-Jacobian block of one component (e.g. `GENROU_1`) |
| `--quiet` | off | One-line summary only (CI-pipeline mode) |

**Exit codes**: 0 = stable, 1 = unstable mode found, 2 = WECC warning only.

---

### A-4 ÔÇö Participation Factors (`compute_participation.py`)

#### What it does

For every eigenmode of the linearised Jacobian **A** it computes the
**participation matrix**:

$$P_{ki} = \phi_{ki} \cdot \psi_{ik}$$

where $\phi_i$ is the right eigenvector and $\psi_i$ is the corresponding row of
$\Phi^{-1}$ (left eigenvector). Each column is normalised so $\sum_k |P_{ki}| = 1$.
A large entry tells you that **state k is the dominant driver of mode i**.

**PH interpretation:**
- High participation of `GENROU.delta` / `GENROU.omega` in a 0.1ÔÇô0.8 Hz mode ÔåÆ electromechanical inter-area mode (J-dominated)
- High participation of `EXDC.Vr` / `ESST3A.VM` ÔåÆ exciter regulator is driving the oscillation (R-dominated)
- High participation of a governor state ÔåÆ slow governor interaction

#### How to run

```bash
python3 tools/compute_participation.py                                    # IEEE14Bus
python3 tools/compute_participation.py --case cases/Kundur/system.json   # Kundur
python3 tools/compute_participation.py --case cases/Kundur/system.json --plot         # heatmap figure
python3 tools/compute_participation.py --case cases/Kundur/system.json --mode 0       # inspect mode 0 in full
python3 tools/compute_participation.py --case cases/Kundur/system.json --save-report  # text report
python3 tools/compute_participation.py --quiet                            # CI one-liner
```

#### Example output (Kundur, Mode 1 ÔÇö lowest damped at ╬Â=0.043)

```
Mode   1  ╬╗ = -0.1773 +4.0868j   f=0.650Hz   ╬Â=0.0433
  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  Rank  Component.State                               |P|
  ÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ  ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
     1  EXDC2_2.Vr                                 0.2180
     2  EXDC2_1.Vr                                 0.1848
     3  EXDC2_2.Vm                                 0.1354
     4  EXDC2_1.Vm                                 0.1272
     5  GENROU_1.omega                             0.0576
     6  GENROU_2.omega                             0.0541
     7  GENROU_2.delta                             0.0538
     8  GENROU_1.delta                             0.0497
```

The exciter voltage states dominate the 0.65 Hz inter-machine oscillation,
identifying them as the primary target for PSS / damping-injection tuning.

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--eps` | `1e-6` | FD step size for Jacobian |
| `--n-modes` | `30` | Modes to display/save |
| `--n-states` | `8` | Top states per mode |
| `--mode N` | off | Inspect a single mode index in full detail |
| `--plot` | off | Save heatmap to `outputs/<CaseName>/tools/participation.png` |
| `--save-report` | off | Save text report to `outputs/<CaseName>/tools/participation_report.txt` |
| `--quiet` | off | One-line summary (CI mode) |

**Exit codes**: 0 = all modes adequately damped, 1 = WECC/unstable mode found.

---

### A-5 ÔÇö Inter-Area Oscillation Detector (`detect_oscillations.py`)

#### What it does

Identifies oscillatory eigenmodes of the Jacobian and classifies them by frequency band, then determines which generators form **coherent groups** that swing together vs. against each other.

For each mode the tool extracts the complex right-eigenvector entries at every generator's `delta` state.  Generators with similar phasor angles ($\angle\phi_{\delta_k} \approx \angle\phi_{\delta_j}$) belong to the same coherent group; generators with $\approx 180┬░$ phase difference are swinging in opposition across a transmission corridor.

**Frequency bands:**

| Band | Range | Physical meaning |
|---|---|---|
| inter-area | 0.1 ÔÇô 0.8 Hz | Groups of generators swinging against each other across tie-lines |
| local | 0.8 ÔÇô 2.0 Hz | One generator swinging against its local network |
| control | 2.0 ÔÇô 4.0 Hz | Exciter / PSS / governor loop interactions |

#### How to run

```bash
python3 tools/detect_oscillations.py                                      # IEEE14Bus
python3 tools/detect_oscillations.py --case cases/Kundur/system.json      # Kundur
python3 tools/detect_oscillations.py --case cases/Kundur/system.json --plot
python3 tools/detect_oscillations.py --case cases/Kundur/system.json --save-report
python3 tools/detect_oscillations.py --quiet
```

#### Example output (Kundur 4-machine, 0.650 Hz inter-area mode)

```
  ÔöÇÔöÇ INTER-AREA MODES  (0.1ÔÇô0.8 Hz) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    ╬╗ = -0.1773 +4.0868j   f = 0.650 Hz   ╬Â = 0.0433
      Group 1: GENROU_3, GENROU_4          ÔåÉ Area 2
      Group 2: GENROU_1, GENROU_2          ÔåÉ Area 1

      Generator              |¤å_╬┤|    Ôêá¤å_╬┤ (┬░)  Group
      GENROU_1             0.27314     -179.31      2
      GENROU_2             0.21570     -179.17      2
      GENROU_3             0.22734        0.37      1
      GENROU_4             0.28724        1.06      1

  Ô£ö  All inter-area modes adequately damped  [PASS]
```

The 180┬░ phase separation between Area 1 (G1, G2) and Area 2 (G3, G4) is the canonical Kundur 2-area inter-area oscillation.

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--eps` | `1e-6` | FD step for Jacobian |
| `--groups` | `2` | Number of coherent groups to identify |
| `--plot` | off | Save phasor + amplitude figure to `outputs/<CaseName>/tools/oscillations.png` |
| `--save-report` | off | Save report to `outputs/<CaseName>/tools/oscillation_report.txt` |
| `--quiet` | off | One-line summary (CI mode) |

**Exit codes**: 0 = all inter-area modes adequately damped, 1 = poorly-damped/unstable inter-area mode found.

---

### A-6 ÔÇö Graph Laplacian Spectrum Analyzer (`graph_laplacian.py`)

#### What it does

Builds the weighted graph Laplacian $L = BKB^\top$ directly from `system.json` (no
dynamic initialisation needed) and performs a spectral analysis.  Key outputs:

| Quantity | Meaning |
|---|---|
| ╬╗Ôéü = 0 | Trivial null mode (always present for a connected graph) |
| **╬╗Ôéé** (Fiedler) | Algebraic connectivity ÔÇö small ╬╗Ôéé ÔåÆ weak tie / near-islanding |
| Spectral gap ╬╗ÔéâÔêÆ╬╗Ôéé | Robustness of the weakest cut |
| **Fiedler vector** | Sign pattern partitions buses into the two sides of the weakest cut |

$B$ is the $(n_{\text{bus}} \times n_{\text{branch}})$ signed incidence matrix;
$K = \text{diag}(b_1, \ldots, b_m)$ uses susceptances $b_k = 1/x_k$.

#### How to run

```bash
python3 tools/graph_laplacian.py                                      # IEEE14Bus
python3 tools/graph_laplacian.py --case cases/Kundur/system.json      # Kundur
python3 tools/graph_laplacian.py --case cases/Kundur/system.json --plot
python3 tools/graph_laplacian.py --case cases/Kundur/system.json --save-report
python3 tools/graph_laplacian.py --quiet
```

#### Example output (Kundur 4-machine)

```
  Buses   : 10    Branches: 15

  Algebraic connectivity  ╬╗Ôéé = 4.058683   Ô£ô  PASS
  Spectral gap           ╬ö╬╗  = 22.571713

  ÔöÇÔöÇ Fiedler partition (weakest cut) ÔöÇÔöÇ
  Side A (+): Gen1_20kV, Gen2_20kV, Bus101_230kV, Bus102_230kV, Bus3_230kV
  Side B (ÔêÆ): Gen3_20kV, Gen4_20kV, Bus13_230kV, Bus112_230kV, Bus111_230kV
```

Side A = Area 1 generators, Side B = Area 2 generators ÔÇö the Fiedler vector
aligns exactly with the 2-area partition known from physical analysis.

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--plot` | off | Save eigenvalue spectrum + Fiedler vector figure to `outputs/<CaseName>/tools/laplacian.png` |
| `--save-report` | off | Save report to `outputs/<CaseName>/tools/laplacian_report.txt` |
| `--quiet` | off | One-line summary (CI mode) |

**Exit codes**: 0 = connected (╬╗Ôéé > threshold), 1 = disconnected / near-islanded.

---

### A-7 ÔÇö Swing Equation PH Formulator (`swing_ph.py`)

#### What it does

Extracts the **swing-equation sub-block** from the full numerical Jacobian and
expresses it in Port-Hamiltonian form:

$$A_\text{sw} = \begin{pmatrix} 0 & \omega_b I \\ -M^{-1}K_s & -M^{-1}D \end{pmatrix} = J - R$$

where $M = \text{diag}(2H_i)$, $D = \text{diag}(D_i)$, $K_s$ is the
synchronising power matrix, and $J, R = (A \mp A^\top)/2$.

Three structural constraints are verified:

| Check | Condition | Meaning |
|---|---|---|
| $\|A[\delta,\delta]\|$ | $\approx 0$ | No angleÔÇôangle coupling |
| $\|A[\delta,\omega] - \omega_b(I - H\mathbf{1}^\top/\Sigma H)\|$ | $\approx 0$ | COI-frame angleÔÇôspeed coupling |
| $K_s$ 2nd eigenvalue | $> 0$ | Synchronising matrix PSD on COI-reduced subspace |
| All $D_i$ | $\geq 0$ | Non-negative per-generator damping |

> **COI frame note.** The framework normalises generator angles to the Centre
> of Inertia, so $A[\delta,\omega] = \omega_b(I - H\mathbf{1}^\top / \Sigma H)$
> (not $\omega_b I$). One near-zero eigenvalue of $K_s$ is expected and accepted.

#### How to run

```bash
python3 tools/swing_ph.py                                      # IEEE14Bus
python3 tools/swing_ph.py --case cases/Kundur/system.json      # Kundur
python3 tools/swing_ph.py --case cases/Kundur/system.json --plot
python3 tools/swing_ph.py --case cases/Kundur/system.json --save-report
python3 tools/swing_ph.py --quiet
```

#### Example output (Kundur 4-machine)

```
  ÔÇû A[╬┤,╬┤] ÔÇû = 0.000e+00   Ô£ô
  ÔÇû A[╬┤,¤ë] ÔêÆ ¤ëb┬À(I ÔêÆ H┬À1ßÁÇ/╬úH) ÔÇû = 2.623e-07   (COI frame)  Ô£ô
  ÔÇû diag(A[¤ë,¤ë]) ÔêÆ (ÔêÆD/2H) ÔÇû = 3.426e-11   Ô£ô
  ╬úH=228.150 s

  Ks eigenvalues: -0.0472, 5.0910, 18.5461, 19.1700
  Ks 2nd-min eigval = 5.091020   PSD on sync. subspace Ô£ô
  All D_i ÔëÑ 0?  Ô£ô YES

  Ô£ô  PH CONDITIONS SATISFIED  [PASS]
```

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--eps` | `1e-6` | Jacobian finite-difference step |
| `--plot` | off | Save 2├ù2 figure to `outputs/<CaseName>/tools/swing_ph.png` |
| `--save-report` | off | Save report to `outputs/<CaseName>/tools/swing_ph_report.txt` |
| `--quiet` | off | One-line summary (CI mode) |

**Exit codes**: 0 = PH conditions satisfied, 1 = PH violation.

---

### A-8 ÔÇö Kirchhoff-Dirac Validator (`validate_kirchhoff_dirac.py`)

#### What it does

Verifies that the power network satisfies the **Kirchhoff-Dirac structure** of
Port-Hamiltonian theory at the power-flow equilibrium:

$$\mathcal{D} = \{ (f_N, e_N) \mid B f_N = 0 \text{ (KCL)},\; e_N = B^\top \lambda \text{ (KVL)} \}$$

where $B$ is the $(n_\text{bus} \times n_\text{branch})$ signed incidence matrix,
$f_N$ are branch currents, $e_N$ are branch voltages, and $\lambda$ are node
voltages.

Two checks are performed:

| Check | Criterion | Result |
|---|---|---|
| **KCL** at non-generator buses | $\|(Y_\text{net} \cdot V_\text{pf})[i] + I_\text{load}[i]\| < \varepsilon$ | 0 = KCL satisfied (PF accuracy) |
| **KVL** via cycle basis | $\|\sum_{k \in \text{cycle}} \pm v_\text{branch}[k]\| < \varepsilon$ | 0 = path-independent voltages |
| **Structural** $B K B^\top \approx Y$ | Off-diagonal mismatch | Detects tap-transformer corrections |

$I_\text{load}[i] = \overline{(P_i + jQ_i)/V_i}$ from PQ data.  
At generator buses $\|(Y_\text{net} V)[i] + I_\text{load}[i]\| = \|I_\text{gen}[i]\|$ (informational).  
Parallel-branch cycles are enumerated separately from topological cycles.

#### How to run

```bash
python3 tools/validate_kirchhoff_dirac.py                                   # IEEE14Bus
python3 tools/validate_kirchhoff_dirac.py --case cases/Kundur/system.json   # Kundur
python3 tools/validate_kirchhoff_dirac.py --case cases/Kundur/system.json --plot
python3 tools/validate_kirchhoff_dirac.py --case cases/Kundur/system.json --save-report
python3 tools/validate_kirchhoff_dirac.py --quiet
```

#### Example output (Kundur 4-machine)

```
  n_bus=10  n_branch=15  n_gen_buses=4  n_load_buses=6
  n_cycles=6  (topo=0, parallel=6)

  KCL max residual (non-gen buses): 7.732e-11 pu  Ô£ô
  KVL max cycle residual          : 0.000e+00 pu  Ô£ô
  B┬Àdiag(y)┬ÀBßÁÇ off-diag err      : 0.000e+00     Ô£ô

  Ô£ô  KIRCHHOFF-DIRAC STRUCTURE VERIFIED  [PASS]
```

and IEEE14Bus (7 topological cycles, tap-transformer B-error is informational):

```
  n_bus=14  n_branch=20  n_gen_buses=5  n_load_buses=9
  n_cycles=7  (topo=7, parallel=0)

  KCL max residual (non-gen buses): 6.469e-08 pu  Ô£ô
  KVL max cycle residual          : 0.000e+00 pu  Ô£ô
  B┬Àdiag(y)┬ÀBßÁÇ off-diag err      : 1.840e-02     (tap transformers ÔÇö informational)

  Ô£ô  KIRCHHOFF-DIRAC STRUCTURE VERIFIED  [PASS]
```

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--eps` | `1e-4` | KCL/KVL violation threshold (pu) |
| `--plot` | off | Save 2├ù2 figure to `outputs/<CaseName>/tools/kirchhoff_dirac.png` |
| `--save-report` | off | Save report to `outputs/<CaseName>/tools/kirchhoff_dirac_report.txt` |
| `--quiet` | off | One-line summary (CI mode) |

**Exit codes**: 0 = KCL and KVL satisfied, 1 = Dirac structure violated.

---

### A-9 ÔÇö Critical Clearing Time Estimator (`cct_estimator.py`)

> *In progress.*

Binary-searches the fault clearing time **t_cl** using repeated simulation runs
to bracket the Critical Clearing Time (CCT) within a user-specified tolerance.

---

### A-10 ÔÇö Component I/O Inspector (`inspect_equilibrium.py`)

#### What it does

Provides deep visibility into the state of every component *at the operating
point* produced by the initialisation pipeline ÔÇö answering questions that
`check_equilibrium` (A-1) does not expose.

Five interlocking inspection modes:

| Mode | What is shown |
|---|---|
| **Params table** | Every parameter inlined as `const double` in the C++ kernel, for every component. Immediately reveals wrong `Efd_0` or `Vref` values before any simulation. |
| **Refinement trace** | Key state values captured after each major refinement pass (power-flow, Eq_p loop, pre/post-lightweight). Flags states that change unexpectedly between passes (e.g. a pure-integral exciter whose `xi` is overwritten by the voltage-transducer update). |
| **I/O at equilibrium** | Compiles and runs a one-shot C++ inspector that calls `system_step(xÔéÇ, ÔÇª)` once, then prints every component's input vector and output vector. Catches signal mis-wiring (e.g. `Vterm=0.577` instead of `1.03`) without needing a full simulation. |
| **Exciter consistency** | For every exciter verifies: `Efd_0` (baked param) == `compute_efd_output(xÔéÇ)`; `Vref` Ôëê `Vterm` at xÔéÇ (zero steady-state error); `dxi/dt Ôëê 0` (integrator quiescent). |
| **Wiring map snapshot** | Lists every `(comp, port) ÔåÆ expr` entry in `compiler.wiring_map` after `generate_cpp()` has run (i.e. after `_refresh_control_params` has resolved `Vref`/`Pref` placeholders). |

Like A-1 the tool is **self-contained**: it drives the full init pipeline
internally and compiles the C++ inspector in memory ÔÇö no prior simulation needed.

#### Port-Hamiltonian context

At the PH equilibrium $x^*$ the system satisfies:

$$( J(x^*) - R(x^*) )\,\nabla H(x^*) + B\,\bar{u} = 0$$

Every component input at $x^*$ must equal its expected design value
$(u^* = \bar{u})$.  Any deviation signals that the interconnection structure
(Dirac structure, VSJ14 ┬º3) is not wired consistently, breaking the PH balance.

#### How to run

```bash
# Default: IEEE14Bus
python3 tools/inspect_equilibrium.py

# Specific case
python3 tools/inspect_equilibrium.py --case cases/SMIB_IdaPBC/system.json

# Inspect a single component
python3 tools/inspect_equilibrium.py --case cases/SMIB_IdaPBC/system.json --component IDAPBC_1

# Save full report
python3 tools/inspect_equilibrium.py --case cases/Kundur/system.json --save-report

# CI one-liner (exit 0 = all consistent, 1 = inconsistency found)
python3 tools/inspect_equilibrium.py --quiet
```

#### Example output ÔÇö SMIB with IDA-PBC exciter

```
=================================================================
  PH Component I/O Inspector ÔÇö A-10
=================================================================
  Case      : cases/SMIB_IdaPBC/system.json
  Component : (all)

  Running initialisation pipeline ...
  Compiling C++ inspector ... OK

ÔöÇÔöÇ IDAPBC_1 ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
  Params
    Kv=10.0  Kd=2.0  Ki=0.5  Efd_0=2.4732  Vref=1.02  Efd_max=6.0

  Refinement trace
    post-powerflow   : xi=0.000000
    post-Eq_p        : xi=0.000000
    pre-lightweight  : xi=0.000000
    post-lightweight : xi=0.000000   Ô£ô  (no spurious overwrite)

  I/O at xÔéÇ
    inputs  [0] Vterm = 1.020000   Ô£ô
    outputs [0] Efd   = 2.473200   Ô£ô

  Exciter consistency
    Efd_0 (param)          = 2.473200
    compute_efd_output(xÔéÇ) = 2.473200   Ô£ô
    Vref - Vterm           = 0.000000   Ô£ô  (zero steady-state error)
    dxi/dt at xÔéÇ           = 6.12e-06   Ô£ô  (< tol 1e-4)

  Ô£ô  IDAPBC_1 CONSISTENT
=================================================================
  Max |ß║ïßÁó|  : 6.12e-06
  Ô£ô  ALL COMPONENTS CONSISTENT
=================================================================
```

#### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--case` | `cases/IEEE14Bus/system.json` | Path to system JSON |
| `--component` | all | Restrict output to one named component (e.g. `IDAPBC_1`, `ESST3A_1`) |
| `--save-report` | off | Save full report to `outputs/<CaseName>/tools/inspect_equilibrium_report.txt` |
| `--quiet` | off | Print only the final PASS/FAIL line (CI mode) |

**Exit codes**: 0 = all components consistent, 1 = at least one inconsistency detected.

#### Use in control design and model validation

| Workflow | How this tool helps |
|---|---|
| **New exciter/controller** | Verify `Efd_0` was correctly computed and `Vref` is consistent before any simulation |
| **IDA-PBC / custom controllers** | Check the integral state `xi` is not corrupted by the lightweight refinement pass |
| **Wiring errors** | Detect mis-mapped ports (e.g. PSS output wired to wrong exciter input) before time-domain simulation |
| **Parameter sensitivity** | Print params table across cases to quickly compare tuning settings |

---

## Dependencies

All tools require only the packages already present in the framework environment:

```
numpy      >= 1.21
scipy      >= 1.7
pandas     >= 1.3        # A-3, A-4
matplotlib >= 3.4        # A-5 plots
networkx   >= 2.6        # A-6
g++        >= 9          # A-1, A-9, A-10  (system package, used via subprocess)
```

The `src/` packages (`SystemCompiler`, `Initializer`, etc.) are part of this
repository and require no separate installation.

---

## Adding a New Tool

1. Place the script in `tools/`.
2. Import `SystemCompiler` from `src.compiler` and `Initializer` from
   `src.initialization`; call `_initialize()` from `check_equilibrium.py`
   (or inline the same pipeline) to get `x0`, `Vd`, `Vq`.
3. Import `ensure_outdir` from `tools._utils` for auto-derived output paths:
   ```python
   from tools._utils import ensure_outdir
   outdir = ensure_outdir(case_path)   # ÔåÆ 'outputs/<CaseName>/toolbox'
   ```
4. Use `compiler.generate_cpp()` to get the kernel and build whatever
   analysis you need on top of it.
5. Follow the exit-code convention: **0 = pass / no anomaly, 1 = issue found**.
6. Add a section to this README following the template above.
