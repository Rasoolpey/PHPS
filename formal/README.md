# PHPSFormal: machine-checked proofs of the simulator's mathematics

Lean 4 (`v4.34.1`) and mathlib (`v4.34.1`) proofs of the port-Hamiltonian claims the simulator rests on. Every theorem
builds with no `sorry` and only Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

## Build

```sh
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
source ~/.elan/env
cd formal
lake exe cache get     # mathlib's compiled files (several GB)
lake build             # prints, for every theorem, the axioms it depends on
```

## What is proved (`PHPSFormal/PH/`)

| File | Content |
|---|---|
| `Balance.lean` | the device form and its power balance; the implicit midpoint and backward Euler identities; discrete gradients, also mixed across coupled devices |
| `Dirac.lean` | the network's Kirchhoff relations form a Dirac structure (any complex incidence, over ℝ and ℂ) |
| `Subsystem.lean` | a group of states as a system of its own; power-conserving ports; composition of devices |
| `Kron.lean` | eliminating the node voltages without a state keeps the form dissipative |
| `CutSet.lean` | the reduction of an inductor-only cut set: energy, rates, losses and port power |
| `Park.lean` | the `dq0` transform; the fault conductance `G_f(θ) ⪰ 0`; phase power = 3/2 × axis power |
| `Network.lean` | network elements, loads and their couplings in the form |
| `Machine.lean` | the synchronous machine for any winding set (the DC shunt exciter as an instance) |
| `Penstock.lean` | the elastic penstock |
| `Brushless.lean` | the brushless exciter's energy balance and non-negative storage |
| `Steam.lean` | the steam unit's elements and their assembly, under stated properties of steam |
| `Storage.lean` | storage gradients, the saturable field, the turbine boundary, the series merge |

`claims/` has one record per group: the statement in the paper, the hypotheses added in Lean, the reviews, and what is
not proved (in particular: that the code computes these equations is checked numerically, not here; the properties of
steam are hypotheses).
