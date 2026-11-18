# The PHPS paper's mathematical claims, and their formal proofs in Lean (plan, 2026-11-18)

**Status 2026-11-18 (end of day):** all of P1-P14 formalised, no `sorry`, standard axioms only (records
[P1-P9.md](P1-P9.md), [P10-P12.md](P10-P12.md), [P13-P14.md](P13-P14.md); reviews 123-127). The steam unit rests on
stated properties of steam. Not machine-checked: that IAPWS-95 satisfies them, the row-by-row identity of the code with
these equations (numerical structure check), a discrete exergy balance, the trajectory equivalence of the reductions.
The results are in the paper's Appendix A and B (`docs/papers/resubmission/paper/appendix_proofs.tex`) and in the PHPS
repository's `formal/` (namespace `PHPSFormal`).

Source: `docs/papers/resubmission/paper/main.tex` (draft of 2026-11-18). Motivation: review 07
(`docs/papers/resubmission/07_rejection_check_merge.md`): both reviewers say the draft repeats the old paper's fault,
propositions without the mathematics that proves them. Goal: every mathematical statement of the paper is either
machine-checked here (Lean 4 + mathlib, no `sorry`, standard axioms only) or explicitly labelled as numerical
evidence.

What Lean can and cannot establish:
- **Can:** the algebra and analysis of the propositions, under hypotheses stated in Lean, for every dimension and
  every admissible parameter value.
- **Cannot:** that the Julia code computes these equations. That link stays the numerical structure check (rates
  against `(J-R)∇H + Gu - P v_f` at sampled states, and at every export). Nor can it check the steam tables' Gibbs
  defect or any simulated number.

Each claim gets a record `formal/claims/<ID>.md` in the format of `L1.md` (statement in the source, hypotheses,
Lean names, refuter pass, build output, what it does not prove).

## Claims, in order of work

| ID | Paper | Statement (informal) | Lean target | Effort |
|---|---|---|---|---|
| P1 | eq. (1)-(3), Cor. 1 | For `x' = (J-R)e + (G-P)u`, `y = (G+P)ᵀe + Su`, with `J` skew: `eᵀx' = yᵀu - [e;u]ᵀ[[R,P],[Pᵀ,S]][e;u]`. With the block PSD: `eᵀx' ≤ yᵀu` | pointwise identity on `Fin n` vectors (`e` stands for `∇H(x)`; with `HasGradientAt` the chain rule gives `dH/dt`) | S |
| P2 | §II-B | Kirchhoff-Dirac: `D = {(f,e) : v_b = Bᴴ v_n, i_n = B i_b}` with `f = (i_b, -i_n)`, `e = (v_b, v_n)` over ℂ: (a) `Re(eᴴf) = 0` on `D`; (b) `D = D^⊥` for the pairing `⟨(f₁,e₁),(f₂,e₂)⟩ = Re(e₁ᴴf₂ + e₂ᴴf₁)` (so `D` is a Dirac structure, maximal isotropic) | complex matrices, `conjTranspose`; (b) proved directly (`D^⊥ ⊆ D` by choosing test elements), no dimension theory needed | M |
| P3 | §II-B | Composition: devices each satisfying P1 with port pairs, joined by an interconnection on which the internal port powers sum to zero, give `Σ e_cᵀx'_c = (external supply) - (Σ losses)` | finite sum of P1 instances | S |
| P4 | §II-B | Cut-set reduction: `L ≻ 0`, `N` full column rank, `A N = 0`, `i = Nη`, `ψ = NᵀL i`: (a) `½ψᵀ(NᵀLN)⁻¹ψ = ½ iᵀL i`; (b) the node-voltage term drops out of `Nᵀφ'`, since `Nᵀ Aᵀ = 0`; (c) `NᵀR_bN ⪰ 0` | matrix algebra; invertibility of `NᵀLN` from `L ≻ 0` and injectivity of `N` | S-M |
| P5 | eq. (gf) | `T Tᵀ = (2/3) I` for the stated `T(θ)`; `G_abc ⪰ 0 ⇒ (3/2) T G_abc Tᵀ ⪰ 0` | trigonometric sums of three phases; PSD congruence | S-M |
| P6 | Prop. 3 | Midpoint step `x₁ - x₀ = h[(J-R)∇H(x_m) + (G-P)u]` with `H` quadratic-plus-affine (symmetric Hessian): `H(x₁) - H(x₀) = h(yᵀu - d)` at the midpoint, `d ≥ 0` the loss term. Summed over devices with linear Kirchhoff relations at the midpoint: the internal powers cancel | `H(x) = ½xᵀQx + cᵀx + k`, `Qᵀ = Q`; exact identity `H(x₁)-H(x₀) = ∇H(x_m)ᵀ(x₁-x₀)` | S |
| P7 | §IV | Backward Euler with quadratic `H`: `H(x₁)-H(x₀) = h(yᵀu - d)|₁ - ½Δxᵀ Q Δx` | as P6 (inventory I3) | S |
| P8 | §IV | Discrete gradient: (a) Gonzalez's `∇̄H = ∇H(x_m) + (ΔH - ∇H(x_m)ᵀΔx)/‖Δx‖² Δx` satisfies `∇̄Hᵀ Δx = ΔH` for `Δx ≠ 0`; (b) the step with `∇̄H` in place of `∇H` keeps the balance of P6 for any `H` | pure algebra; (b) as P6 | S |
| P9 | Prop. 4 | Subsystem: `J` skew, `R_{S S̄} = 0`, `H = H_S(x_S) + H_{S̄}(x_{S̄})`: the group's balance `e_Sᵀ x'_S = y_pᵀ e_N - e_Sᵀ R_{SS} e_S + ...`, and the power entering the group through its ports equals the power the rest gives up | block matrices over an index split | S |
| P10 | Prop. 1, network and loads | π-line, transformer branch, RL load, node capacitance: their equations are of form (1) with the stated `H`, `J`, `R ⪰ 0` | per element, explicit small matrices | S |
| P11 | Prop. 1, machine | the 6-winding machine with momentum and angle, rotor frame, port in the network frame: equations of form (1) with `H = (ω_b/2)λᵀKλ + p²/4H_m`, the rotation a state-dependent skew coupling, `R ⪰ 0` | needs the machine equations written as in the code (docs/presentation machine chapter); then explicit `J(x)`, `R` | M |
| P12 | Prop. 1, DC exciter, hydro unit | saturable shunt field with flux-modulated skew coupling; 28-section penstock ladder with friction | explicit matrices from the code's equations | M |
| P13 | Prop. 1, steam unit | each throttling, turbine section, heat element: rank-one dissipation `γ_e w_e w_eᵀ` with `γ_e ≥ 0`, from stated thermodynamic hypotheses (`∂s/∂p|_h < 0`, concavity of `s(h,p)` in `h`, convexity of `h(s,p)` in `s`, `0 ≤ η ≤ 1`) | real analysis with convexity hypotheses as axioms of the property model (stated as hypotheses, not proved from IAPWS-95) | L |
| P14 | Prop. 2 | brushless exciter: exact energy balance with storage `W ≥ 0` and loss `g V_FE²` | needs the equations written; then a balance identity | M |

Order: P1, P6, P7, P8, P9, P3 (the framework, a day), then P2, P4, P5 (the network), then P10-P12 (the devices
whose equations are short), P14, and P13 last.

## Rules

- A Lean statement is compared with the paper's statement in the claim record; every hypothesis added in Lean (as for
  L1's symmetry of `H_fix`) is reported, and the paper is corrected to state it.
- `lake build` clean; `#print axioms` shows only `propext`, `Classical.choice`, `Quot.sound`; a source scan finds no
  `sorry`, `admit`, `axiom`, `native_decide`.
- Codex reviews this plan (idea), the Lean statements against the paper (code), and the records (result).
