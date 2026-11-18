import PHPSFormal.PH.Balance
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Algebra.Order.Star.Real

/-!
# P11: the synchronous machine (Proposition 1, machine part)

Source: PHPS paper draft, Table I (synchronous machine) and Proposition 1; code:
`src/phsim/system.jl`, `phs_rhs!` (machine rows), `_winding_currents`, `_machine_currents`,
`_torque`; `src/phsim/structure.jl`, `_structure_core` (machines). Claim record:
`formal/claims/P10-P11.md`.

States: the winding flux linkages `λ : W → ℝ` (any finite winding set: the stator pair, the field,
the dampers; at a reduced node also the step-up branch pair, written in the rotor frame), then the
momentum `p` and the angle `δ` (index `Fin 2`: `0 = p`, `1 = δ`). Efforts: the winding currents
`i = ω_b K λ`, the speed `ω = p / 2H_m`, and `0` for the angle (it stores no energy).

The code's rows, in motor convention:

    λ' = v_w - R_w i + ω a,        a = ω_b S λ   (the speed voltages)
    p' = T_shaft - aᵀ i                           (= T_shaft + ω_b (λ_d i_q - λ_q i_d) per pair)
    δ' = ω_b (ω - 1)

with `v_w` the voltages applied to the windings (the terminal pair and the field), `R_w` the
winding resistances (diagonal, or the `2 × 2` block `R_m` of a reduced node), and `S` the
rotation by 90° on each `dq` pair (`(Sλ)_d = λ_q`, `(Sλ)_q = -λ_d`), zero on the rotor windings.

* `machine_rows`: the rows are `(J - R) z + G u` with
  `J = [[0, a e_p], [-(a e_p)ᵀ, J_m]]`, `J_m = [[0, -ω_b], [ω_b, 0]]`, `R = diag(R_w, 0)`.
* `machine_J_skew`: `J` is skew for **every** `a`: the electromechanical conversion conserves
  power, whatever the flux linkages, and with any number of `dq` pairs.
* `machine_R_psd`: `R_w ⪰ 0 ⇒ R ⪰ 0`.
* `pair_torque`: on a `dq` pair, `-aᵀ i = ω_b (λ_d i_q - λ_q i_d)`, the code's `_torque`.
* `machine_energy_nonneg`: `H = (ω_b/2) λᵀ K λ + p²/(4 H_m) ≥ 0` for `K ⪰ 0`, `ω_b ≥ 0`, `H_m > 0`.
-/

namespace PHPSFormal.PH

open Matrix

variable {W : Type*} [Fintype W] [DecidableEq W]

/-- The column of the momentum's effort in the winding rows: `a` in column `p`, `0` in column `δ`. -/
def speedCol (a : W → ℝ) : Matrix W (Fin 2) ℝ := Matrix.of fun i j => if j = 0 then a i else 0

/-- The mechanical block on `(p, δ)`. -/
def mechJ (wb : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![0, -wb; wb, 0]

/-- The machine's `J`. -/
def machJ (a : W → ℝ) (wb : ℝ) : Matrix (W ⊕ Fin 2) (W ⊕ Fin 2) ℝ :=
  fromBlocks 0 (speedCol a) (-(speedCol a)ᵀ) (mechJ wb)

/-- The machine's `R`. -/
def machR (Rw : Matrix W W ℝ) : Matrix (W ⊕ Fin 2) (W ⊕ Fin 2) ℝ := fromBlocks Rw 0 0 0

/-- The input matrix: the winding voltages enter the winding rows, the shaft torque the `p` row,
the frame's rotation (input `1`) the `δ` row with `-ω_b`. Inputs `W ⊕ Fin 2`: `(v_w, T, 1)`. -/
def machG (wb : ℝ) : Matrix (W ⊕ Fin 2) (W ⊕ Fin 2) ℝ :=
  fromBlocks 1 0 0 !![1, 0; 0, -wb]

theorem machine_J_skew (a : W → ℝ) (wb : ℝ) : (machJ a wb)ᵀ = -machJ a wb := by
  have hm : (mechJ wb)ᵀ = -mechJ wb := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [mechJ]
  rw [machJ, fromBlocks_transpose, fromBlocks_neg, transpose_zero, transpose_neg,
    transpose_transpose, hm, neg_zero, neg_neg]

theorem machine_R_psd (Rw : Matrix W W ℝ) (hR : Rw.PosSemidef) : (machR Rw).PosSemidef := by
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · exact IsHermitian.fromBlocks hR.isHermitian (by simp) (by simp)
  · intro y
    rw [← Sum.elim_comp_inl_inr y, machR, fromBlocks_mulVec, star_trivial]
    simp only [Sum.elim_comp_inl, Sum.elim_comp_inr, zero_mulVec, add_zero,
      sumElim_dotProduct_sumElim, dotProduct_zero]
    have h1 := hR.dotProduct_mulVec_nonneg (y ∘ Sum.inl)
    rwa [star_trivial] at h1

/-- **The machine's rows are of the form (1).** -/
theorem machine_rows (a : W → ℝ) (wb : ℝ) (Rw : Matrix W W ℝ) (i vw : W → ℝ) (ω T : ℝ) :
    (machJ a wb - machR Rw) *ᵥ Sum.elim i ![ω, 0] + machG wb *ᵥ Sum.elim vw ![T, 1] =
      Sum.elim (vw - Rw *ᵥ i + ω • a) ![T - a ⬝ᵥ i, wb * (ω - 1)] := by
  ext k
  rcases k with k | k
  · simp [machJ, machR, machG, speedCol, fromBlocks_mulVec, sub_mulVec, mulVec, dotProduct,
      Fin.sum_univ_two, Matrix.vecHead, Matrix.vecTail]
    ring
  · fin_cases k <;>
      simp [machJ, machR, machG, speedCol, mechJ, fromBlocks_mulVec, sub_mulVec, mulVec,
        dotProduct, Fin.sum_univ_two] <;> ring

/-- **The torque on a `dq` pair.** With `a = ω_b S λ` on the pair (`a_d = ω_b λ_q`,
`a_q = -ω_b λ_d`), `-(a_d i_d + a_q i_q) = ω_b (λ_d i_q - λ_q i_d)`. -/
theorem pair_torque (wb ld lq id iq : ℝ) :
    -((wb * lq) * id + (-(wb * ld)) * iq) = wb * (ld * iq - lq * id) := by ring

/-- **The storage is bounded below.** -/
theorem machine_energy_nonneg (K : Matrix W W ℝ) (hK : K.PosSemidef) (wb Hm : ℝ) (hwb : 0 ≤ wb)
    (hH : 0 < Hm) (l : W → ℝ) (p : ℝ) :
    0 ≤ wb / 2 * (l ⬝ᵥ (K *ᵥ l)) + p ^ 2 / (4 * Hm) := by
  have h := hK.dotProduct_mulVec_nonneg l
  rw [star_trivial] at h
  positivity

end PHPSFormal.PH

#print axioms PHPSFormal.PH.machine_J_skew
#print axioms PHPSFormal.PH.machine_R_psd
#print axioms PHPSFormal.PH.machine_rows
#print axioms PHPSFormal.PH.pair_torque
#print axioms PHPSFormal.PH.machine_energy_nonneg
