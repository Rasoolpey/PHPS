import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Algebra.Order.Star.Real

/-!
# P5: the fault conductance in the `dq0` axes

Source: PHPS paper draft, §III, eq. (gf) and the sentences after it. Claim record:
`formal/claims/P5.md`.

`Pk θ` maps `dq0` quantities to the three phase quantities:
`x_k = x_d cos θ_k - x_q sin θ_k + x_0 / √2`, `θ_k = θ - 2πk/3`, `k = 0, 1, 2` (the paper's
formula). `T θ = (2/3) (Pk θ)ᵀ` maps phase quantities to the `dq0` axes.

* `Pk_orth`: `Pkᵀ Pk = (3/2) I`, so `T Tᵀ = (2/3) I` (`T_mul_transpose`) and `T Pk = I`.
* `fault_current`: a fault with phase conductance `G_abc` draws, in the `dq0` axes, the current
  `G_f v` with `G_f = (3/2) T G_abc Tᵀ` (eq. (gf)).
* `Gf_psd`: `G_abc ⪰ 0 ⇒ G_f ⪰ 0` at every frame angle.
* `phase_power`: the phase power is `3/2` times the axis power,
  `v_abcᵀ i_abc = (3/2) (v_d i_d + v_q i_q + v_0 i_0)`: the axis power of the paper is the phase
  power divided by `3/2`, a per-unit normalization the paper must state.
-/

namespace PHPSFormal.PH

open Matrix Real

/-- The phase angles `θ_k = θ - 2πk/3`, written with `cos`, `sin` of `θ`. -/
theorem cos_sub_two_pi_div_three (θ : ℝ) :
    cos (θ - 2 * π / 3) = -(1 / 2) * cos θ + (√3 / 2) * sin θ := by
  have h1 : 2 * π / 3 = π - π / 3 := by ring
  rw [h1, cos_sub, cos_pi_sub, sin_pi_sub, cos_pi_div_three, sin_pi_div_three]
  ring

theorem sin_sub_two_pi_div_three (θ : ℝ) :
    sin (θ - 2 * π / 3) = -(1 / 2) * sin θ - (√3 / 2) * cos θ := by
  have h1 : 2 * π / 3 = π - π / 3 := by ring
  rw [h1, sin_sub, cos_pi_sub, sin_pi_sub, cos_pi_div_three, sin_pi_div_three]
  ring

theorem cos_sub_four_pi_div_three (θ : ℝ) :
    cos (θ - 4 * π / 3) = -(1 / 2) * cos θ - (√3 / 2) * sin θ := by
  have h1 : 4 * π / 3 = π / 3 + π := by ring
  rw [h1, cos_sub, cos_add_pi, sin_add_pi, cos_pi_div_three, sin_pi_div_three]
  ring

theorem sin_sub_four_pi_div_three (θ : ℝ) :
    sin (θ - 4 * π / 3) = -(1 / 2) * sin θ + (√3 / 2) * cos θ := by
  have h1 : 4 * π / 3 = π / 3 + π := by ring
  rw [h1, sin_sub, cos_add_pi, sin_add_pi, cos_pi_div_three, sin_pi_div_three]
  ring

/-- `dq0 → abc`, the paper's formula. Rows are the phases `k = 0, 1, 2`. -/
noncomputable def Pk (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![cos θ, -sin θ, 1 / √2;
     cos (θ - 2 * π / 3), -sin (θ - 2 * π / 3), 1 / √2;
     cos (θ - 4 * π / 3), -sin (θ - 4 * π / 3), 1 / √2]

/-- `abc → dq0`. -/
noncomputable def T (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := (2 / 3 : ℝ) • (Pk θ)ᵀ

theorem Pk_orth (θ : ℝ) : (Pk θ)ᵀ * Pk θ = (3 / 2 : ℝ) • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  have h3 : √3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have h2 : √2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hsc := sin_sq_add_cos_sq θ
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Pk, Matrix.mul_apply, Fin.sum_univ_three, cos_sub_two_pi_div_three,
      sin_sub_two_pi_div_three, cos_sub_four_pi_div_three, sin_sub_four_pi_div_three] <;>
    ring_nf <;> simp only [h3, h2, inv_pow] <;> ring_nf <;> linarith [hsc]

theorem T_mul_Pk (θ : ℝ) : T θ * Pk θ = 1 := by
  rw [T, Matrix.smul_mul, Pk_orth, smul_smul]
  norm_num

theorem T_mul_transpose (θ : ℝ) : T θ * (T θ)ᵀ = (2 / 3 : ℝ) • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  rw [T, transpose_smul, transpose_transpose, Matrix.smul_mul, Matrix.mul_smul, Pk_orth,
    smul_smul, smul_smul]
  norm_num

/-- `Pk = (3/2) Tᵀ`. -/
theorem Pk_eq (θ : ℝ) : Pk θ = (3 / 2 : ℝ) • (T θ)ᵀ := by
  rw [T, transpose_smul, transpose_transpose, smul_smul]
  norm_num

/-- **eq. (gf).** The fault current in the `dq0` axes is `G_f v` with `G_f = (3/2) T G_abc Tᵀ`. -/
theorem fault_current (θ : ℝ) (Gabc : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) :
    T θ *ᵥ (Gabc *ᵥ (Pk θ *ᵥ v)) = ((3 / 2 : ℝ) • (T θ * Gabc * (T θ)ᵀ)) *ᵥ v := by
  rw [Pk_eq, mulVec_mulVec, mulVec_mulVec, Matrix.mul_smul]

/-- **P5.** `G_abc ⪰ 0 ⇒ G_f ⪰ 0`, at every frame angle. -/
theorem Gf_psd (θ : ℝ) (Gabc : Matrix (Fin 3) (Fin 3) ℝ) (hG : Gabc.PosSemidef) :
    ((3 / 2 : ℝ) • (T θ * Gabc * (T θ)ᵀ)).PosSemidef := by
  have h := hG.mul_mul_conjTranspose_same (T θ)
  rw [conjTranspose_eq_transpose_of_trivial] at h
  exact h.smul (by norm_num)

/-- The phase power is `3/2` times the axis power. -/
theorem phase_power (θ : ℝ) (v i : Fin 3 → ℝ) :
    (Pk θ *ᵥ v) ⬝ᵥ (Pk θ *ᵥ i) = (3 / 2 : ℝ) * (v ⬝ᵥ i) := by
  rw [dotProduct_mulVec, ← mulVec_transpose, mulVec_mulVec, Pk_orth, smul_mulVec, one_mulVec,
    smul_dotProduct, smul_eq_mul]

end PHPSFormal.PH

#print axioms PHPSFormal.PH.Pk_orth
#print axioms PHPSFormal.PH.T_mul_transpose
#print axioms PHPSFormal.PH.fault_current
#print axioms PHPSFormal.PH.Gf_psd
#print axioms PHPSFormal.PH.phase_power
