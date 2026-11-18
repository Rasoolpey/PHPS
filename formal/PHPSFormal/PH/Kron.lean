import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.Basic.Real.Basic

/-!
# P3b: eliminating the algebraic node voltages keeps the form dissipative

Source: PHPS paper draft, §II-B, eq. (5) (the descriptor system) and the assembled form; code:
`src/phsim/structure.jl`, `_structure_core`, "the reduction of the algebraic voltages":
`Mr = M_SS - M_SW D⁻¹ M_WS`, `J_r = (Mr - Mrᵀ)/2`, `R_r = -(Mr + Mrᵀ)/2`, with `M = J - R` over the
states `S` and the algebraic node voltages `W` (nodes without capacitance), `D = M_WW`.
Claim record: `formal/claims/P1-P9.md` (P3b).

* `kron_rows`: with `w = -D⁻¹ M_WS x`, the full rows are `M (x, w) = (Mr x, 0)`: the algebraic
  rows `0 = M_WS x + D w` hold and the state rows are `Mr x` (eq. (5) with `λ = w`).
* `kron_power`: the power of the reduced form equals the power of the full form on `(x, w)`:
  `xᵀ Mr x = (x, w)ᵀ M (x, w)`, so the reduced loss `xᵀ R_r x` is the full loss including the
  conductances at the eliminated nodes.
* `kron_dissipative`: if the full form is dissipative (`yᵀ M y ≤ 0` for all `y`, i.e. `R ⪰ 0`,
  with `J` skew) and `D` is invertible, the reduced form is dissipative: `xᵀ Mr x ≤ 0` for all
  `x`, i.e. `R_r ⪰ 0`. `J_r` is skew by construction.
-/

namespace PHPSFormal.PH

open Matrix

variable {s w : Type*} [Fintype s] [Fintype w] [DecidableEq w]

/-- The eliminated voltages. -/
noncomputable def kronW (C : Matrix w s ℝ) (D : Matrix w w ℝ) (x : s → ℝ) : w → ℝ :=
  -(D⁻¹ *ᵥ (C *ᵥ x))

theorem kron_rows (A : Matrix s s ℝ) (B : Matrix s w ℝ) (C : Matrix w s ℝ) (D : Matrix w w ℝ)
    (hD : IsUnit D.det) (x : s → ℝ) :
    fromBlocks A B C D *ᵥ Sum.elim x (kronW C D x) =
      Sum.elim ((A - B * D⁻¹ * C) *ᵥ x) 0 := by
  rw [fromBlocks_mulVec]
  simp only [Sum.elim_comp_inl, Sum.elim_comp_inr, kronW]
  congr 1
  · rw [mulVec_neg, sub_mulVec, ← mulVec_mulVec, ← mulVec_mulVec, sub_eq_add_neg]
  · rw [mulVec_neg, mulVec_mulVec, mul_nonsing_inv _ hD, one_mulVec, add_neg_cancel]

theorem kron_power (A : Matrix s s ℝ) (B : Matrix s w ℝ) (C : Matrix w s ℝ) (D : Matrix w w ℝ)
    (hD : IsUnit D.det) (x : s → ℝ) :
    x ⬝ᵥ ((A - B * D⁻¹ * C) *ᵥ x) =
      Sum.elim x (kronW C D x) ⬝ᵥ (fromBlocks A B C D *ᵥ Sum.elim x (kronW C D x)) := by
  rw [kron_rows A B C D hD x, sumElim_dotProduct_sumElim, dotProduct_zero, add_zero]

/-- **The elimination keeps the form dissipative.** -/
theorem kron_dissipative (A : Matrix s s ℝ) (B : Matrix s w ℝ) (C : Matrix w s ℝ)
    (D : Matrix w w ℝ) (hD : IsUnit D.det)
    (hM : ∀ y : s ⊕ w → ℝ, y ⬝ᵥ (fromBlocks A B C D *ᵥ y) ≤ 0) (x : s → ℝ) :
    x ⬝ᵥ ((A - B * D⁻¹ * C) *ᵥ x) ≤ 0 := by
  rw [kron_power A B C D hD x]
  exact hM _

end PHPSFormal.PH

#print axioms PHPSFormal.PH.kron_rows
#print axioms PHPSFormal.PH.kron_power
#print axioms PHPSFormal.PH.kron_dissipative
