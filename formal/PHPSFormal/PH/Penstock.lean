import PHPSFormal.PH.Balance
import Mathlib.Algebra.Order.Star.Real

/-!
# P12: the elastic penstock of the hydro unit (Proposition 1, hydro part)

Source: PHPS paper draft, Table I (hydro unit); code: `src/phsim/hydro.jl` (header and
`_penstock_rhs!`), `src/phsim/structure.jl` (hydro units). Claim record: `formal/claims/P10-P12.md`.

Sections `n` (in use `Fin N`). Flows `q`, heads relative to the reservoir `e = h - 1` (the datum: the
reservoir's head is `1`, so `e₀ = 0`). Physical rows (code `_penstock_rhs!`):

    M q_k' = h_{k-1} - h_k - r q_k |q_k|          i.e.  M q' = -Aᵀ e - r ⊙ q ⊙ |q|
    C_k h_k' = q_k - q_{k+1}                       i.e.  C ⊙ e' = A q   (+ the turbine's outflow)

with `A` the ladder's incidence (`A_kk = 1`, `A_k,k+1 = -1`); the theorems hold for **any** `A`.
Storage `H = K [Σ M q_k²/2 + Σ C_k e_k²/2]` (`K = units · GMAX`), efforts `z_q = K M q`,
`z_e = K C ⊙ e`.

* `ladder_rows`: the rows are `(J - R)(z_q, z_e)` with `J = [[0, -(1/KM) Aᵀ D], [(1/KM) D A, 0]]`,
  `D = diag(1/C)`, and `R = diag(r |q| / (K M²))` on the flow rows.
* `ladder_J_skew`: `J` is skew for every incidence `A`.
* `ladder_R_psd`: `r ≥ 0`, `K M² > 0 ⇒ R ⪰ 0` (friction is a loss at every flow).

The turbine (the outflow `q_t` at the last node and the power `K q_t h_N / ω` on the rotor) is the
coupling pattern of `Machine.lean` (`machine_J_skew`): an entry `-q_t/(C_N ω)` in the column of the
rotor's effort and its negative transpose, with the reservoir's head a port on the rotor row; it needs
`ω > 0` (Proposition 1's hypothesis).
-/

namespace PHPSFormal.PH

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The ladder's `J` on `(z_q, z_e)`. -/
noncomputable def ladderJ (A : Matrix n n ℝ) (C : n → ℝ) (K M : ℝ) : Matrix (n ⊕ n) (n ⊕ n) ℝ :=
  fromBlocks 0 (-(1 / (K * M)) • (Aᵀ * diagonal (fun k => 1 / C k)))
    ((1 / (K * M)) • (diagonal (fun k => 1 / C k) * A)) 0

/-- Friction on the flow rows. -/
noncomputable def ladderR (r q : n → ℝ) (K M : ℝ) : Matrix (n ⊕ n) (n ⊕ n) ℝ :=
  fromBlocks (diagonal fun k => r k * |q k| / (K * M ^ 2)) 0 0 0

theorem ladder_J_skew (A : Matrix n n ℝ) (C : n → ℝ) (K M : ℝ) :
    (ladderJ A C K M)ᵀ = -ladderJ A C K M := by
  rw [ladderJ, fromBlocks_transpose, fromBlocks_neg, transpose_zero, neg_zero, transpose_smul,
    transpose_smul, transpose_mul, transpose_mul, transpose_transpose, diagonal_transpose]
  simp [neg_smul]

theorem ladder_R_psd (r q : n → ℝ) (K M : ℝ) (hr : ∀ k, 0 ≤ r k) (hKM : 0 < K * M ^ 2) :
    (ladderR r q K M).PosSemidef := by
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · exact IsHermitian.fromBlocks (isHermitian_diagonal_of_self_adjoint _ (by
      ext k; simp)) (by simp) (by simp)
  · intro y
    rw [← Sum.elim_comp_inl_inr y, ladderR, fromBlocks_mulVec, star_trivial]
    simp only [Sum.elim_comp_inl, Sum.elim_comp_inr, zero_mulVec, add_zero,
      sumElim_dotProduct_sumElim, dotProduct_zero]
    simp only [mulVec_diagonal, dotProduct]
    apply Finset.sum_nonneg
    intro k _
    have : 0 ≤ r k * |q k| / (K * M ^ 2) := div_nonneg (mul_nonneg (hr k) (abs_nonneg _)) hKM.le
    rw [Function.comp_apply]
    nlinarith [mul_nonneg this (mul_self_nonneg (y (Sum.inl k)))]

/-- **The penstock's rows are of the form (1).** -/
theorem ladder_rows (A : Matrix n n ℝ) (C r q e : n → ℝ) (K M : ℝ) (hC : ∀ k, C k ≠ 0)
    (hK : K ≠ 0) (hM : M ≠ 0) :
    (ladderJ A C K M - ladderR r q K M) *ᵥ
        Sum.elim ((K * M) • q) (fun k => K * C k * e k) =
      Sum.elim (fun k => (-(Aᵀ *ᵥ e) k - r k * q k * |q k|) / M) (fun k => (A *ᵥ q) k / C k) := by
  have he : (diagonal fun k => 1 / C k) *ᵥ (fun k => K * C k * e k) = K • e := by
    ext j
    rw [mulVec_diagonal, Pi.smul_apply, smul_eq_mul]
    field_simp [hC j]
  ext k
  rcases k with k | k
  · simp only [ladderJ, ladderR, sub_mulVec, fromBlocks_mulVec, Sum.elim_comp_inl,
      Sum.elim_comp_inr, zero_mulVec, zero_add, add_zero, Sum.elim_inl, Pi.sub_apply,
      smul_mulVec, ← mulVec_mulVec, Pi.smul_apply, smul_eq_mul]
    rw [he, mulVec_smul, mulVec_diagonal]
    simp only [Pi.smul_apply, smul_eq_mul]
    field_simp
  · simp only [ladderJ, ladderR, sub_mulVec, fromBlocks_mulVec, Sum.elim_comp_inl,
      Sum.elim_comp_inr, zero_mulVec, add_zero, zero_add, Sum.elim_inr, Pi.sub_apply, sub_zero]
    rw [smul_mulVec, mulVec_smul, ← mulVec_mulVec, smul_smul, Pi.zero_apply, sub_zero,
      Pi.smul_apply, mulVec_diagonal, smul_eq_mul]
    field_simp

end PHPSFormal.PH

#print axioms PHPSFormal.PH.ladder_J_skew
#print axioms PHPSFormal.PH.ladder_R_psd
#print axioms PHPSFormal.PH.ladder_rows
