import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Basic.Real.Basic

/-!
# L1 and L2: the support lemma and its obstruction corollary

Source: `docs/records/early/NOGO_THEOREM.md`, section 2, Lemma 1 (support), file version `197ff2c`.
Claim record: `formal/claims/L1.md`.

Notation as in the source:
* `𝓕_R = {F = Fᵀ : F i j = 0 unless i ∈ R and j ∈ R}` is `Fᵀ = F ∧ SupportedOn R F`;
* `K_R(A) = {v : (A v)_R = 0}` is `kerOn A R`;
* the witness rate `ρ(v) = vᵀ H_fix A v` is `witnessRate H A v`.

Everything is over `ℝ` on a finite index type `n`. Nothing is assumed about `A`, and nothing
about `P` beyond symmetry (in particular `P` need not be positive definite).
-/

namespace PHPSFormal.NoGo

open Matrix

variable {n : Type*} [Fintype n]

/-- `F` vanishes outside `R × R`. With `Fᵀ = F` this is membership in `𝓕_R`. -/
def SupportedOn (R : Set n) (F : Matrix n n ℝ) : Prop :=
  ∀ i j, F i j ≠ 0 → i ∈ R ∧ j ∈ R

/-- `K_R(A) = {v : (A v)_j = 0 for every j ∈ R}`. -/
def kerOn (A : Matrix n n ℝ) (R : Set n) : Set (n → ℝ) :=
  {v | ∀ j ∈ R, (A *ᵥ v) j = 0}

/-- The witness rate `ρ(v) = vᵀ H A v`. -/
def witnessRate (H A : Matrix n n ℝ) (v : n → ℝ) : ℝ :=
  v ⬝ᵥ (H *ᵥ (A *ᵥ v))

/-- Column-support form (stronger than the source): if every nonzero entry of `F` lies in a
column indexed by `R` and `(A v)_R = 0`, then `vᵀ F A v = 0`. No symmetry is needed. -/
theorem support_lemma_col (A F : Matrix n n ℝ) (R : Set n)
    (hF : ∀ i j, F i j ≠ 0 → j ∈ R) (v : n → ℝ) (hv : v ∈ kerOn A R) :
    v ⬝ᵥ (F *ᵥ (A *ᵥ v)) = 0 := by
  have hv' : ∀ j ∈ R, (A *ᵥ v) j = 0 := hv
  generalize A *ᵥ v = w at hv' ⊢
  have h : F *ᵥ w = 0 := by
    ext i
    simp only [mulVec, dotProduct, Pi.zero_apply]
    apply Finset.sum_eq_zero
    intro j _
    by_cases hj : j ∈ R
    · simp [hv' j hj]
    · have hFij : F i j = 0 := by
        by_contra hne
        exact hj (hF i j hne)
      simp [hFij]
  simp [h]

/-- For symmetric `P`, `vᵀ(AᵀP + PA)v = 2 vᵀ P A v`. -/
theorem rate_form_of_symm (A P : Matrix n n ℝ) (hP : Pᵀ = P) (v : n → ℝ) :
    v ⬝ᵥ ((Aᵀ * P + P * A) *ᵥ v) = 2 * (v ⬝ᵥ (P *ᵥ (A *ᵥ v))) := by
  have key : ∀ w, v ⬝ᵥ (P *ᵥ w) = w ⬝ᵥ (P *ᵥ v) := by
    intro w
    conv_lhs => rw [dotProduct_mulVec, ← hP, vecMul_transpose]
    rw [dotProduct_comm]
  rw [add_mulVec, dotProduct_add, ← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec v Aᵀ,
    vecMul_transpose, ← key (A *ᵥ v)]
  ring

/-- **L1, Lemma 1 (support)** as stated in the source: for `F ∈ 𝓕_R` and `v ∈ K_R(A)`,
`vᵀ(AᵀF + FA)v = 0`. -/
theorem support_lemma (A F : Matrix n n ℝ) (R : Set n) (hFsymm : Fᵀ = F)
    (hF : SupportedOn R F) (v : n → ℝ) (hv : v ∈ kerOn A R) :
    v ⬝ᵥ ((Aᵀ * F + F * A) *ᵥ v) = 0 := by
  rw [rate_form_of_symm A F hFsymm,
    support_lemma_col A F R (fun i j h => (hF i j h).2) v hv, mul_zero]

/-- **L2, first consequence.** With `H` symmetric, `F ∈ 𝓕_R`, `v ∈ K_R(A)` and
`P = H + F`: `vᵀ(AᵀP + PA)v = 2ρ(v)`. The symmetry of `H` is implicit in the source
(`H_fix` is a block-diagonal Hessian). -/
theorem rate_eq_two_witnessRate (A H F : Matrix n n ℝ) (R : Set n) (hH : Hᵀ = H)
    (hFsymm : Fᵀ = F) (hF : SupportedOn R F) (v : n → ℝ) (hv : v ∈ kerOn A R) :
    v ⬝ᵥ ((Aᵀ * (H + F) + (H + F) * A) *ᵥ v) = 2 * witnessRate H A v := by
  have hP : (H + F)ᵀ = H + F := by rw [transpose_add, hH, hFsymm]
  rw [rate_form_of_symm A (H + F) hP, add_mulVec, dotProduct_add,
    support_lemma_col A F R (fun i j h => (hF i j h).2) v hv, add_zero]
  rfl

/-- **L2, obstruction.** If some `v ∈ K_R(A)` has `ρ(v) > 0`, then for every `F ∈ 𝓕_R`,
`P = H + F` makes `−(AᵀP + PA)` fail to be positive semidefinite, witnessed by
`vᵀ(−(AᵀP + PA))v < 0`. -/
theorem not_posSemidef_neg_rate (A H F : Matrix n n ℝ) (R : Set n) (hH : Hᵀ = H)
    (hFsymm : Fᵀ = F) (hF : SupportedOn R F) (v : n → ℝ) (hv : v ∈ kerOn A R)
    (hρ : 0 < witnessRate H A v) :
    v ⬝ᵥ ((-(Aᵀ * (H + F) + (H + F) * A)) *ᵥ v) < 0 ∧
      ¬ (-(Aᵀ * (H + F) + (H + F) * A)).PosSemidef := by
  have hrate := rate_eq_two_witnessRate A H F R hH hFsymm hF v hv
  have hneg : v ⬝ᵥ ((-(Aᵀ * (H + F) + (H + F) * A)) *ᵥ v) < 0 := by
    rw [neg_mulVec, dotProduct_neg, hrate]
    linarith
  refine ⟨hneg, fun hpsd => ?_⟩
  have h0 := hpsd.dotProduct_mulVec_nonneg v
  rw [star_trivial] at h0
  linarith

/-- **L2, no decay at any rate.** Under the same hypotheses, no `μ` with `2μP ⪰ 0` gives
`−(AᵀP + PA) ⪰ 2μP`, for `P = H + F`. -/
theorem no_decay_rate (A H F : Matrix n n ℝ) (R : Set n) (hH : Hᵀ = H)
    (hFsymm : Fᵀ = F) (hF : SupportedOn R F) (v : n → ℝ) (hv : v ∈ kerOn A R)
    (hρ : 0 < witnessRate H A v) (μ : ℝ) (hμ : ((2 * μ) • (H + F)).PosSemidef) :
    ¬ (-(Aᵀ * (H + F) + (H + F) * A) - (2 * μ) • (H + F)).PosSemidef := by
  intro h
  have hsum := h.add hμ
  rw [sub_add_cancel] at hsum
  exact (not_posSemidef_neg_rate A H F R hH hFsymm hF v hv hρ).2 hsum

end PHPSFormal.NoGo

#print axioms PHPSFormal.NoGo.support_lemma_col
#print axioms PHPSFormal.NoGo.support_lemma
#print axioms PHPSFormal.NoGo.rate_eq_two_witnessRate
#print axioms PHPSFormal.NoGo.not_posSemidef_neg_rate
#print axioms PHPSFormal.NoGo.no_decay_rate
