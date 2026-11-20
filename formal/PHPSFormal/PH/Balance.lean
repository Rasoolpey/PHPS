import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# The port-Hamiltonian form and its power balance; the time step and its discrete balance

Source: the PHPS paper draft `docs/papers/resubmission/paper/main.tex` (2026-11-18): eq. (1)-(3)
and Corollary 1 (claim P1), Proposition 3 (P6), the backward Euler identity of §IV (P7), the
discrete gradient of §IV (P8). Claim record: `formal/claims/P1_P6_P7_P8.md`.

Everything is over `ℝ` on finite index types: `n` the states, `m` the inputs.

* `e : n → ℝ` stands for the efforts `∇H(x)` (or a discrete gradient);
* the rates are `ẋ = (J - R) e + (G - P) u`, the outputs `y = (G + P)ᵀ e + S u` (eq. (1));
* the loss is the quadratic form of `[[R, P], [Pᵀ, S]]` on `(e, u)` (eq. (2)-(3)).
-/

namespace PHPSFormal.PH

open Matrix

variable {n m : Type*} [Fintype n] [Fintype m]

/-- The rates of eq. (1). -/
def rate (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (e : n → ℝ) (u : m → ℝ) : n → ℝ :=
  (J - R) *ᵥ e + (G - P) *ᵥ u

/-- The outputs of eq. (1). -/
def output (G P : Matrix n m ℝ) (S : Matrix m m ℝ) (e : n → ℝ) (u : m → ℝ) : m → ℝ :=
  (G + P)ᵀ *ᵥ e + S *ᵥ u

/-- The loss: the quadratic form of `[[R, P], [Pᵀ, S]]` on `(e, u)`, written out. -/
def loss (R : Matrix n n ℝ) (P : Matrix n m ℝ) (S : Matrix m m ℝ) (e : n → ℝ) (u : m → ℝ) : ℝ :=
  e ⬝ᵥ (R *ᵥ e) + e ⬝ᵥ (P *ᵥ u) + u ⬝ᵥ (Pᵀ *ᵥ e) + u ⬝ᵥ (S *ᵥ u)

/-- The loss is the quadratic form of the block matrix `fromBlocks R P Pᵀ S`. -/
theorem loss_eq_fromBlocks (R : Matrix n n ℝ) (P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (e : n → ℝ) (u : m → ℝ) :
    loss R P S e u = Sum.elim e u ⬝ᵥ (fromBlocks R P Pᵀ S *ᵥ Sum.elim e u) := by
  rw [fromBlocks_mulVec, sumElim_dotProduct_sumElim]
  simp only [Sum.elim_comp_inl, Sum.elim_comp_inr, dotProduct_add, loss]
  ring

/-- A skew-symmetric matrix does no work: `eᵀ J e = 0`. -/
theorem skew_quadratic_zero (J : Matrix n n ℝ) (hJ : Jᵀ = -J) (e : n → ℝ) :
    e ⬝ᵥ (J *ᵥ e) = 0 := by
  have h : e ⬝ᵥ (J *ᵥ e) = (Jᵀ *ᵥ e) ⬝ᵥ e := by
    rw [dotProduct_mulVec, mulVec_transpose]
  have hc : (J *ᵥ e) ⬝ᵥ e = e ⬝ᵥ (J *ᵥ e) := dotProduct_comm _ _
  rw [hJ, neg_mulVec, neg_dotProduct] at h
  linarith

/-- `uᵀ Pᵀ e = eᵀ P u`. -/
theorem dotProduct_transpose_mulVec (P : Matrix n m ℝ) (e : n → ℝ) (u : m → ℝ) :
    u ⬝ᵥ (Pᵀ *ᵥ e) = e ⬝ᵥ (P *ᵥ u) := by
  rw [mulVec_transpose, dotProduct_comm, dotProduct_mulVec]

/-- **P1, eq. (3).** With `J` skew-symmetric, the power entering the stores, `eᵀ ẋ`, equals the
port power `yᵀ u` less the loss. No other hypothesis is used. -/
theorem power_balance (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (e : n → ℝ) (u : m → ℝ) :
    e ⬝ᵥ rate J R G P e u = output G P S e u ⬝ᵥ u - loss R P S e u := by
  have hskew := skew_quadratic_zero J hJ e
  have hP := dotProduct_transpose_mulVec P e u
  have hG := dotProduct_transpose_mulVec G e u
  simp only [rate, output, loss, sub_mulVec, add_mulVec, dotProduct_add, dotProduct_sub,
    add_dotProduct, transpose_add]
  rw [dotProduct_comm ((Gᵀ) *ᵥ e) u, dotProduct_comm ((Pᵀ) *ᵥ e) u, dotProduct_comm (S *ᵥ u) u,
    hG, hP]
  linarith

/-- **P1, Corollary 1.** If in addition `[[R, P], [Pᵀ, S]] ⪰ 0`, then `eᵀ ẋ ≤ yᵀ u`. -/
theorem power_le_supply (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (hD : (fromBlocks R P Pᵀ S).PosSemidef) (e : n → ℝ) (u : m → ℝ) :
    e ⬝ᵥ rate J R G P e u ≤ output G P S e u ⬝ᵥ u := by
  have h0 := hD.dotProduct_mulVec_nonneg (Sum.elim e u)
  rw [star_trivial, ← loss_eq_fromBlocks] at h0
  rw [power_balance J R G P S hJ e u]
  linarith

/-- **P1, the chain rule.** If `H` has derivative `L` at `x t` with `L v = e ⬝ᵥ v` (so `e` is the
gradient), and the trajectory has velocity `rate … e u`, then `dH/dt` is the power balance. -/
theorem energy_rate (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (H : (n → ℝ) → ℝ) (x : ℝ → n → ℝ) (t : ℝ) (e : n → ℝ) (u : m → ℝ)
    (L : (n → ℝ) →L[ℝ] ℝ) (hL : ∀ v, L v = e ⬝ᵥ v)
    (hH : HasFDerivAt H L (x t)) (hx : HasDerivAt x (rate J R G P e u) t) :
    HasDerivAt (H ∘ x) (output G P S e u ⬝ᵥ u - loss R P S e u) t := by
  have h := hH.comp_hasDerivAt t hx
  rwa [hL, power_balance J R G P S hJ e u] at h

/-! ## Quadratic storage and the time step -/

/-- A quadratic-plus-affine storage `H(x) = ½ xᵀ Q x + cᵀ x + k`. -/
noncomputable def quad (Q : Matrix n n ℝ) (c : n → ℝ) (k : ℝ) (x : n → ℝ) : ℝ :=
  (1 / 2 : ℝ) * (x ⬝ᵥ (Q *ᵥ x)) + c ⬝ᵥ x + k

/-- Its gradient for symmetric `Q`. -/
def quadGrad (Q : Matrix n n ℝ) (c : n → ℝ) (x : n → ℝ) : n → ℝ := Q *ᵥ x + c

/-- For symmetric `Q`: `xᵀ Q y = yᵀ Q x`. -/
theorem symm_bilin (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (x y : n → ℝ) :
    x ⬝ᵥ (Q *ᵥ y) = y ⬝ᵥ (Q *ᵥ x) := by
  rw [dotProduct_mulVec, ← mulVec_transpose, hQ, dotProduct_comm]

/-- **The midpoint identity.** For quadratic-plus-affine `H` with symmetric `Q`,
`H(x₁) - H(x₀) = ∇H(x_m)ᵀ (x₁ - x₀)` exactly, with `x_m = ½(x₀ + x₁)`. -/
theorem quad_midpoint (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (c : n → ℝ) (k : ℝ) (x₀ x₁ : n → ℝ) :
    quad Q c k x₁ - quad Q c k x₀ =
      quadGrad Q c ((1 / 2 : ℝ) • (x₀ + x₁)) ⬝ᵥ (x₁ - x₀) := by
  have hs := symm_bilin Q hQ x₀ x₁
  simp only [quad, quadGrad, mulVec_smul, mulVec_add, add_dotProduct, smul_dotProduct,
    dotProduct_sub, smul_eq_mul]
  rw [dotProduct_comm (Q *ᵥ x₀) x₁, dotProduct_comm (Q *ᵥ x₀) x₀,
    dotProduct_comm (Q *ᵥ x₁) x₁, dotProduct_comm (Q *ᵥ x₁) x₀, dotProduct_comm c x₁,
    dotProduct_comm c x₀]
  linarith

/-- **The backward Euler identity (P7).** For quadratic-plus-affine `H` with symmetric `Q`,
`H(x₁) - H(x₀) = ∇H(x₁)ᵀ (x₁ - x₀) - ½ (x₁ - x₀)ᵀ Q (x₁ - x₀)`. -/
theorem quad_backward (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (c : n → ℝ) (k : ℝ) (x₀ x₁ : n → ℝ) :
    quad Q c k x₁ - quad Q c k x₀ =
      quadGrad Q c x₁ ⬝ᵥ (x₁ - x₀) - (1 / 2 : ℝ) * ((x₁ - x₀) ⬝ᵥ (Q *ᵥ (x₁ - x₀))) := by
  have hs := symm_bilin Q hQ x₀ x₁
  simp only [quad, quadGrad, mulVec_sub, add_dotProduct, dotProduct_sub, sub_dotProduct]
  rw [dotProduct_comm (Q *ᵥ x₁) x₁, dotProduct_comm (Q *ᵥ x₁) x₀, dotProduct_comm c x₁,
    dotProduct_comm c x₀]
  linarith

/-- **P8 (b) and P6, the discrete balance for any effort that satisfies the discrete chain rule.**
If the step is `x₁ - x₀ = h ((J - R) e + (G - P) u)`, `J` is skew, and `e ⬝ᵥ (x₁ - x₀) = ΔH`,
then `ΔH = h (yᵀ u - loss)`, with `y` and the loss evaluated at `e` and `u`. -/
theorem discrete_balance (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (h ΔH : ℝ) (x₀ x₁ e : n → ℝ) (u : m → ℝ)
    (hstep : x₁ - x₀ = h • rate J R G P e u) (hchain : e ⬝ᵥ (x₁ - x₀) = ΔH) :
    ΔH = h * (output G P S e u ⬝ᵥ u - loss R P S e u) := by
  rw [← hchain, hstep, dotProduct_smul, smul_eq_mul, power_balance J R G P S hJ e u]

/-- **P6, Proposition 3.** The implicit midpoint step on a quadratic-plus-affine storage:
`H(x₁) - H(x₀) = h (yᵀ u - loss)` at the midpoint efforts `e = ∇H(x_m)`. -/
theorem midpoint_balance (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (c : n → ℝ) (k h : ℝ) (x₀ x₁ : n → ℝ)
    (u : m → ℝ)
    (hstep : x₁ - x₀ = h • rate J R G P (quadGrad Q c ((1 / 2 : ℝ) • (x₀ + x₁))) u) :
    quad Q c k x₁ - quad Q c k x₀ =
      h * (output G P S (quadGrad Q c ((1 / 2 : ℝ) • (x₀ + x₁))) u ⬝ᵥ u -
        loss R P S (quadGrad Q c ((1 / 2 : ℝ) • (x₀ + x₁))) u) :=
  discrete_balance J R G P S hJ h _ x₀ x₁ _ u hstep (quad_midpoint Q hQ c k x₀ x₁).symm

/-- **P7.** The backward Euler step on a quadratic-plus-affine storage:
`H(x₁) - H(x₀) = h (yᵀ u - loss)|₁ - ½ Δxᵀ Q Δx`. The last term removes energy only when
`Q ⪰ 0` (and strictly only when `Δxᵀ Q Δx > 0`); for indefinite `Q` it can add energy. -/
theorem backward_euler_balance (J R : Matrix n n ℝ) (G P : Matrix n m ℝ) (S : Matrix m m ℝ)
    (hJ : Jᵀ = -J) (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (c : n → ℝ) (k h : ℝ) (x₀ x₁ : n → ℝ)
    (u : m → ℝ) (hstep : x₁ - x₀ = h • rate J R G P (quadGrad Q c x₁) u) :
    quad Q c k x₁ - quad Q c k x₀ =
      h * (output G P S (quadGrad Q c x₁) u ⬝ᵥ u - loss R P S (quadGrad Q c x₁) u) -
        (1 / 2 : ℝ) * ((x₁ - x₀) ⬝ᵥ (Q *ᵥ (x₁ - x₀))) := by
  rw [quad_backward Q hQ c k x₀ x₁]
  rw [discrete_balance J R G P S hJ h (quadGrad Q c x₁ ⬝ᵥ (x₁ - x₀)) x₀ x₁ _ u hstep rfl]

/-- **P8 (a), Gonzalez's discrete gradient.** For any `g` (in use: `∇H` at the midpoint) and
`Δx ≠ 0`, `ḡ = g + ((ΔH - gᵀΔx) / ‖Δx‖²) Δx` satisfies the discrete chain rule `ḡᵀ Δx = ΔH`.
No property of `H` is needed for this identity. -/
theorem gonzalez_chain (g Δx : n → ℝ) (ΔH : ℝ) (hΔ : Δx ⬝ᵥ Δx ≠ 0) :
    (g + ((ΔH - g ⬝ᵥ Δx) / (Δx ⬝ᵥ Δx)) • Δx) ⬝ᵥ Δx = ΔH := by
  rw [add_dotProduct, smul_dotProduct, smul_eq_mul, div_mul_cancel₀ _ hΔ]
  ring

/-- **P8, coupled devices with mixed efforts.** If the states split into blocks `ι` (the devices),
the storage is the sum of the blocks' storages, and every block's effort satisfies its own discrete
chain rule `e_bᵀ Δx_b = ΔH_b` (the midpoint gradient for a quadratic block, a discrete gradient for
any other), then the whole effort satisfies the whole chain rule; with `discrete_balance` the
whole-system balance follows for the step that uses these efforts in every row and output. -/
theorem blockwise_chain {ι : Type*} [Fintype ι] [DecidableEq ι] (blk : n → ι) (e Δx : n → ℝ) (ΔH : ι → ℝ)
    (hblk : ∀ b : ι, ∑ i ∈ Finset.univ.filter (fun i => blk i = b), e i * Δx i = ΔH b) :
    e ⬝ᵥ Δx = ∑ b, ΔH b := by
  simp only [dotProduct]
  rw [← Finset.sum_fiberwise Finset.univ blk (fun i => e i * Δx i)]
  exact Finset.sum_congr rfl (fun b _ => hblk b)

end PHPSFormal.PH

#print axioms PHPSFormal.PH.power_balance
#print axioms PHPSFormal.PH.power_le_supply
#print axioms PHPSFormal.PH.energy_rate
#print axioms PHPSFormal.PH.midpoint_balance
#print axioms PHPSFormal.PH.backward_euler_balance
#print axioms PHPSFormal.PH.gonzalez_chain
#print axioms PHPSFormal.PH.blockwise_chain
