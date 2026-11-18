import PHPSFormal.PH.Balance
import Mathlib.Analysis.Complex.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.LinearAlgebra.Matrix.Notation

/-!
# P10: the network elements and loads (Proposition 1, network part)

Source: PHPS paper draft, Table I (network, constant-impedance load) and Proposition 1; code:
`src/phsim/system.jl`, `phs_rhs!` and `phs_gradient`; `src/phsim/structure.jl`, `_structure_core`.
Claim record: `formal/claims/P10-P11.md`.

Every network element of the case has complex rows (in the `dq` frame rotating at the nominal
frequency) of one shape, with efforts `z = ω_b K x` (`K` real symmetric: `1/C` for a node charge,
`1/L` for a branch or load flux linkage, `M⁻¹` for a reduced cut set):

    ẋ = -j X z - R z + b,

with `X` real symmetric (`C`, `L` or `M`: the frame rotation), `R` real symmetric (`g`, `R`, `R_m`)
and `b` the port terms (node injections, end voltages). Code: node `inj - g v - j C v`; branch
`v_from / a - v_to - R i - j L i`; shunt (load) `v - R_sh i - j L_sh i`; cut set
`c v_far - j ω_b φ - R_m i` with `ω_b φ = M i`.

Split into real and imaginary parts (`z = z_r + j z_i`), the rows are
`(ẋ_r, ẋ_i) = (J - R₂)(z_r, z_i) + (b_r, b_i)` with
`J = [[0, X], [-X, 0]]` and `R₂ = [[R, 0], [0, R]]`.

* `element_rows`: the split rows are the stated form.
* `element_J_skew`: `J` is skew for symmetric `X`: the frame rotation does no work.
* `element_R_psd`: `R ⪰ 0 ⇒ R₂ ⪰ 0`.
* `element_energy`: `H = ½ xᵀ (ω_b K) x` has gradient `ω_b K x` and `H ≥ 0` for `K ⪰ 0`.
* `coupling_skew`, `coupling_rows`: a complex coupling coefficient `c` between a row pair `r` and
  a node pair (the incidence entries, a transformer ratio, a machine's terminal rotation
  `j e^{-jδ}/a`), entered as `c` in the row and `-c̄` in the node, is skew in real form: the
  element receives `c v` and the node `-c̄ i`, and the two powers cancel.
-/

namespace PHPSFormal.PH

open Matrix

variable {k : Type*} [Fintype k]

/-- The real form of `-j X` on `(z_r, z_i)`. -/
def rotJ (X : Matrix k k ℝ) : Matrix (k ⊕ k) (k ⊕ k) ℝ := fromBlocks 0 X (-X) 0

/-- The real form of `R` acting on both parts. -/
def lossR (R : Matrix k k ℝ) : Matrix (k ⊕ k) (k ⊕ k) ℝ := fromBlocks R 0 0 R

/-- **The rows.** Real and imaginary parts of `-j X z - R z + b`. -/
theorem element_rows (X R : Matrix k k ℝ) (zr zi br bi : k → ℝ) :
    (rotJ X - lossR R) *ᵥ Sum.elim zr zi + Sum.elim br bi =
      Sum.elim (X *ᵥ zi - R *ᵥ zr + br) (-(X *ᵥ zr) - R *ᵥ zi + bi) := by
  ext i
  rcases i with i | i <;>
    simp [rotJ, lossR, sub_mulVec, fromBlocks_mulVec, neg_mulVec]

/-- **The frame rotation does no work.** -/
theorem element_J_skew (X : Matrix k k ℝ) (hX : Xᵀ = X) : (rotJ X)ᵀ = -rotJ X := by
  rw [rotJ, fromBlocks_transpose, fromBlocks_neg, transpose_zero, transpose_neg, hX, neg_zero,
    neg_neg]

/-- **The losses.** -/
theorem element_R_psd (R : Matrix k k ℝ) (hR : R.PosSemidef) : (lossR R).PosSemidef := by
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · exact IsHermitian.fromBlocks hR.isHermitian (by simp) hR.isHermitian
  · intro y
    rw [← Sum.elim_comp_inl_inr y, lossR, fromBlocks_mulVec, star_trivial]
    simp only [Sum.elim_comp_inl, Sum.elim_comp_inr, zero_mulVec, add_zero, zero_add,
      sumElim_dotProduct_sumElim]
    have h1 := hR.dotProduct_mulVec_nonneg (y ∘ Sum.inl)
    have h2 := hR.dotProduct_mulVec_nonneg (y ∘ Sum.inr)
    rw [star_trivial] at h1 h2
    linarith

/-- **The storage.** `H = ½ xᵀ (ω_b K) x` with `K` symmetric has the gradient `ω_b K x` (the
midpoint identity of `Balance.lean` applies with `Q = ω_b K`), and `H ≥ 0` if `K ⪰ 0`, `ω_b ≥ 0`. -/
theorem element_energy_nonneg (K : Matrix k k ℝ) (hK : K.PosSemidef) (wb : ℝ) (hwb : 0 ≤ wb)
    (x : k → ℝ) : 0 ≤ quad (wb • K) 0 0 x := by
  have h := hK.dotProduct_mulVec_nonneg x
  rw [star_trivial] at h
  simp only [quad, smul_mulVec, dotProduct_smul, smul_eq_mul, zero_dotProduct, add_zero]
  positivity

/-! ## A complex coupling coefficient -/

/-- Multiplication by `c = a + j b` on `(re, im)`. -/
def cmul (c : ℂ) : Matrix (Fin 2) (Fin 2) ℝ := !![c.re, -c.im; c.im, c.re]

theorem cmul_conj (c : ℂ) : cmul (starRingEnd ℂ c) = (cmul c)ᵀ := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [cmul]

theorem cmul_neg (c : ℂ) : cmul (-c) = -cmul c := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [cmul]

/-- The real form of the product `c v`. -/
theorem cmul_mulVec (c v : ℂ) : cmul c *ᵥ ![v.re, v.im] = ![(c * v).re, (c * v).im] := by
  ext i; fin_cases i <;> simp [cmul, mulVec, dotProduct, Fin.sum_univ_two] <;> ring

/-- **The coupling is skew.** The element's row pair receives `cmul c`, the node's `cmul (-c̄)`;
the latter is minus the transpose of the former, so the assembled block is skew. -/
theorem coupling_skew (c : ℂ) : cmul (-(starRingEnd ℂ c)) = -(cmul c)ᵀ := by
  rw [cmul_neg, cmul_conj]

/-- **The powers cancel.** The element receives `c v` through the coupling and the node `-c̄ i`;
the power into the element, `Re(ī c v)`, and into the node, `Re(v̄ (-c̄ i))`, sum to zero. -/
theorem coupling_power (c v i : ℂ) :
    ((starRingEnd ℂ i) * (c * v)).re + ((starRingEnd ℂ v) * (-(starRingEnd ℂ c) * i)).re = 0 := by
  simp only [Complex.mul_re, Complex.mul_im, Complex.neg_re, Complex.neg_im, Complex.conj_re,
    Complex.conj_im]
  ring

end PHPSFormal.PH

#print axioms PHPSFormal.PH.element_rows
#print axioms PHPSFormal.PH.element_J_skew
#print axioms PHPSFormal.PH.element_R_psd
#print axioms PHPSFormal.PH.element_energy_nonneg
#print axioms PHPSFormal.PH.coupling_skew
#print axioms PHPSFormal.PH.coupling_power
