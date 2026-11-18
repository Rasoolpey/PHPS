import PHPSFormal.PH.Balance
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Algebra.Order.Star.Real

/-!
# The gaps of P10-P12: storage gradients, the saturable field, the turbine boundary, the series merge

Claim record: `formal/claims/P10-P12.md` (section "Gaps closed").

* `quad_rate`: along any differentiable trajectory, a quadratic-plus-affine storage with symmetric
  Hessian changes at the rate `(Q x + c)ᵀ ẋ`: the efforts `ω_b K x` of the network elements and of the
  machine's windings, and the speed `p/2H_m`, are the gradient of their storage (with `Q = ω_b K`,
  `Q = 1/2H_m`).
* `field_storage_hasDerivAt`, `field_storage_nonneg`: a saturable field with a continuous current
  characteristic `i(φ)` that has the sign of `φ` (the code's `[1 + n(K_E + S_E(|cφ|))] φ / (n T_E)`
  with a positive bracket) stores `W(φ) = ∫₀^φ i ≥ 0`, and `W' = i`: the field current is the gradient.
* `turbine_rows`: the turbine coupling `-a`, `a = q_t/(C_N ω)` between the last head and the rotor, with
  the reservoir port `K q_t / ω` on the rotor, gives the head row `-q_t / C_N` and the torque
  `K q_t h_N / ω`; the reservoir's output is the supply `K q_t`.
* `series_merge`: an inductance `L_e ≥ 0` and resistance `R_e ≥ 0` in series with the stator winding `s`
  (same current, same frame) add to the stator's self-inductance and resistance: the energy and the loss
  are the sums, and the merged inductance matrix stays positive definite.
-/

namespace PHPSFormal.PH

open Matrix

variable {n : Type*} [Fintype n]

/-- **Storage gradients.** -/
theorem quad_rate (Q : Matrix n n ℝ) (hQ : Qᵀ = Q) (c : n → ℝ) (k : ℝ) (x : ℝ → n → ℝ)
    (v : n → ℝ) (t : ℝ) (hx : HasDerivAt x v t) :
    HasDerivAt (fun τ => quad Q c k (x τ)) (quadGrad Q c (x t) ⬝ᵥ v) t := by
  have hi : ∀ i, HasDerivAt (fun τ => x τ i) (v i) t := fun i => hasDerivAt_pi.mp hx i
  -- the quadratic part, term by term
  have hq : HasDerivAt (fun τ => x τ ⬝ᵥ (Q *ᵥ x τ))
      (∑ i, (v i * (Q *ᵥ x t) i + x t i * (Q *ᵥ v) i)) t := by
    simp only [dotProduct, mulVec]
    apply HasDerivAt.fun_sum
    intro i _
    apply (hi i).mul
    have : HasDerivAt (fun τ => ∑ j, Q i j * x τ j) (∑ j, Q i j * v j) t :=
      HasDerivAt.fun_sum (fun j _ => (hi j).const_mul (Q i j))
    exact this
  have hl : HasDerivAt (fun τ => c ⬝ᵥ x τ) (c ⬝ᵥ v) t := by
    simp only [dotProduct]
    exact HasDerivAt.fun_sum (fun i _ => (hi i).const_mul (c i))
  have h := ((hq.const_mul (1 / 2 : ℝ)).add hl).add_const k
  refine h.congr_deriv ?_
  have hs := symm_bilin Q hQ (x t) v
  have e1 : ∑ i, (v i * (Q *ᵥ x t) i + x t i * (Q *ᵥ v) i) =
      v ⬝ᵥ (Q *ᵥ x t) + x t ⬝ᵥ (Q *ᵥ v) := by
    simp only [dotProduct, Finset.sum_add_distrib]
  rw [e1, quadGrad, add_dotProduct, dotProduct_comm (Q *ᵥ x t) v, dotProduct_comm c v, hs]
  ring

/-! ## The saturable field -/

/-- The field's stored energy `∫₀^φ i`. -/
noncomputable def fieldW (i : ℝ → ℝ) (φ : ℝ) : ℝ := ∫ s in (0 : ℝ)..φ, i s

theorem field_storage_hasDerivAt (i : ℝ → ℝ) (hi : Continuous i) (φ : ℝ) :
    HasDerivAt (fieldW i) (i φ) φ :=
  (hi.integral_hasStrictDerivAt 0 φ).hasDerivAt

theorem field_storage_nonneg (i : ℝ → ℝ) (hpos : ∀ s, 0 ≤ s → 0 ≤ i s)
    (hneg : ∀ s, s ≤ 0 → i s ≤ 0) (φ : ℝ) : 0 ≤ fieldW i φ := by
  unfold fieldW
  rcases le_total 0 φ with h | h
  · exact intervalIntegral.integral_nonneg h (fun s hs => hpos s hs.1)
  · rw [intervalIntegral.integral_symm, ← intervalIntegral.integral_neg]
    exact intervalIntegral.integral_nonneg h (fun s hs => by linarith [hneg s hs.2])

/-- The code's field characteristic has the sign of the flux when its bracket is positive. -/
theorem field_sign (β : ℝ → ℝ) (hβ : ∀ s, 0 < β s) (s : ℝ) :
    (0 ≤ s → 0 ≤ β s * s) ∧ (s ≤ 0 → β s * s ≤ 0) :=
  ⟨fun h => mul_nonneg (hβ s).le h, fun h => mul_nonpos_of_nonneg_of_nonpos (hβ s).le h⟩

/-! ## The turbine boundary -/

/-- **The turbine rows.** Efforts `z_e = K C_N e_N` (`e_N = h_N - 1`) and `z_p = ω`; the coupling
`J[e_N, p] = -a`, `J[p, e_N] = a`, `a = q_t/(C_N ω)`; the reservoir port `g = K q_t/ω` on the rotor
with the input `1` (the reservoir's head). -/
theorem turbine_rows (K CN qt ω eN : ℝ) (hC : CN ≠ 0) (hω : ω ≠ 0) :
    -(qt / (CN * ω)) * ω = -(qt / CN) ∧
      (qt / (CN * ω)) * (K * CN * eN) + K * qt / ω * 1 = K * qt * (eN + 1) / ω ∧
      K * qt / ω * ω = K * qt := by
  refine ⟨by field_simp, by field_simp, by field_simp⟩

/-! ## The series merge -/

/-- **The series merge.** The stator winding `s` and a series branch with inductance `Le` carry the
same current `i s`; the merged inductance is `L + Le e_s e_sᵀ`. Its energy is the sum of the two
energies; it is positive definite if `L` is and `Le ≥ 0`; the same holds for the resistances. -/
theorem series_merge [DecidableEq n] (L : Matrix n n ℝ) (Le : ℝ) (s : n) (i : n → ℝ) :
    i ⬝ᵥ ((L + Le • vecMulVec (Pi.single s 1) (Pi.single s 1)) *ᵥ i) =
      i ⬝ᵥ (L *ᵥ i) + Le * i s ^ 2 := by
  rw [add_mulVec, dotProduct_add, smul_mulVec]
  congr 1
  simp [vecMulVec, mulVec, dotProduct, Pi.single_apply, Finset.mul_sum]
  ring

theorem series_merge_posDef [DecidableEq n] (L : Matrix n n ℝ) (hL : L.PosDef) (Le : ℝ)
    (hLe : 0 ≤ Le) (s : n) :
    (L + Le • vecMulVec (Pi.single s (1 : ℝ)) (Pi.single s 1)).PosDef := by
  apply hL.add_posSemidef
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · ext a b; simp [vecMulVec, mul_comm]
  · intro y
    rw [star_trivial]
    have := series_merge (0 : Matrix n n ℝ) Le s y
    simp only [zero_add, zero_mulVec, dotProduct_zero] at this
    rw [this]; positivity


/-- **The series merge, resistances.** `R ⪰ 0`, `R_e ≥ 0` give `R + R_e e_s e_sᵀ ⪰ 0`. -/
theorem series_merge_psd [DecidableEq n] (R : Matrix n n ℝ) (hR : R.PosSemidef) (Re : ℝ)
    (hRe : 0 ≤ Re) (s : n) :
    (R + Re • vecMulVec (Pi.single s (1 : ℝ)) (Pi.single s 1)).PosSemidef := by
  apply hR.add
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · ext a b; simp [vecMulVec, mul_comm]
  · intro y
    rw [star_trivial]
    have := series_merge (0 : Matrix n n ℝ) Re s y
    simp only [zero_add, zero_mulVec, dotProduct_zero] at this
    rw [this]; positivity

end PHPSFormal.PH

#print axioms PHPSFormal.PH.quad_rate
#print axioms PHPSFormal.PH.field_storage_hasDerivAt
#print axioms PHPSFormal.PH.field_storage_nonneg
#print axioms PHPSFormal.PH.field_sign
#print axioms PHPSFormal.PH.turbine_rows
#print axioms PHPSFormal.PH.series_merge
#print axioms PHPSFormal.PH.series_merge_posDef
#print axioms PHPSFormal.PH.series_merge_psd
