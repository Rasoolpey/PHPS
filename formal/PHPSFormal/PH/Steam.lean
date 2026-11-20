import PHPSFormal.PH.Balance
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Algebra.Order.Star.Real

/-!
# P13: the steam unit (Proposition 1, steam part), under stated property-model hypotheses

Source: PHPS paper draft, Proposition 1 (steam unit) and its proof; code:
`src/phsim/steam_structure.jl` (`steam_exergy_structure`); derivation `docs/active/PH_B4_EXERGY.md`.
Claim record: `formal/claims/P13-P14.md`.

The realization is built element by element. An element moves a flux `q` along a direction `w` in
the space of the storages' efforts (and, for the throttle and the reheater walls, of the reservoirs'
efforts); its affinity is `A = -wᵀe`, and the code sets `γ = q / A` (`γ = 0` at zero flow). The
properties of steam are **hypotheses** here (IAPWS-95 is not formalised): they are the ones the
paper's proof names.

1. `rank_one_flux`: `-(γ w wᵀ) e = q w`: the rank-one dissipation reproduces the element's flux.
2. `gamma_nonneg`, `rank_one_psd`: `q A ≥ 0` (non-negative production) gives `γ ≥ 0` and
   `γ w wᵀ ⪰ 0`; on the augmented space this is the block `[[R, P], [Pᵀ, S]]` of the throttle
   and of the walls.
3. `skew_pair`: the sections' reversible part, a column `c` on the rotor and its negative transpose,
   is skew.
4. `convex_tangent`, `concave_tangent`: a convex (concave) differentiable function lies above (below)
   its tangent.
5. `section_production`: a turbine section with `η ≤ 1` (`η ≥ 0` is not needed), `h_a ≥ h_is` and `h(·, p_b)` convex with
   slope `T_b` at `s_b` has `h_out ≥ h_rev`, so its dissipated work `L = q (h_out - h_rev) ≥ 0` and
   `γ = L T_b / T₀ ≥ 0`; `section_affinity`: the direction `(−1/ω on the rotor, +1 on U_b)` has
   affinity `T₀ / T_b`.
6. `branch_affinity`, `branch_affinity_nonneg`: an isenthalpic branch from a donor at `(h_d, s_d)` to a
   receiver at `(h_r, s_r, T_r)` has `A = T₀ ((h_d - h_r)/T_r + s_r - s_d)`, and `A ≥ 0` when
   `s(·, p_r)` is concave with slope `1/T_r` at `h_r` and `s(h_d, p_d) ≤ s(h_d, p_r)` (the flow goes
   from the higher to the lower pressure, `∂s/∂p|_h = -v/T < 0`).
7. `heat_production_nonneg`: `k (T_w - T_k) T₀ (1/T_k - 1/T_w) ≥ 0` and `γ_h = k T_w T_k / T₀ ≥ 0`.
8. `exergy_nonneg`: with the dead-state supporting plane `T₀ S_k ≤ U_k + p₀ V_k - μ₀ M_k` of every
   vessel, `H_ex = Σ (U_k - T₀ S_k + p₀ V_k - μ₀ M_k) ≥ 0`.
9. `ledger_step`, `ledger_internal`: a linear ledger (the first-law `Σ U_k`, the mass `Σ M_k`) changes
   over any step `x₁ - x₀ = h f` by exactly `h ℓᵀ f`, and the internal elements (branches, whose
   direction moves mass and enthalpy from one vessel to another) do not enter it.
-/

namespace PHPSFormal.PH

open Matrix

variable {ι : Type*} [Fintype ι]

/-! ## Rank-one elements -/

theorem vecMulVec_self_mulVec (w e : ι → ℝ) : vecMulVec w w *ᵥ e = (w ⬝ᵥ e) • w := by
  ext i
  simp only [mulVec, dotProduct, vecMulVec, of_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_mul]
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-- **1.** The rank-one dissipation reproduces the element's flux. -/
theorem rank_one_flux (w e : ι → ℝ) (q : ℝ) (hA : w ⬝ᵥ e ≠ 0) :
    -(((q / -(w ⬝ᵥ e)) • vecMulVec w w) *ᵥ e) = q • w := by
  rw [smul_mulVec, vecMulVec_self_mulVec, smul_smul]
  have h : q / -(w ⬝ᵥ e) * (w ⬝ᵥ e) = -q := by field_simp
  rw [h, neg_smul, neg_neg]

/-- **2.** Non-negative production gives a non-negative coefficient. -/
theorem gamma_nonneg (q A : ℝ) (hqA : 0 ≤ q * A) : 0 ≤ q / A := by
  rcases lt_trichotomy A 0 with h | h | h
  · have : q ≤ 0 := by nlinarith
    exact div_nonneg_of_nonpos this h.le
  · simp [h]
  · have : 0 ≤ q := by nlinarith
    exact div_nonneg this h.le

/-- **2.** A non-negative multiple of `w wᵀ` is positive semidefinite. -/
theorem rank_one_psd (w : ι → ℝ) (γ : ℝ) (hγ : 0 ≤ γ) : (γ • vecMulVec w w).PosSemidef := by
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · ext i j; simp [vecMulVec, mul_comm]
  · intro x
    rw [star_trivial, smul_mulVec, vecMulVec_self_mulVec, smul_smul, dotProduct_smul, smul_eq_mul,
      dotProduct_comm x w]
    nlinarith [sq_nonneg (w ⬝ᵥ x)]

/-- **3.** A column `c` on one effort and its negative transpose form a skew matrix. -/
theorem skew_pair (c d : ι → ℝ) : (vecMulVec c d - vecMulVec d c)ᵀ = -(vecMulVec c d - vecMulVec d c) := by
  ext i j; simp [vecMulVec]; ring

/-! ## Tangent lines -/

/-- **4.** A convex function lies above its tangent. -/
theorem convex_tangent {S : Set ℝ} {f : ℝ → ℝ} {f' x y : ℝ} (hfc : ConvexOn ℝ S f) (hx : x ∈ S)
    (hy : y ∈ S) (hf : HasDerivAt f f' x) : f x + f' * (y - x) ≤ f y := by
  rcases lt_trichotomy x y with h | h | h
  · have hs := hfc.le_slope_of_hasDerivAt hx hy h hf
    rw [slope_def_field, le_div_iff₀ (by linarith)] at hs
    linarith
  · subst h; simp
  · have hs := hfc.slope_le_of_hasDerivAt hy hx h hf
    rw [slope_def_field, div_le_iff₀ (by linarith)] at hs
    linarith

/-- **4.** A concave function lies below its tangent. -/
theorem concave_tangent {S : Set ℝ} {f : ℝ → ℝ} {f' x y : ℝ} (hfc : ConcaveOn ℝ S f) (hx : x ∈ S)
    (hy : y ∈ S) (hf : HasDerivAt f f' x) : f y ≤ f x + f' * (y - x) := by
  have h := convex_tangent hfc.neg hx hy hf.neg
  simp only [Pi.neg_apply] at h
  linarith

/-! ## The elements' signs -/

/-- **5.** A turbine section dissipates a non-negative work. `hb s` is the enthalpy at the outlet
pressure as a function of entropy (convex on `S`, slope `T_b` at the outlet state `s_b`), `h_is =
hb s_a` the isentropic outlet enthalpy, `h_rev = hb s_b + T_b (s_a - s_b)` the reversible one. -/
theorem section_production {S : Set ℝ} (hb : ℝ → ℝ) (Tb sa sb ha η q T0 : ℝ)
    (hconv : ConvexOn ℝ S hb) (hsa : sa ∈ S) (hsb : sb ∈ S) (hT : HasDerivAt hb Tb sb)
    (hexp : hb sa ≤ ha) (hη1 : η ≤ 1) (hq : 0 ≤ q) (hTb : 0 < Tb) (hT0 : 0 < T0) :
    hb sb + Tb * (sa - sb) ≤ ha - η * (ha - hb sa) ∧
      0 ≤ q * ((ha - η * (ha - hb sa)) - (hb sb + Tb * (sa - sb))) ∧
      0 ≤ q * ((ha - η * (ha - hb sa)) - (hb sb + Tb * (sa - sb))) * Tb / T0 := by
  have htan := convex_tangent hconv hsb hsa hT
  have hout : hb sa ≤ ha - η * (ha - hb sa) := by nlinarith
  have h1 : hb sb + Tb * (sa - sb) ≤ ha - η * (ha - hb sa) := by linarith
  have h2 : 0 ≤ q * ((ha - η * (ha - hb sa)) - (hb sb + Tb * (sa - sb))) :=
    mul_nonneg hq (by linarith)
  exact ⟨h1, h2, div_nonneg (mul_nonneg h2 hTb.le) hT0.le⟩

/-- **5.** The section's dissipative direction has affinity `T₀ / T_b`: with the rotor's effort `ω`
and `U_b`'s effort `1 - T₀/T_b`, `-(−(1/ω) ω + (1 - T₀/T_b)) = T₀/T_b`. -/
theorem section_affinity (ω T0 Tb : ℝ) (hω : ω ≠ 0) :
    -((-(1 / ω)) * ω + 1 * (1 - T0 / Tb)) = T0 / Tb := by
  field_simp
  ring

/-- **6.** The affinity of an isenthalpic branch, from the efforts `b = T₀ μ/T - μ₀`, `a = 1 - T₀/T`,
`μ = h - T s`: the direction is `(-1, -h_d)` on the donor and `(+1, +h_d)` on the receiver. -/
theorem branch_affinity (T0 μ0 hd sd Td hr sr Tr : ℝ) (hTd : Td ≠ 0) (hTr : Tr ≠ 0) :
    -((-1) * (T0 * (hd - Td * sd) / Td - μ0) + (-hd) * (1 - T0 / Td) +
        (1 * (T0 * (hr - Tr * sr) / Tr - μ0) + hd * (1 - T0 / Tr))) =
      T0 * ((hd - hr) / Tr + sr - sd) := by
  field_simp
  ring

/-- **6.** The branch's affinity is non-negative. `sr_of h` is the entropy at the receiver's
pressure as a function of enthalpy (concave, slope `1/T_r` at the receiver's state `h_r`, value
`s_r` there); `sd ≤ sr_of hd` says that the donor's entropy at its own pressure is at most the
entropy at the same enthalpy and the receiver's lower pressure. -/
theorem branch_affinity_nonneg {S : Set ℝ} (sr_of : ℝ → ℝ) (T0 hd sd hr Tr : ℝ)
    (hconc : ConcaveOn ℝ S sr_of) (hhd : hd ∈ S) (hhr : hr ∈ S)
    (hT : HasDerivAt sr_of (1 / Tr) hr) (hp : sd ≤ sr_of hd) (hT0 : 0 ≤ T0) (hTr : Tr ≠ 0) :
    0 ≤ T0 * ((hd - hr) / Tr + sr_of hr - sd) := by
  have htan := concave_tangent hconc hhr hhd hT
  apply mul_nonneg hT0
  have : (hd - hr) / Tr = 1 / Tr * (hd - hr) := by field_simp
  linarith

/-- **7.** Heat from a wall at `T_w` to a vessel at `T_k`. -/
theorem heat_production_nonneg (k Tw Tk T0 : ℝ) (hk : 0 ≤ k) (hTw : 0 < Tw) (hTk : 0 < Tk)
    (hT0 : 0 < T0) :
    0 ≤ k * (Tw - Tk) * T0 * (1 / Tk - 1 / Tw) ∧ 0 ≤ k * Tw * Tk / T0 := by
  constructor
  · have h : (Tw - Tk) * (1 / Tk - 1 / Tw) = (Tw - Tk) ^ 2 / (Tk * Tw) := by
      field_simp
    have : k * (Tw - Tk) * T0 * (1 / Tk - 1 / Tw) = k * T0 * ((Tw - Tk) ^ 2 / (Tk * Tw)) := by
      rw [← h]; ring
    rw [this]; positivity
  · positivity

/-- **8.** The exergy about the dead state is non-negative, given the supporting plane of each
vessel. -/
theorem exergy_nonneg {κ : Type*} [Fintype κ] (U S V M : κ → ℝ) (T0 p0 μ0 : ℝ)
    (hplane : ∀ k, T0 * S k ≤ U k + p0 * V k - μ0 * M k) :
    0 ≤ ∑ k, (U k - T0 * S k + p0 * V k - μ0 * M k) :=
  Finset.sum_nonneg fun k _ => by linarith [hplane k]

/-! ## The first-law ledger over a step -/

/-- **9.** A linear ledger changes over a step by exactly `h ℓᵀ f`. -/
theorem ledger_step (ℓ x₀ x₁ f : ι → ℝ) (h : ℝ) (hstep : x₁ - x₀ = h • f) :
    ℓ ⬝ᵥ x₁ - ℓ ⬝ᵥ x₀ = h * (ℓ ⬝ᵥ f) := by
  rw [← dotProduct_sub, hstep, dotProduct_smul, smul_eq_mul]

/-- **9.** Internal elements do not enter the ledger: if `f = Σ_e q_e w_e + f_ext` and every internal
direction has `ℓᵀ w_e = 0`, the ledger's rate is `ℓᵀ f_ext`. -/
theorem ledger_internal {ε : Type*} [Fintype ε] (ℓ fext : ι → ℝ) (q : ε → ℝ) (w : ε → ι → ℝ)
    (hw : ∀ e, ℓ ⬝ᵥ w e = 0) :
    ℓ ⬝ᵥ (∑ e, q e • w e + fext) = ℓ ⬝ᵥ fext := by
  rw [dotProduct_add, dotProduct_sum]
  simp [dotProduct_smul, hw]


/-! ## Additions after review 126 -/

/-- **Zero flow.** With `γ = 0` the element contributes nothing, which is its flux `0 • w`. -/
theorem rank_one_zero_flow (w e : ι → ℝ) : -(((0 : ℝ) • vecMulVec w w) *ᵥ e) = (0 : ℝ) • w := by
  simp

/-- **A flowing branch has a positive affinity** when the pressure ordering is strict
(`s_d < s(h_d, p_r)`): then `A > 0`, `γ = q/A` is finite and `rank_one_flux` applies. -/
theorem branch_affinity_pos {S : Set ℝ} (sr_of : ℝ → ℝ) (T0 hd sd hr Tr : ℝ)
    (hconc : ConcaveOn ℝ S sr_of) (hhd : hd ∈ S) (hhr : hr ∈ S)
    (hT : HasDerivAt sr_of (1 / Tr) hr) (hp : sd < sr_of hd) (hT0 : 0 < T0) (hTr : Tr ≠ 0) :
    0 < T0 * ((hd - hr) / Tr + sr_of hr - sd) := by
  have htan := concave_tangent hconc hhr hhd hT
  apply mul_pos hT0
  have : (hd - hr) / Tr = 1 / Tr * (hd - hr) := by field_simp
  linarith

/-- **The pressure premises.** With `h(s_a, ·)` monotone in pressure (`∂h/∂p|_s = v > 0`) an
expansion `p_b ≤ p_a` has `h_is = h(s_a, p_b) ≤ h(s_a, p_a) = h_a`; with `s(h_d, ·)` antitone in
pressure (`∂s/∂p|_h = -v/T < 0`) a branch from `p_d` to `p_r ≤ p_d` has `s_d = s(h_d, p_d) ≤
s(h_d, p_r)`, strictly if `s(h_d, ·)` is strictly antitone and `p_r < p_d`. -/
theorem expansion_premise {P : Set ℝ} (h_sa : ℝ → ℝ) (pa pb : ℝ) (hmono : MonotoneOn h_sa P)
    (ha : pa ∈ P) (hb : pb ∈ P) (hp : pb ≤ pa) : h_sa pb ≤ h_sa pa :=
  hmono hb ha hp

theorem branch_premise {P : Set ℝ} (s_hd : ℝ → ℝ) (pd pr : ℝ) (hanti : StrictAntiOn s_hd P)
    (hd : pd ∈ P) (hr : pr ∈ P) (hp : pr < pd) : s_hd pd < s_hd pr :=
  hanti hr hd hp

/-- **The steam efforts are the exergy's gradient.** If the vessel's entropy changes as
`dS = (1/T) dU - (μ/T) dM` at fixed volume (the Gibbs relation), the exergy
`H_ex = U - T₀ S + p₀ V - μ₀ M` changes as `a dU + b dM` with `a = 1 - T₀/T`,
`b = T₀ μ/T - μ₀`: the code's efforts. -/
theorem steam_effort (T0 μ0 T μ dU dM : ℝ) (hT : T ≠ 0) :
    dU - T0 * ((1 / T) * dU - (μ / T) * dM) - μ0 * dM =
      (1 - T0 / T) * dU + (T0 * μ / T - μ0) * dM := by
  field_simp
  ring


/-! ## The assembly of the elements -/

/-- **The assembled dissipation is positive semidefinite.** A sum of non-negative rank-one terms
(on the storages, or on the augmented space of storages and reservoirs: the block
`[[R, P], [Pᵀ, S]]`) is PSD. -/
theorem assembled_psd {ε : Type*} [Fintype ε] (γ : ε → ℝ) (w : ε → ι → ℝ) (hγ : ∀ e, 0 ≤ γ e) :
    (∑ e, γ e • vecMulVec (w e) (w e)).PosSemidef := by
  classical
  apply PosSemidef.of_dotProduct_mulVec_nonneg
  · ext i j; simp [Matrix.sum_apply, vecMulVec, mul_comm]
  · intro x
    rw [star_trivial, Matrix.sum_mulVec, dotProduct_sum]
    apply Finset.sum_nonneg
    intro e _
    have h := (rank_one_psd (w e) (γ e) (hγ e)).dotProduct_mulVec_nonneg x
    rwa [star_trivial] at h

/-- **The assembled reversible coupling is skew.** -/
theorem assembled_skew {ε : Type*} [Fintype ε] (c d : ε → ι → ℝ) :
    (∑ e, (vecMulVec (c e) (d e) - vecMulVec (d e) (c e)))ᵀ =
      -∑ e, (vecMulVec (c e) (d e) - vecMulVec (d e) (c e)) := by
  rw [Matrix.transpose_sum, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl (fun e _ => skew_pair (c e) (d e))

/-- **The assembled rows reproduce the elements' fluxes.** With `γ_e = q_e / A_e` and every
flowing element's affinity non-zero (zero-flow elements carry `γ_e = 0`), `-R e = Σ q_e w_e`. -/
theorem assembled_flux {ε : Type*} [Fintype ε] (q : ε → ℝ) (w : ε → ι → ℝ) (e : ι → ℝ)
    (hA : ∀ k, q k ≠ 0 → w k ⬝ᵥ e ≠ 0) :
    -((∑ k, (q k / -(w k ⬝ᵥ e)) • vecMulVec (w k) (w k)) *ᵥ e) = ∑ k, q k • w k := by
  rw [Matrix.sum_mulVec, ← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hq : q k = 0
  · simp [hq]
  · exact rank_one_flux (w k) e (q k) (hA k hq)

end PHPSFormal.PH

#print axioms PHPSFormal.PH.rank_one_flux
#print axioms PHPSFormal.PH.gamma_nonneg
#print axioms PHPSFormal.PH.rank_one_psd
#print axioms PHPSFormal.PH.skew_pair
#print axioms PHPSFormal.PH.convex_tangent
#print axioms PHPSFormal.PH.concave_tangent
#print axioms PHPSFormal.PH.section_production
#print axioms PHPSFormal.PH.section_affinity
#print axioms PHPSFormal.PH.branch_affinity
#print axioms PHPSFormal.PH.branch_affinity_nonneg
#print axioms PHPSFormal.PH.heat_production_nonneg
#print axioms PHPSFormal.PH.exergy_nonneg
#print axioms PHPSFormal.PH.ledger_step
#print axioms PHPSFormal.PH.ledger_internal
#print axioms PHPSFormal.PH.rank_one_zero_flow
#print axioms PHPSFormal.PH.branch_affinity_pos
#print axioms PHPSFormal.PH.expansion_premise
#print axioms PHPSFormal.PH.branch_premise
#print axioms PHPSFormal.PH.steam_effort
#print axioms PHPSFormal.PH.assembled_psd
#print axioms PHPSFormal.PH.assembled_skew
#print axioms PHPSFormal.PH.assembled_flux
