import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# P14: the brushless exciter (Proposition 2)

Source: PHPS paper draft, Proposition 2; model: `docs/active/PH_BRUSHLESS_EXCITER_MODEL.md` §4-§5
(the IEEE 1981 report's Fig. 6 with the energy reading of §5); code: `src/phsim/exciter.jl`
(`PHBrushlessExciter`). Claim record: `formal/claims/P13-P14.md`.

Quantities (per unit, the report's notation): `x` = `x_E` the exciter's field flux (scaled so that
the exciter voltage is `ω x_E`), `V_R` the regulator's output, `I_FD` the generator's field current,
`K_E`, `K_D`, `T_E` the report's constants, `S_E` the saturation function, `g > 0` the field
circuit's power per unit of `V_R²`.

    V_FE = (K_E + S_E(x)) x + K_D I_FD           (the field current, on the scale of V_R)
    T_E x' = V_R - V_FE                           (the field circuit)
    W(x) = g T_E ∫₀ˣ (K_E + S_E(s)) s ds          (the field's magnetic energy)
    P_reg = g V_R V_FE,  P_tr = g K_D I_FD (V_R - V_FE),  loss g V_FE²

* `brushless_identity`: `g T_E (K_E + S_E) x · x' = P_reg - g V_FE² - P_tr` (algebra).
* `brushless_W_hasDerivAt`: `W' = g T_E (K_E + S_E(x)) x` for continuous `S_E` (calculus).
* `brushless_balance`: along a trajectory of the field circuit,
  `d/dt W(x(t)) = (P_reg - P_tr) - g V_FE²`. With the shaft power `P_sh = P_dc - P_tr + P_reg`
  (the model's definition, §5), `P_reg - P_tr = P_sh - P_dc` is the net power received through the
  shaft and the generator's field: the exciter stores, loses `g V_FE² ≥ 0` and creates nothing.
* `brushless_W_nonneg`: `W(x) ≥ 0` for `x ≥ 0`, `g T_E ≥ 0` and `K_E + S_E ≥ 0` on `[0, x]`.
-/

namespace PHPSFormal.PH

open intervalIntegral

/-- The exciter's stored energy. -/
noncomputable def brushlessW (g TE KE : ℝ) (SE : ℝ → ℝ) (x : ℝ) : ℝ :=
  g * TE * ∫ s in (0 : ℝ)..x, (KE + SE s) * s

/-- **The algebraic identity of §5.** -/
theorem brushless_identity (g TE KE KD IFD VR x SEx : ℝ) (hTE : TE ≠ 0) :
    g * TE * ((KE + SEx) * x) * ((VR - ((KE + SEx) * x + KD * IFD)) / TE) =
      g * VR * ((KE + SEx) * x + KD * IFD) - g * ((KE + SEx) * x + KD * IFD) ^ 2 -
        g * KD * IFD * (VR - ((KE + SEx) * x + KD * IFD)) := by
  field_simp
  ring

/-- **The stored energy's derivative.** -/
theorem brushless_W_hasDerivAt (g TE KE : ℝ) (SE : ℝ → ℝ) (hSE : Continuous SE) (x : ℝ) :
    HasDerivAt (brushlessW g TE KE SE) (g * TE * ((KE + SE x) * x)) x := by
  have hc : Continuous fun s => (KE + SE s) * s := (continuous_const.add hSE).mul continuous_id
  exact (hc.integral_hasStrictDerivAt 0 x).hasDerivAt.const_mul (g * TE)

/-- **Proposition 2: the energy balance along a trajectory.** -/
theorem brushless_balance (g TE KE KD : ℝ) (SE : ℝ → ℝ) (hSE : Continuous SE) (hTE : TE ≠ 0)
    (x VR IFD : ℝ → ℝ) (t : ℝ)
    (hx : HasDerivAt x ((VR t - ((KE + SE (x t)) * x t + KD * IFD t)) / TE) t) :
    HasDerivAt (fun τ => brushlessW g TE KE SE (x τ))
      ((g * VR t * ((KE + SE (x t)) * x t + KD * IFD t) -
          g * KD * IFD t * (VR t - ((KE + SE (x t)) * x t + KD * IFD t))) -
        g * ((KE + SE (x t)) * x t + KD * IFD t) ^ 2) t := by
  have h := (brushless_W_hasDerivAt g TE KE SE hSE (x t)).comp t hx
  exact h.congr_deriv (by rw [brushless_identity g TE KE KD (IFD t) (VR t) (x t) (SE (x t)) hTE]; ring)

/-- **The storage is non-negative** on the model's domain `x ≥ 0`. -/
theorem brushless_W_nonneg (g TE KE : ℝ) (SE : ℝ → ℝ) (x : ℝ) (hx : 0 ≤ x) (hgT : 0 ≤ g * TE)
    (hf : ∀ s ∈ Set.Icc 0 x, 0 ≤ KE + SE s) : 0 ≤ brushlessW g TE KE SE x := by
  unfold brushlessW
  apply mul_nonneg hgT
  apply intervalIntegral.integral_nonneg hx
  intro s hs
  exact mul_nonneg (hf s hs) hs.1

/-- The loss is non-negative. -/
theorem brushless_loss_nonneg (g VFE : ℝ) (hg : 0 ≤ g) : 0 ≤ g * VFE ^ 2 := by positivity

end PHPSFormal.PH

#print axioms PHPSFormal.PH.brushless_identity
#print axioms PHPSFormal.PH.brushless_W_hasDerivAt
#print axioms PHPSFormal.PH.brushless_balance
#print axioms PHPSFormal.PH.brushless_W_nonneg
#print axioms PHPSFormal.PH.brushless_loss_nonneg
