import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Analysis.RCLike.Basic

/-!
# P2: the network's Kirchhoff relations form a Dirac structure

Source: PHPS paper draft, §II-B (eq. (4) and the paragraph after it). Claim record:
`formal/claims/P2.md`.

Over `𝕜 = ℂ` (the `dq` axes, a complex incidence matrix with transformer ratios inside) or
`𝕜 = ℝ` (the zero axis, its own real incidence matrix). Branches `b`, nodes `nd`.

A port element is `(i_b, i_n, v_b, v_n)`; the flows are `f = (i_b, -i_n)`, the efforts
`e = (v_b, v_n)`. The power of an element is `Re(eᴴ f) = Re(v_bᴴ i_b) - Re(v_nᴴ i_n)`, and the
symmetric pairing of two elements is `Re(e₁ᴴ f₂ + e₂ᴴ f₁)`, a real bilinear form on the
underlying real vector space.

`D = {v_b = Bᴴ v_n, i_n = B i_b}` (Kirchhoff's voltage and current laws).

* `power_zero`: the power vanishes on every element of `D` (no rank or sign condition on `B`).
* `dirac`: an element is orthogonal, under the pairing, to every element of `D` **if and only if**
  it lies in `D`. So `D = D^⊥`: `D` is a Dirac structure on the underlying real space.
-/

namespace PHPSFormal.PH

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {b nd : Type*} [Fintype b] [Fintype nd]

/-- `aᴴ (B j) = (Bᴴ a)ᴴ j`. -/
theorem star_dot_mulVec (B : Matrix nd b 𝕜) (a : nd → 𝕜) (j : b → 𝕜) :
    star a ⬝ᵥ (B *ᵥ j) = star (Bᴴ *ᵥ a) ⬝ᵥ j := by
  rw [star_mulVec, conjTranspose_conjTranspose, dotProduct_mulVec]

/-- A vector with `Re(dᴴ d) = 0` is zero. -/
theorem eq_zero_of_re_star_dot_self (d : b → 𝕜) (h : RCLike.re (star d ⬝ᵥ d) = 0) : d = 0 := by
  have hterm : ∀ k, RCLike.re ((star d) k * d k) = ‖d k‖ ^ 2 := by
    intro k
    rw [Pi.star_apply, RCLike.star_def, RCLike.conj_mul]
    norm_cast
  have hsum : ∑ k, ‖d k‖ ^ 2 = 0 := by
    rw [← h, dotProduct, map_sum]
    exact Finset.sum_congr rfl (fun k _ => (hterm k).symm)
  funext k
  have hk := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => sq_nonneg ‖d i‖)).mp hsum k
    (Finset.mem_univ k)
  simpa using hk

/-- The power `Re(eᴴ f)` of a port element `(i_b, i_n, v_b, v_n)`, with `f = (i_b, -i_n)`. -/
def portPower (ib : b → 𝕜) (inn : nd → 𝕜) (vb : b → 𝕜) (vn : nd → 𝕜) : ℝ :=
  RCLike.re (star vb ⬝ᵥ ib) - RCLike.re (star vn ⬝ᵥ inn)

/-- The symmetric pairing `Re(e₁ᴴ f₂ + e₂ᴴ f₁)` of two port elements. -/
def pairing (ib₁ : b → 𝕜) (in₁ : nd → 𝕜) (vb₁ : b → 𝕜) (vn₁ : nd → 𝕜)
    (ib₂ : b → 𝕜) (in₂ : nd → 𝕜) (vb₂ : b → 𝕜) (vn₂ : nd → 𝕜) : ℝ :=
  RCLike.re (star vb₁ ⬝ᵥ ib₂ - star vn₁ ⬝ᵥ in₂ + (star vb₂ ⬝ᵥ ib₁ - star vn₂ ⬝ᵥ in₁))

/-- Membership in `D`: Kirchhoff's voltage and current laws. -/
def InD (B : Matrix nd b 𝕜) (ib : b → 𝕜) (inn : nd → 𝕜) (vb : b → 𝕜) (vn : nd → 𝕜) : Prop :=
  vb = Bᴴ *ᵥ vn ∧ inn = B *ᵥ ib

/-- **P2 (a).** The interconnection neither supplies nor absorbs power: on every element of `D`
the power is zero. Nothing is assumed about `B`. -/
theorem power_zero (B : Matrix nd b 𝕜) (ib : b → 𝕜) (inn : nd → 𝕜) (vb : b → 𝕜) (vn : nd → 𝕜)
    (h : InD B ib inn vb vn) : portPower ib inn vb vn = 0 := by
  obtain ⟨hv, hi⟩ := h
  rw [portPower, hv, hi, star_dot_mulVec, sub_self]

/-- **P2 (b), `D = D^⊥`.** An element is orthogonal under the pairing to every element of `D`
(parametrized by its free branch currents `j` and node voltages `w`) if and only if it lies in
`D`. Nothing is assumed about `B` (no rank condition, any complex transformer ratios). -/
theorem dirac (B : Matrix nd b 𝕜) (ib : b → 𝕜) (inn : nd → 𝕜) (vb : b → 𝕜) (vn : nd → 𝕜) :
    (∀ (j : b → 𝕜) (w : nd → 𝕜), pairing ib inn vb vn j (B *ᵥ j) (Bᴴ *ᵥ w) w = 0) ↔
      InD B ib inn vb vn := by
  -- the pairing with the element of `D` given by `(j, w)`, rewritten
  have key : ∀ (j : b → 𝕜) (w : nd → 𝕜), pairing ib inn vb vn j (B *ᵥ j) (Bᴴ *ᵥ w) w =
      RCLike.re (star (vb - Bᴴ *ᵥ vn) ⬝ᵥ j + star w ⬝ᵥ (B *ᵥ ib - inn)) := by
    intro j w
    rw [pairing, star_dot_mulVec B vn j, ← star_dot_mulVec B w ib, star_sub, sub_dotProduct,
      dotProduct_sub]
  constructor
  · intro h
    have hd : vb - Bᴴ *ᵥ vn = 0 := by
      apply eq_zero_of_re_star_dot_self
      have := h (vb - Bᴴ *ᵥ vn) 0
      rwa [key, star_zero, zero_dotProduct, add_zero] at this
    have hc : B *ᵥ ib - inn = 0 := by
      apply eq_zero_of_re_star_dot_self
      have := h 0 (B *ᵥ ib - inn)
      rwa [key, dotProduct_zero, zero_add] at this
    exact ⟨sub_eq_zero.mp hd, (sub_eq_zero.mp hc).symm⟩
  · rintro ⟨hv, hi⟩ j w
    rw [key, hv, hi, sub_self, sub_self, star_zero, zero_dotProduct, dotProduct_zero, add_zero,
      map_zero]

end PHPSFormal.PH

#print axioms PHPSFormal.PH.power_zero
#print axioms PHPSFormal.PH.dirac
