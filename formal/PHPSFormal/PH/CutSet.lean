import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# P4: elimination of a cut set of inductors

Source: PHPS paper draft, §II-B, "Elimination of a cut set of inductors". Claim record:
`formal/claims/P4.md`.

Branch currents `i : br → ℝ` with constant inductance `L`, constraint `A i = 0` (`A : nd × br`,
the currents at a node with no capacitance sum to zero), `N : br × k` with `A N = 0` and
`range N = ker A`, flux linkages `φ = L i`, reduced flux linkage `ψ = Nᵀ φ`, `M = Nᵀ L N`.

* `reduced_energy`: for `i = N η`, `ψᵀ M⁻¹ ψ = iᵀ L i` (needs `L ≻ 0` and `N` injective, so that
  `M ≻ 0` is invertible; no symmetry of `L` is used beyond `PosDef`).
* `reduced_energy_constrained`: the same for every current with `A i = 0`, if `range N = ker A`.
* `voltage_drops_out`: `Nᵀ Aᵀ λ = 0`: the node voltage (the multiplier of the constraint) does not
  enter the reduced rows.
* `reduced_rates`: with `φ' = -R_b i + Aᵀ λ + E` (branch resistance `R_b`, sources `E`),
  `ψ' = Nᵀ φ' = -(Nᵀ R_b N) η + Nᵀ E`.
* `reduced_loss_psd`: `R_b ⪰ 0 ⇒ Nᵀ R_b N ⪰ 0`.
* `reduced_port_power`: the sources' power is the same, `Eᵀ i = (Nᵀ E)ᵀ η`.

Constant `A`, `L`, `R_b` and a fixed topology are built into the statement: the matrices are fixed.
-/

namespace PHPSFormal.PH

open Matrix

variable {br k nd : Type*} [Fintype br] [Fintype k] [Fintype nd] [DecidableEq k]

theorem reduced_posDef (L : Matrix br br ℝ) (hL : L.PosDef) (N : Matrix br k ℝ)
    (hN : Function.Injective N.mulVec) : (Nᵀ * L * N).PosDef := by
  have h := hL.conjTranspose_mul_mul_same hN
  rwa [conjTranspose_eq_transpose_of_trivial] at h

/-- **P4 (a).** The reduced energy equals the branch energy on the constraint set. -/
theorem reduced_energy (L : Matrix br br ℝ) (hL : L.PosDef) (N : Matrix br k ℝ)
    (hN : Function.Injective N.mulVec) (η : k → ℝ) :
    (Nᵀ *ᵥ (L *ᵥ (N *ᵥ η))) ⬝ᵥ ((Nᵀ * L * N)⁻¹ *ᵥ (Nᵀ *ᵥ (L *ᵥ (N *ᵥ η)))) =
      (N *ᵥ η) ⬝ᵥ (L *ᵥ (N *ᵥ η)) := by
  have hM := reduced_posDef L hL N hN
  have hdet : IsUnit (Nᵀ * L * N).det := hM.isUnit.map Matrix.detMonoidHom
  have hψ : Nᵀ *ᵥ (L *ᵥ (N *ᵥ η)) = (Nᵀ * L * N) *ᵥ η := by
    rw [mulVec_mulVec, mulVec_mulVec]
  have hinv : (Nᵀ * L * N)⁻¹ *ᵥ ((Nᵀ * L * N) *ᵥ η) = η := by
    rw [mulVec_mulVec, nonsing_inv_mul _ hdet, one_mulVec]
  rw [hψ, hinv, ← hψ, mulVec_transpose, ← dotProduct_mulVec, dotProduct_comm]

/-- **P4 (b).** The multiplier (node voltage) drops out of the reduced rows. -/
theorem voltage_drops_out (A : Matrix nd br ℝ) (N : Matrix br k ℝ) (hAN : A * N = 0)
    (lam : nd → ℝ) : Nᵀ *ᵥ (Aᵀ *ᵥ lam) = 0 := by
  rw [mulVec_mulVec, ← transpose_mul, hAN, transpose_zero, zero_mulVec]

/-- **P4 (c).** The reduced rates. -/
theorem reduced_rates (A : Matrix nd br ℝ) (N : Matrix br k ℝ) (hAN : A * N = 0)
    (Rb : Matrix br br ℝ) (E : br → ℝ) (lam : nd → ℝ) (η : k → ℝ) :
    Nᵀ *ᵥ (-(Rb *ᵥ (N *ᵥ η)) + Aᵀ *ᵥ lam + E) = -((Nᵀ * Rb * N) *ᵥ η) + Nᵀ *ᵥ E := by
  rw [mulVec_add, mulVec_add, voltage_drops_out A N hAN, add_zero, mulVec_neg, mulVec_mulVec,
    mulVec_mulVec, Matrix.mul_assoc]

/-- **P4 (d).** The reduced loss matrix is positive semidefinite. -/
theorem reduced_loss_psd (Rb : Matrix br br ℝ) (hR : Rb.PosSemidef) (N : Matrix br k ℝ) :
    (Nᵀ * Rb * N).PosSemidef := by
  have h := hR.conjTranspose_mul_mul_same N
  rwa [conjTranspose_eq_transpose_of_trivial] at h

/-- **P4 (e).** The sources deliver the same power to the reduced system. -/
theorem reduced_port_power (N : Matrix br k ℝ) (E : br → ℝ) (η : k → ℝ) :
    E ⬝ᵥ (N *ᵥ η) = (Nᵀ *ᵥ E) ⬝ᵥ η := by
  rw [dotProduct_mulVec, ← mulVec_transpose]

/-- **P4 (a) on the whole constraint set.** If `range N = ker A` (both inclusions: every constrained
current is some `N η`, and every `N η` is constrained), every constrained current has the same energy
in the reduced coordinates. -/
theorem reduced_energy_constrained (L : Matrix br br ℝ) (hL : L.PosDef) (A : Matrix nd br ℝ)
    (N : Matrix br k ℝ) (hN : Function.Injective N.mulVec)
    (hrange : ∀ i, A *ᵥ i = 0 ↔ ∃ η, i = N *ᵥ η) (i : br → ℝ) (hi : A *ᵥ i = 0) :
    ∃ η, i = N *ᵥ η ∧
      (Nᵀ *ᵥ (L *ᵥ i)) ⬝ᵥ ((Nᵀ * L * N)⁻¹ *ᵥ (Nᵀ *ᵥ (L *ᵥ i))) = i ⬝ᵥ (L *ᵥ i) := by
  obtain ⟨η, rfl⟩ := (hrange i).mp hi
  exact ⟨η, rfl, reduced_energy L hL N hN η⟩

end PHPSFormal.PH

#print axioms PHPSFormal.PH.reduced_energy
#print axioms PHPSFormal.PH.voltage_drops_out
#print axioms PHPSFormal.PH.reduced_rates
#print axioms PHPSFormal.PH.reduced_loss_psd
#print axioms PHPSFormal.PH.reduced_port_power
#print axioms PHPSFormal.PH.reduced_energy_constrained
