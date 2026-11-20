import PHPSFormal.PH.Balance

/-!
# P9 (Proposition 4): a group of states cut out of the assembled model; P3: composition

Source: PHPS paper draft, §V-B, eq. (8) and Proposition 4; §II-B (composition). Claim record:
`formal/claims/P3_P9.md`.

The states split into the group `s` and the rest `r`; the assembled `J` and `R` are block matrices
`fromBlocks Jss Jsr Jrs Jrr` and `fromBlocks Rss Rsr Rrs Rrr`. The group's rows are
`ẋ_s = (Jss - Rss) e_s + (Jsr - Rsr) e_r + G_s u` (eq. (8) without the steam feedthrough). The
group's interconnection ports are `(e_r, y_p)` with `y_p = Gpᵀ e_s`, `Gp = Jsr - Rsr`.

* `skew_blocks`: the skew symmetry of the assembled `J` gives `Jssᵀ = -Jss` and `Jrs = -Jsrᵀ`.
* `group_balance`: with `Jss` skew, `e_sᵀ ẋ_s = y_pᵀ e_r - e_sᵀ Rss e_s + e_sᵀ G_s u`.
* `ports_conserve`: if moreover `Rsr = 0` and `Rrs = 0` (no shared loss), the power the rest
  receives from the group through `Jrs`, `e_rᵀ (Jrs e_s)`, is exactly minus the power the group
  receives through its ports, `y_pᵀ e_r`.
* `composition`: devices that each satisfy their balance, joined so that the internal port powers
  sum to zero, satisfy the balance of the whole with the external supply.

Energy separation `H = H_s(x_s) + H_r(x_r)` is what makes `e_s` the gradient of `H_s` alone, so
that `e_sᵀ ẋ_s` is `dH_s/dt` (chain rule, `energy_rate`); it is a hypothesis on `H` and is not
used by the algebra below.
-/

namespace PHPSFormal.PH

open Matrix

variable {s r m : Type*} [Fintype s] [Fintype r] [Fintype m]

/-- The block reading of the skew symmetry of the assembled `J`. -/
theorem skew_blocks (Jss : Matrix s s ℝ) (Jsr : Matrix s r ℝ) (Jrs : Matrix r s ℝ)
    (Jrr : Matrix r r ℝ) (hJ : (fromBlocks Jss Jsr Jrs Jrr)ᵀ = -fromBlocks Jss Jsr Jrs Jrr) :
    Jssᵀ = -Jss ∧ Jrs = -Jsrᵀ ∧ Jrrᵀ = -Jrr := by
  rw [fromBlocks_transpose, fromBlocks_neg] at hJ
  obtain ⟨h1, h2, h3, h4⟩ := fromBlocks_inj.mp hJ
  refine ⟨h1, ?_, h4⟩
  rw [h3, neg_neg]

/-- **Proposition 4, the group's balance.** -/
theorem group_balance (Jss Rss : Matrix s s ℝ) (Jsr Rsr : Matrix s r ℝ) (Gs : Matrix s m ℝ)
    (hJ : Jssᵀ = -Jss) (es : s → ℝ) (er : r → ℝ) (u : m → ℝ) :
    es ⬝ᵥ ((Jss - Rss) *ᵥ es + (Jsr - Rsr) *ᵥ er + Gs *ᵥ u) =
      ((Jsr - Rsr)ᵀ *ᵥ es) ⬝ᵥ er - es ⬝ᵥ (Rss *ᵥ es) + es ⬝ᵥ (Gs *ᵥ u) := by
  have hskew := skew_quadratic_zero Jss hJ es
  have hp := dotProduct_transpose_mulVec (Jsr - Rsr) es er
  rw [dotProduct_comm, sub_mulVec, dotProduct_sub] at hp
  simp only [dotProduct_add, sub_mulVec, dotProduct_sub]
  linarith

/-- **Proposition 4, the ports conserve power.** With no shared loss, what the rest receives from
the group is minus what the group receives through its ports. -/
theorem ports_conserve (Jsr : Matrix s r ℝ) (Jrs : Matrix r s ℝ) (hJ : Jrs = -Jsrᵀ)
    (es : s → ℝ) (er : r → ℝ) :
    er ⬝ᵥ (Jrs *ᵥ es) = -((Jsrᵀ *ᵥ es) ⬝ᵥ er) := by
  rw [hJ, neg_mulVec, dotProduct_neg, dotProduct_comm]

/-- **P3, composition.** Devices `c` with stored-power `pc c`, port power `yu c` and loss `dc c`
each satisfy `pc c = yu c - dc c`. If the port powers split into internal and external parts and
the internal parts sum to zero (the interconnection is power-conserving, `power_zero`), the whole
satisfies `∑ pc = ∑ external - ∑ dc`. -/
theorem composition {ι : Type*} [Fintype ι] (pc yu dc yint yext : ι → ℝ)
    (hdev : ∀ c, pc c = yu c - dc c) (hsplit : ∀ c, yu c = yint c + yext c)
    (hint : ∑ c, yint c = 0) :
    ∑ c, pc c = ∑ c, yext c - ∑ c, dc c := by
  simp only [hdev, hsplit, Finset.sum_sub_distrib, Finset.sum_add_distrib, hint, zero_add]


/-- **Proposition 4 with the steam feedthrough.** The group's rows
`ẋ_s = (J_ss - R_ss) e_s + (Gp e_r + G_s u) - P_s v_f`, with the feedthrough output
`y_f = P_sᵀ e_s + S_f v_f`, are the form (1) with the inputs `(e_r, u, v_f)` stacked into one
input `w`, the input matrix `G = [Gp, G_s, 0]` and the feedthrough `P = [0, 0, P_s]`,
`S = diag(0, 0, S_f)`: their balance is `power_balance`, and with
`[[R_ss, P], [Pᵀ, S]] ⪰ 0` the group is passive at its ports (`power_le_supply`). -/
theorem group_balance_feedthrough {q : Type*} [Fintype q] (Jss Rss : Matrix s s ℝ)
    (G P : Matrix s q ℝ) (S : Matrix q q ℝ) (hJ : Jssᵀ = -Jss) (es : s → ℝ) (w : q → ℝ) :
    es ⬝ᵥ rate Jss Rss G P es w = output G P S es w ⬝ᵥ w - loss Rss P S es w :=
  power_balance Jss Rss G P S hJ es w

end PHPSFormal.PH

#print axioms PHPSFormal.PH.skew_blocks
#print axioms PHPSFormal.PH.group_balance
#print axioms PHPSFormal.PH.ports_conserve
#print axioms PHPSFormal.PH.composition
#print axioms PHPSFormal.PH.group_balance_feedthrough
