import Mathlib.Tactic.Linarith
import Mathlib.Basic.Real.Basic

/-- Smoke test: mathlib is available and a small real-analysis fact checks. -/
theorem smoke (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by
  nlinarith [sq_nonneg (a - b)]

#print axioms smoke
