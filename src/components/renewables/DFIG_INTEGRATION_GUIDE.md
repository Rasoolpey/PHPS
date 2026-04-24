# DFIG Integration Guide
## Port-Hamiltonian Doubly-Fed Induction Generator

---

## 1. File placement

```
src/components/generators/dfig.py
```

---

## 2. Register in `src/json_compat.py`

Add one entry to `COMPONENT_REGISTRY`:

```python
from src.components.generators.dfig import Dfig

COMPONENT_REGISTRY = {
    ...
    'DFIG': Dfig,
    ...
}
```

---

## 3. JSON system.json entry

### Minimal DFIG (no external rotor voltage — Vrd/Vrq held at constant)

```json
{
  "type": "DFIG",
  "id":   "DFIG_1",
  "bus":  1,
  "params": {
    "Ls":        0.00493,
    "Lr":        0.00490,
    "Lm":        0.00480,
    "Rs":        0.000005,
    "Rr":        0.000005,
    "j_inertia": 0.04,
    "f_damp":    0.0001,
    "np":        2,
    "omega_b":   314.159,
    "omega_s":   1.0,
    "D":         0.0
  }
}
```

> **Parameter note:** The Song & Qu (2011) 2 MW machine uses SI values
> (Ls = 4.93e-3 H, etc.).  Convert to per-unit before passing here using
> your system base (S_base = 2 MVA, V_base = 690 V → Z_base = V²/S = 0.238 Ω,
> L_base = Z_base / ω_b = 0.238 / 314 = 7.58e-4 H).

### Per-unit parameters for the Song & Qu 2 MW example

```python
# Base values: S_base = 2e6 VA, V_base = 690 V, ω_b = 314.16 rad/s
S_base = 2e6
V_base = 690.0
omega_b = 314.159
Z_base = V_base**2 / S_base        # = 0.23805 Ω
L_base = Z_base / omega_b           # = 7.578e-4 H

Ls_pu  = 4.93e-3 / L_base   # ≈ 6.506 pu
Lr_pu  = 4.90e-3 / L_base   # ≈ 6.467 pu
Lm_pu  = 4.80e-3 / L_base   # ≈ 6.335 pu
Rs_pu  = 0.005   / Z_base   # ≈ 0.0210 pu
Rr_pu  = 0.005   / Z_base   # ≈ 0.0210 pu
j_pu   = 120.0 * omega_b**2 / S_base  # inertia in pu: J·ω_b² / S_base ≈ 5.87 pu·s
```

---

## 4. Required connections (system.json "connections" section)

```json
"connections": [
  { "from": "BUS_1.Vd",     "to": "DFIG_1.Vd"  },
  { "from": "BUS_1.Vq",     "to": "DFIG_1.Vq"  },
  { "from": "CONST:0.5",    "to": "DFIG_1.Tm"  },
  { "from": "CONST:0.0",    "to": "DFIG_1.Vrd" },
  { "from": "CONST:0.0",    "to": "DFIG_1.Vrq" }
]
```

> **MSC wiring:** When a machine-side converter (MSC) model is available,
> replace `CONST:0.0` for `Vrd`/`Vrq` with the MSC output:
> ```json
> { "from": "MSC_1.Vrd",  "to": "DFIG_1.Vrd" },
> { "from": "MSC_1.Vrq",  "to": "DFIG_1.Vrq" }
> ```

> **Drive-train wiring:** If a WTDTA1 drive-train is present, wire its `wg`
> speed output to the DFIG mechanical torque (via a torque controller):
> ```json
> { "from": "WTTQA1_1.Pref", "to": "DFIG_1.Tm" }
> ```

---

## 5. Full DFIG + WTDTA1 + WTARA1 wiring example

```json
{
  "components": [
    {
      "type": "DFIG",  "id": "DFIG_1",  "bus": 2,
      "params": { "Ls": 6.51, "Lr": 6.47, "Lm": 6.33,
                  "Rs": 0.021, "Rr": 0.021,
                  "j_inertia": 5.87, "f_damp": 0.00004,
                  "np": 2, "omega_b": 314.159,
                  "omega_s": 1.0, "D": 0.0 }
    },
    {
      "type": "WTDTA1", "id": "WTDTA1_1",
      "params": { "H": 4.0, "D": 1.5, "Kshaft": 0.3, "ree": "DFIG_1" }
    },
    {
      "type": "WTARA1", "id": "WTARA1_1",
      "params": { "Ka": 0.007, "theta_min": 0.0, "theta_max": 1.57,
                  "rego": "DFIG_1" }
    }
  ],

  "connections": [
    { "from": "BUS_2.Vd",           "to": "DFIG_1.Vd"   },
    { "from": "BUS_2.Vq",           "to": "DFIG_1.Vq"   },
    { "from": "WTDTA1_1.wt",        "to": "DFIG_1.Tm"   },
    { "from": "CONST:0.0",          "to": "DFIG_1.Vrd"  },
    { "from": "CONST:0.0",          "to": "DFIG_1.Vrq"  },
    { "from": "WTARA1_1.Pm",        "to": "WTDTA1_1.Pm" },
    { "from": "DFIG_1.Pe",          "to": "WTDTA1_1.Pe" }
  ]
}
```

---

## 6. Observables (auto-available in plot definitions)

| Name        | Description                      | Unit |
|-------------|----------------------------------|------|
| `Pe`        | Active power output              | pu   |
| `Qe`        | Reactive power output            | pu   |
| `omega_pu`  | Rotor mechanical speed           | pu   |
| `slip`      | Slip `(ω_s - ω_m) / ω_s`        | -    |
| `phi_sd`    | Stator d-axis flux linkage       | pu   |
| `phi_sq`    | Stator q-axis flux linkage       | pu   |
| `phi_rd`    | Rotor  d-axis flux linkage       | pu   |
| `phi_rq`    | Rotor  q-axis flux linkage       | pu   |
| `Te_elec`   | Electromagnetic torque           | pu   |
| `V_term`    | Terminal voltage magnitude       | pu   |

---

## 7. Physical model summary (Port-Hamiltonian form)

### Energy variables (states)

```
x = [φ_sd, φ_sq, φ_rd, φ_rq, p_mech]^T
```

### Hamiltonian (total stored electromagnetic + kinetic energy)

```
H_em = ½ x^T M^{-1} x

M = diag(Ls, Ls, Lr, Lr, j_inertia)

∇H_em = M^{-1} x = [i_sd, i_sq, i_rd, i_rq, ω_m]^T
```

### PCH dynamics

```
ẋ = (J_em - R_em) ∇H_em  +  g1·[Vsd, Vsq]^T  +  g2·[Vrd, Vrq]^T  +  g_em·[0, Tm]^T
```

Structure (J_em) and dissipation (R_em):

```
J_em =
⎡  0      ωs·Ls    0       ωs·Lm    0       ⎤
⎢-ωs·Ls   0      -ωs·Lm   0        0       ⎥
⎢  0      ωs·Lm   0       ωs·Lr   -np·φ_rq ⎥
⎢-ωs·Lm   0      -ωs·Lr   0        np·φ_rd ⎥
⎣  0      0       np·φ_rq -np·φ_rd  0       ⎦

R_em = diag(Rs, Rs, Rr, Rr, f_damp)
```

Electromagnetic torque:
```
Te = np · (φ_rd · i_rq - φ_rq · i_rd)
```

Passivity: The system satisfies `dH/dt ≤ u^T y` (power balance inequality)
since J_em is skew-symmetric and R_em ≥ 0 — guaranteed by construction.

---

## 8. Key design decisions

| Decision | Rationale |
|----------|-----------|
| **5 flux states** (not voltage-behind-reactance) | Exact PCH formulation from Song & Qu preserves passivity certificate; no approximation |
| **RI-frame = stator-frame** | DFIG stator is directly grid-connected; no separate Park rotation needed at stator terminals (consistent with Type-3 wind topology) |
| **Norton admittance = stator short-circuit** | `Y_N = 1/(Rs + j·Xs_sigma)`, where `Xs_sigma = (Ls·Lr - Lm²)/Lr` — exact Thevenin equivalent of the stator winding seen from the grid |
| **Vrd/Vrq as explicit inputs** | Keeps the DFIG a standalone `generator`-role component; MSC controller plugs in via JSON wiring without modifying the DFIG class |
| **`contributes_norton_admittance = True`** | Causes the compiler to add `Y_N` to the Y-bus before Kron reduction, exactly like GENROU |
| **`refine_d_axis` is a no-op** | DFIG has no AVR field voltage; Efd concept doesn't apply |
