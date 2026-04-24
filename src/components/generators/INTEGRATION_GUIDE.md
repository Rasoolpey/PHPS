# Integration Guide: GENSAL, GENTPF, GENTPJ

## Step 1 — Place the files

Copy the three new files into your generators folder:

    src/components/generators/gensal.py   ← GENSAL (salient-pole, 5th-order)
    src/components/generators/gentpf.py   ← GENTPF (round-rotor, multiplicative sat.)
    src/components/generators/gentpj.py   ← GENTPJ (GENTPF + Kis armature current sat.)

## Step 2 — Register in json_compat.py

In `_build_registry()`, add the three imports and registry entries:

```python
from src.components.generators.gensal import GenSal
from src.components.generators.gentpf import GenTpf
from src.components.generators.gentpj import GenTpj

# inside the returned dict:
"GENSAL": GenSal,
"GenSal": GenSal,
"GENTPF": GenTpf,
"GenTpf": GenTpf,
"GENTPJ": GenTpj,
"GenTpj": GenTpj,
```

## Step 3 — Parameter schemas

### GENSAL parameters (same as GENROU minus xq_prime + saturation)
| Parameter         | Description                                  |
|-------------------|----------------------------------------------|
| H                 | Inertia constant                             |
| D                 | Damping coefficient                          |
| ra                | Stator resistance                            |
| xd                | d-axis synchronous reactance                 |
| xq                | q-axis synchronous reactance                 |
| xd_prime          | d-axis transient reactance                   |
| xd_double_prime   | d-axis sub-transient reactance               |
| xq_double_prime   | q-axis sub-transient reactance (= xd'' for GENSAL) |
| Td0_prime         | d-axis transient OC time constant            |
| Td0_double_prime  | d-axis sub-transient OC time constant        |
| Tq0_double_prime  | q-axis sub-transient OC time constant        |
| xl                | Leakage reactance                            |
| S10               | Saturation at 1.0 pu flux                    |
| S12               | Saturation at 1.2 pu flux                    |
| omega_b           | Base frequency [rad/s] (e.g. 2*pi*60)        |

**Note:** A_sat and B_sat are computed internally from S10/S12 in `gensal.py`.
For the json_compat parameter normalisation, add a helper similar to
`_normalise_genrou_params` that also converts S10/S12 → A_sat/B_sat.

### GENTPF parameters
Same as GENROU, but replace S10/S12 with A_sat and B_sat (pre-computed), or
add normalisation in json_compat. Does NOT have xq_prime=xq in GENSAL sense —
full Xq' and Xd' are used.

| Parameter         | Description                                  |
|-------------------|----------------------------------------------|
| (all GENROU params except xl)                               |
| xl                | Leakage reactance                            |
| A_sat             | Sat coefficient A (computed from S10, S12)   |
| B_sat             | Sat coefficient B (computed from S10, S12)   |
| omega_b           | Base frequency [rad/s]                       |

### GENTPJ parameters
Identical to GENTPF plus:

| Parameter | Description                                               |
|-----------|-----------------------------------------------------------|
| Kis       | Armature current saturation factor (set 0 → = GENTPF)    |

## Step 4 — json_compat normalisation helper

Add this function alongside `_normalise_genrou_params`:

```python
def _normalise_gentpf_params(params, mva_base, fn=60.0):
    """Normalise GENTPF/GENTPJ params (same as GENROU + sat conversion)."""
    out = _normalise_genrou_params(params, mva_base, fn)
    # Convert S10/S12 → A_sat/B_sat if not already supplied
    S10 = float(out.get('S10', 0.0))
    S12 = float(out.get('S12', 0.0))
    if 'A_sat' not in out or 'B_sat' not in out:
        if S10 > 0 and S12 > 0:
            import math
            ratio = math.sqrt(1.2 * S12 / S10)
            if abs(ratio - 1.0) > 1e-10:
                u = 0.2 / (ratio - 1.0)
                out['A_sat'] = 1.0 - u
                out['B_sat'] = S10 / (u * u)
            else:
                out['A_sat'] = 0.0
                out['B_sat'] = 0.0
        else:
            out.setdefault('A_sat', 0.0)
            out.setdefault('B_sat', 0.0)
    out.setdefault('Kis', 0.0)   # only used by GENTPJ
    return out

def _normalise_gensal_params(params, mva_base, fn=60.0):
    """Normalise GENSAL params."""
    # GENSAL has no xq_prime — if supplied it is ignored
    out = _normalise_gentpf_params(params, mva_base, fn)
    # For GENSAL, xq'' = xd'' is the standard assumption
    out.setdefault('xq_double_prime', out.get('xd_double_prime', 0.2))
    return out
```

## Step 5 — Wiring in JSON (same as GENROU)

The port schema is identical to GENROU, so existing wiring templates work:

```json
{
    "type": "GENTPJ",
    "name": "GENTPJ_1",
    "bus": 1,
    "params": {
        "H": 3.5, "D": 0.0, "ra": 0.003,
        "xd": 1.81, "xq": 1.76,
        "xd_prime": 0.3, "xq_prime": 0.65,
        "xd_double_prime": 0.23, "xq_double_prime": 0.25,
        "Td0_prime": 8.0, "Tq0_prime": 1.0,
        "Td0_double_prime": 0.03, "Tq0_double_prime": 0.07,
        "xl": 0.15, "S10": 0.05, "S12": 0.3,
        "Kis": 0.1,
        "omega_b": 376.99
    }
}
```

## Model Comparison Summary

| Feature                        | GENROU  | GENSAL  | GENTPF  | GENTPJ  |
|-------------------------------|---------|---------|---------|---------|
| Machine type                  | Round   | Salient | Round   | Round   |
| Order                         | 6th     | 5th     | 6th     | 6th     |
| q-axis transient winding      | Yes     | No      | Yes     | Yes     |
| Saturation type               | None    | Additive| Multiplicative | Multiplicative |
| Saturation scope              | —       | dE'q/dt | All flux terms | All flux terms |
| Armature current sat. (Kis)   | No      | No      | No      | Yes     |
| Circuit model for network I/F | Yes     | Yes     | No*     | No*     |
| Transient saliency (Xd''≠Xq'')| Yes    | Yes**   | Yes     | Yes     |

*GENTPF/J use direct Vdterm/Vqterm equations since sat. Xd''≠sat. Xq''
**In GENSAL, Xd''=Xq'' is the standard assumption (subtransient saliency neglected)
