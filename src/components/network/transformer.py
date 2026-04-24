"""
Port-Hamiltonian Two-Winding Transformer Model.

The two-winding transformer is modelled as an ideal transformer
(turns ratio ``tap:1`` with optional phase shift ``phi``) in series with
a leakage reactance ``x_T`` and winding resistance ``r_T``.

The magnetising admittance (iron-core losses + no-load reactance) is
represented as an optional shunt at terminal 1 (high-voltage side).

Port-Hamiltonian Formulation
-----------------------------
Energy storage (Hamiltonian):
    H = (x_T / 2ω_b)(IL_d² + IL_q²)
      + (G_m / 2ω_b·G_m²)(Vm_d² + Vm_q²)   [magnetising branch, if active]

States x = [IL_d, IL_q]          (simplified — no mag branch states)
       x = [IL_d, IL_q, Im_d, Im_q]   (full model with magnetising branch)

  IL_d, IL_q : leakage-inductance current (RI network frame, per-unit)
  Im_d, Im_q : magnetising branch current (RI frame, per-unit) [optional]

Dynamics
--------
    dIL_d/dt = (ω_b/x_T)(V1_eff_d − V2_d − r_T·IL_d) + IL_q
    dIL_q/dt = (ω_b/x_T)(V1_eff_q − V2_q − r_T·IL_q) − IL_d

    V1_eff = V1 / (tap · e^{jφ})   [referred to secondary side]

Parameters
----------
    r_T     : Winding resistance [pu]
    x_T     : Leakage reactance  [pu]
    tap     : Off-nominal turns ratio [pu]  (1.0 = nominal)
    phi     : Phase shift angle  [rad]
    G_m     : Magnetising conductance (core loss)  [pu]  (0 = lossless)
    B_m     : Magnetising susceptance (no-load reactance)  [pu]  (0 = no shunt)
    omega_b : Base angular frequency [rad/s]
    bus1    : Primary (HV) bus index
    bus2    : Secondary (LV) bus index
    mva_base: Component MVA rating (for per-unit conversion)

Note
----
When used in the UI canvas, the transformer is exported to the ``"Line"``
JSON section of the system file, where the simulation kernel treats it as
a complex admittance branch.

For dynamic studies requiring explicit leakage flux transients (e.g.,
geomagnetically induced current analysis), set ``dynamic=true`` in the
component parameters to force export to the ``"components"`` section.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class Transformer2W(PowerComponent):
    """
    Port-Hamiltonian two-winding transformer (2-state leakage model,
    optional 4-state full model with magnetising branch).

    Ports
    -----
    Inputs  : V1_d, V1_q — primary-side voltage   (RI network frame) [pu]
              V2_d, V2_q — secondary-side voltage  (RI network frame) [pu]
    Outputs : I1_d, I1_q — current injection into primary bus   [pu]
              I2_d, I2_q — current injection into secondary bus  [pu]
              IL_d, IL_q — leakage branch current  [pu]
              Vterm1     — primary voltage magnitude   [pu]
              Vterm2     — secondary voltage magnitude [pu]
    """

    @property
    def component_role(self) -> str:
        return 'transformer'

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('V1_d', 'effort', 'pu'),   # Primary voltage  (d-axis, RI frame)
                ('V1_q', 'effort', 'pu'),   # Primary voltage  (q-axis, RI frame)
                ('V2_d', 'effort', 'pu'),   # Secondary voltage (d-axis, RI frame)
                ('V2_q', 'effort', 'pu'),   # Secondary voltage (q-axis, RI frame)
            ],
            'out': [
                ('I1_d',   'flow',   'pu'),  # Current injected into primary bus (d-axis)
                ('I1_q',   'flow',   'pu'),  # Current injected into primary bus (q-axis)
                ('I2_d',   'flow',   'pu'),  # Current injected into secondary bus (d-axis)
                ('I2_q',   'flow',   'pu'),  # Current injected into secondary bus (q-axis)
                ('IL_d',   'flow',   'pu'),  # Leakage branch current (d-axis)
                ('IL_q',   'flow',   'pu'),  # Leakage branch current (q-axis)
                ('Vterm1', 'effort', 'pu'),  # Primary voltage magnitude
                ('Vterm2', 'effort', 'pu'),  # Secondary voltage magnitude
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['IL_d', 'IL_q', 'Im_d', 'Im_q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'r_T':    'Winding resistance [pu]',
            'x_T':    'Leakage reactance [pu]',
            'tap':    'Off-nominal turns ratio [pu] (1.0 = nominal)',
            'phi':    'Phase shift angle [rad] (0 = no phase shift)',
            'G_m':    'Magnetising conductance (core loss) [pu]',
            'B_m':    'Magnetising susceptance (no-load reactance) [pu]',
            'omega_b':'Base angular frequency [rad/s]',
            'bus1':   'Primary (HV) bus index',
            'bus2':   'Secondary (LV) bus index',
            'mva_base':'Component MVA rating [MVA]',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'IL_mag': {
                'description': 'Leakage branch current magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(x[0]*x[0] + x[1]*x[1])'
            },
            'Im_mag': {
                'description': 'Magnetising current magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(x[2]*x[2] + x[3]*x[3])'
            },
            'P1': {
                'description': 'Active power flow at primary terminal',
                'unit': 'pu',
                'cpp_expr': 'inputs[0]*outputs[0] + inputs[1]*outputs[1]'
            },
            'Q1': {
                'description': 'Reactive power at primary terminal',
                'unit': 'pu',
                'cpp_expr': 'inputs[1]*outputs[0] - inputs[0]*outputs[1]'
            },
            'P2': {
                'description': 'Active power flow at secondary terminal',
                'unit': 'pu',
                'cpp_expr': 'inputs[2]*outputs[2] + inputs[3]*outputs[3]'
            },
            'P_loss': {
                'description': 'Winding copper loss',
                'unit': 'pu',
                'cpp_expr': 'r_T * (x[0]*x[0] + x[1]*x[1])'
            },
            'tap_ratio': {
                'description': 'Effective turns ratio',
                'unit': 'pu',
                'cpp_expr': 'tap'
            },
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // Transformer leakage and magnetising branch currents
            double IL_d = x[0];
            double IL_q = x[1];
            double Im_d = x[2];
            double Im_q = x[3];

            double V1_d = inputs[0];
            double V1_q = inputs[1];

            // Effective tap + phase on primary side
            double cos_phi = cos(phi);
            double sin_phi = sin(phi);
            double tap_inv = (tap > 1e-6) ? 1.0 / tap : 1.0;

            // Total current injected into primary bus (referred to primary)
            // = tap_inv * (leakage current referred back) + magnetising current
            double I1_tot_d = tap_inv * (cos_phi * IL_d + sin_phi * IL_q) + Im_d;
            double I1_tot_q = tap_inv * (-sin_phi * IL_d + cos_phi * IL_q) + Im_q;

            outputs[0] = -I1_tot_d;   // I1_d  (positive = into bus)
            outputs[1] = -I1_tot_q;   // I1_q
            outputs[2] =  IL_d;       // I2_d  (current out of secondary, into bus)
            outputs[3] =  IL_q;       // I2_q
            outputs[4] =  IL_d;       // IL_d
            outputs[5] =  IL_q;       // IL_q

            outputs[6] = sqrt(V1_d*V1_d + V1_q*V1_q);     // Vterm1
            outputs[7] = sqrt(inputs[2]*inputs[2] + inputs[3]*inputs[3]);  // Vterm2
        """

    def get_cpp_step_code(self) -> str:
        return """
            // -------------------------------------------------------
            // Port-Hamiltonian 2-winding transformer dynamics
            // -------------------------------------------------------
            double V1_d = inputs[0];
            double V1_q = inputs[1];
            double V2_d = inputs[2];
            double V2_q = inputs[3];

            double IL_d = x[0];
            double IL_q = x[1];
            double Im_d = x[2];
            double Im_q = x[3];

            // Compute effective primary voltage referred to secondary
            double cos_phi = cos(phi);
            double sin_phi = sin(phi);
            double tap_inv = (tap > 1e-6) ? 1.0 / tap : 1.0;
            double V1_eff_d = tap_inv * (V1_d * cos_phi - V1_q * sin_phi);
            double V1_eff_q = tap_inv * (V1_d * sin_phi + V1_q * cos_phi);

            // --- Leakage inductance dynamics (PH series RL) ---
            // dIL_d/dt = (omega_b/x_T)(V1_eff_d - V2_d - r_T*IL_d) + IL_q
            // dIL_q/dt = (omega_b/x_T)(V1_eff_q - V2_q - r_T*IL_q) - IL_d
            double inv_xT = omega_b / x_T;
            dxdt[0] = inv_xT * (V1_eff_d - V2_d - r_T * IL_d) + IL_q;
            dxdt[1] = inv_xT * (V1_eff_q - V2_q - r_T * IL_q) - IL_d;

            // --- Magnetising branch dynamics (PH shunt RL/RC) ---
            // Magnetising admittance: Y_m = G_m + j*B_m  (referred to primary)
            // dIm/dt from mutual flux linkage dynamics:
            // If B_m > 0:  dIm_d/dt = omega_b*(V1_d*G_m - Im_d) + Im_q
            //              dIm_q/dt = omega_b*(V1_d*B_m - Im_q) - Im_d
            // Simple model: Im is purely algebraic if B_m is small (set to 0)
            // Dynamic magnetising current using shunt inductor formulation:
            double Bm_nz = (B_m > 1e-9) ? B_m : 1e-9;   // avoid division by zero
            dxdt[2] = (omega_b / Bm_nz) * (G_m * V1_d - Im_d) + Im_q;
            dxdt[3] = (omega_b / Bm_nz) * (G_m * V1_q - Im_q) - Im_d;
        """

    # ------------------------------------------------------------------
    # Initialization contract
    # ------------------------------------------------------------------

    def init_from_phasor(self, V1: complex, V2: complex) -> Tuple[np.ndarray, Dict]:
        """Steady-state initialization from known terminal phasors."""
        p = self.params
        r_T = p.get('r_T', 0.0)
        x_T = p.get('x_T', 0.1)
        tap = p.get('tap', 1.0)
        phi = p.get('phi', 0.0)
        G_m = p.get('G_m', 0.0)
        B_m = p.get('B_m', 0.0)

        # Effective primary voltage referred to secondary
        V1_eff = V1 / (tap * np.exp(1j * phi))

        # Leakage branch current
        Z_T = complex(r_T, x_T)
        IL = (V1_eff - V2) / Z_T if abs(Z_T) > 1e-9 else 0j

        # Magnetising current
        Y_m = complex(G_m, B_m)
        Im = V1 * Y_m

        x0 = np.array([IL.real, IL.imag, Im.real, Im.imag])
        return x0, {}

    def to_json_line_entry(self) -> Dict[str, Any]:
        """Convert this component to a 'Line' JSON section entry (static model)."""
        p = self.params
        return {
            'idx':  self.name,
            'bus1': p.get('bus1'),
            'bus2': p.get('bus2'),
            'r':    p.get('r_T', 0.0),
            'x':    p.get('x_T', 0.1),
            'b':    0.0,
            'tap':  p.get('tap', 1.0),
            'phi':  p.get('phi', 0.0),
        }
