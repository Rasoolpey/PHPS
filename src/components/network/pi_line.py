"""
Port-Hamiltonian π-Section Transmission Line Model.

The π-section transmission line is modelled as a series RL branch
connected to two buses, with lumped shunt capacitors (B/2) at each end.

Port-Hamiltonian Formulation
-----------------------------
Energy storage (Hamiltonian):
    H = (x_L / 2ω_b)(IL_d² + IL_q²)
      + (1 / b_s ω_b)(Vc1_d² + Vc1_q²)     ... shunt capacitor at terminal 1
      + (1 / b_s ω_b)(Vc2_d² + Vc2_q²)     ... shunt capacitor at terminal 2

where:
    x_L  — series reactance [pu]
    b_s  — total shunt susceptance = b (split b/2 at each end) [pu]
    ω_b  — synchronous base frequency [rad/s]

States x = [IL_d, IL_q, Vc1_d, Vc1_q, Vc2_d, Vc2_q]

  IL_d, IL_q  : series inductor current (d- and q-axis, RI network frame)
  Vc1_d, Vc1_q: shunt capacitor voltage at terminal 1 (RI frame)
  Vc2_d, Vc2_q: shunt capacitor voltage at terminal 2 (RI frame)

Dynamics (PH structure):
    ẋ = [J - R] ∂H/∂x + [B] u

    dIL_d/dt = (ω_b/x)(V1_d - Vc1_d - V2_d + Vc2_d - r·IL_d)
             + IL_q          ... ω₀ cross-coupling
    dIL_q/dt = (ω_b/x)(V1_q - Vc1_q - V2_q + Vc2_q - r·IL_q)
             - IL_d
    dVc1_d/dt = ω_b·(b/2)·(V1_d - Vc1_d) - ω₀·Vc1_q
    dVc1_q/dt = ω_b·(b/2)·(V1_q - Vc1_q) + ω₀·Vc1_d
    dVc2_d/dt = ω_b·(b/2)·(V2_d - Vc2_d) - ω₀·Vc2_q
    dVc2_q/dt = ω_b·(b/2)·(V2_q - Vc2_q) + ω₀·Vc2_d

Note
----
In static (power-flow) contexts, the shunt capacitors are replaced by
an algebraic admittance jB/2 at each terminal bus.  This dynamic model
is used when high-frequency transients or sub-synchronous resonance
studies require explicit L-C line dynamics.

For normal transient-stability studies, use the ``Line`` section of the
system JSON (Ybus-based static handling) instead.

When the PiLine component appears in the UI canvas:
  • its parameters are exported to the ``"Line"`` section of the system JSON
  • the ``bus1`` and ``bus2`` parameters link it to two bus nodes
  • the dynamic version can optionally be exported as a full ``components``
    entry for sub-synchronous studies (set ``dynamic=true`` in params)
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class PiLine(PowerComponent):
    """
    Port-Hamiltonian π-section transmission line (6-state dynamic model).

    Ports
    -----
    Inputs  : V1_d, V1_q — terminal-1 voltage (RI network frame) [pu]
              V2_d, V2_q — terminal-2 voltage (RI network frame) [pu]
    Outputs : I1_d, I1_q — current injection into bus 1 (Norton, RI frame) [pu]
              I2_d, I2_q — current injection into bus 2 (Norton, RI frame) [pu]
              IL_d, IL_q — series branch current (RI frame) [pu]
              Vterm1     — terminal-1 voltage magnitude [pu]
              Vterm2     — terminal-2 voltage magnitude [pu]

    Parameters
    ----------
    r       : Series resistance [pu]
    x       : Series reactance  [pu]
    b       : Total shunt charging susceptance [pu]  (B/2 at each end)
    tap     : Off-nominal turns ratio (1.0 = nominal) [pu]
    phi     : Phase shift angle [rad]  (0 for non-phase-shifting)
    omega_b : Base angular frequency [rad/s]  (2π·50 or 2π·60)
    bus1    : Terminal-1 bus index
    bus2    : Terminal-2 bus index
    """

    @property
    def component_role(self) -> str:
        return 'line'

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('V1_d', 'effort', 'pu'),   # Terminal-1 real-axis voltage
                ('V1_q', 'effort', 'pu'),   # Terminal-1 imag-axis voltage
                ('V2_d', 'effort', 'pu'),   # Terminal-2 real-axis voltage
                ('V2_q', 'effort', 'pu'),   # Terminal-2 imag-axis voltage
            ],
            'out': [
                ('I1_d',   'flow', 'pu'),   # Current injection into bus 1 (d-axis)
                ('I1_q',   'flow', 'pu'),   # Current injection into bus 1 (q-axis)
                ('I2_d',   'flow', 'pu'),   # Current injection into bus 2 (d-axis)
                ('I2_q',   'flow', 'pu'),   # Current injection into bus 2 (q-axis)
                ('IL_d',   'flow', 'pu'),   # Series branch current (d-axis)
                ('IL_q',   'flow', 'pu'),   # Series branch current (q-axis)
                ('Vterm1', 'effort', 'pu'), # Terminal-1 voltage magnitude
                ('Vterm2', 'effort', 'pu'), # Terminal-2 voltage magnitude
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        # 6 PH states: series inductor current + two shunt capacitor voltages
        return ['IL_d', 'IL_q', 'Vc1_d', 'Vc1_q', 'Vc2_d', 'Vc2_q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'r':       'Series resistance [pu]',
            'x':       'Series reactance [pu]',
            'b':       'Total shunt charging susceptance [pu]',
            'tap':     'Off-nominal turns ratio [pu] (1.0 = nominal)',
            'phi':     'Phase shift angle [rad]',
            'omega_b': 'Base angular frequency [rad/s]',
            'bus1':    'Terminal-1 bus index',
            'bus2':    'Terminal-2 bus index',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'IL_mag': {
                'description': 'Series branch current magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(x[0]*x[0] + x[1]*x[1])'
            },
            'P1': {
                'description': 'Active power flow at terminal 1',
                'unit': 'pu',
                'cpp_expr': 'inputs[0]*outputs[0] + inputs[1]*outputs[1]'
            },
            'Q1': {
                'description': 'Reactive power flow at terminal 1',
                'unit': 'pu',
                'cpp_expr': 'inputs[1]*outputs[0] - inputs[0]*outputs[1]'
            },
            'P2': {
                'description': 'Active power flow at terminal 2',
                'unit': 'pu',
                'cpp_expr': 'inputs[2]*outputs[2] + inputs[3]*outputs[3]'
            },
            'P_loss': {
                'description': 'Active power loss in series branch',
                'unit': 'pu',
                'cpp_expr': 'r * (x[0]*x[0] + x[1]*x[1])'
            },
            'Vc1_mag': {
                'description': 'Shunt capacitor-1 voltage magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(x[2]*x[2] + x[3]*x[3])'
            },
            'Vc2_mag': {
                'description': 'Shunt capacitor-2 voltage magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(x[4]*x[4] + x[5]*x[5])'
            },
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // Series branch current (states 0-1)
            double IL_d  = x[0];
            double IL_q  = x[1];
            double Vc1_d = x[2];
            double Vc1_q = x[3];
            double Vc2_d = x[4];
            double Vc2_q = x[5];

            double V1_d = inputs[0];
            double V1_q = inputs[1];
            double V2_d = inputs[2];
            double V2_q = inputs[3];

            // Norton currents into bus 1: series branch + shunt capacitor
            // Series contribution (current flows into bus1 from the left terminal)
            double b_half = b * 0.5;

            // Shunt capacitor Norton current at terminal 1:  Ic1 = j(b/2) V1
            // In RI frame:  Ic1_d = -(b/2)*V1_q,  Ic1_q = (b/2)*V1_d
            double Ic1_d = -b_half * V1_q;
            double Ic1_q =  b_half * V1_d;

            // Shunt capacitor Norton current at terminal 2
            double Ic2_d = -b_half * V2_q;
            double Ic2_q =  b_half * V2_d;

            // Current injection into bus 1 from this line: shunt + series (from bus toward line)
            // Convention: positive = current INTO bus from line element
            outputs[0] = -IL_d + Ic1_d;      // I1_d
            outputs[1] = -IL_q + Ic1_q;      // I1_q
            outputs[2] =  IL_d + Ic2_d;      // I2_d
            outputs[3] =  IL_q + Ic2_q;      // I2_q
            outputs[4] =  IL_d;              // IL_d
            outputs[5] =  IL_q;              // IL_q

            // Terminal voltage magnitudes
            outputs[6] = sqrt(V1_d*V1_d + V1_q*V1_q);    // Vterm1
            outputs[7] = sqrt(V2_d*V2_d + V2_q*V2_q);    // Vterm2
        """

    def get_cpp_step_code(self) -> str:
        return """
            // -------------------------------------------------------
            // Port-Hamiltonian π-line dynamics (dq-frame, RI network)
            // -------------------------------------------------------
            double V1_d  = inputs[0];
            double V1_q  = inputs[1];
            double V2_d  = inputs[2];
            double V2_q  = inputs[3];

            double IL_d  = x[0];
            double IL_q  = x[1];
            double Vc1_d = x[2];
            double Vc1_q = x[3];
            double Vc2_d = x[4];
            double Vc2_q = x[5];

            double b_half = b * 0.5;

            // Apply tap and phase shift to terminal 1 voltage
            // V1_eff = tap * exp(j*phi) * V1
            double cos_phi = cos(phi);
            double sin_phi = sin(phi);
            double tap_inv = (tap > 1e-6) ? 1.0 / tap : 1.0;
            double V1_eff_d = tap_inv * (V1_d * cos_phi - V1_q * sin_phi);
            double V1_eff_q = tap_inv * (V1_d * sin_phi + V1_q * cos_phi);

            // --- Series RL branch dynamics (PH) ---
            // dIL_d/dt = (omega_b/x)(V1_eff_d - Vc1_d - V2_d + Vc2_d - r*IL_d) + IL_q
            // dIL_q/dt = (omega_b/x)(V1_eff_q - Vc1_q - V2_q + Vc2_q - r*IL_q) - IL_d
            double inv_x  = omega_b / x;
            double Vdrv_d = V1_eff_d - Vc1_d - V2_d + Vc2_d;
            double Vdrv_q = V1_eff_q - Vc1_q - V2_q + Vc2_q;

            dxdt[0] = inv_x * (Vdrv_d - r * IL_d) + IL_q;
            dxdt[1] = inv_x * (Vdrv_q - r * IL_q) - IL_d;

            // --- Shunt capacitor dynamics (PH) ---
            // Capacitor 1 at terminal 1: governed by excess current
            // C1 dVc1/dt = I_in1 - IL  (current from bus minus series branch current)
            // b_half = omega_b * C  =>  dVc/dt = omega_b / b_half * (I_in - IL)
            // In PH: dVc1_d/dt = omega_b * b_half * (V1_eff_d - Vc1_d) + Vc1_q
            // (The "+/- Vc cross-coupling comes from the omega_0 reference-frame rotation)
            double inv_C = omega_b * b_half;   // = omega_b * (B/2)

            dxdt[2] = inv_C * (V1_eff_d - Vc1_d) + Vc1_q;
            dxdt[3] = inv_C * (V1_eff_q - Vc1_q) - Vc1_d;

            dxdt[4] = inv_C * (V2_d - Vc2_d) + Vc2_q;
            dxdt[5] = inv_C * (V2_q - Vc2_q) - Vc2_d;
        """

    # ------------------------------------------------------------------
    # Initialization contract
    # ------------------------------------------------------------------

    def init_from_phasor(self, V1: complex, V2: complex) -> Tuple[np.ndarray, Dict]:
        """
        Steady-state initialization from known terminal phasors.

        At steady state:
          ẋ = 0  ⟹  IL = (V1 - V2) / (r + jx),  Vc = V_terminal
        """
        p = self.params
        r = p.get('r', 0.0)
        x = p.get('x', 0.1)
        tap = p.get('tap', 1.0)
        phi = p.get('phi', 0.0)

        # Apply tap/phase to V1
        V1_eff = V1 / (tap * np.exp(1j * phi))

        # Series branch current
        Z = complex(r, x)
        IL = (V1_eff - V2) / Z if abs(Z) > 1e-9 else 0j

        # Shunt capacitor voltages equal terminal voltages at steady state
        x0 = np.array([
            IL.real, IL.imag,       # IL_d, IL_q
            V1_eff.real, V1_eff.imag,  # Vc1_d, Vc1_q
            V2.real, V2.imag,       # Vc2_d, Vc2_q
        ])
        return x0, {}

    def to_json_line_entry(self) -> Dict[str, Any]:
        """Convert this component to a 'Line' JSON section entry."""
        p = self.params
        return {
            'idx':  self.name,
            'bus1': p.get('bus1'),
            'bus2': p.get('bus2'),
            'r':    p.get('r', 0.0),
            'x':    p.get('x', 0.1),
            'b':    p.get('b', 0.0),
            'tap':  p.get('tap', 1.0),
            'phi':  p.get('phi', 0.0),
        }
