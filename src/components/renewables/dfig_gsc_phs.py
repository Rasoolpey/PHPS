"""
DFIG_GSC_PHS — Grid-Side Converter with Filter Inductance (Σ₆)
===============================================================

The GSC connects to the grid through a series filter inductance L_f.
The component combines the physical filter dynamics (PHS, Σ₆) with
the DC-voltage and Q PI controllers.

States (4):
    [0] phi_fd  — filter d-axis flux linkage  [pu]
    [1] phi_fq  — filter q-axis flux linkage  [pu]
    [2] x_Vdc  — DC-voltage PI integrator     [pu]
    [3] x_Q    — Q-control PI integrator      [pu]

Filter PHS (Σ₆):
    H₆ = (φ_fd² + φ_fq²) / (2·L_f)

    J₆ = ω_s·L_f · [[0, 1], [-1, 0]]   (synchronous-frame rotation)
    R₆ = R_f · I₂                        (filter resistance)
    g₆ = (1/L_f) · I₂
    u₆ = [V_fd − V_sd, V_fq − V_sq]

    ∇H₆ = [i_fd, i_fq] = [φ_fd/L_f, φ_fq/L_f]

Expanded dynamics:
    L_f · di_fd/dt = (V_fd − V_sd) − R_f·i_fd + ω_s·L_f·i_fq
    L_f · di_fq/dt = (V_fq − V_sq) − R_f·i_fq − ω_s·L_f·i_fd

where V_fd, V_fq are converter voltage commands from the PI controller.

Ports:
    in:  [Vdc, Vdc_ref, Qref, Vd, Vq]
    out: [Id, Iq, P_gsc, Q_gsc, Pe]

The GSC is a current-source injector (no Norton admittance contribution).
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigGscPHS(PowerComponent):
    """
    Grid-Side Converter (Σ₆) with filter inductance in PH form.

    Four-state model: filter flux linkages + PI integrators.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vdc',     'signal', 'pu'),   # [0]
                ('Vdc_ref', 'signal', 'pu'),   # [1]
                ('Qref',    'signal', 'pu'),   # [2]
                ('Vd',      'effort', 'pu'),   # [3]
                ('Vq',      'effort', 'pu'),   # [4]
            ],
            'out': [
                ('Id',      'flow', 'pu'),     # [0] current injection Re
                ('Iq',      'flow', 'pu'),     # [1] current injection Im
                ('P_gsc',   'flow', 'pu'),     # [2] GSC active power
                ('Q_gsc',   'flow', 'pu'),     # [3] GSC reactive power
                ('Pe',      'flow', 'pu'),     # [4] alias for compat
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['phi_fd', 'phi_fq', 'x_Vdc', 'x_Q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'L_f':     'Filter inductance [pu]',
            'R_f':     'Filter resistance [pu]',
            'omega_s': 'Synchronous speed [pu]',
            'omega_b': 'Base angular frequency [rad/s]',
            'Kp_dc':   'Vdc-loop proportional gain',
            'Ki_dc':   'Vdc-loop integral gain',
            'Kp_Qg':   'Q-loop proportional gain',
            'Ki_Qg':   'Q-loop integral gain',
            'I_max':   'Current magnitude saturation [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    @property
    def contributes_norton_admittance(self) -> bool:
        return False

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'i_fd':   {'description': 'Filter d-current', 'unit': 'pu', 'cpp_expr': 'x[0] / L_f'},
            'i_fq':   {'description': 'Filter q-current', 'unit': 'pu', 'cpp_expr': 'x[1] / L_f'},
            'P_gsc':  {'description': 'GSC active power',  'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'Q_gsc':  {'description': 'GSC reactive power', 'unit': 'pu', 'cpp_expr': 'outputs[3]'},
            'H_filter': {'description': 'Filter energy', 'unit': 'pu',
                         'cpp_expr': '(x[0]*x[0] + x[1]*x[1]) / (2.0 * L_f)'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];

            // ---- PI controller: Vdc loop → d-axis current ref ----
            // Convention: Vdc > Vdc_ref → increase Id_ref → export more to grid → Vdc drops
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref (negative feedback)
            double Id_ref = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → q-axis current ref ----
            // Open-loop feedforward: Iq_ref ∝ Qref
            // (Closed-loop Q feedback requires sign inversion because Q ≈ -Vd·i_fq)
            double e_Q = inputs[2];  // Qref
            double Iq_ref = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((Id_ref > 0.0 && e_Vdc < 0.0) || (Id_ref < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Iq_ref > 0.0 && e_Q < 0.0) || (Iq_ref < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping (prevent runaway buildup) ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        phi_fd, phi_fq = sp.symbols('phi_fd phi_fq')
        x_Vdc, x_Q = sp.symbols('x_Vdc x_Q')
        states = [phi_fd, phi_fq, x_Vdc, x_Q]

        # Inputs: converter voltage minus grid voltage (for filter)
        # + PI error signals
        dV_fd = sp.Symbol('Delta_V_fd')
        dV_fq = sp.Symbol('Delta_V_fq')
        e_Vdc_s = sp.Symbol('e_Vdc')
        e_Q_s = sp.Symbol('e_Q')
        inputs = [dV_fd, dV_fq, e_Vdc_s, e_Q_s]

        Lf = sp.Symbol('L_f', positive=True)
        Rf = sp.Symbol('R_f', nonnegative=True)
        ws = sp.Symbol('omega_s', positive=True)
        wb = sp.Symbol('omega_b', positive=True)

        params = {'L_f': Lf, 'R_f': Rf, 'omega_s': ws, 'omega_b': wb}

        # Hamiltonian: filter energy + PI integrator energy
        H_expr = (phi_fd**2 + phi_fq**2) / (2 * Lf) + sp.Rational(1, 2) * (x_Vdc**2 + x_Q**2)

        # J: synchronous-frame rotation for filter (rows 0,1)
        J = sp.zeros(4, 4)
        J[0, 1] = ws * Lf;  J[1, 0] = -ws * Lf

        # R: filter resistance (rows 0,1), PI integrators lossless
        R = sp.zeros(4, 4)
        R[0, 0] = Rf
        R[1, 1] = Rf

        # g: input coupling
        g = sp.zeros(4, 4)
        g[0, 0] = 1 / Lf  # dV_fd → dφ_fd
        g[1, 1] = 1 / Lf  # dV_fq → dφ_fq
        g[2, 2] = 1        # e_Vdc → dx_Vdc
        g[3, 3] = 1        # e_Q → dx_Q

        return SymbolicPHS(
            name='DFIG_GSC_PHS',
            states=states, inputs=inputs, params=params,
            J=J, R=R, g=g, H=H_expr,
            description='GSC filter inductance (Σ₆) + PI controllers in PH form.',
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def compute_norton_current(self, x_slice: np.ndarray,
                               V_bus_complex: complex = None) -> complex:
        L_f = float(self.params['L_f'])
        i_fd = float(x_slice[0]) / L_f
        i_fq = float(x_slice[1]) / L_f
        return complex(i_fd, i_fq)

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        Ki_dc = p['Ki_dc']
        L_f = p['L_f']
        omega_s = p.get('omega_s', 1.0)

        P_rotor = targets.get('P_rotor', 0.0)
        Vd = targets.get('vd_ri', targets.get('vd', 0.0))
        Vq = targets.get('vq_ri', targets.get('vq', 0.0))
        Vdc_ref = targets.get('Vdc_ref', 1.0)

        if 'bus' in targets:
            p['bus'] = targets['bus']
        p['Vdc_ref0'] = float(Vdc_ref)
        p['Vdc0'] = float(Vdc_ref)

        Vmag = math.hypot(Vd, Vq)

        # At equilibrium: P_gsc = P_rotor, di_f/dt = 0
        # i_fd_ss and i_fq_ss from power balance
        # P_gsc = Vd·i_fd + Vq·i_fq = P_rotor
        # Q_gsc = Vq·i_fd - Vd·i_fq = 0  (Qref=0)
        if Vmag > 1e-6:
            i_fd_ss = P_rotor * Vd / (Vmag**2)
            i_fq_ss = P_rotor * Vq / (Vmag**2)
        else:
            i_fd_ss = 0.0
            i_fq_ss = 0.0

        phi_fd_ss = L_f * i_fd_ss
        phi_fq_ss = L_f * i_fq_ss

        # PI integrator: at equilibrium e_Vdc=0, so Id_ref = Ki_dc·x_Vdc
        # Id_ref ≈ i_fd_ss (assuming proportional error is zero)
        if abs(Ki_dc) > 1e-10:
            x_Vdc = i_fd_ss / Ki_dc
        else:
            x_Vdc = 0.0
        x_Q = 0.0

        return self._init_states({
            'phi_fd': phi_fd_ss,
            'phi_fq': phi_fq_ss,
            'x_Vdc': x_Vdc,
            'x_Q': x_Q,
        })
