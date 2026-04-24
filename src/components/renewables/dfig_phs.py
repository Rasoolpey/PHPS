"""
DFIG_PHS — Doubly-Fed Induction Generator (Σ₃ Port-Hamiltonian Model)
=====================================================================

Electromechanical subsystem of the DFIG wind turbine in explicit PHS form:

    ẋ = (J₃(x) − R₃) ∇H₃(x) + g₃ · u₃

States (5 energy variables):
    x₃ = [φ_sd, φ_sq, φ_rd, φ_rq, p_g]

    φ_sd, φ_sq  — stator  d/q-axis flux linkages  [pu]
    φ_rd, φ_rq  — rotor   d/q-axis flux linkages  [pu]
    p_g         — generator rotor angular momentum  [pu·s]

Hamiltonian (electromagnetic field + kinetic energy):

    H₃ = ½ x₃ᵀ M⁻¹ x₃

    M = [[Ls, 0, Lm, 0, 0],
         [0, Ls, 0, Lm, 0],
         [Lm, 0, Lr, 0, 0],
         [0, Lm, 0, Lr, 0],
         [0,  0,  0,  0, j]]

Gradient:  ∇H₃ = M⁻¹ x₃ = [i_sd, i_sq, i_rd, i_rq, ω_g]

Structure matrix J₃ (skew-symmetric, state-dependent):

    J₃ = ωs × [synchronous-frame rotation block]
       + [electromechanical gyrator block (φ_rd, φ_rq)]

Dissipation:  R₃ = diag(Rs, Rs, Rr, Rr, f_damp)

Ports:  u₃ = [V_sd, V_sq, V_rd, V_rq, T_shaft]

Norton network interface:
    Y_N = 1/(Rs + j·X_σ),   X_σ = σ_LS/Lr,   σ_LS = Ls·Lr − Lm²

References:
    Song & Qu (2011), "Energy-based modelling and control of wind energy
    conversion system with DFIG", Int. J. Control 84(2):281-292.
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigPHS(PowerComponent):
    """
    DFIG electromechanical subsystem (Σ₃) in Port-Hamiltonian form.

    Five-state model with Norton-equivalent grid interface.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        sigma_LS = params['Ls'] * params['Lr'] - params['Lm'] ** 2
        self.params.setdefault('ra', params['Rs'])
        self.params.setdefault('xd_double_prime', sigma_LS / params['Lr'])

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd',   'effort', 'pu'),   # [0] stator V_Re (RI-frame)
                ('Vq',   'effort', 'pu'),   # [1] stator V_Im (RI-frame)
                ('Tm',   'effort', 'pu'),   # [2] shaft torque from drivetrain
                ('Vrd',  'effort', 'pu'),   # [3] rotor d-voltage from RSC
                ('Vrq',  'effort', 'pu'),   # [4] rotor q-voltage from RSC
            ],
            'out': [
                ('Id',         'flow', 'pu'),   # [0]  Norton injection Re
                ('Iq',         'flow', 'pu'),   # [1]  Norton injection Im
                ('omega',      'flow', 'pu'),   # [2]  rotor mechanical speed
                ('Pe',         'flow', 'pu'),   # [3]  active power
                ('Qe',         'flow', 'pu'),   # [4]  reactive power
                ('id_dq',      'flow', 'pu'),   # [5]  stator d-current
                ('iq_dq',      'flow', 'pu'),   # [6]  stator q-current
                ('It_Re',      'flow', 'pu'),   # [7]  terminal current Re
                ('It_Im',      'flow', 'pu'),   # [8]  terminal current Im
                ('i_rd',       'flow', 'pu'),   # [9]  rotor d-current
                ('i_rq',       'flow', 'pu'),   # [10] rotor q-current
                ('phi_sd_out', 'flow', 'pu'),   # [11] stator d-flux (state echo)
                ('phi_sq_out', 'flow', 'pu'),   # [12] stator q-flux (state echo)
                ('P_rotor',    'flow', 'pu'),   # [13] rotor electrical power
                ('slip_out',   'flow', 'pu'),   # [14] slip
                ('Te_out',     'flow', 'pu'),   # [15] electromagnetic torque
                ('Vterm_out',  'flow', 'pu'),   # [16] terminal voltage magnitude
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['phi_sd', 'phi_sq', 'phi_rd', 'phi_rq', 'p_g']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Ls':        'Stator self-inductance [pu]',
            'Lr':        'Rotor self-inductance [pu]',
            'Lm':        'Mutual (magnetising) inductance [pu]',
            'Rs':        'Stator resistance [pu]',
            'Rr':        'Rotor resistance [pu]',
            'j_inertia': 'Generator rotor inertia (2H) [pu·s]',
            'f_damp':    'Rotor mechanical friction [pu]',
            'np':        'Number of pole pairs',
            'omega_b':   'Base angular frequency [rad/s]',
            'omega_s':   'Synchronous electrical speed [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'generator'

    @property
    def contributes_norton_admittance(self) -> bool:
        return True

    @property
    def uses_ri_frame(self) -> bool:
        return True

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pe':       {'description': 'Active power output',    'unit': 'pu', 'cpp_expr': 'outputs[3]'},
            'Qe':       {'description': 'Reactive power output',  'unit': 'pu', 'cpp_expr': 'outputs[4]'},
            'omega_pu': {'description': 'Rotor speed',            'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'slip':     {'description': 'Slip',                   'unit': '-',
                         'cpp_expr': '(omega_s - outputs[2]) / omega_s'},
            'phi_sd':   {'description': 'Stator d-flux',          'unit': 'pu', 'cpp_expr': 'x[0]'},
            'phi_sq':   {'description': 'Stator q-flux',          'unit': 'pu', 'cpp_expr': 'x[1]'},
            'phi_rd':   {'description': 'Rotor d-flux',           'unit': 'pu', 'cpp_expr': 'x[2]'},
            'phi_rq':   {'description': 'Rotor q-flux',           'unit': 'pu', 'cpp_expr': 'x[3]'},
            'Te_elec':  {'description': 'Electromagnetic torque', 'unit': 'pu', 'cpp_expr': 'outputs[15]'},
            'V_term':   {'description': 'Terminal voltage',       'unit': 'pu',
                         'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])'},
            'i_rd':     {'description': 'Rotor d-current',        'unit': 'pu', 'cpp_expr': 'outputs[9]'},
            'i_rq':     {'description': 'Rotor q-current',        'unit': 'pu', 'cpp_expr': 'outputs[10]'},
            'Ir_mag':   {'description': 'Rotor current magnitude','unit': 'pu',
                         'cpp_expr': 'sqrt(outputs[9]*outputs[9] + outputs[10]*outputs[10])'},
            'i_sd':     {'description': 'Stator d-current',       'unit': 'pu', 'cpp_expr': 'outputs[5]'},
            'i_sq':     {'description': 'Stator q-current',       'unit': 'pu', 'cpp_expr': 'outputs[6]'},
            'Is_mag':   {'description': 'Stator current magnitude','unit': 'pu',
                         'cpp_expr': 'sqrt(outputs[5]*outputs[5] + outputs[6]*outputs[6])'},
            'H_total':  {'description': 'Total Hamiltonian',      'unit': 'pu',
                         'cpp_expr': self._hamiltonian_cpp_expr()},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // ---- Currents from flux states: ∇H₃ = M⁻¹·x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * x[0] - Lm * x[2]) / sigma_LS;
            double i_sq =  (Lr * x[1] - Lm * x[3]) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;
            double omega_g = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Voltages ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Power (generator convention: positive = injected) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // ---- Norton equivalent: I_N = I_gen + Y_N × V ----
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            outputs[0] = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            outputs[1] = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[2] = omega_g;
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = x[0];
            outputs[12] = x[1];
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // ---- Inputs ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];
            double T_shaft = inputs[2];
            double Vrd = inputs[3];
            double Vrq = inputs[4];

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_g    = x[4];

            // ---- Hamiltonian gradient: ∇H₃ = M⁻¹ x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_g = p_g / j_inertia;

            // ---- J₃(x)·∇H₃ (skew-symmetric structure) ----
            // Synchronous-frame rotation + electromechanical gyrator
            double J0 =  omega_s * Ls * i_sq  + omega_s * Lm * i_rq;
            double J1 = -omega_s * Ls * i_sd  - omega_s * Lm * i_rd;
            double J2 =  omega_s * Lm * i_sq  + omega_s * Lr * i_rq - np * phi_rq * omega_g;
            double J3 = -omega_s * Lm * i_sd  - omega_s * Lr * i_rd + np * phi_rd * omega_g;
            double J4 =  np * (phi_rq * i_rd - phi_rd * i_rq);  // = -Te (skew row)

            // ---- R₃·∇H₃ (dissipation) ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * omega_g;

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PHS dynamics: ẋ = (1/ωb)(J₃ − R₃)∇H₃ + (1/ωb)g₃·u₃ ----
            // Flux eqs scaled by ωb; mechanical eq in seconds directly
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_shaft;  // = -Te - f_damp*ω + T_shaft

            // ---- Outputs (power, monitoring) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = Vrd * i_rd + Vrq * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        # States
        phi_sd, phi_sq = sp.symbols('phi_sd phi_sq')
        phi_rd, phi_rq = sp.symbols('phi_rd phi_rq')
        p_g = sp.Symbol('p_g')
        states = [phi_sd, phi_sq, phi_rd, phi_rq, p_g]

        # Inputs
        V_sd, V_sq = sp.symbols('V_sd V_sq')
        V_rd, V_rq = sp.symbols('V_rd V_rq')
        T_shaft = sp.Symbol('T_shaft')
        inputs = [V_sd, V_sq, V_rd, V_rq, T_shaft]

        # Parameters
        Ls_s = sp.Symbol('L_s', positive=True)
        Lr_s = sp.Symbol('L_r', positive=True)
        Lm_s = sp.Symbol('L_m', positive=True)
        Rs_s = sp.Symbol('R_s', nonnegative=True)
        Rr_s = sp.Symbol('R_r', nonnegative=True)
        j_s  = sp.Symbol('j', positive=True)
        f_s  = sp.Symbol('f', nonnegative=True)
        np_s = sp.Symbol('n_p', positive=True)
        ws_s = sp.Symbol('omega_s', positive=True)
        wb_s = sp.Symbol('omega_b', positive=True)

        params = {
            'Ls': Ls_s, 'Lr': Lr_s, 'Lm': Lm_s,
            'Rs': Rs_s, 'Rr': Rr_s,
            'j_inertia': j_s, 'f_damp': f_s,
            'np': np_s, 'omega_s': ws_s, 'omega_b': wb_s,
        }

        # Hamiltonian: H = ½ xᵀ M⁻¹ x
        sigma = Ls_s * Lr_s - Lm_s**2
        H_expr = (
            (Lr_s * (phi_sd**2 + phi_sq**2)
             + Ls_s * (phi_rd**2 + phi_rq**2)
             - 2 * Lm_s * (phi_sd * phi_rd + phi_sq * phi_rq))
            / (2 * sigma)
            + p_g**2 / (2 * j_s)
        )

        # J₃(x): synchronous-frame rotation + electromechanical gyrator
        J = sp.zeros(5, 5)
        # Synchronous-frame block
        J[0, 1] =  ws_s * Ls_s;  J[1, 0] = -ws_s * Ls_s
        J[0, 3] =  ws_s * Lm_s;  J[3, 0] = -ws_s * Lm_s
        J[1, 2] = -ws_s * Lm_s;  J[2, 1] =  ws_s * Lm_s
        J[2, 3] =  ws_s * Lr_s;  J[3, 2] = -ws_s * Lr_s
        # Electromechanical gyrator
        J[2, 4] = -np_s * phi_rq;  J[4, 2] =  np_s * phi_rq
        J[3, 4] =  np_s * phi_rd;  J[4, 3] = -np_s * phi_rd

        # R₃: dissipation
        R = sp.diag(Rs_s, Rs_s, Rr_s, Rr_s, f_s)

        # g₃: identity (each input maps to one state equation)
        g = sp.eye(5)

        return SymbolicPHS(
            name='DFIG_PHS',
            states=states, inputs=inputs, params=params,
            J=J, R=R, g=g, H=H_expr,
            description='DFIG electromechanical subsystem (Σ₃) in PH form.',
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _sigma(self) -> float:
        p = self.params
        return p['Ls'] * p['Lr'] - p['Lm'] ** 2

    def _norton_admittance(self) -> complex:
        p = self.params
        Rs = p['Rs']
        Xs = self._sigma() / p['Lr']
        denom = Rs**2 + Xs**2
        return complex(Rs / denom, -Xs / denom)

    def compute_norton_current(self, x_slice: np.ndarray,
                               V_bus_complex: complex = None) -> complex:
        p = self.params
        Lr = p['Lr']; Lm = p['Lm']; Rs = p['Rs']; Ls = p['Ls']
        sigma = self._sigma()

        i_sd = (Lr * x_slice[0] - Lm * x_slice[2]) / sigma
        i_sq = (Lr * x_slice[1] - Lm * x_slice[3]) / sigma

        Igen_re = -i_sd
        Igen_im = -i_sq

        Xs_sig = Ls - Lm**2 / Lr
        Zmag2 = Rs**2 + Xs_sig**2
        if V_bus_complex is not None and Zmag2 > 1e-12:
            Vd = V_bus_complex.real
            Vq = V_bus_complex.imag
            Igen_re += (Rs * Vd + Xs_sig * Vq) / Zmag2
            Igen_im += (Rs * Vq - Xs_sig * Vd) / Zmag2

        return complex(Igen_re, Igen_im)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        sigma = self._sigma()
        Lr = self.params['Lr']; Lm = self.params['Lm']
        i_sd = (Lr * x_slice[0] - Lm * x_slice[2]) / sigma
        i_sq = (Lr * x_slice[1] - Lm * x_slice[3]) / sigma
        return float(i_sd), float(i_sq)

    def compute_te(self, x_slice: np.ndarray, vd: float, vq: float) -> float:
        sigma = self._sigma()
        Ls = self.params['Ls']; Lm = self.params['Lm']
        np_poles = self.params['np']
        i_rd = (-Lm * x_slice[0] + Ls * x_slice[2]) / sigma
        i_rq = (-Lm * x_slice[1] + Ls * x_slice[3]) / sigma
        return float(np_poles * (x_slice[2] * i_rq - x_slice[3] * i_rd))

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        p = self.params
        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        Rs = p['Rs']; Rr = p['Rr']
        np_poles = p['np']
        omega_s = p.get('omega_s', 1.0)

        Vsd = V_phasor.real
        Vsq = V_phasor.imag
        # Motor convention: negate power-flow (generator) current
        i_sd = -I_phasor.real
        i_sq = -I_phasor.imag

        # Rotor currents from stator voltage equilibrium (dφ_s/dt = 0)
        if abs(omega_s * Lm) > 1e-10:
            i_rq = (Rs * i_sd - Vsd - omega_s * Ls * i_sq) / (omega_s * Lm)
            i_rd = (Vsq - omega_s * Ls * i_sd - Rs * i_sq) / (omega_s * Lm)
        else:
            i_rq = 0.0
            i_rd = 0.0

        # Flux linkages
        phi_sd = Ls * i_sd + Lm * i_rd
        phi_sq = Ls * i_sq + Lm * i_rq
        phi_rd = Lr * i_rd + Lm * i_sd
        phi_rq = Lr * i_rq + Lm * i_sq

        # Rotor speed (synchronous start)
        omega_g_init = omega_s
        p_g_init = p['j_inertia'] * omega_g_init

        # Electromagnetic torque
        Te_init = np_poles * (phi_rd * i_rq - phi_rq * i_rd)

        # Power (generator convention)
        Pe_init = -(Vsd * i_sd + Vsq * i_sq)
        Qe_init = -(Vsq * i_sd - Vsd * i_sq)

        # Required rotor voltages for steady state (dφ_r/dt = 0)
        Vrd_ss = -(omega_s * Lm * i_sq + omega_s * Lr * i_rq
                   - np_poles * phi_rq * omega_g_init - Rr * i_rd)
        Vrq_ss = -(- omega_s * Lm * i_sd - omega_s * Lr * i_rd
                   + np_poles * phi_rd * omega_g_init - Rr * i_rq)

        x_init = np.array([phi_sd, phi_sq, phi_rd, phi_rq, p_g_init])

        P_rotor_init = Vrd_ss * i_rd + Vrq_ss * i_rq

        p['Tm0'] = float(Te_init)

        targets = {
            'Tm':        float(Te_init),
            'Pe':        float(Pe_init),
            'Qe':        float(Qe_init),
            'bus':       p.get('bus'),
            'Vt':        float(abs(V_phasor)),
            'omega':     float(omega_g_init),
            'Efd':       0.0,
            'Vrd':       float(Vrd_ss),
            'Vrq':       float(Vrq_ss),
            'id':        float(i_sd),
            'iq':        float(i_sq),
            'vd':        float(Vsd),
            'vq':        float(Vsq),
            'vd_ri':     float(V_phasor.real),
            'vq_ri':     float(V_phasor.imag),
            'i_rd':      float(i_rd),
            'i_rq':      float(i_rq),
            'phi_sd':    float(phi_sd),
            'phi_sq':    float(phi_sq),
            'P_rotor':   float(P_rotor_init),
            'Te':        float(Te_init),
            'slip':      0.0,
            'f_damp_gen': float(p['f_damp']),
        }
        return x_init, targets

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                               vd: float, vq: float) -> np.ndarray:
        p = self.params
        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        Rs = p['Rs']; omega_s = p.get('omega_s', 1.0)
        sigma = self._sigma()

        # Hold rotor currents fixed, re-solve stator equilibrium
        i_rd = (-Lm * x_slice[0] + Ls * x_slice[2]) / sigma
        i_rq = (-Lm * x_slice[1] + Ls * x_slice[3]) / sigma

        # Stator equilibrium: 0 = ωs·Ls·i_sq + ωs·Lm·i_rq − Rs·i_sd + Vsd
        #                      0 = −ωs·Ls·i_sd − ωs·Lm·i_rd − Rs·i_sq + Vsq
        # → 2×2 system: [[Rs, −ωs·Ls], [ωs·Ls, Rs]] · [i_sd, i_sq] = [rhs_d, rhs_q]
        rhs_d = vd + omega_s * Lm * i_rq
        rhs_q = vq - omega_s * Lm * i_rd
        det = Rs**2 + (omega_s * Ls)**2
        i_sd = (Rs * rhs_d + omega_s * Ls * rhs_q) / det
        i_sq = (-omega_s * Ls * rhs_d + Rs * rhs_q) / det

        x_new = x_slice.copy()
        x_new[0] = Ls * i_sd + Lm * i_rd
        x_new[1] = Ls * i_sq + Lm * i_rq
        x_new[2] = Lr * i_rd + Lm * i_sd
        x_new[3] = Lr * i_rq + Lm * i_sq
        return x_new

    def refine_current_source_init(self, x_slice: np.ndarray,
                                   targets: dict,
                                   V_bus: complex) -> np.ndarray:
        vd = V_bus.real
        vq = V_bus.imag
        p = self.params
        sigma = self._sigma()
        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        Rs = p['Rs']; Rr = p['Rr']
        np_poles = p['np']; omega_s = p.get('omega_s', 1.0)

        i_sd = (Lr * x_slice[0] - Lm * x_slice[2]) / sigma
        i_sq = (Lr * x_slice[1] - Lm * x_slice[3]) / sigma
        i_rd = (-Lm * x_slice[0] + Ls * x_slice[2]) / sigma
        i_rq = (-Lm * x_slice[1] + Ls * x_slice[3]) / sigma

        omega_g = x_slice[4] / p['j_inertia']
        Te = np_poles * (x_slice[2] * i_rq - x_slice[3] * i_rd)
        Pe = -(vd * i_sd + vq * i_sq)
        Qe = -(vq * i_sd - vd * i_sq)

        Vrd_ss = -(omega_s * Lm * i_sq + omega_s * Lr * i_rq
                   - np_poles * x_slice[3] * omega_g - Rr * i_rd)
        Vrq_ss = -(- omega_s * Lm * i_sd - omega_s * Lr * i_rd
                   + np_poles * x_slice[2] * omega_g - Rr * i_rq)

        P_rotor = Vrd_ss * i_rd + Vrq_ss * i_rq

        targets.update({
            'Pe': float(Pe), 'Qe': float(Qe), 'Te': float(Te),
            'Tm': float(Te), 'Efd': 0.0, 'omega': float(omega_g),
            'Vrd': float(Vrd_ss), 'Vrq': float(Vrq_ss),
            'id': float(i_sd), 'iq': float(i_sq),
            'vd': float(vd), 'vq': float(vq),
            'vd_ri': float(vd), 'vq_ri': float(vq),
            'i_rd': float(i_rd), 'i_rq': float(i_rq),
            'phi_sd': float(x_slice[0]), 'phi_sq': float(x_slice[1]),
            'P_rotor': float(P_rotor),
            'Vt': float(math.sqrt(vd**2 + vq**2)),
            'f_damp_gen': float(p['f_damp']),
        })
        p['Tm0'] = float(Te)
        return x_slice.copy()

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                      Efd_target: float = None) -> np.ndarray:
        return x_slice

    def rebalance_for_bus_voltage(self, x_slice: np.ndarray,
                                  V_bus_complex: complex) -> tuple:
        vd = V_bus_complex.real
        vq = V_bus_complex.imag
        x_new = self.refine_at_kron_voltage(x_slice, vd, vq)
        targets = {}
        self.refine_current_source_init(x_new, targets, V_bus_complex)
        return x_new, targets
