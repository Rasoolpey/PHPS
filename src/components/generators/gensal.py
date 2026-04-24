import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class GenSal(PowerComponent):
    """
    Salient-Pole Generator Model (GENSAL) with additive saturation.

    Port-Hamiltonian 5th-order synchronous machine model for salient-pole
    (hydro) generators. The salient-pole topology has no q-axis damper winding,
    so the q-axis subtransient flux collapses to an algebraic constraint, giving
    a 5th-order model:

        x = [delta, omega, E'q, psi_d, psi''q]

    Key differences from GENROU:
    - No q-axis transient winding  →  Ed' state removed
    - psi''q is a true state (subtransient q-flux), not blended from Ed'
    - Saturation enters additively in dE'q/dt (GENROU/GENSAL convention)
    - Xd'' = Xq'' (subtransient saliency neglected in GENSAL)

    Physics reference: Sauer & Pai Ch. 3, PowerWorld GENSAL block diagram.

    Port-Hamiltonian interpretation
    --------------------------------
    H = H_mech + H_d + H_q
      H_mech = H*(omega-1)^2
      H_d    = (1/2)*(E'q^2 / (Xd - X'd) + psi_d^2 * ...)
      H_q    = (1/2)*(psi''q^2 / (Xq - Xq''))

    Dissipation: mechanical damping D*(omega-1)^2 + stator copper ra*(id^2+iq^2)
    Ports: mechanical (Tm, omega), field (Efd, id_fd), electrical (Vd/Vq, Id/Iq)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd',  'effort', 'pu'),
                ('Vq',  'effort', 'pu'),
                ('Tm',  'effort', 'pu'),
                ('Efd', 'effort', 'pu'),
            ],
            'out': [
                ('Id',    'flow', 'pu'),   # RI-frame Norton I_Re
                ('Iq',    'flow', 'pu'),   # RI-frame Norton I_Im
                ('omega', 'flow', 'pu'),
                ('Pe',    'flow', 'pu'),
                ('Qe',    'flow', 'pu'),
                ('id_dq', 'flow', 'pu'),   # dq-frame id (for exciter load comp)
                ('iq_dq', 'flow', 'pu'),   # dq-frame iq
                ('It_Re', 'flow', 'pu'),   # actual terminal current Re
                ('It_Im', 'flow', 'pu'),   # actual terminal current Im
            ],
        }

    @property
    def state_schema(self) -> List[str]:
        # 5 states: no Ed' (no q-axis transient winding in salient-pole machine)
        return ['delta', 'omega', 'E_q_prime', 'psi_d', 'psi_q_pp']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H':                'Inertia constant',
            'D':                'Damping coefficient',
            'ra':               'Stator resistance',
            'xd':               'd-axis synchronous reactance',
            'xq':               'q-axis synchronous reactance',
            'xd_prime':         'd-axis transient reactance',
            'xd_double_prime':  'd-axis sub-transient reactance',
            # NOTE: For GENSAL, Xq'' = Xd'' (subtransient saliency not modelled)
            # The user may provide xq_double_prime; if equal to xd_double_prime
            # the Norton circuit simplifies. We keep it explicit for generality.
            'xq_double_prime':  'q-axis sub-transient reactance (= Xd'' for GENSAL)',
            'Td0_prime':        'd-axis transient open-circuit time constant',
            'Td0_double_prime': 'd-axis sub-transient open-circuit time constant',
            'Tq0_double_prime': 'q-axis sub-transient open-circuit time constant',
            'xl':               'Leakage reactance',
            # Saturation parameters (S1, S12 define the sat. curve at 1.0 and 1.2 pu)
            'S10':              'Saturation at 1.0 pu flux (additive, GENSAL convention)',
            'S12':              'Saturation at 1.2 pu flux',
            'omega_b':          'Base frequency [rad/s]',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'delta_deg': {
                'description': 'Rotor angle',
                'unit': 'deg',
                'cpp_expr': 'x[0] * 180.0 / 3.14159265359',
            },
            'Te': {
                'description': 'Electrical torque',
                'unit': 'pu',
                'cpp_expr': 'outputs[3]',
            },
            'Pe': {
                'description': 'Active power',
                'unit': 'pu',
                'cpp_expr': 'outputs[3]',
            },
            'Qe': {
                'description': 'Reactive power',
                'unit': 'pu',
                'cpp_expr': 'outputs[4]',
            },
            'V_term': {
                'description': 'Terminal voltage',
                'unit': 'pu',
                'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])',
            },
        }

    # ------------------------------------------------------------------
    #  Saturation helper (scaled-quadratic, PSS/E convention)
    #  Sat(x) = B*(x-A)^2 / x   [scaled quadratic]
    #  We compute A, B from S10, S12 at init time. During C++ simulation
    #  we embed the formula using the pre-computed A and B parameters.
    #  To keep it simple in generated code we use the exponential form
    #  approximated from S10, S12; but since the additive saturation in
    #  GENSAL enters only as a correction to dE'q/dt it is modest.
    #
    #  For portability we implement the scaled-quadratic directly in C++
    #  using A_sat and B_sat parameters derived from S10/S12 in Python.
    # ------------------------------------------------------------------

    @staticmethod
    def _sat_coeffs(S10: float, S12: float):
        """Return (A, B) for scaled-quadratic Sat(x) = B*(x-A)^2/x."""
        # Solve:  S10 = B*(1.0-A)^2/1.0,  S12 = B*(1.2-A)^2/1.2
        # => S10 = B*(1-A)^2,  S12*1.2 = B*(1.2-A)^2
        # Let u = 1-A, v = 1.2-A = u+0.2
        # S10 = B*u^2,  1.2*S12 = B*(u+0.2)^2
        # ratio: 1.2*S12/S10 = ((u+0.2)/u)^2  => (u+0.2)/u = sqrt(1.2*S12/S10)
        # => 0.2/u = sqrt(1.2*S12/S10) - 1  => u = 0.2/(sqrt(1.2*S12/S10)-1)
        if S10 <= 0.0 or S12 <= 0.0:
            return 0.0, 0.0
        ratio = math.sqrt(1.2 * S12 / S10)
        if abs(ratio - 1.0) < 1e-10:
            return 0.0, 0.0
        u = 0.2 / (ratio - 1.0)
        A = 1.0 - u
        B = S10 / (u * u)
        return A, B

    # ------------------------------------------------------------------
    # Norton current (voltage-independent, used in network solve)
    # For GENSAL:  psi_d'' = Eq'*kd + psi_d*(1-kd)
    #              psi_q'' = psi_q_pp   (direct state)
    # Norton: I = Y_source * EMF_source  (ignoring terminal V)
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
            double delta  = x[0];
            double sin_d  = sin(delta);
            double cos_d  = cos(delta);

            double Eq_p    = x[2];
            double psi_d   = x[3];
            double psi_q_pp = x[4];   // GENSAL: direct subtransient q-flux state

            // Subtransient d-flux blending (same as GENROU d-axis)
            double k_d      = (xd_double_prime - xl) / (xd_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);

            // Norton injection: short-circuit current from subtransient EMF
            // Zsource = ra + j*Xd''  (GENSAL: Xq'' = Xd'' assumed)
            double det    = ra*ra + xd_double_prime * xq_double_prime;
            double id_no  = (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no  = ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re =  id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];   // omega
            outputs[3] = 0.0;    // Pe (updated in step)
            outputs[4] = 0.0;    // Qe (updated in step)
            outputs[5] = id_no;  // dq id (Norton estimate)
            outputs[6] = iq_no;  // dq iq
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm   = inputs[2];
            double Efd  = inputs[3];

            double delta   = x[0];
            double omega   = x[1];
            double Eq_p    = x[2];
            double psi_d   = x[3];
            double psi_q_pp = x[4];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // Park transform: RI → dq
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // Subtransient d-flux
            double k_d      = (xd_double_prime - xl) / (xd_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);

            // Stator equations (Xq'' direct state, so psi_q'' = psi_q_pp)
            // vd = -ra*id + Xq''*iq - psi_q_pp  [neglect speed modulation of flux]
            // vq = +psi_d_pp - Xd''*id - ra*iq
            double rhs_d = vd + psi_q_pp;
            double rhs_q = vq - psi_d_pp;
            double det   = ra*ra + xd_double_prime * xq_double_prime;
            double id    = (-ra * rhs_d - xq_double_prime * rhs_q) / det;
            double iq    = ( xd_double_prime * rhs_d - ra * rhs_q) / det;

            // Electric torque: Te = psi_d''*iq - psi_q''*id
            double Te = psi_d_pp * iq - psi_q_pp * id;

            // ---- Saturation (additive GENSAL convention) ----
            // Compute air-gap flux magnitude for saturation argument
            // psi_ag = sqrt((vq + ra*iq + id*xl)^2 + (vd + ra*id - iq*xl)^2)
            // For the additive GENSAL formulation, sat reduces the d-axis
            // flux linkage driving dE'q/dt.
            // Scaled quadratic: Sat(psi) = B_sat*(psi - A_sat)^2 / psi  if psi > A_sat, else 0
            double psi_d_term = vq + ra * iq;
            double psi_q_term = vd + ra * id;
            double psi_ag = sqrt(psi_d_term*psi_d_term + psi_q_term*psi_q_term);
            double sat_val = 0.0;
            if (psi_ag > A_sat && psi_ag > 1e-6) {
                double diff = psi_ag - A_sat;
                sat_val = B_sat * diff * diff / psi_ag;
            }
            // Additive saturation current (GENSAL: acts on d-axis only via Laad*Ifd)
            // Effective: Eq' -> Eq' + (Xd - Xd')*sat_val  in the field circuit
            double Efd_net = Efd - sat_val * (xd - xl);

            // ---- State derivatives ----
            // Mechanical (swing equation)
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = (Tm - Te - D * (omega - 1.0)) / (2.0 * H);

            // d-axis transient: dE'q/dt
            dxdt[2] = (Efd_net - Eq_p - (xd - xd_prime) * id) / Td0_prime;

            // d-axis subtransient: dpsi_d/dt  (blended flux state)
            dxdt[3] = (-psi_d - (xd_prime - xd_double_prime) * id + Eq_p) / Td0_double_prime;

            // q-axis subtransient: dpsi_q''/dt  (GENSAL: direct state, no q transient)
            // Equivalent to the single q-axis rotor loop:
            //   T''q0 * dpsi_q''/dt = -(psi_q'' + (Xq - Xq'')*iq)
            dxdt[4] = (-(psi_q_pp) - (xq - xq_double_prime) * iq) / Tq0_double_prime;

            // Update outputs with actual computed values
            outputs[3] = vd*id + vq*iq;      // Pe
            outputs[4] = vq*id - vd*iq;      // Qe
            outputs[5] = id;                  // dq id
            outputs[6] = iq;                  // dq iq
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        """

    # ------------------------------------------------------------------
    # Initialization contract (matches GENROU pattern)
    # ------------------------------------------------------------------

    @property
    def component_role(self) -> str:
        return 'generator'

    def _sat_params(self):
        """Return (A_sat, B_sat) pre-computed from S10/S12."""
        S10 = self.params.get('S10', 0.0)
        S12 = self.params.get('S12', 0.0)
        return self._sat_coeffs(S10, S12)

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        p = self.params
        delta   = x_slice[0]
        Eq_p    = x_slice[2]
        psi_d   = x_slice[3]
        psi_q_pp = x_slice[4]
        xd_pp   = p['xd_double_prime']
        xq_pp   = p['xq_double_prime']
        ra      = p.get('ra', 0.0)
        xl      = p['xl']
        xd_p    = p['xd_prime']

        k_d      = (xd_pp - xl) / (xd_p - xl)
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)

        det   = ra**2 + xd_pp * xq_pp
        id_no = (-ra * psi_q_pp + xq_pp * psi_d_pp) / det
        iq_no = (xd_pp * psi_q_pp + ra * psi_d_pp) / det

        sin_d = math.sin(delta)
        cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                       -id_no * cos_d + iq_no * sin_d)

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        p   = self.params
        ra  = p.get('ra', 0.0)
        xd  = p['xd'];    xq  = p['xq']
        xd_p  = p['xd_prime']
        xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']
        xl    = p['xl']

        # Rotor angle (from q-axis)
        Eq_phasor = V_phasor + (ra + 1j * xq) * I_phasor
        delta = float(np.angle(Eq_phasor))

        # Park transform
        dq_factor = np.exp(-1j * (delta - math.pi / 2))
        vd = float((V_phasor * dq_factor).real)
        vq = float((V_phasor * dq_factor).imag)
        id_val = float((I_phasor * dq_factor).real)
        iq_val = float((I_phasor * dq_factor).imag)

        # Subtransient d-flux
        k_d = (xd_pp - xl) / (xd_p - xl)
        psi_d_pp = vq + ra * iq_val + xd_pp * id_val

        # GENSAL: psi_q'' is direct state; at equilibrium dxdt[4]=0:
        # psi_q_pp = -(Xq - Xq'')*iq
        psi_q_pp_eq = -(xq - xq_pp) * iq_val

        # d-axis transient states (dxdt[2]=0, dxdt[3]=0)
        Eq_p  = psi_d_pp + (1.0 - k_d) * (xd_p - xd_pp) * id_val
        psi_d = Eq_p - (xd_p - xd_pp) * id_val

        # Efd (ignoring saturation at init for simplicity)
        Efd_req = Eq_p + (xd - xd_p) * id_val
        Tm_req  = vd * id_val + vq * iq_val

        targets = {
            'Efd': float(Efd_req), 'Tm': float(Tm_req),
            'Vt':  float(abs(V_phasor)),
            'omega': 1.0,
            'vd': vd, 'vq': vq,
            'id': id_val, 'iq': iq_val,
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
        }
        return np.array([delta, 1.0, Eq_p, psi_d, psi_q_pp_eq]), targets

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> np.ndarray:
        x = x_slice.copy()
        p  = self.params
        ra = p.get('ra', 0.0)
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xd_p  = p['xd_prime'];        xq    = p['xq']
        xl    = p['xl']

        Eq_p   = x[2]; psi_d = x[3]
        k_d    = (xd_pp - xl) / (xd_p - xl)
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)

        # Update psi_q_pp from equilibrium condition
        det   = ra**2 + xd_pp * xq_pp
        # Approximate iq from stator
        psi_q_pp_old = x[4]
        rhs_d = vd + psi_q_pp_old
        rhs_q = vq - psi_d_pp
        iq_eq = (xd_pp * rhs_d - ra * rhs_q) / det

        x[4] = -(xq - xq_pp) * iq_eq
        return x

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                       Efd_eff: float, clamped: bool = False) -> np.ndarray:
        x  = x_slice.copy()
        p  = self.params
        ra = p.get('ra', 0.0)
        xd  = p['xd']; xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']
        xl    = p['xl']
        k_d   = (xd_pp - xl) / (xd_p - xl)

        psi_q_pp = x[4]
        psi_d_pp_old = x[2]*k_d + x[3]*(1.0-k_d)
        det   = ra**2 + xd_pp * xq_pp
        rhs_d = vd + psi_q_pp
        rhs_q = vq - psi_d_pp_old
        id_net = (-ra * rhs_d - xq_pp * rhs_q) / det

        Eq_p_new  = Efd_eff - (xd - xd_p) * id_net
        psi_d_new = Eq_p_new - (xd_p - xd_pp) * id_net
        x[2] = Eq_p_new
        x[3] = psi_d_new
        return x
