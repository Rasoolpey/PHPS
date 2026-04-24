import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class GenTpf(PowerComponent):
    """
    Round Rotor Generator Model (GENTPF) with multiplicative saturation.

    Port-Hamiltonian 6th-order synchronous machine model. GENTPF is the
    "Type-F" generator model from PowerWorld/WECC/John Undrill (2007/2012).
    It uses multiplicative (global) saturation applied to all inductance terms
    simultaneously, which more accurately reflects the physics of magnetic
    saturation than the additive approach used in GENROU/GENSAL.

    State vector:
        x = [delta, omega, E''q, E'q, E'd, E''d]

    where E''q = psi_d (d-axis subtransient flux = Eq'' in GENTPF notation),
          E''d = psi_q (q-axis subtransient flux, sign: E''d = -psi_q'').

    Key difference from GENROU:
    - Saturation factors Sat_d, Sat_q multiply ALL flux terms simultaneously
    - Because Xq'' ≠ Xd'' (including sat) the network interface cannot use a
      simple circuit model — the terminal voltages are computed directly:
          Vqterm = E''q*(1+ω) - Id*Xd''sat - Iq*Ra
          Vdterm = E''d*(1+ω) + Iq*Xq''sat - Id*Ra
    - No Kis parameter (armature current saturation) — that is GENTPJ

    Physics reference:
        D.W. Olive (1966, 1968), J. Undrill WECC note (2007/2012),
        PowerWorld GENTPF/GENTPJ document (Weber, 2015).

    Port-Hamiltonian interpretation
    --------------------------------
    H = H_mech + H_d + H_q
    The saturation enters as state-dependent inductances, modifying the
    effective dissipation and interconnection matrices at each time step.
    Passivity is preserved because Sat_d, Sat_q ≥ 1 (inductances are
    reduced, not increased, by saturation).
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
                ('Id',    'flow', 'pu'),
                ('Iq',    'flow', 'pu'),
                ('omega', 'flow', 'pu'),
                ('Pe',    'flow', 'pu'),
                ('Qe',    'flow', 'pu'),
                ('id_dq', 'flow', 'pu'),
                ('iq_dq', 'flow', 'pu'),
                ('It_Re', 'flow', 'pu'),
                ('It_Im', 'flow', 'pu'),
            ],
        }

    @property
    def state_schema(self) -> List[str]:
        # 6 states — expressed in GENTPF notation
        # x[2] = E''q  (d-axis subtransient EMF = +psi_d'')
        # x[3] = E'q   (d-axis transient EMF)
        # x[4] = E'd   (q-axis transient EMF)
        # x[5] = E''d  (q-axis subtransient EMF = -psi_q'')
        return ['delta', 'omega', 'E_q_pp', 'E_q_prime', 'E_d_prime', 'E_d_pp']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H':                'Inertia constant',
            'D':                'Damping coefficient',
            'ra':               'Stator resistance',
            'xd':               'd-axis synchronous reactance',
            'xq':               'q-axis synchronous reactance',
            'xd_prime':         'd-axis transient reactance',
            'xq_prime':         'q-axis transient reactance',
            'xd_double_prime':  'd-axis sub-transient reactance',
            'xq_double_prime':  'q-axis sub-transient reactance',
            'Td0_prime':        'd-axis transient open-circuit time constant',
            'Tq0_prime':        'q-axis transient open-circuit time constant',
            'Td0_double_prime': 'd-axis sub-transient open-circuit time constant',
            'Tq0_double_prime': 'q-axis sub-transient open-circuit time constant',
            'xl':               'Leakage reactance',
            # Saturation curve: scaled-quadratic A, B coefficients
            # Pre-computed by Python from S10/S12 and stored in params
            # (json_compat or user must supply A_sat and B_sat directly,
            #  OR supply S10/S12 and call compute_sat_params() before compile)
            'A_sat':            'Saturation coefficient A (scaled-quadratic)',
            'B_sat':            'Saturation coefficient B (scaled-quadratic)',
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
    #  Saturation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sat_coeffs_from_s(S10: float, S12: float):
        """Compute A_sat, B_sat for scaled-quadratic Sat(x)=B*(x-A)^2/x."""
        if S10 <= 0.0 or S12 <= 0.0:
            return 0.0, 0.0
        ratio = math.sqrt(1.2 * S12 / S10)
        if abs(ratio - 1.0) < 1e-10:
            return 0.0, 0.0
        u = 0.2 / (ratio - 1.0)
        A = 1.0 - u
        B = S10 / (u * u)
        return A, B

    def _sat_at(self, psi: float) -> float:
        """Evaluate scaled-quadratic saturation function Sat(psi)."""
        A = self.params.get('A_sat', 0.0)
        B = self.params.get('B_sat', 0.0)
        if psi <= A or psi < 1e-6 or B <= 0.0:
            return 0.0
        return B * (psi - A) ** 2 / psi

    # ------------------------------------------------------------------
    #  Shared C++ saturation + saturation-factor helper code
    # ------------------------------------------------------------------

    _CPP_SAT_HELPER = r"""
            // ---- Air-gap flux for saturation argument ----
            // psi_ag = sqrt((Vqterm + Iq*Ra + Id*Xl)^2 + (Vdterm + Id*Ra - Iq*Xl)^2)
            // Using terminal voltages at this time step:
            double psi_ag_d = vq + iq * ra + id * xl;
            double psi_ag_q = vd + id * ra - iq * xl;
            double psi_ag   = sqrt(psi_ag_d*psi_ag_d + psi_ag_q*psi_ag_q);

            // Scaled-quadratic saturation: Sat(psi) = B_sat*(psi-A_sat)^2/psi
            double sat_val = 0.0;
            if (psi_ag > A_sat && psi_ag > 1e-6) {
                double diff = psi_ag - A_sat;
                sat_val = B_sat * diff * diff / psi_ag;
            }

            // Saturation factors (GENTPF, no Kis)
            double Sat_d = 1.0 + sat_val;
            double Sat_q = 1.0 + (xq / xd) * sat_val;

            // Saturated subtransient reactances (network interface)
            // X''dsat = (X''d - Xl)/Sat_d + Xl
            // X''qsat = (X''q - Xl)/Sat_q + Xl
            double Xd_pp_sat = (xd_double_prime - xl) / Sat_d + xl;
            double Xq_pp_sat = (xq_double_prime - xl) / Sat_q + xl;
    """

    # ------------------------------------------------------------------
    #  compute_outputs: Norton injection (voltage-independent)
    #  For GENTPF we use E''q, E''d directly as EMF sources
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
            double delta  = x[0];
            double sin_d  = sin(delta);
            double cos_d  = cos(delta);

            double Eq_pp  = x[2];   // E''q = +psi_d''
            double Ed_pp  = x[5];   // E''d = -psi_q''

            // Norton estimate: use unsaturated reactances (sat unknown without Vterm)
            // This is sufficient for the network solve convergence.
            double det   = ra*ra + xd_double_prime * xq_double_prime;
            // psi_d'' = +E''q,  psi_q'' = -E''d
            double psi_d_pp_no = Eq_pp;
            double psi_q_pp_no = -Ed_pp;
            double id_no = (-ra * psi_q_pp_no + xq_double_prime * psi_d_pp_no) / det;
            double iq_no = ( xd_double_prime * psi_q_pp_no + ra * psi_d_pp_no) / det;

            double I_Re =  id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm   = inputs[2];
            double Efd  = inputs[3];

            double delta  = x[0];
            double omega  = x[1];
            double Eq_pp  = x[2];   // E''q
            double Eq_p   = x[3];   // E'q
            double Ed_p   = x[4];   // E'd
            double Ed_pp  = x[5];   // E''d

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // Park transform RI → dq
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // ---- GENTPF network interface (direct, no circuit model) ----
            // First-pass stator solve using current state (unsaturated reactances)
            // to get approximate id/iq for air-gap flux computation.
            // Then recompute with saturated reactances.

            // Pass 1: unsaturated solve
            {
                double det1 = ra*ra + xd_double_prime * xq_double_prime;
                // Vqterm = E''q*(1+ω) - Id*X''d - Iq*Ra
                // Vdterm = E''d*(1+ω) + Iq*X''q - Id*Ra
                double Vq_emf = Eq_pp * (1.0 + omega);
                double Vd_emf = Ed_pp * (1.0 + omega);
                double rhs_q  = vq - Vq_emf;   // -Id*X''d - Iq*Ra = rhs_q
                double rhs_d  = vd - Vd_emf;   // +Iq*X''q - Id*Ra = rhs_d
                // [-Ra, -X''d; X''q, -Ra] * [iq; id] = [rhs_q; rhs_d]  -- rewrite:
                // Ra*id + X''d*iq = -rhs_q    (from q-eqn)
                // Ra*id - X''q*iq = -rhs_d ... wait, let's be careful:
                // Vqterm = E''q*(1+ω) - Id*X''dsat - Iq*Ra  =>  Id*X''d + Iq*Ra = E''q*(1+ω) - vq
                // Vdterm = E''d*(1+ω) + Iq*X''qsat - Id*Ra  =>  -Id*Ra + Iq*X''q = vd - E''d*(1+ω)... 
                // Solve: [X''d, Ra; -Ra, X''q] * [id; iq] = [E''q*(1+ω)-vq; vd-E''d*(1+ω)]
                double b1 = Eq_pp * (1.0 + omega) - vq;
                double b2 = vd - Ed_pp * (1.0 + omega);
                double id = ( xq_double_prime * b1 + ra * b2) / det1;
                double iq = (-ra * b1 + xd_double_prime * b2) / det1;

                // Air-gap flux for saturation
                double psi_ag_d = vq + iq * ra + id * xl;
                double psi_ag_q = vd + id * ra - iq * xl;
                double psi_ag   = sqrt(psi_ag_d*psi_ag_d + psi_ag_q*psi_ag_q);

                // Saturation factors
                double sat_val = 0.0;
                if (psi_ag > A_sat && psi_ag > 1e-6) {
                    double diff = psi_ag - A_sat;
                    sat_val = B_sat * diff * diff / psi_ag;
                }
                double Sat_d = 1.0 + sat_val;
                double Sat_q = 1.0 + (xq / xd) * sat_val;

                // Saturated subtransient reactances
                double Xd_pp_sat = (xd_double_prime - xl) / Sat_d + xl;
                double Xq_pp_sat = (xq_double_prime - xl) / Sat_q + xl;
                double det2 = Xd_pp_sat * Xq_pp_sat + ra * ra;

                // Refined stator solve with saturated reactances
                id = ( Xq_pp_sat * b1 + ra * b2) / det2;
                iq = (-ra * b1 + Xd_pp_sat * b2) / det2;

                // Electric torque: Te = psi_d''*iq - psi_q''*id
                //   psi_d'' = E''q,  psi_q'' = -E''d
                double Te = Eq_pp * iq - (-Ed_pp) * id;

                // ---- State derivatives ----
                dxdt[0] = omega_b * (omega - 1.0);
                dxdt[1] = (Tm - Te - D * (omega - 1.0)) / (2.0 * H);

                // GENTPF dynamics (with multiplicative saturation):
                // dE''q/dt = [Sat_d*(E'q - E''q) - (X'd - Xl)*Id] / T''d0
                dxdt[2] = (Sat_d * (Eq_p - Eq_pp) - (xd_prime - xl) * id) / Td0_double_prime;

                // dE'q/dt = [Efd + Sat_q*(E''q*(Xd-X'd)/(X'd-X''d) - E'q*(Xd-X''d)/(X'd-X''d))] / T'd0
                double coeff_qq = (xd - xd_double_prime) / (xd_prime - xd_double_prime);
                double coeff_qp = (xd - xd_prime)        / (xd_prime - xd_double_prime);
                dxdt[3] = (Efd + Sat_q * (-coeff_qq * Eq_p + coeff_qp * Eq_pp)) / Td0_prime;

                // dE'd/dt = [Sat_d*(E''d*(Xq-X'q)/(X'q-X''q) - E'd*(Xq-X''q)/(X'q-X''q))] / T'q0
                double coeff_dq = (xq - xq_double_prime) / (xq_prime - xq_double_prime);
                double coeff_dp = (xq - xq_prime)        / (xq_prime - xq_double_prime);
                dxdt[4] = Sat_d * (-coeff_dq * Ed_p + coeff_dp * Ed_pp) / Tq0_prime;

                // dE''d/dt = [Sat_d*(E'd - E''d) + (X'q - Xl)*Iq] / T''q0
                dxdt[5] = (Sat_d * (Ed_p - Ed_pp) + (xq_prime - xl) * iq) / Tq0_double_prime;

                // Outputs
                outputs[3] = vd*id + vq*iq;
                outputs[4] = vq*id - vd*iq;
                outputs[5] = id;
                outputs[6] = iq;
                outputs[7] = id * sin_d + iq * cos_d;
                outputs[8] = -id * cos_d + iq * sin_d;
            }
        """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @property
    def component_role(self) -> str:
        return 'generator'

    def _sat_value(self, psi: float) -> float:
        A = self.params.get('A_sat', 0.0)
        B = self.params.get('B_sat', 0.0)
        if psi <= A or psi < 1e-6 or B <= 0.0:
            return 0.0
        return B * (psi - A) ** 2 / psi

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        p     = self.params
        delta = x_slice[0]
        Eq_pp = x_slice[2]
        Ed_pp = x_slice[5]
        ra    = p.get('ra', 0.0)
        xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']

        psi_d_pp = Eq_pp
        psi_q_pp = -Ed_pp
        det   = ra**2 + xd_pp * xq_pp
        id_no = (-ra * psi_q_pp + xq_pp * psi_d_pp) / det
        iq_no = (xd_pp * psi_q_pp + ra * psi_d_pp) / det
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                       -id_no * cos_d + iq_no * sin_d)

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """Initialise 6 GENTPF states from power-flow phasor."""
        p     = self.params
        ra    = p.get('ra', 0.0)
        xd    = p['xd'];   xq    = p['xq']
        xd_p  = p['xd_prime'];   xq_p  = p['xq_prime']
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xl    = p['xl']

        # Rotor angle
        Eq_phasor = V_phasor + (ra + 1j * xq) * I_phasor
        delta = float(np.angle(Eq_phasor))

        # Park transform
        dq = np.exp(-1j * (delta - math.pi / 2))
        vd = float((V_phasor * dq).real)
        vq = float((V_phasor * dq).imag)
        id_val = float((I_phasor * dq).real)
        iq_val = float((I_phasor * dq).imag)

        # Air-gap flux for saturation at initial operating point
        psi_ag_d = vq + iq_val * ra + id_val * xl
        psi_ag_q = vd + id_val * ra - iq_val * xl
        psi_ag   = math.sqrt(psi_ag_d**2 + psi_ag_q**2)
        sat_val  = self._sat_value(psi_ag)
        Sat_d    = 1.0 + sat_val
        Sat_q    = 1.0 + (xq / xd) * sat_val

        # Saturated subtransient reactances
        Xd_pp_sat = (xd_pp - xl) / Sat_d + xl
        Xq_pp_sat = (xq_pp - xl) / Sat_q + xl

        # Stator solve with saturated reactances for refined id/iq
        b1 = (xd_pp + ra * 0) * id_val  # placeholder; use direct phasor id/iq
        # Use the phasor-computed id/iq as starting point (sat approx at init)

        # GENTPF subtransient EMFs
        Eq_pp = vq + ra * iq_val + Xd_pp_sat * id_val
        Ed_pp = -(vd + ra * id_val - Xq_pp_sat * iq_val)

        # Transient EMFs from GENTPF equilibrium (dE''q/dt=0, dE''d/dt=0):
        # At init: Sat_d*(E'q - E''q) = (X'd - Xl)*Id
        Eq_p = Eq_pp + (xd_p - xl) * id_val / Sat_d
        Ed_p = Ed_pp - (xq_p - xl) * iq_val / Sat_d

        # Efd from dE'q/dt=0:
        # dE'q/dt = [Efd + Sat_q*(-coeff_qq*E'q + coeff_qp*E''q)] / T'd0 = 0
        coeff_qq = (xd - xd_pp) / (xd_p - xd_pp)
        coeff_qp = (xd - xd_p)  / (xd_p - xd_pp)
        Efd_req = Sat_q * (coeff_qq * Eq_p - coeff_qp * Eq_pp)

        Tm_req = vd * id_val + vq * iq_val

        targets = {
            'Efd': float(Efd_req), 'Tm': float(Tm_req),
            'Vt':  float(abs(V_phasor)),
            'omega': 1.0,
            'vd': vd, 'vq': vq,
            'id': id_val, 'iq': iq_val,
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
        }
        return np.array([delta, 1.0, Eq_pp, Eq_p, Ed_p, Ed_pp]), targets

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> np.ndarray:
        # Lightweight refinement: re-solve transient states from subtransient states
        x  = x_slice.copy()
        p  = self.params
        ra = p.get('ra', 0.0)
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xd_p  = p['xd_prime'];        xq_p  = p['xq_prime']
        xd    = p['xd'];              xq    = p['xq']
        xl    = p['xl']

        Eq_pp = x[2]; Ed_pp = x[5]
        det   = ra**2 + xd_pp * xq_pp
        b1    = Eq_pp - vq   # approximate (ignoring omega correction)
        b2    = vd - Ed_pp
        id_est = (xq_pp * b1 + ra * b2) / det
        iq_est = (-ra * b1 + xd_pp * b2) / det

        psi_ag_d = vq + iq_est * ra + id_est * xl
        psi_ag_q = vd + id_est * ra - iq_est * xl
        psi_ag   = math.sqrt(psi_ag_d**2 + psi_ag_q**2)
        sat_val  = self._sat_value(psi_ag)
        Sat_d    = 1.0 + sat_val

        x[3] = Eq_pp + (xd_p - xl) * id_est / Sat_d
        x[4] = Ed_pp - (xq_p - xl) * iq_est / Sat_d
        return x

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                       Efd_eff: float, clamped: bool = False) -> np.ndarray:
        x  = x_slice.copy()
        p  = self.params
        xd    = p['xd'];   xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']
        ra    = p.get('ra', 0.0)
        xl    = p['xl']

        Eq_pp = x[2]; Ed_pp = x[5]
        det   = ra**2 + xd_pp * xq_pp
        b1    = Eq_pp * 1.0 - vq
        b2    = vd - Ed_pp * 1.0
        id_est = (xq_pp * b1 + ra * b2) / det

        psi_ag_d = vq + ra * ((-ra * b1 + xd_pp * b2) / det) + id_est * xl
        psi_ag_q = vd + ra * id_est - ((-ra * b1 + xd_pp * b2) / det) * xl
        psi_ag   = math.sqrt(psi_ag_d**2 + psi_ag_q**2)
        sat_val  = self._sat_value(psi_ag)
        Sat_d    = 1.0 + sat_val
        Sat_q    = 1.0 + (p['xq'] / p['xd']) * sat_val

        # dE'q/dt=0 at equilibrium
        coeff_qq = (xd - xd_pp) / (xd_p - xd_pp)
        coeff_qp = (xd - xd_p)  / (xd_p - xd_pp)
        # Efd_eff = Sat_q*(coeff_qq*E'q - coeff_qp*E''q)
        # E'q = (Efd_eff/Sat_q + coeff_qp*E''q) / coeff_qq
        if abs(Sat_q * coeff_qq) > 1e-10:
            Eq_p_new = (Efd_eff / Sat_q + coeff_qp * Eq_pp) / coeff_qq
        else:
            Eq_p_new = x[3]
        x[3] = Eq_p_new
        # Update E''q from dE''q/dt=0: Sat_d*(E'q-E''q) = (X'd-Xl)*Id
        x[2] = Eq_p_new - (xd_p - xl) * id_est / Sat_d
        return x
