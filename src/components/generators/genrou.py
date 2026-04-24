import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent

class GenRou(PowerComponent):
    """
    Round Rotor Generator Model (GENROU) implementation for C++ code generation.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu'),
                ('Tm', 'effort', 'pu'),
                ('Efd', 'effort', 'pu')
            ],
            'out': [
                ('Id',    'flow', 'pu'),    # Network-frame I_Re (for Norton injection)
                ('Iq',    'flow', 'pu'),    # Network-frame I_Im (for Norton injection)
                ('omega', 'flow', 'pu'),
                ('Pe',    'flow', 'pu'),
                ('Qe',    'flow', 'pu'),
                ('id_dq', 'flow', 'pu'),   # dq-frame d-axis current (for exciter VE)
                ('iq_dq', 'flow', 'pu'),   # dq-frame q-axis current (for exciter VE)
                ('It_Re', 'flow', 'pu'),   # Actual terminal current Re (RI frame)
                ('It_Im', 'flow', 'pu'),   # Actual terminal current Im (RI frame)
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['delta', 'omega', 'E_q_prime', 'psi_d', 'psi_q', 'E_d_prime']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H': 'Inertia constant',
            'D': 'Damping coefficient',
            'ra': 'Stator resistance',
            'xd': 'd-axis synchronous reactance',
            'xq': 'q-axis synchronous reactance',
            'xd_prime': 'd-axis transient reactance',
            'xq_prime': 'q-axis transient reactance',
            'xd_double_prime': 'd-axis sub-transient reactance',
            'xq_double_prime': 'q-axis sub-transient reactance',
            'Td0_prime': 'd-axis transient open-circuit time constant',
            'Tq0_prime': 'q-axis transient open-circuit time constant',
            'Td0_double_prime': 'd-axis sub-transient open-circuit time constant',
            'Tq0_double_prime': 'q-axis sub-transient open-circuit time constant',
            'xl': 'Leakage reactance',
            'omega_b': 'Base frequency [rad/s]'
        }
        
    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'delta_deg': {'description': 'Rotor angle', 'unit': 'deg', 'cpp_expr': 'x[0] * 180.0 / 3.14159265359'},
            'Te': {'description': 'Electrical Torque', 'unit': 'pu', 'cpp_expr': 'outputs[3]'},
            'Tm_in': {'description': 'Mechanical Torque input', 'unit': 'pu', 'cpp_expr': 'inputs[2]'},
            'Pe': {'description': 'Active Power', 'unit': 'pu', 'cpp_expr': 'outputs[3]'},
            'Qe': {'description': 'Reactive Power', 'unit': 'pu', 'cpp_expr': 'outputs[4]'},
            'V_term': {'description': 'Terminal Voltage', 'unit': 'pu', 'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])'},
            'Eq_p':   {'description': 'q-axis transient EMF (field flux proxy)', 'unit': 'pu', 'cpp_expr': 'x[2]'},
            'omega':  {'description': 'Rotor speed', 'unit': 'pu', 'cpp_expr': 'x[1]'},
            'H_mech': {'description': 'Mechanical kinetic energy H*(ω-1)²', 'unit': 'pu',
                       'cpp_expr': f'{float(self.params["H"])} * (x[1]-1.0)*(x[1]-1.0)'},
            'H_field':{'description': 'Field magnetic energy Eq_p²/(2*(xd-xl))', 'unit': 'pu',
                       'cpp_expr': f'0.5 * x[2]*x[2] / ({float(self.params["xd"]) - float(self.params["xl"])})'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            double Ed_p = x[5];
            double Eq_p = x[2];
            double psi_d = x[3];
            double psi_q = x[4];
            
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);
            
            // Stator: vd = -ra*id + xq''*iq - psi_q_pp
            //         vq = -ra*iq - xd''*id + psi_d_pp
            // A = [-ra, xq''; -xd'', -ra], b = [-psi_q_pp; psi_d_pp]
            // Norton: I_no = inv(A)*(-b) = inv(A)*[psi_q_pp; -psi_d_pp]
            
            double det = ra*ra + xd_double_prime * xq_double_prime;
            double id_no = (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;
            
            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;
            
            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;   // dq-frame d-axis current (for exciter VE/load comp)
            outputs[6] = iq_no;   // dq-frame q-axis current
            outputs[7] = I_Re;    // placeholder until step updates with actual current
            outputs[8] = I_Im;
        """

    def get_cpp_step_code(self) -> str:
        return """
            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm = inputs[2];
            double Efd = inputs[3];
            
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;
            
            double omega = x[1];
            double Eq_p = x[2];
            double psi_d = x[3];
            double psi_q = x[4];
            double Ed_p = x[5];
            
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);
            
            // Stator: vd = -ra*id + xq''*iq - psi_q_pp
            //         vq = -ra*iq - xd''*id + psi_d_pp
            double rhs_d = vd + psi_q_pp;
            double rhs_q = vq - psi_d_pp;
            double det = ra*ra + xd_double_prime * xq_double_prime;
            
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det;
            
            double Te = vd*id + vq*iq;
            
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = (Tm - Te - D*(omega - 1.0)) / (2.0 * H);
            
            
            dxdt[2] = (Efd - Eq_p - (xd - xd_prime)*(id)) / Td0_prime;
            dxdt[3] = ( -psi_d - (xd_prime - xd_double_prime)*(id) + Eq_p ) / Td0_double_prime;
            dxdt[4] = ( -psi_q - (xq_prime - xq_double_prime)*(iq) - Ed_p ) / Tq0_double_prime;
            dxdt[5] = ( -Ed_p + (xq - xq_prime)*(iq) ) / Tq0_prime;
            
            outputs[3] = vd*id + vq*iq;
            outputs[4] = vq*id - vd*iq;
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;
            outputs[8] = -id * cos_d + iq * sin_d;
        """

    # ------------------------------------------------------------------ #
    # Port-Hamiltonian interface (used by src/dirac/)                      #
    # ------------------------------------------------------------------ #

    def hamiltonian(self, x: np.ndarray) -> float:
        """Evaluate the GENROU Hamiltonian H(x).

        H = H_mech + H_field + H_damper_d + H_damper_q

        where:
            H_mech     = H · (ω − 1)²          (kinetic energy in pu-s²)
            H_field    = Eq'² / [2(xd − xl)]    (field magnetic energy)
            H_damper_d = ψ_d² / [2(xd'− xd'')]  (d-axis damper)
            H_damper_q = ψ_q² / [2(xq'− xq'')]  (q-axis damper)
            H_Ed       = Ed'² / [2(xq − xl)]    (q-axis transient)

        The rotor angle δ is a *cyclic* coordinate — it does not appear
        in the Hamiltonian (energy does not depend on δ).
        """
        p = self.params
        # x = [delta, omega, Eq', psi_d, psi_q, Ed']
        omega = x[1]
        Eq_p  = x[2]
        psi_d = x[3]
        psi_q = x[4]
        Ed_p  = x[5]

        H_val  = p['H'] * (omega - 1.0) ** 2
        H_val += Eq_p ** 2 / (2.0 * (p['xd'] - p['xl']))
        H_val += Ed_p ** 2 / (2.0 * (p['xq'] - p['xl']))

        denom_d = p['xd_prime'] - p['xd_double_prime']
        if abs(denom_d) > 1e-12:
            H_val += psi_d ** 2 / (2.0 * denom_d)

        denom_q = p['xq_prime'] - p['xq_double_prime']
        if abs(denom_q) > 1e-12:
            H_val += psi_q ** 2 / (2.0 * denom_q)

        return float(H_val)

    def grad_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        """∂H/∂x for GENROU (6 components)."""
        p = self.params
        g = np.zeros(6)
        # ∂H/∂δ = 0  (cyclic coordinate)
        g[0] = 0.0
        # ∂H/∂ω = 2H(ω − 1)
        g[1] = 2.0 * p['H'] * (x[1] - 1.0)
        # ∂H/∂Eq' = Eq' / (xd − xl)
        g[2] = x[2] / (p['xd'] - p['xl'])
        # ∂H/∂ψ_d = ψ_d / (xd' − xd'')
        denom_d = p['xd_prime'] - p['xd_double_prime']
        g[3] = x[3] / denom_d if abs(denom_d) > 1e-12 else 0.0
        # ∂H/∂ψ_q = ψ_q / (xq' − xq'')
        denom_q = p['xq_prime'] - p['xq_double_prime']
        g[4] = x[4] / denom_q if abs(denom_q) > 1e-12 else 0.0
        # ∂H/∂Ed' = Ed' / (xq − xl)
        g[5] = x[5] / (p['xq'] - p['xl'])
        return g

    def get_phs_matrices(self, x: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Return PHS structure matrices (J, R, g, Q) for GENROU.

        The dynamics are formulated as:
            ẋ = (J − R) Q ∂H/∂x + g u

        where u = [Vd, Vq, Tm, Efd] and the stator algebraic equations
        (coupling to the network) appear through g and the input u.

        Note: The full GENROU model has stator algebraic constraints that
        make the exact PHS factorisation state-dependent.  Here we provide
        the *intrinsic* J, R capturing the natural energy flow between the
        6 ODE states. The stator coupling is captured in g.
        """
        p = self.params
        n = 6   # states: [delta, omega, Eq', psi_d, psi_q, Ed']
        m = 4   # inputs: [Vd, Vq, Tm, Efd]

        # --- Q: energy-variable scaling (diagonal) ---
        Q = np.eye(n)
        # Q maps state variables to co-energy variables: Q ∂H/∂x
        # For GENROU Q = I because the Hamiltonian is already in state coords.

        # --- J: skew-symmetric interconnection ---
        J = np.zeros((n, n))
        # delta-omega coupling: ẋ₀ = ω_b (ω − 1)
        # In PHS form: J[0,1] * ∂H/∂ω = ω_b * 2H(ω-1) → need J[0,1] = ω_b/(2H)
        # But the swing equation is:  dδ/dt = ω_b(ω-1),  d(ω)/dt = (Tm-Te-D(ω-1))/(2H)
        # With ∂H/∂ω = 2H(ω-1), we get:
        #   J[0,1] = ω_b / (2H)  → dδ/dt = J[0,1] * ∂H/∂ω = ω_b(ω-1)  ✓
        #   J[1,0] = -ω_b / (2H) → skew-symmetric
        wb = p['omega_b']
        H_inertia = p['H']
        J[0, 1] =  wb / (2.0 * H_inertia)
        J[1, 0] = -wb / (2.0 * H_inertia)

        # --- R: dissipation ---
        R = np.zeros((n, n))
        # Damping on omega: D(ω-1)/(2H) dissipates kinetic energy
        R[1, 1] = p['D'] / (2.0 * H_inertia)

        # Field winding dissipation: dEq'/dt has -Eq'/Td0' term
        R[2, 2] = 1.0 / p['Td0_prime']
        # Damper winding d-axis: dψ_d/dt has -ψ_d/Td0'' term
        R[3, 3] = 1.0 / p['Td0_double_prime']
        # Damper winding q-axis: dψ_q/dt has -ψ_q/Tq0'' term
        R[4, 4] = 1.0 / p['Tq0_double_prime']
        # q-axis transient: dEd'/dt has -Ed'/Tq0' term
        R[5, 5] = 1.0 / p['Tq0_prime']

        # --- g: input port matrix ---
        # Coupling to inputs [Vd, Vq, Tm, Efd]
        # The exact form depends on the stator algebraic equations,
        # which are state-dependent.  We provide the direct terms:
        g = np.zeros((n, m))
        # Tm → dω/dt: +Tm/(2H)
        g[1, 2] = 1.0 / (2.0 * H_inertia)
        # Efd → dEq'/dt: +Efd/Td0'
        g[2, 3] = 1.0 / p['Td0_prime']

        return {'J': J, 'R': R, 'g': g, 'Q': Q}

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'generator'

    def _kfactors(self):
        """Sub-transient blending constants (k_d, k_q)."""
        p = self.params
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xd_p  = p['xd_prime'];        xq_p  = p['xq_prime']
        xl    = p['xl']
        return (xd_pp - xl) / (xd_p - xl), (xq_pp - xl) / (xq_p - xl)

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        """RI-frame Norton current from sub-transient fluxes (voltage-independent)."""
        p = self.params
        delta = x_slice[0]
        Eq_p = x_slice[2]; psi_d = x_slice[3]
        psi_q = x_slice[4]; Ed_p = x_slice[5]
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
        psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q)
        det = ra**2 + xd_pp * xq_pp
        id_no = (-ra * psi_q_pp + xq_pp * psi_d_pp) / det
        iq_no = ( xd_pp * psi_q_pp + ra * psi_d_pp) / det
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                       -id_no * cos_d + iq_no * sin_d)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        """Solve stator equations for actual (id, iq) at terminal vd, vq."""
        p = self.params
        Eq_p = x_slice[2]; psi_d = x_slice[3]
        psi_q = x_slice[4]; Ed_p = x_slice[5]
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
        psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q)
        det = ra**2 + xd_pp * xq_pp
        rhs_d = vd + psi_q_pp
        rhs_q = vq - psi_d_pp
        id_act = (-ra * rhs_d - xq_pp * rhs_q) / det
        iq_act = ( xd_pp * rhs_d - ra * rhs_q) / det
        return id_act, iq_act

    def rebalance_for_bus_voltage(self, x_slice: np.ndarray,
                                  V_bus_complex: complex) -> tuple:
        """Compute DAE-consistent Te, Efd targets at the full-bus voltage.

        Called by DiracRunner._rebalance_te_for_fullbus() after the full Y-bus
        network solve overwrites bus voltages with values that differ slightly
        from the Kron-reduced initialisation voltages.

        Keeps the existing rotor angle delta — only re-evaluates the stator
        algebraic equations at the new (vd, vq) to get the correct Te and
        Efd_req, which are then used to reset governor Pref and exciter Vref.

        Returns
        -------
        x_new : ndarray
            State vector (unchanged — delta and flux states are already refined).
        targets : dict
            ``{'Tm': Te, 'Efd': Efd_req, 'Vt': |V|, 'omega': 1.0,
               'vd': vd, 'vq': vq, 'id': id_act, 'iq': iq_act}``
        """
        p = self.params
        delta = float(x_slice[0])
        Eq_p  = float(x_slice[2])

        # Park transform: RI → dq using the existing rotor angle
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        vd = float(V_bus_complex.real) * sin_d - float(V_bus_complex.imag) * cos_d
        vq = float(V_bus_complex.real) * cos_d + float(V_bus_complex.imag) * sin_d

        # Stator algebraic solve at the DAE bus voltage
        id_act, iq_act = self.compute_stator_currents(x_slice, vd, vq)

        # Electrical torque (matches C++ formula: Te = vd*id + vq*iq)
        Te = vd * id_act + vq * iq_act

        # Field-winding SS condition: dEq_p/dt = 0  →  Efd = Eq_p + (xd-xd')*id
        xd   = float(p['xd'])
        xd_p = float(p['xd_prime'])
        Efd_req = Eq_p + (xd - xd_p) * id_act

        targets = {
            'Tm':    Te,
            'Efd':   Efd_req,
            'Vt':    abs(V_bus_complex),
            'omega': 1.0,
            'vd':    vd,
            'vq':    vq,
            'id':    id_act,
            'iq':    iq_act,
            'vd_ri': float(V_bus_complex.real),
            'vq_ri': float(V_bus_complex.imag),
        }
        return x_slice.copy(), targets

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """Init all 6 states from power-flow terminal phasor."""
        p = self.params
        ra    = p.get('ra', 0.0)
        xd    = p['xd'];    xq    = p['xq']
        xd_p  = p['xd_prime']; xq_p  = p['xq_prime']
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']

        # Rotor angle from q-axis phasor
        Eq_phasor = V_phasor + (ra + 1j * xq) * I_phasor
        delta = float(np.angle(Eq_phasor))

        # Park transform: exp(-j*(delta - pi/2))
        dq_factor = np.exp(-1j * (delta - math.pi / 2))
        vd, vq   = float((V_phasor * dq_factor).real), float((V_phasor * dq_factor).imag)
        id_val   = float((I_phasor * dq_factor).real)
        iq_val   = float((I_phasor * dq_factor).imag)

        k_d, k_q = self._kfactors()

        # Stator-consistent sub-transient fluxes
        psi_d_pp = vq + ra * iq_val + xd_pp * id_val
        psi_q_pp = -(vd + ra * id_val - xq_pp * iq_val)

        # d-axis transient states  (dxdt[2]=0, dxdt[3]=0)
        Eq_p  = psi_d_pp + (1.0 - k_d) * (xd_p - xd_pp) * id_val
        psi_d = Eq_p - (xd_p - xd_pp) * id_val

        # q-axis transient states (consistent with stator algebraic psi_q_pp,
        # analogous to d-axis: Eq_p = psi_d_pp + (1-k_d)*(xd_p-xd_pp)*id)
        Ed_p  = -psi_q_pp - (1.0 - k_q) * (xq_p - xq_pp) * iq_val
        psi_q = -Ed_p - (xq_p - xq_pp) * iq_val

        # Efd target (motor convention: id > 0 overexcited)
        Efd_req = Eq_p + (xd - xd_p) * id_val
        Tm_req  = vd * id_val + vq * iq_val

        targets = {
            'Efd': float(Efd_req), 'Tm': float(Tm_req),
            'Vt':  float(abs(V_phasor)),
            'omega': 1.0,
            'vd':  vd,    'vq':  vq,
            'id':  id_val,'iq':  iq_val,
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
        }
        return np.array([delta, 1.0, Eq_p, psi_d, psi_q, Ed_p]), targets

    def refine_q_axis(self, x_slice: np.ndarray,
                      vd: float, vq: float) -> np.ndarray:
        """Closed-form update of q-axis states Ed_p and psi_q."""
        x = x_slice.copy()
        p = self.params
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xq_p  = p['xq_prime'];        xq    = p['xq']
        ra    = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()

        Eq_p  = x[2]; psi_d = x[3]
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)

        # Equilibrium iq from stator d-equation with q-axis at equilibrium
        # psi_q_pp = -C_q * iq_eq  where C_q = (xq-xq') + (xq'-xq'')*(1-k_q)
        C_q  = (xq - xq_p) + (p['xq_prime'] - xq_pp) * (1.0 - k_q)
        det  = ra**2 + xd_pp * xq_pp
        iq_eq = (xd_pp * vd - ra * (vq - psi_d_pp)) / (det + xd_pp * C_q)

        Ed_p_new  = (xq - xq_p) * iq_eq
        psi_q_new = -Ed_p_new - (xq_p - xq_pp) * iq_eq

        x[5] = Ed_p_new
        x[4] = psi_q_new
        return x

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                               vd: float, vq: float) -> np.ndarray:
        """Direct equilibrium update of all generator states at the Kron-network
        terminal voltage (vd, vq).
        """
        x = self.refine_q_axis(x_slice, vd, vq)
        return x

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                      Efd_eff: float, clamped: bool = False) -> np.ndarray:
        """Update Eq_p and psi_d from Efd_eff at terminal vd/vq.

        clamped=True: solve the 2×2 stator+d-axis system (for hard-clamped Efd).
        clamped=False: solve stator equations for id_act, set equilibrium states.
        Returns new x_slice (caller applies under-relaxation).
        """
        x = x_slice.copy()
        p = self.params
        xd    = p['xd']; xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']
        ra    = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()

        if clamped:
            # 2×2: stator d-eqn + d-axis equilibrium substituted into stator q-eqn
            Ed_p    = x[5]; psi_q_s = x[4]
            psi_q_pp = -Ed_p * k_q + psi_q_s * (1.0 - k_q)
            A = k_d * (xd - xd_p) + (1.0 - k_d) * (xd - xd_pp)
            # [stator d]: -ra*id + xq''*iq = vd + psi_q_pp
            # [stator q after d-axis subst]: (A+xd'')*id + ra*iq = Efd_eff - vq
            a11 = -ra;       a12 = xq_pp
            a21 = A + xd_pp; a22 = ra
            b1  = vd + psi_q_pp
            b2  = Efd_eff - vq
            det_s = a11 * a22 - a12 * a21
            if abs(det_s) < 1e-10:
                return x
            id_net = (a22 * b1 - a12 * b2) / det_s
        else:
            # Normal: solve stator for id_act
            Eq_p  = x[2]; psi_d = x[3]
            Ed_p  = x[5]; psi_q_s = x[4]
            psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
            psi_q_pp = -Ed_p * k_q + psi_q_s * (1.0 - k_q)
            det      = ra**2 + xd_pp * xq_pp
            rhs_d    = vd + psi_q_pp
            rhs_q    = vq - psi_d_pp
            id_net   = (-ra * rhs_d - xq_pp * rhs_q) / det

        Eq_p_new  = Efd_eff - (xd - xd_p) * id_net
        psi_d_new = Eq_p_new - (xd_p - xd_pp) * id_net
        x[2] = Eq_p_new
        x[3] = psi_d_new
        return x
