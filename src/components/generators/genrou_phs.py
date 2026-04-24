"""
GENROU — Port-Hamiltonian Formulation.

6th-order round-rotor synchronous machine with explicit (J, R, H) structure.

States: x = [δ, ω, E'_q, ψ_d, E'_d, ψ_q]

Hamiltonian (total stored energy):
    H = H_inertia·(ω − 1)² + E'_q² / [2(x_d − x_l)]
      + ψ_d² / [2(x'_d − x''_d)] + E'_d² / [2(x_q − x_l)]
      + ψ_q² / [2(x'_q − x''_q)]

Gradient:
    ∂H/∂x = [0, 2H(ω−1), E'_q/(x_d−x_l), ψ_d/(x'_d−x''_d),
              E'_d/(x_q−x_l), ψ_q/(x'_q−x''_q)]

PHS Dynamics:
    ẋ = [J(x) − R] ∇H + g(x) · u

where u = [Tm, Efd] are the control inputs (from governor and exciter),
and the network coupling (V_d, V_q → i_d, i_q) enters through the
stator algebraic constraint.

Ports:
    - in:  [Vd, Vq, Tm, Efd]  (network voltages + control inputs)
    - out: [Id, Iq, omega, Pe, Qe, id_dq, iq_dq, It_Re, It_Im]

The stator algebraic equations v = −Z·i + e'' are handled by the
DAE compiler as network constraints, not inside the PHS.

References
----------
- Stegink, De Persis, van der Schaft (2017), "A unifying energy-based
  approach to stability of power grids with market dynamics", IEEE TAC.
- Fiaz, Zonetti, Ortega, Scherpen, van der Schaft (2013),
  "A port-Hamiltonian approach to power network modelling and analysis",
  European Journal of Control.
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


class GenRouPHS(PowerComponent):
    """
    GENROU in Port-Hamiltonian form: ẋ = (J − R)∇H + g·u.

    Dynamics are generated as explicit PHS C++ code where the (J, R, H)
    structure is visible and the energy Hamiltonian is a first-class quantity.
    """

    # ------------------------------------------------------------------ #
    # Schema definitions (identical port/state interface as legacy GenRou) #
    # ------------------------------------------------------------------ #

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
                ('Id',    'flow', 'pu'),
                ('Iq',    'flow', 'pu'),
                ('omega', 'flow', 'pu'),
                ('Pe',    'flow', 'pu'),
                ('Qe',    'flow', 'pu'),
                ('id_dq', 'flow', 'pu'),
                ('iq_dq', 'flow', 'pu'),
                ('It_Re', 'flow', 'pu'),
                ('It_Im', 'flow', 'pu'),
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['delta', 'omega', 'E_q_prime', 'psi_d', 'E_d_prime', 'psi_q']

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
            'delta_deg': {
                'description': 'Rotor angle', 'unit': 'deg',
                'cpp_expr': 'x[0] * 180.0 / 3.14159265359'
            },
            'Te': {
                'description': 'Electrical Torque', 'unit': 'pu',
                'cpp_expr': 'outputs[3]'
            },
            'Tm_in': {
                'description': 'Mechanical Torque input', 'unit': 'pu',
                'cpp_expr': 'inputs[2]'
            },
            'Pe': {
                'description': 'Active Power', 'unit': 'pu',
                'cpp_expr': 'outputs[3]'
            },
            'Qe': {
                'description': 'Reactive Power', 'unit': 'pu',
                'cpp_expr': 'outputs[4]'
            },
            'V_term': {
                'description': 'Terminal Voltage', 'unit': 'pu',
                'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])'
            },
            'Eq_p': {
                'description': 'q-axis transient EMF', 'unit': 'pu',
                'cpp_expr': 'x[2]'
            },
            'omega': {
                'description': 'Rotor speed', 'unit': 'pu',
                'cpp_expr': 'x[1]'
            },
            'H_total': {
                'description': 'Total Hamiltonian', 'unit': 'pu',
                'cpp_expr': self._hamiltonian_cpp_expr()
            },
        }

    # ------------------------------------------------------------------ #
    # PHS C++ code generation                                             #
    # ------------------------------------------------------------------ #

    def get_cpp_compute_outputs_code(self) -> str:
        """Norton current injection (same as legacy — network-frame)."""
        return """
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
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
        """PHS dynamics: ẋ = (J − R) ∇H + g · u.

        The Hamiltonian energy variables and structure matrices are
        computed explicitly so the energy-based structure is preserved
        in the generated C++.

        Stator algebraic coupling:
            The stator equations vd = −ra·id + xq''·iq − ψq''
                                 vq = −ra·iq − xd''·id + ψd''
            are solved for (id, iq) as functions of (x, Vd, Vq).
            These currents then appear in the PHS through the port
            coupling matrix g(x) · u, where u effectively includes
            the current-dependent coupling terms.

        The derivation ensures that the total energy balance is:
            dH/dt = −∇Hᵀ R ∇H + (Tm − D·Δω)·Δω + Efd·(Eq'/(xd−xl))/Td0'
                    − stator_dissipation
        which is the correct passivity inequality.
        """
        return r"""
            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        """

    # ------------------------------------------------------------------ #
    # Python-side PHS interface (for analysis tools)                       #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        delta, omega = sp.symbols('delta omega')
        Eq_p, psi_d = sp.symbols("E'_q psi_d")
        Ed_p, psi_q = sp.symbols("E'_d psi_q")
        states = [delta, omega, Eq_p, psi_d, Ed_p, psi_q]

        # Input symbols
        Tm, Efd = sp.symbols('T_m E_{fd}')
        inputs = [Tm, Efd]

        # Parameter symbols
        H_s = sp.Symbol('H', positive=True)
        D_s = sp.Symbol('D', nonnegative=True)
        xd_s = sp.Symbol('x_d', positive=True)
        xq_s = sp.Symbol('x_q', positive=True)
        xd_p_s = sp.Symbol("x'_d", positive=True)
        xq_p_s = sp.Symbol("x'_q", positive=True)
        xd_pp_s = sp.Symbol("x''_d", positive=True)
        xq_pp_s = sp.Symbol("x''_q", positive=True)
        xl_s = sp.Symbol('x_l', positive=True)
        Td0_p_s = sp.Symbol("T'_{d0}", positive=True)
        Td0_pp_s = sp.Symbol("T''_{d0}", positive=True)
        Tq0_p_s = sp.Symbol("T'_{q0}", positive=True)
        Tq0_pp_s = sp.Symbol("T''_{q0}", positive=True)
        wb_s = sp.Symbol('omega_b', positive=True)

        params = {
            'H': H_s, 'D': D_s, 'xd': xd_s, 'xq': xq_s,
            'xd_prime': xd_p_s, 'xq_prime': xq_p_s,
            'xd_double_prime': xd_pp_s, 'xq_double_prime': xq_pp_s,
            'xl': xl_s, 'Td0_prime': Td0_p_s, 'Td0_double_prime': Td0_pp_s,
            'Tq0_prime': Tq0_p_s, 'Tq0_double_prime': Tq0_pp_s,
            'omega_b': wb_s,
        }

        # Hamiltonian
        H_expr = (H_s * (omega - 1)**2
                  + Eq_p**2 / (2 * (xd_s - xl_s))
                  + psi_d**2 / (2 * (xd_p_s - xd_pp_s))
                  + Ed_p**2 / (2 * (xq_s - xl_s))
                  + psi_q**2 / (2 * (xq_p_s - xq_pp_s)))

        # J matrix (skew-symmetric)
        J = sp.zeros(6, 6)
        J[0, 1] = wb_s / (2 * H_s)
        J[1, 0] = -J[0, 1]
        J[3, 2] = (xd_s - xl_s) / Td0_pp_s
        J[2, 3] = -J[3, 2]
        J[5, 4] = -(xq_s - xl_s) / Tq0_pp_s
        J[4, 5] = -J[5, 4]

        # R matrix (positive semi-definite dissipation)
        R = sp.zeros(6, 6)
        R[1, 1] = D_s / (4 * H_s**2)
        R[2, 2] = (xd_s - xl_s) / Td0_p_s
        R[3, 3] = (xd_p_s - xd_pp_s) / Td0_pp_s
        R[4, 4] = (xq_s - xl_s) / Tq0_p_s
        R[5, 5] = (xq_p_s - xq_pp_s) / Tq0_pp_s

        # g matrix (input coupling)
        g = sp.zeros(6, 2)
        g[1, 0] = 1 / (2 * H_s)    # Tm → dω/dt
        g[2, 1] = 1 / Td0_p_s       # Efd → dEq'/dt

        return SymbolicPHS(
            name='GENROU_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            description=(
                '6th-order round-rotor synchronous generator (GENROU) '
                'in Port-Hamiltonian form. Stator coupling (id, iq) '
                'enters through DAE algebraic constraints, not through '
                'these intrinsic PHS matrices.'
            ),
        )

    # ------------------------------------------------------------------ #
    # Initialization (reuse legacy GenRou logic exactly)                   #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'generator'

    def _kfactors(self):
        p = self.params
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xd_p = p['xd_prime']; xq_p = p['xq_prime']
        xl = p['xl']
        return (xd_pp - xl) / (xd_p - xl), (xq_pp - xl) / (xq_p - xl)

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        p = self.params
        delta = x_slice[0]
        omega = x_slice[1]
        Eq_p = x_slice[2]; psi_d = x_slice[3]
        Ed_p = x_slice[4]; psi_q = x_slice[5]
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
        psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q)
        det = ra**2 + xd_pp * xq_pp
        id_no = omega * (-ra * psi_q_pp + xq_pp * psi_d_pp) / det
        iq_no = omega * ( xd_pp * psi_q_pp + ra * psi_d_pp) / det
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                       -id_no * cos_d + iq_no * sin_d)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        p = self.params
        omega = x_slice[1]
        Eq_p = x_slice[2]; psi_d = x_slice[3]
        Ed_p = x_slice[4]; psi_q = x_slice[5]
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
        psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q)
        det = ra**2 + xd_pp * xq_pp
        rhs_d = vd + omega * psi_q_pp
        rhs_q = vq - omega * psi_d_pp
        id_act = (-ra * rhs_d - xq_pp * rhs_q) / det
        iq_act = ( xd_pp * rhs_d - ra * rhs_q) / det
        return id_act, iq_act

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        p = self.params
        ra = p.get('ra', 0.0)
        xd = p['xd']; xq = p['xq']
        xd_p = p['xd_prime']; xq_p = p['xq_prime']
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']

        Eq_phasor = V_phasor + (ra + 1j * xq) * I_phasor
        delta = float(np.angle(Eq_phasor))

        dq_factor = np.exp(-1j * (delta - math.pi / 2))
        vd = float((V_phasor * dq_factor).real)
        vq = float((V_phasor * dq_factor).imag)
        id_val = float((I_phasor * dq_factor).real)
        iq_val = float((I_phasor * dq_factor).imag)

        k_d, k_q = self._kfactors()

        psi_d_pp = vq + ra * iq_val + xd_pp * id_val
        psi_q_pp = -(vd + ra * id_val - xq_pp * iq_val)

        Eq_p = psi_d_pp + (1.0 - k_d) * (xd_p - xd_pp) * id_val
        psi_d = Eq_p - (xd_p - xd_pp) * id_val

        # q-axis transient states (consistent with stator algebraic psi_q_pp,
        # analogous to d-axis: Eq_p = psi_d_pp + (1-k_d)*(xd_p-xd_pp)*id)
        Ed_p = -psi_q_pp - (1.0 - k_q) * (xq_p - xq_pp) * iq_val
        # psi_q from ODE steady-state: dpsi_q/dt=0 => psi_q = -Ed' - (xq'-xq'')iq
        psi_q = -Ed_p - (xq_p - xq_pp) * iq_val

        Efd_req = Eq_p + (xd - xd_p) * id_val
        Tm_req = vd * id_val + vq * iq_val

        targets = {
            'Efd': float(Efd_req), 'Tm': float(Tm_req),
            'Vt': float(abs(V_phasor)),
            'omega': 1.0,
            'vd': vd, 'vq': vq,
            'id': id_val, 'iq': iq_val,
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
        }
        # Note: state order is [delta, omega, Eq', psi_d, Ed', psi_q]
        return np.array([delta, 1.0, Eq_p, psi_d, Ed_p, psi_q]), targets

    def rebalance_for_bus_voltage(self, x_slice: np.ndarray,
                                  V_bus_complex: complex) -> tuple:
        """Rebalance ALL generator flux states (including Eq') at a
        DAE-consistent bus voltage.

        Given the current generator states and a new complex bus voltage,
        derive the fully self-consistent equilibrium where ALL six flux
        derivatives are zero.  The rotor angle (delta) is held fixed from
        the Kron initialisation; everything else — Eq', psi_d, Ed', psi_q,
        and the resulting Te and Efd — is recomputed.

        Algorithm (one-shot, no iteration required):

        1. Analytical 2×2 solve for (id, iq) using the *current* Eq'
           (preserves psi_d'' exactly — stator currents are stable).
        2. Derive the self-consistent Eq' from the stator equation:
              psi_d'' = vq + ra·iq + xd''·id
              Eq'_new = psi_d'' + alpha_d·id
           Since psi_d'' is invariant under the (Eq', psi_d) update,
           the stator currents from step 1 remain valid.
        3. Update Ed', psi_d, psi_q for equilibrium at (id, iq).

        Returns
        -------
        x_new : np.ndarray
            Updated state vector with ALL flux states at equilibrium.
        targets : dict
            Keys: 'Tm', 'Efd', 'Vt', 'omega'.
        """
        import math as _math

        p = self.params
        V_Re = V_bus_complex.real
        V_Im = V_bus_complex.imag
        Vt = abs(V_bus_complex)

        xd   = p['xd'];       xq   = p['xq']
        xd_p = p['xd_prime'];  xq_p = p['xq_prime']
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xl   = p['xl']
        ra   = p.get('ra', 0.0)

        # Park transform using current rotor angle
        delta = x_slice[0]
        sin_d = _math.sin(delta)
        cos_d = _math.cos(delta)
        vd = V_Re * sin_d - V_Im * cos_d
        vq = V_Re * cos_d + V_Im * sin_d

        k_d, k_q = self._kfactors()
        Eq_p = x_slice[2]         # current Eq' (from Kron-init)

        # ---- Effective damper-winding coupling constants ----
        if abs(xd_p - xl) > 1e-12:
            alpha_d = (xd_p - xd_pp) ** 2 / (xd_p - xl)
        else:
            alpha_d = 0.0

        if abs(xq_p - xl) > 1e-12:
            beta_q = ((xq - xq_p) * k_q
                      + (xq - xq_pp) * (1.0 - k_q))
        else:
            beta_q = 0.0

        # ---- Step 1: Analytical 2×2 solve for (id, iq) ----
        # Uses current Eq' to form the q-axis RHS.  This is stable
        # regardless of machine impedance ratios.
        det_ra = ra * ra + xd_pp * xq_pp

        A11 = det_ra + xq_pp * alpha_d
        A12 = -ra * beta_q
        A21 = ra * alpha_d
        A22 = det_ra + xd_pp * beta_q

        b1 = -ra * vd - xq_pp * (vq - Eq_p)
        b2 =  xd_pp * vd - ra * (vq - Eq_p)

        D = A11 * A22 - A12 * A21
        id_eq = (A22 * b1 - A12 * b2) / D
        iq_eq = (A11 * b2 - A21 * b1) / D

        # ---- Step 2: Derive self-consistent Eq' ----
        # From the stator q-axis algebraic equation:
        #   psi_d'' = vq + ra·iq + xd''·id
        # And the equilibrium relation:
        #   Eq' = psi_d'' + alpha_d·id
        # Since psi_d'' = Eq'_old·k_d + psi_d_old·(1-k_d) is the value
        # used by the 2×2 solve, and the update (Eq'_new, psi_d_new)
        # preserves psi_d'', the stator currents remain valid.
        psi_d_pp = vq + ra * iq_eq + xd_pp * id_eq
        Eq_p_new = psi_d_pp + alpha_d * id_eq

        # ---- Step 3: Update ALL flux states for equilibrium ----
        x_new = x_slice.copy()
        x_new[2] = Eq_p_new                                        # Eq'
        x_new[3] = Eq_p_new - (xd_p - xd_pp) * id_eq              # psi_d
        x_new[4] = (xq - xq_p) * iq_eq                            # Ed'
        x_new[5] = -x_new[4] - (xq_p - xq_pp) * iq_eq            # psi_q

        # Electrical torque
        Te = vd * id_eq + vq * iq_eq

        # Efd required for field-winding equilibrium:
        #   dEq'/dt = 0  →  Efd = Eq' + (xd - xd') * id
        Efd_req = Eq_p_new + (xd - xd_p) * id_eq

        targets = {
            'Tm':    float(Te),
            'Efd':   float(Efd_req),
            'Vt':    float(Vt),
            'omega': float(x_new[1]),
            'vd':    float(vd),
            'vq':    float(vq),
            'id':    float(id_eq),
            'iq':    float(iq_eq),
            'vd_ri': float(V_Re),
            'vq_ri': float(V_Im),
        }

        return x_new, targets

    def adjust_for_target_voltage(self, x_slice: np.ndarray,
                                   V_bus_complex: complex,
                                   V_target_mag: float,
                                   y_diag: float = 0.0,
                                   flux_update: bool = True) -> np.ndarray:
        """Adjust Eq\u2032/psi_d (d-axis) and optionally Ed\u2032/psi_q (q-axis)
        to drive terminal voltage toward *V_target_mag*.

        Parameters
        ----------
        flux_update : bool
            When True (default, post-convergence pass), also writes Ed\u2032
            and psi_q from the current stator currents for ODE steady-
            state consistency.  When False (inside the PV-bus iteration),
            only Eq\u2032 and psi_d are touched \u2014 the q-axis flux is left at
            its Kron-init value.

            The q-axis rewrite ``psi_q = -Ed' - (xq' - xq'')\u00b7iq_act``
            forms a *divergent* fixed-point map when iterated against
            the network solve: iq_act depends on psi_q via psi_q_pp, and
            the loop gain is \u2248 ``-(xq - xq'') / xq''``, which is ~-3 for
            typical GENROU params. So we must not iterate this rewrite;
            one final pass after the d-axis converges is enough.

        The d-axis update is a damped increment ``Eq' += gamma \u00b7 dV``.
        A small fixed fraction (not a Newton step) is used because the
        true ``d|V|/dEq'`` depends on cross-coupling through ``Y^{-1}``
        in multi-generator systems. The caller also applies an outer
        \u03b1-relaxation on ``dV``.
        """
        import math as _math

        x = x_slice.copy()
        V_mag = abs(V_bus_complex)
        dV = V_target_mag - V_mag
        if abs(dV) < 1e-8 and not flux_update:
            return x

        p = self.params
        delta = x[0]
        sin_d = _math.sin(delta)
        cos_d = _math.cos(delta)
        V_Re = V_bus_complex.real
        V_Im = V_bus_complex.imag
        vd = V_Re * sin_d - V_Im * cos_d
        vq = V_Re * cos_d + V_Im * sin_d

        # Current stator currents at present voltage
        id_act, iq_act = self.compute_stator_currents(x, vd, vq)

        xd_p  = p['xd_prime']
        xd_pp = p['xd_double_prime']

        # --- d-axis: damped increment on Eq' with 0.2 pu clamp ---
        if abs(dV) >= 1e-8:
            gamma = 0.3
            step = max(-0.2, min(0.2, gamma * dV))
            x[2] += step
            x[3] = x[2] - (xd_p - xd_pp) * id_act        # psi_d (ODE SS)

        # --- q-axis: only update in the post-convergence pass ---
        if flux_update:
            xq   = p['xq']
            xq_p = p['xq_prime']
            xq_pp = p['xq_double_prime']
            x[4] = (xq - xq_p) * iq_act                  # Ed'
            x[5] = -x[4] - (xq_p - xq_pp) * iq_act       # psi_q

        return x

    def refine_q_axis(self, x_slice: np.ndarray,
                      vd: float, vq: float) -> np.ndarray:
        x = x_slice.copy()
        p = self.params
        xd_pp = p['xd_double_prime']; xq_pp = p['xq_double_prime']
        xq_p = p['xq_prime']; xq = p['xq']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()

        Eq_p = x[2]; psi_d = x[3]
        psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)

        C_q = (xq - xq_p) + (p['xq_prime'] - xq_pp) * (1.0 - k_q)
        det = ra**2 + xd_pp * xq_pp
        iq_eq = (xd_pp * vd - ra * (vq - psi_d_pp)) / (det + xd_pp * C_q)

        Ed_p_new = (xq - xq_p) * iq_eq
        psi_q_new = -Ed_p_new - (xq_p - xq_pp) * iq_eq

        x[4] = Ed_p_new   # Ed' at index 4
        x[5] = psi_q_new  # psi_q at index 5
        return x

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                               vd: float, vq: float) -> np.ndarray:
        return self.refine_q_axis(x_slice, vd, vq)

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                      Efd_eff: float, clamped: bool = False) -> np.ndarray:
        x = x_slice.copy()
        p = self.params
        xd = p['xd']; xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
        xq_pp = p['xq_double_prime']
        ra = p.get('ra', 0.0)
        k_d, k_q = self._kfactors()

        if clamped:
            Ed_p = x[4]; psi_q_s = x[5]
            psi_q_pp = -Ed_p * k_q + psi_q_s * (1.0 - k_q)
            A = k_d * (xd - xd_p) + (1.0 - k_d) * (xd - xd_pp)
            a11 = -ra;       a12 = xq_pp
            a21 = A + xd_pp; a22 = ra
            b1 = vd + psi_q_pp
            b2 = Efd_eff - vq
            det_s = a11 * a22 - a12 * a21
            if abs(det_s) < 1e-10:
                return x
            id_net = (a22 * b1 - a12 * b2) / det_s
        else:
            Eq_p = x[2]; psi_d = x[3]
            Ed_p = x[4]; psi_q_s = x[5]
            psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d)
            psi_q_pp = -Ed_p * k_q + psi_q_s * (1.0 - k_q)
            det = ra**2 + xd_pp * xq_pp
            rhs_d = vd + psi_q_pp
            rhs_q = vq - psi_d_pp
            id_net = (-ra * rhs_d - xq_pp * rhs_q) / det

        Eq_p_new = Efd_eff - (xd - xd_p) * id_net
        psi_d_new = Eq_p_new - (xd_p - xd_pp) * id_net
        x[2] = Eq_p_new
        x[3] = psi_d_new
        return x
