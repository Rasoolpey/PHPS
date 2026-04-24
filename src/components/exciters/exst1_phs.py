"""
EXST1 — Port-Hamiltonian Formulation.

IEEE Type ST1A Exciter with explicit (J, R, H) structure.

States: x = [Vm, LLx, Vr, Vf]
  Vm   – voltage transducer output (1st-order lag at TR)
  LLx  – lead-lag compensator state (TB time constant)
  Vr   – voltage regulator output (KA gain + TA lag)
  Vf   – washout rate feedback filter state (KF/TF)

Efd = Vr with dynamic ceiling.

Storage function (identity metric):
    H = ½||x||²

Dissipation matrix R (diagonal):
    R = diag(1/TR, 1/TB, KA/TA, 1/TF)

References
----------
- IEEE Std 421.5-2016, Type ST1A
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


class Exst1PHS(PowerComponent):
    """
    IEEE EXST1 exciter in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    Storage: H = ½||x||² (identity metric)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [('Vterm', 'signal', 'pu'), ('Vref', 'signal', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'LLx', 'Vr', 'Vf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':    'Voltage transducer time constant [s]',
            'KA':    'Regulator gain',
            'TA':    'Regulator time constant [s]',
            'TC':    'Lead-lag numerator time constant [s]',
            'TB':    'Lead-lag denominator time constant [s]',
            'VRMAX': 'Max regulator output [pu]',
            'VRMIN': 'Min regulator output [pu]',
            'VIMAX': 'Max input limiter [pu]',
            'VIMIN': 'Min input limiter [pu]',
            'KC':    'Commutation reactance factor (ceiling reduction)',
            'KF':    'Rate feedback gain',
            'TF':    'Rate feedback time constant [s]',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage (=Vr)', 'unit': 'pu',
                    'cpp_expr': 'x[2]'},
            'H_exc': {'description': 'Exciter storage function', 'unit': 'pu',
                      'cpp_expr': '(0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]))'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        """

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        Vm, LLx, Vr, Vf = sp.symbols('V_m LLx V_r V_f')
        states = [Vm, LLx, Vr, Vf]

        # Input symbols
        Vterm, Vref = sp.symbols('V_{term} V_{ref}')
        inputs = [Vterm, Vref]

        # Parameter symbols
        TR_s = sp.Symbol('T_R', positive=True)
        KA_s = sp.Symbol('K_A', positive=True)
        TA_s = sp.Symbol('T_A', positive=True)
        TB_s = sp.Symbol('T_B', positive=True)
        TF_s = sp.Symbol('T_F', positive=True)

        params = {
            'TR': TR_s, 'KA': KA_s, 'TA': TA_s, 'TB': TB_s, 'TF': TF_s,
        }

        # Hamiltonian: H = ½||x||² (identity metric)
        H_expr = sp.Rational(1, 2) * (Vm**2 + LLx**2 + Vr**2 + Vf**2)

        # Dissipation matrix R (diagonal)
        R = sp.diag(1/TR_s, 1/TB_s, KA_s/TA_s, 1/TF_s)

        # Interconnection matrix J (skew-symmetric)
        J = sp.zeros(4, 4)
        J[2, 1] = KA_s / (TA_s * TB_s)
        J[1, 2] = -J[2, 1]
        J[3, 2] = 1 / (TA_s * TF_s)
        J[2, 3] = -J[3, 2]

        # Input coupling
        g = sp.zeros(4, 2)
        g[0, 0] = 1 / TR_s   # Vterm → dVm/dt

        sphs = SymbolicPHS(
            name='EXST1_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            description=(
                'IEEE Type ST1A Exciter in Port-Hamiltonian form. '
                'Identity-weighted storage function H = ½||x||².'
            ),
        )

        # Anti-windup limiter on regulator Vr (dynamic ceiling — bounds
        # depend on load current via KC, so the hand-written step code is
        # retained;  annotation is declarative documentation for now)
        sphs.add_limiter(state_idx=2, upper_bound='VRMAX', lower_bound='VRMIN')

        # Lead-lag on LLx (state 1) — documentary; hand-written step code
        # retained due to input limiter (VIMAX/VIMIN) and dynamic ceiling (KC)
        Verr_expr = Vref - Vm
        sphs.add_lead_lag(state_idx=1, input_expr=Verr_expr,
                          output_var='vll_out', tc_param='TC', tb_param='TB')

        def equilibrium_solver(targets: dict, param_values: dict):
            Efd_req = float(targets.get('Efd', 1.0))
            Vt = float(targets.get('Vt', 1.0))
            KA = float(param_values.get('KA', 50.0))
            VRMAX = float(param_values.get('VRMAX', 5.0))
            VRMIN = float(param_values.get('VRMIN', -5.0))
            VIMAX = float(param_values.get('VIMAX', 0.5))
            VIMIN = float(param_values.get('VIMIN', -0.5))

            Efd_eff = max(min(Efd_req, VRMAX), VRMIN)
            Vr_eq = Efd_eff
            vi_eq = max(min((Vr_eq / KA) if KA > 1e-6 else 0.0, VIMAX), VIMIN)
            LLx_eq = vi_eq
            Vf_eq = Vr_eq
            Vm_eq = Vt
            return np.array([Vm_eq, LLx_eq, Vr_eq, Vf_eq]), {'Vref': Vm_eq + vi_eq}

        sphs.set_init_spec(
            target_states={},
            input_bindings={},
            free_param_map={},
            solver_func=equilibrium_solver,
            post_init_func=lambda x0, free_params, targets, param_values: {
                'Efd_eff': float(self.compute_efd_output(x0))
            },
        )

        return sphs

    # ------------------------------------------------------------------ #
    # Initialization (same as legacy Exst1)                                #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd = Vr (state[2])."""
        return float(x_slice[self.state_schema.index('Vr')])

    def efd_output_expr(self, state_offset: int) -> str:
        vr_i = self.state_schema.index('Vr')
        return f"x[{state_offset + vr_i}]"
