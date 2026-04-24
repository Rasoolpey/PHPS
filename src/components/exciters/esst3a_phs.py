"""
ESST3A — Port-Hamiltonian Formulation.

IEEE Type ST3A Exciter with explicit (J, R, H) structure.

States: x = [Vm, LLx, Vr, VM, VB]
  Vm   – voltage transducer output (1st-order lag at TR)
  LLx  – lead-lag compensator state (TB time constant)
  Vr   – voltage regulator output (KA gain + TA lag)
  VM   – inner field voltage regulator output (KM gain + TM lag)
  VB   – rectifier terminal voltage (fast lag ~0.01 s)

The field voltage output is Efd = VB * VM (multiplicative coupling).

Storage function (identity metric):
    H = ½||x||²

Dissipation matrix R (diagonal):
    R = diag(1/TR, 1/TB, KA/TA, KM/TM, 1/τ_VB)

The ESST3A extends the EXDC2 PHS pattern with the multiplicative
VB*VM coupling for Efd and the IEEE FEX rectifier loading function.

References
----------
- IEEE Std 421.5-2016, Type ST3A
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


class Esst3aPHS(PowerComponent):
    """
    IEEE ESST3A exciter in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    Storage: H = ½||x||² (identity metric)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'signal', 'pu'),
                ('Vref',  'signal', 'pu'),
                ('id_dq', 'flow',   'pu'),
                ('iq_dq', 'flow',   'pu'),
                ('Vd',    'effort', 'pu'),
                ('Vq',    'effort', 'pu'),
            ],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'LLx', 'Vr', 'VM', 'VB']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':     'Voltage transducer time constant [s]',
            'KA':     'Voltage regulator gain',
            'TA':     'Voltage regulator time constant [s]',
            'TC':     'Lead-lag numerator time constant [s]',
            'TB':     'Lead-lag denominator time constant [s]',
            'VRMAX':  'Max voltage regulator output [pu]',
            'VRMIN':  'Min voltage regulator output [pu]',
            'VIMAX':  'Max input limiter [pu]',
            'VIMIN':  'Min input limiter [pu]',
            'KM':     'Inner field regulator gain',
            'TM':     'Inner field regulator time constant [s]',
            'VMMAX':  'Max inner regulator output [pu]',
            'VMMIN':  'Min inner regulator output [pu]',
            'KG':     'Feedback gain (field voltage feedback)',
            'VGMAX':  'Max feedback voltage [pu]',
            'KP':     'Load compensator voltage gain',
            'KI':     'Load compensator current gain',
            'KC':     'Commutation reactance factor (FEX demagnetization)',
            'VBMAX':  'Max rectifier terminal voltage [pu]',
            'XL':     'Reactance for load compensation [pu]',
            'THETAP': 'Load compensator angle [deg]',
            'Efd_max': 'Hard ceiling for field voltage [pu]',
            'Efd_min': 'Hard floor for field voltage [pu]',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage (=VB*VM)', 'unit': 'pu',
                    'cpp_expr': 'x[4]*x[3]'},
            'H_exc': {'description': 'Exciter storage function', 'unit': 'pu',
                      'cpp_expr': '(0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]+x[4]*x[4]))'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double VB_o = x[4];
            double VM_o = x[3];
            double Efd_o = VB_o * VM_o;
            if (Efd_o > Efd_max) Efd_o = Efd_max;
            if (Efd_o < Efd_min) Efd_o = Efd_min;
            outputs[0] = Efd_o;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // ============================================================
            // ESST3A Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → natural dissipation via R.
            // The signal path (Vm→Verr→LL→Vr→VG→VM→VB→Efd) provides
            // the interconnection structure.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double VM  = x[3];
            double VB  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];
            double iq    = inputs[3];
            double Vd    = inputs[4];
            double Vq    = inputs[5];

            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;

            // 1. Voltage transducer: dVm/dt = (Vterm − Vm) / TR
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA/TA) with anti-windup
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN_fex = KC * XadIfd / VE;
            double FEX;
            if      (IN_fex <= 0.0)    FEX = 1.0;
            else if (IN_fex <= 0.433)  FEX = 1.0 - 0.577 * IN_fex;
            else if (IN_fex <= 0.75)   FEX = sqrt(0.75 - IN_fex * IN_fex);
            else if (IN_fex <= 1.0)    FEX = 1.732 * (1.0 - IN_fex);
            else                       FEX = 0.0;

            // 7. Rectifier voltage VB — fast lag
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;

            // 8. Field voltage feedback
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM/TM) with anti-windup
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        """

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        Vm, LLx, Vr, VM, VB = sp.symbols('V_m LLx V_r V_M V_B')
        states = [Vm, LLx, Vr, VM, VB]

        Vterm, Vref = sp.symbols('V_{term} V_{ref}')
        # ESST3A has 6 inputs in the numerical interface but the
        # dominant port coupling is Vterm and Vref
        inputs = [Vterm, Vref]

        TR_s = sp.Symbol('T_R', positive=True)
        TB_s = sp.Symbol('T_B', positive=True)
        KA_s = sp.Symbol('K_A', positive=True)
        TA_s = sp.Symbol('T_A', positive=True)
        KM_s = sp.Symbol('K_M', positive=True)
        TM_s = sp.Symbol('T_M', positive=True)
        # tau_VB is a fixed internal constant (0.01 s), not a tunable param
        tau_VB_val = sp.Rational(1, 100)  # 0.01

        params = {
            'TR': TR_s, 'TB': TB_s, 'KA': KA_s, 'TA': TA_s,
            'KM': KM_s, 'TM': TM_s,
        }

        H_expr = sp.Rational(1, 2) * (Vm**2 + LLx**2 + Vr**2 + VM**2 + VB**2)

        R_mat = sp.diag(1/TR_s, 1/TB_s, KA_s/TA_s, KM_s/TM_s, 1/tau_VB_val)

        J = sp.zeros(5, 5)
        J[2, 1] = KA_s / (TA_s * TB_s)
        J[1, 2] = -J[2, 1]

        g = sp.zeros(5, 2)
        g[0, 0] = 1 / TR_s

        sphs = SymbolicPHS(
            name='ESST3A_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R_mat, g=g, H=H_expr,
            description=(
                'IEEE Type ST3A Exciter in Port-Hamiltonian form. '
                'Identity-weighted storage function H = ½||x||². '
                'Multiplicative Efd = VB·VM coupling handled externally.'
            ),
        )

        # Anti-windup limiters (hand-written step code retained due to
        # complex pre-clamping, rate limiters, and FEX rectifier logic)
        sphs.add_limiter(state_idx=2, upper_bound='VRMAX', lower_bound='VRMIN')
        sphs.add_limiter(state_idx=3, upper_bound='VMMAX', lower_bound='VMMIN')
        sphs.add_limiter(state_idx=4, upper_bound='VBMAX', lower_bound='0.0')

        # Lead-lag on LLx (state 1) — documentary; hand-written step code
        # retained due to input limiter, FEX rectifier, rate limiter logic
        vi_expr = Vref - Vm
        sphs.add_lead_lag(state_idx=1, input_expr=vi_expr,
                          output_var='LL_y', tc_param='TC', tb_param='TB')

        def equilibrium_solver(targets: dict, param_values: dict):
            Efd_req = float(targets.get('Efd', 1.0))
            Vt = float(targets.get('Vt', 1.0))
            vd = float(targets.get('vd', targets.get('vd_ri', 0.0)))
            vq = float(targets.get('vq', targets.get('vq_ri', Vt)))
            id_val = float(targets.get('id', 0.0))
            iq_val = float(targets.get('iq', 0.0))

            KP = float(param_values.get('KP', 3.67))
            KI = float(param_values.get('KI', 0.435))
            XL = float(param_values.get('XL', 0.0098))
            THETAP = float(param_values.get('THETAP', 0.0))
            KC = float(param_values.get('KC', 0.01))
            KA = float(param_values.get('KA', 20.0))
            KM = float(param_values.get('KM', 8.0))
            KG = float(param_values.get('KG', 1.0))
            VBMAX = float(param_values.get('VBMAX', 5.48))
            VRMAX = float(param_values.get('VRMAX', 99.0))
            VRMIN = float(param_values.get('VRMIN', -99.0))
            VIMAX = float(param_values.get('VIMAX', 0.2))
            VIMIN = float(param_values.get('VIMIN', -0.2))
            VMMAX = float(param_values.get('VMMAX', 99.0))
            VMMIN = float(param_values.get('VMMIN', 0.0))
            VGMAX = float(param_values.get('VGMAX', 3.86))
            Efd_max = float(param_values.get('Efd_max', 5.0))
            Efd_min = float(param_values.get('Efd_min', -1.0))

            theta_rad = THETAP * math.pi / 180.0
            KPC = KP * complex(math.cos(theta_rad), math.sin(theta_rad))
            VE = abs(KPC * complex(vd, vq) + 1j * (KI + KPC * XL) * complex(id_val, iq_val))
            if VE < 1e-6:
                VE = 1e-6

            VMMAX_eff = min(VMMAX, 10.0) if VMMAX > 90.0 else VMMAX
            Efd_eff = max(min(Efd_req, Efd_max), Efd_min)

            IN = KC * abs(Efd_eff) / VE
            if IN <= 0.0:
                FEX = 1.0
            elif IN <= 0.433:
                FEX = 1.0 - 0.577 * IN
            elif IN <= 0.75:
                FEX = math.sqrt(max(0.0, 0.75 - IN ** 2))
            elif IN <= 1.0:
                FEX = 1.732 * (1.0 - IN)
            else:
                FEX = 0.0
            VB_eq = min(max(VE * FEX, 0.0), VBMAX)

            for _ in range(10):
                VM_eq = (Efd_eff / VB_eq) if VB_eq > 1e-6 else 1.0
                VM_eq = max(min(VM_eq, VMMAX_eff), VMMIN)
                XadIfd = VB_eq * VM_eq
                IN_new = KC * max(XadIfd, 0.0) / VE
                if IN_new <= 0.0:
                    FEX_new = 1.0
                elif IN_new <= 0.433:
                    FEX_new = 1.0 - 0.577 * IN_new
                elif IN_new <= 0.75:
                    FEX_new = math.sqrt(max(0.0, 0.75 - IN_new ** 2))
                elif IN_new <= 1.0:
                    FEX_new = 1.732 * (1.0 - IN_new)
                else:
                    FEX_new = 0.0
                VB_new = min(max(VE * FEX_new, 0.0), VBMAX)
                if abs(VB_new - VB_eq) < 1e-10:
                    break
                VB_eq = VB_new

            VM_eq = (Efd_eff / VB_eq) if VB_eq > 1e-6 else 1.0
            VM_eq = max(min(VM_eq, VMMAX_eff), VMMIN)
            VG_eq = max(min(KG * Efd_eff, VGMAX), 0.0)
            vrs_eq = (VM_eq / KM) if KM > 1e-6 else 0.0
            Vr_eq = max(min(vrs_eq + VG_eq, VRMAX), VRMIN)
            vi_eq = max(min((Vr_eq / KA) if KA > 1e-6 else 0.0, VIMAX), VIMIN)
            LLx_eq = vi_eq
            Vm_eq = Vt
            return np.array([Vm_eq, LLx_eq, Vr_eq, VM_eq, VB_eq]), {'Vref': Vm_eq + vi_eq}

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
    # Initialization (same as legacy Esst3a)                               #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd = clip(VB * VM, Efd_min, Efd_max)."""
        vm_i    = self.state_schema.index('VM')
        vb_i    = self.state_schema.index('VB')
        Efd_max = float(self.params.get('Efd_max', 5.0))
        Efd_min = float(self.params.get('Efd_min', -1.0))
        raw     = float(x_slice[vb_i] * x_slice[vm_i])
        return max(min(raw, Efd_max), Efd_min)

    def efd_output_expr(self, state_offset: int) -> str:
        vm_i    = self.state_schema.index('VM')
        vb_i    = self.state_schema.index('VB')
        Efd_max = float(self.params.get('Efd_max', 5.0))
        Efd_min = float(self.params.get('Efd_min', -1.0))
        raw     = f"(x[{state_offset + vb_i}]*x[{state_offset + vm_i}])"
        return f"fmax({Efd_min:.6f}, fmin({Efd_max:.6f}, {raw}))"
