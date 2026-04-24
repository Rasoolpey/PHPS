import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent
from src.components.exciters.exdc2 import compute_saturation_coeffs


class Ieeex1(PowerComponent):
    """
    IEEE Type DC1A Exciter (IEEEX1) -- 5-state implementation.

    Similar to EXDC2 but with:
      - Voltage-dependent regulator limits: VRMAX_eff = VRMAX * Vt
      - Lead-lag compensation (TC/TB) before regulator
      - Speed compensation on output: Efd = Vp * omega

    States: [Vm, LLx, Vr, Vp, Vf]
      Vm  -- voltage transducer (1st-order lag, TR)
      LLx -- lead-lag compensator state (TC/TB)
      Vr  -- regulator output (KA/TA, V-dependent anti-windup)
      Vp  -- exciter output (KE + saturation SE, TE)
      Vf  -- rate feedback washout state (KF1/TF1)

    Signal path:
      Vterm -> [1/(1+s*TR)] -> Vm
      Vf_out = KF1 * (Vp - Vf) / TF1        (washout feedback)
      Verr = Vref - Vm - Vf_out
      Verr -> [(1+s*TC)/(1+s*TB)] -> LLx_out (lead-lag)
      LLx_out -> [KA/(1+s*TA)] -> Vr         (V-dependent limits)
      Vr -> [1/(KE + SE(Vp) + s*TE)] -> Vp   (exciter with saturation)
      Efd = Vp * omega                        (speed compensation)
      Vp -> [1/(1+s*TF1)] -> Vf              (feedback state)

    Ports in:  [Vterm, Vref, omega]
    Port out:  [Efd]
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [('Vterm', 'signal', 'pu'),
                    ('Vref', 'signal', 'pu'),
                    ('omega', 'flow', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'LLx', 'Vr', 'Vp', 'Vf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':    'Voltage transducer time constant [s]',
            'TC':    'Lead-lag numerator time constant [s]',
            'TB':    'Lead-lag denominator time constant [s]',
            'KA':    'Regulator gain',
            'TA':    'Regulator time constant [s]',
            'VRMAX': 'Max regulator output (multiplied by Vt) [pu]',
            'VRMIN': 'Min regulator output (multiplied by Vt) [pu]',
            'TE':    'Exciter time constant [s]',
            'KE':    'Exciter self-excitation constant',
            'KF1':   'Rate feedback gain',
            'TF1':   'Rate feedback time constant [s]',
            'SAT_A': 'Saturation coefficient A (pre-computed from E1/SE1/E2/SE2)',
            'SAT_B': 'Saturation coefficient B (pre-computed from E1/SE1/E2/SE2)',
        }

    _DEFAULTS = {'TC': 0.0, 'TB': 0.0}

    def __init__(self, name: str, params: Dict[str, Any]):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        if 'SAT_A' not in params or 'SAT_B' not in params:
            E1  = params.pop('E1', 0.0)
            SE1 = params.pop('SE1', 0.0)
            E2  = params.pop('E2', 0.0)
            SE2 = params.pop('SE2', 0.0)
            A, B = compute_saturation_coeffs(E1, SE1, E2, SE2)
            params['SAT_A'] = A
            params['SAT_B'] = B
        super().__init__(name, params)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage', 'unit': 'pu',
                    'cpp_expr': 'x[3] * inputs[2]'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vp    = x[3];
            double omega = inputs[2];
            outputs[0] = Vp * omega;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vp  = x[3];
            double Vf  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double omega = inputs[2];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Rate feedback washout: Vf_out = KF1 * (Vp - Vf) / TF1
            double TF1_eff = (TF1 > 1e-4) ? TF1 : 1e-4;
            double Vf_out = KF1 * (Vp - Vf) / TF1_eff;

            // 3. Voltage error
            double Verr = Vref - Vm - Vf_out;

            // 4. Lead-lag compensator: (1 + s*TC) / (1 + s*TB)
            double LLx_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                LLx_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                LLx_out = (TC > 1e-4) ? TC * Verr : Verr;
            }

            // 5. Regulator with voltage-dependent anti-windup limits
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double vr_max = VRMAX * Vterm;
            double vr_min = VRMIN * Vterm;
            double dVr = (KA * LLx_out - Vr) / TA_eff;
            if (Vr >= vr_max && dVr > 0.0) dVr = 0.0;
            if (Vr <= vr_min && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Exciter with saturation: SE(Vp) = B*(|Vp|-A)^2/|Vp|
            double Vp_abs = (Vp > 0.0) ? Vp : -Vp;
            double SE = 0.0;
            if (SAT_B > 0.0 && Vp_abs > SAT_A) {
                SE = SAT_B * (Vp_abs - SAT_A) * (Vp_abs - SAT_A) / Vp_abs;
            }
            double TE_eff = (TE > 1e-4) ? TE : 1e-4;
            dxdt[3] = (Vr - (KE + SE) * Vp) / TE_eff;

            // 7. Rate feedback filter state: d(Vf)/dt = (Vp - Vf) / TF1
            dxdt[4] = (Vp - Vf) / TF1_eff;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd = Vp * omega.  At initialization omega=1.0, so Efd = Vp."""
        return float(x_slice[self.state_schema.index('Vp')])

    def efd_output_expr(self, state_offset: int) -> str:
        vp_i = self.state_schema.index('Vp')
        return f"x[{state_offset + vp_i}]"

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Back-solve 5 states: [Vm, LLx, Vr, Vp, Vf] from Efd target.

        At equilibrium with omega=1.0:
          Vp = Efd (since Efd = Vp * omega)
          Vf = Vp  (washout equilibrium, so Vf_out = 0)
          SE = saturation(Vp)
          Vr = (KE + SE) * Vp
          LLx = Verr = Vr / KA  (lead-lag at equilibrium)
          Vref = Vm + Vr / KA
        """
        Efd_req = float(targets.get('Efd', 1.0))
        Vt      = float(targets.get('Vt',  1.0))
        omega   = float(targets.get('omega', 1.0))
        p = self.params

        KA    = float(p.get('KA', 40.0))
        KE    = float(p.get('KE', 1.0))
        VRMAX = float(p.get('VRMAX', 8.0))
        VRMIN = float(p.get('VRMIN', -8.0))
        SAT_A = float(p.get('SAT_A', 0.0))
        SAT_B = float(p.get('SAT_B', 0.0))

        Vp_eq = Efd_req / omega if omega > 0.1 else Efd_req
        self.params['Efd_eff'] = float(Efd_req)

        Vp_abs = abs(Vp_eq)
        SE = 0.0
        if SAT_B > 0.0 and Vp_abs > SAT_A:
            SE = SAT_B * (Vp_abs - SAT_A) ** 2 / Vp_abs

        Vr_eq = (KE + SE) * Vp_eq
        Vr_eq = max(min(Vr_eq, VRMAX * Vt), VRMIN * Vt)

        Vf_eq  = Vp_eq
        LLx_eq = Vr_eq / KA if KA > 1e-6 else 0.0
        Vm     = Vt

        self.params['Vref'] = Vm + (Vr_eq / KA if KA > 1e-6 else 0.0)

        return np.array([Vm, LLx_eq, Vr_eq, Vp_eq, Vf_eq])
