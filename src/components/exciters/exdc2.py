import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


def compute_saturation_coeffs(E1, SE1, E2, SE2):
    """Pre-compute quadratic saturation coefficients A, B from IEEE data points.

    SE(E) = B * (E - A)^2 / E  when E > A, else 0.
    Returns (A, B).  When saturation is disabled (E1=0), returns (0, 0).
    """
    if E1 == 0 or E2 == 0 or abs(E2 - E1) < 1e-10:
        return 0.0, 0.0
    sqrt_SE1E1 = math.sqrt(SE1 * E1) if SE1 * E1 > 0 else 0.0
    sqrt_SE2E2 = math.sqrt(SE2 * E2) if SE2 * E2 > 0 else 0.0
    if abs(sqrt_SE2E2 - sqrt_SE1E1) < 1e-10:
        return 0.0, 0.0
    A = (E1 * sqrt_SE2E2 - E2 * sqrt_SE1E1) / (sqrt_SE2E2 - sqrt_SE1E1)
    B_denom = E1 - A
    if abs(B_denom) < 1e-10:
        return 0.0, 0.0
    B = (sqrt_SE1E1 / B_denom) ** 2
    return A, B


class Exdc2(PowerComponent):
    """
    IEEE Type DC2A Exciter (EXDC2) — 4-state implementation with saturation.

    States: [Vm, Vr, Efd, Xf]
      Vm  – voltage transducer output (1st-order lag at TR)
      Vr  – amplifier/regulator output (KA gain + TA lag, VRMAX/VRMIN limits)
      Efd – exciter field voltage (exciter dynamics with KE + saturation SE)
      Xf  – rate feedback filter state (KF/TF1 derivative feedback)

    Signal path:
      Vterm -> [1/(1+s*TR)] -> Vm
      Vf = KF * (Efd - Xf) / TF1            (washout rate feedback)
      Verr = Vref - Vm - Vf
      Verr -> [KA/(1+s*TA)] -> Vr            (VRMAX/VRMIN anti-windup)
      Vr -> [1/(KE + SE(Efd) + s*TE)] -> Efd (exciter with saturation)
      Efd -> [1/(1+s*TF1)] -> Xf             (feedback filter state)

    Output: Efd (state[2])

    Ports in:  [Vterm, Vref]
    Port out:  [Efd]
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [('Vterm', 'signal', 'pu'), ('Vref', 'signal', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'Vr', 'Efd', 'Xf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':    'Voltage transducer time constant [s]',
            'KA':    'Regulator gain',
            'TA':    'Regulator time constant [s]',
            'KE':    'Exciter self-excitation constant',
            'TE':    'Exciter time constant [s]',
            'KF':    'Rate feedback gain',
            'TF1':   'Rate feedback time constant [s]',
            'VRMAX': 'Max regulator output [pu]',
            'VRMIN': 'Min regulator output [pu]',
            'SAT_A': 'Saturation coefficient A (pre-computed)',
            'SAT_B': 'Saturation coefficient B (pre-computed)',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage', 'unit': 'pu', 'cpp_expr': 'x[2]'}
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            outputs[0] = x[2];
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double Vm  = x[0];
            double Vr  = x[1];
            double Efd = x[2];
            double Xf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Rate feedback washout: Vf = KF * (Efd - Xf) / TF1
            double TF1_eff = (TF1 > 1e-4) ? TF1 : 1e-4;
            double Vf = KF * (Efd - Xf) / TF1_eff;

            // 3. Voltage error
            double Verr = Vref - Vm - Vf;

            // 4. Amplifier/regulator with anti-windup limits
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * Verr - Vr) / TA_eff;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[1] = dVr;

            // 5. Exciter with saturation: SE(Efd) = B*(|Efd|-A)^2/|Efd|
            double Efd_abs = (Efd > 0.0) ? Efd : -Efd;
            double SE = 0.0;
            if (SAT_B > 0.0 && Efd_abs > SAT_A) {
                SE = SAT_B * (Efd_abs - SAT_A) * (Efd_abs - SAT_A) / Efd_abs;
            }
            double TE_eff = (TE > 1e-4) ? TE : 1e-4;
            dxdt[2] = (Vr - (KE + SE) * Efd) / TE_eff;

            // 6. Rate feedback filter state
            dxdt[3] = (Efd - Xf) / TF1_eff;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd state (state[2])."""
        return float(x_slice[self.state_schema.index('Efd')])

    def efd_output_expr(self, state_offset: int) -> str:
        efd_i = self.state_schema.index('Efd')
        return f"x[{state_offset + efd_i}]"

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Back-solve 4 states: [Vm, Vr, Efd, Xf] from Efd target."""
        Efd_req = float(targets.get('Efd', 1.0))
        Vt      = float(targets.get('Vt',  1.0))
        p     = self.params
        KA    = float(p.get('KA', 20.0)); KE    = float(p.get('KE', 1.0))
        VRMAX = float(p.get('VRMAX', 5.0)); VRMIN = float(p.get('VRMIN', -5.0))
        SAT_A = float(p.get('SAT_A', 0.0)); SAT_B = float(p.get('SAT_B', 0.0))

        Efd_eq  = float(Efd_req)
        self.params['Efd_eff'] = Efd_eq
        Efd_abs = abs(Efd_eq)
        SE = 0.0
        if SAT_B > 0.0 and Efd_abs > SAT_A:
            SE = SAT_B * (Efd_abs - SAT_A) ** 2 / Efd_abs

        Vr_eq  = max(min((KE + SE) * Efd_eq, VRMAX), VRMIN)
        Xf_eq  = Efd_eq
        Vm     = Vt
        self.params['Vref'] = Vm + (Vr_eq / KA if KA > 1e-6 else 0.0)

        return np.array([Vm, Vr_eq, Efd_eq, Xf_eq])
