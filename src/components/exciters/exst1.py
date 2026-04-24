import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class Exst1(PowerComponent):
    """
    IEEE Type ST1A Exciter (EXST1) — full 4-state implementation.

    States: [Vm, LLx, Vr, Vf]
      Vm   – voltage transducer output (1st-order lag at TR)
      LLx  – lead-lag compensator state (TB time constant)
      Vr   – voltage regulator output (KA gain + TA lag, limited by VRMAX/VRMIN)
      Vf   – washout rate feedback filter state (KF/TF derivative feedback)

    Efd = Vr with dynamic ceiling: Vr_max_dyn = VRMAX - KC * |Vr|.

    The KF/TF washout feedback provides rate (derivative) action on Efd changes.
    This phase-lead compensation damps inter-area oscillations and is the key
    difference from the simplified 3-state model that caused negative damping.

    Signal path:
      Vterm → [TR lag] → Vm
      Vf_out = KF * (Vr - Vf) / TF   (washout output)
      Verr = clip(Vref - Vm - Vf_out, VIMIN, VIMAX)
      Verr → [TC/TB lead-lag] → vll_out
      vll_out → [KA / (1+s*TA)] → Vr (anti-windup, VRMAX-KC*|Vr|)
      Efd = Vr

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
            'Efd': {'description': 'Field Voltage (=Vr)', 'unit': 'pu', 'cpp_expr': 'x[2]'}
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vr_o  = x[2];
            // Dynamic ceiling reduces with field current (XadIfd ~ |Vr|)
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
            // --- States ---
            double Vm  = x[0];   // voltage transducer
            double LLx = x[1];   // lead-lag state
            double Vr  = x[2];   // voltage regulator
            double Vf  = x[3];   // washout feedback filter

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback output: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            //    State evolves at 1/TB; output has TC/TB ratio
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            //    Dynamic ceiling: VRMAX_dyn = VRMAX - KC * XadIfd
            //    XadIfd approximated as |Vr| (field current ~ regulator output)
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

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd = Vr  (state[2])."""
        return float(x_slice[self.state_schema.index('Vr')])

    def efd_output_expr(self, state_offset: int) -> str:
        vr_i = self.state_schema.index('Vr')
        return f"x[{state_offset + vr_i}]"

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Full 4-state back-solve: [Vm, LLx, Vr, Vf] from Efd target."""
        Efd_req = float(targets.get('Efd', 1.0))
        Vt      = float(targets.get('Vt',  1.0))
        p = self.params
        KA    = float(p.get('KA', 50.0))
        VRMAX = float(p.get('VRMAX', 5.0)); VRMIN = float(p.get('VRMIN', -5.0))
        VIMAX = float(p.get('VIMAX', 0.5)); VIMIN = float(p.get('VIMIN', -0.5))

        Efd_eff = max(min(Efd_req, VRMAX), VRMIN)
        self.params['Efd_eff'] = float(Efd_eff)
        Vr_eq   = Efd_eff
        vi_eq   = max(min((Vr_eq / KA) if KA > 1e-6 else 0.0, VIMAX), VIMIN)
        LLx_eq  = vi_eq
        Vf_eq   = Vr_eq          # washout converged: Vf = Vr at SS
        Vm      = Vt
        self.params['Vref'] = Vm + vi_eq

        return np.array([Vm, LLx_eq, Vr_eq, Vf_eq])
