import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class Esst3a(PowerComponent):
    """
    IEEE Type ST3A Exciter (ESST3A) — full 5-state implementation.

    States: [Vm, LLx, Vr, VM, VB]
      Vm   – voltage transducer output (1st-order lag at TR)
      LLx  – lead-lag compensator state (TB time constant)
      Vr   – voltage regulator output (KA gain + TA lag, limited to VRMAX/VRMIN)
      VM   – inner field voltage regulator output (KM gain + TM lag)
      VB   – rectifier terminal voltage (fast lag ~0.01 s, limited to VBMAX)

    The field voltage output is Efd = VB * VM (multiplicative coupling).
    This provides the natural gain rolloff at electromechanical frequencies that
    prevents the AVR-induced negative damping seen in simplified proportional models.

    The TC/TB lead-lag block ahead of KA is the primary stability element:
    with TC < TB it reduces effective gain at inter-area oscillation frequencies
    (0.5-2 Hz) while preserving high DC gain for good voltage regulation.

    Ports in:  [Vterm, Vref, id_dq, iq_dq, Vd, Vq]
    Port out:  [Efd]
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'signal', 'pu'),
                ('Vref',  'signal', 'pu'),
                ('id_dq', 'flow',   'pu'),   # dq-frame d-axis stator current
                ('iq_dq', 'flow',   'pu'),   # dq-frame q-axis stator current
                ('Vd',    'effort', 'pu'),   # dq-frame d-axis terminal voltage
                ('Vq',    'effort', 'pu'),   # dq-frame q-axis terminal voltage
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
                    'cpp_expr': 'x[4]*x[3]'}
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
            // --- States ---
            double Vm  = x[0];   // voltage transducer
            double LLx = x[1];   // lead-lag state
            double Vr  = x[2];   // voltage regulator
            double VM  = x[3];   // inner field regulator
            double VB  = x[4];   // rectifier voltage

            // --- Inputs ---
            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];   // dq-frame d-axis current
            double iq    = inputs[3];   // dq-frame q-axis current
            double Vd    = inputs[4];   // dq-frame d-axis voltage
            double Vq    = inputs[5];   // dq-frame q-axis voltage

            // 1. Voltage transducer lag: dVm/dt = (Vterm - Vm) / TR
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter (VIMAX/VIMIN)
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            //    Transfer fn: H(s) = (1 + s*TC) / (1 + s*TB)
            //    State eq: dLLx/dt = (vi - LLx) / TB
            //    Output:   LL_y = (TC/TB)*(vi - LLx) + LLx
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA gain, TA lag) with anti-windup.
            //    A very small TA is too stiff for explicit RK4 and can explode Vr.
            //    Keep a practical floor and rate-limit dVr for numerical robustness.
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            //    VE = |KPC*(Vd + j*Vq) + j*(KI + KPC*XL)*(id + j*iq)|
            //    where KPC = KP * exp(j*THETAP*pi/180)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            // j*(KI + KPC*XL) expanded: -(KPC_i*XL) + j*(KI + KPC_r*XL)
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function (IEEE Std 421.5 piecewise)
            //    IN = KC * XadIfd / VE  (XadIfd approximated as VB*VM = current Efd)
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN = KC * XadIfd / VE;
            double FEX;
            if      (IN <= 0.0)    FEX = 1.0;
            else if (IN <= 0.433)  FEX = 1.0 - 0.577 * IN;
            else if (IN <= 0.75)   FEX = sqrt(0.75 - IN * IN);
            else if (IN <= 1.0)    FEX = 1.732 * (1.0 - IN);
            else                   FEX = 0.0;

            // 7. Rectifier ceiling VB — fast lag (0.01 s) smooths FEX discontinuities
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;
            
            // 8. Current Efd output and feedback path
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM gain, TM lag) with anti-windup
            //    VM drives the multiplier output: Efd = VB * VM
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd = clip(VB * VM, Efd_min, Efd_max) — matches C++ dynamics."""
        vm_i    = self.state_schema.index('VM')
        vb_i    = self.state_schema.index('VB')
        Efd_max = float(self.params.get('Efd_max', 5.0))
        Efd_min = float(self.params.get('Efd_min', -1.0))
        raw     = float(x_slice[vb_i] * x_slice[vm_i])
        return max(min(raw, Efd_max), Efd_min)

    def efd_output_expr(self, state_offset: int) -> str:
        """C++ expression for clipped Efd — matches the clipping in the dynamics."""
        vm_i    = self.state_schema.index('VM')
        vb_i    = self.state_schema.index('VB')
        Efd_max = float(self.params.get('Efd_max', 5.0))
        Efd_min = float(self.params.get('Efd_min', -1.0))
        raw     = f"(x[{state_offset + vb_i}]*x[{state_offset + vm_i}])"
        return f"fmax({Efd_min:.6f}, fmin({Efd_max:.6f}, {raw}))"

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Full 5-state back-solve: [Vm, LLx, Vr, VM, VB] from Efd target."""
        Efd_req = float(targets.get('Efd', 1.0))
        Vt      = float(targets.get('Vt',  1.0))
        vd      = float(targets.get('vd', targets.get('vd_ri', 0.0)))
        vq      = float(targets.get('vq', targets.get('vq_ri', Vt)))
        id_val  = float(targets.get('id', 0.0))
        iq_val  = float(targets.get('iq', 0.0))

        p = self.params
        KP     = float(p.get('KP', 3.67));  KI    = float(p.get('KI', 0.435))
        XL     = float(p.get('XL', 0.0098)); THETAP = float(p.get('THETAP', 0.0))
        KC     = float(p.get('KC', 0.01));  KA    = float(p.get('KA', 20.0))
        KM     = float(p.get('KM', 8.0));   KG    = float(p.get('KG', 1.0))
        VBMAX  = float(p.get('VBMAX', 5.48))
        VRMAX  = float(p.get('VRMAX', 99.0)); VRMIN = float(p.get('VRMIN', -99.0))
        VIMAX  = float(p.get('VIMAX', 0.2));  VIMIN = float(p.get('VIMIN', -0.2))
        VMMAX  = float(p.get('VMMAX', 99.0)); VMMIN = float(p.get('VMMIN', 0.0))
        VGMAX  = float(p.get('VGMAX', 3.86))
        Efd_max = float(p.get('Efd_max', 5.0))
        Efd_min = float(p.get('Efd_min', -1.0))

        # VE load compensator
        theta_rad = THETAP * math.pi / 180.0
        KPC = KP * complex(math.cos(theta_rad), math.sin(theta_rad))
        VE = abs(KPC * complex(vd, vq) + 1j * (KI + KPC * XL) * complex(id_val, iq_val))
        if VE < 1e-6:
            VE = 1e-6

        VMMAX_eff = min(VMMAX, 10.0) if VMMAX > 90.0 else VMMAX
        Efd_eff   = max(min(Efd_req, Efd_max), Efd_min)
        self.params['Efd_eff'] = float(Efd_eff)

        # FEX rectifier — solve self-consistently: IN = KC*VB*VM/VE where Efd=VB*VM.
        # This matches the C++ which uses XadIfd = VB*VM (not abs(Efd_req)).
        IN = KC * abs(Efd_eff) / VE
        if IN <= 0.0:        FEX = 1.0
        elif IN <= 0.433:    FEX = 1.0 - 0.577 * IN
        elif IN <= 0.75:     FEX = math.sqrt(max(0.0, 0.75 - IN**2))
        elif IN <= 1.0:      FEX = 1.732 * (1.0 - IN)
        else:                FEX = 0.0
        VB_eq = min(max(VE * FEX, 0.0), VBMAX)

        # Solve self-consistently: VB_eq drives VM_eq = Efd/VB, then IN = KC*Efd/VE
        # Iterate a few times to converge VB ↔ IN ↔ FEX ↔ VB loop
        for _ in range(10):
            VM_eq = (Efd_eff / VB_eq) if VB_eq > 1e-6 else 1.0
            VM_eq = max(min(VM_eq, VMMAX_eff), VMMIN)
            XadIfd = VB_eq * VM_eq          # same as Efd_eff when not clamped
            IN_new = KC * max(XadIfd, 0.0) / VE
            if   IN_new <= 0.0:    FEX_new = 1.0
            elif IN_new <= 0.433:  FEX_new = 1.0 - 0.577 * IN_new
            elif IN_new <= 0.75:   FEX_new = math.sqrt(max(0.0, 0.75 - IN_new**2))
            elif IN_new <= 1.0:    FEX_new = 1.732 * (1.0 - IN_new)
            else:                  FEX_new = 0.0
            VB_new = min(max(VE * FEX_new, 0.0), VBMAX)
            if abs(VB_new - VB_eq) < 1e-10:
                break
            VB_eq = VB_new

        VM_eq  = (Efd_eff / VB_eq) if VB_eq > 1e-6 else 1.0
        VM_eq  = max(min(VM_eq, VMMAX_eff), VMMIN)
        VG_eq  = max(min(KG * Efd_eff, VGMAX), 0.0)
        vrs_eq = (VM_eq / KM) if KM > 1e-6 else 0.0
        Vr_eq  = max(min(vrs_eq + VG_eq, VRMAX), VRMIN)
        vi_eq  = max(min((Vr_eq / KA) if KA > 1e-6 else 0.0, VIMAX), VIMIN)
        LLx_eq = vi_eq
        Vm     = Vt
        self.params['Vref'] = Vm + vi_eq

        return np.array([Vm, LLx_eq, Vr_eq, VM_eq, VB_eq])
