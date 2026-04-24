"""
Pmu — Phasor Measurement Unit

Models the measurement chain of a real PMU (IEEE C37.118.1):
  Voltage phasor (magnitude + angle) sampled from bus Vd/Vq
  Branch current phasor (magnitude + angle) sampled from Id/Iq
  Frequency & ROCOF derived from filtered angle rate-of-change
  Active and reactive power computed from V × conj(I)
  All quantities filtered through independent 1st-order low-pass
  filters that represent the PMU's anti-aliasing + reporting delay.

Measurement chain per quantity
──────────────────────────────
                             ┌──────────────┐
  true V_mag  ──────────────►│  LPF  Tf_v  ├──► V_mag_meas
                             └──────────────┘
                             ┌──────────────┐
  true theta  ──────────────►│  LPF  Tf_f  ├──► theta_meas, omega_meas
                             └──────────────┘
                             ┌──────────────┐
  true P      ──────────────►│  LPF  Tf_pq ├──► P_meas
                             └──────────────┘
                             ┌──────────────┐
  true Q      ──────────────►│  LPF  Tf_pq ├──► Q_meas
                             └──────────────┘
                             ┌──────────────┐
  true I_mag  ──────────────►│  LPF  Tf_v  ├──► I_mag_meas
                             └──────────────┘

Typical PMU time constants (60 Hz system, P-class per C37.118):
  Tf_v   = 0.020 s   (20 ms — voltage filter / one cycle)
  Tf_f   = 0.040 s   (40 ms — frequency filter / two cycles)
  Tf_pq  = 0.020 s   (20 ms — power filter)

State variables
───────────────
  x[0] = theta_f   — filtered voltage angle (rad, for omega computation)
  x[1] = V_mag_f   — filtered voltage magnitude (pu)
  x[2] = P_f       — filtered active power (pu)
  x[3] = Q_f       — filtered reactive power (pu)
  x[4] = I_mag_f   — filtered current magnitude (pu)

Input ports
───────────
  Vd   — bus voltage real part (RI frame, pu)  [required]
  Vq   — bus voltage imag part (RI frame, pu)  [required]
  Id   — branch current real part (pu)         [optional, default 0]
  Iq   — branch current imag part (pu)         [optional, default 0]

Output ports
────────────
  V_mag   — filtered voltage magnitude (pu)
  V_ang   — filtered voltage angle (rad)
  omega   — measured frequency (pu, normalised to omega0 = 1.0)
  P_meas  — filtered active power (pu)
  Q_meas  — filtered reactive power (pu)
  I_mag   — filtered current magnitude (pu)

AGC wiring — IMPORTANT note on reference frames
─────────────────────────────────────────────────
This simulator uses a SYNCHRONOUSLY-ROTATING d-q reference frame.
Bus voltage phasors (Vd, Vq) are expressed in a frame that rotates
at ω₀ = 1.0 pu.  When all generators co-drift at ω = 1.0003 pu,
all machine angles advance together and bus voltage angles in the
d-q frame remain constant →  dθ/dt ≈ 0  →  omega_meas ≈ 1.0.

Consequence: PMU.omega correctly tracks INTER-MACHINE oscillations
and fault-induced transients, but cannot detect collective frequency
drift away from ω₀.  For AGC (absolute frequency restoration) use
the machine rotor-speed signal GENROU_x.omega directly.

PMU is best used for:
  • Voltage magnitude monitoring (V_mag)        — works perfectly
  • Voltage angle / phase-angle difference       — works perfectly
  • Inter-machine oscillation detection (omega)  — works perfectly
  • Active/reactive power flows (P_meas, Q_meas) — works when Id/Iq wired
  • Post-fault frequency nadir detection         — works perfectly

AGC wiring (use machine speed, not PMU omega):
  {"from": "GENROU_1.omega",  "to": "AGC_1.omega"}

Wide-area monitoring (any CSV pattern match):
  PMU_BUS1_V_mag  PMU_BUS1_omega  PMU_BUS1_P_meas  ...
"""

import math
import numpy as np
from typing import Dict, List, Tuple
from src.core import PowerComponent


class Pmu(PowerComponent):
    """
    Phasor Measurement Unit — IEEE C37.118.1 measurement model.

    Provides filtered bus-voltage phasor, frequency, current phasor,
    and power measurements.  Id/Iq inputs default to 0.0 if not wired
    (voltage-only PMU mode; P/Q and I_mag will be zero).
    """

    # ------------------------------------------------------------------
    # Component contract
    # ------------------------------------------------------------------

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd',  'effort', 'pu'),   # bus voltage RI real
                ('Vq',  'effort', 'pu'),   # bus voltage RI imag
                ('Id',  'flow',   'pu'),   # branch current RI real (optional)
                ('Iq',  'flow',   'pu'),   # branch current RI imag (optional)
            ],
            'out': [
                ('V_mag',  'signal', 'pu'),   # filtered voltage magnitude
                ('V_ang',  'signal', 'rad'),  # filtered voltage angle
                ('omega',  'signal', 'pu'),   # measured frequency (pu)
                ('P_meas', 'signal', 'pu'),   # filtered active power
                ('Q_meas', 'signal', 'pu'),   # filtered reactive power
                ('I_mag',  'signal', 'pu'),   # filtered current magnitude
            ],
        }

    @property
    def required_ports(self):
        """Id and Iq are optional; default to 0.0 when current measurement is not wired."""
        return ['Vd', 'Vq']

    @property
    def state_schema(self) -> List[str]:
        return ['theta_f', 'V_mag_f', 'P_f', 'Q_f', 'I_mag_f']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Tf_v':   'Voltage/current filter time constant (s, typ. 0.02)',
            'Tf_f':   'Frequency filter time constant (s, typ. 0.04)',
            'Tf_pq':  'Power filter time constant (s, typ. 0.02)',
            'fn':     'Nominal frequency (Hz, default 60)',
            'omega0': 'Nominal angular frequency reference (pu, default 1.0)',
        }

    @property
    def component_role(self) -> str:
        # 'passive' ensures initialization calls init_from_phasor with
        # the correct bus voltage from the power flow solution, so all
        # filter states start at their physical steady-state values.
        return 'passive'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'V_mag':  {'description': 'Measured voltage magnitude', 'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'V_ang':  {'description': 'Measured voltage angle',     'unit': 'rad', 'cpp_expr': 'outputs[1]'},
            'omega':  {'description': 'Measured frequency',         'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'P_meas': {'description': 'Measured active power',      'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'Q_meas': {'description': 'Measured reactive power',    'unit': 'pu',  'cpp_expr': 'outputs[4]'},
            'I_mag':  {'description': 'Measured current magnitude', 'unit': 'pu',  'cpp_expr': 'outputs[5]'},
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        Vd = V_phasor.real
        Vq = V_phasor.imag
        Id = I_phasor.real
        Iq = I_phasor.imag

        theta0 = math.atan2(Vq, Vd)
        V_mag0 = abs(V_phasor)
        P0     = Vd * Id + Vq * Iq           # Re(V * conj(I))
        Q0     = Vq * Id - Vd * Iq           # Im(V * conj(I))
        I_mag0 = abs(I_phasor)

        x_init = np.array([theta0, V_mag0, P0, Q0, I_mag0])

        targets = {
            'V_mag':  V_mag0,
            'V_ang':  theta0,
            'omega':  1.0,
            'P_meas': P0,
            'Q_meas': Q0,
            'I_mag':  I_mag0,
            'Vd':  Vd, 'Vq':  Vq,
            'Id':  Id, 'Iq':  Iq,
            'Efd': 0.0, 'Tm': 0.0,
        }
        return x_init, targets

    def init_from_targets(self, targets: Dict) -> np.ndarray:
        Vd = targets.get('Vd', 1.0)
        Vq = targets.get('Vq', 0.0)
        Id = targets.get('Id', 0.0)
        Iq = targets.get('Iq', 0.0)
        theta0 = math.atan2(Vq, Vd)
        V_mag0 = math.sqrt(Vd**2 + Vq**2)
        P0     = Vd * Id + Vq * Iq
        Q0     = Vq * Id - Vd * Iq
        I_mag0 = math.sqrt(Id**2 + Iq**2)
        return np.array([theta0, V_mag0, P0, Q0, I_mag0])

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        return 0j

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # C++ dynamics — five 1st-order LPF states
    # ------------------------------------------------------------------

    def get_cpp_step_code(self) -> str:
        return r"""
        // ---- PMU: Phasor Measurement Unit (IEEE C37.118.1) ----
        // States: x[0]=theta_f  x[1]=V_mag_f  x[2]=P_f  x[3]=Q_f  x[4]=I_mag_f
        // Inputs: inputs[0]=Vd  inputs[1]=Vq  inputs[2]=Id  inputs[3]=Iq

        const double fn_val  = (fn > 0.0) ? fn    : 60.0;
        const double Tf_v_s  = (Tf_v  > 0.0) ? Tf_v  : 0.020;
        const double Tf_f_s  = (Tf_f  > 0.0) ? Tf_f  : 0.040;
        const double Tf_pq_s = (Tf_pq > 0.0) ? Tf_pq : 0.020;

        double Vd_m = inputs[0];
        double Vq_m = inputs[1];
        double Id_m = inputs[2];   // 0.0 if not wired
        double Iq_m = inputs[3];   // 0.0 if not wired

        // True (instantaneous) measurements
        double theta_true  = std::atan2(Vq_m, Vd_m);
        double V_mag_true  = std::sqrt(Vd_m*Vd_m + Vq_m*Vq_m);
        double P_true      = Vd_m*Id_m + Vq_m*Iq_m;   // Re(V conj(I))
        double Q_true      = Vq_m*Id_m - Vd_m*Iq_m;   // Im(V conj(I))
        double I_mag_true  = std::sqrt(Id_m*Id_m + Iq_m*Iq_m);

        // --- Angle filter (for frequency extraction) ---
        // Unwrap angle difference before filtering
        double dtheta = theta_true - x[0];
        while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        dxdt[0] = dtheta / Tf_f_s;

        // --- Voltage magnitude filter ---
        dxdt[1] = (V_mag_true - x[1]) / Tf_v_s;

        // --- Power filters ---
        dxdt[2] = (P_true - x[2]) / Tf_pq_s;
        dxdt[3] = (Q_true - x[3]) / Tf_pq_s;

        // --- Current magnitude filter ---
        dxdt[4] = (I_mag_true - x[4]) / Tf_v_s;

        // ---- Outputs ----
        double omega_b_hz = 2.0 * M_PI * fn_val;
        // Frequency: omega_meas = omega0 + (d theta_f / dt) / omega_b
        // d(theta_f)/dt evaluated from filter: dxdt[0] = dtheta/Tf_f => at steady state = 0
        // Instantaneous estimate: use angle phase diff / Tf_f
        double omega_meas = omega0 + dtheta / (Tf_f_s * omega_b_hz);
        if (omega_meas < 0.5) omega_meas = 0.5;
        if (omega_meas > 1.5) omega_meas = 1.5;

        outputs[0] = x[1];          // V_mag  (filtered)
        outputs[1] = x[0];          // V_ang  (filtered theta)
        outputs[2] = omega_meas;    // omega  (pu)
        outputs[3] = x[2];          // P_meas (filtered)
        outputs[4] = x[3];          // Q_meas (filtered)
        outputs[5] = x[4];          // I_mag  (filtered)
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
        const double fn_val  = (fn > 0.0) ? fn    : 60.0;
        const double Tf_f_s  = (Tf_f  > 0.0) ? Tf_f  : 0.040;

        double Vd_m = inputs[0];
        double Vq_m = inputs[1];
        double Id_m = inputs[2];
        double Iq_m = inputs[3];

        double theta_true = std::atan2(Vq_m, Vd_m);
        double dtheta = theta_true - x[0];
        while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

        double omega_b_hz = 2.0 * M_PI * fn_val;
        double omega_meas = omega0 + dtheta / (Tf_f_s * omega_b_hz);
        if (omega_meas < 0.5) omega_meas = 0.5;
        if (omega_meas > 1.5) omega_meas = 1.5;

        outputs[0] = x[1];          // V_mag
        outputs[1] = x[0];          // V_ang
        outputs[2] = omega_meas;    // omega
        outputs[3] = x[2];          // P_meas
        outputs[4] = x[3];          // Q_meas
        outputs[5] = x[4];          // I_mag
        """
