"""
VocInverter - Virtual Oscillator Control (VOC) Grid-Forming Inverter
Port-Hamiltonian Formulation

Based on:
  Kong, L., Xue, Y., Qiao, L., & Wang, F. (2024).
  "Control Design of Passive Grid-Forming Inverters in Port-Hamiltonian Framework"
  IEEE Transactions on Power Electronics, 39(1), 332-345.

State variables (synchronous reference frame, polar form):
  x[0] = u_mag     : Virtual oscillator voltage magnitude |u| (pu)
  x[1] = delta_voc : Angle deviation from synchronous frame (rad)
                     Analogous to generator rotor angle delta.
                     d(delta_voc)/dt = 0 at equilibrium.
  x[2] = Pf        : LPF-filtered active power (pu)
  x[3] = Qf        : LPF-filtered reactive power (pu)

Control loops (Section II-B of paper):
  P-f droop (virtual inertia):
    omega_voc = omega0 + mp * (Pref/u_ref² - Pf/u²) * u_ref²
    d(delta_voc)/dt = omega_b * (omega_voc - omega0)

  Q-u oscillator (amplitude regulation):
    du/dt = [xi1*(u_ref² - u²) + xi2_signed*(Qref/u_ref² - Qf/u²)] * u
    xi2_signed switches sign based on passivity (pump/damp condition)

Norton equivalent (voltage source behind LC filter):
  Y_norton = 1 / (Rf + j*Lf)  → added to Y_bus by compiler
  I_norton = E_voc / (Rf + j*Lf)  → source current from compute_norton_current

The compiler uses xd_double_prime = Lf and ra = Rf to compute Y_norton.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class VocInverter(PowerComponent):
    """
    VOC (Virtual Oscillator Control) Grid-Forming Inverter.

    Models a voltage-source GFM inverter with LC output filter.
    Uses Port-Hamiltonian formulation for inherent passivity guarantees.

    The component is treated as a voltage source behind impedance Rf+jLf,
    giving Norton equivalents:
      Y_norton = 1/(Rf+jLf)  (added to Y_bus by compiler)
      I_norton = E_voc/(Rf+jLf)  (returned by compute_norton_current)

    Inputs (RI-frame terminal voltages from network):
      Vd: real part of bus voltage (pu)
      Vq: imaginary part of bus voltage (pu)

    Outputs:
      Id: RI-frame current injection real part (pu)
      Iq: RI-frame current injection imaginary part (pu)
      omega: virtual angular frequency (pu)
      Pe: instantaneous active power (pu)
      Qe: instantaneous reactive power (pu)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu'),
            ],
            'out': [
                ('Id',    'flow',   'pu'),
                ('Iq',    'flow',   'pu'),
                ('omega', 'signal', 'pu'),
                ('Pe',    'flow',   'pu'),
                ('Qe',    'flow',   'pu'),
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['u_mag', 'delta_voc', 'Pf', 'Qf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Rf':        'Filter resistance (pu)',
            'Lf':        'Filter inductance (pu)',
            'Cf':        'Filter capacitance (pu)',
            'u_ref':     'Voltage reference magnitude (pu)',
            'omega0':    'Synchronous frequency reference (pu, 1.0 = 60 Hz)',
            'xi1':       'Voltage convergence gain (amplitude regulation)',
            'xi2':       'Reactive power pump/damp gain (absolute value)',
            'mp':        'Active power - frequency droop gain (pu/pu)',
            'omega_lpf': 'Low-pass filter cutoff for P/Q (rad/s in pu time)',
            'Pref':      'Active power reference (pu)',
            'Qref':      'Reactive power reference (pu)',
            'omega_b':   'Base angular frequency (rad/s)',
            'Imax':      'Current limit (pu)',
            # Used by compiler for Norton admittance: Y = 1/(ra + j*xd_double_prime)
            'ra':               'Filter resistance alias for compiler (= Rf)',
            'xd_double_prime':  'Filter inductance alias for compiler (= Lf)',
        }

    @property
    def component_role(self) -> str:
        return 'generator'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'u_mag':        {'description': 'Virtual oscillator voltage amplitude', 'unit': 'pu',  'cpp_expr': 'x[0]'},
            'delta_voc_deg':{'description': 'VOC angle deviation from sync frame',  'unit': 'deg', 'cpp_expr': 'x[1] * 180.0 / 3.14159265359'},
            'Pf':           {'description': 'LPF-filtered active power',           'unit': 'pu',  'cpp_expr': 'x[2]'},
            'Qf':           {'description': 'LPF-filtered reactive power',         'unit': 'pu',  'cpp_expr': 'x[3]'},
            'omega':        {'description': 'Virtual angular frequency',           'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'Pe':           {'description': 'Instantaneous active power output',   'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'Qe':           {'description': 'Instantaneous reactive power output', 'unit': 'pu',  'cpp_expr': 'outputs[4]'},
            'H_voc':        {'description': 'PH oscillator energy (1/2)*u_mag^2', 'unit': 'pu',  'cpp_expr': '0.5 * x[0] * x[0]'},
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """
        Initialize from power-flow phasor solution.

        Computes the virtual oscillator voltage E_voc = V_bus + I_gen * (Rf + jLf),
        sets states to equilibrium, and calibrates Pref/Qref/u_ref.
        """
        p = self.params
        Rf = p.get('Rf', 0.01)
        Lf = p.get('Lf', 0.08)
        u_ref = p.get('u_ref', 1.0)

        # Compute VOC internal voltage phasor
        Z_filter = complex(Rf, Lf)
        E_voc = V_phasor + I_phasor * Z_filter

        u_mag = abs(E_voc)
        delta  = math.atan2(E_voc.imag, E_voc.real)

        # Power at steady state
        S0 = V_phasor * np.conj(I_phasor)
        P0 = float(S0.real)
        Q0 = float(S0.imag)

        # Calibrate references to match power-flow operating point
        p['Pref'] = P0
        p['Qref'] = Q0
        p['u_ref'] = u_mag   # equilibrium voltage becomes the reference

        # Compiler Norton admittance aliases
        p['ra'] = Rf
        p['xd_double_prime'] = Lf

        # Store terminal voltage for compute_norton_current fallback
        p['_vd_init'] = V_phasor.real
        p['_vq_init'] = V_phasor.imag

        x_init = np.array([u_mag, delta, P0, Q0])

        targets = {
            'Vd': V_phasor.real,
            'Vq': V_phasor.imag,
            'Id': I_phasor.real,
            'Iq': I_phasor.imag,
            'omega': p.get('omega0', 1.0),
            'Pe': P0,
            'Qe': Q0,
            # Required by initialization.py to set Efd0/Tm0 params (unused by VOC)
            'Efd': 0.0,
            'Tm': 0.0,
        }
        return x_init, targets

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Map solver targets back to state variables."""
        p = self.params
        Pref = p.get('Pref', targets.get('Pe', 0.0))
        Qref = p.get('Qref', targets.get('Qe', 0.0))
        u_ref = p.get('u_ref', 1.0)

        Vd = targets.get('Vd', 1.0)
        Vq = targets.get('Vq', 0.0)
        Id = targets.get('Id', 0.0)
        Iq = targets.get('Iq', 0.0)
        Rf = p.get('Rf', 0.01)
        Lf = p.get('Lf', 0.08)

        V_phasor = complex(Vd, Vq)
        I_phasor = complex(Id, Iq)
        Z_filter = complex(Rf, Lf)
        E_voc = V_phasor + I_phasor * Z_filter
        u_mag = abs(E_voc)
        delta = math.atan2(E_voc.imag, E_voc.real)

        return self._init_states({
            'u_mag':     u_mag,
            'delta_voc': delta,
            'Pf':        Pref,
            'Qf':        Qref,
        })

    # ------------------------------------------------------------------
    # Norton current source for network solve
    # ------------------------------------------------------------------

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        """
        Returns Norton current source I = E_voc / (Rf + j*Lf).

        The Norton admittance Y = 1/(Rf+jLf) is added separately to Y_bus
        by the compiler (via xd_double_prime=Lf, ra=Rf).
        """
        p = self.params
        Rf = p.get('Rf', 0.01)
        Lf = p.get('Lf', 0.08)

        if len(x_slice) < 2:
            return 0j

        u_mag = float(x_slice[0])
        delta  = float(x_slice[1])

        if u_mag < 1e-6:
            return 0j

        # VOC internal voltage phasor (RI frame)
        u_a = u_mag * math.cos(delta)
        u_b = u_mag * math.sin(delta)

        # I = E_voc / Z_filter = (u_a + j*u_b) / (Rf + j*Lf)
        Zf2 = Rf * Rf + Lf * Lf
        I_Re = (u_a * Rf + u_b * Lf) / Zf2
        I_Im = (u_b * Rf - u_a * Lf) / Zf2
        return complex(I_Re, I_Im)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        """Return (0, 0) - not used for this RI-frame component."""
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # C++ dynamics kernel
    # ------------------------------------------------------------------

    def get_cpp_step_code(self) -> str:
        return r"""
        // ---- VOC Inverter dynamics ----
        // States: x[0]=u_mag, x[1]=delta_voc, x[2]=Pf, x[3]=Qf
        // Inputs: inputs[0]=Vd (RI real), inputs[1]=Vq (RI imag)

        double u_mag = x[0];
        double delta  = x[1];
        double Pf     = x[2];
        double Qf     = x[3];

        // VOC internal voltage phasor (synchronous RI frame)
        double u_a = u_mag * std::cos(delta);
        double u_b = u_mag * std::sin(delta);

        // Filter impedance squared
        double Zf2   = Rf * Rf + Lf * Lf;

        // Norton current: I_norton = E_voc / Z_f  (injected into Y-bus network)
        // Y_norton = 1/Z_f is already added as a shunt by the compiler, so we
        // must inject the NORTON source current (not the filter current ig).
        // This matches GENROU convention: id_no = psi_pp / Z_norton (not (psi-V)/Z).
        double I_norton_Re = (u_a * Rf + u_b * Lf) / Zf2;
        double I_norton_Im = (u_b * Rf - u_a * Lf) / Zf2;

        // Filter current (for P/Q feedback to control loops)
        // ig = (E_voc - V_bus) / Z_f  — actual current flowing through filter
        double dV_Re = u_a - inputs[0];
        double dV_Im = u_b - inputs[1];
        double ig_Re = (dV_Re * Rf + dV_Im * Lf) / Zf2;
        double ig_Im = (dV_Im * Rf - dV_Re * Lf) / Zf2;

        // Instantaneous power at terminal (V_bus * conj(ig))
        double P_inst = inputs[0] * ig_Re + inputs[1] * ig_Im;
        double Q_inst = inputs[1] * ig_Re - inputs[0] * ig_Im;

        // Power low-pass filter
        dxdt[2] = omega_lpf * (P_inst - Pf);
        dxdt[3] = omega_lpf * (Q_inst - Qf);

        // ---- Voltage amplitude dynamics (Q-u oscillator) ----
        double u2     = u_mag * u_mag;
        double u2_ref = u_ref * u_ref;

        // Pump-or-damp sign selection (instantaneous, no hysteresis state)
        double Phi    = 0.5 * Cf * (u2 - u2_ref);
        double Q_ratio = (u2 > 1e-6) ? (Qref / u2_ref - Qf / u2) : 0.0;
        double xi2_signed = ((Q_ratio * Phi) >= 0.0) ? -xi2 : xi2;

        double A_over_Cf = xi1 * (u2_ref - u2) + xi2_signed * Q_ratio;
        dxdt[0] = A_over_Cf * u_mag;

        // ---- Phase angle / virtual inertia (P-f droop) ----
        double omega_voc = omega0;
        if (u2 > 1e-6) {
            omega_voc = omega0 + mp * (Pref / u2_ref - Pf / u2) * u2_ref;
        }
        dxdt[1] = omega_b * (omega_voc - omega0);

        // ---- Outputs ----
        // Norton current injected into network (Y-bus already has Y_f=1/Z_f shunt)
        outputs[0] = I_norton_Re;
        outputs[1] = I_norton_Im;
        outputs[2] = omega_voc;
        outputs[3] = P_inst;
        outputs[4] = Q_inst;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
        // ---- VOC Inverter outputs (from current states + inputs) ----
        double u_mag = x[0];
        double delta  = x[1];
        double Pf     = x[2];

        double u_a = u_mag * std::cos(delta);
        double u_b = u_mag * std::sin(delta);

        double Zf2   = Rf * Rf + Lf * Lf;

        // Norton current: I_norton = E_voc / Z_f
        double I_norton_Re = (u_a * Rf + u_b * Lf) / Zf2;
        double I_norton_Im = (u_b * Rf - u_a * Lf) / Zf2;

        // Filter current for P/Q feedback
        double dV_Re = u_a - inputs[0];
        double dV_Im = u_b - inputs[1];
        double ig_Re = (dV_Re * Rf + dV_Im * Lf) / Zf2;
        double ig_Im = (dV_Im * Rf - dV_Re * Lf) / Zf2;

        // Terminal power: P = Re(V_bus * conj(ig))
        double P_inst = inputs[0] * ig_Re + inputs[1] * ig_Im;
        double Q_inst = inputs[1] * ig_Re - inputs[0] * ig_Im;

        double u2     = u_mag * u_mag;
        double u2_ref = u_ref * u_ref;
        double omega_voc = omega0;
        if (u2 > 1e-6) {
            omega_voc = omega0 + mp * (Pref / u2_ref - Pf / u2) * u2_ref;
        }

        outputs[0] = I_norton_Re;
        outputs[1] = I_norton_Im;
        outputs[2] = omega_voc;
        outputs[3] = P_inst;
        outputs[4] = Q_inst;
        """
