"""
DFIG — Doubly-Fed Induction Generator (Port-Hamiltonian Model)
==============================================================

Physical model
--------------
Based on Song & Qu (2011), "Energy-based modelling and control of wind
energy conversion system with DFIG", Int. J. Control 84(2):281-292.

The electromechanical subsystem Σ_em is cast in PCH form:

    ẋ = (J_em - R_em) ∇H_em + g1·u1 + g2·u2 + g_em·u_em

where the energy variables (state vector) are the four machine flux-linkages
and the rotor angular momentum:

    x = [φ_sd, φ_sq, φ_rd, φ_rq, j·ω_m]^T

Hamiltonian:
    H_em = ½ x^T M^{-1} x,   M = diag(Ls, Ls, Lr, Lr, j)

The Hamiltonian gradient ∇H_em = M^{-1} x = [i_sd, i_sq, i_rd, i_rq, ω_m]^T.

Structure matrix J_em (skew-symmetric, encodes gyration/speed-coupling):

    J_em =
    ⎡  0      ω_s·Ls    0       ω_s·Lm    0        ⎤
    ⎢-ω_s·Ls   0      -ω_s·Lm   0        0        ⎥
    ⎢  0      ω_s·Lm    0       ω_s·Lr  -np·φ_rq  ⎥
    ⎢-ω_s·Lm   0      -ω_s·Lr   0        np·φ_rd  ⎥
    ⎣  0       0       np·φ_rq -np·φ_rd   0        ⎦

Dissipation matrix R_em (symmetric positive semi-definite):

    R_em = diag(Rs, Rs, Rr, Rr, f)

Stator port input u1 = [Vsd, Vsq]^T  (grid terminal voltages, Park-frame)
Rotor  port input u2 = [Vrd, Vrq]^T  (converter-applied rotor voltages)
Load torque   u_em   = [0, -T_L]^T

Framework integration
----------------------
The DFIG presents a Norton-equivalent interface to the network, exactly like
GENROU.  The Norton admittance is the per-unit stator leakage admittance
seen from the stator terminals:

    Y_norton = 1 / (Rs + j·Xs_sigma)

where Xs_sigma = Ls - Lm^2/Lr  is the stator short-circuit reactance.

Inputs  (indices match port_schema):
    [0] Vd        — terminal voltage d-component (RI-frame: V_Re)
    [1] Vq        — terminal voltage q-component (RI-frame: V_Im)
    [2] Tm        — mechanical torque from drive-train [pu]
    [3] Vrd       — rotor voltage d-component from MSC [pu]
    [4] Vrq       — rotor voltage q-component from MSC [pu]
    [5] omega_s_in — stator synchronous speed (could be wired to BusFreq) [pu]

Outputs (indices match port_schema):
    [0] Id        — Norton injection Re  (for network solve)
    [1] Iq        — Norton injection Im  (for network solve)
    [2] omega     — rotor mechanical speed [pu]
    [3] Pe        — active power output [pu]
    [4] Qe        — reactive power output [pu]
    [5] id_dq     — dq-frame stator d-current (for monitoring/control)
    [6] iq_dq     — dq-frame stator q-current
    [7] It_Re     — actual terminal current Re
    [8] It_Im     — actual terminal current Im

States (5):
    [0] phi_sd   — stator d-axis flux linkage  [pu]
    [1] phi_sq   — stator q-axis flux linkage  [pu]
    [2] phi_rd   — rotor  d-axis flux linkage  [pu]
    [3] phi_rq   — rotor  q-axis flux linkage  [pu]
    [4] p_mech   — rotor angular momentum  p = j·ω_m  [pu·s]

Parameters
----------
  Ls       — stator self-inductance [pu]
  Lr       — rotor  self-inductance [pu]
  Lm       — mutual (magnetising) inductance [pu]
  Rs       — stator resistance [pu]
  Rr       — rotor  resistance [pu]
  j_inertia — rotor inertia (same as 2H in per-unit swing equation) [pu]
  f_damp   — mechanical friction coefficient [pu]
  np       — number of pole pairs
  omega_b  — base angular frequency [rad/s]
  omega_s  — synchronous electrical speed [pu]; default 1.0 (50/60 Hz base)
  D        — additional damping coefficient [pu]; default 0.0
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class Dfig(PowerComponent):
    """
    Doubly-Fed Induction Generator — Port-Hamiltonian flux-linkage model.

    Casts the DFIG electromechanical subsystem in Song & Qu (2011) PCH form,
    then maps it onto the framework's generator-role Norton-injection interface
    so it plugs directly into any bus in system.json.

    The five states are the four flux-linkages (φ_sd, φ_sq, φ_rd, φ_rq) and
    the rotor angular momentum p_mech = j·ω_m.

    The machine uses a **voltage-source Norton equivalent** for the network
    interface (contributes_norton_admittance = True).  Norton admittance is

        Y_N = 1 / (Rs + j·Xs_sigma),   Xs_sigma = Ls - Lm²/Lr

    and the voltage-independent Norton current is computed from the current
    flux states at each time step.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        # Derive compiler-compatible Norton impedance aliases:
        #   ra             → Rs  (stator resistance)
        #   xd_double_prime → Xs_sigma = (Ls·Lr − Lm²)/Lr  (stator leakage)
        sigma_LS = params['Ls'] * params['Lr'] - params['Lm'] ** 2
        self.params.setdefault('ra', params['Rs'])
        self.params.setdefault('xd_double_prime', sigma_LS / params['Lr'])

    # ------------------------------------------------------------------
    # Group 1 — Schema Definitions
    # ------------------------------------------------------------------

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd',        'effort', 'pu'),   # [0] stator terminal V_Re (RI-frame)
                ('Vq',        'effort', 'pu'),   # [1] stator terminal V_Im (RI-frame)
                ('Tm',        'effort', 'pu'),   # [2] mechanical torque in
                ('Vrd',       'effort', 'pu'),   # [3] rotor voltage d (from MSC)
                ('Vrq',       'effort', 'pu'),   # [4] rotor voltage q (from MSC)
            ],
            'out': [
                ('Id',     'flow', 'pu'),   # [0]  Norton injection Re
                ('Iq',     'flow', 'pu'),   # [1]  Norton injection Im
                ('omega',  'flow', 'pu'),   # [2]  rotor mechanical speed
                ('Pe',     'flow', 'pu'),   # [3]  active electrical power (stator)
                ('Qe',     'flow', 'pu'),   # [4]  reactive electrical power (stator)
                ('id_dq',  'flow', 'pu'),   # [5]  stator d-current
                ('iq_dq',  'flow', 'pu'),   # [6]  stator q-current
                ('It_Re',  'flow', 'pu'),   # [7]  terminal current Re
                ('It_Im',  'flow', 'pu'),   # [8]  terminal current Im
                # --- Extended outputs for RSC / DC-link wiring ---
                ('i_rd',       'flow', 'pu'),   # [9]  rotor d-current
                ('i_rq',       'flow', 'pu'),   # [10] rotor q-current
                ('phi_sd_out', 'flow', 'pu'),   # [11] stator d-flux (state echo)
                ('phi_sq_out', 'flow', 'pu'),   # [12] stator q-flux (state echo)
                ('P_rotor',    'flow', 'pu'),   # [13] rotor electrical power
                ('slip_out',   'flow', 'pu'),   # [14] slip value
                ('Te_out',     'flow', 'pu'),   # [15] electromagnetic torque
                ('Vterm_out',  'flow', 'pu'),   # [16] terminal voltage magnitude
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        # Five PH energy variables (flux-linkages + angular momentum)
        return ['phi_sd', 'phi_sq', 'phi_rd', 'phi_rq', 'p_mech']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Ls':        'Stator self-inductance [pu]',
            'Lr':        'Rotor self-inductance [pu]',
            'Lm':        'Mutual (magnetising) inductance [pu]',
            'Rs':        'Stator resistance [pu]',
            'Rr':        'Rotor resistance [pu]',
            'j_inertia': 'Rotor inertia (≈2H in pu-system swing convention) [pu·s]',
            'f_damp':    'Rotor mechanical friction coefficient [pu]',
            'np':        'Number of pole pairs [-] (use 1 in per-unit system)',
            'omega_b':   'Base angular frequency [rad/s]',
            'omega_s':   'Synchronous electrical speed [pu]; typically 1.0',
        }

    @property
    def component_role(self) -> str:
        return 'generator'

    @property
    def contributes_norton_admittance(self) -> bool:
        # DFIG contributes its stator leakage admittance Y_N = 1/(Rs + j·Xs_σ)
        # to the Y-bus.  This reduces the Z-bus diagonal at DFIG buses,
        # improving algebraic loop conditioning and closed-loop stability.
        # The compute_outputs function returns the Norton source current
        # I_N = I_gen + Y_N × V  (the voltage-independent part is I_gen,
        # the Y_N × V correction makes it compatible with the augmented Z-bus).
        return True

    @property
    def uses_ri_frame(self) -> bool:
        # DFIG operates directly in the RI (network) frame — no Park rotation
        return True

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pe':         {'description': 'Active power output',         'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'Qe':         {'description': 'Reactive power output',       'unit': 'pu',  'cpp_expr': 'outputs[4]'},
            'omega_pu':   {'description': 'Rotor speed [pu]',            'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'slip':       {'description': 'Slip = (omega_s - omega)/omega_s', 'unit': '-',
                           'cpp_expr': '(omega_s - outputs[2]) / omega_s'},
            'phi_sd':     {'description': 'Stator d-flux linkage',       'unit': 'pu',  'cpp_expr': 'x[0]'},
            'phi_sq':     {'description': 'Stator q-flux linkage',       'unit': 'pu',  'cpp_expr': 'x[1]'},
            'phi_rd':     {'description': 'Rotor  d-flux linkage',       'unit': 'pu',  'cpp_expr': 'x[2]'},
            'phi_rq':     {'description': 'Rotor  q-flux linkage',       'unit': 'pu',  'cpp_expr': 'x[3]'},
            'Te_elec':    {'description': 'Electromagnetic torque',      'unit': 'pu',
                           'cpp_expr': 'outputs[15]'},
            'V_term':     {'description': 'Terminal voltage magnitude',  'unit': 'pu',
                           'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])'},
        }

    # ------------------------------------------------------------------
    # Group 2 — C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        """
        Compute stator current outputs as a pure current source injection.

        The docstring in `init_from_phasor` describes the PCH motor convention
        and equilibrium equations — all quantities here follow that convention.
        """
        return """
            // ---- Stator & rotor currents from flux states (M^{-1}·φ) ----
            // Motor convention: positive i_sd = current flowing INTO stator.
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd = (x[0] * Lr - x[2] * Lm) / sigma_LS;  // motor convention (i_sd<0 for gen)
            double i_sq = (x[1] * Lr - x[3] * Lm) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;

            // ---- Rotor speed ----
            double omega_m = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Terminal voltage (from algebraic loop — updated each iteration) ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Physical power (positive = injected into network) ----
            // Motor convention: Pe_motor = Vsd*i_sd + Vsq*i_sq < 0 for generator.
            // Physical: Pe = -Pe_motor, Qe = -Qe_motor.
            // Computing here (not deferred to step) so the RSC sees correct
            // Pe/Qe during the algebraic loop and produces correct Vrd/Vrq.
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // Norton equivalent current: I_N = I_gen + Y_N × V
            // where I_gen = (-i_sd, -i_sq) is the physical generator current
            // and Y_N = 1/(Rs + j×Xs_σ) is the stator Norton admittance.
            // Y_N × V = (Rs×Vd + Xs_σ×Vq, Rs×Vq − Xs_σ×Vd) / (Rs² + Xs_σ²)
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            double Iinj_re = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            double Iinj_im = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[0] = Iinj_re;
            outputs[1] = Iinj_im;
            outputs[2] = omega_m;       // rotor mechanical speed [pu]
            outputs[3] = Pe;            // physical active power (positive = generating)
            outputs[4] = Qe;            // physical reactive power (positive = overexcited)
            outputs[5] = i_sd;          // id (motor convention, monitoring)
            outputs[6] = i_sq;          // iq (motor convention, monitoring)
            outputs[7] = -i_sd;         // It_Re (generator convention)
            outputs[8] = -i_sq;         // It_Im (generator convention)
            // ---- Extended outputs for RSC / DC-link ----
            outputs[9]  = i_rd;         // rotor d-current
            outputs[10] = i_rq;         // rotor q-current
            outputs[11] = x[0];         // phi_sd (state echo)
            outputs[12] = x[1];         // phi_sq (state echo)
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;  // P_rotor = Vrd*i_rd + Vrq*i_rq
            outputs[14] = (omega_s - omega_m) / omega_s;  // slip
            outputs[15] = Te;           // electromagnetic torque
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);  // Vterm
        """

    def get_cpp_step_code(self) -> str:
        """
        PCH dynamics:  ẋ = (J_em - R_em) ∇H_em + g1·u1 + g2·u2 + g_em·u_em

        ∇H_em = M^{-1} x  =  [i_sd, i_sq, i_rd, i_rq, ω_m]

        Structure matrix J_em is state-dependent (contains φ_rd, φ_rq)
        — see Song & Qu Eq.(4).

        Equations written out explicitly (row by row):

        φ̇_sd = ωs·Ls·i_sq  + ωs·Lm·i_rq                  - Rs·i_sd  + Vsd
        φ̇_sq = -ωs·Ls·i_sd - ωs·Lm·i_rd                  - Rs·i_sq  + Vsq
        φ̇_rd = ωs·Lm·i_sq  + ωs·Lr·i_rq - np·φ_rq·ω_m   - Rr·i_rd  + Vrd
        φ̇_rq = -ωs·Lm·i_sd - ωs·Lr·i_rd + np·φ_rd·ω_m   - Rr·i_rq  + Vrq
        ṗ     = np·(φ_rq·i_rd - φ_rd·i_rq)·(0) [corrected: Te = np(φ_rd·i_rq - φ_rq·i_rd)]
                - f·ω_m  + T_m - 0

        Electromagnetic torque (from ∇H_em cross-product):
            Te = np · (φ_rd · i_rq - φ_rq · i_rd)

        Swing:
            ṗ_mech = T_m - Te - f·(ω_m - 1)

        Note: Vsd = inputs[0] (V_Re in RI-frame = stator d-voltage)
              Vsq = inputs[1] (V_Im in RI-frame = stator q-voltage)
              Vrd = inputs[3], Vrq = inputs[4]   (from MSC/converter)

        In this model the dq-frame is aligned with the RI-frame (stator voltage
        orientation), i.e. Vsd ≡ V_Re, Vsq ≡ V_Im.  This is valid when the
        stator is directly connected to the infinite bus and no Park rotation is
        needed at the stator terminals — consistent with a Type-3 wind turbine
        whose stator is grid-tied without a full converter.
        """
        return r"""
            // ---- Inputs ----
            double Vsd  = inputs[0];    // stator d-voltage (RI-frame V_Re)
            double Vsq  = inputs[1];    // stator q-voltage (RI-frame V_Im)
            double T_m  = inputs[2];    // mechanical torque [pu]
            double Vrd  = inputs[3];    // rotor d-voltage from MSC [pu]
            double Vrq  = inputs[4];    // rotor q-voltage from MSC [pu]

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_mech = x[4];

            // ---- PH gradient  ∇H_em = M^{-1} x ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            // Invert the 4×4 inductance block:
            //   [i_sd]   1          [Lr   0  -Lm   0 ] [phi_sd]
            //   [i_sq] = ——————  ×  [ 0  Lr    0 -Lm ] [phi_sq]
            //   [i_rd]   sigma_LS   [-Lm  0   Ls   0 ] [phi_rd]
            //   [i_rq]              [ 0 -Lm    0  Ls ] [phi_rq]
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd =  (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq =  (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_m = p_mech / j_inertia;

            // ---- Structure matrix rows J_em × ∇H_em ----
            // Row 0 (φ_sd):  J_em[0,:] · [i_sd,i_sq,i_rd,i_rq,ω_m]
            //   = 0·i_sd + ωs·Ls·i_sq + 0·i_rd + ωs·Lm·i_rq + 0·ω_m
            double J0 =  omega_s * Ls * i_sq + omega_s * Lm * i_rq;

            // Row 1 (φ_sq):
            //   = -ωs·Ls·i_sd + 0 + (-ωs·Lm)·i_rd + 0 + 0
            double J1 = -omega_s * Ls * i_sd - omega_s * Lm * i_rd;

            // Row 2 (φ_rd):
            //   = 0·i_sd + ωs·Lm·i_sq + 0·i_rd + ωs·Lr·i_rq + (-np·φ_rq)·ω_m
            double J2 =  omega_s * Lm * i_sq + omega_s * Lr * i_rq - np * phi_rq * omega_m;

            // Row 3 (φ_rq):
            //   = -ωs·Lm·i_sd + 0 + (-ωs·Lr)·i_rd + 0 + np·φ_rd·ω_m
            double J3 = -omega_s * Lm * i_sd - omega_s * Lr * i_rd + np * phi_rd * omega_m;

            // Row 4 (p_mech):
            //   = 0 + 0 + np·φ_rq·i_rd + (-np·φ_rd)·i_rq + 0
            //   NOTE: J_em[4,2]=+np·φ_rq, J_em[4,3]=-np·φ_rd  (Song & Qu Eq.4)
            //   This equals the electromagnetic torque Te.
            double J4 =  np * phi_rq * i_rd - np * phi_rd * i_rq;

            // ---- Dissipation matrix R_em × ∇H_em ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * (omega_m - 1.0);

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PCH dynamics: ẋ = Ω·[(J - R)∇H + g·u] ----
            // Ω = diag(ωb, ωb, ωb, ωb, 1) converts per-unit flux derivatives
            // from electrical-radian time to seconds:  dψ/dt = ωb·(V − R·i + ωs·ψ⊥).
            // The mechanical equation (row 5) is already in seconds (2H convention).
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_m;  // T_m enters as -T_L (motor sign convention)

            // ---- Power outputs (physical: positive = injected into network) ----
            // Motor-convention: Pe_motor = Vsd*i_sd + Vsq*i_sq  < 0 for generator.
            // Physical Pe (generator convention, positive for generation): Pe = -Pe_motor.
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            double P_rotor = Vrd * i_rd + Vrq * i_rq;

            // outputs[0,1] = network injection, owned by compute_outputs. Do NOT overwrite.
            // outputs[2]   = omega_m, set in compute_outputs. Do NOT overwrite.
            outputs[3] = Pe;            // physical active power (positive = generating)
            outputs[4] = Qe;            // physical reactive power (positive = overexcited)
            outputs[5] = i_sd;          // id motor convention (monitoring)
            outputs[6] = i_sq;          // iq motor convention (monitoring)
            outputs[7] = -i_sd;         // It_Re generator convention
            outputs[8] = -i_sq;         // It_Im generator convention
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = P_rotor;
            outputs[14] = (omega_s - omega_m) / omega_s;
            outputs[15] = Te;
        """

    # ------------------------------------------------------------------
    # Group 3 — Initialization Contract
    # ------------------------------------------------------------------

    def _sigma(self) -> float:
        """Leakage coefficient: σ_LS = Ls·Lr - Lm²."""
        p = self.params
        return p['Ls'] * p['Lr'] - p['Lm'] ** 2

    def _norton_admittance(self) -> complex:
        """
        Norton (Thevenin) admittance seen at stator terminals.

        The stator short-circuit reactance is:
            Xs_sigma = Ls - Lm²/Lr = σ_LS / Lr

        Y_N = 1 / (Rs + j·Xs_sigma)
        """
        p = self.params
        Rs = p['Rs']
        Xs = self._sigma() / p['Lr']
        denom = Rs**2 + Xs**2
        return complex(Rs / denom, -Xs / denom)

    def compute_norton_current(self, x_slice: np.ndarray,
                               V_bus_complex: complex = None) -> complex:
        """
        Current-source injection for the Kron Z-bus.

        The DFIG stator current at any instant is determined entirely by the
        flux-linkage states through the constitutive relation M·i = φ:

            i_sd = (Lr·φ_sd − Lm·φ_rd) / (Ls·Lr − Lm²)
            i_sq = (Lr·φ_sq − Lm·φ_rq) / (Ls·Lr − Lm²)

        The PCH formulation uses MOTOR convention: i_sd is the current flowing
        INTO the stator from the network (positive = motor operation).
        For a generator, i_sd < 0 (current flows from machine to network).

        The current source term is simply the stator current in generator
        convention: I = -(i_sd + j i_sq).  *V_bus_complex* is accepted for
        interface compatibility but intentionally unused.
        """
        p = self.params
        Lr = p['Lr'];  Lm = p['Lm']
        sigma = self._sigma()

        phi_sd, phi_sq = x_slice[0], x_slice[1]
        phi_rd, phi_rq = x_slice[2], x_slice[3]
        Rs = p['Rs']
        Ls = p['Ls']
        # Stator currents from M^{-1}·φ  (motor convention: positive = into machine)
        i_sd = (Lr * phi_sd - Lm * phi_rd) / sigma
        i_sq = (Lr * phi_sq - Lm * phi_rq) / sigma

        # Physical generator current (gen convention)
        Igen_re = -i_sd
        Igen_im = -i_sq

        # Norton correction: I_N = I_gen + Y_N × V
        # Y_N = 1/(Rs + j·Xs_σ),  Xs_σ = Ls − Lm²/Lr
        Xs_sig = Ls - Lm**2 / Lr
        Zmag2  = Rs**2 + Xs_sig**2
        if V_bus_complex is not None and Zmag2 > 1e-12:
            Vd = V_bus_complex.real
            Vq = V_bus_complex.imag
            Igen_re += (Rs * Vd + Xs_sig * Vq) / Zmag2
            Igen_im += (Rs * Vq - Xs_sig * Vd) / Zmag2

        return complex(Igen_re, Igen_im)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        """Recover stator (i_sd, i_sq) from flux states.

        The DFIG operates in RI-frame (no Park rotation), so vd/vq are
        not needed — currents are determined entirely by the flux state.
        The vd/vq arguments are accepted for API compatibility with
        generators that use the Park transform (GENROU, GENCLS, etc.).
        """
        sigma = self._sigma()
        Lr = self.params['Lr']
        Lm = self.params['Lm']
        i_sd = (Lr * x_slice[0] - Lm * x_slice[2]) / sigma
        i_sq = (Lr * x_slice[1] - Lm * x_slice[3]) / sigma
        return float(i_sd), float(i_sq)

    def compute_te(self, x_slice: np.ndarray, vd: float, vq: float) -> float:
        """Compute electromagnetic torque from flux states."""
        sigma = self._sigma()
        Ls = self.params['Ls']
        Lm = self.params['Lm']
        np_poles = self.params['np']
        
        phi_sd, phi_sq = x_slice[0], x_slice[1]
        phi_rd, phi_rq = x_slice[2], x_slice[3]
        
        i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma
        i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma
        
        Te = np_poles * (phi_rd * i_rq - phi_rq * i_rd)
        return float(Te)

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """
        Initialise all 5 states from power-flow terminal phasor V, I.

        Strategy
        --------
        At steady state ẋ = 0.  The DFIG is modelled in a reference frame
        aligned with the network (RI-frame), so:

            Vsd = V.real,  Vsq = V.imag
            i_sd = I.real, i_sq = I.imag  (stator currents from power flow)

        PCH motor convention
        --------------------
        Song & Qu (2011) cast the DFIG in PCH form with MOTOR convention:
        i_sd is the current flowing INTO the stator from the network (positive
        for motor operation).  For a generator, i_sd < 0 (current flows OUT).
        The power-flow I_phasor = conj(S/V) uses generator convention (positive
        = current injected into network), so we negate to get motor convention:

            i_sd = −I_phasor.real,   i_sq = −I_phasor.imag

        Stator voltage equilibrium (dφ_sd/dt = dφ_sq/dt = 0):
            0 = ωs·Ls·i_sq + ωs·Lm·i_rq − Rs·i_sd + Vsd   →  i_rq
            0 = −ωs·Ls·i_sd − ωs·Lm·i_rd − Rs·i_sq + Vsq  →  i_rd

        Flux linkages from constitutive relation M·i = φ:
            φ_sd = Ls·i_sd + Lm·i_rd     (motor-convention currents)
            φ_sq = Ls·i_sq + Lm·i_rq

        Mechanical equilibrium (ṗ = 0):
            Te = np·(φ_rd·i_rq − φ_rq·i_rd) > 0 for generator
            ω_m = ω_s  (synchronous start, slip = 0 initial guess)

        Physical power outputs stored in targets (positive = injected into network):
            Pe_phys = −(Vsd·i_sd + Vsq·i_sq)  > 0 for generator
            Qe_phys = −(Vsq·i_sd − Vsd·i_sq)
        """
        p = self.params
        Ls = p['Ls'];  Lr = p['Lr'];  Lm = p['Lm']
        Rs = p['Rs'];  Rr = p['Rr']
        np_poles = p['np']
        omega_s = p.get('omega_s', 1.0)

        Vsd = V_phasor.real
        Vsq = V_phasor.imag
        # PCH uses MOTOR convention: positive i_sd = current flowing INTO stator.
        # For a generator (injecting power), i_sd < 0.  Negate I_phasor (which
        # is generator-convention current) to get motor-convention stator currents.
        i_sd = -I_phasor.real
        i_sq = -I_phasor.imag

        # ----- Rotor currents from stator equilibrium -----
        # dφ_sd/dt = 0:  ωs·Ls·i_sq + ωs·Lm·i_rq − Rs·i_sd + Vsd = 0
        # → i_rq = (Rs·i_sd − Vsd − ωs·Ls·i_sq) / (ωs·Lm)
        #
        # dφ_sq/dt = 0: −ωs·Ls·i_sd − ωs·Lm·i_rd − Rs·i_sq + Vsq = 0
        # → i_rd = (Vsq − ωs·Ls·i_sd − Rs·i_sq) / (ωs·Lm)
        if abs(omega_s * Lm) > 1e-10:
            i_rq = (Rs * i_sd - Vsd - omega_s * Ls * i_sq) / (omega_s * Lm)
            i_rd = (Vsq - omega_s * Ls * i_sd - Rs * i_sq) / (omega_s * Lm)
        else:
            i_rq = 0.0
            i_rd = 0.0

        # ----- Flux linkages (from constitutive relations, Eq. 2) -----
        phi_sd = Ls * i_sd + Lm * i_rd
        phi_sq = Ls * i_sq + Lm * i_rq
        phi_rd = Lr * i_rd + Lm * i_sd
        phi_rq = Lr * i_rq + Lm * i_sq

        # ----- Rotor speed (assume synchronous at init, slip=0) -----
        omega_m_init = omega_s  # pu mechanical speed ≈ 1.0 initially
        p_mech_init = p['j_inertia'] * omega_m_init

        # ----- Electromagnetic torque at this operating point -----
        Te_init = np_poles * (phi_rd * i_rq - phi_rq * i_rd)

        # ----- Power (physical: positive = injected into network) -----
        # Motor-convention Pe = Vsd*i_sd + Vsq*i_sq  < 0 for generator.
        # Physical Pe (generator convention) = -motor_Pe  > 0 for generator.
        Pe_init = -(Vsd * i_sd + Vsq * i_sq)
        Qe_init = -(Vsq * i_sd - Vsd * i_sq)

        # ----- Required rotor voltages for steady state (dφ_rd/dt = 0) -----
        # dφ_rd/dt = ωs·Lm·i_sq + ωs·Lr·i_rq - np·φ_rq·ωm - Rr·i_rd + Vrd = 0
        Vrd_ss = -(omega_s * Lm * i_sq + omega_s * Lr * i_rq
                   - np_poles * phi_rq * omega_m_init - Rr * i_rd)
        Vrq_ss = -(- omega_s * Lm * i_sd - omega_s * Lr * i_rd
                   + np_poles * phi_rd * omega_m_init - Rr * i_rq)

        x_init = np.array([phi_sd, phi_sq, phi_rd, phi_rq, p_mech_init])

        # Rotor electrical power at equilibrium
        P_rotor_init = Vrd_ss * i_rd + Vrq_ss * i_rq

        # Store Tm0 in params so PARAM:DFIG_x.Tm0 wires resolve correctly.
        # Swing equation: dp/dt = -Te - f*(ω-1) + Tm, so Tm = Te at equilibrium.
        p['Tm0'] = float(Te_init)

        targets = {
            'Tm':    float(Te_init),
            'Pe':    float(Pe_init),
            'Qe':    float(Qe_init),
            'bus':   p.get('bus'),
            'Vt':    float(abs(V_phasor)),
            'omega': float(omega_m_init),
            'Efd':   0.0,          # Not used for DFIG; placeholder for init chain
            'Vrd':   float(Vrd_ss),
            'Vrq':   float(Vrq_ss),
            'id':    float(i_sd),
            'iq':    float(i_sq),
            'vd':    float(Vsd),
            'vq':    float(Vsq),
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
            # Extended targets for RSC / DC-link / GSC initialization
            'i_rd':      float(i_rd),
            'i_rq':      float(i_rq),
            'phi_sd':    float(phi_sd),
            'phi_sq':    float(phi_sq),
            'P_rotor':   float(P_rotor_init),
            'Te':        float(Te_init),
            'slip':      0.0,          # synchronous start
        }
        return x_init, targets

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                               vd: float, vq: float) -> np.ndarray:
        """
        Recompute flux states at the Kron-reduced RI-frame terminal voltage.

        Strategy: hold rotor currents fixed (they are set by the RSC, which
        hasn't changed), solve the 2×2 stator steady-state for new i_sd, i_sq
        at the given (vd, vq) = (V_Re, V_Im), then reconstruct all four flux
        linkages self-consistently.

        Stator equilibrium (dφ_sd/dt = dφ_sq/dt = 0):
            Rs·i_sd − ωs·Ls·i_sq = Vsd + ωs·Lm·i_rq     ... (A)
            ωs·Ls·i_sd + Rs·i_sq = Vsq − ωs·Lm·i_rd      ... (B)

        This is a 2×2 linear system [Rs, -ωLs; ωLs, Rs] · [i_sd; i_sq] = rhs.
        """
        p = self.params
        Ls = p['Ls'];  Lr = p['Lr'];  Lm = p['Lm']
        Rs = p['Rs'];  omega_s = p.get('omega_s', 1.0)
        sigma = self._sigma()

        # Extract current rotor currents (hold fixed)
        i_rd = (-Lm * x_slice[0] + Ls * x_slice[2]) / sigma
        i_rq = (-Lm * x_slice[1] + Ls * x_slice[3]) / sigma

        # Solve 2×2 for stator currents at new terminal voltage
        wLs = omega_s * Ls
        wLm = omega_s * Lm
        rhs_d = vd + wLm * i_rq
        rhs_q = vq - wLm * i_rd
        det = Rs**2 + wLs**2
        if abs(det) > 1e-12:
            i_sd = ( Rs * rhs_d + wLs * rhs_q) / det
            i_sq = (-wLs * rhs_d + Rs * rhs_q) / det
        else:
            # Fallback: keep old stator currents
            i_sd = (Lr * x_slice[0] - Lm * x_slice[2]) / sigma
            i_sq = (Lr * x_slice[1] - Lm * x_slice[3]) / sigma

        # Reconstruct all flux linkages from (i_sd, i_sq, i_rd, i_rq)
        x_new = x_slice.copy()
        x_new[0] = Ls * i_sd + Lm * i_rd   # phi_sd
        x_new[1] = Ls * i_sq + Lm * i_rq   # phi_sq
        x_new[2] = Lr * i_rd + Lm * i_sd   # phi_rd
        x_new[3] = Lr * i_rq + Lm * i_sq   # phi_rq
        # x_new[4] = p_mech  (unchanged — rotor speed fixed during Kron step)
        return x_new

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                      Efd_eff: float, clamped: bool = False) -> np.ndarray:
        """
        No-op for DFIG: Efd is not a DFIG concept.
        The rotor voltage (Vrd/Vrq) is controlled by the MSC, not an AVR.
        This method is required by the initialization contract but is a no-op.
        """
        return x_slice.copy()

    def refine_current_source_init(self, x_slice: np.ndarray,
                                   targets: dict,
                                   V_bus: complex) -> np.ndarray:
        """Update targets dict from post-Kron flux states.

        Called by Initializer.refine_renewable_controllers() after Kron
        equilibrium converged.  The DFIG flux states are already correct
        (updated by refine_at_kron_voltage).  This method re-derives the
        equilibrium rotor voltages and currents so that the RSC controller
        can be re-initialized to match.
        """
        p = self.params
        Ls = p['Ls'];  Lr = p['Lr'];  Lm = p['Lm']
        Rs = p['Rs'];  Rr = p['Rr']
        np_poles = p['np']
        omega_s = p.get('omega_s', 1.0)
        sigma = self._sigma()

        # Currents from post-Kron flux states
        phi_sd = x_slice[0]; phi_sq = x_slice[1]
        phi_rd = x_slice[2]; phi_rq = x_slice[3]
        i_sd = (Lr * phi_sd - Lm * phi_rd) / sigma
        i_sq = (Lr * phi_sq - Lm * phi_rq) / sigma
        i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma
        i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma

        omega_m = x_slice[4] / p['j_inertia']

        # Stator terminal voltage (RI-frame)
        Vsd = V_bus.real;  Vsq = V_bus.imag

        # Electromagnetic torque & power (physical: positive = injected into network)
        Te = np_poles * (phi_rd * i_rq - phi_rq * i_rd)
        Pe = -(Vsd * i_sd + Vsq * i_sq)
        Qe = -(Vsq * i_sd - Vsd * i_sq)

        # Required rotor voltages for dφ_rd/dt = 0, dφ_rq/dt = 0
        Vrd_ss = -(omega_s * Lm * i_sq + omega_s * Lr * i_rq
                   - np_poles * phi_rq * omega_m - Rr * i_rd)
        Vrq_ss = -(- omega_s * Lm * i_sd - omega_s * Lr * i_rd
                   + np_poles * phi_rd * omega_m - Rr * i_rq)

        P_rotor = Vrd_ss * i_rd + Vrq_ss * i_rq

        # Update Tm0 param so PARAM:DFIG_x.Tm0 wires resolve to Kron Te.
        # Swing equation: dp/dt = -Te - f*(ω-1) + Tm, so Tm = Te at equilibrium.
        p['Tm0'] = float(Te)

        # Update targets in-place so RSC sees fresh values
        targets.update({
            'Pe':       float(Pe),
            'Qe':       float(Qe),
            'Tm':       float(Te),
            'bus':      p.get('bus'),
            'omega':    float(omega_m),
            'Vt':       float(abs(V_bus)),
            'Vterm':    float(abs(V_bus)),
            'vd':       float(Vsd),
            'vq':       float(Vsq),
            'vd_ri':    float(Vsd),
            'vq_ri':    float(Vsq),
            'id':       float(i_sd),
            'iq':       float(i_sq),
            'Vrd':      float(Vrd_ss),
            'Vrq':      float(Vrq_ss),
            'i_rd':     float(i_rd),
            'i_rq':     float(i_rq),
            'phi_sd':   float(phi_sd),
            'phi_sq':   float(phi_sq),
            'P_rotor':  float(P_rotor),
            'Te':       float(Te),
            'slip':     float((omega_s - omega_m) / omega_s),
        })

        return x_slice.copy()  # flux states are already correct
