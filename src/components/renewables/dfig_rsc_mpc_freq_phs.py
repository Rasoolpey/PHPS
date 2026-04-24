"""
DFIG_RSC_MPC_FREQ_PHS — Frequency-Based Multi-Horizon MPC RSC Controller (Σ₅_MPC_F)
=====================================================================================

Hierarchical controller separating frequency/torque control from voltage
regulation, mimicking the governor/exciter architecture of synchronous generators:

  Outer loop (N-step MPC):  Optimizes supplementary power ΔP for the APC
    swing equation, predicting coupled frequency-torsional drivetrain dynamics
    over a receding horizon.  Uses wind prediction for anticipatory damping.

  Inner loop (one-step PBQP):  Passivity-based QP computes optimal Vrd/Vrq
    to track rotor flux equilibrium (unchanged from DFIG_RSC_PBQP_PHS).

Key advantage over voltage-based MPC (DFIG_RSC_MPC_PHS):
  - The MPC controls Te INDIRECTLY via frequency/slip (like governor adjusting
    mechanical power), rather than directly perturbing rotor voltage.
  - The PBQP maintains near-equilibrium rotor flux, so terminal voltage
    disturbance is minimized (like exciter maintaining field voltage).
  - Predictive torsional damping works through the physical power-frequency
    channel, not through voltage injection.

MPC prediction model (4-state linearised):
  z = [Δω_vs, twist_rate, Δθ_tw, ξ_P]

  dΔω_vs/dt    = (-Dp·Δω_vs + ΔP + P_net + ξ_P) / J
  d(twist)/dt  = -k_te·Δω/Jg  - D_sh·twist·J_eff_inv - K_sh·Δθ·J_eff_inv + ΔT_aero/Jt
  d(Δθ_tw)/dt  = twist_rate
  dξ_P/dt      = -ki_P · Δω_vs

Decision variable: ΔP = [ΔP_0, ..., ΔP_{N-1}]  (N scalars)
QP size: N×N  (Cholesky-solved).

Cost:
    J = Σ_{k=1}^{N} [α_ω · Δω²_k + α_tw · twist²_k]    (stage + terminal)
      + Σ_{k=0}^{N-1} α_P · ΔP²_k                       (effort penalty)

    Terminal weight: last step uses α_Vf × (stage weights).

States (5, identical to PB-QP / MPC):
    [0] omega_vs   — Virtual angular frequency (APC)
    [1] theta_s    — Virtual angle (integrated from omega_vs)
    [2] phi_Qs     — RPC integral state (ξ_Q)
    [3] xi_P       — APC frequency-error integral (secondary control)
    [4] T_aero_filt — Slow-reference filter for feedforward

Inputs (16):  Same as DfigRscMpcPHS (includes vw_pred for wind forecast).
Outputs (4):  Same as DfigRscMpcPHS.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRscMpcFreqPHS(PowerComponent):
    """
    Frequency-Based Multi-Horizon MPC RSC controller (Σ₅_MPC_F).

    Outer MPC on swing equation (frequency/torque) + inner PBQP on
    rotor voltage (flux tracking).  5 differential states.
    """

    N_MAX = 20  # Maximum prediction horizon (compile-time bound)

    def __init__(self, name: str, params: dict):
        # Outer MPC defaults
        params.setdefault('N_horizon', 10)
        params.setdefault('alpha_omega', 10.0)   # frequency deviation cost
        params.setdefault('alpha_Vf', 5.0)       # terminal multiplier
        params.setdefault('alpha_P', 1.0)         # supplementary power effort cost
        params.setdefault('dP_max', 0.5)          # max supplementary power [pu]
        params.setdefault('dt_pred', 0.05)        # outer MPC prediction step [s]
        # Inner PBQP torsional weight (usually 0 — outer MPC handles it)
        params.setdefault('alpha_tw_pbqp', 0.0)
        # Shared torsional / feedforward defaults
        params.setdefault('alpha_tw', 150.0)
        params.setdefault('gamma_tw', 1.0)
        params.setdefault('alpha_ff', 0.0)
        params.setdefault('Jratio', 0.25)
        params.setdefault('D_t', 0.0)
        params.setdefault('tau_ff', 2.0)
        params.setdefault('K_td', 0.0)
        params.setdefault('K_ff', 0.0)
        # Drivetrain prediction model
        params.setdefault('K_shaft_dt', 0.5)
        params.setdefault('D_shaft_dt', 1.5)
        params.setdefault('Tm0_dt', 1.0)
        params.setdefault('vw_nom_dt', 12.0)
        params.setdefault('eta_kte', 0.15)  # k_te coupling reduction factor
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('P_star',  'signal', 'pu'),   # [0]
                ('Q_star',  'signal', 'pu'),   # [1]
                ('Ps',      'signal', 'pu'),   # [2]
                ('Qs',      'signal', 'pu'),   # [3]
                ('vds',     'signal', 'pu'),   # [4]
                ('vqs',     'signal', 'pu'),   # [5]
                ('ird',     'signal', 'pu'),   # [6]
                ('iqr',     'signal', 'pu'),   # [7]
                ('phi_sd',  'signal', 'pu'),   # [8]
                ('phi_sq',  'signal', 'pu'),   # [9]
                ('omega_r', 'signal', 'pu'),   # [10]
                ('omega_t', 'signal', 'pu'),   # [11]
                ('theta_tw','signal', 'rad'),  # [12]
                ('T_aero',  'signal', 'pu'),   # [13]
                ('T_shaft', 'signal', 'pu'),   # [14]
                ('vw_pred', 'signal', 'pu'),   # [15]  predicted wind speed
            ],
            'out': [
                ('Vrd',     'signal', 'pu'),   # [0]
                ('Vrq',     'signal', 'pu'),   # [1]
                ('P_rotor', 'signal', 'pu'),   # [2]
                ('theta_s', 'signal', 'rad'),  # [3]
                ('dP_opt',  'signal', 'pu'),   # [4]  MPC supplementary power
                ('T_aero_pred','signal','pu'),  # [5]  MPC predicted aero torque
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['omega_vs', 'theta_s', 'phi_Qs', 'xi_P', 'T_aero_filt', 'Cp_filt']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'J':        'Virtual inertia [pu·s²/rad] (APC swing equation)',
            'Dp':       'APC damping / droop gain [pu]',
            'omega_N':  'Nominal angular frequency [pu]',
            'DQ':       'RPC droop coefficient [pu]',
            'kQs':      'RPC integral gain',
            'ki_P':     'APC frequency-error integral gain',
            'omega_b':  'Base angular frequency [rad/s]',
            'Lm':       'DFIG mutual inductance [pu]',
            'Ls':       'DFIG stator inductance [pu]',
            'Lr':       'DFIG rotor inductance [pu]',
            'Rs':       'DFIG stator resistance [pu]',
            'Rr':       'DFIG rotor resistance [pu]',
            'np_poles': 'DFIG pole pairs',
            'j_inertia':'DFIG generator inertia (2H) [pu·s]',
            'f_damp':   'DFIG generator friction [pu]',
            'omega_s':  'Synchronous speed [pu]',
            'Vrd_max':  'Rotor voltage saturation [pu]',
            'I_max':    'Rotor current saturation [pu]',
            'V_nom':    'Nominal terminal voltage from power flow [pu]',
            # Outer MPC parameters
            'N_horizon': 'Prediction horizon (number of steps, max 20)',
            'dt_pred':   'Outer MPC prediction step [s] (decoupled from solver dt)',
            'alpha_omega': 'MPC frequency deviation cost weight',
            'alpha_tw':    'MPC torsional damping cost weight',
            'alpha_P':     'MPC supplementary power effort cost weight',
            'alpha_Vf':    'Terminal multiplier (terminal cost = alpha_Vf x stage cost)',
            'dP_max':      'Maximum supplementary power injection [pu]',
            # Inner PBQP parameters
            'alpha_V':  'PBQP Lyapunov (flux tracking) cost weight',
            'alpha_u':  'PBQP control effort (voltage) cost weight',
            'alpha_tw_pbqp': 'PBQP inner torsional cost (usually 0: outer MPC handles it)',
            'dt_mpc':   'MPC/PBQP prediction step [s]',
            # Feedforward / torsional shared
            'alpha_ff':  'Feedforward torque-matching cost weight (PB-QP compat)',
            'Jratio':    'Generator-to-turbine inertia ratio J_g/J_t',
            'D_t':       'Turbine friction coefficient [pu]',
            'tau_ff':    'Feedforward filter time constant [s]',
            'K_td':      'Freq-domain torsional damping gain [pu] (APC supplementary)',
            'K_ff':      'Freq-domain feedforward gain [pu] (APC supplementary)',
            'gamma_tw':  'Max ratio torsional-to-base Hessian (adaptive clamp)',
            # Drivetrain prediction model
            'K_shaft_dt': 'Drivetrain shaft stiffness (predictive model) [pu/rad]',
            'D_shaft_dt': 'Drivetrain shaft damping (predictive model) [pu]',
            'Tm0_dt':     'Rated per-unit aero torque at vw_nom (predictive model)',
            'vw_nom_dt':  'Nominal wind speed for drivetrain model [m/s]',
            'eta_kte':    'k_Te coupling reduction factor (0..1, default 0.15)',
            # Equilibrium (set during init)
            'phi_sd_ss': 'Stator d-flux at equilibrium [pu]',
            'phi_sq_ss': 'Stator q-flux at equilibrium [pu]',
            'phi_rd_ss': 'Rotor d-flux at equilibrium [pu]',
            'phi_rq_ss': 'Rotor q-flux at equilibrium [pu]',
            'p_g_ss':    'Generator momentum at equilibrium [pu·s]',
            'Vrd_ss':    'Rotor d-voltage at equilibrium [pu]',
            'Vrq_ss':    'Rotor q-voltage at equilibrium [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vrd_cmd':  {'description': 'Rotor d-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Vrq_cmd':  {'description': 'Rotor q-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[1]'},
            'P_rotor':  {'description': 'Rotor electrical power',  'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'theta_s':  {'description': 'Virtual angle (APC)',     'unit': 'rad', 'cpp_expr': 'x[1]'},
            'omega_vs': {'description': 'Virtual frequency (APC)', 'unit': 'pu', 'cpp_expr': 'x[0]'},
            'xi_P':     {'description': 'APC freq-error integral', 'unit': 'pu', 'cpp_expr': 'x[3]'},
            'T_aero_filt': {'description': 'Slow-reference aero torque filter', 'unit': 'pu', 'cpp_expr': 'x[4]'},
            'Cp_filt':     {'description': 'Pitt-Peters Cp inflow filter', 'unit': 'pu', 'cpp_expr': 'x[5]'},
            'dP_opt':   {'description': 'MPC supplementary power', 'unit': 'pu', 'cpp_expr': 'outputs[4]'},
            'T_aero_pred': {'description': 'MPC predicted aero torque', 'unit': 'pu', 'cpp_expr': 'outputs[5]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation — Outer MPC (frequency-torsional)
    # ------------------------------------------------------------------

    def _outer_mpc_solve(self) -> str:
        """C++ block: N-step MPC over swing equation + drivetrain.
        Returns supplementary power dP_opt for the APC.

        Key: uses dt_pred (e.g. 0.05s) for Euler-discretized prediction,
        decoupled from the solver timestep (0.0005s).  This gives the MPC
        enough horizon to see the torsional mode (~8s period).

        4-state prediction model includes ξ_P (secondary frequency integral)
        so the MPC anticipates the integral restoring force and avoids
        overcorrection → oscillation.
        """
        return r"""
            // ============================================================
            // OUTER MPC: Frequency-Torsional Dynamics Prediction
            //
            // Prediction uses dt_pred (>> solver dt) so the horizon
            // covers a meaningful fraction of the torsional period.
            //
            // 4-state model: z = [Δω_vs, twist_rate, Δθ_tw, ξ_P]
            //   dΔω/dt      = (-Dp·Δω + ΔP + P_net + ξ_P) / J
            //   d(twist)/dt = -k_te·Δω/Jg - D_sh·twist·Jinv - K_sh·Δθ·Jinv + ΔT_aero/Jt
            //   d(Δθ)/dt    = twist
            //   dξ_P/dt     = -ki_P · Δω_vs
            //
            // ξ_P is the secondary frequency integral.  Including it as
            // a predicted state lets the MPC anticipate how its own
            // frequency deviation will be partially cancelled by the
            // integral, preventing overshoot and oscillation.
            //
            // Decision: ΔP = [ΔP_0, ..., ΔP_{N-1}]  (N scalars → N×N QP)
            // ============================================================
            double dP_opt;
            double mpc_T_aero_pred;

            {
            // Current measurements (read before APC update)
            double mpc_omega_vs = x[0];
            double mpc_omega_t  = inputs[11];
            double mpc_omega_g  = inputs[10];
            double mpc_theta_tw = inputs[12];
            double mpc_T_aero   = inputs[13];
            double mpc_Ps       = inputs[2];
            double mpc_Pstar    = inputs[0];

            double mpc_twist_rate = mpc_omega_t - mpc_omega_g;

            // Drivetrain inertias
            double mpc_Jg = j_inertia;
            double mpc_Jt = mpc_Jg / fmax(Jratio, 0.01);
            double mpc_Jinv = 1.0 / mpc_Jt + 1.0 / mpc_Jg;  // 1/J_reduced

            // Te sensitivity to ω_vs.
            //
            // The quasi-steady-state derivation gives k_te = Dp/omega_N,
            // but this OVERESTIMATES the actual coupling because:
            //   1. P_s doesn't respond instantly to ω_vs — the inner PB-QP
            //      tracks flux with finite bandwidth (~10 ms settling)
            //   2. The network impedance attenuates the power-angle response
            //   3. The generator inertia smooths ω_g response to T_e changes
            //
            // A reduced effective sensitivity k_te_eff accounts for these
            // dynamics.  Empirically: k_te_eff ≈ 0.15 × k_te_ideal gives
            // MPC predictions consistent with observed twist_rate changes.
            double mpc_k_te_ideal = Dp / fmax(omega_N, 0.5);
            double mpc_k_te = eta_kte * mpc_k_te_ideal;

            // Equilibrium θ_tw (from slow T_aero filter)
            double mpc_T_filt = x[4];
            double mpc_theta_eq = mpc_T_filt / fmax(K_shaft_dt, 0.01);

            // Initial 4-state error
            double mpc_z0[4];
            mpc_z0[0] = mpc_omega_vs - omega_N;        // Δω_vs
            mpc_z0[1] = mpc_twist_rate;                 // twist rate
            mpc_z0[2] = mpc_theta_tw - mpc_theta_eq;    // Δθ_tw
            mpc_z0[3] = x[3];                           // ξ_P (absolute)

            // Net power driving (constant over horizon, ξ_P excluded — it's a state)
            double mpc_Pnet = mpc_Pstar - mpc_Ps;  // P*-Pe (no ξ_P)

            // ============================================================
            // Aerodynamic torque prediction — Heier Cp + inflow filter
            //
            // Uses the SAME Cp(λ,β) model as the drivetrain, with a
            // dynamic Pitt-Peters inflow filter (state x[5] = Cp_filt)
            // that mirrors the drivetrain's Cp_eff dynamics.
            //
            // Constant base torque Tm0_dt is set once during
            // initialisation (from Pe_ss) — never updated live, avoiding
            // the double-counting bug where T_filt already contains the
            // vw³·Cp effect.
            //
            // T_pred = Tm0_dt · vw_pred³ · Cp_filt / ω_t
            // ΔT_aero = T_pred − T_aero_filt
            // ============================================================
            double mpc_vw_pred = inputs[15];
            double mpc_vw_pu = mpc_vw_pred / fmax(vw_nom_dt, 1.0);
            double mpc_omega_t_safe = fmax(fabs(mpc_omega_t), 0.01);

            // Use the Cp inflow filter state (tracks drivetrain dynamics)
            double mpc_Cp_filt = x[5];

            // Predicted aerodynamic torque with constant Tm0 and filtered Cp
            mpc_T_aero_pred = Tm0_dt * mpc_vw_pu * mpc_vw_pu * mpc_vw_pu
                            * mpc_Cp_filt / mpc_omega_t_safe;

            // Disturbance: predicted T_aero minus slow baseline
            double mpc_dT_aero = mpc_T_aero_pred - mpc_T_filt;

            // P_ff (feedforward) — constant over horizon, enters as disturbance
            double mpc_ff_deficit = -Jratio * (mpc_T_aero - mpc_T_filt);
            double mpc_P_ff = K_ff * mpc_ff_deficit;

            // 4×4 continuous-time A matrix
            // A[3] sign: POSITIVE.  Physical chain:
            //   Δω_vs ↑ → slip ↑ → T_e ↑ → generator decelerates
            //   → ω_g ↓ → twist_rate = (ω_t − ω_g) ↑
            //   ∴ d(twist_rate)/d(Δω_vs) = +k_te / J_g  > 0
            double mpc_A[16];
            // Row 0: dΔω/dt
            mpc_A[0]  = -Dp / J;                       // dΔω/dΔω
            mpc_A[1]  = -K_td / J;                     // dΔω/d(twist) from P_td
            mpc_A[2]  =  0.0;                           // dΔω/d(Δθ)
            mpc_A[3]  =  1.0 / J;                       // dΔω/dξ_P  (integral → swing)
            // Row 1: d(twist)/dt
            mpc_A[4]  =  mpc_k_te / mpc_Jg;            // d(twist)/dΔω  [POSITIVE]
            mpc_A[5]  = -D_shaft_dt * mpc_Jinv;        // d(twist)/d(twist)
            mpc_A[6]  = -K_shaft_dt * mpc_Jinv;        // d(twist)/d(Δθ)
            mpc_A[7]  =  0.0;                           // d(twist)/dξ_P
            // Row 2: d(Δθ)/dt
            mpc_A[8]  =  0.0;                           // d(Δθ)/dΔω
            mpc_A[9]  =  1.0;                           // d(Δθ)/d(twist)
            mpc_A[10] =  0.0;                           // d(Δθ)/d(Δθ)
            mpc_A[11] =  0.0;                           // d(Δθ)/dξ_P
            // Row 3: dξ_P/dt = ki_P·(ω_N - ω_vs) = -ki_P·Δω_vs
            mpc_A[12] = -ki_P;                          // dξ_P/dΔω
            mpc_A[13] =  0.0;                           // dξ_P/d(twist)
            mpc_A[14] =  0.0;                           // dξ_P/d(Δθ)
            mpc_A[15] =  0.0;                           // dξ_P/dξ_P

            // 4×1 B vector (ΔP → states: only affects swing equation)
            double mpc_B[4] = {1.0 / J, 0.0, 0.0, 0.0};

            // Disturbance per step (constant over horizon)
            double mpc_w[4];
            mpc_w[0] = dt_pred * (mpc_Pnet + mpc_P_ff) / J;
            mpc_w[1] = dt_pred * mpc_dT_aero / mpc_Jt;
            mpc_w[2] = 0.0;
            mpc_w[3] = 0.0;

            // Discrete-time: Φ = I + dt_pred·A,  Γ = dt_pred·B   (4×4)
            double mpc_Phi[16];
            for (int ii = 0; ii < 16; ii++) mpc_Phi[ii] = dt_pred * mpc_A[ii];
            mpc_Phi[0] += 1.0; mpc_Phi[5] += 1.0; mpc_Phi[10] += 1.0; mpc_Phi[15] += 1.0;

            double mpc_Gam[4];
            for (int ii = 0; ii < 4; ii++) mpc_Gam[ii] = dt_pred * mpc_B[ii];

            // Φ powers: Φ^0, Φ^1, ..., Φ^{N-1}   (each 4×4 = 16 entries)
            int mpc_Nh = (int)fmin(fmax(N_horizon, 1.0), 20.0);

            double mpc_Phipow[320];  // max 20 × 16
            // Φ^0 = I
            for (int ii = 0; ii < 16; ii++) mpc_Phipow[ii] = 0.0;
            mpc_Phipow[0] = 1.0; mpc_Phipow[5] = 1.0; mpc_Phipow[10] = 1.0; mpc_Phipow[15] = 1.0;
            // Φ^1 = Φ
            if (mpc_Nh > 1)
                for (int ii = 0; ii < 16; ii++) mpc_Phipow[16 + ii] = mpc_Phi[ii];
            // Φ^k = Φ · Φ^{k-1}   (4×4 × 4×4)
            for (int kk = 2; kk < mpc_Nh; kk++) {
                for (int rr = 0; rr < 4; rr++) {
                    for (int cc = 0; cc < 4; cc++) {
                        double ss = 0.0;
                        for (int mm = 0; mm < 4; mm++)
                            ss += mpc_Phi[rr*4+mm] * mpc_Phipow[(kk-1)*16 + mm*4+cc];
                        mpc_Phipow[kk*16 + rr*4+cc] = ss;
                    }
                }
            }

            // Free response: propagate z_0 with constant disturbance (no control)
            double mpc_zfree[80];  // max 20 × 4
            double mpc_zcur[4] = {mpc_z0[0], mpc_z0[1], mpc_z0[2], mpc_z0[3]};

            for (int kk = 0; kk < mpc_Nh; kk++) {
                double zn[4];
                for (int rr = 0; rr < 4; rr++) {
                    zn[rr] = mpc_w[rr];
                    for (int cc = 0; cc < 4; cc++)
                        zn[rr] += mpc_Phi[rr*4+cc] * mpc_zcur[cc];
                }
                for (int rr = 0; rr < 4; rr++) {
                    mpc_zcur[rr] = zn[rr];
                    mpc_zfree[kk*4 + rr] = zn[rr];
                }
            }

            // ============================================================
            // Build condensed QP: H · u = rhs
            //
            // Cost = Σ_{k=0}^{N-1} [α_ω_k · c_k[0]² + α_tw_k · c_k[1]²]
            //      + Σ_{k=0}^{N-1} α_P · u_k²
            //
            // c_k = z_free at predicted step k+1
            // Impulse response: Ψ_{k+1,a} = Φ^{k-a} · Γ
            // ============================================================

            double mpc_H[400] = {0.0};  // max 20×20
            double mpc_g[20]  = {0.0};

            for (int kk = 0; kk < mpc_Nh; kk++) {
                // Terminal step gets α_Vf multiplier
                double aw_k = (kk == mpc_Nh - 1) ? alpha_omega * alpha_Vf : alpha_omega;
                double at_k = (kk == mpc_Nh - 1) ? alpha_tw    * alpha_Vf : alpha_tw;

                // Free-response error at predicted step kk+1
                double c0 = mpc_zfree[kk*4];      // Δω_vs
                double c1 = mpc_zfree[kk*4 + 1];  // twist_rate

                // For each control input a that can affect step kk+1 (a ≤ kk)
                for (int a = 0; a <= kk; a++) {
                    int p_idx = kk - a;  // Φ power index

                    // Ψ_{kk+1,a} = Φ^{kk-a} · Γ   (4×1, but we only need rows 0,1)
                    double psi_a0 = 0.0, psi_a1 = 0.0;
                    for (int mm = 0; mm < 4; mm++) {
                        psi_a0 += mpc_Phipow[p_idx*16 + 0*4+mm] * mpc_Gam[mm];
                        psi_a1 += mpc_Phipow[p_idx*16 + 1*4+mm] * mpc_Gam[mm];
                    }

                    // Gradient contribution
                    mpc_g[a] += aw_k * c0 * psi_a0 + at_k * c1 * psi_a1;

                    // Hessian contribution (symmetric)
                    for (int b = 0; b <= a; b++) {
                        int q_idx = kk - b;
                        double psi_b0 = 0.0, psi_b1 = 0.0;
                        for (int mm = 0; mm < 4; mm++) {
                            psi_b0 += mpc_Phipow[q_idx*16 + 0*4+mm] * mpc_Gam[mm];
                            psi_b1 += mpc_Phipow[q_idx*16 + 1*4+mm] * mpc_Gam[mm];
                        }
                        double hv = aw_k * psi_a0 * psi_b0 + at_k * psi_a1 * psi_b1;
                        mpc_H[a * mpc_Nh + b] += hv;
                        if (a != b) mpc_H[b * mpc_Nh + a] += hv;
                    }
                }
            }

            // Regularisation (α_P on diagonal)
            for (int jj = 0; jj < mpc_Nh; jj++)
                mpc_H[jj * mpc_Nh + jj] += alpha_P;

            // RHS
            double mpc_rhs[20];
            for (int jj = 0; jj < mpc_Nh; jj++)
                mpc_rhs[jj] = -mpc_g[jj];

            // ============================================================
            // Cholesky solve (N×N, max 20×20)
            // ============================================================
            double mpc_L[400] = {0.0};
            for (int ii = 0; ii < mpc_Nh; ii++) {
                for (int jj = 0; jj <= ii; jj++) {
                    double ss = 0.0;
                    for (int kk = 0; kk < jj; kk++)
                        ss += mpc_L[ii * mpc_Nh + kk] * mpc_L[jj * mpc_Nh + kk];
                    if (ii == jj) {
                        double dg = mpc_H[ii * mpc_Nh + ii] - ss;
                        mpc_L[ii * mpc_Nh + jj] = (dg > 1e-12) ? sqrt(dg) : 1e-6;
                    } else {
                        mpc_L[ii * mpc_Nh + jj] =
                            (mpc_H[ii * mpc_Nh + jj] - ss) / mpc_L[jj * mpc_Nh + jj];
                    }
                }
            }

            // Forward substitution
            double mpc_y[20] = {0.0};
            for (int ii = 0; ii < mpc_Nh; ii++) {
                double ss = 0.0;
                for (int jj = 0; jj < ii; jj++)
                    ss += mpc_L[ii * mpc_Nh + jj] * mpc_y[jj];
                mpc_y[ii] = (mpc_rhs[ii] - ss) / mpc_L[ii * mpc_Nh + ii];
            }

            // Back substitution
            double mpc_u[20] = {0.0};
            for (int ii = mpc_Nh - 1; ii >= 0; ii--) {
                double ss = 0.0;
                for (int jj = ii + 1; jj < mpc_Nh; jj++)
                    ss += mpc_L[jj * mpc_Nh + ii] * mpc_u[jj];
                mpc_u[ii] = (mpc_y[ii] - ss) / mpc_L[ii * mpc_Nh + ii];
            }

            // Extract first control action and clamp
            dP_opt = mpc_u[0];
            dP_opt = fmax(-dP_max, fmin(dP_max, dP_opt));
            }
"""

    # ------------------------------------------------------------------
    # C++ Code Generation — Inner PBQP (one-step Vrd/Vrq solver)
    # ------------------------------------------------------------------

    def _inner_pbqp_solve(self) -> str:
        """C++ block: one-step passivity-based QP for Vrd/Vrq.
        Uses alpha_tw_pbqp (usually 0) for inner torsional cost.
        Called after frame rotation; expects phi_sd, phi_sq, ird_m, iqr_m,
        vds_m, vqs_m, omega_r_m in scope."""
        return r"""
            // ============================================================
            // INNER PBQP: Reconstruct DFIG EM state from inputs
            // ============================================================
            double sigma_LS = Ls * Lr - Lm * Lm;

            double phi_sd_loc = phi_sd;
            double phi_sq_loc = phi_sq;

            double phi_rd_loc = (sigma_LS * ird_m + Lm * phi_sd_loc) / Ls;
            double phi_rq_loc = (sigma_LS * iqr_m + Lm * phi_sq_loc) / Ls;

            double omega_g = omega_r_m;
            double p_g_loc = j_inertia * omega_g;

            double i_sd_m =  (Lr * phi_sd_loc - Lm * phi_rd_loc) / sigma_LS;
            double i_sq_m =  (Lr * phi_sq_loc - Lm * phi_rq_loc) / sigma_LS;

            // ============================================================
            // INNER PBQP: Equilibrium state
            // ============================================================
            double phi_sd_eq = phi_sd_ss;
            double phi_sq_eq = phi_sq_ss;
            double phi_rd_eq = phi_rd_ss;
            double phi_rq_eq = phi_rq_ss;
            double p_g_eq    = p_g_ss;

            double z0 = phi_sd_loc - phi_sd_eq;
            double z1 = phi_sq_loc - phi_sq_eq;
            double z2 = phi_rd_loc - phi_rd_eq;
            double z3 = phi_rq_loc - phi_rq_eq;
            double z4 = p_g_loc    - p_g_eq;

            // ============================================================
            // INNER PBQP: Autonomous dynamics (J₃ − R₃)∇H₃ + external
            //             using CURRENT virtual frequency ω_vs = x[0]
            // ============================================================
            double omega_vs_cur = x[0];
            double omega_slip = omega_vs_cur - omega_g;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));

            double Vsd_ext = vds_m;
            double Vsq_ext = vqs_m;

            double JdH0 =  omega_vs_cur * (Ls * i_sq_m + Lm * iqr_m);
            double JdH1 = -omega_vs_cur * (Ls * i_sd_m + Lm * ird_m);
            double JdH2 =  omega_vs_cur * (Lm * i_sq_m + Lr * iqr_m)
                           - np_poles * phi_rq_loc * omega_g;
            double JdH3 = -omega_vs_cur * (Lm * i_sd_m + Lr * ird_m)
                           + np_poles * phi_rd_loc * omega_g;
            double JdH4 =  np_poles * (phi_rq_loc * ird_m - phi_rd_loc * iqr_m);

            double RdH0 = Rs * i_sd_m;
            double RdH1 = Rs * i_sq_m;
            double RdH2 = Rr * ird_m;
            double RdH3 = Rr * iqr_m;
            double RdH4 = f_damp * omega_g;

            double f0 = omega_b * (JdH0 - RdH0 + Vsd_ext);
            double f1 = omega_b * (JdH1 - RdH1 + Vsq_ext);
            double f2 = omega_b * (JdH2 - RdH2);
            double f3 = omega_b * (JdH3 - RdH3);
            double f4 = JdH4 - RdH4;

            // Predicted state without control
            double p0 = z0 + dt_mpc * f0;
            double p1 = z1 + dt_mpc * f1;
            double p2 = z2 + dt_mpc * f2;
            double p3 = z3 + dt_mpc * f3;
            double p4 = z4 + dt_mpc * f4;

            // ============================================================
            // INNER PBQP: 2×2 QP (Lyapunov + regularisation)
            //
            // Block-diagonal Q_r: only rotor flux error drives Vrd/Vrq.
            // ============================================================
            double dtw = dt_mpc * omega_b;

            // Torsional damping signals (for inner PBQP — usually alpha_tw_pbqp=0)
            double pbqp_omega_t  = inputs[11];
            double pbqp_omega_g  = omega_g;
            double pbqp_twist    = pbqp_omega_t - pbqp_omega_g;
            double pbqp_T_aero   = inputs[13];

            double a_rd_pbqp = -np_poles * dtw * iqr_m;
            double a_rq_pbqp =  np_poles * dtw * ird_m;
            double r_tw_pbqp = -pbqp_twist;

            // Slow-reference feedforward deficit (for inner PBQP)
            double pbqp_deficit_raw = -Jratio * (pbqp_T_aero - x[4]);
            double pbqp_deficit = pbqp_deficit_raw;
            if (fabs(pbqp_deficit) < 0.02) pbqp_deficit = 0.0;
            r_tw_pbqp += pbqp_deficit;

            // Adaptive torsional weight clamping (inner PBQP)
            double a_sq_pbqp = a_rd_pbqp * a_rd_pbqp + a_rq_pbqp * a_rq_pbqp;
            double atw_eff_pbqp = alpha_tw_pbqp;
            if (a_sq_pbqp > 1e-12 && alpha_tw_pbqp > 1e-12) {
                double H_base_diag_pbqp = alpha_V * dtw * dtw * (Ls / sigma_LS) + alpha_u;
                double atw_max_pbqp = gamma_tw * H_base_diag_pbqp / a_sq_pbqp;
                if (alpha_tw_pbqp > atw_max_pbqp)
                    atw_eff_pbqp = atw_max_pbqp;
            }

            // B^T Q_r p  (block-diagonal rotor-only Lyapunov)
            double BtQp_0 = (Ls * p2) / sigma_LS;
            double BtQp_1 = (Ls * p3) / sigma_LS;
            double BtQB_diag = Ls / sigma_LS;

            // Feedforward RHS (inner PBQP)
            double ff_rhs0_pbqp = 0.0, ff_rhs1_pbqp = 0.0;
            if (a_sq_pbqp > 1e-12 && fabs(alpha_ff) > 1e-12) {
                double a_norm_pbqp = sqrt(a_sq_pbqp);
                double dTe_max_pbqp = a_norm_pbqp * Vrd_max;
                double deficit_clamp_pbqp = fmax(-dTe_max_pbqp, fmin(dTe_max_pbqp, pbqp_deficit));
                ff_rhs0_pbqp = alpha_ff * a_rd_pbqp * deficit_clamp_pbqp;
                ff_rhs1_pbqp = alpha_ff * a_rq_pbqp * deficit_clamp_pbqp;
            }

            // 2×2 QP: H · [Vrd; Vrq] = rhs
            double H00 = alpha_V * dtw * dtw * BtQB_diag + alpha_u
                       + atw_eff_pbqp * a_rd_pbqp * a_rd_pbqp;
            double H01 = atw_eff_pbqp * a_rd_pbqp * a_rq_pbqp;
            double H11 = alpha_V * dtw * dtw * BtQB_diag + alpha_u
                       + atw_eff_pbqp * a_rq_pbqp * a_rq_pbqp;

            double rhs0 = -alpha_V * dtw * BtQp_0 + alpha_u * Vrd_ss
                        + atw_eff_pbqp * a_rd_pbqp * r_tw_pbqp + ff_rhs0_pbqp;
            double rhs1 = -alpha_V * dtw * BtQp_1 + alpha_u * Vrq_ss
                        + atw_eff_pbqp * a_rq_pbqp * r_tw_pbqp + ff_rhs1_pbqp;

            // Solve 2×2 via Cramer's rule
            double det_H = H00 * H11 - H01 * H01;
            if (fabs(det_H) < 1e-12) det_H = 1e-12;
            double Vrd_opt = ( H11 * rhs0 - H01 * rhs1) / det_H;
            double Vrq_opt = (-H01 * rhs0 + H00 * rhs1) / det_H;

            // ============================================================
            // Passivity-based damping injection on rotor flux derivative
            // ============================================================
            double R_inj = 0.05;
            double damp_denom = 1.0 + R_inj * omega_b;
            Vrd_opt = (Vrd_opt - R_inj * f2) / damp_denom;
            Vrq_opt = (Vrq_opt - R_inj * f3) / damp_denom;

            // ============================================================
            // Constraint projection
            // ============================================================
            Vrd_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrd_opt));
            Vrq_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrq_opt));

            double sigma_r = Lr - Lm * Lm / Ls;
            double i_rd_pred = ird_m + dtw * Vrd_opt / sigma_r;
            double i_rq_pred = iqr_m + dtw * Vrq_opt / sigma_r;
            double I_pred_mag = sqrt(i_rd_pred * i_rd_pred + i_rq_pred * i_rq_pred);
            if (I_pred_mag > I_max && I_pred_mag > 1e-6) {
                double sc = I_max / I_pred_mag;
                Vrd_opt = Vrd_ss + sc * (Vrd_opt - Vrd_ss);
                Vrq_opt = Vrq_ss + sc * (Vrq_opt - Vrq_ss);
            }

            double Vrd_cmd = Vrd_opt;
            double Vrq_cmd = Vrq_opt;
"""

    # ------------------------------------------------------------------
    # compute_outputs  (algebraic: inner PBQP only, no outer MPC)
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];
""" + self._inner_pbqp_solve() + """
            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
            outputs[4] = 0.0;   // dP_opt not available in compute_outputs
            outputs[5] = 0.0;   // T_aero_pred not available in compute_outputs
        """

    # ------------------------------------------------------------------
    # step  (dynamic: outer MPC → APC → frame rotation → RPC → inner PBQP)
    # ------------------------------------------------------------------

    def get_cpp_step_code(self) -> str:
        return r"""
            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];
""" + self._outer_mpc_solve() + r"""
            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //         Includes MPC supplementary power dP_opt
            //
            // J·dω_vs/dt = P* - Ps - Dp·(ω_vs-ω_N) + ξ_P + dP_opt
            //              + P_td + P_ff     (legacy: usually K_td=K_ff=0)
            // ==========================================================
            double omega_vs = x[0];
            double xi_P = x[3];

            // Legacy supplementary signals (usually disabled: K_td=K_ff=0)
            double omega_t_apc = inputs[11];
            double omega_g_apc = inputs[10];
            double twist_rate  = omega_t_apc - omega_g_apc;
            double P_td = -K_td * twist_rate;

            double T_aero_apc  = inputs[13];
            double ff_deficit  = -Jratio * (T_aero_apc - x[4]);
            double P_ff = K_ff * ff_deficit;

            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)
                       + xi_P + dP_opt + P_td + P_ff) / J;
            dxdt[1] = omega_vs - omega_N;

            // Secondary frequency control: integral of frequency error
            dxdt[3] = ki_P * (omega_N - omega_vs);
            double xi_P_max = 0.5;
            if (x[3] > xi_P_max && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -xi_P_max && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ==========================================================
            // STEP 2: Frame rotation: RI → virtual dq (theta_s)
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 3: RPC — Reactive Power Controller
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3b: Slow-reference filter on T_aero for feedforward
            // ==========================================================
            double T_aero_step = inputs[13];
            if (tau_ff > 1e-12) {
                double ff_err = T_aero_step - x[4];
                double tau_eff = (fabs(ff_err) > 0.1) ? 0.01 : tau_ff;
                dxdt[4] = ff_err / tau_eff;
            } else {
                dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 3c: Cp inflow filter (Pitt-Peters dynamic inflow)
            //
            // Mirrors the drivetrain's Cp_eff filter so the MPC's T_aero
            // prediction matches the actual dynamic Cp behaviour.
            //   dCp_filt/dt = (Cp_emp − Cp_filt) / τ_i
            //   τ_i = 1 / vw_pu   (per-unit Pitt-Peters)
            // ==========================================================
            {
                double vw_step = inputs[15];  // predicted wind (= actual for now)
                double vw_pu_step = vw_step / fmax(vw_nom_dt, 1.0);
                double omega_t_step = fmax(fabs(inputs[11]), 0.01);
                double lambda_step = vw_pu_step / omega_t_step;
                double beta_step = 0.0;
                double li_step = 1.0 / (1.0 / (lambda_step + 0.08 * beta_step)
                                        - 0.035 / (beta_step * beta_step + 1.0));
                double Cp_raw_step = 0.5176 * (116.0 * li_step - 0.4 * beta_step - 5.0)
                                   * exp(-21.0 * li_step) + 0.0068 * lambda_step;
                // Normalise by rated Cp (same constants as MPC block)
                double li0 = 1.0 / (1.0 - 0.035);
                double Cp0 = 0.5176 * (116.0 * li0 - 5.0) * exp(-21.0 * li0) + 0.0068;
                double Cp_emp_step = fmax(0.0, Cp_raw_step / fmax(Cp0, 1e-6));
                if (Cp_emp_step > 2.0) Cp_emp_step = 2.0;
                double tau_i_step = 1.0 / fmax(vw_pu_step, 0.1);
                dxdt[5] = (Cp_emp_step - x[5]) / fmax(tau_i_step, 0.01);
            }
""" + self._inner_pbqp_solve() + r"""
            // ==========================================================
            // STEP 4: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];
            outputs[4] = dP_opt;
            outputs[5] = mpc_T_aero_pred;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS (outer-loop structure, identical to PB-QP)
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        omega_vs_, theta_s_ = sp.symbols('omega_vs theta_s')
        phi_Qs_ = sp.Symbol('phi_Qs')
        xi_P_ = sp.Symbol('xi_P')
        states = [omega_vs_, theta_s_, phi_Qs_, xi_P_]

        J_sym = sp.Symbol('J', positive=True)
        Dp_sym = sp.Symbol('Dp', nonnegative=True)
        ki_P_sym = sp.Symbol('ki_P', nonnegative=True)

        H_expr = (sp.Rational(1, 2) * J_sym * omega_vs_**2
                  + sp.Rational(1, 2) * (theta_s_**2 + phi_Qs_**2 + xi_P_**2))

        J_mat = sp.zeros(4, 4)
        J_mat[0, 1] = -1
        J_mat[1, 0] = 1
        J_mat[0, 3] = 1
        J_mat[3, 0] = -1

        R_mat = sp.zeros(4, 4)
        R_mat[0, 0] = Dp_sym / J_sym

        g_mat = sp.eye(4)

        u_Ps = sp.Symbol('u_Ps')
        u_theta = sp.Symbol('u_theta')
        u_Qs = sp.Symbol('u_Qs')
        u_xi = sp.Symbol('u_xi')
        inputs = [u_Ps, u_theta, u_Qs, u_xi]

        return SymbolicPHS(
            name='DFIG_RSC_MPC_FREQ_PHS',
            states=states, inputs=inputs,
            params={'J': J_sym, 'Dp': Dp_sym, 'ki_P': ki_P_sym},
            J=J_mat, R=R_mat, g=g_mat, H=H_expr,
            description=(
                'Frequency-based MPC RSC (Σ₅_MPC_F): VSG + RPC + freq integral, '
                '4 PHS states.  Outer MPC on frequency, inner PBQP on rotor voltage.'
            ),
        )

    # ------------------------------------------------------------------
    # Initialization (same algorithm as PB-QP / voltage-based MPC)
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params

        # Defaults (backward compatible)
        p.setdefault('alpha_tw', 150.0)
        p.setdefault('alpha_tw_pbqp', 0.0)
        p.setdefault('gamma_tw', 1.0)
        p.setdefault('alpha_ff', 0.0)
        p.setdefault('tau_ff', 2.0)
        p.setdefault('K_td', 0.0)
        p.setdefault('K_ff', 0.0)
        p.setdefault('N_horizon', 10)
        p.setdefault('alpha_Vf', 5.0)
        p.setdefault('alpha_omega', 10.0)
        p.setdefault('alpha_P', 1.0)
        p.setdefault('dP_max', 0.5)
        p.setdefault('dt_pred', 0.05)

        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        Rs = p['Rs']; Rr = p['Rr']
        omega_s_nom = p['omega_s']
        DQ = p['DQ']; kQs = p['kQs']
        omega_N = p['omega_N']
        np_poles = p['np_poles']
        j_inertia = p['j_inertia']
        sigma_LS = Ls * Lr - Lm**2

        # Steady-state values from power flow (RI frame)
        i_rd_ri   = targets.get('i_rd', 0.0)
        i_rq_ri   = targets.get('i_rq', 0.0)
        Pe_ss     = targets.get('Pe', 0.0)
        Qe_ss     = targets.get('Qe', 0.0)
        omega_ss  = targets.get('omega', omega_s_nom)
        phi_sd_ri = targets.get('phi_sd', 0.0)
        phi_sq_ri = targets.get('phi_sq', 0.0)
        Vrd_ri    = targets.get('Vrd', 0.0)
        Vrq_ri    = targets.get('Vrq', 0.0)
        vds_ri    = targets.get('vd_ri', targets.get('vd', 1.0))
        vqs_ri    = targets.get('vq_ri', targets.get('vq', 0.0))

        Pref = targets.get('Pref', Pe_ss)
        Qref = targets.get('Qref', Qe_ss)
        self.params['Pref0'] = float(Pref)
        self.params['Qref0'] = float(Qref)

        # APC equilibrium: omega_s = omega_N
        omega_s_ss = omega_N
        theta_s_ss = targets.get('theta_s', targets.get('a0', 0.0))

        # Rotate all RI-frame quantities to theta_s frame
        cos_ts = np.cos(theta_s_ss)
        sin_ts = np.sin(theta_s_ss)

        vds_ss   =  vds_ri * cos_ts + vqs_ri * sin_ts
        vqs_ss   = -vds_ri * sin_ts + vqs_ri * cos_ts
        i_rd_ss  =  i_rd_ri * cos_ts + i_rq_ri * sin_ts
        i_rq_ss  = -i_rd_ri * sin_ts + i_rq_ri * cos_ts
        phi_sd_ss =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts
        phi_sq_ss = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts
        Vrd_ss   =  Vrd_ri * cos_ts + Vrq_ri * sin_ts
        Vrq_ss   = -Vrd_ri * sin_ts + Vrq_ri * cos_ts

        self.params['V_nom'] = float(vds_ss)

        # Rotor fluxes from currents
        phi_rd_ss = (sigma_LS * i_rd_ss + Lm * phi_sd_ss) / Ls
        phi_rq_ss = (sigma_LS * i_rq_ss + Lm * phi_sq_ss) / Ls
        p_g_ss = j_inertia * omega_ss

        # Store as compiled constants
        self.params['phi_sd_ss'] = float(phi_sd_ss)
        self.params['phi_sq_ss'] = float(phi_sq_ss)
        self.params['phi_rd_ss'] = float(phi_rd_ss)
        self.params['phi_rq_ss'] = float(phi_rq_ss)
        self.params['p_g_ss']    = float(p_g_ss)
        self.params['Vrd_ss']    = float(Vrd_ss)
        self.params['Vrq_ss']    = float(Vrq_ss)

        # Auto-compute Tm0 for the predictive model.
        # Must match the drivetrain's formula:
        #   Tm0 = Te_gen + f_damp_gen + D_t
        # where Te_gen ≈ Pe_ss, f_damp_gen accounts for mechanical damping
        # on the generator side, and D_t is the turbine friction.
        f_damp_gen = float(targets.get('f_damp_gen', 0.0))
        D_t_dt = float(self.params.get('D_t_dt', 0.05))
        self.params['Tm0_dt'] = float(abs(Pe_ss) + f_damp_gen + D_t_dt)

        return self._init_states({
            'omega_vs': omega_s_ss,
            'theta_s':  theta_s_ss,
            'phi_Qs':   0.0,
            'xi_P':     0.0,
            'T_aero_filt': float(Pe_ss),
            'Cp_filt':  1.0,
        })
