"""
DFIG_RSC_MPC_PHS — Multi-Horizon MPC Rotor-Side Converter Controller (Σ₅_MPC)
=====================================================================================

True Model Predictive Controller with N-step prediction horizon for the
DFIG rotor-side converter.  Unlike the PB-QP controller (Σ₅_PBQP) which
solves a one-step analytical QP, this controller:

  1. Linearises the Σ₃ (DFIG EM) dynamics around the current operating point.
  2. Propagates the frozen-linearisation model N steps forward (condensed form).
  3. Minimises a multi-step Lyapunov cost with a heavier terminal weight
     (Lyapunov-MPC guarantees passivity).
  4. Adds predictive torsional damping using a predicted wind speed profile to
     anticipate T_aero changes and pre-emptively adjust Te via the RSC.
  5. Solves the resulting 2N×2N dense QP via Cholesky factorisation.
  6. Applies only the first control step (receding-horizon principle).

**Retained from PB-QP / GFM (unchanged):**
  - VSG swing equation (Active Power Controller)
  - Reactive Power Controller (droop + integral)
  - Torsional damping / feedforward in APC supplementary loop
  - Passivity-based damping injection on rotor voltage
  - Block-diagonal Q_r (rotor-only Lyapunov gradient for fault robustness)

**New compared to PB-QP:**
  - Multi-step prediction horizon N (default 3)
  - Terminal Lyapunov weight α_Vf >> α_V (stability guarantee)
  - Explicit model-based propagation via linearised Σ₃ dynamics
  - Predictive torsional damping: uses predicted wind to anticipate T_aero,
    adjusts Te over the horizon to minimise shaft tension (ΔTe ≈ −ΔT_shaft)
  - Dense QP solver (Cholesky + box projection)

Cost function (N-step):

    J = Σ_{k=1}^{N-1} α_V · V₃(z_k) + α_Vf · V₃(z_N)
        + Σ_{k=0}^{N-1} α_u · ||u_k − u*||²
        + Σ_{k=0}^{N-1} α_tw · (ΔTe_k + twist_target_k)²

    where twist_target_k incorporates predicted T_aero change from wind forecast.

    s.t.  z_{k+1} = Φ z_k + Γ Δu_k + d   (linearised Σ₃)
          |V_rd|, |V_rq| ≤ V_max
          i_rd² + i_rq² ≤ I_max²

States (6):
    [0] omega_vs  — Virtual angular frequency (APC)
    [1] theta_s   — Virtual angle (APC, integrated from omega_s)
    [2] phi_Qs    — RPC integral state (ξ_Q)
    [3] xi_P      — APC frequency-error integral (secondary control)
    [4] T_aero_filt — Slow-reference filter for feedforward
    [5] Cp_filt   — Dynamic inflow Cp filter (Pitt-Peters)

Inputs (16):  Same as DfigRscPbqpPHS [0..14] + vw_pred [15].
Outputs (4):  Same as DfigRscPbqpPHS.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRscMpcPHS(PowerComponent):
    """
    Multi-Horizon MPC RSC controller (Σ₅_MPC) in Port-Hamiltonian form.

    VSG swing equation + RPC droop + N-step Lyapunov-MPC with
    Cholesky-based dense QP solver.  5 differential states.
    """

    # Maximum prediction horizon (compile-time bound for static C++ arrays)
    N_MAX = 5

    def __init__(self, name: str, params: dict):
        # MPC-specific defaults
        params.setdefault('N_horizon', 3)
        params.setdefault('alpha_Vf', 50.0)
        # PB-QP-inherited defaults
        params.setdefault('alpha_tw', 0.0)
        params.setdefault('gamma_tw', 1.0)
        params.setdefault('alpha_ff', 0.0)
        params.setdefault('Jratio', 0.25)
        params.setdefault('D_t', 0.0)
        params.setdefault('tau_ff', 2.0)
        params.setdefault('K_td', 0.0)
        params.setdefault('K_ff', 0.0)
        # Drivetrain model params for predictive torsional damping
        params.setdefault('K_shaft_dt', 0.5)
        params.setdefault('D_shaft_dt', 1.5)
        params.setdefault('Tm0_dt', 1.0)
        params.setdefault('vw_nom_dt', 12.0)
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
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['omega_vs', 'theta_s', 'phi_Qs', 'xi_P', 'T_aero_filt', 'Cp_filt']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'J':        'Virtual inertia [pu·s²/rad]',
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
            # MPC multi-horizon parameters
            'N_horizon': 'Prediction horizon (number of steps, max 5)',
            'alpha_V':   'Stage Lyapunov cost weight',
            'alpha_u':   'Control effort cost weight',
            'alpha_Vf':  'Terminal Lyapunov cost weight (>> alpha_V for stability)',
            'alpha_tw':  'Torsional damping cost weight (PB-QP compat, step 0 only)',
            'alpha_ff':  'Feedforward torque-matching cost weight',
            'Jratio':    'Generator-to-turbine inertia ratio J_g/J_t',
            'D_t':       'Turbine friction coefficient [pu]',
            'tau_ff':    'Feedforward filter time constant [s]',
            'K_td':      'Freq-domain torsional damping gain [pu] (APC supplementary)',
            'K_ff':      'Freq-domain feedforward gain [pu] (APC supplementary)',
            'gamma_tw':  'Max ratio torsional-to-base Hessian (adaptive clamp)',
            'K_shaft_dt':'Drivetrain shaft stiffness (predictive model) [pu/rad]',
            'D_shaft_dt':'Drivetrain shaft damping (predictive model) [pu]',
            'Tm0_dt':    'Rated per-unit aero torque at vw_nom (predictive model)',
            'vw_nom_dt': 'Nominal wind speed for drivetrain model [m/s]',
            'dt_mpc':    'MPC prediction step [s]',
            # Equilibrium state (set during init)
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
            'Cp_filt':  {'description': 'Dynamic Cp inflow filter', 'unit': '--', 'cpp_expr': 'x[5]'},
            'T_aero_filt': {'description': 'Slow T_aero filter',   'unit': 'pu', 'cpp_expr': 'x[4]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation — Multi-Horizon MPC Core
    # ------------------------------------------------------------------

    def _mpc_reconstruct_and_solve(self) -> str:
        """C++ code block: reconstruct DFIG state, linearise, propagate
        N-step horizon, build & solve condensed QP, return rotor voltage
        commands in the theta_s frame."""
        return r"""
            // ============================================================
            // MPC: Reconstruct DFIG electromagnetic state from inputs
            // ============================================================
            double sigma_LS = Ls * Lr - Lm * Lm;

            // Stator fluxes (already in theta_s frame from rotation above)
            double phi_sd_loc = phi_sd;
            double phi_sq_loc = phi_sq;

            // Rotor fluxes from currents + stator fluxes:
            //   φ_rd = (σ·i_rd + Lm·φ_sd) / Ls
            double phi_rd_loc = (sigma_LS * ird_m + Lm * phi_sd_loc) / Ls;
            double phi_rq_loc = (sigma_LS * iqr_m + Lm * phi_sq_loc) / Ls;

            // Generator momentum
            double omega_g = omega_r_m;
            double p_g_loc = j_inertia * omega_g;

            // Co-energy variables (currents = M⁻¹ x)
            double i_sd_m =  (Lr * phi_sd_loc - Lm * phi_rd_loc) / sigma_LS;
            double i_sq_m =  (Lr * phi_sq_loc - Lm * phi_rq_loc) / sigma_LS;

            // ============================================================
            // MPC: Equilibrium state (from power-flow initialization)
            // ============================================================
            double phi_sd_eq = phi_sd_ss;
            double phi_sq_eq = phi_sq_ss;
            double phi_rd_eq = phi_rd_ss;
            double phi_rq_eq = phi_rq_ss;
            double p_g_eq    = p_g_ss;

            // Error states z = x - x_eq (shifted Hamiltonian coordinates)
            double z_em[5];
            z_em[0] = phi_sd_loc - phi_sd_eq;
            z_em[1] = phi_sq_loc - phi_sq_eq;
            z_em[2] = phi_rd_loc - phi_rd_eq;
            z_em[3] = phi_rq_loc - phi_rq_eq;
            z_em[4] = p_g_loc    - p_g_eq;

            // ============================================================
            // MPC: Autonomous dynamics f(x_0) = (J₃ − R₃)∇H₃ + external
            //       (everything except V_rd, V_rq which we control)
            //       Using CURRENT virtual frequency ω_vs = x[0] for
            //       better accuracy than nominal ω_s.
            // ============================================================
            double omega_vs_cur = x[0];
            double omega_slip = omega_vs_cur - omega_g;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));

            // Stator voltages (in theta_s frame) are external inputs
            double Vsd_ext = vds_m;
            double Vsq_ext = vqs_m;

            // Simplified forms (J₃·∇H₃ reduces to ωs·φ products):
            //   JdH0 = ωs·φsq,  JdH1 = −ωs·φsd
            //   JdH2 = (ωs − np·ωg)·φrq,  JdH3 = −(ωs − np·ωg)·φrd
            //   JdH4 = np·Lm/σ·(φsq·φrd − φsd·φrq)
            double JdH0 =  omega_vs_cur * (Ls * i_sq_m + Lm * iqr_m);
            double JdH1 = -omega_vs_cur * (Ls * i_sd_m + Lm * ird_m);
            double JdH2 =  omega_vs_cur * (Lm * i_sq_m + Lr * iqr_m)
                           - np_poles * phi_rq_loc * omega_g;
            double JdH3 = -omega_vs_cur * (Lm * i_sd_m + Lr * ird_m)
                           + np_poles * phi_rd_loc * omega_g;
            double JdH4 =  np_poles * (phi_rq_loc * ird_m - phi_rd_loc * iqr_m);

            // Dissipation R₃·∇H₃
            double RdH0 = Rs * i_sd_m;
            double RdH1 = Rs * i_sq_m;
            double RdH2 = Rr * ird_m;
            double RdH3 = Rr * iqr_m;
            double RdH4 = f_damp * omega_g;

            // Autonomous dynamics (WITHOUT rotor voltage control)
            double f_auto[5];
            f_auto[0] = omega_b * (JdH0 - RdH0 + Vsd_ext);
            f_auto[1] = omega_b * (JdH1 - RdH1 + Vsq_ext);
            f_auto[2] = omega_b * (JdH2 - RdH2);
            f_auto[3] = omega_b * (JdH3 - RdH3);
            f_auto[4] = JdH4 - RdH4;

            // ============================================================
            // MPC: Linearise Σ₃ dynamics — Jacobian A = ∂f/∂x |_{x_0}
            //
            // The EM dynamics in flux coordinates simplify to:
            //   f₀ = ωb·(ωs·φsq − Rs·isd + Vsd)
            //   f₁ = ωb·(−ωs·φsd − Rs·isq + Vsq)
            //   f₂ = ωb·((ωs−np·ωg)·φrq + Rr·Lm/σ·φsd − Rr·Ls/σ·φrd)
            //   f₃ = ωb·(−(ωs−np·ωg)·φrd + Rr·Lm/σ·φsq − Rr·Ls/σ·φrq)
            //   f₄ = np·Lm/σ·(φsq·φrd − φsd·φrq) − f_damp·ωg
            //
            // ωs uses the LIVE virtual frequency x[0] (not nominal).
            // ============================================================
            double ws = omega_vs_cur;               // virtual sync speed
            double wr = omega_g;                     // rotor speed = pg/j
            double ws_np_wr = ws - np_poles * wr;    // combined rotational freq

            double A_mat[25];  // 5×5 flat, row-major

            // Row 0: ∂f₀/∂[φsd, φsq, φrd, φrq, pg]
            A_mat[0]  = -omega_b * Rs * Lr / sigma_LS;  // ∂f₀/∂φsd
            A_mat[1]  =  omega_b * ws;                   // ∂f₀/∂φsq
            A_mat[2]  =  omega_b * Rs * Lm / sigma_LS;  // ∂f₀/∂φrd
            A_mat[3]  =  0.0;                            // ∂f₀/∂φrq
            A_mat[4]  =  0.0;                            // ∂f₀/∂pg

            // Row 1: ∂f₁/∂[φsd, φsq, φrd, φrq, pg]
            A_mat[5]  = -omega_b * ws;                   // ∂f₁/∂φsd
            A_mat[6]  = -omega_b * Rs * Lr / sigma_LS;  // ∂f₁/∂φsq
            A_mat[7]  =  0.0;                            // ∂f₁/∂φrd
            A_mat[8]  =  omega_b * Rs * Lm / sigma_LS;  // ∂f₁/∂φrq
            A_mat[9]  =  0.0;                            // ∂f₁/∂pg

            // Row 2: ∂f₂/∂[φsd, φsq, φrd, φrq, pg]
            A_mat[10] =  omega_b * Rr * Lm / sigma_LS;                // ∂f₂/∂φsd
            A_mat[11] =  0.0;                                          // ∂f₂/∂φsq
            A_mat[12] = -omega_b * Rr * Ls / sigma_LS;                // ∂f₂/∂φrd
            A_mat[13] =  omega_b * ws_np_wr;                          // ∂f₂/∂φrq
            A_mat[14] = -omega_b * np_poles * phi_rq_loc / j_inertia; // ∂f₂/∂pg

            // Row 3: ∂f₃/∂[φsd, φsq, φrd, φrq, pg]
            A_mat[15] =  0.0;                                          // ∂f₃/∂φsd
            A_mat[16] =  omega_b * Rr * Lm / sigma_LS;                // ∂f₃/∂φsq
            A_mat[17] = -omega_b * ws_np_wr;                          // ∂f₃/∂φrd
            A_mat[18] = -omega_b * Rr * Ls / sigma_LS;                // ∂f₃/∂φrq
            A_mat[19] =  omega_b * np_poles * phi_rd_loc / j_inertia;  // ∂f₃/∂pg

            // Row 4: ∂f₄/∂[φsd, φsq, φrd, φrq, pg]
            A_mat[20] = -np_poles * Lm * phi_rq_loc / sigma_LS;       // ∂f₄/∂φsd
            A_mat[21] =  np_poles * Lm * phi_rd_loc / sigma_LS;       // ∂f₄/∂φsq
            A_mat[22] =  np_poles * Lm * phi_sq_loc / sigma_LS;       // ∂f₄/∂φrd
            A_mat[23] = -np_poles * Lm * phi_sd_loc / sigma_LS;       // ∂f₄/∂φrq
            A_mat[24] = -f_damp / j_inertia;                           // ∂f₄/∂pg

            // ============================================================
            // MPC: Discrete-time matrices
            //   Φ = I + dt · A   (5×5 state transition)
            //   Γ = dt · B_u     (5×2, only rows 2,3 nonzero)
            //   d = dt · f_auto  (5×1 affine disturbance per step)
            // ============================================================
            double dtw = dt_mpc * omega_b;
            double Phi[25];  // 5×5 flat
            for (int ii = 0; ii < 25; ii++) Phi[ii] = dt_mpc * A_mat[ii];
            for (int ii = 0; ii < 5; ii++) Phi[ii * 5 + ii] += 1.0;

            // Γ[2,0] = dtw, Γ[3,1] = dtw, rest zero
            // (implicit — used via Phi powers below)

            double d_aff[5];  // affine term per step
            for (int ii = 0; ii < 5; ii++) d_aff[ii] = dt_mpc * f_auto[ii];

            // ============================================================
            // MPC: Precompute Phi powers  Φ^k  for k = 0, 1, ..., N-1
            //       Stored flat: Phi_pow[k*25 + r*5 + c]
            // ============================================================
            int Nh = (int)fmin(fmax(N_horizon, 1.0), 5.0);

            double Phi_pow[125];  // max 5 × 25
            // Φ⁰ = I
            for (int ii = 0; ii < 25; ii++) Phi_pow[ii] = 0.0;
            for (int ii = 0; ii < 5; ii++) Phi_pow[ii * 5 + ii] = 1.0;
            // Φ¹ = Phi
            if (Nh > 1)
                for (int ii = 0; ii < 25; ii++) Phi_pow[25 + ii] = Phi[ii];
            // Φ^k = Φ · Φ^{k−1}
            for (int kk = 2; kk < Nh; kk++) {
                for (int rr = 0; rr < 5; rr++) {
                    for (int cc = 0; cc < 5; cc++) {
                        double s = 0.0;
                        for (int mm = 0; mm < 5; mm++)
                            s += Phi[rr * 5 + mm] * Phi_pow[(kk-1) * 25 + mm * 5 + cc];
                        Phi_pow[kk * 25 + rr * 5 + cc] = s;
                    }
                }
            }

            // ============================================================
            // MPC: Free-response error state propagation
            //
            //   δ_free_{k+1} = Φ · δ_free_k + d,   δ_free_0 = 0
            //   c_k = z_em + δ_free_k   (error state at step k)
            //
            //   Lyapunov cost uses only rotor components c_k[2], c_k[3].
            // ============================================================
            double delta_free[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            double c_arr[25];  // max 5 steps × 5 states

            for (int kk = 0; kk < Nh; kk++) {
                double tmp[5];
                for (int rr = 0; rr < 5; rr++) {
                    tmp[rr] = d_aff[rr];
                    for (int ss = 0; ss < 5; ss++)
                        tmp[rr] += Phi[rr * 5 + ss] * delta_free[ss];
                }
                for (int rr = 0; rr < 5; rr++) {
                    delta_free[rr] = tmp[rr];
                    c_arr[kk * 5 + rr] = z_em[rr] + delta_free[rr];
                }
            }

            // ============================================================
            // MPC: Build condensed-form QP
            //
            // Decision variables: U = [Vrd_0, Vrq_0, ..., Vrd_{N-1}, Vrq_{N-1}]
            //
            // Predicted error at step k:
            //   z_k = c_k + Ψ_{u,k} · U
            //
            // Block (k,j) of Ψ_u is Φ^{k−j−1} Γ for j ≤ k−1, else 0.
            // Rows 2,3 of Φ^p Γ:
            //   Ψ_r2(p, 0) = Φ^p[2,2]·dtw,  Ψ_r2(p, 1) = Φ^p[2,3]·dtw
            //   Ψ_r3(p, 0) = Φ^p[3,2]·dtw,  Ψ_r3(p, 1) = Φ^p[3,3]·dtw
            //
            // Cost: J = Σ_k α_k · q · (c_k[2] + Ψ_r2 · U)² + (c_k[3] + Ψ_r3 · U)²
            //           + α_u · ||U − U_eq||²
            //
            // where q = Ls/σ (block-diagonal rotor-only Lyapunov weight).
            //
            // QP: H · U = rhs
            //   H = Σ_k α_k · q · (Psi_r2^T Psi_r2 + Psi_r3^T Psi_r3) + α_u I
            //   rhs = −Σ_k α_k · q · (c_k[2] · Psi_r2 + c_k[3] · Psi_r3) + α_u U_eq
            // ============================================================
            int n_u = 2 * Nh;
            double q_rotor = Ls / sigma_LS;

            double H_qp[100] = {0.0};  // max 10×10
            double g_qp[10]  = {0.0};  // gradient (Lyapunov part)

            for (int kk = 0; kk < Nh; kk++) {
                // Step weight: terminal step gets alpha_Vf, others get alpha_V
                double alpha_k = (kk == Nh - 1) ? alpha_Vf : alpha_V;
                double aq = alpha_k * q_rotor;

                // Free-response rotor errors at step kk+1
                double cr2 = c_arr[kk * 5 + 2];
                double cr3 = c_arr[kk * 5 + 3];

                // For each control index a that affects step kk+1 (a ≤ kk)
                for (int a = 0; a <= kk; a++) {
                    int pow_a = kk - a;  // Φ power for control a

                    // Sensitivity of rotor flux rows to control a
                    double sa20 = Phi_pow[pow_a * 25 + 2 * 5 + 2] * dtw;
                    double sa21 = Phi_pow[pow_a * 25 + 2 * 5 + 3] * dtw;
                    double sa30 = Phi_pow[pow_a * 25 + 3 * 5 + 2] * dtw;
                    double sa31 = Phi_pow[pow_a * 25 + 3 * 5 + 3] * dtw;

                    // Gradient contribution: g[2a+m] += aq * (cr2·sa2m + cr3·sa3m)
                    g_qp[2 * a]     += aq * (cr2 * sa20 + cr3 * sa30);
                    g_qp[2 * a + 1] += aq * (cr2 * sa21 + cr3 * sa31);

                    // Hessian contribution (lower-triangle + mirror)
                    for (int b = 0; b <= a; b++) {
                        int pow_b = kk - b;
                        double sb20 = Phi_pow[pow_b * 25 + 2 * 5 + 2] * dtw;
                        double sb21 = Phi_pow[pow_b * 25 + 2 * 5 + 3] * dtw;
                        double sb30 = Phi_pow[pow_b * 25 + 3 * 5 + 2] * dtw;
                        double sb31 = Phi_pow[pow_b * 25 + 3 * 5 + 3] * dtw;

                        double h00 = aq * (sa20 * sb20 + sa30 * sb30);
                        double h01 = aq * (sa20 * sb21 + sa30 * sb31);
                        double h10 = aq * (sa21 * sb20 + sa31 * sb30);
                        double h11 = aq * (sa21 * sb21 + sa31 * sb31);

                        H_qp[(2*a)   * n_u + 2*b]     += h00;
                        H_qp[(2*a)   * n_u + 2*b + 1] += h01;
                        H_qp[(2*a+1) * n_u + 2*b]     += h10;
                        H_qp[(2*a+1) * n_u + 2*b + 1] += h11;

                        if (a != b) {
                            H_qp[(2*b)   * n_u + 2*a]     += h00;
                            H_qp[(2*b)   * n_u + 2*a + 1] += h10;
                            H_qp[(2*b+1) * n_u + 2*a]     += h01;
                            H_qp[(2*b+1) * n_u + 2*a + 1] += h11;
                        }
                    }
                }
            }

            // Add regularization (α_u · I) to Hessian diagonal
            for (int jj = 0; jj < n_u; jj++)
                H_qp[jj * n_u + jj] += alpha_u;

            // Build RHS: rhs = −g_lyap + α_u · U_eq
            double rhs_qp[10];
            for (int jj = 0; jj < Nh; jj++) {
                rhs_qp[2 * jj]     = -g_qp[2 * jj]     + alpha_u * Vrd_ss;
                rhs_qp[2 * jj + 1] = -g_qp[2 * jj + 1] + alpha_u * Vrq_ss;
            }

            // ============================================================
            // MPC: Predictive torsional damping
            //
            // This is the KEY advantage of MPC over PB-QP.  Using the
            // predicted wind speed vw_pred, we anticipate the future
            // aerodynamic torque T_aero_pred and add a multi-step
            // torsional cost to the QP:
            //
            //   J_tw = Σ_k α_tw · (ΔTe_k + r_tw_k)²
            //
            // where:
            //   ΔTe_k ≈ a_rd · Vrd_k + a_rq · Vrq_k   (linearised)
            //     a_rd = −np · dtw · iqr
            //     a_rq =  np · dtw · ird
            //   r_tw_k = −(ω_t − ω_g) − Jratio · (T_aero_pred − T_aero_filt)
            //
            // The twist_rate term (ω_t − ω_g) damps existing oscillations
            // (same as PB-QP).  The T_aero prediction term PRE-EMPTIVELY
            // adjusts Te to counteract the coming wind gust BEFORE the
            // shaft feels it.
            //
            // T_aero_pred uses the Heier Cp model with Pitt-Peters
            // dynamic inflow filter (state x[5] = Cp_filt):
            //   T_aero_pred = Tm0 · vw_pu³ · Cp_filt / ωt
            // This matches the drivetrain's actual T_aero computation.
            //
            // Hessian:  H_tw += α_tw · [a_rd², a_rd·a_rq; a_rd·a_rq, a_rq²]
            //                   (same block added to each step's diagonal)
            // RHS:     rhs_tw_k += α_tw · [a_rd; a_rq] · r_tw_k
            //
            // Adaptive clamp (from PB-QP): cap α_tw_eff so torsional
            // Hessian ≤ γ_tw · base Lyapunov Hessian.
            // ============================================================
            if (alpha_tw > 1e-12) {
                // Torque sensitivity to rotor voltage (linearised)
                double a_rd = -np_poles * dtw * iqr_m;
                double a_rq =  np_poles * dtw * ird_m;
                double a_sq = a_rd * a_rd + a_rq * a_rq;

                // Adaptive clamping (same logic as PB-QP)
                double alpha_tw_eff = alpha_tw;
                if (a_sq > 1e-12) {
                    double H_base_diag = alpha_V * dtw * dtw * q_rotor + alpha_u;
                    double alpha_damp_max = gamma_tw * H_base_diag / a_sq;
                    if (alpha_tw > alpha_damp_max)
                        alpha_tw_eff = alpha_damp_max;
                }

                // Current drivetrain state
                double omega_t_m    = inputs[11];
                double theta_tw_m   = inputs[12];
                double T_aero_now   = inputs[13];
                double T_shaft_now  = inputs[14];
                double twist_rate_m = omega_t_m - omega_g;

                // Predicted wind → predicted T_aero (Heier Cp + Pitt-Peters)
                double vw_pred_m = inputs[15];
                double vw_pu_pred = vw_pred_m / fmax(vw_nom_dt, 1.0);
                double omega_t_safe = fmax(fabs(omega_t_m), 0.01);
                double Cp_filt_val = x[5];  // dynamic inflow filter state
                double T_aero_pred = Tm0_dt * vw_pu_pred * vw_pu_pred * vw_pu_pred
                                   * Cp_filt_val / omega_t_safe;

                // Feedforward deficit: predicted T_aero change
                double T_aero_filt_val = x[4];  // slow-reference filter state
                double ff_deficit = -Jratio * (T_aero_pred - T_aero_filt_val);
                if (fabs(ff_deficit) < 0.02) ff_deficit = 0.0;

                // Add torsional cost to each step's H and rhs
                for (int kk = 0; kk < Nh; kk++) {
                    // Target: damp twist + anticipate T_aero change
                    double r_tw_k = -twist_rate_m + ff_deficit;

                    // Hessian: α_tw · A·A^T on the (kk,kk) diagonal block
                    H_qp[(2*kk)   * n_u + 2*kk]     += alpha_tw_eff * a_rd * a_rd;
                    H_qp[(2*kk)   * n_u + 2*kk + 1] += alpha_tw_eff * a_rd * a_rq;
                    H_qp[(2*kk+1) * n_u + 2*kk]     += alpha_tw_eff * a_rd * a_rq;
                    H_qp[(2*kk+1) * n_u + 2*kk + 1] += alpha_tw_eff * a_rq * a_rq;

                    // RHS: α_tw · A · r_tw_k
                    rhs_qp[2 * kk]     += alpha_tw_eff * a_rd * r_tw_k;
                    rhs_qp[2 * kk + 1] += alpha_tw_eff * a_rq * r_tw_k;
                }
            }

            // ============================================================
            // MPC: Solve dense QP via Cholesky factorisation
            //       H = L L^T,  L y = rhs,  L^T U = y
            //       Then project onto box constraints.
            // ============================================================
            double L_chol[100] = {0.0};  // max 10×10

            // Cholesky decomposition (column-by-column)
            for (int ii = 0; ii < n_u; ii++) {
                for (int jj = 0; jj <= ii; jj++) {
                    double s = 0.0;
                    for (int kk = 0; kk < jj; kk++)
                        s += L_chol[ii * n_u + kk] * L_chol[jj * n_u + kk];
                    if (ii == jj) {
                        double diag = H_qp[ii * n_u + ii] - s;
                        L_chol[ii * n_u + jj] = (diag > 1e-12) ? sqrt(diag) : 1e-6;
                    } else {
                        L_chol[ii * n_u + jj] = (H_qp[ii * n_u + jj] - s)
                                                / L_chol[jj * n_u + jj];
                    }
                }
            }

            // Forward substitution: L y = rhs
            double y_sol[10] = {0.0};
            for (int ii = 0; ii < n_u; ii++) {
                double s = 0.0;
                for (int jj = 0; jj < ii; jj++)
                    s += L_chol[ii * n_u + jj] * y_sol[jj];
                y_sol[ii] = (rhs_qp[ii] - s) / L_chol[ii * n_u + ii];
            }

            // Back substitution: L^T U = y
            double U_sol[10] = {0.0};
            for (int ii = n_u - 1; ii >= 0; ii--) {
                double s = 0.0;
                for (int jj = ii + 1; jj < n_u; jj++)
                    s += L_chol[jj * n_u + ii] * U_sol[jj];
                U_sol[ii] = (y_sol[ii] - s) / L_chol[ii * n_u + ii];
            }

            // ============================================================
            // MPC: Extract first control and apply constraints
            // ============================================================
            double Vrd_opt = U_sol[0];
            double Vrq_opt = U_sol[1];

            // ============================================================
            // Passivity-based damping on rotor flux derivative
            //   u_final = (u_opt − R_inj · f₂) / (1 + R_inj · ωb)
            //   Vanishes at equilibrium (f₂ → 0).
            // ============================================================
            double R_inj = 0.05;
            double damp_denom = 1.0 + R_inj * omega_b;
            Vrd_opt = (Vrd_opt - R_inj * f_auto[2]) / damp_denom;
            Vrq_opt = (Vrq_opt - R_inj * f_auto[3]) / damp_denom;

            // ============================================================
            // Constraint projection
            // ============================================================

            // Rotor voltage saturation
            Vrd_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrd_opt));
            Vrq_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrq_opt));

            // Current limit: predict currents after applying this voltage
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
""" + self._mpc_reconstruct_and_solve() + """
            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        """

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

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG) + integral
            //         (identical to PB-QP — outer loops unchanged)
            // ==========================================================
            double omega_vs = x[0];
            double xi_P = x[3];

            // Torsional damping: supplementary power signal
            double omega_t_apc = inputs[11];
            double omega_g_apc = inputs[10];
            double twist_rate  = omega_t_apc - omega_g_apc;
            double P_td = -K_td * twist_rate;

            // Feedforward: supplementary power signal
            double T_aero_apc  = inputs[13];
            double ff_deficit  = -Jratio * (T_aero_apc - x[4]);
            double P_ff = K_ff * ff_deficit;

            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)
                       + xi_P + P_td + P_ff) / J;
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
                double vw_step = inputs[15];  // predicted wind speed
                double vw_pu_step = vw_step / fmax(vw_nom_dt, 1.0);
                double omega_t_step = fmax(fabs(inputs[11]), 0.01);
                double lambda_step = vw_pu_step / omega_t_step;
                double beta_step = 0.0;
                double li_step = 1.0 / (1.0 / (lambda_step + 0.08 * beta_step)
                                        - 0.035 / (beta_step * beta_step + 1.0));
                double Cp_raw_step = 0.5176 * (116.0 * li_step - 0.4 * beta_step - 5.0)
                                   * exp(-21.0 * li_step) + 0.0068 * lambda_step;
                // Normalise by rated Cp (same as drivetrain)
                double li0 = 1.0 / (1.0 - 0.035);
                double Cp0 = 0.5176 * (116.0 * li0 - 5.0) * exp(-21.0 * li0) + 0.0068;
                double Cp_emp_step = fmax(0.0, Cp_raw_step / fmax(Cp0, 1e-6));
                if (Cp_emp_step > 2.0) Cp_emp_step = 2.0;
                double tau_i_step = 1.0 / fmax(vw_pu_step, 0.1);
                dxdt[5] = (Cp_emp_step - x[5]) / fmax(tau_i_step, 0.01);
            }
""" + self._mpc_reconstruct_and_solve() + r"""
            // ==========================================================
            // STEP 4: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];
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
            name='DFIG_RSC_MPC_PHS',
            states=states, inputs=inputs,
            params={'J': J_sym, 'Dp': Dp_sym, 'ki_P': ki_P_sym},
            J=J_mat, R=R_mat, g=g_mat, H=H_expr,
            description=(
                'Real MPC RSC (Σ₅_MPC): VSG + RPC + freq integral, 4 PHS states. '
                'PI cascades replaced by N-step Lyapunov-MPC with Cholesky QP solver.'
            ),
        )

    # ------------------------------------------------------------------
    # Initialization (same algorithm as PB-QP)
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params

        # Default values (backward compatible)
        p.setdefault('alpha_tw', 0.0)
        p.setdefault('gamma_tw', 1.0)
        p.setdefault('alpha_ff', 0.0)
        p.setdefault('tau_ff', 2.0)
        p.setdefault('K_td', 0.0)
        p.setdefault('K_ff', 0.0)
        p.setdefault('N_horizon', 3)
        p.setdefault('alpha_Vf', 50.0)

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

        # Store as compiled constants for the MPC prediction model
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
        f_damp_gen = float(targets.get('f_damp_gen', 0.0))
        D_t_param = float(self.params.get('D_t', 0.05))
        self.params['Tm0_dt'] = float(abs(Pe_ss) + f_damp_gen + D_t_param)

        return self._init_states({
            'omega_vs': omega_s_ss,
            'theta_s':  theta_s_ss,
            'phi_Qs':   0.0,
            'xi_P':     0.0,
            'T_aero_filt': 0.0,
            'Cp_filt':  1.0,
        })
