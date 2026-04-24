"""
DFIG_DRIVETRAIN_PHS — Wind Turbine Rotor + Drive Train (Σ₁ + Σ₂)
=================================================================

Composed mechanical subsystem: wind turbine rotor inertia (Σ₁), dynamic
inflow model, and two-mass drive train with gearbox (Σ₂).

States (3 energy variables):
    x = [p_t, Cp_eff, theta_tw]

    p_t       — turbine angular momentum H_t·ω_t  [pu·s]
    Cp_eff    — effective power coefficient (dynamic inflow state) [-]
    theta_tw  — torsional shaft twist angle  [rad]

Hamiltonian:
    H = p_t²/(2·H_t) + k_cp·Cp_eff²/2 + K_s·theta_tw²/2

Gradient:
    ∇H = [ω_t, k_cp·Cp_eff, K_s·theta_tw]
       = [ω_t, k_cp·Cp_eff, T_spring]

Dynamics:
    ṗ_t      = T_aero − T_shaft − D_t·ω_t
    Ċp_eff   = (Cp_emp − Cp_eff) / τ_i
    θ̇_tw     = ω_t − ω_g

where:
    T_shaft  = K_s·theta_tw + D_s·(ω_t − ω_g)
    T_aero   = Cp_eff · ½ρπR²V_w³ / ω_t
    Cp_emp   = standard empirical formula (Heier polynomial-exponential)
    τ_i      = R / V_w  (Pitt–Peters dynamic inflow time constant)

Ports:
    in:  [omega_g, V_w, beta]
    out: [T_shaft, omega_t, T_aero, theta_tw]

The gear ratio is absorbed by per-unit normalisation (N_pu = 1).
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigDrivetrainPHS(PowerComponent):
    """
    Wind turbine rotor + drive train (Σ₁+Σ₂) in Port-Hamiltonian form.

    Three-state model: turbine momentum, dynamic inflow, shaft twist.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega_g', 'effort', 'pu'),   # [0] generator rotor speed
                ('V_w',     'effort', 'm/s'),  # [1] wind speed
                ('beta',    'effort', 'deg'),   # [2] pitch angle
            ],
            'out': [
                ('T_gen',    'flow', 'pu'),    # [0] shaft torque → DFIG.Tm
                ('omega_t',  'flow', 'pu'),    # [1] turbine speed
                ('T_aero',   'flow', 'pu'),    # [2] aerodynamic torque
                ('theta_out','flow', 'rad'),   # [3] shaft twist (monitoring)
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['p_t', 'Cp_eff', 'theta_tw']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H_t':       'Turbine inertia constant (2H) [pu·s]',
            'K_shaft':   'Torsional shaft stiffness [pu/rad]',
            'D_shaft':   'Mutual shaft damping [pu]',
            'D_t':       'Turbine aerodynamic drag [pu]',
            'k_cp':      'Dynamic inflow filter gain [pu]',
            'vw_nom':    'Nominal wind speed for pu normalisation [m/s]',
            'gear_ratio':'Physical gearbox ratio (absorbed in pu)',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'omega_t':    {'description': 'Turbine rotor speed',       'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'theta_tw':   {'description': 'Shaft twist angle',         'unit': 'rad', 'cpp_expr': 'x[2]'},
            'Cp_eff':     {'description': 'Effective Cp (dynamic)',    'unit': '-',   'cpp_expr': 'x[1]'},
            'T_shaft':    {'description': 'Shaft torque',              'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'T_aero':     {'description': 'Aerodynamic torque',        'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'twist_rate': {'description': 'Shaft twist rate (omega_t - omega_g)',
                           'unit': 'pu',  'cpp_expr': 'outputs[1] - inputs[0]'},
        }

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        Tm0 = self.params.get('Tm0', 0.5)
        return f"double Tm0 = {Tm0};\n" + r"""
            double omega_t = x[0] / fmax(H_t, 1e-6);
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Per-unit wind speed ----
            double vw_pu = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Aerodynamic torque from dynamic Cp_eff (per-unit) ----
            // P_aero = Tm0 * vw_pu³ * Cp_eff_val,  T_aero = P_aero / omega_t
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Shaft torque: spring + damping ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            outputs[0] = T_shaft;
            outputs[1] = omega_t;
            outputs[2] = T_aero;
            outputs[3] = theta_tw;
        """

    def get_cpp_step_code(self) -> str:
        Tm0 = self.params.get('Tm0', 0.5)
        return f"double Tm0 = {Tm0};\n" + r"""
            // ---- States ----
            double omega_t  = x[0] / fmax(H_t, 1e-6);
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Inputs ----
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double vw_pu   = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Shaft torque ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // ---- Aerodynamic torque (per-unit, uses dynamic Cp_eff) ----
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Empirical Cp (quasi-static target, per-unit lambda) ----
            double lambda_pu = vw_pu / omega_t_safe;
            double lambda_i = 1.0 / (1.0 / (lambda_pu + 0.08 * beta) - 0.035 / (beta * beta + 1.0));
            double Cp_raw = 0.5176 * (116.0 * lambda_i - 0.4 * beta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * lambda_pu;

            // Base Cp at rated (vw=1, omega=1, beta=0)
            double lambda_i_0 = 1.0 / (1.0 - 0.035);
            double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
            double Cp_emp = fmax(0.0, Cp_raw / fmax(Cp_0, 1e-6));
            if (Cp_emp > 2.0) Cp_emp = 2.0;

            // ---- Dynamic inflow time constant ----
            double tau_i = 1.0 / fmax(vw_pu, 0.1);

            // ---- PHS dynamics ----
            dxdt[0] = T_aero - T_shaft - D_t * omega_t;
            dxdt[1] = (Cp_emp - Cp_eff_val) / fmax(tau_i, 0.01);
            dxdt[2] = omega_t - omega_g;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        p_t, Cp_eff, theta_tw = sp.symbols('p_t C_p^{eff} theta_tw')
        states = [p_t, Cp_eff, theta_tw]

        omega_g = sp.Symbol('omega_g')
        V_w = sp.Symbol('V_w', positive=True)
        beta = sp.Symbol('beta')
        T_aero = sp.Symbol('T_aero')
        inputs = [T_aero, omega_g, V_w]

        Ht = sp.Symbol('H_t', positive=True)
        Ks = sp.Symbol('K_s', positive=True)
        Ds = sp.Symbol('D_s', nonnegative=True)
        Dt = sp.Symbol('D_t', nonnegative=True)
        kcp = sp.Symbol('k_cp', positive=True)
        tau = sp.Symbol('tau_i', positive=True)

        params = {
            'H_t': Ht, 'K_shaft': Ks, 'D_shaft': Ds,
            'D_t': Dt, 'k_cp': kcp, 'tau_i': tau,
        }

        # Hamiltonian
        H_expr = p_t**2 / (2 * Ht) + kcp * Cp_eff**2 / 2 + Ks * theta_tw**2 / 2

        # J: shaft coupling between p_t and theta_tw
        J = sp.zeros(3, 3)
        J[0, 2] = -1;  J[2, 0] = 1  # spring coupling

        # R: turbine drag + dynamic inflow dissipation
        R = sp.zeros(3, 3)
        R[0, 0] = Dt
        R[1, 1] = 1 / (kcp * tau)

        # g: input coupling
        g = sp.zeros(3, 3)
        g[0, 0] = 1   # T_aero → dp_t
        g[2, 1] = -1  # -omega_g → dtheta_tw (flow from generator)
        g[1, 2] = 1 / tau  # V_w-dependent Cp input (simplified)

        return SymbolicPHS(
            name='DFIG_DRIVETRAIN_PHS',
            states=states, inputs=inputs, params=params,
            J=J, R=R, g=g, H=H_expr,
            description='Wind turbine rotor + drive train (Σ₁+Σ₂) in PH form.',
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        Te_gen = float(targets.get('Tm', targets.get('Pe', targets.get('Pref', 0.5))))
        K_shaft = float(self.params['K_shaft'])
        H_t = float(self.params['H_t'])
        D_t = float(self.params.get('D_t', 0.0))
        f_damp_gen = float(targets.get('f_damp_gen', 0.0))

        # At equilibrium (omega_t = omega_g = 1.0):
        #   DFIG:  T_shaft = Te + f_damp_gen * omega_g
        #   DT:    T_aero  = T_shaft + D_t * omega_t
        T_shaft_eq = Te_gen + f_damp_gen
        theta_tw_0 = T_shaft_eq / max(K_shaft, 1e-6)
        p_t_0 = H_t  # H_t * omega_t_0, omega_t_0 = 1.0

        # Store Tm0 for C++ code generation — rated per-unit torque.
        # Tm0 = T_aero at rated (vw_pu=1, omega_t=1, Cp_eff=1).
        self.params['Tm0'] = T_shaft_eq + D_t

        # At rated conditions (vw_pu=1, omega_t=1, beta=0), Cp_eff should
        # equal the normalised Cp_factor ≈ 1.0 so that T_aero = Tm0·1³·1 = Tm0.
        Cp_eff_0 = 1.0

        return self._init_states({
            'p_t': p_t_0,
            'Cp_eff': Cp_eff_0,
            'theta_tw': theta_tw_0,
        })
