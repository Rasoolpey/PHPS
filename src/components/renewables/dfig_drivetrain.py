"""
DFIG_DRIVETRAIN — Two-Mass Drive-Train Model (Port-Hamiltonian)
===============================================================

Models the flexible shaft between the wind turbine rotor (low-speed side)
and the DFIG generator rotor (high-speed side) including the gearbox.

In per-unit, both speeds are normalised to synchronous speed (1.0 pu);
the physical gear ratio is therefore implicit in the per-unit base
(omega_t_pu = omega_g_pu = 1.0 at rated) but stored as ``gear_ratio``
for post-processing physical shaft torques (fatigue analysis).

Port-Hamiltonian formulation
----------------------------
The turbine-side mass + torsional spring form a two-state PH subsystem.
Generator-side dynamics are handled by the DFIG's own p_mech state.

Hamiltonian (energy stored in turbine inertia + shaft spring):

    H_dt = p_t²/(2·H_t) + K_shaft/2 · theta_tw²

Energy variables (state vector):

    x = [p_t, theta_tw]^T

    p_t      = H_t · omega_t     (turbine angular momentum)
    theta_tw = shaft twist angle  (positive = turbine leads generator)

Gradient  ∇H_dt = [omega_t, K_shaft·theta_tw]^T
                = [p_t/H_t,  K_shaft·x[1]  ]^T

PCH structure  J_dt = [[0, -1], [1, 0]]  (skew-symmetric, unit gyration)
PCH dissipation R_dt = [[D_t + D_shaft, -D_shaft], [-D_shaft, 0]]

The generator speed omega_g enters as an exogenous input (from DFIG.omega),
breaking the closed PH structure intentionally — the DFIG provides the
complementary PH subsystem.

Differential equations
----------------------

    dp_t/dt      = T_aero - T_shaft - D_t·(omega_t - 1)
    dtheta_tw/dt = omega_t - omega_g

    omega_t = p_t / H_t
    T_shaft = K_shaft·theta_tw + D_shaft·(omega_t - omega_g)

The shaft torque T_shaft is the mechanical torque transmitted to the
generator rotor (T_gen = T_shaft in pu; the low-speed physical torque
is T_lss = T_shaft · gear_ratio).

States (2):
    [0] p_t       — turbine angular momentum  [pu·s]
    [1] theta_tw  — shaft twist angle          [rad (electrical)]

Inputs (2):
    [0] T_aero  — aerodynamic torque from WIND_AERO.Tm  [pu]
    [1] omega_g — DFIG generator rotor speed (DFIG.omega) [pu]

Outputs (4):
    [0] T_gen    — shaft torque applied to generator rotor (→ DFIG.Tm) [pu]
    [1] omega_t  — turbine rotor speed (→ WIND_AERO.omega)             [pu]
    [2] theta_out — shaft twist angle (monitoring)                      [rad]
    [3] T_lss    — low-speed-shaft torque = T_shaft·gear_ratio (fatigue)[pu]

Parameters
----------
    H_t        — turbine-side inertia constant (= 2·H_turbine in pu·s)
    K_shaft    — torsional shaft stiffness [pu/rad]
    D_shaft    — mutual shaft damping coefficient [pu]
    D_t        — turbine self-damping (aerodynamic drag) [pu]
    gear_ratio — physical gearbox ratio (omega_g_phys / omega_t_phys);
                 stored for low-speed-shaft torque calculation only
"""

import math
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigDrivetrain(PowerComponent):
    """
    Two-mass turbine drive-train for the DFIG wind turbine chain.

    Sits between WIND_AERO (aerodynamic torque source) and the DFIG machine
    (generator rotor), adding turbine inertia and shaft torsional dynamics.

    Wiring:
        WIND_AERO.Tm   → DFIG_DRIVETRAIN.T_aero
        DFIG.omega     → DFIG_DRIVETRAIN.omega_g
        DFIG_DRIVETRAIN.T_gen   → DFIG.Tm
        DFIG_DRIVETRAIN.omega_t → WIND_AERO.omega
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('T_aero',  'effort', 'pu'),   # [0] aerodynamic torque
                ('omega_g', 'effort', 'pu'),   # [1] generator rotor speed
            ],
            'out': [
                ('T_gen',   'flow', 'pu'),     # [0] torque to DFIG rotor
                ('omega_t', 'flow', 'pu'),     # [1] turbine speed to WIND_AERO
                ('theta_out','flow', 'rad'),   # [2] shaft twist (monitoring)
                ('T_lss',   'flow', 'pu'),     # [3] low-speed shaft torque
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['p_t', 'theta_tw']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H_t':       'Turbine-side inertia constant (= 2H_turbine) [pu·s]',
            'K_shaft':   'Torsional shaft stiffness [pu/rad]',
            'D_shaft':   'Mutual shaft damping coefficient [pu]',
            'D_t':       'Turbine self-damping (aerodynamic drag) [pu]',
            'gear_ratio':'Physical gearbox speed ratio (generator/turbine)',
            # NOTE: 'dfig' is intentionally excluded — it is a string component
            # reference used only by get_associated_generator() on the Python side.
            # Including it here would cause the compiler to emit `const double dfig
            # = DFIG_N;` which is invalid C++.
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'omega_t':    {'description': 'Turbine rotor speed',                  'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'theta_tw':   {'description': 'Shaft twist angle',                    'unit': 'rad', 'cpp_expr': 'x[1]'},
            'T_shaft':    {'description': 'Shaft torque delivered to generator',  'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'T_lss':      {'description': 'Low-speed shaft torque',               'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'T_aero_in':  {'description': 'Aerodynamic torque input to shaft',    'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'T_net':      {'description': 'Net shaft torque (T_aero - T_shaft) — drives turbine speed change',
                           'unit': 'pu',  'cpp_expr': '(inputs[0] - outputs[0])'},
            'delta_omega':{'description': 'Speed difference (omega_t - omega_g)', 'unit': 'pu',
                           'cpp_expr': '(outputs[1] - inputs[1])'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the DFIG generator that this drivetrain is connected to.
        Requires 'dfig' parameter in JSON pointing to the DFIG machine name."""
        return self.params.get('dfig', None)

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        """
        Equilibrium initialisation.

        At steady state:
            omega_t = omega_g = 1.0 pu
            dtheta_tw/dt = 0  →  theta_tw constant
            dp_t/dt = 0       →  T_aero = T_shaft = K_shaft·theta_tw

        So: theta_tw_0 = T_aero_0 / K_shaft  (from power/torque at omega=1)
            p_t_0      = H_t · 1.0
        """
        # T_aero_0 ≈ Pe at omega_m=1 (torque = power when omega=1 pu)
        T_aero_0 = float(targets.get('Tm', targets.get('Pe', targets.get('Pref', 0.5))))
        K_shaft = float(self.params['K_shaft'])
        H_t     = float(self.params['H_t'])

        theta_tw_0 = T_aero_0 / max(K_shaft, 1e-6)
        p_t_0      = H_t  # H_t * omega_t_0, omega_t_0 = 1.0

        return self._init_states({'p_t': p_t_0, 'theta_tw': theta_tw_0})

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double omega_t = x[0] / fmax(H_t, 0.001);
            double T_shaft = K_shaft * x[1] + D_shaft * (omega_t - inputs[1]);
            outputs[0] = T_shaft;             // T_gen  → DFIG.Tm
            outputs[1] = omega_t;             // omega_t → WIND_AERO.omega
            outputs[2] = x[1];               // theta_tw (monitoring)
            outputs[3] = T_shaft * gear_ratio; // T_lss (low-speed shaft torque)
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // States
            double omega_t = x[0] / fmax(H_t, 0.001);  // p_t / H_t
            double theta_tw = x[1];

            // Shaft torque (spring + mutual damping)
            double omega_g  = inputs[1];
            double T_shaft  = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // Turbine equation: dp_t/dt = T_aero - T_shaft - D_t*(omega_t - 1)
            dxdt[0] = inputs[0] - T_shaft - D_t * (omega_t - 1.0);

            // Twist angle: dtheta/dt = omega_t - omega_g
            dxdt[1] = omega_t - omega_g;
        """
