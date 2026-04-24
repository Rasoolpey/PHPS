"""
IEEEG1 — Port-Hamiltonian Formulation.

IEEE Type G1 Steam Turbine Governor with explicit (J, R, H) structure.

States: x = [x1, x2, x3]
  x1 – governor/servo output          (first-order lag at T1)
  x2 – HP turbine / steam-chest       (first-order lag at T3)
  x3 – reheater / IP+LP turbine       (first-order lag at T7)

Three-stage cascade matching the standard IEEEG1 turbine path:
  gov(T1) → HP(T3) → reheater(T7)

Mechanical torque output (two-fraction split):
  Tm = K5·x2 + K7·x3

where K5 is the HP fraction and K7 is the reheated fraction (K5+K7 ≈ 1).
For the IEEE 14-bus default: K5=0.3, K7=0.7, T7=8.72 s.

Storage function (identity metric):
    H = ½||x||²

Port inputs: u = [omega, Pref, u_agc]
Output:      y = [Tm] = K5·x2 + K7·x3

References
----------
- IEEE Std 421.5-2016 / WECC IEEEG1 model
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


class Ieeeg1PHS(PowerComponent):
    """
    IEEE Type G1 Steam Turbine Governor in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow',   'pu'),
                ('Pref',  'signal', 'pu'),
                ('u_agc', 'signal', 'pu'),
            ],
            'out': [('Tm', 'effort', 'pu')]
        }

    @property
    def required_ports(self):
        """u_agc is optional; defaults to 0.0 when no AGC is wired."""
        return ['omega', 'Pref']

    @property
    def state_schema(self) -> List[str]:
        return ['x1', 'x2', 'x3']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'K': 'Gain',
            'T1': 'Time Constant 1 (governor servo)',
            'T2': 'Time Constant 2 (unused)',
            'T3': 'Time Constant 3 (HP steam chest)',
            'UO': 'Max opening rate',
            'UC': 'Max closing rate',
            'PMAX': 'Max Power',
            'PMIN': 'Min Power',
            'T4': '', 'K1': '', 'K2': '', 'T5': '', 'K3': '', 'K4': '',
            'T6': '', 'K5': 'HP turbine fraction', 'K6': '',
            'T7': 'Reheater time constant (s)', 'K7': 'Reheated fraction', 'K8': '',
            'PMAX': 'Max power output (pu)', 'PMIN': 'Min power output (pu)',
            'UO':   'Max valve opening rate (pu/s)', 'UC': 'Max valve closing rate (pu/s)'
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Tm': {'description': 'Mechanical Torque', 'unit': 'pu',
                   'cpp_expr': 'K5*x[1] + K7*x[2]'},
            'Valve': {'description': 'Governor valve (x1)', 'unit': 'pu',
                      'cpp_expr': 'x[0]'},
            'H_gov': {'description': 'Governor storage function', 'unit': 'pu',
                      'cpp_expr': '(0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        """Auto-generated from SymbolicPHS output_map."""
        return super().get_cpp_compute_outputs_code()

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        x1, x2, x3 = sp.symbols('x_1 x_2 x_3')
        states = [x1, x2, x3]

        # Input symbols
        omega_s, Pref_s, u_agc_s = sp.symbols('omega P_{ref} u_{agc}')
        inputs = [omega_s, Pref_s, u_agc_s]

        # Parameter symbols
        K_s  = sp.Symbol('K',  positive=True)
        T1_s = sp.Symbol('T_1', positive=True)
        T3_s = sp.Symbol('T_3', positive=True)
        T7_s = sp.Symbol('T_7', positive=True)
        K5_s = sp.Symbol('K_5', nonnegative=True)
        K7_s = sp.Symbol('K_7', nonnegative=True)

        params = {'K': K_s, 'T1': T1_s, 'T3': T3_s, 'T7': T7_s,
                  'K5': K5_s, 'K7': K7_s}

        # Hamiltonian: H = ½||x||² (identity metric)
        H_expr = sp.Rational(1, 2) * (x1**2 + x2**2 + x3**2)

        # Dissipation matrix R (diagonal)
        R = sp.diag(1/T1_s, 1/T3_s, 1/T7_s)

        # Interconnection matrix J (skew-symmetric, cascaded feedforward)
        J = sp.zeros(3, 3)
        J[1, 0] =  1 / (T1_s * T3_s)
        J[0, 1] = -J[1, 0]
        J[2, 1] =  1 / (T3_s * T7_s)
        J[1, 2] = -J[2, 1]

        # Input coupling
        g = sp.zeros(3, 3)
        g[0, 0] = -K_s / T1_s   # omega → dx1
        g[0, 1] =  K_s / T1_s   # Pref → dx1
        g[0, 2] =  K_s / T1_s   # u_agc → dx1

        # Explicit dynamics: three cascaded first-order lags
        err = Pref_s + u_agc_s - omega_s
        dynamics_expr = sp.Matrix([
            (K_s * err - x1) / T1_s,   # governor servo
            (x1 - x2) / T3_s,          # HP steam chest
            (x2 - x3) / T7_s,          # reheater
        ])

        sphs = SymbolicPHS(
            name='IEEEG1_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            dynamics_expr=dynamics_expr,
            description=(
                'IEEE Type G1 Steam Turbine Governor in Port-Hamiltonian form. '
                'Three-stage cascade: governor(T1) → HP(T3) → reheater(T7). '
                'Tm = K5·x2 + K7·x3.'
            ),
        )

        # Anti-windup limiters: valve and turbine stages must stay in [PMIN, PMAX]
        # x1: governor valve position
        sphs.add_limiter(state_idx=0, upper_bound='PMAX', lower_bound='PMIN')
        # x2: HP steam-chest output — cannot be negative (no negative steam flow)
        sphs.add_limiter(state_idx=1, upper_bound='PMAX', lower_bound='PMIN')
        # x3: reheater output — same physical bounds
        sphs.add_limiter(state_idx=2, upper_bound='PMAX', lower_bound='PMIN')

        # Output: Tm = K5*x2 + K7*x3
        sphs.set_output_map({0: K5_s * x2 + K7_s * x3})

        # Equilibrium specification:
        #   At steady-state: x1=x2=x3=Tm_target (all states equal Tm)
        #   Pref is free, omega=1, u_agc=0
        sphs.set_init_spec(
            target_states={'Tm': 2},       # state 2 (x3) = targets['Tm']
            input_bindings={
                'omega': 'omega',
                'P_{ref}': None,           # free — solve for this
                'u_{agc}': 0.0,
            },
            free_param_map={'P_{ref}': 'Pref'},
            target_bounds={'Tm': ('PMIN', 'PMAX')},
        )

        return sphs

    # ------------------------------------------------------------------ #
    # Initialization (same as legacy Ieeeg1)                               #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'governor'

    def update_from_te(self, x_slice: np.ndarray, Te: float) -> tuple:
        """Update governor states and Pref so Tm = Te at equilibrium.

        At steady-state all three states equal the load point:
          x1 = x2 = x3 = Tm_target  (so dx/dt = 0 everywhere)
        and Tm = K5*x2 + K7*x3 = (K5+K7)*Tm_target = Tm_target (since K5+K7≈1).
        """
        x = x_slice.copy()
        K    = float(self.params.get('K',  20.0))
        K5   = float(self.params.get('K5',  0.3))
        K7   = float(self.params.get('K7',  0.7))
        PMIN = float(self.params.get('PMIN', -999.0))
        PMAX = float(self.params.get('PMAX',  999.0))
        Tm = Te
        if Tm < PMIN:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} < PMIN={PMIN}. Clamping.")
            Tm = PMIN
        elif Tm > PMAX:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} > PMAX={PMAX}. Clamping.")
            Tm = PMAX
        old_Pref = float(self.params.get('Pref', 0.0))
        # At equilibrium: K*(Pref - omega)/T1 = x1/T1 → Pref = omega + x1/K = 1 + Tm/K
        Pref_new = 1.0 + Tm / K
        self.params['Pref'] = Pref_new
        x[0] = Tm   # governor valve
        x[1] = Tm   # HP output
        x[2] = Tm   # reheater output (Tm = K5*Tm + K7*Tm = Tm since K5+K7=1)
        print(f"  [Init] {self.name}: Pref {old_Pref:.3f}\u2192{Pref_new:.3f}, Tm\u2192Te={Te:.4f}")
        return x, Pref_new
