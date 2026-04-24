"""
TGOV1 — Port-Hamiltonian Formulation.

Steam Turbine Governor with explicit (J, R, H) structure.

States: x = [x1, x2, xi]
  x1 : valve position
  x2 : turbine / reheater output
  xi : integral frequency correction (isochronous mode)

Storage function (generalised Hamiltonian):
    H = ½·T1·x1² + ½·T3·x2² + ½·(1/Ki)·xi²  (Ki>0)
      = ½·T1·x1² + ½·T3·x2²                    (Ki=0)

This weights each state by its dominant time constant, giving:
    ∂H/∂x1 = T1·x1
    ∂H/∂x2 = T3·x2
    ∂H/∂xi = (1/Ki)·xi   (Ki>0)

Dissipation matrix R (diagonal, ≥ 0):
    R[0,0] = 1/T1   →  −R00·∂H/∂x1 = −x1    (valve lag)
    R[1,1] = 1/T3   →  −R11·∂H/∂x2 = −x2    (reheater lag)
    R[2,2] = Ki      →  −R22·∂H/∂xi = −xi     (integral action)

Port inputs: u = [omega, Pref, u_agc]
Output:      y = [Tm]

    Tm = x2 + (T2/T3)·(x1 − x2) + xi

References
----------
- IEEE Std 421.5-2016 / WECC TGOV1 model specification
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


class Tgov1PHS(PowerComponent):
    """
    TGOV1 Steam Turbine Governor in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u
    """

    _DEFAULTS = {'Ki': 0.0, 'wref0': 1.0, 'xi_max': 1.0, 'xi_min': -1.0}

    def __init__(self, name: str, params: Dict[str, Any]):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow', 'pu'),
                ('Pref',  'signal', 'pu'),
                ('u_agc', 'signal', 'pu'),
            ],
            'out': [('Tm', 'effort', 'pu')]
        }

    @property
    def required_ports(self):
        return ['omega', 'Pref']

    @property
    def state_schema(self) -> List[str]:
        return ['x1', 'x2', 'xi']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'R':     'Droop (pu)',
            'T1':    'Governor Time Constant (s)',
            'T2':    'Turbine Lead Time Constant (s)',
            'T3':    'Reheater / Turbine Lag Time Constant (s)',
            'Dt':    'Turbine Damping',
            'VMAX':  'Max Valve Position',
            'VMIN':  'Min Valve Position',
            'Ki':    'Integral Gain (0=droop only, >0=isochronous)',
            'wref0': 'Frequency Reference (pu, default 1.0)',
            'xi_max': 'Integral state upper limit (anti-windup)',
            'xi_min': 'Integral state lower limit (anti-windup)',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Tm':    {'description': 'Mechanical Torque',
                      'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Valve': {'description': 'Valve Position',
                      'unit': 'pu', 'cpp_expr': 'x[0]'},
            'xi':    {'description': 'Integral Frequency Correction',
                      'unit': 'pu', 'cpp_expr': 'x[2]'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        """

    def get_cpp_step_code(self) -> str:
        """Auto-generated from SymbolicPHS with limiter annotations.

        The base class generates C++ from the PHS dynamics_expr and wraps
        limited states (x1, xi) with anti-windup guards.
        """
        return super().get_cpp_step_code()

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        x1, x2, xi = sp.symbols('x_1 x_2 xi')
        states = [x1, x2, xi]

        # Input symbols
        omega_in, Pref, u_agc = sp.symbols('omega P_{ref} u_{agc}')
        inputs = [omega_in, Pref, u_agc]

        # Parameter symbols
        R_s = sp.Symbol('R_d', positive=True)   # droop
        T1_s = sp.Symbol('T_1', positive=True)
        T3_s = sp.Symbol('T_3', positive=True)
        Ki_s = sp.Symbol('K_i', nonnegative=True)

        params = {
            'R': R_s, 'T1': T1_s, 'T3': T3_s, 'Ki': Ki_s,
        }

        # Hamiltonian: H = ½T1·x1² + ½T3·x2² + ½(1/Ki)·xi² (Ki>0)
        # For symbolic generality, use identity-weighted form
        H_expr = sp.Rational(1, 2) * (x1**2 + x2**2)

        # Dissipation
        R_mat = sp.diag(1/T1_s, 1/T3_s, sp.Integer(0))

        # Interconnection (feedforward x1 → x2)
        J = sp.zeros(3, 3)
        J[1, 0] = 1 / (T1_s * T3_s)
        J[0, 1] = -J[1, 0]

        # Input coupling
        g = sp.zeros(3, 3)
        g[0, 0] = -1 / (T1_s * R_s)  # omega → dx1
        g[0, 1] = 1 / (T1_s * R_s)   # Pref → dx1
        g[0, 2] = 1 / (T1_s * R_s)   # u_agc → dx1
        g[2, 0] = -Ki_s               # omega → dxi

        # wref0 is a parameter (not an input port), so the constant term
        # Ki*wref0 can't appear in g*u.  Use dynamics_expr override.
        wref0_s = sp.Symbol('wref0')
        params['wref0'] = wref0_s

        speed_error = (Pref + u_agc - omega_in) / R_s
        dynamics_expr = sp.Matrix([
            (speed_error - x1) / T1_s,
            (x1 - x2) / T3_s,
            Ki_s * (wref0_s - omega_in),
        ])

        sphs = SymbolicPHS(
            name='TGOV1_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R_mat, g=g, H=H_expr,
            dynamics_expr=dynamics_expr,
            description=(
                'Steam turbine governor (TGOV1) in Port-Hamiltonian form. '
                'Identity-weighted storage for the valve and reheater states.'
            ),
        )

        # Anti-windup limiters
        sphs.add_limiter(state_idx=0, upper_bound='VMAX', lower_bound='VMIN')
        # Reheater output cannot go below VMIN (no negative steam power)
        sphs.add_limiter(state_idx=1, upper_bound='VMAX', lower_bound='VMIN')
        sphs.add_limiter(state_idx=2, upper_bound='xi_max', lower_bound='xi_min')

        # Equilibrium specification for auto-derived init_from_targets():
        #   x2 = Tm target, Pref is free, omega from targets, u_agc = 0
        #   xi is underdetermined (dxi/dt=0 when omega=wref0) → default 0
        sphs.set_init_spec(
            target_states={'Tm': 1},       # state 1 (x2) = targets['Tm']
            input_bindings={
                'omega': 'omega',          # from targets dict
                'P_{ref}': None,           # free — solve for this
                'u_{agc}': 0.0,            # literal zero
            },
            free_param_map={'P_{ref}': 'Pref'},
            state_defaults={2: 0.0},       # xi = 0 at equilibrium
            target_bounds={'Tm': ('VMIN', 'VMAX')},
        )

        return sphs

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'governor'

    def update_from_te(self, x_slice: np.ndarray, Te: float) -> tuple:
        x = x_slice.copy()
        R = float(self.params.get('R', 0.05))
        for k, v in self._DEFAULTS.items():
            self.params.setdefault(k, v)
        VMIN = float(self.params.get('VMIN', 0.0))
        VMAX = float(self.params.get('VMAX', 999.0))
        Tm = Te
        if Tm < VMIN:
            print(f"  [WARN] {self.name}: Tm={Tm:.6f} < VMIN={VMIN}. Clamping.")
            Tm = VMIN
        elif Tm > VMAX:
            print(f"  [WARN] {self.name}: Tm={Tm:.6f} > VMAX={VMAX}. Clamping.")
            Tm = VMAX
        old_Pref = float(self.params.get('Pref', 0.0))
        Pref_new = 1.0 + Tm * R
        self.params['Pref'] = Pref_new
        x[0] = Tm; x[1] = Tm; x[2] = 0.0
        print(f"  [Init] {self.name}: Pref {old_Pref:.3f}->{Pref_new:.3f}, Tm->Te={Te:.4f}")
        return x, Pref_new
