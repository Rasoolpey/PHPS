"""
IDA-PBC Exciter Controller
==========================
Replaces the classical AVR with a Port-Hamiltonian energy-shaping controller
derived from Interconnection and Damping Assignment Passivity-Based Control
(IDA-PBC; Ortega, Galaz, Astolfi et al., 2005).

Theory
------
The GENROU field-circuit ODE is:

    Td0' * dEq'/dt = Efd - Eq' - (xd - xd') * id

At the operating point x* the equilibrium field voltage is:

    Efd* = Eq'* + (xd - xd') * id*

The PH desired energy function for the closed-loop system is:

    H_d(x) = H*(ω-1)²  +  ½ * c_v * (Vt - Vref)²  +  cross terms

Energy-shaping assigns a "virtual spring" in the field direction
(driving Vt → Vref) and damping injection couples the mechanical
speed port into the field voltage:

    Efd = Efd_0
          + Kv * (Vref - Vt)          ← voltage shaping  (replaces KA roll)
          + Kd * (ω  − 1.0)           ← speed cross-damping  (PH-only term)
          + Ki * ξ                     ← integral anti-windup action

where ξ integrates the voltage error:

    dξ/dt = Vref − Vt

The Kd term is the PH-specific contribution absent in classical AVRs.
Physically it pre-excites the field when the machine accelerates (ω > 1),
injecting reactive power that brakes the rotor via electromagnetic torque
– precisely the IDA mechanism in energy coordinates.

Stability guarantee (Lyapunov)
-------------------------------
With the Kd cross term the closed-loop Hamiltonian decreases along every
trajectory (for K_v > 0, K_d > 0).  The convergence rate improves with Kd:
the effective post-fault damping is approximately D_eff ≈ D + Kd * ∂Te/∂ω.

Ports  (same interface as IEEEX1 / ESST3A for drop-in compatibility)
-----
in:  Vterm — terminal voltage magnitude [pu]
     Vref  — voltage reference setpoint [pu]  (auto-wired from params + PSS)
     omega — rotor speed [pu]
out: Efd   — field voltage [pu]

Parameters
----------
Kv       : Voltage proportional gain                (default 10.0)
Kd       : Speed cross-damping gain  (PH-specific)  (default  2.0)
Ki       : Integral gain                            (default  0.5)
Efd_min  : Lower field voltage limit                (default -6.0 pu)
Efd_max  : Upper field voltage limit                (default  6.0 pu)
xi_min   : Integrator anti-windup lower limit       (default -3.0 pu)
xi_max   : Integrator anti-windup upper limit       (default  3.0 pu)
Efd_0    : Equilibrium field voltage — SET DURING INIT
Vref     : Voltage reference — SET DURING INIT

References
----------
  Ortega R., Galaz M., Astolfi A., Sun Y., Shen T. (2005).
  "Transient Stabilization of Multimachine Power Systems With Nontrivial
   Transfer Conductances."  IEEE Trans. Autom. Control 50(1):60-75.

  Galaz M., Ortega R., Bazanella A.S., Stankovic A.M. (2003).
  "An energy-shaping approach to excitation control of synchronous
   generators."  Automatica 39(1):111-119.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class IdaPbcExciter(PowerComponent):
    """
    IDA-PBC Exciter — energy-shaping + damping-injection voltage regulator.

    Drop-in replacement for ESST3A / IEEEX1 with the same port interface
    (Vterm, Vref, omega → Efd) but designed from Port-Hamiltonian principles.

    One state: ξ (voltage-error integral for steady-state tracking).
    """

    # Signal to initialization.py that this exciter has NO voltage-transducer
    # state at index 0  (unlike IEEEX1 / ESST3A whose x[0] = Vm).
    # refine_exciter_voltages checks this flag to skip the lightweight Vm update.
    _has_voltage_transducer: bool = False

    _DEFAULTS: Dict[str, Any] = {
        'Kv':      10.0,   # voltage proportional gain
        'Kd':       2.0,   # speed cross-damping gain (PH-specific term)
        'Ki':       0.5,   # integral gain
        'Efd_min': -6.0,   # field voltage lower limit [pu]
        'Efd_max':  6.0,   # field voltage upper limit [pu]
        'xi_min':  -3.0,   # integrator anti-windup lower limit
        'xi_max':   3.0,   # integrator anti-windup upper limit
        # Set during init_from_targets(); defaults are placeholders only
        'Efd_0':    1.0,   # equilibrium field voltage  [overwritten at init]
        'Vref':     1.0,   # voltage reference          [overwritten at init]
    }

    def __init__(self, name: str, params: Dict[str, Any]):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        super().__init__(name, params)

    # ------------------------------------------------------------------
    # Port / state / param schemas
    # ------------------------------------------------------------------

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [
                ('Vterm', 'signal', 'pu'),
                ('Vref',  'signal', 'pu'),   # auto-wired from params (+ PSS if present)
                ('omega', 'flow',   'pu'),   # rotor speed from GENROU
            ],
            'out': [
                ('Efd', 'effort', 'pu'),
            ],
        }

    @property
    def state_schema(self) -> List[str]:
        return ['xi']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kv':      'Voltage proportional gain (energy-shaping)',
            'Kd':      'Speed cross-damping gain (IDA-PBC term, damps swing)',
            'Ki':      'Voltage error integral gain (anti-offset)',
            'Efd_min': 'Field voltage lower limit [pu]',
            'Efd_max': 'Field voltage upper limit [pu]',
            'xi_min':  'Integrator state lower limit',
            'xi_max':  'Integrator state upper limit',
            'Efd_0':   'Equilibrium field voltage — set at initialization [pu]',
            'Vref':    'Voltage reference — set at initialization [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'exciter'

    # ------------------------------------------------------------------
    # Exciter output helpers (used by Initializer.refine_Eq_p)
    # ------------------------------------------------------------------

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Return the Efd output at current exciter state.

        IDA-PBC control law at near-equilibrium (ω≈1, Vt≈Vref):
            Efd ≈ Efd_0 + Ki * ξ

        The proportional (Kv) and speed-damping (Kd) terms vanish at the
        operating point, so this approximation is exact at equilibrium and
        a good first-order estimate during the refinement sweeps.

        Called by Initializer.refine_Eq_p() to determine how much field
        flux (Eq_prime) the generator should carry.
        """
        xi = float(x_slice[0])   # x_slice[0] is ξ
        return float(self.params['Efd_0']) + float(self.params.get('Ki', 0.5)) * xi

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'xi': {
                'description': 'Voltage-error integral state',
                'unit':        'pu',
                'cpp_expr':    'x[0]',
            },
            'Efd_out': {
                'description': 'Field voltage output',
                'unit':        'pu',
                'cpp_expr':    'outputs[0]',
            },
            'v_err': {
                'description': 'Voltage error (Vref − Vt)',
                'unit':        'pu',
                'cpp_expr':    'inputs[1] - inputs[0]',
            },
            'speed_dev': {
                'description': 'Speed deviation fed to damping term (ω−1)',
                'unit':        'pu',
                'cpp_expr':    'inputs[2] - 1.0',
            },
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: Dict) -> np.ndarray:
        """
        Set equilibrium parameters from the power-flow targets dict.

        The GENROU init_from_phasor() provides:
          targets['Efd'] — required equilibrium field voltage
          targets['Vt']  — terminal voltage magnitude at operating point

        Both are baked into the C++ code as const parameters.
        The integral state starts at 0 (already at equilibrium).
        """
        efd_eq = float(targets.get('Efd', 1.0))
        vt_eq  = float(targets.get('Vt',  1.0))

        self.params['Efd_0'] = efd_eq
        self.params['Vref']  = vt_eq   # compiler auto-wires this + optional PSS

        return np.array([0.0])         # ξ₀ = 0 at equilibrium

    # ------------------------------------------------------------------
    # C++ code generation
    # ------------------------------------------------------------------

    def get_cpp_step_code(self) -> str:
        return r"""
        // ── IDA-PBC Exciter ──────────────────────────────────────────
        // State:   x[0] = ξ  (voltage-error integral)
        // Inputs:  inputs[0] = Vterm   (terminal voltage magnitude)
        //          inputs[1] = Vref    (setpoint  ≈ Vt_eq + PSS signal)
        //          inputs[2] = omega   (rotor speed)
        //
        // IDA-PBC control law:
        //   Efd = Efd_0
        //         + Kv  * (Vref − Vt)       ← voltage energy shaping
        //         + Kd  * (ω   − 1.0)       ← speed cross-damping (PH term)
        //         + Ki  * ξ                 ← integral anti-offset
        //
        // The Kd term is unique to the PH formulation: it pre-excites the
        // field when ω > 1 (machine accelerating), injecting stabilising
        // reactive power that backs off the rotor via electromagnetic torque.

        double Vt    = inputs[0];
        double Vref_sig = inputs[1];
        double omega = inputs[2];

        double v_err = Vref_sig - Vt;
        double xi    = x[0];

        // ── Integrator with anti-windup ────────────────────────────
        double dxi = v_err;
        if (xi >= xi_max && dxi > 0.0) dxi = 0.0;
        if (xi <= xi_min && dxi < 0.0) dxi = 0.0;
        dxdt[0] = dxi;

        // ── IDA-PBC field voltage ──────────────────────────────────
        double Efd = Efd_0
                   + Kv * v_err
                   + Kd * (omega - 1.0)
                   + Ki * xi;

        // Output limits
        if (Efd > Efd_max) Efd = Efd_max;
        if (Efd < Efd_min) Efd = Efd_min;
        outputs[0] = Efd;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
        // IDA-PBC output (algebraic, given current state + inputs)
        double Vt    = inputs[0];
        double Vref_sig = inputs[1];
        double omega = inputs[2];
        double xi    = x[0];

        double Efd = Efd_0
                   + Kv * (Vref_sig - Vt)
                   + Kd * (omega - 1.0)
                   + Ki * xi;

        if (Efd > Efd_max) Efd = Efd_max;
        if (Efd < Efd_min) Efd = Efd_min;
        outputs[0] = Efd;
        """
