"""
Agc - Automatic Generation Control (single-area)

Implements a PI-like outer frequency regulation loop that drives
steady-state frequency back to exactly omega_ref (= 1.0 pu) after
a load disturbance.  Governor droop control alone settles at a
non-unity speed offset (e.g. 1.00035 pu); AGC corrects this by
slowly moving the governor Pref setpoints.

Architecture
------------
                    ┌─────────────────────────────┐
  omega ──────────► │  ACE = omega − omega_ref    │
                    │  d(xi)/dt = −Ki_agc * ACE   │
                    │  u_agc  = xi_agc            │
                    └──────────┬──────────────────┘
                               │  u_agc (correction, pu)
              ┌────────────────┼─────────────────────┐
              ▼                ▼                      ▼
         TGOV1_1.u_agc   TGOV1_2.u_agc   IEEEG1_4.u_agc  ...

Each governor adds u_agc to its droop setpoint:
    speed_error = (Pref + u_agc − omega) / R

At steady state omega = omega_ref  →  d(xi)/dt = 0  →  u_agc
is constant at whatever value holds the generators in balance.

Tuning guidance
---------------
  Ki_agc = 0.05   → restores frequency in ~20-30 s  (gentle, stable)
  Ki_agc = 0.1    → restores frequency in ~10-15 s  (faster)
  Ki_agc > 0.3    → risk of interaction with governor/PSS dynamics

Anti-windup
-----------
  u_agc is clamped to [u_agc_min, u_agc_max].  Default ±0.20 pu
  (covers all realistic 5%-droop governor offsets at full load).
"""

import numpy as np
from typing import Dict, List, Tuple
from src.core import PowerComponent


class Agc(PowerComponent):
    """
    Single-area Automatic Generation Control.

    One integrator state, one omega input, one u_agc output.
    The output is a Pref correction signal wired to all participating
    governors' u_agc input ports.

    Ports
    -----
    in:   omega  — system frequency measurement (pu, from any GENROU or BusFreq)
    out:  u_agc  — Pref correction sent to govornors (pu, starts at 0)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'signal', 'pu'),
            ],
            'out': [
                ('u_agc', 'signal', 'pu'),
            ],
        }

    @property
    def state_schema(self) -> List[str]:
        return ['xi_agc']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Ki_agc':    'Integral gain (pu/s per pu frequency error)',
            'omega_ref': 'Frequency reference (pu, normally 1.0)',
            'u_agc_max': 'Anti-windup upper limit on correction (pu)',
            'u_agc_min': 'Anti-windup lower limit on correction (pu)',
        }

    @property
    def component_role(self) -> str:
        return 'control'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'xi_agc': {
                'description': 'AGC integrator state (Pref correction)',
                'unit':        'pu',
                'cpp_expr':    'x[0]',
            },
            'u_agc': {
                'description': 'AGC output correction signal',
                'unit':        'pu',
                'cpp_expr':    'outputs[0]',
            },
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: Dict) -> np.ndarray:
        """AGC starts at zero correction — governors are already at equilibrium."""
        return np.array([0.0])

    # ------------------------------------------------------------------
    # C++ code generation
    # ------------------------------------------------------------------

    def get_cpp_step_code(self) -> str:
        return r"""
        // ---- AGC: Area Automatic Generation Control ----
        // State: x[0] = xi_agc  (integrated frequency error)
        // Input: inputs[0] = omega (system frequency, pu)
        //
        // d(xi_agc)/dt = -Ki_agc * (omega - omega_ref)
        // Output: u_agc = xi_agc  (Pref correction signal, pu)

        double omega_meas = inputs[0];
        double ace        = omega_meas - omega_ref;   // Area Control Error

        double xi = x[0];

        // Anti-windup: freeze integrator when saturated and error  
        // would push it further into saturation
        double dxi = -Ki_agc * ace;
        if (xi >= u_agc_max && dxi > 0.0) dxi = 0.0;
        if (xi <= u_agc_min && dxi < 0.0) dxi = 0.0;
        dxdt[0] = dxi;

        // Output: correction signal (clamped)
        double u_agc = xi;
        if (u_agc > u_agc_max) u_agc = u_agc_max;
        if (u_agc < u_agc_min) u_agc = u_agc_min;
        outputs[0] = u_agc;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return r"""
        double xi = x[0];
        double u_agc = xi;
        if (u_agc > u_agc_max) u_agc = u_agc_max;
        if (u_agc < u_agc_min) u_agc = u_agc_min;
        outputs[0] = u_agc;
        """
