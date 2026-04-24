"""
ComplexLoad — PowerFactory-compatible static load with frequency dependency.

Implements the PowerFactory Complex Load model (ElmLod) static part for
RMS simulation (Section 5.1.2 of the PF Technical Reference).

Load Model (constant-impedance with frequency dependency):
    P_load = P0 · (u/u0)^2 · (1 + kpf · (fe - 1))
    Q_load = Q0 · (u/u0)^2 · (1 + kqf · (fe - 1))

where fe is the system electrical frequency in p.u. (COI omega).

In the PHPS DAE framework, the base constant-impedance load Y = (P-jQ)/V0²
is already embedded in the Y-bus matrix.  This component adds ONLY the
frequency-dependent correction current:

    ΔI_d = kpf·Δω·G_load·Vd - kqf·Δω·B_load·Vq
    ΔI_q = kpf·Δω·G_load·Vq + kqf·Δω·B_load·Vd

where G_load = P0/V0², B_load = -Q0/V0², Δω = ω_COI - 1.

This is a zero-state (purely algebraic) component: no differential equations,
only an output function that computes the frequency-dependent current
correction.  The correction current is injected into the KCL at the bus.

Note: The frequency-dependent injection is handled directly in the DAE
compiler's KCL residual (using the per-bus LOAD_G/LOAD_B arrays and
KPF/KQF constants from config), NOT through this component's output ports.
This component serves as the declarative specification in the system JSON
and as the parameter container for per-load kpf/kqf values.

Ports:
    in:  [Vd, Vq]        — bus voltage (RI frame)
    out: [Id, Iq, Pload, Qload]  — diagnostic outputs

Parameters:
    P0   : Nominal active power consumption [pu on system base]
    Q0   : Nominal reactive power consumption [pu on system base]
    V0   : Nominal voltage magnitude [pu]
    kpf  : Active power frequency factor (PF default: 0.0)
    kqf  : Reactive power frequency factor (PF default: 0.0)
    bus  : Bus index where the load is connected

References
----------
- DIgSILENT PowerFactory 2024, Technical Reference: Complex Load
  (ElmLod, TypLodind), Section 5.1.2 (RMS Simulation, Static Part)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class ComplexLoadPHS(PowerComponent):
    """PF-compatible complex load with voltage and frequency dependency.

    Zero-state component: no differential equations.
    The frequency-dependent load correction is handled at the KCL level
    in the DAE compiler using per-bus kpf/kqf parameters read from
    this component.
    """

    _DEFAULTS = {
        'P0': 0.0,
        'Q0': 0.0,
        'V0': 1.0,
        'kpf': 0.0,
        'kqf': 0.0,
    }

    def __init__(self, name: str, params: dict):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu'),
            ],
            'out': [
                ('Id',    'flow', 'pu'),
                ('Iq',    'flow', 'pu'),
                ('Pload', 'flow', 'pu'),
                ('Qload', 'flow', 'pu'),
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return []

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'P0':  'Nominal active power [pu]',
            'Q0':  'Nominal reactive power [pu]',
            'V0':  'Nominal voltage magnitude [pu]',
            'kpf': 'Active power frequency factor',
            'kqf': 'Reactive power frequency factor',
            'bus': 'Bus index',
        }

    @property
    def component_role(self) -> str:
        return 'load'

    @property
    def contributes_norton_admittance(self) -> bool:
        return False

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pload': {'description': 'Active power consumed',   'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'Qload': {'description': 'Reactive power consumed', 'unit': 'pu', 'cpp_expr': 'outputs[3]'},
        }

    def get_cpp_step_code(self) -> str:
        """No differential states — empty step function."""
        return ""

    def get_cpp_compute_outputs_code(self) -> str:
        """Compute diagnostic output: load P, Q at current voltage.

        The actual frequency-dependent current injection is handled
        in the DAE compiler KCL residual, not here.  The outputs here
        are for monitoring/logging only.
        """
        P0 = float(self.params.get('P0', 0.0))
        Q0 = float(self.params.get('Q0', 0.0))
        V0 = float(self.params.get('V0', 1.0))
        V02 = max(V0 * V0, 1e-12)
        G_load = P0 / V02
        B_load = -Q0 / V02

        return f"""
        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = {G_load:.12e};
        double B_load = {B_load:.12e};

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        """

    def init_from_targets(self, targets: Dict[str, float]) -> np.ndarray:
        """No states to initialize."""
        return np.zeros(0)
