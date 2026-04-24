"""
DFIG_DCLINK_PHS — DC-Link Capacitor (Σ₄ Port-Hamiltonian Model)
================================================================

The DC-link capacitor stores electrostatic energy between the RSC and GSC.

State (1 energy variable):
    x₄ = V_dc  (DC-bus voltage)

Hamiltonian:
    H₄ = ½ C_dc V_dc²

Gradient:
    ∇H₄ = C_dc V_dc = q_dc  (stored charge)

PHS structure:
    J₄ = [0],  R₄ = [R_esr / V_dc²],  g₄ = [1/V_dc, −1/V_dc]

Dynamics:
    V̇_dc = (P_rsc − P_gsc) / (C_dc·V_dc) − R_esr·(P_rsc − P_gsc) / (C_dc·V_dc³)

Ports:
    in:  [P_rsc, P_gsc]
    out: [V_dc, P_net]

Power balance:
    dH₄/dt = P_rsc − P_gsc − R_esr·i_dc²
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigDclinkPHS(PowerComponent):
    """
    DC-link capacitor (Σ₄) in Port-Hamiltonian form.

    Single-state model using V_dc as the energy variable.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('P_rsc', 'effort', 'pu'),   # [0] power from RSC
                ('P_gsc', 'effort', 'pu'),   # [1] power to GSC
            ],
            'out': [
                ('Vdc',   'flow', 'pu'),     # [0] DC-bus voltage
                ('P_net', 'flow', 'pu'),     # [1] net power (monitoring)
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['V_dc']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'C_dc':    'DC-link capacitance [pu]',
            'Vdc_nom': 'Nominal DC voltage [pu]',
            'R_esr':   'Equivalent series resistance [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'passive'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vdc':    {'description': 'DC-bus voltage',       'unit': 'pu', 'cpp_expr': 'x[0]'},
            'P_net':  {'description': 'Net DC power',         'unit': 'pu', 'cpp_expr': 'outputs[1]'},
            'H_dc':   {'description': 'DC-link energy',       'unit': 'pu',
                       'cpp_expr': '0.5 * C_dc * x[0] * x[0]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vdc = fmax(x[0], 0.01);
            outputs[0] = Vdc;
            outputs[1] = inputs[0] - inputs[1];  // P_net = P_rsc - P_gsc
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double Vdc   = fmax(x[0], 0.01);
            double P_rsc = inputs[0];
            double P_gsc = inputs[1];

            // DC-link current
            double P_net = P_rsc - P_gsc;
            double i_dc  = P_net / Vdc;

            // ESR power loss
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper brake (overvoltage protection)
            double P_brake = 0.0;
            if (Vdc > 1.15 * Vdc_nom) {
                P_brake = (Vdc - 1.15 * Vdc_nom) * 10.0;
            }

            // C_dc · dVdc/dt = (P_rsc - P_gsc - P_loss - P_brake) / Vdc
            dxdt[0] = (P_net - P_loss - P_brake) / (C_dc * Vdc);
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        V_dc = sp.Symbol('V_dc', positive=True)
        states = [V_dc]

        P_rsc = sp.Symbol('P_rsc')
        P_gsc = sp.Symbol('P_gsc')
        inputs = [P_rsc, P_gsc]

        C_dc = sp.Symbol('C_dc', positive=True)
        R_esr = sp.Symbol('R_esr', nonnegative=True)
        Vdc_nom = sp.Symbol('V_{dc,nom}', positive=True)

        params = {'C_dc': C_dc, 'R_esr': R_esr, 'Vdc_nom': Vdc_nom}

        # Hamiltonian
        H_expr = sp.Rational(1, 2) * C_dc * V_dc**2

        # PHS matrices
        J = sp.Matrix([[0]])
        R = sp.Matrix([[R_esr / V_dc**2]])
        g = sp.Matrix([[1 / V_dc, -1 / V_dc]])

        return SymbolicPHS(
            name='DFIG_DCLINK_PHS',
            states=states, inputs=inputs, params=params,
            J=J, R=R, g=g, H=H_expr,
            description='DC-link capacitor (Σ₄) in PH form.',
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        Vdc_nom = float(self.params['Vdc_nom'])
        return self._init_states({'V_dc': Vdc_nom})
