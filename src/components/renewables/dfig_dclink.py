"""
DFIG_DCLink — DC-Link Capacitor Model (Port-Hamiltonian)
========================================================

Connects the Rotor-Side Converter (RSC) and Grid-Side Converter (GSC)
through a capacitor energy-storage element.

Port-Hamiltonian formulation
-----------------------------
The DC link is a *natural* PH component:

    State:   E_dc = ½ C V²_dc   (energy stored in capacitor)
    Hamiltonian:  H = E_dc ≥ 0
    Dynamics:  dE_dc/dt = P_rsc − P_gsc − R_esr · i²_dc

where P_rsc is the power flowing from the rotor circuit into the DC link
and P_gsc is the power extracted by the GSC for grid injection.

The structure/dissipation matrices reduce to scalars:
    J = 0   (no gyration for a single capacitor)
    R = R_esr · P²/(V²_dc)  ≥ 0   (ESR loss)

State (1):
    [0] E_dc   — energy stored in DC-link capacitor  [pu·s]

Inputs (2):
    [0] P_rsc   — power from RSC into DC link [pu]
    [1] P_gsc   — power extracted by GSC from DC link [pu]

Outputs (2):
    [0] Vdc     — DC-link voltage  [pu]
    [1] E_dc    — stored energy (for monitoring)  [pu·s]

Parameters:
    C_dc      — DC-link capacitance [pu·s]  (= C_phys · V²_base / S_base)
    Vdc_nom   — nominal DC-link voltage [pu]
    R_esr     — equivalent series resistance [pu] (typically very small)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigDclink(PowerComponent):
    """
    DC-link capacitor connecting RSC and GSC.

    Natural Port-Hamiltonian: H = E_dc = ½ C V²_dc ≥ 0.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('P_rsc',  'effort', 'pu'),   # [0] power from RSC into DC link
                ('P_gsc',  'effort', 'pu'),   # [1] power extracted by GSC
            ],
            'out': [
                ('Vdc',    'flow', 'pu'),     # [0] DC-link voltage
                ('E_dc',   'flow', 'pu'),     # [1] stored energy
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['E_dc']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'C_dc':    'DC-link capacitance [pu·s]',
            'Vdc_nom': 'Nominal DC-link voltage [pu]',
            'R_esr':   'Equivalent series resistance [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'passive'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vdc':   {'description': 'DC-link voltage',  'unit': 'pu',
                      'cpp_expr': 'outputs[0]'},
            'E_dc':  {'description': 'DC-link energy',   'unit': 'pu·s',
                      'cpp_expr': 'outputs[1]'},
            'P_net': {'description': 'DC-link net power (P_rsc - P_gsc)',
                      'unit': 'pu',
                      'cpp_expr': '(inputs[0] - inputs[1])'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // V_dc = sqrt(2 E_dc / C_dc), clamped to avoid sqrt(negative)
            double E_safe = fmax(x[0], 1e-6);
            outputs[0] = sqrt(2.0 * E_safe / C_dc);     // Vdc
            outputs[1] = x[0];                           // E_dc
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // PH dynamics: dE_dc/dt = P_rsc − P_gsc − P_loss
            double E_safe = fmax(x[0], 1e-6);
            double Vdc = sqrt(2.0 * E_safe / C_dc);

            // ESR power loss ≈ R_esr · (P_net / Vdc)²
            double P_net = inputs[0] - inputs[1];
            double i_dc = P_net / fmax(Vdc, 0.01);
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper/brake resistor: when Vdc exceeds 1.15 pu of Vdc_nom,
            // dump excess energy proportionally to limit voltage rise.
            double Vdc_trip = 1.15 * Vdc_nom;
            double P_brake = 0.0;
            if (Vdc > Vdc_trip) {
                // Proportional braking: ramp from 0 to full-power over 15% band
                double overshoot = (Vdc - Vdc_trip) / (0.15 * Vdc_nom);
                if (overshoot > 1.0) overshoot = 1.0;
                P_brake = overshoot * fabs(P_net + 0.5);  // dump up to (|P_net|+0.5) pu
            }

            // Physical capacitor energy dynamics with chopper protection.
            dxdt[0] = P_net - P_loss - P_brake;

            outputs[0] = Vdc;
            outputs[1] = x[0];
        """

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Initialise DC link at nominal voltage.

        At equilibrium P_rsc = P_gsc (power balance), so E_dc is constant
        at the nominal value E_dc0 = ½ C_dc V²_dc_nom.
        """
        p = self.params
        E_dc0 = 0.5 * p['C_dc'] * p['Vdc_nom'] ** 2
        return self._init_states({'E_dc': E_dc0})
