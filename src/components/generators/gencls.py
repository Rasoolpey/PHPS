import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class GenCls(PowerComponent):
    """
    Classical Generator Model (GENCLS) implementation for C++ code generation.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd', 'effort', 'pu'), 
                ('Vq', 'effort', 'pu'),
                ('Tm', 'effort', 'pu')
            ],
            'out': [
                ('Id', 'flow', 'pu'),
                ('Iq', 'flow', 'pu'),
                ('omega', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['delta', 'omega']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H': 'Inertia constant',
            'D': 'Damping coefficient',
            'ra': 'Stator resistance',
            'xd1': 'Transient reactance',
            'E_p': 'Internal EMF magnitude',
            'omega_b': 'Base frequency'
        }
        
    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'delta_deg': {'description': 'Rotor angle', 'unit': 'deg', 'cpp_expr': 'x[0] * 180.0 / 3.14159265359'},
            'Te': {'description': 'Electrical Torque', 'unit': 'pu', 'cpp_expr': '(inputs[0] * outputs[0] + inputs[1] * outputs[1])'}
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            // Norton Current (Independent of V)
            // I_no = E / Z
            // E = j E_p (Rotor Frame) => ed=0, eq=E_p
            // Z = ra + j xd1
            // i_d = E_p * xd1 / (ra^2 + xd1^2)
            // i_q = E_p * ra / (ra^2 + xd1^2)
            
            double det = ra*ra + xd1*xd1;
            double id_no = E_p * xd1 / det;
            double iq_no = E_p * ra / det;
            
            // Rotate to Network Frame
            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;
            
            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1]; 
        """

    def get_cpp_step_code(self) -> str:
        return """
            double omega = x[1];
            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm = inputs[2];
            
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;
            
            // Solve Terminal Current
            // ed=0, eq=E_p
            // vd = -ra id + xd1 iq + ed -> vd = -ra id + xd1 iq
            // vq = -xd1 id - ra iq + eq -> vq - E_p = -xd1 id - ra iq
            
            // [vd; vq-E_p] = [-ra, xd1; -xd1, -ra] [id; iq]
            // inv = 1/det [-ra, -xd1; xd1, -ra]
            
            double det = ra*ra + xd1*xd1;
            double rhs_d = vd;
            double rhs_q = vq - E_p;
            
            double id = (-ra * rhs_d - xd1 * rhs_q) / det;
            double iq = (xd1 * rhs_d - ra * rhs_q) / det;
            
            double Te = vd*id + vq*iq;

            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = (Tm - Te - D*(omega - 1.0)) / (2.0 * H);
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'generator'

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        """RI-frame Norton current: E_p behind xd1."""
        p = self.params
        delta = x_slice[0]
        E_p = p.get('E_p', 1.0); ra = p.get('ra', 0.0); xd1 = p.get('xd1', 0.3)
        det  = ra**2 + xd1**2
        id_no = E_p * xd1 / det
        iq_no = E_p * ra  / det
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                       -id_no * cos_d + iq_no * sin_d)

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        """Solve stator equations for actual (id, iq) at terminal vd, vq."""
        p = self.params
        E_p  = p.get('E_p', 1.0); ra = p.get('ra', 0.0); xd1 = p.get('xd1', 0.3)
        det  = ra**2 + xd1**2
        rhs_d = vd
        rhs_q = vq - E_p
        id_act = (-ra * rhs_d - xd1 * rhs_q) / det
        iq_act = ( xd1 * rhs_d - ra  * rhs_q) / det
        return id_act, iq_act

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """Init delta and omega from power-flow phasor."""
        p   = self.params
        ra  = p.get('ra', 0.0)
        xd1 = p.get('xd1', 0.3)
        Z   = ra + 1j * xd1
        E   = V_phasor + Z * I_phasor
        delta  = float(np.angle(E))
        Efd_req = float(abs(E))
        Tm_req  = float((V_phasor * np.conj(I_phasor)).real)
        p['E_p'] = Efd_req   # store for Norton current computation

        targets = {
            'Efd': Efd_req, 'Tm': Tm_req,
            'Vt':  float(abs(V_phasor)),
            'omega': 1.0,
            'vd':  0.0, 'vq': float(abs(V_phasor)),
            'id':  0.0, 'iq': 0.0,
            'vd_ri': float(V_phasor.real),
            'vq_ri': float(V_phasor.imag),
        }
        return np.array([delta, 1.0]), targets
