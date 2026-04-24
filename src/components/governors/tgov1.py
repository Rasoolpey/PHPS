import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Tgov1(PowerComponent):
    """
    TGOV1 Steam Turbine Governor implementation for C++ code generation.
    """

    _DEFAULTS = {'Ki': 0.0, 'wref0': 1.0, 'xi_max': 1.0, 'xi_min': -1.0}

    def __init__(self, name: str, params: Dict[str, Any]):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        # u_agc is optional: defaults to 0.0 via compiler when not connected
        return {
            'in': [
                ('omega', 'flow', 'pu'), 
                ('Pref',  'signal', 'pu'),
                ('u_agc', 'signal', 'pu'),   # AGC correction (+ve = increase setpoint)
            ],
            'out': [
                ('Tm', 'effort', 'pu')
            ]
        }

    @property
    def required_ports(self):
        """u_agc is optional; defaults to 0.0 when no AGC is wired."""
        return ['omega', 'Pref']

    @property
    def state_schema(self) -> List[str]:
        # x[0]=x1 (valve position), x[1]=x2 (reheater), x[2]=xi (frequency error integral)
        return ['x1', 'x2', 'xi']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'R':     'Droop (pu)',
            'T1':    'Governor Time Constant (s)',
            'T2':    'Turbine Time Constant (s)',
            'T3':    'Reheater Time Constant (s)',
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
            # Use outputs[0] (already computed by get_cpp_compute_outputs_code)
            # so this works correctly in the CSV-logging context.
            'Tm':    {'description': 'Mechanical Torque (droop+integral)',
                      'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Valve': {'description': 'Valve Position',
                      'unit': 'pu', 'cpp_expr': 'x[0]'},
            'xi':    {'description': 'Integral Frequency Correction',
                      'unit': 'pu', 'cpp_expr': 'x[2]'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            // Clamp total mechanical torque to valve limits
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        """

    def get_cpp_step_code(self) -> str:
        return """
            double omega  = inputs[0];
            double Pref   = inputs[1];
            double u_agc  = inputs[2];   // AGC Pref correction (0.0 when AGC not wired)

            double x1 = x[0];   // valve position
            double x2 = x[1];   // reheater output
            double xi = x[2];   // integral correction

            // ---- Droop path (standard TGOV1 + AGC correction) ----
            double speed_error = (Pref + u_agc - omega) / R;
            double dx1 = (speed_error - x1) / T1;
            if (x1 >= VMAX && dx1 > 0) dx1 = 0;
            if (x1 <= VMIN && dx1 < 0) dx1 = 0;
            dxdt[0] = dx1;
            dxdt[1] = (x1 - x2) / T3;

            // ---- Integral correction with anti-windup ----
            double dxi = Ki * (wref0 - omega);
            // Anti-windup: freeze integrator when at limit and error drives it further
            if (xi >= xi_max && dxi > 0) dxi = 0;
            if (xi <= xi_min && dxi < 0) dxi = 0;
            dxdt[2] = dxi;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'governor'

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """
        Initialise governor states.

        x[0]=x1 and x[1]=x2 follow the standard TGOV1 droop initialisation
        (valve position = reheater output = Tm_0).

        x[2]=xi is the fast integral correction and starts at **0.0**:
        it is an additive bias on Tm that integrates away any residual
        frequency error over time.  At t=0 the output is:
          Tm(0) = x2(0) + (T2/T3)*(x1(0)-x2(0)) + xi(0) = Tm_0 + 0 = Tm_0
        """
        Tm    = float(targets.get('Tm', 0.0))
        R     = float(self.params.get('R', 0.05))
        for k, v in self._DEFAULTS.items():
            self.params.setdefault(k, v)
        VMIN = float(self.params.get('VMIN', 0.0))
        VMAX = float(self.params.get('VMAX', 999.0))
        if Tm < VMIN:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} < VMIN={VMIN}. "
                  f"Clamping governor output to VMIN.")
            Tm = VMIN
        elif Tm > VMAX:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} > VMAX={VMAX}. "
                  f"Clamping governor output to VMAX.")
            Tm = VMAX
        self.params['Pref'] = 1.0 + Tm * R
        return np.array([Tm, Tm, 0.0])

    def update_from_te(self, x_slice: np.ndarray, Te: float) -> tuple:
        """Update governor states so Tm = Te at equilibrium."""
        x = x_slice.copy()
        R        = float(self.params.get('R', 0.05))
        for k, v in self._DEFAULTS.items():
            self.params.setdefault(k, v)
        VMIN = float(self.params.get('VMIN', 0.0))
        VMAX = float(self.params.get('VMAX', 999.0))
        Tm = Te
        if Tm < VMIN:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} < VMIN={VMIN}. "
                  f"Clamping governor output to VMIN.")
            Tm = VMIN
        elif Tm > VMAX:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} > VMAX={VMAX}. "
                  f"Clamping governor output to VMAX.")
            Tm = VMAX
        old_Pref = float(self.params.get('Pref', 0.0))
        Pref_new = 1.0 + Tm * R
        self.params['Pref'] = Pref_new
        x[0] = Tm
        x[1] = Tm
        x[2] = 0.0
        print(f"  [Init] {self.name}: Pref {old_Pref:.3f}->{Pref_new:.3f}, Tm->Te={Te:.4f}")
        return x, Pref_new
