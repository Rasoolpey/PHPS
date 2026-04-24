import math
import os
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


def _load_wind_profile_file(path: str) -> Tuple[List[float], List[float]]:
    """Parse a wind profile txt file with lines of the form 't, v'.

    Blank lines and comment lines (starting with '#') are skipped.
    Returns (times, speeds) as lists of floats.
    """
    times, speeds = [], []
    abs_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"WindFileReader: profile_file not found: {abs_path!r}")
    with open(abs_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            times.append(float(parts[0].strip()))
            speeds.append(float(parts[1].strip()))
    if not times:
        raise ValueError(f"WindFileReader: no data found in {abs_path!r}")
    return times, speeds


class WindFileReader(PowerComponent):
    """
    Component that generates a dynamic wind speed output by interpolating from a profile.
    It takes an embedded C++ time-series array rather than reading from disk dynamically to maximize C++ performance.

    The profile can be supplied in two ways:
      1. Directly as ``times``, ``speeds``, and ``num_points`` scalar/array params.
      2. Via a ``profile_file`` param pointing to a text file with lines ``"t, v"``.
         The file is resolved relative to the working directory.  When
         ``profile_file`` is used, ``times``, ``speeds``, and ``num_points``
         are derived from the file automatically.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        # If a profile_file is given, load it and populate times/speeds/num_points
        # before the base-class validation runs.
        if 'profile_file' in params and params['profile_file']:
            params = dict(params)   # don't mutate original
            ts, vs = _load_wind_profile_file(str(params['profile_file']))
            params['times'] = ', '.join(f'{t}' for t in ts)
            params['speeds'] = ', '.join(f'{v}' for v in vs)
            params['num_points'] = len(ts)
            print(f"  [WindFileReader] {name}: loaded {len(ts)} points "
                  f"from '{params['profile_file']}'  "
                  f"(t={ts[0]}..{ts[-1]} s, v={vs[0]}..{max(vs):.4g} pu)")
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [],
            'out': [('vw', 'flow', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]: return []

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'times': 'Comma separated array of time values',
            'speeds': 'Comma separated array of wind speed values',
            'num_points': 'Number of points'
        }

    @property
    def component_role(self) -> str: return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'vw_out': {'description': 'Output wind speed', 'unit': 'pu', 'cpp_expr': 'outputs[0]'}
        }

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        return self._init_states({})

    def get_cpp_step_code(self) -> str: return ""

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        // Look up time (t) in arrays and interpolate
        // Since parameters are constants, we use static arrays to define our time series
        int n_pts = (int)num_points;
        double v = speeds[0];
        if (t <= times[0]) {
            v = speeds[0];
        } else if (t >= times[n_pts - 1]) {
            v = speeds[n_pts - 1];
        } else {
            for (int i = 0; i < n_pts - 1; ++i) {
                if (t >= times[i] && t < times[i+1]) {
                    double dt = times[i+1] - times[i];
                    double wt = (t - times[i]) / dt;
                    v = speeds[i] * (1.0 - wt) + speeds[i+1] * wt;
                    break;
                }
            }
        }
        outputs[0] = v;
        """
