"""
Dirac DAE Compiler — C++ code generation for the DAE formulation.

Generates a self-contained C++ kernel where the network constraints
(KCL at every bus, voltage source at slack buses) appear as **explicit
algebraic equations** in a DAE residual, rather than being solved via
an iterative V = Z·I loop.

The generated C++ contains:
  1. Per-component step functions (identical to the ODE compiler)
  2. Full Y-bus (no Kron reduction) embedded as constant arrays
  3. A ``dae_residual()`` function computing F(t, y, ẏ) = 0
  4. A BDF-1 (backward Euler) implicit solver with Newton iteration
  5. A ``main()`` driver with CSV output

DAE state vector layout::

    y = [ x_diff_0, ..., x_diff_{n-1},     (differential: component ODE states)
          Vd_bus0, Vq_bus0,                  (algebraic: bus voltages RI frame)
          Vd_bus1, Vq_bus1,
          ... ]

    For differential states:  res[i] = ẏ[i] − f(x, V)
    For algebraic states:
      - Generator/load bus i: res = KCL mismatch = I_inj − Y_bus·V
      - Slack bus i:          res = V − V_ref  (fixed voltage)

This module does NOT modify any existing code.  It delegates to
``SystemCompiler`` for structure building and initialization, then
generates its own C++ independently.
"""

from __future__ import annotations

import math
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.system_graph import SystemGraph, build_system_graph
from src.compiler import SystemCompiler
from src.core import PowerComponent
from src.ybus import YBusBuilder


DAE_SOLVER_LABELS = {
    "bdf1":   "BDF-1 (implicit Newton)",
    "ida":    "SUNDIALS IDA (variable-order BDF, adaptive step)",
    "midpoint": "Implicit Midpoint (structure-preserving)",
    "scipy":  "Python scipy Radau (network-reduced ODE)",
    "jit":    "Numba JIT BDF-1 fixed-step (network-reduced ODE)",
}

# Solvers that are handled entirely in Python (no C++ generation or compilation)
PYTHON_SOLVERS = {"scipy", "jit"}

_DAE_SOLVER_ALIASES = {
    "bdf1": "bdf1",
    "be": "bdf1",
    "backward-euler": "bdf1",
    "backward_euler": "bdf1",
    "backwardeuler": "bdf1",
    "ida": "ida",
    "midpoint": "midpoint",
    "mid": "midpoint",
    "im": "midpoint",
    "imr": "midpoint",
    "implicit-midpoint": "midpoint",
    "implicit_midpoint": "midpoint",
    "implicitmidpoint": "midpoint",
    "scipy": "scipy",
    "python": "scipy",
    "python-scipy": "scipy",
    "jit": "jit",
    "python-jit": "jit",
    "jit-bdf1": "jit",
}

_ODE_ONLY_SOLVERS = {
    "rk4",
    "rk2",
    "heun",
    "sdirk2",
    "dirk",
    "radau",
}


def normalize_dae_solver_name(solver: str) -> str:
    """Normalize a DAE solver token to one of: bdf1, ida, midpoint."""
    token = str(solver).strip().lower()
    if token in _DAE_SOLVER_ALIASES:
        return _DAE_SOLVER_ALIASES[token]
    if token in _ODE_ONLY_SOLVERS:
        raise ValueError(
            f"Solver '{solver}' is ODE-only. "
            f"For DAE use one of: {', '.join(DAE_SOLVER_LABELS)}. "
            f"Use tools/run_simulation.py for ODE methods."
        )
    raise ValueError(
        f"Unknown solver '{solver}'. "
        f"Supported DAE solvers: {', '.join(DAE_SOLVER_LABELS)}."
    )


def get_dae_solver_label(solver: str) -> str:
    """Return human-readable label for a DAE solver token."""
    canonical = normalize_dae_solver_name(solver)
    return DAE_SOLVER_LABELS[canonical]


def _port_names(comp: PowerComponent, direction: str) -> List[str]:
    return [p[0] for p in comp.port_schema[direction]]


def _fmt_cpp_double(v: float) -> str:
    """Format a float for C++ code, handling NaN and inf safely."""
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    return f"{v:.12f}"


class DiracCompiler:
    """Generate a C++ DAE kernel from a SystemGraph.

    Usage::

        dc = DiracCompiler("cases/SMIB/system.json")
        dc.build()
        cpp_code = dc.generate_cpp(dt=0.0005, duration=10.0)
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.graph = build_system_graph(json_path)

        # Use the existing SystemCompiler for structure + initialization
        self.base_compiler = SystemCompiler(json_path)

        # Mirror component list
        self.components: List[PowerComponent] = list(self.graph.components.values())
        self.comp_map: Dict[str, PowerComponent] = dict(self.graph.components)

        # State layout
        self.state_offsets: Dict[str, int] = {}
        self.n_diff = 0        # total differential states
        self.n_alg = 0         # total algebraic states (2 per bus)
        self.n_total = 0       # n_diff + n_alg

        # Network
        self.n_bus = 0
        self.bus_indices: List[int] = []
        self.bus_map: Dict[int, int] = {}
        self.Y_full: Optional[np.ndarray] = None  # full Y-bus (complex, n_bus × n_bus)

        # Slack bus info
        self.slack_bus_ids: List[int] = []
        self.slack_V: Dict[int, complex] = {}    # bus_id → V_ref phasor

        # Generator bus info
        self.gen_bus_map: Dict[str, int] = {}     # comp_name → bus_id

        # Wiring map (reuse from base compiler after build)
        self.wiring_map: Dict[Tuple[str, str], str] = {}

        # Bus fault events (populated by _parse_bus_faults)
        self.fault_events_dae: List[dict] = []

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def build(self):
        """Build the DAE structure: components, Y-bus, wiring.

        Must be called before ``generate_cpp()``.
        """
        # 1. Build base compiler structure (memory layout, Y-bus, wiring)
        self.base_compiler.build_structure()

        # Mirror state offsets and component ordering from base compiler
        self.components = self.base_compiler.components
        self.comp_map = self.base_compiler.comp_map
        self.state_offsets = dict(self.base_compiler.state_offsets)

        # Count differential states (component states + delta_COI)
        self.n_diff = self.base_compiler.total_states
        self.delta_coi_idx = self.base_compiler.delta_coi_idx

        # 2. Build FULL Y-bus (no Kron reduction, no generator impedance)
        self._build_full_ybus()

        # 3. Read frequency-dependent load parameters
        #    Priority: per-bus kpf/kqf from ComplexLoad components,
        #    fallback to global kpf/kqf from config
        cfg = self.graph.config
        global_kpf = float(cfg.get('kpf', 0.0))
        global_kqf = float(cfg.get('kqf', 0.0))
        if self._has_complex_loads:
            # Per-bus kpf/kqf already populated from ComplexLoad components
            self.kpf = 1.0  # flag: use per-bus arrays
            self.kqf = 1.0
            has_any = np.any(self.load_kpf != 0.0) or np.any(self.load_kqf != 0.0)
            if has_any:
                print(f"  [DAE] Frequency-dependent loads from ComplexLoad components")
                for i in range(self.n_bus):
                    if self.load_kpf[i] != 0.0 or self.load_kqf[i] != 0.0:
                        bus_id = self.bus_indices[i]
                        print(f"        Bus {bus_id}: kpf={self.load_kpf[i]:.3f}, kqf={self.load_kqf[i]:.3f}")
        else:
            # Global kpf/kqf from config (backward compatible)
            self.kpf = global_kpf
            self.kqf = global_kqf
            # Fill per-bus arrays with global values
            self.load_kpf = np.full(self.n_bus, global_kpf)
            self.load_kqf = np.full(self.n_bus, global_kqf)
            if global_kpf != 0.0 or global_kqf != 0.0:
                print(f"  [DAE] Frequency-dependent loads (global): kpf={global_kpf}, kqf={global_kqf}")

        # 4. Set up algebraic variable mapping
        self.n_alg = 2 * self.n_bus   # Vd, Vq for each bus
        self.n_total = self.n_diff + self.n_alg

        # 5. Resolve wiring — copy from base, then remap __BUS_ placeholders
        #    to full-bus indices (no Kron reduction).
        self.wiring_map = dict(self.base_compiler.wiring_map)
        self._resolve_bus_placeholders()

        # 5. Refresh control parameters (Vref, Pref, etc.)
        self._refresh_control_params()

        # 6. Identify generators and slack buses
        self._identify_generators()
        self._identify_slack_buses()

        # 7. Parse bus fault events
        self._parse_bus_faults()

    def _parse_bus_faults(self):
        """Parse BusFault events and compute fault shunt admittances.

        During a fault, a large shunt admittance Y_fault = 1/(r+jx) is added
        at the faulted bus.  The KCL constraint becomes:

            I_inj − (Y_bus + Y_fault_diag) · V = 0

        This smoothly drives the bus voltage toward zero for a bolted fault.
        """
        bus_faults = self.base_compiler.data.get('BusFault', [])
        self.fault_events_dae = []

        for bf in bus_faults:
            bus_id = int(bf['bus'])
            t_start = float(bf['t_start'])
            if 't_duration' in bf:
                t_end = t_start + float(bf['t_duration'])
            else:
                t_end = float(bf.get('t_end', t_start + 0.1))

            r = float(bf.get('r', 0.0))
            x = float(bf.get('x', 0.0001))  # small impedance for bolted fault

            # Y_fault = 1 / (r + jx) = (r - jx) / (r² + x²)
            z_sq = r * r + x * x
            if z_sq < 1e-20:
                z_sq = 1e-20
            g_fault = r / z_sq
            b_fault = -x / z_sq     # negative imaginary

            bus_idx = self.bus_map.get(bus_id, -1)
            if bus_idx >= 0:
                self.fault_events_dae.append({
                    'bus_id': bus_id,
                    'bus_idx': bus_idx,
                    't_start': t_start,
                    't_end': t_end,
                    'g': g_fault,
                    'b': b_fault,
                })
                print(f"  [DAE] Bus fault: bus {bus_id} (idx {bus_idx})  "
                      f"t=[{t_start:.4f}, {t_end:.4f}]  "
                      f"Y_fault={g_fault:.2f}{b_fault:+.2f}j")

    def _build_full_ybus(self):
        """Build the full Y-bus including loads and generator admittances.

        Unlike the existing pipeline, we do NOT Kron-reduce.  All buses
        remain as algebraic variables in the DAE.

        If ComplexLoad components are present, their per-load kpf/kqf
        parameters are used instead of a global config value.
        """
        data = self.base_compiler.data
        self.bus_indices = sorted([b['idx'] for b in data.get('Bus', [])])
        self.n_bus = len(self.bus_indices)
        self.bus_map = {b_idx: i for i, b_idx in enumerate(self.bus_indices)}

        # Build Y-bus from scratch (lines + shunts + loads)
        yb = YBusBuilder(data)
        yb.build(include_loads=True)

        # Add generator Norton admittances (sub-transient impedance)
        for comp in self.components:
            if comp.component_role != 'generator':
                continue
            if not comp.contributes_norton_admittance:
                continue
            bus_id = comp.params.get('bus')
            ra = comp.params.get('ra', 0.0)
            xd_pp = comp.params.get('xd_double_prime',
                                    comp.params.get('xd1', 0.2))
            yb.add_generator_impedance(bus_id, ra, xd_pp)

        self.Y_full = yb.Y.copy()

        # ----------------------------------------------------------
        # Per-bus load admittance for frequency-dependent loads
        # G_load[i] = P0/V0²,  B_load[i] = -Q0/V0²
        # Also collect per-bus kpf/kqf from ComplexLoad components
        # ----------------------------------------------------------
        from collections import defaultdict
        pq_list = data.get('PQ', [])
        bus_data_map = {b['idx']: b for b in data.get('Bus', [])}
        bus_load_p = defaultdict(float)
        bus_load_q = defaultdict(float)
        for pq in pq_list:
            bus = pq['bus']
            if bus not in self.bus_map:
                continue
            bus_load_p[bus] += pq.get('p0', 0.0)
            bus_load_q[bus] += pq.get('q0', 0.0)
        self.load_G = np.zeros(self.n_bus)
        self.load_B = np.zeros(self.n_bus)
        for bus, P in bus_load_p.items():
            Q = bus_load_q[bus]
            i = self.bus_map[bus]
            V0 = float(bus_data_map.get(bus, {}).get('v0', 1.0))
            V02 = max(V0 * V0, 1e-12)
            self.load_G[i] = P / V02
            self.load_B[i] = -Q / V02

        # Collect per-bus kpf/kqf from ComplexLoad components
        self.load_kpf = np.zeros(self.n_bus)
        self.load_kqf = np.zeros(self.n_bus)
        self._has_complex_loads = False
        for comp in self.components:
            if comp.component_role == 'load':
                self._has_complex_loads = True
                bus_id = comp.params.get('bus')
                if bus_id in self.bus_map:
                    i = self.bus_map[bus_id]
                    kpf = float(comp.params.get('kpf', 0.0))
                    kqf = float(comp.params.get('kqf', 0.0))
                    self.load_kpf[i] = kpf
                    self.load_kqf[i] = kqf
                    P0 = float(comp.params.get('P0', 0.0))
                    Q0 = float(comp.params.get('Q0', 0.0))
                    V0 = float(comp.params.get('V0', 1.0))
                    V02 = max(V0 * V0, 1e-12)
                    # Override load_G/B for this bus from the component
                    if P0 != 0.0 or Q0 != 0.0:
                        self.load_G[i] = P0 / V02
                        self.load_B[i] = -Q0 / V02

    def _resolve_bus_placeholders(self):
        """Resolve ``__BUS_<id>_<signal>`` placeholders to full-bus arrays.

        In the ODE compiler these map to Kron-reduced indices.  Here they
        map to the full bus indices: ``Vd_net[i]``, ``Vq_net[i]``,
        ``Vterm_net[i]`` where *i* is the position in our ``bus_map``.
        """
        for key, val in list(self.wiring_map.items()):
            if isinstance(val, str) and val.startswith('__BUS_'):
                stripped = val[len('__BUS_'):]
                last_us = stripped.rfind('_')
                bus_id = int(stripped[:last_us])
                sig = stripped[last_us + 1:]
                if bus_id in self.bus_map:
                    idx = self.bus_map[bus_id]
                    if sig == 'Vd':
                        self.wiring_map[key] = f"Vd_net[{idx}]"
                    elif sig == 'Vq':
                        self.wiring_map[key] = f"Vq_net[{idx}]"
                    elif sig == 'Vterm':
                        self.wiring_map[key] = f"Vterm_net[{idx}]"
                    else:
                        self.wiring_map[key] = "0.0"
                else:
                    self.wiring_map[key] = "0.0"

    def _refresh_control_params(self):
        """Refresh Vref, Pref, and other control set-points in the wiring map.

        Same logic as ``SystemCompiler._refresh_control_params()`` but
        without the Kron-reduction bus index remapping (already done).
        """
        for comp in self.components:
            role = comp.component_role
            if role == 'exciter' and 'Vref' in [p[0] for p in comp.port_schema['in']]:
                if 'Vref' in comp.params:
                    self.wiring_map[(comp.name, 'Vref')] = self._emit_float(
                        comp.params['Vref'], comp.name, 'Vref')
            if role == 'governor' and 'Pref' in [p[0] for p in comp.port_schema['in']]:
                if 'Pref' in comp.params:
                    self.wiring_map[(comp.name, 'Pref')] = self._emit_float(
                        comp.params['Pref'], comp.name, 'Pref')

    @staticmethod
    def _emit_float(value, comp_name: str, field: str) -> str:
        """Format a Python float for safe C++ injection.

        Refuses non-finite values up front: ``str(float('nan'))`` is
        ``'nan'``, which in C++ ``<cmath>`` resolves to ``double nan(const
        char*)`` rather than a float literal, yielding a cryptic compile
        error far from the real cause.
        """
        fval = float(value)
        if not math.isfinite(fval):
            raise RuntimeError(
                f"Refusing to emit non-finite {field}={value!r} for "
                f"component {comp_name!r}. The initialization pipeline "
                f"produced a NaN/Inf set-point; fix the upstream init "
                f"rather than writing it into generated C++."
            )
        return repr(fval)

    def _identify_generators(self):
        """Map generator component names to bus IDs."""
        for comp in self.components:
            if comp.component_role == 'generator':
                bus_id = comp.params.get('bus')
                if bus_id is not None:
                    self.gen_bus_map[comp.name] = int(bus_id)

    def _identify_slack_buses(self):
        """Identify slack buses and their reference voltages."""
        data = self.base_compiler.data
        self.slack_bus_ids = []
        self.slack_V = {}
        for sl in data.get('Slack', []):
            bus_id = sl['bus']
            v0 = sl.get('v0', 1.0)
            a0 = sl.get('a0', 0.0)
            self.slack_bus_ids.append(bus_id)
            self.slack_V[bus_id] = v0 * np.exp(1j * a0)

    # ------------------------------------------------------------------
    # C++ code generation
    # ------------------------------------------------------------------

    def _oos_check_snippet(self, varname: str = 'y') -> str:
        """Return C++ snippet that detects out-of-step (pole-slip) instability.

        Checks:
          1. Any |delta_i - delta_j| > π  (angle spread > 180°) → pole-slip
          2. Any Vterm > 5 pu or NaN      → numerical blowup
        Uses generator delta state indices from self.state_offsets.
        """
        gen_comps = [c for c in self.components
                     if c.component_role == 'generator'
                     and 'delta' in c.state_schema]
        if len(gen_comps) < 2:
            # Single generator: only voltage check
            return (
                'if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {\n'
                '    std::cout << "Stability limit reached. Stopping." << std::endl;\n'
                '    aborted = true;\n'
                '    break;\n'
                '}'
            )
        delta_indices = [self.state_offsets[c.name] for c in gen_comps]
        v = varname
        lines = []
        lines.append('// --- Out-of-step detection ---')
        lines.append('double _oos_delta_min = {}[{}];'.format(v, delta_indices[0]))
        lines.append('double _oos_delta_max = {}[{}];'.format(v, delta_indices[0]))
        for idx in delta_indices[1:]:
            lines.append('if ({v}[{i}] < _oos_delta_min) _oos_delta_min = {v}[{i}];'.format(v=v, i=idx))
            lines.append('if ({v}[{i}] > _oos_delta_max) _oos_delta_max = {v}[{i}];'.format(v=v, i=idx))
        lines.append('double _oos_spread = _oos_delta_max - _oos_delta_min;')
        lines.append('bool _oos = (_oos_spread > M_PI);')
        lines.append('bool _blowup = (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0]));')
        lines.append('if (_oos || _blowup) {')
        lines.append('    if (_oos)')
        lines.append('        std::cout << "OUT-OF-STEP detected: max angle spread="')
        lines.append('                  << (_oos_spread * 180.0 / M_PI) << " deg. Stopping." << std::endl;')
        lines.append('    else')
        lines.append('        std::cout << "Stability limit reached. Stopping." << std::endl;')
        lines.append('    aborted = true;')
        lines.append('    break;')
        lines.append('}')
        return '\n'.join(lines)

    def generate_cpp(self, dt: float = 0.0005, duration: float = 10.0,
                     x0: np.ndarray = None,
                     Vd_init: np.ndarray = None,
                     Vq_init: np.ndarray = None,
                     solver: str = "bdf1") -> str:
        """Generate the complete C++ source file.

        Parameters
        ----------
        dt : float
            Time step for the implicit solver.
        duration : float
            Total simulation time [s].
        x0 : ndarray, optional
            Initial differential state vector.
        Vd_init, Vq_init : ndarray, optional
            Initial bus voltages (all buses, ordered by bus_map).
        solver : str
            Solver backend: ``"bdf1"`` (built-in backward Euler + Newton)
            or ``"ida"`` (SUNDIALS IDA, variable-order BDF, adaptive step).

        Returns
        -------
        cpp_code : str
            Complete C++ source code (kernel + solver + main).
        """
        solver = normalize_dae_solver_name(solver)

        code = []
        if solver == "ida":
            code.append(self._emit_headers_ida())
        else:
            code.append(self._emit_headers())
        code.append(self._emit_constants())
        code.append(self._emit_ybus_arrays())
        code.append(self._emit_slack_constants())
        code.append(self._emit_fault_constants())
        code.append(self._emit_buffers())
        code.append(self._emit_component_functions())
        code.append(self._emit_dae_residual())
        if solver == "ida":
            code.append(self._emit_main_ida(dt, duration, x0, Vd_init, Vq_init))
        elif solver == "midpoint":
            code.append(self._emit_main_midpoint(dt, duration, x0, Vd_init, Vq_init))
        else:
            code.append(self._emit_main(dt, duration, x0, Vd_init, Vq_init))
        return "\n".join(code)

    def _emit_headers(self) -> str:
        return """#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
"""

    def _emit_headers_ida(self) -> str:
        return """#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

// SUNDIALS IDA
#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
"""

    def _emit_constants(self) -> str:
        lines = []
        lines.append(f"const int N_DIFF  = {self.n_diff};")
        lines.append(f"const int N_BUS   = {self.n_bus};")
        lines.append(f"const int N_ALG   = {self.n_alg};")
        lines.append(f"const int N_TOTAL = {self.n_total};")
        lines.append("")
        # Frequency-dependent load constants (per-bus arrays)
        lines.append(f"const double LOAD_G[{self.n_bus}] = {{ {', '.join(f'{v:.10e}' for v in self.load_G)} }};")
        lines.append(f"const double LOAD_B[{self.n_bus}] = {{ {', '.join(f'{v:.10e}' for v in self.load_B)} }};")
        lines.append(f"const double LOAD_KPF[{self.n_bus}] = {{ {', '.join(f'{v:.10e}' for v in self.load_kpf)} }};")
        lines.append(f"const double LOAD_KQF[{self.n_bus}] = {{ {', '.join(f'{v:.10e}' for v in self.load_kqf)} }};")
        lines.append("")
        return "\n".join(lines)

    def _emit_ybus_arrays(self) -> str:
        """Emit the full Y-bus as constant arrays (real and imaginary parts)."""
        lines = []
        Y = self.Y_full
        n = self.n_bus

        lines.append(f"// Full Y-bus ({n}x{n}) — NO Kron reduction")
        lines.append(f"const double Y_real[{n * n}] = {{")
        lines.append(f"    {', '.join(f'{v:.10e}' for v in Y.real.flatten())}")
        lines.append("};")
        lines.append(f"const double Y_imag[{n * n}] = {{")
        lines.append(f"    {', '.join(f'{v:.10e}' for v in Y.imag.flatten())}")
        lines.append("};")
        lines.append("")
        return "\n".join(lines)

    def _emit_slack_constants(self) -> str:
        """Emit slack bus reference voltage constants.

        A slack bus that has a dynamic generator (GENROU etc.) connected
        is treated with KCL during transient simulation — **not** as a
        fixed voltage source.  Only slack buses *without* a dynamic
        generator are kept as infinite-bus voltage constraints (e.g. the
        infinite bus in an SMIB system).
        """
        lines = []
        lines.append("// Slack bus configuration")

        # Buses that have a dynamic generator model
        gen_bus_ids = set(self.gen_bus_map.values())

        # Build arrays: is_slack[bus], Vd_ref[bus], Vq_ref[bus]
        is_slack = [0] * self.n_bus
        Vd_ref = [0.0] * self.n_bus
        Vq_ref = [0.0] * self.n_bus

        for bus_id, V_ref in self.slack_V.items():
            if bus_id in self.bus_map:
                # Only treat as infinite bus if NO dynamic generator
                if bus_id in gen_bus_ids:
                    continue   # use KCL — generator provides dynamics
                i = self.bus_map[bus_id]
                is_slack[i] = 1
                Vd_ref[i] = V_ref.real
                Vq_ref[i] = V_ref.imag

        lines.append(f"const int IS_SLACK[{self.n_bus}] = "
                      f"{{ {', '.join(str(s) for s in is_slack)} }};")
        lines.append(f"const double Vd_slack_ref[{self.n_bus}] = "
                      f"{{ {', '.join(f'{v:.10e}' for v in Vd_ref)} }};")
        lines.append(f"const double Vq_slack_ref[{self.n_bus}] = "
                      f"{{ {', '.join(f'{v:.10e}' for v in Vq_ref)} }};")
        lines.append("")
        return "\n".join(lines)

    def _emit_fault_constants(self) -> str:
        """Emit bus fault event arrays (shunt admittance, time window)."""
        faults = self.fault_events_dae
        n = len(faults)
        lines = []
        lines.append("// Bus fault events")
        lines.append(f"const int N_FAULTS = {n};")
        if n > 0:
            bus_vals = ', '.join(str(ev['bus_idx']) for ev in faults)
            ts_vals = ', '.join(f"{ev['t_start']:.10e}" for ev in faults)
            te_vals = ', '.join(f"{ev['t_end']:.10e}" for ev in faults)
            g_vals = ', '.join(f"{ev['g']:.10e}" for ev in faults)
            b_vals = ', '.join(f"{ev['b']:.10e}" for ev in faults)
            lines.append(f"const int FAULT_BUS[{n}] = {{ {bus_vals} }};")
            lines.append(f"const double FAULT_T_START[{n}] = {{ {ts_vals} }};")
            lines.append(f"const double FAULT_T_END[{n}] = {{ {te_vals} }};")
            lines.append(f"const double FAULT_G[{n}] = {{ {g_vals} }};")
            lines.append(f"const double FAULT_B[{n}] = {{ {b_vals} }};")
        else:
            # Emit dummy arrays so the residual code compiles even with no faults
            lines.append("const int FAULT_BUS[1] = {0};")
            lines.append("const double FAULT_T_START[1] = {0.0};")
            lines.append("const double FAULT_T_END[1] = {0.0};")
            lines.append("const double FAULT_G[1] = {0.0};")
            lines.append("const double FAULT_B[1] = {0.0};")
        # Mutable flag array: set by the integration loop at each segment boundary.
        # The residual checks this flag instead of comparing against time,
        # which avoids a residual discontinuity at the exact segment boundary.
        sz = max(n, 1)
        lines.append(f"static int fault_active[{sz}] = {{0}};")
        lines.append("")
        return "\n".join(lines)

    def _emit_buffers(self) -> str:
        """Emit global output/input buffers for components."""
        lines = []
        for comp in self.components:
            n_out = len(comp.port_schema['out'])
            if n_out > 0:
                lines.append(f"double outputs_{comp.name}[{n_out}];")
            n_in = len(comp.port_schema['in'])
            if n_in > 0:
                lines.append(f"double inputs_{comp.name}[{n_in}];")
        lines.append("")
        return "\n".join(lines)

    def _emit_component_functions(self) -> str:
        """Emit per-component step and output functions (same as existing compiler)."""
        lines = []
        for comp in self.components:
            lines.append(self._generate_instance_step(comp))
            lines.append(self._generate_instance_output(comp))
        return "\n".join(lines)

    def _generate_instance_step(self, comp: PowerComponent) -> str:
        lines = []
        func_name = f"step_{comp.name}"
        lines.append(f"void {func_name}(const double* x, double* dxdt, "
                      f"const double* inputs, double* outputs, double t) {{")
        for p_name, p_val in comp.params.items():
            if p_name in comp.param_schema:
                if isinstance(p_val, (int, float)):
                    if not math.isnan(p_val):
                        lines.append(f"    const double {p_name} = {p_val};")
                elif isinstance(p_val, str):
                    if "," in p_val:
                        lines.append(f"    const double {p_name}[] = {{{p_val}}};")
                    else:
                        lines.append(f"    const double {p_name} = {p_val};")
        lines.append("    // --- Kernel ---")
        lines.append(comp.get_cpp_step_code())
        lines.append("}")
        return "\n".join(lines)

    def _generate_instance_output(self, comp: PowerComponent) -> str:
        lines = []
        func_name = f"step_{comp.name}_out"
        lines.append(f"void {func_name}(const double* x, const double* inputs, "
                      f"double* outputs, double t) {{")
        for p_name, p_val in comp.params.items():
            if p_name in comp.param_schema:
                if isinstance(p_val, (int, float)):
                    if not math.isnan(p_val):
                        lines.append(f"    const double {p_name} = {p_val};")
                elif isinstance(p_val, str):
                    if "," in p_val:
                        lines.append(f"    const double {p_name}[] = {{{p_val}}};")
                    else:
                        lines.append(f"    const double {p_name} = {p_val};")
        lines.append("    // --- Kernel ---")
        lines.append(comp.get_cpp_compute_outputs_code())
        lines.append("}")
        return "\n".join(lines)

    def _generate_input_gathering_dae(self, comp: PowerComponent) -> str:
        """Generate input-gathering code using DAE bus voltages.

        The key difference from the ODE compiler: bus voltages come from
        the algebraic part of the DAE state vector ``y[N_DIFF + 2*bus + 0/1]``
        rather than from separate ``Vd_net[]`` arrays.
        """
        lines = []
        n_in = len(comp.port_schema['in'])
        if n_in == 0:
            return ""

        for i, (p_name, _, _) in enumerate(comp.port_schema['in']):
            key = (comp.name, p_name)
            if key in self.wiring_map:
                val = self.wiring_map[key]
                # Translate bus voltage references to DAE state vector
                val = self._translate_bus_ref_to_dae(val)
                lines.append(f"        inputs_{comp.name}[{i}] = {val}; // {p_name}")
            else:
                lines.append(f"        inputs_{comp.name}[{i}] = 0.0; // UNWIRED {p_name}")
        return "\n".join(lines)

    def _translate_bus_ref_to_dae(self, expr: str) -> str:
        """Translate Vd_net[k], Vq_net[k], Vterm_net[k] references
        to DAE state vector y[N_DIFF + 2*bus + offset]."""
        import re

        # Vd_net[k] → y[N_DIFF + 2*k]
        expr = re.sub(
            r'Vd_net\[(\d+)\]',
            lambda m: f'Vd_net[{m.group(1)}]',
            expr
        )
        # Vq_net[k] → y[N_DIFF + 2*k + 1]
        expr = re.sub(
            r'Vq_net\[(\d+)\]',
            lambda m: f'Vq_net[{m.group(1)}]',
            expr
        )
        # Vterm_net[k] → sqrt(Vd² + Vq²)
        expr = re.sub(
            r'Vterm_net\[(\d+)\]',
            lambda m: f'Vterm_net[{m.group(1)}]',
            expr
        )
        return expr

    def _emit_dae_residual(self) -> str:
        """Generate the dae_residual() C++ function.

        Computes F(t, y, ẏ) where:
          - y = [x_diff; Vd0; Vq0; Vd1; Vq1; ...]
          - Differential residuals: F[i] = ẏ[i] − f_i(x, V)
          - Algebraic residuals (non-slack): F[n+2i] = KCL_d, F[n+2i+1] = KCL_q
          - Algebraic residuals (slack): F[n+2i] = Vd − Vd_ref, F[n+2i+1] = Vq − Vq_ref
        """
        na = self.n_bus
        code = []

        code.append("void dae_residual(const double* y, const double* ydot, "
                     "double* res, double t) {")
        code.append("    // --- Extract state partitions ---")
        code.append("    const double* x = y;")
        code.append(f"    double Vd_net[{na}], Vq_net[{na}], Vterm_net[{na}];")
        code.append(f"    for (int i = 0; i < N_BUS; ++i) {{")
        code.append(f"        Vd_net[i]   = y[N_DIFF + 2*i];")
        code.append(f"        Vq_net[i]   = y[N_DIFF + 2*i + 1];")
        code.append(f"        Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + "
                     f"Vq_net[i]*Vq_net[i]);")
        code.append(f"    }}")
        code.append("")

        # Pre-compute dq-frame voltages for generators that need them
        gen_comps_for_dq = [
            comp for comp in self.components
            if comp.component_role == 'generator'
            and not getattr(comp, 'uses_ri_frame', False)
            and 'id_dq' in _port_names(comp, 'out')
            and comp.params.get('bus') in self.bus_map
        ]

        if gen_comps_for_dq:
            code.append("    // dq-frame voltages for exciter inputs")
            for comp in gen_comps_for_dq:
                off = self.state_offsets[comp.name]
                bus_id = comp.params['bus']
                bus_idx = self.bus_map[bus_id]
                code.append(f"    double vd_dq_{comp.name} = "
                            f"Vd_net[{bus_idx}]*sin(x[{off}]) - "
                            f"Vq_net[{bus_idx}]*cos(x[{off}]);")
                code.append(f"    double vq_dq_{comp.name} = "
                            f"Vd_net[{bus_idx}]*cos(x[{off}]) + "
                            f"Vq_net[{bus_idx}]*sin(x[{off}]);")
            code.append("")

        # Current injection accumulators
        code.append(f"    double Id_inj[{na}] = {{0}};")
        code.append(f"    double Iq_inj[{na}] = {{0}};")
        code.append("")

        # 1. Compute outputs and gather Norton current injections
        code.append("    // --- 1. Compute Outputs & Gather Injections ---")
        for comp in self.components:
            in_def = self._generate_input_gathering_dae(comp)
            n_in = len(comp.port_schema['in'])
            in_buf = f"inputs_{comp.name}" if n_in > 0 else "nullptr"
            code.append(f"    {{ // {comp.name}")
            if in_def:
                code.append(in_def)
            code.append(f"        step_{comp.name}_out(&x[{self.state_offsets[comp.name]}], "
                        f"{in_buf}, outputs_{comp.name}, t);")

            # Gather Norton current injections
            bus_id = comp.params.get('bus')
            if bus_id is None and comp.component_role == 'renewable_controller':
                gen_name = comp.get_associated_generator(self.comp_map)
                gen_comp = self.comp_map.get(gen_name) if gen_name else None
                if gen_comp is not None:
                    bus_id = gen_comp.params.get('bus')

            if bus_id in self.bus_map:
                bus_idx = self.bus_map[bus_id]
                out_names = _port_names(comp, 'out')
                if 'Id' in out_names and 'Iq' in out_names:
                    id_idx = out_names.index('Id')
                    iq_idx = out_names.index('Iq')
                    code.append(f"        Id_inj[{bus_idx}] += "
                                f"outputs_{comp.name}[{id_idx}];")
                    code.append(f"        Iq_inj[{bus_idx}] += "
                                f"outputs_{comp.name}[{iq_idx}];")
            code.append(f"    }}")
        code.append("")

        # Refresh dq-frame stator currents for exciter inputs
        if gen_comps_for_dq:
            code.append("    // Refresh actual dq-frame stator currents")
            for comp in gen_comps_for_dq:
                off = self.state_offsets[comp.name]
                bus_id = comp.params['bus']
                bus_idx = self.bus_map[bus_id]
                ra = float(comp.params.get('ra', 0.0))
                xd_pp = float(comp.params.get('xd_double_prime',
                              comp.params.get('xd1', 0.2)))
                xq_pp = float(comp.params.get('xq_double_prime', xd_pp))
                xd_p = float(comp.params.get('xd_prime',
                             comp.params.get('xd1', 0.3)))
                xq_p = float(comp.params.get('xq_prime',
                             comp.params.get('xq1', 0.3)))
                xl = float(comp.params.get('xl', 0.0))
                det = ra * ra + xd_pp * xq_pp
                kd = (xd_pp - xl) / (xd_p - xl) if (xd_p - xl) != 0 else 1.0
                kq = (xq_pp - xl) / (xq_p - xl) if (xq_p - xl) != 0 else 1.0
                out_names = _port_names(comp, 'out')
                id_dq_idx = out_names.index('id_dq')
                iq_dq_idx = out_names.index('iq_dq')

                # Resolve state indices from schema (supports both
                # legacy GenRou and GenRouPHS state orderings)
                schema = comp.state_schema
                idx_Eq = schema.index('E_q_prime')
                idx_pd = schema.index('psi_d')
                idx_Ed = schema.index('E_d_prime')
                idx_pq = schema.index('psi_q')

                # Helper to wrap negative literals in parentheses for C++
                def _cpp(v): return f"({v:.6f})" if v < 0 else f"{v:.6f}"

                code.append(f"    {{")
                code.append(f"        double psi_d_pp = x[{off+idx_Eq}]*{_cpp(kd)} + "
                            f"x[{off+idx_pd}]*(1.0-{_cpp(kd)});")
                code.append(f"        double psi_q_pp = -x[{off+idx_Ed}]*{_cpp(kq)} + "
                            f"x[{off+idx_pq}]*(1.0-{_cpp(kq)});")
                code.append(f"        double rhs_d = vd_dq_{comp.name} + psi_q_pp;")
                code.append(f"        double rhs_q = vq_dq_{comp.name} - psi_d_pp;")
                code.append(f"        outputs_{comp.name}[{id_dq_idx}] = "
                            f"({_cpp(-ra)}*rhs_d - {_cpp(xq_pp)}*rhs_q) / {det:.6f};")
                code.append(f"        outputs_{comp.name}[{iq_dq_idx}] = "
                            f"({_cpp(xd_pp)}*rhs_d + {_cpp(-ra)}*rhs_q) / {det:.6f};")
                code.append(f"    }}")
            code.append("")

        # 2. Compute dynamics (same as existing compiler)
        code.append("    // --- 2. Compute Dynamics (dxdt) ---")
        code.append(f"    double dxdt[{self.n_diff}];")
        for comp in self.components:
            in_def = self._generate_input_gathering_dae(comp)
            n_in = len(comp.port_schema['in'])
            in_buf = f"inputs_{comp.name}" if n_in > 0 else "nullptr"
            code.append(f"    {{ // {comp.name} dynamics")
            if in_def:
                code.append(in_def)
            code.append(f"        step_{comp.name}(&x[{self.state_offsets[comp.name]}], "
                        f"&dxdt[{self.state_offsets[comp.name]}], "
                        f"{in_buf}, outputs_{comp.name}, t);")
            code.append(f"    }}")

        # COI correction (same as existing compiler)
        gen_comps = [(comp, self.state_offsets[comp.name])
                     for comp in self.components
                     if comp.component_role == 'generator'
                     and 'omega' in comp.state_schema]
        if len(gen_comps) > 1:
            code.append("\n    // COI Reference Frame Correction")
            total_2H_parts = []
            weighted_omega_parts = []
            delta_dot_refs = []
            for comp, off in gen_comps:
                H = float(comp.params.get('H', 3.0))
                total_2H_parts.append(f"{2.0 * H:.6f}")
                weighted_omega_parts.append(f"{2.0 * H:.6f} * x[{off + 1}]")
                delta_dot_refs.append((off, H))
            omega_b_val = gen_comps[0][0].params.get('omega_b', '2.0 * M_PI * 60.0')
            code.append(f"    double coi_total_2H = {' + '.join(total_2H_parts)};")
            code.append(f"    double coi_omega = ({' + '.join(weighted_omega_parts)}) "
                        f"/ coi_total_2H;")
            code.append(f"    double omega_b_sys = {omega_b_val};")
            for delta_idx, _ in delta_dot_refs:
                code.append(f"    dxdt[{delta_idx}] -= omega_b_sys * "
                            f"(coi_omega - 1.0);")
            code.append(f"    dxdt[{self.delta_coi_idx}] = omega_b_sys * "
                        f"(coi_omega - 1.0);")
        else:
            # Single or no generator: still need coi_omega for freq-dep loads
            if gen_comps:
                comp0, off0 = gen_comps[0]
                code.append(f"    double coi_omega = x[{off0 + 1}];")
            else:
                code.append(f"    double coi_omega = 1.0;")
            code.append(f"    dxdt[{self.delta_coi_idx}] = 0.0;")
        code.append("")

        # 3. Differential residuals: res[i] = ydot[i] − dxdt[i]
        code.append("    // --- 3. Differential Residuals ---")
        code.append(f"    for (int i = 0; i < N_DIFF; ++i)")
        code.append(f"        res[i] = ydot[i] - dxdt[i];")
        code.append("")

        # 4. Compute per-bus fault shunt admittance (active during fault)
        code.append("    // --- Fault shunt admittance (segment-based flag) ---")
        code.append(f"    double Yf_g[{na}] = {{0}};")
        code.append(f"    double Yf_b[{na}] = {{0}};")
        code.append(f"    for (int f = 0; f < N_FAULTS; ++f) {{")
        code.append(f"        if (fault_active[f]) {{")
        code.append(f"            Yf_g[FAULT_BUS[f]] += FAULT_G[f];")
        code.append(f"            Yf_b[FAULT_BUS[f]] += FAULT_B[f];")
        code.append(f"        }}")
        code.append(f"    }}")
        code.append("")

        # 5. Algebraic residuals: KCL or voltage constraint
        code.append("    // --- 5. Algebraic Residuals (KCL / Slack) ---")
        code.append(f"    for (int i = 0; i < N_BUS; ++i) {{")
        code.append(f"        if (IS_SLACK[i]) {{")
        code.append(f"            // Slack bus: V = V_ref (voltage source)")
        code.append(f"            res[N_DIFF + 2*i]     = Vd_net[i] - Vd_slack_ref[i];")
        code.append(f"            res[N_DIFF + 2*i + 1] = Vq_net[i] - Vq_slack_ref[i];")
        code.append(f"        }} else {{")
        code.append(f"            // KCL: I_inj − (Y_bus + Y_fault) · V = 0")
        code.append(f"            double Id_ybus = 0.0, Iq_ybus = 0.0;")
        code.append(f"            for (int j = 0; j < N_BUS; ++j) {{")
        code.append(f"                double G = Y_real[i*N_BUS + j];")
        code.append(f"                double B = Y_imag[i*N_BUS + j];")
        code.append(f"                Id_ybus += G*Vd_net[j] - B*Vq_net[j];")
        code.append(f"                Iq_ybus += G*Vq_net[j] + B*Vd_net[j];")
        code.append(f"            }}")
        code.append(f"            // Add fault shunt current: I_fault = Y_fault · V_local")
        code.append(f"            Id_ybus += Yf_g[i]*Vd_net[i] - Yf_b[i]*Vq_net[i];")
        code.append(f"            Iq_ybus += Yf_g[i]*Vq_net[i] + Yf_b[i]*Vd_net[i];")
        code.append(f"            // Frequency-dependent load: extra current = kpf*dw*G_load*V (P) + kqf*dw*B_load*V (Q)")
        code.append(f"            if (LOAD_KPF[i] != 0.0 || LOAD_KQF[i] != 0.0) {{")
        code.append(f"                double dw = coi_omega - 1.0;")
        code.append(f"                double dP = LOAD_KPF[i] * dw * LOAD_G[i];")
        code.append(f"                double dQ = LOAD_KQF[i] * dw * LOAD_B[i];")
        code.append(f"                Id_ybus += dP*Vd_net[i] - dQ*Vq_net[i];")
        code.append(f"                Iq_ybus += dP*Vq_net[i] + dQ*Vd_net[i];")
        code.append(f"            }}")
        code.append(f"            res[N_DIFF + 2*i]     = Id_inj[i] - Id_ybus;")
        code.append(f"            res[N_DIFF + 2*i + 1] = Iq_inj[i] - Iq_ybus;")
        code.append(f"        }}")
        code.append(f"    }}")

        code.append("}")
        code.append("")
        return "\n".join(code)

    def _emit_main(self, dt: float, duration: float,
                   x0: np.ndarray = None,
                   Vd_init: np.ndarray = None,
                   Vq_init: np.ndarray = None) -> str:
        """Generate main() with BDF-1 solver."""

        # Default initial values (zeros) — real values injected by runner
        if x0 is None:
            x0 = np.zeros(self.n_diff)
        if Vd_init is None:
            Vd_init = np.ones(self.n_bus)
        if Vq_init is None:
            Vq_init = np.zeros(self.n_bus)

        # Build y0 = [x0; Vd0; Vq0; Vd1; Vq1; ...]
        y0 = np.zeros(self.n_total)
        y0[:self.n_diff] = x0
        for i in range(self.n_bus):
            y0[self.n_diff + 2 * i] = Vd_init[i]
            y0[self.n_diff + 2 * i + 1] = Vq_init[i]

        y0_str = ", ".join(_fmt_cpp_double(v) for v in y0)
        steps = int(duration / dt)

        # CSV columns
        csv_cols = self._build_csv_columns()
        header = ",".join(h for h, _ in csv_cols)
        log_parts = " << \",\" << ".join(f"({expr})" for _, expr in csv_cols)

        return f"""
// =================================================================
// Dense LU solver for Ax = b (in-place, small N)
// =================================================================
void lu_solve(double* A, double* b, int n) {{
    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n; ++k) {{
        // Find pivot
        int pivot = k;
        double pmax = fabs(A[k*n+k]);
        for (int i = k+1; i < n; ++i) {{
            if (fabs(A[i*n+k]) > pmax) {{ pmax = fabs(A[i*n+k]); pivot = i; }}
        }}
        // Swap rows in A and b
        if (pivot != k) {{
            for (int j = 0; j < n; ++j) {{
                double tmp = A[k*n+j]; A[k*n+j] = A[pivot*n+j]; A[pivot*n+j] = tmp;
            }}
            double tmp = b[k]; b[k] = b[pivot]; b[pivot] = tmp;
        }}
        // Eliminate below
        double akk = A[k*n+k];
        if (fabs(akk) < 1e-30) akk = 1e-30;
        for (int i = k+1; i < n; ++i) {{
            double factor = A[i*n+k] / akk;
            for (int j = k+1; j < n; ++j)
                A[i*n+j] -= factor * A[k*n+j];
            b[i] -= factor * b[k];
        }}
    }}
    // Back substitution
    for (int i = n-1; i >= 0; --i) {{
        double sum = b[i];
        for (int j = i+1; j < n; ++j)
            sum -= A[i*n+j] * b[j];
        double aii = A[i*n+i];
        if (fabs(aii) < 1e-30) aii = 1e-30;
        b[i] = sum / aii;
    }}
}}

// =================================================================
// BDF-1 (Backward Euler) DAE Solver with Full Newton Iteration
// =================================================================

void solve_bdf1(double* y, double dt, int n_steps) {{
    double y_old[N_TOTAL];
    double ydot[N_TOTAL];
    double res[N_TOTAL];
    double res_pert[N_TOTAL];
    double y_pert[N_TOTAL];
    double dy[N_TOTAL];
    double J[N_TOTAL * N_TOTAL];  // Dense Jacobian (column-major: J[i*N+j])

    const double newton_tol = 1e-8;
    const int max_newton = 20;
    const double eps_fd = 1e-7;

    // Output File
    std::ofstream outfile("simulation_results.csv");
    outfile << "{header}" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double t = 0.0;
    const int log_every = {max(1, int(0.01 / dt))};

    // Extract Vd/Vq arrays for logging
    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Initial diagnostics
    for (int i = 0; i < N_TOTAL; ++i) ydot[i] = 0.0;
    dae_residual(y, ydot, res, 0.0);
    double max_res = 0.0;
    int max_res_idx = -1;
    for (int i = 0; i < N_TOTAL; ++i) {{
        if (fabs(res[i]) > max_res) {{ max_res = fabs(res[i]); max_res_idx = i; }}
    }}
    std::cout << "[DAE] Initial max |residual| = " << max_res
              << " at index " << max_res_idx << std::endl;

    for (int step = 0; step < n_steps; ++step) {{
        // Update Vd/Vq for logging
        for (int i = 0; i < N_BUS; ++i) {{
            Vd_net[i]   = y[N_DIFF + 2*i];
            Vq_net[i]   = y[N_DIFF + 2*i + 1];
            Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
        }}

        // Log
        if (step % log_every == 0) {{
            const double* x = y;
            outfile << {log_parts} << std::endl;
        }}

        // Save current state
        for (int i = 0; i < N_TOTAL; ++i) y_old[i] = y[i];

        double t_new = t + dt;

        // Update fault flags based on current time
        for (int f = 0; f < N_FAULTS; ++f) {{
            fault_active[f] = (t_new >= FAULT_T_START[f]
                            && t_new <  FAULT_T_END[f]) ? 1 : 0;
        }}

        // Newton iteration to solve: G(y) = F(t_new, y, (y-y_old)/dt) = 0
        for (int nit = 0; nit < max_newton; ++nit) {{
            // Compute ydot = (y - y_old) / dt
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = (y[i] - y_old[i]) / dt;

            // Compute residual
            dae_residual(y, ydot, res, t_new);

            // Check convergence
            double res_norm = 0.0;
            for (int i = 0; i < N_TOTAL; ++i)
                res_norm += res[i] * res[i];
            res_norm = sqrt(res_norm);
            if (res_norm < newton_tol) break;

            // Build full Jacobian via finite differences: J[i][j] = dG_i/dy_j
            // G(y) = F(t_new, y, (y-y_old)/dt)
            // dG/dy_j = dF/dy_j + (1/dt) * dF/dydot_j  (for variable j)
            // We compute this numerically by perturbing y[j] and recomputing G.
            for (int j = 0; j < N_TOTAL; ++j) {{
                for (int k = 0; k < N_TOTAL; ++k) y_pert[k] = y[k];
                double h = eps_fd * (1.0 + fabs(y[j]));
                y_pert[j] += h;

                double ydot_pert[N_TOTAL];
                for (int k = 0; k < N_TOTAL; ++k)
                    ydot_pert[k] = (y_pert[k] - y_old[k]) / dt;

                dae_residual(y_pert, ydot_pert, res_pert, t_new);

                for (int i = 0; i < N_TOTAL; ++i)
                    J[i*N_TOTAL + j] = (res_pert[i] - res[i]) / h;
            }}

            // Solve J * dy = -res  →  dy = J^{{-1}} * (-res)
            for (int i = 0; i < N_TOTAL; ++i) dy[i] = -res[i];
            lu_solve(J, dy, N_TOTAL);

            // Line search with backtracking for robustness
            double alpha = 1.0;
            for (int ls = 0; ls < 5; ++ls) {{
                for (int i = 0; i < N_TOTAL; ++i)
                    y_pert[i] = y[i] + alpha * dy[i];
                double ydot_ls[N_TOTAL];
                for (int i = 0; i < N_TOTAL; ++i)
                    ydot_ls[i] = (y_pert[i] - y_old[i]) / dt;
                dae_residual(y_pert, ydot_ls, res_pert, t_new);
                double new_norm = 0.0;
                for (int i = 0; i < N_TOTAL; ++i)
                    new_norm += res_pert[i]*res_pert[i];
                new_norm = sqrt(new_norm);
                if (new_norm < res_norm) break;
                alpha *= 0.5;
            }}

            for (int i = 0; i < N_TOTAL; ++i)
                y[i] += alpha * dy[i];
        }}

        t = t_new;

        // Progress check every ~1 second of simulation time
        if (step % (int)(1.0 / dt) == 0) {{
            for (int i = 0; i < N_BUS; ++i) {{
                Vd_net[i]   = y[N_DIFF + 2*i];
                Vq_net[i]   = y[N_DIFF + 2*i + 1];
                Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
            }}

            // Compute residual norm for monitoring
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = (y[i] - y_old[i]) / dt;
            dae_residual(y, ydot, res, t);
            double rn = 0.0;
            for (int i = 0; i < N_TOTAL; ++i) rn += res[i]*res[i];
            std::cout << "t=" << t << " |res|=" << sqrt(rn)
                      << " Vterm[0]=" << Vterm_net[0] << std::endl;

            if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {{
                std::cout << "Stability limit reached. Stopping." << std::endl;
                break;
            }}
        }}
    }}

    outfile.close();
    std::cout << "Done. Results in simulation_results.csv" << std::endl;
}}

int main() {{
    double y[N_TOTAL] = {{ {y0_str} }};

    std::cout << "Dirac DAE Simulation" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  dt = {dt},  T = {duration} s" << std::endl;
    std::cout << std::endl;

    solve_bdf1(y, {dt}, {steps});

    return 0;
}}
"""

    def _emit_main_midpoint(self, dt: float, duration: float,
                            x0: np.ndarray = None,
                            Vd_init: np.ndarray = None,
                            Vq_init: np.ndarray = None) -> str:
        """Generate main() with structure-preserving implicit midpoint solver.

        The implicit midpoint rule preserves the Dirac / port-Hamiltonian
        structure of the DAE.  At each time step we solve:

            F( t + dt/2,  (y_old + y_new)/2,  (y_new - y_old)/dt ) = 0

        for ``y_new`` via Newton iteration with a full finite-difference
        Jacobian and dense LU factorisation (identical linear-algebra
        infrastructure as the BDF-1 solver).
        """

        if x0 is None:
            x0 = np.zeros(self.n_diff)
        if Vd_init is None:
            Vd_init = np.ones(self.n_bus)
        if Vq_init is None:
            Vq_init = np.zeros(self.n_bus)

        y0 = np.zeros(self.n_total)
        y0[:self.n_diff] = x0
        for i in range(self.n_bus):
            y0[self.n_diff + 2 * i] = Vd_init[i]
            y0[self.n_diff + 2 * i + 1] = Vq_init[i]

        y0_str = ", ".join(_fmt_cpp_double(v) for v in y0)
        steps = int(duration / dt)

        csv_cols = self._build_csv_columns()
        header = ",".join(h for h, _ in csv_cols)
        log_parts = " << \",\" << ".join(f"({expr})" for _, expr in csv_cols)

        return f"""
// =================================================================
// Dense LU solver for Ax = b (in-place, small N)
// =================================================================
void lu_solve(double* A, double* b, int n) {{
    for (int k = 0; k < n; ++k) {{
        int pivot = k;
        double pmax = fabs(A[k*n+k]);
        for (int i = k+1; i < n; ++i) {{
            if (fabs(A[i*n+k]) > pmax) {{ pmax = fabs(A[i*n+k]); pivot = i; }}
        }}
        if (pivot != k) {{
            for (int j = 0; j < n; ++j) {{
                double tmp = A[k*n+j]; A[k*n+j] = A[pivot*n+j]; A[pivot*n+j] = tmp;
            }}
            double tmp = b[k]; b[k] = b[pivot]; b[pivot] = tmp;
        }}
        double akk = A[k*n+k];
        if (fabs(akk) < 1e-30) akk = 1e-30;
        for (int i = k+1; i < n; ++i) {{
            double factor = A[i*n+k] / akk;
            for (int j = k+1; j < n; ++j)
                A[i*n+j] -= factor * A[k*n+j];
            b[i] -= factor * b[k];
        }}
    }}
    for (int i = n-1; i >= 0; --i) {{
        double sum = b[i];
        for (int j = i+1; j < n; ++j)
            sum -= A[i*n+j] * b[j];
        double aii = A[i*n+i];
        if (fabs(aii) < 1e-30) aii = 1e-30;
        b[i] = sum / aii;
    }}
}}

// =================================================================
// Implicit Midpoint DAE Solver  (structure-preserving)
//   Solve:  F( t+dt/2, (y_old+y_new)/2, (y_new-y_old)/dt ) = 0
// =================================================================

void solve_midpoint(double* y, double dt, int n_steps) {{
    double y_old[N_TOTAL];
    double y_mid[N_TOTAL];
    double ydot[N_TOTAL];
    double res[N_TOTAL];
    double res_pert[N_TOTAL];
    double y_pert[N_TOTAL];
    double y_mid_pert[N_TOTAL];
    double dy[N_TOTAL];
    double J[N_TOTAL * N_TOTAL];

    const double newton_tol = 1e-8;
    const int max_newton = 20;
    const double eps_fd = 1e-7;

    std::ofstream outfile("simulation_results.csv");
    outfile << "{header}" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double t = 0.0;
    const int log_every = {max(1, int(0.01 / dt))};

    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Initial diagnostics
    for (int i = 0; i < N_TOTAL; ++i) ydot[i] = 0.0;
    dae_residual(y, ydot, res, 0.0);
    double max_res = 0.0;
    int max_res_idx = -1;
    for (int i = 0; i < N_TOTAL; ++i) {{
        if (fabs(res[i]) > max_res) {{ max_res = fabs(res[i]); max_res_idx = i; }}
    }}
    std::cout << "[DAE-Midpoint] Initial max |residual| = " << max_res
              << " at index " << max_res_idx << std::endl;

    for (int step = 0; step < n_steps; ++step) {{
        // Update Vd/Vq for logging
        for (int i = 0; i < N_BUS; ++i) {{
            Vd_net[i]   = y[N_DIFF + 2*i];
            Vq_net[i]   = y[N_DIFF + 2*i + 1];
            Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
        }}

        if (step % log_every == 0) {{
            const double* x = y;
            outfile << {log_parts} << std::endl;
        }}

        // Save current state
        for (int i = 0; i < N_TOTAL; ++i) y_old[i] = y[i];

        double t_mid = t + 0.5 * dt;

        // Newton iteration:  G(y_new) = F(t_mid, (y_old+y_new)/2, (y_new-y_old)/dt) = 0
        for (int nit = 0; nit < max_newton; ++nit) {{
            // y_mid = (y_old + y_new) / 2
            for (int i = 0; i < N_TOTAL; ++i)
                y_mid[i] = 0.5 * (y_old[i] + y[i]);

            // ydot = (y_new - y_old) / dt
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = (y[i] - y_old[i]) / dt;

            // Evaluate residual at the midpoint
            dae_residual(y_mid, ydot, res, t_mid);

            double res_norm = 0.0;
            for (int i = 0; i < N_TOTAL; ++i)
                res_norm += res[i] * res[i];
            res_norm = sqrt(res_norm);
            if (res_norm < newton_tol) break;

            // Build Jacobian dG/dy_new[j]
            // Perturbing y_new[j] affects both y_mid (by 0.5) and ydot (by 1/dt).
            for (int j = 0; j < N_TOTAL; ++j) {{
                for (int k = 0; k < N_TOTAL; ++k) y_pert[k] = y[k];
                double h = eps_fd * (1.0 + fabs(y[j]));
                y_pert[j] += h;

                for (int k = 0; k < N_TOTAL; ++k)
                    y_mid_pert[k] = 0.5 * (y_old[k] + y_pert[k]);

                double ydot_pert[N_TOTAL];
                for (int k = 0; k < N_TOTAL; ++k)
                    ydot_pert[k] = (y_pert[k] - y_old[k]) / dt;

                dae_residual(y_mid_pert, ydot_pert, res_pert, t_mid);

                for (int i = 0; i < N_TOTAL; ++i)
                    J[i*N_TOTAL + j] = (res_pert[i] - res[i]) / h;
            }}

            // Solve J * dy = -res
            for (int i = 0; i < N_TOTAL; ++i) dy[i] = -res[i];
            lu_solve(J, dy, N_TOTAL);

            // Line search
            double alpha = 1.0;
            for (int ls = 0; ls < 5; ++ls) {{
                for (int i = 0; i < N_TOTAL; ++i)
                    y_pert[i] = y[i] + alpha * dy[i];

                for (int k = 0; k < N_TOTAL; ++k)
                    y_mid_pert[k] = 0.5 * (y_old[k] + y_pert[k]);

                double ydot_ls[N_TOTAL];
                for (int i = 0; i < N_TOTAL; ++i)
                    ydot_ls[i] = (y_pert[i] - y_old[i]) / dt;

                dae_residual(y_mid_pert, ydot_ls, res_pert, t_mid);
                double new_norm = 0.0;
                for (int i = 0; i < N_TOTAL; ++i)
                    new_norm += res_pert[i]*res_pert[i];
                new_norm = sqrt(new_norm);
                if (new_norm < res_norm) break;
                alpha *= 0.5;
            }}

            for (int i = 0; i < N_TOTAL; ++i)
                y[i] += alpha * dy[i];
        }}

        t += dt;

        // Progress monitoring
        if (step % (int)(1.0 / dt) == 0) {{
            for (int i = 0; i < N_BUS; ++i) {{
                Vd_net[i]   = y[N_DIFF + 2*i];
                Vq_net[i]   = y[N_DIFF + 2*i + 1];
                Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
            }}
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = 0.0;
            dae_residual(y, ydot, res, t);
            double rn = 0.0;
            for (int i = 0; i < N_TOTAL; ++i) rn += res[i]*res[i];
            std::cout << "t=" << t << " |res|=" << sqrt(rn)
                      << " Vterm[0]=" << Vterm_net[0] << std::endl;

            if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {{
                std::cout << "Stability limit reached. Stopping." << std::endl;
                break;
            }}
        }}
    }}

    outfile.close();
    std::cout << "Done. Results in simulation_results.csv" << std::endl;
}}

int main() {{
    double y[N_TOTAL] = {{ {y0_str} }};

    std::cout << "Dirac DAE Simulation (Implicit Midpoint)" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  dt = {dt},  T = {duration} s" << std::endl;
    std::cout << std::endl;

    solve_midpoint(y, {dt}, {steps});

    return 0;
}}
"""

    def _emit_main_ida(self, dt: float, duration: float,
                       x0: np.ndarray = None,
                       Vd_init: np.ndarray = None,
                       Vq_init: np.ndarray = None) -> str:
        """Generate main() with SUNDIALS IDA solver.

        Uses variable-order BDF (up to order 5) with adaptive time stepping.
        The existing dae_residual() function is wrapped in an IDA-compatible
        callback.  IDA provides its own Newton solver with dense direct linear
        algebra.

        Fault event discontinuities are handled by collecting all fault
        boundary times (t_start, t_start + t_duration) and breaking the
        integration into smooth segments.  IDA is reinitialized at each
        discontinuity with ``IDAReInit`` + ``IDACalcIC``.
        """
        # Default initial values
        if x0 is None:
            x0 = np.zeros(self.n_diff)
        if Vd_init is None:
            Vd_init = np.ones(self.n_bus)
        if Vq_init is None:
            Vq_init = np.zeros(self.n_bus)

        # Build y0 = [x0; Vd0; Vq0; Vd1; Vq1; ...]
        y0 = np.zeros(self.n_total)
        y0[:self.n_diff] = x0
        for i in range(self.n_bus):
            y0[self.n_diff + 2 * i] = Vd_init[i]
            y0[self.n_diff + 2 * i + 1] = Vq_init[i]

        y0_str = ", ".join(_fmt_cpp_double(v) for v in y0)

        # Build differential/algebraic ID vector: 1.0 = differential, 0.0 = algebraic
        var_id = []
        for i in range(self.n_total):
            if i < self.n_diff:
                var_id.append("1.0")
            else:
                var_id.append("0.0")
        var_id_str = ", ".join(var_id)

        # Collect fault event boundary times for IDA restart
        event_times = set()
        for ev in self.fault_events_dae:
            event_times.add(ev['t_start'])
            event_times.add(ev['t_end'])
        # Remove times outside [0, duration] and add duration
        event_times = sorted(t for t in event_times if 0 < t < duration)
        event_times.append(duration)
        n_segments = len(event_times)
        event_times_str = ", ".join(f"{t:.12f}" for t in event_times)

        # CSV columns
        csv_cols = self._build_csv_columns()
        header = ",".join(h for h, _ in csv_cols)
        log_parts = " << \",\" << ".join(f"({expr})" for _, expr in csv_cols)

        log_every_time = 0.01  # log every 10 ms of simulation time

        # ── IC torque-balance diagnostics for governors with xi state ──────────
        # After IDACalcIC adjusts algebraic bus voltages, yp[omega] reflects the
        # startup torque mismatch (Tm - Te_IDA) / (2H).
        # This block is diagnostic-only: it reports the residual dω/dt seen by
        # IDA after consistent-IC calculation, but does not modify the states.
        _bc = self.base_compiler
        _xi_lines: list = []
        for _gov in _bc.components:
            if _gov.component_role != 'governor':
                continue
            if 'xi' not in list(_gov.state_schema):
                continue
            _gen_name = _gov.params.get('syn')
            if not _gen_name:
                continue
            _gen = _bc.comp_map.get(_gen_name)
            if _gen is None or 'omega' not in list(_gen.state_schema):
                continue
            _H = float(_gen.params.get('H', 0.0))
            if _H <= 0.0:
                continue
            _omega_idx = _bc.state_offsets[_gen.name] + list(_gen.state_schema).index('omega')
            _xi_idx    = _bc.state_offsets[_gov.name] + list(_gov.state_schema).index('xi')
            _xi_lines.append(
                f'    // IC-diag: {_gov.name} residual torque mismatch of {_gen.name}')
            _xi_lines.append(
                f'    std::printf("[IC-diag] {_gov.name} torque_residual=%.6e  (omega_dot=%.6e)\\n",'
                f' yp_data[{_omega_idx}] * (2.0 * {_H}), yp_data[{_omega_idx}]);'
            )
        xi_diag_code = (
            "\n".join(_xi_lines)
            if _xi_lines else
            "    // (no governors with xi state — no IC torque-balance diagnostic needed)"
        )

        # ── IC Vm diagnostics for exciters ─────────────────────────────────────
        # After IDACalcIC shifts algebraic bus voltages, Vm (a differential
        # controller state initialised from Python-side Vt) can differ slightly
        # from the final algebraic terminal magnitude Vterm_IDA.
        # This block reports that residual; it does not modify the exciter state.
        _vm_lines: list = []
        for _exc in _bc.components:
            if _exc.component_role != 'exciter':
                continue
            if 'Vm' not in list(_exc.state_schema):
                continue
            _gen_name_exc = _exc.params.get('syn')
            if not _gen_name_exc:
                continue
            _gen_exc = _bc.comp_map.get(_gen_name_exc)
            if _gen_exc is None:
                continue
            _bus_id_exc = _gen_exc.params.get('bus')
            if _bus_id_exc is None:
                continue
            _bi_exc = self.bus_map.get(_bus_id_exc)
            if _bi_exc is None:
                continue
            _vm_state_idx = _bc.state_offsets[_exc.name] + list(_exc.state_schema).index('Vm')
            _vm_lines.append(
                f'    // IC-diag: {_exc.name}.Vm residual at bus {_bus_id_exc}')
            _vm_lines.append(
                f'    {{'
            )
            _vm_lines.append(
                f'        double Vd_ic = yy_data[N_DIFF + {2 * _bi_exc}];'
            )
            _vm_lines.append(
                f'        double Vq_ic = yy_data[N_DIFF + {2 * _bi_exc + 1}];'
            )
            _vm_lines.append(
                f'        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);'
            )
            _vm_lines.append(
                f'        std::printf("[IC-diag] {_exc.name}.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\\n",'
                f' yy_data[{_vm_state_idx}], Vt_ic, Vt_ic - yy_data[{_vm_state_idx}]);'
            )
            _vm_lines.append(
                f'    }}'
            )
        vm_diag_code = (
            "\n".join(_vm_lines)
            if _vm_lines else
            "    // (no exciter Vm diagnostic needed)"
        )

        return f"""
// =================================================================
// IDA residual wrapper — adapts our dae_residual to SUNDIALS signature
// =================================================================
int ida_residual(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr,
                 void* /*user_data*/) {{
    const double* y_data    = N_VGetArrayPointer(yy);
    const double* ydot_data = N_VGetArrayPointer(yp);
    double*       res_data  = N_VGetArrayPointer(rr);
    dae_residual(y_data, ydot_data, res_data, (double)t);
    return 0;
}}

// Helper: log one data point
static inline void log_state(std::ofstream& outfile, double t_val,
                              const double* yy_data,
                              double* Vd_net, double* Vq_net, double* Vterm_net) {{
    double t = t_val;
    const double* y = yy_data;
    const double* x = y;
    for (int i = 0; i < N_BUS; ++i) {{
        Vd_net[i]   = y[N_DIFF + 2*i];
        Vq_net[i]   = y[N_DIFF + 2*i + 1];
        Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
    }}
    outfile << {log_parts} << std::endl;
}}

// =================================================================
// SUNDIALS IDA solver (variable-order BDF, adaptive step)
//   with event-driven reinitialization at fault boundaries
// =================================================================
int main() {{
    // --- Context ---
    SUNContext sunctx = NULL;
    // SUN_COMM_NULL was introduced in SUNDIALS 7.0; on 6.x SUNContext_Create
    // takes void* and accepts NULL. Define it ourselves when missing so the
    // same generated code compiles against either ABI.
#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif
    int ierr = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (ierr) {{ std::cerr << "SUNContext_Create failed" << std::endl; return 1; }}

    // --- Initial state y0 ---
    N_Vector yy = N_VNew_Serial(N_TOTAL, sunctx);
    N_Vector yp = N_VNew_Serial(N_TOTAL, sunctx);
    sunrealtype* yy_data = N_VGetArrayPointer(yy);
    sunrealtype* yp_data = N_VGetArrayPointer(yp);

    double y0_arr[N_TOTAL] = {{ {y0_str} }};
    for (int i = 0; i < N_TOTAL; ++i) yy_data[i] = y0_arr[i];

    // --- Compute consistent initial yp from residual ---
    double res0[N_TOTAL];
    for (int i = 0; i < N_TOTAL; ++i) yp_data[i] = 0.0;
    dae_residual(yy_data, yp_data, res0, 0.0);
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
    for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;

    // --- Variable type ID (1 = differential, 0 = algebraic) ---
    N_Vector var_id = N_VNew_Serial(N_TOTAL, sunctx);
    sunrealtype* var_id_data = N_VGetArrayPointer(var_id);
    double var_id_arr[N_TOTAL] = {{ {var_id_str} }};
    for (int i = 0; i < N_TOTAL; ++i) var_id_data[i] = var_id_arr[i];

    // --- Dense linear solver (shared across reinits) ---
    SUNMatrix A = SUNDenseMatrix(N_TOTAL, N_TOTAL, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(yy, A, sunctx);

    // --- Create IDA solver ---
    void* ida_mem = IDACreate(sunctx);
    if (!ida_mem) {{ std::cerr << "IDACreate failed" << std::endl; return 1; }}

    ierr = IDAInit(ida_mem, ida_residual, 0.0, yy, yp);
    if (ierr < 0) {{ std::cerr << "IDAInit failed: " << ierr << std::endl; return 1; }}

    // Tolerances: relax for larger systems to avoid excessive Jacobian updates
    double ida_rtol = (N_TOTAL > 30) ? 1e-4 : 1e-6;
    double ida_atol = (N_TOTAL > 30) ? 1e-6 : 1e-8;
    ierr = IDASStolerances(ida_mem, ida_rtol, ida_atol);
    if (ierr < 0) {{ std::cerr << "IDASStolerances failed" << std::endl; return 1; }}

    ierr = IDASetLinearSolver(ida_mem, LS, A);
    if (ierr < 0) {{ std::cerr << "IDASetLinearSolver failed" << std::endl; return 1; }}

    ierr = IDASetId(ida_mem, var_id);
    if (ierr < 0) {{ std::cerr << "IDASetId failed" << std::endl; return 1; }}

    // Max internal step: allow IDA freedom, capped at output interval
    double ida_max_step = (N_TOTAL > 30) ? fmax({dt}, 0.005) : {dt};
    IDASetMaxStep(ida_mem, ida_max_step);
    IDASetMaxNumSteps(ida_mem, 5000000);
    // Limit BDF order to 3 for large stiff systems (more stable)
    if (N_TOTAL > 30) IDASetMaxOrd(ida_mem, 3);

    // --- Correct initial conditions ---
    ierr = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, {min(dt * 10, 0.01)});
    if (ierr < 0) {{
        std::cerr << "WARNING: IDACalcIC returned " << ierr
                  << " (proceeding anyway)" << std::endl;
    }}
    IDAGetConsistentIC(ida_mem, yy, yp);

    // === IC torque-balance diagnostics (all governors with xi state) ===
    // After _sync_voltages_to_states, the KCL residual should be ~0.
    // These diagnostics verify that yp[omega] (= dω/dt at t=0) is negligible.
{xi_diag_code}

    // === IC exciter Vm diagnostic ===
    // After _sync_voltages_to_states, Vm should match Vterm_IDA.
    // These diagnostics verify the residual is negligible.
{vm_diag_code}

    // Re-sync yp from the (unmodified) state, then tell IDA about the IC.
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = 0.0;
    dae_residual(yy_data, yp_data, res0, 0.0);
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
    for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;
    IDAReInit(ida_mem, 0.0, yy, yp);

    // --- Output setup ---
    std::ofstream outfile("simulation_results.csv");
    outfile << "{header}" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Print header
    std::cout << "Dirac DAE Simulation (SUNDIALS IDA)" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  max_dt = {dt},  T = {duration} s" << std::endl;
    std::cout << "  Solver: SUNDIALS IDA (variable-order BDF, adaptive step)"
              << std::endl;
    std::cout << "  rtol = " << ida_rtol << ",  atol = " << ida_atol
              << ",  max_step = " << ida_max_step << std::endl;
    std::cout << std::endl;

    // Initial diagnostics
    dae_residual(yy_data, yp_data, res0, 0.0);
    double max_res = 0.0;
    int max_res_idx = -1;
    for (int i = 0; i < N_TOTAL; ++i) {{
        if (fabs(res0[i]) > max_res) {{ max_res = fabs(res0[i]); max_res_idx = i; }}
    }}
    std::cout << "[IDA] Initial max |residual| = " << max_res
              << " at index " << max_res_idx << std::endl;

    // Log initial state
    log_state(outfile, 0.0, yy_data, Vd_net, Vq_net, Vterm_net);
    long n_logged = 1;

    // --- Event-driven integration segments ---
    // Fault boundaries create discontinuities; IDA must be restarted
    // at each boundary to avoid step-size collapse.
    const double seg_ends[{n_segments}] = {{ {event_times_str} }};
    const int n_segments = {n_segments};
    double t_current = 0.0;
    double t_next_log = {log_every_time};
    int last_sec = -1;
    bool aborted = false;

    for (int seg = 0; seg < n_segments && !aborted; ++seg) {{
        double t_seg_end = seg_ends[seg];

        if (t_seg_end <= t_current + 1e-12) continue;

        // Update fault flags for this segment (must happen BEFORE residual
        // evaluations so IDACalcIC / IDAReInit see the correct topology)
        for (int f = 0; f < N_FAULTS; ++f) {{
            fault_active[f] = (t_current >= FAULT_T_START[f]
                               && t_current < FAULT_T_END[f]) ? 1 : 0;
        }}

        // Reinitialize IDA at segment boundary (except first segment)
        if (seg > 0) {{
            // Recompute yp for the new regime (fault topology changed)
            for (int i = 0; i < N_TOTAL; ++i) yp_data[i] = 0.0;
            dae_residual(yy_data, yp_data, res0, t_current);
            for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
            for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;

            ierr = IDAReInit(ida_mem, t_current, yy, yp);
            if (ierr < 0) {{
                std::cerr << "IDAReInit failed at t=" << t_current
                          << " flag=" << ierr << std::endl;
                break;
            }}

            // Fix algebraic variables for new regime
            ierr = IDACalcIC(ida_mem, IDA_YA_YDP_INIT,
                             t_current + {min(dt * 10, 0.01)});
            if (ierr < 0) {{
                std::cerr << "WARNING: IDACalcIC at t=" << t_current
                          << " returned " << ierr << std::endl;
            }}
            IDAGetConsistentIC(ida_mem, yy, yp);

            std::cout << "[IDA] Reinit at t=" << t_current
                      << " (segment " << seg << "/" << n_segments << ")"
                      << std::endl;
        }}

        // Integrate within this segment
        sunrealtype t_ret = t_current;
        while (t_ret < t_seg_end - 1e-12) {{
            double t_out = t_ret + {dt};
            if (t_out > t_seg_end) t_out = t_seg_end;

            ierr = IDASolve(ida_mem, t_out, &t_ret, yy, yp, IDA_NORMAL);
            if (ierr < 0) {{
                std::cerr << "IDASolve failed at t=" << t_ret
                          << " flag=" << ierr << std::endl;
                aborted = true;
                break;
            }}

            // Log at regular intervals
            if (t_ret >= t_next_log - 1e-12 || t_ret >= {duration} - 1e-12) {{
                log_state(outfile, t_ret, yy_data, Vd_net, Vq_net, Vterm_net);
                n_logged++;
                while (t_next_log <= t_ret + 1e-12)
                    t_next_log += {log_every_time};
            }}

            // Progress check every ~1 second
            int sec = (int)t_ret;
            if (sec > last_sec) {{
                last_sec = sec;
                for (int i = 0; i < N_BUS; ++i) {{
                    Vd_net[i]   = yy_data[N_DIFF + 2*i];
                    Vq_net[i]   = yy_data[N_DIFF + 2*i + 1];
                    Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i]
                                        + Vq_net[i]*Vq_net[i]);
                }}
                std::cout << "t=" << t_ret
                          << " Vterm[0]=" << Vterm_net[0] << std::endl;

                if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {{
                    std::cout << "Stability limit reached. Stopping." << std::endl;
                    goto ida_done;
                }}
            }}
        }}

        t_current = t_seg_end;
    }}

ida_done:
    outfile.close();

    // --- Statistics ---
    long nst, nre, nje, nni, ncfn;
    IDAGetNumSteps(ida_mem, &nst);
    IDAGetNumResEvals(ida_mem, &nre);
    IDAGetNumJacEvals(ida_mem, &nje);
    IDAGetNumNonlinSolvIters(ida_mem, &nni);
    IDAGetNumNonlinSolvConvFails(ida_mem, &ncfn);
    std::cout << std::endl;
    std::cout << "[IDA] Steps: " << nst << "  Residual evals: " << nre
              << "  Jacobian evals: " << nje << std::endl;
    std::cout << "[IDA] Newton iters: " << nni
              << "  Conv fails: " << ncfn << std::endl;
    std::cout << "[IDA] Logged " << n_logged << " data points" << std::endl;
    std::cout << "Done. Results in simulation_results.csv" << std::endl;

    // --- Cleanup ---
    IDAFree(&ida_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(var_id);
    SUNContext_Free(&sunctx);

    return 0;
}}
"""

    def _build_csv_columns(self) -> List[Tuple[str, str]]:
        """Build CSV column definitions (header, C++ expression) for logging."""
        cols = [("t", "t")]

        for comp in self.components:
            off = self.state_offsets[comp.name]
            # States
            for si, sname in enumerate(comp.state_schema):
                cols.append((f"{comp.name}.{sname}", f"x[{off + si}]"))
            # Observables
            for obs_name, obs_info in comp.observables.items():
                cpp_expr = obs_info.get('cpp_expr', '0.0')
                # Rewrite x[], inputs[], outputs[] refs for this component
                cpp_expr_full = self._rewrite_obs_expr(comp, cpp_expr)
                cols.append((f"{comp.name}.{obs_name}", cpp_expr_full))

        # Bus voltages
        for bus_id in self.bus_indices:
            bi = self.bus_map[bus_id]
            cols.append((f"Vd_Bus{bus_id}", f"Vd_net[{bi}]"))
            cols.append((f"Vq_Bus{bus_id}", f"Vq_net[{bi}]"))
            cols.append((f"Vterm_Bus{bus_id}", f"Vterm_net[{bi}]"))

        return cols

    def _rewrite_obs_expr(self, comp: PowerComponent, expr: str) -> str:
        """Rewrite an observable C++ expression with global array references."""
        off = self.state_offsets[comp.name]
        import re
        # x[k] → y[off + k]
        expr = re.sub(r'x\[(\d+)\]',
                       lambda m: f'y[{off + int(m.group(1))}]', expr)
        # inputs[k] → inputs_COMP[k]
        expr = re.sub(r'inputs\[(\d+)\]',
                       lambda m: f'inputs_{comp.name}[{m.group(1)}]', expr)
        # outputs[k] → outputs_COMP[k]
        expr = re.sub(r'outputs\[(\d+)\]',
                       lambda m: f'outputs_{comp.name}[{m.group(1)}]', expr)
        # Substitute bare parameter names with literal numeric values
        # (parameters are local to component functions, not visible in CSV scope)
        for pname in sorted(comp.param_schema.keys(), key=len, reverse=True):
            if pname in comp.params and re.search(r'\b' + re.escape(pname) + r'\b', expr):
                expr = re.sub(r'\b' + re.escape(pname) + r'\b',
                              f'{float(comp.params[pname]):.15g}', expr)
        return expr
