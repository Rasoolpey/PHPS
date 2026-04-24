"""Numba JIT-accelerated ODE solver for the PHS-DAE system.

Flattens all component data (parameters, states, wiring) into contiguous
numpy arrays so that the entire ODE right-hand side and BDF-1 time-stepping
loop can run in compiled machine code via Numba ``@njit``.

The Y-bus LU solve is the only call that exits Numba — it uses
``scipy.linalg.lu_solve`` (compiled Fortran/LAPACK under the hood).
"""

from __future__ import annotations

import csv
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.integrate

from numba import njit, types
from numba.typed import List as NumbaList

from src.dirac.py_codegen import translate_cpp_kernel

# ======================================================================
# Wiring expression types (encoded as integers)
# ======================================================================
W_ZERO        = 0   # constant 0.0
W_CONST       = 1   # constant float value
W_VD_NET      = 2   # Vd_net[bus_idx]
W_VQ_NET      = 3   # Vq_net[bus_idx]
W_VTERM_NET   = 4   # Vterm_net[bus_idx]
W_VD_DQ       = 5   # vd_dq[gen_idx]
W_VQ_DQ       = 6   # vq_dq[gen_idx]
W_OUTPUT      = 7   # flat_outputs[out_offset + out_idx]
W_CONST_PLUS_OUTPUT = 8  # const_val + flat_outputs[out_offset + out_idx]


def _parse_wiring_expr(
    expr: str,
    comp_out_offsets: Dict[str, int],
    gen_idx_map: Dict[str, int],
) -> Tuple[int, int, float]:
    """Parse a wiring expression to (type, index, const_val).

    Returns
    -------
    (w_type, w_index, w_const)
        - W_ZERO: (0, 0, 0.0)
        - W_CONST: (1, 0, value)
        - W_VD_NET: (2, bus_idx, 0.0)
        - W_VQ_NET: (3, bus_idx, 0.0)
        - W_VTERM_NET: (4, bus_idx, 0.0)
        - W_VD_DQ: (5, gen_idx, 0.0)
        - W_VQ_DQ: (6, gen_idx, 0.0)
        - W_OUTPUT: (7, flat_idx, 0.0)
        - W_CONST_PLUS_OUTPUT: (8, flat_idx, const_val)
    """
    expr = expr.strip()

    if expr == '0.0' or expr == '0':
        return (W_ZERO, 0, 0.0)

    # Vd_net[i]
    m = re.match(r'^Vd_net\[(\d+)\]$', expr)
    if m:
        return (W_VD_NET, int(m.group(1)), 0.0)

    # Vq_net[i]
    m = re.match(r'^Vq_net\[(\d+)\]$', expr)
    if m:
        return (W_VQ_NET, int(m.group(1)), 0.0)

    # Vterm_net[i]
    m = re.match(r'^Vterm_net\[(\d+)\]$', expr)
    if m:
        return (W_VTERM_NET, int(m.group(1)), 0.0)

    # vd_dq_GENROU_N
    m = re.match(r'^vd_dq_(\w+)$', expr)
    if m:
        gen_name = m.group(1)
        if gen_name in gen_idx_map:
            return (W_VD_DQ, gen_idx_map[gen_name], 0.0)

    # vq_dq_GENROU_N
    m = re.match(r'^vq_dq_(\w+)$', expr)
    if m:
        gen_name = m.group(1)
        if gen_name in gen_idx_map:
            return (W_VQ_DQ, gen_idx_map[gen_name], 0.0)

    # const + outputs_COMP[i]  (PSS correction)
    m = re.match(
        r'^([0-9eE.+-]+)\s*\+\s*outputs_(\w+)\[(\d+)\]$', expr
    )
    if m:
        const_val = float(m.group(1))
        comp_name = m.group(2)
        out_idx = int(m.group(3))
        if comp_name in comp_out_offsets:
            flat_idx = comp_out_offsets[comp_name] + out_idx
            return (W_CONST_PLUS_OUTPUT, flat_idx, const_val)

    # outputs_COMP[i]
    m = re.match(r'^outputs_(\w+)\[(\d+)\]$', expr)
    if m:
        comp_name = m.group(1)
        out_idx = int(m.group(2))
        if comp_name in comp_out_offsets:
            flat_idx = comp_out_offsets[comp_name] + out_idx
            return (W_OUTPUT, flat_idx, 0.0)

    # Plain float constant
    try:
        val = float(expr)
        if val == 0.0:
            return (W_ZERO, 0, 0.0)
        return (W_CONST, 0, val)
    except ValueError:
        pass

    # Fallback: zero
    print(f'[JIT] WARNING: unhandled wiring expr: {expr!r} → 0.0')
    return (W_ZERO, 0, 0.0)


# ======================================================================
# Numba JIT kernels
# ======================================================================

@njit(cache=True)
def _resolve_wiring_value(
    w_type: int, w_index: int, w_const: float,
    Vd: np.ndarray, Vq: np.ndarray, Vterm: np.ndarray,
    vd_dq: np.ndarray, vq_dq: np.ndarray,
    flat_outputs: np.ndarray,
) -> float:
    """Evaluate a pre-compiled wiring expression."""
    if w_type == 0:       # W_ZERO
        return 0.0
    elif w_type == 1:     # W_CONST
        return w_const
    elif w_type == 2:     # W_VD_NET
        return Vd[w_index]
    elif w_type == 3:     # W_VQ_NET
        return Vq[w_index]
    elif w_type == 4:     # W_VTERM_NET
        return Vterm[w_index]
    elif w_type == 5:     # W_VD_DQ
        return vd_dq[w_index]
    elif w_type == 6:     # W_VQ_DQ
        return vq_dq[w_index]
    elif w_type == 7:     # W_OUTPUT
        return flat_outputs[w_index]
    elif w_type == 8:     # W_CONST_PLUS_OUTPUT
        return w_const + flat_outputs[w_index]
    return 0.0


@njit(cache=True)
def _compute_norton_genrou(
    x: np.ndarray, off: int,
    ra: float, xd_pp: float, xq_pp: float,
    kd: float, kq: float,
) -> Tuple[float, float]:
    """Compute Norton current (I_Re, I_Im) for a GENROU generator."""
    delta = x[off]
    Eq_p = x[off + 2]
    psi_d = x[off + 3]
    Ed_p = x[off + 4]
    psi_q = x[off + 5]

    psi_d_pp = Eq_p * kd + psi_d * (1.0 - kd)
    psi_q_pp = -Ed_p * kq + psi_q * (1.0 - kq)

    det = ra * ra + xd_pp * xq_pp
    id_no = (-ra * psi_q_pp + xq_pp * psi_d_pp) / det
    iq_no = (xd_pp * psi_q_pp + ra * psi_d_pp) / det

    sin_d = math.sin(delta)
    cos_d = math.cos(delta)

    I_Re = id_no * sin_d + iq_no * cos_d
    I_Im = -id_no * cos_d + iq_no * sin_d
    return I_Re, I_Im


@njit(cache=True)
def _compute_dq_frame_jit(
    x: np.ndarray, off: int,
    Vd_bus: float, Vq_bus: float,
    idx_delta: int, idx_Eq: int, idx_psi_d: int,
    idx_Ed: int, idx_psi_q: int,
    kd: float, kq: float,
    ra: float, xd_pp: float, xq_pp: float, det: float,
) -> Tuple[float, float, float, float]:
    """Return (vd_dq, vq_dq, id_dq, iq_dq) in machine dq frame."""
    delta = x[off + idx_delta]
    sin_d = math.sin(delta)
    cos_d = math.cos(delta)

    vd_dq = Vd_bus * sin_d - Vq_bus * cos_d
    vq_dq = Vd_bus * cos_d + Vq_bus * sin_d

    Eq_p = x[off + idx_Eq]
    psi_d = x[off + idx_psi_d]
    Ed_p = x[off + idx_Ed]
    psi_q = x[off + idx_psi_q]

    psi_d_pp = Eq_p * kd + psi_d * (1.0 - kd)
    psi_q_pp = -Ed_p * kq + psi_q * (1.0 - kq)

    rhs_d = vd_dq + psi_q_pp
    rhs_q = vq_dq - psi_d_pp

    id_dq = (-ra * rhs_d - xq_pp * rhs_q) / det
    iq_dq = (xd_pp * rhs_d - ra * rhs_q) / det

    return vd_dq, vq_dq, id_dq, iq_dq


@njit(cache=True)
def _compute_norton_all(
    x: np.ndarray,
    n_gen: int,
    gen_offsets: np.ndarray,
    gen_bus_indices: np.ndarray,
    gen_ra: np.ndarray,
    gen_xd_pp: np.ndarray,
    gen_xq_pp: np.ndarray,
    gen_kd: np.ndarray,
    gen_kq: np.ndarray,
    n_bus: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all Norton current injections (Re, Im arrays)."""
    I_Re = np.zeros(n_bus)
    I_Im = np.zeros(n_bus)
    for g in range(n_gen):
        off = gen_offsets[g]
        bi = gen_bus_indices[g]
        ir, ii = _compute_norton_genrou(
            x, off,
            gen_ra[g], gen_xd_pp[g], gen_xq_pp[g],
            gen_kd[g], gen_kq[g],
        )
        I_Re[bi] += ir
        I_Im[bi] += ii
    return I_Re, I_Im


@njit(cache=True)
def _rhs_core(
    x: np.ndarray,
    Vd: np.ndarray,
    Vq: np.ndarray,
    # Generator dq params (flattened)
    n_gen: int,
    gen_offsets: np.ndarray,
    gen_bus_indices: np.ndarray,
    gen_ra: np.ndarray,
    gen_xd_pp: np.ndarray,
    gen_xq_pp: np.ndarray,
    gen_kd: np.ndarray,
    gen_kq: np.ndarray,
    gen_det: np.ndarray,
    gen_idx_delta: np.ndarray,
    gen_idx_Eq: np.ndarray,
    gen_idx_psi_d: np.ndarray,
    gen_idx_Ed: np.ndarray,
    gen_idx_psi_q: np.ndarray,
    gen_out_id_dq: np.ndarray,
    gen_out_iq_dq: np.ndarray,
    gen_out_flat_offset: np.ndarray,
    gen_H: np.ndarray,
    # Wiring tables
    wiring_types: np.ndarray,     # (total_inputs,)
    wiring_indices: np.ndarray,   # (total_inputs,)
    wiring_consts: np.ndarray,    # (total_inputs,)
    # Component layout
    n_comp: int,
    comp_offsets: np.ndarray,     # (n_comp,) state offset
    comp_n_states: np.ndarray,    # (n_comp,)
    comp_n_inputs: np.ndarray,    # (n_comp,)
    comp_n_outputs: np.ndarray,   # (n_comp,)
    comp_input_start: np.ndarray, # (n_comp,) index into wiring tables
    comp_output_start: np.ndarray,# (n_comp,) index into flat_outputs
    # Outputs & inputs buffers
    flat_outputs: np.ndarray,
    # COI
    delta_coi_idx: int,
    omega_b: float,
    coi_2H: float,
    n_diff: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core RHS computation — all numpy, no Python objects.

    Returns (vd_dq, vq_dq, Vterm) to be consumed by the kernel pass.
    The actual dxdt computation happens in the kernel execution phase
    (outside this function, since we can't JIT the exec'd kernels directly).
    """
    Vterm = np.sqrt(Vd * Vd + Vq * Vq)

    # Compute dq-frame voltages and currents for generators
    vd_dq = np.zeros(n_gen)
    vq_dq = np.zeros(n_gen)
    for g in range(n_gen):
        off = gen_offsets[g]
        bi = gen_bus_indices[g]
        vd, vq, id_dq, iq_dq = _compute_dq_frame_jit(
            x, off, Vd[bi], Vq[bi],
            gen_idx_delta[g], gen_idx_Eq[g], gen_idx_psi_d[g],
            gen_idx_Ed[g], gen_idx_psi_q[g],
            gen_kd[g], gen_kq[g],
            gen_ra[g], gen_xd_pp[g], gen_xq_pp[g], gen_det[g],
        )
        vd_dq[g] = vd
        vq_dq[g] = vq
        # Inject dq currents into flat outputs
        fo = gen_out_flat_offset[g]
        flat_outputs[fo + gen_out_id_dq[g]] = id_dq
        flat_outputs[fo + gen_out_iq_dq[g]] = iq_dq

    return vd_dq, vq_dq, Vterm


@njit(cache=True)
def _gather_inputs_jit(
    comp_idx: int,
    comp_input_start: np.ndarray,
    comp_n_inputs: np.ndarray,
    wiring_types: np.ndarray,
    wiring_indices: np.ndarray,
    wiring_consts: np.ndarray,
    Vd: np.ndarray,
    Vq: np.ndarray,
    Vterm: np.ndarray,
    vd_dq: np.ndarray,
    vq_dq: np.ndarray,
    flat_outputs: np.ndarray,
    inputs_buf: np.ndarray,
) -> None:
    """Fill the inputs buffer for component comp_idx."""
    start = comp_input_start[comp_idx]
    n_in = comp_n_inputs[comp_idx]
    for i in range(n_in):
        wi = start + i
        inputs_buf[i] = _resolve_wiring_value(
            wiring_types[wi], wiring_indices[wi], wiring_consts[wi],
            Vd, Vq, Vterm, vd_dq, vq_dq, flat_outputs,
        )


@njit(cache=True)
def _apply_coi(
    dxdt: np.ndarray,
    x: np.ndarray,
    n_gen: int,
    gen_offsets: np.ndarray,
    gen_H: np.ndarray,
    coi_2H: float,
    omega_b: float,
    delta_coi_idx: int,
) -> None:
    """Apply COI correction to dxdt."""
    if n_gen <= 1:
        dxdt[delta_coi_idx] = 0.0
        return
    omega_coi = 0.0
    for g in range(n_gen):
        omega_coi += 2.0 * gen_H[g] * x[gen_offsets[g] + 1]
    omega_coi /= coi_2H
    for g in range(n_gen):
        dxdt[gen_offsets[g]] -= omega_b * (omega_coi - 1.0)
    dxdt[delta_coi_idx] = omega_b * (omega_coi - 1.0)


# ======================================================================
# Main JIT Solver Class
# ======================================================================

class JitDAESolver:
    """Numba JIT-accelerated ODE solver for the PHS-DAE system.

    Flattens all component data into contiguous numpy arrays, JIT-compiles
    Norton current injection, dq-frame computation, wiring resolution,
    and COI correction.  Component dynamics kernels are called as Python
    functions (since they are generated via exec()), but all surrounding
    orchestration is JIT-compiled.

    Parameters
    ----------
    runner : DiracRunner
        A fully built DiracRunner.
    """

    def __init__(self, runner):
        dc = runner.dae_compiler
        self.output_dir = runner.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # --- System dimensions ---
        self.n_diff = dc.n_diff
        self.n_bus = dc.n_bus
        self.delta_coi_idx = dc.delta_coi_idx

        # --- Network ---
        self.Y_full = dc.Y_full.copy()
        self.bus_map = dict(dc.bus_map)
        self.bus_ids = sorted(dc.bus_map, key=dc.bus_map.__getitem__)

        # --- Components ---
        self.components = dc.components
        self.state_offsets = dict(dc.state_offsets)
        self.wiring_map = dict(dc.wiring_map)

        # --- Initial conditions ---
        self.x0 = runner.x0.copy()

        # --- Fault events ---
        self.fault_events: List[dict] = []
        for ev in dc.fault_events_dae:
            self.fault_events.append({
                'bus_idx': int(ev['bus_idx']),
                't_start': float(ev['t_start']),
                't_end': float(ev['t_end']),
                'G': float(ev.get('G', ev.get('g', 0.0))),
                'B': float(ev.get('B', ev.get('b', 0.0))),
            })

        # === Flatten generator data ===
        self._gen_comps = []
        gen_idx_map: Dict[str, int] = {}
        for comp in self.components:
            if comp.component_role == 'generator' and 'omega' in comp.state_schema:
                gen_idx_map[comp.name] = len(self._gen_comps)
                self._gen_comps.append(comp)

        n_gen = len(self._gen_comps)
        self.n_gen = n_gen
        self._gen_offsets = np.zeros(n_gen, dtype=np.int64)
        self._gen_bus_indices = np.zeros(n_gen, dtype=np.int64)
        self._gen_ra = np.zeros(n_gen)
        self._gen_xd_pp = np.zeros(n_gen)
        self._gen_xq_pp = np.zeros(n_gen)
        self._gen_kd = np.zeros(n_gen)
        self._gen_kq = np.zeros(n_gen)
        self._gen_det = np.zeros(n_gen)
        self._gen_H = np.zeros(n_gen)
        self._gen_idx_delta = np.zeros(n_gen, dtype=np.int64)
        self._gen_idx_Eq = np.zeros(n_gen, dtype=np.int64)
        self._gen_idx_psi_d = np.zeros(n_gen, dtype=np.int64)
        self._gen_idx_Ed = np.zeros(n_gen, dtype=np.int64)
        self._gen_idx_psi_q = np.zeros(n_gen, dtype=np.int64)
        self._gen_out_id_dq = np.zeros(n_gen, dtype=np.int64)
        self._gen_out_iq_dq = np.zeros(n_gen, dtype=np.int64)
        self._gen_out_flat_offset = np.zeros(n_gen, dtype=np.int64)

        self._omega_b = 2.0 * math.pi * 60.0
        if n_gen > 0:
            ob = self._gen_comps[0].params.get('omega_b', None)
            if ob is not None:
                try:
                    self._omega_b = float(eval(str(ob), {'M_PI': math.pi}))
                except Exception:
                    pass
        self._coi_2H = sum(
            2.0 * float(c.params.get('H', 3.0)) for c in self._gen_comps
        )

        # === Flatten component layout ===
        n_comp = len(self.components)
        self.n_comp = n_comp
        self._comp_offsets = np.zeros(n_comp, dtype=np.int64)
        self._comp_n_states = np.zeros(n_comp, dtype=np.int64)
        self._comp_n_inputs = np.zeros(n_comp, dtype=np.int64)
        self._comp_n_outputs = np.zeros(n_comp, dtype=np.int64)
        self._comp_input_start = np.zeros(n_comp, dtype=np.int64)
        self._comp_output_start = np.zeros(n_comp, dtype=np.int64)

        # Compute flat output offsets
        comp_out_offsets: Dict[str, int] = {}
        out_offset = 0
        for ci, comp in enumerate(self.components):
            n_out = len(comp.port_schema['out'])
            comp_out_offsets[comp.name] = out_offset
            self._comp_output_start[ci] = out_offset
            self._comp_n_outputs[ci] = n_out
            out_offset += n_out
        self.total_outputs = out_offset
        self._flat_outputs = np.zeros(out_offset)

        # Compute flat input offsets and wiring tables
        inp_offset = 0
        for ci, comp in enumerate(self.components):
            n_in = len(comp.port_schema['in'])
            self._comp_offsets[ci] = self.state_offsets[comp.name]
            self._comp_n_states[ci] = len(comp.state_schema)
            self._comp_n_inputs[ci] = n_in
            self._comp_input_start[ci] = inp_offset
            inp_offset += n_in
        self.total_inputs = inp_offset

        # Build wiring tables
        self._wiring_types = np.zeros(self.total_inputs, dtype=np.int64)
        self._wiring_indices = np.zeros(self.total_inputs, dtype=np.int64)
        self._wiring_consts = np.zeros(self.total_inputs)

        for ci, comp in enumerate(self.components):
            start = self._comp_input_start[ci]
            for i, (p_name, _, _) in enumerate(comp.port_schema['in']):
                key = (comp.name, p_name)
                expr = self.wiring_map.get(key, '0.0')
                w_type, w_index, w_const = _parse_wiring_expr(
                    expr, comp_out_offsets, gen_idx_map
                )
                self._wiring_types[start + i] = w_type
                self._wiring_indices[start + i] = w_index
                self._wiring_consts[start + i] = w_const

        # Fill generator flat arrays
        for g, comp in enumerate(self._gen_comps):
            p = comp.params
            off = self.state_offsets[comp.name]
            bus_id = int(p['bus'])
            self._gen_offsets[g] = off
            self._gen_bus_indices[g] = self.bus_map[bus_id]
            self._gen_ra[g] = float(p.get('ra', 0.0))
            self._gen_xd_pp[g] = float(p.get('xd_double_prime', 0.2))
            self._gen_xq_pp[g] = float(p.get('xq_double_prime', self._gen_xd_pp[g]))
            xd_p = float(p.get('xd_prime', 0.3))
            xq_p = float(p.get('xq_prime', self._gen_xq_pp[g]))
            xl = float(p.get('xl', 0.0))
            self._gen_kd[g] = (self._gen_xd_pp[g] - xl) / (xd_p - xl) if (xd_p - xl) != 0 else 1.0
            self._gen_kq[g] = (self._gen_xq_pp[g] - xl) / (xq_p - xl) if (xq_p - xl) != 0 else 1.0
            self._gen_det[g] = self._gen_ra[g]**2 + self._gen_xd_pp[g] * self._gen_xq_pp[g]
            self._gen_H[g] = float(p.get('H', 3.0))

            sch = comp.state_schema
            self._gen_idx_delta[g] = sch.index('delta')
            self._gen_idx_Eq[g] = sch.index('E_q_prime')
            self._gen_idx_psi_d[g] = sch.index('psi_d')
            self._gen_idx_Ed[g] = sch.index('E_d_prime')
            self._gen_idx_psi_q[g] = sch.index('psi_q')
            out_names = [po[0] for po in comp.port_schema['out']]
            self._gen_out_id_dq[g] = out_names.index('id_dq')
            self._gen_out_iq_dq[g] = out_names.index('iq_dq')
            self._gen_out_flat_offset[g] = comp_out_offsets[comp.name]

        # Temporary input buffer (max input size across comps)
        max_inputs = max(len(c.port_schema['in']) for c in self.components) if self.components else 0
        self._inp_buf = np.zeros(max_inputs)

        # === Build Python component functions ===
        print('[JitSolver] Translating component kernels...')
        self._step_fns = []
        self._out_fns = []
        from src.dirac.py_codegen import make_step_func, make_out_func
        for comp in self.components:
            self._step_fns.append(make_step_func(comp))
            self._out_fns.append(make_out_func(comp))
        print(f'[JitSolver] {len(self.components)} components compiled.')

        # === Pre-factor Y-bus ===
        self._Y_prefactored: Dict[frozenset, object] = {}
        self._prefactor_ybus(frozenset())
        for ev in self.fault_events:
            self._prefactor_ybus(frozenset([ev['bus_idx']]))

        # === Warm up JIT ===
        print('[JitSolver] Warming up Numba JIT...')
        t_warmup = time.time()
        self._warmup_jit()
        print(f'[JitSolver] JIT warmup done in {time.time() - t_warmup:.2f}s.')

    def _warmup_jit(self):
        """Call JIT functions once with real data to trigger compilation."""
        x = self.x0.copy()
        I_Re, I_Im = _compute_norton_all(
            x, self.n_gen,
            self._gen_offsets, self._gen_bus_indices,
            self._gen_ra, self._gen_xd_pp, self._gen_xq_pp,
            self._gen_kd, self._gen_kq, self.n_bus,
        )
        Vd = np.zeros(self.n_bus)
        Vq = np.zeros(self.n_bus)
        vd_dq, vq_dq, Vterm = _rhs_core(
            x, Vd, Vq,
            self.n_gen,
            self._gen_offsets, self._gen_bus_indices,
            self._gen_ra, self._gen_xd_pp, self._gen_xq_pp,
            self._gen_kd, self._gen_kq, self._gen_det,
            self._gen_idx_delta, self._gen_idx_Eq, self._gen_idx_psi_d,
            self._gen_idx_Ed, self._gen_idx_psi_q,
            self._gen_out_id_dq, self._gen_out_iq_dq, self._gen_out_flat_offset,
            self._gen_H,
            self._wiring_types, self._wiring_indices, self._wiring_consts,
            self.n_comp,
            self._comp_offsets, self._comp_n_states, self._comp_n_inputs, self._comp_n_outputs,
            self._comp_input_start, self._comp_output_start,
            self._flat_outputs,
            self.delta_coi_idx, self._omega_b, self._coi_2H, self.n_diff,
        )
        dxdt = np.zeros(self.n_diff)
        _apply_coi(
            dxdt, x, self.n_gen,
            self._gen_offsets, self._gen_H,
            self._coi_2H, self._omega_b, self.delta_coi_idx,
        )
        inp = np.zeros(4)
        _gather_inputs_jit(
            0,
            self._comp_input_start, self._comp_n_inputs,
            self._wiring_types, self._wiring_indices, self._wiring_consts,
            Vd, Vq, Vterm, vd_dq, vq_dq, self._flat_outputs,
            inp,
        )

    # ------------------------------------------------------------------
    # Y-bus
    # ------------------------------------------------------------------

    def _prefactor_ybus(self, active_faults: frozenset) -> None:
        Y_eff = self.Y_full.copy()
        for ev in self.fault_events:
            if ev['bus_idx'] in active_faults:
                bi = ev['bus_idx']
                Y_eff[bi, bi] += complex(ev['G'], ev['B'])
        self._Y_prefactored[active_faults] = scipy.linalg.lu_factor(Y_eff)

    def _get_lu(self, t: float):
        active = frozenset(
            ev['bus_idx'] for ev in self.fault_events
            if ev['t_start'] <= t < ev['t_end']
        )
        if active not in self._Y_prefactored:
            self._prefactor_ybus(active)
        return self._Y_prefactored[active]

    def _solve_network(self, x: np.ndarray, t: float):
        """Solve Y-bus using JIT Norton currents + scipy LU."""
        I_Re, I_Im = _compute_norton_all(
            x, self.n_gen,
            self._gen_offsets, self._gen_bus_indices,
            self._gen_ra, self._gen_xd_pp, self._gen_xq_pp,
            self._gen_kd, self._gen_kq, self.n_bus,
        )
        I_inj = I_Re + 1j * I_Im
        lu = self._get_lu(t)
        V = scipy.linalg.lu_solve(lu, I_inj)
        return V.real.copy(), V.imag.copy()

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Full ODE right-hand side with JIT-accelerated core."""
        Vd, Vq = self._solve_network(x, t)

        # JIT core: dq-frame, inject dq currents into flat_outputs
        vd_dq, vq_dq, Vterm = _rhs_core(
            x, Vd, Vq,
            self.n_gen,
            self._gen_offsets, self._gen_bus_indices,
            self._gen_ra, self._gen_xd_pp, self._gen_xq_pp,
            self._gen_kd, self._gen_kq, self._gen_det,
            self._gen_idx_delta, self._gen_idx_Eq, self._gen_idx_psi_d,
            self._gen_idx_Ed, self._gen_idx_psi_q,
            self._gen_out_id_dq, self._gen_out_iq_dq, self._gen_out_flat_offset,
            self._gen_H,
            self._wiring_types, self._wiring_indices, self._wiring_consts,
            self.n_comp,
            self._comp_offsets, self._comp_n_states, self._comp_n_inputs, self._comp_n_outputs,
            self._comp_input_start, self._comp_output_start,
            self._flat_outputs,
            self.delta_coi_idx, self._omega_b, self._coi_2H, self.n_diff,
        )

        fo = self._flat_outputs
        inp = self._inp_buf

        # Pass 1: evaluate component outputs (Python kernels)
        for ci, comp in enumerate(self.components):
            off = self._comp_offsets[ci]
            n_st = self._comp_n_states[ci]
            x_c = x[off:off + n_st]
            _gather_inputs_jit(
                ci, self._comp_input_start, self._comp_n_inputs,
                self._wiring_types, self._wiring_indices, self._wiring_consts,
                Vd, Vq, Vterm, vd_dq, vq_dq, fo, inp,
            )
            out_start = self._comp_output_start[ci]
            out_end = out_start + self._comp_n_outputs[ci]
            out_slice = fo[out_start:out_end]
            self._out_fns[ci](x_c, inp[:self._comp_n_inputs[ci]], out_slice, t)
            # Re-inject dq currents for generators (output fn may overwrite)
            if comp.name in {c.name for c in self._gen_comps}:
                g = next(i for i, c in enumerate(self._gen_comps) if c.name == comp.name)
                bi = self._gen_bus_indices[g]
                vd, vq, id_dq, iq_dq = _compute_dq_frame_jit(
                    x, off, Vd[bi], Vq[bi],
                    self._gen_idx_delta[g], self._gen_idx_Eq[g], self._gen_idx_psi_d[g],
                    self._gen_idx_Ed[g], self._gen_idx_psi_q[g],
                    self._gen_kd[g], self._gen_kq[g],
                    self._gen_ra[g], self._gen_xd_pp[g], self._gen_xq_pp[g], self._gen_det[g],
                )
                fo[self._gen_out_flat_offset[g] + self._gen_out_id_dq[g]] = id_dq
                fo[self._gen_out_flat_offset[g] + self._gen_out_iq_dq[g]] = iq_dq

        # Pass 2: evaluate component dynamics (Python kernels)
        dxdt = np.zeros(self.n_diff)
        for ci, comp in enumerate(self.components):
            off = self._comp_offsets[ci]
            n_st = self._comp_n_states[ci]
            x_c = x[off:off + n_st]
            _gather_inputs_jit(
                ci, self._comp_input_start, self._comp_n_inputs,
                self._wiring_types, self._wiring_indices, self._wiring_consts,
                Vd, Vq, Vterm, vd_dq, vq_dq, fo, inp,
            )
            out_start = self._comp_output_start[ci]
            out_end = out_start + self._comp_n_outputs[ci]
            out_slice = fo[out_start:out_end]
            buf = dxdt[off:off + n_st]
            self._step_fns[ci](x_c, buf, inp[:self._comp_n_inputs[ci]], out_slice, t)

        # COI correction (JIT)
        _apply_coi(
            dxdt, x, self.n_gen,
            self._gen_offsets, self._gen_H,
            self._coi_2H, self._omega_b, self.delta_coi_idx,
        )

        return dxdt

    # ------------------------------------------------------------------
    # BDF-1 solver
    # ------------------------------------------------------------------

    def run(
        self,
        duration: float = 15.0,
        dt: float = 5e-4,
        csv_filename: str = 'simulation_results.csv',
    ) -> str:
        """Run fixed-step BDF-1 with JIT-accelerated RHS."""
        n_steps = int(round(duration / dt))
        log_every = max(1, n_steps // 200)

        print(f'[JitSolver] BDF-1: dt={dt}, n_steps={n_steps}')

        t_log = []
        x_log = []
        x = self.x0.copy()
        t = 0.0
        t_wall0 = time.time()

        for step in range(n_steps + 1):
            if step % log_every == 0:
                t_log.append(t)
                x_log.append(x.copy())

            # BDF-1 with one Newton correction
            x_pred = x + dt * self.rhs(t + dt, x)
            f_corr = self.rhs(t + dt, x_pred)
            x = x + dt * f_corr
            t += dt

        t_wall = time.time() - t_wall0
        print(f'[JitSolver] BDF-1 finished in {t_wall:.2f}s, '
              f'{len(t_log)} output points.')

        t_arr = np.array(t_log)
        x_arr = np.column_stack(x_log)
        csv_path = self._write_csv(t_arr, x_arr, csv_filename)
        return csv_path

    # ------------------------------------------------------------------
    # scipy Radau solver (uses same JIT RHS)
    # ------------------------------------------------------------------

    def run_scipy(
        self,
        duration: float = 15.0,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        max_step: float = 5e-3,
        csv_filename: str = 'simulation_results.csv',
    ) -> str:
        """Run scipy Radau integrator with JIT-accelerated RHS."""
        t_span = (0.0, duration)
        t_eval = np.arange(0.0, duration + 1e-9, max_step * 10)

        # Include fault boundary times
        for ev in self.fault_events:
            for tt in (ev['t_start'], ev['t_end']):
                if 0.0 < tt < duration:
                    t_eval = np.sort(np.append(t_eval, tt))

        # Break simulation into fault segments
        breakpoints = set([0.0, duration])
        for ev in self.fault_events:
            if 0.0 < ev['t_start'] < duration:
                breakpoints.add(ev['t_start'])
            if 0.0 < ev['t_end'] < duration:
                breakpoints.add(ev['t_end'])
        breakpoints = sorted(breakpoints)

        print(f'[JitSolver] scipy Radau: t=[0, {duration}] s, rtol={rtol}, '
              f'atol={atol}, max_step={max_step}')

        all_t = []
        all_x = []
        x_curr = self.x0.copy()
        t_wall0 = time.time()

        for seg_i in range(len(breakpoints) - 1):
            t0_seg = breakpoints[seg_i]
            t1_seg = breakpoints[seg_i + 1]
            t_seg_eval = t_eval[(t_eval >= t0_seg) & (t_eval <= t1_seg)]
            if len(t_seg_eval) == 0:
                t_seg_eval = np.array([t0_seg, t1_seg])

            sol = scipy.integrate.solve_ivp(
                self.rhs,
                (t0_seg, t1_seg),
                x_curr,
                method='Radau',
                t_eval=t_seg_eval,
                rtol=rtol, atol=atol,
                max_step=max_step,
            )
            if not sol.success:
                print(f'[JitSolver] WARNING: Radau failed on segment '
                      f'[{t0_seg}, {t1_seg}]: {sol.message}')

            all_t.append(sol.t)
            all_x.append(sol.y)
            x_curr = sol.y[:, -1].copy()

        t_all = np.concatenate(all_t)
        x_all = np.concatenate(all_x, axis=1)
        t_wall = time.time() - t_wall0

        print(f'[JitSolver] scipy Radau finished in {t_wall:.2f}s, '
              f'{t_all.size} output points.')

        csv_path = self._write_csv(t_all, x_all, csv_filename)
        return csv_path

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_csv(self, t_arr, x_arr, filename):
        csv_path = os.path.join(self.output_dir, filename)
        headers = ['t']
        for comp in self.components:
            for sname in comp.state_schema:
                headers.append(f'{comp.name}.{sname}')
        headers.append('delta_COI')
        for bus_id in self.bus_ids:
            headers.append(f'Vd_bus{bus_id}')
            headers.append(f'Vq_bus{bus_id}')
        for bus_id in self.bus_ids:
            headers.append(f'Vterm_bus{bus_id}')

        n_t = t_arr.size
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for k in range(n_t):
                t_k = t_arr[k]
                x_k = x_arr[:, k]
                Vd_k, Vq_k = self._solve_network(x_k, t_k)
                Vterm_k = np.sqrt(Vd_k**2 + Vq_k**2)
                row = [f'{t_k:.6f}']
                for j in range(self.n_diff):
                    row.append(f'{x_k[j]:.8e}')
                for bi in range(self.n_bus):
                    row.append(f'{Vd_k[bi]:.8e}')
                    row.append(f'{Vq_k[bi]:.8e}')
                for bi in range(self.n_bus):
                    row.append(f'{Vterm_k[bi]:.8e}')
                writer.writerow(row)

        print(f'[JitSolver] Results written to {csv_path}')
        return csv_path
