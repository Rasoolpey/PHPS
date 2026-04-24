"""Python ODE solver for the DAE simulation pipeline.

Reduces the full DAE (differential + algebraic bus-voltage states) to an
explicit ODE by solving the network Y-bus at every RHS evaluation, then
integrates with scipy Radau (scipy backend) or a fixed-step BDF-1 (jit backend).

Usage::

    solver = PyDAESolver(runner)          # runner must have build() completed
    csv_path = solver.run_scipy(rtol=1e-4, atol=1e-6)
    # or
    csv_path = solver.run_jit(dt=0.001)
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.integrate

from src.dirac.py_codegen import make_step_func, make_out_func


class PyDAESolver:
    """Network-reduced Python ODE solver for the PHS-DAE system.

    The DAE has the structure:
        ẋ = f(x, V)                    differential equations
        0 = I_inj(x) − Y · V          algebraic KCL constraint

    This class eliminates the algebraic layer by solving the network at each
    ODE step:
        V = (Y + Y_fault)⁻¹ · I_inj(x, t)

    Parameters
    ----------
    runner : DiracRunner
        A fully built DiracRunner (``build()`` must have been called).
    """

    def __init__(self, runner):
        dc = runner.dae_compiler
        self.output_dir = runner.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # --- System dimensions ---
        self.n_diff = dc.n_diff
        self.n_bus  = dc.n_bus
        self.delta_coi_idx = dc.delta_coi_idx

        # --- Network ---
        self.Y_full   = dc.Y_full.copy()   # complex (n_bus × n_bus)
        self.bus_map  = dict(dc.bus_map)   # bus_id → bus_index
        self.bus_ids  = sorted(dc.bus_map, key=dc.bus_map.__getitem__)

        # --- Components ---
        self.components   = dc.components
        self.state_offsets = dict(dc.state_offsets)
        self.wiring_map   = dict(dc.wiring_map)

        # --- Initial conditions ---
        self.x0     = runner.x0.copy()
        self.Vd0    = runner.Vd_init.copy()
        self.Vq0    = runner.Vq_init.copy()

        # --- Fault events ---
        self.fault_events: List[dict] = []
        for ev in dc.fault_events_dae:
            self.fault_events.append({
                'bus_idx': int(ev['bus_idx']),
                't_start': float(ev['t_start']),
                't_end':   float(ev['t_end']),
                'G': float(ev.get('G', ev.get('g', 0.0))),
                'B': float(ev.get('B', ev.get('b', 0.0))),
            })

        # --- COI info (for multi-gen COI correction) ---
        self._gen_comps = [
            (comp, self.state_offsets[comp.name])
            for comp in self.components
            if comp.component_role == 'generator'
            and 'omega' in comp.state_schema
        ]
        self._omega_b = 2.0 * math.pi * 60.0
        if self._gen_comps:
            g0 = self._gen_comps[0][0]
            ob = g0.params.get('omega_b', None)
            if ob is not None:
                try:
                    self._omega_b = float(eval(str(ob), {'M_PI': math.pi}))
                except Exception:
                    pass
        self._coi_2H = sum(
            2.0 * float(comp.params.get('H', 3.0))
            for comp, _ in self._gen_comps
        )

        # --- GENROU dq-frame stator parameters (for vd_dq / iq_dq computation) ---
        self._gen_dq_params: Dict[str, dict] = {}
        for comp in self.components:
            if comp.component_role != 'generator':
                continue
            if 'id_dq' not in [p[0] for p in comp.port_schema['out']]:
                continue
            p = comp.params
            ra    = float(p.get('ra', 0.0))
            xd_pp = float(p.get('xd_double_prime', p.get('xd1', 0.2)))
            xq_pp = float(p.get('xq_double_prime', xd_pp))
            xd_p  = float(p.get('xd_prime', p.get('xd1', 0.3)))
            xq_p  = float(p.get('xq_prime', xq_pp))
            xl    = float(p.get('xl', 0.0))
            bus_id = int(p['bus'])
            bi     = self.bus_map[bus_id]
            kd = (xd_pp - xl) / (xd_p - xl) if (xd_p - xl) != 0 else 1.0
            kq = (xq_pp - xl) / (xq_p - xl) if (xq_p - xl) != 0 else 1.0
            det = ra * ra + xd_pp * xq_pp
            sch = comp.state_schema
            self._gen_dq_params[comp.name] = {
                'bus_idx': bi,
                'ra': ra, 'xd_pp': xd_pp, 'xq_pp': xq_pp,
                'kd': kd, 'kq': kq, 'det': det,
                'idx_delta': sch.index('delta'),
                'idx_Eq':    sch.index('E_q_prime'),
                'idx_psi_d': sch.index('psi_d'),
                'idx_Ed':    sch.index('E_d_prime'),
                'idx_psi_q': sch.index('psi_q'),
                'out_id_dq': [p[0] for p in comp.port_schema['out']].index('id_dq'),
                'out_iq_dq': [p[0] for p in comp.port_schema['out']].index('iq_dq'),
            }

        # --- Pre-factor Y-bus for all fault segments ---
        self._Y_prefactored: Dict[frozenset, object] = {}
        self._prefactor_ybus(frozenset())  # nominal (no fault)
        for ev in self.fault_events:
            self._prefactor_ybus(frozenset([ev['bus_idx']]))

        # --- Build Python component functions ---
        print('[PySolver] Translating component kernels to Python...')
        self._step_fns = {}
        self._out_fns  = {}
        for comp in self.components:
            self._step_fns[comp.name] = make_step_func(comp)
            self._out_fns[comp.name]  = make_out_func(comp)
        print(f'[PySolver] {len(self.components)} components compiled.')

        # --- Output buffers ---
        self._outputs: Dict[str, np.ndarray] = {
            comp.name: np.zeros(len(comp.port_schema['out']))
            for comp in self.components
        }
        self._inputs: Dict[str, np.ndarray] = {
            comp.name: np.zeros(len(comp.port_schema['in']))
            for comp in self.components
        }

    # ------------------------------------------------------------------
    # Y-bus factorization
    # ------------------------------------------------------------------

    def _prefactor_ybus(self, active_faults: frozenset) -> None:
        """Compute and cache LU factorization for a given set of active faults."""
        Y_eff = self.Y_full.copy()
        for ev in self.fault_events:
            if ev['bus_idx'] in active_faults:
                bi = ev['bus_idx']
                Y_eff[bi, bi] += complex(ev['G'], ev['B'])
        key = active_faults
        self._Y_prefactored[key] = scipy.linalg.lu_factor(Y_eff)

    def _get_lu(self, t: float) -> object:
        """Return (possibly cached) LU decomposition for the current fault state."""
        active = frozenset(
            ev['bus_idx'] for ev in self.fault_events
            if ev['t_start'] <= t < ev['t_end']
        )
        if active not in self._Y_prefactored:
            self._prefactor_ybus(active)
        return self._Y_prefactored[active]

    # ------------------------------------------------------------------
    # Network solve
    # ------------------------------------------------------------------

    def _solve_network(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve network KCL → return (Vd, Vq) arrays [n_bus].

        Compute Norton current injections from generator states, then solve
        (Y + Y_fault) · V_complex = I_norton_complex.
        """
        I_inj = np.zeros(self.n_bus, dtype=complex)

        for comp in self.components:
            if comp.component_role != 'generator':
                continue
            off = self.state_offsets[comp.name]
            n_st = len(comp.state_schema)
            x_gen = x[off:off + n_st]
            bus_id = int(comp.params['bus'])
            bi = self.bus_map[bus_id]
            I_inj[bi] += comp.compute_norton_current(x_gen)

        lu = self._get_lu(t)
        V = scipy.linalg.lu_solve(lu, I_inj)

        return V.real.copy(), V.imag.copy()

    # ------------------------------------------------------------------
    # Component evaluation helpers
    # ------------------------------------------------------------------

    def _compute_dq_frame(
        self,
        comp_name: str,
        x: np.ndarray,
        Vd: np.ndarray,
        Vq: np.ndarray,
        off: int,
    ) -> Tuple[float, float, float, float]:
        """Return (vd_dq, vq_dq, id_dq, iq_dq) in machine dq frame."""
        p = self._gen_dq_params[comp_name]
        bi = p['bus_idx']
        delta = x[off + p['idx_delta']]
        sin_d = math.sin(delta)
        cos_d = math.cos(delta)

        Vd_bus = Vd[bi]
        Vq_bus = Vq[bi]

        vd_dq = Vd_bus * sin_d - Vq_bus * cos_d
        vq_dq = Vd_bus * cos_d + Vq_bus * sin_d

        Eq_p  = x[off + p['idx_Eq']]
        psi_d = x[off + p['idx_psi_d']]
        Ed_p  = x[off + p['idx_Ed']]
        psi_q = x[off + p['idx_psi_q']]

        kd, kq = p['kd'], p['kq']
        psi_d_pp = Eq_p * kd + psi_d * (1.0 - kd)
        psi_q_pp = -Ed_p * kq + psi_q * (1.0 - kq)

        ra    = p['ra']
        xd_pp = p['xd_pp']
        xq_pp = p['xq_pp']
        det   = p['det']

        rhs_d = vd_dq + psi_q_pp
        rhs_q = vq_dq - psi_d_pp

        id_dq = (-ra * rhs_d - xq_pp * rhs_q) / det
        iq_dq = (xd_pp * rhs_d - ra * rhs_q) / det

        return vd_dq, vq_dq, id_dq, iq_dq

    def _eval_wiring(
        self,
        expr: str,
        Vd_net: np.ndarray,
        Vq_net: np.ndarray,
        Vterm_net: np.ndarray,
        vd_dq_map: Dict[str, float],
        vq_dq_map: Dict[str, float],
    ) -> float:
        """Evaluate a wiring expression string in the current context."""
        # Build eval namespace
        ns = {
            'Vd_net': Vd_net,
            'Vq_net': Vq_net,
            'Vterm_net': Vterm_net,
            'math': math,
        }
        # Inject vd_dq_* / vq_dq_* locals
        for name, val in vd_dq_map.items():
            ns[f'vd_dq_{name}'] = val
        for name, val in vq_dq_map.items():
            ns[f'vq_dq_{name}'] = val
        # Inject outputs_COMPNAME arrays
        for comp in self.components:
            ns[f'outputs_{comp.name}'] = self._outputs[comp.name]

        try:
            return float(eval(expr, {'__builtins__': {}}, ns))
        except Exception:
            return 0.0

    def _gather_inputs(
        self,
        comp,
        Vd: np.ndarray,
        Vq: np.ndarray,
        Vterm: np.ndarray,
        vd_dq: Dict[str, float],
        vq_dq: Dict[str, float],
    ) -> np.ndarray:
        """Build the inputs array for a component from wiring expressions."""
        inp = self._inputs[comp.name]
        inp[:] = 0.0
        for i, (p_name, _, _) in enumerate(comp.port_schema['in']):
            key = (comp.name, p_name)
            expr = self.wiring_map.get(key, '0.0')
            inp[i] = self._eval_wiring(expr, Vd, Vq, Vterm, vd_dq, vq_dq)
        return inp

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Compute dx/dt for the network-reduced ODE.

        Steps:
          1. Solve network → Vd, Vq at each bus
          2. Compute dq-frame voltages and currents for each generator
          3. Compute component outputs (pass 1)
          4. Compute component dynamics (pass 2)
          5. Apply COI correction
        """
        Vd, Vq = self._solve_network(x, t)
        Vterm = np.sqrt(Vd**2 + Vq**2)

        # Precompute machine dq-frame voltages and refresh id_dq / iq_dq
        vd_dq: Dict[str, float] = {}
        vq_dq: Dict[str, float] = {}
        for comp in self.components:
            if comp.name not in self._gen_dq_params:
                continue
            off = self.state_offsets[comp.name]
            vd, vq, id_dq_v, iq_dq_v = self._compute_dq_frame(comp.name, x, Vd, Vq, off)
            vd_dq[comp.name] = vd
            vq_dq[comp.name] = vq
            p = self._gen_dq_params[comp.name]
            # Inject refreshed dq currents into outputs buffer (mirrors C++ pass)
            self._outputs[comp.name][p['out_id_dq']] = id_dq_v
            self._outputs[comp.name][p['out_iq_dq']] = iq_dq_v

        # Pass 1: evaluate all component outputs
        for comp in self.components:
            off = self.state_offsets[comp.name]
            n_st = len(comp.state_schema)
            x_c = x[off:off + n_st]
            inp = self._gather_inputs(comp, Vd, Vq, Vterm, vd_dq, vq_dq)
            out = self._outputs[comp.name]
            self._out_fns[comp.name](x_c, inp, out, t)
            # Re-inject refreshed dq currents (output fn may overwrite them)
            if comp.name in self._gen_dq_params:
                p = self._gen_dq_params[comp.name]
                vd, vq, id_dq_v, iq_dq_v = self._compute_dq_frame(
                    comp.name, x, Vd, Vq, off)
                out[p['out_id_dq']] = id_dq_v
                out[p['out_iq_dq']] = iq_dq_v

        # Pass 2: evaluate all component dynamics
        dxdt = np.zeros(self.n_diff)
        for comp in self.components:
            off = self.state_offsets[comp.name]
            n_st = len(comp.state_schema)
            x_c = x[off:off + n_st]
            inp = self._gather_inputs(comp, Vd, Vq, Vterm, vd_dq, vq_dq)
            out = self._outputs[comp.name]
            buf = dxdt[off:off + n_st]
            self._step_fns[comp.name](x_c, buf, inp, out, t)

        # COI correction
        if len(self._gen_comps) > 1:
            omega_coi = sum(
                2.0 * float(comp.params.get('H', 3.0)) * x[off + 1]
                for comp, off in self._gen_comps
            ) / self._coi_2H
            for _, off in self._gen_comps:
                dxdt[off] -= self._omega_b * (omega_coi - 1.0)
            dxdt[self.delta_coi_idx] = self._omega_b * (omega_coi - 1.0)
        else:
            dxdt[self.delta_coi_idx] = 0.0

        return dxdt

    # ------------------------------------------------------------------
    # Scipy Radau solver
    # ------------------------------------------------------------------

    def run_scipy(
        self,
        duration: Optional[float] = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        max_step: float = 5e-3,
        csv_filename: str = 'simulation_results_scipy.csv',
    ) -> str:
        """Run scipy Radau integrator and write results CSV.

        Parameters
        ----------
        duration : float, optional
            Simulation duration [s]. Defaults to the value stored by runner.
        rtol, atol : float
            Relative and absolute tolerances.
        max_step : float
            Maximum internal step size [s].
        csv_filename : str
            Output CSV filename (written to output_dir).

        Returns
        -------
        str
            Path to the CSV results file.
        """
        if duration is None:
            duration = getattr(self, '_duration', 15.0)

        t_span = (0.0, duration)
        t_eval = np.arange(0.0, duration + 1e-9, max_step * 10)
        # Make sure key fault times are included in output
        for ev in self.fault_events:
            for tt in (ev['t_start'], ev['t_end']):
                if 0.0 < tt < duration:
                    t_eval = np.sort(np.unique(np.append(t_eval, tt)))

        print(f'[PySolver] scipy Radau: t=[0, {duration}] s, '
              f'rtol={rtol}, atol={atol}, max_step={max_step}')

        # Split at fault boundaries to reinitialize (same as C++ IDA)
        t_boundaries = [0.0]
        for ev in self.fault_events:
            if 0.0 < ev['t_start'] < duration:
                t_boundaries.append(ev['t_start'])
            if 0.0 < ev['t_end'] < duration:
                t_boundaries.append(ev['t_end'])
        t_boundaries.append(duration)
        t_boundaries = sorted(set(t_boundaries))

        all_t = []
        all_x = []
        x_cur = self.x0.copy()
        t_wall0 = time.time()

        for seg_idx in range(len(t_boundaries) - 1):
            t0_seg = t_boundaries[seg_idx]
            t1_seg = t_boundaries[seg_idx + 1]
            t_eval_seg = t_eval[(t_eval >= t0_seg) & (t_eval <= t1_seg)]
            if t_eval_seg.size == 0:
                t_eval_seg = np.array([t0_seg, t1_seg])

            sol = scipy.integrate.solve_ivp(
                self.rhs,
                (t0_seg, t1_seg),
                x_cur,
                method='Radau',
                t_eval=t_eval_seg,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=False,
            )
            if not sol.success:
                print(f'[PySolver] WARNING: Radau failed in segment '
                      f'[{t0_seg:.4f}, {t1_seg:.4f}]: {sol.message}')

            all_t.append(sol.t)
            all_x.append(sol.y)  # shape (n_diff, n_t)
            x_cur = sol.y[:, -1].copy()

        t_wall = time.time() - t_wall0

        t_all = np.concatenate(all_t)
        x_all = np.concatenate(all_x, axis=1)   # (n_diff, N)

        print(f'[PySolver] scipy Radau finished in {t_wall:.2f}s, '
              f'{t_all.size} output points.')

        csv_path = self._write_csv(t_all, x_all, csv_filename)
        return csv_path

    # ------------------------------------------------------------------
    # Fixed-step BDF-1 (JIT-friendly / Numba-compatible skeleton)
    # ------------------------------------------------------------------

    def run_jit(
        self,
        duration: Optional[float] = None,
        dt: float = 5e-4,
        csv_filename: str = 'simulation_results_jit.csv',
    ) -> str:
        """Run fixed-step implicit BDF-1 (Backward Euler) with Python dynamics.

        This backend uses the same Python ODE RHS as the scipy backend but
        drives it with a simple fixed-step Newton iteration — structurally
        identical to the C++ BDF-1 emitted by DiracCompiler.

        Parameters
        ----------
        duration : float, optional
            Simulation duration [s].
        dt : float
            Fixed step size [s].
        csv_filename : str
            Output CSV filename.

        Returns
        -------
        str
            Path to the CSV results file.
        """
        if duration is None:
            duration = getattr(self, '_duration', 15.0)

        n_steps = int(round(duration / dt))
        log_every = max(1, n_steps // 200)

        print(f'[PySolver] JIT BDF-1: dt={dt}, n_steps={n_steps}')

        t_log = []
        x_log = []
        x = self.x0.copy()
        t = 0.0
        t_wall0 = time.time()

        for step in range(n_steps + 1):
            if step % log_every == 0:
                t_log.append(t)
                x_log.append(x.copy())

            # BDF-1: x_{n+1} − x_n − dt·f(t_{n+1}, x_{n+1}) = 0
            # Newton iteration (fixed-point approximation with one step)
            x_next = x + dt * self.rhs(t + dt, x)

            # One Newton correction step
            f_next = self.rhs(t + dt, x_next)
            x_next = x + dt * f_next

            x = x_next
            t += dt

        t_wall = time.time() - t_wall0
        print(f'[PySolver] JIT BDF-1 finished in {t_wall:.2f}s, '
              f'{len(t_log)} output points.')

        t_arr = np.array(t_log)
        x_arr = np.column_stack(x_log)  # (n_diff, N)
        csv_path = self._write_csv(t_arr, x_arr, csv_filename)
        return csv_path

    # ------------------------------------------------------------------
    # CSV output (same column layout as C++ binary)
    # ------------------------------------------------------------------

    def _write_csv(
        self,
        t_arr: np.ndarray,
        x_arr: np.ndarray,
        filename: str,
    ) -> str:
        """Write results to CSV matching the C++ binary output format."""
        csv_path = os.path.join(self.output_dir, filename)

        # Build column header matching C++ output
        headers = ['t']

        # State names per component
        for comp in self.components:
            off = self.state_offsets[comp.name]
            for sname in comp.state_schema:
                headers.append(f'{comp.name}.{sname}')
        headers.append('delta_COI')

        # Bus voltages
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

        print(f'[PySolver] Results written to {csv_path}')
        return csv_path
