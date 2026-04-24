"""
Dirac DAE Runner — build, compile and execute the DAE simulation.

This is the Dirac analogue of ``SimulationRunner``.  It reuses the
existing initialization pipeline (power flow + state refinement) and
then generates a DAE-based C++ kernel via ``DiracCompiler`` instead of
the ODE-based kernel from ``SystemCompiler``.

Usage::

    from src.dirac import DiracRunner

    runner = DiracRunner("cases/SMIB/system.json")
    runner.build(dt=0.0005, duration=10.0)
    runner.run()
"""

from __future__ import annotations

import math
import os
import subprocess
import numpy as np
import time

from src.compiler import SystemCompiler
from src.initialization import Initializer
from src.errors import FrameworkError
from src.dirac.dae_compiler import (
    DiracCompiler,
    get_dae_solver_label,
    normalize_dae_solver_name,
    PYTHON_SOLVERS,
)


class DiracRunner:
    """End-to-end DAE simulation pipeline.

    Parameters
    ----------
    json_path : str
        Path to the system JSON (``cases/SMIB/system.json``).
    output_dir : str, optional
        Where to write the generated C++ and CSV results.
        If ``None``, defaults to ``outputs/<case_name>``.
    events : list, optional
        Event dicts (``BusFault``, ``Toggler``, etc.).
    """

    def __init__(self, json_path: str, output_dir: str = None,
                 events: list = None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found")

        self.json_path = json_path

        # Set up output directory
        if output_dir is None:
            case_name = os.path.basename(os.path.dirname(os.path.abspath(json_path)))
            output_dir = os.path.join(os.path.dirname(os.path.abspath(json_path)),
                                      '..', '..', 'outputs', f"{case_name}_dirac")
            output_dir = os.path.abspath(output_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.source_path = os.path.join(self.output_dir, "dae_sim.cpp")
        import platform as _plt
        _bin_ext = ".exe" if _plt.system() == "Windows" else ""
        self.binary_path = os.path.join(self.output_dir, f"dae_sim{_bin_ext}")
        self.csv_path = os.path.join(self.output_dir, "simulation_results.csv")

        # DAE compiler (handles the Dirac-specific C++ generation)
        self.dae_compiler = DiracCompiler(json_path)

        # Keep reference to the base compiler (for initialization)
        self.base_compiler = self.dae_compiler.base_compiler

        # Inject events (same as SimulationRunner)
        if events:
            self._inject_events(events)

        # Filled by build()
        self.x0 = None
        self.Vd_init = None
        self.Vq_init = None

    def _inject_events(self, events):
        """Inject fault/toggler events into both base and DAE compilers."""
        toggler_events = [
            {k: v for k, v in ev.items() if k != "type"}
            for ev in events if ev.get("type") == "Toggler"
        ]
        if toggler_events:
            existing = self.base_compiler.data.get("Toggler", [])
            self.base_compiler.data["Toggler"] = existing + toggler_events

        bus_fault_events = [
            {k: v for k, v in ev.items() if k != "type"}
            for ev in events if ev.get("type") == "BusFault"
        ]
        if bus_fault_events:
            existing = self.base_compiler.data.get("BusFault", [])
            self.base_compiler.data["BusFault"] = existing + bus_fault_events

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def build(self, dt: float = 0.0005, duration: float = 10.0,
              solver: str = "bdf1"):
        """Full build pipeline: init + C++ generation + compilation.

        This mirrors ``SimulationRunner.build()`` but generates a DAE kernel
        instead of the ODE kernel.

        Pipeline:
          1. Build component structure (via base compiler)
          2. Validate SystemGraph wiring
          3. Run power flow + initialize component states
          4. State refinement (Kron equilibrium, exciter, governor)
          5. Extract full-bus initial voltages from power flow
          6. Generate DAE C++ kernel
          7. Compile with g++

        Parameters
        ----------
        dt : float
            Time step for the implicit solver.
        duration : float
            Total simulation time [s].
        solver : str
            Solver backend: ``"bdf1"``, ``"ida"`` (SUNDIALS IDA),
            or ``"midpoint"`` (structure-preserving implicit midpoint).

        Returns
        -------
        x0 : ndarray
            Initial differential state vector.
        Vd_init, Vq_init : ndarray
            Initial bus voltages (all buses, ordered by bus_map).
        """
        solver = normalize_dae_solver_name(solver)
        solver_label = get_dae_solver_label(solver)
        self._solver = solver
        self._dt = dt
        self._duration = duration

        print(f"[DiracRunner] Building DAE system from {self.json_path}...")
        print(f"[DiracRunner] Solver backend: {solver_label}")
        graph = self.base_compiler.graph
        print(f"[DiracRunner] SystemGraph: {graph}")

        # 1. Build Structure
        print("[DiracRunner] Phase 1: Building component structure...")
        self.base_compiler.build_structure()

        # 2. Validate
        print("[DiracRunner] Phase 2: Validating system graph...")
        try:
            graph.validate()
            print("[DiracRunner] Graph validation passed.")
        except FrameworkError as e:
            print(f"[DiracRunner] Graph validation ERROR:\n{e}")
            raise

        # 3. Initialize (power flow + component states)
        print("[DiracRunner] Phase 3: Initializing system states (Power Flow)...")
        init = Initializer(self.base_compiler)
        x0 = init.run()
        # DAE pipeline uses the full Y-bus — Kron network reduction is never
        # needed, so finalize_network() (which builds the ODE Z-bus) is skipped.

        # 4. DAE pipeline does NOT use Kron reduction at any stage.
        #    The full Y-bus is the authoritative network model throughout.
        #    Generator flux states from init.run() (phasor-based) are already
        #    self-consistent with the PF voltages; the DAE-consistent voltage
        #    solve below (phase 6) will correct any remaining mismatch.
        #    Nothing to do here.

        # Ensure x0 has the correct size (including delta_COI)
        if len(x0) < self.base_compiler.total_states:
            x0 = np.append(x0, np.zeros(
                self.base_compiler.total_states - len(x0)))

        self.x0 = x0

        # 5. Build DAE compiler structure (mirrors base compiler's structure)
        #    Must happen BEFORE computing bus voltages, since build() populates
        #    the bus_map and bus_indices.
        #
        #    CRITICAL: update Bus v0/a0 with converged PF voltages first.
        #    The DAE Y-bus converts PQ loads to constant-impedance via
        #    Y_load = (P - jQ) / V0^2.  If v0 comes from stale initial
        #    guesses (often much lower than PF solution), load admittances
        #    are over-sized, causing generators to be overloaded in the DAE.
        ybus_map = self.base_compiler.ybus_builder.bus_map
        for bus_entry in self.base_compiler.data.get('Bus', []):
            bus_id = bus_entry['idx']
            ybi = ybus_map.get(bus_id)
            if ybi is not None and ybi < len(init.pf.V):
                bus_entry['v0'] = float(init.pf.V[ybi])
                bus_entry['a0'] = float(init.pf.theta[ybi])

        print("[DiracRunner] Building DAE structure (full Y-bus)...")
        self.dae_compiler.build()

        # 6. Compute DAE-consistent initial bus voltages.
        #
        #     Solve V = Y_bus^{-1} * I_norton iteratively until PV-bus
        #     voltage setpoints are met.  Generator flux states (Eq' etc.)
        #     are adjusted by each generator's adjust_for_target_voltage()
        #     until convergence.  Only Tm, governor and exciter are then
        #     rebalanced to the final DAE-consistent operating point.
        print("[DiracRunner] Computing DAE-consistent initial bus voltages...")
        self._compute_full_bus_voltages(init)            # PF voltages (fallback)
        self._compute_dae_consistent_voltages(init)      # overwrite with network solve

        print("[DiracRunner] Rebalance Tm=Te + exciter/governor...")
        self._rebalance_te_for_fullbus(init)

        # --- Outer iteration: re-solve bus voltages after rebalance ---
        # The rebalance changes generator flux states (including Eq'),
        # which changes Norton currents.  Re-solving the bus voltages
        # and re-rebalancing gives a mutually consistent set.
        for outer in range(3):
            x0_before = self.x0.copy()
            self._compute_dae_consistent_voltages(init)
            self._rebalance_te_for_fullbus(init)
            dx = float(np.max(np.abs(self.x0 - x0_before)))
            print(f"  [Outer iter {outer+1}] max |dx| = {dx:.2e}")
            if dx < 1e-6:
                print(f"  Voltage-rebalance converged in {outer+1} "
                      f"iteration(s).")
                break

        # Final voltage sync: V = Y^{-1} · I_norton(x0_final).
        # The outer loop exits with V from the LAST voltage solve, but x0
        # was modified by the LAST rebalance.  One more solve ensures
        # the KCL residual is exactly zero, making IDACalcIC a no-op.
        self._sync_voltages_to_states(init)

        # Diagnostic: show initial bus voltages
        for bus_id in self.dae_compiler.bus_indices:
            bi = self.dae_compiler.bus_map[bus_id]
            Vm = math.sqrt(self.Vd_init[bi]**2 + self.Vq_init[bi]**2)
            Va = math.degrees(math.atan2(self.Vq_init[bi], self.Vd_init[bi]))
            tag = " (slack)" if bus_id in self.dae_compiler.slack_bus_ids else ""
            print(f"  Bus {bus_id}: |V|={Vm:.5f}, ang={Va:.3f}deg{tag}")

        # --- Optional omega perturbation (set all generators to omega0_perturb) ---
        _omega0_perturb = getattr(self, '_omega0_perturb', None)
        if _omega0_perturb is not None:
            for comp in self.base_compiler.components:
                if comp.component_role == 'generator' and 'omega' in comp.state_schema:
                    off = self.base_compiler.state_offsets[comp.name]
                    idx = comp.state_schema.index('omega')
                    self.x0[off + idx] = float(_omega0_perturb)
                    print(f"  [Perturb] {comp.name}: omega0 set to {_omega0_perturb}")

        # 7. Generate C++ (skipped for Python solvers)
        # Refresh Pref/Vref in the wiring map so the DAE-consistent values
        # computed by _rebalance_te_for_fullbus (which updated comp.params) are
        # actually compiled into the C++ binary — not the stale Kron-based ones
        # that were embedded when build() called _refresh_control_params().
        self.dae_compiler._refresh_control_params()

        if solver in PYTHON_SOLVERS:
            print(f"[DiracRunner] Python solver '{solver}' — skipping C++ generation.")
            print("[DiracRunner] Build successful.")
            return self.x0, self.Vd_init, self.Vq_init

        print(f"[DiracRunner] Generating DAE C++ kernel (solver={solver})...")
        cpp_code = self.dae_compiler.generate_cpp(
            dt=dt, duration=duration,
            x0=self.x0,
            Vd_init=self.Vd_init,
            Vq_init=self.Vq_init,
            solver=solver,
        )

        with open(self.source_path, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        print(f"[DiracRunner] C++ source written to {self.source_path}")

        # 8. Compile
        print(f"[DiracRunner] Compiling binary: {self.binary_path}")
        import platform as _plt2, copy as _copy, glob as _glob
        _GPP = "g++"
        _SUNDIALS_INC = []
        _SUNDIALS_LIB = []
        _SUNDIALS_LIB_DIR = None
        _COMPILE_ENV = None
        if _plt2.system() == "Windows":
            # Search for a UCRT64 g++/SUNDIALS installation.
            # Priority: RTools44 → MSYS2.  The first one that has g++.exe wins.
            _UCRT64_CANDIDATES = [
                (r"C:\rtools44\ucrt64\bin", r"C:\rtools44\usr\bin"),
                (r"C:\msys64\ucrt64\bin",   r"C:\msys64\usr\bin"),
            ]
            for _UCRT64_BIN, _USR_BIN in _UCRT64_CANDIDATES:
                _UCRT64_GPP = os.path.join(_UCRT64_BIN, "g++.exe")
                _UCRT64_ROOT = os.path.dirname(_UCRT64_BIN)
                _SUNDIALS_HEADER = os.path.join(_UCRT64_ROOT, "include", "ida", "ida.h")
                if os.path.exists(_UCRT64_GPP) and os.path.exists(_SUNDIALS_HEADER):
                    _GPP = _UCRT64_GPP
                    _SUNDIALS_LIB_DIR = os.path.join(_UCRT64_ROOT, "lib")
                    _SUNDIALS_INC = [f"-I{_UCRT64_ROOT}\\include"]
                    _SUNDIALS_LIB = [f"-L{_SUNDIALS_LIB_DIR}"]
                    _COMPILE_ENV = _copy.copy(os.environ)
                    _COMPILE_ENV["PATH"] = (
                        _UCRT64_BIN + os.pathsep
                        + _USR_BIN + os.pathsep
                        + _COMPILE_ENV.get("PATH", "")
                    )
                    break
        # sundials_core only exists in SUNDIALS >= 7.0; on 6.x its symbols
        # are bundled into the per-module libs. Only link it if we can see it.
        def _has_sundials_core():
            if _SUNDIALS_LIB_DIR is None:
                return True  # unknown lib dir — assume system has it
            patterns = ("libsundials_core.*", "sundials_core.*")
            return any(_glob.glob(os.path.join(_SUNDIALS_LIB_DIR, p)) for p in patterns)
        cmd = [_GPP, "-O3"] + _SUNDIALS_INC + [
            self.source_path, "-o", self.binary_path
        ] + _SUNDIALS_LIB + ["-lm"]
        if solver.lower() == "ida":
            cmd += [
                "-lsundials_ida",
                "-lsundials_nvecserial",
                "-lsundials_sunlinsoldense",
                "-lsundials_sunmatrixdense",
            ]
            if _has_sundials_core():
                cmd.append("-lsundials_core")
        result = subprocess.run(cmd, capture_output=True,
                                encoding="utf-8", errors="replace",
                                env=_COMPILE_ENV)
        if result.returncode != 0:
            print("COMPILATION FAILED:")
            print(result.stderr)
            raise RuntimeError("C++ compilation failed")

        print("[DiracRunner] Build successful.")
        return self.x0, self.Vd_init, self.Vq_init

    def _rebalance_te_for_fullbus(self, init: Initializer):
        """Rebalance Tm = Te and exciter Vref using DAE-consistent voltages.

        After Kron-refined initialisation the generator states are tuned for
        a reduced network.  The DAE solver sees DAE-consistent voltages
        (computed by ``_compute_dae_consistent_voltages``), which produce a
        different Te via the stator algebraic equations.

        This method is **model-agnostic**: each generator component implements
        ``rebalance_for_bus_voltage(x, V_complex)`` which returns updated
        states and a targets dict.  The runner then generically passes
        targets to associated governors and exciters.
        """
        bc = self.base_compiler
        dc = self.dae_compiler

        for comp in bc.components:
            if comp.component_role != 'generator':
                continue

            off = bc.state_offsets[comp.name]
            n_st = len(comp.state_schema)
            x_gen = self.x0[off:off + n_st]

            # Get DAE-consistent bus voltage as complex number
            bus_id = comp.params['bus']
            bi = dc.bus_map[bus_id]
            V_complex = complex(self.Vd_init[bi], self.Vq_init[bi])

            # --- Generic component call ---
            x_new, targets = comp.rebalance_for_bus_voltage(x_gen, V_complex)
            self.x0[off:off + n_st] = x_new

            if not targets:
                continue  # non-rebalancing component

            Te = targets['Tm']
            Efd_req = targets['Efd']
            Vt_dae = targets['Vt']
            old_Tm = comp.params.get('Tm0', Te)
            comp.params['Tm0'] = float(Te)
            print(f"  {comp.name} bus {bus_id}: Te(DAE)={Te:.6f}, "
                  f"old_Tm={old_Tm:.6f}, diff={Te - old_Tm:.6f}, "
                  f"Vt_dae={Vt_dae:.5f}, Efd_req={Efd_req:.6f}")

            # --- Update governor: set its states so that Tm = Te ---
            for gov_comp in bc.components:
                if gov_comp.component_role != 'governor':
                    continue
                gen_for_gov = init._get_generator_for_comp(gov_comp.name)
                if gen_for_gov is None or gen_for_gov.name != comp.name:
                    continue

                gov_off = bc.state_offsets[gov_comp.name]
                n_gov = len(gov_comp.state_schema)

                x_gov = gov_comp.init_from_targets(targets)
                self.x0[gov_off:gov_off + n_gov] = x_gov

                if hasattr(gov_comp, 'params') and 'Pref' in gov_comp.params:
                    print(f"    Governor {gov_comp.name}: "
                          f"Pref={gov_comp.params['Pref']:.6f}")

            # --- Update exciter: set Vref to match DAE terminal voltage ---
            for exc_comp in bc.components:
                if exc_comp.component_role != 'exciter':
                    continue
                gen_for_exc = init._get_generator_for_comp(exc_comp.name)
                if gen_for_exc is None or gen_for_exc.name != comp.name:
                    continue

                exc_off = bc.state_offsets[exc_comp.name]
                n_exc = len(exc_comp.state_schema)

                x_exc = exc_comp.init_from_targets(targets)
                self.x0[exc_off:exc_off + n_exc] = x_exc

                print(f"    Exciter {exc_comp.name}: "
                      f"Vref={exc_comp.params.get('Vref', '?'):.6f}, "
                      f"Efd={Efd_req:.6f}")

        # Refresh Pref / Vref in the DAE wiring map
        self.dae_compiler._refresh_control_params()

    def _sync_voltages_to_states(self, init: Initializer):
        """Final voltage sync using the same algebraic constraints as the DAE.

        After the voltage-rebalance loop, x0 was updated by the last
        rebalance but V was not re-computed.  This one-shot solve closes
        the gap so that the (x0, V) pair satisfies KCL exactly, making
        IDACalcIC a no-op.

        Static slack buses are held at their reference voltages, exactly as
        in the generated C++ DAE residual. Slack buses that host dynamic
        generators remain KCL buses.
        """
        V = self._solve_dae_network_voltages()

        dVd = float(np.max(np.abs(V.real - self.Vd_init)))
        dVq = float(np.max(np.abs(V.imag - self.Vq_init)))
        print(f"  [Final sync] max |dVd|={dVd:.2e}, max |dVq|={dVq:.2e}")

        self.Vd_init = V.real.copy()
        self.Vq_init = V.imag.copy()

    def _solve_dae_network_voltages(self, V_prev=None) -> np.ndarray:
        """Solve the DAE network algebraics for the current differential state.

        Matches the generated DAE residual exactly:

        - Dynamic generator buses use KCL.
        - Static slack buses are pinned to their reference voltage.

        Parameters
        ----------
        V_prev : ndarray or None
            Previous bus voltage estimate (complex, n_bus).  Used to compute
            Norton source currents for components whose ``compute_norton_current``
            accepts a ``V_bus_complex`` kwarg (e.g. current-source generators
            like DFIG_PHS whose Norton source current includes a Y_N*V term).
            When ``None``, voltages from ``self.Vd_init`` / ``self.Vq_init``
            are used, falling back to zero if those are not yet set.
        """
        dc = self.dae_compiler
        bc = self.base_compiler
        n = dc.n_bus

        # Build voltage estimate for Norton-source correction
        if V_prev is None:
            if self.Vd_init is not None and self.Vq_init is not None:
                V_prev = self.Vd_init + 1j * self.Vq_init
            else:
                V_prev = np.zeros(n, dtype=complex)

        Y = dc.Y_full.copy()
        I_inj = np.zeros(n, dtype=complex)

        for comp in bc.components:
            if not hasattr(comp, 'compute_norton_current'):
                continue
            if comp.component_role not in ('generator', 'renewable_controller'):
                continue
            bus_id = comp.params.get('bus')
            if bus_id is None:
                continue
            bi = dc.bus_map.get(bus_id)
            if bi is None:
                continue
            off = bc.state_offsets[comp.name]
            x_gen = self.x0[off:off + len(comp.state_schema)]

            # Components with V_bus_complex param need voltage for Norton source
            try:
                I_inj[bi] += comp.compute_norton_current(
                    x_gen, V_bus_complex=complex(V_prev[bi]))
            except TypeError:
                I_inj[bi] += comp.compute_norton_current(x_gen)

        dynamic_gen_buses = set(dc.gen_bus_map.values())
        fixed_bus_ids = [
            bus_id for bus_id in dc.slack_bus_ids
            if bus_id not in dynamic_gen_buses and bus_id in dc.bus_map
        ]

        if not fixed_bus_ids:
            return np.linalg.solve(Y, I_inj)

        V = np.zeros(n, dtype=complex)
        fixed_idx = [dc.bus_map[bus_id] for bus_id in fixed_bus_ids]
        free_idx = [i for i in range(n) if i not in fixed_idx]

        for bus_id, bi in zip(fixed_bus_ids, fixed_idx):
            V[bi] = complex(dc.slack_V[bus_id])

        if free_idx:
            Y_ff = Y[np.ix_(free_idx, free_idx)]
            rhs = I_inj[free_idx].copy()
            if fixed_idx:
                rhs -= Y[np.ix_(free_idx, fixed_idx)] @ V[fixed_idx]
            V[free_idx] = np.linalg.solve(Y_ff, rhs)

        return V

    def _compute_full_bus_voltages(self, init: Initializer):
        """Extract initial bus voltages for ALL buses from power flow.

        Unlike the ODE runner which only needs active (Kron-reduced) buses,
        we need voltages at every bus in the full Y-bus.
        """
        V_pf = init.pf.V         # magnitude per bus (indexed by ybus bus order)
        theta_pf = init.pf.theta  # angle per bus

        ybus_map = self.base_compiler.ybus_builder.bus_map  # bus_id → ybus_idx

        n_bus = len(self.dae_compiler.bus_indices)
        Vd = np.zeros(n_bus)
        Vq = np.zeros(n_bus)

        for bus_id in self.dae_compiler.bus_indices:
            bi = self.dae_compiler.bus_map[bus_id]  # DAE bus index
            ybi = ybus_map.get(bus_id)
            if ybi is not None and ybi < len(V_pf):
                Vd[bi] = V_pf[ybi] * math.cos(theta_pf[ybi])
                Vq[bi] = V_pf[ybi] * math.sin(theta_pf[ybi])
            else:
                # Fallback: use bus nominal voltage
                Vd[bi] = 1.0
                Vq[bi] = 0.0

        self.Vd_init = Vd
        self.Vq_init = Vq

    def _compute_dae_consistent_voltages(self, init: Initializer):
        """Compute initial bus voltages that satisfy KCL at t = 0,
        with iterative PV-bus voltage regulation and static slack constraints.

        The DAE algebraic system is solved repeatedly. Static slack buses are
        pinned to their reference voltage, while generator buses use KCL.
        After each solve, *only* the generator Eq' state is adjusted (under-
        relaxed) so that bus voltage magnitudes converge to their power-flow
        setpoints.  Flux states (psi_d, Ed', psi_q) are left unchanged during
        the iteration to avoid accumulated inconsistency from the alpha
        relaxation.  Once converged, flux states are recomputed for ODE
        steady-state consistency.

        This ensures PV-bus generators produce the correct reactive power
        to maintain voltage even though the DAE network uses constant-
        impedance loads (no algebraic PV constraint at run-time).
        """
        dc = self.dae_compiler
        bc = self.base_compiler
        n = dc.n_bus
        ybus_map = bc.ybus_builder.bus_map

        # ---- Identify PV-bus voltage targets (including slack bus) ----
        pv_targets = {}   # bus_id → V_setpoint (magnitude)
        gen_at_bus = {}   # bus_id → [(comp, off, n_st), ...]
        for comp in bc.components:
            if comp.component_role != 'generator':
                continue
            bus_id = comp.params['bus']
            ybi = ybus_map.get(bus_id, 0)
            pv_targets[bus_id] = float(init.pf.V[ybi])
            off = bc.state_offsets[comp.name]
            n_st = len(comp.state_schema)
            gen_at_bus.setdefault(bus_id, []).append((comp, off, n_st))

        # ---- Adaptive relaxation parameters ----
        # Compute a safe alpha based on the maximum generator Norton
        # admittance relative to the Y-bus diagonal.  Generators with very
        # small sub-transient impedance (large 1/xd'') create high-gain
        # feedback that can destabilise the fixed-point iteration.
        alpha_base = 0.3
        max_gain = 0.0
        for bus_id in pv_targets:
            bi = dc.bus_map[bus_id]
            Y_diag = abs(dc.Y_full[bi, bi])
            for comp, off, n_st in gen_at_bus[bus_id]:
                xd_pp = comp.params.get('xd_double_prime',
                                        comp.params.get('xd1', 0.2))
                ra = comp.params.get('ra', 0.0)
                y_gen = 1.0 / math.sqrt(ra**2 + xd_pp**2) if xd_pp > 0 else 5.0
                gain = y_gen / Y_diag if Y_diag > 0 else 1.0
                max_gain = max(max_gain, gain)

        # Choose alpha so that alpha * max_gain < 0.5 (safe margin)
        if max_gain > 0:
            alpha = min(alpha_base, 0.45 / max_gain)
        else:
            alpha = alpha_base
        alpha = max(alpha, 0.01)  # floor

        max_iter = 200   # allow more iterations for small alpha
        tol = 1e-4       # PV bus voltage tolerance [pu]
        V_complex = None

        # Bound on plausible bus voltage magnitude during the iteration.
        # Anything above this means the scheme has diverged — fail fast instead
        # of letting values run to inf/NaN and poisoning downstream codegen.
        V_DIVERGE_LIMIT = 1e3

        for itr in range(max_iter):
            # 1. Solve the DAE algebraic network for the current generator state
            V_complex = self._solve_dae_network_voltages()

            # Lightweight per-iteration diagnostic: first 3 iters then every 50.
            if itr < 3 or itr % 50 == 0:
                mx = float(np.max(np.abs(V_complex)))
                worst = 0.0
                for bid, Vt in pv_targets.items():
                    worst = max(worst, abs(Vt - abs(V_complex[dc.bus_map[bid]])))
                print(f"    [PV iter {itr:3d}] max|V|={mx:.4f}  max|dV|={worst:.4e}")

            # Guard: abort early on non-finite state or runaway magnitudes.
            if not np.all(np.isfinite(V_complex)):
                bad = [bus_id for bus_id in dc.bus_indices
                       if not np.isfinite(V_complex[dc.bus_map[bus_id]])]
                raise RuntimeError(
                    f"PV-bus init diverged at iteration {itr}: non-finite bus "
                    f"voltage(s) at {bad}. This is usually caused by an "
                    f"unstable Eq' update in a generator's "
                    f"adjust_for_target_voltage(). Check the GENROU/exciter "
                    f"wiring and the PV-bus relaxation gain."
                )
            max_abs = float(np.max(np.abs(V_complex)))
            if max_abs > V_DIVERGE_LIMIT:
                raise RuntimeError(
                    f"PV-bus init diverged at iteration {itr}: max |V| = "
                    f"{max_abs:.3e} (limit {V_DIVERGE_LIMIT:.0e}). The "
                    f"fixed-point iteration is unstable — reduce the "
                    f"adjustment gain or fix the generator sensitivity model."
                )

            # Update stored voltages so next solve uses the latest V
            self.Vd_init = V_complex.real.copy()
            self.Vq_init = V_complex.imag.copy()

            # 2. Check PV-bus convergence
            # Use np.nanmax-style accumulation: Python's max(0.0, nan) returns
            # 0.0 (NaN comparisons are False), which would silently "converge"
            # on a poisoned state. Track NaN explicitly.
            max_err = 0.0
            saw_nan = False
            for bus_id, V_target in pv_targets.items():
                bi = dc.bus_map[bus_id]
                err = abs(V_target - abs(V_complex[bi]))
                if not math.isfinite(err):
                    saw_nan = True
                    continue
                if err > max_err:
                    max_err = err
            if saw_nan:
                raise RuntimeError(
                    f"PV-bus init produced non-finite error at iteration "
                    f"{itr}. State or target became NaN/Inf."
                )

            if max_err < tol:
                print(f"    PV-bus voltage converged in {itr + 1} "
                      f"iteration(s), max |dV| = {max_err:.2e} "
                      f"(alpha={alpha:.4f})")
                break

            if itr == max_iter - 1:
                print(f"    PV-bus voltage: {max_iter} iters, "
                      f"max |dV| = {max_err:.5f} (not converged, "
                      f"alpha={alpha:.4f})")
                break

            # 3. Adjust generator states at PV buses.
            #    Delegate the state adjustment to each component's
            #    adjust_for_target_voltage() method, which handles its
            #    own state layout (GENROU adjusts Eq'/fluxes, GENCLS and
            #    DFIG return unchanged).  Alpha-relaxation is achieved by
            #    passing a relaxed target: V_relaxed = V_mag + alpha*dV.
            for bus_id in pv_targets:
                V_target = pv_targets[bus_id]
                bi = dc.bus_map[bus_id]
                y_diag = abs(dc.Y_full[bi, bi])
                V_mag = abs(V_complex[bi])
                dV = V_target - V_mag
                V_relaxed = V_mag + alpha * dV
                for comp, off, n_st in gen_at_bus[bus_id]:
                    x_gen = self.x0[off:off + n_st]
                    # flux_update=False inside the iteration — see GENROU
                    # adjust_for_target_voltage docstring.
                    x_adj = comp.adjust_for_target_voltage(
                        x_gen, V_complex[bi], V_relaxed,
                        y_diag=y_diag, flux_update=False)
                    self.x0[off:off + n_st] = x_adj

        # ---- Post-convergence: recompute consistent flux states ----
        # Now that Eq' values have converged, update psi_d, Ed', psi_q
        # on the ODE steady-state manifold using the converged bus voltage.
        if V_complex is not None:
            for bus_id in pv_targets:
                bi = dc.bus_map[bus_id]
                y_diag = abs(dc.Y_full[bi, bi])
                for comp, off, n_st in gen_at_bus[bus_id]:
                    x_gen = self.x0[off:off + n_st]
                    x_adj = comp.adjust_for_target_voltage(
                        x_gen, V_complex[bi], pv_targets[bus_id],
                        y_diag=y_diag)
                    # Apply full adjustment (no relaxation needed at convergence;
                    # dV ≈ 0 so Eq' is unchanged, only fluxes are updated)
                    self.x0[off:off + n_st] = x_adj

        # ---- Store final voltages ----
        self.Vd_init = V_complex.real.copy()
        self.Vq_init = V_complex.imag.copy()

        # Diagnostic: compare with PF values
        print("  DAE-consistent vs PF voltages:")
        for bus_id in dc.bus_indices:
            bi = dc.bus_map[bus_id]
            Vm_dae = abs(V_complex[bi])
            ybi = ybus_map.get(bus_id, 0)
            Vm_pf = init.pf.V[ybi] if ybi < len(init.pf.V) else 1.0
            tag = " (slack)" if bus_id in dc.slack_bus_ids else ""
            pv = " (PV)" if bus_id in pv_targets else ""
            print(f"    Bus {bus_id}: |V|_pf={Vm_pf:.5f} -> |V|_dae={Vm_dae:.5f} "
                  f"(d={Vm_dae - Vm_pf:+.5f}){tag}{pv}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, cwd: str = None) -> str:
        """Execute the simulation and return the CSV path.

        For C++ solvers (bdf1, ida, midpoint) this runs the compiled binary.
        For Python solvers (scipy, jit) this runs the Python ODE solver.
        """
        if getattr(self, '_solver', None) in PYTHON_SOLVERS:
            return self._run_python()
        return self._run_cpp(cwd)

    def _run_python(self) -> str:
        """Run the Python ODE solver (scipy or jit backend)."""
        if self._solver == 'jit':
            return self._run_jit()

        from src.dirac.py_solver import PyDAESolver

        solver = PyDAESolver(self)
        solver._duration = self._duration

        if self._solver == 'scipy':
            csv_path = solver.run_scipy(
                duration=self._duration,
                rtol=1e-4,
                atol=1e-6,
                max_step=5e-3,
                csv_filename='simulation_results.csv',
            )
        else:
            csv_path = solver.run_jit(
                duration=self._duration,
                dt=self._dt,
                csv_filename='simulation_results.csv',
            )
        self.csv_path = csv_path
        return csv_path

    def _run_jit(self) -> str:
        """Run the Numba JIT-accelerated BDF-1 solver."""
        from src.dirac.jit_solver import JitDAESolver

        solver = JitDAESolver(self)
        csv_path = solver.run(
            duration=self._duration,
            dt=self._dt,
            csv_filename='simulation_results.csv',
        )
        self.csv_path = csv_path
        return csv_path

    def _run_cpp(self, cwd: str = None) -> str:
        """Execute the compiled DAE binary and return the CSV path.

        Parameters
        ----------
        cwd : str, optional
            Working directory for the binary (defaults to output_dir).

        Returns
        -------
        csv_path : str
            Path to the CSV results file.
        """
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"Binary not found at {self.binary_path}. Call build() first.")

        if cwd is None:
            cwd = self.output_dir

        print(f"[DiracRunner] Running DAE simulation...")
        t0 = time.time()

        import platform as _plt3, copy as _copy3
        _run_env = None
        if _plt3.system() == "Windows":
            _UCRT64_CANDIDATES = [
                r"C:\rtools44\ucrt64\bin",
                r"C:\msys64\ucrt64\bin",
            ]
            for _UCRT64_BIN in _UCRT64_CANDIDATES:
                _UCRT64_ROOT = os.path.dirname(_UCRT64_BIN)
                _SUNDIALS_HEADER = os.path.join(_UCRT64_ROOT, "include", "ida", "ida.h")
                if os.path.isdir(_UCRT64_BIN) and os.path.exists(_SUNDIALS_HEADER):
                    _run_env = _copy3.copy(os.environ)
                    _run_env["PATH"] = _UCRT64_BIN + os.pathsep + _run_env.get("PATH", "")
                    break

        result = subprocess.run(
            [self.binary_path],
            cwd=cwd,
            capture_output=True,
            encoding="utf-8", errors="replace",
            env=_run_env,
            timeout=600,
        )

        elapsed = time.time() - t0
        print(f"[DiracRunner] Simulation finished in {elapsed:.2f}s")

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  [C++] {line}")

        if result.returncode != 0:
            print(f"RUNTIME ERROR (exit code {result.returncode}):")
            if result.stderr:
                print(result.stderr)
            raise RuntimeError("DAE simulation failed")

        self.csv_path = os.path.join(cwd, "simulation_results.csv")
        if os.path.exists(self.csv_path):
            print(f"[DiracRunner] Results: {self.csv_path}")
        else:
            print("[DiracRunner] WARNING: CSV not found after simulation")

        return self.csv_path

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def load_results(self):
        """Load the CSV results into a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            Simulation results with columns for time, states, voltages.
        """
        import pandas as pd
        return pd.read_csv(self.csv_path)

    def compute_hamiltonian(self):
        """Compute total Hamiltonian from simulation results.

        Uses the ``HamiltonianAssembler`` to evaluate H(t) at each time
        step from the CSV data.

        Returns
        -------
        t : ndarray
            Time array.
        H : ndarray
            Total Hamiltonian at each time step.
        H_comp : dict
            Per-component Hamiltonian contributions.
        """
        from src.dirac.hamiltonian import HamiltonianAssembler

        assembler = HamiltonianAssembler(self.dae_compiler.graph)
        t, H_comp = assembler.from_csv(self.csv_path, self.dae_compiler.state_offsets)

        H_total = np.zeros_like(t)
        for name, h_arr in H_comp.items():
            H_total += h_arr

        return t, H_total, H_comp
