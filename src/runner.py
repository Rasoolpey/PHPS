import os
import sys
import time
import json
import math
import subprocess
import shutil
import numpy as np
from src.compiler import SystemCompiler
from src.initialization import Initializer
from src.errors import FrameworkError


class SimulationRunner:
    def __init__(self, json_path: str, output_dir: str = None,
                 events: list = None):
        """
        Args:
            json_path:  Path to the system JSON (physical model).
            output_dir: Where to write outputs. Derived from json_path if None.
            events:     Optional list of event dicts to inject (e.g. Toggler).
        """
        self.json_path = json_path
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found")

        # SystemCompiler now builds SystemGraph internally.
        # graph is accessible via self.compiler.graph.
        self.compiler = SystemCompiler(json_path)

        if events:
            toggler_events = [
                {k: v for k, v in ev.items() if k != "type"}
                for ev in events if ev.get("type") == "Toggler"
            ]
            if toggler_events:
                existing = self.compiler.data.get("Toggler", [])
                self.compiler.data["Toggler"] = existing + toggler_events

            bus_fault_events = [
                {k: v for k, v in ev.items() if k != "type"}
                for ev in events if ev.get("type") == "BusFault"
            ]
            if bus_fault_events:
                existing = self.compiler.data.get("BusFault", [])
                self.compiler.data["BusFault"] = existing + bus_fault_events

            line_fault_events = [
                {k: v for k, v in ev.items() if k != "type"}
                for ev in events if ev.get("type") == "LineFault"
            ]
            if line_fault_events:
                from src.system_graph import BusNode
                max_bus_idx = max([b["idx"] for b in self.compiler.data.get("Bus", [])])
                
                for lf in line_fault_events:
                    line_idx = lf["line_idx"]
                    distance = lf.get("distance", 0.5)
                    
                    line = next((l for l in self.compiler.data.get("Line", []) if l["idx"] == line_idx), None)
                    if not line:
                        raise ValueError(f"Line {line_idx} not found for LineFault")
                    
                    dummy_bus_idx = 99 # Use 99 to match the user's plot config
                    if dummy_bus_idx <= max_bus_idx:
                        dummy_bus_idx = max_bus_idx + 1
                    max_bus_idx = max(max_bus_idx, dummy_bus_idx)
                    
                    dummy_bus = {
                        "idx": dummy_bus_idx,
                        "name": f"Bus{dummy_bus_idx}",
                        "Vn": 1.0,
                        "v0": 1.0,
                        "a0": 0.0
                    }
                    self.compiler.data.setdefault("Bus", []).append(dummy_bus)
                    
                    dummy_bus_node = BusNode(
                        idx=dummy_bus_idx,
                        name=f"Bus{dummy_bus_idx}",
                        Vn=1.0,
                        v0=1.0,
                        a0=0.0,
                        vmax=1.1,
                        vmin=0.9
                    )
                    self.compiler.graph.buses[dummy_bus_idx] = dummy_bus_node
                    
                    line1 = dict(line)
                    line1["idx"] = f"{line_idx}_part1"
                    line1["bus2"] = dummy_bus_idx
                    line1["r"] = line.get("r", 0.0) * distance
                    line1["x"] = line.get("x", 0.001) * distance
                    line1["b"] = line.get("b", 0.0) * distance
                    
                    line2 = dict(line)
                    line2["idx"] = f"{line_idx}_part2"
                    line2["bus1"] = dummy_bus_idx
                    line2["r"] = line.get("r", 0.0) * (1 - distance)
                    line2["x"] = line.get("x", 0.001) * (1 - distance)
                    line2["b"] = line.get("b", 0.0) * (1 - distance)
                    
                    self.compiler.data["Line"] = [l for l in self.compiler.data["Line"] if l["idx"] != line_idx]
                    self.compiler.data["Line"].extend([line1, line2])

                    # Support both t_end (absolute) and t_duration (relative)
                    t0 = lf["t_start"]
                    if "t_duration" in lf:
                        t_end_val = t0 + float(lf["t_duration"])
                    else:
                        t_end_val = float(lf.get("t_end", t0 + 0.1))

                    bus_fault = {
                        "bus": dummy_bus_idx,
                        "r": lf.get("r", 0.0),
                        "x": lf.get("x", 0.001),
                        "t_start": t0,
                        "t_end": t_end_val
                    }
                    self.compiler.data.setdefault("BusFault", []).append(bus_fault)
                
                # Re-initialize YBusBuilder
                from src.ybus import YBusBuilder
                self.compiler.ybus_builder = YBusBuilder(self.compiler.data)
                self.compiler.n_bus = self.compiler.ybus_builder.n_bus

        if output_dir is None:
            base = os.path.splitext(os.path.basename(json_path))[0]
            self.output_dir = os.path.join("outputs", base)
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.source_path = os.path.join(self.output_dir, "system.cpp")
        _exe = ".exe" if sys.platform == "win32" else ""
        self.binary_path = os.path.join(self.output_dir, "sim_run" + _exe)
        self.csv_path = "simulation_results.csv"

        # Subscription system
        self.subscriptions = set()

        # Warm-start: set via set_warm_start() before build()
        self._warm_start = None

    def subscribe(self, comp_name: str, obs_name: str):
        """Subscribe to a specific observable or state of a component."""
        self.subscriptions.add((comp_name, obs_name))

    def subscribe_all(self, obs_name: str):
        """Subscribe to an observable or state across all components that have it."""
        for comp in self.compiler.components:
            if obs_name in comp.state_schema or obs_name in comp.observables:
                self.subscriptions.add((comp.name, obs_name))

    def subscribe_pattern(self, pattern: str):
        """Subscribe to all observables matching a fnmatch pattern."""
        from fnmatch import fnmatch
        for comp in self.compiler.components:
            # Check states
            for sname in comp.state_schema:
                if fnmatch(f"{comp.name}_{sname}", pattern):
                    self.subscriptions.add((comp.name, sname))
            # Check observables
            for obs_name in comp.observables:
                if fnmatch(f"{comp.name}_{obs_name}", pattern):
                    self.subscriptions.add((comp.name, obs_name))
            # Check Vterm
            if fnmatch(f"Vterm_{comp.name}", pattern):
                self.subscriptions.add((comp.name, 'Vterm'))
            # Check Pe, Qe, It_Re, It_Im
            for port in ('Pe', 'Qe', 'It_Re', 'It_Im'):
                if fnmatch(f"{port}_{comp.name}", pattern):
                    self.subscriptions.add((comp.name, port))
            # Check Efd_out
            if fnmatch(f"{comp.name}_Efd_out", pattern):
                self.subscriptions.add((comp.name, 'Efd_out'))
                
        # Check network variables
        for bus_id in self.compiler.graph.buses.keys():
            for prefix in ('Vterm_Bus', 'V_Re_Bus', 'V_Im_Bus'):
                if fnmatch(f"{prefix}{bus_id}", pattern):
                    self.subscriptions.add(('Network', f"{prefix}{bus_id}"))

    def list_observables(self, comp_name: str = None):
        """Print a formatted table of all observable names, descriptions, and units."""
        cmp = self.compiler
        print(f"{'Component':<15} | {'Signal':<20} | {'Unit':<10} | {'Description'}")
        print("-" * 80)
        for comp in cmp.components:
            if comp_name and comp.name != comp_name:
                continue
            
            # States
            for state in comp.state_schema:
                print(f"{comp.name:<15} | {state:<20} | {'':<10} | State variable")
            
            # Observables
            for obs, info in comp.observables.items():
                unit = info.get('unit', '')
                desc = info.get('description', '')
                print(f"{comp.name:<15} | {obs:<20} | {unit:<10} | {desc}")
            
            # Standard ports
            if comp.component_role == 'generator':
                print(f"{comp.name:<15} | {'Vterm':<20} | {'pu':<10} | Terminal Voltage")
                print(f"{comp.name:<15} | {'Pe':<20} | {'pu':<10} | Active Power")
                print(f"{comp.name:<15} | {'Qe':<20} | {'pu':<10} | Reactive Power")
                print(f"{comp.name:<15} | {'It_Re':<20} | {'pu':<10} | Real Terminal Current")
                print(f"{comp.name:<15} | {'It_Im':<20} | {'pu':<10} | Imaginary Terminal Current")
            
            if comp.component_role == 'exciter':
                print(f"{comp.name:<15} | {'Efd_out':<20} | {'pu':<10} | Exciter Field Voltage Output")
            
            print("-" * 80)

    def set_warm_start(self, csv_path: str, t: float = None,
                       delta_unwrap: dict = None,
                       omega_override: float = None):
        """Configure warm-start from a previous simulation CSV.

        Args:
            csv_path:       Path to the simulation_results.csv.
            t:              Time to extract state from.  None = last row.
            delta_unwrap:   Dict mapping component name -> number of 2π
                            rotations to subtract from delta, e.g.
                            {"GENROU_2": 1} subtracts 2π from GENROU_2 delta.
            omega_override: If set, force all generator omega to this value.
        """
        self._warm_start = {
            'csv_path': csv_path,
            't': t,
            'delta_unwrap': delta_unwrap or {},
            'omega_override': omega_override,
        }

    def _apply_warm_start(self, x0: np.ndarray,
                          init: 'Initializer') -> np.ndarray:
        """Override x0 with converged states from a previous run's CSV."""
        import pandas as pd

        ws = self._warm_start
        df = pd.read_csv(ws['csv_path'])
        t_col = df.columns[0]

        if ws['t'] is not None:
            mask = df[t_col] <= ws['t']
            row = df[mask].iloc[-1]
        else:
            row = df.iloc[-1]

        print(f"  [WarmStart] Loading state from t={row[t_col]:.4f}")

        # Build column-name → (component, state) mapping from compiler
        col_map = {}  # csv_column_name -> (comp_name, state_name, state_idx)
        for comp in self.compiler.components:
            off = self.compiler.state_offsets[comp.name]
            for i, sname in enumerate(comp.state_schema):
                # CSV columns use patterns like "GENROU_1_delta",
                # "EXST1_1_Vm", etc.
                col_map[f"{comp.name}_{sname}"] = (comp.name, sname, off + i)

        n_matched = 0
        for csv_col in df.columns:
            # Strip unit suffixes like " (pu)" or " (deg)"
            clean = csv_col.split(' (')[0].strip()
            if clean in col_map:
                comp_name, sname, idx = col_map[clean]
                val = float(row[csv_col])

                # Apply delta unwrap correction
                if sname == 'delta' and comp_name in ws['delta_unwrap']:
                    n_rot = ws['delta_unwrap'][comp_name]
                    correction = n_rot * 2.0 * math.pi
                    print(f"  [WarmStart] {comp_name} delta: {val:.4f} "
                          f"-> {val - correction:.4f} (unwrap {n_rot} rev)")
                    val -= correction

                x0[idx] = val
                n_matched += 1

        # Force omega to a specific value if requested
        if ws.get('omega_override') is not None:
            omega_val = ws['omega_override']
            for comp in self.compiler.components:
                if 'omega' not in comp.state_schema:
                    continue
                off = self.compiler.state_offsets[comp.name]
                omega_idx = off + comp.state_schema.index('omega')
                old_w = x0[omega_idx]
                x0[omega_idx] = omega_val
                print(f"  [WarmStart] {comp.name} omega: {old_w:.6f} -> {omega_val:.6f}")

        # Also update Tm0 for generators (use Te from the warm-start point)
        for comp in self.compiler.components:
            if comp.component_role != 'generator':
                continue
            pe_col = f"Pe_{comp.name}"
            if pe_col in df.columns:
                Te_warm = float(row[pe_col])
                comp.params['Tm0'] = Te_warm
                print(f"  [WarmStart] {comp.name}: Tm0={Te_warm:.6f}")

        # Update governor Pref to match warm-start Te
        for comp in self.compiler.components:
            if comp.component_role != 'governor':
                continue
            gen_comp = None
            for gc in self.compiler.components:
                if gc.name == comp.params.get('syn'):
                    gen_comp = gc
                    break
            if gen_comp is None:
                continue
            Te_act = gen_comp.params.get('Tm0', 0.0)
            off = self.compiler.state_offsets[comp.name]
            n = len(comp.state_schema)
            x_new, _ = comp.update_from_te(x0[off:off + n].copy(), Te_act)
            x0[off:off + n] = x_new

        print(f"  [WarmStart] Matched {n_matched} state variables")
        return x0

    def build(self, dt: float = 0.001, duration: float = 10.0, method: str = "rk4"):
        """
        Compiles the system and generates the C++ driver with specific settings.

        Pipeline:
          1. Build component structure (memory layout, Y-bus, wiring)
          2. Validate SystemGraph wires (port connections, types)
          3. Run power flow + initialize component states
          4. Finalize Kron-reduced network with PF voltages
          5. Iterative state refinement (Kron equilibrium, exciter, governor)
          6. Generate and compile C++ kernel
        """
        print(f"[Runner] Compiling system from {self.json_path}...")
        graph = self.compiler.graph
        print(f"[Runner] SystemGraph: {graph}")

        # 1. Build Structure (Parse + Wiring + YBus)
        print("[Runner] Building component structure...")
        self.compiler.build_structure()

        # 2. Validate SystemGraph wiring (before spending time on power flow)
        print("[Runner] Validating system graph connections...")
        try:
            graph.validate()
            print("[Runner] Graph validation passed.")
        except FrameworkError as e:
            print(f"[Runner] Graph validation ERROR:\n{e}")
            raise
        
        # 2. Run Power Flow & Init (Python side)
        # This updates component parameters (Vref, Pref) to match equilibrium
        print("[Runner] Initializing system states (Power Flow)...")
        init = Initializer(self.compiler)
        # Run power flow first (inside Initializer.run() -> pf.solve())
        # then finalize the Kron-reduced network with actual PF voltages so
        # load admittances use |V0|^2 instead of assuming |V|=1 (Issue #1 fix).
        x0 = init.run()
        self.compiler.finalize_network(pf_V=init.pf.V, pf_theta=init.pf.theta)

        # Phase 1: Correct generator states to Kron-reduced network operating point.
        # init_from_phasor() uses PF terminal voltages; the Kron-reduced network
        # produces slightly different terminal voltages due to load lumping and
        # generator internal impedances. This converging fixed-point iteration
        # updates each component's states via comp.refine_at_kron_voltage().
        print("[Runner] Correcting states to Kron-reduced network equilibrium...")
        x0 = init.refine_kron_equilibrium(x0, n_iter=30, alpha=0.5)

        # Phase 2: Correct exciter states to the actual Kron-network terminal voltage.
        x0 = init.refine_exciter_voltages(x0, full_reinit=True)

        # Phase 3: Handle generators where the exciter clips Efd (hard clamp).
        # Re-initialize d-axis states to the clamped-Efd equilibrium, then
        # re-run exciter back-solve at the new operating point.
        x0 = init.refine_clamped_d_axis(x0)
        x0 = init.refine_exciter_voltages(x0, full_reinit=True)

        # Phase 4: Interleaved d-axis / Eq_p / psi_d / exciter / q-axis refinement.
        # Updating Eq_p (via Efd) changes psi_d_pp → changes stator iq → invalidates
        # q-axis equilibrium. We therefore re-run refine_kron_equilibrium after each
        # d-axis update to keep all states mutually consistent.
        #
        # NOTE: For larger systems (n_active > 4) the interleaved updates can
        # build up numerical errors across rounds due to stronger machine
        # coupling.  We use fewer rounds and a smaller relaxation factor for
        # those cases, and we always verify that states remain physical before
        # accepting each round.
        print("[Runner] Refining d-axis and Efd-coupled states (interleaved)...")
        n_active = self.compiler.n_active
        if n_active <= 1:
            alpha_d    = 0.5
            max_rounds = 5
        elif n_active <= 4:
            alpha_d    = 0.4
            max_rounds = 4
        else:
            # Large multi-machine systems: be conservative to avoid divergence
            alpha_d    = 0.3
            max_rounds = 2

        # For large multi-machine systems (n_active > 4) the Kron-reduced network
        # equilibrium differs significantly from the full PF operating point (e.g.
        # the slack machine can show 25 % more power in Kron than in PF due to
        # transformer-bus machines absorbing different reactive power).  Running
        # refine_exciter_voltages(full_reinit=True) inside Phase 4 inflates the
        # exciter Efd (~14 % higher for IEEE14) as it re-solves VB/VM at the
        # Kron-inflated stator currents, pushing the operating point into a region
        # that can be unstable for the ESST3A high-gain parameters.
        # Solution: freeze exciter VB/VM during Phase 4 (use full_reinit=False);
        # the final Phase-6 step will then do a lightweight transducer correction.
        full_reinit_p4 = (n_active <= 4)  # Only full VB/VM re-solve for small systems

        x0_phase4_start = x0.copy()   # fall-back snapshot before Phase 4
        for _round in range(max_rounds):
            x0_round_start = x0.copy()
            x0 = init.refine_Eq_p(x0, n_iter=10, alpha=alpha_d)
            x0 = init.refine_psi_d(x0)
            x0 = init.refine_exciter_voltages(x0, full_reinit=full_reinit_p4)
            # Restore q-axis equilibrium after d-axis states changed
            x0 = init.refine_kron_equilibrium(x0, n_iter=20, alpha=0.5)

            # Sanity check: abandon round if any generator state has blown up.
            # Norton |I| > threshold for any generator signals numerical
            # instability.  The normal Norton current for a machine is
            # E'/xd'', which can be 200-300 pu on system base for large
            # machines with small sub-transient reactance.  Scale the
            # threshold by the number of generators and max machine size.
            max_norton = 0.0
            norton_threshold = max(100.0, 50.0 * n_active)
            for _comp in self.compiler.components:
                if _comp.component_role != 'generator':
                    continue
                _off = self.compiler.state_offsets[_comp.name]
                _n   = len(_comp.state_schema)
                _Ino = abs(_comp.compute_norton_current(x0[_off:_off + _n]))
                max_norton = max(max_norton, _Ino)

            if max_norton > norton_threshold:
                print(f"  [Runner] Phase 4 round {_round+1}: Norton |I|={max_norton:.1f} "
                      f"pu (threshold={norton_threshold:.0f}) -- instability detected, "
                      f"reverting to round start.")
                x0 = x0_round_start
                break

        # After Phase 4: converge the exciter-flux-VB coupling:
        # 1) full_reinit → updates Efd/Eq_p from new network operating point
        # 2) refine_rectifier_VB → sets VB = VE*FEX exactly (eliminates dVB residual)
        # 3) refine_psi_d → re-aligns psi_d to new Eq_p so dxdt[psi_d] = 0
        # Iterate a few rounds until the total change is negligible.
        for _cleanup in range(4):
            x0_prev = x0.copy()
            x0 = init.refine_exciter_voltages(x0, full_reinit=True)
            x0 = init.refine_rectifier_VB(x0)
            x0 = init.refine_psi_d(x0)
            if np.max(np.abs(x0 - x0_prev)) < 1e-7:
                break

        # Phase 5: Synchronize governor Pref to the actual electrical torque
        # Phase 5b: Re-initialize renewable controllers to the actual Kron network voltages
        print("[Runner] Refining renewable controllers...")
        for _ in range(10):
            x0_prev = x0.copy()
            x0 = init.refine_renewable_controllers(x0)
            x0 = init.refine_kron_equilibrium(x0, n_iter=50, alpha=0.5)
            if np.max(np.abs(x0 - x0_prev)) < 1e-6:
                break

        # Phase 5a: Re-initialize PSS states to the actual electrical torque
        x0 = init.refine_pss(x0)

        # 4. Refine governor Pref values
        print("[Runner] Refining governor Pref values to match actual Te...")
        x0 = init.refine_governor_pref(x0)

        # Phase 5c: Adjust rotor angle delta to achieve Te = Tm in the Kron network.
        # Only needed for SMIB / systems with an external slack bus (non-zero
        # slack_norton_I), where the Kron voltage angle differs from the PF angle.
        # For multi-machine systems (no external slack) the Kron-reduced network
        # IS the self-consistent equilibrium; governors use Te_kron directly
        # (see refine_governor_pref), so the delta does not need correction here.
        slack_I_mag = float(np.max(np.abs(self.compiler.slack_norton_I))) \
            if hasattr(self.compiler, 'slack_norton_I') else 0.0
        if slack_I_mag > 1e-6:
            print("[Runner] Adjusting delta for torque balance in Kron network...")
            x0 = init.refine_delta_for_torque_balance(x0, n_iter=60, alpha=0.15)
            # After delta changed: re-converge all states using the same proven
            # sequence as Phase 4 (q-axis, Eq_p, psi_d, exciter).
            for _post in range(20):
                x0 = init.iterative_refinement(x0, n_iter=5, alpha=0.8)   # psi_q, Ed_p
                x0 = init.refine_Eq_p(x0, n_iter=10, alpha=0.4)           # Eq_p
                x0 = init.refine_psi_d(x0)                                # psi_d
                x0 = init.refine_exciter_voltages(x0, full_reinit=True)   # VB, VM, Vr
            # Re-sync governor after delta update (Te may have changed)
            x0 = init.refine_governor_pref(x0)

        # Phase 6: Final coupled exciter-VB-governor convergence loop.
        # The VB state depends on stator currents (id/iq) via the VE load
        # compensator.  The Python Z-bus solve gives slightly different id/iq
        # than the C++ iterative algebraic loop, leaving a small VB residual
        # that drives an Efd transient → Te shift → omega offset via droop.
        # Iterating (exciter → VB → Kron → governor) converges this coupling.
        for _final in range(6):
            x0_prev = x0.copy()
            x0 = init.refine_exciter_voltages(x0, full_reinit=False)
            x0 = init.refine_rectifier_VB(x0)
            x0 = init.refine_kron_equilibrium(x0, n_iter=50, alpha=0.5)
            x0 = init.refine_governor_pref(x0)
            if np.max(np.abs(x0 - x0_prev)) < 1e-8:
                break

        # Phase 7: Re-initialize passive components with the final Kron-reduced network voltages.
        x0 = init.refine_passive_components(x0)

        # Phase 8: Final governor Pref re-sync after passive component init.
        x0 = init.refine_governor_pref(x0)

        # Phase 9: Final DFIG equilibrium convergence.
        # Earlier phases (governor Pref, exciter, VB) change the GENCLS
        # Norton current, shifting all bus voltages.  Phase 6's Kron
        # equilibrium re-adjusts DFIG flux states but the RSC integrators
        # are stale.  This final loop ensures DFIG flux states AND RSC
        # controller states are mutually consistent at the true network
        # voltage: dφ/dt = 0 for all 4 flux states and the RSC produces
        # exactly the Vrd/Vrq needed for rotor equilibrium.
        print("[Runner] Final DFIG equilibrium convergence...")
        for _ren in range(10):
            x0_prev = x0.copy()
            x0 = init.refine_kron_equilibrium(x0, n_iter=100, alpha=0.5)
            x0 = init.refine_renewable_controllers(x0)
            chg = np.max(np.abs(x0 - x0_prev))
            if chg < 1e-8:
                print(f"  [Runner] Phase 9 converged in {_ren+1} round(s) (max|dx|={chg:.2e}).")
                break

        # Diagnostic: print per-generator Tm vs Te balance
        V_diag = init._get_converged_network_voltages(x0)
        for comp in self.compiler.components:
            if comp.component_role != 'generator' or 'omega' not in comp.state_schema:
                continue
            off = self.compiler.state_offsets[comp.name]
            n = len(comp.state_schema)
            V_bus = init._gen_bus_voltage(comp, V_diag)
            if V_bus is None:
                continue
            vd, vq = init._park_transform(V_bus, float(x0[off]))
            id_act, iq_act = comp.compute_stator_currents(x0[off:off+n], vd, vq)
            Te = vd * id_act + vq * iq_act
            Tm0 = float(comp.params.get('Tm0', 0.0))
            D = float(comp.params.get('D', 0.0))
            omega = float(x0[off + comp.state_schema.index('omega')])
            H = float(comp.params.get('H', 1.0))
            dw_dt = (Tm0 - Te - D * (omega - 1.0)) / (2.0 * H)
            print(f"  [Diag] {comp.name}: Tm0={Tm0:.8f} Te={Te:.8f} "
                  f"Tm-Te={Tm0-Te:.2e} dw/dt={dw_dt:.2e} "
                  f"|V|={abs(V_bus):.6f}")

        print("[Runner] Computing consistent initial network voltages from x0...")
        Vd_init, Vq_init = init.compute_initial_network_voltages(x0)

        # ── Warm-start override ──────────────────────────────────────
        # If a warm_start dict is attached, load the converged state
        # from a previous simulation CSV and overwrite x0 + Vd/Vq.
        if self._warm_start is not None:
            print("[Runner] Applying warm-start override...")
            x0 = self._apply_warm_start(x0, init)
            # Refine flux states to equilibrium at the warm-start operating
            # point.  The warm-start CSV may have been generated with a
            # different model variant, leaving small flux residuals.
            print("[Runner] Refining warm-start states to current-model equilibrium...")
            for _ws_round in range(5):
                x0_prev = x0.copy()
                x0 = init.refine_kron_equilibrium(x0, n_iter=50, alpha=0.5)
                x0 = init.refine_psi_d(x0)
                x0 = init.refine_governor_pref(x0)
                if np.max(np.abs(x0 - x0_prev)) < 1e-8:
                    break
            Vd_init, Vq_init = init.compute_initial_network_voltages(x0)

        # Ensure x0 has the correct size (including delta_COI)
        if len(x0) < self.compiler.total_states:
            x0 = np.append(x0, np.zeros(self.compiler.total_states - len(x0)))

        x0_str = ", ".join(f"{v:.12f}" for v in x0)
        n_active = self.compiler.n_active
        Vd_str = ", ".join(f"{v:.12f}" for v in Vd_init)
        Vq_str = ", ".join(f"{v:.12f}" for v in Vq_init)
        
        # 3. Generate Kernel (Now using the UPDATED parameters)
        print("[Runner] Generating C++ kernel...")
        kernel_code = self.compiler.generate_cpp()
        
        # 4. Generate Main Driver (with initial Vd, Vq from power flow)
        csv_cols = self._build_csv_columns()
        driver_code = self._generate_driver(x0_str, dt, duration, method, Vd_str, Vq_str, n_active, csv_cols)
        
        # 5. Write complete C++ file
        with open(self.source_path, 'w', encoding='utf-8') as f:
            f.write(kernel_code + "\n" + driver_code)
            
        # 6. Compile C++
        print(f"[Runner] Compiling C++ binary: {self.binary_path}")
        cmd = ["g++", "-O3", self.source_path, "-o", self.binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("COMPILATION FAILED:")
            print(result.stderr)
            raise RuntimeError("C++ compilation failed")
            
        print("[Runner] Build successful.")
        return x0, Vd_init, Vq_init

    # ------------------------------------------------------------------
    # Tier 2 Co-Simulation build
    # ------------------------------------------------------------------

    def build_cosim(self, cosim_config, dt: float = 0.001,
                    duration: float = 10.0) -> None:
        """
        Build a co-simulation shared library (``plant.so``).

        The initialisation pipeline is identical to ``build()``.  After the
        binary has been compiled the method additionally generates ``plant.cpp``
        with the co-simulation ABI and links it with ``-shared -fPIC``.

        Parameters
        ----------
        cosim_config : CosimConfig
            Port declarations (controls + measurements).
        dt : float
            Physics step size hint (forwarded to ``build()``).
        duration : float
            Simulation duration hint (forwarded to ``build()``; ignored by
            plant.so — the orchestrator controls the end time).
        """
        print(f"[Runner] build_cosim() -- config='{cosim_config.name}'"
              f"  n_ctrl={cosim_config.n_ctrl}  n_meas={cosim_config.n_meas}")

        # 1. Full init pipeline (power flow + all refinement phases)
        #    This also leaves compiler.wiring_map at post-init equilibrium values
        #    which generate_cosim_cpp() reads when overriding control ports.
        self.build(dt=dt, duration=duration)

        # 2. Capture initial conditions from the driver source file.
        self.x0_arr = self._extract_ic_from_source('x')
        self.Vd0    = self._extract_ic_from_source('Vd')
        self.Vq0    = self._extract_ic_from_source('Vq')

        # 3. Generate co-simulation C++ kernel.
        print("[Runner] Generating co-simulation C++ kernel (plant.cpp)...")
        kernel_code = self.compiler.generate_cosim_cpp(cosim_config)
        plant_cpp_path = os.path.join(self.output_dir, "plant.cpp")
        with open(plant_cpp_path, 'w', encoding='utf-8') as f:
            f.write(kernel_code)

        # 4. Compile shared library.
        self.so_path = os.path.join(self.output_dir, "plant.so")
        print(f"[Runner] Compiling shared library: {self.so_path}")
        cmd = ["g++", "-O3", "-shared", "-fPIC", plant_cpp_path,
               "-o", self.so_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("SHARED LIBRARY COMPILATION FAILED:")
            print(result.stderr)
            raise RuntimeError("plant.so compilation failed")

        print("[Runner] build_cosim() successful.")
        print(f"  plant.so  : {self.so_path}")
        print(f"  x0 shape  : {self.x0_arr.shape}")
        print(f"  n_bus     : {self.compiler.n_active}")

    def _extract_ic_from_source(self, var_name: str) -> np.ndarray:
        """
        Parse an initial-condition vector from the generated C++ driver source.

        The driver (``_generate_driver``) emits::

            double x[N_STATES] = { v0, v1, ... };
            double Vd[N_BUS]   = { v0, v1, ... };
            double Vq[N_BUS]   = { v0, v1, ... };

        ``var_name`` should be ``'x'``, ``'Vd'``, or ``'Vq'``.
        """
        import re
        fallback_size = {
            'x':  self.compiler.total_states,
            'Vd': self.compiler.n_active,
            'Vq': self.compiler.n_active,
        }.get(var_name, 1)
        try:
            with open(self.source_path, 'r') as f:
                src = f.read()
            # Matches: double x[N_STATES] = { v, v, ... };
            pattern = (rf'double\s+{re.escape(var_name)}\s*\[[^\]]+\]\s*=\s*'
                       rf'\{{\s*([^}}]+)\s*\}}')
            m = re.search(pattern, src)
            if m:
                vals = [float(v.strip()) for v in m.group(1).split(',')
                        if v.strip()]
                return np.array(vals, dtype=np.float64)
        except Exception as exc:
            print(f"[Runner] _extract_ic_from_source('{var_name}'): {exc}")
        return np.zeros(fallback_size, dtype=np.float64)

    def run(self):
        """Executes the compiled binary."""
        print(f"[Runner] Executing simulation in {self.output_dir}...")
        
        # Run inside output dir so CSV is generated there
        cwd = os.getcwd()
        abs_binary = os.path.abspath(self.binary_path)
        try:
            os.chdir(self.output_dir)
            if not os.path.exists(abs_binary):
                raise RuntimeError("Binary not found. Call build() first.")
            
            # Execute using absolute path so it works on both Linux and Windows
            subprocess.run([abs_binary], check=True)
            print(f"[Runner] Results saved to {os.path.join(self.output_dir, self.csv_path)}")
            
            # Save config for plotter
            with open(os.path.join(cwd, "tools/last_run_config.txt"), "w") as f:
                f.write(self.output_dir)
                
        finally:
            os.chdir(cwd)

    def _build_csv_columns(self):
        """Build ordered list of (header_name, cpp_expr) for CSV output.

        Fully type-agnostic: uses component_role and port schema to decide
        what to log.  Adding a new component type requires no changes here.

        Logged columns:
          - t
          - Vterm_<gen> for every generator bus
          - Vterm_Bus<N>, V_Re_Bus<N>, V_Im_Bus<N> for observer buses
          - It_Re_<gen>, It_Im_<gen>, Pe_<gen>, Qe_<gen> for generators
          - <comp>_<state> for every state of every component
          - <comp>_<observable> for every declared observable
        """
        cols = [("t", "t")]

        cmp = self.compiler
        active_bus_map = cmp.active_bus_map
        ybus_map = cmp.ybus_builder.bus_map

        # --- Terminal voltage for each generator bus ---
        for c in cmp.components:
            if c.component_role != 'generator':
                continue
            if self.subscriptions and (c.name, 'Vterm') not in self.subscriptions:
                continue
            bus_id = c.params.get('bus')
            if bus_id in ybus_map:
                full_idx = ybus_map[bus_id]
                if full_idx in active_bus_map:
                    red_idx = active_bus_map[full_idx]
                    cols.append((f"Vterm_{c.name} (pu)", f"Vt[{red_idx}]"))

        # --- Observer bus voltages ---
        for i, bus_id in enumerate(cmp.observer_bus_names):
            if not self.subscriptions or ('Network', f'Vterm_Bus{bus_id}') in self.subscriptions:
                cols.append((f"Vterm_Bus{bus_id} (pu)", f"Vterm_obs[{i}]"))
            if not self.subscriptions or ('Network', f'V_Re_Bus{bus_id}') in self.subscriptions:
                cols.append((f"V_Re_Bus{bus_id} (pu)", f"Vd_obs[{i}]"))
            if not self.subscriptions or ('Network', f'V_Im_Bus{bus_id}') in self.subscriptions:
                cols.append((f"V_Im_Bus{bus_id} (pu)", f"Vq_obs[{i}]"))

        # --- Generator terminal current and power (from port schema) ---
        for c in cmp.components:
            if c.component_role != 'generator':
                continue
            out_names = [p[0] for p in c.port_schema['out']]
            for port in ('It_Re', 'It_Im'):
                if port in out_names:
                    if not self.subscriptions or (c.name, port) in self.subscriptions:
                        idx = out_names.index(port)
                        cols.append((f"{port}_{c.name}", f"outputs_{c.name}[{idx}]"))
            for port in ('Pe', 'Qe'):
                if port in out_names:
                    if not self.subscriptions or (c.name, port) in self.subscriptions:
                        idx = out_names.index(port)
                        cols.append((f"{port}_{c.name}", f"outputs_{c.name}[{idx}]"))

        # --- All state variables and observables (every component) ---
        import re
        for c in cmp.components:
            off = cmp.state_offsets[c.name]
            
            # 1. Raw states
            for si, sname in enumerate(c.state_schema):
                if not self.subscriptions or (c.name, sname) in self.subscriptions:
                    cols.append((f"{c.name}_{sname}", f"x[{off + si}]"))
                
            # 2. Declared observables
            obs_dict = c.observables
            for obs_name, obs_info in obs_dict.items():
                if not self.subscriptions or (c.name, obs_name) in self.subscriptions:
                    expr = obs_info.get('cpp_expr', '')
                    if not expr:
                        continue
                    
                    # Replace x[i] with x[off + i]
                    expr = re.sub(r'x\[(\d+)\]', lambda m: f"x[{off + int(m.group(1))}]", expr)
                    # Replace inputs[i] with inputs_C[i]
                    expr = re.sub(r'inputs\[(\d+)\]', lambda m: f"inputs_{c.name}[{m.group(1)}]", expr)
                    # Replace outputs[i] with outputs_C[i]
                    expr = re.sub(r'outputs\[(\d+)\]', lambda m: f"outputs_{c.name}[{m.group(1)}]", expr)
                    # Replace named params with numeric literals (params are local to
                    # system_step but observables are emitted in main())
                    for pname in c.param_schema:
                        val = c.params.get(pname)
                        if val is not None:
                            try:
                                expr = re.sub(r'\b' + re.escape(pname) + r'\b',
                                              repr(float(val)), expr)
                            except (ValueError, TypeError):
                                pass  # skip non-numeric params (e.g. '2.0 * M_PI * 60.0')
                    
                    unit = obs_info.get('unit', '')
                    header = f"{c.name}_{obs_name} ({unit})" if unit else f"{c.name}_{obs_name}"
                    cols.append((header, expr))
                
            # Efd output for exciters: use the efd_output_expr() contract.
            # Each exciter knows how to express its Efd as a C++ expression.
            if c.component_role == 'exciter':
                if not self.subscriptions or (c.name, 'Efd_out') in self.subscriptions:
                    expr = c.efd_output_expr(off)
                    if expr:
                        cols.append((f"{c.name}_Efd_out", expr))

        # --- Add COI angle and absolute angles ---
        cols.append(("delta_COI (deg)", f"x[{cmp.delta_coi_idx}] * 180.0 / M_PI"))
        for c in cmp.components:
            if c.component_role == 'generator' and 'omega' in c.state_schema:
                off = cmp.state_offsets[c.name]
                # delta is always the first state (x[0]) for generators
                cols.append((f"{c.name}_delta_abs_deg (deg)", f"(x[{off}] + x[{cmp.delta_coi_idx}]) * 180.0 / M_PI"))

        return cols

    def _generate_driver(self, x0_str, dt, duration, method, Vd_str, Vq_str, n_bus, csv_cols=None):
        steps = int(duration / dt)
        
        # Build CSV header and logging line
        if csv_cols is None:
            csv_cols = [("t", "t")]
        header = ",".join(h for h, _ in csv_cols)
        log_parts = " << \",\" << ".join(f"({expr})" for _, expr in csv_cols)
        
        # Diagnostics block to print large derivatives at t=0
        gen_diag_lines = []
        for c in self.compiler.components:
            if c.component_role == 'generator' and 'omega' in c.state_schema:
                off = self.compiler.state_offsets[c.name]
                omega_idx = off + c.state_schema.index('omega')
                H2 = 2.0 * float(c.params.get('H', 1.0))
                gen_diag_lines.append(
                    f'    std::cout << "  {c.name}: dw/dt=" << dxdt[{omega_idx}]'
                    f' << " Tm-Te=" << dxdt[{omega_idx}]*{H2:.1f}'
                    f' << std::endl;'
                )
        gen_diag_code = '\n'.join(gen_diag_lines) if gen_diag_lines else ''

        debug_block = """
    // --- DIAGNOSTICS: Check Initial Derivatives ---
    std::cout << "\\n[Diagnostics] Checking Equilibrium Quality..." << std::endl;
    system_step(x, dxdt, 0.0, Vd, Vq, Vt);
    
    bool stable = true;
    double max_dxdt_init = 0.0;
    int max_idx = -1;
    for(int i=0; i<N_STATES; ++i) {
        if(fabs(dxdt[i]) > 0.05) {
            std::cout << "  WARNING: State[" << i << "] large dxdt=" << dxdt[i] << std::endl;
            stable = false;
        }
        if(fabs(dxdt[i]) > max_dxdt_init) { max_dxdt_init = fabs(dxdt[i]); max_idx = i; }
    }
    std::cout << "  Max |dxdt| = " << max_dxdt_init << " at state[" << max_idx << "]" << std::endl;
    if(stable) std::cout << "  Initial equilibrium looks good (all |dx/dt| < 0.05)." << std::endl;
    else std::cout << "  SYSTEM NOT IN EQUILIBRIUM. Expect transients." << std::endl;
""" + gen_diag_code + """
    std::cout << "  V after algebraic solve:" << std::endl;
    for(int b=0; b<N_BUS; ++b)
        std::cout << "    Bus[" << b << "]: Vd=" << Vd[b] << " Vq=" << Vq[b]
                  << " |V|=" << Vt[b] << std::endl;
    std::cout << "------------------------------------------\\n" << std::endl;
"""

        # Integration Loop Selection
        m = method.lower()
        if m == "euler":
            solver_loop = f"""
        system_step(x, dxdt, t, Vd, Vq, Vt);
        for(int j=0; j<N_STATES; ++j) x[j] += dxdt[j] * dt;
"""
        elif m == "rk2":
            # Heun's method (explicit trapezoidal / RK2)
            solver_loop = f"""
        // RK2 Stage 1
        system_step(x, k1, t, Vd, Vq, Vt);

        // RK2 Stage 2 (predictor)
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + dt * k1[j];
        system_step(x_temp, k2, t + dt, Vd, Vq, Vt);

        // Final Update (corrector)
        for(int j=0; j<N_STATES; ++j) x[j] += 0.5 * dt * (k1[j] + k2[j]);
"""
        elif m == "midpoint":
            # Implicit Midpoint Rule — structure-preserving symplectic integrator.
            #
            # The implicit midpoint rule preserves quadratic invariants exactly,
            # which makes it ideal for Port-Hamiltonian systems where
            # H(x) is quadratic: the discrete energy balance
            #   H(x^{n+1}) - H(x^n) ≤ dt · y^n^T u^n
            # is satisfied exactly at every timestep.
            #
            # The rule: x_{n+1} = x_n + dt · f(x_{mid}, t_{mid})
            #   where x_{mid} = (x_n + x_{n+1})/2,  t_{mid} = t + dt/2
            #
            # Solved via Newton iteration on:
            #   G(x_{n+1}) = x_{n+1} - x_n - dt · f((x_n + x_{n+1})/2, t + dt/2) = 0
            #
            # Uses the same diagonal FD Newton approach as SDIRK-2.
            solver_loop = f"""
        // Implicit Midpoint Rule (structure-preserving)
        {{
            const double t_mid = t + 0.5 * dt;
            const double tol_newton = 1e-10;
            const int    max_newton = 50;

            // Predictor: explicit Euler step
            system_step(x, k1, t, Vd, Vq, Vt);
            for(int j=0; j<N_STATES; ++j) k2[j] = x[j] + dt * k1[j]; // k2 = x_{n+1} guess

            for(int nit = 0; nit < max_newton; ++nit) {{
                // Midpoint state: x_mid = (x + x_{n+1}) / 2
                for(int j=0; j<N_STATES; ++j)
                    x_temp[j] = 0.5 * (x[j] + k2[j]);

                // f(x_mid, t_mid)
                system_step(x_temp, k3, t_mid, Vd, Vq, Vt);

                // Residual: r = x_{n+1} - x_n - dt * f(x_mid, t_mid)
                double res_norm = 0.0;
                for(int j=0; j<N_STATES; ++j) {{
                    res[j] = k2[j] - x[j] - dt * k3[j];
                    res_norm += res[j]*res[j];
                }}
                res_norm = sqrt(res_norm);
                if(res_norm < tol_newton) break;

                // Diagonal Newton: d(res)/d(x_{n+1})[j] = 1 - dt * 0.5 * df_j/dx_j
                double eps_fd = 1e-7;
                for(int j=0; j<N_STATES; ++j) {{
                    double xnp1_save = k2[j];
                    k2[j] += eps_fd;
                    for(int jj=0; jj<N_STATES; ++jj)
                        x_temp[jj] = 0.5 * (x[jj] + k2[jj]);
                    system_step(x_temp, k4, t_mid, Vd, Vq, Vt);
                    // df_j/dx_{n+1}[j] = 0.5 * (f_j(x_mid+) - f_j(x_mid)) / eps
                    double dfdx = (k4[j] - k3[j]) / eps_fd;
                    double diag = 1.0 - dt * dfdx;   // dfdx already has 0.5 factor from midpoint
                    if(fabs(diag) < 1e-14) diag = 1e-14;
                    k2[j] = xnp1_save - res[j] / diag;
                }}
            }}

            // Update: x = x_{n+1}
            for(int j=0; j<N_STATES; ++j) x[j] = k2[j];

            // Keep dxdt updated for the progress monitor
            system_step(x, dxdt, t+dt, Vd, Vq, Vt);
        }}
"""
        elif m in ("sdirk2", "dirk", "radau"):
            # SDIRK-2: 2-stage, L-stable, A-stable implicit solver (same family as Radau IIA).
            # Butcher tableau (gamma = 1 - 1/sqrt(2) ~ 0.2929):
            #   gamma  |  gamma     0
            #   1      |  1-gamma   gamma
            #   -------+------------------
            #          |  1-gamma   gamma
            #
            # Stage 1 (implicit): K1 = f(x + dt*gamma*K1,  t + gamma*dt)
            # Stage 2 (implicit): K2 = f(x + dt*(1-gamma)*K1 + dt*gamma*K2,  t + dt)
            # Update:  x_new = x + dt*((1-gamma)*K1 + gamma*K2)
            #        = x + dt*K2  (since stage-2 and output share weights)
            #
            # Each stage is solved with a Jacobian-free Newton iteration using
            # finite-difference directional derivatives (Jacobian-vector products).
            # This avoids storing the full N×N Jacobian while still converging
            # quadratically for smooth problems.
            solver_loop = f"""
        // SDIRK-2 (L-stable, A-stable, same family as Radau IIA)
        {{
            const double gamma_s = 1.0 - 1.0/sqrt(2.0); // ~0.2929
            const double t1 = t + gamma_s * dt;
            const double t2 = t + dt;
            const double tol_newton = 1e-10;
            const int    max_newton = 50;

            // -- Stage 1: solve  G1(K1) = K1 - f(x + dt*gamma*K1, t1) = 0 --
            // Initialise with explicit Euler prediction
            system_step(x, k1, t, Vd, Vq, Vt);   // k1 = f(x, t)  (initial guess)
            for(int nit = 0; nit < max_newton; ++nit) {{
                // Residual:  r = k1 - f(x + dt*gamma*k1,  t1)
                for(int j=0; j<N_STATES; ++j)
                    x_temp[j] = x[j] + dt * gamma_s * k1[j];
                system_step(x_temp, k2, t1, Vd, Vq, Vt);  // k2 = f(x_temp, t1)
                double res_norm = 0.0;
                for(int j=0; j<N_STATES; ++j) {{
                    res[j]  = k1[j] - k2[j];
                    res_norm += res[j]*res[j];
                }}
                res_norm = sqrt(res_norm);
                if(res_norm < tol_newton) break;

                // Jacobian-free Newton: approximate (I - dt*gamma*J)*delta = -r
                // using finite-difference Jacobian-vector product (Jv ~ (f(x+eps*v)-f(x))/eps)
                // We solve with a simple fixed-point / chord-Newton step:
                //   delta = -r / (1 - dt*gamma*df/dk)
                // For each component independently (diagonal approximation):
                double eps_fd = 1e-7;
                for(int j=0; j<N_STATES; ++j) {{
                    double k1j_save = k1[j];
                    k1[j] += eps_fd;
                    for(int jj=0; jj<N_STATES; ++jj)
                        x_temp[jj] = x[jj] + dt * gamma_s * k1[jj];
                    system_step(x_temp, k3, t1, Vd, Vq, Vt);
                    // df/dk1[j] ~ (f_j(k1+eps) - f_j(k1)) / eps
                    double dfdkj = (k3[j] - k2[j]) / eps_fd;
                    double diag  = 1.0 - dt * gamma_s * dfdkj;
                    if(fabs(diag) < 1e-14) diag = 1e-14;
                    k1[j] = k1j_save - res[j] / diag;
                }}
            }}
            // k1 now holds Stage-1 slope K1

            // -- Stage 2: solve  G2(K2) = K2 - f(x + dt*(1-gamma)*K1 + dt*gamma*K2, t2) = 0 --
            // Predictor: start from explicit estimate using K1
            for(int j=0; j<N_STATES; ++j)
                x_temp[j] = x[j] + dt * k1[j];   // full Euler step as guess
            system_step(x_temp, k2, t2, Vd, Vq, Vt);  // k2 = initial guess for K2

            for(int nit = 0; nit < max_newton; ++nit) {{
                for(int j=0; j<N_STATES; ++j)
                    x_temp[j] = x[j] + dt*(1.0-gamma_s)*k1[j] + dt*gamma_s*k2[j];
                system_step(x_temp, k3, t2, Vd, Vq, Vt);  // k3 = f(stage-2 point)
                double res_norm = 0.0;
                for(int j=0; j<N_STATES; ++j) {{
                    res[j]  = k2[j] - k3[j];
                    res_norm += res[j]*res[j];
                }}
                res_norm = sqrt(res_norm);
                if(res_norm < tol_newton) break;

                double eps_fd = 1e-7;
                for(int j=0; j<N_STATES; ++j) {{
                    double k2j_save = k2[j];
                    k2[j] += eps_fd;
                    for(int jj=0; jj<N_STATES; ++jj)
                        x_temp[jj] = x[jj] + dt*(1.0-gamma_s)*k1[jj] + dt*gamma_s*k2[jj];
                    system_step(x_temp, k4, t2, Vd, Vq, Vt);
                    double dfdkj = (k4[j] - k3[j]) / eps_fd;
                    double diag  = 1.0 - dt * gamma_s * dfdkj;
                    if(fabs(diag) < 1e-14) diag = 1e-14;
                    k2[j] = k2j_save - res[j] / diag;
                }}
            }}
            // k2 now holds Stage-2 slope K2

            // -- Update: x_new = x + dt*((1-gamma)*K1 + gamma*K2) --
            for(int j=0; j<N_STATES; ++j)
                x[j] += dt * ((1.0-gamma_s)*k1[j] + gamma_s*k2[j]);

            // Keep dxdt updated for the progress monitor
            system_step(x, dxdt, t+dt, Vd, Vq, Vt);
        }}
"""
        else:  # rk4 (default)
            solver_loop = f"""
        // RK4 Stage 1
        system_step(x, k1, t, Vd, Vq, Vt);
        
        // RK4 Stage 2
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + 0.5 * dt * k1[j];
        system_step(x_temp, k2, t + 0.5*dt, Vd, Vq, Vt);
        
        // RK4 Stage 3
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + 0.5 * dt * k2[j];
        system_step(x_temp, k3, t + 0.5*dt, Vd, Vq, Vt);
        
        // RK4 Stage 4
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + dt * k3[j];
        system_step(x_temp, k4, t + dt, Vd, Vq, Vt);
        
        // Final Update
        for(int j=0; j<N_STATES; ++j) x[j] += (dt/6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
"""

        return f"""
#include <fstream>
#include <iomanip>
#include <cmath>

int main() {{
    double x[N_STATES] = {{ {x0_str} }};
    double dxdt[N_STATES];
    
    // Solver stage buffers (used by RK2, RK4, and SDIRK-2)
    double k1[N_STATES], k2[N_STATES], k3[N_STATES], k4[N_STATES];
    double x_temp[N_STATES];
    double res[N_STATES];   // Newton residual (SDIRK-2)
    
    // Initial network voltages from power flow (Vd = V*cos(theta), Vq = V*sin(theta))
    double Vd[N_BUS] = {{ {Vd_str} }};
    double Vq[N_BUS] = {{ {Vq_str} }};
    double Vt[N_BUS] = {{0}};
    
    // Output File
    std::ofstream outfile("simulation_results.csv");
    outfile << "{header}" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double t = 0.0;
    const double dt = {dt};
    const int steps = {steps};
    const int log_every = 10; // log every 10 steps = every {dt*10:.4f} s
    
    {debug_block}
    
    std::cout << "Starting Simulation ({method.upper()}, T={duration}s, dt={dt})..." << std::endl;
    
    for(int i=0; i<steps; ++i) {{
        // Log (Decimated)
        if (i % log_every == 0) {{
            outfile << {log_parts} << std::endl;
        }}

        {solver_loop}
        
        t += dt;
        
        // Progress & safety check every 10 seconds
        if (i % (int)(10.0 / dt) == 0) {{
             // Recompute derivative at current state (dxdt not updated by RK4/RK2)
             system_step(x, dxdt, t, Vd, Vq, Vt);
             double max_d = 0.0;
             int max_d_idx = -1;
             for(int k=0; k<N_STATES; ++k) {{
                 if(fabs(dxdt[k]) > max_d) {{
                     max_d = fabs(dxdt[k]);
                     max_d_idx = k;
                 }}
             }}
             std::cout << "t=" << t << " max_d=" << max_d << " at state[" << max_d_idx << "] Vterm[0]=" << Vt[0] << std::endl;
             
             if(Vt[0] > 5.0 || std::isnan(Vt[0])) {{
                 std::cout << "Stability Limit Reached. Stopping." << std::endl;
                 break;
             }}
        }}
    }}
    
    outfile.close();
    std::cout << "Done. Results in simulation_results.csv" << std::endl;
    return 0;
}}
"""
