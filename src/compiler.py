import json
import os
import math
from typing import Dict, List, Any, Tuple
from src.core import PowerComponent
from src.ybus import YBusBuilder
from src.system_graph import SystemGraph, Wire, build_system_graph
import numpy as np


def _port_names(comp: PowerComponent, direction: str) -> List[str]:
    """Return list of port names for 'in' or 'out' direction."""
    return [p[0] for p in comp.port_schema[direction]]

def _has_port(comp: PowerComponent, direction: str, name: str) -> bool:
    """Return True if comp has a port named *name* in *direction*."""
    return name in _port_names(comp, direction)


class SystemCompiler:
    """
    Compiles a Power System defined in JSON into a high-performance C++ simulation kernel.

    Uses a ``SystemGraph`` as the single source of truth for component topology
    and signal wiring.  All case-specific wiring logic has been removed from
    this class and moved into the JSON ``connections`` block (or derived
    automatically by ``json_compat.to_new_format()``).
    """

    def __init__(self, system_json_path: str):
        self.json_path = system_json_path
        if not os.path.exists(system_json_path):
            raise FileNotFoundError(f"System JSON not found: {system_json_path}")

        # Build SystemGraph (upgrades old-format JSON transparently)
        self.graph: SystemGraph = build_system_graph(system_json_path)

        # Expose the (possibly upgraded) raw dict for downstream consumers
        # (YBusBuilder, PowerFlow) that still expect the old dict interface.
        self.data = self.graph.raw_data

        # Ordered component list and name→comp map (mirror from graph)
        self.components: List[PowerComponent] = list(self.graph.components.values())
        self.comp_map: Dict[str, PowerComponent] = dict(self.graph.components)

        # Wiring Map: { (target_comp, port_name): source_expr }
        # Populated by _resolve_wiring() from graph.wires.
        self.wiring_map: Dict[Tuple[str, str], str] = {}

        # PSS → exciter map: avr_name → PSS component (for Vref additive correction)
        self._pss_for_avr: Dict[str, PowerComponent] = {}

        # Memory Layout
        self.state_offsets: Dict[str, int] = {}
        self.total_states = 0

        # Network
        self.ybus_builder = YBusBuilder(self.data)
        self.n_bus = self.ybus_builder.n_bus
        self.z_bus = None  # Computed later

        # Fault event topologies: list of dicts sorted by event time
        self.fault_topologies: List[Dict] = []
        self.fault_events: List[Dict] = []

    def build_structure(self):
        """
        Parses system, builds component list, resolves wiring, and builds Y-bus.
        Must be called BEFORE initialization and code generation.
        """
        self._build_pss_map()
        self._assign_memory()
        self._resolve_wiring()
        self._parse_fault_events()

        # ── PHS structural validation ──────────────────────────────────
        # For every component that exposes a symbolic PHS definition,
        # verify the structural invariants (J skew-sym, R PSD, etc.)
        # BEFORE any C++ is generated.
        for comp in self.components:
            sphs = comp.get_symbolic_phs()
            if sphs is None:
                continue
            from src.symbolic.validation import validate_phs_structure
            report = validate_phs_structure(sphs)
            if not report.all_passed:
                failed = [c for c in report.checks if not c['passed']]
                msgs = "; ".join(
                    f"{c['name']}: {c['detail']}" for c in failed
                )
                raise ValueError(
                    f"PHS validation failed for {comp.name} "
                    f"({comp.__class__.__name__}): {msgs}"
                )
            print(f"  [PHS] {comp.name}: structural validation OK "
                  f"({report.n_passed}/{report.n_passed + report.n_failed} checks)")

        # Build base Y-Bus (lines, shunts -- NO loads for Kron approach)
        self.ybus_builder.build(include_loads=False)
        
        # Add Generator Impedances to Y-Bus (Norton Equivalent).
        # Use 'xd_double_prime' if present (GENROU), otherwise 'xd1' (GENCLS).
        # This is type-agnostic: any component with component_role='generator'
        # and a bus parameter contributes a Norton admittance.
        self._gen_bus_internal = []
        for comp in self.components:
            if comp.component_role != 'generator':
                continue
            if not comp.contributes_norton_admittance:
                # Current-source models (REGCA1, PVD1, …): no Norton admittance shunt.
                # Their bus still needs to be registered as an internal gen bus.
                bus_id = comp.params.get('bus')
                if bus_id in self.ybus_builder.bus_map:
                    self._gen_bus_internal.append(self.ybus_builder.bus_map[bus_id])
                continue
            bus_id = comp.params.get('bus')
            ra = comp.params.get('ra', 0.0)
            xd_pp = comp.params.get('xd_double_prime',
                    comp.params.get('xd1', 0.2))
            self.ybus_builder.add_generator_impedance(bus_id, ra, xd_pp)
            if bus_id in self.ybus_builder.bus_map:
                self._gen_bus_internal.append(self.ybus_builder.bus_map[bus_id])
        
        # Kron-reduced Z-bus is deferred until after power flow so we can
        # pass actual bus voltages to the load admittance calculation.
        # Call finalize_network() after running the power flow.
        self.z_bus = None
        self.active_buses = None
        self.n_active = None
        self.active_bus_map = None

    def _build_pss_map(self):
        """Populate _pss_for_avr and _observer_buses from the graph."""
        # PSS → exciter map (for Vref additive injection in wiring and CSV)
        self._pss_for_avr = {}
        for pss_name, pss_comp in self.graph.pss_components():
            avr_id = pss_comp.params.get("avr")
            if avr_id:
                self._pss_for_avr[avr_id] = pss_comp
                K1 = float(pss_comp.params.get("K1", pss_comp.params.get("KS", 0.0)))
                print(f"  [Compiler] PSS {pss_name} (K1/KS={K1}) wired to exciter {avr_id}")

        # Observer buses: buses without generators get voltage recovery
        gen_buses = set()
        for gen_name, gen_comp in self.graph.generator_components():
            bus_id = gen_comp.params.get("bus")
            if bus_id is not None:
                gen_buses.add(bus_id)

        self._observer_buses = []
        all_bus_ids = set(self.graph.buses.keys())
        for bus in sorted(all_bus_ids):
            if bus not in gen_buses:
                self._observer_buses.append(bus)
        if self._observer_buses:
            print(f"  [Compiler] Observer buses (voltage recovery): {self._observer_buses}")

    def _parse_fault_events(self):
        """Parse Toggler and BusFault entries to build fault topology timeline.

        Each point in time has a topology state described by:
          - which lines are disconnected (Toggler)
          - which bus-fault shunts are active (BusFault)

        A separate Kron-reduced Z-bus is computed for each unique state.
        """
        togglers = self.data.get('Toggler', [])
        bus_faults = self.data.get('BusFault', [])
        if not togglers and not bus_faults:
            return

        # Build a unified event timeline: list of (t, action) sorted by time.
        # action = ('toggle_line', dev_idx)
        #        | ('bus_fault_on',  bus, r, x, fault_id)
        #        | ('bus_fault_off', fault_id)
        timeline = []
        for ev in togglers:
            if ev.get('model', '') == 'Line':
                timeline.append((ev['t'], ('toggle_line', ev['dev'])))

        for i, bf in enumerate(bus_faults):
            fid = f"bf_{i}"
            t0 = float(bf['t_start'])
            # Support both 't_duration' (preferred) and 't_end' (legacy)
            if 't_duration' in bf:
                dur = float(bf['t_duration'])
            else:
                dur = float(bf.get('t_end', t0 + 0.1)) - t0
            timeline.append((t0,       ('bus_fault_on', bf['bus'],
                                        bf.get('r', 0.0), bf.get('x', 0.01),
                                        fid)))
            timeline.append((t0 + dur, ('bus_fault_off', fid)))

        timeline.sort(key=lambda x: x[0])

        # Walk timeline to build topology snapshots
        disconnected_lines: set = set()
        active_bus_faults: Dict[str, dict] = {}   # fid -> {bus, r, x}
        topo_changes = []                          # (t, disconnected, bus_faults)

        for t_ev, action in timeline:
            kind = action[0]
            if kind == 'toggle_line':
                dev = action[1]
                if dev in disconnected_lines:
                    disconnected_lines.discard(dev)
                else:
                    disconnected_lines.add(dev)
            elif kind == 'bus_fault_on':
                _, bus, r, x, fid = action
                active_bus_faults[fid] = {'bus': bus, 'r': r, 'x': x}
            elif kind == 'bus_fault_off':
                fid = action[1]
                active_bus_faults.pop(fid, None)

            snap_lines = frozenset(disconnected_lines)
            snap_faults = frozenset(
                (d['bus'], d['r'], d['x'])
                for d in active_bus_faults.values()
            )
            topo_changes.append((t_ev, snap_lines, snap_faults))

        self.fault_events = [{'t': tc[0]} for tc in topo_changes]

        # Deduplicate (keep first occurrence of each unique state)
        seen = set()
        unique = []
        for t_ev, lines, faults in topo_changes:
            key = (lines, faults)
            if key not in seen:
                seen.add(key)
                unique.append((t_ev, lines, faults))
        self._fault_topo_changes = unique

    def finalize_network(self, pf_V=None, pf_theta=None):
        """Build the Kron-reduced Z-bus using power-flow voltages.
        
        Must be called after build_structure() AND after the power flow solve
        so that load admittances use the correct |V0|^2 denominator (fixes the
        ~0.2 pu initial voltage mismatch, Issue #1).
        
        Also builds faulted Z-bus topologies for each Toggler event (Issue #4).
        
        Args:
            pf_V:     power-flow voltage magnitude array (n_bus, internal indices)
            pf_theta: power-flow voltage angle array (n_bus, internal indices)
        """
        self.z_bus, self.active_buses = self.ybus_builder.get_z_bus_kron(
            self._gen_bus_internal, pf_V=pf_V, pf_theta=pf_theta
        )
        self.n_active = len(self.active_buses)
        self.active_bus_map = {ab: i for i, ab in enumerate(self.active_buses)}
        # Slack (infinite) bus Norton current in Kron-reduced active bus frame
        kron = getattr(self.ybus_builder, '_last_kron', {})
        self.slack_norton_I = kron.get('slack_norton_I',
                                       np.zeros(self.n_active, dtype=complex))

        # Voltage recovery matrix for observer buses (Slack buses without generators)
        self.observer_recovery = None
        self.observer_bus_names = []
        obs_buses = getattr(self, '_observer_buses', [])
        if obs_buses:
            M, names = self.ybus_builder.get_voltage_recovery_matrix(obs_buses)
            if M is not None:
                self.observer_recovery = M
                self.observer_bus_names = names
                print(f"  [Compiler] Voltage recovery matrix for observer buses: {names}")
        
        # Build faulted Z-bus matrices for each unique topology
        self.fault_topologies = []
        topo_changes = getattr(self, '_fault_topo_changes', [])
        for t_ev, disconnected_lines, bus_fault_shunts in topo_changes:
            faulted_data = dict(self.data)
            faulted_data['Line'] = [
                ln for ln in self.data.get('Line', [])
                if ln['idx'] not in disconnected_lines
            ]
            faulted_ybus = YBusBuilder(faulted_data)
            faulted_ybus.build(include_loads=False)
            for comp in self.components:
                if comp.component_role != 'generator':
                    continue
                if not comp.contributes_norton_admittance:
                    # Current-source models: no Norton admittance, same as build_structure
                    continue
                bus_id = comp.params.get('bus')
                ra = comp.params.get('ra', 0.0)
                xd_pp = comp.params.get('xd_double_prime',
                        comp.params.get('xd1', 0.2))
                faulted_ybus.add_generator_impedance(bus_id, ra, xd_pp)
            for bus, r, x in bus_fault_shunts:
                faulted_ybus.add_fault_shunt(bus, r, x)
            z_faulted, _ = faulted_ybus.get_z_bus_kron(
                self._gen_bus_internal, pf_V=pf_V, pf_theta=pf_theta
            )
            # Recovery matrix for this faulted topology
            fault_M = None
            if obs_buses:
                fM, fnames = faulted_ybus.get_voltage_recovery_matrix(obs_buses)
                if fM is not None:
                    fault_M = fM
            parts = []
            if disconnected_lines:
                parts.append(f"no_{'+'.join(sorted(str(l) for l in disconnected_lines))}")
            if bus_fault_shunts:
                parts.append(f"fault@bus{'_'.join(str(b) for b,_,_ in bus_fault_shunts)}")
            label = '_'.join(parts) if parts else "restored"
            fkron = getattr(faulted_ybus, '_last_kron', {})
            f_slack_I = fkron.get('slack_norton_I',
                                   np.zeros(self.n_active, dtype=complex))
            self.fault_topologies.append({
                't': t_ev, 'z_bus': z_faulted, 'label': label,
                'observer_M': fault_M,
                'slack_norton_I': f_slack_I,
            })

    def generate_cpp(self) -> str:
        """
        Generates the C++ kernel code.
        Must be called AFTER build_structure() and AFTER initialization (to capture updated params).
        """
        if self.z_bus is None:
            raise RuntimeError("Must call build_structure() before generate_cpp()")
        
        self._resolve_wiring()
        self._refresh_control_params()
        return self._generate_cpp()

    def _refresh_control_params(self):
        """Re-read Vref/Pref/Efd0/Tm0 and remap bus indices to reduced network."""
        # Remap __BUS_X_Vd/Vq/Vterm placeholders to reduced indices
        for key, val in list(self.wiring_map.items()):
            if isinstance(val, str) and val.startswith('__BUS_'):
                stripped = val[len('__BUS_'):]
                last_us = stripped.rfind('_')
                bus_id = int(stripped[:last_us])
                sig = stripped[last_us+1:]
                full_idx = self.ybus_builder.bus_map[bus_id]
                if full_idx in self.active_bus_map:
                    red_idx = self.active_bus_map[full_idx]
                    if sig == 'Vd':
                        self.wiring_map[key] = f"Vd_net[{red_idx}]"
                    elif sig == 'Vq':
                        self.wiring_map[key] = f"Vq_net[{red_idx}]"
                    elif sig == 'Vterm':
                        self.wiring_map[key] = f"Vterm_net[{red_idx}]"
                else:
                    self.wiring_map[key] = "0.0"
        
        for comp in self.components:
            role = comp.component_role
            if role == 'exciter' and _has_port(comp, 'in', 'Vref'):
                if 'Vref' in comp.params:
                    vref_val = str(comp.params['Vref'])
                    pss_comp = self._pss_for_avr.get(comp.name)
                    if pss_comp is not None:
                        self.wiring_map[(comp.name, 'Vref')] = \
                            f"{vref_val} + outputs_{pss_comp.name}[0]"
                    else:
                        self.wiring_map[(comp.name, 'Vref')] = vref_val
            if role == 'governor' and _has_port(comp, 'in', 'Pref'):
                if 'Pref' in comp.params:
                    self.wiring_map[(comp.name, 'Pref')] = str(comp.params['Pref'])
            # Default u_agc to 0.0 for governors not connected to an AGC
            if role == 'governor' and _has_port(comp, 'in', 'u_agc'):
                self.wiring_map.setdefault((comp.name, 'u_agc'), '0.0')
            # Default Id/Iq to 0.0 for PMUs without current measurement wired
            if role == 'measurement' and _has_port(comp, 'in', 'Id'):
                self.wiring_map.setdefault((comp.name, 'Id'), '0.0')
            if role == 'measurement' and _has_port(comp, 'in', 'Iq'):
                self.wiring_map.setdefault((comp.name, 'Iq'), '0.0')
            if role == 'generator':
                if 'Tm0' in comp.params:
                    curr_tm = self.wiring_map.get((comp.name, 'Tm'))
                    # Only skip update if Tm is already wired to a live component
                    # output (i.e., a governor).  Constants — including "0.0" from
                    # an early PARAM: evaluation before init set Tm0 — must be
                    # overwritten with the post-init equilibrium value.
                    if curr_tm is None or not curr_tm.startswith('outputs_'):
                        self.wiring_map[(comp.name, 'Tm')] = str(comp.params['Tm0'])
                if 'Efd0' in comp.params:
                    curr_efd = self.wiring_map.get((comp.name, 'Efd'))
                    # Same logic for Efd: skip only if wired to an exciter output.
                    if curr_efd is None or not curr_efd.startswith('outputs_'):
                        self.wiring_map[(comp.name, 'Efd')] = str(comp.params['Efd0'])
            if role == 'renewable_controller':
                if 'Pref0' in comp.params:
                    curr_pref = self.wiring_map.get((comp.name, 'Pref'))
                    # Only fall back to the frozen init constant if Pref is NOT
                    # already wired to a live component output (e.g. WIND_AERO.Pm).
                    if curr_pref is None or not curr_pref.startswith('outputs_'):
                        self.wiring_map[(comp.name, 'Pref')] = str(comp.params['Pref0'])
                if 'Qref0' in comp.params:
                    curr_qref = self.wiring_map.get((comp.name, 'Qref'))
                    if curr_qref is None or not curr_qref.startswith('outputs_'):
                        self.wiring_map[(comp.name, 'Qref')] = str(comp.params['Qref0'])

    # ------------------------------------------------------------------
    # Tier 2 Co-Simulation: generate shared-library C++ kernel
    # ------------------------------------------------------------------

    def generate_cosim_cpp(self, cosim_config) -> str:
        """
        Generate a co-simulation C++ kernel that can be compiled as a
        shared library (``plant.so``).

        Differences from ``generate_cpp()``:

        1. Selected ``wiring_map`` entries are overridden to read from
           ``cosim_u[i]`` (runtime control inputs from the Python side).
        2. The ``system_step()`` signature gains two extra parameters:
           ``double* cosim_u, double* cosim_y``.
        3. Z-matrix arrays become ``static double`` (writable) so
           ``plant_swap_topology()`` can push a new Kron Z at runtime.
        4. A ``cosim_y`` population block is injected at the end of
           ``system_step()`` to expose measurement outputs.
        5. Constants ``N_CTRL`` / ``N_MEAS`` are prepended.
        6. A full ``extern "C"`` interface block is appended (see
           ``_generate_cosim_interface()``).

        Parameters
        ----------
        cosim_config : CosimConfig
            Port declarations.  All ``CosimControlPort`` entries whose
            ``(component_name, port_name)`` key exists in ``wiring_map``
            will be redirected to ``cosim_u[index]``.

        Returns
        -------
        str
            Complete C++ source for ``plant.cpp``.
        """
        if self.z_bus is None:
            raise RuntimeError("Must call build_structure() before generate_cosim_cpp()")

        # 1. Let _refresh_control_params() resolve all equilibrium values first
        #    (Pref, Vref, Efd0, Tm0 get set from post-init params).
        self._resolve_wiring()
        self._refresh_control_params()

        # 2. Now override the chosen wiring_map entries with cosim_u[i].
        #    We do this AFTER refresh so the refresh doesn't overwrite them.
        wiring_backup = dict(self.wiring_map)
        for cp in cosim_config.ctrl_ports:
            key = (cp.component_name, cp.port_name)
            self.wiring_map[key] = f"cosim_u[{cp.index}]"

        # 2. Generate the base C++ (caller wiring_map is now patched)
        cpp = self._generate_cpp()

        # 3. Restore wiring_map so subsequent calls are unaffected
        self.wiring_map = wiring_backup

        # 4. Patch Z-matrix arrays: const → static so swap_topology can write
        cpp = cpp.replace(
            "const double Z_real_all[",
            "static double Z_real_all["
        )
        cpp = cpp.replace(
            "const double Z_imag_all[",
            "static double Z_imag_all["
        )

        # 5. Patch system_step signature to accept cosim_u and cosim_y
        old_sig = ("void system_step(double* x, double* dxdt, double t, "
                   "double* Vd_net, double* Vq_net, double* Vterm_net) {")
        new_sig = ("void system_step(double* x, double* dxdt, double t, "
                   "double* Vd_net, double* Vq_net, double* Vterm_net, "
                   "double* cosim_u, double* cosim_y) {")
        cpp = cpp.replace(old_sig, new_sig)

        # 6. Inject cosim_y population block before the final closing brace
        #    (which is the closing } of system_step — always the last line)
        meas_lines = []
        if cosim_config.n_meas > 0:
            meas_lines.append("    // --- CoSim Measurements (cosim_y) ---")
            for mp in cosim_config.meas_ports:
                meas_lines.append(
                    f"    if(cosim_y) cosim_y[{mp.index}] = {mp.cpp_expr};"
                )
        if meas_lines:
            injection = "\n" + "\n".join(meas_lines) + "\n"
            # Replace the last bare "}" (closing brace of system_step)
            last_brace = cpp.rfind("\n}")
            if last_brace != -1:
                cpp = cpp[:last_brace] + injection + "\n}" + cpp[last_brace + 2:]

        # 7. Prepend N_CTRL / N_MEAS constants
        header = (
            f"const int N_CTRL = {cosim_config.n_ctrl};\n"
            f"const int N_MEAS = {cosim_config.n_meas};\n\n"
        )
        cpp = header + cpp

        # 8. Append extern "C" interface
        cpp += "\n\n" + self._generate_cosim_interface(cosim_config)

        return cpp

    def _generate_cosim_interface(self, cosim_config) -> str:
        """
        Emit the ``extern "C"`` ABI expected by ``PlantInterface``.

        Emitted symbols
        ---------------
        ``plant_init``         — copy x0/Vd0/Vq0 into static buffers, set
                                 default cosim_u values
        ``plant_set_inputs``   — write cosim_u buffer
        ``plant_step_rk4``     — RK4 integrator (calls system_step 4×)
        ``plant_get_outputs``  — read cosim_y buffer (populated by last step)
        ``plant_get_state``    — read full state vector x
        ``plant_swap_topology``— overwrite Z_real_all[0] / Z_imag_all[0]
        ``plant_n_states``     — return N_STATES
        ``plant_n_bus``        — return N_BUS
        ``plant_n_ctrl``       — return N_CTRL
        ``plant_n_meas``       — return N_MEAS
        """
        na = self.n_active
        ns = self.total_states
        n_ctrl = cosim_config.n_ctrl
        n_meas = cosim_config.n_meas
        defaults = [cp.default_value for cp in cosim_config.ctrl_ports]
        defaults_str = ", ".join(f"{v:.10e}" for v in defaults)

        lines = []
        lines.append('// ============================================================')
        lines.append('// extern "C" Plant Interface  (Tier 2 Co-Simulation ABI)')
        lines.append('// ============================================================')
        lines.append('extern "C" {')
        lines.append('')

        # Static buffers
        lines.append(f'static double _x[{ns}];')
        lines.append(f'static double _dxdt[{ns}];')
        lines.append(f'static double _Vd[{na}];')
        lines.append(f'static double _Vq[{na}];')
        lines.append(f'static double _Vterm[{na}];')
        lines.append(f'static double _cosim_u[{max(n_ctrl, 1)}]'
                     f' = {{{defaults_str if n_ctrl > 0 else "0.0"}}};')
        lines.append(f'static double _cosim_y[{max(n_meas, 1)}];')
        lines.append(f'static double _k1[{ns}], _k2[{ns}], _k3[{ns}], _k4[{ns}];')
        lines.append(f'static double _xt[{ns}];')
        lines.append('')

        # plant_init
        lines.append('void plant_init(double* x0, int n, double* Vd0, double* Vq0, int nb) {')
        lines.append(f'    for(int i=0; i<n; ++i) _x[i] = x0[i];')
        lines.append(f'    for(int i=0; i<nb; ++i) {{ _Vd[i]=Vd0[i]; _Vq[i]=Vq0[i]; }}')
        if n_ctrl > 0:
            lines.append(f'    // Initialise cosim_u to defaults')
            for cp in cosim_config.ctrl_ports:
                lines.append(f'    _cosim_u[{cp.index}] = {cp.default_value:.10e};')
        lines.append('    // Run one step to populate outputs and cosim_y')
        lines.append(f'    for(int i=0; i<{na}; ++i) _Vterm[i]=0.0;')
        lines.append('    system_step(_x, _dxdt, 0.0, _Vd, _Vq, _Vterm, _cosim_u, _cosim_y);')
        lines.append('}')
        lines.append('')

        # plant_set_inputs
        lines.append('void plant_set_inputs(double* u, int n) {')
        lines.append(f'    for(int i=0; i<n; ++i) _cosim_u[i]=u[i];')
        lines.append('}')
        lines.append('')

        # plant_step_rk4  (classic RK4, updates _x and repopulates _cosim_y)
        lines.append('void plant_step_rk4(double dt, double t) {')
        lines.append(f'    int ns = {ns}, nb = {na};')
        # k1
        lines.append('    system_step(_x, _k1, t, _Vd, _Vq, _Vterm, _cosim_u, _cosim_y);')
        # k2
        lines.append(f'    for(int i=0; i<ns; ++i) _xt[i] = _x[i] + 0.5*dt*_k1[i];')
        lines.append('    system_step(_xt, _k2, t+0.5*dt, _Vd, _Vq, _Vterm, _cosim_u, nullptr);')
        # k3
        lines.append(f'    for(int i=0; i<ns; ++i) _xt[i] = _x[i] + 0.5*dt*_k2[i];')
        lines.append('    system_step(_xt, _k3, t+0.5*dt, _Vd, _Vq, _Vterm, _cosim_u, nullptr);')
        # k4
        lines.append(f'    for(int i=0; i<ns; ++i) _xt[i] = _x[i] + dt*_k3[i];')
        lines.append('    system_step(_xt, _k4, t+dt,     _Vd, _Vq, _Vterm, _cosim_u, nullptr);')
        # Final update
        lines.append(f'    for(int i=0; i<ns; ++i)')
        lines.append(f'        _x[i] += (dt/6.0)*(_k1[i] + 2.0*_k2[i] + 2.0*_k3[i] + _k4[i]);')
        # Re-run outputs at final state (populates _cosim_y with updated values)
        lines.append('    system_step(_x, _dxdt, t+dt, _Vd, _Vq, _Vterm, _cosim_u, _cosim_y);')
        lines.append('}')
        lines.append('')

        # plant_get_outputs
        lines.append('void plant_get_outputs(double* y_out, int n) {')
        lines.append(f'    for(int i=0; i<n; ++i) y_out[i] = _cosim_y[i];')
        lines.append('}')
        lines.append('')

        # plant_get_state
        lines.append('void plant_get_state(double* x_out, int n) {')
        lines.append(f'    for(int i=0; i<n; ++i) x_out[i] = _x[i];')
        lines.append('}')
        lines.append('')

        # plant_swap_topology
        lines.append('void plant_swap_topology(double* Zr, double* Zi, int nb) {')
        lines.append(f'    int nn = nb * nb;')
        lines.append(f'    for(int i=0; i<nn; ++i) {{')
        lines.append(f'        Z_real_all[0][i] = Zr[i];')
        lines.append(f'        Z_imag_all[0][i] = Zi[i];')
        lines.append(f'    }}')
        lines.append('}')
        lines.append('')

        # Dimension queries
        lines.append(f'int plant_n_states() {{ return {ns}; }}')
        lines.append(f'int plant_n_bus()    {{ return {na}; }}')
        lines.append(f'int plant_n_ctrl()   {{ return {n_ctrl}; }}')
        lines.append(f'int plant_n_meas()   {{ return {n_meas}; }}')
        lines.append('')
        lines.append('} // extern "C"')

        return "\n".join(lines)

    def compile(self) -> str:
        """Legacy helper for single-shot compilation (no init injection)."""
        self.build_structure()
        return self.generate_cpp()

    def _assign_memory(self):
        offset = 0
        for comp in self.components:
            self.state_offsets[comp.name] = offset
            offset += len(comp.state_schema)
        self.delta_coi_idx = offset
        self.total_states = offset + 1

    def _resolve_wiring(self):
        """Translate SystemGraph wires to C++ wiring_map expressions.

        Iterates over ``self.graph.wires`` and calls
        ``_wire_src_to_placeholder()`` to convert each wire source to a
        placeholder string.  Network bus sources use a ``__BUS_`` placeholder
        that is later resolved to concrete Kron-reduced indices by
        ``_refresh_control_params()`` (called after finalize_network()).

        Later rules (lower in the wire list) override earlier ones for the
        same destination, allowing the JSON author to override defaults.

        Special handling:
          - ``DQ_<gen>.{Vd|Vq}`` sources → ``vd_dq_<gen>`` / ``vq_dq_<gen>``
            (dq-frame voltages computed inside the iterative network loop)
          - Exciter ``Vref`` with an associated PSS → adds the PSS signal
        """
        for wire in self.graph.wires:
            dst_comp_name = wire.dst_component()
            dst_port      = wire.dst_port()
            src           = wire.src

            expr = self._wire_src_to_placeholder(src, dst_comp_name, dst_port)
            if expr is not None:
                self.wiring_map[(dst_comp_name, dst_port)] = expr

        # Post-pass: inject PSS correction into exciter Vref expressions.
        # The PSS signal is added on top of the base Vref constant.
        for avr_name, pss_comp in self._pss_for_avr.items():
            key = (avr_name, "Vref")
            base = self.wiring_map.get(key, "1.0")
            # Only inject if base is a plain number (not already wired to a component)
            try:
                float(base)
                self.wiring_map[key] = f"{base} + outputs_{pss_comp.name}[0]"
            except ValueError:
                pass  # Already complex expression — leave untouched

    def _wire_src_to_placeholder(self, src: str, dst_comp: str, dst_port: str):
        """Convert a wire source string to a C++ expression / placeholder.

        Returns None to skip (e.g. invalid source), a string placeholder
        otherwise.  ``__BUS_<id>_<signal>`` placeholders are resolved later
        by ``_refresh_control_params()``.
        """
        # --- DQ_<gen>.{Vd|Vq}: dq-frame terminal voltage from generator ---
        if src.startswith("DQ_"):
            rest = src[len("DQ_"):]
            gen_name, sig = rest.split(".", 1)
            if sig in ("Vd", "Vd_dq"):
                return f"vd_dq_{gen_name}"
            if sig in ("Vq", "Vq_dq"):
                return f"vq_dq_{gen_name}"
            return None

        kind = Wire(src=src, dst="X.y").src_kind()

        # --- CONST:<float> ---
        if kind == "const":
            return src[len("CONST:"):]

        # --- PARAM:<comp>.<key> ---
        if kind == "param":
            rest = src[len("PARAM:"):]
            comp_name, param_key = rest.split(".", 1)
            comp = self.comp_map.get(comp_name)
            if comp is None:
                return "0.0"
            # If the parameter is not yet set (e.g. Pref0 before init), return a placeholder
            # that will be overwritten by _refresh_control_params later.
            if param_key not in comp.params:
                return "0.0"
            return str(float(comp.params.get(param_key, 0.0)))

        # --- BUS_<id>.<signal>: placeholder resolved after Kron ---
        if kind == "bus":
            parts = src.split(".")
            bus_id_str = parts[0][len("BUS_"):]
            signal     = parts[1]
            try:
                bus_id = int(bus_id_str)
            except ValueError:
                return "0.0"
            return f"__BUS_{bus_id}_{signal}"

        # --- <comp>.<port>: output of another component ---
        if kind == "comp":
            src_comp_name = Wire(src=src, dst="X.y").src_component()
            src_port_name = Wire(src=src, dst="X.y").src_port()
            src_comp = self.comp_map.get(src_comp_name)
            if src_comp is None:
                return "0.0"
            out_ports = _port_names(src_comp, "out")
            if src_port_name not in out_ports:
                return "0.0"
            idx = out_ports.index(src_port_name)
            return f"outputs_{src_comp_name}[{idx}]"

        return None

    def _generate_cpp(self) -> str:
        na = self.n_active
        n_topos = 1 + len(self.fault_topologies)  # pre-fault + each event
        code = []
        code.append("#define _USE_MATH_DEFINES")
        code.append("#include <math.h>")
        code.append("#include <vector>")
        code.append("#include <iostream>")
        code.append("")
        
        code.append(f"const int N_STATES = {self.total_states};")
        code.append(f"const int N_COMPONENTS = {len(self.components)};")
        code.append(f"const int N_BUS = {na};")
        code.append(f"const int N_TOPOS = {n_topos};")
        code.append("")
        
        # Emit pre-fault Z-bus (topology 0)
        code.append(f"// Topology 0: Pre-fault Kron-reduced Z-Bus ({na}x{na})")
        all_z_real = [self.z_bus.real.flatten()]
        all_z_imag = [self.z_bus.imag.flatten()]
        for topo in self.fault_topologies:
            all_z_real.append(topo['z_bus'].real.flatten())
            all_z_imag.append(topo['z_bus'].imag.flatten())
        
        # Flatten into 2D arrays: Z_real_all[topo][bus*N_BUS + bus]
        code.append(f"const double Z_real_all[{n_topos}][{na * na}] = {{")
        for i, zr in enumerate(all_z_real):
            label = "pre-fault" if i == 0 else self.fault_topologies[i-1]['label']
            code.append(f"    {{ // topo {i}: {label}")
            code.append(f"        {', '.join(f'{v:.6e}' for v in zr)}")
            code.append(f"    }}{',' if i < n_topos - 1 else ''}")
        code.append("};")
        
        code.append(f"const double Z_imag_all[{n_topos}][{na * na}] = {{")
        for i, zi in enumerate(all_z_imag):
            label = "pre-fault" if i == 0 else self.fault_topologies[i-1]['label']
            code.append(f"    {{ // topo {i}: {label}")
            code.append(f"        {', '.join(f'{v:.6e}' for v in zi)}")
            code.append(f"    }}{',' if i < n_topos - 1 else ''}")
        code.append("};")

        # Slack (infinite) bus Norton current bias — constant per topology.
        # V_active = Z_red * (I_gen + I_slack_bias), so we pre-add the bias.
        # Format: I_slack_d[topo][bus] = Re(slack_norton_I), I_slack_q[topo][bus] = Im(...)
        all_slack_d = [self.slack_norton_I.real]
        all_slack_q = [self.slack_norton_I.imag]
        for topo in self.fault_topologies:
            si = topo.get('slack_norton_I', np.zeros(na, dtype=complex))
            all_slack_d.append(si.real)
            all_slack_q.append(si.imag)
        code.append(f"// Slack-bus Norton current bias per topology (network Re/Im frame)")
        code.append(f"const double I_slack_d_all[{n_topos}][{na}] = {{")
        for i, sd in enumerate(all_slack_d):
            code.append(f"    {{ {', '.join(f'{v:.10e}' for v in sd)} }}"
                        f"{',' if i < n_topos - 1 else ''}")
        code.append("};")
        code.append(f"const double I_slack_q_all[{n_topos}][{na}] = {{")
        for i, sq in enumerate(all_slack_q):
            code.append(f"    {{ {', '.join(f'{v:.10e}' for v in sq)} }}"
                        f"{',' if i < n_topos - 1 else ''}")
        code.append("};")

        # Per-bus frequency-dependent load arrays (for kpf/kqf correction)
        # Aggregate load admittance and frequency factors on active (retained) buses.
        load_G = np.zeros(na)
        load_B = np.zeros(na)
        load_KPF = np.zeros(na)
        load_KQF = np.zeros(na)
        for comp in self.components:
            if comp.component_role == 'load':
                bus_id = comp.params.get('bus')
                if bus_id in self.ybus_builder.bus_map:
                    full_idx = self.ybus_builder.bus_map[bus_id]
                    if full_idx in self.active_bus_map:
                        red_idx = self.active_bus_map[full_idx]
                        P0 = float(comp.params.get('P0', 0.0))
                        Q0 = float(comp.params.get('Q0', 0.0))
                        V0 = float(comp.params.get('V0', 1.0))
                        V02 = max(V0 * V0, 1e-12)
                        kpf = float(comp.params.get('kpf', 0.0))
                        kqf = float(comp.params.get('kqf', 0.0))
                        load_G[red_idx] += P0 / V02
                        load_B[red_idx] += -Q0 / V02
                        # Weighted average kpf/kqf if multiple loads on same bus
                        if abs(P0) > 1e-12:
                            load_KPF[red_idx] = kpf
                        if abs(Q0) > 1e-12:
                            load_KQF[red_idx] = kqf
        has_freq_dep_load = np.any(load_KPF != 0.0) or np.any(load_KQF != 0.0)
        if has_freq_dep_load:
            code.append(f"// Frequency-dependent load: G_load, B_load, kpf, kqf per active bus")
            code.append(f"const double LOAD_G[{na}] = {{ {', '.join(f'{v:.10e}' for v in load_G)} }};")
            code.append(f"const double LOAD_B[{na}] = {{ {', '.join(f'{v:.10e}' for v in load_B)} }};")
            code.append(f"const double LOAD_KPF[{na}] = {{ {', '.join(f'{v:.10e}' for v in load_KPF)} }};")
            code.append(f"const double LOAD_KQF[{na}] = {{ {', '.join(f'{v:.10e}' for v in load_KQF)} }};")
        code.append("")

        # Event time table (topology switches at these times)
        if self.fault_topologies:
            event_times = [topo['t'] for topo in self.fault_topologies]
            code.append(f"// Fault event times: topology switches from 0->1->2->... at these times")
            code.append(f"const double FAULT_TIMES[{len(event_times)}] = {{")
            code.append(f"    {', '.join(f'{t:.6f}' for t in event_times)}")
            code.append("};")
            code.append(f"const int N_FAULT_EVENTS = {len(event_times)};")
        else:
            code.append("const int N_FAULT_EVENTS = 0;")
        code.append("")

        # Observer bus voltage recovery matrices (one per topology)
        n_obs = len(self.observer_bus_names) if self.observer_recovery is not None else 0
        code.append(f"const int N_OBS = {n_obs};")
        if n_obs > 0:
            all_M = [self.observer_recovery]
            for topo in self.fault_topologies:
                fM = topo.get('observer_M')
                all_M.append(fM if fM is not None else np.zeros((n_obs, na)))
            code.append(f"const double OBS_M_real[{n_topos}][{n_obs * na}] = {{")
            for ti, M in enumerate(all_M):
                mr = M.real.flatten()
                code.append(f"    {{{', '.join(f'{v:.10e}' for v in mr)}}}"
                            f"{',' if ti < n_topos - 1 else ''}")
            code.append("};")
            code.append(f"const double OBS_M_imag[{n_topos}][{n_obs * na}] = {{")
            for ti, M in enumerate(all_M):
                mi = M.imag.flatten()
                code.append(f"    {{{', '.join(f'{v:.10e}' for v in mi)}}}"
                            f"{',' if ti < n_topos - 1 else ''}")
            code.append("};")
        code.append(f"double Vd_obs[{max(n_obs, 1)}];")
        code.append(f"double Vq_obs[{max(n_obs, 1)}];")
        code.append(f"double Vterm_obs[{max(n_obs, 1)}];")
        code.append("")

        # Global output buffers (accessible from both system_step and main)
        for comp in self.components:
            n_out = len(comp.port_schema['out'])
            if n_out > 0:
                code.append(f"double outputs_{comp.name}[{n_out}];")
            n_in = len(comp.port_schema['in'])
            if n_in > 0:
                code.append(f"double inputs_{comp.name}[{n_in}];")
        code.append("")
        
        for comp in self.components:
            code.append(self._generate_instance_step(comp))
            code.append(self._generate_instance_output(comp))
            
        code.append("void system_step(double* x, double* dxdt, double t, double* Vd_net, double* Vq_net, double* Vterm_net) {")
        
        # Topology selection based on fault events
        code.append("    int topo_idx = 0;")
        if self.fault_topologies:
            code.append("    // --- Topology Selection (Fault Events) ---")
            code.append("    for(int e = 0; e < N_FAULT_EVENTS; ++e) {")
            code.append("        if(t >= FAULT_TIMES[e]) topo_idx = e + 1;")
            code.append("    }")
        code.append("    const double* Z_real = Z_real_all[topo_idx];")
        code.append("    const double* Z_imag = Z_imag_all[topo_idx];")
        code.append("    const double* I_slack_d = I_slack_d_all[topo_idx];")
        code.append("    const double* I_slack_q = I_slack_q_all[topo_idx];")
        
        # Injection Accumulators (sized to N_BUS = n_active)
        code.append(f"    double Id_inj[{na}];")
        code.append(f"    double Iq_inj[{na}];")
        
        # Pre-compute list of generators that expose dq-frame current ports.
        # These need a Park-transformed terminal voltage and actual stator
        # currents (not Norton currents) for correct exciter VE computation.
        # NOTE: RI-frame generators (e.g. DFIG) are EXCLUDED — they do not
        # use a Park transform and their id_dq/iq_dq are already correct
        # from compute_outputs().  Applying a GENROU-style dq refresh to
        # an RI-frame machine would use phi_sd (a flux) as a rotation angle
        # and overwrite correct stator currents with garbage.
        gen_comps_for_dq = [
            comp for comp in self.components
            if comp.component_role == 'generator'
            and not getattr(comp, 'uses_ri_frame', False)
            and comp.params.get('bus') in self.ybus_builder.bus_map
            and self.ybus_builder.bus_map[comp.params['bus']] in self.active_bus_map
            and 'id_dq' in _port_names(comp, 'out')
        ]

        # Pre-declare dq-frame voltage variables at function scope so they are
        # visible both inside the iterative loop (for the exciter output step)
        # and in section 3 (dynamics).  Initialised from the seed voltage.
        if gen_comps_for_dq:
            code.append(f"\n    // dq-frame voltages for exciter inputs (updated inside loop)")
            for comp in gen_comps_for_dq:
                off     = self.state_offsets[comp.name]
                bus_id  = comp.params['bus']
                red_idx = self.active_bus_map[self.ybus_builder.bus_map[bus_id]]
                code.append(f"    double vd_dq_{comp.name} = Vd_net[{red_idx}]*sin(x[{off}]) - Vq_net[{red_idx}]*cos(x[{off}]);")
                code.append(f"    double vq_dq_{comp.name} = Vd_net[{red_idx}]*cos(x[{off}]) + Vq_net[{red_idx}]*sin(x[{off}]);")

        # Helper: emit code to (re)compute dq voltages and actual id/iq for one gen
        def _emit_dq_update(comp, indent="        "):
            off     = self.state_offsets[comp.name]
            bus_id  = comp.params['bus']
            red_idx = self.active_bus_map[self.ybus_builder.bus_map[bus_id]]
            ra    = float(comp.params.get('ra', 0.0))
            xd_pp = float(comp.params.get('xd_double_prime', comp.params.get('xd1', 0.2)))
            xq_pp = float(comp.params.get('xq_double_prime', xd_pp))
            xd_p  = float(comp.params.get('xd_prime', comp.params.get('xd1', 0.3)))
            xq_p  = float(comp.params.get('xq_prime', comp.params.get('xq1', 0.3)))
            xl    = float(comp.params.get('xl', 0.0))
            det   = ra*ra + xd_pp*xq_pp
            kd    = (xd_pp - xl) / (xd_p - xl) if (xd_p - xl) != 0 else 1.0
            kq    = (xq_pp - xl) / (xq_p - xl) if (xq_p - xl) != 0 else 1.0
            out_names = _port_names(comp, 'out')
            id_dq_idx = out_names.index('id_dq')
            iq_dq_idx = out_names.index('iq_dq')
            lines = []
            # Helper to wrap negative literals in parentheses for C++
            def _cpp(v): return f"({v:.6f})" if v < 0 else f"{v:.6f}"
            lines.append(f"{indent}vd_dq_{comp.name} = Vd_net[{red_idx}]*sin(x[{off}]) - Vq_net[{red_idx}]*cos(x[{off}]);")
            lines.append(f"{indent}vq_dq_{comp.name} = Vd_net[{red_idx}]*cos(x[{off}]) + Vq_net[{red_idx}]*sin(x[{off}]);")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}    double psi_d_pp_x = x[{off+2}]*{_cpp(kd)} + x[{off+3}]*(1.0-{_cpp(kd)});")
            lines.append(f"{indent}    double psi_q_pp_x = -x[{off+5}]*{_cpp(kq)} + x[{off+4}]*(1.0-{_cpp(kq)});")
            lines.append(f"{indent}    double rhs_d_x = vd_dq_{comp.name} + psi_q_pp_x;")
            lines.append(f"{indent}    double rhs_q_x = vq_dq_{comp.name} - psi_d_pp_x;")
            lines.append(f"{indent}    outputs_{comp.name}[{id_dq_idx}] = ({_cpp(-ra)}*rhs_d_x - {_cpp(xq_pp)}*rhs_q_x) / {det:.6f};")
            lines.append(f"{indent}    outputs_{comp.name}[{iq_dq_idx}] = ({_cpp(xd_pp)}*rhs_d_x + {_cpp(-ra)}*rhs_q_x) / {det:.6f};")
            lines.append(f"{indent}}}")
            return lines

        # Iterative Solver Loop
        code.append(f"\n    // --- Iterative Network Solution (Algebraic Loop) ---")
        code.append(f"    const double ALG_RELAX = 0.30; // under-relaxation for nonlinear current injections")
        code.append(f"    for(int iter=0; iter<200; ++iter) {{")
        
        code.append(f"        // Zero Accumulators then add constant slack-bus Norton bias")
        code.append(f"        for(int i=0; i<N_BUS; ++i) {{ Id_inj[i]=I_slack_d[i]; Iq_inj[i]=I_slack_q[i]; }}")

        # 1. Compute Outputs & Gather Injections
        # Generator Norton currents are identified by 'Id'/'Iq' output ports.
        code.append("\n        // --- 1. Compute Outputs & Gather Injections ---")
        for comp in self.components:
             in_def = self._generate_input_gathering(comp)
             code.append(f"        // {comp.name} Outputs")
             code.append(f"        {{")
             code.append(in_def)
             code.append(f"            step_{comp.name}_out(&x[{self.state_offsets[comp.name]}], inputs_{comp.name}, outputs_{comp.name}, t);")

             bus_id = comp.params.get('bus')
             if bus_id is None and comp.component_role == 'renewable_controller':
                 gen_name = comp.get_associated_generator(self.comp_map)
                 gen_comp = self.comp_map.get(gen_name) if gen_name else None
                 if gen_comp is not None:
                     bus_id = gen_comp.params.get('bus')

             if bus_id in self.ybus_builder.bus_map:
                 full_idx = self.ybus_builder.bus_map[bus_id]
                 if full_idx in self.active_bus_map:
                     red_idx = self.active_bus_map[full_idx]
                     out_names = _port_names(comp, 'out')
                     if 'Id' in out_names and 'Iq' in out_names:
                         id_idx = out_names.index('Id')
                         iq_idx = out_names.index('Iq')
                         code.append(f"            Id_inj[{red_idx}] += outputs_{comp.name}[{id_idx}];")
                         code.append(f"            Iq_inj[{red_idx}] += outputs_{comp.name}[{iq_idx}];")
                     elif 'It_Re' in out_names and 'It_Im' in out_names:
                         re_idx = out_names.index('It_Re')
                         im_idx = out_names.index('It_Im')
                         code.append(f"            Id_inj[{red_idx}] += outputs_{comp.name}[{re_idx}];")
                         code.append(f"            Iq_inj[{red_idx}] += outputs_{comp.name}[{im_idx}];")
             code.append(f"        }}")

        # 1b. Frequency-dependent load current injection (kpf/kqf)
        # ΔP = kpf * Δω * G_load,  ΔQ = kqf * Δω * B_load
        # ΔI_d = ΔP*Vd - ΔQ*Vq,   ΔI_q = ΔP*Vq + ΔQ*Vd
        # where Δω = ω_COI - 1.0
        if has_freq_dep_load:
            # Compute COI omega from generator speeds (use current state)
            gen_comps_coi = [(comp, self.state_offsets[comp.name])
                            for comp in self.components
                            if comp.component_role == 'generator' and 'omega' in comp.state_schema]
            if gen_comps_coi:
                total_2H_parts = []
                weighted_parts = []
                for comp, off in gen_comps_coi:
                    H = float(comp.params.get('H', 3.0))
                    total_2H_parts.append(f"{2.0 * H:.6f}")
                    weighted_parts.append(f"{2.0 * H:.6f} * x[{off + 1}]")
                code.append(f"\n        // --- 1b. Frequency-dependent load correction ---")
                code.append(f"        {{")
                code.append(f"            double coi_2H = {' + '.join(total_2H_parts)};")
                code.append(f"            double coi_w = ({' + '.join(weighted_parts)}) / coi_2H;")
                code.append(f"            double dw = coi_w - 1.0;")
                code.append(f"            for(int i=0; i<N_BUS; ++i) {{")
                code.append(f"                double dP = LOAD_KPF[i] * dw * LOAD_G[i];")
                code.append(f"                double dQ = LOAD_KQF[i] * dw * LOAD_B[i];")
                code.append(f"                // Negative sign: load consumes more → less net injection")
                code.append(f"                Id_inj[i] -= dP * Vd_net[i] - dQ * Vq_net[i];")
                code.append(f"                Iq_inj[i] -= dP * Vq_net[i] + dQ * Vd_net[i];")
                code.append(f"            }}")
                code.append(f"        }}")

        # 2. Network Solve
        code.append("\n        // --- 2. Solve Network (V = Z * I) ---")
        code.append(f"        double max_err = 0.0;")
        code.append(f"        for(int i=0; i<N_BUS; ++i) {{")
        code.append(f"            double Vd_raw = 0.0; double Vq_raw = 0.0;")
        code.append(f"            for(int j=0; j<N_BUS; ++j) {{")
        code.append(f"                double R = Z_real[i*N_BUS + j];")
        code.append(f"                double X = Z_imag[i*N_BUS + j];")
        code.append(f"                double Id = Id_inj[j];")
        code.append(f"                double Iq = Iq_inj[j];")
        code.append(f"                Vd_raw += R*Id - X*Iq;")
        code.append(f"                Vq_raw += R*Iq + X*Id;")
        code.append(f"            }}")
        code.append(f"            double Vd_new = (1.0 - ALG_RELAX) * Vd_net[i] + ALG_RELAX * Vd_raw;")
        code.append(f"            double Vq_new = (1.0 - ALG_RELAX) * Vq_net[i] + ALG_RELAX * Vq_raw;")
        code.append(f"            double err = fabs(Vd_new - Vd_net[i]) + fabs(Vq_new - Vq_net[i]);")
        code.append(f"            if(err > max_err) max_err = err;")
        code.append(f"            Vd_net[i] = Vd_new;")
        code.append(f"            Vq_net[i] = Vq_new;")
        code.append(f"            Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);")
        code.append(f"        }}")
        # 2b-inside: after each V update, refresh dq voltages and actual id/iq
        # so the NEXT iteration's exciter output step gets correct inputs.
        if gen_comps_for_dq:
            code.append(f"        // 2b: refresh dq voltages/currents from updated V")
            for comp in gen_comps_for_dq:
                code.extend(_emit_dq_update(comp, indent="        "))
        code.append(f"        if(max_err < 1e-6) break;")
        code.append(f"    }}") # End Iter Loop

        # Observer bus voltage recovery: V_obs = M[topo] @ V_active
        if n_obs > 0:
            code.append(f"\n    // --- Observer Bus Voltage Recovery ---")
            code.append(f"    const double* obs_mr = OBS_M_real[topo_idx];")
            code.append(f"    const double* obs_mi = OBS_M_imag[topo_idx];")
            code.append(f"    for(int i = 0; i < N_OBS; ++i) {{")
            code.append(f"        double vr = 0.0, vi = 0.0;")
            code.append(f"        for(int j = 0; j < N_BUS; ++j) {{")
            code.append(f"            double mr = obs_mr[i*N_BUS + j];")
            code.append(f"            double mi = obs_mi[i*N_BUS + j];")
            code.append(f"            vr += mr * Vd_net[j] - mi * Vq_net[j];")
            code.append(f"            vi += mr * Vq_net[j] + mi * Vd_net[j];")
            code.append(f"        }}")
            code.append(f"        Vd_obs[i] = vr; Vq_obs[i] = vi;")
            code.append(f"        Vterm_obs[i] = sqrt(vr*vr + vi*vi);")
            code.append(f"    }}")

        # 3. Compute Dynamics (using Solved V)
        code.append("\n    // --- 3. Compute Dynamics (dxdt) using Solved V ---")
        for comp in self.components:
             in_def = self._generate_input_gathering(comp)
             code.append(f"    // {comp.name} Dynamics")
             code.append(f"    {{")
             code.append(in_def)
             code.append(f"        step_{comp.name}(&x[{self.state_offsets[comp.name]}], &dxdt[{self.state_offsets[comp.name]}], inputs_{comp.name}, outputs_{comp.name}, t);")
             code.append(f"    }}")

        # 4. COI (Center of Inertia) reference frame correction.
        # Prevents net system angle from numerical drift.
        # omega_COI = sum(2H_i * omega_i) / sum(2H_i)
        # Each generator's d_delta/dt is shifted by omega_b * (omega_COI - 1.0)
        # so that the COI angle remains bounded.
        gen_comps = [(comp, self.state_offsets[comp.name])
                     for comp in self.components if comp.component_role == 'generator' and 'omega' in comp.state_schema]
        if len(gen_comps) > 1:
            code.append("\n    // --- 4. COI Reference Frame Correction ---")
            total_2H_parts = []
            weighted_omega_parts = []
            delta_dot_refs = []  # (offset_of_delta_dot, H)
            for comp, off in gen_comps:
                H = float(comp.params.get('H', 3.0))
                # state layout: [0]=delta, [1]=omega
                delta_idx = off + 0
                omega_idx = off + 1
                total_2H_parts.append(f"{2.0 * H:.6f}")
                weighted_omega_parts.append(f"{2.0 * H:.6f} * x[{omega_idx}]")
                delta_dot_refs.append((delta_idx, H))
            total_2H_str = " + ".join(total_2H_parts)
            weighted_omega_str = " + ".join(weighted_omega_parts)
            
            # Get omega_b from the first generator (assume uniform across system)
            omega_b_val = gen_comps[0][0].params.get('omega_b', '2.0 * M_PI * 60.0')
            
            code.append(f"    double coi_total_2H = {total_2H_str};")
            code.append(f"    double coi_omega = ({weighted_omega_str}) / coi_total_2H;")
            code.append(f"    double omega_b_sys = {omega_b_val};")
            for delta_idx, _ in delta_dot_refs:
                code.append(f"    dxdt[{delta_idx}] -= omega_b_sys * (coi_omega - 1.0);")
            code.append(f"    dxdt[{self.delta_coi_idx}] = omega_b_sys * (coi_omega - 1.0);")
        else:
            code.append(f"    dxdt[{self.delta_coi_idx}] = 0.0;")

        code.append("}")
        return "\n".join(code)

    def _generate_input_gathering(self, comp: PowerComponent) -> str:
        lines = []
        n_in = len(comp.port_schema['in'])
        if n_in == 0:
            return f"        double* inputs_{comp.name} = nullptr;"
        for i, (p_name, _, _) in enumerate(comp.port_schema['in']):
            key = (comp.name, p_name)
            if key in self.wiring_map:
                val = self.wiring_map[key]
                lines.append(f"        inputs_{comp.name}[{i}] = {val}; // {p_name}")
            else:
                # Default 0.0 for unwired ports
                lines.append(f"        inputs_{comp.name}[{i}] = 0.0; // UNWIRED {p_name}")
        return "\n".join(lines)

    def _generate_instance_step(self, comp: PowerComponent) -> str:
        lines = []
        func_name = f"step_{comp.name}"
        lines.append(f"void {func_name}(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {{")
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
        lines.append(f"void {func_name}(const double* x, const double* inputs, double* outputs, double t) {{")
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
