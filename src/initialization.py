import numpy as np
import math
from src.compiler import SystemCompiler
from src.powerflow import PowerFlow


class Initializer:
    """Generic power-system state initializer.

    Uses the PowerComponent initialization contract (component_role,
    init_from_phasor, init_from_targets, compute_norton_current, …)
    so that no model-specific logic or isinstance checks appear here.
    Adding or swapping a component only requires updating that component's
    class — this file never needs to change.

    Wiring is read from ``compiler.graph`` (a SystemGraph).  The
    exciter/governor → generator linkage is discovered by inspecting the
    graph's wire list rather than relying on the ``syn`` parameter key.
    """

    def __init__(self, compiler: SystemCompiler):
        self.compiler = compiler
        # Convenience alias: the graph is the authoritative model
        self.graph = compiler.graph
        self.pf = PowerFlow(compiler.data)

    # ------------------------------------------------------------------
    # Graph-aware link helpers
    # ------------------------------------------------------------------

    def _get_generator_for_comp(self, comp_name: str):
        """Return the generator PowerComponent linked to *comp_name*, or None.

        Looks for a wire that connects the named component's output to a
        generator's input (exciter → gen.Efd) or a wire to the named
        component's input from a generator (gen.omega → gov).
        Falls back to the ``syn`` param for backward compatibility.
        """
        comp = self.compiler.comp_map.get(comp_name)
        if comp is None:
            return None
        role = comp.component_role

        if role == "exciter":
            # Exciter feeds its Efd output to a generator input
            gen_name = self.graph.get_generator_for_exciter(comp_name)
            if gen_name is None:
                # Fall back to 'syn' param
                gen_name = comp.params.get("syn")
            return self.compiler.comp_map.get(gen_name)

        if role == "governor":
            # Governor reads a generator's omega and outputs Tm to it
            gen_name = None
            for wire in self.graph.wires:
                if wire.dst_component() == comp_name and wire.dst_port() == "omega":
                    if wire.src_kind() == "comp":
                        src_comp = self.compiler.comp_map.get(wire.src_component())
                        if src_comp and src_comp.component_role == "generator":
                            gen_name = wire.src_component()
                            break
            if gen_name is None:
                gen_name = comp.params.get("syn")
            return self.compiler.comp_map.get(gen_name)

        if role == "pss":
            # PSS reads generator signals; find via avr→syn chain or syn param
            avr_name = comp.params.get("avr")
            avr_comp = self.compiler.comp_map.get(avr_name)
            if avr_comp:
                gen_name = self.graph.get_generator_for_exciter(avr_name)
                if gen_name is None:
                    gen_name = avr_comp.params.get("syn")
                return self.compiler.comp_map.get(gen_name)
            gen_name = comp.params.get("syn")
            return self.compiler.comp_map.get(gen_name)

        return None

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------


    def _get_converged_network_voltages(self, x0: np.ndarray) -> np.ndarray:
        """Return the converged network voltage matching the C++ algebraic loop."""
        V_net = self._compute_network_voltages(x0)
        for _viter in range(50):
            V_new = self._compute_network_voltages(x0, V_net_prev=V_net)
            if len(V_new) > 0 and len(V_net) > 0:
                dV = float(np.max(np.abs(V_new - V_net)))
            else:
                dV = 0.0
            V_net = V_new
            if dV < 1e-6:
                break
        return V_net

    def _compute_network_voltages(self, x0: np.ndarray,
                                   V_net_prev: np.ndarray = None) -> np.ndarray:
        """Single Z-bus solve: V = Z * (I_gen_norton + I_slack_norton).

        I_gen_norton comes from each generator's compute_norton_current().
        For RI-frame generators (e.g. DFIG) *V_net_prev* — the Z-bus result
        from the previous Kron iteration — is passed through so the Norton
        formula I = I_stator + Y_N*V uses the actual network voltage, matching
        the C++ algebraic loop.  On the first call V_net_prev is None and those
        components fall back to an equilibrium-voltage estimate automatically.
        I_slack_norton is the pre-computed contribution of all slack (infinite)
        buses folded through the Kron reduction; it is stored in
        ybus_builder._last_kron['slack_norton_I'] after finalize_network().

        Returns complex voltage array indexed by active-bus position.
        """
        compiler = self.compiler
        Id_inj = np.zeros(compiler.n_active, dtype=complex)

        # Generator Norton currents
        for comp in compiler.components:
            bus_id = comp.params.get('bus')
            if bus_id is None:
                continue
            offset = compiler.state_offsets[comp.name]
            x_slice = x0[offset:offset + len(comp.state_schema)]
            full_idx = compiler.ybus_builder.bus_map.get(bus_id)
            if full_idx is None:
                continue
            red_idx = compiler.active_bus_map.get(full_idx)
            # For RI-frame Norton generators (e.g. DFIG), pass the Z-bus voltage
            # from the previous iteration so Python matches the C++ algebraic loop:
            #   I_norton = I_stator + Y_N * V_actual  (not V_eq from dφ/dt=0).
            # On the first iteration V_net_prev is None — the component falls back
            # to the equilibrium-voltage formula automatically.
            if (getattr(comp, 'uses_ri_frame', False)
                    and getattr(comp, 'contributes_norton_admittance', False)
                    and V_net_prev is not None
                    and red_idx is not None):
                I = comp.compute_norton_current(x_slice, V_net_prev[red_idx])
            else:
                I = comp.compute_norton_current(x_slice)
            if abs(I) == 0.0:
                continue
            if red_idx is not None:
                Id_inj[red_idx] += I

        # Slack (infinite) bus Norton current injection (folded through Kron)
        kron = getattr(compiler.ybus_builder, '_last_kron', None)
        if kron is not None:
            slack_I = kron.get('slack_norton_I')
            if slack_I is not None and len(slack_I) == compiler.n_active:
                Id_inj += slack_I

        return compiler.z_bus @ Id_inj

    def _gen_bus_voltage(self, comp, V_net: np.ndarray):
        """Return the complex network voltage at comp's bus, or None."""
        compiler = self.compiler
        bus_id = comp.params.get('bus')
        if bus_id is None:
            return None
        full_idx = compiler.ybus_builder.bus_map.get(bus_id)
        if full_idx is None:
            return None
        red_idx = compiler.active_bus_map.get(full_idx)
        if red_idx is None:
            return None
        return V_net[red_idx]

    @staticmethod
    def _park_transform(V_bus: complex, delta: float):
        """RI-frame network voltage → dq-frame (vd, vq)."""
        sin_d = math.sin(delta); cos_d = math.cos(delta)
        vd = V_bus.real * sin_d - V_bus.imag * cos_d
        vq = V_bus.real * cos_d + V_bus.imag * sin_d
        return vd, vq

    # ------------------------------------------------------------------
    # Main initialization pass
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Run power flow then initialize all component states.

        Uses the SystemGraph wire list to discover generator↔exciter and
        generator↔governor relationships generically, without relying on
        hard-coded 'syn' parameter checks.

        Raises structured FrameworkError subclasses on configuration problems
        so the caller can surface actionable messages to the user.
        """
        # Validate graph wiring before spending time on power flow
        try:
            self.graph.validate()
        except Exception as e:
            # Re-raise with a clear prefix
            raise type(e)(f"[Initializer] Graph validation failed: {e}") from e

        skip_pf = self.compiler.data.get("config", {}).get("skip_pf_solve", False)
        if skip_pf:
            print("Bypassing NR Power Flow — using Bus v0/a0 from JSON...")
            self.pf.load_bus_overrides()
        else:
            print("Running Power Flow...")
            self.pf.solve()
        for i in range(self.pf.n_bus):
            print(f"  [PF] Bus {self.pf.buses[i]} V={self.pf.V[i]:.5f} ang={self.pf.theta[i]*180/np.pi:.3f}")
        V = self.pf.V
        theta = self.pf.theta

        x0 = np.zeros(self.compiler.total_states)
        # targets: {gen_name -> {'Efd', 'Tm', 'Vt', 'vd', 'vq', 'id', 'iq',
        #                        'vd_ri', 'vq_ri', 'exciter_comp', 'exciter_offset'}}
        self.targets = {}

        # Pass 1: initialize generators (they produce the targets dict)
        # Group generators by bus to correctly split power if multiple exist
        gens_by_bus = {}
        for comp in self.compiler.components:
            if comp.component_role == "generator":
                bus_id = comp.params["bus"]
                gens_by_bus.setdefault(bus_id, []).append(comp)

        for bus_id, gens in gens_by_bus.items():
            bus_idx  = self.pf.bus_map[bus_id]
            Vt       = V[bus_idx]
            ang      = theta[bus_idx]
            V_phasor = Vt * np.exp(1j * ang)

            S_calc_net = self.pf.calculate_power()[bus_idx]
            P_load = 0.0; Q_load = 0.0
            for pq in self.graph.pq_loads:
                if pq.bus == bus_id:
                    P_load += pq.p0
                    Q_load += pq.q0

            S_total_gen = (S_calc_net.real + P_load) + 1j * (S_calc_net.imag + Q_load)

            # Apply p_override / q_override: if a generator has these in its params,
            # force the exact PF values instead of the internal PF result.
            if len(gens) == 1:
                comp = gens[0]
                p_ov = comp.params.get("p_override")
                q_ov = comp.params.get("q_override")
                P_val = float(p_ov) if p_ov is not None else S_total_gen.real
                Q_val = float(q_ov) if q_ov is not None else S_total_gen.imag
                if p_ov is not None or q_ov is not None:
                    print(f"  [Init] {comp.name} bus {bus_id}: S override "
                          f"({S_total_gen.real:.6f}+j{S_total_gen.imag:.6f}) -> "
                          f"({P_val:.6f}+j{Q_val:.6f})")
                    S_total_gen = P_val + 1j * Q_val
                S_gens = [S_total_gen]
            else:
                # Multiple generators: those with p0/q0 take their specified power,
                # the rest split the remaining power equally.
                S_gens = [0j] * len(gens)
                S_remaining = S_total_gen
                unspecified_idx = []
                
                for i, comp in enumerate(gens):
                    p0 = comp.params.get("p0")
                    q0 = comp.params.get("q0")
                    if p0 is not None:
                        # If q0 is not specified, assume it takes a proportional share of Q,
                        # or just 0 for now. Let's assume 0 if not specified, but usually
                        # renewable generators might only specify p0.
                        # Actually, if q0 is missing, let's just use 0.0.
                        q_val = q0 if q0 is not None else 0.0
                        S_gens[i] = p0 + 1j * q_val
                        S_remaining -= S_gens[i]
                    else:
                        unspecified_idx.append(i)
                
                if unspecified_idx:
                    S_per_unspecified = S_remaining / len(unspecified_idx)
                    for i in unspecified_idx:
                        S_gens[i] = S_per_unspecified
                else:
                    # If all specified, but they don't add up to S_total_gen,
                    # we have a mismatch. We'll just add the remainder to the first one.
                    S_gens[0] += S_remaining

            for comp, S_gen in zip(gens, S_gens):
                offset = self.compiler.state_offsets[comp.name]
                I_phasor = np.conj(S_gen / V_phasor)

                x_init, targets = comp.init_from_phasor(V_phasor, I_phasor)
                x0[offset:offset + len(x_init)] = x_init
                self.targets[comp.name] = targets
                comp.params["Efd0"] = float(targets["Efd"])
                comp.params["Tm0"]  = float(targets["Tm"])

        # Pass 2: initialize exciters using the generator's targets dict
        for comp in self.compiler.components:
            if comp.component_role != "exciter":
                continue
            offset = self.compiler.state_offsets[comp.name]
            gen_comp = self._get_generator_for_comp(comp.name)
            if gen_comp is None:
                continue
            gen_name = gen_comp.name
            if gen_name not in self.targets:
                continue
            x_init = comp.init_from_targets(self.targets[gen_name])
            x0[offset:offset + len(x_init)] = x_init
            self.targets[gen_name]["exciter_name"]   = comp.name
            self.targets[gen_name]["exciter_comp"]   = comp
            self.targets[gen_name]["exciter_offset"] = offset

        # Pass 3: PSS — initialise from steady-state generator signals
        for comp in self.compiler.components:
            if comp.component_role != "pss":
                continue
            offset = self.compiler.state_offsets[comp.name]
            gen_comp = self._get_generator_for_comp(comp.name)
            if gen_comp is None:
                x0[offset:offset + len(comp.state_schema)] = 0.0
                continue
            gen_name = gen_comp.name
            tgt = self.targets.get(gen_name)
            if tgt is None:
                x0[offset:offset + len(comp.state_schema)] = 0.0
                continue
            x_init = comp.init_from_targets(tgt)
            print(f"  [Init] PSS {comp.name} initialized to: {x_init} with Pe={tgt.get('Pe')}, Tm={tgt.get('Tm')}")
            x0[offset:offset + len(x_init)] = x_init

        # Pass 4: governors
        for comp in self.compiler.components:
            if comp.component_role != "governor":
                continue
            offset = self.compiler.state_offsets[comp.name]
            gen_comp = self._get_generator_for_comp(comp.name)
            if gen_comp is None:
                continue
            gen_name = gen_comp.name
            if gen_name not in self.targets:
                continue
            x_init = comp.init_from_targets(self.targets[gen_name])
            x0[offset:offset + len(x_init)] = x_init

        # Pass 5: renewable controllers (initial fill; refine_renewable_controllers corrects later)
        for comp in self.compiler.components:
            if comp.component_role != "renewable_controller":
                continue
            offset = self.compiler.state_offsets[comp.name]

            # Dispatch through component protocol — no class-name checks needed
            gen_name = comp.get_associated_generator(self.compiler.comp_map)
            if gen_name is None or gen_name not in self.targets:
                continue
            x_init = comp.init_from_targets(self.targets[gen_name])
            x0[offset:offset + len(x_init)] = x_init

        # Pass 6: passive components
        for comp in self.compiler.components:
            if comp.component_role != "passive":
                continue
            offset = self.compiler.state_offsets[comp.name]
            if len(comp.state_schema) == 0:
                continue
            
            # Try to find the bus voltage
            bus_id = comp.params.get("bus")
            if bus_id is None:
                # Try to find the bus from connected components
                for wire in self.compiler.graph.wires:
                    if wire.dst_component() == comp.name and wire.src.startswith("BUS_"):
                        # wire.src format: "BUS_<id>.<port>"
                        bus_id = int(wire.src.split(".")[0].split("_", 1)[1])
                        break
            
            if bus_id is not None and bus_id in self.pf.bus_map:
                bus_idx = self.pf.bus_map[bus_id]
                Vt = self.pf.V[bus_idx]
                ang = self.pf.theta[bus_idx]
                V_phasor = Vt * np.exp(1j * ang)
                targets = {
                    'Vd': V_phasor.real,
                    'Vq': V_phasor.imag,
                    'Vterm': abs(V_phasor)
                }
                x_init = comp.init_from_targets(targets)
                x0[offset:offset + len(x_init)] = x_init
            else:
                # No bus found — still call init_from_targets with empty dict
                # so components like DCLink can self-initialise from their own params.
                x_init = comp.init_from_targets({})
                x0[offset:offset + len(x_init)] = x_init

        return x0

    # ------------------------------------------------------------------
    # Refinement passes
    # ------------------------------------------------------------------

    def refine_kron_equilibrium(self, x0: np.ndarray,
                                n_iter: int = 20,
                                alpha: float = 0.5) -> np.ndarray:
        """Correct component states to the Kron-reduced network operating point.

        After finalize_network(), the Kron-reduced bus voltages differ from the
        full power-flow terminal voltages used by init_from_phasor().  This pass
        calls comp.refine_at_kron_voltage(x_slice, vd, vq) on every component,
        applying under-relaxed updates until convergence.

        The method is fully type-agnostic: each component decides what (if any)
        states to correct; the base class default is a no-op.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        x0_backup = x0.copy()          # snapshot before any iteration
        prev_max_change = None          # for divergence detection
        diverge_count = 0               # consecutive diverging iterations

        V_net_prev = None       # convergence tracking
        V_norton_seed = None    # Z-bus result fed back into RI-frame Norton each iter
        for iteration in range(n_iter):
            V_net      = self._compute_network_voltages(x0, V_net_prev=V_norton_seed)
            V_norton_seed = V_net  # next iteration uses this V for RI-frame Norton
            max_change = 0.0
            
            if V_net_prev is not None and len(V_net) > 0:
                max_change = max(max_change, float(np.max(np.abs(V_net - V_net_prev))))
            V_net_prev = V_net.copy()

            for comp in compiler.components:
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                if n == 0:
                    continue
                bus_id = comp.params.get('bus')
                if bus_id is None:
                    # Try graph-based linkage first, then fall back to 'syn'
                    gen_comp = self._get_generator_for_comp(comp.name)
                    if gen_comp is None:
                        syn_name = comp.params.get('syn')
                        gen_comp = compiler.comp_map.get(syn_name) if syn_name else None
                    bus_id = gen_comp.params.get('bus') if gen_comp else None
                if bus_id is None:
                    continue
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue
                
                if 'vd_ri' in comp.params:
                    comp.params['vd_ri'] = V_bus.real
                if 'vq_ri' in comp.params:
                    comp.params['vq_ri'] = V_bus.imag
                    
                if comp.uses_ri_frame:
                    vd, vq = V_bus.real, V_bus.imag
                else:
                    vd, vq = self._park_transform(V_bus, float(x0[off]))
                x_slice   = x0[off:off + n].copy()
                x_new     = comp.refine_at_kron_voltage(x_slice, vd, vq)
                d_states  = alpha * (x_new - x_slice)
                max_change = max(max_change, float(np.max(np.abs(d_states))))
                x0[off:off + n] += d_states

            # Divergence detection: revert if iteration is making things worse
            if prev_max_change is not None and max_change > 1.5 * prev_max_change:
                diverge_count += 1
            else:
                diverge_count = 0
            if diverge_count >= 3 or max_change > 1e10:
                print(f"  [Init] Kron equilibrium DIVERGING at iter {iteration + 1} "
                      f"(max |dx|={max_change:.2e}), reverting to pre-Kron state.")
                return x0_backup
            prev_max_change = max_change

            if iteration > 0 and max_change < 1e-6:
                print(f"  [Init] Kron equilibrium converged after {iteration + 1} iter(s) "
                      f"(max |dx|={max_change:.2e}).")
                break
        else:
            print(f"  [Init] Kron equilibrium: {n_iter} iters, max |dx|={max_change:.2e}.")

        return x0

    def refine_delta_dispatch(self, x0: np.ndarray,
                              n_outer: int = 8,
                              kron_iters: int = 15,
                              alpha_delta: float = 0.3) -> np.ndarray:
        """After Kron equilibrium settles q-axis, adjust generator delta to
        maintain the original PF active power dispatch (Tm0).

        Uses a nested iteration: outer loop adjusts delta via Newton steps,
        inner Kron equilibrium re-settles q-axis at the new angle.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        # Identify generator components with Tm0 targets
        gen_comps = []
        for comp in compiler.components:
            if (hasattr(comp, 'compute_stator_currents')
                    and 'Tm0' in comp.params
                    and not comp.uses_ri_frame):
                gen_comps.append(comp)
        if not gen_comps:
            return x0

        for outer in range(n_outer):
            V_net = self._compute_network_voltages(x0)
            max_dp = 0.0

            for comp in gen_comps:
                off = compiler.state_offsets[comp.name]
                n   = len(comp.state_schema)
                x_slice = x0[off:off+n]
                Tm0 = comp.params['Tm0']
                delta = float(x_slice[0])

                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue

                vd, vq = self._park_transform(V_bus, delta)
                id_a, iq_a = comp.compute_stator_currents(x_slice, vd, vq)
                te = vd * id_a + vq * iq_a
                dp = Tm0 - te
                max_dp = max(max_dp, abs(dp))

                if abs(dp) < 1e-5:
                    continue

                # Newton step: numerical dTe/ddelta
                eps = 1e-6
                vd2, vq2 = self._park_transform(V_bus, delta + eps)
                x_pert = comp.refine_q_axis(x_slice.copy(), vd2, vq2)
                id2, iq2 = comp.compute_stator_currents(x_pert, vd2, vq2)
                te2 = vd2 * id2 + vq2 * iq2
                dTe_dd = (te2 - te) / eps

                if abs(dTe_dd) > 1e-8:
                    d_delta = alpha_delta * dp / dTe_dd
                    d_delta = max(-0.05, min(0.05, d_delta))
                    x0[off] += d_delta
                    # Re-settle q-axis at new delta
                    vd3, vq3 = self._park_transform(V_bus, float(x0[off]))
                    x_new = comp.refine_q_axis(x0[off:off+n].copy(), vd3, vq3)
                    x0[off+4] = x_new[4]  # psi_q
                    x0[off+5] = x_new[5]  # Ed_p

            if max_dp < 1e-4:
                print(f"  [Init] Delta dispatch converged after {outer+1} "
                      f"outer iter(s) (max |dP|={max_dp:.2e}).")
                break

            # Re-settle full Kron equilibrium after delta changes
            x0 = self.refine_kron_equilibrium(x0, n_iter=kron_iters, alpha=0.5)

        else:
            print(f"  [Init] Delta dispatch: {n_outer} outer iters, max |dP|={max_dp:.2e}.")

        return x0

    def refine_rectifier_VB(self, x0: np.ndarray) -> np.ndarray:
        """Correct the VB state of every ESST3A exciter so that dVB/dt = 0 at t=0.

        At equilibrium:  VB = VE * FEX(IN)   where IN = KC * (VB*VM) / VE.
        The init_from_targets() back-solve already does this, but it uses PF-based
        stator currents (id/iq from the full-network operating point) to compute VE.
        The C++ simulation uses Kron-reduced stator currents which differ slightly.
        This pass recomputes VB directly from the Kron-network stator currents so
        the C++ diagnostics show dVB ≈ 0 at t = 0.

        Only ESST3A exciters have a VB state (index 4).  Other exciter types are
        skipped gracefully.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        V_net = self._get_converged_network_voltages(x0)

        for gen_comp in compiler.components:
            if gen_comp.component_role != 'generator':
                continue
            tgt = self.targets.get(gen_comp.name, {})
            exc_comp = tgt.get('exciter_comp')
            if exc_comp is None:
                continue
            # Only ESST3A has a 'VB' state
            if 'VB' not in exc_comp.state_schema:
                continue

            exc_off = compiler.state_offsets[exc_comp.name]
            gen_off = compiler.state_offsets[gen_comp.name]
            gn      = len(gen_comp.state_schema)

            V_bus = self._gen_bus_voltage(gen_comp, V_net)
            if V_bus is None:
                continue

            delta = float(x0[gen_off])
            sin_d = math.sin(delta); cos_d = math.cos(delta)
            vd_dq = float(V_bus.real) * sin_d - float(V_bus.imag) * cos_d
            vq_dq = float(V_bus.real) * cos_d + float(V_bus.imag) * sin_d
            id_kron, iq_kron = gen_comp.compute_stator_currents(
                x0[gen_off:gen_off + gn], vd_dq, vq_dq)

            p = exc_comp.params
            KP     = float(p.get('KP', 3.67));  KI    = float(p.get('KI', 0.435))
            XL     = float(p.get('XL', 0.0098)); THETAP = float(p.get('THETAP', 0.0))
            KC     = float(p.get('KC', 0.01));  VBMAX = float(p.get('VBMAX', 5.48))
            theta_rad = THETAP * math.pi / 180.0
            KPC_r = KP * math.cos(theta_rad); KPC_i = KP * math.sin(theta_rad)
            jc_r  = -(KPC_i * XL); jc_i = KI + KPC_r * XL
            # VE — same expression as C++ get_cpp_step_code
            v_r = KPC_r*vd_dq - KPC_i*vq_dq + jc_r*id_kron - jc_i*iq_kron
            v_i = KPC_r*vq_dq + KPC_i*vd_dq + jc_r*iq_kron + jc_i*id_kron
            VE  = math.sqrt(v_r**2 + v_i**2)
            if VE < 1e-6: VE = 1e-6

            vb_idx = exc_comp.state_schema.index('VB')
            vm_idx = exc_comp.state_schema.index('VM')
            VB_cur = float(x0[exc_off + vb_idx])
            VM_cur = float(x0[exc_off + vm_idx])

            # Self-consistent solve: VB_eq so that FEX(KC*VB*VM/VE)*VE = VB,
            # where VM stays fixed at current value (it's handled by dVM dynamics).
            # A few Newton iterations converge quickly (IN << 1 for typical operation).
            VB_eq = VB_cur
            for _ in range(15):
                XadIfd = max(VB_eq * VM_cur, 0.0)
                IN = KC * XadIfd / VE
                if   IN <= 0.0:    FEX = 1.0
                elif IN <= 0.433:  FEX = 1.0 - 0.577 * IN
                elif IN <= 0.75:   FEX = math.sqrt(max(0.0, 0.75 - IN**2))
                elif IN <= 1.0:    FEX = 1.732 * (1.0 - IN)
                else:              FEX = 0.0
                VB_new = min(max(VE * FEX, 0.0), VBMAX)
                if abs(VB_new - VB_eq) < 1e-12:
                    break
                VB_eq = VB_new

            if abs(VB_eq - VB_cur) > 1e-8:
                x0[exc_off + vb_idx] = VB_eq

        return x0

    def refine_exciter_voltages(self, x0: np.ndarray,
                                full_reinit: bool = False) -> np.ndarray:
        """Refine exciter states using the actual Kron-network voltage at t=0.

        full_reinit=False: lightweight — only update Vm and shift Vref.
        full_reinit=True:  full back-solve of all exciter states at the new
                           network operating point (use after refine_clamped_d_axis).
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        V_net = self._get_converged_network_voltages(x0)

        # Collect per-generator network data (Norton currents and RI voltages)
        gen_data = {}
        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            off = compiler.state_offsets[comp.name]
            n   = len(comp.state_schema)
            V_bus = self._gen_bus_voltage(comp, V_net)
            if V_bus is None:
                continue
            Vd_ri   = float(V_bus.real)
            Vq_ri   = float(V_bus.imag)
            Vterm   = float(abs(V_bus))
            delta   = float(x0[off])
            sin_d   = math.sin(delta); cos_d = math.cos(delta)
            # Park-transform RI → dq frame
            vd_dq = Vd_ri * sin_d - Vq_ri * cos_d
            vq_dq = Vd_ri * cos_d + Vq_ri * sin_d
            # Actual terminal dq currents from stator equations
            id_act, iq_act = comp.compute_stator_currents(x0[off:off + n], vd_dq, vq_dq)
            gen_data[comp.name] = {
                'Vterm': Vterm, 'Vd_ri': Vd_ri, 'Vq_ri': Vq_ri,
                'vd_dq': vd_dq, 'vq_dq': vq_dq,
                'id_act': id_act, 'iq_act': iq_act,
            }

        for gen_name, tgt in self.targets.items():
            if 'exciter_offset' not in tgt:
                continue
            gd = gen_data.get(gen_name)
            if not gd:
                continue
            exc_comp   = tgt['exciter_comp']
            exc_offset = tgt['exciter_offset']
            Vterm_actual = gd['Vterm']

            if full_reinit:
                Efd_eff = float(exc_comp.params.get('Efd_eff', tgt.get('Efd', 0.0)))
                updated = {
                    **tgt,
                    'Efd': Efd_eff,
                    'Vt':  Vterm_actual,
                    # dq-frame voltages and actual terminal currents (C++ uses dq)
                    'vd':     gd['vd_dq'],
                    'vq':     gd['vq_dq'],
                    'vd_ri':  gd['Vd_ri'],
                    'vq_ri':  gd['Vq_ri'],
                    'id':     gd['id_act'],
                    'iq':     gd['iq_act'],
                }
                x_exc = exc_comp.init_from_targets(updated)
                x0[exc_offset:exc_offset + len(x_exc)] = x_exc
            else:
                # Only update Vm state for classical exciters that store a
                # voltage-transducer measurement at state index 0 (e.g. IEEEX1,
                # ESST3A).  Algebraic / PH exciters (e.g. IDA-PBC) signal this
                # via _has_voltage_transducer = False — skip the Vm write so
                # their integral state xi is not corrupted.
                has_vm = getattr(exc_comp, '_has_voltage_transducer', True)
                if has_vm:
                    old_Vm = float(x0[exc_offset])
                    x0[exc_offset] = Vterm_actual
                    if 'Vref' in exc_comp.params:
                        old_error = float(exc_comp.params['Vref']) - old_Vm
                        exc_comp.params['Vref'] = Vterm_actual + old_error
                else:
                    # Algebraic exciter (IDA-PBC etc.): just sync Vref to
                    # actual terminal voltage so the operating-point offset
                    # is correct after Kron reduction.
                    if 'Vref' in exc_comp.params:
                        exc_comp.params['Vref'] = Vterm_actual

        return x0

    def refine_clamped_d_axis(self, x0: np.ndarray,
                               n_iter: int = 20) -> np.ndarray:
        """Iterative d-axis update for generators whose exciter clamps Efd.

        Uses comp.refine_d_axis(x_slice, vd, vq, Efd_eff, clamped=True)
        per generator; applies damped updates until convergence.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        clamped = []
        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            tgt = self.targets.get(comp.name, {})
            exc_comp = tgt.get('exciter_comp')
            if exc_comp is None:
                continue
            Efd_req = comp.params.get('Efd0', 0.0)
            Efd_eff = float(exc_comp.params.get('Efd_eff', Efd_req))
            if abs(Efd_eff - Efd_req) < 0.01:
                continue
            clamped.append((comp, float(Efd_eff), float(Efd_req)))

        if not clamped:
            return x0

        alpha_d = 0.4
        for iteration in range(n_iter):
            V_net = self._get_converged_network_voltages(x0)
            max_change = 0.0
            for comp, Efd_eff, Efd_req in clamped:
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue
                vd, vq  = self._park_transform(V_bus, float(x0[off]))
                x_slice = x0[off:off + n].copy()
                x_new   = comp.refine_d_axis(x_slice, vd, vq, Efd_eff, clamped=True)
                dEq_p   = alpha_d * (x_new[2] - x0[off + 2])
                dpsi_d  = alpha_d * (x_new[3] - x0[off + 3])
                max_change = max(max_change, abs(dEq_p), abs(dpsi_d))
                x0[off + 2] += dEq_p
                x0[off + 3] += dpsi_d

            if max_change < 1e-5:
                off0 = compiler.state_offsets[clamped[0][0].name]
                print(f"  [Init] {clamped[0][0].name}: Efd clamped "
                      f"{clamped[0][2]:.3f}->{clamped[0][1]:.3f}. "
                      f"D-axis converged after {iteration+1} iters: "
                      f"Eq_p={x0[off0+2]:.3f}, psi_d={x0[off0+3]:.3f}")
                break
        else:
            off0 = compiler.state_offsets[clamped[0][0].name]
            print(f"  [Init] {clamped[0][0].name}: Efd clamped "
                  f"{clamped[0][2]:.3f}->{clamped[0][1]:.3f}. "
                  f"D-axis ({n_iter} iters, max_change={max_change:.3e}): "
                  f"Eq_p={x0[off0+2]:.3f}, psi_d={x0[off0+3]:.3f}")
        return x0

    def reinit_from_kron_network(self, x0: np.ndarray,
                                  n_iter: int = 30,
                                  alpha: float = 0.5) -> np.ndarray:
        """Fully re-derive ALL generator states from the Kron-network voltage.

        At each outer iteration:
          1. Compute Norton currents → Z-bus → V_kron per bus.
          2. For each 6-state generator: get stator currents at current delta,
             convert to RI phasor, call comp.init_from_phasor(V_kron, I_phasor).
          3. Apply damped update; repeat until convergence.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        for iteration in range(n_iter):
            V_net = self._get_converged_network_voltages(x0)
            max_change = 0.0

            for comp in compiler.components:
                if comp.component_role != 'generator':
                    continue
                # GenCls has only 2 states and E_p is fixed — skip full reinit
                if len(comp.state_schema) < 6:
                    continue
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None or abs(V_bus) < 1e-6:
                    continue

                delta_cur = float(x0[off])
                vd, vq    = self._park_transform(V_bus, delta_cur)
                id_s, iq_s = comp.compute_stator_currents(x0[off:off + n], vd, vq)

                # dq stator currents → RI phasor
                sin_d = math.sin(delta_cur); cos_d = math.cos(delta_cur)
                I_phasor = complex(id_s * sin_d + iq_s * cos_d,
                                   -id_s * cos_d + iq_s * sin_d)

                x_new, _ = comp.init_from_phasor(V_bus, I_phasor)
                d_states  = alpha * (x_new - x0[off:off + n])
                max_change = max(max_change, float(np.max(np.abs(d_states))))
                x0[off:off + n] += d_states

            if max_change < 1e-6:
                print(f"  [Init] Full Kron reinit converged in {iteration+1} iters "
                      f"(max_change={max_change:.2e}).")
                break
        else:
            print(f"  [Init] Full Kron reinit done ({n_iter} iters, "
                  f"max_change={max_change:.2e}).")
        return x0

    def iterative_refinement(self, x0: np.ndarray,
                              n_iter: int = 8,
                              alpha: float = 0.5) -> np.ndarray:
        """Closed-form q-axis refinement: update Ed_p and psi_q.

        At each pass:
          1. Network solve → V_kron per generator bus.
          2. Park-transform → vd, vq.
          3. comp.refine_q_axis(x_slice, vd, vq) → analytical Ed_p, psi_q.
          4. Apply under-relaxation.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        for iteration in range(n_iter):
            V_net = self._get_converged_network_voltages(x0)
            max_dq_change = 0.0

            for comp in compiler.components:
                if comp.component_role != 'generator':
                    continue
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue
                vd, vq   = self._park_transform(V_bus, float(x0[off]))
                x_slice  = x0[off:off + n].copy()
                x_new    = comp.refine_q_axis(x_slice, vd, vq)
                # Only states 4 and 5 (psi_q, Ed_p) are updated by refine_q_axis
                if n >= 6:
                    dEd_p  = alpha * (x_new[5] - x0[off + 5])
                    dpsi_q = alpha * (x_new[4] - x0[off + 4])
                    max_dq_change = max(max_dq_change, abs(dEd_p), abs(dpsi_q))
                    x0[off + 5] += dEd_p
                    x0[off + 4] += dpsi_q

            if max_dq_change < 1e-6:
                print(f"  [Init] q-axis refinement converged after {iteration+1} iteration(s).")
                break
        else:
            print(f"  [Init] q-axis refinement done ({n_iter} iters, "
                  f"max change per iter={max_dq_change:.2e})")
        return x0

    def refine_psi_d(self, x0: np.ndarray,
                     n_iter: int = 20,
                     alpha: float = 0.6) -> np.ndarray:
        """Update psi_d for 6-state generators to satisfy dxdt[3]=0.

        At equilibrium: psi_d = Eq_p - (xd' - xd'') * id_act.
        id_act comes from comp.compute_stator_currents at the Kron network.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        for iteration in range(n_iter):
            V_net = self._get_converged_network_voltages(x0)
            max_dpsi_d = 0.0

            for comp in compiler.components:
                if comp.component_role != 'generator':
                    continue
                if len(comp.state_schema) < 6:
                    continue   # GenCls has no psi_d state
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue
                vd, vq   = self._park_transform(V_bus, float(x0[off]))
                id_act, _ = comp.compute_stator_currents(x0[off:off + n], vd, vq)

                p        = comp.params
                xd_p     = p['xd_prime'];  xd_pp = p['xd_double_prime']
                Eq_p     = x0[off + 2]
                psi_d_tgt = Eq_p - (xd_p - xd_pp) * id_act
                dpsi_d    = alpha * (psi_d_tgt - x0[off + 3])
                max_dpsi_d = max(max_dpsi_d, abs(dpsi_d))
                x0[off + 3] += dpsi_d

            if max_dpsi_d < 1e-6:
                print(f"  [Init] psi_d refined for all GENROU "
                      f"(converged after {iteration+1} iters, "
                      f"max |dpsi_d|={max_dpsi_d:.2e}).")
                return x0

        print(f"  [Init] psi_d refined for all GENROU "
              f"({n_iter} iters, max |dpsi_d|={max_dpsi_d:.4f}).")
        return x0

    def refine_Eq_p(self, x0: np.ndarray,
                    n_iter: int = 20,
                    alpha: float = 0.7) -> np.ndarray:
        """Update Eq_p and psi_d from current exciter Efd output.

        At equilibrium: Eq_p = Efd - (xd - xd') * id_act.
        Efd is obtained via comp.compute_efd_output(exciter_states).
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        for iteration in range(n_iter):
            V_net = self._get_converged_network_voltages(x0)
            max_dEqp = 0.0

            for comp in compiler.components:
                if comp.component_role != 'generator':
                    continue
                if len(comp.state_schema) < 6:
                    continue
                off   = compiler.state_offsets[comp.name]
                n     = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue
                vd, vq   = self._park_transform(V_bus, float(x0[off]))
                id_act, _ = comp.compute_stator_currents(x0[off:off + n], vd, vq)

                # Get Efd from exciter
                tgt      = self.targets.get(comp.name, {})
                exc_comp = tgt.get('exciter_comp')
                if exc_comp is not None:
                    exc_off = compiler.state_offsets[exc_comp.name]
                    exc_n   = len(exc_comp.state_schema)
                    Efd_eff = exc_comp.compute_efd_output(x0[exc_off:exc_off + exc_n])
                else:
                    Efd_eff = float(tgt.get('Efd', 1.0))

                p        = comp.params
                xd       = p['xd']; xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
                Eq_p_new  = Efd_eff - (xd - xd_p) * id_act
                psi_d_new = Eq_p_new - (xd_p - xd_pp) * id_act

                dEq_p  = alpha * (Eq_p_new  - x0[off + 2])
                dpsi_d = alpha * (psi_d_new - x0[off + 3])
                max_dEqp = max(max_dEqp, abs(dEq_p))
                x0[off + 2] += dEq_p
                x0[off + 3] += dpsi_d

            if max_dEqp < 1e-6:
                break
        return x0

    def align_daxis_to_seed_voltages(
        self,
        x0: np.ndarray,
        Vd_seed: list,
        Vq_seed: list,
    ) -> np.ndarray:
        """Set Eq_p and psi_d so that dxdt[2]=dxdt[3]=0 at the C++ seed voltages.

        The C++ network solver starts from (Vd_seed, Vq_seed) and converges
        quickly to a nearby point.  If Eq_p is initialised at the equilibrium
        consistent with *those* seed voltages, the d-axis residual at t=0
        is essentially zero regardless of the tiny Gauss-Seidel correction.
        """
        compiler = self.compiler
        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            if len(comp.state_schema) < 6:
                continue
            off = compiler.state_offsets[comp.name]
            n   = len(comp.state_schema)
            bus_id  = comp.params.get('bus')
            if bus_id is None:
                continue
            bus_idx = compiler.active_bus_map.get(bus_id)
            if bus_idx is None:
                continue
            # Use the exact seed voltages embedded in C++.
            Vd_ri = Vd_seed[bus_idx]
            Vq_ri = Vq_seed[bus_idx]
            delta = float(x0[off])
            vd, vq = self._park_transform(complex(Vd_ri, Vq_ri), delta)

            # Exciter Efd at the current exciter state.
            tgt      = self.targets.get(comp.name, {})
            exc_comp = tgt.get('exciter_comp')
            if exc_comp is not None:
                exc_off = compiler.state_offsets[exc_comp.name]
                exc_n   = len(exc_comp.state_schema)
                Efd_eff = exc_comp.compute_efd_output(x0[exc_off:exc_off + exc_n])
            else:
                Efd_eff = float(tgt.get('Efd', 1.0))

            p       = comp.params
            xd      = p['xd']; xd_p = p['xd_prime']; xd_pp = p['xd_double_prime']
            ra      = p.get('ra', 0.0)
            k_d, _  = comp._kfactors()
            omega   = float(x0[off + 1])

            # Self-consistent analytical d-axis solve.
            # At equilibrium: Eq_p = Efd - (xd-xd')*id, psi_d = Eq_p - (xd'-xd'')*id
            # psi_d_pp = Efd - C_d*id  where C_d = (xd-xd')*k_d + (xd-xd'')*(1-k_d)
            # Stator (ra≈0): id = -(vq - omega*psi_d_pp)/xd_pp
            # Solving: id = (omega*Efd - vq) / (xd_pp + omega*C_d)
            C_d = (xd - xd_p) * k_d + (xd - xd_pp) * (1 - k_d)
            denom = xd_pp + omega * C_d
            if abs(ra) > 1e-10:
                # General case: use current psi_q_pp for rhs_d coupling
                Ed_p = x0[off + 4]; psi_q = x0[off + 5]
                _, k_q = comp._kfactors()
                psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q)
                rhs_d = vd + omega * psi_q_pp
                # Solve 2x2 system with ra coupling (simplified for small ra)
                id_eq = (omega * Efd_eff - vq - ra * rhs_d / xq_pp) / (denom + ra**2 / xq_pp)
            else:
                id_eq = (omega * Efd_eff - vq) / denom

            Eq_p_eq  = Efd_eff - (xd - xd_p) * id_eq
            psi_d_eq = Eq_p_eq - (xd_p - xd_pp) * id_eq

            old_Eq_p = x0[off + 2]
            x0[off + 2] = Eq_p_eq
            x0[off + 3] = psi_d_eq
            print(
                f"  [Init] align_daxis: Eq_p {old_Eq_p:.6f} -> {Eq_p_eq:.6f}"
                f"  (d={Eq_p_eq - old_Eq_p:+.6f},"
                f" Efd={Efd_eff:.4f}, id_eq={id_eq:.4f})"
            )
        return x0

    def refine_governor_pref(self, x0: np.ndarray) -> np.ndarray:
        """Set Tm = Te at the actual network operating point.

        Computes actual Te (= vd*id + vq*iq) via comp.compute_stator_currents,
        then calls comp.update_from_te(x_slice, Te) on each governor.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        V_net = self._get_converged_network_voltages(x0)
        gen_Te = {}

        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            off   = compiler.state_offsets[comp.name]
            n     = len(comp.state_schema)
            V_bus = self._gen_bus_voltage(comp, V_net)
            if V_bus is None:
                continue
            vd, vq     = self._park_transform(V_bus, float(x0[off]))
            id_act, iq_act = comp.compute_stator_currents(x0[off:off + n], vd, vq)
            
            if hasattr(comp, 'compute_te'):
                Te_kron = comp.compute_te(x0[off:off + n], vd, vq)
            else:
                Te_kron = vd * id_act + vq * iq_act
                
            gen_Te[comp.name] = Te_kron

            # Update the generator's Tm0 parameter to match the actual Te
            # This ensures generators without governors are perfectly balanced
            Tm_pf = self.targets.get(comp.name, {}).get('Tm', Te_kron)
            S_system = float(compiler.data.get('config', {}).get('mva_base', 100.0))
            Te_max = 1.5 * float(comp.params.get('Sn', S_system)) / S_system

            # Check both absolute magnitude AND consistency with PF dispatch.
            # The Kron-reduced equilibrium can produce Te values that pass the
            # magnitude check but are wildly different from the PF operating
            # point (e.g. 1.17 instead of 0.35) due to unphysical Kron voltages.
            rel_err = abs(Te_kron - Tm_pf) / max(abs(Tm_pf), 0.01)
            te_physical = abs(Te_kron) < Te_max and rel_err < 0.5

            if te_physical:
                comp.params['Tm0'] = float(Te_kron)
            else:
                comp.params['Tm0'] = float(Tm_pf)
                print(f"  [Init] {comp.name}: Kron Te={Te_kron:.4f} unphysical "
                      f"(PF Tm={Tm_pf:.4f}, rel_err={rel_err:.1f}), "
                      f"using PF Tm={Tm_pf:.4f}")

        for comp in compiler.components:
            if comp.component_role != 'governor':
                continue
            gen_comp = self._get_generator_for_comp(comp.name)
            # Fall back to 'syn' param for old-format compatibility
            syn_name = gen_comp.name if gen_comp else comp.params.get('syn')
            if syn_name not in gen_Te:
                continue

            gen_comp = compiler.comp_map.get(syn_name)
            Te_act = gen_comp.params['Tm0'] if gen_comp else gen_Te[syn_name]

            off   = compiler.state_offsets[comp.name]
            n     = len(comp.state_schema)
            x_new, _ = comp.update_from_te(x0[off:off + n].copy(), Te_act)
            x0[off:off + n] = x_new

        return x0

    def refine_pss(self, x0: np.ndarray) -> np.ndarray:
        """Re-initialize PSS states using the actual network operating point.
        
        This is necessary because the Kron-reduced network equilibrium Te
        can differ from the power-flow Pe, and PSS states (especially for MODE=3/4)
        must be initialized to the actual Te to avoid large initial transients.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        V_net = self._get_converged_network_voltages(x0)
        gen_Te = {}

        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            off   = compiler.state_offsets[comp.name]
            n     = len(comp.state_schema)
            V_bus = self._gen_bus_voltage(comp, V_net)
            if V_bus is None:
                continue
            vd, vq     = self._park_transform(V_bus, float(x0[off]))
            id_act, iq_act = comp.compute_stator_currents(x0[off:off + n], vd, vq)
            gen_Te[comp.name] = vd * id_act + vq * iq_act

        for comp in compiler.components:
            if comp.component_role != 'pss':
                continue
            gen_comp = self._get_generator_for_comp(comp.name)
            if gen_comp is None:
                continue
            syn_name = gen_comp.name
            if syn_name not in gen_Te:
                continue

            Te_kron = gen_Te[syn_name]
            tgt = self.targets.get(syn_name, {}).copy()
            tgt['Pe'] = Te_kron
            tgt['Tm'] = Te_kron
            
            off   = compiler.state_offsets[comp.name]
            n     = len(comp.state_schema)
            x_init = comp.init_from_targets(tgt)
            x0[off:off + n] = x_init
            print(f"  [Init] {comp.name}: Re-initialized to Te={Te_kron:.4f}")

        return x0


    def refine_renewable_controllers(self, x0: np.ndarray) -> np.ndarray:
        """Re-initialize renewable controllers using the actual network operating point.
        
        Keeps the desired active and reactive power (Pe, Qe) constant from the Power Flow,
        but updates the current commands (Ipcmd, Iqcmd) and voltage (Vterm) to match
        the Kron-reduced network equilibrium.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        # Iterate V_net with Y_N×V feedback so RI-frame Norton generators
        # (DFIG with large Y_N ≈ 5 pu) get correct bus voltages.
        # A single-pass Z-bus solve without this feedback produces
        # dramatically wrong voltages (e.g. 0.17 pu instead of 1.03 pu),
        # causing Pe/Qe to be computed at the wrong operating point and
        # the RSC to be initialized with Pref ≠ Pe.
        V_net = self._compute_network_voltages(x0)
        for _viter in range(50):
            V_new = self._compute_network_voltages(x0, V_net_prev=V_net)
            if len(V_new) > 0 and np.max(np.abs(V_new - V_net)) < 1e-8:
                break
            V_net = V_new

        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            off   = compiler.state_offsets[comp.name]
            n     = len(comp.state_schema)
            V_bus = self._gen_bus_voltage(comp, V_net)
            if V_bus is None:
                continue
            tgt = self.targets.setdefault(comp.name, {})
            x_slice = x0[off:off + n].copy()
            # Dispatch to per-component Kron refinement (no-op for non-IBR generators)
            x0[off:off + n] = comp.refine_current_source_init(x_slice, tgt, V_bus)

        for comp in compiler.components:
            if comp.component_role != "renewable_controller":
                continue
            offset = self.compiler.state_offsets[comp.name]

            # Dispatch through component protocol — no class-name checks needed
            gen_name = comp.get_associated_generator(self.compiler.comp_map)
            if gen_name is None or gen_name not in self.targets:
                continue

            # Inject Pref/Qref constants from wiring map so init_from_targets can
            # account for non-zero steady-state errors (e_P = Pref - Pe_ss, etc.).
            # ONLY inject if the original wire was a CONST:. If it's a PARAM:,
            # we want init_from_targets to calculate the equilibrium value (Pe_ss)
            # and store it in the parameter.
            tgt = dict(self.targets[gen_name])
            for port in ('Pref', 'Qref'):
                # Find the original wire in the graph
                is_const = False
                for wire in compiler.graph.wires:
                    if wire.dst_component() == comp.name and wire.dst_port() == port:
                        if wire.src.startswith("CONST:"):
                            is_const = True
                        break
                
                if is_const:
                    wire_val = compiler.wiring_map.get((comp.name, port))
                    if wire_val is not None:
                        try:
                            tgt[port] = float(wire_val)
                        except (ValueError, TypeError):
                            pass  # non-constant wire (expression), skip

            x_init = comp.init_from_targets(tgt)
            x0[offset:offset + len(x_init)] = x_init
            print(f"  [Init] {comp.name}: Re-initialized to Pe={tgt['Pe']:.4f}, Qref={tgt.get('Qref', 0.0):.4f}")

        return x0

    def refine_delta_for_torque_balance(self, x0: np.ndarray,
                                        n_iter: int = 60,
                                        alpha: float = 0.15) -> np.ndarray:
        """Adjust generator rotor angle delta to achieve Te = Tm equilibrium
        in the Kron-reduced network.

        This is needed when the Kron-network terminal voltage angle differs
        from the power-flow angle (e.g. SMIB with a local load on the gen bus),
        causing the Park-transform decomposition to give wrong vd/vq and thus
        Te ≠ Tm despite correct q-axis states.

        The method is fully type-agnostic: it works with any generator component
        that has 'delta' as state[0] and provides compute_stator_currents().
        Generators without a paired governor (no Tm target) are skipped.
        Generators where Te already equals Tm (within 1e-4) are also skipped.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0

        # Check whether any generator needs delta correction
        V_net_check = self._get_converged_network_voltages(x0)
        needs_correction = False
        for comp in compiler.components:
            if comp.component_role != 'generator':
                continue
            Tm = self.targets.get(comp.name, {}).get('Tm')
            if Tm is None:
                continue
            off = compiler.state_offsets[comp.name]
            n   = len(comp.state_schema)
            V_bus = self._gen_bus_voltage(comp, V_net_check)
            if V_bus is None:
                continue
            delta = float(x0[off])
            vd, vq = self._park_transform(V_bus, delta)
            id_act, iq_act = comp.compute_stator_currents(x0[off:off+n], vd, vq)
            Te = vd * id_act + vq * iq_act
            if abs(Te - Tm) > 1e-4:
                needs_correction = True
                break

        if not needs_correction:
            return x0

        max_delta_change = 0.0
        for iteration in range(n_iter):
            # Re-compute Kron voltage at start of each outer iteration
            V_net = self._get_converged_network_voltages(x0)
            max_delta_change = 0.0

            for comp in compiler.components:
                if comp.component_role != 'generator':
                    continue
                syn_name = comp.name
                Tm = self.targets.get(syn_name, {}).get('Tm')
                if Tm is None:
                    continue

                off = compiler.state_offsets[syn_name]
                n   = len(comp.state_schema)
                V_bus = self._gen_bus_voltage(comp, V_net)
                if V_bus is None:
                    continue

                x_gen  = x0[off:off + n]
                delta  = float(x_gen[0])

                # Compute Te at current delta
                vd, vq     = self._park_transform(V_bus, delta)
                id_act, iq_act = comp.compute_stator_currents(x_gen, vd, vq)
                Te_curr    = vd * id_act + vq * iq_act
                Te_err     = Te_curr - Tm

                if abs(Te_err) < 1e-7:
                    continue

                # Numerical gradient dTe/d(delta) via central difference
                # (at fixed V_bus — the outer loop handles V_bus coupling)
                eps = 1e-4
                vd_p, vq_p = self._park_transform(V_bus, delta + eps)
                vd_m, vq_m = self._park_transform(V_bus, delta - eps)
                id_p, iq_p = comp.compute_stator_currents(x_gen, vd_p, vq_p)
                id_m, iq_m = comp.compute_stator_currents(x_gen, vd_m, vq_m)
                Te_p = vd_p * id_p + vq_p * iq_p
                Te_m = vd_m * id_m + vq_m * iq_m
                dTe_ddelta = (Te_p - Te_m) / (2 * eps)

                if abs(dTe_ddelta) < 1e-8:
                    continue

                # Newton step, under-relaxed
                d_delta = -alpha * Te_err / dTe_ddelta

                # Also update q-axis states for the new delta (for consistency)
                delta_new = delta + d_delta
                vd_n, vq_n = self._park_transform(V_bus, delta_new)
                x_new = comp.refine_at_kron_voltage(x_gen, vd_n, vq_n)
                x_new[0] = delta_new

                max_delta_change = max(max_delta_change, abs(d_delta))
                x0[off:off + n] = x_new

            if max_delta_change < 1e-7:
                print(f"  [Init] Delta torque balance converged after {iteration+1} iter(s) "
                      f"(max |ddelta|={max_delta_change:.2e}).")
                break
        else:
            print(f"  [Init] Delta torque balance: {n_iter} iters "
                  f"(max |ddelta|={max_delta_change:.2e}).")

        return x0

    def refine_passive_components(self, x0: np.ndarray) -> np.ndarray:
        """Re-initialize passive components with the final Kron-reduced network voltages."""
        compiler = self.compiler
        if compiler.z_bus is None:
            return x0
            
        V_net = self._get_converged_network_voltages(x0)
        
        for comp in compiler.components:
            if comp.component_role != "passive":
                continue
            offset = compiler.state_offsets[comp.name]
            if len(comp.state_schema) == 0:
                continue
                
            bus_id = comp.params.get("bus")
            if bus_id is None:
                for wire in compiler.graph.wires:
                    if wire.dst_component() == comp.name and wire.src.startswith("BUS_"):
                        # wire.src format: "BUS_<id>.<port>"
                        bus_id = int(wire.src.split(".")[0].split("_", 1)[1])
                        break
                        
            if bus_id is not None:
                full_idx = compiler.ybus_builder.bus_map.get(bus_id)
                if full_idx is not None:
                    red_idx = compiler.active_bus_map.get(full_idx)
                    if red_idx is not None:
                        V_phasor = V_net[red_idx]
                        targets = {
                            'Vd': V_phasor.real,
                            'Vq': V_phasor.imag,
                            'Vterm': abs(V_phasor)
                        }
                        x_init = comp.init_from_targets(targets)
                        x0[offset:offset + len(x_init)] = x_init
            else:
                # No bus found — still call init_from_targets with empty dict
                x_init = comp.init_from_targets({})
                x0[offset:offset + len(x_init)] = x_init
                        
        return x0

    def compute_initial_network_voltages(self, x0: np.ndarray) -> tuple:
        """Compute consistent Vd/Vq at each active bus from the final x0.

        The Z-bus solve is iterated with Y_N×V feedback so that the embedded
        Vd0/Vq0 match the C++ algebraic-loop equilibrium.  Without this
        iteration, RI-frame Norton generators (DFIG) whose Y_N×V contribution
        is voltage-dependent produce dramatically wrong single-pass voltages,
        causing a large initial voltage spike.

        Returns (Vd_list, Vq_list) indexed by active-bus position.
        """
        compiler = self.compiler
        if compiler.z_bus is None:
            V = self.pf.V; theta = self.pf.theta
            return (
                [V[ab] * math.cos(theta[ab]) for ab in compiler.active_buses],
                [V[ab] * math.sin(theta[ab]) for ab in compiler.active_buses],
            )

        # Iterate the Z-bus solve with Y_N×V feedback (mirrors C++ algebraic loop)
        V_net = self._compute_network_voltages(x0)
        for _it in range(50):
            V_new = self._compute_network_voltages(x0, V_net_prev=V_net)
            if len(V_new) > 0 and len(V_net) > 0:
                dV = float(np.max(np.abs(V_new - V_net)))
            else:
                dV = 0.0
            V_net = V_new
            if dV < 1e-6:
                print(f"  [Init] Initial voltage iteration converged after {_it + 1} iter(s) "
                      f"(max|dV|={dV:.2e}).")
                break
        else:
            print(f"  [Init] Initial voltage iteration: 50 iters (max|dV|={dV:.2e}).")

        Vd_list = [V_net[i].real for i in range(compiler.n_active)]
        Vq_list = [V_net[i].imag for i in range(compiler.n_active)]

        print(f"  [Init] Initial network voltages computed from x0 (Z-bus):")
        for i in range(compiler.n_active):
            Vmag = abs(V_net[i])
            Vang = math.degrees(math.atan2(V_net[i].imag, V_net[i].real))
            print(f"    Bus[{i}]: |V|={Vmag:.5f}, ang={Vang:.3f}deg")
        return Vd_list, Vq_list
