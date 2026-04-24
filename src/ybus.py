import numpy as np
import math
from collections import defaultdict

class YBusBuilder:
    """
    Constructs the Nodal Admittance Matrix (Y-Bus) from ANDES/JSON system data.
    """
    def __init__(self, data):
        self.data = data
        self.buses = sorted([b['idx'] for b in data.get('Bus', [])])
        self.n_bus = len(self.buses)
        self.bus_map = {b_idx: i for i, b_idx in enumerate(self.buses)}
        
        # Initialize Y matrix (Complex)
        self.Y = np.zeros((self.n_bus, self.n_bus), dtype=complex)
        self._built = False
        
    def build(self, include_loads=True) -> np.ndarray:
        """Parses Lines, Transformers, Shunts to build Y.
        
        Args:
            include_loads: If True, add constant-impedance load admittance.
                           Set False for power flow (loads modeled as P/Q specs).
                           Set True for dynamic simulation reduced network.
        """
        if self._built:
            return self.Y
        self._add_lines()
        self._add_shunts()
        if include_loads:
            self._add_load_admittance()
        self._built = True
        return self.Y

    def _add_lines(self):
        """Process 'Line' entries."""
        if 'Line' not in self.data:
            return

        for line in self.data['Line']:
            from_idx = line['bus1']
            to_idx = line['bus2']
            
            if from_idx not in self.bus_map or to_idx not in self.bus_map:
                continue
                
            i = self.bus_map[from_idx]
            j = self.bus_map[to_idx]
            
            # Parameters
            r = line.get('r', 0.0)
            x = line.get('x', 0.001)
            b = line.get('b', 0.0) # Total line charging
            tap = line.get('tap', 1.0) # Off-nominal turns ratio
            phi = line.get('phi', 0.0) # Phase shift in rad
            
            # Series Admittance
            z_series = complex(r, x)
            y_series = 1.0 / z_series
            
            # Shunt Admittance (Half per side)
            y_shunt = complex(0, b/2.0)
            
            # Complex Tap Ratio: a = tap * exp(j*phi)
            a = tap * (math.cos(phi) + 1j * math.sin(phi))
            a_conj = a.conjugate()
            mag_a2 = abs(a)**2
            
            # Pi-Model Elements with Transformer
            # Y_ii += (y_series + j*b/2) / |a|^2  (Wait, standard formulation?)
            # Standard Pi Model with Off-Nominal Tap at 'i' side:
            # Y_ii = (y_series + y_shunt) / |a|^2 ??
            # Usually Tap is on one side. Assuming 'bus1' is tap side.
            
            # Element:
            # [ I_i ]   [  y_s/|a|^2      -y_s/a*   ] [ V_i ]
            # [ I_j ] = [ -y_s/a           y_s      ] [ V_j ]
            # (ignoring shunt for a moment)
            
            # With Shunts:
            # Y_ii += (y_series / mag_a2) + y_shunt
            # Y_jj += y_series + y_shunt
            # Y_ij -= y_series / a_conj
            # Y_ji -= y_series / a
            
            self.Y[i, i] += (y_series / mag_a2) + y_shunt
            self.Y[j, j] += y_series + y_shunt
            self.Y[i, j] -= y_series / a_conj
            self.Y[j, i] -= y_series / a

    def _add_shunts(self):
        """Process 'Shunt' entries."""
        if 'Shunt' not in self.data:
            return
            
        for shunt in self.data['Shunt']:
            bus_idx = shunt['bus']
            if bus_idx not in self.bus_map:
                continue
            i = self.bus_map[bus_idx]
            
            g = shunt.get('g', 0.0)
            b = shunt.get('b', 0.0)
            self.Y[i, i] += complex(g, b)

    def _add_load_admittance(self):
        """Add constant-impedance load admittance at PQ buses (P - jQ)/V0^2 so network solve matches power flow."""
        pq_list = self.data.get('PQ', [])
        if not pq_list:
            return
        bus_to_pq = defaultdict(lambda: [0.0, 0.0])  # bus -> [P, Q] (consumption)
        for pq in pq_list:
            bus = pq['bus']
            if bus not in self.bus_map:
                continue
            bus_to_pq[bus][0] += pq.get('p0', 0.0)
            bus_to_pq[bus][1] += pq.get('q0', 0.0)
        bus_data = {b['idx']: b for b in self.data.get('Bus', [])}
        for bus, (P, Q) in bus_to_pq.items():
            if P == 0 and Q == 0:
                continue
            i = self.bus_map[bus]
            V0 = float(bus_data.get(bus, {}).get('v0', 1.0))
            V02 = max(V0 * V0, 1e-12)
            # Constant impedance: I = conj(S/V) => Y_load = conj(S)/|V|^2 = (P - j*Q)/V0^2
            self.Y[i, i] += (P - 1j * Q) / V02

    def add_fault_shunt(self, bus_idx, r: float, x: float):
        """Add a fault shunt admittance Y = 1/(r + jx) at a bus."""
        if bus_idx not in self.bus_map:
            return
        i = self.bus_map[bus_idx]
        z = complex(r, x)
        if abs(z) < 1e-12:
            z = complex(0.0, 1e-6)
        self.Y[i, i] += 1.0 / z

    def add_generator_impedance(self, bus_idx, ra, xd_pp):
        """Adds generator sub-transient admittance to diagonal."""
        if bus_idx not in self.bus_map:
            return
        i = self.bus_map[bus_idx]
        
        # Z = ra + j*xd''
        # Y = 1/Z
        z = complex(ra, xd_pp)
        if abs(z) < 1e-6:
            z = complex(0.0, 0.0001) # Avoid div by zero
            
        y = 1.0 / z
        self.Y[i, i] += y

    def get_z_bus_kron(self, gen_bus_indices, pf_V=None, pf_theta=None):
        """Build reduced Z-bus via Kron reduction (like Archive/system_coordinator.py).
        
        Keeps generator buses as active nodes, eliminates load and slack buses.
        Loads are modelled as constant-impedance shunts using PF voltages.
        Slack buses (infinite buses) are modelled as Norton equivalents:
          admittance Y_slack = 1/(ra + j*xs) shunted to ground, plus the
          Norton current I_slack = V_slack * Y_slack stored in _last_kron so
          that _compute_network_voltages can inject it into the Kron network.
        
        Args:
            gen_bus_indices: list of internal bus indices where generators are connected
            pf_V:     array of power-flow voltage magnitudes (indexed by internal bus).
                      Pass None to fall back to the v0 field in the Bus JSON (or 1.0).
            pf_theta: array of power-flow voltage angles (radians, internal bus index).
                      Required for correct slack-bus Norton current direction.
            
        Returns:
            Z_reduced: n_gen_bus x n_gen_bus complex Z matrix
            active_buses: sorted list of active bus internal indices
        """
        active_buses = sorted(set(gen_bus_indices))
        load_buses = [i for i in range(self.n_bus) if i not in active_buses]
        
        if len(load_buses) == 0:
            Y_red = self.Y[np.ix_(active_buses, active_buses)].copy()
            self._last_kron = {
                'Yll_inv': np.zeros((0, 0)), 'Yla': np.zeros((0, len(active_buses))),
                'load_buses': [], 'active_buses': active_buses,
                'slack_norton_I': np.zeros(len(active_buses), dtype=complex),
            }
            return np.linalg.inv(Y_red), active_buses
        
        Yaa = self.Y[np.ix_(active_buses, active_buses)]
        Yal = self.Y[np.ix_(active_buses, load_buses)]
        Yla = self.Y[np.ix_(load_buses, active_buses)]
        Yll = self.Y[np.ix_(load_buses, load_buses)].copy()
        
        pq_list = self.data.get('PQ', [])
        bus_data = {b['idx']: b for b in self.data.get('Bus', [])}
        bus_to_pq = defaultdict(lambda: [0.0, 0.0])
        for pq in pq_list:
            bus = pq['bus']
            if bus in self.bus_map:
                bus_to_pq[bus][0] += pq.get('p0', 0.0)
                bus_to_pq[bus][1] += pq.get('q0', 0.0)
        
        inv_map = {v: k for k, v in self.bus_map.items()}
        
        # Identify slack (infinite) buses among the eliminated load buses.
        # Slack buses have a *fixed* terminal voltage V_sl (Thevenin sources).
        # The correct Kron reduction separates load buses into:
        #   L' = passive load buses (constant-impedance loads, I_L' = 0)
        #   S  = slack buses (fixed voltage V_s, inject current into the network)
        # Reference: Bergen & Vittal "Power Systems Analysis" §11.3;
        #            Kundur "Power System Stability and Control" §12.2.
        slack_list = self.data.get('Slack', [])
        slack_bus_V = {}   # internal_idx -> complex V_sl
        for sl in slack_list:
            bus_id = sl.get('bus')
            if bus_id not in self.bus_map:
                continue
            ib = self.bus_map[bus_id]
            if ib in active_buses:
                continue  # generator also on this bus
            if pf_V is not None and pf_theta is not None:
                V_sl = pf_V[ib] * np.exp(1j * pf_theta[ib])
            else:
                v0  = float(bus_data.get(bus_id, {}).get('v0', 1.0))
                a0  = float(bus_data.get(bus_id, {}).get('a0', 0.0))
                V_sl = v0 * np.exp(1j * a0)
            slack_bus_V[ib] = V_sl

        # Partition load_buses into passive (L') and slack (S)
        passive_indices = [i for i, lb in enumerate(load_buses) if lb not in slack_bus_V]
        slack_indices   = [i for i, lb in enumerate(load_buses) if lb in slack_bus_V]

        # Extract sub-matrices for the correct partitioned Kron reduction
        # Y_ll' = Y[passive, passive], Y_ls = Y[passive, slack]
        # Y_al' = Y[active, passive],  Y_as = Y[active, slack]
        Yll_p = Yll[np.ix_(passive_indices, passive_indices)].copy()  # passive x passive
        Yal_p = Yal[:, passive_indices]    # active x passive
        Yla_p = Yla[passive_indices, :]    # passive x active
        Yll_s = Yll[np.ix_(passive_indices, slack_indices)]  # passive x slack
        Yal_s = Yal[:, slack_indices]      # active x slack

        # Add load admittances at PASSIVE load buses into Yll_p diagonal.
        for i_loc, i_glb in enumerate(passive_indices):
            lb = load_buses[i_glb]
            bus_id = inv_map.get(lb)
            if bus_id and bus_id in bus_to_pq:
                P, Q = bus_to_pq[bus_id]
                if abs(P) > 1e-6 or abs(Q) > 1e-6:
                    S = P + 1j * Q
                    if pf_V is not None:
                        V02 = max(pf_V[lb] ** 2, 1e-12)
                    else:
                        v0 = float(bus_data.get(bus_id, {}).get('v0', 1.0))
                        V02 = max(v0 * v0, 1e-12)
                    Yll_p[i_loc, i_loc] += np.conj(S) / V02

        # Also add load admittances at GENERATOR (active) buses into Yaa diagonal.
        Yaa = Yaa.copy()
        for i, ab in enumerate(active_buses):
            bus_id = inv_map.get(ab)
            if bus_id and bus_id in bus_to_pq:
                P, Q = bus_to_pq[bus_id]
                if abs(P) > 1e-6 or abs(Q) > 1e-6:
                    S = P + 1j * Q
                    if pf_V is not None:
                        V02 = max(pf_V[ab] ** 2, 1e-12)
                    else:
                        v0 = float(bus_data.get(bus_id, {}).get('v0', 1.0))
                        V02 = max(v0 * v0, 1e-12)
                    Yaa[i, i] += np.conj(S) / V02

        # Tiny regularisation to prevent singular Yll_p.
        n_p = len(passive_indices)
        if n_p > 0:
            Yll_p += np.eye(n_p) * 1e-8
            Yll_p_inv = np.linalg.inv(Yll_p)
            # Kron-reduced admittance (active buses only, passive loads eliminated)
            Y_reduced = Yaa - Yal_p @ Yll_p_inv @ Yla_p
        else:
            Yll_p_inv = np.zeros((0, 0))
            Y_reduced = Yaa.copy()

        Y_reduced += np.eye(len(active_buses)) * 1e-10

        # Slack (Thevenin) contribution to the active bus current injections:
        #   I_slack_eff = (Y_as - Y_al' @ Yll_p^{-1} @ Y_ls) @ V_s
        # This is the standard result from partitioned Kron reduction with
        # fixed-voltage slack buses.
        slack_norton_I = np.zeros(len(active_buses), dtype=complex)
        if slack_indices:
            V_sl_vec = np.array([slack_bus_V[load_buses[i]] for i in slack_indices])
            # Derivation (partitioned KCL with fixed slack voltage V_s):
            #   Y_red @ Va = I_a + (Yal_p @ Yll_p_inv @ Yll_s - Yal_s) @ V_s
            # so the effective Norton injection from the slack bus into the active
            # buses is:  I_slack_eff = +(Yal_p @ Yll_p_inv @ Yll_s - Yal_s) @ V_s
            #                        = -(Yal_s - Yal_p @ Yll_p_inv @ Yll_s) @ V_s
            if n_p > 0:
                transfer = Yal_s - Yal_p @ (Yll_p_inv @ Yll_s)
            else:
                transfer = Yal_s
            slack_norton_I = -transfer @ V_sl_vec

        # For the voltage-recovery intermediates, we still need Yll_inv and Yla for
        # ALL load buses (including slack) so observer buses can be recovered.
        # Fall back to the simple whole-Yll inverse for the recovery matrix.
        Yll_full = Yll.copy()
        Yll_full += np.eye(Yll_full.shape[0]) * 1e-8
        Yll_inv = np.linalg.inv(Yll_full)

        # Save intermediates for voltage recovery at eliminated buses
        self._last_kron = {
            'Yll_inv': Yll_inv, 'Yla': Yla,
            'load_buses': load_buses, 'active_buses': active_buses,
            'slack_norton_I': slack_norton_I,
        }
        
        return np.linalg.inv(Y_reduced), active_buses

    def get_voltage_recovery_matrix(self, observer_bus_ids):
        """Return a matrix M such that V_obs = M @ V_active (complex).

        Uses the saved Kron-reduction intermediates:
            V_load = -Yll_inv @ Yla @ V_active
        then extracts the rows corresponding to the requested observer buses.

        Returns:
            M: (n_obs, n_active) complex matrix
            obs_names: list of bus IDs in order
        """
        kron = self._last_kron
        M_full = -kron['Yll_inv'] @ kron['Yla']  # (n_load, n_active)
        load_buses = kron['load_buses']

        rows = []
        obs_names = []
        for bus_id in observer_bus_ids:
            internal = self.bus_map.get(bus_id)
            if internal is None:
                continue
            try:
                idx_in_load = load_buses.index(internal)
            except ValueError:
                continue
            rows.append(M_full[idx_in_load])
            obs_names.append(bus_id)

        if not rows:
            return None, []
        return np.array(rows), obs_names
