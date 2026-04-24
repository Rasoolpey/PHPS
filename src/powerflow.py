import numpy as np
import math
from src.ybus import YBusBuilder

class PowerFlow:
    """
    Newton-Raphson Power Flow Solver.
    """
    def __init__(self, system_data):
        self.data = system_data
        
        # 1. Build Y-Bus
        self.ybus_builder = YBusBuilder(system_data)
        self.Y = self.ybus_builder.build(include_loads=False)
        self.buses = self.ybus_builder.buses
        self.n_bus = len(self.buses)
        self.bus_map = self.ybus_builder.bus_map
        
        # 2. Parse Bus Types & Specs
        self.V = np.ones(self.n_bus)
        self.theta = np.zeros(self.n_bus)
        self.P_spec = np.zeros(self.n_bus)
        self.Q_spec = np.zeros(self.n_bus)
        self.bus_types = np.zeros(self.n_bus, dtype=int) # 0=PQ, 1=PV, 2=Slack
        
        self._load_specs()
        
    def _load_specs(self):
        """Loads P, Q, V specs from JSON."""
        # Defaults
        for i in range(self.n_bus):
            self.bus_types[i] = 0 # Default PQ
            self.V[i] = 1.0
            self.theta[i] = 0.0
            
        # Slack
        if 'Slack' in self.data:
            for s in self.data['Slack']:
                if s['bus'] in self.bus_map:
                    idx = self.bus_map[s['bus']]
                    self.bus_types[idx] = 2
                    self.V[idx] = s.get('v0', 1.0)
                    self.theta[idx] = s.get('a0', 0.0)
                    
        # PV
        if 'PV' in self.data:
            for p in self.data['PV']:
                if p['bus'] in self.bus_map:
                    idx = self.bus_map[p['bus']]
                    if self.bus_types[idx] != 2: # Don't overwrite Slack
                        self.bus_types[idx] = 1
                        self.V[idx] = p.get('v0', 1.0)
                        self.P_spec[idx] += p.get('p0', 0.0)
                        
        # PQ (Loads)
        if 'PQ' in self.data:
            for p in self.data['PQ']:
                if p['bus'] in self.bus_map:
                    idx = self.bus_map[p['bus']]
                    # Loads subtract from injection
                    self.P_spec[idx] -= p.get('p0', 0.0)
                    self.Q_spec[idx] -= p.get('q0', 0.0)

    def load_bus_overrides(self):
        """Load v0/a0 from Bus section for ALL buses (bypass NR).
        Use when Bus v0/a0 come from an external power flow (e.g. PowerFactory)."""
        if 'Bus' not in self.data:
            return False
        bus_map_by_name = {}
        for bus_id, idx in self.bus_map.items():
            bus_map_by_name[f"BUS{bus_id}"] = idx
            bus_map_by_name[str(bus_id)] = idx
            bus_map_by_name[bus_id] = idx
        loaded = 0
        for b in self.data['Bus']:
            name = b.get('name', b.get('idx'))
            idx = bus_map_by_name.get(name)
            if idx is None:
                continue
            if 'v0' in b:
                self.V[idx] = b['v0']
            if 'a0' in b:
                self.theta[idx] = b['a0']
            loaded += 1
        print(f"  [PF] Loaded v0/a0 overrides for {loaded}/{self.n_bus} buses (skip NR)")
        return loaded > 0

    def solve(self, tol=1e-6, max_iter=20):
        """Solves power flow."""
        for it in range(max_iter):
            # Calc P, Q mismatches
            S_calc = self.calculate_power()
            P_calc = S_calc.real
            Q_calc = S_calc.imag
            
            dP = self.P_spec - P_calc
            dQ = self.Q_spec - Q_calc
            
            unknowns = []
            mismatch = []
            theta_map = {}
            v_map = {}
            col = 0
            
            # 1. Thetas (for PQ and PV)
            for i in range(self.n_bus):
                if self.bus_types[i] != 2: # Not Slack
                    theta_map[i] = col
                    unknowns.append(f"t{i}")
                    mismatch.append(dP[i])
                    col += 1
                    
            # 2. Voltages (for PQ only)
            for i in range(self.n_bus):
                if self.bus_types[i] == 0: # PQ
                    v_map[i] = col
                    unknowns.append(f"v{i}")
                    mismatch.append(dQ[i])
                    col += 1
            
            if len(mismatch) == 0:
                break
                
            norm = np.max(np.abs(mismatch))
            if norm < tol:
                print(f"Power Flow Converged in {it} iterations. Norm: {norm:.6e}")
                return True
                
            # Jacobian
            J = np.zeros((len(unknowns), len(unknowns)))
            
            for i in range(self.n_bus):
                if i in theta_map:
                    row = theta_map[i]
                    for j in range(self.n_bus):
                        if j in theta_map:
                            col_j = theta_map[j]
                            if i == j:
                                val = -Q_calc[i] - (self.V[i]**2) * self.Y[i,i].imag
                                J[row, col_j] = val
                            else:
                                tij = self.theta[i] - self.theta[j]
                                Gij = self.Y[i,j].real
                                Bij = self.Y[i,j].imag
                                val = self.V[i] * self.V[j] * (Gij * np.sin(tij) - Bij * np.cos(tij))
                                J[row, col_j] = val
                                
                if i in theta_map: 
                    row = theta_map[i]
                    for j in range(self.n_bus):
                        if j in v_map:
                            col_j = v_map[j]
                            if i == j:
                                val = P_calc[i]/self.V[i] + self.V[i]*self.Y[i,i].real
                                J[row, col_j] = val
                            else:
                                tij = self.theta[i] - self.theta[j]
                                Gij = self.Y[i,j].real
                                Bij = self.Y[i,j].imag
                                val = self.V[i] * (Gij * np.cos(tij) + Bij * np.sin(tij))
                                J[row, col_j] = val

                if i in v_map:
                    row = v_map[i]
                    for j in range(self.n_bus):
                        if j in theta_map:
                            col_j = theta_map[j]
                            if i == j:
                                val = P_calc[i] - (self.V[i]**2)*self.Y[i,i].real
                                J[row, col_j] = val
                            else:
                                tij = self.theta[i] - self.theta[j]
                                Gij = self.Y[i,j].real
                                Bij = self.Y[i,j].imag
                                val = -self.V[i] * self.V[j] * (Gij * np.cos(tij) + Bij * np.sin(tij))
                                J[row, col_j] = val
                                
                    for j in range(self.n_bus):
                        if j in v_map:
                            col_j = v_map[j]
                            if i == j:
                                val = Q_calc[i]/self.V[i] - self.V[i]*self.Y[i,i].imag
                                J[row, col_j] = val
                            else:
                                tij = self.theta[i] - self.theta[j]
                                Gij = self.Y[i,j].real
                                Bij = self.Y[i,j].imag
                                val = self.V[i] * (Gij * np.sin(tij) - Bij * np.cos(tij))
                                J[row, col_j] = val
                                
            dx = np.linalg.solve(J, mismatch)
            
            for i, d_val in zip(theta_map.keys(), dx[:len(theta_map)]):
                self.theta[i] += d_val
            
            for i, d_val in zip(v_map.keys(), dx[len(theta_map):]):
                self.V[i] += d_val
                
        return False

    def calculate_power(self):
        """Calculate power injections at all buses."""
        V_complex = self.V * np.exp(1j * self.theta)
        I = self.Y @ V_complex
        S = V_complex * np.conj(I)
        return S

    def get_results(self):
        """Returns dict with bus_idx -> {v, angle}."""
        res = {}
        for b_idx, i in self.bus_map.items():
            res[b_idx] = {'v': self.V[i], 'angle': self.theta[i]}
        return res
