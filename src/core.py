from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.system_graph import Wire


class PowerComponent:
    """
    Base class for all power system components in the compiler-first architecture.
    
    This class serves as the 'Source of Truth' for:
    1. Interface definition (ports, states, parameters).
    2. C++ code generation (equations).
    3. Metadata for tooling (plotting, validation).
    
    It separates the Model (class definition) from the Data (instance parameters).
    The logic is defined via C++ snippets, not Python methods.
    """
    
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        Initialize the component with a specific name and parameter set.
        
        Args:
            name: Unique identifier for this instance (e.g., "G1").
            params: Dictionary of physical parameters (e.g., {'H': 3.5, 'D': 0.0}).
                    Must match param_schema.
        """
        self.name = name
        self.params = params
        self._validate_params()

    # --- Schema Definitions (Metadata) ---

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Defines the input and output ports.
        
        Returns:
            Dict with keys 'in' and 'out'.
            Values are lists of tuples: (port_name, kind, units).
            kind should be 'effort', 'flow', or 'signal'.
            
            Example:
            {
                'in':  [('Tm', 'effort', 'pu'), ('Vf', 'effort', 'pu')],
                'out': [('omega', 'flow', 'pu'), ('Pe', 'flow', 'pu')]
            }
        """
        raise NotImplementedError

    @property
    def state_schema(self) -> List[str]:
        """
        List of state variable names. 
        The order here determines the index in the C++ 'x' array.
        
        Example: ['delta', 'omega', 'e_q_prime']
        """
        raise NotImplementedError

    @property
    def param_schema(self) -> Dict[str, str]:
        """
        Dictionary of required parameters and their descriptions/units.
        Used for validation and to generate C++ struct definitions.
        
        Example: {'H': 'Inertia constant [MWs/MVA]', 'D': 'Damping coeff'}
        """
        raise NotImplementedError
        
    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        """
        Dictionary of observable signals that can be computed from states/inputs.
        Used for plotting and analysis without recording full state history.
        
        Format:
            name: {
                'description': str,
                'unit': str,
                'cpp_expr': str  # C++ expression to compute it
            }
        """
        return {}

    # --- Code Generation (The Kernel) ---

    def get_cpp_step_code(self) -> str:
        """
        Returns the C++ implementation of the state dynamics: dx/dt = f(x, u).
        
        When a component provides ``get_signal_flow_graph()`` or
        ``get_symbolic_phs()``, the C++ code is auto-generated from that
        declarative model — no hand-written C++ is needed. Signal-flow graphs
        take precedence for runtime code generation; the symbolic PHS remains
        available for analysis and equilibrium solving.

        Components may still override this method when the auto-generated code
        must be augmented.
        
        Context provided to this snippet:
        - double* x: Pointer to current states (indexed 0 to n_states-1)
        - double* dxdt: Pointer to state derivatives (write output here)
        - double* u: Pointer to inputs (flattened based on port_schema['in'])
        - <ParamName>: All keys in param_schema are available as const doubles
        """
        sfg = self.get_signal_flow_graph()
        if sfg is not None:
            from src.signal_flow.codegen import generate_signal_flow_cpp_dynamics
            state_c_names = self.state_schema
            input_c_names = [p[0] for p in self.port_schema['in']]
            return generate_signal_flow_cpp_dynamics(sfg, state_c_names, input_c_names)

        sphs = self.get_symbolic_phs()
        if sphs is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_cpp_step_code() "
                f"or provide get_symbolic_phs()/get_signal_flow_graph()"
            )
        from src.symbolic.codegen import generate_phs_cpp_dynamics
        state_c_names = self.state_schema
        input_c_names = [p[0] for p in self.port_schema['in']]
        return generate_phs_cpp_dynamics(sphs, state_c_names, input_c_names)

    def get_cpp_compute_outputs_code(self) -> str:
        """
        Returns the C++ implementation of output equations: y = g(x, u).

        When a component's ``get_signal_flow_graph()`` or ``get_symbolic_phs()``
        provides an output map, the C++ code is auto-generated from that
        declarative representation. Components may still override this method
        for complex output logic.
        
        Context provided to this snippet:
        - double* x: Pointer to current states
        - double* u: Pointer to inputs
        - double* y: Pointer to outputs (flattened based on port_schema['out'])
        - <ParamName>: Parameters available as const doubles
        """
        sfg = self.get_signal_flow_graph()
        if sfg is not None and sfg.output_map:
            from src.signal_flow.codegen import generate_signal_flow_cpp_outputs
            state_c_names = self.state_schema
            input_c_names = [p[0] for p in self.port_schema['in']]
            return generate_signal_flow_cpp_outputs(sfg, state_c_names, input_c_names)

        sphs = self.get_symbolic_phs()
        if sphs is not None and sphs.output_map is not None:
            from src.symbolic.codegen import generate_phs_cpp_outputs
            state_c_names = self.state_schema
            input_c_names = [p[0] for p in self.port_schema['in']]
            return generate_phs_cpp_outputs(sphs, state_c_names, input_c_names)
        raise NotImplementedError
    
    # --- Port validation helpers ---

    @property
    def required_ports(self) -> List[str]:
        """List of input port names that *must* be connected.

        By default all input ports are required.  Override in a subclass to
        make certain ports optional (e.g. a PSS Tm input that defaults to 0).
        The SystemGraph validator uses this list when checking wires.
        """
        return [p[0] for p in self.port_schema["in"]]

    def validate_connection(self, wire: "Wire") -> None:
        """Validate that *wire* is a legal connection to or from this component.

        Called by SystemGraph.validate() for each wire whose source or
        destination is this component.  Should raise a FrameworkError subclass
        (PortNotFoundError, PortTypeMismatchError, …) if the wire is illegal.

        The base implementation checks that the referenced port name exists in
        the appropriate direction.  Subclasses can tighten constraints (e.g.
        check signal units, cardinality, or domain-specific rules).

        Parameters
        ----------
        wire : Wire
            The wire being validated.  Call wire.src_component() /
            wire.dst_component() to determine whether this component is on
            the source or destination side.
        """
        from src.errors import PortNotFoundError, PortTypeMismatchError

        if wire.dst_component() == self.name:
            # This component is the destination: check the input port exists
            in_port_names = [p[0] for p in self.port_schema["in"]]
            port = wire.dst_port()
            if port not in in_port_names:
                raise PortNotFoundError(self.name, port, "in", in_port_names)

        if wire.src_kind() == "comp" and wire.src_component() == self.name:
            # This component is the source: check the output port exists
            out_port_names = [p[0] for p in self.port_schema["out"]]
            port = wire.src_port()
            if port not in out_port_names:
                raise PortNotFoundError(self.name, port, "out", out_port_names)

    # --- Helper Methods ---

    def _validate_params(self):
        """Checks if self.params contains all keys in self.param_schema."""
        missing = [k for k in self.param_schema if k not in self.params]
        if missing:
            raise ValueError(f"Component '{self.name}' ({self.__class__.__name__}) missing parameters: {missing}")

    def get_cpp_struct_name(self) -> str:
        """Returns the C++ struct name, usually matching the class name."""
        return self.__class__.__name__

    # ------------------------------------------------------------------ #
    # Port-Hamiltonian System (PHS) interface                              #
    # Used by the Dirac-structure analysis / DAE pipeline (src/dirac/).    #
    #                                                                      #
    # These default implementations derive everything from the symbolic    #
    # PHS layer (get_symbolic_phs).  PH-aware components only need to      #
    # implement get_symbolic_phs(); the Hamiltonian, gradient, PHS         #
    # matrices, and C++ Hamiltonian expression are all auto-derived here.  #
    # ------------------------------------------------------------------ #

    def get_phs_matrices(self, x: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Return the PHS structure matrices {J, R, g, Q} at state *x*.

        Auto-derived from ``get_symbolic_phs()`` — no hand-coding needed.
        Returns ``None`` for components without a PHS formulation.
        """
        sphs = self.get_symbolic_phs()
        if sphs is None:
            return None
        from src.symbolic.codegen import evaluate_phs_matrices
        return evaluate_phs_matrices(sphs, self.params, x)

    def hamiltonian(self, x: np.ndarray) -> float:
        """Evaluate the storage function H(x).

        Auto-derived from the symbolic Hamiltonian in ``get_symbolic_phs()``.
        The lambdified function is cached on first call for performance.
        """
        if not hasattr(self, '_cached_H_func'):
            sphs = self.get_symbolic_phs()
            if sphs is None:
                return 0.0
            from src.symbolic.codegen import make_hamiltonian_func
            self._cached_H_func = make_hamiltonian_func(sphs, self.params)
        return self._cached_H_func(x)

    def grad_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ∂H/∂x at state *x*.

        Auto-derived from the symbolic Hamiltonian in ``get_symbolic_phs()``.
        The lambdified function is cached on first call for performance.
        """
        if not hasattr(self, '_cached_gradH_func'):
            sphs = self.get_symbolic_phs()
            if sphs is None:
                return np.zeros(len(self.state_schema))
            from src.symbolic.codegen import make_grad_hamiltonian_func
            self._cached_gradH_func = make_grad_hamiltonian_func(
                sphs, self.params
            )
        return self._cached_gradH_func(x)

    def _hamiltonian_cpp_expr(self) -> str:
        """C++ expression for H(x), used in observables.

        Auto-derived from the symbolic Hamiltonian.  Returns ``"0.0"``
        for components without a PHS formulation.
        """
        sphs = self.get_symbolic_phs()
        if sphs is None:
            return "0.0"
        from src.symbolic.codegen import generate_hamiltonian_cpp_expr
        return generate_hamiltonian_cpp_expr(sphs, self.params)

    def get_symbolic_phs(self):
        """Return a SymbolicPHS object for this component.

        PH-aware components (those with ``_phs.py`` variants) override this
        to return a ``src.symbolic.SymbolicPHS`` with SymPy matrices for
        J, R, g, Q and a SymPy expression for H(x).

        Returns ``None`` for components without a PHS formulation.
        """
        return None

    def get_signal_flow_graph(self):
        """Return a SignalFlowGraph object for this component.

        Components with ordered block-diagram runtime logic can expose a
        declarative signal-flow graph here. The base class uses it to
        auto-generate step and output C++ code.

        Returns ``None`` for components without a signal-flow formulation.
        """
        return None

    # --- Initialization Contract ---
    # Default no-op implementations for passive components.
    # Generators, exciters, governors, and PSS models override what they need.

    @property
    def component_role(self) -> str:
        """One of: 'generator' | 'exciter' | 'governor' | 'pss' | 'passive'."""
        return 'passive'

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        """RI-frame Norton current injection.  Generators implement this."""
        return 0j

    def compute_stator_currents(self, x_slice: np.ndarray,
                                vd: float, vq: float) -> tuple:
        """Solve stator equations for (id_act, iq_act) at terminal vd, vq.
        Generators implement this; used for Te computation."""
        return 0.0, 0.0

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """Init states from power-flow phasor operating point.
        Returns (x0_slice: np.ndarray, targets: dict).
        targets must contain at minimum: Efd, Tm, Vt, vd, vq, id, iq,
        vd_ri, vq_ri (RI-frame voltages for ESST3A VE)."""
        return np.zeros(len(self.state_schema)), {}

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Back-solve equilibrium states from targets dict.

        If the component's :class:`SymbolicPHS` declares an
        :class:`InitSpec` (via ``set_init_spec``), the equilibrium is
        solved generically from the symbolic dynamics.  Otherwise
        subclasses should override this method.

        Also sets ``self.params['Vref']`` / ``self.params['Pref']``.
        """
        # --- Try auto-derived path via SymbolicPHS init_spec -----------
        try:
            sphs = self.get_symbolic_phs()
        except (NotImplementedError, AttributeError):
            return np.zeros(len(self.state_schema))

        init_spec = getattr(sphs, 'init_spec', None)
        if init_spec is not None:
            from src.symbolic.codegen import solve_equilibrium
            x0, free_params = solve_equilibrium(sphs, self.params, targets)
            self.params.update(free_params)
            if init_spec.post_init_func is not None:
                extra_params = init_spec.post_init_func(
                    x0, free_params, targets, self.params
                )
                if extra_params:
                    self.params.update(extra_params)
            return x0

        return np.zeros(len(self.state_schema))

    def rebalance_for_bus_voltage(self, x_slice: np.ndarray,
                                   V_bus_complex: complex) -> tuple:
        """Rebalance generator states and compute controller targets at a
        given DAE-consistent complex bus voltage.

        Called by the DAE runner after the network voltage solve overwrites
        the bus voltages.  The generator uses its own model equations (Park
        transform, stator algebraic, Te formula, Efd equilibrium) to:

          1. Compute (vd, vq) from V_bus_complex via the Park transform.
          2. Solve stator algebraic for (id_act, iq_act).
          3. Compute Te = vd·id + vq·iq.
          4. Compute Efd_req from the field-winding ODE SS condition.
          5. Return updated x_slice and a targets dict for controllers.

        The targets dict must contain at minimum:

            {'Tm': float, 'Efd': float, 'Vt': float, 'omega': float}

        so that the runner can generically re-initialise associated
        governors and exciters via ``init_from_targets(targets)``.

        Parameters
        ----------
        x_slice : np.ndarray
            Current state vector for this generator.
        V_bus_complex : complex
            DAE-consistent bus voltage (Re + j Im in RI frame).

        Returns
        -------
        x_new : np.ndarray
            Updated state vector (unchanged for base class).
        targets : dict
            Controller targets: ``{'Tm': …, 'Efd': …, 'Vt': …, 'omega': …}``.
            Empty dict for non-generator components (base class).
        """
        return x_slice.copy(), {}

    def adjust_for_target_voltage(self, x_slice: np.ndarray,
                                   V_bus_complex: complex,
                                   V_target_mag: float,
                                   y_diag: float = 0.0,
                                   flux_update: bool = True) -> np.ndarray:
        """Adjust field-winding states to drive terminal voltage toward
        *V_target_mag*.

        Called iteratively by the DAE runner during the PV-bus voltage
        regulation loop.  The generator adjusts its own d-axis states
        (e.g. Eq\u2032, psi_d for GENROU) and q-axis states (Ed\u2032, psi_q)
        so that a subsequent network solve produces a bus voltage
        magnitude closer to *V_target_mag*.

        Non-generator components (or generators without voltage-regulation
        capability, such as GENCLS) return *x_slice* unchanged.

        Parameters
        ----------
        x_slice : np.ndarray
            Current state vector for this generator.
        V_bus_complex : complex
            Current DAE-consistent bus voltage (Re + j\u202fIm).
        V_target_mag : float
            Desired voltage magnitude (from power-flow setpoint).
        y_diag : float, optional
            Magnitude of the Y-bus diagonal entry at this generator's bus,
            used by generators to size the internal-state update against
            actual network sensitivity.  Default 0.0 means "no hint";
            implementations must remain safe in that case.

        Returns
        -------
        x_adjusted : np.ndarray
            State vector with adjusted field states.
        """
        return x_slice.copy()

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Return the Efd output value from current exciter states.
        Exciters implement this so callers need no isinstance checks."""
        return 0.0

    def efd_output_expr(self, state_offset: int) -> str:
        """Return a C++ expression string for the Efd output observable,
        given the component's base state offset in the global x[] array.
        Returns '' (empty) for non-exciter components (base class default).
        Exciters override this to emit the correct expression for their model.

        Examples:
          ESST3A: '(x[vb_offset] * x[vm_offset])'
          EXST1 / EXDC2: 'x[efd_offset]'
        """
        return ''

    def refine_q_axis(self, x_slice: np.ndarray,
                      vd: float, vq: float) -> np.ndarray:
        """One-shot closed-form update of q-axis states (Ed_p, psi_q)
        at the given terminal vd/vq.  GenRou implements; others return x_slice."""
        return x_slice.copy()

    def refine_d_axis(self, x_slice: np.ndarray, vd: float, vq: float,
                      Efd_eff: float, clamped: bool = False) -> np.ndarray:
        """Update d-axis states (Eq_p, psi_d) from Efd_eff at terminal vd/vq.
        clamped=True uses the 2×2 stator+d-axis system (for hard-clamped Efd).
        Returns new x_slice (caller applies under-relaxation)."""
        return x_slice.copy()

    def update_from_te(self, x_slice: np.ndarray, Te: float) -> tuple:
        """Set governor equilibrium states from actual electrical torque Te.
        Returns (x_slice_new, Pref_new).  Governors implement this."""
        return x_slice.copy(), self.params.get('Pref', Te)

    def refine_at_kron_voltage(self, x_slice: np.ndarray,
                               vd: float, vq: float) -> np.ndarray:
        """One-shot direct update of any states that must match the Kron-network
        operating point rather than the original power-flow terminal voltage.

        Called by Initializer.refine_kron_equilibrium() after finalize_network().
        Generators use this to correct q-axis states (Ed_p, psi_q) and d-axis
        states (Eq_p, psi_d) to the Kron-reduced terminal voltage.
        Controllers that need no correction return x_slice unchanged.

        Returns updated x_slice (caller applies under-relaxation)."""
        return x_slice.copy()

    # ------------------------------------------------------------------ #
    # Initialization helpers                                               #
    # ------------------------------------------------------------------ #

    def _init_states(self, d: dict) -> np.ndarray:
        """Convert a {state_name: value} dict to a correctly-ordered ndarray.

        Use this as the final return statement in every ``init_from_targets``
        override to guarantee the method always returns ``np.ndarray``:

            return self._init_states({'s0_y': Ipcmd0, 's1_y': Iqcmd0, ...})

        Keys that are not in ``state_schema`` are silently ignored (useful for
        passing extra targets like 'Ipcmd' or 'Iqcmd' without a separate
        conversion step).  Missing keys default to 0.0.
        """
        arr = np.zeros(len(self.state_schema))
        for i, name in enumerate(self.state_schema):
            arr[i] = d.get(name, 0.0)
        return arr

    # ------------------------------------------------------------------ #
    # Port-agnostic protocol extensions (added for IBR / current-source   #
    # generators and renewable controller chains).                         #
    # ------------------------------------------------------------------ #

    @property
    def uses_ri_frame(self) -> bool:
        """True if this generator operates directly in the RI (network) frame
        without a Park transform.  The initializer will pass RI-frame voltages
        (Vd=V_Re, Vq=V_Im) directly to refine_at_kron_voltage() instead of
        Park-transforming through delta=x[0].

        DFIG operates in RI-frame (stator aligned with network); GENROU/GENCLS
        use the Park transform.  Default is False (Park transform)."""
        return False

    @property
    def contributes_norton_admittance(self) -> bool:
        """True if this generator adds a Norton admittance shunt to the Y-bus
        (voltage-source / Norton-equivalent model, e.g. GENROU, GENCLS).
        False for pure current-source injectors (REGCA1, PVD1, VocInverter)
        that drive the network directly without an internal impedance term.

        Used by the compiler to decide whether to call
        ybus_builder.add_generator_impedance() for this component."""
        return True

    def get_associated_generator(self, comp_map: dict):
        """Return the name of the generator (REGCA1/REECA1 chain root) that
        this renewable controller is chained to, or None if not applicable.

        Used by the initializer to propagate power-flow targets (Pe, Qe,
        Vterm, Ipcmd, Iqcmd) through the WT3 / IBR controller hierarchy
        without any class-name checks.

        Override in every renewable controller subclass:
          - Reeca1  → self.params['reg']
          - Repca1  → comp_map[self.params['ree']].params['reg']
          - Wtdta1  → comp_map[self.params['ree']].params['reg']
          - Wtara1  → self.params['rego']
          - Wtpta1  → comp_map[self.params['rea']].params['rego']
          - Wttqa1  → comp_map[comp_map[self.params['rep']].params['ree']].params['reg']
        """
        return None

    def refine_current_source_init(self, x_slice: np.ndarray,
                                   targets: dict,
                                   V_bus: complex) -> np.ndarray:
        """Update initial states and the *targets* dict from the Kron-reduced
        network bus voltage *V_bus* for current-source generator models.

        Called by Initializer.refine_renewable_controllers() on every
        generator-role component.  The default is a no-op: x_slice is
        returned unchanged and targets is not modified.

        REGCA1 overrides this to recompute Ipcmd/Iqcmd accounting for the
        Low-Voltage Guard (LVG) and High-Voltage Guard (HVG) at the true
        Kron-network terminal voltage."""
        return x_slice.copy()
