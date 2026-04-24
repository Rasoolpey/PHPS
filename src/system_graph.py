"""
SystemGraph — the Python-friendly intermediate model of a power system.

This module sits between the raw JSON (Layer 1) and both the Initializer
(Layer 3) and C++ Compiler (Layer 4).  It is the single authoritative
representation of:

  • Network topology  : buses, lines, loads, PV generators, slack buses
  • Component models  : instantiated PowerComponent objects keyed by name
  • Signal wiring     : explicit Wire objects connecting component ports
  • Hard constraints  : e.g. PQ setpoints, limits

The SystemGraph is built by ``build_system_graph(json_path)`` which calls
``json_compat.to_new_format()`` internally so that old-style JSON files
(with implicit ``syn``/``avr`` linkage) work transparently.

Key public API
--------------
  validate()
      Check that every required input port has a source.  Raises structured
      FrameworkError subclasses so callers can give actionable messages.

  get_source_wire(comp_name, port_name) -> Wire | None
      Return the Wire driving a given destination port, or None.

  resolve_src_to_cpp_expr(src, active_bus_map, ybus_map, port_schemas) -> str
      Translate a wire source string to a C++ expression string for the
      generated kernel.

  to_dict() / from_dict(d)
      (De-)serialise the graph to a plain dict (JSON-ready, UI-friendly).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core import PowerComponent
from src.errors import (
    ComponentNotFoundError,
    PortNotFoundError,
    PortTypeMismatchError,
    PortUnconnectedError,
    SchemaError,
    WireSourceError,
)


# ---------------------------------------------------------------------------
# Data classes for network topology elements
# ---------------------------------------------------------------------------

@dataclass
class BusNode:
    idx: int
    name: str
    Vn: float          # nominal kV
    v0: float = 1.0    # power-flow initial voltage magnitude
    a0: float = 0.0    # power-flow initial angle [rad]
    vmax: float = 1.1
    vmin: float = 0.9

    def to_dict(self) -> dict:
        return {
            "idx": self.idx, "name": self.name, "Vn": self.Vn,
            "v0": self.v0, "a0": self.a0,
            "vmax": self.vmax, "vmin": self.vmin,
        }


@dataclass
class LineData:
    idx: Any            # string or int
    bus1: int
    bus2: int
    r: float
    x: float
    b: float = 0.0
    g: float = 0.0
    tap: float = 1.0
    phi: float = 0.0
    trans: int = 0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.raw)


@dataclass
class PQLoad:
    idx: Any
    bus: int
    p0: float
    q0: float
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.raw)


@dataclass
class PVGen:
    idx: Any
    bus: int
    p0: float
    v0: float
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.raw)


@dataclass
class SlackBus:
    idx: Any
    bus: int
    v0: float
    a0: float
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.raw)


# ---------------------------------------------------------------------------
# Wire: a directed signal connection
# ---------------------------------------------------------------------------

@dataclass
class Wire:
    """A directed connection from one port to another.

    Source formats (``src``):
      BUS_<id>.<Vd|Vq|Vterm>        — network bus voltage (filled in after Kron)
      <ComponentName>.<port>          — output port of a registered component
      CONST:<float>                   — compile-time constant
      PARAM:<ComponentName>.<key>     — parameter value from a component instance

    Destination format (``dst``):
      <ComponentName>.<port>          — input port of a registered component
    """
    src: str
    dst: str

    # --- convenience helpers ------------------------------------------------

    def src_kind(self) -> str:
        """Return the broad category of the source: 'bus', 'dq', 'comp', 'const', 'param'."""
        if self.src.startswith("BUS_"):
            return "bus"
        if self.src.startswith("DQ_"):
            # dq-frame terminal voltage from a generator (Park-transformed)
            return "dq"
        if self.src.startswith("CONST:"):
            return "const"
        if self.src.startswith("PARAM:"):
            return "param"
        return "comp"

    def dst_component(self) -> str:
        return self.dst.split(".")[0]

    def dst_port(self) -> str:
        return self.dst.split(".", 1)[1]

    def src_component(self) -> Optional[str]:
        """Return the component name for 'comp' sources, else None."""
        if self.src_kind() == "comp":
            return self.src.split(".")[0]
        return None

    def src_port(self) -> Optional[str]:
        """Return the port name for 'comp' sources, else None."""
        if self.src_kind() == "comp":
            return self.src.split(".", 1)[1]
        return None

    def to_dict(self) -> dict:
        return {"from": self.src, "to": self.dst}

    @staticmethod
    def from_dict(d: dict) -> "Wire":
        return Wire(src=d["from"], dst=d["to"])


# ---------------------------------------------------------------------------
# SystemGraph
# ---------------------------------------------------------------------------

class SystemGraph:
    """Python-friendly intermediate model of the power system.

    Attributes
    ----------
    buses       : dict[int, BusNode]
    pq_loads    : list[PQLoad]
    pv_gens     : list[PVGen]
    slack_buses : list[SlackBus]
    lines       : list[LineData]
    components  : dict[str, PowerComponent]   name → instantiated component
    wires       : list[Wire]                  ordered list of explicit connections
    constraints : dict[str, Any]              hard constraints (PQ setpoints, limits)
    config      : dict[str, Any]              global config (mva_base, fn, …)
    raw_data    : dict                        the full (possibly upgraded) JSON dict
    """

    def __init__(self):
        self.buses: Dict[int, BusNode] = {}
        self.pq_loads: List[PQLoad] = []
        self.pv_gens: List[PVGen] = []
        self.slack_buses: List[SlackBus] = []
        self.lines: List[LineData] = []
        self.components: Dict[str, PowerComponent] = {}
        self.wires: List[Wire] = []
        self.constraints: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.raw_data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, strict_types: bool = False):
        """Validate the graph.  Raises FrameworkError subclasses on failure.

        Checks:
          1. Every wire destination references a known component.
          2. Every wire destination port exists on that component.
          3. Every wire source follows a recognised format.
          4. Every wire source component/port exists (for 'comp' sources).
          5. Every *required* input port of every component has at least one wire.
          6. (Optional, strict_types=True) Port kinds match across each wire.
        """
        comp_names = list(self.components.keys())

        # Build lookup: dst_key → Wire
        wired_inputs: Dict[Tuple[str, str], Wire] = {}
        for wire in self.wires:
            dst_comp = wire.dst_component()
            dst_port = wire.dst_port()

            # 1. Destination component must exist
            if dst_comp not in self.components:
                raise ComponentNotFoundError(dst_comp, "destination", comp_names)

            # 2. Destination port must exist on that component
            comp = self.components[dst_comp]
            in_ports = [p[0] for p in comp.port_schema["in"]]
            if dst_port not in in_ports:
                raise PortNotFoundError(dst_comp, dst_port, "in", in_ports)

            wired_inputs[(dst_comp, dst_port)] = wire

            # 3. Source format must be recognised
            src = wire.src
            kind = wire.src_kind()
            if kind == "bus":
                # BUS_<id>.<signal> — validate signal name
                parts = src.split(".", 1)
                if len(parts) != 2 or parts[1] not in ("Vd", "Vq", "Vterm"):
                    raise WireSourceError(src)
            elif kind == "dq":
                # DQ_<gen_name>.<signal> — Park-transformed bus voltage
                # Just check that the referenced generator exists
                rest = src[len("DQ_"):]
                gen_name = rest.split(".", 1)[0]
                if gen_name not in self.components:
                    raise ComponentNotFoundError(gen_name, "source (DQ)", comp_names)
            elif kind == "const":
                try:
                    float(src[len("CONST:"):])
                except ValueError:
                    raise WireSourceError(src)
            elif kind == "param":
                rest = src[len("PARAM:"):]
                if "." not in rest:
                    raise WireSourceError(src)
                param_comp, _param_key = rest.split(".", 1)
                if param_comp not in self.components:
                    raise ComponentNotFoundError(param_comp, "source (PARAM)", comp_names)
            elif kind == "comp":
                # 4. Source component and port must exist
                src_comp_name = wire.src_component()
                src_port_name = wire.src_port()
                if src_comp_name not in self.components:
                    raise ComponentNotFoundError(src_comp_name, "source", comp_names)
                src_comp = self.components[src_comp_name]
                out_ports = [p[0] for p in src_comp.port_schema["out"]]
                if src_port_name not in out_ports:
                    raise PortNotFoundError(src_comp_name, src_port_name, "out", out_ports)

                # 6. (Optional) port kinds
                if strict_types:
                    dst_kind = dict(
                        (p[0], p[1]) for p in comp.port_schema["in"]
                    ).get(dst_port, "signal")
                    src_kind_val = dict(
                        (p[0], p[1]) for p in src_comp.port_schema["out"]
                    ).get(src_port_name, "signal")
                    if dst_kind != src_kind_val:
                        raise PortTypeMismatchError(src, wire.dst, src_kind_val, dst_kind)

        # 5. Every required input port must have a wire
        for comp_name, comp in self.components.items():
            required = getattr(comp, "required_ports", None)
            if required is None:
                # Fall back: all input ports are required
                required = [p[0] for p in comp.port_schema["in"]]
            for port_name in required:
                if (comp_name, port_name) not in wired_inputs:
                    raise PortUnconnectedError(
                        comp_name, port_name,
                        suggestion=self._suggest_source(comp_name, port_name)
                    )

    def _suggest_source(self, comp_name: str, port_name: str) -> str:
        """Heuristic suggestion for an unconnected port."""
        if port_name in ("Vd", "Vq", "Vterm"):
            return (
                f"Wire a bus voltage: "
                f'{{\"from\": \"BUS_<id>.{port_name}\", \"to\": \"{comp_name}.{port_name}\"}}'
            )
        if port_name in ("Efd",):
            return "Wire from an exciter output: {\"from\": \"<EXCITER>.Efd\", ...}"
        if port_name in ("Tm",):
            return "Wire from a governor output: {\"from\": \"<GOV>.Tm\", ...}"
        if port_name in ("omega",):
            return "Wire from a generator output: {\"from\": \"<GEN>.omega\", ...}"
        if port_name in ("Vref",):
            return (
                f"Add a constant: "
                f'{{\"from\": \"CONST:1.0\", \"to\": \"{comp_name}.Vref\"}}'
            )
        return ""

    # ------------------------------------------------------------------
    # Port / wire lookup
    # ------------------------------------------------------------------

    def get_source_wire(self, comp_name: str, port_name: str) -> Optional[Wire]:
        """Return the Wire driving (comp_name, port_name), or None."""
        for wire in self.wires:
            if wire.dst_component() == comp_name and wire.dst_port() == port_name:
                return wire
        return None

    def get_all_wires_to(self, comp_name: str) -> List[Wire]:
        """Return all wires whose destination is comp_name."""
        return [w for w in self.wires if w.dst_component() == comp_name]

    def get_all_wires_from(self, comp_name: str) -> List[Wire]:
        """Return all wires whose source is the comp_name component."""
        return [w for w in self.wires if w.src_kind() == "comp"
                and w.src_component() == comp_name]

    # ------------------------------------------------------------------
    # C++ expression resolution (used by compiler)
    # ------------------------------------------------------------------

    def resolve_src_to_cpp_expr(
        self,
        src: str,
        active_bus_map: Dict[int, int],   # full_bus_idx → reduced_idx
        ybus_map: Dict[int, int],          # bus_id → full_bus_idx
        port_out_indices: Dict[str, Dict[str, int]],  # comp_name → {port: output_idx}
    ) -> str:
        """Translate a wire source string to a C++ expression.

        Parameters
        ----------
        src               : wire source string (see Wire docstring)
        active_bus_map    : maps full_bus_idx → Kron-reduced index
        ybus_map          : maps bus_id (int) → full_bus_idx
        port_out_indices  : maps comp_name → {port_name: output_array_index}

        Returns a C++ rvalue expression.
        """
        kind = Wire(src=src, dst="X.y").src_kind()

        if kind == "const":
            return src[len("CONST:"):]

        if kind == "param":
            rest = src[len("PARAM:"):]
            comp_name, param_key = rest.split(".", 1)
            comp = self.components.get(comp_name)
            if comp is None:
                return "0.0"
            val = comp.params.get(param_key, 0.0)
            return str(float(val))

        if kind == "dq":
            # DQ_<gen_name>.<Vd_dq|Vq_dq> → dq-frame voltage C++ variable
            rest = src[len("DQ_"):]
            gen_name, sig = rest.split(".", 1)
            if "Vd" in sig:
                return f"vd_dq_{gen_name}"
            if "Vq" in sig:
                return f"vq_dq_{gen_name}"
            return "0.0"

        if kind == "bus":
            # BUS_<id>.<Vd|Vq|Vterm>
            parts = src.split(".")
            bus_id_str = parts[0][len("BUS_"):]
            signal = parts[1]
            try:
                bus_id = int(bus_id_str)
            except ValueError:
                return "0.0"
            full_idx = ybus_map.get(bus_id)
            if full_idx is None:
                return "0.0"
            red_idx = active_bus_map.get(full_idx)
            if red_idx is None:
                return "0.0"
            if signal == "Vd":
                return f"Vd_net[{red_idx}]"
            if signal == "Vq":
                return f"Vq_net[{red_idx}]"
            if signal == "Vterm":
                return f"Vterm_net[{red_idx}]"
            return "0.0"

        # kind == "comp"
        comp_name = Wire(src=src, dst="X.y").src_component()
        port_name = Wire(src=src, dst="X.y").src_port()
        indices = port_out_indices.get(comp_name, {})
        idx = indices.get(port_name)
        if idx is None:
            return "0.0"
        return f"outputs_{comp_name}[{idx}]"

    def resolve_src_to_python_value(
        self,
        src: str,
        x0: np.ndarray,
        state_offsets: Dict[str, int],
    ) -> Any:
        """Evaluate a wire source at the current state x0 (Python-side).

        Used by the initializer for generic port value lookup.
        Returns a scalar (float or complex), or 0.0 on failure.
        """
        kind = Wire(src=src, dst="X.y").src_kind()

        if kind == "const":
            try:
                return float(src[len("CONST:"):])
            except ValueError:
                return 0.0

        if kind == "param":
            rest = src[len("PARAM:"):]
            comp_name, param_key = rest.split(".", 1)
            comp = self.components.get(comp_name)
            if comp is None:
                return 0.0
            return float(comp.params.get(param_key, 0.0))

        if kind in ("bus", "dq"):
            # Cannot resolve network/dq voltages in Python without a network solve;
            # callers that need this use _compute_network_voltages() instead.
            return None

        # kind == "comp"
        comp_name = Wire(src=src, dst="X.y").src_component()
        port_name = Wire(src=src, dst="X.y").src_port()
        comp = self.components.get(comp_name)
        if comp is None:
            return 0.0
        out_ports = [p[0] for p in comp.port_schema["out"]]
        if port_name not in out_ports:
            return 0.0
        # Return from params if it's a scalar observable stored there
        return comp.params.get(port_name, 0.0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-compatible, UI-friendly)."""
        return {
            "config":      self.config,
            "Bus":         [b.to_dict() for b in self.buses.values()],
            "PQ":          [p.to_dict() for p in self.pq_loads],
            "PV":          [p.to_dict() for p in self.pv_gens],
            "Slack":       [s.to_dict() for s in self.slack_buses],
            "Line":        [l.to_dict() for l in self.lines],
            "components":  {
                name: {
                    "type":   comp.__class__.__name__,
                    "params": comp.params,
                }
                for name, comp in self.components.items()
            },
            "connections": [w.to_dict() for w in self.wires],
            "constraints": self.constraints,
        }

    @staticmethod
    def from_dict(d: dict, component_registry: Dict[str, type]) -> "SystemGraph":
        """Deserialise from a plain dict, creating component instances via
        *component_registry* which maps type-name → PowerComponent subclass."""
        g = SystemGraph()
        g.config = d.get("config", {})
        g.constraints = d.get("constraints", {})

        for bd in d.get("Bus", []):
            b = BusNode(
                idx=bd["idx"], name=bd.get("name", str(bd["idx"])),
                Vn=bd.get("Vn", 1.0), v0=bd.get("v0", 1.0),
                a0=bd.get("a0", 0.0), vmax=bd.get("vmax", 1.1),
                vmin=bd.get("vmin", 0.9),
            )
            g.buses[b.idx] = b

        for ld in d.get("PQ", []):
            g.pq_loads.append(PQLoad(idx=ld["idx"], bus=ld["bus"],
                                     p0=ld.get("p0", 0.0), q0=ld.get("q0", 0.0),
                                     raw=ld))
        for pv in d.get("PV", []):
            g.pv_gens.append(PVGen(idx=pv["idx"], bus=pv["bus"],
                                   p0=pv.get("p0", 0.0), v0=pv.get("v0", 1.0),
                                   raw=pv))
        for sl in d.get("Slack", []):
            g.slack_buses.append(SlackBus(idx=sl["idx"], bus=sl["bus"],
                                          v0=sl.get("v0", 1.0), a0=sl.get("a0", 0.0),
                                          raw=sl))
        for ln in d.get("Line", []):
            g.lines.append(LineData(idx=ln["idx"], bus1=ln["bus1"], bus2=ln["bus2"],
                                    r=ln.get("r", 0.0), x=ln.get("x", 0.01),
                                    b=ln.get("b", 0.0), g=ln.get("g", 0.0),
                                    tap=ln.get("tap", 1.0), phi=ln.get("phi", 0.0),
                                    trans=ln.get("trans", 0), raw=ln))

        for name, cd in d.get("components", {}).items():
            type_name = cd["type"]
            cls = component_registry.get(type_name)
            if cls is None:
                from src.errors import UnknownComponentTypeError
                raise UnknownComponentTypeError(type_name, list(component_registry.keys()))
            comp = cls(name, cd["params"])
            g.components[name] = comp

        for wd in d.get("connections", []):
            g.wires.append(Wire.from_dict(wd))

        return g

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_component_bus(self, comp_name: str) -> Optional[int]:
        """Return the bus id for a component (from its 'bus' param), or None."""
        comp = self.components.get(comp_name)
        if comp is None:
            return None
        return comp.params.get("bus")

    def generator_components(self) -> List[Tuple[str, PowerComponent]]:
        """Return (name, comp) pairs for all generator-role components."""
        return [(n, c) for n, c in self.components.items()
                if c.component_role == "generator"]

    def exciter_components(self) -> List[Tuple[str, PowerComponent]]:
        return [(n, c) for n, c in self.components.items()
                if c.component_role == "exciter"]

    def governor_components(self) -> List[Tuple[str, PowerComponent]]:
        return [(n, c) for n, c in self.components.items()
                if c.component_role == "governor"]

    def pss_components(self) -> List[Tuple[str, PowerComponent]]:
        return [(n, c) for n, c in self.components.items()
                if c.component_role == "pss"]

    def get_generator_for_exciter(self, exc_name: str) -> Optional[str]:
        """Return the generator name that drives this exciter's Efd input."""
        # The exciter is wired TO the generator (exciter.Efd → gen.Efd)
        for wire in self.wires:
            if wire.src_kind() == "comp" and wire.src_component() == exc_name:
                if wire.src_port() == "Efd":
                    return wire.dst_component()
        return None

    def get_exciter_for_generator(self, gen_name: str) -> Optional[str]:
        """Return the exciter whose Efd output drives gen_name.Efd."""
        for wire in self.wires:
            if wire.dst_component() == gen_name and wire.dst_port() == "Efd":
                if wire.src_kind() == "comp":
                    return wire.src_component()
        return None

    def get_governor_for_generator(self, gen_name: str) -> Optional[str]:
        """Return the governor whose Tm output drives gen_name.Tm."""
        for wire in self.wires:
            if wire.dst_component() == gen_name and wire.dst_port() == "Tm":
                if wire.src_kind() == "comp":
                    return wire.src_component()
        return None

    def get_pss_for_exciter(self, exc_name: str) -> Optional[str]:
        """Return PSS component name wired to exciter's Vref (additive)."""
        for wire in self.wires:
            if wire.dst_component() == exc_name and wire.dst_port() == "Vref":
                # Check for additive PSS — the source would be the PSS component
                if wire.src_kind() == "comp":
                    src_comp = self.components.get(wire.src_component())
                    if src_comp and src_comp.component_role == "pss":
                        return wire.src_component()
        return None

    def __repr__(self) -> str:
        return (
            f"SystemGraph("
            f"buses={len(self.buses)}, "
            f"components={list(self.components.keys())}, "
            f"wires={len(self.wires)})"
        )


# ---------------------------------------------------------------------------
# Factory: build a SystemGraph from a JSON file
# ---------------------------------------------------------------------------

def build_system_graph(json_path: str) -> SystemGraph:
    """Parse a system JSON file (old or new format) and return a SystemGraph.

    Internally calls ``json_compat.to_new_format()`` so that both legacy
    (implicit ``syn``/``avr``) and new (explicit ``connections``) JSON files
    work transparently.

    The returned graph is *not* validated — call ``graph.validate()``
    explicitly when strict checking is desired.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"System JSON not found: {json_path}")

    with open(json_path, "r") as fh:
        raw = json.load(fh)

    # Upgrade to new format if necessary
    from src.json_compat import to_new_format
    data = to_new_format(raw)

    # Build the graph object
    from src.json_compat import instantiate_components
    g = SystemGraph()
    g.config = data.get("config", {})
    g.raw_data = data

    for bd in data.get("Bus", []):
        b = BusNode(
            idx=bd["idx"],
            name=bd.get("name", str(bd["idx"])),
            Vn=bd.get("Vn", 1.0),
            v0=bd.get("v0", 1.0),
            a0=bd.get("a0", 0.0),
            vmax=bd.get("vmax", 1.1),
            vmin=bd.get("vmin", 0.9),
        )
        g.buses[b.idx] = b

    for ld in data.get("PQ", []):
        g.pq_loads.append(PQLoad(
            idx=ld["idx"], bus=ld["bus"],
            p0=ld.get("p0", 0.0), q0=ld.get("q0", 0.0), raw=ld,
        ))
    for pv in data.get("PV", []):
        g.pv_gens.append(PVGen(
            idx=pv["idx"], bus=pv["bus"],
            p0=pv.get("p0", 0.0), v0=pv.get("v0", 1.0), raw=pv,
        ))
    for sl in data.get("Slack", []):
        g.slack_buses.append(SlackBus(
            idx=sl["idx"], bus=sl["bus"],
            v0=sl.get("v0", 1.0), a0=sl.get("a0", 0.0), raw=sl,
        ))
    for ln in data.get("Line", []):
        g.lines.append(LineData(
            idx=ln["idx"], bus1=ln["bus1"], bus2=ln["bus2"],
            r=ln.get("r", 0.0), x=ln.get("x", 0.01),
            b=ln.get("b", 0.0), g=ln.get("g", 0.0),
            tap=ln.get("tap", 1.0), phi=ln.get("phi", 0.0),
            trans=ln.get("trans", 0), raw=ln,
        ))

    g.components = instantiate_components(data)

    for wd in data.get("connections", []):
        g.wires.append(Wire.from_dict(wd))

    return g
