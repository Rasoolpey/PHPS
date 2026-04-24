#!/usr/bin/env python3
"""Plot SystemGraph topology for a PHPS case.

This tool accepts either:
1) A scenario JSON (contains key "system"), or
2) A system JSON (contains keys like "Bus" / "components").

It generates one figure with:
- Electrical topology: buses, lines, and component-to-bus attachments
- Signal wiring: directed wires between component ports and external sources

Usage examples:
  python tools/plot_system_graph.py cases/IEEE14Bus/no_fault_phs.json
  python tools/plot_system_graph.py cases/IEEE14Bus/system_phs.json --view signal
  python tools/plot_system_graph.py cases/Kundur/system_phs.json --out outputs/Kundur/tools/system_graph.svg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# Ensure "src" imports work even when script is launched from another directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.system_graph import SystemGraph, build_system_graph  # noqa: E402


ROLE_COLORS: Dict[str, str] = {
    "generator": "#f4a261",
    "exciter": "#2a9d8f",
    "governor": "#e76f51",
    "pss": "#457b9d",
    "renewable": "#8ab17d",
    "controller": "#6d597a",
    "other": "#adb5bd",
}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_system_json_path(input_json_path: str) -> str:
    data = _load_json(input_json_path)

    # If it already looks like a system file, use it directly.
    if "Bus" in data or "components" in data:
        return input_json_path

    # Otherwise treat it as scenario JSON and resolve "system" relative to it.
    system_rel = data.get("system")
    if not isinstance(system_rel, str):
        raise ValueError(
            "Input JSON is neither a system file nor a scenario file with a 'system' key."
        )

    system_path = os.path.abspath(os.path.join(os.path.dirname(input_json_path), system_rel))
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"Resolved system JSON does not exist: {system_path}")
    return system_path


def _default_output_path(system_json_path: str) -> str:
    case_name = os.path.basename(os.path.dirname(system_json_path))
    out_dir = os.path.join(REPO_ROOT, "outputs", case_name, "tools")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "system_graph.png")


def _component_role(comp) -> str:
    role = getattr(comp, "component_role", None)
    if not isinstance(role, str) or not role:
        return "other"
    return role if role in ROLE_COLORS else "other"


def _build_electrical_graph(graph: SystemGraph) -> nx.Graph:
    g = nx.Graph()

    for bus_id in sorted(graph.buses.keys()):
        g.add_node(f"bus:{bus_id}", label=f"Bus {bus_id}", kind="bus")

    for line in graph.lines:
        b1 = f"bus:{line.bus1}"
        b2 = f"bus:{line.bus2}"
        if g.has_node(b1) and g.has_node(b2):
            g.add_edge(b1, b2, kind="line")

    for comp_name, comp in graph.components.items():
        role = _component_role(comp)
        comp_node = f"comp:{comp_name}"
        g.add_node(comp_node, label=comp_name, kind="component", role=role)

        bus_id = graph.get_component_bus(comp_name)
        bus_node = f"bus:{bus_id}" if bus_id is not None else None
        if bus_node and g.has_node(bus_node):
            g.add_edge(comp_node, bus_node, kind="attachment")

    return g


def _electrical_layout(g: nx.Graph) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}

    bus_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "bus"]
    if bus_nodes:
        bus_sub = g.subgraph(bus_nodes)
        bus_pos = nx.circular_layout(bus_sub, scale=2.0)
        for n, xy in bus_pos.items():
            pos[n] = (float(xy[0]), float(xy[1]))

        for bus_node in bus_nodes:
            attached_components = [
                nbr for nbr in g.neighbors(bus_node)
                if g.nodes[nbr].get("kind") == "component"
            ]
            attached_components.sort()
            count = len(attached_components)
            if count == 0:
                continue

            bx, by = pos[bus_node]
            radius = 0.6
            for i, comp_node in enumerate(attached_components):
                angle = (2.0 * math.pi * i) / max(count, 1)
                pos[comp_node] = (
                    bx + radius * math.cos(angle),
                    by + radius * math.sin(angle),
                )

    orphan_components = [
        n for n, d in g.nodes(data=True)
        if d.get("kind") == "component" and n not in pos
    ]
    for i, comp_node in enumerate(sorted(orphan_components)):
        pos[comp_node] = (-2.5 + 0.6 * i, -2.8)

    return pos


def _draw_electrical(ax, graph: SystemGraph):
    g = _build_electrical_graph(graph)
    pos = _electrical_layout(g)

    line_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "line"]
    attach_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "attachment"]

    nx.draw_networkx_edges(g, pos, edgelist=line_edges, width=1.8, edge_color="#495057", ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=attach_edges,
        width=1.2,
        edge_color="#868e96",
        style="dashed",
        ax=ax,
    )

    bus_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "bus"]
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=bus_nodes,
        node_size=700,
        node_color="#90caf9",
        edgecolors="#1d3557",
        linewidths=1.0,
        ax=ax,
    )

    for role, color in ROLE_COLORS.items():
        comp_nodes = [
            n for n, d in g.nodes(data=True)
            if d.get("kind") == "component" and d.get("role", "other") == role
        ]
        if not comp_nodes:
            continue
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=comp_nodes,
            node_size=520,
            node_shape="s",
            node_color=color,
            edgecolors="#343a40",
            linewidths=0.9,
            ax=ax,
        )

    labels = {n: d.get("label", n) for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title("SystemGraph Electrical Topology")
    ax.axis("off")


def _short_external_label(src: str) -> str:
    if src.startswith("BUS_"):
        return src
    if src.startswith("CONST:"):
        return src.replace("CONST:", "K=")
    if src.startswith("PARAM:"):
        return src.replace("PARAM:", "P:")
    if len(src) <= 24:
        return src
    return src[:21] + "..."


def _build_signal_graph(graph: SystemGraph) -> nx.DiGraph:
    g = nx.DiGraph()

    for comp_name, comp in graph.components.items():
        g.add_node(
            f"comp:{comp_name}",
            label=comp_name,
            kind="component",
            role=_component_role(comp),
        )

    for wire in graph.wires:
        dst_comp = wire.dst_component()
        dst_node = f"comp:{dst_comp}"
        if dst_node not in g:
            continue

        if wire.src_kind() == "comp":
            src_node = f"comp:{wire.src_component()}"
            if src_node not in g:
                continue
        else:
            src_node = f"ext:{wire.src}"
            if src_node not in g:
                g.add_node(
                    src_node,
                    label=_short_external_label(wire.src),
                    kind="external",
                    role="other",
                )

        g.add_edge(src_node, dst_node, label=wire.dst_port())

    return g


def _draw_signal(ax, graph: SystemGraph, edge_labels: bool):
    g = _build_signal_graph(graph)

    if len(g.nodes) == 0:
        ax.set_title("SystemGraph Signal Wiring")
        ax.axis("off")
        return

    pos = nx.spring_layout(g, seed=7, k=1.5 / max(math.sqrt(len(g.nodes)), 1.0), iterations=350)

    comp_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "component"]
    ext_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "external"]

    for role, color in ROLE_COLORS.items():
        role_nodes = [n for n in comp_nodes if g.nodes[n].get("role", "other") == role]
        if not role_nodes:
            continue
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=role_nodes,
            node_size=680,
            node_shape="o",
            node_color=color,
            edgecolors="#2b2d42",
            linewidths=1.0,
            ax=ax,
        )

    if ext_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=ext_nodes,
            node_size=520,
            node_shape="D",
            node_color="#dee2e6",
            edgecolors="#6c757d",
            linewidths=0.9,
            ax=ax,
        )

    nx.draw_networkx_edges(
        g,
        pos,
        arrows=True,
        arrowsize=12,
        width=1.0,
        edge_color="#495057",
        connectionstyle="arc3,rad=0.04",
        ax=ax,
    )

    labels = {n: d.get("label", n) for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)

    if edge_labels:
        edge_label_map = {(u, v): d.get("label", "") for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_label_map, font_size=6, ax=ax)

    ax.set_title("SystemGraph Signal Wiring")
    ax.axis("off")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SystemGraph from scenario/system JSON."
    )
    parser.add_argument(
        "input_json",
        help="Path to scenario JSON (with 'system') or system JSON file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path. Default: outputs/<CaseName>/tools/system_graph.png",
    )
    parser.add_argument(
        "--view",
        choices=["electrical", "signal", "both"],
        default="both",
        help="Which graph to render.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI (default: 220).",
    )
    parser.add_argument(
        "--edge-labels",
        action="store_true",
        help="Show destination port labels on signal edges.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    input_path = os.path.abspath(args.input_json)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    system_json_path = _resolve_system_json_path(input_path)
    graph = build_system_graph(system_json_path)
    graph.validate(strict_types=False)

    if args.view == "both":
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        _draw_electrical(axes[0], graph)
        _draw_signal(axes[1], graph, edge_labels=args.edge_labels)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if args.view == "electrical":
            _draw_electrical(ax, graph)
        else:
            _draw_signal(ax, graph, edge_labels=args.edge_labels)

    fig.suptitle(f"SystemGraph | {os.path.basename(system_json_path)}", fontsize=13)
    fig.tight_layout()

    out_path = os.path.abspath(args.out) if args.out else _default_output_path(system_json_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot_system_graph] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
