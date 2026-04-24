"""
Run a power system simulation from a simulation config JSON file.

Usage:
    python tools/run_simulation.py cases/SMIB/line_trip.json
    python tools/run_simulation.py cases/SMIB/bus_fault.json
    python tools/run_simulation.py cases/SMIB/no_fault.json --no-plot
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.runner import SimulationRunner


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def resolve_path(path: str, config_dir: str) -> str:
    """Resolve a path relative to the project root (cwd), falling back to
    config file directory."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    alt = os.path.join(config_dir, path)
    if os.path.exists(alt):
        return alt
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Power System Simulation from a config JSON")
    parser.add_argument("config", type=str,
                        help="Path to simulation config JSON "
                             "(e.g. simulations/smib_fault.json)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip automatic plotting after simulation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    system_path = resolve_path(cfg["system"], config_dir)
    solver = cfg.get("solver", {})
    output = cfg.get("output", {})
    events = cfg.get("events", [])

    dt       = solver.get("dt", 0.001)
    duration = solver.get("duration", 10.0)
    method   = solver.get("method", "rk4")
    out_dir  = output.get("directory", None)

    print(f"[Config] {cfg.get('description', args.config)}")
    print(f"[Config] System : {system_path}")
    print(f"[Config] Solver : {method}, dt={dt}, T={duration} s")
    print(f"[Config] Output : {out_dir or '(auto)'}")
    if events:
        print(f"[Config] Events : {len(events)}")

    runner = SimulationRunner(system_path, output_dir=out_dir, events=events)

    # Warm-start from a previous simulation's CSV
    warm = cfg.get("warm_start", None)
    if warm:
        ws_csv = resolve_path(warm["csv"], config_dir)
        ws_t = warm.get("t", None)
        ws_unwrap = warm.get("delta_unwrap", {})
        ws_omega = warm.get("omega_override", None)
        print(f"[Config] Warm-start: {ws_csv} @ t={ws_t}")
        runner.set_warm_start(ws_csv, t=ws_t, delta_unwrap=ws_unwrap,
                              omega_override=ws_omega)

    # Auto-subscribe based on figures
    figures = cfg.get("figures", [])
    if figures:
        for fig in figures:
            for panel in fig.get("panels", []):
                for sig in panel.get("signals", []):
                    pattern = sig.get("pattern", "")
                    if pattern:
                        runner.subscribe_pattern(pattern)
                    expr = sig.get("expr", "")
                    if expr:
                        # Extract variable names from expression
                        import re
                        vars_in_expr = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                        for var in vars_in_expr:
                            if var not in ('sqrt', 'sin', 'cos', 'abs', 'arctan2', 'pi', 'np'):
                                runner.subscribe_pattern(var)
        
    runner.build(dt=dt, duration=duration, method=method)
    runner.run()

    if not args.no_plot:
        from tools.plot_results import plot_from_config
        csv_path = os.path.join(runner.output_dir, "simulation_results.csv")
        plot_from_config(cfg, csv_path, runner.output_dir)
