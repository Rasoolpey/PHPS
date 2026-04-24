"""
Run a DAE (Dirac) power system simulation from a scenario JSON file.

Uses the exact same JSON format as the ODE pipeline (run_simulation.py):
same system file, same events, same plot configuration.  The only
difference is the solver backend: DAE implicit solvers (BDF-1, IDA,
or implicit midpoint) instead of ODE methods like RK4.

Usage:
    python tools/run_dae_simulation.py cases/SMIB/bus_fault_dae.json
    python tools/run_dae_simulation.py cases/SMIB/bus_fault_dae.json --no-plot

Config notes:
    solver.method can be either:
      - a single solver name: "ida"
      - a list for smoke tests: ["ida", "bdf1", "midpoint"]
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dirac import DiracRunner
from src.dirac.dae_compiler import get_dae_solver_label, normalize_dae_solver_name


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


def normalize_solver_sequence(method_raw):
    """Normalize solver config into an ordered list of DAE solver names."""
    if isinstance(method_raw, list):
        if not method_raw:
            raise ValueError("solver.method list cannot be empty")
        methods = []
        for method in method_raw:
            canonical = normalize_dae_solver_name(method)
            if canonical not in methods:
                methods.append(canonical)
        return methods
    return [normalize_dae_solver_name(method_raw)]


def solver_output_directory(base_dir: str, config_name: str,
                            solver: str, multi_solver: bool) -> str:
    """Choose per-run output directory for single vs multi-solver runs."""
    if not multi_solver:
        return base_dir
    if base_dir:
        return f"{base_dir}_{solver}"
    stem = os.path.splitext(os.path.basename(config_name))[0]
    return os.path.join("outputs", f"{stem}_{solver}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DAE (Dirac) Power System Simulation from a config JSON")
    parser.add_argument("config", type=str,
                        help="Path to simulation config JSON "
                             "(e.g. cases/SMIB/bus_fault_dae.json)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip automatic plotting after simulation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    system_path = resolve_path(cfg["system"], config_dir)
    solver = cfg.get("solver", {})
    output = cfg.get("output", {})
    events = cfg.get("events", [])

    dt       = solver.get("dt", 0.0005)
    duration = solver.get("duration", 10.0)
    method_raw = solver.get("method", "bdf1")
    out_dir = output.get("directory", None)

    try:
        methods = normalize_solver_sequence(method_raw)
    except ValueError as exc:
        print(f"[DAE Config] ERROR: {exc}")
        sys.exit(2)

    multi_solver = len(methods) > 1
    print(f"[DAE Config] {cfg.get('description', args.config)}")
    print(f"[DAE Config] System : {system_path}")
    if multi_solver:
        labels = [f"{m} ({get_dae_solver_label(m)})" for m in methods]
        print(f"[DAE Config] Solvers: {', '.join(labels)}")
    else:
        solver_label = get_dae_solver_label(methods[0])
        print(f"[DAE Config] Solver : {solver_label}, dt={dt}, T={duration} s")
    print(f"[DAE Config] Output : {out_dir or '(auto)'}")
    if events:
        print(f"[DAE Config] Events : {len(events)}")

    omega_perturb = solver.get("omega0_perturb", None)
    run_results = []
    run_failures = []

    for method in methods:
        run_out_dir = solver_output_directory(
            out_dir, args.config, method, multi_solver)
        solver_label = get_dae_solver_label(method)
        print(f"[DAE Run] Starting {method} ({solver_label})...")

        try:
            runner = DiracRunner(system_path, output_dir=run_out_dir,
                                 events=events if events else None)
            if omega_perturb is not None:
                runner._omega0_perturb = float(omega_perturb)
            runner.build(dt=dt, duration=duration, solver=method)
            csv_path = runner.run()
            run_results.append((method, csv_path, runner.output_dir))

            if not args.no_plot:
                from tools.plot_results import plot_from_config
                plot_from_config(cfg, csv_path, runner.output_dir)

        except Exception as exc:
            run_failures.append((method, str(exc)))
            print(f"[DAE Run] ERROR ({method}): {exc}")
            if not multi_solver:
                raise

    if multi_solver:
        print("[DAE Smoke] Summary")
        for method, csv_path, out_path in run_results:
            print(f"  PASS {method}: {csv_path} (dir: {out_path})")
        for method, err in run_failures:
            print(f"  FAIL {method}: {err}")

    if run_failures:
        sys.exit(1)
