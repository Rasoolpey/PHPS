"""
Plot simulation results from the PH Power System Framework.

Supports two modes:
  1. Config-driven: reads plot definitions from a simulation config JSON.
  2. Standalone:    auto-detects columns and generates default panels.

Signal patterns use fnmatch syntax against CSV column names:
  - "Vterm_*"    ÔåÆ all terminal voltages
  - "*_omega"    ÔåÆ all rotor speeds
  - "*_delta"    ÔåÆ all rotor angles
  - "GENROU_1_*" ÔåÆ all states of GENROU_1

Transforms (applied per-signal):
  - "rad2deg"           ÔåÆ value ├ù 180/¤Ç
  - "pu_deviation"      ÔåÆ value ÔêÆ 1.0
  - "percent_deviation" ÔåÆ (value ÔêÆ 1.0) ├ù 100

Usage:
    # Auto-called by run_simulation.py (config-driven)
    # Standalone with a config file:
    python tools/plot_results.py simulations/smib_fault.json
    # Standalone from directory (default panels):
    python tools/plot_results.py --dir outputs/smib_fault
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
import re
import json
from fnmatch import fnmatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_columns(df, pattern: str) -> list:
    """Return column names matching an fnmatch pattern.
    Ignores the ' (unit)' suffix if present in the column name.
    """
    matches = []
    for c in df.columns:
        if c == 't':
            continue
        # Strip unit suffix for matching, e.g. "G1_Te (pu)" -> "G1_Te"
        base_name = c.split(' (')[0] if ' (' in c else c
        if fnmatch(base_name, pattern) or fnmatch(c, pattern):
            matches.append(c)
    return matches


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform is None:
        return series
    t = transform.lower()
    if t == "rad2deg":
        return series * (180.0 / np.pi)
    if t == "pu_deviation":
        return series - 1.0
    if t == "percent_deviation":
        return (series - 1.0) * 100.0
    return series


def _auto_label(col: str) -> str:
    """Derive a short legend label from a CSV column name."""
    # Strip unit suffix if present
    base_name = col.split(' (')[0] if ' (' in col else col
    for suffix in ('_delta', '_omega', '_E_q_prime', '_psi_d', '_psi_q',
                    '_E_d_prime', '_Efd_out', '_Efd', '_Vm', '_x1', '_x2', '_Tm', '_Valve', '_delta_deg'):
        if base_name.endswith(suffix):
            return base_name[:-len(suffix)]
    if base_name.startswith('Vterm_'):
        return base_name[len('Vterm_'):]
    return base_name


def _fault_times_from_events(events: list) -> list:
    """Extract unique fault times from the events list in a config."""
    times = set()
    for ev in events:
        if 't' in ev:
            times.add(ev['t'])
        if 't_start' in ev:
            times.add(ev['t_start'])
            if 't_duration' in ev:
                times.add(ev['t_start'] + ev['t_duration'])
    return sorted(times)


def _fault_times_from_cpp(out_dir: str) -> list:
    """Try to read fault event times from the generated C++ source."""
    src = os.path.join(out_dir, "system.cpp")
    if not os.path.exists(src):
        return []
    with open(src, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r"FAULT_TIMES\[.*?\]\s*=\s*\{([^}]+)\}", line)
            if m:
                return [float(t.strip()) for t in m.group(1).split(",")
                        if t.strip()]
    return []


def _find_csv():
    """Locate the CSV from the most recent run."""
    cfg_path = "tools/last_run_config.txt"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            out_dir = f.read().strip()
        candidate = os.path.join(out_dir, "simulation_results.csv")
        if os.path.exists(candidate):
            return candidate, out_dir
    if os.path.exists("simulation_results.csv"):
        return "simulation_results.csv", "."
    return None, None


# ---------------------------------------------------------------------------
# Default plot panels (used when no config is provided)
# ---------------------------------------------------------------------------

_DEFAULT_PANELS = [
    {"title": "Terminal Voltages",  "y_label": "Vterm [pu]",
     "signals": [{"pattern": "Vterm_*"}]},
    {"title": "Rotor Angles",      "y_label": "╬┤ [deg]",
     "signals": [{"pattern": "*_delta", "transform": "rad2deg"}]},
    {"title": "Rotor Speed",       "y_label": "¤ë [pu]",
     "signals": [{"pattern": "*_omega"}]},
    {"title": "Exciter Field Voltage", "y_label": "Efd [pu]",
     "signals": [{"pattern": "*_Efd_out"}]},
    {"title": "Governor State",    "y_label": "x1 [pu]",
     "signals": [{"pattern": "*_x1"}]},
]


# ---------------------------------------------------------------------------
# Core plot engine
# ---------------------------------------------------------------------------

def _eval_expr(df, expr: str) -> pd.Series:
    """Evaluate a math expression over DataFrame columns.
    
    Supports column names as variables plus numpy math functions
    (sqrt, sin, cos, abs, arctan2, pi, etc.).
    """
    # Strip unit suffixes from column names for evaluation
    ns = {c.split(' (')[0] if ' (' in c else c: df[c] for c in df.columns}
    ns.update({'sqrt': np.sqrt, 'abs': np.abs, 'sin': np.sin,
               'cos': np.cos, 'arctan2': np.arctan2, 'pi': np.pi,
               'np': np})
    return pd.eval(expr, local_dict=ns, engine='python')


def _build_figure(df, panels, fault_ts, title_prefix=""):
    t = df['t']

    active_panels = []
    for panel in panels:
        has_data = False
        for sig in panel.get("signals", []):
            if "expr" in sig:
                has_data = True
            elif _match_columns(df, sig.get("pattern", "")):
                has_data = True
        if has_data:
            active_panels.append(panel)

    if not active_panels:
        print("Warning: no matching columns for any plot panel.")
        return None

    n = len(active_panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
        
    axes[0].set_xlim(t.iloc[0], t.iloc[-1])

    dt_ms = t.diff().median() * 1000
    title = f"{title_prefix}  |  0 ÔÇô {t.iloc[-1]:.1f} s  |  dt = {dt_ms:.1f} ms"
    fig.suptitle(title.strip(), fontsize=13, fontweight='bold', y=1.002)

    cmap = plt.get_cmap('tab10')
    color_idx = 0

    def add_fault_lines(ax):
        for ft in fault_ts:
            ax.axvline(ft, color='red', lw=0.8, ls='--', alpha=0.6)

    for ax, panel in zip(axes, active_panels):
        # Disable matplotlib's offset/scientific notation on y-axis so values
        # are always shown as absolute numbers (e.g. 24.86 deg, not 1e-5+2.4862e1)
        ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        for sig in panel.get("signals", []):
            if "expr" in sig:
                try:
                    y = _eval_expr(df, sig["expr"])
                    transform = sig.get("transform", None)
                    y = _apply_transform(y, transform)
                    label = sig.get("label", sig["expr"][:40])
                    ax.plot(t, y, label=label, color=cmap(color_idx % 10), lw=1.2)
                    color_idx += 1
                except Exception as e:
                    print(f"  Warning: expr '{sig['expr']}' failed: {e}")
            else:
                matched = _match_columns(df, sig.get("pattern", ""))
                transform = sig.get("transform", None)
                for col in matched:
                    y = _apply_transform(df[col], transform)
                    label = sig.get("label", _auto_label(col))
                    ax.plot(t, y, label=label, color=cmap(color_idx % 10), lw=1.2)
                    color_idx += 1

        y_label = panel.get("y_label", "")
        if not y_label:
            # Try to infer unit from the first matched column
            for sig in panel.get("signals", []):
                if "pattern" in sig:
                    matched = _match_columns(df, sig["pattern"])
                    if matched:
                        col = matched[0]
                        if " (" in col and col.endswith(")"):
                            unit = col.split(" (")[-1][:-1]
                            y_label = f"[{unit}]"
                            break

        ax.set_ylabel(y_label)
        ax.set_title(panel.get("title", ""))
        ax.legend(fontsize=8, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.3)
        add_fault_lines(ax)

    if fault_ts:
        patch = mpatches.Patch(color='red', alpha=0.6,
                               label=f"Events @ {fault_ts} s")
        leg = axes[0].get_legend()
        existing = getattr(leg, 'legend_handles',
                           getattr(leg, 'legendHandles', []))
        axes[0].legend(handles=list(existing) + [patch],
                       fontsize=8, ncol=4, loc='upper right')

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_from_config(config: dict, csv_path: str, out_dir: str,
                     save: bool = True, show: bool = False):
    """Plot using definitions from a simulation config dict.

    The config "plots" key can be a list of panels (single figure) or the
    config can include a "figures" key ÔÇö a list of figure defs, each with
    its own "name" (filename stem) and "panels" list.  Both formats work.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns, "
          f"t = {df['t'].iloc[0]:.3f} .. {df['t'].iloc[-1]:.3f} s")

    # Only drop rows that are fully NaN or contain Inf (diverged steps).
    # A hard magnitude threshold like 1e4 incorrectly discards valid data
    # for simulations with large accumulated angles or long run times.
    numeric_cols = df.select_dtypes(include='number').columns
    df = df[~df[numeric_cols].isin([np.inf, -np.inf]).any(axis=1)].copy()
    df = df.dropna(subset=numeric_cols, how='all').copy()

    events = config.get("events", [])
    fault_ts = _fault_times_from_events(events)
    if not fault_ts:
        fault_ts = _fault_times_from_cpp(out_dir)

    label = config.get("description", os.path.basename(out_dir))

    # Support "figures" (multiple output files) or "plots" (single figure).
    figures_def = config.get("figures", None)
    if figures_def is None:
        figures_def = [{"name": "simulation_plot",
                        "panels": config.get("plots", _DEFAULT_PANELS)}]

    saved_paths = []
    for fdef in figures_def:
        fname = fdef.get("name", "simulation_plot")
        panels = fdef.get("panels", [])
        t_limit = fdef.get('t_limit', None)
        df_fig = df[(df['t'] >= t_limit[0]) & (df['t'] <= t_limit[1])] if t_limit else df
        fig = _build_figure(df_fig, panels, fault_ts, title_prefix=label)
        if fig is None:
            continue
        if save:
            out_path = os.path.join(out_dir, f"{fname}.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved ÔåÆ {os.path.abspath(out_path)}")
            saved_paths.append(os.path.abspath(out_path))
        if show:
            plt.show()
        plt.close(fig)

    return saved_paths[0] if saved_paths else None


def plot(csv_path=None, out_dir=None, save=True, show=False, case_label=None):
    """Standalone plot with auto-detected default panels (backward compat)."""
    if csv_path is None:
        csv_path, out_dir = _find_csv()
    if csv_path is None or not os.path.exists(csv_path):
        print("Error: simulation_results.csv not found. Run the simulation first.")
        return

    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."

    config = {
        "plots": _DEFAULT_PANELS,
        "events": [],
        "description": case_label or os.path.basename(out_dir),
    }
    return plot_from_config(config, csv_path, out_dir, save=save, show=show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot PH Power System simulation results")
    parser.add_argument("config", nargs="?", default=None,
                        help="Simulation config JSON (reads plot definitions)")
    parser.add_argument("--csv", default=None,
                        help="Path to simulation_results.csv")
    parser.add_argument("--dir", default=None,
                        help="Output directory (reads simulation_results.csv)")
    parser.add_argument("--show", action="store_true",
                        help="Display interactive window")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
        out_dir = cfg.get("output", {}).get("directory", args.dir or ".")
        csv = args.csv or os.path.join(out_dir, "simulation_results.csv")
        plot_from_config(cfg, csv, out_dir, show=args.show)
    else:
        csv = args.csv
        out_dir = args.dir
        if csv is None and out_dir is not None:
            csv = os.path.join(out_dir, "simulation_results.csv")
        case_label = os.path.basename(out_dir) if out_dir else None
        plot(csv_path=csv, out_dir=out_dir, show=args.show,
             case_label=case_label)
