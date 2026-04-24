"""
Backward-compatibility shim for system JSON files.

Old-format JSON uses model-type keys at the top level (``GENROU``, ``ESST3A``,
``TGOV1``, …) with implicit linkage via ``syn`` / ``avr`` fields.

New-format JSON uses a flat ``components`` dict (component name → type + params)
and an explicit ``connections`` list of ``{from, to}`` wires.

``to_new_format(raw_data)`` upgrades old JSON in-memory.  The result is
exactly equivalent to writing a new-format JSON by hand, so no information
is lost.  The original JSON file is **never modified on disk**.

``instantiate_components(data)`` creates live PowerComponent objects from the
(possibly upgraded) ``components`` dict using the global component registry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from src.core import PowerComponent


# ---------------------------------------------------------------------------
# Component registry: type-name → class
# ---------------------------------------------------------------------------

def _build_registry() -> Dict[str, type]:
    from src.components.generators.gencls import GenCls
    from src.components.generators.genrou import GenRou
    from src.components.generators.genrou_phs import GenRouPHS
    from src.components.exciters.avr import AvrType1
    from src.components.exciters.esst3a import Esst3a
    from src.components.exciters.exst1 import Exst1
    from src.components.exciters.exdc2 import Exdc2
    from src.components.exciters.exdc2_phs import Exdc2PHS
    from src.components.exciters.esst3a_phs import Esst3aPHS
    from src.components.exciters.exst1_phs import Exst1PHS
    from src.components.exciters.ieeex1 import Ieeex1
    from src.components.exciters.ieeex1_phs import Ieeex1PHS
    from src.components.exciters.ida_pbc import IdaPbcExciter
    from src.components.governors.tgov1 import Tgov1
    from src.components.governors.tgov1_phs import Tgov1PHS
    from src.components.governors.ieeeg1 import Ieeeg1
    from src.components.governors.ieeeg1_phs import Ieeeg1PHS
    from src.components.pss.st2cut import St2cut
    from src.components.pss.st2cut_phs import St2cutPHS
    from src.components.pss.ieeest import Ieeest
    from src.components.pss.ieeest_phs import IeeestPHS
    from src.components.network.pi_line import PiLine
    from src.components.network.transformer import Transformer2W
    from src.components.renewables.regca1 import Regca1
    from src.components.renewables.reeca1 import Reeca1
    from src.components.renewables.repca1 import Repca1
    from src.components.renewables.wtdta1 import Wtdta1
    from src.components.renewables.wtara1 import Wtara1
    from src.components.renewables.wtpta1 import Wtpta1
    from src.components.renewables.wttqa1 import Wttqa1
    from src.components.renewables.pvd1 import Pvd1
    from src.components.renewables.dgprct1 import Dgprct1
    from src.components.renewables.voc_inverter import VocInverter
    from src.components.renewables.dfig import Dfig
    from src.components.renewables.dfig_rsc import DfigRsc
    from src.components.renewables.dfig_dclink import DfigDclink
    from src.components.renewables.dfig_gsc import DfigGsc
    from src.components.renewables.wind_aero import WindAero
    from src.components.renewables.wind_file_reader import WindFileReader
    from src.components.renewables.dfig_drivetrain import DfigDrivetrain
    from src.components.renewables.dfig_phs import DfigPHS
    from src.components.renewables.dfig_rsc_phs import DfigRscPHS
    from src.components.renewables.dfig_dclink_phs import DfigDclinkPHS
    from src.components.renewables.dfig_gsc_phs import DfigGscPHS
    from src.components.renewables.dfig_rsc_gfm_phs import DfigRscGfmPHS
    from src.components.renewables.dfig_rsc_mpc_phs import DfigRscMpcPHS
    from src.components.renewables.dfig_rsc_mpc_freq_phs import DfigRscMpcFreqPHS
    from src.components.renewables.dfig_rsc_pbqp_phs import DfigRscPbqpPHS
    from src.components.renewables.dfig_gsc_gfm_phs import DfigGscGfmPHS
    from src.components.renewables.dfig_drivetrain_phs import DfigDrivetrainPHS
    from src.components.control.agc import Agc
    from src.components.measurements.busfreq import BusFreq
    from src.components.measurements.pmu import Pmu
    from src.components.loads.complex_load_phs import ComplexLoadPHS

    return {
        "GENCLS": GenCls,
        "GenCls": GenCls,
        "GENROU": GenRou,
        "GenRou": GenRou,
        "GENROU_PHS": GenRouPHS,
        "GenRouPHS": GenRouPHS,
        "IEEEX1": Ieeex1,
        "Ieeex1": Ieeex1,
        "IEEEX1_PHS": Ieeex1PHS,
        "Ieeex1PHS": Ieeex1PHS,
        "IDAPBC": IdaPbcExciter,
        "IdaPbcExciter": IdaPbcExciter,
        "AvrType1": AvrType1,
        "ESST3A": Esst3a,
        "Esst3a": Esst3a,
        "EXST1": Exst1,
        "Exst1": Exst1,
        "EXDC2": Exdc2,
        "Exdc2": Exdc2,
        "EXDC2_PHS": Exdc2PHS,
        "Exdc2PHS": Exdc2PHS,
        "TGOV1": Tgov1,
        "Tgov1": Tgov1,
        "TGOV1_PHS": Tgov1PHS,
        "Tgov1PHS": Tgov1PHS,
        "IEEEG1": Ieeeg1,
        "Ieeeg1": Ieeeg1,
        "IEEEG1_PHS": Ieeeg1PHS,
        "Ieeeg1PHS": Ieeeg1PHS,
        "ST2CUT": St2cut,
        "St2cut": St2cut,
        "ST2CUT_PHS": St2cutPHS,
        "St2cutPHS": St2cutPHS,
        "IEEEST": Ieeest,
        "Ieeest": Ieeest,
        "IEEEST_PHS": IeeestPHS,
        "IeeestPHS": IeeestPHS,
        "ESST3A_PHS": Esst3aPHS,
        "Esst3aPHS": Esst3aPHS,
        "EXST1_PHS": Exst1PHS,
        "Exst1PHS": Exst1PHS,
        # Network elements (PH dynamic models)
        "PILINE": PiLine,
        "PiLine": PiLine,
        "XFMR2W": Transformer2W,
        "Transformer2W": Transformer2W,
        "REGCA1": Regca1,
        "Regca1": Regca1,
        "REECA1": Reeca1,
        "Reeca1": Reeca1,
        "REPCA1": Repca1,
        "Repca1": Repca1,
        "WTDTA1": Wtdta1,
        "Wtdta1": Wtdta1,
        "WTARA1": Wtara1,
        "Wtara1": Wtara1,
        "WTPTA1": Wtpta1,
        "Wtpta1": Wtpta1,
        "WTTQA1": Wttqa1,
        "Wttqa1": Wttqa1,
        "PVD1": Pvd1,
        "Pvd1": Pvd1,
        "DGPRCT1": Dgprct1,
        "Dgprct1": Dgprct1,
        "VOC_INVERTER": VocInverter,
        "VocInverter": VocInverter,
        "DFIG": Dfig,
        "Dfig": Dfig,
        "DFIG_RSC": DfigRsc,
        "DfigRsc": DfigRsc,
        "DFIG_DCLINK": DfigDclink,
        "DfigDclink": DfigDclink,
        "DFIG_GSC": DfigGsc,
        "DfigGsc": DfigGsc,
        "WIND_AERO": WindAero,
        "WindAero": WindAero,
        "WIND_FILE": WindFileReader,
        "DFIG_DRIVETRAIN": DfigDrivetrain,
        "DfigDrivetrain": DfigDrivetrain,
        "DFIG_PHS": DfigPHS,
        "DfigPHS": DfigPHS,
        "DFIG_RSC_PHS": DfigRscPHS,
        "DfigRscPHS": DfigRscPHS,
        "DFIG_DCLINK_PHS": DfigDclinkPHS,
        "DfigDclinkPHS": DfigDclinkPHS,
        "DFIG_GSC_PHS": DfigGscPHS,
        "DfigGscPHS": DfigGscPHS,
        "DFIG_RSC_GFM_PHS": DfigRscGfmPHS,
        "DfigRscGfmPHS": DfigRscGfmPHS,
        "DFIG_RSC_MPC_PHS": DfigRscMpcPHS,      # multi-horizon Lyapunov-MPC
        "DfigRscMpcPHS": DfigRscMpcPHS,
        "DFIG_RSC_MPC_FREQ_PHS": DfigRscMpcFreqPHS, # frequency-based MPC
        "DfigRscMpcFreqPHS": DfigRscMpcFreqPHS,
        "DFIG_RSC_PBQP_PHS": DfigRscPbqpPHS,    # one-step passivity-based QP
        "DfigRscPbqpPHS": DfigRscPbqpPHS,
        "DFIG_GSC_GFM_PHS": DfigGscGfmPHS,
        "DfigGscGfmPHS": DfigGscGfmPHS,
        "DFIG_DRIVETRAIN_PHS": DfigDrivetrainPHS,
        "DfigDrivetrainPHS": DfigDrivetrainPHS,
        "AGC": Agc,
        "Agc": Agc,
        "BusFreq": BusFreq,
        "BUSFREQ": BusFreq,
        "PMU": Pmu,
        "Pmu": Pmu,
        "ComplexLoad": ComplexLoadPHS,
        "COMPLEXLOAD": ComplexLoadPHS,
        "ComplexLoadPHS": ComplexLoadPHS,
    }


COMPONENT_REGISTRY: Dict[str, type] = {}


def _registry() -> Dict[str, type]:
    global COMPONENT_REGISTRY
    if not COMPONENT_REGISTRY:
        COMPONENT_REGISTRY = _build_registry()
    return COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Parameter normalisation helpers (match compiler._map_* logic)
# ---------------------------------------------------------------------------

def _normalise_genrou_params(params: Dict[str, Any], mva_base: float, fn: float = 60.0) -> Dict[str, Any]:
    mapping = {
        "xd1": "xd_prime", "xq1": "xq_prime",
        "xd2": "xd_double_prime", "xq2": "xq_double_prime",
        "Td10": "Td0_prime", "Tq10": "Tq0_prime",
        "Td20": "Td0_double_prime", "Tq20": "Tq0_double_prime",
    }
    out = dict(params)

    # --- Detect already-normalised params ---
    # If both the old-style alias (xd1) and the new-style full name (xd_prime)
    # are present and xd_prime ≈ xd1 * (mva_base/Sn) within 1 %, the params
    # have already been converted from machine base to system base (e.g. by a
    # previous call to this function or by an external JSON generator).
    # In that case we must NOT scale again — just ensure all new-style keys
    # exist, re-run the sanity bounds, recompute Kfd_scale, and return.
    _Sn = float(params.get("Sn", mva_base))
    _Z  = mva_base / _Sn
    _already_normalised = False
    if abs(_Z - 1.0) > 0.01 and "xd1" in params and "xd_prime" in params:
        _xd1 = float(params["xd1"])
        _xdp = float(params["xd_prime"])
        if _xd1 > 0 and abs(_xdp - _xd1 * _Z) / abs(_xdp) < 0.01:
            _already_normalised = True

    if _already_normalised:
        # Params are already in system base.  Only fill missing new-style keys
        # from their old-style aliases (no scaling), re-run sanity bounds, and
        # recompute Kfd_scale.
        for k_old, k_new in mapping.items():
            if k_old in params and k_new not in params:
                out[k_new] = params[k_old]
        out["omega_b"] = f"2.0 * M_PI * {fn}"
        if "M" in params and "H" not in params:
            out["H"] = params["M"] / 2.0
        xd_pp = float(out.get("xd_double_prime", 0.2))
        xd_p  = float(out.get("xd_prime",        0.3))
        xq_pp = float(out.get("xq_double_prime", 0.2))
        xq_p  = float(out.get("xq_prime",        0.55))
        xl    = float(out.get("xl",               0.15))
        if xd_pp <= xl:
            out["xd_double_prime"] = xl + 0.5 * (xd_p - xl)
        if xq_pp <= xl:
            out["xq_double_prime"] = xl + 0.5 * (xq_p - xl)
        xd_val  = float(out.get("xd",      1.8))
        xl_val  = float(out.get("xl",      0.15))
        xd1_val = float(out.get("xd_prime", 0.3))
        Xad = xd_val - xl_val
        if Xad > 1e-6 and (xd1_val - xl_val) > 1e-6:
            denom = Xad - (xd1_val - xl_val)
            if abs(denom) > 1e-6:
                Xfl = (Xad * (xd1_val - xl_val)) / denom
                out["Kfd_scale"] = (Xad + Xfl) / Xad
            else:
                out["Kfd_scale"] = 1.0
        else:
            out["Kfd_scale"] = 1.0
        return out

    # --- First-time normalisation: rename old-style aliases then scale ---
    for k_old, k_new in mapping.items():
        if k_old in params:
            out[k_new] = params[k_old]
    if "M" in params:
        out["H"] = params["M"] / 2.0
    out["omega_b"] = f"2.0 * M_PI * {fn}"

    Sn = float(out.get("Sn", mva_base))
    Z_scale = mva_base / Sn
    M_scale = Sn / mva_base

    for key in ("ra", "xd", "xq", "xd_prime", "xq_prime",
                "xd_double_prime", "xq_double_prime", "xl"):
        if key in out:
            out[key] = float(out[key]) * Z_scale
            
    if "M" in out:
        out["M"] = float(out["M"]) * M_scale
        out["H"] = out["M"] / 2.0
    elif "H" in out:
        out["H"] = float(out["H"]) * M_scale

    D_raw = float(out.get("D", 0.0))
    D_scaled = D_raw * M_scale
    out["D"] = max(D_scaled, 2.0 * M_scale)

    xd_pp = out.get("xd_double_prime", out.get("xd2", 0.2))
    xq_pp = out.get("xq_double_prime", out.get("xq2", 0.2))
    xd_p  = out.get("xd_prime",        out.get("xd1", 0.3))
    xq_p  = out.get("xq_prime",        out.get("xq1", 0.55))
    xl    = out.get("xl", 0.15)
    if xd_pp <= xl:
        out["xd_double_prime"] = xl + 0.5 * (xd_p - xl)
    if xq_pp <= xl:
        out["xq_double_prime"] = xl + 0.5 * (xq_p - xl)

    xd_val  = out.get("xd", 1.8)
    xl_val  = out.get("xl", 0.15)
    xd1_val = out.get("xd_prime", out.get("xd1", 0.3))
    Xad = xd_val - xl_val
    if Xad > 1e-6 and (xd1_val - xl_val) > 1e-6:
        denom = Xad - (xd1_val - xl_val)
        if abs(denom) > 1e-6:
            Xfl = (Xad * (xd1_val - xl_val)) / denom
            out["Kfd_scale"] = (Xad + Xfl) / Xad
        else:
            out["Kfd_scale"] = 1.0
    else:
        out["Kfd_scale"] = 1.0
    return out


def _normalise_gencls_params(params: Dict[str, Any], mva_base: float, fn: float = 60.0) -> Dict[str, Any]:
    out = dict(params)
    out["omega_b"] = f"2.0 * M_PI * {fn}"
    Sn = float(out.get("Sn", mva_base))
    Z_scale = mva_base / Sn
    M_scale = Sn / mva_base
    
    for key in ("ra", "xd1"):
        if key in out:
            out[key] = float(out[key]) * Z_scale
            
    if "M" in out:
        out["M"] = float(out["M"]) * M_scale
        out["H"] = out["M"] / 2.0
    elif "H" in out:
        out["H"] = float(out["H"]) * M_scale
        
    D_raw = float(out.get("D", 0.0))
    D_scaled = D_raw * M_scale
    out["D"] = max(D_scaled, 2.0 * M_scale)
    
    if "E_p" not in out:
        out["E_p"] = 1.0  # Default value, should be calculated from power flow
        
    return out


def _normalise_esst3a_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.setdefault("TC", 0.0)
    out.setdefault("TB", 0.0)
    out.setdefault("Efd_max", 5.0)
    out.setdefault("Efd_min", -1.0)
    return out


def _normalise_exst1_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.setdefault("TC", 0.0)
    out.setdefault("TB", 0.0)
    out.setdefault("VIMAX", 0.5)
    out.setdefault("VIMIN", -0.5)
    return out


def _normalise_exdc2_params(params: Dict[str, Any]) -> Dict[str, Any]:
    from src.components.exciters.exdc2 import compute_saturation_coeffs
    out = dict(params)
    out["KF"] = out.pop("KF1", out.get("KF", 0.0754))
    out.setdefault("TF1", 1.0)
    out.setdefault("TR", 0.02)
    out.setdefault("TA", 0.02)
    out.setdefault("TE", 0.83)
    out.setdefault("KA", 20.0)
    out.setdefault("KE", 1.0)
    out.setdefault("VRMAX", 5.0)
    out.setdefault("VRMIN", -5.0)
    E1  = float(out.get("E1",  0.0))
    SE1 = float(out.get("SE1", 0.0))
    E2  = float(out.get("E2",  1.0))
    SE2 = float(out.get("SE2", 1.0))
    sat_A, sat_B = compute_saturation_coeffs(E1, SE1, E2, SE2)
    out["SAT_A"] = sat_A
    out["SAT_B"] = sat_B
    return out


def _normalise_ieeex1_params(params: Dict[str, Any]) -> Dict[str, Any]:
    from src.components.exciters.exdc2 import compute_saturation_coeffs
    out = dict(params)
    out.setdefault("TC", 0.0)
    out.setdefault("TB", 0.0)
    out.setdefault("TR", 0.01)
    out.setdefault("TA", 0.04)
    out.setdefault("TE", 0.8)
    out.setdefault("KA", 40.0)
    out.setdefault("KE", 1.0)
    out.setdefault("KF1", out.get("KF1", 0.03))
    out.setdefault("TF1", 1.0)
    out.setdefault("VRMAX", 7.3)
    out.setdefault("VRMIN", -7.3)
    E1  = float(out.get("E1",  0.0))
    SE1 = float(out.get("SE1", 0.0))
    E2  = float(out.get("E2",  1.0))
    SE2 = float(out.get("SE2", 1.0))
    sat_A, sat_B = compute_saturation_coeffs(E1, SE1, E2, SE2)
    out["SAT_A"] = sat_A
    out["SAT_B"] = sat_B
    return out


def _normalise_ieeest_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.setdefault("MODE", 1)
    for tc in ("A1", "A2", "A4", "A6", "T2", "T4"):
        val = float(out.get(tc, 0.0))
        if 0.0 < val < 1e-4:
            out[tc] = 1e-4
    return out


def _normalise_st2cut_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    for tc in ("T1", "T2", "T4", "T6", "T8", "T10"):
        val = float(out.get(tc, 0.0))
        if 0.0 < val < 1e-4:
            out[tc] = 1e-4
    return out


# ---------------------------------------------------------------------------
# Old-format → new-format upgrade
# ---------------------------------------------------------------------------

def to_new_format(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an old-style system JSON dict to the new explicit-wiring format.

    If the JSON already contains a ``components`` key AND a ``connections``
    key it is assumed to be in the new format and is returned as-is (after a
    shallow copy so the caller's dict is not mutated).

    Otherwise the function:
      1. Reads all GENROU/GENCLS entries → components dict (type + params).
      2. Reads all ESST3A/EXST1/EXDC2/IEEEX1 entries → components dict.
      3. Reads all TGOV1/IEEEG1 entries → components dict.
      4. Reads all ST2CUT/IEEEST PSS entries → components dict.
      5. Derives explicit ``connections`` wires from the implicit ``syn``/``avr``
         linkage fields and from the bus assignments.
      6. Returns a new dict containing the topology fields (Bus, PQ, PV,
         Slack, Line, Toggler, BusFault, config) plus the new ``components``
         and ``connections`` keys.
    """
    # Already in new format?
    if "components" in raw and "connections" in raw:
        return dict(raw)

    mva_base = float(raw.get("config", {}).get("mva_base", 100.0))
    fn = float(raw.get("config", {}).get("fn", 60.0))

    components: Dict[str, Dict[str, Any]] = {}
    connections: List[Dict[str, str]] = []

    # Track which components we've registered for cross-referencing
    gen_ids: Dict[str, int] = {}    # comp_name → bus_id
    exc_syn: Dict[str, str] = {}    # exc_name → gen_name
    pss_avr: Dict[str, str] = {}    # pss_name → exc_name

    # Build a map of bus_id -> p0 from Slack and PV
    bus_p0: Dict[int, float] = {}
    for entry in raw.get("Slack", []):
        bus_p0[entry["bus"]] = entry.get("p0", 0.5)
    for entry in raw.get("PV", []):
        bus_p0[entry["bus"]] = entry.get("p0", 0.5)

    # ------------------------------------------------------------------
    # 1. Generators
    # ------------------------------------------------------------------
    for model in ("GENCLS", "GENROU"):
        for entry in raw.get(model, []):
            name = entry["idx"]
            bus_id = entry.get("bus")
            gen_ids[name] = bus_id

            if model == "GENCLS":
                params = _normalise_gencls_params(entry, mva_base, fn)
            else:
                params = _normalise_genrou_params(entry, mva_base, fn)
            params["bus"] = bus_id
            params['_params_normalized'] = True

            components[name] = {"type": model, "params": params}

            # Generator ← Bus voltage wires (Vd, Vq from the network)
            connections.append({"from": f"BUS_{bus_id}.Vd",    "to": f"{name}.Vd"})
            connections.append({"from": f"BUS_{bus_id}.Vq",    "to": f"{name}.Vq"})

            # Default fallback wires for Tm and Efd (overridden below by governor/exciter)
            connections.append({"from": f"PARAM:{name}.Tm0",   "to": f"{name}.Tm"})
            if model == "GENROU":
                connections.append({"from": f"PARAM:{name}.Efd0", "to": f"{name}.Efd"})

    # ------------------------------------------------------------------
    # 2. Exciters
    # ------------------------------------------------------------------
    for model in ("ESST3A", "EXST1", "EXDC2", "IEEEX1"):
        for entry in raw.get(model, []):
            name   = entry["idx"]
            syn_id = entry.get("syn")
            if syn_id is None or syn_id not in components:
                print(f"  [json_compat] Warning: Exciter {name} links to "
                      f"unknown generator '{syn_id}'. Skipping.")
                continue

            if model == "ESST3A":
                params = _normalise_esst3a_params(entry)
            elif model == "EXST1":
                params = _normalise_exst1_params(entry)
            elif model == "EXDC2":
                params = _normalise_exdc2_params(entry)
            elif model == "IEEEX1":
                params = _normalise_ieeex1_params(entry)
            else:
                params = dict(entry)

            # Carry the syn link so the initializer can still find the generator
            params["syn"] = syn_id

            components[name] = {"type": model, "params": params}
            exc_syn[name] = syn_id

            bus_id = gen_ids.get(syn_id)

            # Exciter receives terminal voltage from bus
            if bus_id is not None:
                connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})

            # ESST3A: dq-frame voltages and currents from generator
            if model in ("ESST3A",):
                connections.append({"from": f"{syn_id}.Vd_dq", "to": f"{name}.Vd"})
                connections.append({"from": f"{syn_id}.Vq_dq", "to": f"{name}.Vq"})
                connections.append({"from": f"{syn_id}.id_dq", "to": f"{name}.id_dq"})
                connections.append({"from": f"{syn_id}.iq_dq", "to": f"{name}.iq_dq"})

            # IEEEX1: speed compensation (Efd = Vp * omega)
            if model in ("IEEEX1",):
                connections.append({"from": f"{syn_id}.omega", "to": f"{name}.omega"})

            # Exciter Vref default (overridden later if PSS present)
            vref = float(entry.get("v0", 1.0))
            connections.append({"from": f"CONST:{vref}", "to": f"{name}.Vref"})

            # Exciter → generator Efd (override the CONST default)
            _remove_const_wire(connections, f"{syn_id}.Efd")
            connections.append({"from": f"{name}.Efd", "to": f"{syn_id}.Efd"})

    # ------------------------------------------------------------------
    # 3. Governors
    # ------------------------------------------------------------------
    for model in ("TGOV1", "IEEEG1"):
        for entry in raw.get(model, []):
            name   = entry["idx"]
            syn_id = entry.get("syn")
            if syn_id is None or syn_id not in components:
                continue

            params = dict(entry)
            params["syn"] = syn_id

            gen_params = components[syn_id]["params"]
            Sn_gen = float(gen_params.get("Sn", mva_base))
            M_scale = Sn_gen / mva_base
            if M_scale > 1.01 or M_scale < 0.99:
                for vlim in ("VMAX", "VMIN"):
                    if vlim in params:
                        params[vlim] = float(params[vlim]) * M_scale

            components[name] = {"type": model, "params": params}

            # omega wire: governor reads generator speed
            connections.append({"from": f"{syn_id}.omega", "to": f"{name}.omega"})

            # Pref default (initializer will overwrite the param later)
            pref = float(entry.get("wref0", 1.0))
            connections.append({"from": f"CONST:{pref}", "to": f"{name}.Pref"})

            # Governor → generator Tm (override the CONST default)
            _remove_const_wire(connections, f"{syn_id}.Tm")
            connections.append({"from": f"{name}.Tm", "to": f"{syn_id}.Tm"})

    # ------------------------------------------------------------------
    # 4. PSS models
    # ------------------------------------------------------------------
    for model in ("ST2CUT", "IEEEST"):
        for entry in raw.get(model, []):
            name   = entry["idx"]
            avr_id = entry.get("avr")
            K1     = float(entry.get("K1", 0.0))
            K2     = float(entry.get("K2", 0.0))
            KS     = float(entry.get("KS", 0.0))
            # For ST2CUT: skip if both K1=K2=0 (no gain).
            # For IEEEST: skip if KS=0 (no gain).
            if model == "ST2CUT" and abs(K1) < 1e-9 and abs(K2) < 1e-9:
                continue
            if model == "IEEEST" and abs(KS) < 1e-9:
                continue
            if avr_id is None or avr_id not in components:
                continue

            syn_id = exc_syn.get(avr_id)
            if syn_id is None:
                continue

            if model == "ST2CUT":
                params = _normalise_st2cut_params(entry)
            elif model == "IEEEST":
                params = _normalise_ieeest_params(entry)
            else:
                params = dict(entry)
            params["avr"] = avr_id
            params["syn"] = syn_id
            components[name] = {"type": model, "params": params}
            pss_avr[name] = avr_id

            # PSS reads generator omega and Pe
            connections.append({"from": f"{syn_id}.omega", "to": f"{name}.omega"})
            connections.append({"from": f"{syn_id}.Pe",    "to": f"{name}.Pe"})

            # Governor Tm (if any) to PSS
            gov_tm_src = _find_wire_src(connections, f"{syn_id}.Tm")
            if gov_tm_src and not gov_tm_src.startswith("CONST:"):
                connections.append({"from": gov_tm_src, "to": f"{name}.Tm"})
            else:
                connections.append({"from": "CONST:0.0", "to": f"{name}.Tm"})

            # PSS signal added to exciter Vref
            # Replace the constant Vref wire with one that includes PSS correction.
            # We model this as: the Vref wire points to the PSS output, and the
            # compiler/initializer knows to add Vref_base + Vss.  For now, we
            # encode the PSS Vss output as an additive correction by using the
            # PARAM approach: wire PARAM:<exc>.Vref as the base, then add PSS.
            # Simpler: keep the CONST Vref wire and add a secondary PSS wire.
            # The compiler uses _pss_for_avr map to inject the PSS signal; we
            # preserve this by keeping 'avr' in the PSS params.
            print(f"  [json_compat] PSS {name} (K1={K1}, KS={KS}) wired to exciter {avr_id}")

    # ------------------------------------------------------------------
    # 5. Renewable models
    # ------------------------------------------------------------------
    def get_regca1(comp_type, comp_id):
        try:
            if comp_type == "REGCA1":
                return comp_id
            elif comp_type == "REECA1":
                return next(c["reg"] for c in raw.get("REECA1", []) if c["idx"] == comp_id)
            elif comp_type == "REPCA1":
                ree_id = next(c["ree"] for c in raw.get("REPCA1", []) if c["idx"] == comp_id)
                return get_regca1("REECA1", ree_id)
            elif comp_type == "WTDTA1":
                ree_id = next(c["ree"] for c in raw.get("WTDTA1", []) if c["idx"] == comp_id)
                return get_regca1("REECA1", ree_id)
            elif comp_type == "WTARA1":
                return next(c["rego"] for c in raw.get("WTARA1", []) if c["idx"] == comp_id)
            elif comp_type == "WTPTA1":
                rea_id = next(c["rea"] for c in raw.get("WTPTA1", []) if c["idx"] == comp_id)
                return get_regca1("WTARA1", rea_id)
            elif comp_type == "WTTQA1":
                rep_id = next(c["rep"] for c in raw.get("WTTQA1", []) if c["idx"] == comp_id)
                return get_regca1("REPCA1", rep_id)
        except StopIteration:
            pass
        return None

    wt_groups = {}
    for ctype in ["REGCA1", "REECA1", "REPCA1", "WTDTA1", "WTARA1", "WTPTA1", "WTTQA1"]:
        for c in raw.get(ctype, []):
            reg_id = get_regca1(ctype, c["idx"])
            if reg_id:
                if reg_id not in wt_groups:
                    wt_groups[reg_id] = {}
                wt_groups[reg_id][ctype] = c["idx"]

    for reg_id, group in wt_groups.items():
        # REGCA1
        if "REGCA1" in group:
            name = group["REGCA1"]
            entry = next(c for c in raw["REGCA1"] if c["idx"] == name)
            bus_id = entry.get("bus")
            components[name] = {"type": "REGCA1", "params": dict(entry)}
            connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})
            connections.append({"from": f"BUS_{bus_id}.Vd",    "to": f"{name}.Vd"})
            connections.append({"from": f"BUS_{bus_id}.Vq",    "to": f"{name}.Vq"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Ipcmd"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Iqcmd"})

        # REECA1
        if "REECA1" in group:
            name = group["REECA1"]
            entry = next(c for c in raw["REECA1"] if c["idx"] == name)
            components[name] = {"type": "REECA1", "params": dict(entry)}
            bus_id = next(c["bus"] for c in raw["REGCA1"] if c["idx"] == reg_id)
            connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})
            connections.append({"from": f"{reg_id}.Pe", "to": f"{name}.Pe"})
            connections.append({"from": f"{reg_id}.Qe", "to": f"{name}.Qe"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Pext"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Qext"})
            connections.append({"from": "CONST:1.0", "to": f"{name}.wg"})
            _remove_const_wire(connections, f"{reg_id}.Ipcmd")
            _remove_const_wire(connections, f"{reg_id}.Iqcmd")
            connections.append({"from": f"{name}.Ipcmd", "to": f"{reg_id}.Ipcmd"})
            connections.append({"from": f"{name}.Iqcmd", "to": f"{reg_id}.Iqcmd"})

        # REPCA1
        if "REPCA1" in group:
            name = group["REPCA1"]
            entry = next(c for c in raw["REPCA1"] if c["idx"] == name)
            components[name] = {"type": "REPCA1", "params": dict(entry)}
            bus_id = next(c["bus"] for c in raw["REGCA1"] if c["idx"] == reg_id)
            connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})
            connections.append({"from": "CONST:1.0", "to": f"{name}.f"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Pline"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Qline"})
            if "REECA1" in group:
                ree_id = group["REECA1"]
                _remove_const_wire(connections, f"{ree_id}.Pext")
                _remove_const_wire(connections, f"{ree_id}.Qext")
                connections.append({"from": f"{name}.Pext", "to": f"{ree_id}.Pext"})
                connections.append({"from": f"{name}.Qext", "to": f"{ree_id}.Qext"})

        # WTDTA1
        if "WTDTA1" in group:
            name = group["WTDTA1"]
            entry = next(c for c in raw["WTDTA1"] if c["idx"] == name)
            components[name] = {"type": "WTDTA1", "params": dict(entry)}
            connections.append({"from": f"{reg_id}.Pe", "to": f"{name}.Pe"})
            connections.append({"from": "CONST:0.0", "to": f"{name}.Pm"})
            if "REECA1" in group:
                ree_id = group["REECA1"]
                _remove_const_wire(connections, f"{ree_id}.wg")
                connections.append({"from": f"{name}.wg", "to": f"{ree_id}.wg"})

        # WTARA1
        if "WTARA1" in group:
            name = group["WTARA1"]
            entry = next(c for c in raw["WTARA1"] if c["idx"] == name)
            components[name] = {"type": "WTARA1", "params": dict(entry)}
            connections.append({"from": "CONST:0.0", "to": f"{name}.theta"})
            if "WTDTA1" in group:
                wtdta1_name = group["WTDTA1"]
                _remove_const_wire(connections, f"{wtdta1_name}.Pm")
                connections.append({"from": f"{name}.Pm", "to": f"{wtdta1_name}.Pm"})

        # WTPTA1
        if "WTPTA1" in group:
            name = group["WTPTA1"]
            entry = next(c for c in raw["WTPTA1"] if c["idx"] == name)
            components[name] = {"type": "WTPTA1", "params": dict(entry)}
            if "WTDTA1" in group:
                connections.append({"from": f"{group['WTDTA1']}.wt", "to": f"{name}.wt"})
            else:
                connections.append({"from": "CONST:1.0", "to": f"{name}.wt"})
            if "REECA1" in group:
                connections.append({"from": f"{group['REECA1']}.Pord", "to": f"{name}.Pord"})
            else:
                connections.append({"from": "CONST:0.0", "to": f"{name}.Pord"})
            if "WTTQA1" in group:
                connections.append({"from": f"{group['WTTQA1']}.Pref", "to": f"{name}.Pref"})
            else:
                connections.append({"from": "CONST:0.0", "to": f"{name}.Pref"})
            if "WTARA1" in group:
                wtara1_name = group["WTARA1"]
                _remove_const_wire(connections, f"{wtara1_name}.theta")
                connections.append({"from": f"{name}.theta", "to": f"{wtara1_name}.theta"})

        # WTTQA1
        if "WTTQA1" in group:
            name = group["WTTQA1"]
            entry = next(c for c in raw["WTTQA1"] if c["idx"] == name)
            components[name] = {"type": "WTTQA1", "params": dict(entry)}
            connections.append({"from": f"{reg_id}.Pe", "to": f"{name}.Pe"})
            if "WTDTA1" in group:
                connections.append({"from": f"{group['WTDTA1']}.wg", "to": f"{name}.wg"})
            else:
                connections.append({"from": "CONST:1.0", "to": f"{name}.wg"})

    # ------------------------------------------------------------------
    # PVD1 and BusFreq
    # ------------------------------------------------------------------
    for entry in raw.get("BusFreq", []):
        name = entry["idx"]
        bus_id = entry.get("bus")
        components[name] = {"type": "BusFreq", "params": dict(entry)}
        connections.append({"from": f"BUS_{bus_id}.Vd", "to": f"{name}.Vd"})
        connections.append({"from": f"BUS_{bus_id}.Vq", "to": f"{name}.Vq"})

    for entry in raw.get("PVD1", []):
        name = entry["idx"]
        bus_id = entry.get("bus")
        busf = entry.get("busf")
        params = dict(entry)
        params["Pref"] = params.get("p0", 0.0)
        params["Qref"] = params.get("q0", 0.0)
        components[name] = {"type": "PVD1", "params": params}
        connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})
        connections.append({"from": f"BUS_{bus_id}.Vd", "to": f"{name}.Vd"})
        connections.append({"from": f"BUS_{bus_id}.Vq", "to": f"{name}.Vq"})
        if busf:
            connections.append({"from": f"{busf}.f_pu", "to": f"{name}.f_pu"})
        else:
            connections.append({"from": "CONST:1.0", "to": f"{name}.f_pu"})
        
        # Check if there is a DGPRCT1 for this PVD1
        dgprct1_name = None
        for dg in raw.get("DGPRCT1", []):
            if dg.get("dev") == name:
                dgprct1_name = dg["idx"]
                break
        if dgprct1_name:
            connections.append({"from": f"{dgprct1_name}.ue", "to": f"{name}.ue"})
        else:
            connections.append({"from": "CONST:0.0", "to": f"{name}.ue"})

    for entry in raw.get("DGPRCT1", []):
        name = entry["idx"]
        dev = entry.get("dev")
        busf = entry.get("busfreq")
        params = dict(entry)
        if "fn" not in params:
            params["fn"] = fn
            
        # Find the bus of the device
        bus_id = None
        for pvd1 in raw.get("PVD1", []):
            if pvd1["idx"] == dev:
                bus_id = pvd1.get("bus")
                break
                
        if bus_id is not None:
            params["bus"] = bus_id
            
        components[name] = {"type": "DGPRCT1", "params": params}
        
        if bus_id is not None:
            connections.append({"from": f"BUS_{bus_id}.Vterm", "to": f"{name}.Vterm"})
        else:
            connections.append({"from": "CONST:1.0", "to": f"{name}.Vterm"})
            
        if busf:
            connections.append({"from": f"{busf}.f_pu", "to": f"{name}.f_pu"})
        else:
            connections.append({"from": "CONST:1.0", "to": f"{name}.f_pu"})

    # ------------------------------------------------------------------
    # Fix dangling Vd_dq / Vq_dq port references.
    # For GENROU, id_dq/iq_dq are output ports, but Vd_dq/Vq_dq are not.
    # Those are the *dq-frame terminal voltages*, computed inside the C++
    # kernel.  For the explicit wire, we use BUS_<id>.Vd/Vq with a note
    # that the compiler will do the Park transform to get the dq-frame.
    # The ESST3A model's Vd/Vq inputs actually want dq-frame voltages
    # (vd_dq_<gen>, vq_dq_<gen> in C++), NOT RI-frame.  The compiler
    # handles this mapping in _generate_input_gathering via wiring_map.
    # We annotate the wire source as a special "DQ" bus source so the
    # compiler can detect and emit the correct expression.
    # Replace BUS_<id>.{Vd,Vq} → DQ_<gen_name>.{Vd,Vq} for exciters.
    for i, w in enumerate(connections):
        dst = w["to"]
        # Identify exciter Vd/Vq wires (written as {syn_id}.Vd_dq above)
        if ".Vd_dq" in w["from"] or ".Vq_dq" in w["from"]:
            # Rewrite src to the special DQ_ prefix so the compiler can
            # distinguish dq-frame from RI-frame bus voltages.
            gen_name, sig = w["from"].split(".")
            connections[i] = {"from": f"DQ_{gen_name}.{sig}", "to": dst}

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    out = {}
    # Preserve all topology and config fields
    for key in ("config", "Bus", "PQ", "PV", "Slack", "Shunt", "Line",
                "Toggler", "BusFault"):
        if key in raw:
            out[key] = raw[key]

    out["components"] = components
    out["connections"] = connections
    return out


def _remove_const_wire(connections: List[Dict], dst: str):
    """Remove any CONST:* → dst wire from the connections list (in-place)."""
    to_remove = [
        i for i, w in enumerate(connections)
        if w["to"] == dst and w["from"].startswith("CONST:")
    ]
    for i in reversed(to_remove):
        del connections[i]


def _find_wire_src(connections: List[Dict], dst: str) -> str | None:
    """Return the 'from' of the last wire pointing to dst, or None."""
    for w in reversed(connections):
        if w["to"] == dst:
            return w["from"]
    return None


# ---------------------------------------------------------------------------
# Component instantiation
# ---------------------------------------------------------------------------

def instantiate_components(
    data: Dict[str, Any],
    registry: Dict[str, type] | None = None,
) -> Dict[str, PowerComponent]:
    """Create live PowerComponent instances from the ``components`` dict.

    Works with both old-format JSON (upgrading via ``to_new_format`` first)
    and new-format JSON that already has a ``components`` key.

    When the JSON is already in new format, generator parameters are
    normalised from machine base (Sn) to system base (mva_base) if needed.
    Old-format JSON is normalised inside ``to_new_format`` already.
    """
    already_new_format = "components" in data and "connections" in data
    if not already_new_format:
        data = to_new_format(data)

    if registry is None:
        registry = _registry()

    mva_base = float(data.get('config', {}).get('mva_base', 100.0))
    fn = float(data.get('config', {}).get('fn', 60.0))

    # Types that require Sn → mva_base normalisation
    _GENROU_TYPES = {'GENROU', 'GENROU_PHS', 'GENSAL', 'GENTPF', 'GENTPJ'}
    _GENCLS_TYPES = {'GENCLS'}

    result: Dict[str, PowerComponent] = {}
    for name, comp_spec in data["components"].items():
        type_name = comp_spec["type"]
        cls = registry.get(type_name)
        if cls is None:
            from src.errors import UnknownComponentTypeError
            raise UnknownComponentTypeError(type_name, list(registry.keys()))
        params = dict(comp_spec["params"])

        # Apply per-unit base normalisation for generators when loading
        # new-format JSON directly (old-format is normalised in to_new_format).
        if already_new_format and not params.get('_params_normalized'):
            if type_name in _GENROU_TYPES:
                params = _normalise_genrou_params(params, mva_base, fn)
                params['_params_normalized'] = True
            elif type_name in _GENCLS_TYPES:
                params = _normalise_gencls_params(params, mva_base, fn)
                params['_params_normalized'] = True

        result[name] = cls(name, params)

    return result


# ---------------------------------------------------------------------------
# Utility: check whether a raw JSON dict is in the old format
# ---------------------------------------------------------------------------

def is_old_format(raw: Dict[str, Any]) -> bool:
    """Return True if the dict looks like old-style (no ``connections`` key)."""
    return "connections" not in raw or "components" not in raw
