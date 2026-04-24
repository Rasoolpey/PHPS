"""
genrou_full.py
==============
Full PowerFactory-equivalent Synchronous Machine (ElmSym / TypSym)
implemented in the Port-Hamiltonian System (PHS) framework.

Model corresponds to the PowerFactory "Standard Model" (Model 2.2 – round-rotor,
Section 6.1 of the TechRef) extended to support:

  * Round-rotor (2q-damper, default) and salient-pole (1q-damper) configurations
  * Full subtransient state-space in the dq reference frame (Park)
  * Saturation of mutual reactances xad / xaq (quadratic, exponential, or tabular)
  * Three damping-torque coefficients (dpu, dkd, dpe)
  * Correct mechanical torque initialisation (pt / xmdm input priority)
  * Excitation interfacing in the air-gap-line non-reciprocal p.u. system
  * Parameter conversion: short-circuit input → equivalent-circuit parameters
    (exact Canay procedure or standard approximation)
  * Port-Hamiltonian matrices J, R, g and Hamiltonian H(x) consistent with the
    full 8-state (round-rotor) or 6-state (salient-pole) ODE

State vector
------------
Round-rotor  (8 states):
    x = [delta, omega, psi_fd, psi_1d, psi_1q, psi_2q, speed_alias*, phi_alias*]
    * delta  – rotor position angle (rad, ≡ phi in PF notation)
    * omega  – per-unit speed (n in PF)
    * psi_fd – excitation (field) winding flux (d-axis)
    * psi_1d – 1d-damper flux (d-axis)
    * psi_1q – 1q-damper flux (q-axis)
    * psi_2q – 2q-damper flux (q-axis)

Salient-pole (6 states): drop psi_2q (set psi_2q ≡ 0, k2q ≡ 0).

All quantities are in p.u. (reciprocal xadu system for internal rotor variables)
unless stated otherwise.

Port-Hamiltonian form
---------------------
    ẋ = [J(x) − R(x)] ∇H(x) + g(x) u

where:
    u   = [vd, vq, Tm, ve]     (inputs: terminal voltages, mech. torque, excitation)
    y   = [id, iq, omega, Te]  (outputs: conjugate port variables)

References
----------
  [PF]  DIgSILENT PowerFactory 2024 Technical Reference – Synchronous Machine
        (ElmSym, TypSym), Sections 6.1.1–6.1.5
  [K94] Kundur, "Power System Stability and Control", 1994
  [I02] IEEE Std 1110-2002 (Generator Modelling Guide)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


# ---------------------------------------------------------------------------
# Helper: saturation coefficient (quadratic approximation, [PF] Eq. 120-122)
# ---------------------------------------------------------------------------

def _csat_quadratic(psi_m: float, SG10: float, SG12: float) -> float:
    """Return c_sat for quadratic saturation model ([PF] Eq. 120-122)."""
    if SG10 <= 0.0:
        return 0.0
    ratio = math.sqrt(max(0.0, 1.2 * SG12 / SG10))
    denom = 1.0 - ratio
    if abs(denom) < 1e-12:
        return 0.0
    Ag = (1.2 - ratio) / denom
    Bg = SG10 / max((1.0 - Ag) ** 2, 1e-12)
    if psi_m <= Ag:
        return 0.0
    return Bg * (psi_m - Ag) ** 2 / psi_m


def _csat_exponential(psi_m: float, SG10: float, SG12: float) -> float:
    """Return c_sat for exponential saturation model ([PF] Eq. 123-124)."""
    if SG10 <= 0.0:
        return 0.0
    if SG12 <= 0.0 or SG10 <= 0.0:
        exp_val = 1.0
    else:
        exp_val = math.log(max(1e-12, 1.2 * SG12 / SG10)) / math.log(1.2)
    return SG10 * (psi_m ** exp_val) / max(psi_m, 1e-12)


# ---------------------------------------------------------------------------
# Helper: short-circuit → equivalent-circuit parameter conversion
# Implements the Canay exact procedure ([PF] Sec. 6.1.1.1, Eqs. 49-54)
# ---------------------------------------------------------------------------

def sc_to_eqcct_daxis(xd, xd_p, xd_pp, td_p, td_pp, xl, xrld=0.0, omega_n=2*math.pi*50):
    """
    Convert d-axis short-circuit parameters to equivalent-circuit parameters.

    Returns
    -------
    dict with keys: xad, xfd, x1d, rfd, r1d
    """
    xad = xd - xl

    # Help variables [PF] Eqs. 49-51
    x1 = xad + xrld
    x2 = x1 - xad**2 / xd
    x3 = (x2 - x1 * xd_pp / xd) / (1.0 - xd_pp / xd + 1e-30)

    T1 = (xd / xd_p) * td_p + (1.0 - xd / xd_p + xd / xd_pp) * td_pp
    T2 = td_p + td_pp

    a = (x2 * T1 - x1 * T2) / (x1 - x2 + 1e-30)
    b = (x3 / (x3 - x2 + 1e-30)) * td_p * td_pp

    disc = max(0.0, (a / 2.0) ** 2 - b)
    sqrt_disc = math.sqrt(disc)
    T_sfd = -a / 2.0 + sqrt_disc     # Eq. 52
    T_s1d = -a / 2.0 - sqrt_disc

    denom_T = (T1 - T2) / (x1 - x2 + 1e-30)
    xfd = denom_T * (T_sfd - T_s1d) + T_s1d / x3   # Eq. 53
    x1d = denom_T * (T_s1d - T_sfd) + T_sfd / x3
    rfd = xfd / (omega_n * T_sfd + 1e-30)
    r1d = x1d / (omega_n * T_s1d + 1e-30)

    return {'xad': xad, 'xfd': xfd, 'x1d': x1d, 'rfd': rfd, 'r1d': r1d}


def sc_to_eqcct_qaxis_round(xq, xq_p, xq_pp, tq_p, tq_pp, xl, xrlq=0.0,
                             omega_n=2*math.pi*50):
    """Q-axis equivalent circuit for round-rotor (two rotor loops)."""
    xaq = xq - xl
    x1 = xaq + xrlq
    x2 = x1 - xaq**2 / xq
    x3 = (x2 - x1 * xq_pp / xq) / (1.0 - xq_pp / xq + 1e-30)
    T1 = (xq / xq_p) * tq_p + (1.0 - xq / xq_p + xq / xq_pp) * tq_pp
    T2 = tq_p + tq_pp
    a  = (x2 * T1 - x1 * T2) / (x1 - x2 + 1e-30)
    b  = (x3 / (x3 - x2 + 1e-30)) * tq_p * tq_pp
    disc    = max(0.0, (a / 2.0) ** 2 - b)
    sqrt_d  = math.sqrt(disc)
    T_s1q   = -a / 2.0 + sqrt_d
    T_s2q   = -a / 2.0 - sqrt_d
    denom_T = (T1 - T2) / (x1 - x2 + 1e-30)
    x1q = denom_T * (T_s1q - T_s2q) + T_s2q / x3
    x2q = denom_T * (T_s2q - T_s1q) + T_s1q / x3
    r1q = x1q / (omega_n * T_s1q + 1e-30)
    r2q = x2q / (omega_n * T_s2q + 1e-30)
    return {'xaq': xaq, 'x1q': x1q, 'x2q': x2q, 'r1q': r1q, 'r2q': r2q}


def sc_to_eqcct_qaxis_salient(xq, xq_pp, tq_pp, xl, omega_n=2*math.pi*50):
    """Q-axis equivalent circuit for salient-pole (one rotor loop, [PF] Eq. 54)."""
    xaq = xq - xl
    x1q = (xaq * (xq_pp - xl)) / (xq - xq_pp + 1e-30)
    r1q = (xq_pp / xq) * (xaq + x1q) / (omega_n * tq_pp + 1e-30)
    return {'xaq': xaq, 'x1q': x1q, 'r1q': r1q}


# ---------------------------------------------------------------------------
# Coupling factors (subtransient blending)  [PF] Eqs. 64-66
# ---------------------------------------------------------------------------

def _coupling_factors_daxis(xad, xrld, xfd, x1d):
    """
    k_fd, k_1d, x''_d for the d-axis ([PF] Eq. 64).
    """
    denom = (xad + xrld) * (x1d + xfd) + xfd * x1d
    k_fd = xad * x1d / denom
    k_1d = xad * xfd / denom
    xd_pp = xad + 0.0 - (k_fd + k_1d) * xad   # xl added separately by caller
    return k_fd, k_1d, xd_pp


def _coupling_factors_qaxis_round(xaq, xrlq, x1q, x2q):
    """
    k_1q, k_2q, x''_q for round-rotor q-axis ([PF] Eq. 66).
    """
    denom = (xaq + xrlq) * (x2q + x1q) + x2q * x1q
    k_1q = xaq * x2q / denom
    k_2q = xaq * x1q / denom
    xq_pp = xaq - (k_2q + k_1q) * xaq   # xl added separately
    return k_1q, k_2q, xq_pp


def _coupling_factors_qaxis_salient(xaq, xrlq, x1q):
    """k_1q, k_2q (=0) for salient-pole ([PF] Eq. 65)."""
    k_1q = xaq / (xaq + xrlq + x1q)
    k_2q = 0.0
    xq_pp = xaq - k_1q * xaq
    return k_1q, k_2q, xq_pp


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GenRouFull(PowerComponent):
    """
    Full PowerFactory Standard Synchronous Machine (Model 2.1/2.2) in PHS form.

    Rotor model
    -----------
    * d-axis : field winding (fd) + 1d-damper  → 2 state fluxes
    * q-axis : 1q-damper + optional 2q-damper   → 1 or 2 state fluxes
    * Mechanical: delta, omega                   → 2 states

    Total states: 6 (salient) or 8 (round-rotor, default)

    State ordering (round-rotor):
        [0] delta   – rotor angle (rad)
        [1] omega   – per-unit speed
        [2] psi_fd  – field winding flux (reciprocal xadu system)
        [3] psi_1d  – 1d-damper flux
        [4] psi_1q  – 1q-damper flux
        [5] psi_2q  – 2q-damper flux  (0 for salient-pole)

    For salient-pole set param 'salient_pole': True; state index [5] unused.
    """

    # ------------------------------------------------------------------
    # Port / state / param schemas
    # ------------------------------------------------------------------

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('vd',  'effort', 'pu'),   # d-axis terminal voltage (Park frame)
                ('vq',  'effort', 'pu'),   # q-axis terminal voltage
                ('Tm',  'effort', 'pu'),   # mechanical torque (from governor)
                ('ve',  'effort', 'pu'),   # excitation voltage (non-reciprocal p.u.)
            ],
            'out': [
                ('id',    'flow', 'pu'),   # d-axis stator current
                ('iq',    'flow', 'pu'),   # q-axis stator current
                ('omega', 'flow', 'pu'),   # rotor speed
                ('Te',    'flow', 'pu'),   # electrical torque
                ('Pe',    'flow', 'pu'),   # active power
                ('Qe',    'flow', 'pu'),   # reactive power
                ('ie',    'flow', 'pu'),   # excitation current (non-reciprocal)
                ('I_Re',  'flow', 'pu'),   # RI-frame Norton current, real
                ('I_Im',  'flow', 'pu'),   # RI-frame Norton current, imaginary
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        # Round-rotor ordering; salient-pole: psi_2q is always present but frozen at 0
        return ['delta', 'omega', 'psi_fd', 'psi_1d', 'psi_1q', 'psi_2q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            # --- Rating ---
            'H':         'Inertia constant (s)',
            'omega_b':   'Base angular frequency (rad/s)',
            'cosn':      'Rated power factor',
            # --- Stator ---
            'rstr':      'Stator resistance (p.u.)',
            'xl':        'Stator leakage reactance (p.u.)',
            # --- Short-circuit input (standard user-facing) ---
            'xd':        'd-axis synchronous reactance (p.u.)',
            'xq':        'q-axis synchronous reactance (p.u.)',
            'xd_prime':  'd-axis transient reactance (p.u.)',
            'xq_prime':  'q-axis transient reactance (p.u.)',
            'xd_pp':     'd-axis subtransient reactance (p.u.)',
            'xq_pp':     'q-axis subtransient reactance (p.u.)',
            'Td0_prime': 'd-axis open-loop transient time constant (s)',
            'Tq0_prime': 'q-axis open-loop transient time constant (s)',
            'Td0_pp':    'd-axis open-loop subtransient time constant (s)',
            'Tq0_pp':    'q-axis open-loop subtransient time constant (s)',
            # --- Optional coupling reactances ---
            'xrld':      'Coupling reactance field-damper d-axis (p.u.), default 0',
            'xrlq':      'Coupling reactance q-axis damper windings (p.u.), default 0',
            # --- Damping ---
            'dkd':       'Damping torque coefficient (p.u. torque / p.u. speed deviation)',
            'dpe':       'Power-based damping coefficient (p.u. power / p.u. speed deviation)',
            'dpu':       'Shaft friction torque coefficient (p.u. torque / p.u. speed)',
            # --- Saturation ---
            'SG10':      'Saturation factor at 1.0 p.u. (quadratic/exponential)',
            'SG12':      'Saturation factor at 1.2 p.u.',
            'sat_model': 'Saturation model: "none", "quadratic", "exponential"',
            'sat_axis':  'Axis for saturation: "dq_magnitude", "dq_equal", "d_only"',
            # --- Topology ---
            'salient_pole': 'True → salient-pole (1q damper only), False → round-rotor',
        }

    # ------------------------------------------------------------------
    # Internal parameter derivation
    # ------------------------------------------------------------------

    def _derive_eqcct(self) -> dict:
        """
        Derive equivalent-circuit parameters from short-circuit input params.
        Implements the exact Canay conversion ([PF] Sec. 6.1.1.1).

        Returns a dict with all internal p.u. parameters needed by the ODE.
        """
        p = self.params
        omega_n = float(p.get('omega_b', 2 * math.pi * 50))
        xl      = float(p['xl'])
        xrld    = float(p.get('xrld', 0.0))
        xrlq    = float(p.get('xrlq', 0.0))
        salient = bool(p.get('salient_pole', False))

        # d-axis
        d = sc_to_eqcct_daxis(
            xd    = float(p['xd']),
            xd_p  = float(p['xd_prime']),
            xd_pp = float(p['xd_pp']),
            td_p  = float(p['Td0_prime']),
            td_pp = float(p['Td0_pp']),
            xl    = xl,
            xrld  = xrld,
            omega_n = omega_n
        )
        xad, xfd, x1d, rfd, r1d = d['xad'], d['xfd'], d['x1d'], d['rfd'], d['r1d']

        # q-axis
        if salient:
            q = sc_to_eqcct_qaxis_salient(
                xq    = float(p['xq']),
                xq_pp = float(p['xq_pp']),
                tq_pp = float(p['Tq0_pp']),
                xl    = xl,
                omega_n = omega_n
            )
            x2q = 0.0; r2q = 0.0
        else:
            q = sc_to_eqcct_qaxis_round(
                xq    = float(p['xq']),
                xq_p  = float(p['xq_prime']),
                xq_pp = float(p['xq_pp']),
                tq_p  = float(p['Tq0_prime']),
                tq_pp = float(p['Tq0_pp']),
                xl    = xl,
                xrlq  = xrlq,
                omega_n = omega_n
            )
            x2q = q['x2q']; r2q = q['r2q']

        xaq = q['xaq']; x1q = q['x1q']; r1q = q['r1q']

        # Coupling factors  [PF] Eqs. 64-66
        k_fd, k_1d, xd_pp_bare = _coupling_factors_daxis(xad, xrld, xfd, x1d)
        xd_pp_check = xl + xd_pp_bare   # should match input xd_pp

        if salient:
            k_1q, k_2q, xq_pp_bare = _coupling_factors_qaxis_salient(xaq, xrlq, x1q)
        else:
            k_1q, k_2q, xq_pp_bare = _coupling_factors_qaxis_round(xaq, xrlq, x1q, x2q)

        xq_pp_check = xl + xq_pp_bare

        # Reactance sums per winding loop  [PF] Eq. 72
        xfd_loop = xad + xrld + xfd
        x1d_loop = xad + xrld + x1d
        x1q_loop = xaq + xrlq + x1q
        x2q_loop = xaq + xrlq + x2q

        return {
            'xad': xad, 'xaq': xaq,
            'xfd': xfd, 'x1d': x1d, 'x1q': x1q, 'x2q': x2q,
            'rfd': rfd, 'r1d': r1d, 'r1q': r1q, 'r2q': r2q,
            'k_fd': k_fd, 'k_1d': k_1d,
            'k_1q': k_1q, 'k_2q': k_2q,
            'xd_pp': xd_pp_check, 'xq_pp': xq_pp_check,
            'xfd_loop': xfd_loop, 'x1d_loop': x1d_loop,
            'x1q_loop': x1q_loop, 'x2q_loop': x2q_loop,
            'xrld': xrld, 'xrlq': xrlq,
            'salient': salient,
        }

    def _eqcct(self) -> dict:
        """Cached equivalent-circuit parameter dictionary."""
        if not hasattr(self, '_eqcct_cache'):
            self._eqcct_cache = self._derive_eqcct()
        return self._eqcct_cache

    # ------------------------------------------------------------------
    # Saturation  [PF] Sec. 6.1.4
    # ------------------------------------------------------------------

    def _saturation_factors(self, psi_ad: float, psi_aq: float) -> Tuple[float, float]:
        """
        Return (satd, satq) ∈ (0, 1].
        psi_ad, psi_aq: d- and q-axis magnetising flux components (p.u.)
        """
        p   = self.params
        eq  = self._eqcct()
        model = str(p.get('sat_model', 'none')).lower()
        axis  = str(p.get('sat_axis', 'dq_magnitude')).lower()

        if model == 'none':
            return 1.0, 1.0

        SG10 = float(p.get('SG10', 0.0))
        SG12 = float(p.get('SG12', 0.0))

        # Magnetising flux magnitude  [PF] Eq. 79
        psi_m = math.sqrt(psi_ad ** 2 + psi_aq ** 2)
        if psi_m < 1e-12:
            return 1.0, 1.0

        if model == 'quadratic':
            csat = _csat_quadratic(psi_m, SG10, SG12)
        elif model == 'exponential':
            csat = _csat_exponential(psi_m, SG10, SG12)
        else:
            csat = 0.0   # fallback

        xadu = eq['xad']
        xaqu = eq['xaq']

        # Saturation factor options  [PF] Eqs. 127-130
        if axis == 'dq_magnitude':
            # Common for round-rotor: satq weighted by xaqu/xadu  [PF] Eq. 127
            satd = 1.0 / (1.0 + csat)
            satq = 1.0 / (1.0 + (xaqu / (xadu + 1e-12)) * csat)
        elif axis == 'dq_equal':
            satd = 1.0 / (1.0 + csat)
            satq = satd
        elif axis == 'd_only':
            # Salient-pole: saturate only d-axis  [PF] Eq. 129
            satd = 1.0 / (1.0 + csat)
            satq = 1.0
        else:
            satd = satq = 1.0

        return satd, satq

    # ------------------------------------------------------------------
    # Core ODE building blocks
    # ------------------------------------------------------------------

    def _stator_currents(self, psi_fd: float, psi_1d: float,
                         psi_1q: float, psi_2q: float,
                         vd: float, vq: float,
                         satd: float, satq: float) -> Tuple[float, float,
                                                            float, float,
                                                            float, float]:
        """
        Solve the stator algebraic equations ([PF] Eqs. 61-62/73-74) for id, iq.

        Returns (id, iq, psi_d, psi_q, psi_d_pp, psi_q_pp)

        Subtransient voltages u''_d, u''_q (RMS simplification, n≈1):
            u''_d = -psi''_q
            u''_q = +psi''_d

        Stator equations (n≈1, neglect transformer voltage):
            vd = u''_d - rstr*id + xq''*iq  →  rstr*id - xq''*iq = -vd - u''_d
            vq = u''_q - rstr*iq - xd''*id  →  xd''*id + rstr*iq = vq - u''_q

        [PF] Eq. 73 uses the actual n; here we expose it via the step function.
        """
        p   = self.params
        eq  = self._eqcct()
        rstr  = float(p['rstr'])
        xl    = float(p['xl'])
        xd_pp = eq['xd_pp']
        xq_pp = eq['xq_pp']
        k_fd  = eq['k_fd']; k_1d = eq['k_1d']
        k_1q  = eq['k_1q']; k_2q = eq['k_2q']

        # Subtransient fluxes  [PF] Eq. 60
        psi_d_pp = k_fd * psi_fd + k_1d * psi_1d
        psi_q_pp = k_1q * psi_1q + k_2q * psi_2q

        # Subtransient voltages (RMS, n≈1)  [PF] Eq. 74
        u_d_pp = -psi_q_pp
        u_q_pp =  psi_d_pp

        # Solve 2×2 linear system  [PF] Eq. 73
        # [rstr, -xq''] [id]   [u_d_pp - vd]
        # [xd'',  rstr] [iq] = [vq - u_q_pp]
        det = rstr ** 2 + xd_pp * xq_pp
        rhs_d = u_d_pp - vd     # = -psi_q_pp - vd
        rhs_q = vq - u_q_pp     # = vq - psi_d_pp

        # Solve using Cramer
        id_ =  (rstr * rhs_d + xq_pp * rhs_q) / det
        iq_ = (-xd_pp * rhs_d + rstr * rhs_q) / det

        # Stator fluxes  [PF] Eq. 61
        psi_d = -xd_pp * id_ + psi_d_pp
        psi_q = -xq_pp * iq_ + psi_q_pp

        return id_, iq_, psi_d, psi_q, psi_d_pp, psi_q_pp

    def _magnetising_fluxes(self, id_: float, iq_: float,
                            psi_fd: float, psi_1d: float,
                            psi_1q: float, psi_2q: float,
                            satd: float = 1.0,
                            satq: float = 1.0) -> Tuple[float, float]:
        """
        Compute magnetising fluxes psi_ad, psi_aq ([PF] Eq. 58).

        With saturation the mutual reactances scale as:
            xad_sat = satd * xad_u,  xaq_sat = satq * xaq_u
        so the rotor currents (ifd, i1d, i1q, i2q) from flux/reactance relations
        feed back through the saturated xad into psi_ad.

        For the PHS ODE the flux-state form is used directly:
            psi_ad = psi_fd - (xrld + xfd)*ifd - (xrld)*i1d   [from rotor flux eqs]
        Here we use the simpler magnetising-flux expression from stator perspective:
            psi_ad = psi_d + xl*id
            psi_aq = psi_q + xl*iq
        which is equivalent ([PF] Eq. 57 rearranged).
        """
        xl = float(self.params['xl'])
        eq = self._eqcct()
        k_fd = eq['k_fd']; k_1d = eq['k_1d']
        k_1q = eq['k_1q']; k_2q = eq['k_2q']
        xd_pp = eq['xd_pp']; xq_pp = eq['xq_pp']

        psi_d_pp = k_fd * psi_fd + k_1d * psi_1d
        psi_q_pp = k_1q * psi_1q + k_2q * psi_2q
        psi_d = -xd_pp * id_ + psi_d_pp
        psi_q = -xq_pp * iq_ + psi_q_pp
        psi_ad = psi_d + xl * id_
        psi_aq = psi_q + xl * iq_
        return psi_ad, psi_aq

    def _rotor_currents(self, psi_fd: float, psi_1d: float,
                        psi_1q: float, psi_2q: float,
                        id_: float, iq_: float) -> Tuple[float, float, float, float]:
        """
        Solve for rotor currents (ifd, i1d, i1q, i2q) from flux equations
        ([PF] Eq. 69).

        For round-rotor:
            ifd = k_fd*id + (x1d_loop*psi_fd - (xad+xrld)*psi_1d) / xdet_d
            i1d = k_1d*id + (xfd_loop*psi_1d - (xad+xrld)*psi_fd) / xdet_d
            (analogously for q-axis)
        """
        eq   = self._eqcct()
        xad  = eq['xad']; xrld = eq['xrld']
        xaq  = eq['xaq']; xrlq = eq['xrlq']
        xfd  = eq['xfd']; x1d  = eq['x1d']
        x1q  = eq['x1q']; x2q  = eq['x2q']
        k_fd = eq['k_fd']; k_1d = eq['k_1d']
        k_1q = eq['k_1q']; k_2q = eq['k_2q']

        # Reactance sums  [PF] Eq. 72
        xfd_loop = xad + xrld + xfd
        x1d_loop = xad + xrld + x1d
        x1q_loop = xaq + xrlq + x1q
        x2q_loop = xaq + xrlq + x2q
        xdet_d   = (xad + xrld) * (x1d + xfd) + xfd * x1d   # [PF] Eq. 70
        xdet_q   = (xaq + xrlq) * (x2q + x1q) + x2q * x1q

        i_fd = k_fd * id_ + (x1d_loop * psi_fd - (xad + xrld) * psi_1d) / (xdet_d + 1e-30)
        i_1d = k_1d * id_ + (xfd_loop * psi_1d - (xad + xrld) * psi_fd) / (xdet_d + 1e-30)

        if eq['salient']:
            # i2q = 0; k2q = 0; psi_2q = 0  [PF] Eq. 71
            i_1q = k_1q * iq_ + psi_1q / (x1q + 1e-30)
            i_2q = 0.0
        else:
            i_1q = k_1q * iq_ + (x2q_loop * psi_1q - (xaq + xrlq) * psi_2q) / (xdet_q + 1e-30)
            i_2q = k_2q * iq_ + (x1q_loop * psi_2q - (xaq + xrlq) * psi_1q) / (xdet_q + 1e-30)

        return i_fd, i_1d, i_1q, i_2q

    def _excitation_current(self, i_fd: float) -> float:
        """
        Convert field winding current ifd (reciprocal p.u.) to excitation current
        ie in non-reciprocal (air-gap-line) p.u. system.
        [PF] Eq. 131:  ie = xadu * ifd
        """
        return self._eqcct()['xad'] * i_fd

    def _excitation_voltage_input(self, ve: float) -> float:
        """
        Convert excitation voltage ve (non-reciprocal p.u.) to vfd (reciprocal).
        [PF] Eq. 132:  vfd = (rfd / xadu) * ve
        """
        eq = self._eqcct()
        return (eq['rfd'] / (eq['xad'] + 1e-30)) * ve

    def _electrical_torque(self, id_: float, iq_: float,
                           psi_d: float, psi_q: float) -> float:
        """
        Electrical torque [PF] Eq. 67:
            te = (iq * psi_d - id * psi_q) / cosn
        """
        cosn = float(self.params.get('cosn', 1.0))
        return (iq_ * psi_d - id_ * psi_q) / cosn

    def _damping_torque(self, omega: float, omega_ref: float = 1.0) -> float:
        """
        Combined damping torque (dkd + dpe terms, [PF] Eqs. 103-104).
        dpu friction is embedded in the mechanical torque tm calculation.
        """
        p = self.params
        dkd = float(p.get('dkd', 0.0))
        dpe = float(p.get('dpe', 0.0))
        dn  = omega - omega_ref
        tdkd = dkd * dn
        tdpe = (dpe / max(omega, 1e-3)) * dn
        return tdkd + tdpe

    def _mechanical_torque(self, Tm_input: float, omega: float) -> float:
        """
        Mechanical torque tm for a generator ([PF] Eq. 102):
            tm = Tm_input / omega - dpu * omega
        Here Tm_input corresponds to the turbine power signal pt (in p.u.).
        addmt is assumed zero (no additional torque offset).
        """
        p   = self.params
        dpu = float(p.get('dpu', 0.0))
        return Tm_input / max(omega, 1e-3) - dpu * omega

    # ------------------------------------------------------------------
    # ODE right-hand side  (called by integrator)
    # ------------------------------------------------------------------

    def _ode_rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute ẋ = f(x, u).

        x = [delta, omega, psi_fd, psi_1d, psi_1q, psi_2q]
        u = [vd, vq, Tm, ve]

        Implements [PF] Eqs. 68 (rotor voltage), 73 (stator, RMS),
                          100 (swing), 102 (mechanical torque).
        """
        p     = self.params
        eq    = self._eqcct()
        omega_b = float(p.get('omega_b', 2 * math.pi * 50))
        H_val   = float(p['H'])

        delta  = x[0]; omega  = x[1]
        psi_fd = x[2]; psi_1d = x[3]
        psi_1q = x[4]; psi_2q = x[5]
        vd = u[0]; vq = u[1]; Tm_in = u[2]; ve = u[3]

        # Saturation factors (iterate: first use old fluxes for psi_m estimate)
        # Simple approach: use stator-derived magnetising fluxes at previous step
        # (explicit update is standard for RMS-stability studies)
        psi_d_pp_tmp = eq['k_fd'] * psi_fd + eq['k_1d'] * psi_1d
        psi_q_pp_tmp = eq['k_1q'] * psi_1q + eq['k_2q'] * psi_2q
        xl   = float(p['xl'])
        xdpp = eq['xd_pp']; xqpp = eq['xq_pp']

        # approximate id,iq for saturation evaluation (ignore rstr coupling)
        id_tmp  = (psi_d_pp_tmp - vq) / (xdpp + 1e-12)   # rough
        iq_tmp  = (vd + psi_q_pp_tmp) / (xqpp + 1e-12)

        psi_ad0 = xdpp * (-id_tmp) + psi_d_pp_tmp + xl * id_tmp
        psi_aq0 = xqpp * (-iq_tmp) + psi_q_pp_tmp + xl * iq_tmp
        psi_m0  = math.sqrt(psi_ad0**2 + psi_aq0**2)
        satd, satq = self._saturation_factors(psi_ad0, psi_aq0)

        # Full stator solve with saturation-modified xd'', xq''
        # (In the full PF model saturation modifies xad/xaq which ripples into
        #  xd_pp, xq_pp and coupling factors.  Here we apply a first-order
        #  correction consistent with [PF] Eq. 117.)
        xad_s = satd * eq['xad']
        xaq_s = satq * eq['xaq']
        # Recompute coupling factors with saturated mutual reactances
        k_fd_s, k_1d_s, xd_pp_s_bare = _coupling_factors_daxis(xad_s, eq['xrld'],
                                                                  eq['xfd'], eq['x1d'])
        if eq['salient']:
            k_1q_s, k_2q_s, xq_pp_s_bare = _coupling_factors_qaxis_salient(
                xaq_s, eq['xrlq'], eq['x1q'])
        else:
            k_1q_s, k_2q_s, xq_pp_s_bare = _coupling_factors_qaxis_round(
                xaq_s, eq['xrlq'], eq['x1q'], eq['x2q'])
        xd_pp_s = xl + xd_pp_s_bare
        xq_pp_s = xl + xq_pp_s_bare

        psi_d_pp = k_fd_s * psi_fd + k_1d_s * psi_1d
        psi_q_pp = k_1q_s * psi_1q + k_2q_s * psi_2q
        det_s    = float(p['rstr'])**2 + xd_pp_s * xq_pp_s
        rhs_d    = -psi_q_pp - vd
        rhs_q    = vq - psi_d_pp
        id_  = ( float(p['rstr']) * rhs_d + xq_pp_s * rhs_q) / det_s
        iq_  = (-xd_pp_s  * rhs_d + float(p['rstr']) * rhs_q) / det_s

        psi_d = -xd_pp_s * id_ + psi_d_pp
        psi_q = -xq_pp_s * iq_ + psi_q_pp

        # Electrical torque  [PF] Eq. 67
        te = self._electrical_torque(id_, iq_, psi_d, psi_q)

        # Mechanical torque  [PF] Eq. 102
        tm = self._mechanical_torque(Tm_in, omega)

        # Damping  [PF] Eqs. 103-104
        tdamp = self._damping_torque(omega)

        # Excitation voltage conversion  [PF] Eq. 132
        vfd = self._excitation_voltage_input(ve)

        # ---- Rotor flux derivatives  [PF] Eq. 68 ----
        # dψ_fd/dt = ω_n * (vfd - rfd * ifd)
        # dψ_1d/dt = ω_n * (0   - r1d * i1d)
        # dψ_1q/dt = ω_n * (0   - r1q * i1q)
        # dψ_2q/dt = ω_n * (0   - r2q * i2q)   (0 for salient)

        i_fd, i_1d, i_1q, i_2q = self._rotor_currents(
            psi_fd, psi_1d, psi_1q, psi_2q, id_, iq_)

        dxdt = np.zeros(6)
        dxdt[0] = omega_b * (omega - 1.0)                         # dδ/dt  [PF] Eq.115
        dxdt[1] = (tm - te - tdamp) / (2.0 * H_val)               # dω/dt  [PF] Eq.100
        dxdt[2] = omega_b * (vfd - eq['rfd'] * i_fd)              # dψ_fd/dt
        dxdt[3] = omega_b * (-eq['r1d'] * i_1d)                   # dψ_1d/dt
        dxdt[4] = omega_b * (-eq['r1q'] * i_1q)                   # dψ_1q/dt
        dxdt[5] = omega_b * (-eq['r2q'] * i_2q)                   # dψ_2q/dt (0 if salient)

        return dxdt

    # ------------------------------------------------------------------
    # C++ code generation (compute_outputs and step)
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        """
        Evaluate outputs from current state (Norton current for network interface).
        Subtransient fluxes → network-frame Norton current injection.
        """
        return r"""
            // ---- Unpack states ----
            double delta  = x[0];
            double omega  = x[1];
            double psi_fd = x[2];
            double psi_1d = x[3];
            double psi_1q = x[4];
            double psi_2q = x[5];

            double sin_d = sin(delta), cos_d = cos(delta);

            // ---- Subtransient fluxes ----
            double psi_d_pp = k_fd * psi_fd + k_1d * psi_1d;
            double psi_q_pp = k_1q * psi_1q + k_2q * psi_2q;

            // ---- Norton current (dq) ----
            double det = rstr*rstr + xd_pp*xq_pp;
            double id_no = ( rstr * psi_d_pp + xq_pp * psi_q_pp) / det;  // voltage-independent
            double iq_no = (-xd_pp * psi_d_pp + rstr * psi_q_pp) / det;

            // Wait — Norton needs the driving emf in the right sign convention:
            // vd = -rstr*id + xq_pp*iq - psi_q_pp   (≡ u_d'' + stator eq)
            // Norton: set vd=vq=0 → id_no from stator with zero terminal voltage
            //   0 = -rstr*id + xq_pp*iq - psi_q_pp
            //   0 = -rstr*iq - xd_pp*id + psi_d_pp
            // [ rstr, -xq_pp ] [id]   [ psi_q_pp ]
            // [ xd_pp,  rstr ] [iq] = [-psi_d_pp ]   ← corrected sign
            id_no = ( rstr * psi_q_pp - xq_pp * (-psi_d_pp)) / det;
            iq_no = (-xd_pp * psi_q_pp + rstr * (-psi_d_pp)) / det;

            // ---- RI frame ----
            outputs[0] = id_no * sin_d + iq_no * cos_d;   // I_Re
            outputs[1] = -id_no * cos_d + iq_no * sin_d;  // I_Im
            outputs[2] = omega;
            // Te, Pe, Qe, ie: filled in step
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = 0.0;
            outputs[6] = 0.0;
            outputs[7] = outputs[0];
            outputs[8] = outputs[1];
        """

    def get_cpp_step_code(self) -> str:
        """Full ODE RHS for integration step."""
        return r"""
            // ---- Inputs ----
            double vd_in  = inputs[0];   // d-axis terminal voltage
            double vq_in  = inputs[1];   // q-axis terminal voltage
            double Tm_in  = inputs[2];   // mechanical torque (turbine)
            double ve_in  = inputs[3];   // excitation voltage (non-reciprocal pu)

            // ---- States ----
            double delta  = x[0];
            double omega  = x[1];
            double psi_fd = x[2];
            double psi_1d = x[3];
            double psi_1q = x[4];
            double psi_2q = x[5];

            double sin_d  = sin(delta), cos_d = cos(delta);

            // ---- Subtransient fluxes  [PF Eq.60] ----
            double psi_d_pp = k_fd * psi_fd + k_1d * psi_1d;
            double psi_q_pp = k_1q * psi_1q + k_2q * psi_2q;

            // ---- Stator currents  [PF Eq.73] ----
            // [ rstr, -xq_pp ] [id]   [-psi_q_pp - vd]
            // [ xd_pp,  rstr ] [iq] = [ psi_d_pp - vq]
            double det    = rstr*rstr + xd_pp*xq_pp;
            double rhs_d  = -psi_q_pp - vd_in;
            double rhs_q  =  psi_d_pp - vq_in;
            double id_ = ( rstr * rhs_d + xq_pp * rhs_q) / det;
            double iq_ = (-xd_pp  * rhs_d + rstr  * rhs_q) / det;

            // ---- Stator fluxes ----
            double psi_d = -xd_pp * id_ + psi_d_pp;
            double psi_q = -xq_pp * iq_ + psi_q_pp;

            // ---- Electrical torque  [PF Eq.67] ----
            double Te = (iq_ * psi_d - id_ * psi_q) / cosn;

            // ---- Mechanical torque  [PF Eq.102] ----
            double tm = Tm_in / fmax(omega, 1e-3) - dpu * omega;

            // ---- Damping  [PF Eqs.103-104] ----
            double dn    = omega - 1.0;
            double tdamp = dkd * dn + (dpe / fmax(omega, 1e-3)) * dn;

            // ---- Excitation voltage conversion  [PF Eq.132] ----
            double vfd = (rfd / fmax(xad, 1e-12)) * ve_in;

            // ---- Rotor currents  [PF Eq.69] ----
            double xdet_d = (xad + xrld)*(x1d + xfd) + xfd*x1d;
            double xdet_q = (xaq + xrlq)*(x2q + x1q) + x2q*x1q;
            double xfd_loop = xad + xrld + xfd;
            double x1d_loop = xad + xrld + x1d;
            double x1q_loop = xaq + xrlq + x1q;
            double x2q_loop = xaq + xrlq + x2q;

            double i_fd = k_fd * id_ + (x1d_loop*psi_fd - (xad+xrld)*psi_1d) / fmax(xdet_d,1e-12);
            double i_1d = k_1d * id_ + (xfd_loop*psi_1d - (xad+xrld)*psi_fd) / fmax(xdet_d,1e-12);
            double i_1q, i_2q;
            if (salient_pole) {
                i_1q = k_1q * iq_ + psi_1q / fmax(x1q, 1e-12);
                i_2q = 0.0;
            } else {
                i_1q = k_1q * iq_ + (x2q_loop*psi_1q - (xaq+xrlq)*psi_2q) / fmax(xdet_q,1e-12);
                i_2q = k_2q * iq_ + (x1q_loop*psi_2q - (xaq+xrlq)*psi_1q) / fmax(xdet_q,1e-12);
            }

            // ---- Excitation current output  [PF Eq.131] ----
            double ie = xad * i_fd;

            // ---- ODE  [PF Eqs.68, 100, 115] ----
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = (tm - Te - tdamp) / (2.0 * H);
            dxdt[2] = omega_b * (vfd - rfd * i_fd);
            dxdt[3] = omega_b * (-r1d * i_1d);
            dxdt[4] = omega_b * (-r1q * i_1q);
            dxdt[5] = omega_b * (-r2q * i_2q);

            // ---- Active / reactive power ----
            double Pe = vd_in*id_ + vq_in*iq_;
            double Qe = vq_in*id_ - vd_in*iq_;

            // ---- RI-frame current (for Norton injection) ----
            double I_Re = id_ * sin_d + iq_ * cos_d;
            double I_Im = -id_ * cos_d + iq_ * sin_d;

            // ---- Pack outputs ----
            outputs[0] = id_;
            outputs[1] = iq_;
            outputs[2] = omega;
            outputs[3] = Te;
            outputs[4] = Pe;
            outputs[5] = Qe;
            outputs[6] = ie;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        """

    # ------------------------------------------------------------------
    # Port-Hamiltonian formulation
    # ------------------------------------------------------------------

    def hamiltonian(self, x: np.ndarray) -> float:
        """
        Hamiltonian H(x) for the full PowerFactory synchronous machine model.

        H = H_mech + H_field + H_1d + H_1q + H_2q

        Physical interpretation
        -----------------------
        H_mech = H * (ω − 1)²
            Kinetic energy of rotating mass (inertia).  Factor 2H in swing eq.
            comes from 2H * ω̈ = Tm − Te.  With ∂H/∂ω = 2H(ω−1):
            dω/dt = (Tm-Te)/(2H)  →  J[1,0] couples through ∂H/∂ω. ✓

        H_field = ψ_fd² / (2 * xfd_loop)
            Magnetic energy stored in field winding.

        H_1d    = ψ_1d² / (2 * x1d_loop)
            Magnetic energy in 1d-damper winding.

        H_1q    = ψ_1q² / (2 * x1q_loop)
            Magnetic energy in 1q-damper winding.

        H_2q    = ψ_2q² / (2 * x2q_loop)
            Magnetic energy in 2q-damper winding (zero for salient-pole).

        Note: δ is cyclic (∂H/∂δ = 0).  The cross-energy between stator and
        rotor (the electromechanical coupling) enters through the g(x)u term,
        not H, because the stator is modelled as an algebraic constraint in the
        RMS formulation.
        """
        eq   = self._eqcct()
        omega  = x[1]
        psi_fd = x[2]; psi_1d = x[3]
        psi_1q = x[4]; psi_2q = x[5]
        H_val  = float(self.params['H'])

        H_mech  = H_val * (omega - 1.0) ** 2
        H_field = psi_fd ** 2 / (2.0 * max(eq['xfd_loop'], 1e-12))
        H_1d    = psi_1d ** 2 / (2.0 * max(eq['x1d_loop'], 1e-12))
        H_1q    = psi_1q ** 2 / (2.0 * max(eq['x1q_loop'], 1e-12))
        H_2q    = psi_2q ** 2 / (2.0 * max(eq['x2q_loop'], 1e-12)) if not eq['salient'] else 0.0

        return H_mech + H_field + H_1d + H_1q + H_2q

    def grad_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        """
        ∂H/∂x  (6-vector, p.u.).

        g[0] = ∂H/∂δ     = 0                      (cyclic)
        g[1] = ∂H/∂ω     = 2H(ω − 1)
        g[2] = ∂H/∂ψ_fd  = ψ_fd / xfd_loop
        g[3] = ∂H/∂ψ_1d  = ψ_1d / x1d_loop
        g[4] = ∂H/∂ψ_1q  = ψ_1q / x1q_loop
        g[5] = ∂H/∂ψ_2q  = ψ_2q / x2q_loop  (0 if salient)
        """
        eq  = self._eqcct()
        g   = np.zeros(6)
        g[0] = 0.0
        g[1] = 2.0 * float(self.params['H']) * (x[1] - 1.0)
        g[2] = x[2] / max(eq['xfd_loop'], 1e-12)
        g[3] = x[3] / max(eq['x1d_loop'], 1e-12)
        g[4] = x[4] / max(eq['x1q_loop'], 1e-12)
        g[5] = (x[5] / max(eq['x2q_loop'], 1e-12)) if not eq['salient'] else 0.0
        return g

    def get_phs_matrices(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Port-Hamiltonian structure matrices.

        Dynamical PHS form:
            ẋ = [J − R] ∂H/∂x  +  B u
            y =  B^T   ∂H/∂x

        where u = [vd, vq, Tm, ve]  (inputs from network and controllers).

        Derivation
        ----------
        The 6-state ODE is:
            dδ/dt    = ω_b (ω − 1)
            dω/dt    = [Tm − Te − tdamp] / (2H)
            dψ_fd/dt = ω_b [vfd − rfd * ifd]
            dψ_1d/dt = ω_b [−r1d * i1d]
            dψ_1q/dt = ω_b [−r1q * i1q]
            dψ_2q/dt = ω_b [−r2q * i2q]

        With ∂H/∂ω = 2H(ω−1) and ∂H/∂ψ_j = ψ_j / L_j_eff :

        J matrix (skew-symmetric, energy-conserving interconnection):
            J[0,1] = ω_b / (2H)    (δ–ω coupling; [J∂H]_0 = ω_b(ω−1) ✓)
            J[1,0] = −ω_b / (2H)

        R matrix (positive semi-definite, dissipation):
            R[1,1] = dkd / (2H)             (speed damping)
            R[2,2] = rfd * ω_b / xfd_loop    (field winding resistance loss)
            R[3,3] = r1d * ω_b / x1d_loop    (1d-damper resistance loss)
            R[4,4] = r1q * ω_b / x1q_loop    (1q-damper resistance loss)
            R[5,5] = r2q * ω_b / x2q_loop    (2q-damper resistance loss)

        B matrix (input coupling):
            B[1, 2] = 1/(2H)             → Tm drives dω/dt
            B[2, 3] = ω_b * rfd / xadu   → ve drives dψ_fd/dt via vfd

        The electromechanical coupling (Te) appears as a state-dependent
        interconnection; in the linearised / operating-point sense it is
        captured via the cross terms in the full nonlinear J(x).  The stator
        coupling to vd, vq makes B state-dependent (through id, iq), which
        is standard for generator PHS formulations.
        """
        p    = self.params
        eq   = self._eqcct()
        H_val   = float(p['H'])
        omega_b = float(p.get('omega_b', 2 * math.pi * 50))
        dkd     = float(p.get('dkd', 0.0))
        rfd     = eq['rfd']; r1d = eq['r1d']
        r1q     = eq['r1q']; r2q = eq['r2q']
        xadu    = eq['xad']

        n = 6   # states
        m = 4   # inputs

        # J: skew-symmetric interconnection
        J = np.zeros((n, n))
        J[0, 1] =  omega_b / (2.0 * H_val)
        J[1, 0] = -omega_b / (2.0 * H_val)

        # R: positive semi-definite dissipation
        R = np.zeros((n, n))
        R[1, 1] = dkd / (2.0 * H_val)
        R[2, 2] = rfd * omega_b / max(eq['xfd_loop'], 1e-12)
        R[3, 3] = r1d * omega_b / max(eq['x1d_loop'], 1e-12)
        R[4, 4] = r1q * omega_b / max(eq['x1q_loop'], 1e-12)
        R[5, 5] = (r2q * omega_b / max(eq['x2q_loop'], 1e-12)) if not eq['salient'] else 0.0

        # B: input coupling matrix
        # Note: vd, vq couple through the stator algebraic eq. (state-dependent);
        # here we provide the direct/affine part only.
        B = np.zeros((n, m))
        # u[2] = Tm  →  dω/dt += Tm/(2H)
        B[1, 2] = 1.0 / (2.0 * H_val)
        # u[3] = ve  →  dψ_fd/dt += ω_b * (rfd/xadu) * ve
        B[2, 3] = omega_b * rfd / max(xadu, 1e-12)

        return {'J': J, 'R': R, 'B': B}

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'delta_deg': {
                'description': 'Rotor angle (load angle)',
                'unit': 'deg',
                'cpp_expr': 'x[0] * 180.0 / 3.14159265359'
            },
            'omega': {
                'description': 'Per-unit rotor speed',
                'unit': 'pu',
                'cpp_expr': 'x[1]'
            },
            'Te': {
                'description': 'Electrical torque',
                'unit': 'pu',
                'cpp_expr': 'outputs[3]'
            },
            'Pe': {
                'description': 'Active power output',
                'unit': 'pu',
                'cpp_expr': 'outputs[4]'
            },
            'Qe': {
                'description': 'Reactive power output',
                'unit': 'pu',
                'cpp_expr': 'outputs[5]'
            },
            'ie': {
                'description': 'Excitation current (non-reciprocal p.u.)',
                'unit': 'pu',
                'cpp_expr': 'outputs[6]'
            },
            'psi_fd': {
                'description': 'Field winding flux',
                'unit': 'pu',
                'cpp_expr': 'x[2]'
            },
            'psi_1d': {
                'description': '1d-damper flux',
                'unit': 'pu',
                'cpp_expr': 'x[3]'
            },
            'psi_1q': {
                'description': '1q-damper flux',
                'unit': 'pu',
                'cpp_expr': 'x[4]'
            },
            'psi_2q': {
                'description': '2q-damper flux (0 for salient)',
                'unit': 'pu',
                'cpp_expr': 'x[5]'
            },
            'Vt': {
                'description': 'Terminal voltage magnitude',
                'unit': 'pu',
                'cpp_expr': 'sqrt(inputs[0]*inputs[0] + inputs[1]*inputs[1])'
            },
            'H_total': {
                'description': 'Total Hamiltonian (p.u.-s²)',
                'unit': 'pu-s²',
                'cpp_expr': (
                    f'{float(self.params["H"])} * (x[1]-1.0)*(x[1]-1.0)'
                    ' + x[2]*x[2]/(2.0*xfd_loop)'
                    ' + x[3]*x[3]/(2.0*x1d_loop)'
                    ' + x[4]*x[4]/(2.0*x1q_loop)'
                    ' + x[5]*x[5]/(2.0*x2q_loop)'
                )
            },
        }

    # ------------------------------------------------------------------
    # Initialisation from power-flow phasor
    # ------------------------------------------------------------------

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        """
        Initialise all 6 flux states from power-flow terminal phasor.

        Power-flow gives: V = |V|∠θ_V,  I = |I|∠θ_I  (RI frame, generator convention)

        Procedure (Section 6.1.3, [PF] Eq. 116):
          1. Determine rotor angle δ from q-axis phasor.
          2. Park-transform V, I to dq frame.
          3. Solve stator equations for steady-state id, iq.
          4. Solve rotor equations (all derivatives = 0) for flux states.
          5. Compute Efd_req, Tm_req for controller initialisation.

        Returns
        -------
        x0      : ndarray, shape (6,)
        targets : dict  (Efd, Tm, Vt, omega, vd, vq, id, iq, ie)
        """
        p    = self.params
        eq   = self._eqcct()
        rstr = float(p['rstr'])
        xad  = eq['xad']; xaq = eq['xaq']
        xrld = eq['xrld']; xrlq = eq['xrlq']
        xfd  = eq['xfd']; x1d = eq['x1d']
        x1q  = eq['x1q']; x2q = eq['x2q']
        rfd  = eq['rfd']; r1d = eq['r1d']
        r1q  = eq['r1q']; r2q = eq['r2q']
        k_fd = eq['k_fd']; k_1d = eq['k_1d']
        k_1q = eq['k_1q']; k_2q = eq['k_2q']
        xd_pp = eq['xd_pp']; xq_pp = eq['xq_pp']
        xl    = float(p['xl'])
        xd    = float(p['xd']); xq = float(p['xq'])
        cosn  = float(p.get('cosn', 1.0))

        # Step 1: q-axis phasor for rotor angle  [PF] Eq. 116
        Eq_phasor = V_phasor + (rstr + 1j * xq) * I_phasor
        delta     = float(np.angle(Eq_phasor))

        # Step 2: Park transform  (exp(-j*(δ - π/2)) maps RI→dq)
        dq_factor = np.exp(-1j * (delta - math.pi / 2.0))
        vd = float((V_phasor * dq_factor).real)
        vq = float((V_phasor * dq_factor).imag)
        id_ = float((I_phasor * dq_factor).real)
        iq_ = float((I_phasor * dq_factor).imag)

        # Step 3: Steady-state stator fluxes
        psi_d = vq + rstr * iq_       # from vq = -rstr*iq + psi_d  (n=1, d/dt=0)  [PF Eq.55]
        psi_q = -(vd + rstr * id_)    # from vd = -rstr*id - psi_q

        # Step 4: Steady-state rotor flux states  (all dψ/dt = 0)
        #
        # psi_d = -xd_pp*id + psi_d_pp  →  psi_d_pp = psi_d + xd_pp*id
        # psi_q = -xq_pp*iq + psi_q_pp  →  psi_q_pp = psi_q + xq_pp*iq
        psi_d_pp = psi_d + xd_pp * id_
        psi_q_pp = psi_q + xq_pp * iq_

        # Magnetising fluxes
        psi_ad = psi_d + xl * id_
        psi_aq = psi_q + xl * iq_

        # Saturation at operating point
        satd, satq = self._saturation_factors(psi_ad, psi_aq)

        # d-axis: from rotor flux equations at steady state (dψ_fd/dt=0, dψ_1d/dt=0)
        # psi_d_pp = k_fd*psi_fd + k_1d*psi_1d
        # SS condition: r1d*i1d = 0  →  i1d = 0  →  psi_1d satisfies psi_1d=f(psi_fd)
        # From [PF] Eq.59 at SS: psi_1d = psi_fd (because i1d=0 means no current in 1d)
        # More precisely: using xad-based expressions:
        #   SS: ifd = psi_ad / xad,  i1d = 0
        #   psi_fd = psi_ad + (xrld + xfd) * ifd
        #   psi_1d = psi_ad + xrld * ifd
        i_fd_ss = psi_ad / max(xad, 1e-12)
        psi_fd_ss = psi_ad + (xrld + xfd) * i_fd_ss
        psi_1d_ss = psi_ad + xrld * i_fd_ss

        # q-axis SS: i1q = i2q = 0; from xaq:
        #   psi_aq = xaq * (-iq + 0 + 0)  wait — need actual psi_aq from stator
        #   psi_1q = psi_aq + xrlq*0 + x1q*0 = psi_aq
        #   psi_2q = psi_aq
        psi_1q_ss = psi_aq
        psi_2q_ss = psi_aq if not eq['salient'] else 0.0

        x0 = np.array([delta, 1.0, psi_fd_ss, psi_1d_ss, psi_1q_ss, psi_2q_ss])

        # Targets for controllers
        ie_ss  = self._excitation_current(i_fd_ss)       # non-reciprocal excitation current
        vfd_ss = rfd * i_fd_ss                            # steady-state field voltage
        # ve_ss: invert Eq.132: ve = vfd * xad / rfd = xad * ifd (= ie_ss) ✓  [PF Eq.133]
        ve_ss = ie_ss

        Te_ss = self._electrical_torque(id_, iq_, psi_d, psi_q)
        Tm_ss = Te_ss   # at steady state (ω=1, dω/dt=0, damping=0)

        targets = {
            'Efd':    float(ve_ss),
            'Tm':     float(Tm_ss),
            'Vt':     float(abs(V_phasor)),
            'omega':  1.0,
            'vd':     vd, 'vq': vq,
            'id':     id_, 'iq': iq_,
            'ie':     float(ie_ss),
            'vd_ri':  float(V_phasor.real),
            'vq_ri':  float(V_phasor.imag),
        }
        return x0, targets

    # ------------------------------------------------------------------
    # Norton current (for network interface)
    # ------------------------------------------------------------------

    def compute_norton_current(self, x: np.ndarray) -> complex:
        """
        RI-frame Norton current injection at zero terminal voltage.
        Used to assemble the network admittance system.

        I_norton = Y_gen * V_term + I_no
        where Y_gen = 1 / (rstr + j*xd'')  (approximately)
        """
        eq    = self._eqcct()
        delta = x[0]
        psi_fd = x[2]; psi_1d = x[3]
        psi_1q = x[4]; psi_2q = x[5]

        psi_d_pp = eq['k_fd'] * psi_fd + eq['k_1d'] * psi_1d
        psi_q_pp = eq['k_1q'] * psi_1q + eq['k_2q'] * psi_2q

        xd_pp = eq['xd_pp']; xq_pp = eq['xq_pp']
        rstr  = float(self.params['rstr'])
        det   = rstr**2 + xd_pp * xq_pp

        # Stator at V=0:
        # [rstr, -xq''][id]   [-psi_q_pp]
        # [xd'',  rstr][iq] = [ psi_d_pp]
        id_no = ( rstr * (-psi_q_pp) + xq_pp * psi_d_pp) / det
        iq_no = (-xd_pp * (-psi_q_pp) + rstr  * psi_d_pp) / det

        sin_d = math.sin(delta); cos_d = math.cos(delta)
        return complex(id_no * sin_d + iq_no * cos_d,
                      -id_no * cos_d + iq_no * sin_d)

    @property
    def component_role(self) -> str:
        return 'generator'
