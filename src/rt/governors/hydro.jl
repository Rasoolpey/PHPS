# =====================================================================================================================
# The hydro unit (PHHydroUnit: IEEEG3 governor, penstock, ideal turbine) on the FPGA: one build for the type
# =====================================================================================================================
# Record: docs/realtime/models/HYDRO.md; plan and reviews: docs/realtime/reviews/fpga/43*_hydro_idea_*.md (go at 43d).

# ---------------------------------------------------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------------------------------------------------
# Source: PHHydroUnit, src/phsim/hydro.jl (`_hydro_rhs!`, `_hydro_power`, `_hydro_powers`); limiter primitives
# src/components/primitives.jl 88-120. States x = (xp, xr, at, q); input omega; Pref a parameter.
#   at_l = clamp_mode(at, GMIN, GMAX), xp_l = clamp_mode(xp, RDN, RUP)
#   e = Pref - R (omega - 1) - Sigma at_l - Delta (at_l - xr)
#   xp' = outband_relax(nonwindup(xp_l, (e / Tg - xp_l) / Tp)),  xr' = (at_l - xr) / Tr,  at' = outband_relax(nonwindup(at_l, xp_l))
#   Tw q' = 1 - c q^2, c = (GMAX / at_l)^2;  output P = GMAX c_P q^3 with the gate reclamped at the midpoint (unheld)
# The twelve decision sites are taken at x_n with the step's input and held (ModeLog / ModeReplay). The governor rows are
# then affine in z = (xp, xr, at): 16 J-structures from (A_in, X_in, NX, NA); the clamped constants xpc, atc enter as inputs.
# The penstock: midpoint quadratic q_m = q + b (1 - c q_m^2), b = h / (2 Tw): delta = A - A D, A = b - b c q^2, D = 2 b c q.
# One reciprocal of the midpoint gate per step (ROM 1024 + 2 Newton); row c held, output c reclamped (43b, 43c).

const HYDRO_RT_DIR = joinpath(@__DIR__)
const _HYPKG = parentmodule(@__MODULE__)   # the package (its PHHydroUnit rows for the corner parity check)
const Bt = Float64           # b = h / (2 Tw) (C++ ap_fixed<48, -20>)
const Cw = Float64           # c = (GMAX / gate)^2, heads (ap_fixed<48, 11>)
const Gc = Float64           # GMAX c_P (ap_fixed<48, 13>)
const Rr = Float64           # the reciprocal r = 1 / gate (ap_fixed<48, 4>)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Data and preparation (BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""    hydro_rt_data(i): unit i of src/rt/governors/data/ieee39_hydro.toml; i = 0: the synthetic unit (the case's with
PMIN = 0.1: GMIN = 0.85) for the lower-limit corners (43, 5)."""
function hydro_rt_data(i::Int = 1; path = joinpath(HYDRO_RT_DIR, "data", "ieee39_hydro.toml"))
    d = copy(TOML.parsefile(path)["unit"][max(i, 1)])
    if i == 0
        d["PMIN"] = 0.1; d["name"] *= " (synthetic, PMIN 0.1)"; d["machine"] *= "*"
    end
    d["GMIN"] = d["PMIN"] * d["R_base"]; d["RUP"] = d["UO"] * d["R_base"]; d["RDN"] = d["UC"] * d["R_base"]
    return d
end
hydro_rt_ninstances(; path = joinpath(HYDRO_RT_DIR, "data", "ieee39_hydro.toml")) = length(TOML.parsefile(path)["unit"])

# the bank index of the decision bits (A_in, X_in, NX, NA), 1..16
_hy_bank(ain, xin, nx, na) = 1 + 8ain + 4xin + 2nx + na

# the affine rows z' = J z + g (omega - 1) + kp + ex xpc + ea atc of one decision structure (BigFloat)
function _hy_rows(d, ain::Bool, xin::Bool, nx::Bool, na::Bool)
    b(k) = BigFloat(d[k])
    Tg, Tp, s, dl, Tr, R, Pref = b("Tg"), b("Tp"), b("Sigma"), b("Delta"), b("Tr"), b("R_base"), b("Pref")
    J = zeros(BigFloat, 3, 3); g = zeros(BigFloat, 3); kp = zeros(BigFloat, 3); ex = zeros(BigFloat, 3); ea = zeros(BigFloat, 3)
    if !nx                                   # (e / Tg - xp) / Tp (in band, or out of band with the relax: the same row)
        J[1, 1] = -1 / Tp; J[1, 2] = dl / (Tg * Tp); g[1] = -R / (Tg * Tp); kp[1] = Pref / (Tg * Tp)
        ain ? (J[1, 3] = -(s + dl) / (Tg * Tp)) : (ea[1] = -(s + dl) / (Tg * Tp))
    elseif !xin                              # frozen out of band: the relax (xpc - xp) / Tp only
        J[1, 1] = -1 / Tp; ex[1] = 1 / Tp
    end                                      # frozen in band (xp exactly at its limit): xp' = 0
    J[2, 2] = -1 / Tr
    ain ? (J[2, 3] = 1 / Tr) : (ea[2] = 1 / Tr)
    if !na
        xin ? (J[3, 1] = 1) : (ex[3] = 1)
    end
    if !ain                                  # the gate out of band: relax (atc - at) / Tg
        J[3, 3] -= 1 / Tg; ea[3] += 1 / Tg
    end
    return J, g, kp, ex, ea
end

# the midpoint step of one structure: d = P z + W (omega - 1) + K + EX xpc + EA atc, M = (I - h J / 2)^-1 (BigFloat)
function _hy_bankmats(d, h::BigFloat, ain, xin, nx, na)
    J, g, kp, ex, ea = _hy_rows(d, ain, xin, nx, na)
    M = inv(Matrix{BigFloat}(I, 3, 3) - h / 2 * J)
    return (; P = M * h * J, W = M * h * g, K = M * h * kp, EX = M * h * ex, EA = M * h * ea)
end
_hy_bits(bk) = (ain = (bk - 1) >> 3 & 1 == 1, xin = (bk - 1) >> 2 & 1 == 1, nx = (bk - 1) >> 1 & 1 == 1, na = (bk - 1) & 1 == 1)
_hy_bigbanks(d, h::BigFloat) = [(bt = _hy_bits(bk); _hy_bankmats(d, h, bt.ain, bt.xin, bt.nx, bt.na)) for bk in 1:16]

# ---------------------------------------------------------------------------------------------------------------------
# 3. The coefficients (the C++ struct HydroCoefs) and the reciprocal's seed ROM
# ---------------------------------------------------------------------------------------------------------------------
@hls struct HydroCoefs
    P::Small[16, 9]     # bank = 1 + 8 A_in + 4 X_in + 2 NX + NA: h M J, row by row
    W::Small[16, 3]     # (omega - 1)
    K::Small[16, 3]     # Pref
    EX::Small[16, 3]    # the clamped pilot valve xpc
    EA::Small[16, 3]    # the clamped gate atc
    kc::Coef[11]        # 1 Pref, 2 R_base, 3 Sigma + Delta, 4 Delta, 5 Tg, 6 RUP, 7 RDN, 8 GMAX, 9 GMIN, 10 0.05 GMAX, 11 1.05 GMAX
    kb::Bt[1]           # b = h / (2 Tw)
    kw::Cw[1]           # c at GMIN, (GMAX / GMIN)^2 (0 when GMIN = 0: never valid)
end

const HYDRO_RT_I = (St = 6, Sig = 8, Small = -7, Inc = -4, Coef = 8, Bt = -20, Rq = 4, Rr = 8, Cw = 11, Gc = 13, Pw = 8)
const HYDRO_RT_FIELDS = [:P => :Small, :W => :Small, :K => :Small, :EX => :Small, :EA => :Small, :kc => :Coef, :kb => :Bt,
                         :kw => :Cw]

function hydro_rt_coefstruct(d, h::BigFloat; grid = true)
    r(v, T) = grid ? _q48(v, getfield(HYDRO_RT_I, T)) : Float64(v)
    P = zeros(16, 9); W = zeros(16, 3); K = zeros(16, 3); EX = zeros(16, 3); EA = zeros(16, 3)
    for (bk, m) in enumerate(_hy_bigbanks(d, h)), i in 1:3
        for j in 1:3; P[bk, 3(i-1)+j] = r(m.P[i, j], :Small); end
        W[bk, i] = r(m.W[i], :Small); K[bk, i] = r(m.K[i], :Small); EX[bk, i] = r(m.EX[i], :Small); EA[bk, i] = r(m.EA[i], :Small)
    end
    b(k) = BigFloat(d[k])
    kc = [r(v, :Coef) for v in (b("Pref"), b("R_base"), b("Sigma") + b("Delta"), b("Delta"), b("Tg"), b("RUP"), b("RDN"),
                                b("GMAX"), b("GMIN"), b("GMAX") / 20, b("GMAX") * 21 / 20)]
    kb = [r(h / (2 * b("Tw")), :Bt)]
    kw = [r(d["GMIN"] > 0 ? (b("GMAX") / b("GMIN"))^2 : zero(BigFloat), :Cw)]
    return HydroCoefs(P, W, K, EX, EA, kc, kb, kw)
end

# seeds of 1/m, m in [1/2, 1), 1024 bins: 2 / (m_lo + m_hi) (relative error <= 2^-11), rounded to Rq
hydro_rt_seeds() = [_q48(2 / (BigFloat(1024 + j) / 2048 + BigFloat(1025 + j) / 2048), HYDRO_RT_I.Rq) for j in 0:1023]
hls_rom!(:HY_SEED, hydro_rt_seeds())

# ---------------------------------------------------------------------------------------------------------------------
# 4. THE RECIPROCAL, THE DECISIONS AND THE STEP (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
# 1 / a for 2^-6 <= a < 16: a = m 2^k, m in [1/2, 1) (exact scalings), seed from the ROM, two Newton steps y <- y (2 - m y)
# (error <= 2^-44 before rounding), 1 / a = y 2^-k
@hls function hy_recip(a::Sig)::Rr
    seed::Rq[1024] = rom(:HY_SEED)
    two::Rq = 2
    m::Rq = 0
    k::Int = 0
    c8::Sig = 8
    c4::Sig = 4
    c2::Sig = 2
    c1::Sig = 1
    ch::Sig = 0.5
    cq::Sig = 0.25
    ce::Sig = 0.125
    c16::Sig = 0.0625
    c32::Sig = 0.03125
    c64::Sig = 0.015625
    if a >= c8
        m = scale2(a, -4)
        k = 4
    elseif a >= c4
        m = scale2(a, -3)
        k = 3
    elseif a >= c2
        m = scale2(a, -2)
        k = 2
    elseif a >= c1
        m = scale2(a, -1)
        k = 1
    elseif a >= ch
        m = a
        k = 0
    elseif a >= cq
        m = scale2(a, 1)
        k = -1
    elseif a >= ce
        m = scale2(a, 2)
        k = -2
    elseif a >= c16
        m = scale2(a, 3)
        k = -3
    elseif a >= c32
        m = scale2(a, 4)
        k = -4
    elseif a >= c64
        m = scale2(a, 5)
        k = -5
    else
        m = scale2(a, 6)
        k = -6
    end
    j::Int = ifloor(scale2(m, 11)) - 1024
    if j < 0
        j = 0
    end
    if j > 1023
        j = 1023
    end
    y::Rq = seed[j + 1]
    for it in 1:2
        t::Rq = m * y
        u::Rq = two - t
        y = y * u
    end
    r::Rr = y
    if k == 4
        r = scale2(y, -4)
    elseif k == 3
        r = scale2(y, -3)
    elseif k == 2
        r = scale2(y, -2)
    elseif k == 1
        r = scale2(y, -1)
    elseif k == -1
        r = scale2(y, 1)
    elseif k == -2
        r = scale2(y, 2)
    elseif k == -3
        r = scale2(y, 3)
    elseif k == -4
        r = scale2(y, 4)
    elseif k == -5
        r = scale2(y, 5)
    elseif k == -6
        r = scale2(y, 6)
    end
    return r
end

# clamp_mode's two sites: 0 within, 1 above (v > hi), 2 below (v < lo, tested on the value after the first site)
@hls function hy_region(v::St, lo::Coef, hi::Coef)::Int
    rg::Int = 0
    if v > hi
        rg = 1
    elseif v < lo
        rg = 2
    end
    return rg
end

# nonwindup's two sites: 1 when frozen (vl >= hi and rate > 0, or vl <= lo and rate < 0); the rate's sign as pos / neg
# (1 or 0), formed in the rate's own type: no narrowing of the rate (review 44, 2)
@hls function hy_freeze(vl::St, pos::Int, neg::Int, lo::Coef, hi::Coef)::Int
    fz::Int = 0
    if vl >= hi && pos == 1
        fz = 1
    elseif vl <= lo && neg == 1
        fz = 1
    end
    return fz
end

# one step of h: x = (xp, xr, at, q) changed in place; u = (omega); y = (P) at the midpoint; md = (bank, domain flag,
# crossing code: 0 none, 1 held-in gate out at the midpoint, 2 held-out gate back in)
@hls function hydro_step!(x::St[4], xc::Small[4], u::Sig[1], y::Pw[1], md::Int[3], kP::Small[16, 9], kW::Small[16, 3], kK::Small[16, 3],
                          kEX::Small[16, 3], kEA::Small[16, 3], kc::Coef[11], kb::Bt[1], kw::Cw[1])
    om::Sig = u[1]
    one::Sig = 1
    om1::Sig = om - one
    # the decisions at x_n with this step's input (held for the step)
    ar::Int = hy_region(x[3], kc[9], kc[8])
    xr_::Int = hy_region(x[1], kc[7], kc[6])
    atl::St = x[3]                                   # St: a state one LSB off its limit stays off it (C++ corners)
    atc::Coef = 0
    ain::Int = 1
    if ar == 1
        atl = kc[8]
        atc = kc[8]
        ain = 0
    elseif ar == 2
        atl = kc[9]
        atc = kc[9]
        ain = 0
    end
    xpl::St = x[1]
    xpc::Coef = 0
    xin::Int = 1
    if xr_ == 1
        xpl = kc[6]
        xpc = kc[6]
        xin = 0
    elseif xr_ == 2
        xpl = kc[7]
        xpc = kc[7]
        xin = 0
    end
    # the pilot valve's rate sign: s = e - Tg xp_l = Pref - R (omega - 1) - (Sigma + Delta) at_l + Delta xr - Tg xp_l
    e1::Sig = kc[2] * om1
    e2::Sig = kc[3] * atl
    e3::Sig = kc[4] * x[2]
    e4::Sig = kc[5] * xpl
    ea::Sig = kc[1] - e1
    eb::Sig = e3 - e2
    ec::Sig = ea - e4
    s::Sig = ec + eb
    sp::Int = 0
    sn::Int = 0
    if s > 0
        sp = 1
    elseif s < 0
        sn = 1
    end
    xq::Int = 0
    xn_::Int = 0
    if xpl > 0
        xq = 1
    elseif xpl < 0
        xn_ = 1
    end
    nx::Int = hy_freeze(xpl, sp, sn, kc[7], kc[6])
    na::Int = hy_freeze(atl, xq, xn_, kc[9], kc[8])
    bk::Int = 1 + 8 * ain + 4 * xin + 2 * nx + na
    # the governor's increment (one product level on x after the bank choice)
    d::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=d complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        l1::Inc = kP[bk, 3 * i - 2] * x[1]
        l2::Inc = kP[bk, 3 * i - 1] * x[2]
        l3::Inc = kP[bk, 3 * i] * x[3]
        l4::Inc = kW[bk, i] * om1
        l5::Inc = kEX[bk, i] * xpc
        l6::Inc = kEA[bk, i] * atc
        la::Inc = l1 + l2
        lb::Inc = l3 + l4
        lc::Inc = l5 + l6
        ld::Inc = lc + kK[bk, i]
        le::Inc = la + lb
        d[i] = le + ld
    end
    # the midpoint gate; one reciprocal (its operand a safe constant when neither selection needs it, 43c)
    atm::St = x[3] + scale2(d[3], -1)
    inr::Int = 0
    if atm >= kc[9] && atm <= kc[8]
        inr = 1
    end
    ro::Sig = kc[8]
    if ain == 1 || inr == 1
        ro = atm
    end
    r::Rr = hy_recip(ro)
    g1::Sig = kc[8] * r
    c::Cw = g1 * g1
    crow::Cw = c                                     # the row: held clamp
    if ar == 1
        crow = 1
    elseif ar == 2
        crow = kw[1]
    end
    cP::Cw = c                                       # the output: the gate reclamped at the midpoint (the package's)
    if inr == 0
        if atm > kc[8]
            cP = 1
        else
            cP = kw[1]
        end
    end
    # the penstock: delta = A - A D
    q::St = x[4]
    bc::Small = kb[1] * crow
    q2::St = q * q
    bq2::Small = bc * q2
    aa::Small = kb[1] - bq2
    bq::Small = bc * q
    dd::Small = scale2(bq, 1)
    ad::Small = aa * dd
    dl::Small = aa - ad
    qm::St = q + dl
    # the output at the midpoint
    qm2::St = qm * qm
    qm3::St = qm2 * qm
    gc::Gc = kc[8] * cP
    y[1] = gc * qm3
    # the domain guards at the midpoint (43c): a sticky flag in the board top
    fl::Int = 0
    atr::St = atm
    if ain == 0
        atr = atc
    end
    if atr < kc[10] || atr > kc[11]
        fl = 1
    end
    hr::Cw = crow * qm2
    hp::Cw = cP * qm2
    four::Cw = 4
    if hr > four || hp > four
        fl = 1
    end
    if inr == 1 && atm < kc[10]
        fl = 1
    end
    zero::St = 0
    qmax::St = 1.5
    if qm <= zero || qm > qmax
        fl = 1
    end
    omin::Sig = 0.5
    if om < omin
        fl = 1
    end
    cr::Int = 0
    if ain == 1 && inr == 0
        cr = 1
    elseif ain == 0 && inr == 1
        cr = 2
    end
    # compensated summation (Kahan): the increment plus the last remainder; the new remainder exactly (a constant
    # increment, e.g. the rate-limited gate, otherwise rounds the same way every step and drifts)
    dq::Inc = scale2(dl, 1)
    for i in 1:4
        @pragma "HLS UNROLL"
        di::Inc = dq
        if i < 4
            di = d[i]
        end
        ei::Inc = di + xc[i]
        xn::St = x[i] + ei
        xc[i] = x[i] - xn + ei                       # (x - x_new) + e: exact in Float64 too (x + e - x_new gave 0)
        x[i] = xn
    end
    md[1] = bk
    md[2] = fl
    md[3] = cr
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. THE BOARD TOP FUNCTION
# ---------------------------------------------------------------------------------------------------------------------
# As the exciters: input omega in 4 segments (us the first step's midpoint value, du = h omega'); flg[1] a sticky domain
# flag (written 0 by the host, 1 if any step left the domain, 43c). A call returns the four states.
@hls function hydro_run!(k::HydroCoefs, x::St[4], xc::Small[4], us::Sig[4], du::Small[4], n_on::Int, n_mid::Int, n_off::Int, n0::Int,
                         nsteps::Int, decim::Int, y::Pw[1], flg::Int[1], at_out::Out{St}, p_out::Out{Pw}, strobe::Out{Bool},
                         sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=xc"
    @pragma "HLS INTERFACE mode=s_axilite port=us"
    @pragma "HLS INTERFACE mode=s_axilite port=du"
    @pragma "HLS INTERFACE mode=s_axilite port=n_on"
    @pragma "HLS INTERFACE mode=s_axilite port=n_mid"
    @pragma "HLS INTERFACE mode=s_axilite port=n_off"
    @pragma "HLS INTERFACE mode=s_axilite port=n0"
    @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=s_axilite port=flg"
    @pragma "HLS INTERFACE mode=ap_none port=at_out"
    @pragma "HLS INTERFACE mode=ap_none port=p_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::St[4]
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    cs::Small[4]                                     # the remainders of the compensated summation
    @pragma "HLS ARRAY_PARTITION variable=cs complete"
    for i in 1:4
        @pragma "HLS UNROLL"
        xs[i] = x[i]
        cs[i] = xc[i]
    end
    u::Sig[1]
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    yy::Pw[1]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    md::Int[3]
    @pragma "HLS ARRAY_PARTITION variable=md complete"
    kP::Small[16, 9]                                 # the coefficients in registers for the run
    @pragma "HLS ARRAY_PARTITION variable=kP complete dim=0"
    kW::Small[16, 3]
    @pragma "HLS ARRAY_PARTITION variable=kW complete dim=0"
    kK::Small[16, 3]
    @pragma "HLS ARRAY_PARTITION variable=kK complete dim=0"
    kEX::Small[16, 3]
    @pragma "HLS ARRAY_PARTITION variable=kEX complete dim=0"
    kEA::Small[16, 3]
    @pragma "HLS ARRAY_PARTITION variable=kEA complete dim=0"
    kc::Coef[11]
    @pragma "HLS ARRAY_PARTITION variable=kc complete"
    kb::Bt[1]
    kw::Cw[1]
    for b in 1:16
        for j in 1:9
            kP[b, j] = k.P[b, j]
        end
        for j in 1:3
            kW[b, j] = k.W[b, j]
            kK[b, j] = k.K[b, j]
            kEX[b, j] = k.EX[b, j]
            kEA[b, j] = k.EA[b, j]
        end
    end
    for j in 1:11
        kc[j] = k.kc[j]
    end
    kb[1] = k.kb[1]
    kw[1] = k.kw[1]
    fs::Int = flg[1]
    st::Bool = false
    cnt::Int = 0
    pat::St = xs[3]
    ppw::Pw = 0
    psa::Bool = false
    pst::Bool = false
    for i in 1:nsteps
        @pragma "HLS PIPELINE II=24"                 # HYDRO_RT_STEP_CYCLES (checked at export; sweeps 2-3: 21-23 below 125 MHz)
        @pragma "HLS ALLOCATION operation instances=mul limit=12"
        at_out[] = pat
        p_out[] = ppw
        sample[] = psa
        strobe[] = pst
        m1::Int = n0 + i - 1
        sg::Int = 1
        s0::Int = 0
        if m1 >= n_on
            sg = 2
            s0 = n_on
        end
        if m1 >= n_mid
            sg = 3
            s0 = n_mid
        end
        if m1 >= n_off
            sg = 4
            s0 = n_off
        end
        ns::Int = m1 - s0
        u[1] = us[sg] + du[sg] * ns
        hydro_step!(xs, cs, u, yy, md, kP, kW, kK, kEX, kEA, kc, kb, kw)
        if md[2] == 1
            fs = 1
        end
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            psa = true
        else
            psa = false
        end
        pat = xs[3]
        ppw = yy[1]
        pst = st
    end
    for i in 1:4
        @pragma "HLS UNROLL"
        x[i] = xs[i]
        xc[i] = cs[i]
    end
    y[1] = yy[1]
    flg[1] = fs
end

# ---------------------------------------------------------------------------------------------------------------------
# 6. Export: bench, checks, C++, testbench, board program
# ---------------------------------------------------------------------------------------------------------------------
const HYDRO_RT_CLOCK_NS = 8
const HYDRO_RT_STEP_CYCLES = 24
const HYDRO_RT_H = "$(HYDRO_RT_STEP_CYCLES * HYDRO_RT_CLOCK_NS)e-9"
const HYDRO_RT_TYPES = [:St => "ap_fixed<48, 6, AP_RND_CONV>",          # xp, xr, at, q: +-32, grid 2.3e-13
                        :Sig => "ap_fixed<48, 8, AP_RND_CONV>",         # omega, e, gate at the midpoint, GMAX r: +-128
                        :Small => "ap_fixed<48, -7, AP_RND_CONV>",      # prepared entries, b c, A, D, delta: +-3.9e-3
                        :Inc => "ap_fixed<48, -4, AP_RND_CONV>",        # increments: +-0.031
                        :Coef => "ap_fixed<48, 8, AP_RND_CONV>",        # gains, limits: +-128
                        :Bt => "ap_fixed<48, -20, AP_RND_CONV>",        # b = h / (2 Tw) ~ 4e-8: +-4.8e-7, grid 1.7e-21
                        :Rq => "ap_fixed<48, 4, AP_RND_CONV>",          # normalised m, seed, Newton (up to 2): +-8
                        :Rr => "ap_fixed<48, 8, AP_RND_CONV>",          # r = 1 / gate up to 64 (C++ sweep: +-8 wrapped at 1/0.125)
                        :Cw => "ap_fixed<48, 11, AP_RND_CONV>",         # c, heads: +-1024
                        :Gc => "ap_fixed<48, 13, AP_RND_CONV>",         # GMAX c_P: +-4096
                        :Pw => "ap_fixed<48, 8, AP_RND_CONV>"]          # P <= 51: +-128
const HYDRO_RT_SCALE = (x = 10.0, P = 10.0)
const HYDRO_RT_LIM = 5e-8
const HYDRO_RT_BOUND = 1e-15

function hydro_rt_bench(d, case::Symbol, h::Float64)
    non, nmid, noff = round(Int, 0.1 / h), round(Int, 0.15 / h), round(Int, 0.2 / h)
    w0 = d["omega"]
    o = case === :A ? [(w0, w0), (w0, 0.97), (0.97, w0), (w0, w0)] :
        case === :B ? [(w0, w0), (w0, 1.03), (1.03, w0), (w0, w0)] : [(w0, w0) for _ in 1:4]
    case === :C && (non = nmid = noff = 1 << 30)
    len = [non, nmid - non, noff - nmid, 1]
    us = zeros(4); du = zeros(4)
    for s in 1:4
        rate = case === :C ? 0.0 : (o[s][2] - o[s][1]) / len[s]
        du[s] = _q48(rate, HYDRO_RT_I.Small); us[s] = _q48(o[s][1] + rate / 2, HYDRO_RT_I.Sig)
    end
    return (; non, nmid, noff, us, du)
end

_hy_x0(d) = map(v -> _q48(v, HYDRO_RT_I.St), d["x"])
_hy_input(b, m1) = m1 >= b.noff ? (4, b.noff) : m1 >= b.nmid ? (3, b.nmid) : m1 >= b.non ? (2, b.non) : (1, 0)
_hy_u(b, m1) = (t = _hy_input(b, m1); [b.us[t[1]] + b.du[t[1]] * (m1 - t[2])])
_hy_call!(k, x, b, n0, nsteps; decim = 32, flg = [0], xc = zeros(4)) =
    hydro_run!(k, x, xc, b.us, b.du, b.non, b.nmid, b.noff, n0, nsteps, decim, zeros(1), flg, Ref(0.0), Ref(0.0), Ref(false), Ref(false))

function _hydro_rt_julia(k, x0, b, cuts)
    x = copy(x0); xc = zeros(4); rec = [copy(x)]; outs = [zeros(1)]; y = zeros(1); n = 0
    for c in cuts
        hydro_run!(k, x, xc, b.us, b.du, b.non, b.nmid, b.noff, n, c - n, 32, y, [0], Ref(0.0), Ref(0.0), Ref(false), Ref(false))
        push!(rec, copy(x)); push!(outs, copy(y)); n = c
    end
    return rec, outs
end

function _hydro_rt_trace(k, x0, b, nsteps)
    x = copy(x0); xc = zeros(4); y = zeros(1); md = zeros(Int, 3)
    X = Vector{Vector{Float64}}(); Xn = similar(X); U = similar(X); Y = similar(X); M = Vector{Vector{Int}}()
    for n in 1:nsteps
        u = _hy_u(b, n - 1)
        push!(X, copy(x)); push!(U, u)
        hydro_step!(x, xc, u, y, md, k.P, k.W, k.K, k.EX, k.EA, k.kc, k.kb, k.kw)
        push!(Xn, copy(x)); push!(Y, copy(y)); push!(M, copy(md))
    end
    return (; X, Xn, U, Y, M)
end

function _hydro_rt_cuts(h, b, tr = nothing)
    nend = round(Int, 0.3 / h)
    c = vcat([round(Int, g * 0.01 / h) for g in 1:30], b.non-1:b.non+2, b.nmid-1:b.nmid+2, b.noff-1:b.noff+2)
    if tr !== nothing
        for n in 2:length(tr.M)
            tr.M[n][1] != tr.M[n-1][1] && append!(c, n-3:n+2)
        end
    end
    return sort(unique(filter(v -> 0 < v <= nend, c)))
end

# the reciprocal in the C++ arithmetic (Julia mirror, every assignment rounded to its type) -> (r, j, k)
function _hy_recip_q(a::Float64)
    q(v, T) = _q48(v, getfield(HYDRO_RT_I, T))
    k = a >= 8 ? 4 : a >= 4 ? 3 : a >= 2 ? 2 : a >= 1 ? 1 : a >= 0.5 ? 0 : a >= 0.25 ? -1 : a >= 0.125 ? -2 :
        a >= 0.0625 ? -3 : a >= 0.03125 ? -4 : a >= 0.015625 ? -5 : -6
    m = q(a * 2.0^-k, :Rq); j = clamp(floor(Int, m * 2048) - 1024, 0, 1023)
    y = HLS_ROMS[:HY_SEED][j+1]
    for _ in 1:2
        t = q(m * y, :Rq); u = q(2 - t, :Rq); y = q(y * u, :Rq)
    end
    return q(y * 2.0^-k, :Rr), j, k
end

function _hy_recip_sweep()
    g = 2.0^-40; pts = Float64[]
    for k in -6:3, j in 0:1024
        a = (0.5 + j / 2048) * 2.0^(k + 1)
        append!(pts, (a - g, a, a + g))
    end
    rng = Random.MersenneTwister(43); append!(pts, 2.0^-6 .+ (16 - 2.0^-6) .* rand(rng, 1_000_000))
    w = 0.0; wf = 0.0; at = 0.0
    for a in filter(v -> 2.0^-6 <= v < 16, pts)
        aq = _q48(a, HYDRO_RT_I.Sig)
        r, _, _ = _hy_recip_q(aq); e = Float64(abs(big(r) * big(aq) - 1))
        e > w && (w = e; at = aq)
        wf = max(wf, Float64(abs(big(hy_recip(aq)) * big(aq) - 1)))
    end
    return w, wf, at
end

# every intermediate of hydro_step! with its C++ type (the range mirror); returns the domain quantities
function _hy_intermediates!(mx::Dict{Symbol,Float64}, k, x, u)
    put(T, a) = (mx[T] = max(get(mx, T, 0.0), abs(a)); a)
    P, W, K, EX, EA, kc, kb, kw = k.P, k.W, k.K, k.EX, k.EA, k.kc, k.kb, k.kw
    om = u[1]; om1 = put(:Sig, om - 1)
    ar = hy_region(x[3], kc[9], kc[8]); xr = hy_region(x[1], kc[7], kc[6])
    ain = ar == 0 ? 1 : 0; xin = xr == 0 ? 1 : 0
    atl = ar == 1 ? kc[8] : ar == 2 ? kc[9] : x[3]; atc = ain == 1 ? 0.0 : atl
    xpl = xr == 1 ? kc[6] : xr == 2 ? kc[7] : x[1]; xpc = xin == 1 ? 0.0 : xpl
    e1 = put(:Sig, kc[2] * om1); e2 = put(:Sig, kc[3] * atl); e3 = put(:Sig, kc[4] * x[2]); e4 = put(:Sig, kc[5] * xpl)
    s = put(:Sig, put(:Sig, put(:Sig, kc[1] - e1) - e4) + put(:Sig, e3 - e2))
    nx = hy_freeze(xpl, Int(s > 0), Int(s < 0), kc[7], kc[6]); na = hy_freeze(atl, Int(xpl > 0), Int(xpl < 0), kc[9], kc[8])
    bk = 1 + 8ain + 4xin + 2nx + na
    d = zeros(3)
    for i in 1:3
        l = [put(:Inc, P[bk, 3i-3+j] * x[j]) for j in 1:3]
        l4 = put(:Inc, W[bk, i] * om1); l5 = put(:Inc, EX[bk, i] * xpc); l6 = put(:Inc, EA[bk, i] * atc)
        d[i] = put(:Inc, put(:Inc, put(:Inc, l[1] + l[2]) + put(:Inc, l[3] + l4)) + put(:Inc, put(:Inc, l5 + l6) + K[bk, i]))
    end
    atm = put(:Sig, x[3] + d[3] / 2)
    inr = kc[9] <= atm <= kc[8] ? 1 : 0
    ro = (ain == 1 || inr == 1) ? atm : kc[8]
    r = put(:Rr, hy_recip(ro))
    if 2.0^-6 <= ro < 16                                             # the reciprocal's internals
        rq, _, kk = _hy_recip_q(ro); put(:Rq, ro * 2.0^-kk); put(:Rq, 2.0)
    end
    g1 = put(:Sig, kc[8] * r); c = put(:Cw, g1 * g1)
    crow = ar == 1 ? 1.0 : ar == 2 ? kw[1] : c
    cP = inr == 1 ? c : atm > kc[8] ? 1.0 : kw[1]
    q = x[4]; bc = put(:Small, kb[1] * crow); q2 = put(:St, q * q); bq2 = put(:Small, bc * q2); aa = put(:Small, kb[1] - bq2)
    dd = put(:Small, 2 * put(:Small, bc * q)); ad = put(:Small, aa * dd); dl = put(:Small, aa - ad); qm = put(:St, q + dl)
    qm2 = put(:St, qm * qm); qm3 = put(:St, qm2 * qm); gc = put(:Gc, kc[8] * cP); P_ = put(:Pw, gc * qm3)
    put(:Cw, crow * qm2); put(:Cw, cP * qm2)
    return (; atm, ro, ain, inr, crow, cP, qm, P = P_)
end

# ---------------------------------------------------------------------------------------------------------------------
# The certified per-step bound (43b, 43c, 43d): the penstock's algebra against the exact midpoint of the same decisions
# (the governor's banks are exact affine midpoints); MPFR outward rounding (`_Iv`, shunt.jl). Inputs (Float64 run values,
# 256-bit banks) exact. Returns an upper bound of |q+ - q+_exact| (endpoint), rounded up.
# ---------------------------------------------------------------------------------------------------------------------
function _hy_stepbound(d, bb, k, x, u, md)
    bk, _, _ = md
    bt = _hy_bits(bk); m = bb[bk]
    xb = _iv.(x); om1 = _iv(u[1]) - _iv(1.0)
    GMAX, GMIN = _iv(d["GMAX"]), _iv(d["GMIN"])
    xpl_side = x[1] > d["RUP"] ? d["RUP"] : x[1] < d["RDN"] ? d["RDN"] : 0.0
    at_side = x[3] > d["GMAX"] ? d["GMAX"] : x[3] < d["GMIN"] ? d["GMIN"] : 0.0
    xpc, atc = _iv(bt.xin ? 0.0 : xpl_side), _iv(bt.ain ? 0.0 : at_side)
    d3 = _iv(m.P[3, 1]) * xb[1] + _iv(m.P[3, 2]) * xb[2] + _iv(m.P[3, 3]) * xb[3] + _iv(m.W[3]) * om1 + _iv(m.K[3]) +
         _iv(m.EX[3]) * xpc + _iv(m.EA[3]) * atc
    atm = xb[3] + _ivhalf(d3)
    atrow = bt.ain ? atm : atc
    atrow.lo > 0 || error("bound: the row's gate is not positive")
    ctrue = (GMAX * GMAX) * _ivsq_recip(atrow)
    # the hardware algorithm's c (exact arithmetic): the reciprocal from the ROM seed and two Newton steps, or the side value
    if bt.ain
        r = _hy_recip_exact(atm)
        g1 = GMAX * r; chw = g1 * g1
    else
        chw = at_side == d["GMAX"] ? _iv(1.0) : _iv(k.kw[1])                # the prepared, rounded side value
    end
    b = _Iv(d["__bbig"], d["__bbig"])                                # b = h / (2 Tw) unrounded (256 bits, the export)
    q = xb[4]
    bc = b * chw
    A = b - bc * (q * q); D = _iv(2.0) * bc * q
    dl = A - A * D
    qh = q + dl
    res = _ivmag(A * D * D) |> v -> _mpadd(v, _ivmag(bc * dl * dl), _RU)
    res = _mpadd(res, _ivmag(b * (ctrue - chw) * (qh * qh)), _RU)
    return Float64(_mpmul(BigFloat(2; precision = 256), res, _RU), RoundUp)
end
_ivsq_recip(a::_Iv) = (a.lo > 0 || error("reciprocal of a non-positive interval");
                       one = BigFloat(1; precision = 256);
                       _Iv(_mpdiv(one, _mpmul(a.hi, a.hi, _RU), _RD), _mpdiv(one, _mpmul(a.lo, a.lo, _RD), _RU)))
# the hardware reciprocal in exact arithmetic on an interval (the same bin and scaling for both ends, else an error)
function _hy_recip_exact(a::_Iv)
    kf(v) = v >= 8 ? 4 : v >= 4 ? 3 : v >= 2 ? 2 : v >= 1 ? 1 : v >= 0.5 ? 0 : v >= 0.25 ? -1 : v >= 0.125 ? -2 :
            v >= 0.0625 ? -3 : v >= 0.03125 ? -4 : v >= 0.015625 ? -5 : -6
    k1, k2 = kf(a.lo), kf(a.hi); k1 == k2 || error("bound: the gate interval straddles a scaling boundary")
    sc = BigFloat(2; precision = 256)^(-k1)
    m = _Iv(a.lo * sc, a.hi * sc)
    j1, j2 = floor(Int, m.lo * 2048) - 1024, floor(Int, m.hi * 2048) - 1024
    j1 == j2 || error("bound: the gate interval straddles a seed bin")
    y = _iv(HLS_ROMS[:HY_SEED][clamp(j1, 0, 1023)+1]); two = _iv(2.0)
    for _ in 1:2
        y = y * (two - m * y)
    end
    return _Iv(y.lo * sc, y.hi * sc)
end

# ---------------------------------------------------------------------------------------------------------------------
# The corner suite (43c, 43d): one step from constructed states; (name, unit index (1 case, 0 synthetic), x, omega, expect)
# where expect = (:pass, (ain, xin, nx, na), crossing) or (:reject,)
# ---------------------------------------------------------------------------------------------------------------------
function hydro_rt_corners(h)
    q(v) = _q48(v, HYDRO_RT_I.St); w(v) = _q48(v, HYDRO_RT_I.Sig); L = 2.0^-42
    cs = []
    for unit in (1, 0)
        d = hydro_rt_data(unit); GMAX, GMIN, RUP, RDN = d["GMAX"], d["GMIN"], d["RUP"], d["RDN"]
        a0 = d["x"][3]
        X(xp, at; qv = clamp(at, GMIN, GMAX) / GMAX) = [q(xp), q(at), q(at), q(qv)]
        P(name, x, om, bits, cr = 0) = push!(cs, (name, unit, x, w(om), (:pass, bits, cr)))
        P("free", X(0.0, a0), 1.0, (1, 1, 0, 0))
        P("xp exactly RUP, large inward rate (omega 5: s ~ -34, beyond St)", X(RUP, a0), 5.0, (1, 1, 0, 0))
        for (side, lim, oo, oi) in (("RUP", RUP, 0.97, 1.03), ("RDN", RDN, 1.03, 0.97))
            P("xp beyond $side, rate out", X(lim + sign(lim) * 0.01, a0), oo, (1, 0, 1, 0))
            P("xp beyond $side, rate in", X(lim + sign(lim) * 0.01, a0), oi, (1, 0, 0, 0))
            P("xp exactly $side, rate out", X(lim, a0), oo, (1, 1, 1, 0))
            P("xp exactly $side, rate in", X(lim, a0), oi, (1, 1, 0, 0))
            P("xp one LSB beyond $side, rate out", X(lim + sign(lim) * L, a0), oo, (1, 0, 1, 0))
            P("xp one LSB inside $side, rate out", X(lim - sign(lim) * L, a0), oo, (1, 1, 0, 0))
        end
        # the gate at and above GMAX (xp within band)
        P("gate exactly GMAX, xp > 0", X(0.5, GMAX), 1.0, (1, 1, 0, 1))
        P("gate exactly GMAX, xp < 0", X(-0.5, GMAX), 1.0, (1, 1, 0, 0))
        P("gate one LSB above GMAX, xp > 0", X(0.5, GMAX + L), 1.0, (0, 1, 0, 1))
        P("gate above GMAX, xp > 0", X(0.5, GMAX + 0.01), 1.0, (0, 1, 0, 1))
        P("gate above GMAX, xp < 0", X(-0.5, GMAX + 0.01), 1.0, (0, 1, 0, 0))
        # crossings within the step (rate +-RUP: the gate moves h/2 RUP to the midpoint)
        P("gate crossing out above GMAX", X(RUP, GMAX - 0.25 * h * RUP), 0.97, (1, 1, 0, 0), 1)   # near GMAX the rate is inward: xp not frozen
        P("gate crossing in below GMAX", X(RDN, GMAX + 0.25 * h * abs(RDN)), 1.03, (0, 1, 1, 0), 2)
        if GMIN > 0                                                  # the synthetic unit only (43c, 4)
            P("gate exactly GMIN, xp < 0", X(-0.5, GMIN), 1.0, (1, 1, 0, 1))
            P("gate exactly GMIN, xp > 0", X(0.5, GMIN), 1.0, (1, 1, 0, 0))
            P("gate one LSB below GMIN, xp < 0", X(-0.5, GMIN - L), 1.0, (0, 1, 0, 1))
            P("gate below GMIN, xp < 0", X(-0.5, GMIN - 0.01), 1.0, (0, 1, 0, 1))
            P("gate below GMIN, xp > 0", X(0.5, GMIN - 0.01), 1.0, (0, 1, 0, 0))
            P("gate crossing out below GMIN", X(RDN, GMIN + 0.25 * h * abs(RDN)), 1.03, (1, 1, 1, 0), 1)
            P("gate crossing in above GMIN", X(RUP, GMIN - 0.25 * h * RUP), 0.97, (0, 1, 1, 0), 2)
            P("held low, gate state <= 0 (safe reciprocal operand)", X(-0.5, -0.1; qv = GMIN / GMAX), 1.0, (0, 1, 0, 1))
        end
    end
    # expected rejections (the case's unit): the flag must be set
    d = hydro_rt_data(1); GMAX = d["GMAX"]
    R(name, x, om) = push!(cs, (name, 1, x, w(om), (:reject,)))
    R("gate below 0.05 GMAX", [q(0.0), q(0.03GMAX), q(0.03GMAX), q(0.03)], 1.0)
    R("head above 4", [q(0.0), q(0.5GMAX), q(0.5GMAX), q(1.05)], 1.0)
    R("flow not positive", [q(0.0), q(0.5GMAX), q(0.5GMAX), q(-0.01)], 1.0)
    R("flow above 1.5", [q(0.0), q(GMAX), q(GMAX), q(1.6)], 1.0)
    R("speed below 0.5", [q(0.0), q(0.5GMAX), q(0.5GMAX), q(0.5)], 0.4)
    R("case unit: gate below GMIN = 0", [q(-0.5), q(-0.1), q(-0.1), q(0.1)], 1.0)
    return cs
end

function _hy_corner_step(k, x, om)
    xx = copy(x); y = zeros(1); md = zeros(Int, 3)
    hydro_step!(xx, zeros(4), [om], y, md, k.P, k.W, k.K, k.EX, k.EA, k.kc, k.kb, k.kw)
    return xx, y, md
end

# the package's step from x (4) at omega (decisions recorded at x_n, replayed, fixed point to 1e-15): x+, P at the midpoint
function _hy_pkg_step(d, x, om, h)
    p = _HYPKG.Ieeeg3Params(d["Tg"], d["Tp"], d["Sigma"], d["Delta"], d["Tr"], 1.0, d["UO"], d["UC"], d["PMAX"], d["PMIN"],
                             d["R_base"], 0.0, 0.0, 0.0, 0.0, 0.0)
    g = _HYPKG.PHHydroUnit(d["name"], p, d["GMAX"], d["Tw"], d["Tw_case"])
    dx = zeros(4); log = _HYPKG.ModeLog(Bool[])
    _HYPKG._hydro_rhs!(dx, g, x, om, d["Pref"], log)
    xn = x .+ h .* dx
    for it in 1:80
        _HYPKG._hydro_rhs!(dx, g, (x .+ xn) ./ 2, om, d["Pref"], _HYPKG.ModeReplay(log.decisions, 0))
        x1 = x .+ h .* dx
        maximum(abs, x1 .- xn) <= 1e-15 * max(1.0, maximum(abs, x1)) && (xn = x1; break)
        xn = x1
        it == 80 && error("package midpoint did not converge")
    end
    return xn, _HYPKG._hydro_power(g, (x .+ xn) ./ 2)
end

# the exact-input tests of hy_region and hy_freeze: (args, expected)
const HYDRO_RT_REGION = (L = 2.0^-42; [((1.0, -1.0, 1.0), 0), ((1.0 + L, -1.0, 1.0), 1), ((-1.0, -1.0, 1.0), 0),
                         ((-1.0 - L, -1.0, 1.0), 2), ((0.0, -1.0, 1.0), 0)])
const HYDRO_RT_FREEZE = [((1.0, 1, 0, -1.0, 1.0), 1), ((1.0, 0, 0, -1.0, 1.0), 0), ((1.0, 0, 1, -1.0, 1.0), 0),
                         ((-1.0, 0, 1, -1.0, 1.0), 1), ((-1.0, 0, 0, -1.0, 1.0), 0), ((-1.0, 1, 0, -1.0, 1.0), 0),
                         ((0.5, 1, 0, -1.0, 1.0), 0)]                    # (vl, pos, neg, lo, hi)

"""
    hydro_rt_export(; h = HYDRO_RT_H, dir = src/rt/governors/CPP/hydro, bound = true)

Prepares the unit for h (BigFloat), runs the Julia checks fixed in the idea gate (43-43d) and writes the C++, the seed ROM,
the coefficients (the case's unit and the synthetic unit), the C simulation testbench, the board program, the ILA reference
and flow.cfg. The check against the package over the bench and the refinement: scripts/realtime/hydro_checks.jl.
"""
function hydro_rt_export(; h = HYDRO_RT_H, dir = joinpath(HYDRO_RT_DIR, "CPP", "hydro"), bound = true)
    occursin("II=$(HYDRO_RT_STEP_CYCLES)\"", string(HLS_DEFS[:hydro_run!])) ||
        error("the PIPELINE II in hydro_run! differs from HYDRO_RT_STEP_CYCLES = $HYDRO_RT_STEP_CYCLES")
    hb = setprecision(() -> parse(BigFloat, h), BigFloat, 256); hh = Float64(hb)
    units = [hydro_rt_data(1), hydro_rt_data(0)]                     # the case's unit, the synthetic unit
    for d in units
        d["__bbig"] = setprecision(() -> hb / (2 * BigFloat(d["Tw"])), BigFloat, 256); d["__b"] = Float64(d["__bbig"])
    end
    ks = [setprecision(() -> hydro_rt_coefstruct(d, hb), BigFloat, 256) for d in units]
    ku = [setprecision(() -> hydro_rt_coefstruct(d, hb; grid = false), BigFloat, 256) for d in units]
    bbs = [setprecision(() -> _hy_bigbanks(d, hb), BigFloat, 256) for d in units]
    d, k, kun, bb = units[1], ks[1], ku[1], bbs[1]
    x0 = _hy_x0(d); nend = round(Int, 0.3 / hh); S_ = HYDRO_RT_SCALE
    println("h = $hh s; unit $(d["machine"]) (and the synthetic unit with PMIN 0.1 for corners)")
    half(T) = 2.0^(getfield(HYDRO_RT_I, T) - 2)
    for (i, kk) in enumerate(ks), (fld, T) in HYDRO_RT_FIELDS
        mxv = maximum(abs, getfield(kk, fld))
        mxv < half(T) || error("unit $i: coefficient $fld reaches $mxv (half range of $T: $(half(T)))")
    end
    # (e) the reciprocal
    wq, wf, at = _hy_recip_sweep()
    println("reciprocal: max relative error $(round(wq; sigdigits = 3)) in the C++ arithmetic (Julia mirror, at $at), ",
            "$(round(wf; sigdigits = 3)) in Float64 (limit 1e-10)")
    (wq <= 1e-10 && wf <= 1e-10) || error("reciprocal error $wq / $wf")
    for (a, e) in HYDRO_RT_REGION; hy_region(a...) == e || error("hy_region$a != $e"); end
    for (a, e) in HYDRO_RT_FREEZE; hy_freeze(a...) == e || error("hy_freeze$a != $e"); end
    # the corner suite: passing corners against the package (states, P), the bits, the crossing code, the bound, the ranges
    wc = 0.0; npass = 0; nrej = 0; bcmax = 0.0
    for (name, unit, x, om, ex) in hydro_rt_corners(hh)
        iu = unit == 1 ? 1 : 2; du = units[iu]; kk = ks[iu]
        xx, y, md = _hy_corner_step(kk, x, om)
        if ex[1] === :reject
            md[2] == 1 || error("corner '$name': the domain flag is not set")
            nrej += 1; continue
        end
        bits = _hy_bits(md[1]); want = ex[2]
        (Int(bits.ain), Int(bits.xin), Int(bits.nx), Int(bits.na)) == want || error("corner '$name': bits $(bits) expected $want")
        md[3] == ex[3] || error("corner '$name': crossing code $(md[3]) expected $(ex[3])")
        md[2] == 0 || error("corner '$name': the domain flag is set")
        xp_, Pp = _hy_pkg_step(du, x, om, hh)
        _, yu, _ = _hy_corner_step(ku[iu], x, om); xu, _, _ = _hy_corner_step(ku[iu], x, om)
        e = max(maximum(abs, xu .- xp_) / S_.x, abs(yu[1] - Pp) / S_.P)
        e <= 1e-9 || error("corner '$name': differs from the package by $e (limit 1e-9)")
        wc = max(wc, e)
        mxr = Dict{Symbol,Float64}(); _hy_intermediates!(mxr, kk, x, [om])
        for (T, v) in mxr; v < half(T) || error("corner '$name': a $T value reaches $v"); end
        if bound
            bv = setprecision(() -> _hy_stepbound(du, bbs[iu], ks[iu], x, [om], md), BigFloat, 256)   # rounded side c (44, 3)
            bv <= HYDRO_RT_BOUND || error("corner '$name': certified bound $bv > $HYDRO_RT_BOUND")
            bcmax = max(bcmax, bv)
        end
        npass += 1
    end
    println("corner suite: $npass passing corners (bits, crossings, package parity <= $(round(wc; sigdigits = 3)), ranges, ",
            "bound <= $(round(bcmax; sigdigits = 3))), $nrej expected rejections flagged")
    for case in (:A, :B)
        b = hydro_rt_bench(d, case, hh)
        tr = _hydro_rt_trace(k, x0, b, nend); tu = _hydro_rt_trace(kun, x0, b, nend)
        mx = Dict{Symbol,Float64}(); occ = Dict{Int,Int}(); ncr = 0
        H(qv) = big(d["GMAX"]) * big(d["Tw"]) * big(qv)^2 / 2; H0 = H(x0[4]); Dmax = big(0.0); Dsum = big(0.0); Pdiff = 0.0
        for n in 1:nend
            x, xn, u, y, md = tr.X[n], tr.Xn[n], tr.U[n], tr.Y[n], tr.M[n]
            md[2] == 0 || error("case $case step $n: outside the domain")
            occ[md[1]] = get(occ, md[1], 0) + 1; md[3] != 0 && (ncr += 1)
            iv = _hy_intermediates!(mx, k, x, u)
            foreach(v -> (mx[:St] = max(get(mx, :St, 0.0), abs(v))), xn)
            # energy on the row's head (43b, 1): D = H+ - H - h (GMAX q_m - GMAX c_row q_m^3)
            qm = (big(x[4]) + big(xn[4])) / 2
            atm = big(iv.atm); crow = iv.ain == 1 ? (big(d["GMAX"]) / atm)^2 : (x[3] > d["GMAX"] ? big(1) : (big(d["GMAX"]) / big(d["GMIN"]))^2)
            D = H(xn[4]) - H(x[4]) - big(hh) * big(d["GMAX"]) * (qm - crow * qm^3)
            Dmax = max(Dmax, abs(D)); Dsum += abs(D)
            iv.inr != (iv.ain) && (Pdiff = max(Pdiff, abs(iv.cP - iv.crow) * d["GMAX"] * iv.qm^3))
        end
        for (T, v) in mx; v < half(T) || error("case $case: a $T value reaches $v (half range $(half(T)))"); end
        bmax = 0.0
        if bound
            bs = zeros(nend)
            Threads.@threads for n in 1:nend
                bs[n] = _hy_stepbound(d, bb, k, tu.X[n], tu.U[n], tu.M[n])
            end
            bmax, nb_ = findmax(bs)
            bmax <= HYDRO_RT_BOUND || error("case $case step $nb_: certified bound $bmax")
        end
        db = maximum(n -> max(maximum(abs, tr.Xn[n] - tu.Xn[n]) / S_.x, abs(tr.Y[n][1] - tu.Y[n][1]) / S_.P), 1:nend)
        db <= 5e-8 || error("case $case: rounding moves the run by $db")
        (Dmax <= 1e-12 * H0 && Dsum <= 1e-9 * H0) || error("case $case: energy defect max $Dmax, sum $Dsum (H0 $H0)")
        xs_ = reduce(hcat, tr.Xn)
        println("case $case: banks ", join(("$b=>$(occ[b])" for b in sort(collect(keys(occ)))), ", "), "; crossings $ncr; ",
                "gate [$(round(minimum(xs_[3, :]); sigdigits = 6)), $(round(maximum(xs_[3, :]); sigdigits = 6))], ",
                "q [$(round(minimum(xs_[4, :]); sigdigits = 6)), $(round(maximum(xs_[4, :]); sigdigits = 6))]; rounding ",
                "$(round(db; sigdigits = 3)) (limit 5e-8); energy max |D| $(round(Float64(Dmax / H0); sigdigits = 3)) H0, sum ",
                "$(round(Float64(Dsum / H0); sigdigits = 3)) H0; certified bound $(bound ? round(bmax; sigdigits = 3) : "skipped") ",
                "(limit 1e-15); output vs row power at crossings $(Pdiff)")
    end
    # continuous = segmented; short calls (the shunt's windows)
    nshort = 0
    for case in (:A, :B)
        b = hydro_rt_bench(d, case, hh); tr = _hydro_rt_trace(k, x0, b, nend); cuts = _hydro_rt_cuts(hh, b, tr)
        rec, _ = _hydro_rt_julia(k, x0, b, cuts)
        xq = copy(x0); _hy_call!(k, xq, b, 0, cuts[end])
        (xq == rec[end] && xq == tr.Xn[cuts[end]]) || error("case $case: continuous differs from segmented")
        trm = (; M = [[m[1]] for m in tr.M])
        x = copy(x0); xcs = zeros(4); prev = 0; y = zeros(1)
        for (st, ends) in _shunt_rt_windows(trm, nend)
            if st > prev
                hydro_run!(k, x, xcs, b.us, b.du, b.non, b.nmid, b.noff, prev, st - prev, 32, y, [0], Ref(0.0), Ref(0.0), Ref(false), Ref(false))
                (x == tr.Xn[st] && y == tr.Y[st]) || error("case $case: call to $st differs"); prev = st
            end
            for e in ends
                hydro_run!(k, x, xcs, b.us, b.du, b.non, b.nmid, b.noff, prev, e - prev, 32, y, [0], Ref(0.0), Ref(0.0), Ref(false), Ref(false))
                (x == tr.Xn[e] && y == tr.Y[e]) || error("case $case: call to $e differs"); prev = e; nshort += 1
            end
        end
    end
    println("continuous = segmented and $nshort short calls: bit for bit (states and P)")
    # the C++
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "hydro_rom.h"), [:HY_SEED]; header = "the reciprocal's seeds (hydro.jl section 3)")
    hls_write(joinpath(dir, "hydro.cpp"), [:HydroCoefs, :hy_recip, :hy_region, :hy_freeze, :hydro_step!, :hydro_run!];
              types = HYDRO_RT_TYPES, local_includes = ["rt_util.h", "hydro_rom.h"],
              header = "the hydro unit (PHHydroUnit), one run of steps of h = $hh s; from src/rt/governors/hydro.jl")
    p = joinpath(dir, "hydro_coefs.h")
    open(p, "w") do io
        println(io, "// hydro_coefs.h: written by PHPS.RT.hydro_rt_export; do not edit\n// h = $hh s")
        for (i, (du, kk)) in enumerate(zip(units, ks))
            println(io, "// $i: $(du["name"]) on $(du["machine"])")
            println(io, "const HydroCoefs HYDRO_COEFS_$i = ", _cinit(kk), ";")
        end
    end
    println("written $p")
    hydro_rt_testbench(joinpath(dir, "hydro_tb.cpp"), units, ks, x0; h = hh)
    hydro_rt_board_script(joinpath(dir, "hydro_board.py"), d, k, x0; h = hh)
    hydro_rt_ilaref(joinpath(dir, "hydro_ila_ref.csv"), d, k, x0; h = hh)
    I_ = HYDRO_RT_I
    flow_config(joinpath(dir, "flow.cfg"); model = "hydro", top = "hydro_run", clock_ns = HYDRO_RT_CLOCK_NS,
                ip_version = "1.0", ip_name = "Hydro unit real-time step",
                probes = ["at_out" => 48, "p_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                plot = ["at_out" => (48 - I_.St, "gate [pu]"), "p_out" => (48 - I_.Pw, "P [pu]")],
                sample_dt = HYDRO_RT_DECIM * hh, board_args = "--no-download",
                impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = HYDRO_RT_STEP_CYCLES, ila_ref = "hydro_ila_ref.csv",
                ila_check = ["at_out:$(48 - I_.St):$(HYDRO_RT_LIM * S_.x)", "p_out:$(48 - I_.Pw):$(HYDRO_RT_LIM * S_.P)"],
                strict = true)
end

const HYDRO_RT_BOARD_RUNS = (:A, :B)
const HYDRO_RT_DECIM = 96

function _hy_ila_call!(rec, k, x, b, n0, nsteps, decim, nmax)
    pat = x[3]; ppw = 0.0; psa = false; cnt = 0; y = zeros(1); md = zeros(Int, 3); xc = zeros(4)
    ha = fill(NaN, 8); hp = fill(NaN, 8)
    for i in 1:nsteps
        ha = vcat(pat, ha[1:7]); hp = vcat(ppw, hp[1:7])
        psa && length(rec) < nmax && push!(rec, (copy(ha), copy(hp), n0 + i - 1))
        hydro_step!(x, xc, _hy_u(b, n0 + i - 1), y, md, k.P, k.W, k.K, k.EX, k.EA, k.kc, k.kb, k.kw)
        cnt += 1; psa = cnt >= decim; psa && (cnt = 0)
        pat = x[3]; ppw = y[1]
    end
    return x
end

function hydro_rt_ilaref(path, d, k, x0; h, decim = HYDRO_RT_DECIM, nmax = 16384)
    rec = Tuple{Vector{Float64},Vector{Float64},Int}[]; tag = String[]
    for case in HYDRO_RT_BOARD_RUNS
        b = hydro_rt_bench(d, case, h); cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        x = copy(x0); n = 0
        for c in cuts
            l0 = length(rec); xr = copy(x)
            _hy_call!(k, xr, b, n, c - n; decim)
            _hy_ila_call!(rec, k, x, b, n, c - n, decim, nmax)
            x == xr || error("the ILA mirror differs from hydro_run! (case $case)")
            append!(tag, fill("1$case", length(rec) - l0)); n = c
            length(rec) >= nmax && break
        end
        length(rec) >= nmax && break
    end
    open(path, "w") do io
        println(io, "k,run,step,", join(("at_out_$L" for L in 0:7), ","), ",", join(("p_out_$L" for L in 0:7), ","))
        for (j, (a, v, st)) in enumerate(rec)
            println(io, "$j,$(tag[j]),$st,", join(map(repr, a), ","), ",", join(map(repr, v), ","))
        end
    end
    println("written $path ($(length(rec)) samples, decim $decim)")
    return path
end

# the C simulation: cases A and B (cuts every 0.01 s, events, every bank change), continuous = segmented and the short-call
# windows bit for bit, the reciprocal sweep, hy_region / hy_freeze with exact inputs, the corner suite (bits, crossing,
# states and P within the limit of Julia's, the flag of the rejections). Returns 1 on failure.
function hydro_rt_testbench(path, units, ks, x0; h)
    S_ = HYDRO_RT_SCALE; d, k = units[1], ks[1]
    blocks = String[]; calls = String[]; corners = String[]
    nend = round(Int, 0.3 / h)
    for case in (:A, :B)
        b = hydro_rt_bench(d, case, h); tr = _hydro_rt_trace(k, x0, b, nend); cuts = _hydro_rt_cuts(h, b, tr)
        rec, outs = _hydro_rt_julia(k, x0, b, cuts)
        W = _shunt_rt_windows((; M = [[m[1]] for m in tr.M]), nend); wends = reduce(vcat, (e for (_, e) in W))
        tag = "$(case)"
        push!(blocks, """
        // case $case
        const int NC_$tag = $(length(cuts)), ON_$tag = $(b.non), MID_$tag = $(b.nmid), OFF_$tag = $(b.noff);
        const int CUTS_$tag[NC_$tag] = {$(join(cuts, ", "))};
        const double US_$tag[4] = $(_sh_vec(b.us));
        const double DU_$tag[4] = $(_sh_vec(b.du));
        const double J_$tag[NC_$tag][5] = {$(join((_sh_vec(vcat(rec[g+1], outs[g+1])) for g in eachindex(cuts)), ",\n    "))};
        const int NW_$tag = $(length(W));
        const int WS_$tag[NW_$tag] = {$(join((st for (st, _) in W), ", "))};
        const int WK_$tag[NW_$tag] = {$(join((length(e) for (_, e) in W), ", "))};
        const int WE_$tag[$(length(wends))] = {$(join(wends, ", "))};
        const double WJ_$tag[$(length(wends))][5] = {$(join((_sh_vec(vcat(tr.Xn[e], tr.Y[e])) for e in wends), ",\n    "))};
        """)
        push!(calls, "    w = fmax(w, run_case(\"$case\", HYDRO_COEFS_1, X0, US_$tag, DU_$tag, ON_$tag, MID_$tag, OFF_$tag, NC_$tag, CUTS_$tag, J_$tag, &segok));\n" *
                     "    w = fmax(w, run_windows(\"$case\", HYDRO_COEFS_1, X0, US_$tag, DU_$tag, ON_$tag, MID_$tag, OFF_$tag, NW_$tag, WS_$tag, WK_$tag, WE_$tag, WJ_$tag, &segok));")
    end
    for (name, unit, x, om, ex) in hydro_rt_corners(h)
        iu = unit == 1 ? 1 : 2
        xx, y, md = _hy_corner_step(ks[iu], x, om)
        push!(corners, "    {$iu, \"$(replace(name, '"' => '\''))\", $(_sh_vec(x)), $(_sh_f(om)), $(_sh_vec(vcat(xx, y))), $(md[1]), $(md[2]), $(md[3])},")
    end
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.hydro_rt_testbench (src/rt/governors/hydro.jl); do not edit
        // C simulation of the hydro unit, h = $h s: cases A (omega -> 0.97) and B (omega -> 1.03), 0.1-0.2 s; the C++ states
        // and P next to the same @hls code in Julia. Limit (fixed before results, 43): $(HYDRO_RT_LIM) relative to the scales
        // x $(S_.x), P $(S_.P). Continuous = segmented, short-call windows bit for bit; reciprocal sweep (1e-10); hy_region /
        // hy_freeze with exact inputs; corner suite (bits, crossing code, flag, states and P). Returns 1 on failure.
        #include <cstdio>
        #include <cmath>
        #include <random>
        #include "hydro.h"
        #include "hydro_coefs.h"

        const double X0[4] = $(_sh_vec(x0));
        $(join(blocks, "\n"))
        const HydroCoefs *COEFS[2] = {&HYDRO_COEFS_1, &HYDRO_COEFS_2};

        double run_case(const char *name, const HydroCoefs &k, const double *x0, const double *us0, const double *du0,
                        int on, int mid, int off, int nc, const int *cuts, const double J[][5], int *segok) {
            St x[4], xc[4]; Small r[4], rc[4]; Sig us[4]; Small du[4]; Pw y[1]; int flg[1] = {0}; St a; Pw p; bool st, sa;
            for (int i = 0; i < 4; i++) { x[i] = x0[i]; xc[i] = x0[i]; r[i] = 0; rc[i] = 0; us[i] = us0[i]; du[i] = du0[i]; }
            double w = 0; int prev = 0;
            for (int g = 0; g < nc; g++) {
                hydro_run(k, x, r, us, du, on, mid, off, prev, cuts[g] - prev, 32, y, flg, a, p, st, sa);
                prev = cuts[g];
                double e = 0;
                for (int i = 0; i < 4; i++) e = fmax(e, fabs(x[i].to_double() - J[g][i]) / $(S_.x));
                e = fmax(e, fabs(y[0].to_double() - J[g][4]) / $(S_.P));
                w = fmax(w, e);
            }
            Pw ys = y[0]; int f1 = flg[0];
            hydro_run(k, xc, rc, us, du, on, mid, off, 0, cuts[nc - 1], 32, y, flg, a, p, st, sa);
            int same = (y[0] == ys) && f1 == 0 && flg[0] == 0;
            for (int i = 0; i < 4; i++) same = same && (x[i] == xc[i]) && (r[i] == rc[i]);
            if (!same) *segok = 0;
            printf("case %s %4d checkpoints: xp %.10f xr %.10f gate %.10f q %.12f P %.10f, max relative difference %.3e, segmented %s\\n",
                   name, nc, x[0].to_double(), x[1].to_double(), x[2].to_double(), x[3].to_double(), y[0].to_double(), w,
                   same ? "= continuous" : "DIFFERS");
            return w;
        }

        double run_windows(const char *name, const HydroCoefs &k, const double *x0, const double *us0, const double *du0,
                           int on, int mid, int off, int nw, const int *ws, const int *wk, const int *we, const double WJ[][5], int *segok) {
            St x[4], xw[4]; Small r[4], rw[4]; Sig us[4]; Small du[4]; Pw y[1], yw[1]; int flg[1] = {0}; St a; Pw p; bool st, sa;
            for (int i = 0; i < 4; i++) { x[i] = x0[i]; r[i] = 0; us[i] = us0[i]; du[i] = du0[i]; }
            double w = 0; int prev = 0, q = 0, same = 1, calls = 0;
            for (int g = 0; g < nw; g++) {
                if (ws[g] > prev) { hydro_run(k, x, r, us, du, on, mid, off, prev, ws[g] - prev, 32, y, flg, a, p, st, sa); prev = ws[g]; }
                for (int i = 0; i < 4; i++) { xw[i] = x[i]; rw[i] = r[i]; }
                int s0 = prev;
                for (int c = 0; c < wk[g]; c++, q++) {
                    hydro_run(k, x, r, us, du, on, mid, off, prev, we[q] - prev, 32, y, flg, a, p, st, sa);
                    prev = we[q]; calls++;
                    double e = 0;
                    for (int i = 0; i < 4; i++) e = fmax(e, fabs(x[i].to_double() - WJ[q][i]) / $(S_.x));
                    e = fmax(e, fabs(y[0].to_double() - WJ[q][4]) / $(S_.P));
                    w = fmax(w, e);
                }
                hydro_run(k, xw, rw, us, du, on, mid, off, s0, prev - s0, 32, yw, flg, a, p, st, sa);
                int ok = (y[0] == yw[0]);
                for (int i = 0; i < 4; i++) ok = ok && (x[i] == xw[i]) && (r[i] == rw[i]);
                same = same && ok;
            }
            if (!same) *segok = 0;
            printf("case %s %4d short calls in %3d windows: max relative difference %.3e, one call over each window %s\\n", name, calls, nw, w,
                   same ? "= short calls (bit for bit)" : "DIFFERS");
            return w;
        }

        double recip_sweep() {
            double w = 0, wa = 0; const double g = ldexp(1.0, -40);
            auto one = [&](double av) {
                if (av < ldexp(1.0, -6) || av >= 16) return;
                Sig a = av; double aq = a.to_double();
                double e = fabs(hy_recip(a).to_double() * aq - 1);
                if (e > w) { w = e; wa = aq; }
            };
            for (int k = -6; k <= 3; k++) for (int j = 0; j <= 1024; j++) { double a = (0.5 + j / 2048.0) * ldexp(1.0, k + 1); one(a - g); one(a); one(a + g); }
            std::mt19937_64 rng(43); std::uniform_real_distribution<double> U(ldexp(1.0, -6), 16);
            for (int i = 0; i < 1000000; i++) one(U(rng));
            printf("reciprocal sweep: max relative error %.3e at a = %.12f (limit 1e-10)\\n", w, wa);
            return w;
        }

        struct Corner { int unit; const char *name; double x[4]; double om; double ref[5]; int bk, fl, cr; };
        const Corner CORNERS[] = {
        $(join(corners, "\n"))
        };

        int main() {
            double w = 0; int segok = 1;
            int fail_rec = !(recip_sweep() <= 1e-10);
            int fail_dec = 0;
        $(join(("    fail_dec |= hy_region($(join(map(_sh_f, a), ", "))) != $e;" for (a, e) in HYDRO_RT_REGION), "\n"))
        $(join(("    fail_dec |= hy_freeze($(_sh_f(a[1])), $(a[2]), $(a[3]), $(_sh_f(a[4])), $(_sh_f(a[5]))) != $e;" for (a, e) in HYDRO_RT_FREEZE), "\n"))
            printf("hy_region / hy_freeze with exact inputs: %s\\n", fail_dec ? "FAIL" : "ok");
            int fail_cor = 0; double wc = 0;
            for (const Corner &c : CORNERS) {
                HydroCoefs k = *COEFS[c.unit - 1];   // a copy: the step takes non-const arrays
                St x[4]; Small r[4]; Sig u[1]; Pw y[1]; int md[3];
                for (int i = 0; i < 4; i++) { x[i] = c.x[i]; r[i] = 0; }
                u[0] = c.om;
                hydro_step(x, r, u, y, md, k.P, k.W, k.K, k.EX, k.EA, k.kc, k.kb, k.kw);
                double e = 0;
                if (c.fl == 0) {
                    for (int i = 0; i < 4; i++) e = fmax(e, fabs(x[i].to_double() - c.ref[i]) / $(S_.x));
                    e = fmax(e, fabs(y[0].to_double() - c.ref[4]) / $(S_.P));
                    wc = fmax(wc, e);
                }
                if (md[0] != c.bk || md[1] != c.fl || md[2] != c.cr || !(e <= $(HYDRO_RT_LIM))) {
                    fail_cor = 1; printf("corner %s: bank %d flag %d crossing %d (expected %d %d %d), difference %.3e\\n", c.name, md[0], md[1], md[2], c.bk, c.fl, c.cr, e);
                }
            }
            printf("corner suite: %d corners, max relative difference %.3e, banks / flags / crossings %s\\n", (int)(sizeof(CORNERS) / sizeof(Corner)), wc, fail_cor ? "FAIL" : "as expected");
        $(join(calls, "\n"))
            int fail = !(w <= $(HYDRO_RT_LIM)) || !segok || fail_rec || fail_dec || fail_cor;
            printf("C sim check: max relative difference %.3e (limit %.0e), continuous = segmented %s, reciprocal %s, decisions %s, corners %s: %s\\n",
                   w, $(HYDRO_RT_LIM), segok ? "yes" : "NO", fail_rec ? "FAIL" : "ok", fail_dec ? "FAIL" : "ok", fail_cor ? "FAIL" : "ok", fail ? "FAIL" : "PASS");
            return fail;
        }
        """)
    end
    println("written $path")
    return path
end

# the board program: cases A and B, then case C (10^7 steps at the operating point)
function hydro_rt_board_script(path, d, k, x0; h, decim = HYDRO_RT_DECIM)
    S_ = HYDRO_RT_SCALE; F = map(I -> 48 - I, HYDRO_RT_I)
    runs = String[]
    for case in HYDRO_RT_BOARD_RUNS
        b = hydro_rt_bench(d, case, h); cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        rec, outs = _hydro_rt_julia(k, x0, b, cuts)
        push!(runs, "(\"$case\", $(repr(collect(b.us))), $(repr(collect(b.du))), $(b.non), $(b.nmid), $(b.noff), $cuts, " *
                    "$(repr([vcat(rec[g+1], outs[g+1]) for g in eachindex(cuts)])))")
    end
    bC = hydro_rt_bench(d, :C, h)
    c = Tuple{Float64,Int}[]
    for (fld, T) in HYDRO_RT_FIELDS
        v = getfield(k, fld); f = getfield(F, T)
        vals = v isa AbstractMatrix ? [v[i, j] for i in 1:size(v, 1) for j in 1:size(v, 2)] : collect(v)
        foreach(e -> push!(c, (e, f)), vals)
    end
    coefs = "[" * join(("($(_sh_f(v)), $f)" for (v, f) in c), ", ") * "]"
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.hydro_rt_board_script (src/rt/governors/hydro.jl); do not edit.
        # Runs the hydro unit IP on the PYNQ-Z2 (next to it: hydro.bit, hydro.hwh, xhydro_run_hw.h).
        #   python3 hydro_board.py [--no-download]
        import sys, time, re
        from pynq import Overlay, Clocks

        H, DECIM, LIM = $(_sh_f(h)), $decim, $(HYDRO_RT_LIM)
        SX, SP = $(S_.x), $(S_.P)
        FST, FSIG, FSMALL, FPW = $(F.St), $(F.Sig), $(F.Small), $(F.Pw)
        COEFS = $coefs
        X0 = $(repr(x0))
        RUNS = [$(join(runs, ",\n        "))]
        US_C, DU_C = $(repr(collect(bC.us))), $(repr(collect(bC.du)))

        A = {}
        for m in re.finditer(r"#define XHYDRO_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xhydro_run_hw.h").read()):
            A[m.group(1)] = int(m.group(3), 16)

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("hydro.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / HYDRO_RT_CLOCK_NS)) > 0.01:
            Clocks.fclk0_mhz = $(1000 / HYDRO_RT_CLOCK_NS)
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.hydro_run_0

        def w48(addr, v, F):
            r = int(round(v * 2**F)) & ((1 << 48) - 1)
            ip.write(addr, r & 0xFFFFFFFF)
            ip.write(addr + 4, r >> 32)
        def r48(addr, F):
            r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
            if r >> 47:
                r -= 1 << 48
            return r / 2**F

        for j, (v, F) in enumerate(COEFS):
            w48(A["K"] + 8 * j, v, F)
        def run(x, xc, us, du, on, mid, off, n0, nsteps):
            for j, v in enumerate(x):
                w48(A["X"] + 8 * j, v, FST)
            for j, v in enumerate(xc):
                w48(A["XC"] + 8 * j, v, FSMALL)               # the remainders of the compensated summation
            for j in range(4):
                w48(A["US"] + 8 * j, us[j], FSIG); w48(A["DU"] + 8 * j, du[j], FSMALL)
            for nm, v in (("N_ON", on), ("N_MID", mid), ("N_OFF", off), ("N0", n0), ("NSTEPS", nsteps), ("DECIM", DECIM)):
                ip.write(A[nm], v)
            ip.write(A["FLG_I"], 0)                           # the sticky flag in / out (HLS: FLG_I, FLG_O; review 44, 1)
            t0 = time.perf_counter()
            ip.write(0x00, 1)
            while not (ip.read(0x00) & 0x2):
                time.sleep(0.001)
            t = time.perf_counter() - t0
            return t, [r48(A["X"] + 8 * j, FST) for j in range(4)], [r48(A["XC"] + 8 * j, FSMALL) for j in range(4)], r48(A["Y"], FPW), ip.read(A["FLG_O"])

        worst, flags = 0.0, 0
        for (case, us, du, on, mid, off, cuts, ref) in RUNS:
            print("case %s: compared at %s s" % (case, [round(c * H, 3) for c in cuts]))
            x, xc, n, tt = list(X0), [0.0] * 4, 0, 0.0
            for g, c in enumerate(cuts):
                t, x, xc, p, fl = run(x, xc, us, du, on, mid, off, n, c - n)
                tt += t; n = c; flags |= fl
                e = max(max(abs(x[j] - ref[g][j]) for j in range(4)) / SX, abs(p - ref[g][4]) / SP)
                worst = max(worst, e)
                print("  t = %.3f s: xp %.10f xr %.10f gate %.10f q %.12f P %.10f flag %d; max relative |board - Julia| %.3e"
                      % (c * H, x[0], x[1], x[2], x[3], p, fl, e))
            print("  host elapsed time %.4f s (simulated %.4f s)" % (tt, cuts[-1] * H))

        N2 = 10_000_000
        print("case C: the operating point,", N2, "steps (%.3f s simulated)" % (N2 * H))
        t, x, xc, p, fl = run(X0, [0.0] * 4, US_C, DU_C, 1 << 30, 1 << 30, 1 << 30, 0, N2)
        flags |= fl
        dx = max(abs(x[j] - X0[j]) for j in range(4))
        print("  host elapsed time %.4f s, %.4f us per step (h = %.4f us); max |x - x0| %.3e (limit 1e-9)" % (t, t / N2 * 1e6, H * 1e6, dx))
        ok = worst <= LIM and dx <= 1e-9 and flags == 0
        print("board check: max relative |board - Julia| %.3e (limit %.0e), case C |x - x0| %.3e (limit 1e-9), domain flag %d: %s"
              % (worst, LIM, dx, flags, "PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
        """)
    end
    println("written $path")
    return path
end
