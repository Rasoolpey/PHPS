# =====================================================================================================================
# The steam unit (PHSteamUnit: IEEEG1 governor, steam chest, sections, IAPWS-95 tables) on the FPGA: one build per type
# =====================================================================================================================
# Plan and reviews: docs/realtime/reviews/fpga/45*_steam_idea_*.md (go at 45f for body A). The frame (outputs/realtime/
# steam/frame.jl): four processors in a frame of P = 110 cycles (0.88 us): A the 12 dry storages (this file, so far),
# W the wet-capable slot, B the 11 sections, C the rest (reductions, resistances, extraction, heat, the S1 update, the
# governor). Record: docs/realtime/models/STEAM.md (to be written with the full build).

# ---------------------------------------------------------------------------------------------------------------------
# 1. Body A: one dry storage (package steam_state -> _vapour_state; src/phsim/steamprops.jl 430-459)
# ---------------------------------------------------------------------------------------------------------------------
# Item = one dry storage slot k of the instance (its V_k, its table window): from the states (M, U) of the slot
#   u = U / M, v = V / M, X = log(M / V)                      (rho = M / V, X = log rho)
#   ug, ut = gas table outputs 2, 3 at X (Hermite, 1-D)        clear = u >= ug (1 + 1e-6)
#   xi = (u - ug) / (ut - ug)                                  p, s = vapour map outputs 2, 3 at (X, xi) (Hermite, 2-D)
#   h = u + p v                                                flag = !clear or a coordinate outside its window
# Scales (the 48-bit words): per slot a power of two Mref_k = 2^e_k (M0_k / Mref_k in (1/2, 1]); M~ = M / Mref_k,
# U~ = U / (Mref_k 1e6), so u~ = U~ / M~ is u in MJ/kg for every slot, v = (V_k / Mref_k) / M~ in m3/kg,
# X = log M~ + (log Mref_k - log V_k); table words: u, h, p in 1e6 (J/kg, Pa), s in 1e3 J/(kg K).
# Units (no divider): a = m 2^k, m in [1/2, 1) (a ladder of exact scalings); one ROM of 1024 words 1/c_j (c_j the
# centre of m's cell j) seeds the reciprocal (two Newton steps y <- y (2 - m y)) and gives log m = log(1 + r) - log w_j
# with r = m w_j - 1, |r| <= 2^-11, log(1 + r) = r (1 + r (-1/2 + r (1/3 - r / 4))) (error <= 6e-18); the first product
# m w_j is shared by the log and the reciprocal. 70 products per item (the count graph: 71).

const STEAM_RT_DIR = joinpath(@__DIR__)
const _STPKG = parentmodule(@__MODULE__)   # the package (tables, PHSteamUnit, steam_state)
const Wt = Float64           # cell coordinates and Hermite weights (ap_fixed<48, 2>)
const Tw = Float64           # table words (ap_fixed<48, 8>)
const Tc = Float64           # a cell's unclamped coordinate (ap_fixed<48, 10, AP_RND_CONV, AP_SAT>)
const Lw = Float64           # log c_j words (ap_fixed<48, 1>)
const Rs = Float64           # the log unit's r, |r| < 2^-10 (ap_fixed<48, -9>)
const Cf = Float64           # coefficients (ap_fixed<48, 8>)
const Dq = Float64           # body B: scaled pressures, D~, z, p v, the root units' arguments (ap_fixed<48, 3>)
const Sd = Float64           # body B: the root units' seeds (ap_ufixed<18, 2>: 16 fraction bits)
const Tx = Float64           # body W: the exp unit's argument t in [0, 16) (ap_fixed<48, 6>)

const STEAM_RT_I = (St = 5, Sig = 8, Wt = 2, Tw = 8, Tc = 10, Rq = 4, Lw = 1, Rs = -9, Cf = 8, Dq = 3, Tx = 6, Kc = -4, Ic = -3)
const STEAM_RT_TYPES = [:St => "ap_fixed<48, 5, AP_RND_CONV>",            # M~, U~ (+-16)
                        :Sig => "ap_fixed<48, 8, AP_RND_CONV>",           # u, v, p, h, s, X (+-128)
                        :Wt => "ap_fixed<48, 2, AP_RND_CONV>",            # t, Hermite weights (+-2)
                        :Tw => "ap_fixed<48, 8, AP_RND_CONV>",            # table words
                        :Tc => "ap_fixed<48, 10, AP_RND_CONV, AP_SAT>",   # cell coordinates (saturating: no wrap)
                        :Rq => "ap_fixed<48, 4, AP_RND_CONV>",            # m, y, 2 - m y
                        :Lw => "ap_fixed<48, 1, AP_RND_CONV>",            # log c_j
                        :Rs => "ap_fixed<48, -9, AP_RND_CONV>",           # r
                        :Cf => "ap_fixed<48, 8, AP_RND_CONV>",            # coefficients
                        :Dq => "ap_fixed<48, 3, AP_RND_CONV>",            # body B: p~, D~, z, p v (+-4)
                        :Sd => "ap_ufixed<18, 2>",                        # body B: root seeds (exact on 2^-16)
                        :Tx => "ap_fixed<48, 6, AP_RND_CONV>",            # body W: the exp unit's t
                        :Kc => "ap_fixed<48, -4, AP_RND_CONV>",           # body C: S1 coefficients
                        :Ic => "ap_fixed<48, -3, AP_RND_CONV>"]           # body C: S1 increments, remainders
const STEAM_RT_NA = 12                      # dry storage items (path C: 13 storages, 1 wet slot; path A: 8 used)
const STEAM_RT_WIN = (nx = 14, ny = 18)     # window nodes (13 x 17 cells)
const STEAM_RT_WW = STEAM_RT_NA * 2 * STEAM_RT_WIN.nx * STEAM_RT_WIN.ny   # words per bank: 6,048
const STEAM_RT_WET = Dict(:A => "lp_ext", :C => "lp_ext3")
const STEAM_RT_SC = (u = 1e-6, p = 1e-6, s = 1e-3)
const STEAM_RT_KMIN, STEAM_RT_KMAX = -9, 3  # the normalisation ladder: 2^-10 <= a < 2^3

# ROMs: the reciprocal / log words (Rq), the logs (Lw: -log of the quantised word, so log m is exact up to r's series),
# k ln 2 for the ladder, the gas table (rows V2, V3, dx D2, dx D3 in 1e6)
steam_rt_lgi() = [_q48(2 / (BigFloat(1024 + j) / 2048 + BigFloat(1025 + j) / 2048), STEAM_RT_I.Rq) for j in 0:1023]
steam_rt_lgl() = [_q48(-log(BigFloat(w)), STEAM_RT_I.Lw) for w in steam_rt_lgi()]
steam_rt_kln2() = [_q48(k * log(BigFloat(2)), STEAM_RT_I.Cf) for k in STEAM_RT_KMIN:STEAM_RT_KMAX]
function steam_rt_gasrom(tb)
    g = tb.gas; S = STEAM_RT_SC.u
    return [_q48(r == 1 ? g.V[2, i] * S : r == 2 ? g.V[3, i] * S : r == 3 ? g.dx * g.D[2, i] * S : g.dx * g.D[3, i] * S,
                 STEAM_RT_I.Tw) for r in 1:4, i in 1:g.n]
end
const _STEAM_TB = Ref{Any}(nothing)
steam_rt_tables() = (_STEAM_TB[] === nothing && (_STEAM_TB[] = _STPKG.steam_tables()); _STEAM_TB[])
hls_rom!(:STA_LGI, steam_rt_lgi())
hls_rom!(:STA_LGL, steam_rt_lgl())
hls_rom!(:STA_KLN2, steam_rt_kln2())
hls_rom!(:STA_GAS, steam_rt_gasrom(steam_rt_tables()))
# the instance's windows (run-time data: written by the load mode of steam_a_run!)
for b in (:STA_VF, :STA_VX, :STA_VY, :STA_VZ); hls_keep!(b, zeros(STEAM_RT_WW)); end

# ---------------------------------------------------------------------------------------------------------------------
# 2. The coefficient struct and the units (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
@hls struct SteamACoefs
    vt::Cf[12]          # V_k / Mref_k
    cx::Cf[12]          # log Mref_k - log V_k
    xlo::Int[12]        # the X window: first and last cell (0-based table cells)
    xhi::Int[12]
    xflo::Cf[12]        # first cell, last cell + 1 (as numbers)
    xfhi::Cf[12]
    xtlo::Cf[12]        # in the window: xtlo <= t <= xthi (the table edge's tolerance included)
    xthi::Cf[12]
    xcl::Wt[12]         # the coordinate's clamp outside the window: [xcl, xch]
    xch::Wt[12]
    ylo::Int[12]        # the same for xi
    yhi::Int[12]
    yflo::Cf[12]
    yfhi::Cf[12]
    ytlo::Cf[12]
    ythi::Cf[12]
    ycl::Wt[12]
    ych::Wt[12]
    gx0::Cf             # the X axis (gas table and vapour map): x0, 1 / dx
    gidx::Cf
    vy0::Cf             # the xi axis: y0, 1 / dy
    vidy::Cf
    eps1::Cf            # 1 + 1e-6 (the clear test)
end

# a = m 2^k, m in [1/2, 1), for 2^-10 <= a < 2^3 (exact scalings); k = 99 outside (the flag), m then 3/4
@hls function sta_norm(a::Sig, mo::Rq[1])::Int
    @pragma "HLS INLINE"
    k::Int = 99
    m::Rq = 0.75
    c8::Sig = 8
    c4::Sig = 4
    c2::Sig = 2
    c1::Sig = 1
    c0::Sig = 0.5
    cm1::Sig = 0.25
    cm2::Sig = 0.125
    cm3::Sig = 0.0625
    cm4::Sig = 0.03125
    cm5::Sig = 0.015625
    cm6::Sig = 0.0078125
    cm7::Sig = 0.00390625
    cm8::Sig = 0.001953125
    cm9::Sig = 0.0009765625
    if a >= c8
        k = 99
    elseif a >= c4
        m = scale2(a, -3)
        k = 3
    elseif a >= c2
        m = scale2(a, -2)
        k = 2
    elseif a >= c1
        m = scale2(a, -1)
        k = 1
    elseif a >= c0
        m = a
        k = 0
    elseif a >= cm1
        m = scale2(a, 1)
        k = -1
    elseif a >= cm2
        m = scale2(a, 2)
        k = -2
    elseif a >= cm3
        m = scale2(a, 3)
        k = -3
    elseif a >= cm4
        m = scale2(a, 4)
        k = -4
    elseif a >= cm5
        m = scale2(a, 5)
        k = -5
    elseif a >= cm6
        m = scale2(a, 6)
        k = -6
    elseif a >= cm7
        m = scale2(a, 7)
        k = -7
    elseif a >= cm8
        m = scale2(a, 8)
        k = -8
    elseif a >= cm9
        m = scale2(a, 9)
        k = -9
    end
    mo[1] = m
    return k
end

# x 2^-k for the ladder's k (k = 99: x)
@hls function sta_sh(x::Sig, k::Int)::Sig
    @pragma "HLS INLINE"
    r::Sig = x
    if k == 3
        r = scale2(x, -3)
    elseif k == 2
        r = scale2(x, -2)
    elseif k == 1
        r = scale2(x, -1)
    elseif k == -1
        r = scale2(x, 1)
    elseif k == -2
        r = scale2(x, 2)
    elseif k == -3
        r = scale2(x, 3)
    elseif k == -4
        r = scale2(x, 4)
    elseif k == -5
        r = scale2(x, 5)
    elseif k == -6
        r = scale2(x, 6)
    elseif k == -7
        r = scale2(x, 7)
    elseif k == -8
        r = scale2(x, 8)
    elseif k == -9
        r = scale2(x, 9)
    end
    return r
end

# the ROM cell of m in [1/2, 1): j = floor(2048 m) - 1024 in [0, 1023]
@hls function sta_j(m::Rq)::Int
    @pragma "HLS INLINE"
    j::Int = ifloor(scale2(m, 11)) - 1024
    if j < 0
        j = 0
    end
    if j > 1023
        j = 1023
    end
    return j
end

# 1 / m from the ROM word w (the seed) and its first product e = m w (two Newton steps)
@hls function sta_rcp(m::Rq, w::Rq, e::Rq)::Rq
    @pragma "HLS INLINE"
    two::Rq = 2
    t1::Rq = two - e
    y1::Rq = w * t1
    e2::Rq = m * y1
    t2::Rq = two - e2
    y::Rq = y1 * t2
    return y
end

# a cell of a window: t = (x - x0) / dx (given), the index clamped into [lo, hi], the local index i - lo in ci; the
# coordinate t - i, clamped into [cl, ch] when outside the window; returns whether t was inside (tlo <= t <= thi)
@hls function sta_cell(tt::Tc, lo::Int, hi::Int, flo::Cf, fhi::Cf, tlo::Cf, thi::Cf, cl::Wt, ch::Wt, ci::Int[1], ct::Wt[1])::Bool
    @pragma "HLS INLINE"
    ok::Bool = tt >= tlo && tt <= thi
    i::Int = lo
    if tt < flo
        i = lo
    elseif tt >= fhi
        i = hi
    else
        i = ifloor(tt - flo) + lo
    end
    tf::Tc = tt - i
    t::Wt = 0
    if ok
        t = tf
    elseif tf < cl
        t = cl
    elseif tf > ch
        t = ch
    else
        t = tf
    end
    ci[1] = i - lo
    ct[1] = t
    return ok
end

# Hermite weights of t: w = (h00, h01, h10, h11)
@hls function sta_basis(t::Wt, w::Wt[4])
    @pragma "HLS INLINE"
    one::Wt = 1
    t2::Wt = t * t
    t3::Wt = t2 * t
    a::Wt = t3 - t2
    h01::Wt = t2 + t2 + t2 - t3 - t3
    w[1] = one - h01
    w[2] = h01
    w[3] = a - t2 + t
    w[4] = a
end

# one output of the 2-D window at node address b (0-based, node (i, j)); a, c the X weights (h00, h01), (h10, h11);
# bb, d the xi weights; words F, dx Fx, dy Fy, dx dy Fxy (node (i + 1, j): b + 18; (i, j + 1): b + 1)
@hls function sta_v2(F::Tw[6048], X::Tw[6048], Y::Tw[6048], Z::Tw[6048], b::Int, a0::Wt, a1::Wt, c0::Wt, c1::Wt,
                     b0::Wt, b1::Wt, d0::Wt, d1::Wt)::Tw
    @pragma "HLS INLINE"
    s1::Tw = a0 * F[b + 1] + a1 * F[b + 19] + c0 * X[b + 1] + c1 * X[b + 19]
    s2::Tw = a0 * Y[b + 1] + a1 * Y[b + 19] + c0 * Z[b + 1] + c1 * Z[b + 19]
    t1::Tw = a0 * F[b + 2] + a1 * F[b + 20] + c0 * X[b + 2] + c1 * X[b + 20]
    t2::Tw = a0 * Y[b + 2] + a1 * Y[b + 20] + c0 * Z[b + 2] + c1 * Z[b + 20]
    r::Tw = b0 * s1 + d0 * s2 + b1 * t1 + d1 * t2
    return r
end

# ---------------------------------------------------------------------------------------------------------------------
# 3. BODY A: the 12 dry storages, one item per II_A = 9 cycles (the item loop; the frame stalls it, frame.jl)
# ---------------------------------------------------------------------------------------------------------------------
# mode 1: write the word lw at la of the windows (banks F, X, Y, Z one after another, 6,048 words each)
# mode 0: the 12 items: (M~, U~) -> p, h, v, s (in 1e6 / 1e6 / m3/kg / 1e3), flag; for the T lookups of the heated
#         slots (processor C) the 8 Hermite weights (X then xi) and the 2 local cells
@hls function steam_a_run!(k::SteamACoefs, mode::Int, la::Int, lw::Tw, M::St[12], U::St[12], p::Sig[12], hh::Sig[12],
                           v::Sig[12], s::Sig[12], fl::Int[12], wo::Wt[12, 8], co::Int[12, 2])
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=mode"
    @pragma "HLS INTERFACE mode=s_axilite port=la"
    @pragma "HLS INTERFACE mode=s_axilite port=lw"
    @pragma "HLS INTERFACE mode=s_axilite port=M"
    @pragma "HLS INTERFACE mode=s_axilite port=U"
    @pragma "HLS INTERFACE mode=s_axilite port=p"
    @pragma "HLS INTERFACE mode=s_axilite port=hh"
    @pragma "HLS INTERFACE mode=s_axilite port=v"
    @pragma "HLS INTERFACE mode=s_axilite port=s"
    @pragma "HLS INTERFACE mode=s_axilite port=fl"
    @pragma "HLS INTERFACE mode=s_axilite port=wo"
    @pragma "HLS INTERFACE mode=s_axilite port=co"
    VF::Tw[6048] = keep(:STA_VF)
    VX::Tw[6048] = keep(:STA_VX)
    VY::Tw[6048] = keep(:STA_VY)
    VZ::Tw[6048] = keep(:STA_VZ)
    @pragma "HLS BIND_STORAGE variable=VF type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VX type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VY type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VZ type=ram_t2p impl=bram"
    LGI::Rq[1024] = rom(:STA_LGI)
    LGL::Lw[1024] = rom(:STA_LGL)
    KL::Cf[13] = rom(:STA_KLN2)
    G::Tw[4, 91] = rom(:STA_GAS)
    @pragma "HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=G type=rom_2p impl=bram"
    if mode == 1
        if la < 6048
            VF[la + 1] = lw
        elseif la < 12096
            VX[la - 6047] = lw
        elseif la < 18144
            VY[la - 12095] = lw
        else
            VZ[la - 18143] = lw
        end
    else
        # the per-item coefficients in local arrays (the struct is one single-port AXI memory: 18 reads per item
        # there forced II 18, synthesis 1)
        cvt::Cf[12]
        ccx::Cf[12]
        cxlo::Int[12]
        cxhi::Int[12]
        cxflo::Cf[12]
        cxfhi::Cf[12]
        cxtlo::Cf[12]
        cxthi::Cf[12]
        cxcl::Wt[12]
        cxch::Wt[12]
        cylo::Int[12]
        cyhi::Int[12]
        cyflo::Cf[12]
        cyfhi::Cf[12]
        cytlo::Cf[12]
        cythi::Cf[12]
        cycl::Wt[12]
        cych::Wt[12]
        for i in 1:12
            cvt[i] = k.vt[i]
            ccx[i] = k.cx[i]
            cxlo[i] = k.xlo[i]
            cxhi[i] = k.xhi[i]
            cxflo[i] = k.xflo[i]
            cxfhi[i] = k.xfhi[i]
            cxtlo[i] = k.xtlo[i]
            cxthi[i] = k.xthi[i]
            cxcl[i] = k.xcl[i]
            cxch[i] = k.xch[i]
            cylo[i] = k.ylo[i]
            cyhi[i] = k.yhi[i]
            cyflo[i] = k.yflo[i]
            cyfhi[i] = k.yfhi[i]
            cytlo[i] = k.ytlo[i]
            cythi[i] = k.ythi[i]
            cycl[i] = k.ycl[i]
            cych[i] = k.ych[i]
        end
        one::Rq = 1
        third::Rq = 0.333333333333333333
        half::Rq = 0.5
        for it in 1:12
            @pragma "HLS PIPELINE II=9"
            @pragma "HLS ALLOCATION operation instances=mul limit=8"
            Mi::St = M[it]
            Ui::St = U[it]
            # 1 / M~ and log M~ (one normalisation, one ROM word)
            mo::Rq[1]
            km::Int = sta_norm(Mi, mo)
            m::Rq = mo[1]
            j::Int = sta_j(m)
            w::Rq = LGI[j + 1]
            e::Rq = m * w
            r::Rs = e - one
            y::Rq = sta_rcp(m, w, e)
            q1::Rq = third - scale2(r, -2)
            q2::Rq = r * q1 - half
            q3::Rq = r * q2 + one
            lg::Rs = r * q3
            kl::Int = km
            if km == 99
                kl = 0
            end
            X::Sig = lg + LGL[j + 1] + KL[kl + 10] + ccx[it]
            uy::Sig = Ui * y
            u::Sig = sta_sh(uy, km)
            vy::Sig = cvt[it] * y
            vv::Sig = sta_sh(vy, km)
            # the X cell (gas table, window)
            tx::Tc = (X - k.gx0) * k.gidx
            ci::Int[1]
            ct::Wt[1]
            okx::Bool = sta_cell(tx, cxlo[it], cxhi[it], cxflo[it], cxfhi[it], cxtlo[it], cxthi[it], cxcl[it], cxch[it], ci, ct)
            lx::Int = ci[1]
            ix::Int = lx + cxlo[it]
            wx::Wt[4]
            sta_basis(ct[1], wx)
            ug::Sig = wx[1] * G[1, ix + 1] + wx[2] * G[1, ix + 2] + wx[3] * G[3, ix + 1] + wx[4] * G[3, ix + 2]
            ut::Sig = wx[1] * G[2, ix + 1] + wx[2] * G[2, ix + 2] + wx[3] * G[4, ix + 1] + wx[4] * G[4, ix + 2]
            uge::Sig = ug * k.eps1
            clear::Bool = u >= uge
            # xi = (u - ug) / (ut - ug)
            dd::Sig = ut - ug
            mo2::Rq[1]
            kd::Int = sta_norm(dd, mo2)
            m2::Rq = mo2[1]
            j2::Int = sta_j(m2)
            w2::Rq = LGI[j2 + 1]
            e2::Rq = m2 * w2
            y2::Rq = sta_rcp(m2, w2, e2)
            du::Sig = u - ug
            xy::Tc = du * y2
            xi::Tc = sta_sh(xy, kd)
            # the xi cell (window)
            ty::Tc = (xi - k.vy0) * k.vidy
            cj::Int[1]
            cu::Wt[1]
            oky::Bool = sta_cell(ty, cylo[it], cyhi[it], cyflo[it], cyfhi[it], cytlo[it], cythi[it], cycl[it], cych[it], cj, cu)
            ly::Int = cj[1]
            wy::Wt[4]
            sta_basis(cu[1], wy)
            # p, s from the window (node (lx, ly); outputs 2 then 3 of the slot)
            bp::Int = (it - 1) * 504 + lx * 18 + ly
            pp::Sig = sta_v2(VF, VX, VY, VZ, bp, wx[1], wx[2], wx[3], wx[4], wy[1], wy[2], wy[3], wy[4])
            ss::Sig = sta_v2(VF, VX, VY, VZ, bp + 252, wx[1], wx[2], wx[3], wx[4], wy[1], wy[2], wy[3], wy[4])
            pv::Sig = pp * vv
            p[it] = pp
            hh[it] = u + pv
            v[it] = vv
            s[it] = ss
            f::Int = 0
            if !clear || !okx || !oky || km == 99 || kd == 99
                f = 1
            end
            fl[it] = f
            for q in 1:4
                @pragma "HLS UNROLL"
                wo[it, q] = wx[q]
                wo[it, q + 4] = wy[q]
            end
            co[it, 1] = lx
            co[it, 2] = ly
        end
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 3b. BODY B: one turbine section (package _steam_eval, sections; steam_h_isentropic), 11 items at II_B = 10
# ---------------------------------------------------------------------------------------------------------------------
# Item = section j: inlet storage a, outlet b (terminal: the condenser at p_cond), from the storage outputs of A and W,
# p (1e6 Pa), h (1e6 J/kg), v (m3/kg), s (1e3 J/(kg K)), in storage order:
#   p~ = p / 2^ep_j at both ends (2^ep_j >= the inlet pressure at x0 in MPa: p~ in (1/2, 1] at x0); a pressure
#        >= 1.375 x 2^ep_j is flagged (p~ < 1.375 keeps z < 4; with the instances' x0, at least 1.41 x the x0 pressure)
#   D~ = p~in^2 - p~out^2, fwd = D~ >= 0
#   z = D~^2 + delta~_j^2 (delta~ = delta_j / (1e6 2^ep_j)^2); pd, vd = the donor's p, v (the inlet when fwd or terminal)
#   q~ = Kq_j D~ z^(-1/4) (pd vd)^(-1/2), Kq_j = K_j 1e3 2^ep_j / Qref (Qref = 2^e >= the valve's full flow cv);
#        the terminal section reversed: q~ = 0 (no flow back from the condenser)
#   l = log pout (terminal: log p_cond); its cell in the section's window; T, h_f, s_f, s_g, s_top at l (1-D Hermite)
#   eta = (s_a - s_g) / (s_top - s_g); hs = the isentrope window at (l, eta) (2-D Hermite); the wet end h_f + T (s_a - s_f)
#   hs = the dry value when s_a >= s_g or the item has no wet end; hout = h_a - eta_j (h_a - hs)
#   e1 = q~ (fwd ? h_a : hb), e2 = q~ (fwd ? hout : hb), hb = (terminal ? h_a : h_b); C sums them (review 45d, 2)
#   flag (only where the isentrope's result is used, fwd and q~ != 0): l or eta outside its window, an end in the dome
#   where the item has none, entropy below saturated liquid; a root argument outside its range
# Units (no divider): z^(-1/4) and (p v)^(-1/2) from 18-bit seeds (1024 per octave) and two Newton steps, the argument
# normalised in steps of 2^4 (z) and 2^2 (p v) by compare-and-shift ladders; log p and 1 / (s_top - s_g) as in body A
# (the 1/c_j ROM, one normalisation ladder over 2^-31 .. 2^7). 80 products per item (the count graph: 80).
const STEAM_RT_NB = 11                      # section items (path C: 11; path A: 8, the last repeated)
const STEAM_RT_EWIN = (nl = 19, ne = 13)    # isentrope window nodes (18 x 12 cells)
const STEAM_RT_EW = STEAM_RT_NB * STEAM_RT_EWIN.nl * STEAM_RT_EWIN.ne   # words per bank: 2,717

# the seeds of m^(-1/4), m in [1/16, 1) (octave o = [2^(o-4), 2^(o-3)), 1024 bins), and of m^(-1/2), m in [1/4, 1)
# (2 octaves); the value at the bin's centre on 2^-16 (relative error <= 2.5e-4 and 4.9e-4: two Newton steps 5e-14, 2e-13)
function steam_rt_seeds(no, e0, pw)
    s = zeros(1024no)
    for o in 0:no-1, j in 0:1023
        c = BigFloat(2)^(o + e0) * (1 + (BigFloat(j) + 0.5) / 1024)
        s[1024o+j+1] = Float64(round(c^pw * 2^16) / 2^16)
    end
    return s
end
steam_rt_klb() = [_q48(k * log(BigFloat(2)) + log(BigFloat(10)^6), STEAM_RT_I.Cf) for k in -31:7]
# the 1-D tables of body B at l: rows T (1e3 K), h_f (1e6), s_f, s_g (1e3, saturation), s_top (1e3, the top isentrope);
# values V and dx-scaled slopes dx D
function steam_rt_satrom(tb, slope::Bool)
    rows = ((tb.sat, 1, 1e-3), (tb.sat, 6, 1e-6), (tb.sat, 8, 1e-3), (tb.sat, 9, 1e-3), (tb.istop, 1, 1e-3))
    return [_q48(slope ? t.dx * t.D[r, i] * S : t.V[r, i] * S, STEAM_RT_I.Tw) for (t, r, S) in rows, i in 1:tb.sat.n]
end
hls_rom!(:STB_QS, steam_rt_seeds(4, -4, -1 / 4))
hls_rom!(:STB_RS, steam_rt_seeds(2, -2, -1 / 2))
hls_rom!(:STB_KLN, steam_rt_klb())
hls_rom!(:STB_SV, steam_rt_satrom(steam_rt_tables(), false))
hls_rom!(:STB_SD, steam_rt_satrom(steam_rt_tables(), true))
for b in (:STB_IF, :STB_IX, :STB_IY, :STB_IZ); hls_keep!(b, zeros(STEAM_RT_EW)); end
# the per-item coefficients, loaded once (mode 2) into kept arrays (no per-call copy)
const STEAM_RT_BINT = (:IA, :IB, :TM, :WO, :EP, :LLO, :LHI, :ELO, :EHI)
const STEAM_RT_BNUM = (:PM, :KQ, :D2, :ET, :LFLO, :LFHI, :LTLO, :LTHI, :LCL, :LCH, :EFLO, :EFHI, :ETLO, :ETHI, :ECL, :ECH)
for n in STEAM_RT_BINT; hls_keep!(Symbol(:STB_, n), zeros(Int, STEAM_RT_NB)); end
for n in STEAM_RT_BNUM; hls_keep!(Symbol(:STB_, n), zeros(STEAM_RT_NB)); end

@hls struct SteamBCoefs
    ia::Int[11]         # the inlet storage (0-based, storage order)
    ib::Int[11]         # the outlet storage (terminal: the inlet's, not used)
    tm::Int[11]         # 1: into the condenser
    wo::Int[11]         # 1: the item has a wet end
    ep::Int[11]         # p~ = p / 2^ep
    pm::Sig[11]         # the pressure limit 1.375 x 2^ep (1e6 Pa)
    kq::Cf[11]          # Kq_j
    d2::Dq[11]          # delta~_j^2
    et::Wt[11]          # the section's isentropic efficiency
    llo::Int[11]        # the l window (as body A's cells)
    lhi::Int[11]
    lflo::Cf[11]
    lfhi::Cf[11]
    ltlo::Cf[11]
    lthi::Cf[11]
    lcl::Wt[11]
    lch::Wt[11]
    elo::Int[11]        # the eta window
    ehi::Int[11]
    eflo::Cf[11]
    efhi::Cf[11]
    etlo::Cf[11]
    ethi::Cf[11]
    ecl::Wt[11]
    ech::Wt[11]
    pc::Sig             # p_cond (1e6 Pa)
    lpc::Cf             # log p_cond (Pa)
    sx0::Cf             # the l axis: x0, 1 / dx (saturation, top isentrope and isentrope map share it)
    sidx::Cf
    ey0::Cf             # the eta axis: y0, 1 / dy
    eidy::Cf
end

# a = m 2^k, m in [1/2, 1), for 2^-31 <= a < 128 (compare-and-shift ladder); k = 99 outside (m then 3/4)
@hls function stb_norm(a::Sig, mo::Rq[1])::Int
    @pragma "HLS INLINE"
    m::Sig = a
    k::Int = 0
    c8::Sig = 8
    c2::Sig = 2
    c1::Sig = 1
    e16::Sig = 0.0000152587890625
    e8::Sig = 0.00390625
    e4::Sig = 0.0625
    e2::Sig = 0.25
    e1::Sig = 0.5
    if m >= c8
        m = scale2(m, -4)
        k = 4
    end
    if m >= c2
        m = scale2(m, -2)
        k = k + 2
    end
    if m >= c1
        m = scale2(m, -1)
        k = k + 1
    end
    if m < e16
        m = scale2(m, 16)
        k = k - 16
    end
    if m < e8
        m = scale2(m, 8)
        k = k - 8
    end
    if m < e4
        m = scale2(m, 4)
        k = k - 4
    end
    if m < e2
        m = scale2(m, 2)
        k = k - 2
    end
    if m < e1
        m = scale2(m, 1)
        k = k - 1
    end
    r::Rq = m
    tq::Rq = 0.75
    if m < e1
        k = 99
        r = tq
    end
    mo[1] = r
    return k
end

# z = m 16^(-n), m in [1/16, 1), for 2^-32 <= z < 4 (steps of 2^4): z^(-1/4) = m^(-1/4) 2^n; n = 99 outside (m 1/2)
@hls function stb_nq(z::Dq, mo::Rq[1])::Int
    @pragma "HLS INLINE"
    m::Dq = z
    n::Int = 0
    one::Dq = 1
    e16::Dq = 0.0000152587890625
    e8::Dq = 0.00390625
    e4::Dq = 0.0625
    if m >= one
        m = scale2(m, -4)
        n = -1
    end
    if m < e16
        m = scale2(m, 16)
        n = n + 4
    end
    if m < e8
        m = scale2(m, 8)
        n = n + 2
    end
    if m < e4
        m = scale2(m, 4)
        n = n + 1
    end
    r::Rq = m
    hq::Rq = 0.5
    if m < e4
        n = 99
        r = hq
    end
    mo[1] = r
    return n
end

# x = m 4^(-n), m in [1/4, 1), for 2^-14 <= x < 4 (steps of 2^2): x^(-1/2) = m^(-1/2) 2^n; n = 99 outside (m 1/2)
@hls function stb_nr(x::Dq, mo::Rq[1])::Int
    @pragma "HLS INLINE"
    m::Dq = x
    n::Int = 0
    one::Dq = 1
    e8::Dq = 0.00390625
    e4::Dq = 0.0625
    e2::Dq = 0.25
    if m >= one
        m = scale2(m, -2)
        n = -1
    end
    if m < e8
        m = scale2(m, 8)
        n = n + 4
    end
    if m < e4
        m = scale2(m, 4)
        n = n + 2
    end
    if m < e2
        m = scale2(m, 2)
        n = n + 1
    end
    r::Rq = m
    hq::Rq = 0.5
    if m < e2
        n = 99
        r = hq
    end
    mo[1] = r
    return n
end

# the seed bins: m in [1/16, 1) -> octave * 1024 + bin (4,096); m in [1/4, 1) -> 2,048
@hls function stb_jq(m::Rq)::Int
    @pragma "HLS INLINE"
    half::Rq = 0.5
    quarter::Rq = 0.25
    eighth::Rq = 0.125
    j::Int = 0
    if m >= half
        j = ifloor(scale2(m, 11)) + 2048
    elseif m >= quarter
        j = ifloor(scale2(m, 12)) + 1024
    elseif m >= eighth
        j = ifloor(scale2(m, 13))
    else
        j = ifloor(scale2(m, 14)) - 1024
    end
    if j < 0
        j = 0
    end
    if j > 4095
        j = 4095
    end
    return j
end
@hls function stb_jr(m::Rq)::Int
    @pragma "HLS INLINE"
    half::Rq = 0.5
    j::Int = 0
    if m >= half
        j = ifloor(scale2(m, 11))
    else
        j = ifloor(scale2(m, 12)) - 1024
    end
    if j < 0
        j = 0
    end
    if j > 2047
        j = 2047
    end
    return j
end

# m^(-1/4) from the seed: y <- y (5 - m y^4) / 4, twice (4 products each)
@hls function stb_qrt(m::Rq, y0::Rq)::Rq
    @pragma "HLS INLINE"
    five::Rq = 5
    y::Rq = y0
    for i in 1:2
        y2::Rq = y * y
        a::Rq = m * y2
        b::Rq = a * y2
        t::Rq = five - b
        yt::Rq = y * t
        y = scale2(yt, -2)
    end
    return y
end
# m^(-1/2) from the seed: y <- y (3 - m y^2) / 2, twice (3 products each)
@hls function stb_rsq(m::Rq, y0::Rq)::Rq
    @pragma "HLS INLINE"
    three::Rq = 3
    y::Rq = y0
    for i in 1:2
        y2::Rq = y * y
        a::Rq = m * y2
        t::Rq = three - a
        yt::Rq = y * t
        y = scale2(yt, -1)
    end
    return y
end

# x 2^-e for the section's pressure exponent e in [-7, 6] (VHP 28.8 MPa: 5; LP 22 kPa: -5) (one exact scaling, selected)
@hls function stb_ps(x::Sig, e::Int)::Dq
    @pragma "HLS INLINE"
    r::Dq = x
    if e == 6
        r = scale2(x, -6)
    elseif e == 5
        r = scale2(x, -5)
    elseif e == 4
        r = scale2(x, -4)
    elseif e == 3
        r = scale2(x, -3)
    elseif e == 2
        r = scale2(x, -2)
    elseif e == 1
        r = scale2(x, -1)
    elseif e == -1
        r = scale2(x, 1)
    elseif e == -2
        r = scale2(x, 2)
    elseif e == -3
        r = scale2(x, 3)
    elseif e == -4
        r = scale2(x, 4)
    elseif e == -5
        r = scale2(x, 5)
    elseif e == -6
        r = scale2(x, 6)
    elseif e == -7
        r = scale2(x, 7)
    end
    return r
end

# x 2^n for |n| < 16 (a ladder of exact scalings)
@hls function stb_sh(x::Sig, n::Int)::Sig
    @pragma "HLS INLINE"
    r::Sig = x
    s::Int = n
    if n < 0
        s = 0 - n
        if s >= 8
            r = scale2(r, -8)
            s = s - 8
        end
        if s >= 4
            r = scale2(r, -4)
            s = s - 4
        end
        if s >= 2
            r = scale2(r, -2)
            s = s - 2
        end
        if s >= 1
            r = scale2(r, -1)
        end
    else
        if s >= 8
            r = scale2(r, 8)
            s = s - 8
        end
        if s >= 4
            r = scale2(r, 4)
            s = s - 4
        end
        if s >= 2
            r = scale2(r, 2)
            s = s - 2
        end
        if s >= 1
            r = scale2(r, 1)
        end
    end
    return r
end

# the isentrope window at node address b (0-based node (i, j); (i + 1, j): b + 13; (i, j + 1): b + 1)
@hls function stb_v2(F::Tw[2717], X::Tw[2717], Y::Tw[2717], Z::Tw[2717], b::Int, a0::Wt, a1::Wt, c0::Wt, c1::Wt,
                     b0::Wt, b1::Wt, d0::Wt, d1::Wt)::Tw
    @pragma "HLS INLINE"
    s1::Tw = a0 * F[b + 1] + a1 * F[b + 14] + c0 * X[b + 1] + c1 * X[b + 14]
    s2::Tw = a0 * Y[b + 1] + a1 * Y[b + 14] + c0 * Z[b + 1] + c1 * Z[b + 14]
    t1::Tw = a0 * F[b + 2] + a1 * F[b + 15] + c0 * X[b + 2] + c1 * X[b + 15]
    t2::Tw = a0 * Y[b + 2] + a1 * Y[b + 15] + c0 * Z[b + 2] + c1 * Z[b + 15]
    r::Tw = b0 * s1 + d0 * s2 + b1 * t1 + d1 * t2
    return r
end

# mode 1: write the word lw at la of the isentrope windows (banks F, X, Y, Z, 2,717 words each)
# mode 2: load the per-item coefficients of k into the kept arrays
# mode 0: the 11 items: storage outputs (p, hh, v, s in storage order) -> q~, e1, e2 (MW / Qref), fwd, flag
@hls function steam_b_run!(k::SteamBCoefs, mode::Int, la::Int, lw::Tw, p::Sig[13], hh::Sig[13], v::Sig[13], s::Sig[13],
                           q::Sig[11], e1::Sig[11], e2::Sig[11], fw::Int[11], fl::Int[11])
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=mode"
    @pragma "HLS INTERFACE mode=s_axilite port=la"
    @pragma "HLS INTERFACE mode=s_axilite port=lw"
    @pragma "HLS INTERFACE mode=s_axilite port=p"
    @pragma "HLS INTERFACE mode=s_axilite port=hh"
    @pragma "HLS INTERFACE mode=s_axilite port=v"
    @pragma "HLS INTERFACE mode=s_axilite port=s"
    @pragma "HLS INTERFACE mode=s_axilite port=q"
    @pragma "HLS INTERFACE mode=s_axilite port=e1"
    @pragma "HLS INTERFACE mode=s_axilite port=e2"
    @pragma "HLS INTERFACE mode=s_axilite port=fw"
    @pragma "HLS INTERFACE mode=s_axilite port=fl"
    IF::Tw[2717] = keep(:STB_IF)
    IX::Tw[2717] = keep(:STB_IX)
    IY::Tw[2717] = keep(:STB_IY)
    IZ::Tw[2717] = keep(:STB_IZ)
    @pragma "HLS BIND_STORAGE variable=IF type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=IX type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=IY type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=IZ type=ram_t2p impl=bram"
    cia::Int[11] = keep(:STB_IA)
    cib::Int[11] = keep(:STB_IB)
    ctm::Int[11] = keep(:STB_TM)
    cwo::Int[11] = keep(:STB_WO)
    cep::Int[11] = keep(:STB_EP)
    cpm::Sig[11] = keep(:STB_PM)
    ckq::Cf[11] = keep(:STB_KQ)
    cd2::Dq[11] = keep(:STB_D2)
    cet::Wt[11] = keep(:STB_ET)
    cllo::Int[11] = keep(:STB_LLO)
    clhi::Int[11] = keep(:STB_LHI)
    clflo::Cf[11] = keep(:STB_LFLO)
    clfhi::Cf[11] = keep(:STB_LFHI)
    cltlo::Cf[11] = keep(:STB_LTLO)
    clthi::Cf[11] = keep(:STB_LTHI)
    clcl::Wt[11] = keep(:STB_LCL)
    clch::Wt[11] = keep(:STB_LCH)
    celo::Int[11] = keep(:STB_ELO)
    cehi::Int[11] = keep(:STB_EHI)
    ceflo::Cf[11] = keep(:STB_EFLO)
    cefhi::Cf[11] = keep(:STB_EFHI)
    cetlo::Cf[11] = keep(:STB_ETLO)
    cethi::Cf[11] = keep(:STB_ETHI)
    cecl::Wt[11] = keep(:STB_ECL)
    cech::Wt[11] = keep(:STB_ECH)
    LGI::Rq[1024] = rom(:STA_LGI)
    LGL::Lw[1024] = rom(:STA_LGL)
    KLB::Cf[39] = rom(:STB_KLN)
    QS::Sd[4096] = rom(:STB_QS)
    RS::Sd[2048] = rom(:STB_RS)
    SV::Tw[5, 121] = rom(:STB_SV)
    SD::Tw[5, 121] = rom(:STB_SD)
    @pragma "HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=QS type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=RS type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SV type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SD type=rom_2p impl=bram"
    if mode == 1
        if la < 2717
            IF[la + 1] = lw
        elseif la < 5434
            IX[la - 2716] = lw
        elseif la < 8151
            IY[la - 5433] = lw
        else
            IZ[la - 8150] = lw
        end
    elseif mode == 2
        for i in 1:11
            cia[i] = k.ia[i]
            cib[i] = k.ib[i]
            ctm[i] = k.tm[i]
            cwo[i] = k.wo[i]
            cep[i] = k.ep[i]
            cpm[i] = k.pm[i]
            ckq[i] = k.kq[i]
            cd2[i] = k.d2[i]
            cet[i] = k.et[i]
            cllo[i] = k.llo[i]
            clhi[i] = k.lhi[i]
            clflo[i] = k.lflo[i]
            clfhi[i] = k.lfhi[i]
            cltlo[i] = k.ltlo[i]
            clthi[i] = k.lthi[i]
            clcl[i] = k.lcl[i]
            clch[i] = k.lch[i]
            celo[i] = k.elo[i]
            cehi[i] = k.ehi[i]
            ceflo[i] = k.eflo[i]
            cefhi[i] = k.efhi[i]
            cetlo[i] = k.etlo[i]
            cethi[i] = k.ethi[i]
            cecl[i] = k.ecl[i]
            cech[i] = k.ech[i]
        end
    else
        one::Rq = 1
        third::Rq = 0.333333333333333333
        half::Rq = 0.5
        zq::Dq = 0
        zs::Sig = 0
        pc::Sig = k.pc
        lpc::Sig = k.lpc
        sx0::Cf = k.sx0
        sidx::Cf = k.sidx
        ey0::Cf = k.ey0
        eidy::Cf = k.eidy
        for it in 1:11
            @pragma "HLS PIPELINE II=10"
            @pragma "HLS ALLOCATION operation instances=mul limit=8"
            ia::Int = cia[it]
            ib::Int = cib[it]
            tm::Bool = ctm[it] == 1
            pin::Sig = p[ia + 1]
            pb::Sig = p[ib + 1]
            va::Sig = v[ia + 1]
            vb::Sig = v[ib + 1]
            ha::Sig = hh[ia + 1]
            hbv::Sig = hh[ib + 1]
            sa::Sig = s[ia + 1]
            pob::Sig = pb
            if tm
                pob = pc
            end
            # the flow: q~ = Kq D~ z^(-1/4) (pd vd)^(-1/2)
            pti::Dq = stb_ps(pin, cep[it])
            pto::Dq = stb_ps(pob, cep[it])
            dd::Dq = pti * pti - pto * pto
            fwd::Bool = dd >= zq
            z::Dq = dd * dd + cd2[it]
            mz::Rq[1]
            nz::Int = stb_nq(z, mz)
            jz::Int = stb_jq(mz[1])
            sz::Rq = QS[jz + 1]
            y4::Rq = stb_qrt(mz[1], sz)
            pd::Sig = pin
            vd::Sig = va
            if !fwd && !tm
                pd = pb
                vd = vb
            end
            x::Dq = pd * vd
            mx::Rq[1]
            nx::Int = stb_nr(x, mx)
            jx::Int = stb_jr(mx[1])
            sx::Rq = RS[jx + 1]
            y2::Rq = stb_rsq(mx[1], sx)
            t1::Dq = ckq[it] * dd
            t2::Dq = t1 * y4
            t3::Dq = t2 * y2
            qq::Sig = stb_sh(t3, nz + nx)
            if !fwd && tm
                qq = zs
            end
            # l = log pout (one normalisation, one ROM word; the terminal's exact log p_cond)
            ml::Rq[1]
            kp::Int = stb_norm(pob, ml)
            m::Rq = ml[1]
            j::Int = sta_j(m)
            w::Rq = LGI[j + 1]
            e::Rq = m * w
            r::Rs = e - one
            q1::Rq = third - scale2(r, -2)
            q2::Rq = r * q1 - half
            q3::Rq = r * q2 + one
            lg::Rs = r * q3
            kl::Int = kp
            if kp == 99
                kl = 0
            end
            l::Sig = lg + LGL[j + 1] + KLB[kl + 32]
            if tm
                l = lpc
            end
            # the l cell (window), the 1-D tables
            tl::Tc = (l - sx0) * sidx
            ci::Int[1]
            ct::Wt[1]
            okl::Bool = sta_cell(tl, cllo[it], clhi[it], clflo[it], clfhi[it], cltlo[it], clthi[it], clcl[it], clch[it], ci, ct)
            li::Int = ci[1]
            il::Int = li + cllo[it]
            wl::Wt[4]
            sta_basis(ct[1], wl)
            # the 1-D tables at cell il (written out: a ROM is not passed to a function, the C++ has no const parameters)
            tt::Sig = wl[1] * SV[1, il + 1] + wl[2] * SV[1, il + 2] + wl[3] * SD[1, il + 1] + wl[4] * SD[1, il + 2]
            hf::Sig = wl[1] * SV[2, il + 1] + wl[2] * SV[2, il + 2] + wl[3] * SD[2, il + 1] + wl[4] * SD[2, il + 2]
            sf::Sig = wl[1] * SV[3, il + 1] + wl[2] * SV[3, il + 2] + wl[3] * SD[3, il + 1] + wl[4] * SD[3, il + 2]
            sg::Sig = wl[1] * SV[4, il + 1] + wl[2] * SV[4, il + 2] + wl[3] * SD[4, il + 1] + wl[4] * SD[4, il + 2]
            stp::Sig = wl[1] * SV[5, il + 1] + wl[2] * SV[5, il + 2] + wl[3] * SD[5, il + 1] + wl[4] * SD[5, il + 2]
            dry::Bool = sa >= sg
            # eta = (s_a - s_g) / (s_top - s_g)
            ds::Sig = stp - sg
            md::Rq[1]
            kd::Int = stb_norm(ds, md)
            m2::Rq = md[1]
            j2::Int = sta_j(m2)
            w2::Rq = LGI[j2 + 1]
            ee2::Rq = m2 * w2
            yd::Rq = sta_rcp(m2, w2, ee2)
            dsa::Sig = sa - sg
            ey::Sig = dsa * yd
            et::Sig = stb_sh(ey, 0 - kd)
            te::Tc = (et - ey0) * eidy
            cj::Int[1]
            cu::Wt[1]
            oke::Bool = sta_cell(te, celo[it], cehi[it], ceflo[it], cefhi[it], cetlo[it], cethi[it], cecl[it], cech[it], cj, cu)
            le::Int = cj[1]
            we::Wt[4]
            sta_basis(cu[1], we)
            bi::Int = (it - 1) * 247 + li * 13 + le
            hsd::Sig = stb_v2(IF, IX, IY, IZ, bi, wl[1], wl[2], wl[3], wl[4], we[1], we[2], we[3], we[4])
            hsw::Sig = hf + tt * (sa - sf)
            hs::Sig = hsd
            if !dry && cwo[it] == 1
                hs = hsw
            end
            hout::Sig = ha - cet[it] * (ha - hs)
            hb::Sig = hbv
            if tm
                hb = ha
            end
            h1::Sig = hb
            h2::Sig = hb
            if fwd
                h1 = ha
                h2 = hout
            end
            q[it] = qq
            e1[it] = qq * h1
            e2[it] = qq * h2
            f::Int = 0
            if fwd
                fw[it] = 1
            else
                fw[it] = 0
            end
            if nz == 99 || nx == 99 || pin >= cpm[it] || pob >= cpm[it]
                f = 1
            end
            if fwd && qq != zs
                if !okl || kp == 99
                    f = 1
                end
                if dry && (!oke || kd == 99)
                    f = 1
                end
                if !dry && (cwo[it] == 0 || sa < sf)
                    f = 1
                end
            end
            fl[it] = f
        end
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 3c. BODY W: the wet-capable storage slot (package steam_state -> _vapour_state, _sat_l_of_vg, _wet_state), one item per
#     frame (a pipeline at II = P = 110 on 3 multipliers; its latency spans several frames, so the seeds come from d = 3
#     frames back, frame.jl)
# ---------------------------------------------------------------------------------------------------------------------
# From (M~, U~) of the slot and the seeds (lg0, l0) of an earlier evaluation (as hw_lib.jl's hw_storage, sl.wet):
#   the dry state as body A (p, s from the slot's vapour window; clear, okv)
#   the dry boundary lg: v_g(lg) = v, two Newton steps from lg0; u_g at lg; rg = v_g(lg) - v
#   wet = !clear && v_g(l_max) <= v <= v_g(l_min) && u < u_g(lg)
#   the wet state l: u = u_f + x (u_g - u_f), x = (v - v_f) / (v_g - v_f), two Newton steps from l0; at the final l:
#   x, rl, s = s_f + x (s_g - s_f), p = exp(l), h = u + p v
#   flag: wet ? !(-1e-9 <= x <= 1 + 1e-6, l in the table, |rl| <= 2^-34 |u|, |rg| <= 2^-30 v)
#             : (!okv || (in range && !clear && |rg| > 2^-30 v))      (2^-34 < 1e-10, 2^-30 < 1e-9: the package's tolerances
#   made a little stricter, a shift instead of a product)
# Saturation words (rows v_f, v_g, u_f, u_g, s_f, s_g): v / 16 (v_g reaches 181 m3/kg, its E words 991), u in 1e6, s in 1e3;
# per row the values V, dx D (the Hermite value), E = (6 / dx)(V_i - V_i+1) and D (the derivative in l). p = exp(l) by
# 2^t (t = l log2 e - log2 1e6 + 11 split into k, j / 256, r; 2^(j/256) from a ROM; a cubic in r, as the shunt exciter's
# unit, error <= 2.2e-12) and a shift by k - 11. 244 products (the count graph: 245).
const STEAM_RT_WR = (2, 3, 4, 5, 8, 9)           # the saturation rows W reads
const STEAM_RT_WSC = (1 / 16, 1 / 16, 1e-6, 1e-6, 1e-3, 1e-3)
function steam_rt_wsat(tb, w::Symbol)
    s = tb.sat; n = w === :E ? s.n - 1 : s.n
    return [_q48(w === :V ? s.V[r, i] * S : w === :X ? s.dx * s.D[r, i] * S : w === :D ? s.D[r, i] * S :
                 6 / s.dx * (s.V[r, i] - s.V[r, i+1]) * S, STEAM_RT_I.Tw) for (r, S) in zip(STEAM_RT_WR, STEAM_RT_WSC), i in 1:n]
end
steam_rt_ex2() = [_q48(BigFloat(2)^(BigFloat(j) / 256), STEAM_RT_I.Rq) for j in 0:255]
steam_rt_exc() = (l = log(BigFloat(2)); [_q48(l, STEAM_RT_I.Rq), _q48(l^2 / 2, STEAM_RT_I.Rq), _q48(l^3 / 6, STEAM_RT_I.Rq)])
hls_rom!(:STW_V, steam_rt_wsat(steam_rt_tables(), :V))
hls_rom!(:STW_X, steam_rt_wsat(steam_rt_tables(), :X))
hls_rom!(:STW_E, steam_rt_wsat(steam_rt_tables(), :E))
hls_rom!(:STW_D, steam_rt_wsat(steam_rt_tables(), :D))
hls_rom!(:STW_EX2, steam_rt_ex2())
hls_rom!(:STW_EXC, steam_rt_exc())
const STEAM_RT_WWW = 2 * STEAM_RT_WIN.nx * STEAM_RT_WIN.ny      # words per window bank: 504 (p, s)
for b in (:STW_VF, :STW_VX, :STW_VY, :STW_VZ); hls_keep!(b, zeros(STEAM_RT_WWW)); end

@hls struct SteamWCoefs
    vt::Cf              # V / Mref
    cx::Cf              # log Mref - log V
    xlo::Int            # the X window (as body A)
    xhi::Int
    xflo::Cf
    xfhi::Cf
    xtlo::Cf
    xthi::Cf
    xcl::Wt
    xch::Wt
    ylo::Int            # the xi window
    yhi::Int
    yflo::Cf
    yfhi::Cf
    ytlo::Cf
    ythi::Cf
    ycl::Wt
    ych::Wt
    gx0::Cf             # the X and xi axes, the clear test (as body A)
    gidx::Cf
    vy0::Cf
    vidy::Cf
    eps1::Cf
    sx0::Cf             # the saturation table's l axis: x0, 1 / dx, the in-table limits of t (tolerances included)
    sidx::Cf
    stlo::Cf
    sthi::Cf
    vlo::Sig            # v_g at l_max and at l_min (v / 16)
    vhi::Sig
    l2e::Cf             # log2 e; 11 - log2 1e6
    l2c::Cf
end

# x 2^-k for body A's ladder (k in [-9, 3]) is sta_sh; the cell of a whole 1-D table (0 .. 119) with the coordinate unclamped
# (the Newton iterates: clamped index, t = tt - i); returns whether tt was inside [tlo, thi]
@hls function stw_cell(tt::Tc, tlo::Cf, thi::Cf, ci::Int[1], ct::Wt[1])::Bool
    @pragma "HLS INLINE"
    ok::Bool = tt >= tlo && tt <= thi
    zero::Tc = 0
    top::Tc = 119
    i::Int = 0
    if tt < zero
        i = 0
    elseif tt >= top
        i = 119
    else
        i = ifloor(tt)
    end
    tf::Tc = tt - i
    ci[1] = i
    ct[1] = tf
    return ok
end

# the derivative weights of t (the Hermite derivative in l: g E_i + d10 D_i + d11 D_i+1): w = (g, d10, d11)
@hls function stw_dbasis(t::Wt, w::Wt[3])
    @pragma "HLS INLINE"
    one::Wt = 1
    t2::Wt = t * t
    w[1] = t2 - t
    w[2] = one - t - t - t - t + t2 + t2 + t2
    w[3] = t2 + t2 + t2 - t - t
end

# a signed reciprocal 1 / a (|a| in [2^-31, 128)): body B's ladder, body A's seed and Newton steps; nk[1] = 99 outside
@hls function stw_srcp(a::Sig, nk::Int[1])::Sig
    @pragma "HLS INLINE"
    LGI::Rq[1024] = rom(:STA_LGI)
    zs::Sig = 0
    b::Sig = a
    if a < zs
        b = zs - a
    end
    mo::Rq[1]
    k::Int = stb_norm(b, mo)
    m::Rq = mo[1]
    j::Int = sta_j(m)
    w::Rq = LGI[j + 1]
    e::Rq = m * w
    y::Rq = sta_rcp(m, w, e)
    kk::Int = 0 - k
    if k == 99
        kk = 0
    end
    r::Sig = stb_sh(y, kk)
    if a < zs
        r = zs - r
    end
    nk[1] = k
    return r
end

# p = exp(l) in 1e6 Pa for l in [6.5, 17]: t = l log2 e + (11 - log2 1e6) in [0.5, 16), k = floor t, j = floor(256 f),
# r = f - j / 256; 2^(j/256) ((1 + c1 r) + r^2 (c2 + c3 r)) 2^(k - 11)
@hls function stw_exp(l::Sig, l2e::Cf, l2c::Cf)::Sig
    @pragma "HLS INLINE"
    EX2::Rq[256] = rom(:STW_EX2)
    EXC::Rq[3] = rom(:STW_EXC)
    one::Rq = 1
    zero::Tx = 0
    top::Tx = 15.999999
    t1::Tx = l * l2e
    t::Tx = t1 + l2c
    if t < zero
        t = zero
    end
    if t > top
        t = top
    end
    k::Int = ifloor(t)
    f::Tx = t - k
    f8::Tc = scale2(f, 8)
    j::Int = ifloor(f8)
    r8::Tc = f8 - j
    r::Wt = scale2(r8, -8)
    r2::Rq = r * r
    q1::Rq = EXC[1] * r
    q3::Rq = EXC[3] * r
    lo::Rq = one + q1
    hi::Rq = EXC[2] + q3
    q2::Rq = r2 * hi
    pl::Rq = lo + q2
    mm::Sig = EX2[j + 1] * pl
    p::Sig = stb_sh(mm, k - 11)
    return p
end

# the evaluation: one item per II = 110 on 3 multipliers; o = (p, h, v, s, lg, l, rl, rg), fo = (flag, wet, ...)
@hls function steam_w_eval!(k::SteamWCoefs, VF::Tw[504], VX::Tw[504], VY::Tw[504], VZ::Tw[504], Mi::St, Ui::St, lg0::Sig,
                            l0::Sig, o::Sig[8], fo::Int[2])
    @pragma "HLS INLINE off"
    @pragma "HLS PIPELINE II=110"
    @pragma "HLS ALLOCATION operation instances=mul limit=3"
    LGI::Rq[1024] = rom(:STA_LGI)
    LGL::Lw[1024] = rom(:STA_LGL)
    KL::Cf[13] = rom(:STA_KLN2)
    G::Tw[4, 91] = rom(:STA_GAS)
    SV::Tw[6, 121] = rom(:STW_V)
    SX::Tw[6, 121] = rom(:STW_X)
    SE::Tw[6, 120] = rom(:STW_E)
    SD::Tw[6, 121] = rom(:STW_D)
    @pragma "HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=G type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SV type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SX type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SE type=rom_2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=SD type=rom_2p impl=bram"
    one::Rq = 1
    third::Rq = 0.333333333333333333
    half::Rq = 0.5
    zs::Sig = 0
    # --- the dry state (body A's item for the slot) ---
    mo::Rq[1]
    km::Int = sta_norm(Mi, mo)
    m::Rq = mo[1]
    j::Int = sta_j(m)
    w::Rq = LGI[j + 1]
    e::Rq = m * w
    r::Rs = e - one
    y::Rq = sta_rcp(m, w, e)
    q1::Rq = third - scale2(r, -2)
    q2::Rq = r * q1 - half
    q3::Rq = r * q2 + one
    lgm::Rs = r * q3
    kl::Int = km
    if km == 99
        kl = 0
    end
    X::Sig = lgm + LGL[j + 1] + KL[kl + 10] + k.cx
    uy::Sig = Ui * y
    u::Sig = sta_sh(uy, km)
    vy::Sig = k.vt * y
    vv::Sig = sta_sh(vy, km)
    vs::Sig = scale2(vv, -4)
    tx::Tc = (X - k.gx0) * k.gidx
    ci::Int[1]
    ct::Wt[1]
    okx::Bool = sta_cell(tx, k.xlo, k.xhi, k.xflo, k.xfhi, k.xtlo, k.xthi, k.xcl, k.xch, ci, ct)
    lx::Int = ci[1]
    ix::Int = lx + k.xlo
    wx::Wt[4]
    sta_basis(ct[1], wx)
    ug::Sig = wx[1] * G[1, ix + 1] + wx[2] * G[1, ix + 2] + wx[3] * G[3, ix + 1] + wx[4] * G[3, ix + 2]
    ut::Sig = wx[1] * G[2, ix + 1] + wx[2] * G[2, ix + 2] + wx[3] * G[4, ix + 1] + wx[4] * G[4, ix + 2]
    uge::Sig = ug * k.eps1
    clear::Bool = u >= uge
    dd::Sig = ut - ug
    mo2::Rq[1]
    kd::Int = sta_norm(dd, mo2)
    m2::Rq = mo2[1]
    j2::Int = sta_j(m2)
    w2::Rq = LGI[j2 + 1]
    e2::Rq = m2 * w2
    y2::Rq = sta_rcp(m2, w2, e2)
    du::Sig = u - ug
    xy::Tc = du * y2
    xi::Tc = sta_sh(xy, kd)
    ty::Tc = (xi - k.vy0) * k.vidy
    cj::Int[1]
    cu::Wt[1]
    oky::Bool = sta_cell(ty, k.ylo, k.yhi, k.yflo, k.yfhi, k.ytlo, k.ythi, k.ycl, k.ych, cj, cu)
    ly::Int = cj[1]
    wy::Wt[4]
    sta_basis(cu[1], wy)
    bp::Int = lx * 18 + ly
    pd::Sig = sta_v2(VF, VX, VY, VZ, bp, wx[1], wx[2], wx[3], wx[4], wy[1], wy[2], wy[3], wy[4])
    sd::Sig = sta_v2(VF, VX, VY, VZ, bp + 252, wx[1], wx[2], wx[3], wx[4], wy[1], wy[2], wy[3], wy[4])
    pvd::Sig = pd * vv
    hd::Sig = u + pvd
    okv::Bool = okx && oky && km != 99 && kd != 99
    # --- the dry boundary lg: v_g(lg) = v (row 2 of W's table), two Newton steps from the seed ---
    inr::Bool = vs >= k.vlo && vs <= k.vhi
    lg::Sig = lg0
    nk::Int[1]
    nbad::Bool = false
    cc::Int[1]
    tc::Wt[1]
    wb::Wt[4]
    wd::Wt[3]
    for it in 1:2
        tg::Tc = (lg - k.sx0) * k.sidx
        okg::Bool = stw_cell(tg, k.stlo, k.sthi, cc, tc)
        ig::Int = cc[1]
        sta_basis(tc[1], wb)
        stw_dbasis(tc[1], wd)
        fg::Sig = wb[1] * SV[2, ig + 1] + wb[2] * SV[2, ig + 2] + wb[3] * SX[2, ig + 1] + wb[4] * SX[2, ig + 2] - vs
        dg::Sig = wd[1] * SE[2, ig + 1] + wd[2] * SD[2, ig + 1] + wd[3] * SD[2, ig + 2]
        rg1::Sig = stw_srcp(dg, nk)
        if nk[1] == 99
            nbad = true
        end
        lg = lg - fg * rg1
    end
    tg2::Tc = (lg - k.sx0) * k.sidx
    okg2::Bool = stw_cell(tg2, k.stlo, k.sthi, cc, tc)
    ig2::Int = cc[1]
    sta_basis(tc[1], wb)
    ugs::Sig = wb[1] * SV[4, ig2 + 1] + wb[2] * SV[4, ig2 + 2] + wb[3] * SX[4, ig2 + 1] + wb[4] * SX[4, ig2 + 2]
    rg::Sig = wb[1] * SV[2, ig2 + 1] + wb[2] * SV[2, ig2 + 2] + wb[3] * SX[2, ig2 + 1] + wb[4] * SX[2, ig2 + 2] - vs
    wet::Bool = !clear && inr && !(u >= ugs)
    # --- the wet state l, two Newton steps from the seed (rows 1 v_f, 2 v_g, 3 u_f, 4 u_g) ---
    l::Sig = l0
    sv::Sig[4]
    ev::Sig[4]
    for it in 1:2
        tw::Tc = (l - k.sx0) * k.sidx
        okw::Bool = stw_cell(tw, k.stlo, k.sthi, cc, tc)
        iw::Int = cc[1]
        sta_basis(tc[1], wb)
        stw_dbasis(tc[1], wd)
        for q in 1:4
            sv[q] = wb[1] * SV[q, iw + 1] + wb[2] * SV[q, iw + 2] + wb[3] * SX[q, iw + 1] + wb[4] * SX[q, iw + 2]
            ev[q] = wd[1] * SE[q, iw + 1] + wd[2] * SD[q, iw + 1] + wd[3] * SD[q, iw + 2]
        end
        rdv::Sig = stw_srcp(sv[2] - sv[1], nk)
        if nk[1] == 99
            nbad = true
        end
        xw::Sig = (vs - sv[1]) * rdv
        dxa::Sig = ev[1] + xw * (ev[2] - ev[1])
        dxl::Sig = zs - dxa * rdv
        duw::Sig = sv[4] - sv[3]
        R::Sig = (u - sv[3]) - xw * duw
        dR::Sig = (zs - ev[3] - dxl * duw) - xw * (ev[4] - ev[3])
        rR::Sig = stw_srcp(dR, nk)
        if nk[1] == 99
            nbad = true
        end
        l = l - R * rR
    end
    tf::Tc = (l - k.sx0) * k.sidx
    okl::Bool = stw_cell(tf, k.stlo, k.sthi, cc, tc)
    il::Int = cc[1]
    sta_basis(tc[1], wb)
    for q in 1:4
        sv[q] = wb[1] * SV[q, il + 1] + wb[2] * SV[q, il + 2] + wb[3] * SX[q, il + 1] + wb[4] * SX[q, il + 2]
    end
    rxq::Sig = stw_srcp(sv[2] - sv[1], nk)
    if nk[1] == 99
        nbad = true
    end
    xq::Sig = (vs - sv[1]) * rxq
    rl::Sig = (u - sv[3]) - xq * (sv[4] - sv[3])
    s8::Sig = wb[1] * SV[5, il + 1] + wb[2] * SV[5, il + 2] + wb[3] * SX[5, il + 1] + wb[4] * SX[5, il + 2]
    s9::Sig = wb[1] * SV[6, il + 1] + wb[2] * SV[6, il + 2] + wb[3] * SX[6, il + 1] + wb[4] * SX[6, il + 2]
    sw::Sig = s8 + xq * (s9 - s8)
    pw::Sig = stw_exp(l, k.l2e, k.l2c)
    pvw::Sig = pw * vv
    hw::Sig = u + pvw
    # --- the decision and the flag ---
    arl::Sig = rl
    if rl < zs
        arl = zs - rl
    end
    arg::Sig = rg
    if rg < zs
        arg = zs - rg
    end
    tolr::Sig = scale2(u, -34)
    tolg::Sig = scale2(vs, -30)
    rgok::Bool = arg <= tolg
    xlo::Sig = -0.000000001
    xhi::Sig = 1.000001
    f::Int = 0
    if wet
        if !(xq >= xlo && xq <= xhi && okl && arl <= tolr && rgok) || nbad
            f = 1
        end
    else
        if !okv || (inr && !clear && (!rgok || nbad))
            f = 1
        end
    end
    pp::Sig = pd
    hh::Sig = hd
    ss::Sig = sd
    iwet::Int = 0
    if wet
        pp = pw
        hh = hw
        ss = sw
        iwet = 1
    end
    o[1] = pp
    o[2] = hh
    o[3] = vv
    o[4] = ss
    o[5] = lg
    o[6] = l
    o[7] = rl
    o[8] = rg
    fo[1] = f
    fo[2] = iwet
end

# mode 1: write the word lw at la of the slot's window (banks F, X, Y, Z, 504 words each); mode 0: one evaluation
@hls function steam_w_run!(k::SteamWCoefs, mode::Int, la::Int, lw::Tw, Mi::St, Ui::St, lg0::Sig, l0::Sig, o::Sig[8],
                           fo::Int[2])
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=mode"
    @pragma "HLS INTERFACE mode=s_axilite port=la"
    @pragma "HLS INTERFACE mode=s_axilite port=lw"
    @pragma "HLS INTERFACE mode=s_axilite port=Mi"
    @pragma "HLS INTERFACE mode=s_axilite port=Ui"
    @pragma "HLS INTERFACE mode=s_axilite port=lg0"
    @pragma "HLS INTERFACE mode=s_axilite port=l0"
    @pragma "HLS INTERFACE mode=s_axilite port=o"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    VF::Tw[504] = keep(:STW_VF)
    VX::Tw[504] = keep(:STW_VX)
    VY::Tw[504] = keep(:STW_VY)
    VZ::Tw[504] = keep(:STW_VZ)
    @pragma "HLS BIND_STORAGE variable=VF type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VX type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VY type=ram_t2p impl=bram"
    @pragma "HLS BIND_STORAGE variable=VZ type=ram_t2p impl=bram"
    if mode == 1
        if la < 504
            VF[la + 1] = lw
        elseif la < 1008
            VX[la - 503] = lw
        elseif la < 1512
            VY[la - 1007] = lw
        else
            VZ[la - 1511] = lw
        end
    else
        steam_w_eval!(k, VF, VX, VY, VZ, Mi, Ui, lg0, l0, o, fo)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 3d. BODY C: the evaluation's tail (valve, section sums, resistances, extraction, heat -> f_n) and the step (S1 with
#     compensated sums, the shaft power, the governor's held-decision midpoint) (package _steam_eval, _steam_rhs!;
#     outputs/realtime/steam/stepped.jl run_s1, gov_mid)
# ---------------------------------------------------------------------------------------------------------------------
# Units of f (one scale per instance, Qref = 2^e >= cv): f = (dM / Qref, dU / (Qref 1e6)) per storage, W / (Qref 1e6);
# the S1 coefficients carry h Qref / Mref_k into each state's scale (M~ = M / Mref_k, U~ = U / (Mref_k 1e6)).
#   valve   x1l = clamp(x1, PMIN, PMAX); y = the characteristic at x1l (binary search over its nodes, cubic Hermite;
#           below the first node y_1 d / d_1, above the last y_end); eps = p_1 / p_T, z = (eps - eps_cr) / (1 - eps_cr),
#           beta = 1 (eps <= eps_cr), 0 (z >= 1), w (w^2 + dv^2)^(-1/4) / sqd(1, dv) (w = 1 - z^2) otherwise;
#           q~v = (cv / Qref) y beta, into storage 1 with h_T (q~v >= 0 always: y, beta >= 0)
#   sections (B's q~, e1, e2, fw): dM[a] -= q, dU[a] -= e1; dM[b] += q, dU[b] += e2 (not terminal); the inflows qin;
#           W += fw (e1 - e2)
#   resistances  q~ = Kr~ D (D^2 + dr~^2)^(-1/4), D = p_a - p_b (1e6 Pa), Kr~ = K 1e3 / Qref; e = q (q >= 0 ? h_a : h_b)
#   extraction   qe = beta_k qin_k; dM -= qe, dU -= qe h_k        heat  dU += kh~ (Tw~ - T~_k) (T~ = T / 1e3 from the
#           slot's vapour window with A's weights and cells; kh~ = kh 1e3 / (Qref 1e6))
# Step: x~ += ca_k fa - cb_k fb with a compensated sum (the remainder cc carried), ca, cb = (L + 3/2, L + 1/2) h Qref /
# Mref_k; P = pa Wa - pb Wb (pu); the governor: decisions at (x0, x1, err) (the package's eight sites) select one of 18
# banks: the increment (y - x) = Pb [x0, x1] + Qb err + Rb, exact for the affine rows with those decisions held (BigFloat,
# rounded once); err = (Pref - 1 + R) - R omega (omega the speed at the step's midpoint).
const STEAM_RT_NS = 13                          # storages (path A: 9 used)
const STEAM_RT_NVC = 64                         # valve characteristic nodes (padded with the last; 6 search levels)
const STEAM_RT_NG = 18                          # governor banks: 6 m1 + 2 m2 + nz + 1
const Kc = Float64           # body C: the S1 coefficients (ap_fixed<48, -4>: |c| < 2^-5)
const Ic = Float64           # body C: the S1 increments and remainders (ap_fixed<48, -3>)
for b in (:STC_TF, :STC_TX, :STC_TY, :STC_TZ); hls_keep!(b, zeros(STEAM_RT_WWW)); end      # T windows (2 heated slots)
hls_keep!(:STC_VC, zeros(6, STEAM_RT_NVC))      # per segment: d_i, 1 / h_i, y_i, y_i+1, h_i s_i, h_i s_i+1
hls_keep!(:STC_GB, zeros(8, STEAM_RT_NG))       # per bank: P00 P01 P10 P11 Q0 Q1 R0 R1

@hls mutable struct SteamCCoefs
    ns::Int             # storages
    nsec::Int           # sections (B's items 1..nsec)
    sa::Int[11]         # section inlet, outlet (0-based), terminal
    sb::Int[11]
    st::Int[11]
    nres::Int           # resistances (<= 2): inlet, outlet, Kr~, dr~^2
    ra::Int[2]
    rb::Int[2]
    rk::Cf[2]
    rd2::Dq[2]
    beta::Wt[13]        # extraction fractions
    nh::Int             # heated slots (<= 2): storage, A item, kh~, Tw~
    hk::Int[2]
    hit::Int[2]
    hkh::Cf[2]
    htw::Sig[2]
    cvq::Cf             # valve: cv / Qref, 1e6 / p_T, eps_cr, 1 / (1 - eps_cr), dv^2, 1 / sqd(1, dv), h_T (1e6)
    pti::Cf
    ecr::Cf
    iecr::Cf
    dv2::Dq
    bno::Cf
    hT::Sig
    pmin::Sig           # governor and the characteristic's limits
    pmax::Sig
    vd1::Sig            # first and last characteristic node: d_1, y_1 / d_1, y_end
    vk1::Cf
    vyn::Sig
    vdn::Sig
    uo::Sig
    uc::Sig
    g1::Cf              # GV = g1 x0 + g2 err (1 - T2/T1, K T2/T1); 1 / T3
    g2::Cf
    it3::Cf
    e0::Sig             # err = e0 - R omega
    rr::Cf
    ca::Kc[26]          # S1 per state
    cb::Kc[26]
    pa::Cf              # shaft power (pu): (L + 3/2, L + 1/2) Qref 1e6 / S_b
    pb::Cf
end

# the evaluation's tail: storage outputs (storage order), A's weights and cells, B's items, the servo state -> f (27)
@hls function steam_c_eval!(k::SteamCCoefs, p::Sig[13], hh::Sig[13], wo::Wt[12, 8], co::Int[12, 2], q::Sig[11],
                            e1::Sig[11], e2::Sig[11], fw::Int[11], x1::Sig, f::Sig[27], fl::Int[1])
    TF::Tw[504] = keep(:STC_TF)
    TX::Tw[504] = keep(:STC_TX)
    TY::Tw[504] = keep(:STC_TY)
    TZ::Tw[504] = keep(:STC_TZ)
    VC::Sig[6, 64] = keep(:STC_VC)
    QS::Sd[4096] = rom(:STB_QS)
    zs::Sig = 0
    one::Sig = 1
    dM::Sig[13]
    dU::Sig[13]
    qin::Sig[13]
    for i in 1:13
        dM[i] = zs
        dU[i] = zs
        qin[i] = zs
    end
    flag::Int = 0
    # the valve opening y at x1l (the characteristic: binary search, cubic Hermite)
    x1l::Sig = x1
    if x1 > k.pmax
        x1l = k.pmax
    end
    if x1 < k.pmin
        x1l = k.pmin
    end
    lo::Int = 0
    for s in 1:6
        st::Int = 64
        if s == 1
            st = 32
        elseif s == 2
            st = 16
        elseif s == 3
            st = 8
        elseif s == 4
            st = 4
        elseif s == 5
            st = 2
        else
            st = 1
        end
        if x1l >= VC[1, lo + st + 1]
            lo = lo + st
        end
    end
    ty::Wt = (x1l - VC[1, lo + 1]) * VC[2, lo + 1]
    wv::Wt[4]
    sta_basis(ty, wv)
    yc::Sig = wv[1] * VC[3, lo + 1] + wv[2] * VC[4, lo + 1] + wv[3] * VC[5, lo + 1] + wv[4] * VC[6, lo + 1]
    y::Sig = yc
    if x1l <= k.vd1
        y = k.vk1 * x1l
    end
    if x1l >= k.vdn
        y = k.vyn
    end
    # beta(eps)
    eps::Sig = p[1] * k.pti
    z::Sig = (eps - k.ecr) * k.iecr
    w::Dq = one - z * z
    zz::Dq = w * w + k.dv2
    mz::Rq[1]
    nz::Int = stb_nq(zz, mz)
    jz::Int = stb_jq(mz[1])
    sz::Rq = QS[jz + 1]
    y4::Rq = stb_qrt(mz[1], sz)
    bw::Sig = w * y4
    bn::Sig = bw * k.bno
    beta::Sig = stb_sh(bn, nz)
    if z >= one
        beta = zs
    end
    if eps <= k.ecr
        beta = one
    end
    if nz == 99
        flag = 1
    end
    yb::Sig = y * beta
    qv::Sig = k.cvq * yb
    hv::Sig = hh[1]
    if qv >= zs
        hv = k.hT
    end
    dM[1] = dM[1] + qv
    dU[1] = dU[1] + qv * hv
    if qv > zs
        qin[1] = qin[1] + qv
    end
    # the sections
    W::Sig = zs
    for j in 1:11
        if j <= k.nsec
            a::Int = k.sa[j]
            dM[a + 1] = dM[a + 1] - q[j]
            dU[a + 1] = dU[a + 1] - e1[j]
            if k.st[j] == 0
                b::Int = k.sb[j]
                dM[b + 1] = dM[b + 1] + q[j]
                dU[b + 1] = dU[b + 1] + e2[j]
            end
            if fw[j] == 1
                if k.st[j] == 0
                    qin[k.sb[j] + 1] = qin[k.sb[j] + 1] + q[j]
                end
                W = W + (e1[j] - e2[j])
            else
                qin[a + 1] = qin[a + 1] - q[j]
            end
        end
    end
    # the resistances
    for r in 1:2
        if r <= k.nres
            ia::Int = k.ra[r]
            ib::Int = k.rb[r]
            dr::Dq = p[ia + 1] - p[ib + 1]
            zr::Dq = dr * dr + k.rd2[r]
            mr::Rq[1]
            nr::Int = stb_nq(zr, mr)
            jr::Int = stb_jq(mr[1])
            sr::Rq = QS[jr + 1]
            yr::Rq = stb_qrt(mr[1], sr)
            t1::Sig = k.rk[r] * dr
            t2::Sig = t1 * yr
            qr::Sig = stb_sh(t2, nr)
            if nr == 99
                flag = 1
            end
            hr::Sig = hh[ib + 1]
            if qr >= zs
                hr = hh[ia + 1]
            end
            er::Sig = qr * hr
            dM[ia + 1] = dM[ia + 1] - qr
            dU[ia + 1] = dU[ia + 1] - er
            dM[ib + 1] = dM[ib + 1] + qr
            dU[ib + 1] = dU[ib + 1] + er
            if qr >= zs
                qin[ib + 1] = qin[ib + 1] + qr
            else
                qin[ia + 1] = qin[ia + 1] - qr
            end
        end
    end
    # the extractions and the heat
    for i in 1:13
        if i <= k.ns
            if k.beta[i] > zs
                qe::Sig = k.beta[i] * qin[i]
                dM[i] = dM[i] - qe
                dU[i] = dU[i] - qe * hh[i]
            end
        end
    end
    for m in 1:2
        if m <= k.nh
            it::Int = k.hit[m]
            bt::Int = (m - 1) * 252 + co[it + 1, 1] * 18 + co[it + 1, 2]
            tt::Sig = sta_v2(TF, TX, TY, TZ, bt, wo[it + 1, 1], wo[it + 1, 2], wo[it + 1, 3], wo[it + 1, 4], wo[it + 1, 5],
                             wo[it + 1, 6], wo[it + 1, 7], wo[it + 1, 8])
            hk::Int = k.hk[m]
            dU[hk + 1] = dU[hk + 1] + k.hkh[m] * (k.htw[m] - tt)
        end
    end
    for i in 1:13
        f[2 * i - 1] = dM[i]
        f[2 * i] = dU[i]
    end
    f[27] = W
    fl[1] = flag
end

# the step: storage states (St) with their remainders, f of frames n - L and n - L - 1, the governor (x0, x1) at x_n and
# the midpoint speed -> the next states, remainders, governor, shaft power (pu)
@hls function steam_c_step!(k::SteamCCoefs, x::St[26], cc::Ic[26], fa::Sig[27], fb::Sig[27], gx::Sig[2], om::Sig,
                            xo::St[26], cco::Ic[26], go::Sig[2], pm::Sig[1], bk::Int[1])
    GB::Cf[8, 18] = keep(:STC_GB)
    zs::Sig = 0
    for i in 1:26
        if i <= 2 * k.ns
            ia::Ic = k.ca[i] * fa[i]
            ib::Ic = k.cb[i] * fb[i]
            e::Ic = ia - ib + cc[i]
            xn::St = x[i] + e
            d::Ic = x[i] - xn
            cco[i] = d + e
            xo[i] = xn
        end
    end
    pa::Sig = k.pa * fa[27]
    pb::Sig = k.pb * fb[27]
    pm[1] = pa - pb
    # the governor: decisions at x_n (clamp x1; clamp srv; non-windup; out-of-band relax tied to the x1 clamp)
    x0::Sig = gx[1]
    x1::Sig = gx[2]
    rw::Sig = k.rr * om
    err::Sig = k.e0 - rw
    m1::Int = 0
    x1l::Sig = x1
    if x1 > k.pmax
        m1 = 1
        x1l = k.pmax
    end
    if x1 < k.pmin
        m1 = 2
        x1l = k.pmin
    end
    gv1::Sig = k.g1 * x0
    gv2::Sig = k.g2 * err
    dg::Sig = gv1 + gv2 - x1l
    srv::Sig = dg * k.it3
    m2::Int = 0
    if srv > k.uo
        m2 = 1
        srv = k.uo
    end
    if srv < k.uc
        m2 = 2
        srv = k.uc
    end
    nz::Int = 0
    if x1l >= k.pmax && srv > zs
        nz = 1
    end
    if x1l <= k.pmin && srv < zs
        nz = 1
    end
    b::Int = 6 * m1 + 2 * m2 + nz
    p00::Sig = GB[1, b + 1] * x0
    p01::Sig = GB[2, b + 1] * x1
    q0::Sig = GB[5, b + 1] * err
    p10::Sig = GB[3, b + 1] * x0
    p11::Sig = GB[4, b + 1] * x1
    q1::Sig = GB[6, b + 1] * err
    go[1] = x0 + (p00 + p01 + q0 + GB[7, b + 1])
    go[2] = x1 + (p10 + p11 + q1 + GB[8, b + 1])
    bk[1] = b
end

# ---------------------------------------------------------------------------------------------------------------------
# 4. Data: the instances, their slot scales, windows and coefficients (on the PC)
# ---------------------------------------------------------------------------------------------------------------------
steam_rt_file() = joinpath(STEAM_RT_DIR, "data", "ieee39_steam.toml")
steam_rt_windows() = TOML.parsefile(joinpath(STEAM_RT_DIR, "data", "steam_windows.toml"))

"""    steam_rt_unit(d, D): the PHSteamUnit of instance table d (D the whole file: eps_cr)."""
function steam_rt_unit(d, D)
    st = d["storage"]; tb = steam_rt_tables()
    _STPKG.PHSteamUnit(d["name"], Symbol(d["path"]), d["K"], d["T1"], d["T2"], d["T3"], d["UO"], d["UC"], d["PMAX"], d["PMIN"],
        d["R_base"], d["n"], [s["name"] for s in st], [s["V"] for s in st], [s["beta"] for s in st],
        [s["kh"] for s in st], [s["Tw"] for s in st], d["cv"], d["p_T"], d["h_T"], D["eps_cr"], d["delta_v"],
        Int.(d["sec_up"]), Int.(d["sec_down"]), Float64.(d["sec_K"]), Float64.(d["sec_eta"]), Float64.(d["sec_delta"]),
        String.(d["sec_names"]), Int.(d["res_up"]), Int.(d["res_down"]), Float64.(d["res_K"]), Float64.(d["res_delta"]),
        d["p_cond"], [s["node"] for s in st], (;), tb, :characteristic, Float64.(d["vc_d"]), Float64.(d["vc_y"]),
        Float64.(d["vc_s"]), Dict{String,Any}())
end

"""    steam_rt_aslots(g, x0): the dry slots of body A (12 items; path A's 8 repeated to fill) and their scale exponents."""
function steam_rt_aslots(g, x0)
    ks = [k for k in eachindex(g.V) if g.names[k] != STEAM_RT_WET[g.path]]
    length(ks) <= STEAM_RT_NA || error("$(g.name): $(length(ks)) dry slots > $STEAM_RT_NA")
    items = vcat(ks, fill(ks[end], STEAM_RT_NA - length(ks)))
    e = [ceil(Int, log2(x0[2 + 2k - 1])) for k in items]                  # Mref_k = 2^e_k: M0 / Mref in (1/2, 1]
    return (; ks, items, e)
end

"""    steam_rt_acoefs(g, x0): SteamACoefs and the window words (4 banks) of instance g (body A)."""
function steam_rt_acoefs(g, x0; win = steam_rt_windows())
    tb = steam_rt_tables(); sl = steam_rt_aslots(g, x0); I_ = STEAM_RT_I; q(v, T) = _q48(v, getfield(I_, T))
    W = win[g.name]["storage"]
    nX, nY = tb.vap.nx, tb.vap.ny
    f(name) = zeros(STEAM_RT_NA)
    vt = f(:vt); cx = f(:cx); xlo = zeros(Int, 12); xhi = zeros(Int, 12); ylo = zeros(Int, 12); yhi = zeros(Int, 12)
    xflo, xfhi, xtlo, xthi, xcl, xch = f(1), f(2), f(3), f(4), f(5), f(6)
    yflo, yfhi, ytlo, ythi, ycl, ych = f(1), f(2), f(3), f(4), f(5), f(6)
    banks = [zeros(STEAM_RT_WW) for _ in 1:4]
    for (it, kk) in enumerate(sl.items)
        Mref = 2.0^sl.e[it]
        vt[it] = q(g.V[kk] / Mref, :Cf); cx[it] = q(log(BigFloat(Mref)) - log(BigFloat(g.V[kk])), :Cf)
        w = W[g.names[kk]]
        for (lo_, n, wd, below, LO, HI, FLO, FHI, TLO, THI, CL, CH) in
                ((w[1], nX, w[2], 1e-9, xlo, xhi, xflo, xfhi, xtlo, xthi, xcl, xch),
                 (w[3], nY, w[4], _STPKG._STEAM_BELOW_EDGE, ylo, yhi, yflo, yfhi, ytlo, ythi, ycl, ych))
            hi_ = min(lo_ + wd - 1, n - 2)
            blo = lo_ == 0 ? below : 0.0; bhi = hi_ == n - 2 ? 1e-9 : 0.0
            LO[it] = lo_; HI[it] = hi_; FLO[it] = lo_; FHI[it] = hi_ + 1
            TLO[it] = q(lo_ - blo, :Cf); THI[it] = q(hi_ + 1 + bhi, :Cf); CL[it] = q(-blo, :Wt); CH[it] = q(1 + bhi, :Wt)
        end
        V = tb.vap
        for (oo, out, S) in ((0, 2, STEAM_RT_SC.p), (1, 3, STEAM_RT_SC.s)), a in 0:STEAM_RT_WIN.nx-1, b in 0:STEAM_RT_WIN.ny-1
            i, jj = xlo[it] + a + 1, ylo[it] + b + 1                       # 1-based table nodes
            (i <= nX && jj <= nY) || continue                              # beyond the table: never read
            ad = (it - 1) * 504 + oo * 252 + a * 18 + b + 1
            banks[1][ad] = q(V.F[out, i, jj] * S, :Tw)
            banks[2][ad] = q(V.dx * V.Fx[out, i, jj] * S, :Tw)
            banks[3][ad] = q(V.dy * V.Fy[out, i, jj] * S, :Tw)
            banks[4][ad] = q(V.dx * V.dy * V.Fxy[out, i, jj] * S, :Tw)
        end
    end
    k = SteamACoefs(vt, cx, xlo, xhi, xflo, xfhi, xtlo, xthi, xcl, xch, ylo, yhi, yflo, yfhi, ytlo, ythi, ycl, ych,
                    q(tb.vap.x0, :Cf), q(1 / tb.vap.dx, :Cf), q(tb.vap.y0, :Cf), q(1 / tb.vap.dy, :Cf), q(1 + 1e-6, :Cf))
    return (; k, banks, sl)
end

"""    steam_rt_aload!(banks): writes the window words through the load mode (as the board program will)."""
function steam_rt_aload!(k, banks)
    z = zeros(12); zi = zeros(Int, 12); wo = zeros(12, 8); co = zeros(Int, 12, 2)
    for (b, w) in enumerate(banks), a in eachindex(w)
        steam_a_run!(k, 1, (b - 1) * STEAM_RT_WW + a - 1, w[a], z, z, z, z, z, z, zi, wo, co)
    end
end

"""    steam_rt_astates(g, x0, sl): the scaled item inputs (M~, U~) of the slots."""
steam_rt_astates(x, sl) = ([_q48(x[2 + 2k - 1] / 2.0^e, STEAM_RT_I.St) for (k, e) in zip(sl.items, sl.e)],
                           [_q48(x[2 + 2k] / (2.0^e * 1e6), STEAM_RT_I.St) for (k, e) in zip(sl.items, sl.e)])

"""
    steam_rt_acheck(; ndraw = 200, rel = 0.01, lim = 1e-10)

Body A in Julia (Float64, the @hls code with its units) against the package's steam_state for every instance: at x0 and
`ndraw` states with M, U of every slot scaled by 1 + rel (2 rand - 1). For each item: p, h, s, v relative differences
(limit `lim`, fixed before the run), flags; an unflagged item whose package state is wet, or outside its window, fails.
"""
function steam_rt_acheck(; ndraw = 200, rel = 0.01, lim = 1e-10, seed = 45)
    D = TOML.parsefile(steam_rt_file()); tb = steam_rt_tables(); S = STEAM_RT_SC
    rng = _STPKG.Random.MersenneTwister(seed)
    worst = 0.0; nfl = 0; nok = 0; bad = 0
    for d in D["unit"]
        g = steam_rt_unit(d, D); x0 = Float64.(d["x"])
        c = steam_rt_acoefs(g, x0); steam_rt_aload!(c.k, c.banks)
        wu = 0.0; fu = 0
        for n in 0:ndraw
            x = copy(x0)
            n > 0 && for k in eachindex(g.V); x[2+2k-1] *= 1 + rel * (2rand(rng) - 1); x[2+2k] *= 1 + rel * (2rand(rng) - 1); end
            Mi, Ui = steam_rt_astates(x, c.sl)
            p = zeros(12); h = zeros(12); v = zeros(12); s = zeros(12); fl = zeros(Int, 12); wo = zeros(12, 8); co = zeros(Int, 12, 2)
            steam_a_run!(c.k, 0, 0, 0.0, Mi, Ui, p, h, v, s, fl, wo, co)
            for (it, kk) in enumerate(c.sl.ks)
                M = Mi[it] * 2.0^c.sl.e[it]; U = Ui[it] * 2.0^c.sl.e[it] * 1e6       # the quantised state itself
                st = try _STPKG.steam_state(tb, M / g.V[kk], U / M) catch; nothing end
                if fl[it] == 1
                    nfl += 1; fu += 1; continue
                end
                (st === nothing || st.x != 1) && (bad += 1; println("  $(g.name) slot $(g.names[kk]): unflagged, package ", st === nothing ? "outside the tables" : "wet"); continue)
                e = max(abs(p[it] / S.p - st.p) / abs(st.p), abs(h[it] / S.u - st.h) / abs(st.h),
                        abs(s[it] / S.s - st.s) / abs(st.s), abs(v[it] - g.V[kk] / M) / abs(g.V[kk] / M))
                wu = max(wu, e); nok += 1
            end
        end
        worst = max(worst, wu)
        println("  $(g.name) (path $(g.path), $(length(c.sl.ks)) dry slots): max relative difference $(round(wu; sigdigits = 3)), flagged items $fu")
    end
    ok = worst <= lim && bad == 0
    println("body A vs package: $nok items compared, max relative difference $(round(worst; sigdigits = 3)) (limit $lim), ",
            "flagged $nfl, unflagged failures $bad: ", ok ? "PASS" : "FAIL")
    return ok
end

# ---------------------------------------------------------------------------------------------------------------------
# 4b. Data of body B: the sections' coefficients and isentrope windows; the check against the package's formulas
# ---------------------------------------------------------------------------------------------------------------------
"""    steam_rt_bscale(g, x0): per section the pressure exponent ep_j (2^ep_j >= p_in(x0) in MPa) and the flow scale Qref."""
function steam_rt_bscale(g, x0)
    e = _STPKG._steam_eval(g, x0[3:end], _STPKG._steam_stroke(g, x0[2]))
    ep = [ceil(Int, log2(e.st[g.sec_up[j]].p / 1e6)) for j in eachindex(g.sec_K)]
    return (; ep, Qref = 2.0^ceil(Int, log2(g.cv)))
end

"""    steam_rt_bcoefs(g, x0): SteamBCoefs and the isentrope window words (4 banks) of instance g (body B)."""
function steam_rt_bcoefs(g, x0; win = steam_rt_windows())
    tb = steam_rt_tables(); sc = steam_rt_bscale(g, x0); q(v, T) = _q48(v, getfield(STEAM_RT_I, T))
    ns = length(g.sec_K); ns <= STEAM_RT_NB || error("$(g.name): $ns sections > $STEAM_RT_NB")
    items = vcat(1:ns, fill(ns, STEAM_RT_NB - ns))
    W = win[g.name]["section"]; E = tb.ise
    N = STEAM_RT_NB; zi() = zeros(Int, N); zf() = zeros(N)
    ia, ib, tm, wo, ep = zi(), zi(), zi(), zi(), zi(); pm, kq, d2, et = zf(), zf(), zf(), zf()
    L = (zi(), zi(), zf(), zf(), zf(), zf(), zf(), zf()); Ee = (zi(), zi(), zf(), zf(), zf(), zf(), zf(), zf())
    banks = [zeros(STEAM_RT_EW) for _ in 1:4]
    for (it, j) in enumerate(items)
        a, b = g.sec_up[j], g.sec_down[j]
        ia[it] = a - 1; ib[it] = b == 0 ? a - 1 : b - 1; tm[it] = b == 0; wo[it] = g.sec_names[j] in WET_ENDS_RT[g.path]
        ep[it] = sc.ep[j]; S = BigFloat(10)^6 * BigFloat(2)^sc.ep[j]; pm[it] = q(1.375 * 2.0^sc.ep[j], :Sig)
        kq[it] = q(g.sec_K[j] * BigFloat(1000) * BigFloat(2)^sc.ep[j] / sc.Qref, :Cf)
        d2[it] = q((g.sec_delta[j] / S^2)^2, :Dq); et[it] = q(g.sec_eta[j], :Wt)
        w = W[g.sec_names[j]]
        for (lo_, n, wd, C) in ((w[1], E.nx, w[2], L), (w[3], E.ny, w[4], Ee))
            hi_ = min(lo_ + wd - 1, n - 2)
            blo = lo_ == 0 ? 1e-9 : 0.0; bhi = hi_ == n - 2 ? 1e-9 : 0.0
            C[1][it] = lo_; C[2][it] = hi_; C[3][it] = lo_; C[4][it] = hi_ + 1
            C[5][it] = q(lo_ - blo, :Cf); C[6][it] = q(hi_ + 1 + bhi, :Cf); C[7][it] = q(-blo, :Wt); C[8][it] = q(1 + bhi, :Wt)
        end
        for aa in 0:STEAM_RT_EWIN.nl-1, bb in 0:STEAM_RT_EWIN.ne-1
            i, jj = L[1][it] + aa + 1, Ee[1][it] + bb + 1
            (i <= E.nx && jj <= E.ny) || continue
            ad = (it - 1) * 247 + aa * 13 + bb + 1
            banks[1][ad] = q(E.F[1, i, jj] * 1e-6, :Tw)
            banks[2][ad] = q(E.dx * E.Fx[1, i, jj] * 1e-6, :Tw)
            banks[3][ad] = q(E.dy * E.Fy[1, i, jj] * 1e-6, :Tw)
            banks[4][ad] = q(E.dx * E.dy * E.Fxy[1, i, jj] * 1e-6, :Tw)
        end
    end
    k = SteamBCoefs(ia, ib, tm, wo, ep, pm, kq, d2, et, L..., Ee..., q(g.p_cond * 1e-6, :Sig), q(log(BigFloat(g.p_cond)), :Cf),
                    q(E.x0, :Cf), q(1 / E.dx, :Cf), q(E.y0, :Cf), q(1 / E.dy, :Cf))
    return (; k, banks, items, ns, sc)
end
const WET_ENDS_RT = Dict(:A => ("LP1", "LP2"), :C => ("LP3", "LP4"))   # the sections whose end may be wet (count_1.log)

"""    steam_rt_bload!(k, banks): writes the windows (mode 1) and the coefficients (mode 2), as the board program will."""
function steam_rt_bload!(k, banks)
    z = zeros(13); o = zeros(11); oi = zeros(Int, 11)
    for (b, w) in enumerate(banks), a in eachindex(w)
        steam_b_run!(k, 1, (b - 1) * STEAM_RT_EW + a - 1, w[a], z, z, z, z, o, o, o, oi, oi)
    end
    steam_b_run!(k, 2, 0, 0.0, z, z, z, z, o, o, o, oi, oi)
end

# the package's section (_steam_eval) at given storage outputs (Pa, J/kg, m3/kg, J/(kg K)): q, e1, e2 as body B defines
# them; nothing when the package would stop (an isentropic end outside its tables, below saturated liquid)
function steam_rt_bref(g, j, P, H, V, S, tb)
    a, b = g.sec_up[j], g.sec_down[j]
    pin = P[a]; pout = b == 0 ? g.p_cond : P[b]
    D = pin^2 - pout^2
    sq = _STPKG._sqd(D, g.sec_delta[j])
    if D >= 0 || b == 0
        q = b == 0 && D < 0 ? 0.0 : g.sec_K[j] * sq / sqrt(pin * V[a])
    else
        q = g.sec_K[j] * sq / sqrt(pout * V[b])
    end
    hb = b == 0 ? H[a] : H[b]
    D >= 0 || return (; q, e1 = q * hb, e2 = q * hb, wet = false)
    hs = try _STPKG.steam_h_isentropic(tb, pout, S[a]) catch; return nothing end
    hout = H[a] - g.sec_eta[j] * (H[a] - hs.h)
    return (; q, e1 = q * H[a], e2 = q * hout, wet = hs.x < 1)
end

"""
    steam_rt_bcheck(; ndraw = 200, rel = 0.01, lim = 1e-10)

Body B in Julia (Float64, the @hls code with its units) against the package's section formulas at the same storage
outputs (the package's steam_state of every slot, quantised to the words B reads): at x0, `ndraw` states with M, U of
every slot scaled by 1 + rel (2 rand - 1), and per non-terminal section a reversed state (outlet pressure 1.02 x inlet)
and per terminal section an inlet below p_cond. q, e1, e2 relative differences (limit `lim`, fixed before the run);
flags: an unflagged item the package rejects fails; a flagged item it accepts is counted.
"""
function steam_rt_bcheck(; ndraw = 200, rel = 0.01, lim = 1e-10, seed = 46)
    D = TOML.parsefile(steam_rt_file()); tb = steam_rt_tables()
    rng = _STPKG.Random.MersenneTwister(seed)
    worst = 0.0; nok = 0; nfl = 0; bad = 0; nrev = 0; nwet = 0; nfa = 0; nfl0 = 0
    q(v, T) = _q48(v, getfield(STEAM_RT_I, T))
    for d in D["unit"]
        g = steam_rt_unit(d, D); x0 = Float64.(d["x"]); ns = length(g.V)
        c = steam_rt_bcoefs(g, x0); steam_rt_bload!(c.k, c.banks)
        cases = Any[(0, 0)]; append!(cases, [(n, 0) for n in 1:ndraw])
        append!(cases, [(-1, j) for j in 1:c.ns])
        wu = 0.0; fu = 0
        for (n, jr) in cases
            x = copy(x0)
            n > 0 && for k in 1:ns; x[2+2k-1] *= 1 + rel * (2rand(rng) - 1); x[2+2k] *= 1 + rel * (2rand(rng) - 1); end
            st = [_STPKG.steam_state(tb, x[2+2k-1] / g.V[k], x[2+2k] / x[2+2k-1]) for k in 1:ns]
            p = zeros(13); h = zeros(13); v = zeros(13); s = zeros(13)
            for k in 1:ns
                p[k] = q(st[k].p * 1e-6, :Sig); h[k] = q(st[k].h * 1e-6, :Sig); v[k] = q(g.V[k] / x[2+2k-1], :Sig)
                s[k] = q(st[k].s * 1e-3, :Sig)
            end
            if jr > 0                               # a reversed section
                a, b = g.sec_up[jr], g.sec_down[jr]
                b == 0 ? (p[a] = q(0.9 * g.p_cond * 1e-6, :Sig)) : (p[b] = q(1.02 * p[a], :Sig))
            end
            P = p[1:ns] .* 1e6; H = h[1:ns] .* 1e6; V = v[1:ns]; S = s[1:ns] .* 1e3     # the words themselves
            qo = zeros(11); e1 = zeros(11); e2 = zeros(11); fw = zeros(Int, 11); fl = zeros(Int, 11)
            steam_b_run!(c.k, 0, 0, 0.0, p, h, v, s, qo, e1, e2, fw, fl)
            for j in 1:c.ns
                r = steam_rt_bref(g, j, P, H, V, S, tb)
                jr > 0 && j == jr && (nrev += 1)
                if fl[j] == 1
                    nfl += 1; fu += 1; r === nothing || (nfa += 1); n >= 0 && (nfl0 += 1)
                    n >= 0 && println("  $(g.name) $(g.sec_names[j]) case $n: flagged (package ", r === nothing ? "rejects" :
                                      r.wet ? "wet end" : "dry end", "; fwd $(fw[j]), p_in $(P[g.sec_up[j]]), s_in $(S[g.sec_up[j]]))")
                    continue
                end
                if r === nothing || (r.wet && !(g.sec_names[j] in WET_ENDS_RT[g.path]))
                    bad += 1; println("  $(g.name) $(g.sec_names[j]) case $n/$jr: unflagged, the package rejects it"); continue
                end
                r.wet && (nwet += 1)
                Q = c.sc.Qref; E = Q * 1e6
                err = max(abs(qo[j] * Q - r.q) / max(abs(r.q), 1e-300), abs(e1[j] * E - r.e1) / max(abs(r.e1), 1e-300),
                          abs(e2[j] * E - r.e2) / max(abs(r.e2), 1e-300))
                r.q == 0 && (err = max(abs(qo[j]), abs(e1[j]), abs(e2[j])))
                err > lim && println("  $(g.name) $(g.sec_names[j]) case $n/$jr: relative difference $(round(err; sigdigits = 3)) ",
                                     "(q $(r.q), B $(qo[j] * Q))")
                wu = max(wu, err); nok += 1
            end
        end
        worst = max(worst, wu)
        println("  $(g.name) (path $(g.path), $(c.ns) sections, Qref $(c.sc.Qref)): max relative difference ",
                "$(round(wu; sigdigits = 3)), flagged items $fu")
    end
    ok = worst <= lim && bad == 0
    println("body B vs package: $nok items compared ($nrev reversed, $nwet wet ends), max relative difference ",
            "$(round(worst; sigdigits = 3)) (limit $lim), flagged $nfl (in the x0 and perturbed states $nfl0; the package accepts $nfa), ",
            "unflagged failures $bad: ", ok ? "PASS" : "FAIL")
    return ok
end

# ---------------------------------------------------------------------------------------------------------------------
# 4c. Data of body W: the wet slot's scale, window and coefficients; the check against the package's steam_state
# ---------------------------------------------------------------------------------------------------------------------
"""    steam_rt_wcoefs(g, x0): SteamWCoefs, the window words (4 banks of 504) and the slot (k, Mref exponent e)."""
function steam_rt_wcoefs(g, x0; win = steam_rt_windows())
    tb = steam_rt_tables(); q(v, T) = _q48(v, getfield(STEAM_RT_I, T))
    kk = findfirst(==(STEAM_RT_WET[g.path]), g.names)
    e = ceil(Int, log2(x0[2 + 2kk - 1])); Mref = 2.0^e
    w = win[g.name]["storage"][g.names[kk]]; nX, nY = tb.vap.nx, tb.vap.ny
    C = Any[]
    for (lo_, n, wd, below) in ((w[1], nX, w[2], 1e-9), (w[3], nY, w[4], _STPKG._STEAM_BELOW_EDGE))
        hi_ = min(lo_ + wd - 1, n - 2)
        blo = lo_ == 0 ? below : 0.0; bhi = hi_ == n - 2 ? 1e-9 : 0.0
        append!(C, Any[lo_, hi_, q(lo_, :Cf), q(hi_ + 1, :Cf), q(lo_ - blo, :Cf), q(hi_ + 1 + bhi, :Cf), q(-blo, :Wt), q(1 + bhi, :Wt)])
    end
    banks = [zeros(STEAM_RT_WWW) for _ in 1:4]; V = tb.vap
    for (oo, out, S) in ((0, 2, STEAM_RT_SC.p), (1, 3, STEAM_RT_SC.s)), a in 0:STEAM_RT_WIN.nx-1, b in 0:STEAM_RT_WIN.ny-1
        i, jj = C[1] + a + 1, C[9] + b + 1
        (i <= nX && jj <= nY) || continue
        ad = oo * 252 + a * 18 + b + 1
        banks[1][ad] = q(V.F[out, i, jj] * S, :Tw); banks[2][ad] = q(V.dx * V.Fx[out, i, jj] * S, :Tw)
        banks[3][ad] = q(V.dy * V.Fy[out, i, jj] * S, :Tw); banks[4][ad] = q(V.dx * V.dy * V.Fxy[out, i, jj] * S, :Tw)
    end
    s = tb.sat
    k = SteamWCoefs(q(g.V[kk] / Mref, :Cf), q(log(BigFloat(Mref)) - log(BigFloat(g.V[kk])), :Cf), C...,
                    q(tb.vap.x0, :Cf), q(1 / tb.vap.dx, :Cf), q(tb.vap.y0, :Cf), q(1 / tb.vap.dy, :Cf), q(1 + 1e-6, :Cf),
                    q(s.x0, :Cf), q(1 / s.dx, :Cf), q(-1e-9, :Cf), q(s.n - 1 + 1e-9, :Cf),
                    q(VHI / 16, :Sig), q(VLO / 16, :Sig), q(1 / log(BigFloat(2)), :Cf), q(11 - log2(BigFloat(10)^6), :Cf))
    return (; k, banks, kk, e)
end
const VLO, VHI = (tb = steam_rt_tables(); (_STPKG._interp(tb.sat, 3, tb.sat.x0, ""),
                                            _STPKG._interp(tb.sat, 3, tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx, "")))

"""    steam_rt_wload!(k, banks): writes the window words through mode 1."""
function steam_rt_wload!(k, banks)
    o = zeros(8); fo = zeros(Int, 2)
    for (b, w) in enumerate(banks), a in eachindex(w)
        steam_w_run!(k, 1, (b - 1) * STEAM_RT_WWW + a - 1, w[a], 0.0, 0.0, 0.0, 0.0, o, fo)
    end
end

# the seeds (lg, l) of the package at state (M, U) of the slot (as frame.jl's pkg_seeds)
function steam_rt_wseeds(tb, V, M, U)
    lg = _STPKG._sat_l_of_vg(tb, V / M)
    lg = lg === nothing ? tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx / 2 : lg
    st = _STPKG.steam_state(tb, M / V, U / M)
    return (lg, st.x != 1.0 ? log(st.p) : lg)
end

"""
    steam_rt_wcheck(; ndraw = 400, rel = 0.01, drift = 1e-4, lim = 1e-10)

Body W in Julia (Float64, the @hls code with its units) against the package's steam_state for the wet slot of every
instance: at x0 and `ndraw` states with M, U scaled by 1 + rel (2 rand - 1); the seeds are the package's (lg, l) at a
state further moved by `drift` (2 rand - 1) relative (the frames between the seeds' evaluation and this one). For each:
p, h, s, v relative differences (limit `lim`, fixed before the run), the wet decision against the package's (x < 1),
flags; an unflagged item whose decision differs, or which the package rejects, fails.
"""
function steam_rt_wcheck(; ndraw = 400, rel = 0.01, drift = 1e-4, lim = 1e-10, seed = 47)
    D = TOML.parsefile(steam_rt_file()); tb = steam_rt_tables(); S = STEAM_RT_SC
    rng = _STPKG.Random.MersenneTwister(seed)
    worst = 0.0; nok = 0; nfl = 0; bad = 0; nwet = 0
    for d in D["unit"]
        g = steam_rt_unit(d, D); x0 = Float64.(d["x"])
        c = steam_rt_wcoefs(g, x0); steam_rt_wload!(c.k, c.banks)
        kk = c.kk; Mref = 2.0^c.e; wu = 0.0; fu = 0; wn = 0
        for n in 0:ndraw
            M = x0[2+2kk-1]; U = x0[2+2kk]
            n > 0 && (M *= 1 + rel * (2rand(rng) - 1); U *= 1 + rel * (2rand(rng) - 1))
            Mi = _q48(M / Mref, STEAM_RT_I.St); Ui = _q48(U / (Mref * 1e6), STEAM_RT_I.St)
            M = Mi * Mref; U = Ui * Mref * 1e6                      # the quantised state itself
            sd = steam_rt_wseeds(tb, g.V[kk], M * (1 + drift * (2rand(rng) - 1)), U * (1 + drift * (2rand(rng) - 1)))
            o = zeros(8); fo = zeros(Int, 2)
            steam_w_run!(c.k, 0, 0, 0.0, Mi, Ui, sd[1], sd[2], o, fo)
            st = try _STPKG.steam_state(tb, M / g.V[kk], U / M) catch; nothing end
            if fo[1] == 1
                nfl += 1; fu += 1; continue
            end
            if st === nothing || (st.x < 1) != (fo[2] == 1)
                bad += 1
                println("  $(g.name) case $n: unflagged, ", st === nothing ? "the package rejects it" :
                        "decision wet $(fo[2]) vs package x = $(st.x)"); continue
            end
            fo[2] == 1 && (wn += 1)
            e = max(abs(o[1] / S.p - st.p) / st.p, abs(o[2] / S.u - st.h) / abs(st.h), abs(o[4] / S.s - st.s) / abs(st.s),
                    abs(o[3] - g.V[kk] / M) / (g.V[kk] / M))
            e > lim && println("  $(g.name) case $n: relative difference $(round(e; sigdigits = 3)) (wet $(fo[2]), x $(st.x))")
            wu = max(wu, e); nok += 1
        end
        worst = max(worst, wu); nwet += wn
        println("  $(g.name) (slot $(g.names[kk])): max relative difference $(round(wu; sigdigits = 3)), wet $wn of ",
                "$(ndraw + 1), flagged $fu")
    end
    ok = worst <= lim && bad == 0
    println("body W vs package: $nok states compared ($nwet wet), max relative difference $(round(worst; sigdigits = 3)) ",
            "(limit $lim), flagged $nfl, unflagged failures $bad: ", ok ? "PASS" : "FAIL")
    return ok
end

# ---------------------------------------------------------------------------------------------------------------------
# 4d. Data of body C; the whole evaluation (A, W, B, C in Julia) against the package; the governor banks
# ---------------------------------------------------------------------------------------------------------------------
# the governor's increment for the decisions (m1, m2, nz), exact for the held affine rows: (y - x) = P x + Q err + R
function steam_rt_gbank(g, h::BigFloat, m1, m2, nz)
    B(v) = BigFloat(v)
    T1, T2, T3, K = B(g.T1), B(g.T2), B(g.T3), B(g.K)
    g1, g2 = 1 - T2 / T1, K * T2 / T1
    A = zeros(BigFloat, 2, 2); Bv = zeros(BigFloat, 2); c = zeros(BigFloat, 2)
    A[1, 1] = -1 / T1; Bv[1] = K / T1
    lim = m1 == 1 ? B(g.PMAX) : m1 == 2 ? B(g.PMIN) : B(0)
    if nz == 0
        if m2 == 0
            A[2, 1] = g1 / T3; A[2, 2] = m1 == 0 ? -1 / T3 : B(0); Bv[2] = g2 / T3; c[2] = -lim / T3
        else
            c[2] = m2 == 1 ? B(g.UO) : B(g.UC)
        end
    end
    m1 != 0 && (A[2, 2] -= 1 / T3; c[2] += lim / T3)           # out-of-band relax toward the limit
    M = inv(Matrix{BigFloat}(I, 2, 2) - h / 2 * A)
    return (; P = M * h * A, Q = M * h * Bv, R = M * h * c)
end

"""    steam_rt_ccoefs(g, x0; L, h): SteamCCoefs, the valve table, the governor banks and the T windows of instance g."""
function steam_rt_ccoefs(g, x0; L = 7, h = 110 * 8e-9, win = steam_rt_windows())
    tb = steam_rt_tables(); q(v, T) = _q48(v, getfield(STEAM_RT_I, T)); qk(v) = _q48(v, STEAM_RT_I.Kc)
    sa = steam_rt_aslots(g, x0); sc = steam_rt_bscale(g, x0); cw = steam_rt_wcoefs(g, x0; win)
    ns = length(g.V); nsec = length(g.sec_K); Q = sc.Qref
    Mref = zeros(ns); for (it, k) in enumerate(sa.ks); Mref[k] = 2.0^sa.e[it]; end; Mref[cw.kk] = 2.0^cw.e
    pad(v, n, x) = vcat(v, fill(x, n - length(v)))
    sA = pad(g.sec_up .- 1, 11, 0); sB = pad([b == 0 ? 0 : b - 1 for b in g.sec_down], 11, 0); sT = pad(Int.(g.sec_down .== 0), 11, 1)
    nres = length(g.res_K); nres <= 2 || error("$(g.name): $nres resistances")
    heated = findall(>(0), g.kh); length(heated) <= 2 || error("$(g.name): $(length(heated)) heated slots")
    hit = [findfirst(==(k), sa.ks) - 1 for k in heated]
    # the T windows of the heated slots (vapour map output 1, in 1e3 K), the slot's own window
    banks = [zeros(STEAM_RT_WWW) for _ in 1:4]; V = tb.vap
    for (m, k) in enumerate(heated)
        w = win[g.name]["storage"][g.names[k]]
        for a in 0:STEAM_RT_WIN.nx-1, b in 0:STEAM_RT_WIN.ny-1
            i, jj = w[1] + a + 1, w[3] + b + 1
            (i <= V.nx && jj <= V.ny) || continue
            ad = (m - 1) * 252 + a * 18 + b + 1
            banks[1][ad] = q(V.F[1, i, jj] * 1e-3, :Tw); banks[2][ad] = q(V.dx * V.Fx[1, i, jj] * 1e-3, :Tw)
            banks[3][ad] = q(V.dy * V.Fy[1, i, jj] * 1e-3, :Tw); banks[4][ad] = q(V.dx * V.dy * V.Fxy[1, i, jj] * 1e-3, :Tw)
        end
    end
    # the valve characteristic per segment (padded: d = 1e3 never reached)
    dd, yy, ss = g.vc_d, g.vc_y, g.vc_s; n = length(dd); n <= STEAM_RT_NVC || error("$(g.name): $n characteristic nodes")
    VC = zeros(6, STEAM_RT_NVC)
    for i in 1:STEAM_RT_NVC
        j = min(i, n - 1); hj = dd[j+1] - dd[j]
        VC[:, i] = [i <= n - 1 ? q(dd[i], :Sig) : 1e3, q(1 / hj, :Sig), q(yy[j], :Sig), q(yy[j+1], :Sig), q(hj * ss[j], :Sig),
                    q(hj * ss[j+1], :Sig)]
    end
    VC[1, n] = q(dd[n], :Sig)                  # the last node itself (x1l >= d_n: y_end)
    # the governor banks (BigFloat, rounded once)
    hb = BigFloat(h); GB = zeros(8, STEAM_RT_NG)
    for m1 in 0:2, m2 in 0:2, nz in 0:1
        bk = steam_rt_gbank(g, hb, m1, m2, nz)
        GB[:, 6m1+2m2+nz+1] = [q(bk.P[1, 1], :Cf), q(bk.P[1, 2], :Cf), q(bk.P[2, 1], :Cf), q(bk.P[2, 2], :Cf), q(bk.Q[1], :Cf),
                               q(bk.Q[2], :Cf), q(bk.R[1], :Cf), q(bk.R[2], :Cf)]
    end
    ca = zeros(26); cb = zeros(26)
    for k in 1:ns, r in 0:1
        ca[2k-1+r] = qk((L + 1.5) * h * Q / Mref[k]); cb[2k-1+r] = qk((L + 0.5) * h * Q / Mref[k])
    end
    sqd1 = 1 / _STPKG._sqd(1.0, g.delta_v)
    k = SteamCCoefs(ns, nsec, sA, sB, sT, nres, pad(g.res_up .- 1, 2, 0), pad(g.res_down .- 1, 2, 0),
                    pad([q(K * 1e3 / Q, :Cf) for K in g.res_K], 2, 0.0), pad([q((d / 1e6)^2, :Dq) for d in g.res_delta], 2, 0.0),
                    pad([q(b, :Wt) for b in g.beta], 13, 0.0), length(heated), pad(heated .- 1, 2, 0), pad(hit, 2, 0),
                    pad([q(g.kh[k] * 1e3 / (Q * 1e6), :Cf) for k in heated], 2, 0.0), pad([q(g.Tw[k] * 1e-3, :Sig) for k in heated], 2, 0.0),
                    q(g.cv / Q, :Cf), q(1e6 / g.p_T, :Cf), q(g.eps_cr, :Cf), q(1 / (1 - g.eps_cr), :Cf), q(g.delta_v^2, :Dq),
                    q(sqd1, :Cf), q(g.h_T * 1e-6, :Sig), q(g.PMIN, :Sig), q(g.PMAX, :Sig), q(dd[1], :Sig), q(yy[1] / dd[1], :Cf),
                    q(yy[end], :Sig), q(dd[end], :Sig), q(g.UO, :Sig), q(g.UC, :Sig), q(1 - g.T2 / g.T1, :Cf),
                    q(g.K * g.T2 / g.T1, :Cf), q(1 / g.T3, :Cf), q(0.0, :Sig), q(g.R_base, :Cf), ca, cb,
                    q((L + 1.5) * Q * 1e6 / _STPKG.STEAM_SB, :Cf), q((L + 0.5) * Q * 1e6 / _STPKG.STEAM_SB, :Cf))
    return (; k, banks, VC, GB, Q, Mref, sa, cw)
end
"""    steam_rt_cload!(c, Pref): the T windows, the valve table and the banks into the kept tables; err's constant."""
function steam_rt_cload!(c, g, Pref)
    for (b, nm) in zip(c.banks, (:STC_TF, :STC_TX, :STC_TY, :STC_TZ)); HLS_KEEPS[nm] .= b; end
    HLS_KEEPS[:STC_VC] .= c.VC; HLS_KEEPS[:STC_GB] .= c.GB
    c.k.e0 = _q48(Pref - 1 + g.R_base, STEAM_RT_I.Sig)
    return c
end

"""    steam_rt_feval(g, x, c, ca, cb, cw): the whole evaluation in Julia (A, W, B, C's tail) at the unit state x -> f, flags."""
function steam_rt_feval(g, x, c, ca, cb, cw)
    tb = steam_rt_tables(); ns = length(g.V)
    Mi, Ui = steam_rt_astates(x, ca.sl)
    pa = zeros(12); ha = zeros(12); va = zeros(12); sa_ = zeros(12); fa = zeros(Int, 12); wo = zeros(12, 8); co = zeros(Int, 12, 2)
    steam_a_run!(ca.k, 0, 0, 0.0, Mi, Ui, pa, ha, va, sa_, fa, wo, co)
    kk = cw.kk; Mref = 2.0^cw.e
    Mw = _q48(x[2+2kk-1] / Mref, STEAM_RT_I.St); Uw = _q48(x[2+2kk] / (Mref * 1e6), STEAM_RT_I.St)
    sd = steam_rt_wseeds(tb, g.V[kk], Mw * Mref, Uw * Mref * 1e6)
    o = zeros(8); fo = zeros(Int, 2)
    steam_w_run!(cw.k, 0, 0, 0.0, Mw, Uw, sd[1], sd[2], o, fo)
    p = zeros(13); h = zeros(13); v = zeros(13); s = zeros(13)
    for (it, k) in enumerate(ca.sl.ks); p[k] = pa[it]; h[k] = ha[it]; v[k] = va[it]; s[k] = sa_[it]; end
    p[kk] = o[1]; h[kk] = o[2]; v[kk] = o[3]; s[kk] = o[4]
    q = zeros(11); e1 = zeros(11); e2 = zeros(11); fw = zeros(Int, 11); fb = zeros(Int, 11)
    steam_b_run!(cb.k, 0, 0, 0.0, p, h, v, s, q, e1, e2, fw, fb)
    f = zeros(27); fl = zeros(Int, 1)
    steam_c_eval!(c.k, p, h, wo, co, q, e1, e2, fw, x[2], f, fl)
    flag = any(==(1), fa[1:length(ca.sl.ks)]) || fo[1] == 1 || any(==(1), fb[1:cb.ns]) || fl[1] == 1
    return f, flag
end

"""
    steam_rt_ccheck(; ndraw = 100, rel = 0.01, lim = 1e-10)

The whole evaluation in Julia (bodies A, W, B and C's tail, the @hls code) against the package's _steam_eval at x0 and
`ndraw` states with every storage's M, U scaled by 1 + rel (2 rand - 1) and the servo state by +-5 %: per storage
|dM - dM_pkg| / throughflow and |dU - dU_pkg| / (throughflow |h|), and |W - W_pkg| / |W_pkg| (limit `lim`, fixed
before the run); flagged states counted. Then the governor: 2,000 random (x0, x1, omega) over every bank against the
package's rows by the held-decision midpoint (stepped.jl gov_mid, 3 iterations; limit 1e-12 relative to PMAX).
"""
function steam_rt_ccheck(; ndraw = 100, rel = 0.01, lim = 1e-10, seed = 48)
    D = TOML.parsefile(steam_rt_file()); tb = steam_rt_tables()
    rng = _STPKG.Random.MersenneTwister(seed)
    worst = 0.0; nfl = 0; nok = 0
    for d in D["unit"]
        g = steam_rt_unit(d, D); x0 = Float64.(d["x"]); ns = length(g.V)
        ca = steam_rt_acoefs(g, x0); steam_rt_aload!(ca.k, ca.banks)
        cb = steam_rt_bcoefs(g, x0); steam_rt_bload!(cb.k, cb.banks)
        cw = steam_rt_wcoefs(g, x0); steam_rt_wload!(cw.k, cw.banks)
        c = steam_rt_cload!(steam_rt_ccoefs(g, x0), g, d["Pref"])
        wu = 0.0; fu = 0
        for n in 0:ndraw
            x = copy(x0)
            if n > 0
                for k in 1:ns; x[2+2k-1] *= 1 + rel * (2rand(rng) - 1); x[2+2k] *= 1 + rel * (2rand(rng) - 1); end
                x[2] *= 1 + 0.05 * (2rand(rng) - 1)
            end
            # the package at the quantised state the bodies read
            xq = copy(x)
            for (it, k) in enumerate(c.sa.ks)
                xq[2+2k-1] = _q48(x[2+2k-1] / c.Mref[k], STEAM_RT_I.St) * c.Mref[k]
                xq[2+2k] = _q48(x[2+2k] / (c.Mref[k] * 1e6), STEAM_RT_I.St) * c.Mref[k] * 1e6
            end
            kw = c.cw.kk
            xq[2+2kw-1] = _q48(x[2+2kw-1] / c.Mref[kw], STEAM_RT_I.St) * c.Mref[kw]
            xq[2+2kw] = _q48(x[2+2kw] / (c.Mref[kw] * 1e6), STEAM_RT_I.St) * c.Mref[kw] * 1e6
            xq[2] = _q48(x[2], STEAM_RT_I.Sig)
            f, flag = steam_rt_feval(g, xq, c, ca, cb, cw)
            if flag
                nfl += 1; fu += 1; continue
            end
            e = _STPKG._steam_eval(g, xq[3:end], _STPKG._steam_stroke(g, xq[2]))
            thr = zeros(ns); thr[1] += abs(e.qv)
            for j in eachindex(g.sec_K); thr[g.sec_up[j]] += abs(e.qsec[j]); g.sec_down[j] > 0 && (thr[g.sec_down[j]] += abs(e.qsec[j])); end
            for r in eachindex(g.res_K); thr[g.res_up[r]] += abs(e.qres[r]); thr[g.res_down[r]] += abs(e.qres[r]); end
            er = 0.0
            for k in 1:ns
                er = max(er, abs(f[2k-1] * c.Q - e.dM[k]) / thr[k], abs(f[2k] * c.Q * 1e6 - e.dU[k]) / (thr[k] * abs(e.st[k].h)))
            end
            er = max(er, abs(f[27] * c.Q * 1e6 - e.W) / abs(e.W))
            er > lim && println("  $(g.name) case $n: difference $(round(er; sigdigits = 3))")
            wu = max(wu, er); nok += 1
        end
        worst = max(worst, wu)
        println("  $(g.name): f against the package, max $(round(wu; sigdigits = 3)) (of throughflow / W), flagged $fu")
    end
    okf = worst <= lim
    println("whole evaluation (A, W, B, C) vs package: $nok states, max $(round(worst; sigdigits = 3)) (limit $lim), flagged $nfl: ",
            okf ? "PASS" : "FAIL")
    # the governor banks against the package rows (held-decision midpoint)
    wg = 0.0; seen = Set{Int}()
    for d in D["unit"]
        g = steam_rt_unit(d, D); x0 = Float64.(d["x"]); c = steam_rt_cload!(steam_rt_ccoefs(g, x0), g, d["Pref"]); h = 110 * 8e-9
        for _ in 1:250
            gx = [x0[1] * (1 + 0.3 * (2rand(rng) - 1)), g.PMAX * (1.3 * rand(rng) - 0.15)]
            om = 1 + 0.1 * (2rand(rng) - 1)
            go = zeros(2); pm = zeros(1); bk = zeros(Int, 1)
            steam_c_step!(c.k, zeros(26), zeros(26), zeros(27), zeros(27), gx, om, zeros(26), zeros(26), go, pm, bk)
            push!(seen, bk[1])
            lgr = _STPKG.ModeLog(); rows(z0, z1, rec) = begin
                err = d["Pref"] - 1.0 - g.R_base * (om - 1.0)
                d0 = (g.K * err - z0) / g.T1; GV = z0 * (1 - g.T2 / g.T1) + g.K * (g.T2 / g.T1) * err
                x1l = _STPKG.clamp_mode(rec, z1, g.PMIN, g.PMAX); srv = (GV - x1l) / g.T3
                srv = _STPKG.clamp_mode(rec, srv, g.UC, g.UO); srv = _STPKG.nonwindup(rec, x1l, srv, g.PMIN, g.PMAX)
                srv = _STPKG.outband_relax(rec, srv, z1, g.PMIN, g.PMAX, g.T3); (d0, srv) end
            rows(gx[1], gx[2], lgr)
            a0, a1 = rows(gx[1], gx[2], _STPKG.ModeReplay(lgr.decisions, 0)); y0, y1 = gx[1] + h * a0, gx[2] + h * a1
            for _ in 1:3
                a0, a1 = rows((gx[1] + y0) / 2, (gx[2] + y1) / 2, _STPKG.ModeReplay(lgr.decisions, 0)); y0, y1 = gx[1] + h * a0, gx[2] + h * a1
            end
            wg = max(wg, abs(go[1] - y0) / g.PMAX, abs(go[2] - y1) / g.PMAX)
        end
    end
    okg = wg <= 1e-12
    println("governor banks vs held-decision midpoint: 2,000 states, banks used $(sort(collect(seen))), max $(round(wg; sigdigits = 3)) ",
            "of PMAX (limit 1e-12): ", okg ? "PASS" : "FAIL")
    return okf && okg
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. Export (body A alone: the C++, the ROMs, the coefficients of one instance; HLS synthesis by steam_a_synth.sh)
# ---------------------------------------------------------------------------------------------------------------------
const STEAM_RT_CLOCK_NS = 8
const STEAM_RT_IIA = 9

function steam_rt_aexport(; dir = joinpath(STEAM_RT_DIR, "CPP", "steam_a"), inst = "IEEEG1_3")
    occursin("II=$(STEAM_RT_IIA)\"", string(HLS_DEFS[:steam_a_run!])) || error("the PIPELINE II in steam_a_run! differs from $STEAM_RT_IIA")
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "steam_a_rom.h"), [:STA_LGI, :STA_LGL, :STA_KLN2, :STA_GAS])
    hls_write(joinpath(dir, "steam_a.cpp"), [:SteamACoefs, :sta_norm, :sta_sh, :sta_j, :sta_rcp, :sta_cell, :sta_basis,
              :sta_v2, :steam_a_run!]; types = STEAM_RT_TYPES, local_includes = ["rt_util.h", "steam_a_rom.h"],
              header = "Body A of the steam unit (12 dry storages at II $(STEAM_RT_IIA)); src/rt/governors/steam.jl")
    D = TOML.parsefile(steam_rt_file()); d = only(filter(u -> u["name"] == inst, D["unit"]))
    c = steam_rt_acoefs(steam_rt_unit(d, D), Float64.(d["x"]))
    hls_write_const(joinpath(dir, "steam_a_coefs.h"), "STEAM_A_COEFS", c.k)
    return dir
end

const STEAM_RT_IIB = 10

function steam_rt_bexport(; dir = joinpath(STEAM_RT_DIR, "CPP", "steam_b"), inst = "IEEEG1_3")
    occursin("II=$(STEAM_RT_IIB)\"", string(HLS_DEFS[:steam_b_run!])) || error("the PIPELINE II in steam_b_run! differs from $STEAM_RT_IIB")
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "steam_b_rom.h"), [:STA_LGI, :STA_LGL, :STB_KLN, :STB_QS, :STB_RS, :STB_SV, :STB_SD])
    hls_write(joinpath(dir, "steam_b.cpp"), [:SteamBCoefs, :sta_j, :sta_rcp, :sta_cell, :sta_basis, :stb_norm, :stb_nq,
              :stb_nr, :stb_jq, :stb_jr, :stb_qrt, :stb_rsq, :stb_ps, :stb_sh, :stb_v2, :steam_b_run!];
              types = STEAM_RT_TYPES, local_includes = ["rt_util.h", "steam_b_rom.h"],
              header = "Body B of the steam unit (11 sections at II $(STEAM_RT_IIB)); src/rt/governors/steam.jl")
    D = TOML.parsefile(steam_rt_file()); d = only(filter(u -> u["name"] == inst, D["unit"]))
    c = steam_rt_bcoefs(steam_rt_unit(d, D), Float64.(d["x"]))
    hls_write_const(joinpath(dir, "steam_b_coefs.h"), "STEAM_B_COEFS", c.k)
    return dir
end

function steam_rt_wexport(; dir = joinpath(STEAM_RT_DIR, "CPP", "steam_w"), inst = "IEEEG1_3")
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "steam_w_rom.h"), [:STA_LGI, :STA_LGL, :STA_KLN2, :STA_GAS, :STW_V, :STW_X, :STW_E, :STW_D,
                                                    :STW_EX2, :STW_EXC])
    hls_write(joinpath(dir, "steam_w.cpp"), [:SteamWCoefs, :sta_norm, :sta_sh, :sta_j, :sta_rcp, :sta_cell, :sta_basis,
              :sta_v2, :stb_norm, :stb_sh, :stw_cell, :stw_dbasis, :stw_srcp, :stw_exp, :steam_w_eval!, :steam_w_run!];
              types = STEAM_RT_TYPES, local_includes = ["rt_util.h", "steam_w_rom.h"],
              header = "Body W of the steam unit (the wet-capable slot, II 110 on 3 multipliers); src/rt/governors/steam.jl")
    D = TOML.parsefile(steam_rt_file()); d = only(filter(u -> u["name"] == inst, D["unit"]))
    c = steam_rt_wcoefs(steam_rt_unit(d, D), Float64.(d["x"]))
    hls_write_const(joinpath(dir, "steam_w_coefs.h"), "STEAM_W_COEFS", c.k)
    return dir
end

# ---------------------------------------------------------------------------------------------------------------------
# 6. The shared engine (review 46): the primitive census of the bodies (finding 1), from the @hls source itself
# ---------------------------------------------------------------------------------------------------------------------
for f in (:sta_norm, :stb_norm, :stb_nq, :stb_nr); HG_INTRINSICS[f] = :norm; end
for f in (:sta_sh, :stb_sh, :stb_ps); HG_INTRINSICS[f] = :vshift; end
for f in (:sta_j, :stb_jq, :stb_jr); HG_INTRINSICS[f] = :sidx; end
for f in (:sta_cell, :stw_cell); HG_INTRINSICS[f] = :cell; end
for n in (STEAM_RT_BINT..., STEAM_RT_BNUM...); push!(HG_CONSTKEEPS, Symbol(:STB_, n)); end

_hgin(g, name, n, fmt = :St) = Any[gnode!(g, :in, Int[], fmt, (name, i)) for i in 1:n]
_hgout(dims...) = Array{Any}(nothing, dims...)

"""    steam_rt_graph(d, D): the flat operation graph of one instance's frame: A, W, B, C (eval and step) linked (tags
:A, :W, :B, :C); the states, remainders, f history, seeds and the speed are inputs tagged :state (the state memory)."""
function steam_rt_graph(d, D)
    g = steam_rt_unit(d, D); x0 = Float64.(d["x"]); G = HGraph()
    ca = steam_rt_acoefs(g, x0); cb = steam_rt_bcoefs(g, x0); steam_rt_bload!(cb.k, cb.banks); cw = steam_rt_wcoefs(g, x0)
    c = steam_rt_cload!(steam_rt_ccoefs(g, x0), g, d["Pref"])
    st(name, n, fmt) = (G.curtag = :state; v = _hgin(G, name, n, fmt); v)
    X = st(:x, 26, :St); CC = st(:cc, 26, :Ic); FA = st(:fa, 27, :Sig); FB = st(:fb, 27, :Sig); GX = st(:gx, 2, :Sig)
    OM = st(:om, 1, :Sig)[1]; SD = st(:seed, 2, :Sig)
    ns = length(g.V); kw = cw.kk
    G.curtag = :A
    Ma = Any[X[2k-1] for k in ca.sl.items]; Ua = Any[X[2k] for k in ca.sl.items]
    pa, ha, va, sa_, fa = _hgout(12), _hgout(12), _hgout(12), _hgout(12), _hgout(12); wo, co = _hgout(12, 8), _hgout(12, 2)
    hg_call(G, :steam_a_run!, Any[ca.k, 0, 0, 0.0, Ma, Ua, pa, ha, va, sa_, fa, wo, co])
    G.curtag = :W
    o, fo = _hgout(8), _hgout(2)
    hg_call(G, :steam_w_run!, Any[cw.k, 0, 0, 0.0, X[2kw-1], X[2kw], SD[1], SD[2], o, fo])
    P = Any[0.0 for _ in 1:13]; H = copy(P); V = copy(P); S = copy(P)
    for (it, k) in enumerate(ca.sl.ks); P[k] = pa[it]; H[k] = ha[it]; V[k] = va[it]; S[k] = sa_[it]; end
    P[kw] = o[1]; H[kw] = o[2]; V[kw] = o[3]; S[kw] = o[4]
    G.curtag = :B
    q, e1, e2, fw, fb = _hgout(11), _hgout(11), _hgout(11), _hgout(11), _hgout(11)
    hg_call(G, :steam_b_run!, Any[cb.k, 0, 0, 0.0, P, H, V, S, q, e1, e2, fw, fb])
    G.curtag = :C
    f, fl = _hgout(27), _hgout(1)
    hg_call(G, :steam_c_eval!, Any[c.k, P, H, wo, co, q, e1, e2, fw, GX[2], f, fl])
    xo, cco, go, pm, bk = _hgout(26), _hgout(26), _hgout(2), _hgout(1), _hgout(1)
    G.curtag = :S
    hg_call(G, :steam_c_step!, Any[c.k, X, CC, FA, FB, GX, OM, xo, cco, go, pm, bk])
    outs = (; f, xo = xo[1:2ns], cco = cco[1:2ns], go, pm, seeds = (o[5], o[6]),
            flags = vcat(fa[1:length(ca.sl.ks)], fo[1:1], fb[1:cb.ns], fl))
    return G, outs
end

"""    steam_rt_census(): the primitive counts per body and instance (review 46, finding 1)."""
function steam_rt_census()
    D = TOML.parsefile(steam_rt_file())
    cls = [:mul, :add, :sub, :neg, :cvt, :shiftc, :cmp, :sel, :pand, :por, :pnot, :psel, :floor, :iadd, :isub, :imul, :ineg,
           :icmp, :isel, :norm, :vshift, :sidx, :cell, :mem, :const, :in]
    for d in D["unit"]
        G, outs = steam_rt_graph(d, D)
        nst = count(==(:state), G.tag); nw = length(outs.f) + length(outs.xo) + length(outs.cco) + 2 + 1 + 2
        println(d["name"], " (path ", d["path"], "): state memory reads $nst, writes $nw (f, x, remainders, governor, P, seeds)")
        for t in (:A, :W, :B, :C)
            idx = findall(==(t), G.tag)
            c = Dict{Symbol,Int}(); m = Dict{Symbol,Int}(); inner = 0
            for i in idx
                o = G.op[i]; o === :proj && continue
                c[o] = get(c, o, 0) + 1
                o === :mem && (m[G.info[i][1]] = get(m, G.info[i][1], 0) + 1)
                G.fmt[i] === :exact && o in (:mul, :add, :sub, :neg, :shiftc, :vshift) && (inner += 1)
            end
            println("  $t: ", join(["$o $(c[o])" for o in cls if haskey(c, o)], ", "), "; intermediate (exact) results $inner")
            println("     reads: ", join(["$k $v" for (k, v) in sort(collect(m), by = first)], ", "))
        end
    end
end

"""
    steam_rt_engine(; P = 110, kw...)

The scheduling experiment (review 46b, 5): every instance's frame graph scheduled on the machine (EngineMachine(; P,
kw...)); per instance the frame times (f by tf, x_n+1 by tx, seeds by ts), L, d, whether x_n+1 is ready for the next
frame, the widest MAC accumulator; unit occupancy.
"""
function steam_rt_engine(; P = 110, insts = nothing, kw...)
    D = TOML.parsefile(steam_rt_file()); m = EngineMachine(; P, kw...)
    I = Dict{Symbol,Int}(pairs(STEAM_RT_I)); I[:Sd] = 2; I[:int] = 32; I[:bool] = 1
    res = Any[]
    for d in D["unit"]
        insts === nothing || d["name"] in insts || continue
        G, outs = steam_rt_graph(d, D)
        ops, opof = es_ops(G, outs; I)
        sf = Set(k for (k, op) in enumerate(ops) if (op.node > 0 && G.tag[op.node] === :S) ||
                                                    (endswith(String(op.cls), "W") && startswith(String(op.cls), "ST_") &&
                                                     G.tag[op.reads[1]] === :S))
        s = es_schedule(G, outs, m; I, stepfirst = sf)
        r = es_frame_report(G, outs, s, m)
        cnt = Dict{Symbol,Int}(); for op in s.ops; c = startswith(String(op.cls), "MEM_") ? :MEM : startswith(String(op.cls), "ST_") ? :STATE : op.cls
            cnt[c] = get(cnt, c, 0) + (op.cls === :MAC ? op.nterms : 1); end
        println("$(d["name"]) P $P: f by $(r.tf), x_n+1 by $(r.tx) (", r.xok ? "ready" : "NOT ready", "), seeds by $(r.ts); ",
                "L $(r.L), d $(r.d); widest accumulator $(r.wacc) bits; slots ",
                join(["$c $(cnt[c])" for c in sort(collect(keys(cnt)))], ", "))
        push!(res, (; name = d["name"], r, s, G, outs))
    end
    return res
end

"""
    steam_rt_connect(; P = 104, MAC = 22, Fs = (4, 6, 8, 64), kw...)

The connectivity experiment (46b, 5): schedule every instance at P, then for each fan-in F learn one shared
producer -> file adjacency over all instances (capped at F), freeze it and bind every instance again. Per F: operand
deliveries, writes, reuses, failures (deliveries needing a move), lifetimes > P, peak file occupancy, files, fan-in.
"""
function steam_rt_connect(; P = 104, MAC = 22, Fs = (4, 6, 8, 64), hold = 1, kw...)
    m = EngineMachine(; P, MAC, hold, kw...)
    res = steam_rt_engine(; P, MAC, hold, kw...)
    # lifetimes in the files (write to last read) of the full-connectivity binding: their distribution against P
    for r in res[1:min(2, end)]
        b = es_bind(r.s, m; F = 1000, adj = Dict{Tuple,Set{Tuple}}(), learn = true)
        println("  $(r.name) unlimited fan-in: failures $(b.fails) of $(b.deliveries), lifetimes > P $(b.long), peak $(b.peak)")
    end
    for F in Fs
        adj = Dict{Tuple,Set{Tuple}}()
        for r in res; es_bind(r.s, m; F, adj, learn = true); end
        tot = [es_bind(r.s, m; F, adj = deepcopy(adj), learn = false) for r in res]
        println("F $F (hold $hold): files $(tot[1].nfiles), max fan-in $(maximum(t.fanin for t in tot)); per instance deliveries ",
                "$(maximum(t.deliveries for t in tot)), failures $(join([t.fails for t in tot], " ")), lifetimes > P ",
                "$(maximum(t.long for t in tot)), peak occupancy $(maximum(t.peak for t in tot))")
    end
    return res
end
