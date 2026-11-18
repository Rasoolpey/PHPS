# =====================================================================================================================
# The grid-forming VSM converter's controller call (PHVsm, sampled; REALTIME_TODO item 3; reviews 48, 48b)
# =====================================================================================================================
# Source: src/phsim/converter.jl, `vsm_sample!` (lines 256-269) with `vsm_divider` (149-158), `vsm_rhs!` (169-177),
# `_held_from` / `_radial_clip` (273-276); the DLL gfm_vsm_pf (v1.4, VI-only) as ported there; the record
# docs/active/PH_CONVERTER_MODEL.md sections 2, 10, 12-14. The reference form: the DLL called every Ts = 50 us, its output
# from the states as they are, then explicit Euler; the output (the modulation, with the physical DC side) held in
# alpha-beta until the next call.
#
# One call (this file's map), in alpha-beta (the package works in the simulator frame turning at omega_b; every operation
# is a rotation, a product Re(u conj(i)) or a magnitude, so only the angle changes: phi = (theta + omega_b t) / 2 pi):
#   E = u_mag e^{j 2 pi phi};  D = E - v_k;  the predictive virtual impedance: i |Z(i)| = |D|, Z(i) = r_vi + kpr ov +
#   j (Zs + x_vi + kpx ov), ov = max(0, i - i_lim);  ip = D / Z(i);  u_c = E - (Z(i) - j Zs) ip
#   u_c clipped radially to u_max (if set); m = u_c / v_dc clipped radially to m_max (with the DC side), else held u_c
#   p = Re(v_k conj(i_c));  omega += Ts/Ta (p_set - p) - Ts Dp/Ta (omega_f - 1);  omega_f += Ts omega_c (omega - omega_f)
#   u_mag += Ts/Tv (u_set - u_mag);  phi += Ts f_n omega  (mod 1)            (all from the old states)
#
# Hardware form (48 merge, items 3-7):
#   - phi in turns, ap_fixed<48, 2>, wrapped explicitly into [0, 1); cos / sin from a 1024-entry ROM at the bin centres
#     and the remainder's rotation (|delta| <= pi / 1024; sin to delta^5, cos to delta^4: truncation <= 1.1e-18);
#   - engagement by squares: i0 = |D| / |Z0| > i_lim <=> |D|^2 > (i_lim |Z0|)^2 (exact; D = 0 is off: ip = 0, u_c = E);
#   - engaged: the root ov* of F(ov) = |(i_lim + ov) Z(ov)|^2 - |D|^2 (convex, increasing). Start ov0 from the lower
#     bound |Z| >= c0' + c1 ov (c1 = |kpr + j kpx|, c0' = Re(Z0 conj(kp)) / c1), which makes ov0 >= ov*: the root of
#     (i_lim + ov)(c0' + c1 ov) = |D|. Then NCHORD chord steps ov <- ov - g F(ov), g ~ 1 / F'(ov0) (seed only, 2^-11):
#     the contraction is certified per instance (`vsm_certify`), the slope's error changes only the rate;
#   - ip from the root's identity |Z| = |D| / i: ip = (D i / |D|) conj(Z) (i / |D|) (no reciprocal of |Z|^2);
#   - the clips by squared comparisons (exact decisions); one rsqrt of |u_c|^2 when a clip acts; 1 / v_dc by a seeded
#     Newton reciprocal; v_dc outside [v_min, v_max] is flagged (and clamped).
# =====================================================================================================================

const VSM_DIR = @__DIR__
const Turn = Float64         # the angle in turns (ap_fixed<48, 2>)
const Zc = Float64           # impedance-sized coefficients (ap_fixed<48, 0>: |c| < 1/2)
const Rc = Float64           # the cos / sin ROM words (ap_fixed<48, 2>)
const Tq = Float64           # the phase increment's second word (ap_fixed<48, -54>)
const Pq = Float64           # the set-points p_set, u_set (ap_fixed<48, 5>)
const Ka = Float64           # Ts / Ta (ap_fixed<48, -16>)
const Sd = Float64           # the rsqrt seeds (ap_ufixed<18, 2>: 16 fraction bits)

const VSM_I = (Sig = 8, St = 3, Turn = 2, Small = -7, Ang = -6, Coef = 10, Zc = 0, Rq = 4, Rc = 2, Tq = -54, Pq = 5, Ka = -16)
const VSM_NCHORD = 9         # chord steps (certified: vsm_certify)
const VSM_VDC = (0.5, 1.99)  # the DC voltage's domain for the call (flagged outside)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------------------------------------------------
"""    vsm_data(i) -> Dict: converter i of src/rt/converters/data/ieee39_vsm.toml (scripts/realtime/vsm_extract.jl)."""
vsm_data(i::Int; path = joinpath(VSM_DIR, "data", "ieee39_vsm.toml")) = TOML.parsefile(path)["converter"][i]
vsm_ninstances(; path = joinpath(VSM_DIR, "data", "ieee39_vsm.toml")) = length(TOML.parsefile(path)["converter"])

# ---------------------------------------------------------------------------------------------------------------------
# 2. Coefficients (BigFloat, rounded once to their types)
# ---------------------------------------------------------------------------------------------------------------------
@hls struct VsmCoefs
    kz::Zc[10]       # r_vi, b = Zs + x_vi, Zs, x_vi, kpr, kpx, c1, A2 = c1^2, ab = r_vi kpr + b kpx, unused
    kg::Sig[12]      # i_lim, thr2 = (i_lim |Z0|)^2, y0 = 1 / Z0 (re, im), p_set, u_set, u_max, umax2, m_max, mmax2, v_min, v_max
    kc::Coef[5]      # beta = c0' + c1 i_lim, beta2, 4 c1, gam0 = c0' i_lim, inv2c1 = 1 / (2 c1)
    ks::Small[5]     # Ts / Ta, Ts Dp / Ta, Ts omega_c, Ts / Tv, Ts f_n  (f_n = omega_n / 2 pi)
    kt::Tq[1]        # Ts f_n - its Small rounding (the phase increment's second word)
    kx::Pq[2]        # p_set, u_set (finer than Sig: omega and u_mag settle on them; 49, 3)
    ka::Ka[1]        # Ts / Ta (~3.6e-6: its own fine grid; its rounding otherwise drifts the phase, 49, 3)
    kf::Sig[3]       # flags as words (1 or 0): u_max set, the DC side (held m), the VI grows (c1 > 0)
end

function vsm_coefs_big(d)
    b(k) = BigFloat(d[k])
    r, x, Zs, kpr, kpx = b("r_vi"), b("x_vi"), b("Zs"), b("kpr"), b("kpx")
    bb = Zs + x
    c1 = sqrt(kpr^2 + kpx^2)
    Z0 = complex(r, bb)
    c0p = c1 > 0 ? (r * kpr + bb * kpx) / c1 : zero(c1)
    il = b("i_lim")
    beta = c0p + c1 * il
    fn = b("omega_n") / (2 * BigFloat(pi))
    Ts = b("Ts")
    umax = d["u_max"] > 0 ? b("u_max") : BigFloat(8)
    y0 = 1 / Z0
    return (kz = [r, bb, Zs, x, kpr, kpx, c1, c1^2, r * kpr + bb * kpx, zero(r)],
            kg = [il, (il * abs(Z0))^2, real(y0), imag(y0), b("p_set"), b("u_set"), umax, umax^2, b("m_max"), b("m_max")^2,
                  BigFloat(VSM_VDC[1]), BigFloat(VSM_VDC[2])],
            kc = [beta, beta^2, 4c1, c0p * il, c1 > 0 ? 1 / (2c1) : zero(c1)],
            ks = [Ts / b("Ta"), Ts * b("Dp") / b("Ta"), Ts * b("omega_c"), Ts / b("Tv"), Ts * fn],
            kt = [Ts * fn - BigFloat(_q48(Ts * fn, VSM_I.Small))], kx = [b("p_set"), b("u_set")], ka = [Ts / b("Ta")],
            kf = [d["u_max"] > 0 ? 1 : 0, d["dc_mode"] == "held" ? 1 : 0, c1 > 0 ? 1 : 0])
end

"""    vsm_coefs(d; check = true) -> VsmCoefs, each entry rounded to its type's grid (range-checked)."""
function vsm_coefs(d; check = true)
    c = vsm_coefs_big(d)
    q(v, T) = begin
        I_ = getfield(VSM_I, T)
        m = Float64(maximum(abs, v))
        (!check || m < 2.0^(I_ - 1)) || error("$(d["name"]): a coefficient (max |value| $m) does not fit $T = ap_fixed<48, $I_>")
        [_q48(w, I_) for w in v]
    end
    d["dc_mode"] in ("held",) || error("$(d["name"]): DC mode $(d["dc_mode"]) is not the held modulation built here")
    Float64(c.kg[2]) >= 2.0^-8 || error("$(d["name"]): the engagement threshold |D|^2 = $(Float64(c.kg[2])) is below the rsqrt range")
    return VsmCoefs(q(c.kz, :Zc), q(c.kg, :Sig), q(c.kc, :Coef), q(c.ks, :Small), q(c.kt, :Tq), q(c.kx, :Pq), q(c.ka, :Ka),
                    Float64.(c.kf))
end

# ROMs: cos / sin at the bin centres (j + 1/2) / 1024 turns; rsqrt seeds (m in [1/4, 1): 2 octaves x 512 bins, centre
# value 2 / (sqrt lo + sqrt hi)); reciprocal seeds (m in [1/2, 1): 1024 bins, 2 / (lo + hi))
hls_rom!(:VS_COS, [_q48(cospi(2 * (BigFloat(j) + BigFloat(1) / 2) / 1024), VSM_I.Rc) for j in 0:1023])
hls_rom!(:VS_SIN, [_q48(sinpi(2 * (BigFloat(j) + BigFloat(1) / 2) / 1024), VSM_I.Rc) for j in 0:1023])
hls_rom!(:VS_RSQ, [begin                              # 2 octaves x 2048 bins, 18-bit words (16 fraction bits)
                       lo = BigFloat(2)^(oct - 2) * (1 + BigFloat(j) / 2048); hi = BigFloat(2)^(oct - 2) * (1 + BigFloat(j + 1) / 2048)
                       g = BigFloat(2)^-16
                       Float64(round(2 / (sqrt(lo) + sqrt(hi)) / g) * g)
                   end for oct in 0:1 for j in 0:2047])
hls_rom!(:VS_REC, [_q48(2 / (BigFloat(1024 + j) / 2048 + BigFloat(1025 + j) / 2048), VSM_I.Rq) for j in 0:1023])

# ---------------------------------------------------------------------------------------------------------------------
# 3. THE UNITS AND THE CALL (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
# 1 / sqrt(a) for 2^-8 <= a < 64: a = m 4^e, m in [1/4, 1), seed from the ROM (4096 x 18 bits: relative <= 1.4e-4), two
# Newton steps y <- y (3 - m y^2) / 2 (<= 1.4e-15 before rounding), 1 / sqrt(a) = y 2^-e
@hls function vs_rsqrt(a::Sig)::Sig
    @pragma "HLS INLINE off"                         # one shared unit (its products are one dependent chain)
    @pragma "HLS ALLOCATION operation instances=mul limit=1"
    seed::Sd[4096] = rom(:VS_RSQ)
    quarter::Rq = 0.25
    half::Rq = 0.5
    three::Rq = 3
    four::Sig = 4
    sixteen::Sig = 16
    onesig::Sig = 1
    lo::Sig = 0.00390625
    aa::Sig = a
    if aa < lo                                       # outside the range (flagged by the range check): clamped
        aa = lo
    end
    m::Rq = 0
    e::Int = 0
    if aa >= sixteen
        m = scale2(aa, -6)
        e = 3
    elseif aa >= four
        m = scale2(aa, -4)
        e = 2
    elseif aa >= onesig
        m = scale2(aa, -2)
        e = 1
    else
        m = aa
    end
    for i in 1:4                                     # a >= 2^-8: at most four multiplications by 4
        @pragma "HLS UNROLL"
        if m < quarter
            m = scale2(m, 2)
            e -= 1
        end
    end
    ix::Int = 0
    if m >= half
        ix = ifloor(scale2(m, 12))                   # [1/2, 1): 2048 .. 4095
    else
        ix = ifloor(scale2(m, 13)) - 2048            # [1/4, 1/2): 0 .. 2047
    end
    if ix > 4095                                     # a at the range's top edge (m rounded to 1)
        ix = 4095
    end
    y::Rq = seed[ix + 1]
    for it in 1:2
        y2::Rq = y * y
        my2::Rq = m * y2
        t::Rq = three - my2
        yt::Rq = y * t
        y = scale2(yt, -1)
    end
    r::Sig = y                                       # 1 / sqrt(m) in (1, 2]; then 2^-e, shifted from the finer Rq word in
    if e == 3                                        # one step, so the result is rounded to Sig once (49c)
        r = scale2(y, -3)
    elseif e == 2
        r = scale2(y, -2)
    elseif e == 1
        r = scale2(y, -1)
    elseif e == -1
        r = scale2(y, 1)
    elseif e == -2
        r = scale2(y, 2)
    elseif e == -3
        r = scale2(y, 3)
    elseif e == -4
        r = scale2(y, 4)
    end
    return r
end

# ~ 1 / a for 2^-5 <= a <= 63 (the chord's slope, 0.17-17 over the instances: vsm_checks (0)): a = m 2^k, m in [1/2, 1),
# the ROM seed only (relative <= 2^-11)
@hls function vs_recip0(a::Sig)::Sig
    seed::Rq[1024] = rom(:VS_REC)
    half::Rq = 0.5
    onesig::Sig = 1
    lo::Sig = 0.03125
    hi::Sig = 63
    aa::Sig = a
    if aa < lo
        aa = lo
    end
    if aa > hi
        aa = hi
    end
    m::Rq = 0
    k::Int = 0
    if aa >= onesig                                  # a < 64: a / 2^6, then up
        m = scale2(aa, -6)
        k = 6
    else
        m = aa
    end
    for i in 1:6
        @pragma "HLS UNROLL"
        if m < half
            m = scale2(m, 1)
            k -= 1
        end
    end
    j::Int = ifloor(scale2(m, 11)) - 1024
    y::Rq = seed[j + 1]
    r::Sig = y                                       # 1 / m in (1, 2]; 1 / a = y 2^-k
    for i in 1:6
        @pragma "HLS UNROLL"
        if k > 0
            r = scale2(r, -1)
            k -= 1
        end
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        if k < 0
            r = scale2(r, 1)
            k += 1
        end
    end
    return r
end

# 1 / v for 1/2 <= v < 2 (the DC voltage, clamped to its domain first): seed and two Newton steps (<= 2^-44)
@hls function vs_recipv(v::Sig)::Sig
    seed::Rq[1024] = rom(:VS_REC)
    two::Rq = 2
    onesig::Sig = 1
    m::Rq = v
    k::Int = 0
    if v >= onesig
        m = scale2(v, -1)
        k = 1
    end
    j::Int = ifloor(scale2(m, 11)) - 1024
    y::Rq = seed[j + 1]
    for it in 1:2
        t::Rq = m * y
        u::Rq = two - t
        y = y * u
    end
    r::Sig = y
    if k == 1
        r = scale2(y, -1)
    end
    return r
end

# One controller call. x = (omega, omega_f, u_mag), ph = (phi) in turns, pr = (the phase's remainder); u = (v_re, v_im, i_re, i_im, v_dc) in
# alpha-beta at the call; y = the held output (m, or u_c without a DC side) in alpha-beta; fl = (engaged, u_max clip,
# m_max clip, v_dc outside its domain)
@hls function vsm_call!(x::St[3], ph::Turn[1], pr::Ang[1], u::Sig[5], y::Sig[2], fl::Int[4], k::VsmCoefs)
    @pragma "HLS ARRAY_PARTITION variable=x complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    @pragma "HLS ALLOCATION operation instances=mul limit=6"   # latency-bound: one call per 6,250 clocks
    cs::Rc[1024] = rom(:VS_COS)
    sn::Rc[1024] = rom(:VS_SIN)
    one::Sig = 1
    zero::Sig = 0
    half::Sig = 0.5
    # --- the EMF E = u_mag e^{j 2 pi phi}
    phi::Turn = ph[1]
    jb::Int = ifloor(scale2(phi, 10))                # the bin (phi in [0, 1))
    if jb > 1023
        jb = 1023
    end
    hf::Turn = 0.5
    ft::Turn = scale2(phi, 10) - jb                  # the fraction within the bin, [0, 1) (the index never enters Turn)
    fh::Turn = ft - hf
    dl::Turn = scale2(fh, -10)                       # the remainder from the bin centre, |dl| <= 1/2048 turn (exact)
    twopi::Coef = 6.283185307179586
    dr::Ang = twopi * dl
    d2::Ang = dr * dr
    s6::Coef = 0.16666666666666666
    s20::Coef = 0.05
    s12::Coef = 0.08333333333333333
    sa::Ang = d2 * s20
    sb::Rq = one - sa
    sc::Ang = d2 * s6
    sd::Ang = sc * sb
    se::Rq = one - sd
    sdl::Ang = dr * se                               # sin(dr) = dr (1 - d2/6 (1 - d2/20))
    ca::Ang = d2 * s12
    cbb::Rq = one - ca
    cc::Ang = scale2(d2, -1)
    cd::Ang = cc * cbb
    cdl::Rq = one - cd                               # cos(dr) = 1 - d2/2 (1 - d2/12)
    c0::Rc = cs[jb + 1]
    s0::Rc = sn[jb + 1]
    t1::Sig = c0 * cdl
    t2::Sig = s0 * sdl
    t3::Sig = s0 * cdl
    t4::Sig = c0 * sdl
    cph::Sig = t1 - t2
    sph::Sig = t3 + t4
    um::St = x[3]
    er::Sig = um * cph
    ei::Sig = um * sph
    # --- the virtual-impedance divider
    dre::Sig = er - u[1]
    dim::Sig = ei - u[2]
    dr2::Sig = dre * dre
    di2::Sig = dim * dim
    dd2::Sig = dr2 + di2
    on::Int = 0
    if k.kf[3] > half && dd2 > k.kg[2]
        on = 1
    end
    zr::Sig = k.kz[1]                                # Z = (zr, zx) at the root (engaged) or Z0
    zx::Sig = k.kz[2]
    ipr::Sig = 0
    ipi::Sig = 0
    if on == 1
        rd::Sig = vs_rsqrt(dd2)                      # 1 / |D|
        dm::Sig = dd2 * rd                           # |D|
        gm::Sig = dm - k.kc[4]                       # |D| - c0' i_lim > 0
        q4::Sig = k.kc[3] * gm
        ds::Sig = k.kc[2] + q4                       # beta^2 + 4 c1 gamma
        rs::Sig = vs_rsqrt(ds)
        sq::Sig = ds * rs
        sbt::Sig = sq - k.kc[1]
        ov::Sig = sbt * k.kc[5]                      # ov0 >= ov*
        # the slope F'(ov0) = 2 a (re + i kpr) + 2 w (xe + i kpx), a = i re, w = i xe
        ii::Sig = k.kg[1] + ov
        pkr::Sig = k.kz[5] * ov
        px::Sig = k.kz[6] * ov
        re::Sig = k.kz[1] + pkr
        xe::Sig = k.kz[2] + px
        aa::Sig = ii * re
        ww::Sig = ii * xe
        ikr::Sig = ii * k.kz[5]
        ikx::Sig = ii * k.kz[6]
        gr::Sig = re + ikr
        gx::Sig = xe + ikx
        f1::Sig = aa * gr
        f2::Sig = ww * gx
        fs::Sig = f1 + f2
        sl::Sig = scale2(fs, 1)
        g::Sig = vs_recip0(sl)
        for it in 1:9                                # VSM_NCHORD chord steps
            ic::Sig = k.kg[1] + ov
            cr::Sig = k.kz[5] * ov
            cx::Sig = k.kz[6] * ov
            rr::Sig = k.kz[1] + cr
            xx::Sig = k.kz[2] + cx
            ua::Sig = ic * rr
            uw::Sig = ic * xx
            ua2::Sig = ua * ua
            uw2::Sig = uw * uw
            fa::Sig = ua2 + uw2
            fv::Sig = fa - dd2
            st::Sig = g * fv
            ov = ov - st
        end
        ir::Sig = k.kg[1] + ov                       # the root i, and Z(i)
        qr::Sig = k.kz[5] * ov
        qx::Sig = k.kz[6] * ov
        zr = k.kz[1] + qr
        zx = k.kz[2] + qx
        dhr::Sig = dre * rd                          # D / |D|
        dhi::Sig = dim * rd
        a2::Sig = dhr * ir                           # D i / |D| ...
        a3::Sig = dhi * ir
        zir0::Sig = zr * ir                          # ... times conj(Z) i / |D| = conj(Z) / |Z| at the root
        zix0::Sig = zx * ir
        b1::Sig = zir0 * rd
        b2::Sig = zix0 * rd
        m1::Sig = a2 * b1
        m2::Sig = a3 * b2
        m3::Sig = a3 * b1
        m4::Sig = a2 * b2
        ipr = m1 + m2
        ipi = m3 - m4
    else
        n1::Sig = dre * k.kg[3]                      # ip = D / Z0 = D y0
        n2::Sig = dim * k.kg[4]
        n3::Sig = dim * k.kg[3]
        n4::Sig = dre * k.kg[4]
        ipr = n1 - n2
        ipi = n3 + n4
    end
    zex::Sig = zx - k.kz[3]                          # ze = Z - j Zs
    w1::Sig = zr * ipr
    w2::Sig = zex * ipi
    w3::Sig = zr * ipi
    w4::Sig = zex * ipr
    zir::Sig = w1 - w2
    zii::Sig = w3 + w4
    ucr::Sig = er - zir
    uci::Sig = ei - zii
    # --- the limits and the held quantity
    ur2::Sig = ucr * ucr
    ui2::Sig = uci * uci
    uu2::Sig = ur2 + ui2
    vd::Sig = u[5]
    fdc::Int = 0
    if vd < k.kg[11]
        vd = k.kg[11]
        fdc = 1
    end
    if vd > k.kg[12]
        vd = k.kg[12]
        fdc = 1
    end
    cu::Int = 0
    if k.kf[1] > half && uu2 > k.kg[8]
        cu = 1
    end
    up2::Sig = uu2                                   # |u_c'|^2 after the u_max clip
    if cu == 1
        up2 = k.kg[8]
    end
    vd2::Sig = vd * vd
    lim2::Sig = k.kg[10] * vd2                       # (m_max v_dc)^2
    cm::Int = 0
    if k.kf[2] > half && up2 > lim2
        cm = 1
    end
    ru::Sig = vs_rsqrt(uu2)                          # 1 / |u_c| (used only when a clip acts)
    gv::Sig = vs_recipv(vd)                          # 1 / v_dc
    fac::Sig = one                                   # the held quantity = u_c fac
    if k.kf[2] > half
        if cm == 1
            fac = k.kg[9] * ru                       # m = m_max u_c / |u_c|
        elseif cu == 1
            fu::Sig = k.kg[7] * ru
            fac = fu * gv                            # m = u_max (u_c / |u_c|) / v_dc
        else
            fac = gv                                 # m = u_c / v_dc
        end
    else
        if cu == 1
            fac = k.kg[7] * ru
        end
    end
    y[1] = ucr * fac
    y[2] = uci * fac
    fl[1] = on
    fl[2] = cu
    fl[3] = cm
    fl[4] = fdc
    # --- the Euler step from the old states
    vr::Sig = u[1] * u[3]
    vi::Sig = u[2] * u[4]
    p::Sig = vr + vi
    om::St = x[1]
    of::St = x[2]
    e1::Sig = k.kx[1] - p
    e2::St = of - one
    e3::St = om - of
    e4::St = k.kx[2] - um
    g1::St = k.ka[1] * e1
    g2::St = k.ks[2] * e2
    g3::St = k.ks[3] * e3
    g4::St = k.ks[4] * e4
    x[1] = om + g1 - g2
    x[2] = of + g3
    x[3] = um + g4
    # the phase, compensated (49, 3): the increment (Ts f_n) omega with Ts f_n in two words, plus the remainder the
    # last call's rounding left; the new remainder is what the sum's rounding to Turn drops (exact)
    inc::Ang = k.ks[5] * om + k.kt[1] * om + pr[1]
    pn::Turn = phi + inc
    dd::Ang = pn - phi
    pr[1] = inc - dd
    if pn >= one
        pn = pn - one
    end
    if pn < zero
        pn = pn + one
    end
    ph[1] = pn
end

# ---------------------------------------------------------------------------------------------------------------------
# 3b. THE BOARD TOP (becomes the C++): a batch of up to VSM_NB calls of one instance (its coefficients loaded over AXI-Lite);
# the five state words in and out (the host carries them between batches: continuous = segmented); per call the held output,
# the flags packed (engaged + 2 u_max clip + 4 m clip + 8 v_dc domain); probes for the ILA: the held output's alpha part
# (m_out), a strobe toggled per call (the every-clock capture measures each call's cycles) and a sample pulse per call
# ---------------------------------------------------------------------------------------------------------------------
const VSM_NB = 256
@hls function vsm_run!(k::VsmCoefs, x::St[3], ph::Turn[1], pr::Ang[1], u::Sig[256, 5], y::Sig[256, 2], fo::Int[256],
                       ncall::Int, m_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=ph"
    @pragma "HLS INTERFACE mode=s_axilite port=pr"
    @pragma "HLS INTERFACE mode=s_axilite port=u"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    @pragma "HLS INTERFACE mode=s_axilite port=ncall"
    @pragma "HLS INTERFACE mode=ap_none port=m_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::St[3]
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    phs::Turn[1]
    prs::Ang[1]
    for i in 1:3
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    phs[1] = ph[1]
    prs[1] = pr[1]
    uu::Sig[5]
    @pragma "HLS ARRAY_PARTITION variable=uu complete"
    yy::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    ff::Int[4]
    @pragma "HLS ARRAY_PARTITION variable=ff complete"
    st::Bool = false
    for c in 1:ncall
        @pragma "HLS LOOP_TRIPCOUNT min=1 max=256"
        sample[] = false
        for i in 1:5
            uu[i] = u[c, i]
        end
        vsm_call!(xs, phs, prs, uu, yy, ff, k)
        y[c, 1] = yy[1]
        y[c, 2] = yy[2]
        fo[c] = ff[1] + 2 * ff[2] + 4 * ff[3] + 8 * ff[4]
        m_out[] = yy[1]
        st = !st
        strobe[] = st
        sample[] = true
    end
    for i in 1:3
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
    ph[1] = phs[1]
    pr[1] = prs[1]
end

# ---------------------------------------------------------------------------------------------------------------------
# 4. The Julia replay (alpha-beta), the package map, and the export
# ---------------------------------------------------------------------------------------------------------------------
"""    vsm_state(xc, omega_b, t) -> (x, ph, pr): the call unit's states from the package's (theta, omega, omega_f, u_mag) at t."""
vsm_state(xc, omega_b, t) = ([xc[2], xc[3], xc[4]], [mod((xc[1] + omega_b * t) / (2pi), 1.0)], [0.0])

const VSM_TYPES = [:Sig => "ap_fixed<48, $(VSM_I.Sig), AP_RND_CONV>",       # signals, currents, ov, |D|^2: +-128
                   :St => "ap_fixed<48, $(VSM_I.St), AP_RND_CONV>",         # omega, omega_f, u_mag: +-4
                   :Turn => "ap_fixed<48, $(VSM_I.Turn), AP_RND_CONV>",     # phi in turns: +-2
                   :Small => "ap_fixed<48, $(VSM_I.Small), AP_RND_CONV>",   # Euler gains: +-7.8e-3
                   :Ang => "ap_fixed<48, $(VSM_I.Ang), AP_RND_CONV>",       # the remainder angle and its powers
                   :Coef => "ap_fixed<48, $(VSM_I.Coef), AP_RND_CONV>",     # start-point coefficients: +-512
                   :Zc => "ap_fixed<48, $(VSM_I.Zc), AP_RND_CONV>",         # impedances and gains: +-0.5
                   :Rq => "ap_fixed<48, $(VSM_I.Rq), AP_RND_CONV>",         # the units' normalised values: +-8
                   :Rc => "ap_fixed<48, $(VSM_I.Rc), AP_RND_CONV>",         # cos / sin words
                   :Tq => "ap_fixed<48, $(VSM_I.Tq), AP_RND_CONV>",         # the increment's second word: +-2^-55
                   :Pq => "ap_fixed<48, $(VSM_I.Pq), AP_RND_CONV>",         # p_set, u_set: +-16
                   :Ka => "ap_fixed<48, $(VSM_I.Ka), AP_RND_CONV>",         # Ts / Ta: +-7.6e-6
                   :Sd => "ap_ufixed<18, 2>"]                               # rsqrt seeds

"Writes src/rt/converters/CPP/vsm/ (the call, its units, the board top, the ROMs, the first instance's coefficients, flow.cfg)."
function vsm_export(; i = 1, dir = joinpath(VSM_DIR, "CPP", "vsm"))
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "vsm_rom.h"), [:VS_COS, :VS_SIN, :VS_RSQ, :VS_REC]; header = "vsm.jl section 2 ROMs")
    hls_write(joinpath(dir, "vsm.cpp"), [:VsmCoefs, :vs_rsqrt, :vs_recip0, :vs_recipv, :vsm_call!, :vsm_run!];
              types = VSM_TYPES, local_includes = ["rt_util.h", "vsm_rom.h"],
              header = "the VSM converter's controller call (PHVsm, sampled); from src/rt/converters/vsm.jl")
    for j in 1:vsm_ninstances()
        vsm_coefs(vsm_data(j))                       # every instance fits the types
    end
    hls_write_const(joinpath(dir, "vsm_coefs.h"), "VSM_COEFS", vsm_coefs(vsm_data(i)))
    flow_config(joinpath(dir, "flow.cfg"); model = "vsm", top = "vsm_run", clock_ns = 8, ip_version = "1.0",
                ip_name = "VSM converter controller call", probes = ["m_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["m_out" => (40, "m alpha (held)")], sample_dt = 5e-5,
                board_args = "--no-download", hls_extra = String[], impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = 0, ila_ref = "", ila_check = String[], strict = true)
    vsm_board_script(joinpath(dir, "vsm_board.py"))
    println("written $dir (coefficients of all $(vsm_ninstances()) VSM converters fit their types)")
end
vsm_rt_export() = vsm_export()                       # the flow's name (scripts/realtime/export_cpp.jl vsm)

# the coefficient words in the struct's order with their fraction bits (one 64-bit slot each over AXI-Lite)
function vsm_coef_words(k::VsmCoefs)
    w = Tuple{Float64,Int}[]
    for (f, T) in ((:kz, :Zc), (:kg, :Sig), (:kc, :Coef), (:ks, :Small), (:kt, :Tq), (:kx, :Pq), (:ka, :Ka), (:kf, :Sig))
        foreach(v -> push!(w, (v, 48 - getfield(VSM_I, T))), getfield(k, f))
    end
    return w
end

"Writes the board program: the six converters' replay (vsm_board_data.bin, scripts/realtime/vsm_board_data.jl), every call against the C++ (bit for bit) and Julia (5e-8)."
function vsm_board_script(path)
    sets = [vsm_coef_words(vsm_coefs(vsm_data(j))) for j in 1:vsm_ninstances()]
    names = [vsm_data(j)["name"] for j in 1:vsm_ninstances()]
    F = (St = 48 - VSM_I.St, Turn = 48 - VSM_I.Turn, Ang = 48 - VSM_I.Ang, Sig = 48 - VSM_I.Sig)
    open(path, "w") do io
        print(io, """
# vsm_board.py: written by PHPS.RT.vsm_board_script (src/rt/converters/vsm.jl); do not edit.
# Runs the VSM controller-call IP on the PYNQ-Z2: the replay of the PHPS simulator's bus-16 fault run (six converters,
# 20,000 calls each; vsm_board_data.bin beside this file), in batches of $VSM_NB calls with the states carried by the host.
# Every call's held output against the C++ (C simulation build, bit for bit) and Julia (limit 5e-8); flags against Julia.
#   python3 vsm_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM = $VSM_NB, 5e-8
FST, FTURN, FANG, FSIG = $(F.St), $(F.Turn), $(F.Ang), $(F.Sig)
NAMES = $(repr(names))
SETS = [$(join(("[" * join(("($(repr(v)), $f)" for (v, f) in s), ", ") * "]" for s in sets), ",\n        "))]
A = {}
for m in re.finditer(r"#define XVSM_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xvsm_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("vsm.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.vsm_run_0
mem = ip.mmio.array                                    # the register space as uint32 words
def raw(v, F):
    return int(round(v * 2**F)) & ((1 << 48) - 1)
def w48(addr, v, F):
    r = raw(v, F)
    ip.write(addr, r & 0xFFFFFFFF)
    ip.write(addr + 4, r >> 32)
def r48raw(addr):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    return r - (1 << 48) if r >> 47 else r
def wvec(base, vals, F):                               # 48-bit words in 64-bit slots, vectorised
    r = np.round(np.asarray(vals, dtype=np.float64) * 2.0**F).astype(np.int64) & ((1 << 48) - 1)
    o = base // 4
    mem[o:o + 2 * len(r):2] = (r & 0xFFFFFFFF).astype(np.uint32)
    mem[o + 1:o + 2 * len(r):2] = (r >> 32).astype(np.uint32)
def rvec(base, n):
    o = base // 4
    lo = mem[o:o + 2 * n:2].astype(np.int64); hi = mem[o + 1:o + 2 * n:2].astype(np.int64) & 0xFFFF
    r = lo | (hi << 32)
    return np.where(r >> 47, r - (1 << 48), r)
def rint(base, n):
    o = base // 4
    return mem[o:o + n].astype(np.int64)

D = np.fromfile("vsm_board_data.bin", dtype=np.float64)
nconv, ncall = int(D[0]), int(D[1]); p = 2
if "--timing" in sys.argv:
    # one batch from inside the fault (calls 2001-2256 of the converter whose virtual impedance acts most, from the
    # Julia state there), for the every-clock capture of the cycles per call on the long path; no checks
    j = int(dict(a.split("=") for a in sys.argv if a.startswith("--conv=")).get("--conv", "3")) - 1
    q = 2 + j * (6 + 10 * ncall); s = int(D[q]); R = D[q + 6:q + 6 + 10 * ncall].reshape(ncall, 10)
    for i, (v, F) in enumerate(SETS[s]):
        w48(A["K"] + 8 * i, v, F)
    S = np.fromfile("vsm_board_state2000.bin", dtype=np.float64).reshape(nconv, 5)[j]
    for i in range(3):
        w48(A["X"] + 8 * i, S[i], FST)
    w48(A["PH_I"], S[3], FTURN); w48(A["PR_I"], S[4], FANG)
    wvec(A["U"], R[2000:2000 + NB, 0:5].reshape(-1), FSIG)
    ip.write(A["NCALL"], NB)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass
    fl = rint(A["FO"], NB)
    print("timing batch: converter %d, calls 2001-%d, engaged in %d of %d calls" % (j + 1, 2000 + NB, int(np.sum(fl & 1)), NB))
    sys.exit(0)
# the replay, the six converters interleaved batch by batch through the one IP (51, 3): each batch loads that converter's
# coefficients and its five state words, runs, and reads back the outputs, the packed flags and the five state words;
# outputs and states against the C++ (bit for bit: the C simulation's board top, vsm_board_bstates.bin), flags against Julia
BS = np.fromfile("vsm_board_bstates.bin", dtype=np.float64).reshape(nconv, -1, 5)
conv = []
for j in range(nconv):
    s = int(D[p]); init = D[p + 1:p + 6]; p += 6
    R = D[p:p + 10 * ncall].reshape(ncall, 10); p += 10 * ncall      # u (5), y C++ (2), y Julia (2), flags Julia
    conv.append(dict(s=s, R=R, st=list(init), Y=[], F=[], dS=0))
FS = (FST, FST, FST, FTURN, FANG)
nb_all = (ncall + NB - 1) // NB
t_run, t_all, nbat = 0.0, 0.0, 0
for b in range(nb_all):
    b0 = b * NB; n = min(NB, ncall - b0)
    for j, c in enumerate(conv):
        ta = time.perf_counter()
        for i, (v, F) in enumerate(SETS[c["s"]]):        # this instance's coefficients
            w48(A["K"] + 8 * i, v, F)
        x, ph, pr = c["st"][:3], c["st"][3], c["st"][4]
        for i in range(3):
            w48(A["X"] + 8 * i, x[i], FST)
        w48(A["PH_I"], ph, FTURN); w48(A["PR_I"], pr, FANG)
        wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG)
        ip.write(A["NCALL"], n)
        t0 = time.perf_counter()
        ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        t_run += time.perf_counter() - t0
        c["Y"].append(rvec(A["Y"], 2 * n).reshape(n, 2)); c["F"].append(rint(A["FO"], n))
        rws = [r48raw(A["X"] + 8 * i) for i in range(3)] + [r48raw(A["PH_O"]), r48raw(A["PR_O"])]
        t_all += time.perf_counter() - ta; nbat += 1
        ref = [int(round(v * 2.0**F)) for v, F in zip(BS[j, b], FS)]   # the C++ state after this batch, raw
        c["dS"] = max(c["dS"], max(abs(a - r) for a, r in zip(rws, ref)))
        c["st"] = [rws[i] / 2.0**FS[i] for i in range(5)]
ok, worst_c, worst_j, nflag, worst_s = True, 0, 0.0, 0, 0
ALLY = []                                              # the board's held outputs (raw), every call, for the figures
for j, c in enumerate(conv):
    R = c["R"]; Y = np.vstack(c["Y"]); fl = np.concatenate(c["F"])
    rc = np.round(R[:, 5:7] * 2.0**FSIG).astype(np.int64)
    dc = int(np.max(np.abs(Y - rc)))                    # board vs C++ in units of the last bit
    dj = float(np.max(np.abs(Y / 2.0**FSIG - R[:, 7:9]) / np.maximum(1.0, np.abs(R[:, 7:9]))))
    nf = int(np.sum(fl != R[:, 9].astype(np.int64)))
    pj = dc == 0 and c["dS"] == 0 and dj <= LIM and nf == 0
    print("converter %d %-28s %d calls: outputs board - C++ %d LSB, returned states (5 words, %d batches) board - C++ %d LSB, "
          "board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, NAMES[c["s"]], ncall, dc, nb_all, c["dS"], dj, LIM, nf, "PASS" if pj else "FAIL"))
    ok &= pj; worst_c = max(worst_c, dc); worst_s = max(worst_s, c["dS"]); worst_j = max(worst_j, dj); nflag += nf
    ALLY.append(Y)
np.vstack(ALLY).astype(np.int64).tofile("vsm_board_y.bin")
print("interleaved: %d batches of up to %d calls, the six converters in turn (coefficients and states reloaded every batch)"
      % (nbat, NB))
print("host time: IP runs (start to done, polled) %.3f s = %.2f us per call; whole batches (coefficients, states and inputs "
      "written, run, outputs, flags and states read) %.3f s = %.2f us per call (AXI-Lite from the processor)"
      % (t_run, t_run / (nconv * ncall) * 1e6, t_all, t_all / (nconv * ncall) * 1e6))
print("board check: max |board - Julia| %.3e (limit %.0e), outputs board - C++ %d LSB, returned states board - C++ %d LSB, "
      "flag differences %d: %s" % (worst_j, LIM, worst_c, worst_s, nflag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
""")
    end
    println("written $path")
end
