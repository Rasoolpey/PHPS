# =====================================================================================================================
# The grid-forming droop converter's controller call (PHDroop, sampled, DC rejection; REALTIME_TODO item 2; review 56)
# =====================================================================================================================
# Source: src/phsim/converter.jl, `droop_sample!` (515-529) with `droop_output` (462-486), `droop_rates!` (493-506),
# `_dcr_constants` (415-419), `_held_from` / `_radial_clip` (273-276); the DLL variant gfm_droop_pf_dcr as ported there;
# record docs/active/PH_CONVERTER_MODEL.md sections 16, 25, 45. Reference configuration: Ts = 50 us, dc_reject with
# dcr_on = :both, the physical DC side held (the call's output is the modulation m).
#
# One call (this file's map), in alpha-beta: rot = e^{j 2 pi phi}, phi = (theta + the frame angle) / 2 pi in turns. Every
# quantity of `droop_output` is rot times a controller-frame one, so the limiter runs in the controller frame and the
# result is rotated once:
#   vmf = dcc (vm - dg V_v),  imf = dcc (im - dg V_i)                     (dcr_on = both: Stage 2 reads imf too)
#   U = u_mag - (kpr + j kpx)(|imf| - i_lim) imf   if |imf| > i_lim, else u_mag
#   d = U - vmf;  |d| > dmax = Zs i_con_lim:  U = vmf + d dmax / |d|
#   |imf| > i_con_lim:  U = vmf + (U - vmf) i_con_lim / |imf|
#   m = rot U / v_dc, clipped radially to m_max
# then the Euler step from the old states: P + jQ = v conj(i); v_c = v conj(rot), i_cc = i conj(rot);
#   p_f += aP (P - p_f), q_f += aQ (Q - q_f), u_mag += aU (u_set - mq (q_f - q_set) - u_mag), vm += aV (v_c - vm),
#   im += aI (i_cc - im), V_v += (lam - 1) V_v + vm, V_i += (lam - 1) V_i + im (the old vm, im),
#   phi += Ts f_n - (Ts f_n mp)(p_f - p_set)   (mod 1)
# Hardware form (56): the filter states summed with compensation (x - x_new) + e (the lags are contractive: a per-step
# rounding would be amplified 1 / a, up to 2,000, and 53 for the DC rejection); the DC-rejection filter in increment form
# with lam - 1 on a fine grid; magnitudes by a normalising unit (the vector scaled by 2^-k into [1/2, 1), its square, the
# seeded rsqrt), so |imf| and |d| keep their relative precision at any size; no divider; 1 / v_dc by the VSM's reciprocal.
# =====================================================================================================================

const DROOP_DIR = @__DIR__
const Pf = Float64           # p_f, q_f, the set-points (ap_fixed<48, 6>)
const Vb = Float64           # the DC-rejection filter's current state V_i (ap_fixed<48, 13>)
const Gn = Float64           # lam - 1, dg (integer bits from the instances)
const Ga = Float64           # aP, aQ = Ts / Tf (each gain on a grid sized to its own magnitude: a shared type sized for
const Gu = Float64           # aU = Ts / Tv            the largest left aP ~5e-4 with 2e-13 relative precision, which moved
const Gv = Float64           # aV, aI = Ts / T_V, T_I  the lags by ~2e-13 of their swing in the fault replay)
const Gm = Float64           # mq
const Km = Float64           # Ts f_n mp (integer bits from the instances)
const Dcf = Float64          # dcc (ap_fixed<48, 1>)
const Df = Float64           # increments of p_f, q_f, u_mag, vm (ap_fixed<48, -2>)
const Dh = Float64           # increments of im, V_v (ap_fixed<48, 3>)
const Dg = Float64           # increments of V_i (ap_fixed<48, 8>)
const Rm = Float64           # the remainders of the compensated sums (ap_fixed<48, -30>)
const Rn = Float64           # normalised vector components (ap_fixed<48, 2>)
const Ry = Float64           # the normalising unit's rsqrt and magnitude (ap_fixed<48, 3>)
const Fr = Float64           # the Stage 2 factor i_con_lim / |imf| (ap_fixed<48, 2>: it reaches 1 at the boundary)

const DROOP_I = Dict{Symbol,Int}(:Sig => 8, :St => 3, :Turn => 2, :Small => -7, :Ang => -6, :Coef => 10, :Rq => 4, :Rc => 2,
                                 :Tq => -54, :Zc => 0, :Pf => 6, :Vb => 13, :Gn => -4, :Ga => -9, :Gu => -7, :Gv => -4, :Gm => -6, :Km => -14, :Dcf => 1, :Df => -2,
                                 :Dh => 3, :Dg => 8, :Rm => -30, :Rn => 2, :Ry => 3, :Fr => 2)
const DROOP_VDC = (0.5, 1.99)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Data and coefficients (BigFloat, rounded once to their types)
# ---------------------------------------------------------------------------------------------------------------------
droop_data(i::Int; path = joinpath(DROOP_DIR, "data", "ieee39_droop.toml")) = TOML.parsefile(path)["converter"][i]
droop_ninstances(; path = joinpath(DROOP_DIR, "data", "ieee39_droop.toml")) = length(TOML.parsefile(path)["converter"])

@hls struct DroopCoefs
    kl::Gn[4]        # lam - 1 (re, im); dg (re, im)
    ka::Ga[2]        # aP, aQ = Ts / (Tf_p, Tf_q)
    ku::Gu[1]        # aU = Ts / Tv
    kv::Gv[2]        # aV, aI = Ts / (T_V, T_I)
    kd::Dcf[2]       # dcc (re, im)
    kk::Zc[2]        # kpr, kpx
    kg::Sig[8]       # i_lim, i_con_lim, dmax = Zs i_con_lim, m_max, m_max^2, v_min, v_max, unused
    kp::Pf[3]        # p_set, q_set, u_set
    km::Gm[1]        # mq
    ks::Small[1]     # c0 = Ts f_n (1 + mp p_set) (f_n = omega_n / 2 pi): the phase increment's constant part
    kt::Tq[1]        # c0 - its Small rounding (the second word: a rounded p_set would drift the phase ~2e-18 turn per call)
    kpm::Km[1]       # Ts f_n mp (the increment is c0 - (Ts f_n mp) p_f)
end

function droop_coefs_big(d)
    b(k) = BigFloat(d[k])
    bc(k) = complex(BigFloat(d[k][1]), BigFloat(d[k][2]))
    Ts = b("Ts")
    lam, dg, dcc = bc("dlam"), bc("dg"), bc("dcc")
    fn = b("omega_n") / (2 * BigFloat(pi))
    return (kl = [real(lam) - 1, imag(lam), real(dg), imag(dg)], ka = [Ts / b("Tf_p"), Ts / b("Tf_q")], ku = [Ts / b("Tv")],
            kv = [Ts / b("T_V"), Ts / b("T_I")],
            kd = [real(dcc), imag(dcc)], kk = [b("kpr"), b("kpx")],
            kg = [b("i_lim"), b("i_con_lim"), b("Zs") * b("i_con_lim"), b("m_max"), b("m_max")^2, BigFloat(DROOP_VDC[1]),
                  BigFloat(DROOP_VDC[2]), zero(Ts)],
            kp = [b("p_set"), b("q_set"), b("u_set")], km = [b("mq")], ks = [Ts * fn * (1 + b("mp") * b("p_set"))],
            kt = [Ts * fn * (1 + b("mp") * b("p_set")) - BigFloat(_q48(Ts * fn * (1 + b("mp") * b("p_set")), DROOP_I[:Small]))],
            kpm = [Ts * fn * b("mp")])
end

"    droop_coefs(d; check = true) -> DroopCoefs, each entry rounded to its type's grid (range-checked)."
function droop_coefs(d; check = true)
    (d["dcr"] && d["dcr_on"] == "both" && d["dc_mode"] == "held") ||
        error("$(d["name"]): the build covers dc_reject with dcr_on = both and the held modulation")
    d["i_lim"] > 4 || error("$(d["name"]): i_lim <= 4 (the magnitude unit's output range assumes |imf| > 4 where it matters)")
    c = droop_coefs_big(d)
    q(v, T) = begin
        I_ = DROOP_I[T]
        m = Float64(maximum(abs, v))
        (!check || m < 2.0^(I_ - 1)) || error("$(d["name"]): a coefficient (max |value| $m) does not fit $T = ap_fixed<48, $I_>")
        [_q48(w, I_) for w in v]
    end
    return DroopCoefs(q(c.kl, :Gn), q(c.ka, :Ga), q(c.ku, :Gu), q(c.kv, :Gv), q(c.kd, :Dcf), q(c.kk, :Zc), q(c.kg, :Sig),
                      q(c.kp, :Pf), q(c.km, :Gm), q(c.ks, :Small), q(c.kt, :Tq), q(c.kpm, :Km))
end

# the coefficient types' integer bits from the instances (each below half its range, the tightest that fits)
function droop_types!()
    mx = Dict(:Gn => 0.0, :Ga => 0.0, :Gu => 0.0, :Gv => 0.0, :Gm => 0.0, :Km => 0.0)
    for i in 1:droop_ninstances()
        c = droop_coefs_big(droop_data(i))
        for (T, v) in ((:Gn, c.kl), (:Ga, c.ka), (:Gu, c.ku), (:Gv, c.kv), (:Gm, c.km), (:Km, c.kpm))
            mx[T] = max(mx[T], Float64(maximum(abs, v)))
        end
    end
    for (T, m) in mx
        DROOP_I[T] = ceil(Int, log2(m)) + 1               # |c| < 2^(I - 1)
    end
    return mx
end

# 2^-k and 2^k for k = -6 .. 7 (index k + 7): the normalising unit's scales (exact powers of two)
hls_rom!(:DR_PINV, [Float64(BigFloat(2)^(-k)) for k in -6:7])
hls_rom!(:DR_P2, [Float64(BigFloat(2)^k) for k in -6:7])

# ---------------------------------------------------------------------------------------------------------------------
# 2. THE UNITS AND THE CALL (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
# The normalising magnitude unit: x = xr + j xi (|x| < 128) -> k with max(|xr|, |xi|) in [2^(k-1), 2^k) (k in -6 .. 7;
# smaller vectors clamped to k = -6), n = x 2^-k (one rounding), q = |n|^2 in [1/4, 2), y = 1 / sqrt(q) by the VSM's seeds
# (4096 x 18 bits) and two Newton steps; outputs on = (n re, n im), oy = (1 / |n|, |n|), kk = k: |x| = |n| 2^k.
@hls function dr_mag!(xr::Sig, xi::Sig, on::Rn[2], oy::Ry[2], kk::Int[1])
    @pragma "HLS INLINE off"
    @pragma "HLS ALLOCATION operation instances=mul limit=1"
    seed::Sd[4096] = rom(:VS_RSQ)
    pinv::Coef[14] = rom(:DR_PINV)
    quarter::Ry = 0.25
    half::Ry = 0.5
    onen::Ry = 1
    three::Rq = 3
    ar::Sig = xr
    if ar < 0
        ar = -ar
    end
    ai::Sig = xi
    if ai < 0
        ai = -ai
    end
    mx::Sig = ar
    if ai > ar
        mx = ai
    end
    k::Int = -6
    t::Coef = 0.015625                               # thresholds 2^-6 .. 2^6 (Coef: the last doubling reaches 128)
    for i in 1:13
        @pragma "HLS UNROLL"
        if mx >= t
            k += 1
        end
        t = scale2(t, 1)
    end
    ps::Coef = pinv[k + 7]
    nr::Rn = xr * ps
    ni::Rn = xi * ps
    q1::Ry = nr * nr                                 # q in Ry <48,3>: |n| components < 1, but n may round to 1 at |x| ~ 128
    q2::Ry = ni * ni
    q0::Ry = q1 + q2
    qv::Ry = q0
    if qv < quarter                                  # only when |x| < 2^-7 (clamped for the seed; the callers' decisions are
        qv = quarter                                 # off there: thresholds >= 0.15)
    end
    ix::Int = 0
    hi::Int = 0
    if qv >= onen                                    # [1, 2): the seed of q / 4, halved
        ix = ifloor(scale2(qv, 11)) - 2048
        hi = 1
    elseif qv >= half                                # [1/2, 1)
        ix = ifloor(scale2(qv, 12))
    else                                             # [1/4, 1/2)
        ix = ifloor(scale2(qv, 13)) - 2048
    end
    if ix > 4095
        ix = 4095
    end
    s0::Sd = seed[ix + 1]
    y::Ry = s0
    if hi == 1
        y = scale2(y, -1)
    end
    for it in 1:2
        y2::Rq = y * y                               # up to 4 (q = 1/4): Rq <48,4>
        my2::Ry = qv * y2
        tt::Rq = three - my2
        yt::Rq = y * tt                              # up to 4 (q = 1/4 exactly: n on an axis at 1/2): Rq <48,4>
        y = scale2(yt, -1)
    end
    nm::Ry = q0 * y                                  # |n| = q / sqrt(q); x = 0 gives 0 (the zero bypass, 56, 3)
    on[1] = nr
    on[2] = ni
    oy[1] = y
    oy[2] = nm
    kk[1] = k
end

# One controller call. ph = (phi) in turns, pr = (the phase's remainder); xp = (p_f, q_f), xu = (u_mag), xs = (vm re, im,
# im re, im, V_v re, im), xv = (V_i re, im), xr = the eleven remainders (p_f, q_f, u_mag, vm 2, im 2, V_v 2, V_i 2);
# u = (v_re, v_im, i_re, i_im, v_dc) in alpha-beta at the call; y = the held modulation m in alpha-beta;
# fl = (boost, Stage 1, Stage 2, m clip, v_dc outside its domain)
@hls function droop_call!(ph::Turn[1], pr::Ang[1], xp::Pf[2], xu::St[1], xs::Sig[6], xv::Vb[2], xr::Rm[11], u::Sig[5],
                          y::Sig[2], fl::Int[5], k::DroopCoefs)
    @pragma "HLS ARRAY_PARTITION variable=xp complete"
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    @pragma "HLS ARRAY_PARTITION variable=xv complete"
    @pragma "HLS ARRAY_PARTITION variable=xr complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    @pragma "HLS ALLOCATION operation instances=mul limit=6"   # latency-bound: one call per 6,250 clocks
    cs::Rc[1024] = rom(:VS_COS)
    sn::Rc[1024] = rom(:VS_SIN)
    p2::Coef[14] = rom(:DR_P2)
    pinv::Coef[14] = rom(:DR_PINV)
    one::Sig = 1
    zero::Sig = 0
    # --- rot = e^{j 2 pi phi} (as the VSM: the ROM at the bin centre and the remainder's rotation)
    phi::Turn = ph[1]
    jb::Int = ifloor(scale2(phi, 10))
    if jb > 1023
        jb = 1023
    end
    hf::Turn = 0.5
    ft::Turn = scale2(phi, 10) - jb
    fh::Turn = ft - hf
    dl::Turn = scale2(fh, -10)
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
    sdl::Ang = dr * se
    ca::Ang = d2 * s12
    cbb::Rq = one - ca
    cc::Ang = scale2(d2, -1)
    cd::Ang = cc * cbb
    cdl::Rq = one - cd
    c0::Rc = cs[jb + 1]
    s0::Rc = sn[jb + 1]
    t1::Sig = c0 * cdl
    t2::Sig = s0 * sdl
    t3::Sig = s0 * cdl
    t4::Sig = c0 * sdl
    cph::Sig = t1 - t2
    sph::Sig = t3 + t4
    # --- the DC-rejected measurements (controller frame): vmf = dcc (vm - dg V_v), imf = dcc (im - dg V_i)
    gv1::Sig = k.kl[3] * xs[5]
    gv2::Sig = k.kl[4] * xs[6]
    gv3::Sig = k.kl[3] * xs[6]
    gv4::Sig = k.kl[4] * xs[5]
    av::Sig = xs[1] - (gv1 - gv2)
    bv::Sig = xs[2] - (gv3 + gv4)
    gi1::Sig = k.kl[3] * xv[1]
    gi2::Sig = k.kl[4] * xv[2]
    gi3::Sig = k.kl[3] * xv[2]
    gi4::Sig = k.kl[4] * xv[1]
    ai::Sig = xs[3] - (gi1 - gi2)
    bi::Sig = xs[4] - (gi3 + gi4)
    hv1::Sig = k.kd[1] * av
    hv2::Sig = k.kd[2] * bv
    hv3::Sig = k.kd[1] * bv
    hv4::Sig = k.kd[2] * av
    vfr::Sig = hv1 - hv2
    vfi::Sig = hv3 + hv4
    hi1::Sig = k.kd[1] * ai
    hi2::Sig = k.kd[2] * bi
    hi3::Sig = k.kd[1] * bi
    hi4::Sig = k.kd[2] * ai
    ifr::Sig = hi1 - hi2
    ifi::Sig = hi3 + hi4
    # --- |imf| (the normalising unit), the boost
    ni::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=ni complete"
    yi::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yi complete"
    ki::Int[1]
    dr_mag!(ifr, ifi, ni, yi, ki)
    sci::Coef = p2[ki[1] + 7]
    ia::Sig = yi[2] * sci                            # |imf|
    bo::Int = 0
    ur::Sig = xu[1]                                  # U = u_mag (+ j 0)
    ui::Sig = 0
    if ia > k.kg[1]
        bo = 1
        ov::Sig = ia - k.kg[1]
        kr::Sig = k.kk[1] * ov
        kx::Sig = k.kk[2] * ov
        du1::Sig = kr * ifr
        du2::Sig = kx * ifi
        du3::Sig = kr * ifi
        du4::Sig = kx * ifr
        ur = ur - (du1 - du2)
        ui = zero - (du3 + du4)
    end
    # --- Stage 1: |U - vmf| <= dmax
    dre::Sig = ur - vfr
    dim::Sig = ui - vfi
    nd::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=nd complete"
    yd::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yd complete"
    kd::Int[1]
    dr_mag!(dre, dim, nd, yd, kd)
    scd::Coef = p2[kd[1] + 7]
    dmg::Sig = yd[2] * scd                           # |d|
    s1::Int = 0
    if dmg > k.kg[3]
        s1 = 1
        f1::Ry = k.kg[3] * yd[1]                     # dmax / |n_d|: U = vmf + n_d dmax / |n_d|
        e1::Sig = f1 * nd[1]
        e2::Sig = f1 * nd[2]
        ur = vfr + e1
        ui = vfi + e2
    end
    # --- Stage 2: above i_con_lim, U - vmf scaled by i_con_lim / |imf|
    s2::Int = 0
    if ia > k.kg[2]
        s2 = 1
        g0::Sig = k.kg[2] * yi[1]
        pin::Coef = pinv[ki[1] + 7]
        fc::Fr = g0 * pin                            # i_con_lim / |imf|
        wr::Sig = ur - vfr
        wi::Sig = ui - vfi
        w1::Sig = wr * fc
        w2::Sig = wi * fc
        ur = vfr + w1
        ui = vfi + w2
    end
    # --- the held modulation: m = rot U / v_dc, clipped to m_max (as the VSM)
    ur2::Sig = ur * ur
    ui2::Sig = ui * ui
    uu2::Sig = ur2 + ui2
    vd::Sig = u[5]
    fdc::Int = 0
    if vd < k.kg[6]
        vd = k.kg[6]
        fdc = 1
    end
    if vd > k.kg[7]
        vd = k.kg[7]
        fdc = 1
    end
    vd2::Sig = vd * vd
    lim2::Sig = k.kg[5] * vd2
    cm::Int = 0
    if uu2 > lim2
        cm = 1
    end
    ru::Sig = vs_rsqrt(uu2)
    gv::Sig = vs_recipv(vd)
    fac::Sig = gv
    if cm == 1
        fac = k.kg[4] * ru
    end
    r1::Sig = cph * ur
    r2::Sig = sph * ui
    r3::Sig = sph * ur
    r4::Sig = cph * ui
    uar::Sig = r1 - r2
    uai::Sig = r3 + r4
    y[1] = uar * fac
    y[2] = uai * fac
    fl[1] = bo
    fl[2] = s1
    fl[3] = s2
    fl[4] = cm
    fl[5] = fdc
    # --- the Euler step from the old states
    pv1::Sig = u[1] * u[3]
    pv2::Sig = u[2] * u[4]
    pp::Sig = pv1 + pv2                              # P = Re(v conj i)
    qv1::Sig = u[2] * u[3]
    qv2::Sig = u[1] * u[4]
    qq::Sig = qv1 - qv2                              # Q = Im(v conj i)
    vc1::Sig = u[1] * cph
    vc2::Sig = u[2] * sph
    vc3::Sig = u[2] * cph
    vc4::Sig = u[1] * sph
    vcr::Sig = vc1 + vc2                             # v conj(rot)
    vci::Sig = vc3 - vc4
    ic1::Sig = u[3] * cph
    ic2::Sig = u[4] * sph
    ic3::Sig = u[4] * cph
    ic4::Sig = u[3] * sph
    icr::Sig = ic1 + ic2                             # i conj(rot)
    ici::Sig = ic3 - ic4
    # the old states
    pf0::Pf = xp[1]
    qf0::Pf = xp[2]
    um0::St = xu[1]
    vr0::Sig = xs[1]
    vi0::Sig = xs[2]
    ir0::Sig = xs[3]
    ii0::Sig = xs[4]
    wv0::Sig = xs[5]
    wv1::Sig = xs[6]
    zi0::Vb = xv[1]
    zi1::Vb = xv[2]
    # the phase increment c0 - (Ts f_n mp) p_f, c0 = Ts f_n (1 + mp p_set) in two words, p_f with its remainder, plus the last
    # call's phase remainder (one expression: exact in C++ up to its one rounding)
    inc::Ang = k.ks[1] + k.kt[1] - k.kpm[1] * (pf0 + xr[1]) + pr[1]
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
    # the compensated sums, each in one expression (59 below): x_new = x + increment + e, e_new = (x - x_new) + increment
    # + e. In C++ the products and sums inside an expression are exact (ap_fixed widens), so the only roundings are x_new to
    # its grid (recovered exactly by e_new) and e_new to the remainder's grid (2^-78); the increment is not rounded on its own
    # (a rounded increment's error would be amplified 1 / a by the lags: 100 for im, then 53 by the DC rejection)
    qe::Pf = qf0 - k.kp[2]
    mqe::Sig = k.km[1] * qe
    ue0::Sig = k.kp[3] - mqe                         # u_ref (the target of u_mag)
    n01::Pf = pf0 + k.ka[1] * (pp - pf0) + xr[1]
    xr[1] = pf0 - n01 + k.ka[1] * (pp - pf0) + xr[1]
    xp[1] = n01
    n02::Pf = qf0 + k.ka[2] * (qq - qf0) + xr[2]
    xr[2] = qf0 - n02 + k.ka[2] * (qq - qf0) + xr[2]
    xp[2] = n02
    n03::St = um0 + k.ku[1] * (ue0 - um0) + xr[3]
    xr[3] = um0 - n03 + k.ku[1] * (ue0 - um0) + xr[3]
    xu[1] = n03
    n04::Sig = vr0 + k.kv[1] * (vcr - vr0) + xr[4]
    xr[4] = vr0 - n04 + k.kv[1] * (vcr - vr0) + xr[4]
    xs[1] = n04
    n05::Sig = vi0 + k.kv[1] * (vci - vi0) + xr[5]
    xr[5] = vi0 - n05 + k.kv[1] * (vci - vi0) + xr[5]
    xs[2] = n05
    n06::Sig = ir0 + k.kv[2] * (icr - ir0) + xr[6]
    xr[6] = ir0 - n06 + k.kv[2] * (icr - ir0) + xr[6]
    xs[3] = n06
    n07::Sig = ii0 + k.kv[2] * (ici - ii0) + xr[7]
    xr[7] = ii0 - n07 + k.kv[2] * (ici - ii0) + xr[7]
    xs[4] = n07
    # V_v += (lam - 1) V_v + vm, V_i += (lam - 1) V_i + im (the old vm, im)
    n08::Sig = wv0 + k.kl[1] * wv0 - k.kl[2] * wv1 + vr0 + xr[8]
    xr[8] = wv0 - n08 + k.kl[1] * wv0 - k.kl[2] * wv1 + vr0 + xr[8]
    xs[5] = n08
    n09::Sig = wv1 + k.kl[1] * wv1 + k.kl[2] * wv0 + vi0 + xr[9]
    xr[9] = wv1 - n09 + k.kl[1] * wv1 + k.kl[2] * wv0 + vi0 + xr[9]
    xs[6] = n09
    n10::Vb = zi0 + k.kl[1] * zi0 - k.kl[2] * zi1 + ir0 + xr[10]
    xr[10] = zi0 - n10 + k.kl[1] * zi0 - k.kl[2] * zi1 + ir0 + xr[10]
    xv[1] = n10
    n11::Vb = zi1 + k.kl[1] * zi1 + k.kl[2] * zi0 + ii0 + xr[11]
    xr[11] = zi1 - n11 + k.kl[1] * zi1 + k.kl[2] * zi0 + ii0 + xr[11]
    xv[2] = n11
end

# ---------------------------------------------------------------------------------------------------------------------
# 2b. THE BOARD TOP (becomes the C++): a batch of up to DROOP_NB calls of one instance (coefficients over AXI-Lite); the
# state words in and out (the host carries them between batches); per call the held output and the packed flags (boost +
# 2 Stage 1 + 4 Stage 2 + 8 m clip + 16 v_dc domain); probes: m_out (the held output's alpha part), a strobe toggled per
# call, a sample pulse per call
# ---------------------------------------------------------------------------------------------------------------------
const DROOP_NB = 256
@hls function droop_run!(k::DroopCoefs, ph::Turn[1], pr::Ang[1], xp::Pf[2], xu::St[1], xs::Sig[6], xv::Vb[2], xr::Rm[11],
                         u::Sig[256, 5], y::Sig[256, 2], fo::Int[256], ncall::Int, m_out::Out{Sig}, strobe::Out{Bool},
                         sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=ph"
    @pragma "HLS INTERFACE mode=s_axilite port=pr"
    @pragma "HLS INTERFACE mode=s_axilite port=xp"
    @pragma "HLS INTERFACE mode=s_axilite port=xu"
    @pragma "HLS INTERFACE mode=s_axilite port=xs"
    @pragma "HLS INTERFACE mode=s_axilite port=xv"
    @pragma "HLS INTERFACE mode=s_axilite port=xr"
    @pragma "HLS INTERFACE mode=s_axilite port=u"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    @pragma "HLS INTERFACE mode=s_axilite port=ncall"
    @pragma "HLS INTERFACE mode=ap_none port=m_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    phs::Turn[1]
    prs::Ang[1]
    xps::Pf[2]
    @pragma "HLS ARRAY_PARTITION variable=xps complete"
    xus::St[1]
    xss::Sig[6]
    @pragma "HLS ARRAY_PARTITION variable=xss complete"
    xvs::Vb[2]
    @pragma "HLS ARRAY_PARTITION variable=xvs complete"
    xrs::Rm[11]
    @pragma "HLS ARRAY_PARTITION variable=xrs complete"
    phs[1] = ph[1]
    prs[1] = pr[1]
    xus[1] = xu[1]
    for i in 1:2
        @pragma "HLS UNROLL"
        xps[i] = xp[i]
        xvs[i] = xv[i]
    end
    for i in 1:6
        @pragma "HLS UNROLL"
        xss[i] = xs[i]
    end
    for i in 1:11
        @pragma "HLS UNROLL"
        xrs[i] = xr[i]
    end
    uu::Sig[5]
    @pragma "HLS ARRAY_PARTITION variable=uu complete"
    yy::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    ff::Int[5]
    @pragma "HLS ARRAY_PARTITION variable=ff complete"
    st::Bool = false
    for c in 1:ncall
        @pragma "HLS LOOP_TRIPCOUNT min=1 max=256"
        sample[] = false
        for i in 1:5
            uu[i] = u[c, i]
        end
        droop_call!(phs, prs, xps, xus, xss, xvs, xrs, uu, yy, ff, k)
        y[c, 1] = yy[1]
        y[c, 2] = yy[2]
        fo[c] = ff[1] + 2 * ff[2] + 4 * ff[3] + 8 * ff[4] + 16 * ff[5]
        m_out[] = yy[1]
        st = !st
        strobe[] = st
        sample[] = true
    end
    ph[1] = phs[1]
    pr[1] = prs[1]
    xu[1] = xus[1]
    for i in 1:2
        @pragma "HLS UNROLL"
        xp[i] = xps[i]
        xv[i] = xvs[i]
    end
    for i in 1:6
        @pragma "HLS UNROLL"
        xs[i] = xss[i]
    end
    for i in 1:11
        @pragma "HLS UNROLL"
        xr[i] = xrs[i]
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 3. The package's states <-> the call unit's; export
# ---------------------------------------------------------------------------------------------------------------------
"    droop_state(xc, clk) -> (ph, pr, xp, xu, xs, xv, xr): the call unit's states from the package's at the frame angle clk."
droop_state(xc, clk) = ([mod((xc[1] + clk) / (2pi), 1.0)], [0.0], [xc[2], xc[3]], [xc[4]],
                        [xc[5], xc[6], xc[7], xc[8], xc[11], xc[12]], [xc[13], xc[14]], zeros(11))

droop_types() = [T => (T === :Sd ? "ap_ufixed<18, 2>" : "ap_fixed<48, $(DROOP_I[T]), AP_RND_CONV>")
                 for T in (:Sig, :St, :Turn, :Small, :Ang, :Coef, :Rq, :Rc, :Tq, :Zc, :Pf, :Vb, :Gn, :Ga, :Gu, :Gv, :Gm, :Km, :Dcf, :Df, :Dh, :Dg,
                           :Rm, :Rn, :Ry, :Fr, :Sd)]

"Writes src/rt/converters/CPP/droop/ (the call, its units, the board top, the ROMs, the first instance's coefficients)."
function droop_export(; dir = joinpath(DROOP_DIR, "CPP", "droop"))
    setprecision(() -> droop_types!(), BigFloat, 256)
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "droop_rom.h"), [:VS_COS, :VS_SIN, :VS_RSQ, :VS_REC, :DR_PINV, :DR_P2];
                   header = "vsm.jl and droop.jl ROMs")
    hls_write(joinpath(dir, "droop.cpp"), [:DroopCoefs, :vs_rsqrt, :vs_recipv, :dr_mag!, :droop_call!, :droop_run!];
              types = droop_types(), local_includes = ["rt_util.h", "droop_rom.h"],
              header = "the droop converter's controller call (PHDroop, sampled, DC rejection); from src/rt/converters/droop.jl")
    ks = [setprecision(() -> droop_coefs(droop_data(j)), BigFloat, 256) for j in 1:droop_ninstances()]
    hls_write_const(joinpath(dir, "droop_coefs.h"), "DROOP_COEFS", ks[1])
    println("written $dir (coefficients of all $(length(ks)) droop converters fit; Gn I=$(DROOP_I[:Gn]), Km I=$(DROOP_I[:Km]))")
    return ks
end

"The flow's export (stage 1): the C++, flow.cfg and the board program."
function droop_rt_export(; dir = joinpath(DROOP_DIR, "CPP", "droop"))
    ks = droop_export(; dir)
    flow_config(joinpath(dir, "flow.cfg"); model = "droop", top = "droop_run", clock_ns = 8, ip_version = "1.0",
                ip_name = "Droop converter controller call", probes = ["m_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["m_out" => (48 - DROOP_I[:Sig], "m alpha (held)")], sample_dt = 5e-5,
                board_args = "--no-download", hls_extra = String[], impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = 0, ila_ref = "", ila_check = String[], strict = true)
    droop_board_script(joinpath(dir, "droop_board.py"), ks)
end

# the coefficient fields in the struct's order: (values in C order, fraction bits); one 64-bit AXI-Lite slot per word
droop_coef_fields(k::DroopCoefs) = [(collect(getfield(k, f)), 48 - DROOP_I[T]) for (f, T) in
    ((:kl, :Gn), (:ka, :Ga), (:ku, :Gu), (:kv, :Gv), (:kd, :Dcf), (:kk, :Zc), (:kg, :Sig), (:kp, :Pf), (:km, :Gm), (:ks, :Small),
     (:kt, :Tq), (:kpm, :Km))]

"Writes the droop board program (conv_board_script)."
function droop_board_script(path, ks)
    F(T) = 48 - DROOP_I[T]
    sw = vcat([("PH", 0, F(:Turn)), ("PR", 0, F(:Ang))], [("XP", i, F(:Pf)) for i in 0:1], [("XU", 0, F(:St))],
              [("XS", i, F(:Sig)) for i in 0:5], [("XV", i, F(:Vb)) for i in 0:1], [("XR", i, F(:Rm)) for i in 0:10])
    conv_board_script(path; tag = "droop", nb = DROOP_NB, sets = [droop_coef_fields(k) for k in ks], sw,
                      names = [droop_data(j)["name"] for j in 1:length(ks)], scalars = ["PH", "PR", "XU"], fsig = F(:Sig))
end

"""Writes a converter call's board program (tag droop / voc): the six converters' replay (<tag>_board_data.bin from
scripts/realtime/conv_board_data.jl), every call against the C++ (bit for bit) and Julia (5e-8), the returned state after every
batch against the C++. sets: per instance the coefficient fields [(values, fraction bits)]; sw: the state words (register,
index, fraction bits) in the flat order; scalars: the registers of one-word states (HLS names them _I / _O)."""
function conv_board_script(path; tag, nb, sets, sw, names, scalars, fsig)
    pyset(fs) = "[" * join(("(" * repr(v) * ", $f)" for (v, f) in fs), ", ") * "]"
    T = uppercase(tag)
    open(path, "w") do io
        print(io, """
# $(tag)_board.py: written by PHPS.RT.conv_board_script (src/rt/converters/droop.jl); do not edit.
# Runs the $tag controller-call IP on the PYNQ-Z2: the replay of the PHPS simulator's bus-16 fault run (six converters,
# 20,000 calls each; $(tag)_board_data.bin beside this file), in batches of $nb calls, the six converters interleaved
# batch by batch with their coefficients and $(length(sw)) state words reloaded. Every call's held output against the C++ (bit for
# bit) and Julia (5e-8); flags against Julia; the returned state after every batch against the C++ (bit for bit).
#   python3 $(tag)_board.py [--no-download] [--timing]   (--timing: one batch with flag bit 1 set most, for the every-clock capture)
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM, NS = $nb, 5e-8, $(length(sw))
FSIG = $fsig
NAMES = $(repr(names))
SETS = [$(join((pyset(fs) for fs in sets), ",\n        "))]
SW = $(repr(sw))   # the state words: (register, index, fraction bits)
SCALARS = $(repr(Tuple(scalars)))
A = {}
for m in re.finditer(r"#define X$(T)_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("x$(tag)_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
def reg(name, io):                                     # a scalar register: NAME_I / NAME_O, or NAME (in and out)
    return A[name + "_" + io] if name + "_" + io in A else A[name]
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("$(tag).bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.$(tag)_run_0
mem = ip.mmio.array
def raw(v, F):
    return np.round(np.asarray(v, dtype=np.float64) * 2.0**F).astype(np.int64)
def wvec(base, vals, F):
    r = raw(vals, F) & ((1 << 48) - 1)
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
def wcoefs(s):
    a = A["K"]
    for vals, F in SETS[s]:
        wvec(a, vals, F); a += 8 * len(vals)
def wstate(v):
    for (nm, i, F), x in zip(SW, v):
        wvec((reg(nm, "I") if nm in SCALARS else A[nm]) + 8 * i, [x], F)
def rstate():
    return np.array([rvec((reg(nm, "O") if nm in SCALARS else A[nm]) + 8 * i, 1)[0] for (nm, i, F) in SW])

D = np.fromfile("$(tag)_board_data.bin", dtype=np.float64)
nconv, ncall = int(D[0]), int(D[1]); p = 2
BS = np.fromfile("$(tag)_board_bstates.bin", dtype=np.float64).reshape(nconv, -1, NS)   # the C++ state after every batch
conv = []
for j in range(nconv):
    s = int(D[p]); init = D[p + 1:p + 1 + NS]; p += 1 + NS
    R = D[p:p + 10 * ncall].reshape(ncall, 10); p += 10 * ncall   # u (5), y C++ (2), y Julia (2), Julia's packed flags
    conv.append(dict(s=s, R=R, st=init.copy(), Y=[], F=[], dS=0))
FS = np.array([F for (_, _, F) in SW], dtype=np.float64)
nb_all = (ncall + NB - 1) // NB
if "--timing" in sys.argv:   # one batch where flag bit 1 is set most (e.g. the VOC's virtual impedance), from the C++ state
    best = max(((np.sum(c["R"][b * NB:(b + 1) * NB, 9].astype(np.int64) & 1), j, b) for j, c in enumerate(conv)
                for b in range(1, nb_all)))
    cnt, j, b = best; c = conv[j]; b0 = b * NB; n = min(NB, ncall - b0)
    wcoefs(c["s"]); wstate(BS[j, b - 1])
    wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG); ip.write(A["NCALL"], n)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass
    print("timing batch: converter %d, batch %d (%d calls, flag bit 1 set in %d)" % (j + 1, b, n, cnt))
    sys.exit(0)
t_run, t_all, nbat = 0.0, 0.0, 0
for b in range(nb_all):
    b0 = b * NB; n = min(NB, ncall - b0)
    for j, c in enumerate(conv):
        ta = time.perf_counter()
        wcoefs(c["s"]); wstate(c["st"])
        wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG)
        ip.write(A["NCALL"], n)
        t0 = time.perf_counter()
        ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        t_run += time.perf_counter() - t0
        c["Y"].append(rvec(A["Y"], 2 * n).reshape(n, 2)); c["F"].append(rint(A["FO"], n))
        rws = rstate()
        t_all += time.perf_counter() - ta; nbat += 1
        ref = np.round(BS[j, b] * 2.0**FS).astype(np.int64)
        c["dS"] = max(c["dS"], int(np.max(np.abs(rws - ref))))
        c["st"] = rws / 2.0**FS
ok, worst_c, worst_j, nflag, worst_s = True, 0, 0.0, 0, 0
ALLY = []
for j, c in enumerate(conv):
    R = c["R"]; Y = np.vstack(c["Y"]); fl = np.concatenate(c["F"])
    dc = int(np.max(np.abs(Y - raw(R[:, 5:7], FSIG))))
    dj = float(np.max(np.abs(Y / 2.0**FSIG - R[:, 7:9]) / np.maximum(1.0, np.abs(R[:, 7:9]))))
    nf = int(np.sum(fl != R[:, 9].astype(np.int64)))
    pj = dc == 0 and c["dS"] == 0 and dj <= LIM and nf == 0
    print("converter %d %-28s %d calls: outputs board - C++ %d LSB, returned states (%d words, %d batches) board - C++ %d LSB, "
          "board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, NAMES[c["s"]], ncall, dc, NS, nb_all, c["dS"], dj, LIM, nf, "PASS" if pj else "FAIL"))
    ok &= pj; worst_c = max(worst_c, dc); worst_s = max(worst_s, c["dS"]); worst_j = max(worst_j, dj); nflag += nf
    ALLY.append(Y)
np.vstack(ALLY).astype(np.int64).tofile("$(tag)_board_y.bin")
print("interleaved: %d batches of up to %d calls, the six converters in turn (coefficients and states reloaded every batch)" % (nbat, NB))
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
