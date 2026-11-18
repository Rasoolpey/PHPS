# =====================================================================================================================
# The saturating transformer core (PHCore): the discrete-gradient currents of a cored converter branch (REALTIME_TODO item
# 5; record docs/active/PH_CONVERTER_MODEL.md sections 19-24; review 70)
# =====================================================================================================================
# Source: src/phsim/transformer.jl `core_solve` (188-227), `_core_hnl` (236-242), `core_dgrad` (252-266); the step's use in
# src/phsim/midpoint.jl `_core_context` (71-89). The unit's map: from the rotated loop fluxes of a step's two ends
# (psi~0 = e^{-j beta} psi(x0), psi~1 = e^{j beta} psi(x1), psi = omega_b x, H and S) and the frozen mid angle theta_m, the
# branch currents (i_H, i_S) = Gonzalez's discrete gradient of g = h(., theta_m):
#   lambda*(z) = argmin |psi_H - l|^2 / (2 L2) + |l - psi_S|^2 / (2 L1) + W(l, theta_m),  W = (2/3) sum_k w(l_k),
#   l_k = Re(l e^{j(theta_m + p_k)}), p = (0, -2 pi/3, 2 pi/3); grad g = ((psi_H - lambda*) / L2, -(lambda* - psi_S) / L1);
#   dgrad = grad g(z_m) + c dz, c = [g(z1) - g(z0) - grad g(z_m) . dz] / |dz|^2, the numerator on the non-linear part
#   h_nl = kappa |delta|^2 / 2 + W_nl(lambda*), delta = lambda* - lambda_l (the package's form, record section 23).
# Hardware form (70):
#   - the inner minimisation: Newton from the linear minimiser lambda_l, NIT iterations (fixed; certified on the replay),
#     segments by comparisons, the 2x2 solve through a normalised seeded reciprocal of det K (no divider); in the linear
#     region the residual is exactly zero at lambda_l, so Newton stays there;
#   - the correction is exactly zero when the three solves (z0, z_m, z1) have the same segments in every phase (g is
#     quadratic there); otherwise its numerator is formed from exact differences, each scaled by 4^-k (|dz| ~ 2^k) before
#     its one rounding: kappa Re((delta1 - delta0) conj(delta1 + delta0)) / 2, the per-phase W_nl differences as sums over
#     the segments' overlaps of (b - a)(g1_s + g2_s ((a - lam_s) + (b - lam_s))), and the gradient term; then
#     c dz = (num 4^-k)(1 / |n|^2) n 2^k with n = dz 2^-k (dr_mag!-style normalisation of the 4-vector).
# =====================================================================================================================

const CORE_DIR = @__DIR__
const Fx = Float64           # fluxes psi, lambda (ap_fixed<48, 4>)
const Cg = Float64           # coefficients up to ~300: 1/L1, 1/L2, kappa (ap_fixed<48, 10>: Coef)
const Kd = Float64           # det K (ap_fixed<48, 20>)
const Wd = Float64           # the numerator's terms scaled by 2^-k (|dz| ~ 2^k): O(1) (ap_fixed<48, 7>)
const Pu = Float64           # 2^m, m = 0 .. 47 (ap_fixed<48, 48>: an integer grid)
const Pd = Float64           # 2^-m, m = 0 .. 47 (ap_fixed<48, 1>)
const Dl = Float64           # delta, lambda differences (ap_fixed<48, 2>)
const Ad = Float64           # adj(K) r in the Newton step (ap_fixed<48, 12>)
const Wx = Float64           # the exact intermediate of the shifts and the numerator's sums (ap_fixed<128, 40>: wiring only)
const CORE_NIT = 6
const CORE_I = Dict{Symbol,Int}(:Sig => 8, :St => 3, :Turn => 2, :Ang => -6, :Coef => 10, :Rq => 4, :Rc => 2, :Fx => 4,
                                :Kd => 20, :Wd => 7, :Pu => 48, :Pd => 1, :Dl => 2, :Wx => 40, :Ad => 12, :Zc => 0, :Small => -7, :Rn => 2, :Ry => 3)

core_data(i::Int; path = joinpath(CORE_DIR, "data", "ieee39_cores.toml")) = TOML.parsefile(path)["core"][i]
core_ninstances(; path = joinpath(CORE_DIR, "data", "ieee39_cores.toml")) = length(TOML.parsefile(path)["core"])

@hls struct CoreCoefs
    kl::Coef[5]      # 1/L2, 1/L1, kappa' = 1/L1 + 1/L2, kappa = kappa' + 1/Lm, kappa / 2
    kc::Pd[2]        # c_H = 1 / (L2 kappa), c_S = 1 / (L1 kappa): lambda_l = c_H psi_H + c_S psi_S (<48,1>: up to 0.73)
    lam::Fx[10]      # the knots (flux; lam[10] a cap above any flux)
    cur::Sig[9]      # the knot currents
    sl::Sig[9]       # the slopes
    g1::Sig[9]       # cur_s - lam_s / Lm
    g2::Sig[9]       # (sl_s - 1/Lm) / 2
end

function core_coefs_big(d)
    b(k) = BigFloat(d[k])
    L1, L2, Lm = b("L1"), b("L2"), b("Lm")
    ks = 1 / L1 + 1 / L2; kap = ks + 1 / Lm
    lam = BigFloat.(d["lam"]); cur = BigFloat.(d["cur"]); sl = BigFloat.(d["sl"])
    return (kl = [1 / L2, 1 / L1, ks, kap, kap / 2], kc = [1 / (L2 * kap), 1 / (L1 * kap)], lam = vcat(lam, BigFloat(7)),
            cur, sl, g1 = cur .- lam ./ Lm, g2 = (sl .- 1 / Lm) ./ 2)
end
function core_coefs(d; check = true)
    c = core_coefs_big(d)
    q(v, T) = begin
        I_ = CORE_I[T]; m = Float64(maximum(abs, v))
        (!check || m < 2.0^(I_ - 1)) || error("$(d["name"]): a coefficient (max $m) does not fit $T = ap_fixed<48, $I_>")
        [_q48(w, I_) for w in v]
    end
    length(d["lam"]) == 9 || error("$(d["name"]): the build has nine knots")
    return CoreCoefs(q(c.kl, :Coef), q(c.kc, :Pd), q(c.lam, :Fx), q(c.cur, :Sig), q(c.sl, :Sig), q(c.g1, :Sig), q(c.g2, :Sig))
end

# exact powers of two: CR_UP[m + 1] = 2^m, CR_DN[m + 1] = 2^-m (m = 0 .. 47)
hls_rom!(:CR_UP, [Float64(BigFloat(2)^m) for m in 0:47])
hls_rom!(:CR_DN, [Float64(BigFloat(2)^(-m)) for m in 0:47])

# x 2^m for |m| <= 31 by conditional constant shifts (wiring and muxes; a multiply by a table power of two is a full
# multiplier to HLS)
@hls function cr_shift(x::Wx, m::Int)::Wx
    @pragma "HLS INLINE"
    v::Wx = x
    r::Int = m
    if r >= 16
        v = scale2(v, 16)
        r -= 16
    end
    if r >= 8
        v = scale2(v, 8)
        r -= 8
    end
    if r >= 4
        v = scale2(v, 4)
        r -= 4
    end
    if r >= 2
        v = scale2(v, 2)
        r -= 2
    end
    if r >= 1
        v = scale2(v, 1)
        r -= 1
    end
    if r <= -16
        v = scale2(v, -16)
        r += 16
    end
    if r <= -8
        v = scale2(v, -8)
        r += 8
    end
    if r <= -4
        v = scale2(v, -4)
        r += 4
    end
    if r <= -2
        v = scale2(v, -2)
        r += 2
    end
    if r <= -1
        v = scale2(v, -1)
        r += 1
    end
    return v
end

# ---------------------------------------------------------------------------------------------------------------------
# THE INNER MINIMISATION (becomes the C++): psi = (psi_H re, im, psi_S re, im); ce = (c_k, s_k) of the three phases; out
# lo = (lambda* re, im, delta re, im), am = |lambda*_k| (3), sg = the segments (3)
# ---------------------------------------------------------------------------------------------------------------------
@hls function core_solve!(psi::Fx[4], ce::Rc[6], lo::Fx[4], am::Fx[3], sg::Int[3], k::CoreCoefs)
    @pragma "HLS INLINE off"
    @pragma "HLS ALLOCATION operation instances=mul limit=4"
    seed::Rq[1024] = rom(:VS_REC)
    two::Rq = 2
    twothirds::Rq = 0.6666666666666666
    zero::Fx = 0
    llr::Fx = k.kc[1] * psi[1] + k.kc[2] * psi[3]   # lambda_l
    lli::Fx = k.kc[1] * psi[2] + k.kc[2] * psi[4]
    br::Ad = k.kl[1] * psi[1] + k.kl[2] * psi[3]     # b = psi_H / L2 + psi_S / L1 (to ~280 on the replays: Ad)
    bi::Ad = k.kl[1] * psi[2] + k.kl[2] * psi[4]
    lr::Fx = llr
    li::Fx = lli
    for it in 1:6                                    # CORE_NIT Newton steps
        imr::Sig = 0
        imi::Sig = 0
        k11::Coef = k.kl[3]
        k22::Coef = k.kl[3]
        k12::Coef = 0
        for p in 1:3
            @pragma "HLS UNROLL"
            lk::Fx = lr * ce[2 * p - 1] - li * ce[2 * p]
            ak::Fx = lk
            if lk < zero
                ak = zero - lk
            end
            s::Int = 1
            for j in 2:9
                @pragma "HLS UNROLL"
                if ak >= k.lam[j]
                    s = j
                end
            end
            fk::Sig = k.cur[s] + k.sl[s] * (ak - k.lam[s])
            if lk < zero
                fk = zero - fk
            end
            imr += twothirds * fk * ce[2 * p - 1]
            imi -= twothirds * fk * ce[2 * p]
            ssl::Sig = twothirds * k.sl[s]
            k11 += ssl * ce[2 * p - 1] * ce[2 * p - 1]
            k22 += ssl * ce[2 * p] * ce[2 * p]
            k12 -= ssl * ce[2 * p - 1] * ce[2 * p]
            sg[p] = s
            am[p] = ak
        end
        rr::Sig = k.kl[3] * lr - br + imr
        ri::Sig = k.kl[3] * li - bi + imi
        det::Kd = k11 * k22 - k12 * k12
        # 1 / det: det = m 2^q, m in [1/2, 1) (q = 10 .. 17 over the cores: det in [kappa'^2, (kappa' + 2 sl_max)^2]),
        # the reciprocal seed and two Newton steps
        q::Int = 10
        t::Kd = 1024
        for j in 1:8
            @pragma "HLS UNROLL"
            if det >= t
                q += 1
            end
            t = scale2(t, 1)
        end
        dw::Wx = det
        mn::Rq = cr_shift(dw, -q)
        ix::Int = ifloor(scale2(mn, 11)) - 1024
        if ix > 1023
            ix = 1023
        end
        if ix < 0
            ix = 0
        end
        y::Rq = seed[ix + 1]
        for nt in 1:2
            ty::Rq = mn * y
            uy::Rq = two - ty
            y = y * uy
        end
        ar::Ad = k22 * rr - k12 * ri                # adj(K) r (to ~1000 on the replays: Ad <48,12>)
        ai::Ad = k11 * ri - k12 * rr
        ary::Wx = ar * y
        aiy::Wx = ai * y
        dr::Fx = cr_shift(ary, -q)
        di::Fx = cr_shift(aiy, -q)
        lr = lr - dr
        li = li - di
    end
    lo[1] = lr
    lo[2] = li
    lo[3] = lr - llr
    lo[4] = li - lli
end

# ---------------------------------------------------------------------------------------------------------------------
# THE UNIT (becomes the C++): z = (H0 re, im, S0 re, im, H1 re, im, S1 re, im) the rotated ends in flux units; th = theta_m
# in turns; y = (i_H re, im, i_S re, im); fl = (segments differ (the correction formed), a phase crossed more than one knot)
# ---------------------------------------------------------------------------------------------------------------------
@hls function core_unit!(z::Fx[8], th::Turn[1], y::Sig[4], fl::Int[2], k::CoreCoefs)
    @pragma "HLS ARRAY_PARTITION variable=z complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    @pragma "HLS ALLOCATION operation instances=mul limit=6"
    @pragma "HLS ALLOCATION function instances=core_solve limit=1"
    cs::Rc[1024] = rom(:VS_COS)
    sn::Rc[1024] = rom(:VS_SIN)
    seed::Rq[1024] = rom(:VS_REC)
    one::Sig = 1
    two::Rq = 2
    zero::Fx = 0
    half::Rq = 0.5
    r3::Rq = 0.8660254037844386
    # --- the phases at theta_m (the VSM's ROM and remainder rotation), then +-2 pi / 3
    phi::Turn = th[1]
    jb::Int = ifloor(scale2(phi, 10))
    if jb > 1023
        jb = 1023
    end
    hf::Turn = 0.5
    ft::Turn = scale2(phi, 10) - jb
    fh::Turn = ft - hf
    dl::Turn = scale2(fh, -10)
    twopi::Coef = 6.283185307179586
    dra::Ang = twopi * dl
    d2::Ang = dra * dra
    s6::Coef = 0.16666666666666666
    s20::Coef = 0.05
    s12::Coef = 0.08333333333333333
    sa::Ang = d2 * s20
    sb::Rq = one - sa
    sc::Ang = d2 * s6
    sd::Ang = sc * sb
    se::Rq = one - sd
    sdl::Ang = dra * se
    ca::Ang = d2 * s12
    cbb::Rq = one - ca
    cc::Ang = scale2(d2, -1)
    cd::Ang = cc * cbb
    cdl::Rq = one - cd
    c0::Rc = cs[jb + 1]
    s0::Rc = sn[jb + 1]
    ce::Rc[6]
    @pragma "HLS ARRAY_PARTITION variable=ce complete"
    ce[1] = c0 * cdl - s0 * sdl                      # cos theta_m
    ce[2] = s0 * cdl + c0 * sdl                      # sin theta_m
    ce[3] = r3 * ce[2] - half * ce[1]                # cos(theta_m - 2 pi / 3)
    ce[4] = zero - half * ce[2] - r3 * ce[1]         # sin(theta_m - 2 pi / 3)
    ce[5] = zero - half * ce[1] - r3 * ce[2]         # cos(theta_m + 2 pi / 3)
    ce[6] = r3 * ce[1] - half * ce[2]                # sin(theta_m + 2 pi / 3)
    # --- the three solves: the midpoint, the two ends
    pm::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=pm complete"
    p0::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=p0 complete"
    p1::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=p1 complete"
    for i in 1:4
        @pragma "HLS UNROLL"
        p0[i] = z[i]
        p1[i] = z[4 + i]
        pm[i] = scale2(z[i] + z[4 + i], -1)
    end
    om::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=om complete"
    o0::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=o0 complete"
    o1::Fx[4]
    @pragma "HLS ARRAY_PARTITION variable=o1 complete"
    am::Fx[3]
    a0::Fx[3]
    a1::Fx[3]
    sm::Int[3]
    s0g::Int[3]
    s1g::Int[3]
    core_solve!(pm, ce, om, am, sm, k)
    core_solve!(p0, ce, o0, a0, s0g, k)
    core_solve!(p1, ce, o1, a1, s1g, k)
    # --- the gradient at the midpoint: i_H = (psi_H - lambda*) / L2, i_S = (lambda* - psi_S) / L1
    ihr::Sig = k.kl[1] * (pm[1] - om[1])
    ihi::Sig = k.kl[1] * (pm[2] - om[2])
    isr::Sig = k.kl[2] * (om[1] - pm[3])
    isi::Sig = k.kl[2] * (om[2] - pm[4])
    diff::Int = 0
    multi::Int = 0
    for p in 1:3
        @pragma "HLS UNROLL"
        if s0g[p] != s1g[p] || s0g[p] != sm[p]
            diff = 1
        end
        dsg::Int = s1g[p] - s0g[p]
        if dsg > 1 || dsg < -1
            multi = 1
        end
    end
    fl[1] = diff
    fl[2] = multi
    if diff == 1
        # dz = z1 - z0 normalised: k_d with max |dz_i| in [2^(k_d - 1), 2^k_d), n = dz 2^-k_d (exact)
        dz::Fx[4]
        @pragma "HLS ARRAY_PARTITION variable=dz complete"
        mx::Fx = 0
        for i in 1:4
            @pragma "HLS UNROLL"
            dz[i] = z[4 + i] - z[i]
            ad::Fx = dz[i]
            if ad < zero
                ad = zero - ad
            end
            if ad > mx
                mx = ad
            end
        end
        kd::Int = -23                                # |dz| in [2^-24, 1): kd in -23 .. 0
        t::Coef = 0.00000011920928955078125            # thresholds 2^-23 .. 2^-1
        for j in 1:23
            @pragma "HLS UNROLL"
            if mx >= t
                kd += 1
            end
            t = scale2(t, 1)
        end
        nz::Rn[4]
        @pragma "HLS ARRAY_PARTITION variable=nz complete"
        for i in 1:4
            @pragma "HLS UNROLL"
            dzx::Wx = dz[i]
            nz[i] = cr_shift(dzx, 0 - kd)             # dz 2^-kd (exact)
        end
        nn::Ry = nz[1] * nz[1] + nz[2] * nz[2] + nz[3] * nz[3] + nz[4] * nz[4]   # in [1/4, 4)
        # 1 / nn: nn / 4 in [1/16, 1): normalised to [1/2, 1) by shifts, the reciprocal seed and two Newton steps
        e4::Int = 0
        mm::Rq = scale2(nn, -2)
        for j in 1:4
            @pragma "HLS UNROLL"
            if mm < half
                mm = scale2(mm, 1)
                e4 += 1
            end
        end
        jx::Int = ifloor(scale2(mm, 11)) - 1024
        if jx > 1023
            jx = 1023
        end
        yr::Rq = seed[jx + 1]
        for nt in 1:2
            tr::Rq = mm * yr
            ur::Rq = two - tr
            yr = yr * ur
        end
        yrx::Wx = yr
        rn::Coef = cr_shift(yrx, e4 - 2)              # 1 / nn = y 2^e4 / 4 (about 4: Coef)
        # the numerator scaled by 2^-kd (each term O(1) before its one rounding): c dz = (num 2^-kd) n / |n|^2, the powers
        # of two cancel (num = O(|dz|^2), n2 = |n|^2 4^kd, dz = n 2^kd)
        pkx::Wx = (o1[3] - o0[3]) * (o1[3] + o0[3]) + (o1[4] - o0[4]) * (o1[4] + o0[4])   # exact in Wx
        pks::Wd = cr_shift(pkx, 0 - kd)
        nk::Wd = k.kl[5] * pks
        gax::Wx = k.kl[1] * dz[1] + k.kl[2] * dz[3]
        gbx::Wx = k.kl[1] * dz[2] + k.kl[2] * dz[4]
        gas::Coef = cr_shift(gax, 0 - kd)             # up to ~150 (Coef)
        gbs::Coef = cr_shift(gbx, 0 - kd)
        gt::Wd = om[3] * gas + om[4] * gbs
        nwx::Wx = 0
        for p in 1:3
            @pragma "HLS UNROLL"
            lo_::Fx = a0[p]
            hi_::Fx = a1[p]
            sgn::Int = 1
            if a1[p] < a0[p]
                lo_ = a1[p]
                hi_ = a0[p]
                sgn = -1
            end
            wp::Wx = 0
            for j in 2:9                              # segment 1 has W_nl = 0
                @pragma "HLS UNROLL"
                ja::Fx = lo_
                if ja < k.lam[j]
                    ja = k.lam[j]
                end
                if ja > k.lam[j + 1]
                    ja = k.lam[j + 1]
                end
                jbb::Fx = hi_
                if jbb < k.lam[j]
                    jbb = k.lam[j]
                end
                if jbb > k.lam[j + 1]
                    jbb = k.lam[j + 1]
                end
                wp += (jbb - ja) * (k.g1[j] + k.g2[j] * ((ja - k.lam[j]) + (jbb - k.lam[j])))   # exact in Wx
            end
            if sgn < 0
                wp = zero - wp
            end
            nwx += wp
        end
        nw::Wd = cr_shift(nwx, 0 - kd)
        twothirds::Rq = 0.6666666666666666
        num::Wd = nk + twothirds * nw + gt           # [g(z1) - g(z0) - grad g(z_m) . dz] 2^-kd (gt: - grad_nl . dz)
        cg::Sig = num * rn                           # c dz = cg n
        ihr = ihr + cg * nz[1]
        ihi = ihi + cg * nz[2]
        isr = isr - cg * nz[3]
        isi = isi - cg * nz[4]
    end
    y[1] = ihr
    y[2] = ihi
    y[3] = isr
    y[4] = isi
end

# ---------------------------------------------------------------------------------------------------------------------
# THE BOARD TOP (becomes the C++): a batch of up to CORE_NB independent steps of one core (the unit is a map: no state), one
# pipelined loop at CORE_II; per step the rotated ends (8) and theta_m in, the currents (4) and flags out; probes: the i_H
# alpha part, a strobe toggled per step, a sample pulse every decim steps. Inside the twin the unit sits in the network's
# implicit step, where its latency (the loop's depth) is what counts (the composition, P5).
# ---------------------------------------------------------------------------------------------------------------------
const CORE_NB = 256
const CORE_II = 64
@hls function core_run!(k::CoreCoefs, z::Fx[256, 8], th::Turn[256], y::Sig[256, 4], fo::Int[256], n::Int, decim::Int,
                        i_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=z"
    @pragma "HLS INTERFACE mode=s_axilite port=th"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    @pragma "HLS INTERFACE mode=s_axilite port=n"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=i_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    zz::Fx[8]
    @pragma "HLS ARRAY_PARTITION variable=zz complete"
    tt::Turn[1]
    yy::Sig[4]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    ff::Int[2]
    @pragma "HLS ARRAY_PARTITION variable=ff complete"
    st::Bool = false
    cnt::Int = 0
    for s in 1:n
        @pragma "HLS PIPELINE II=64"
        @pragma "HLS LOOP_TRIPCOUNT min=1 max=256"
        for i in 1:8
            zz[i] = z[s, i]
        end
        tt[1] = th[s]
        core_unit!(zz, tt, yy, ff, k)
        for i in 1:4
            y[s, i] = yy[i]
        end
        fo[s] = ff[1] + 2 * ff[2]
        i_out[] = yy[1]
        st = !st
        strobe[] = st
        cnt += 1
        if cnt >= decim
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
    end
end

core_types() = [T => (T === :Sd ? "ap_ufixed<18, 2>" : T === :Wx ? "ap_fixed<128, 40, AP_RND_CONV>" : "ap_fixed<48, $(CORE_I[T]), AP_RND_CONV>")
                for T in (:Sig, :St, :Turn, :Ang, :Coef, :Rq, :Rc, :Fx, :Kd, :Wd, :Pu, :Pd, :Dl, :Zc, :Small, :Rn, :Ry, :Wx, :Ad)]

"Writes src/rt/converters/CPP/core/ (the unit, its solve, the board top, the ROMs, the first core's coefficients)."
function core_export(; dir = joinpath(CORE_DIR, "CPP", "core"))
    occursin("II=$(CORE_II)\"", string(HLS_DEFS[:core_run!])) || error("the PIPELINE II in core_run! differs from CORE_II")
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "core_rom.h"), [:VS_COS, :VS_SIN, :VS_REC, :CR_UP, :CR_DN]; header = "vsm.jl and core.jl ROMs")
    hls_write(joinpath(dir, "core.cpp"), [:CoreCoefs, :cr_shift, :core_solve!, :core_unit!, :core_run!]; types = core_types(),
              local_includes = ["rt_util.h", "core_rom.h"],
              header = "the saturating transformer core (PHCore): the discrete-gradient currents; from src/rt/converters/core.jl")
    ks = [core_coefs(core_data(j)) for j in 1:core_ninstances()]
    hls_write_const(joinpath(dir, "core_coefs.h"), "CORE_COEFS", ks[1])
    println("written $dir (coefficients of all $(length(ks)) cores fit)")
    return ks
end
core_rt_export() = core_export()
