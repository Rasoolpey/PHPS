# =====================================================================================================================
# The converters' physical DC side (PHConverterDC): DC link, battery behind Rb with up to three RC branches, the averaged
# bridge (REALTIME_TODO item 3, the second converter build; reviews 52-52c)
# =====================================================================================================================
# Source (the PHPS simulator): `PHConverterDC` in src/phsim/converter.jl (`_battery_current`, `_dc_rows!`,
# `converter_dc_powers`, `_bridge_voltage`, `_modulation`, `gfl_bridge`); record docs/active/PH_CONVERTER_MODEL.md sections 12,
# 13, 32-34, 43-46. States x = (v, v_1, v_2, v_3): the link voltage and the battery's RC branch voltages:
#   (c / omega_b) v'     = i_b - G v - w,        i_b = (Eb - sum v_k - v) / Rb
#   (cb_k / omega_b) v_k' = i_b - v_k / R_k
#   bridge u_x = m v (to the network), w = Re(m conj(i_c)) (out of the link)
#   held mode (VSM, droop, VOC): m held (an input);  continuous mode (GFL): u_x = clip(u_c, m_max v), m = u_x / v
# The step: implicit midpoint, as the simulator. Linear in x with the scalar w: x_{n+1} = x_n + P x_n + q + r w (P, q, r
# prepared in BigFloat at h), so v_mid = alpha + rho w with alpha = v_n + (P x_n + q)_1 / 2, rho = r_1 / 2.
#   held:       w = Re(m conj(i_c)); u_x = m v_mid
#   continuous: unclipped w = P_c / v_mid (P_c = Re(u_c conj(i_c))): v_mid^2 - alpha v_mid - rho P_c = 0, the continuation
#               root v_mid = (alpha + sqrt(alpha^2 + 4 rho P_c)) / 2; clipped w = m_max P_c / |u_c|: v_mid = alpha + rho w.
#               Acceptance (52b merge): unclipped if its discriminant > 0, v_mid in [(v_n + v_min) / 2, (v_n + v_max) / 2]
#               and |u_c|^2 <= (m_max v_mid)^2; else clipped if in the interval and |u_c|^2 > (m_max v_mid)^2; else the
#               complete state is held (with the remainders), the outputs from the held state, and the sticky flag set.
# The four states are summed with compensation (Kahan; the slow branches move ~1e-11 per step, 52 merge, 5).
# =====================================================================================================================

const DCSIDE_DIR = @__DIR__
const Lk = Float64           # the link row of P and q (ap_fixed<48, I> chosen at export from the instances at h)
const Lr = Float64           # the link row of r (the w coefficient)
const Br = Float64           # the branch rows of P and q
const Brr = Float64          # the branch rows of r over rho
const Rinv = Float64         # 1 / rho
const Ep = Float64           # eps and its products (ap_fixed<48, -3>)
const Fs = Float64           # the series terms near 1 and 2 (ap_fixed<48, 3>)
const Inc = Float64          # the increments (link)
const Ib = Float64           # the increments (branches)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Data and the prepared step (BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
dcside_data(i::Int; path = joinpath(DCSIDE_DIR, "data", "ieee39_dcside.toml")) = TOML.parsefile(path)["instance"][i]
dcside_ninstances(; path = joinpath(DCSIDE_DIR, "data", "ieee39_dcside.toml")) = length(TOML.parsefile(path)["instance"])

"The midpoint step's matrices at h (BigFloat): x_{n+1} - x_n = P x_n + q + r w (four states; absent branches zero)."
function dcside_prepare(d, h)
    B(v) = BigFloat(v)
    wb, c, Rb, Eb, G = B(d["omega_b"]), B(d["c"]), B(d["Rb"]), B(d["Eb"]), B(d["G"])
    nb = length(d["R"])
    A = zeros(BigFloat, 4, 4); b = zeros(BigFloat, 4); e = zeros(BigFloat, 4)
    # i_b = (Eb - sum v_k - v) / Rb
    A[1, 1] = wb / c * (-1 / Rb - G); b[1] = wb / c * Eb / Rb; e[1] = -wb / c
    for k in 1:nb
        A[1, 1+k] = -wb / (c * Rb)
        ck, Rk = B(d["cb"][k]), B(d["R"][k])
        A[1+k, 1] = -wb / (ck * Rb); b[1+k] = wb / ck * Eb / Rb
        for j in 1:nb
            A[1+k, 1+j] = -wb / (ck * Rb)
        end
        A[1+k, 1+k] -= wb / (ck * Rk)
    end
    M = Matrix{BigFloat}(I, 4, 4) - B(h) / 2 * A
    Mi = inv(M)
    return (P = B(h) * Mi * A, q = B(h) * Mi * b, r = B(h) * Mi * e, A, b, e)
end

# ---------------------------------------------------------------------------------------------------------------------
# 2. The coefficients (rounded once to their types; the types' integer bits are chosen at export, DCSIDE_I)
# ---------------------------------------------------------------------------------------------------------------------
@hls struct DcCoefs
    pv::Lk[4]        # the link row of P
    qv::Lk[1]        # the link row of q
    rv::Lr[1]        # the link row of r
    pb::Br[3, 4]     # the branch rows of P
    qb::Br[3]        # the branch rows of q
    rbr::Brr[3]      # the branch rows of r over rho (r_b / rho: the branches' increments from delta = rho w)
    kq::Rinv[1]      # 1 / rho (w = delta / rho, an output only)
    kc::Coef[3]      # 1 / Rb, Eb, unused (the battery current i_b = (Eb - sum v_k - v) / Rb)
    kg::Sig[6]       # m_max, m_max^2, v_min, v_max, the mode (1 continuous, 0 held), 1 / m_max
end

# the integer bits of the types (set by dcside_types!, from every instance at the build's h; defaults for h ~ 0.1 us)
const DCSIDE_I = Dict{Symbol,Int}(:Sig => 8, :St => 3, :Small => -14, :Coef => 10, :Rq => 4, :Pw => 10, :Lk => -9, :Lr => -16,
                                  :Br => -16, :Brr => -24, :Rinv => 22, :Inc => -6, :Ib => -18, :Ep => -3, :Fs => 3)
const DCSIDE_VDC = (0.5, 1.99)

function dcside_coefs(d, h; check = true)
    p = dcside_prepare(d, h)
    I_(T) = DCSIDE_I[T]
    q(v, T) = begin
        m = Float64(maximum(abs, v; init = BigFloat(0)))
        (!check || m < 2.0^(I_(T) - 1)) || error("$(d["name"]): a coefficient (max |value| $m) does not fit $T = ap_fixed<48, $(I_(T))>")
        [_q48(w, I_(T)) for w in v]
    end
    B(v) = BigFloat(v)
    mode = d["mode"] == "continuous" ? 1 : 0
    rho = p.r[1] / 2
    return DcCoefs(q(p.P[1, :], :Lk), q([p.q[1]], :Lk), q([p.r[1]], :Lr), reshape(q(vec(p.P[2:4, :]), :Br), 3, 4),
                   q(p.q[2:4], :Br), q(p.r[2:4] ./ rho, :Brr), q([1 / rho], :Rinv), q([1 / B(d["Rb"]), B(d["Eb"]), 0], :Coef),
                   q([B(d["m_max"]), B(d["m_max"])^2, B(DCSIDE_VDC[1]), B(DCSIDE_VDC[2]), mode, 1 / B(d["m_max"])], :Sig))
end

# the types' integer bits from the instances at h: the largest |coefficient| of each array kept below a quarter of the range
function dcside_types!(h)
    mx = Dict(:Lk => 0.0, :Lr => 0.0, :Br => 0.0, :Brr => 0.0, :Rinv => 0.0)
    for i in 1:dcside_ninstances()
        p = dcside_prepare(dcside_data(i), h)
        mx[:Lk] = max(mx[:Lk], Float64(maximum(abs, vcat(p.P[1, :], p.q[1]))))
        mx[:Lr] = max(mx[:Lr], Float64(abs(p.r[1])))
        mx[:Br] = max(mx[:Br], Float64(maximum(abs, vcat(vec(p.P[2:4, :]), p.q[2:4]))))
        mx[:Brr] = max(mx[:Brr], Float64(maximum(abs, p.r[2:4] ./ (p.r[1] / 2))))
        mx[:Rinv] = max(mx[:Rinv], Float64(abs(2 / p.r[1])))
    end
    for (T, m) in mx
        DCSIDE_I[T] = ceil(Int, log2(m)) + 3                 # |c| < 2^(I - 3): a quarter of the range
    end
    return mx
end

# ---------------------------------------------------------------------------------------------------------------------
# 3. THE STEP (becomes the C++). x = (v, v_1, v_2, v_3, z): the link and branch voltages and z ~ 1 / alpha of the last
# step (refreshed by two Newton steps; it keeps the reciprocal off the state's recurrence); xc the four remainders of the
# compensated sums; u = (i_re, i_im, a_re, a_im) with a = m (held) or u_c (continuous), all at the step's midpoint,
# alpha-beta; y = (u_x re, u_x im, w), yb = (i_b) at the midpoint; fl = (rejected (sticky, carried in), clipped, this step
# rejected).
# Continuous, unclipped (review 53): with kappa = rho P_c and eps = kappa / alpha^2 (|eps| <= 3.5e-4, certified),
#   delta = v_mid - alpha = (kappa / alpha) (1 - eps + 2 eps^2 - 5 eps^3 + 14 eps^4)   (truncation 42 eps^5: certified)
# clipped: delta = rho m_max P_c / |u_c| (from the inputs only); held: delta = rho P_c. Then v_mid = alpha + delta,
# x_{n+1} - x_n = d0 + (r / rho) delta (r_1 / rho = 2), w = delta / rho.
# ---------------------------------------------------------------------------------------------------------------------
@hls function dcside_core!(x::St[5], xc::Small[4], u::Sig[4], y::Sig[3], yb::Pw[1], fl::Int[3], pv::Lk[4], qv::Lk[1],
                           rv::Lr[1], pb::Br[3, 4], qb::Br[3], rbr::Brr[3], kq::Rinv[1], kc::Coef[3], kg::Sig[6])
    @pragma "HLS ARRAY_PARTITION variable=x complete"
    @pragma "HLS ARRAY_PARTITION variable=xc complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    half::Sig = 0.5
    two::St = 2
    one::Sig = 1
    lim::Ep = 0.01                                   # |eps| admitted for the series (certified <= 3.5e-4)
    # --- from the inputs only (off the state's recurrence)
    pr1::Sig = u[3] * u[1]
    pr2::Sig = u[4] * u[2]
    pc::Sig = pr1 + pr2                              # Re(a conj(i_c))
    rh::Lr = scale2(rv[1], -1)                       # rho = r_1 / 2
    kap::Inc = rh * pc                               # kappa = rho P_c (held: delta = kappa)
    a1::Sig = u[3] * u[3]
    a2::Sig = u[4] * u[4]
    uu2::Sig = a1 + a2                               # |u_c|^2
    rU::Sig = vs_rsqrt(uu2)                          # 1 / |u_c| (clamped at 2^-8 for |u_c| ~ 0: the clipped path needs |u_c| > m_max v_min)
    ua::Sig = uu2 * rU                               # |u_c|
    aa::Sig = ua * kg[6]                             # |u_c| / m_max: unclipped iff aa <= v_mid
    wc0::Sig = kg[1] * pc
    wc::Sig = wc0 * rU                               # m_max P_c / |u_c|
    dlc::Inc = rh * wc                               # the clipped candidate's delta
    sc0::Sig = kg[1] * rU                            # m_max / |u_c| (the clipped bridge voltage's factor, times v_mid)
    # --- the state's recurrence
    pv1::Inc = pv[1] * x[1]                          # the increments without w: each product its own variable,
    pv2::Inc = pv[2] * x[2]                          # summed as a tree (no serial chain on the recurrence)
    pv3::Inc = pv[3] * x[3]
    pv4::Inc = pv[4] * x[4]
    sa::Inc = pv1 + pv2
    sb::Inc = pv3 + pv4
    sc1::Inc = sa + sb
    dv0::Inc = sc1 + qv[1]
    db0::Ib[3]
    @pragma "HLS ARRAY_PARTITION variable=db0 complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        b1::Ib = pb[i, 1] * x[1]
        b2::Ib = pb[i, 2] * x[2]
        b3::Ib = pb[i, 3] * x[3]
        b4::Ib = pb[i, 4] * x[4]
        ba::Ib = b1 + b2
        bb2::Ib = b3 + b4
        bc::Ib = ba + bb2
        db0[i] = bc + qb[i]
    end
    vn::St = x[1]
    al::St = vn + scale2(dv0, -1)                    # v_mid when w = 0
    lo::St = scale2(vn + kg[3], -1)                  # the admitted v_mid interval
    hi::St = scale2(vn + kg[4], -1)
    z0::St = x[5]                                    # ~ 1 / alpha of the last step: two Newton steps to 1 / alpha
    n1::St = al * z0
    m1::St = two - n1
    z1::St = z0 * m1
    n2::St = al * z1
    m2::St = two - n2
    ai::St = z1 * m2
    gz::St = n1 - one                                # the carried reciprocal's admission (53, 4): |1 - alpha z| <= 1e-3, so
    if gz < 0                                        # two Newton steps leave <= 1e-12 relative (else the unclipped
        gz = -gz                                     # candidate is not admitted)
    end
    zl::St = 0.001
    okz::Bool = gz <= zl
    dl::Inc = kap                                    # delta (held)
    clp::Int = 0
    acc::Int = 1
    vm::St = al
    if kg[5] > half
        t::Inc = kap * ai                            # kappa / alpha
        ep::Ep = t * ai                              # eps (Ep <48,-3>: the 0.01 admission and 5 eps representable, 53, 1; 53c)
        ee::Ep = ep * ep                             # eps^2; the series in Estrin form with the constants as shifts:
        e5::Ep = scale2(ep, 2) + ep                  # 5 eps
        e14::Ep = scale2(ee, 4) - scale2(ee, 1)      # 14 eps^2
        e5h::Ep = scale2(e5, -1)                     # (halved: the terms stay near 1 in Fs <48,3>, a half unit 1.4e-14)
        e14h::Ep = scale2(e14, -1)
        g1::Fs = one - e5h                           # 1 - 5 eps / 2
        g2::Fs = g1 + e14h                           # (2 - 5 eps + 14 eps^2) / 2
        g3h::Ep = ee * g2
        g3::Ep = scale2(g3h, 1)                      # eps^2 (2 - 5 eps + 14 eps^2)
        g4::Fs = one - ep
        f::Fs = g4 + g3                              # 1 - eps + 2 eps^2 - 5 eps^3 + 14 eps^4 (truncation 42 eps^5)
        du::Inc = t * f
        vu::St = al + du
        vc::St = al + dlc
        epa::Ep = ep
        if epa < 0
            epa = -epa
        end
        oku::Bool = epa <= lim && okz && vu >= lo && vu <= hi && aa <= vu
        okc::Bool = vc >= lo && vc <= hi && aa > vc
        if oku
            dl = du
            vm = vu
        elseif okc
            dl = dlc
            vm = vc
            clp = 1
        else
            acc = 0
        end
    else
        vm = al + kap
        if vm < lo || vm > hi
            acc = 0
        end
    end
    if acc == 1
        dv::Inc = dv0 + scale2(dl, 1)                # r_1 w = 2 delta
        ev::Inc = dv + xc[1]
        xv::St = x[1] + ev
        xc[1] = x[1] - xv + ev                       # (x - x_new) + e: exact in Float64 (Julia) and fixed point (C++)
        x[1] = xv
        bm::Sig = 0                                  # the branches' midpoint sum (for i_b)
        for i in 1:3
            @pragma "HLS UNROLL"
            di::Ib = db0[i] + rbr[i] * dl
            ei::Ib = di + xc[1+i]
            xn::St = x[1+i] + ei
            bm += x[1+i] + scale2(di, -1)
            xc[1+i] = x[1+i] - xn + ei
            x[1+i] = xn
        end
        x[5] = ai
        w::Sig = dl * kq[1]                          # w = delta / rho
        if kg[5] > half
            if clp == 1
                sc::Sig = sc0 * vm                   # m_max v_mid / |u_c|
                y[1] = u[3] * sc
                y[2] = u[4] * sc
            else
                y[1] = u[3]
                y[2] = u[4]
            end
            y[3] = w
        else
            y[1] = u[3] * vm
            y[2] = u[4] * vm
            y[3] = pc
        end
        t1::Sig = bm + vm
        t2::Sig = kc[2] - t1                         # Eb - sum v_k - v (small: no large cancellation)
        yb[1] = kc[1] * t2                           # i_b at the midpoint
    else
        # rejected: the complete state held (x, z and the remainders); the outputs from the held state
        fl[1] = 1
        bs::Sig = x[2] + x[3] + x[4] + vn
        t3::Sig = kc[2] - bs
        yb[1] = kc[1] * t3
        if kg[5] > half
            q1::St = vn * ai                         # 1 / v_n by two Newton steps from 1 / alpha
            q2::St = two - q1
            zv1::St = ai * q2
            q3::St = vn * zv1
            q4::St = two - q3
            zv::St = zv1 * q4
            if aa <= vn
                y[1] = u[3]
                y[2] = u[4]
                y[3] = pc * zv
            else
                sn::Sig = sc0 * vn
                y[1] = u[3] * sn
                y[2] = u[4] * sn
                y[3] = wc
            end
        else
            y[1] = u[3] * vn
            y[2] = u[4] * vn
            y[3] = pc
        end
    end
    fl[2] = clp
    fl[3] = 1 - acc
end

# the step with the coefficient struct (Julia; the checks)
dcside_step!(x, xc, u, y, yb, fl, k::DcCoefs) = dcside_core!(x, xc, u, y, yb, fl, k.pv, k.qv, k.rv, k.pb, k.qb, k.rbr, k.kq, k.kc, k.kg)

# ---------------------------------------------------------------------------------------------------------------------
# 3b. THE BOARD TOP (becomes the C++): a run of up to DCSIDE_NB steps of one instance, the midpoint inputs per step from
# AXI-Lite memory (the replay), the outputs per step back; the coefficients copied into registers; one pipelined loop at
# II = DCSIDE_II (h = II x 8 ns); probes: v (the link voltage), a strobe toggled per step, a sample pulse every decim steps
# ---------------------------------------------------------------------------------------------------------------------
const DCSIDE_NB = 512
const DCSIDE_II = 76
@hls function dcside_run!(k::DcCoefs, x::St[5], xc::Small[4], u::Sig[512, 4], y::Sig[512, 3], yb::Pw[512], fo::Int[512],
                          flg::Int[1], nstep::Int, decim::Int, v_out::Out{St}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=xc"
    @pragma "HLS INTERFACE mode=s_axilite port=u"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=s_axilite port=yb"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    @pragma "HLS INTERFACE mode=s_axilite port=flg"
    @pragma "HLS INTERFACE mode=s_axilite port=nstep"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=v_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::St[5]
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    cs::Small[4]
    @pragma "HLS ARRAY_PARTITION variable=cs complete"
    for i in 1:4
        @pragma "HLS UNROLL"
        cs[i] = xc[i]
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    pv::Lk[4]
    @pragma "HLS ARRAY_PARTITION variable=pv complete"
    qv::Lk[1]
    rv::Lr[1]
    pb::Br[3, 4]
    @pragma "HLS ARRAY_PARTITION variable=pb complete dim=0"
    qb::Br[3]
    @pragma "HLS ARRAY_PARTITION variable=qb complete"
    rbr::Brr[3]
    @pragma "HLS ARRAY_PARTITION variable=rbr complete"
    kq::Rinv[1]
    kc::Coef[3]
    @pragma "HLS ARRAY_PARTITION variable=kc complete"
    kg::Sig[6]
    @pragma "HLS ARRAY_PARTITION variable=kg complete"
    for j in 1:4
        pv[j] = k.pv[j]
        for i in 1:3
            pb[i, j] = k.pb[i, j]
        end
    end
    qv[1] = k.qv[1]
    rv[1] = k.rv[1]
    kq[1] = k.kq[1]
    for i in 1:3
        qb[i] = k.qb[i]
        rbr[i] = k.rbr[i]
        kc[i] = k.kc[i]
    end
    for i in 1:6
        kg[i] = k.kg[i]
    end
    uu::Sig[4]
    @pragma "HLS ARRAY_PARTITION variable=uu complete"
    yy::Sig[3]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    bb::Pw[1]
    ff::Int[3]
    @pragma "HLS ARRAY_PARTITION variable=ff complete"
    ff[1] = flg[1]                                   # the sticky rejection flag, carried across calls (53, 6)
    ff[3] = 0
    st::Bool = false
    cnt::Int = 0
    for n in 1:nstep
        @pragma "HLS PIPELINE II=76"
        @pragma "HLS LOOP_TRIPCOUNT min=1 max=512"
        for i in 1:4
            uu[i] = u[n, i]
        end
        dcside_core!(xs, cs, uu, yy, bb, ff, pv, qv, rv, pb, qb, rbr, kq, kc, kg)
        y[n, 1] = yy[1]
        y[n, 2] = yy[2]
        y[n, 3] = yy[3]
        yb[n] = bb[1]
        fo[n] = ff[3] + 2 * ff[2]                    # this step: rejected + 2 clipped
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
        v_out[] = xs[1]
        strobe[] = st
    end
    for i in 1:4
        @pragma "HLS UNROLL"
        xc[i] = cs[i]
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
    flg[1] = ff[1]
end

# ---------------------------------------------------------------------------------------------------------------------
# 4. Julia references: the exact midpoint (BigFloat) for the same inputs, and the replay driver
# ---------------------------------------------------------------------------------------------------------------------
"""The exact implicit-midpoint step (BigFloat) of the simulator's rows with the midpoint inputs (i_c, a), with the acceptance
rule of the 52b merge (the exact root, guarded): returns (x_{n+1}, w, clipped, accepted); rejected: x unchanged."""
function dcside_exact_step(d, p, x, ic::Complex, a::Complex)
    B = BigFloat
    xb = B.(x)
    d0 = p.P * xb .+ p.q
    al = xb[1] + d0[1] / 2; rh = p.r[1] / 2
    lo = (xb[1] + B(DCSIDE_VDC[1])) / 2; hi = (xb[1] + B(DCSIDE_VDC[2])) / 2
    pc = B(real(a)) * B(real(ic)) + B(imag(a)) * B(imag(ic))
    if d["mode"] == "continuous"
        mm = B(d["m_max"]); ua = sqrt(B(real(a))^2 + B(imag(a))^2)
        disc = al^2 + 4rh * pc
        if disc > 0
            vu = (al + sqrt(disc)) / 2
            lo <= vu <= hi && ua <= mm * vu && return (xb .+ d0 .+ p.r .* (pc / vu), pc / vu, false, true)
        end
        if ua > 0
            wc = mm * pc / ua; vc = al + rh * wc
            lo <= vc <= hi && ua > mm * vc && return (xb .+ d0 .+ p.r .* wc, wc, true, true)
        end
        return (xb, B(0), false, false)
    end
    vm = al + rh * pc
    lo <= vm <= hi || return (xb, pc, false, false)
    return (xb .+ d0 .+ p.r .* pc, pc, false, true)
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. Export
# ---------------------------------------------------------------------------------------------------------------------
dcside_types() = [T => (T === :Sd ? "ap_ufixed<18, 2>" : "ap_fixed<48, $(DCSIDE_I[T]), AP_RND_CONV>")
                  for T in (:Sig, :St, :Small, :Coef, :Rq, :Pw, :Lk, :Lr, :Br, :Brr, :Rinv, :Inc, :Ib, :Ep, :Fs, :Sd)]

"Writes src/rt/converters/CPP/dcside/ at h = II x 8 ns (the types sized from every instance at that h)."
function dcside_export(; ii = DCSIDE_II, dir = joinpath(DCSIDE_DIR, "CPP", "dcside"))
    occursin("II=$(DCSIDE_II)\"", string(HLS_DEFS[:dcside_run!])) || error("the PIPELINE II in dcside_run! differs from DCSIDE_II")
    h = setprecision(() -> big(ii) * 8 / big(10)^9, BigFloat, 256)
    mx = setprecision(() -> dcside_types!(h), BigFloat, 256)
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "dcside_rom.h"), [:VS_RSQ, :VS_REC]; header = "the rsqrt and reciprocal seeds (vsm.jl)")
    hls_write(joinpath(dir, "dcside.cpp"), [:DcCoefs, :vs_rsqrt, :dcside_core!, :dcside_run!];
              types = dcside_types(), local_includes = ["rt_util.h", "dcside_rom.h"],
              header = "the converters' DC side (PHConverterDC), one step of h = $(Float64(h)) s; from src/rt/converters/dcside.jl")
    ks = [setprecision(() -> dcside_coefs(dcside_data(i), h), BigFloat, 256) for i in 1:dcside_ninstances()]
    hls_write_const(joinpath(dir, "dcside_coefs.h"), "DCSIDE_COEFS", ks[1])
    println("written $dir at h = $(Float64(h)) s (II $ii); types: ", join(("$T I=$(DCSIDE_I[T])" for T in (:Lk, :Lr, :Br, :Brr)), ", "),
            "; coefficients of all $(length(ks)) instances fit")
    return ks
end

"The flow's export (stage 1): the C++, flow.cfg (STEP_CYCLES = II, strict) and the board program (gate 54 merge, 3-4)."
function dcside_rt_export(; ii = DCSIDE_II, dir = joinpath(DCSIDE_DIR, "CPP", "dcside"))
    ks = dcside_export(; ii, dir)
    hb = setprecision(() -> big(ii) * 8 / big(10)^9, BigFloat, 256)
    d2 = copy(dcside_data(1)); d2["R"] = d2["R"][1:2]; d2["cb"] = d2["cb"][1:2]       # set 25: padding (as the C simulation)
    push!(ks, setprecision(() -> dcside_coefs(d2, hb), BigFloat, 256))
    h = Float64(hb)
    flow_config(joinpath(dir, "flow.cfg"); model = "dcside", top = "dcside_run", clock_ns = 8, ip_version = "1.0",
                ip_name = "Converter DC side", probes = ["v_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["v_out" => (48 - DCSIDE_I[:St], "v_dc (link)")], sample_dt = DCSIDE_DECIM * h,
                board_args = "--no-download", hls_extra = String[], impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = ii, ila_ref = "", ila_check = String[], strict = true)
    dcside_board_script(joinpath(dir, "dcside_board.py"), ks)
end

const DCSIDE_DECIM = 100                              # the ILA sample pulse every 100 steps

# the coefficient fields in the struct's order: (register offset in 64-bit slots, values in C order, fraction bits)
function dcside_coef_fields(k::DcCoefs)
    F(T) = 48 - DCSIDE_I[T]
    fs = Tuple{Vector{Float64},Int}[]
    push!(fs, (collect(k.pv), F(:Lk)), (collect(k.qv), F(:Lk)), (collect(k.rv), F(:Lr)),
          ([k.pb[i, j] for i in 1:3 for j in 1:4], F(:Br)), (collect(k.qb), F(:Br)), (collect(k.rbr), F(:Brr)),
          (collect(k.kq), F(:Rinv)), (collect(k.kc), F(:Coef)), (collect(k.kg), F(:Sig)))
    return fs
end

"""Writes the board program: the replays of dcside_board_data.bin (scripts/realtime/dcside_board_data.jl; all 24 DC sides,
106,000 steps each, and the synthetic S1-S3), streamed from a memory map in batches of DCSIDE_NB, the instances interleaved
batch by batch through the one IP with their coefficients and returned state (x, xc, flg) reloaded; every step's outputs
and packed flags against the C++ (bit for bit) and Julia (5e-8); the returned state after every batch against the C++."""
function dcside_board_script(path, ks)
    nset = length(ks)
    sets = [[(v, f) for (v, f) in dcside_coef_fields(k)] for k in ks]
    F = Dict(T => 48 - DCSIDE_I[T] for T in (:St, :Small, :Sig, :Pw))
    pyset(s) = "[" * join(("(" * repr(v) * ", $f)" for (v, f) in s), ", ") * "]"
    open(path, "w") do io
        print(io, """
# dcside_board.py: written by PHPS.RT.dcside_board_script (src/rt/converters/dcside.jl); do not edit.
# Runs the DC-side IP on the PYNQ-Z2: the replays of the PHPS simulator's bus-16 fault runs (the 24 DC sides, 106,000
# steps each, and three synthetic replays; dcside_board_data.bin beside this file, memory-mapped), in batches of
# $DCSIDE_NB steps, the instances interleaved batch by batch with their coefficients and returned state reloaded. Every
# step's outputs (u_x, w, i_b) and packed flags against the C++ (C simulation, bit for bit) and Julia (limit 5e-8); the
# returned state (x 5, xc 4, the sticky flag) after every batch against the C++ (dcside_board_bstates.bin), bit for bit.
#   python3 dcside_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM, NR, DECIM = $DCSIDE_NB, 5e-8, 14, $DCSIDE_DECIM
FST, FSM, FSIG, FPW = $(F[:St]), $(F[:Small]), $(F[:Sig]), $(F[:Pw])
SETS = [$(join((pyset(s) for s in sets), ",\n        "))]   # per set: the struct's fields (values in C order, fraction bits)
A = {}
for m in re.finditer(r"#define XDCSIDE_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xdcside_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
FLGI, FLGO = ("FLG_I", "FLG_O") if "FLG_I" in A else ("FLG", "FLG")
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("dcside.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.dcside_run_0
mem = ip.mmio.array                                    # the register space as uint32 words
def raw(v, F):
    return np.round(np.asarray(v, dtype=np.float64) * 2.0**F).astype(np.int64)
def wvec(base, vals, F):                               # 48-bit words in 64-bit slots
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

D = np.memmap("dcside_board_data.bin", dtype=np.float64, mode="r")
nrep = int(D[0]); H = np.array(D[1:1 + 11 * nrep]).reshape(nrep, 11)   # per replay: set, steps, x (5), xc (4)
BS = np.fromfile("dcside_board_bstates.bin", dtype=np.float64).reshape(-1, 10)   # the C++ state after every batch
reps, p, q = [], 1 + 11 * nrep, 0
for j in range(nrep):
    s, ns = int(H[j, 0]), int(H[j, 1]); nb = (ns + NB - 1) // NB
    reps.append(dict(s=s, ns=ns, R=D[p:p + NR * ns].reshape(ns, NR), x=H[j, 2:7].copy(), xc=H[j, 7:11].copy(), flg=0,
                     B=BS[q:q + nb], dC=0, dS=0, dJ=0.0, nf=0, nfs=0, yb=np.zeros(ns, dtype=np.int64), vb=[]))
    p += NR * ns; q += nb
assert q == len(BS), "dcside_board_bstates.bin does not match the data file"
ip.write(A["DECIM"], DECIM)
t_run, t_all, nbat, nstep = 0.0, 0.0, 0, 0
for b in range(max((r["ns"] + NB - 1) // NB for r in reps)):
    for r in reps:
        b0 = b * NB
        if b0 >= r["ns"]:
            continue
        n = min(NB, r["ns"] - b0); R = r["R"][b0:b0 + n]
        ta = time.perf_counter()
        wcoefs(r["s"])
        wvec(A["X"], r["x"], FST); wvec(A["XC"], r["xc"], FSM); ip.write(A[FLGI], r["flg"])
        wvec(A["U"], R[:, 0:4].reshape(-1), FSIG)
        ip.write(A["NSTEP"], n)
        t0 = time.perf_counter()
        ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        t_run += time.perf_counter() - t0
        Y = rvec(A["Y"], 3 * n).reshape(n, 3); YB = rvec(A["YB"], n); FO = rint(A["FO"], n)
        xs, xcs, fl = rvec(A["X"], 5), rvec(A["XC"], 4), int(ip.read(A[FLGO]))
        t_all += time.perf_counter() - ta; nbat += 1; nstep += n
        # against the C++ (raw, bit for bit) and Julia (5e-8 absolute + relative)
        dc = max(int(np.max(np.abs(Y - raw(R[:, 4:7], FSIG)))), int(np.max(np.abs(YB - raw(R[:, 7], FPW)))))
        r["dC"] = max(r["dC"], dc)
        r["nf"] += int(np.sum(FO != R[:, 8].astype(np.int64)))                    # per step: rejected + 2 clipped
        yb = YB / 2.0**FPW; yy = Y / 2.0**FSIG
        r["dJ"] = max(r["dJ"], float(np.max(np.abs(yy - R[:, 9:12]) / np.maximum(1.0, np.abs(R[:, 9:12])))),
                      float(np.max(np.abs(yb - R[:, 12]) / np.maximum(1.0, np.abs(R[:, 12])))))
        r["nf"] += int(np.sum(FO != (R[:, 13].astype(np.int64) & 3)))           # Julia's per-step flags
        ref = r["B"][b]
        ds = max(int(np.max(np.abs(xs - raw(ref[0:5], FST)))), int(np.max(np.abs(xcs - raw(ref[5:9], FSM)))))
        r["dS"] = max(r["dS"], ds)
        r["nfs"] += int(fl != int(ref[9])) + int(fl != (int(R[-1, 13]) >> 2))   # the sticky flag vs C++ and Julia
        r["yb"][b0:b0 + n] = YB; r["vb"].append(xs[0])
        r["x"], r["xc"], r["flg"] = xs / 2.0**FST, xcs / 2.0**FSM, fl
ok = True; wC = wS = 0; wJ = 0.0; nfa = 0
for j, r in enumerate(reps):
    pj = r["dC"] == 0 and r["dS"] == 0 and r["dJ"] <= LIM and r["nf"] == 0 and r["nfs"] == 0
    print("replay %2d (set %2d) %6d steps: outputs board - C++ %d LSB, returned states (%d batches) board - C++ %d LSB, "
          "board - Julia %.3e (limit %.0e), step flag differences %d, sticky flag differences %d, sticky at the end %d: %s"
          % (j + 1, r["s"] + 1, r["ns"], r["dC"], len(r["B"]), r["dS"], r["dJ"], LIM, r["nf"], r["nfs"], r["flg"],
             "PASS" if pj else "FAIL"))
    ok &= pj; wC = max(wC, r["dC"]); wS = max(wS, r["dS"]); wJ = max(wJ, r["dJ"]); nfa += r["nf"] + r["nfs"]
np.concatenate([r["yb"] for r in reps]).tofile("dcside_board_yb.bin")           # i_b every step (raw), for the figure
np.concatenate([np.array(r["vb"], dtype=np.int64) for r in reps]).tofile("dcside_board_vb.bin")   # v after every batch
print("interleaved: %d batches of up to %d steps, %d steps in all (coefficients and states reloaded every batch)" % (nbat, NB, nstep))
print("host time: IP runs (start to done, polled) %.3f s = %.3f us per step; whole batches (coefficients, state and inputs "
      "written, run, outputs, flags and state read; the comparison not timed) %.3f s = %.2f us per step (AXI-Lite from the processor)"
      % (t_run, t_run / nstep * 1e6, t_all, t_all / nstep * 1e6))
print("board check: max |board - Julia| %.3e (limit %.0e), outputs board - C++ %d LSB, returned states board - C++ %d LSB, "
      "flag differences %d: %s" % (wJ, LIM, wC, wS, nfa, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
""")
    end
    println("written $path ($nset coefficient sets)")
end
