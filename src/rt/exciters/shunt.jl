# =====================================================================================================================
# The DC shunt exciter (PHShuntExciter, saturated) on the FPGA: one build for the type, six instances
# =====================================================================================================================
# Record: docs/realtime/models/SHUNT.md; plan and reviews: docs/realtime/reviews/fpga/41*_shunt_idea_*.md
# (idea gate closed at 41c: go).

# ---------------------------------------------------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------------------------------------------------
# Source: PHShuntExciter, src/phsim/exciter.jl lines 1-170 (`_exc_rhs!`, `_exc_powers`, `_exc_energy`). Package states
# (phi_f, xa, xf); here y = x_E = c phi_f (c = Lad / (Rfd n TE); the midpoint commutes with the scaling). With c b = 1/TE:
#   TE y' = VR - (KE + SE(y)) y + (omega - 1) y / n,   SE(y) = A exp(B y),   VR = clamp(xa, VRMIN, VRMAX) (unheld)
#   TA xa' = KA (Vref - vterm - KF (omega y - xf) / TF) - xa   (non-windup N, held over the step)
#   TF xf' = omega y - xf
#   outputs at the midpoint: v_f = (Rfd/Lad) omega y,  i_f = ((1/n + KE) y + A y exp(B y)) / (c TE rF),  P = v_f (i_fd + i_f)
# Inputs u = (vterm, omega, i_fd), sampled at the step's midpoint time. Every instance has A > 0 (A = 0 refused).
# The step (41, 41b, 41c): implicit midpoint of the affine part (omega = 1, SE = 0) prepared per bank (BigFloat):
# banks 1 (N free, VR = xa_mid: candidate W), 2 (N free, VR constant: H = VRMAX, L = VRMIN), 3, 4 the same with N frozen.
# Increment d = x+ - x = P x + w vterm + c (+ wv VR) + Q (omega - 1) y_m + S N(y_m), Q = h M g, S = -h M e1 / TE,
# g = (1/(n TE), -KA KF/(TF TA) (0 frozen), 1/TF). The saturation is linearised about the previous step's y* (history
# state, review 41d: y* = y_{n-2}): N(y_m) ~ N0 + N'* y_m, N'* = SE* (1 + B y*), N0 = -SE* B y*^2. With D = dlin + S N0, eps = (Q1 (omega - 1)
# + S1 N'*) / 2, R = x1 + D1 / 2: y_m = R (1 + eps + eps^2), d1 = D1 + 2 R (eps + eps^2), d_k = D_k + a_k y_m (k = 2, 3),
# a = Q (omega - 1) + S N'*. The exp unit computes SE(y_n) during step n for step n + 1 (a two-step recurrence).

const SHUNT_RT_DIR = joinpath(@__DIR__)
const Targ = Float64         # the exp argument t' = t + 16 and its fraction (C++ ap_fixed<48, 6>)
const Frac8 = Float64        # 256 f (ap_fixed<48, 10>)
const Mant = Float64         # 2^f in [1, 2) and the polynomial (ap_fixed<48, 3>)
const Sat = Float64          # SE and the powers of two 2^(k - 16) (ap_fixed<48, 4>)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Data and preparation (BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""    shunt_rt_data(i) -> Dict: instance i (1..6) of src/rt/exciters/data/ieee39_shunt.toml, with y0 = c phi_f0."""
function shunt_rt_data(i::Int = 1; path = joinpath(SHUNT_RT_DIR, "data", "ieee39_shunt.toml"))
    d = TOML.parsefile(path)["exciter"][i]
    (d["A"] > 0 && d["B"] >= 0) || error("shunt exciter $(d["name"]): A = $(d["A"]), B = $(d["B"]) is outside the type " *
                                         "built (the saturated PHShuntExciter, A > 0; SHUNT.md 1)")
    d["y0"] = d["c"] * d["x"][1]
    d["x0"] = [d["y0"], d["x"][2], d["x"][3], d["y0"], d["y0"]]   # (y, xa, xf, y_{n-1}, y_{n-2}): history = y0 at the start
    return d
end
shunt_rt_ninstances(; path = joinpath(SHUNT_RT_DIR, "data", "ieee39_shunt.toml")) = length(TOML.parsefile(path)["exciter"])

# the affine rows z' = J z + G vterm + k + e VR (omega = 1, SE = 0), the speed vector g and the saturation vector es
function _sh_affine(d, frozen::Bool, sat::Bool)
    b(k) = BigFloat(d[k])
    KA, TA, KE, TE, KF, TF, Vref, n = b("KA"), b("TA"), b("KE"), b("TE"), b("KF"), b("TF"), b("Vref"), b("n")
    J = zeros(BigFloat, 3, 3); G = zeros(BigFloat, 3); k = zeros(BigFloat, 3); e = zeros(BigFloat, 3)
    g = zeros(BigFloat, 3); es = zeros(BigFloat, 3)
    J[1, 1] = -KE / TE
    sat ? (e[1] = 1 / TE) : (J[1, 2] = 1 / TE)
    g[1] = 1 / (n * TE); es[1] = -1 / TE
    if !frozen
        J[2, 1] = -KA * KF / (TF * TA); J[2, 2] = -1 / TA; J[2, 3] = KA * KF / (TF * TA)
        G[2] = -KA / TA; k[2] = KA * Vref / TA; g[2] = -KA * KF / (TF * TA)
    end
    J[3, 1] = 1 / TF; J[3, 3] = -1 / TF; g[3] = 1 / TF
    return J, G, k, e, g, es
end

# the prepared vectors of one bank: P = h M J, W = h M G, c = h M k, w = h M e, Q = h M g, S = h M es (BigFloat)
function _sh_bank(d, h::BigFloat, frozen::Bool, sat::Bool)
    J, G, k, e, g, es = _sh_affine(d, frozen, sat)
    M = inv(Matrix{BigFloat}(I, 3, 3) - h / 2 * J)
    return (; P = M * h * J, W = M * h * G, c = M * h * k, w = M * h * e, Q = M * h * g, S = M * h * es)
end
const SHUNT_RT_BANKS = ((false, false), (false, true), (true, false), (true, true))   # (frozen, sat) of banks 1-4

# ---------------------------------------------------------------------------------------------------------------------
# 3. The coefficients (the C++ struct ShuntCoefs) and the ROMs of the exp unit
# ---------------------------------------------------------------------------------------------------------------------
@hls struct ShuntCoefs
    P::Small[4, 9]      # banks 1 free-W, 2 free-sat, 3 frozen-W, 4 frozen-sat: h M J, row by row
    W::Small[4, 3]      # the vterm column h M G
    c::Small[4, 3]      # the constant (Vref) term
    wmax::Small[2, 3]   # sat banks (free, frozen): h M e VRMAX
    wmin::Small[2, 3]   # and h M e VRMIN
    Q::Small[4, 3]      # the speed vector h M g
    S::Small[4, 3]      # the saturation vector -h M e1 / TE
    kc::Coef[10]        # 1 KA Vref, 2 KA, 3 c3 = KA KF / TF, 4 VRMAX, 5 VRMIN, 6 Bl = B log2(e), 7 Al = log2(A) + 16,
                        # 8 B, 9 1/n + KE, 10 1 / (c TE rF)
    kt::Tiny[1]         # 1 Rfd / Lad
end

const SHUNT_RT_I = (St = 6, Sig = 8, Small = -7, Inc = -4, Coef = 8, Tiny = -8, Targ = 6, Frac8 = 10, Mant = 3, Sat = 4,
                    Pw = 2, Wd = 10)
const SHUNT_RT_FIELDS = [:P => :Small, :W => :Small, :c => :Small, :wmax => :Small, :wmin => :Small, :Q => :Small,
                         :S => :Small, :kc => :Coef, :kt => :Tiny]

"""
    shunt_rt_coefstruct(d, h; grid = true) -> ShuntCoefs

The coefficients of instance `d` for the step h (BigFloat), rounded to their C++ types (grid = false: unrounded).
"""
function shunt_rt_coefstruct(d, h::BigFloat; grid = true)
    I_ = SHUNT_RT_I
    r(v, T) = grid ? _q48(v, getfield(I_, T)) : Float64(v)
    P = zeros(4, 9); W = zeros(4, 3); c = zeros(4, 3); wmax = zeros(2, 3); wmin = zeros(2, 3); Q = zeros(4, 3); S = zeros(4, 3)
    for (bk, (fz, sat)) in enumerate(SHUNT_RT_BANKS)
        bb = _sh_bank(d, h, fz, sat)
        for i in 1:3
            for j in 1:3; P[bk, 3(i-1)+j] = r(bb.P[i, j], :Small); end
            W[bk, i] = r(bb.W[i], :Small); c[bk, i] = r(bb.c[i], :Small)
            Q[bk, i] = r(bb.Q[i], :Small); S[bk, i] = r(bb.S[i], :Small)
            if sat
                s = fz ? 2 : 1
                wmax[s, i] = r(bb.w[i] * BigFloat(d["VRMAX"]), :Small); wmin[s, i] = r(bb.w[i] * BigFloat(d["VRMIN"]), :Small)
            end
        end
    end
    b(k) = BigFloat(d[k])
    l2e = 1 / log(BigFloat(2))
    kc = [r(v, :Coef) for v in (b("KA") * b("Vref"), b("KA"), b("KA") * b("KF") / b("TF"), b("VRMAX"), b("VRMIN"),
                                b("B") * l2e, log2(b("A")) + 16, b("B"), 1 / b("n") + b("KE"), 1 / (b("c") * b("TE") * b("rF")))]
    kt = [r(b("b") / b("n"), :Tiny)]                                   # Rfd / Lad = b / n
    return ShuntCoefs(P, W, c, wmax, wmin, Q, S, kc, kt)
end

# the exp unit's tables, rounded to Mant (2^(j/256), 256 entries; the cubic's coefficients ln(2)^i / i!) and the exact
# powers of two 2^(k - 16), k = 0..17 (41b brief, 2)
shunt_rt_exp2() = [_q48(BigFloat(2)^(BigFloat(j) / 256), SHUNT_RT_I.Mant) for j in 0:255]
shunt_rt_expc() = (l = log(BigFloat(2)); [_q48(l, SHUNT_RT_I.Mant), _q48(l^2 / 2, SHUNT_RT_I.Mant), _q48(l^3 / 6, SHUNT_RT_I.Mant)])
shunt_rt_pow2() = [Float64(BigFloat(2)^(k - 16)) for k in 0:17]
hls_rom!(:SH_EXP2, shunt_rt_exp2())
hls_rom!(:SH_EXPC, shunt_rt_expc())
hls_rom!(:SH_POW2, shunt_rt_pow2())

# ---------------------------------------------------------------------------------------------------------------------
# 4. THE EXP UNIT, THE DECISIONS AND THE STEP (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
# 2^(t' - 16) = M 2^(k - 16), t' in [0, 18): k = floor(t'), f = t' - k, j = floor(256 f), r = f - j / 256 in [0, 2^-8);
# M = 2^(j/256) ((1 + l1 r) + r^2 (l2 + l3 r)) (Estrin), error <= (ln 2 / 256)^4 / 24 = 2.2e-12 (41b, 2; 41c, 1). Every step exact
# until the polynomial. mk[1] = M, kk[1] = k.
@hls function sh_exp!(tp::Targ, mk::Mant[1], kk::Int[1])
    ex2::Mant[256] = rom(:SH_EXP2)
    lc::Mant[3] = rom(:SH_EXPC)
    k::Int = ifloor(tp)
    f::Targ = tp - k
    f8::Frac8 = scale2(f, 8)
    j::Int = ifloor(f8)
    r8::Frac8 = f8 - j
    r::Targ = scale2(r8, -8)
    one::Mant = 1                                    # Estrin: (1 + l1 r) + r^2 (l2 + l3 r), two product levels
    r2::Mant = r * r
    q1::Mant = lc[1] * r
    q3::Mant = lc[3] * r
    lo::Mant = one + q1
    hi::Mant = lc[2] + q3
    q2::Mant = r2 * hi
    pl::Mant = lo + q2
    mk[1] = ex2[j + 1] * pl
    kk[1] = k
end

# SE(y) = A exp(B y) = 2^(t' - 16), t' = Bl y + Al; t' clamped to the table's range [0, 18) (outside the guard: the export
# fails on it, 41c, 3)
@hls function sh_se(yv::St, kc::Coef[10])::Sat
    @pragma "HLS INLINE off"                         # one unit for the call before the loop and the one inside it
    @pragma "HLS ALLOCATION operation instances=mul limit=3"   # 60 DSP unshared (sweep 2); its II (3) must divide the loop II (sweep 10)
    p2::Sat[18] = rom(:SH_POW2)
    zero::Targ = 0
    top::Targ = 17.999999
    t1::Targ = kc[6] * yv
    tp::Targ = t1 + kc[7]
    if tp < zero
        tp = zero
    end
    if tp > top
        tp = top
    end
    mk::Mant[1]
    kk::Int[1]
    sh_exp!(tp, mk, kk)
    se::Sat = mk[1] * p2[kk[1] + 1]
    return se
end

# the non-windup decision at x_n with this step's inputs: 0 free, 1 frozen high, 2 frozen low
@hls function sh_decide(xa::St, r::Sig, vmax::Coef, vmin::Coef)::Int
    nb::Int = 0
    if xa >= vmax && r > 0
        nb = 1
    elseif xa <= vmin && r < 0
        nb = 2
    end
    return nb
end

# the clamp at the midpoint: the consistent candidate, W (VR = xa_mid) preferred at both boundaries; 0 W, 1 H (VRMAX),
# 2 L (VRMIN), 3 none consistent. Tested on the candidates' xa increments g = 2 (xa_mid - xa_n) against umax = 2 (VRMAX -
# xa_n) and umin = 2 (VRMIN - xa_n), formed at the step's start (xa_mid >= VRMAX <=> g >= umax): no add on the chain (sweep 7)
@hls function sh_select(gw::Inc, gh::Inc, gl::Inc, umax::St, umin::St)::Int
    cs::Int = 3
    if gw >= umin && gw <= umax
        cs = 0
    elseif gh >= umax
        cs = 1
    elseif gl <= umin
        cs = 2
    end
    return cs
end

# one step of h: x = (y, xa, xf, y_{n-1}, y_{n-2}) changed in place; u = (vterm, omega, i_fd); se = SE(y_{n-2}) (the exp
# unit's result of two iterations before: the saturation is linearised about y* = y_{n-2}, review 41d);
# y = (v_f, P) at the step's midpoint; md = (N: 0 free, 1 frozen high, 2 frozen low; C: 0 W, 1 H, 2 L, 3 none)
@hls function shunt_step!(x::St[5], u::Sig[3], y::Pw[2], md::Int[2], se::Sat, kP::Small[4, 9], kW::Small[4, 3],
                          kk::Small[4, 3], kwx::Small[2, 3], kwn::Small[2, 3], kQ::Small[4, 3], kS::Small[4, 3],
                          kc::Coef[10], kt::Tiny[1])
    vt::Sig = u[1]
    om::Sig = u[2]
    ifd::Sig = u[3]
    one::Sig = 1
    zero::Wd = 0
    om1::Sig = om - one
    # the history terms (inputs and y_prev only: off the one-step chain)
    ys::St = x[5]
    by::Sig = kc[8] * ys
    sb::Sig = se * by
    np::Sig = se + sb                                # N'* = SE (1 + B y*)
    sby::Wd = sb * ys
    n0::Wd = zero - sby                              # N0 = N* - N'* y* = -SE B y*^2
    # the decision N at x_n with this step's inputs (held for the step)
    r1::Sig = kc[2] * vt
    ra::Sig = kc[1] - r1                             # KA (Vref - vterm)
    cw::Sig = kc[3] * om                             # KA KF omega / TF (input only)
    r3::Sig = cw * x[1]                              # one level of products on the loop path
    r4::Sig = kc[3] * x[3]
    tq::Sig = ra - x[2]                              # ready before the products
    vmx::St = kc[4] - x[2]                           # the clamp test's bounds (sh_select), from x_n
    vmn::St = kc[5] - x[2]
    umax::St = scale2(vmx, 1)
    umin::St = scale2(vmn, 1)
    rb::Sig = r4 - r3
    r::Sig = tq + rb                                 # two adds after the products (sweep 5)
    nb::Int = sh_decide(x[2], r, kc[4], kc[5])
    fz::Int = 0
    if nb > 0
        fz = 1
    end
    bW::Int = 1 + 2 * fz                             # the bank of candidate W, and of the constant-VR candidates
    bS::Int = 2 + 2 * fz
    # per bank, for all four banks (inputs and history only: off the one-step chain; selected by fz afterwards, so the
    # decision does not precede these products, sweep 4): the constant part K = w vterm + c + S N0, a = Q (omega - 1)
    # + S N'*, g = eps + eps^2 (eps = a1 / 2), hat a_k = a_k (1 + g) / 2
    kK::Inc[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kK complete dim=0"
    gB::Small[4]
    @pragma "HLS ARRAY_PARTITION variable=gB complete"
    hB::Small[4, 2]
    @pragma "HLS ARRAY_PARTITION variable=hB complete dim=0"
    for b in 1:4
        @pragma "HLS UNROLL"
        aa::Small[3]
        @pragma "HLS ARRAY_PARTITION variable=aa complete"
        for i in 1:3
            @pragma "HLS UNROLL"
            l4::Inc = kW[b, i] * vt
            l5::Inc = kS[b, i] * n0
            lc::Inc = l4 + l5
            kK[b, i] = lc + kk[b, i]
            qa::Small = kQ[b, i] * om1
            pa::Small = kS[b, i] * np
            aa[i] = qa + pa
        end
        eb::Small = scale2(aa[1], -1)
        e2::Small = eb * eb
        gb::Small = eb + e2
        gB[b] = gb
        for i in 1:2
            @pragma "HLS UNROLL"
            ag::Small = aa[i + 1] * gb
            ta::Small = aa[i + 1] + ag
            hB[b, i] = scale2(ta, -1)
        end
    end
    # the linear part on the chain: D = P x + K, a balanced tree of separately rounded products
    # the constant parts of the H and L candidates, K + w VRMAX and K + w VRMIN, before the products (sweep 8)
    kH::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=kH complete"
    kL::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=kL complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        kH[i] = kK[bS, i] + kwx[fz + 1, i]
        kL[i] = kK[bS, i] + kwn[fz + 1, i]
    end
    dW::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=dW complete"
    dH::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=dH complete"
    dL::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=dL complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        l1::Inc = kP[bW, 3 * i - 2] * x[1]
        l2::Inc = kP[bW, 3 * i - 1] * x[2]
        l3::Inc = kP[bW, 3 * i] * x[3]
        s1::Inc = kP[bS, 3 * i - 2] * x[1]
        s2::Inc = kP[bS, 3 * i - 1] * x[2]
        s3::Inc = kP[bS, 3 * i] * x[3]
        la::Inc = l1 + l2
        lb::Inc = l3 + kK[bW, i]
        dW[i] = la + lb
        sa::Inc = s1 + s2
        sh::Inc = s3 + kH[i]
        sl::Inc = s3 + kL[i]
        dH[i] = sa + sh
        dL[i] = sa + sl
    end
    gW::Small = gB[bW]
    gS::Small = gB[bS]
    hW2::Small = hB[bW, 1]
    hW3::Small = hB[bW, 2]
    hS2::Small = hB[bS, 1]
    hS3::Small = hB[bS, 2]
    # the three candidates, one product level after D: R = x1 + D1 / 2, d1 = D1 + 2 R g, d_k = D_k + 2 hat a_k R
    # (= D_k + a_k y_m with y_m = R (1 + g))
    RW::St = x[1] + scale2(dW[1], -1)
    RH::St = x[1] + scale2(dH[1], -1)
    RL::St = x[1] + scale2(dL[1], -1)
    rgW::Inc = RW * gW
    rgH::Inc = RH * gS
    rgL::Inc = RL * gS
    mW2::Inc = hW2 * RW
    mH2::Inc = hS2 * RH
    mL2::Inc = hS2 * RL
    mW3::Inc = hW3 * RW
    mH3::Inc = hS3 * RH
    mL3::Inc = hS3 * RL
    fW::Inc = dW[1] + scale2(rgW, 1)
    fH::Inc = dH[1] + scale2(rgH, 1)
    fL::Inc = dL[1] + scale2(rgL, 1)
    gW2::Inc = dW[2] + scale2(mW2, 1)
    gH2::Inc = dH[2] + scale2(mH2, 1)
    gL2::Inc = dL[2] + scale2(mL2, 1)
    cs::Int = sh_select(gW2, gH2, gL2, umax, umin)
    gW3::Inc = dW[3] + scale2(mW3, 1)
    gH3::Inc = dH[3] + scale2(mH3, 1)
    gL3::Inc = dL[3] + scale2(mL3, 1)
    d1::Inc = fW
    d2::Inc = gW2
    d3::Inc = gW3
    if cs == 1
        d1 = fH
        d2 = gH2
        d3 = gH3
    elseif cs == 2
        d1 = fL
        d2 = gL2
        d3 = gL3
    end
    ym::St = x[1] + scale2(d1, -1)                   # y_m for the outputs
    # the outputs at the midpoint (not fed back: outside the recurrence)
    oy::Sig = om * ym
    vf::Pw = kt[1] * oy
    npy::Wd = np * ym
    nm::Wd = n0 + npy                                # N(y_m), linearised
    ky::Sig = kc[9] * ym
    fs::Wd = ky + nm
    ifm::Sig = kc[10] * fs                           # i_f
    it::Sig = ifd + ifm
    y[1] = vf
    y[2] = vf * it
    x[5] = x[4]
    x[4] = x[1]
    x[1] = x[1] + d1
    x[2] = x[2] + d2
    x[3] = x[3] + d3
    md[1] = nb
    md[2] = cs
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. THE BOARD TOP FUNCTION (the IP on the FPGA): the step in a loop, the bench's inputs from a segment table
# ---------------------------------------------------------------------------------------------------------------------
# As brushless_run! (segments at steps 0, n_on, n_mid, n_off; us the first step's midpoint value, du = h u'). The
# history (41b, 3; 41d): SE(x[5]) and SE(x[4]) before the loop; in each iteration SE(y_n) is computed for the step two
# iterations later (se1 -> se2). A call returns all five states, so a run cut into calls equals the continuous run bit for bit.
@hls function shunt_run!(k::ShuntCoefs, x::St[5], us::Sig[4, 3], du::Small[4, 3], n_on::Int, n_mid::Int,
                         n_off::Int, n0::Int, nsteps::Int, decim::Int, y::Pw[2], xa_out::Out{St},
                         vf_out::Out{Pw}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=us"
    @pragma "HLS INTERFACE mode=s_axilite port=du"
    @pragma "HLS INTERFACE mode=s_axilite port=n_on"
    @pragma "HLS INTERFACE mode=s_axilite port=n_mid"
    @pragma "HLS INTERFACE mode=s_axilite port=n_off"
    @pragma "HLS INTERFACE mode=s_axilite port=n0"
    @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=s_axilite port=y"
    @pragma "HLS INTERFACE mode=ap_none port=xa_out"
    @pragma "HLS INTERFACE mode=ap_none port=vf_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::St[5]                                        # the states, in registers during the run
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    for i in 1:5
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    u::Sig[3]
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    yy::Pw[2]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    md::Int[2]
    @pragma "HLS ARRAY_PARTITION variable=md complete"
    kP::Small[4, 9]                                  # the coefficients in registers for the run (brushless synth 1)
    @pragma "HLS ARRAY_PARTITION variable=kP complete dim=0"
    kW::Small[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kW complete dim=0"
    kk::Small[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kk complete dim=0"
    kQ::Small[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kQ complete dim=0"
    kS::Small[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kS complete dim=0"
    kwx::Small[2, 3]
    @pragma "HLS ARRAY_PARTITION variable=kwx complete dim=0"
    kwn::Small[2, 3]
    @pragma "HLS ARRAY_PARTITION variable=kwn complete dim=0"
    kc::Coef[10]
    @pragma "HLS ARRAY_PARTITION variable=kc complete"
    kt::Tiny[1]
    @pragma "HLS ARRAY_PARTITION variable=kt complete"
    for b in 1:4
        for j in 1:9
            kP[b, j] = k.P[b, j]
        end
        for j in 1:3
            kW[b, j] = k.W[b, j]
            kk[b, j] = k.c[b, j]
            kQ[b, j] = k.Q[b, j]
            kS[b, j] = k.S[b, j]
        end
    end
    for b in 1:2
        for j in 1:3
            kwx[b, j] = k.wmax[b, j]
            kwn[b, j] = k.wmin[b, j]
        end
    end
    for j in 1:10
        kc[j] = k.kc[j]
    end
    kt[1] = k.kt[1]
    seh::Sat[2]                                      # SE(y_{n-1}), SE(y_{n-2}) for the first two steps: one call site,
    @pragma "HLS ARRAY_PARTITION variable=seh complete"   # so one exp unit before the loop (sweep 6: 18 DSP per site)
    for j in 1:2
        seh[j] = sh_se(xs[3 + j], kc)
    end
    se1::Sat = seh[1]
    se2::Sat = seh[2]
    st::Bool = false
    cnt::Int = 0
    pxa::St = xs[2]                                  # the probes' registers (review 39, 1)
    pvf::Pw = 0
    psa::Bool = false
    pst::Bool = false
    for i in 1:nsteps
        @pragma "HLS PIPELINE II=27"                 # SHUNT_RT_STEP_CYCLES (checked at export; sweeps 1-10)
        @pragma "HLS ALLOCATION operation instances=mul limit=9"    # in the loop body (PHMachine lesson; sweep 10)
        xa_out[] = pxa
        vf_out[] = pvf
        sample[] = psa
        strobe[] = pst
        m1::Int = n0 + i - 1                         # steps before this one
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
        for j in 1:3
            @pragma "HLS UNROLL"
            u[j] = us[sg, j] + du[sg, j] * ns
        end
        sn::Sat = sh_se(xs[1], kc)                   # SE(y_n), for the step two iterations later
        shunt_step!(xs, u, yy, md, se2, kP, kW, kk, kwx, kwn, kQ, kS, kc, kt)
        se2 = se1
        se1 = sn
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            psa = true
        else
            psa = false
        end
        pxa = xs[2]
        pvf = yy[1]
        pst = st
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
    y[1] = yy[1]
    y[2] = yy[2]
end

# ---------------------------------------------------------------------------------------------------------------------
# 6. Export: bench cases, checks, C++, testbench, board program
# ---------------------------------------------------------------------------------------------------------------------
const SHUNT_RT_CLOCK_NS = 8
const SHUNT_RT_STEP_CYCLES = 27
const SHUNT_RT_H = "$(SHUNT_RT_STEP_CYCLES * SHUNT_RT_CLOCK_NS)e-9"
const SHUNT_RT_TYPES = [:St => "ap_fixed<48, 6, AP_RND_CONV>",          # states: +-32, grid 2.3e-13
                        :Sig => "ap_fixed<48, 8, AP_RND_CONV>",         # inputs, xa_mid, N'*, decision terms: +-128
                        :Small => "ap_fixed<48, -7, AP_RND_CONV>",      # prepared step entries, a, eps: +-3.9e-3
                        :Inc => "ap_fixed<48, -4, AP_RND_CONV>",        # increments: +-0.031, 2.2e-16
                        :Coef => "ap_fixed<48, 8, AP_RND_CONV>",        # gains: +-128
                        :Tiny => "ap_fixed<48, -8, AP_RND_CONV>",       # Rfd / Lad: +-1.95e-3
                        :Targ => "ap_fixed<48, 6, AP_RND_CONV>",        # exp argument t' in [0, 18), r: grid 2^-42
                        :Frac8 => "ap_fixed<48, 10, AP_RND_CONV>",      # 256 f in [0, 256): grid 2^-38
                        :Mant => "ap_fixed<48, 3, AP_RND_CONV>",        # 2^f, polynomial: +-4, grid 2.8e-14
                        :Sat => "ap_fixed<48, 4, AP_RND_CONV>",         # SE, 2^(k-16) (2^-16 exact): +-8
                        :Pw => "ap_fixed<48, 2, AP_RND_CONV>",          # v_f, P: +-2, 3.6e-15
                        :Wd => "ap_fixed<48, 10, AP_RND_CONV>"]         # N0, N_m, (1/n + KE) y + N_m: +-512
const SHUNT_RT_SCALE = (x = 10.0, vf = 1e-3, P = 1e-2)              # the scales of the relative limits (41, 6)
const SHUNT_RT_LIM = 5e-8                                            # C sim and board: relative to the scales
const SHUNT_RT_BOUND = 1e-15                                         # certified endpoint bound per step (41c, 2)

# the bench's segment table of case :A, :B or :C for instance d (midpoint-sampled; rounded to the C++ grids)
function shunt_rt_bench(d, case::Symbol, h::Float64)
    non, nmid, noff = round(Int, 0.1 / h), round(Int, 0.15 / h), round(Int, 0.2 / h)
    v0, w0, i0 = d["vterm"], d["omega"], d["ifd"]
    if case === :A                                   # upper limit, low speed, i_fd x 2
        a = [(v0, v0), (0.5, 0.5), (0.5, 0.5), (v0, v0)]
        o = [(w0, w0), (w0, 0.97), (0.97, w0), (w0, w0)]
        f = [(i0, i0), (i0, 2 * i0), (2 * i0, i0), (i0, i0)]
    elseif case === :B                               # lower limit, overspeed, i_fd x 0.5
        a = [(v0, v0), (1.4, 1.4), (1.4, 1.4), (v0, v0)]
        o = [(w0, w0), (w0, 1.03), (1.03, w0), (w0, w0)]
        f = [(i0, i0), (i0, 0.5 * i0), (0.5 * i0, i0), (i0, i0)]
    else
        a = [(v0, v0) for _ in 1:4]; o = [(w0, w0) for _ in 1:4]; f = [(i0, i0) for _ in 1:4]
        non = nmid = noff = 1 << 30
    end
    st = [0, non, nmid, noff]; len = [non, nmid - non, noff - nmid, 1]
    us = zeros(4, 3); du = zeros(4, 3)
    for s in 1:4, (j, v) in enumerate((a, o, f))
        rate = case === :C ? 0.0 : (v[s][2] - v[s][1]) / len[s]          # per step
        du[s, j] = _q48(rate, SHUNT_RT_I.Small)
        us[s, j] = _q48(v[s][1] + rate / 2, SHUNT_RT_I.Sig)              # the first step's midpoint value
    end
    return (; non, nmid, noff, us, du)
end

_sh_x0(d) = map(v -> _q48(v, SHUNT_RT_I.St), d["x0"])

# the inputs of step m1 + 1 (m1 steps before it), as shunt_run! forms them
function _sh_input!(u, b, m1)
    sg, s0 = m1 >= b.noff ? (4, b.noff) : m1 >= b.nmid ? (3, b.nmid) : m1 >= b.non ? (2, b.non) : (1, 0)
    for j in 1:3; u[j] = b.us[sg, j] + b.du[sg, j] * (m1 - s0); end
    return u
end

_sh_call!(k, x, b, n0, nsteps; decim = 32) =
    shunt_run!(k, x, b.us, b.du, b.non, b.nmid, b.noff, n0, nsteps, decim, zeros(2), Ref(0.0), Ref(0.0), Ref(false), Ref(false))

# the Julia run of shunt_run! (Float64) cut at `cuts`: the states after each cut and the outputs of its last step
function _shunt_rt_julia(k, x0, b, cuts)
    x = copy(x0); rec = [copy(x)]; outs = [zeros(2)]
    y = zeros(2); r = (Ref(0.0), Ref(0.0), Ref(false), Ref(false)); n = 0
    for c in cuts
        shunt_run!(k, x, b.us, b.du, b.non, b.nmid, b.noff, n, c - n, 32, y, r...)
        push!(rec, copy(x)); push!(outs, copy(y)); n = c
    end
    return rec, outs
end

# every step of a case, as shunt_run! does it: per step x before and after, u, y, md and the SE used
function _shunt_rt_trace(k, x0, b, nsteps)
    x = copy(x0); y = zeros(2); md = zeros(Int, 2); u = zeros(3)
    X = Vector{Vector{Float64}}(); Xn = similar(X); U = similar(X); Y = similar(X); M = Vector{Vector{Int}}(); SE = Float64[]
    se2 = sh_se(x[5], k.kc); se1 = sh_se(x[4], k.kc)
    for n in 1:nsteps
        _sh_input!(u, b, n - 1)
        push!(X, copy(x)); push!(U, copy(u)); push!(SE, se2)
        sn = sh_se(x[1], k.kc)
        shunt_step!(x, u, y, md, se2, k.P, k.W, k.c, k.wmax, k.wmin, k.Q, k.S, k.kc, k.kt)
        se2 = se1; se1 = sn
        push!(Xn, copy(x)); push!(Y, copy(y)); push!(M, copy(md))
    end
    return (; X, Xn, U, Y, M, SE)
end

# cuts every 0.01 s, around the events and within two steps of every mode change of the trace (41b, 3; 41d, 3)
function _shunt_rt_cuts(h, b, tr = nothing)
    nend = round(Int, 0.3 / h)
    c = vcat([round(Int, g * 0.01 / h) for g in 1:30], b.non-1:b.non+2, b.nmid-1:b.nmid+2, b.noff-1:b.noff+2)
    if tr !== nothing
        for n in 2:length(tr.M)
            tr.M[n] != tr.M[n-1] && append!(c, n-3:n+2)
        end
    end
    return sort(unique(filter(v -> 0 < v <= nend, c)))
end

# the short-call windows (41d, 3; 42, 2): (start, ends): from the start and around every mode change, calls of 1, 2, 3
# steps (mixed); a window starts where the previous one ended if they overlap
function _shunt_rt_windows(tr, nend)
    W = Tuple{Int,Vector{Int}}[]; e = Int[]; c = 0
    for L in (1, 1, 1, 2, 3, 1, 2, 3, 2, 1); c += L; push!(e, c); end
    push!(W, (0, e)); last = c
    for n in 2:nend
        tr.M[n] != tr.M[n-1] || continue
        s = max(n - 4, last); e = Int[]; c = s
        for L in (1, 2, 1, 3, 1, 1); c += L; c <= nend && push!(e, c); end
        isempty(e) || (push!(W, (s, e)); last = e[end])
    end
    return W
end

# every intermediate of shunt_step! and sh_se at (x, u, se), with the C++ type it is stored in (the range check,
# mirroring the @hls code statement by statement); returns the values the domain checks need
function _sh_intermediates!(mx::Dict{Symbol,Float64}, k, x, u, se)
    P, W, cc, wx, wn, Q, S, kc, kt = k.P, k.W, k.c, k.wmax, k.wmin, k.Q, k.S, k.kc, k.kt
    put(T, a) = (mx[T] = max(get(mx, T, 0.0), abs(a)); a)
    # sh_se of x[1] (the unit computed in this iteration)
    t1 = put(:Targ, kc[6] * x[1]); tpr = put(:Targ, t1 + kc[7])
    tp = clamp(tpr, 0.0, 17.999999); kx = ifloor(tp); f = put(:Targ, tp - kx); f8 = put(:Frac8, 256 * f)
    j = ifloor(f8); r8 = put(:Frac8, f8 - j); r = put(:Targ, r8 / 256); lc = HLS_ROMS[:SH_EXPC]
    r2 = put(:Mant, r * r); q1 = put(:Mant, lc[1] * r); q3 = put(:Mant, lc[3] * r); lo = put(:Mant, 1 + q1)
    hi = put(:Mant, lc[2] + q3); q2 = put(:Mant, r2 * hi); pl = put(:Mant, lo + q2); m = put(:Mant, HLS_ROMS[:SH_EXP2][j+1] * pl)
    put(:Sat, m * HLS_ROMS[:SH_POW2][kx+1])
    vt, om, ifd = u
    om1 = put(:Sig, om - 1); ys = x[5]
    by = put(:Sig, kc[8] * ys); sb = put(:Sig, se * by); np = put(:Sig, se + sb); sby = put(:Wd, sb * ys); n0 = put(:Wd, -sby)
    r1 = put(:Sig, kc[2] * vt); ra = put(:Sig, kc[1] - r1); cw = put(:Sig, kc[3] * om); r3 = put(:Sig, cw * x[1])
    r4 = put(:Sig, kc[3] * x[3]); ta = put(:Sig, ra - x[2]); rb = put(:Sig, r4 - r3); rr = put(:Sig, ta + rb)
    nb = sh_decide(x[2], rr, kc[4], kc[5]); fz = nb > 0 ? 1 : 0; bW, bS = 1 + 2fz, 2 + 2fz
    kK = zeros(4, 3); gB = zeros(4); hB = zeros(4, 2)
    for bk in 1:4
        aa = zeros(3)
        for i in 1:3
            l4 = put(:Inc, W[bk, i] * vt); l5 = put(:Inc, S[bk, i] * n0); lcc = put(:Inc, l4 + l5); kK[bk, i] = put(:Inc, lcc + cc[bk, i])
            aa[i] = put(:Small, put(:Small, Q[bk, i] * om1) + put(:Small, S[bk, i] * np))
        end
        eb = put(:Small, aa[1] / 2); gB[bk] = put(:Small, eb + put(:Small, eb * eb))
        for i in 1:2; hB[bk, i] = put(:Small, put(:Small, aa[i+1] + put(:Small, aa[i+1] * gB[bk])) / 2); end
    end
    kH = [put(:Inc, kK[bS, i] + wx[fz+1, i]) for i in 1:3]; kL = [put(:Inc, kK[bS, i] + wn[fz+1, i]) for i in 1:3]
    dW = zeros(3); dH = zeros(3); dL = zeros(3)
    for i in 1:3
        l = [put(:Inc, P[bW, 3i-3+jj] * x[jj]) for jj in 1:3]; dW[i] = put(:Inc, put(:Inc, l[1] + l[2]) + put(:Inc, l[3] + kK[bW, i]))
        sv = [put(:Inc, P[bS, 3i-3+jj] * x[jj]) for jj in 1:3]; sa = put(:Inc, sv[1] + sv[2])
        dH[i] = put(:Inc, sa + put(:Inc, sv[3] + kH[i])); dL[i] = put(:Inc, sa + put(:Inc, sv[3] + kL[i]))
    end
    gW, gS = gB[bW], gB[bS]; hW = [0.0, hB[bW, 1], hB[bW, 2]]; hS = [0.0, hB[bS, 1], hB[bS, 2]]
    out = Dict{Int,Any}()
    for (cand, D, hh, g) in ((0, dW, hW, gW), (1, dH, hS, gS), (2, dL, hS, gS))
        R = put(:St, x[1] + D[1] / 2); rg = put(:Inc, R * g); m2 = put(:Inc, hh[2] * R); m3 = put(:Inc, hh[3] * R)
        fq = put(:Inc, D[1] + 2rg); g2 = put(:Inc, D[2] + 2m2); g3 = put(:Inc, D[3] + 2m3)
        ym = put(:St, x[1] + fq / 2)
        out[cand] = (; ym, g2)
    end
    umax = put(:St, 2 * put(:St, kc[4] - x[2])); umin = put(:St, 2 * put(:St, kc[5] - x[2]))
    cs = sh_select(out[0].g2, out[1].g2, out[2].g2, umax, umin)
    ym = out[cs == 3 ? 0 : cs].ym
    oy = put(:Sig, om * ym); vf = put(:Pw, kt[1] * oy); npy = put(:Wd, np * ym); nm = put(:Wd, n0 + npy)
    ky = put(:Sig, kc[9] * ym); fs = put(:Wd, ky + nm); ifm = put(:Sig, kc[10] * fs); it = put(:Sig, ifd + ifm)
    put(:Pw, vf * it)
    return (; tpr, ym, om)
end

# ---------------------------------------------------------------------------------------------------------------------
# The certified per-step bound (41c, 2; 42, 1): the step's algebra (exact arithmetic) against the exact midpoint of the
# same bank
# ---------------------------------------------------------------------------------------------------------------------
# BigFloat banks of instance d (unrounded) at h
function _sh_bigbanks(d, h::BigFloat)
    [_sh_bank(d, h, fz, sat) for (fz, sat) in SHUNT_RT_BANKS]
end

# Outward-rounded interval arithmetic on 256-bit BigFloat (review 42, 1): every operation calls MPFR with an explicit
# rounding mode (down for lower ends, up for upper ends), so the enclosure holds whatever the global rounding mode and
# in threads. The inputs (Float64 run values, the 256-bit prepared vectors, the instance data) are taken as exact: the
# bound is that of the step's algebra with these coefficients against the exact midpoint of the same bank.
struct _Iv
    lo::BigFloat
    hi::BigFloat
end
const _RU = Base.MPFR.MPFRRoundUp
const _RD = Base.MPFR.MPFRRoundDown
_nb() = BigFloat(; precision = 256)
_mpadd(a, b, r) = (z = _nb(); ccall((:mpfr_add, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, b, r); z)
_mpsub(a, b, r) = (z = _nb(); ccall((:mpfr_sub, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, b, r); z)
_mpmul(a, b, r) = (z = _nb(); ccall((:mpfr_mul, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, b, r); z)
_mpdiv(a, b, r) = (z = _nb(); ccall((:mpfr_div, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, b, r); z)
_mpexp(a, r) = (z = _nb(); ccall((:mpfr_exp, Base.MPFR.libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, a, r); z)
_iv(v::Real) = (b = BigFloat(v; precision = 256); _Iv(b, b))      # exact for Float64 and 256-bit BigFloat
Base.:+(a::_Iv, b::_Iv) = _Iv(_mpadd(a.lo, b.lo, _RD), _mpadd(a.hi, b.hi, _RU))
Base.:-(a::_Iv, b::_Iv) = _Iv(_mpsub(a.lo, b.hi, _RD), _mpsub(a.hi, b.lo, _RU))
Base.:-(a::_Iv) = _Iv(-a.hi, -a.lo)
function Base.:*(a::_Iv, b::_Iv)
    ps = ((a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi))
    return _Iv(minimum(_mpmul(p..., _RD) for p in ps), maximum(_mpmul(p..., _RU) for p in ps))
end
_ivexp(a::_Iv) = _Iv(_mpexp(a.lo, _RD), _mpexp(a.hi, _RU))
_ivhalf(a::_Iv) = _Iv(a.lo / 2, a.hi / 2)                          # exact (binary)
_ivabs(a::_Iv) = a.lo >= 0 ? a : a.hi <= 0 ? -a : _Iv(zero(a.lo), max(-a.lo, a.hi))
_ivmag(a::_Iv) = _ivabs(a).hi                                     # an upper bound of |a|

# the bound of one step from x (5 Float64 values), u and the modes (nb, cs): returns an upper bound of the largest
# endpoint error of the three states (rounded up to Float64), after checking the root enclosure (41c, 2; y* = x[5], 41d)
function _sh_stepbound(d, bb, x, u, nb, cs; delta = 1e-9)
    A, B, one = _iv(d["A"]), _iv(d["B"]), _iv(1.0)
    xb = _iv.(x); vt, om = _iv(u[1]), _iv(u[2]); dl = _iv(delta)
    fz = nb > 0; bank = bb[(cs == 0 ? 1 : 2) + (fz ? 2 : 0)]
    ys = xb[5]; byv = B * ys
    se = A * _ivexp(byv)
    np = se * (one + byv); n0 = -(se * byv * ys)
    vr = _iv(cs == 1 ? d["VRMAX"] : cs == 2 ? d["VRMIN"] : 0.0)
    D = [_iv(bank.P[i, 1]) * xb[1] + _iv(bank.P[i, 2]) * xb[2] + _iv(bank.P[i, 3]) * xb[3] + _iv(bank.W[i]) * vt +
         _iv(bank.c[i]) + _iv(bank.w[i]) * vr + _iv(bank.S[i]) * n0 for i in 1:3]
    q = [_ivhalf(_iv(bank.Q[i])) for i in 1:3]; sv = [_ivhalf(_iv(bank.S[i])) for i in 1:3]
    om1 = om - one
    eps = q[1] * om1 + sv[1] * np
    R = xb[1] + _ivhalf(D[1])
    yh = R * (one + eps + eps * eps)
    min(ys.lo, yh.lo) > 0 || error("bound: y <= 0 (N', N'' monotone only on y > -2/B; the domain is y > 0)")
    # N'(y) = A e^{By} (1 + B y), N''(y) = A B e^{By} (2 + B y): increasing on y > 0; upper bounds at the upper end
    ytop = _Iv(max(ys.hi, yh.hi), max(ys.hi, yh.hi)) + dl
    N2 = (A * B * _ivexp(B * ytop) * (_iv(2.0) + B * ytop)).hi
    yt1 = _Iv(yh.hi, yh.hi) + dl
    N1 = (A * _ivexp(B * yt1) * (one + B * yt1)).hi
    dy = yh - ys; dy2 = _ivhalf(dy * dy).hi
    e3 = _ivmag(eps)
    rho = _mpadd(_mpmul(_ivmag(R), _mpmul(e3, _mpmul(e3, e3, _RU), _RU), _RU), _mpmul(_mpmul(_ivmag(sv[1]), N2, _RU), dy2, _RU), _RU)
    gmin = _mpsub(_mpsub(BigFloat(1; precision = 256), _ivmag(om1 * q[1]), _RD), _mpmul(_ivmag(sv[1]), N1, _RU), _RD)
    gmin > 0 || error("bound: g_min <= 0")
    dn = _mpdiv(rho, gmin, _RU)
    dn <= dl.lo || error("bound: rho / g_min = $dn > delta")
    e = [_mpmul(BigFloat(2; precision = 256), dn, _RU)]
    for kk in 2:3
        t1 = _mpmul(_mpadd(_ivmag(om1 * q[kk]), _mpmul(_ivmag(sv[kk]), N1, _RU), _RU), dn, _RU)
        t2 = _mpmul(_mpmul(_ivmag(sv[kk]), N2, _RU), dy2, _RU)
        push!(e, _mpmul(BigFloat(2; precision = 256), _mpadd(t1, t2, _RU), _RU))
    end
    return Float64(maximum(e), RoundUp)
end

# ---------------------------------------------------------------------------------------------------------------------
# The exp unit in the C++ arithmetic (Julia mirror with every assignment rounded to its type, 41b, 2)
# ---------------------------------------------------------------------------------------------------------------------
function _sh_exp_q(tp::Float64)
    I_ = SHUNT_RT_I; q(v, T) = _q48(v, getfield(I_, T)); lc = HLS_ROMS[:SH_EXPC]; ex2 = HLS_ROMS[:SH_EXP2]
    tp = q(tp, :Targ); k = floor(Int, tp); f = q(tp - k, :Targ); f8 = q(256 * f, :Frac8); j = floor(Int, f8)
    r8 = q(f8 - j, :Frac8); r = q(r8 / 256, :Targ)
    r2 = q(r * r, :Mant); q1 = q(lc[1] * r, :Mant); q3 = q(lc[3] * r, :Mant); lo = q(1 + q1, :Mant)
    hi = q(lc[2] + q3, :Mant); q2 = q(r2 * hi, :Mant); pl = q(lo + q2, :Mant); m = q(ex2[j+1] * pl, :Mant)
    return m, k, tp
end

# the sweep points of t' (41b, 2): every integer and ROM-bin boundary with its grid neighbours, the guard ends, random
function _sh_exp_points(; nrand = 1_000_000)
    g = 2.0^-42; pts = Float64[]
    for k in 0:17, j in 0:255
        t = k + j / 256
        append!(pts, (t - g, t, t + g))
    end
    append!(pts, (18 - g, 18 - 2g))
    rng = Random.MersenneTwister(41)
    append!(pts, 18 .* rand(rng, nrand))
    return filter(t -> 0 <= t < 18, pts)
end

function _sh_exp_sweep()
    wq = 0.0; wf = 0.0; at = 0.0
    for t in _sh_exp_points()
        m, k, tq = _sh_exp_q(t)
        ex = BigFloat(2)^(BigFloat(tq) - 16)
        e = Float64(abs(BigFloat(m) * BigFloat(2)^(k - 16) / ex - 1))
        e > wq && (wq = e; at = tq)
        mk = [0.0]; kk = [0]; sh_exp!(tq, mk, kk)                    # the @hls code itself in Float64
        wf = max(wf, Float64(abs(BigFloat(mk[1]) * BigFloat(2)^(kk[1] - 16) / ex - 1)))
    end
    return wq, wf, at
end

# ---------------------------------------------------------------------------------------------------------------------
# The corner suite (41.5, 41b, 4): one step from constructed states; the decision and selection with exact inputs
# ---------------------------------------------------------------------------------------------------------------------
# (name, x (4), u, expected (nb, cs)) for instance d, k at h
function shunt_rt_corners(d, k, h)
    q(v, T) = _q48(v, getfield(SHUNT_RT_I, T))
    y0 = d["y0"]; vmax, vmin = d["VRMAX"], d["VRMIN"]; TA = d["TA"]; KA = d["KA"]; Vref = d["Vref"]
    u(vt) = [q(vt, :Sig), 1.0, q(d["ifd"], :Sig)]
    rate(vt, xa) = (KA * (Vref - vt) - xa) / TA                # xa' at y = xf (omega = 1)
    cs = Tuple{String,Vector{Float64},Vector{Float64},Tuple{Int,Int}}[]
    X(xa) = [q(y0, :St), q(xa, :St), q(y0, :St), q(y0, :St), q(y0, :St)]
    push!(cs, ("free W", X(d["x"][2]), u(d["vterm"]), (0, 0)))
    # histories with y_n, y_{n-1}, y_{n-2} distinct, differences ~ h |y'| (|y'| = 4 pu/s): rising, falling, turning (41d, 2)
    dh = 4 * h
    for (nm, y1, y2) in (("rising", y0 - dh, y0 - 2dh), ("falling", y0 + dh, y0 + 2dh), ("turning", y0 + dh, y0 - dh))
        push!(cs, ("free W, $nm history", [q(y0, :St), q(d["x"][2], :St), q(y0, :St), q(y1, :St), q(y2, :St)], u(d["vterm"]), (0, 0)))
        r = rate(0.5, vmax)
        push!(cs, ("free H crossing, $nm history", [q(y0, :St), q(vmax - 0.25 * h * r, :St), q(y0, :St), q(y1, :St), q(y2, :St)], u(0.5), (0, 1)))
    end
    r = rate(0.5, vmax); push!(cs, ("free H crossing", X(vmax - 0.25 * h * r), u(0.5), (0, 1)))
    r = rate(1.4, vmin); push!(cs, ("free L crossing", X(vmin + 0.25 * h * abs(r)), u(1.4), (0, 2)))
    # released inward from outside: xa_mid ~ xa_n - (h/2) |rate|, so 1.25 (h/2) |rate| outside stays clamped at the
    # midpoint and 0.75 (h/2) |rate| crosses back in
    r = rate(1.4, vmax); push!(cs, ("release from above, H", X(vmax + 1.25 * h / 2 * abs(r)), u(1.4), (0, 1)))
    push!(cs, ("release from above, W", X(vmax + 0.75 * h / 2 * abs(r)), u(1.4), (0, 0)))
    r = rate(0.5, vmin); push!(cs, ("release from below, L", X(vmin - 1.25 * h / 2 * r), u(0.5), (0, 2)))
    push!(cs, ("release from below, W", X(vmin - 0.75 * h / 2 * r), u(0.5), (0, 0)))
    push!(cs, ("frozen H", X(vmax + 1e-3), u(0.5), (1, 1)))
    push!(cs, ("frozen L", X(vmin - 1e-3), u(1.4), (2, 2)))
    push!(cs, ("frozen at VRMAX, W", X(vmax), u(0.5), (1, 0)))
    push!(cs, ("frozen at VRMIN, W", X(vmin), u(1.4), (2, 0)))
    return cs
end

# one corner step in Julia: (x+, y, md)
function _sh_corner_step(k, x, u)
    xx = copy(x); y = zeros(2); md = zeros(Int, 2)
    se = sh_se(xx[5], k.kc)
    shunt_step!(xx, u, y, md, se, k.P, k.W, k.c, k.wmax, k.wmin, k.Q, k.S, k.kc, k.kt)
    return xx, y, md
end

# the exact-input tests of sh_decide and sh_select: (args, expected)
const SHUNT_RT_DECIDE = [((1.0, 1.0, 1.0, -1.0), 1), ((1.0, -1.0, 1.0, -1.0), 0), ((1.0, 0.0, 1.0, -1.0), 0),
                         ((-1.0, -1.0, 1.0, -1.0), 2), ((-1.0, 1.0, 1.0, -1.0), 0), ((-1.0, 0.0, 1.0, -1.0), 0),
                         ((0.5, 1.0, 1.0, -1.0), 0), ((1.5, 1.0, 1.0, -1.0), 1), ((-1.5, -1.0, 1.0, -1.0), 2)]
const SHUNT_RT_SELECT = (u = 2.0^-7; [((u, u, u, u, -u), 0), ((-u, -u, -u, u, -u), 0),       # (g_W, g_H, g_L, umax, umin)
                         ((1.25u, 1.25u, 1.25u, u, -u), 1), ((-1.25u, -1.25u, -1.25u, u, -u), 2),
                         ((1.25u, 0.75u, 0.75u, u, -u), 3), ((0.0, 2u, -2u, u, -u), 0), ((u, u, u, u, u), 0)])

"""
    shunt_rt_export(; h = SHUNT_RT_H, dir = src/rt/exciters/CPP/shunt, bound = true)

Prepares the six instances for h (BigFloat), runs the Julia checks fixed in the idea gate (41, 41b, 41c: ranges, domain,
occupancy, corners, candidate consistency, rounding, energy, certified bound, exp sweep, continuous = segmented) and
writes the C++, the ROMs, the coefficients, the C simulation testbench, the board program, the ILA reference and
flow.cfg. The check against the package's rows and the refinement: scripts/realtime/shunt_checks.jl.
"""
function shunt_rt_export(; h = SHUNT_RT_H, dir = joinpath(SHUNT_RT_DIR, "CPP", "shunt"), bound = true)
    occursin("II=$(SHUNT_RT_STEP_CYCLES)\"", string(HLS_DEFS[:shunt_run!])) ||
        error("the PIPELINE II in shunt_run! differs from SHUNT_RT_STEP_CYCLES = $SHUNT_RT_STEP_CYCLES")
    hb = setprecision(() -> parse(BigFloat, h), BigFloat, 256); hh = Float64(hb)
    ni = shunt_rt_ninstances()
    ds = [shunt_rt_data(i) for i in 1:ni]
    ks = [setprecision(() -> shunt_rt_coefstruct(d, hb), BigFloat, 256) for d in ds]
    ku = [setprecision(() -> shunt_rt_coefstruct(d, hb; grid = false), BigFloat, 256) for d in ds]
    bbs = [setprecision(() -> _sh_bigbanks(d, hb), BigFloat, 256) for d in ds]
    x0s = [_sh_x0(d) for d in ds]
    nend = round(Int, 0.3 / hh)
    S_ = SHUNT_RT_SCALE
    println("h = $hh s; $ni instances: ", join((d["machine"] for d in ds), ", "))
    I_ = SHUNT_RT_I
    half(T) = 2.0^(getfield(I_, T) - 2)                              # half of the type's range
    for (i, k) in enumerate(ks), (fld, T) in SHUNT_RT_FIELDS
        m = maximum(abs, getfield(k, fld))
        m < half(T) || error("instance $i: coefficient $fld reaches $m (half range of $T: $(half(T)))")
    end
    # the exp unit (check e)
    wq, wf, at = _sh_exp_sweep()
    println("exp unit: max relative error $(round(wq; sigdigits = 3)) in the C++ arithmetic (Julia mirror, at t' = $at), ",
            "$(round(wf; sigdigits = 3)) in Float64 (limit 1e-10)")
    (wq <= 1e-10 && wf <= 1e-10) || error("exp unit error $wq / $wf")
    # decision and selection with exact inputs (41b, 4)
    for (a, e) in SHUNT_RT_DECIDE; sh_decide(a...) == e || error("sh_decide$a != $e"); end
    for (a, e) in SHUNT_RT_SELECT; sh_select(a...) == e || error("sh_select$a != $e"); end
    # the corner suite: every reachable (N, C) pair for every instance, with the certified bound
    for (i, (d, k, kuu, bb)) in enumerate(zip(ds, ks, ku, bbs))
        seen = Set{Tuple{Int,Int}}()
        for (name, x, u, want) in shunt_rt_corners(d, k, hh)
            all(v -> 0 < v <= 5, x[[1, 4, 5]]) || error("$(d["machine"]) corner '$name': y or history outside (0, 5]")
            all(v -> 0 <= d["B"] * v / log(2) + log2(d["A"]) + 16 < 18, x[[1, 4, 5]]) || error("$(d["machine"]) corner '$name': history outside the exp guard")
            xx, _, md = _sh_corner_step(k, x, u)
            (xx[4] == x[1] && xx[5] == x[4]) || error("$(d["machine"]) corner '$name': the history did not shift exactly")
            _, _, mu = _sh_corner_step(kuu, x, u)
            (md[1], md[2]) == want == (mu[1], mu[2]) || error("$(d["machine"]) corner '$name': modes $md (unrounded $mu), expected $want")
            push!(seen, want)
            bound && (e = setprecision(() -> _sh_stepbound(d, bb, x, u, want...), BigFloat, 256);
                      e <= SHUNT_RT_BOUND || error("$(d["machine"]) corner '$name': bound $e > $SHUNT_RT_BOUND"))
        end
        for p in ((0, 0), (0, 1), (0, 2), (1, 1), (2, 2), (1, 0), (2, 0))
            p in seen || error("$(d["machine"]): corner pair $p missing")
        end
    end
    println("corner suite: all (N, C) pairs for every instance, modes as expected (rounded and unrounded)")
    for (ii, (d, k, ku_, x0, bb)) in enumerate(zip(ds, ks, ku, x0s, bbs)), case in (:A, :B)
        b = shunt_rt_bench(d, case, hh)
        tr = _shunt_rt_trace(k, x0, b, nend)
        tu = _shunt_rt_trace(ku_, x0, b, nend)
        mx = Dict{Symbol,Float64}()
        up!(s, v) = (mx[s] = max(get(mx, s, 0.0), abs(v)))
        Dmax = big(0.0); Dsum = big(0.0); occ = zeros(Int, 3, 4); bmax = 0.0
        Wf(yv) = _sh_energy(d, yv); W0 = Wf(big(x0[1]))
        tmin, tmax = Inf, -Inf; ymin = Inf; ymax = -Inf
        for n in 1:nend
            x, xn, u, y, md = tr.X[n], tr.Xn[n], tr.U[n], tr.Y[n], tr.M[n]
            foreach(v -> up!(:St, v), xn); foreach(v -> up!(:Inc, v), xn[1:3] - x[1:3]); foreach(v -> up!(:Sig, v), u)
            up!(:Pw, y[1]); up!(:Pw, y[2])
            occ[md[1]+1, md[2]+1] += 1
            iv = _sh_intermediates!(mx, k, x, u, tr.SE[n])
            tmin = min(tmin, iv.tpr); tmax = max(tmax, iv.tpr); ymin = min(ymin, xn[1]); ymax = max(ymax, xn[1])
            (0 < xn[1] <= 5) || error("$(d["machine"]) case $case step $n: y = $(xn[1]) outside (0, 5]")
            (0 <= iv.tpr < 18) || error("$(d["machine"]) case $case step $n: t' = $(iv.tpr) outside the guard [0, 18)")
            iv.om >= 0.5 || error("$(d["machine"]) case $case step $n: omega < 0.5")
            md[2] == 3 && error("$(d["machine"]) case $case step $n: no consistent clamp candidate")
            # energy (41, 6c; 41b, 5): D = W(y+) - W(y) - h i_f (v_f + b VR - rF i_f) at the midpoint, BigFloat
            ym = (big(x[1]) + big(xn[1])) / 2
            vr = md[2] == 1 ? big(d["VRMAX"]) : md[2] == 2 ? big(d["VRMIN"]) : clamp((big(x[2]) + big(xn[2])) / 2, big(d["VRMIN"]), big(d["VRMAX"]))
            i_f = _sh_if(d, ym); vf = big(d["b"]) / big(d["n"]) * big(u[2]) * ym
            D = Wf(big(xn[1])) - Wf(big(x[1])) - big(hh) * i_f * (vf + big(d["b"]) * vr - big(d["rF"]) * i_f)
            Dmax = max(Dmax, abs(D)); Dsum += abs(D)
        end
        # the certified bound on the unrounded run (41c, 2), every step, in threads (BigFloat at its default 256 bits)
        if bound
            precision(BigFloat) == 256 || error("BigFloat precision $(precision(BigFloat)) != 256")
            bs = zeros(nend)
            Threads.@threads for n in 1:nend
                bs[n] = _sh_stepbound(d, bb, tu.X[n], tu.U[n], tu.M[n][1], tu.M[n][2])
            end
            bmax, nb_ = findmax(bs)
            bmax <= SHUNT_RT_BOUND || error("$(d["machine"]) case $case step $nb_: certified bound $bmax > $SHUNT_RT_BOUND")
        end
        for (T, v) in mx
            v < half(T) || error("$(d["machine"]) case $case: a $T value reaches $v (half range $(half(T)))")
        end
        need = case === :A ? [(0, 0), (1, 1)] : [(0, 0), (2, 2)]
        for (a, c) in need
            occ[a+1, c+1] > 0 || error("$(d["machine"]) case $case: (N, C) = ($a, $c) never occurs")
        end
        db = maximum(n -> max(maximum(abs, tr.Xn[n] - tu.Xn[n]) / S_.x, abs(tr.Y[n][1] - tu.Y[n][1]) / S_.vf,
                              abs(tr.Y[n][2] - tu.Y[n][2]) / S_.P), 1:nend)
        db <= 5e-8 || error("$(d["machine"]) case $case: rounding moves the run by $db (relative)")
        (Dmax <= 1e-12 * W0 && Dsum <= 1e-9 * W0) || error("$(d["machine"]) case $case: energy defect max $Dmax, sum $Dsum (W0 $W0)")
        println("$(d["machine"]) case $case: occupancy (N, C) free W/H/L $(occ[1, 1:3]), frozen-high W/H $(occ[2, 1:2]), ",
                "frozen-low W/L $(occ[3, [1, 3]]); y [$(round(ymin; sigdigits = 5)), $(round(ymax; sigdigits = 5))], ",
                "t' [$(round(tmin; sigdigits = 5)), $(round(tmax; sigdigits = 5))]; rounding $(round(db; sigdigits = 3)) (limit 5e-8); ",
                "energy max |D| $(round(Float64(Dmax / W0); sigdigits = 3)) W0, sum $(round(Float64(Dsum / W0); sigdigits = 3)) W0 ",
                "(limits 1e-12, 1e-9); certified bound $(bound ? round(bmax; sigdigits = 3) : "skipped") (limit 1e-15)")
    end
    # continuous = segmented, every instance, case A and B, cuts at the mode changes (41b, 3)
    for (i, (d, k, x0)) in enumerate(zip(ds, ks, x0s)), case in (:A, :B)
        b = shunt_rt_bench(d, case, hh); tr = _shunt_rt_trace(k, x0, b, nend); cuts = _shunt_rt_cuts(hh, b, tr)
        rec, _ = _shunt_rt_julia(k, x0, b, cuts)
        xc = copy(x0); _sh_call!(k, xc, b, 0, cuts[end])
        (xc == rec[end] && xc == tr.Xn[cuts[end]]) || error("$(d["machine"]) case $case: the continuous run differs from the segmented one")
    end
    println("continuous = segmented: bit for bit, every instance in A and B (cuts at every mode change)")
    # short calls (41d, 3): from the start and around every mode change, calls of 1, 2, 3 steps (mixed), each compared with
    # the continuous trace in all five states and both outputs, bit for bit; then a reload: instance 2 after instance 1
    nshort = 0
    for (i, (d, k, x0)) in enumerate(zip(ds, ks, x0s)), case in (:A, :B)
        b = shunt_rt_bench(d, case, hh); tr = _shunt_rt_trace(k, x0, b, nend)
        x = copy(x0); prev = 0; y = zeros(2)
        for (st, ends) in _shunt_rt_windows(tr, nend)
            if st > prev                                 # one long call to the window's start
                shunt_run!(k, x, b.us, b.du, b.non, b.nmid, b.noff, prev, st - prev, 32, y, Ref(0.0), Ref(0.0), Ref(false), Ref(false))
                (x == tr.Xn[st] && y == tr.Y[st]) || error("$(d["machine"]) case $case: a call ending at step $st differs from the continuous run")
                prev = st
            end
            for e in ends
                shunt_run!(k, x, b.us, b.du, b.non, b.nmid, b.noff, prev, e - prev, 32, y, Ref(0.0), Ref(0.0), Ref(false), Ref(false))
                (x == tr.Xn[e] && y == tr.Y[e]) || error("$(d["machine"]) case $case: a call ending at step $e differs from the continuous run")
                prev = e; nshort += 1
            end
        end
    end
    xr = copy(x0s[1]); _sh_call!(ks[1], xr, shunt_rt_bench(ds[1], :A, hh), 0, nend)       # a nontrivial run, then a reload
    x2 = copy(x0s[2]); b2 = shunt_rt_bench(ds[2], :A, hh); _sh_call!(ks[2], x2, b2, 0, 1000)
    x2 == _shunt_rt_trace(ks[2], x0s[2], b2, 1000).Xn[end] || error("reload after a run differs from a fresh run")
    println("short calls: $nshort calls of 1-3 steps (start and every mode change) = continuous, five states and outputs bit for bit; reload ok")
    # the C++
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "shunt_rom.h"), [:SH_EXP2, :SH_EXPC, :SH_POW2]; header = "the exp unit's tables (shunt.jl section 3)")
    hls_write(joinpath(dir, "shunt.cpp"), [:ShuntCoefs, :sh_exp!, :sh_se, :sh_decide, :sh_select, :shunt_step!, :shunt_run!];
              types = SHUNT_RT_TYPES, local_includes = ["rt_util.h", "shunt_rom.h"],
              header = "the DC shunt exciter (saturated PHShuntExciter), one run of steps of h = $hh s; from src/rt/exciters/shunt.jl")
    p = joinpath(dir, "shunt_coefs.h")
    open(p, "w") do io
        println(io, "// shunt_coefs.h: written by PHPS.RT.shunt_rt_export; do not edit\n// h = $hh s; instances of src/rt/exciters/data/ieee39_shunt.toml")
        for (i, (d, k)) in enumerate(zip(ds, ks))
            println(io, "// $i: $(d["name"]) on $(d["machine"])")
            println(io, "const ShuntCoefs SHUNT_COEFS_$i = ", _cinit(k), ";")
        end
    end
    println("written $p")
    shunt_rt_testbench(joinpath(dir, "shunt_tb.cpp"), ds, ks, x0s; h = hh)
    shunt_rt_board_script(joinpath(dir, "shunt_board.py"), ds, ks, x0s; h = hh)
    shunt_rt_ilaref(joinpath(dir, "shunt_ila_ref.csv"), ds, ks, x0s; h = hh)
    flow_config(joinpath(dir, "flow.cfg"); model = "shunt", top = "shunt_run", clock_ns = SHUNT_RT_CLOCK_NS,
                ip_version = "1.0", ip_name = "DC shunt exciter real-time step",
                probes = ["xa_out" => 48, "vf_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                plot = ["xa_out" => (48 - I_.St, "xa [pu]"), "vf_out" => (48 - I_.Pw, "v_f [pu]")],
                sample_dt = SHUNT_RT_DECIM * hh, board_args = "--no-download",
                impl_strategy = "Performance_ExplorePostRoutePhysOpt",   # Vivado 1: WNS -0.058 ns on a shared multiplier's input mux
                step_cycles = SHUNT_RT_STEP_CYCLES, ila_ref = "shunt_ila_ref.csv",
                ila_check = ["xa_out:$(48 - I_.St):$(SHUNT_RT_LIM * SHUNT_RT_SCALE.x)",
                             "vf_out:$(48 - I_.Pw):$(SHUNT_RT_LIM * SHUNT_RT_SCALE.vf)"], strict = true)
end

# the field energy W(phi_f) = integral_0^phi i_f dphi as a function of y = c phi (package `_exc_energy`), BigFloat
function _sh_energy(d, y::BigFloat)
    n, KE, A, TE, rF, c, B = big(d["n"]), big(d["KE"]), big(d["A"]), big(d["TE"]), big(d["rF"]), big(d["c"]), big(d["B"])
    phi = y / c; z = B * y
    mom = iszero(z) ? big(0.5) : (z * exp(z) - expm1(z)) / z^2
    return ((1 + n * KE) / 2 + n * A * mom) * phi^2 / (n * TE * rF)
end
# the field current at y (exact saturation), BigFloat
_sh_if(d, y::BigFloat) = ((1 / big(d["n"]) + big(d["KE"])) * y + big(d["A"]) * y * exp(big(d["B"]) * y)) / (big(d["c"]) * big(d["TE"]) * big(d["rF"]))

# the board program's run order: (instance, case), then case C
const SHUNT_RT_BOARD_RUNS = ((1, :A), (1, :B), (2, :A), (3, :A), (4, :A), (5, :A), (6, :A))
const SHUNT_RT_DECIM = 96                            # ILA: one sample per 96 steps

# what the ILA records (as _bx_ila_call!): for each sample the values written at the start of this iteration and of
# the 7 before it (lags 0..7); the flow picks each probe's lag from the schedule
function _sh_ila_call!(rec, k, x, b, n0, nsteps, decim, nmax)
    pxa = x[2]; pvf = 0.0; psa = false; cnt = 0; y = zeros(2); md = zeros(Int, 2); u = zeros(3)
    ha = fill(NaN, 8); hv = fill(NaN, 8)
    se2 = sh_se(x[5], k.kc); se1 = sh_se(x[4], k.kc)
    for i in 1:nsteps
        ha = vcat(pxa, ha[1:7]); hv = vcat(pvf, hv[1:7])
        psa && length(rec) < nmax && push!(rec, (copy(ha), copy(hv), n0 + i - 1))
        _sh_input!(u, b, n0 + i - 1)
        sn = sh_se(x[1], k.kc)
        shunt_step!(x, u, y, md, se2, k.P, k.W, k.c, k.wmax, k.wmin, k.Q, k.S, k.kc, k.kt)
        se2 = se1; se1 = sn
        cnt += 1
        psa = cnt >= decim; psa && (cnt = 0)
        pxa = x[2]; pvf = y[1]
    end
    return x
end

function shunt_rt_ilaref(path, ds, ks, x0s; h, decim = SHUNT_RT_DECIM, nmax = 16384)
    rec = Tuple{Vector{Float64},Vector{Float64},Int}[]; tag = String[]
    for (i, case) in SHUNT_RT_BOARD_RUNS
        b = shunt_rt_bench(ds[i], case, h); cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        x = copy(x0s[i]); n = 0
        for c in cuts
            l0 = length(rec)
            xr = copy(x)
            _sh_call!(ks[i], xr, b, n, c - n; decim)
            _sh_ila_call!(rec, ks[i], x, b, n, c - n, decim, nmax)
            x == xr || error("the ILA mirror differs from shunt_run! (instance $i, case $case)")
            append!(tag, fill("$i$case", length(rec) - l0)); n = c
            length(rec) >= nmax && break
        end
        length(rec) >= nmax && break
    end
    open(path, "w") do io
        println(io, "k,run,step,", join(("xa_out_$L" for L in 0:7), ","), ",", join(("vf_out_$L" for L in 0:7), ","))
        for (j, (a, v, st)) in enumerate(rec)
            println(io, "$j,$(tag[j]),$st,", join(map(repr, a), ","), ",", join(map(repr, v), ","))
        end
    end
    println("written $path ($(length(rec)) samples, decim $decim)")
    return path
end

_sh_f(v) = repr(Float64(v))
_sh_arr(a::AbstractMatrix) = "{" * join(("{" * join(map(_sh_f, a[i, :]), ", ") * "}" for i in 1:size(a, 1)), ", ") * "}"
_sh_vec(a) = "{" * join(map(_sh_f, a), ", ") * "}"

# the C simulation: every instance in A and B through shunt_run!, cut at every 0.01 s, the events and every mode change;
# continuous = segmented (bit for bit) per case; the exp sweep, the decision and selection tests and the corner suite in the
# C++ types. Limit 5e-8 relative to the scales; nonzero exit on failure.
function shunt_rt_testbench(path, ds, ks, x0s; h)
    S_ = SHUNT_RT_SCALE
    blocks = String[]; calls = String[]; corners = String[]
    for (i, (d, k, x0)) in enumerate(zip(ds, ks, x0s)), case in (:A, :B)
        b = shunt_rt_bench(d, case, h); nend = round(Int, 0.3 / h)
        tr = _shunt_rt_trace(k, x0, b, nend); cuts = _shunt_rt_cuts(h, b, tr)
        rec, outs = _shunt_rt_julia(k, x0, b, cuts)
        tag = "I$(i)$(case)"
        W = _shunt_rt_windows(tr, nend)
        wends = reduce(vcat, (e for (_, e) in W))
        push!(blocks, """
        // instance $i ($(d["machine"])), case $case
        const int NC_$tag = $(length(cuts)), ON_$tag = $(b.non), MID_$tag = $(b.nmid), OFF_$tag = $(b.noff);
        const int CUTS_$tag[NC_$tag] = {$(join(cuts, ", "))};
        const double US_$tag[4][3] = $(_sh_arr(b.us));
        const double DU_$tag[4][3] = $(_sh_arr(b.du));
        const double X0_$tag[5] = $(_sh_vec(x0));
        const double J_$tag[NC_$tag][7] = {$(join((_sh_vec(vcat(rec[g+1][1:5], outs[g+1])) for g in eachindex(cuts)), ",\n    "))};
        const int NW_$tag = $(length(W));
        const int WS_$tag[NW_$tag] = {$(join((st for (st, _) in W), ", "))};
        const int WK_$tag[NW_$tag] = {$(join((length(e) for (_, e) in W), ", "))};
        const int WE_$tag[$(length(wends))] = {$(join(wends, ", "))};
        const double WJ_$tag[$(length(wends))][7] = {$(join((_sh_vec(vcat(tr.Xn[e], tr.Y[e])) for e in wends), ",\n    "))};
        """)
        push!(calls, "    w = fmax(w, run_case(\"$(d["machine"]) $case\", SHUNT_COEFS_$i, X0_$tag, US_$tag, DU_$tag, ON_$tag, MID_$tag, OFF_$tag, NC_$tag, CUTS_$tag, J_$tag, &segok));\n" *
                     "    w = fmax(w, run_windows(\"$(d["machine"]) $case\", SHUNT_COEFS_$i, X0_$tag, US_$tag, DU_$tag, ON_$tag, MID_$tag, OFF_$tag, NW_$tag, WS_$tag, WK_$tag, WE_$tag, WJ_$tag, &segok));")
    end
    for (i, (d, k)) in enumerate(zip(ds, ks))
        for (name, x, u, want) in shunt_rt_corners(d, k, h)
            xx, y, md = _sh_corner_step(k, x, u)
            push!(corners, "    {$i, \"$(d["machine"]): $name\", $(_sh_vec(x)), $(_sh_vec(u)), $(_sh_vec(vcat(xx, y))), $(want[1]), $(want[2])},")
        end
    end
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.shunt_rt_testbench (src/rt/exciters/shunt.jl); do not edit
        // C simulation: every instance in bench cases A (vterm 0.5 pu, omega to 0.97, i_fd x 2, 0.1-0.2 s) and B (vterm
        // 1.4 pu, omega to 1.03, i_fd x 0.5), h = $h s, cut every 0.01 s, at the events and around every mode change; the
        // C++ states and outputs next to the same @hls code in Julia (Float64). Limit (fixed before results, 41): $(SHUNT_RT_LIM)
        // relative to the scales x $(S_.x), v_f $(S_.vf), P $(S_.P). Continuous = segmented per case (bit for bit). The exp
        // unit swept in the C++ types (limit 1e-10 relative), sh_decide / sh_select with exact inputs, the corner suite
        // (modes equal to Julia's, states and outputs within the limit). Returns 1 on failure.
        #include <cstdio>
        #include <cmath>
        #include <random>
        #include "shunt.h"
        #include "shunt_coefs.h"

        $(join(blocks, "\n"))
        const ShuntCoefs *COEFS[$(length(ds))] = {$(join(("&SHUNT_COEFS_$i" for i in eachindex(ds)), ", "))};

        double run_case(const char *name, const ShuntCoefs &k, const double *x0, const double us0[4][3], const double du0[4][3],
                        int on, int mid, int off, int nc, const int *cuts, const double J[][7], int *segok) {
            St x[5], xc[5]; Sig us[4][3]; Small du[4][3]; Pw y[2]; St xa; Pw vf; bool st, sa;
            for (int i = 0; i < 5; i++) { x[i] = x0[i]; xc[i] = x0[i]; }
            for (int s = 0; s < 4; s++) for (int j = 0; j < 3; j++) { us[s][j] = us0[s][j]; du[s][j] = du0[s][j]; }
            double w = 0; int prev = 0;
            for (int g = 0; g < nc; g++) {
                shunt_run(k, x, us, du, on, mid, off, prev, cuts[g] - prev, 32, y, xa, vf, st, sa);
                prev = cuts[g];
                double e = 0;
                for (int i = 0; i < 5; i++) e = fmax(e, fabs(x[i].to_double() - J[g][i]) / $(S_.x));
                e = fmax(e, fabs(y[0].to_double() - J[g][5]) / $(S_.vf));
                e = fmax(e, fabs(y[1].to_double() - J[g][6]) / $(S_.P));
                w = fmax(w, e);
            }
            Pw ys[2] = {y[0], y[1]};                     // the segmented run's outputs (review 42, 2)
            shunt_run(k, xc, us, du, on, mid, off, 0, cuts[nc - 1], 32, y, xa, vf, st, sa);
            int same = (y[0] == ys[0]) && (y[1] == ys[1]);
            for (int i = 0; i < 5; i++) same = same && (x[i] == xc[i]);
            if (!same) *segok = 0;
            printf("%-16s %4d checkpoints: y %.10f xa %.10f xf %.10f v_f %.6e P %.6e, max relative difference %.3e, segmented %s\\n",
                   name, nc, x[0].to_double(), x[1].to_double(), x[2].to_double(), y[0].to_double(), y[1].to_double(), w,
                   same ? "= continuous" : "DIFFERS");
            return w;
        }

        // short calls (41d, 3; 42, 2): per window, from its start, calls of 1-3 steps; at every end the state and outputs
        // against Julia's continuous run (limit); at the window's end the short-call result against ONE call over the
        // window from the same start, bit for bit in all five states and both outputs
        double run_windows(const char *name, const ShuntCoefs &k, const double *x0, const double us0[4][3], const double du0[4][3],
                           int on, int mid, int off, int nw, const int *ws, const int *wk, const int *we, const double WJ[][7], int *segok) {
            St x[5], xw[5]; Sig us[4][3]; Small du[4][3]; Pw y[2], yw[2]; St xa; Pw vf; bool st, sa;
            for (int i = 0; i < 5; i++) x[i] = x0[i];
            for (int s = 0; s < 4; s++) for (int j = 0; j < 3; j++) { us[s][j] = us0[s][j]; du[s][j] = du0[s][j]; }
            double w = 0; int prev = 0, p = 0, same = 1, calls = 0;
            for (int g = 0; g < nw; g++) {
                if (ws[g] > prev) { shunt_run(k, x, us, du, on, mid, off, prev, ws[g] - prev, 32, y, xa, vf, st, sa); prev = ws[g]; }
                for (int i = 0; i < 5; i++) xw[i] = x[i];
                int s0 = prev;
                for (int c = 0; c < wk[g]; c++, p++) {
                    shunt_run(k, x, us, du, on, mid, off, prev, we[p] - prev, 32, y, xa, vf, st, sa);
                    prev = we[p]; calls++;
                    double e = 0;
                    for (int i = 0; i < 5; i++) e = fmax(e, fabs(x[i].to_double() - WJ[p][i]) / $(S_.x));
                    e = fmax(e, fabs(y[0].to_double() - WJ[p][5]) / $(S_.vf));
                    e = fmax(e, fabs(y[1].to_double() - WJ[p][6]) / $(S_.P));
                    w = fmax(w, e);
                }
                shunt_run(k, xw, us, du, on, mid, off, s0, prev - s0, 32, yw, xa, vf, st, sa);
                int ok = (y[0] == yw[0]) && (y[1] == yw[1]);
                for (int i = 0; i < 5; i++) ok = ok && (x[i] == xw[i]);
                same = same && ok;
            }
            if (!same) *segok = 0;
            printf("%-16s %4d short calls in %3d windows: max relative difference %.3e, one call over each window %s\\n", name, calls, nw, w,
                   same ? "= short calls (bit for bit)" : "DIFFERS");
            return w;
        }

        // the exp unit in the C++ types: M 2^(k-16) against 2^(t'-16) of the quantized t'; every integer and ROM-bin
        // boundary with its grid neighbours, the guard end, 10^6 random points; limit 1e-10 relative
        double exp_sweep() {
            double w = 0, wt = 0; const double g = ldexp(1.0, -42);
            auto one = [&](double tv) {
                if (tv < 0 || tv >= 18) return;
                Targ t = tv; Mant mk[1]; int kk[1];
                sh_exp(t, mk, kk);
                double e = fabs(mk[0].to_double() * ldexp(1.0, kk[0] - 16) / exp2(t.to_double() - 16) - 1);
                if (e > w) { w = e; wt = t.to_double(); }
            };
            for (int k = 0; k < 18; k++) for (int j = 0; j < 256; j++) { double t = k + j / 256.0; one(t - g); one(t); one(t + g); }
            one(18 - g); one(18 - 2 * g);
            std::mt19937_64 rng(41); std::uniform_real_distribution<double> U(0, 18);
            for (int i = 0; i < 1000000; i++) one(U(rng));
            printf("exp unit sweep: max relative error %.3e at t' = %.12f (limit 1e-10)\\n", w, wt);
            return w;
        }

        struct Corner { int inst; const char *name; double x[5]; double u[3]; double ref[7]; int nb, cs; };
        const Corner CORNERS[] = {
        $(join(corners, "\n"))
        };

        int main() {
            double w = 0; int segok = 1;
            int fail_exp = !(exp_sweep() <= 1e-10);
            int fail_dec = 0;
        $(join(("    fail_dec |= sh_decide($(join(map(_sh_f, a), ", "))) != $e;" for (a, e) in SHUNT_RT_DECIDE), "\n"))
        $(join(("    fail_dec |= sh_select($(join(map(_sh_f, a), ", "))) != $e;" for (a, e) in SHUNT_RT_SELECT), "\n"))
            printf("sh_decide / sh_select with exact inputs: %s\\n", fail_dec ? "FAIL" : "ok");
            int fail_cor = 0; double wc = 0;
            for (const Corner &c : CORNERS) {
                ShuntCoefs k = *COEFS[c.inst - 1];   // a copy: the step takes non-const arrays
                St x[5]; Sig u[3]; Pw y[2]; int md[2];
                for (int i = 0; i < 5; i++) x[i] = c.x[i];
                for (int i = 0; i < 3; i++) u[i] = c.u[i];
                Sat se = sh_se(x[4], k.kc);
                shunt_step(x, u, y, md, se, k.P, k.W, k.c, k.wmax, k.wmin, k.Q, k.S, k.kc, k.kt);
                double e = 0;
                for (int i = 0; i < 5; i++) e = fmax(e, fabs(x[i].to_double() - c.ref[i]) / $(S_.x));
                e = fmax(e, fabs(y[0].to_double() - c.ref[5]) / $(S_.vf));
                e = fmax(e, fabs(y[1].to_double() - c.ref[6]) / $(S_.P));
                wc = fmax(wc, e);
                int shift = (x[3] == St(c.x[0])) && (x[4] == St(c.x[3]));    // the history shifted exactly (41d, 2)
                if (md[0] != c.nb || md[1] != c.cs || !(e <= $(SHUNT_RT_LIM)) || !shift) { fail_cor = 1; printf("corner %s: modes %d %d (expected %d %d), difference %.3e\\n", c.name, md[0], md[1], c.nb, c.cs, e); }
            }
            printf("corner suite: %d corners, max relative difference %.3e, modes %s\\n", (int)(sizeof(CORNERS) / sizeof(Corner)), wc, fail_cor ? "FAIL" : "as expected");
        $(join(calls, "\n"))
            int fail = !(w <= $(SHUNT_RT_LIM)) || !segok || fail_exp || fail_dec || fail_cor;
            printf("C sim check: max relative difference %.3e (limit %.0e), continuous = segmented %s, exp %s, decide/select %s, corners %s: %s\\n",
                   w, $(SHUNT_RT_LIM), segok ? "yes" : "NO", fail_exp ? "FAIL" : "ok", fail_dec ? "FAIL" : "ok", fail_cor ? "FAIL" : "ok", fail ? "FAIL" : "PASS");
            return fail;
        }
        """)
    end
    println("written $path")
    return path
end

# the board program: instance 1 in cases A and B, instances 2-6 in case A (coefficients reloaded on the same bitstream),
# then case C (10^7 steps at the operating point); against the embedded Julia values
function shunt_rt_board_script(path, ds, ks, x0s; h, decim = SHUNT_RT_DECIM)
    S_ = SHUNT_RT_SCALE; F = map(I -> 48 - I, SHUNT_RT_I)
    runs = String[]
    for (i, case) in SHUNT_RT_BOARD_RUNS
        d, k, x0 = ds[i], ks[i], x0s[i]
        b = shunt_rt_bench(d, case, h)
        cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        rec, outs = _shunt_rt_julia(k, x0, b, cuts)
        push!(runs, "($i, \"$case\", \"$(d["machine"])\", $(repr(vec(permutedims(b.us)))), $(repr(vec(permutedims(b.du)))), " *
                    "$(b.non), $(b.nmid), $(b.noff), $(repr(x0)), $cuts, " *
                    "$(repr([vcat(rec[g+1], outs[g+1]) for g in eachindex(cuts)])))")
    end
    bC = shunt_rt_bench(ds[1], :C, h)
    coefs = map(ks) do k
        c = Tuple{Float64,Int}[]
        for (fld, T) in SHUNT_RT_FIELDS
            v = getfield(k, fld); f = getfield(F, T)
            vals = v isa AbstractMatrix ? [v[i, j] for i in 1:size(v, 1) for j in 1:size(v, 2)] : v isa AbstractVector ? collect(v) : [v]
            foreach(e -> push!(c, (e, f)), vals)
        end
        "[" * join(("($(_sh_f(v)), $f)" for (v, f) in c), ", ") * "]"
    end
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.shunt_rt_board_script (src/rt/exciters/shunt.jl); do not edit.
        # Runs the DC shunt exciter IP on the PYNQ-Z2 (next to it: shunt.bit, shunt.hwh, xshunt_run_hw.h).
        #   python3 shunt_board.py [--no-download]
        import sys, time, re
        from pynq import Overlay, Clocks

        H, DECIM, LIM = $(_sh_f(h)), $decim, $(SHUNT_RT_LIM)
        SX, SVF, SP = $(S_.x), $(S_.vf), $(S_.P)
        FST, FSIG, FSMALL, FPW = $(F.St), $(F.Sig), $(F.Small), $(F.Pw)
        COEFS = [$(join(coefs, ",\n         "))]
        RUNS = [$(join(runs, ",\n        "))]
        US_C, DU_C, X0_C = $(repr(vec(permutedims(bC.us)))), $(repr(vec(permutedims(bC.du)))), $(repr(x0s[1]))

        A = {}
        for m in re.finditer(r"#define XSHUNT_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xshunt_run_hw.h").read()):
            A[m.group(1)] = int(m.group(3), 16)

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("shunt.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / SHUNT_RT_CLOCK_NS)) > 0.01:
            Clocks.fclk0_mhz = $(1000 / SHUNT_RT_CLOCK_NS)
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.shunt_run_0

        def w48(addr, v, F):
            r = int(round(v * 2**F)) & ((1 << 48) - 1)
            ip.write(addr, r & 0xFFFFFFFF)
            ip.write(addr + 4, r >> 32)
        def r48(addr, F):
            r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
            if r >> 47:
                r -= 1 << 48
            return r / 2**F

        def load(i):
            for j, (v, F) in enumerate(COEFS[i - 1]):
                w48(A["K"] + 8 * j, v, F)
        def run(x, us, du, on, mid, off, n0, nsteps):
            for j, v in enumerate(x):
                w48(A["X"] + 8 * j, v, FST)
            for j in range(12):
                w48(A["US"] + 8 * j, us[j], FSIG); w48(A["DU"] + 8 * j, du[j], FSMALL)
            for nm, v in (("N_ON", on), ("N_MID", mid), ("N_OFF", off), ("N0", n0), ("NSTEPS", nsteps), ("DECIM", DECIM)):
                ip.write(A[nm], v)
            t0 = time.perf_counter()
            ip.write(0x00, 1)
            while not (ip.read(0x00) & 0x2):
                time.sleep(0.001)
            t = time.perf_counter() - t0
            return t, [r48(A["X"] + 8 * j, FST) for j in range(5)], [r48(A["Y"] + 8 * j, FPW) for j in range(2)]

        worst = 0.0
        for (i, case, mach, us, du, on, mid, off, x0, cuts, ref) in RUNS:
            load(i)
            print("instance %d (%s), case %s: compared at %s s" % (i, mach, case, [round(c * H, 3) for c in cuts]))
            x, n, tt = list(x0), 0, 0.0
            for g, c in enumerate(cuts):
                t, x, y = run(x, us, du, on, mid, off, n, c - n)
                tt += t; n = c
                e = max(max(abs(x[j] - ref[g][j]) for j in range(5)) / SX, abs(y[0] - ref[g][5]) / SVF, abs(y[1] - ref[g][6]) / SP)
                worst = max(worst, e)
                print("  t = %.3f s: y %.10f xa %.10f xf %.10f v_f %.6e P %.6e; max relative |board - Julia| %.3e"
                      % (c * H, x[0], x[1], x[2], y[0], y[1], e))
            print("  host elapsed time %.4f s (simulated %.4f s)" % (tt, cuts[-1] * H))

        load(1)
        N2 = 10_000_000
        print("case C: instance 1 at its operating point,", N2, "steps (%.3f s simulated)" % (N2 * H))
        t, x, y = run(X0_C, US_C, DU_C, 1 << 30, 1 << 30, 1 << 30, 0, N2)
        dx = max(abs(x[j] - X0_C[j]) for j in range(5))
        print("  host elapsed time %.4f s, %.4f us per step (h = %.4f us); max |x - x0| %.3e (limit 1e-9)" % (t, t / N2 * 1e6, H * 1e6, dx))
        ok = worst <= LIM and dx <= 1e-9
        print("board check: max relative |board - Julia| %.3e (limit %.0e), case C |x - x0| %.3e (limit 1e-9): %s"
              % (worst, LIM, dx, "PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
        """)
    end
    println("written $path")
    return path
end
