# =====================================================================================================================
# PHMachine, the type extended to the cut-node machines (reviews 47, 47b): one build for all ten IEEE-39 generators
# =====================================================================================================================
# The model of phmachine.jl with each axis a group of FOUR windings (d: stator, fd, 1d, T; q: stator, 1q, 2q, T): at a
# reduced inductor-only node (GENROU_2 at bus 31, GENROU_5+6 at bus 20; src/phsim/system.jl header lines 44-66, PHCut,
# _extended_K) the step-up transformer's branch pair T is a fourth winding in the rotor frame, coupled through the
# extended inductance and the 2 x 2 resistance Rm (stator, branch), and the port is the far bus; an ordinary machine has
# winding 4 decoupled (its rows and columns of K and R zero, no input, its flux 0 at the start and exactly 0 after).
#   psi_g' = omega_b (e_pw v_g + e_2 v_f [d] - R_g i_g) +- omega_b omega psi_other (pairs 1 and 4),  i_g = K_g psi_g
#   2H omega' = T_m + sum over pairs 1, 4 (psi_d i_q - psi_q i_d),  theta' = omega_b omega
#   port: v_pw = kappa j e^{-j theta} v (kappa = 1/a ordinary, c cut), y = kappa (-j e^{j theta} i_pw); the bus gets -y
# Step: phmachine.jl's frozen-skew linearly implicit midpoint, generalised (47b section 2): two speed-coupling columns,
# both pairs in the torque, the speed's implicit factor S = psi^_q' G_d psi^_q + psi^_d' G_q psi^_d (G symmetric).
# States x (18): psi d group 1-4, psi q group 5-8, omega 9, midpoint phasor (cos, sin) 10-11, predictor history
# (psi_d1, psi_q1, psi_d4, psi_q4) 12-15, a_n 16, the bench's source phasor 17-18.
# =====================================================================================================================

const PHM4_NX = 18

"""    phm4_data(name): one machine of ieee39_machines.toml (cut machines included), with omega_b."""
function phm4_data(name; path = joinpath(@__DIR__, "data", "ieee39_machines.toml"))
    t = TOML.parsefile(path)
    i = findfirst(m -> m["name"] == name, t["machine"]); i === nothing && error("no machine $name")
    d = t["machine"][i]
    d["a_im"] == 0 || error("$name: complex ratio a not in this build")
    return merge(d, Dict("omega_b" => t["omega_b"]))
end
phm4_names(; path = joinpath(@__DIR__, "data", "ieee39_machines.toml")) = [m["name"] for m in TOML.parsefile(path)["machine"]]

"""    phm4_KR(d): per axis the 4 x 4 inverse inductance K and resistance R (BigFloat); the port winding and kappa."""
function phm4_KR(d)
    B(k) = BigFloat(d[k])
    Ll, Lad, Laq = B("Ll"), B("Lad"), B("Laq")
    Ld = [Ll+Lad Lad Lad; Lad B("Lfd")+Lad Lad; Lad Lad B("L1d")+Lad]
    Lq = d["nq"] == 2 ? [Ll+Laq Laq Laq; Laq B("L1q")+Laq Laq; Laq Laq B("L2q")+Laq] : [Ll+Laq Laq; Laq B("L1q")+Laq]
    Rd = [B("Ra"), B("Rfd"), B("R1d")]; Rq = [B("Ra"), B("R1q"), d["nq"] == 2 ? B("R2q") : BigFloat(0)]
    Kd = zeros(BigFloat, 4, 4); Kq = zeros(BigFloat, 4, 4)
    RdM = zeros(BigFloat, 4, 4); RqM = zeros(BigFloat, 4, 4)
    for i in 1:3; RdM[i, i] = Rd[i]; RqM[i, i] = Rq[i]; end
    nqw = size(Lq, 1)
    if d["cut"] == 0
        Kd[1:3, 1:3] = inv(Ld); Kq[1:nqw, 1:nqw] = inv(Lq)
        return (; Kd, Kq, Rd = RdM, Rq = RqM, pw = 1, kappa = 1 / B("a_re"))
    end
    M = permutedims(reshape(BigFloat.(d["M"]), 2, 2)); Rm = permutedims(reshape(BigFloat.(d["Rm"]), 2, 2))
    ext(L, n) = (E = zeros(BigFloat, 4, 4); E[1:n, 1:n] = L; E[1, 1] += M[1, 1]; E[1, 4] = E[4, 1] = M[1, 2];
                 E[4, 4] = M[2, 2]; E)
    Ed = ext(Ld, 3); Eq = ext(Lq, nqw)
    nqw == 2 && (Eq[3, 3] = 1)                         # an absent 2q: an identity placeholder, its K row zeroed below
    Kd = inv(Ed); Kq = inv(Eq)
    nqw == 2 && (Kq[3, :] .= 0; Kq[:, 3] .= 0)
    for R in (RdM, RqM); R[1, 1] = Rm[1, 1]; R[1, 4] = Rm[1, 2]; R[4, 1] = Rm[2, 1]; R[4, 4] = Rm[2, 2]; end
    return (; Kd, Kq, Rd = RdM, Rq = RqM, pw = 4, kappa = BigFloat(d["c"]))
end

"""
    phm4_rows(d, KR, psid, psiq, om, th, v, vf, Tm) -> NamedTuple

The continuous model (the reference of checks a, a', b): rates of psi (4 + 4), omega, theta; currents, torque, the
port's y = kappa (-j e^{j theta} i_pw) and the energy terms. `v` the port bus voltage (complex, the frame of th).
"""
function phm4_rows(d, KR, psid, psiq, om, th, v, vf, Tm)
    wb = d["omega_b"]; T = promote_type(eltype(psid), typeof(om), typeof(real(v)))
    Kd, Kq, Rd, Rq = (T.(KR.Kd), T.(KR.Kq), T.(KR.Rd), T.(KR.Rq)); pw = KR.pw; ka = T(KR.kappa)
    id = Kd * psid; iq = Kq * psiq
    vp = ka * im * cis(-th) * v
    dd = -wb .* (Rd * id); dq = -wb .* (Rq * iq)
    dd[pw] += wb * real(vp); dq[pw] += wb * imag(vp); dd[2] += wb * vf
    for i in (1, 4); dd[i] += wb * om * psiq[i]; dq[i] -= wb * om * psid[i]; end
    Te = sum(psid[i] * iq[i] - psiq[i] * id[i] for i in (1, 4))
    dom = (Tm + Te) / (2 * d["H"])
    y = ka * (-im * cis(th) * complex(id[pw], iq[pw]))
    Pport = real(vp) * id[pw] + imag(vp) * iq[pw]
    loss = dot(id, Rd * id) + dot(iq, Rq * iq)
    Hm = (dot(psid, Kd * psid) + dot(psiq, Kq * psiq)) / (2wb) + d["H"] * om^2
    dH = (dot(dd, id) + dot(dq, iq)) / wb + 2 * d["H"] * om * dom
    # each row's own terms' magnitudes (review 47c, 2): resistances, the port voltage on the port row only, the field on
    # row 2 of d only, the rotation on the pair rows only
    sd = wb .* (abs.(Rd) * abs.(id)); sq = wb .* (abs.(Rq) * abs.(iq))
    sd[pw] += wb * abs(real(vp)); sq[pw] += wb * abs(imag(vp)); sd[2] += wb * abs(vf)
    for i in (1, 4); sd[i] += wb * abs(om * psiq[i]); sq[i] += wb * abs(om * psid[i]); end
    return (; dd, dq, dom, dth = wb * om, id, iq, Te, y, Pport, ifd = id[2], loss, Hm, dH, sd, sq,
            bal = (Pport + vf * id[2] + Tm * om - loss))
end
dot(a, b) = sum(a .* b)

"""    phm4_coefs(d, h) -> NamedTuple of BigFloat: the generalised step's preparation (47b section 2)."""
function phm4_coefs(d, h)
    KR = phm4_KR(d); wb = BigFloat(d["omega_b"])
    function axis(R, K)
        A = -wb * R * K; Mi = inv(Matrix{BigFloat}(I, 4, 4) - (h / 2) * A)
        (; P = h * Mi * A, C = h * wb * Mi)          # C[:, j] = h omega_b M^-1 e_j
    end
    ad = axis(KR.Rd, KR.Kd); aq = axis(KR.Rq, KR.Kq); pw = KR.pw; ka = KR.kappa
    cdv = ad.C[:, pw] * ka; cqv = aq.C[:, pw] * ka; wf = ad.C[:, 2]
    k(K, i) = K[i, :]
    ar(K, P, i) = k(K, i) + transpose(P) * k(K, i) / 2
    Gd = [sum(k(KR.Kd, i) .* ad.C[:, j]) for i in (1, 4), j in (1, 4)]
    Gq = [sum(k(KR.Kq, i) .* aq.C[:, j]) for i in (1, 4), j in (1, 4)]
    hq = h / (2 * BigFloat(d["H"])); be = hq * hq / 4
    return (; P_d = ad.P, P_q = aq.P, cdv, cqv, wf, c1d = ad.C[:, 1], c4d = ad.C[:, 4], c1q = aq.C[:, 1], c4q = aq.C[:, 4],
            a1d = ar(KR.Kd, ad.P, 1), a4d = ar(KR.Kd, ad.P, 4), a1q = ar(KR.Kq, aq.P, 1), a4q = ar(KR.Kq, aq.P, 4),
            alphad1 = sum(k(KR.Kd, 1) .* cdv) / 2, alphad4 = sum(k(KR.Kd, 4) .* cdv) / 2,
            alphaq1 = sum(k(KR.Kq, 1) .* cqv) / 2, alphaq4 = sum(k(KR.Kq, 4) .* cqv) / 2,
            bf1 = sum(k(KR.Kd, 1) .* wf) / 2, bf4 = sum(k(KR.Kd, 4) .* wf) / 2,
            kd11 = Gd[1, 1] / 2, kd14 = (Gd[1, 2] + Gd[2, 1]) / 2, kd44 = Gd[2, 2] / 2,
            kq11 = Gq[1, 1] / 2, kq14 = (Gq[1, 2] + Gq[2, 1]) / 2, kq44 = Gq[2, 2] / 2,
            bd11 = be * Gd[1, 1], bd14 = be * (Gd[1, 2] + Gd[2, 1]), bd44 = be * Gd[2, 2],
            bq11 = be * Gq[1, 1], bq14 = be * (Gq[1, 2] + Gq[2, 1]), bq44 = be * Gq[2, 2],
            kp1d = sum(k(KR.Kd, pw) .* ad.C[:, 1]), kp4d = sum(k(KR.Kd, pw) .* ad.C[:, 4]),
            kp1q = sum(k(KR.Kq, pw) .* aq.C[:, 1]), kp4q = sum(k(KR.Kq, pw) .* aq.C[:, 4]),
            kappa = ka, pw, hq, kap1 = 3 * h * wb / 2, kap2 = 2 * h * wb, wbh2 = h * wb / 2, Gd, Gq, KR)
end

@hls struct PHM4Coefs
    Pd::Small[4, 4]     # d group step matrix (stator, fd, 1d, T)
    Pq::Small[4, 4]     # q group (stator, 1q, 2q, T)
    cdv::Small[4]       # the port voltage's column (kappa h omega_b M^-1 e_pw), d and q
    cqv::Small[4]
    wf::Small[4]        # the field voltage's column (d)
    c1d::Small[4]       # the speed-coupling columns of the pairs (h omega_b M^-1 e_1, e_4)
    c4d::Small[4]
    c1q::Small[4]
    c4q::Small[4]
    a1d::Coef[4]        # currents of windings 1 and 4 at (psi + y / 2) from psi: k_i + P' k_i / 2
    a4d::Coef[4]
    a1q::Coef[4]
    a4q::Coef[4]
    alphad1::Mid        # k_i . cdv / 2, k_i . cqv / 2: the same currents per unit port voltage
    alphad4::Mid
    alphaq1::Mid
    alphaq4::Mid
    bf1::Mid            # k_i . wf / 2 (field)
    bf4::Mid
    kd11::Mid           # G / 2 (cross term G): t3 = omega_n S / 2 by terms
    kd14::Mid
    kd44::Mid
    kq11::Mid
    kq14::Mid
    kq44::Mid
    kp1d::Mid           # the port winding's current per unit psi^ of each pair (speed coupling)
    kp4d::Mid
    kp1q::Mid
    kp4q::Mid
    kappa::Coef         # 1 / a (ordinary) or c (cut)
    pw::Int             # the port winding: 1 or 4
    hq::Rate            # h / 2H
    bd11::Beta          # (hq^2 / 4) G (cross term 2 G): q = hq - bs
    bd14::Beta
    bd44::Beta
    bq11::Beta
    bq14::Beta
    bq44::Beta
    kap1::Ang
    kap2::Ang
    wbh2::Ang
end

const PHM4_I = (Sig = 11, St = 3, Small = -8, Coef = 9, Ang = -9, Mid = -2, Rate = -24, Beta = -59, Beta2 = -55)
const PHM4_FIELDS = [:Pd => :Small, :Pq => :Small, :cdv => :Small, :cqv => :Small, :wf => :Small, :c1d => :Small,
                     :c4d => :Small, :c1q => :Small, :c4q => :Small, :a1d => :Coef, :a4d => :Coef, :a1q => :Coef,
                     :a4q => :Coef, :alphad1 => :Mid, :alphad4 => :Mid, :alphaq1 => :Mid, :alphaq4 => :Mid, :bf1 => :Mid,
                     :bf4 => :Mid, :kd11 => :Mid, :kd14 => :Mid, :kd44 => :Mid, :kq11 => :Mid, :kq14 => :Mid, :kq44 => :Mid,
                     :kp1d => :Mid, :kp4d => :Mid, :kp1q => :Mid, :kp4q => :Mid, :kappa => :Coef, :pw => :Int,
                     :hq => :Rate, :bd11 => :Beta, :bd14 => :Beta, :bd44 => :Beta, :bq11 => :Beta, :bq14 => :Beta,
                     :bq44 => :Beta, :kap1 => :Ang, :kap2 => :Ang, :wbh2 => :Ang]

"""    phm4_coefstruct(d, h; grid = true, check = grid): PHM4Coefs, each coefficient range-checked (check) and (grid)
rounded to its type (the order check runs at large h with unrounded coefficients: no range check there)."""
function phm4_coefstruct(d, h; grid = true, check = grid)
    c = phm4_coefs(d, h)
    v = Dict(:Pd => c.P_d, :Pq => c.P_q, (f => getfield(c, f) for (f, _) in PHM4_FIELDS if !(f in (:Pd, :Pq)))...)
    args = map(PHM4_FIELDS) do (f, T)
        T === :Int && return v[f]
        I_ = getfield(PHM4_I, T)
        m = Float64(maximum(abs, v[f]))
        (!check || m < 2.0^(I_ - 1)) || error("$(d["name"]): coefficient $f (max |value| $m) does not fit $T = ap_fixed<48, $I_>")
        grid ? _q.(v[f], 48 - I_) : Float64.(v[f])
    end
    return PHM4Coefs(args...)
end

# ---------------------------------------------------------------------------------------------------------------------
# THE STEP (becomes the C++): phmachine_step! generalised to two 4-winding groups, two pairs, a port winding
# ---------------------------------------------------------------------------------------------------------------------
@hls function phm4_step!(x::St[18], u::Sig[4], y::Sig[2], hold::Bool, hold_next::Bool, k::PHM4Coefs)
    @pragma "HLS ARRAY_PARTITION variable=x complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    pd1::Sig = (3 * x[1] - x[12]) / 2                # predicted midpoint fluxes of the pairs (windings 1 and 4)
    pq1::Sig = (3 * x[5] - x[13]) / 2
    pd4::Sig = (3 * x[4] - x[14]) / 2
    pq4::Sig = (3 * x[8] - x[15]) / 2
    if hold                                          # the step after an input edge: no extrapolation
        pd1 = x[1]
        pq1 = x[5]
        pd4 = x[4]
        pq4 = x[8]
    end
    mpd1::Sig = -pd1
    mpd4::Sig = -pd4
    c::St = x[10]                                    # (cos, sin) of the step's midpoint angle
    s::St = x[11]
    gn::St = (3 - (c * c + s * s)) / 2
    cn::St = c * gn
    sn::St = s * gn
    sixth::Sig = 0.16666666666666666
    cn6::Sig = cn * sixth
    sn6::Sig = sn * sixth
    kap::Ang = k.kap1
    asum::Sig = x[16]
    if hold_next
        kap = k.kap2
        asum = x[9] + x[16]
    end
    b::Ang = k.wbh2 * asum

    vd::Sig = s * u[1] - c * u[2]                    # Park: j e^{-j theta} v (kappa in the coefficients)
    vq::Sig = c * u[1] + s * u[2]

    xd::Sig[4] = (x[1], x[2], x[3], x[4])
    xq::Sig[4] = (x[5], x[6], x[7], x[8])
    yd::Ang[4]
    yq::Ang[4]
    for i in 1:4
        sd::W4y = k.cdv[i] * vd + k.wf[i] * u[3]      # summed exactly on a wide word, rounded once to Ang (each rounded
        sq::W4y = k.cqv[i] * vq                       # partial sum cost an adder and a rounder: the routing congested)
        for j in 1:4
            sd += k.Pd[i, j] * xd[j]
            sq += k.Pq[i, j] * xq[j]
        end
        yd[i] = sd
        yq[i] = sq
    end

    a1w::W4c = k.bf1 * u[3]                          # currents of windings 1, 4 at (psi + y / 2) without v terms (wide sums)
    a4w::W4c = k.bf4 * u[3]
    b1w::W4c = 0
    b4w::W4c = 0
    for j in 1:4
        a1w += k.a1d[j] * xd[j]
        a4w += k.a4d[j] * xd[j]
        b1w += k.a1q[j] * xq[j]
        b4w += k.a4q[j] * xq[j]
    end
    D1::Sig = a1w                                    # rounded once
    D4::Sig = a4w
    Q1::Sig = b1w
    Q4::Sig = b4w
    Dp::Sig = D1                                     # the port winding's
    Qp::Sig = Q1
    adp::Mid = k.alphad1
    aqp::Mid = k.alphaq1
    if k.pw == 4
        Dp = D4
        Qp = Q4
        adp = k.alphad4
        aqp = k.alphaq4
    end
    ibd::Sig = Dp + adp * vd
    ibq::Sig = Qp + aqp * vq
    e1::Sig = k.alphad1 * u[1]                       # the torque's voltage terms, the phasor entering once:
    e4::Sig = k.alphad4 * u[1]                       # X = u1 A_q - u2 A_d, Y = -u2 A_q - u1 A_d
    f1::Sig = k.alphad1 * u[2]
    f4::Sig = k.alphad4 * u[2]
    g1::Sig = k.alphaq1 * u[1]
    g4::Sig = k.alphaq4 * u[1]
    h1::Sig = k.alphaq1 * u[2]
    h4::Sig = k.alphaq4 * u[2]
    X::Sig = pq1 * e1 + pq4 * e4 - pd1 * h1 - pd4 * h4
    Y::Sig = mpd1 * g1 + mpd4 * g4 - pq1 * f1 - pq4 * f4
    q11::Sig = pq1 * pq1
    q14::Sig = pq1 * pq4
    q44::Sig = pq4 * pq4
    d11::Sig = pd1 * pd1
    d14::Sig = pd1 * pd4
    d44::Sig = pd4 * pd4
    w11::Mid = x[9] * k.kd11                         # omega_n G / 2
    w14::Mid = x[9] * k.kd14
    w44::Mid = x[9] * k.kd44
    v11::Mid = x[9] * k.kq11
    v14::Mid = x[9] * k.kq14
    v44::Mid = x[9] * k.kq44
    T1::Sig = pq1 * D1 + pq4 * D4 - pd1 * Q1 - pd4 * Q4 + s * X + c * Y
    t3::Sig = w11 * q11 + w14 * q14 + w44 * q44 + v11 * d11 + v14 * d14 + v44 * d44
    f::Sig = u[4] - T1 - t3
    bs::Beta2 = k.bd11 * q11 + k.bd14 * q14 + k.bd44 * q44 + k.bq11 * d11 + k.bq14 * d14 + k.bq44 * d44
    q::Rate = k.hq - bs
    dw::Small = q * f
    wm::Sig = x[9] + dw / 2

    ird::Sig = k.kp1d * pq1 + k.kp4d * pq4           # the port winding's speed-coupling current per omega_m
    irq::Sig = k.kp1q * mpd1 + k.kp4q * mpd4
    hwm::Sig = wm / 2
    idm::Sig = ibd + hwm * ird
    iqm::Sig = ibq + hwm * irq
    t4::Sig = s * idm + c * iqm                      # -j e^{j theta} (i_d + j i_q), times kappa
    t5::Sig = s * iqm - c * idm
    y[1] = t4 * k.kappa
    y[2] = t5 * k.kappa

    dl::Ang = kap * wm - b
    dl2::Small = dl * dl
    hdl2::Small = dl2 / 2
    sdl::Small = sn6 * dl
    cdl::Small = cn6 * dl
    c1::St = cn - cn * hdl2 - sn * dl + sdl * dl2
    s1::St = sn - sn * hdl2 + cn * dl - cdl * dl2

    ecd::Mid[4]
    ecq::Mid[4]
    for i in 1:4
        ecd[i] = k.c1d[i] * pq1 + k.c4d[i] * pq4
        ecq[i] = k.c1q[i] * mpd1 + k.c4q[i] * mpd4
    end
    x[12] = x[1]
    x[13] = x[5]
    x[14] = x[4]
    x[15] = x[8]
    for i in 1:4
        x[i] = x[i] + yd[i] + ecd[i] * wm
        x[i + 4] = x[i + 4] + yq[i] + ecq[i] * wm
    end
    x[9] = x[9] + dw
    x[10] = c1
    x[11] = s1
    x[16] = wm
    if hold_next
        x[16] = x[9]
    end
end

"""    phm4_initial(d, h) -> (x0, bench): the operating point (synchronous frame at t = 0) and the bench's port source."""
function phm4_initial(d, h; grid = true)
    wb = d["omega_b"]; lam = d["lam"]
    lq2 = d["nq"] == 2 ? lam[6] : 0.0
    lT = d["cut"] > 0 ? d["lamT"] : [0.0, 0.0]
    psid = wb .* [lam[1], lam[3], lam[4], lT[1]]; psiq = wb .* [lam[2], lam[5], lq2, lT[2]]
    om = d["p"] / (2 * d["H"])
    v = d["cut"] > 0 ? complex(d["far_v_re"], d["far_v_im"]) : complex(d["v_re"], d["v_im"])
    ph = angle(v) + wb * h / 2
    th = d["delta"] + wb * h * om / 2
    x0 = vcat(psid, psiq, om, cos(th), sin(th), psid[1], psiq[1], psid[4], psiq[4], om, cos(ph), sin(ph))
    F = map(I -> 48 - I, PHM4_I)
    r(v, T) = grid ? _q(v, getfield(F, T)) : v
    return r.(x0, :St), (; vmag = r(abs(v), :Sig), cm = r(2 * sin(wb * h / 2)^2, :Small), srot = r(sin(wb * h), :Small),
                          vf = r(d["vf"], :Sig), tm = r(d["Tm"], :Sig), v)
end

# the bench in Julia: n steps of phm4_step! with the rotating port source and the dip (as phmachine_run!)
function phm4_bench!(x, k, b, n0, nsteps; vdip = 0.5, non, noff, out = nothing)
    u = zeros(4); y = zeros(2)
    for i in 1:nsteps
        n = n0 + i
        vs = (n > non && n <= noff) ? b.vmag * vdip : b.vmag
        hold = n == non + 1 || n == noff + 1
        hn = n == non || n == noff
        u[1] = vs * x[17]; u[2] = vs * x[18]; u[3] = b.vf; u[4] = b.tm
        phm4_step!(x, u, y, hold, hn, k)
        sc1 = x[17] - x[17] * b.cm - x[18] * b.srot; ss1 = x[18] - x[18] * b.cm + x[17] * b.srot
        gs = (3 - (sc1 * sc1 + ss1 * ss1)) / 2
        x[17] = sc1 * gs; x[18] = ss1 * gs
        out === nothing || out(n, x, y)
    end
    return x
end

# ---------------------------------------------------------------------------------------------------------------------
# THE BOARD TOP (the IP): phmachine_run! for the generalised step (18 states; the port source in x[17], x[18])
# ---------------------------------------------------------------------------------------------------------------------
@hls function phm4_run!(k::PHM4Coefs, x::St[18], vmag::Sig, cm::Small, srot::Small, vf::Sig, tm::Sig, vdip::Sig, n0::Int,
                        n_on::Int, n_off::Int, nsteps::Int, decim::Int, omega_out::Out{Sig}, ia_out::Out{Sig},
                        ib_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=vmag"
    @pragma "HLS INTERFACE mode=s_axilite port=cm"
    @pragma "HLS INTERFACE mode=s_axilite port=srot"
    @pragma "HLS INTERFACE mode=s_axilite port=vf"
    @pragma "HLS INTERFACE mode=s_axilite port=tm"
    @pragma "HLS INTERFACE mode=s_axilite port=vdip"
    @pragma "HLS INTERFACE mode=s_axilite port=n0"
    @pragma "HLS INTERFACE mode=s_axilite port=n_on"
    @pragma "HLS INTERFACE mode=s_axilite port=n_off"
    @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=omega_out"
    @pragma "HLS INTERFACE mode=ap_none port=ia_out"
    @pragma "HLS INTERFACE mode=ap_none port=ib_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::St[18]
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    for i in 1:18
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    u::Sig[4]
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    y::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    st::Bool = false
    cnt::Int = 0
    for i in 1:nsteps
        @pragma "HLS PIPELINE II=43"
        @pragma "HLS ALLOCATION operation instances=mul limit=16"   # 16: 153 DSP (19: 180 DSP did not route, congestion 32 x 32)
        n::Int = n0 + i
        vs::Sig = vmag
        if n > n_on && n <= n_off
            vs = vmag * vdip
        end
        hold::Bool = false
        if n == n_on + 1
            hold = true
        end
        if n == n_off + 1
            hold = true
        end
        hold_next::Bool = false
        if n == n_on
            hold_next = true
        end
        if n == n_off
            hold_next = true
        end
        u[1] = vs * xs[17]
        u[2] = vs * xs[18]
        u[3] = vf
        u[4] = tm
        phm4_step!(xs, u, y, hold, hold_next, k)
        sc1::St = xs[17] - xs[17] * cm - xs[18] * srot   # the bench's source phasor on the state grid (St): on Sig <48,11>
        ss1::St = xs[18] - xs[18] * cm + xs[17] * srot   # (widened for GENROU_1's currents) its 7.3e-12 rounding random-
        gs::St = (3 - (sc1 * sc1 + ss1 * ss1)) / 2      # walked ~4e-9 in angle over 0.3 s of the dip bench
        xs[17] = sc1 * gs
        xs[18] = ss1 * gs
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
        omega_out[] = xs[9]
        ia_out[] = y[1]
        ib_out[] = y[2]
        strobe[] = st
    end
    for i in 1:18
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
end

const PHM4_CLOCK_NS = 8
const PHM4_STEP_CYCLES = 43   # 43 since the wide exact sums (42 with the per-term rounding did not route)
# Sig widened to +-1024 (review 47c, 1: GENROU_1's stator currents reach 256 on the 50 % dip bench; the range check (d)
# of scripts/realtime/phmachine4_checks.jl); grid 7.3e-12
const W4y = Float64       # the flux increments' exact sums (ap_fixed<96, 0>: wiring), rounded once to Ang
const W4c = Float64       # the winding currents' sums (ap_fixed<96, 14>), rounded once to Sig
const PHM4_TYPES = [:W4y => "ap_fixed<96, 0, AP_RND_CONV>", :W4c => "ap_fixed<96, 14, AP_RND_CONV>",
                    :Sig => "ap_fixed<48, 11, AP_RND_CONV>", :St => "ap_fixed<48, 3, AP_RND_CONV>",
                    :Small => "ap_fixed<48, -8, AP_RND_CONV>", :Coef => "ap_fixed<48, 9, AP_RND_CONV>",
                    :Ang => "ap_fixed<48, -9, AP_RND_CONV>", :Mid => "ap_fixed<48, -2, AP_RND_CONV>",
                    :Rate => "ap_fixed<48, -24, AP_RND_CONV>", :Beta => "ap_fixed<48, -59, AP_RND_CONV>",
                    :Beta2 => "ap_fixed<48, -55, AP_RND_CONV>"]

"""    phm4_export(name = "GENROU_2"; dir): the C++ of the generalised step and board top, and one machine's coefficients
(at h = PHM4_STEP_CYCLES x 8 ns, every coefficient range-checked) for C synthesis."""
function phm4_export(name = "GENROU_2"; dir = joinpath(@__DIR__, "CPP", "phmachine4"))
    occursin("II=$(PHM4_STEP_CYCLES)\"", string(HLS_DEFS[:phm4_run!])) || error("PIPELINE II differs from PHM4_STEP_CYCLES")
    h = setprecision(() -> BigFloat(PHM4_STEP_CYCLES) * PHM4_CLOCK_NS / 10^9, BigFloat, 256)
    ks = Dict(nm => setprecision(() -> phm4_coefstruct(phm4_data(nm), h), BigFloat, 256) for nm in phm4_names())
    mkpath(dir)
    hls_write_util(dir)
    hls_write(joinpath(dir, "phmachine4.cpp"), [:PHM4Coefs, :phm4_step!, :phm4_run!]; types = PHM4_TYPES,
              header = "PHMachine, the type with the branch winding (all ten generators); src/rt/machines/phmachine4.jl")
    hls_write_const(joinpath(dir, "phmachine4_coefs.h"), "PHM4_COEFS", ks[name]; header = "machine $name, h = $(Float64(h)) s")
    println("coefficients of all ten machines fit their types at h = $(Float64(h)) s")
    return dir
end

# ---------------------------------------------------------------------------------------------------------------------
# THE BOARD STAGE: flow.cfg and the board program (all ten machine sets through the one bitstream)
# ---------------------------------------------------------------------------------------------------------------------
# One (value, fraction bits) per 64-bit slot of the coefficient struct, in the struct's order (matrices row by row; the
# port winding's index as an integer, fraction bits 0).
function phm4_coef_words(k::PHM4Coefs)
    w = Tuple{Float64,Int}[]
    for (f, T) in PHM4_FIELDS
        v = getfield(k, f); F = T === :Int ? 0 : 48 - getfield(PHM4_I, T)
        vals = v isa AbstractMatrix ? [v[i, j] for i in 1:size(v, 1) for j in 1:size(v, 2)] : collect(v)
        foreach(e -> push!(w, (Float64(e), F)), vals)
    end
    return w
end

"""    phm4_rt_export(): the C++ (phm4_export), flow.cfg and the board program phmachine4_board.py. The board data
(phmachine4_board_data.bin: the ten sets, the bench, Julia's checkpoints) is written by scripts/realtime/phm4_board_data.jl,
the C++ checkpoints (phmachine4_board_cpp.bin) by the C simulation (scripts/realtime/csim_phm4.sh)."""
function phm4_rt_export(; dir = joinpath(@__DIR__, "CPP", "phmachine4"))
    phm4_export(; dir)
    F = map(I -> 48 - I, PHM4_I)
    open(joinpath(dir, "phmachine4_board.py"), "w") do io
        print(io, """
# phmachine4_board.py: written by PHPS.RT.phm4_rt_export (src/rt/machines/phmachine4.jl); do not edit.
# Runs the PHMachine4 IP (one build, all ten IEEE-39 machine sets) on the PYNQ-Z2: each set on the dip bench (the port
# source x 0.5 for 0.1-0.2 s, 0.3 s), cut at the checkpoints of the C simulation; the 18 returned state words at every
# checkpoint against the C++ (bit for bit, phmachine4_board_cpp.bin) and Julia (5e-8 relative, phmachine4_board_data.bin).
#   python3 phmachine4_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
FST, FSIG, FSM = $(F.St), $(F.Sig), $(F.Small)
A = {}
for m in re.finditer(r"#define XPHM4_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xphm4_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("phmachine4.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.phm4_run_0
def raw(v, F):
    return int(round(v * 2.0**F)) & ((1 << 48) - 1)
def w48(addr, v, F):
    r = raw(v, F); ip.write(addr, r & 0xFFFFFFFF); ip.write(addr + 4, r >> 32)
def r48(addr):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    return r - (1 << 48) if r >> 47 else r
D = np.fromfile("phmachine4_board_data.bin", dtype=np.float64)
nset, ncut, non, noff = (int(v) for v in D[:4]); p = 4
cuts = [int(v) for v in D[p:p + ncut]]; p += ncut
C = np.fromfile("phmachine4_board_cpp.bin", dtype=np.int64).reshape(nset, ncut, 20) & ((1 << 48) - 1)   # raw 48-bit words
C = np.where(C >> 47, C - (1 << 48), C)
ok = True; worst = 0.0; nbits = 0; ttot = 0.0
for s in range(nset):
    name = "".join(chr(int(c)) for c in D[p:p + 12]).strip(); p += 12
    nw = int(D[p]); p += 1
    W = D[p:p + 2 * nw].reshape(nw, 2); p += 2 * nw
    x0 = D[p:p + 18]; p += 18
    vmag, cm, srot, vf, tm = D[p:p + 5]; p += 5
    J = D[p:p + 18 * ncut].reshape(ncut, 18); p += 18 * ncut
    for i, (v, F) in enumerate(W):
        if F == 0:
            ip.write(A["K"] + 8 * i, int(v)); ip.write(A["K"] + 8 * i + 4, 0)
        else:
            w48(A["K"] + 8 * i, v, int(F))
    w48(A["VMAG"], vmag, FSIG); w48(A["CM"], cm, FSM); w48(A["SROT"], srot, FSM)
    w48(A["VF"], vf, FSIG); w48(A["TM"], tm, FSIG); w48(A["VDIP"], 0.5, FSIG)
    for i in range(18):
        w48(A["X"] + 8 * i, x0[i], FST)
    ip.write(A["N_ON"], non); ip.write(A["N_OFF"], noff); ip.write(A["DECIM"], 32)
    prev, ex, nb = 0, 0.0, 0
    for g, c in enumerate(cuts):
        ip.write(A["N0"], prev); ip.write(A["NSTEPS"], c - prev)
        t0 = time.perf_counter(); ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        ttot += time.perf_counter() - t0
        xr = [r48(A["X"] + 8 * i) for i in range(18)]
        nb += sum(1 for i in range(18) if xr[i] != C[s, g, i])
        ex = max(ex, max(abs(xr[i] / 2.0**FST - J[g, i]) / max(1.0, abs(J[g, i])) for i in range(18)))
        prev = c
    pas = nb == 0 and ex <= 5e-8
    ok = ok and pas; worst = max(worst, ex); nbits += nb
    print("  %-10s %d checkpoints x 18 words: board - C++ %s, board - Julia %.3e: %s" % (name, ncut, "0 LSB" if nb == 0 else "%d words differ" % nb, ex, "PASS" if pas else "FAIL"))
print("  wall-clock %.3f s for %d sets x %d steps (%.4f us per step incl. the AXI-Lite control)" % (ttot, nset, cuts[-1], ttot / (nset * cuts[-1]) * 1e6))
print("board check: %s (board = C++ bit for bit at %d checkpoints x 18 words x %d sets: %s; max board - Julia %.3e, limit 5e-8)" % ("PASS" if ok else "FAIL", ncut, nset, "yes" if nbits == 0 else "no", worst))
""")
    end
    println("written ", joinpath(dir, "phmachine4_board.py"))
    flow_config(joinpath(dir, "flow.cfg"); model = "phmachine4", top = "phm4_run", clock_ns = PHM4_CLOCK_NS, ip_version = "1.0",
                ip_name = "PHMachine4 real-time step", probes = ["omega_out" => 48, "ia_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["omega_out" => (48 - PHM4_I.Sig, "omega [pu]"), "ia_out" => (48 - PHM4_I.Sig, "i_alpha [pu]")],
                sample_dt = 32 * PHM4_STEP_CYCLES * PHM4_CLOCK_NS * 1e-9, board_args = "--no-download",
                impl_strategy = "Performance_ExplorePostRoutePhysOpt", step_cycles = PHM4_STEP_CYCLES, strict = true)
    return dir
end
