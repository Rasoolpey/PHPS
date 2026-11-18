# =====================================================================================================================
# The grid-following converter (PHGfl): the DLL's sampled call and the static generator's h-paced current loop, one build
# (REALTIME_TODO item 4; review 48 merge, 1; review 64)
# =====================================================================================================================
# Source: src/phsim/converter.jl, `gfl_call` (774-806), `gfl_sample!` (809-821), `gfl_emf` (827-833), `gfl_rates!` (840-848);
# record docs/active/PH_CONVERTER_MODEL.md sections 36-43. Reference configuration: Ts = 50 us, advance = 1, the physical DC
# side continuous (the loop's EMF u_c is the DC side's command, dcside.jl).
#
# Part 1, the call (every Ts), in alpha-beta: psi = (theta + the frame angle) / 2 pi in turns; v = v_k e^{-j 2 pi psi};
#   omega / omega_n = 1 + (kp vq + ki phi) / omega_n; the phase increment Ts omega / 2 pi = c1 + c2 vq + c3 phi;
#   v_f, f_f lags; dead-band droops; ramps p_r, q_r (clamped at r Ts per call); eP = (p_r + dP) - vd id_l,
#   eQ = (q_r + dQ) + vd iq_l; id_raw = kp_P eP + ki_P gP, iq_raw = -(kp_Q eQ + ki_Q gQ) + Kqf vq; the circular limit
#   i_max; gate = clamp((vd - 0.1) / 0.2, 0, 1); anti-windup aw = sat ? 0 : gate; gP += Ts aw eP, gQ += Ts aw eQ;
#   id_l += (Ts / T_cc)(gate id - id_l), iq_l alike; latched (from the states before the call): ch = e^{j 2 pi (psi +
#   advance Ts omega / 2 pi)}, ir = (id_l, iq_l), omega_call; then psi += Ts omega / 2 pi, phi += Ts vq (clamped +-phi_max).
# Part 2, the current loop (every h): ih = i conj(ch); e = ir - ih; z += h (Kd / Td, Kq / Tq) e (exact midpoint: the rows
#   do not depend on z); u_c = (Kd e_d + z_mid,d - x1 ih_q + j (Kq e_q + z_mid,q + x1 ih_d)) ch at the midpoint.
# Hardware form as the droop / VOC calls: the phase in turns with a two-word constant and a compensated sum; |v_k| and
# |i_ref| by `dr_mag!`; the lags and integrators summed with compensation in one expression; each gain on its own type.
# =====================================================================================================================

const GFL_DIR = @__DIR__
const Gl = Float64           # Ts / T_meas, Ts / T_cc
const Gp = Float64           # c2 = Ts kp / 2 pi, c3 = Ts ki / 2 pi (turns per unit vq, phi)
const Go = Float64           # kp / omega_n, ki / omega_n
const Gc = Float64           # the loop's h Kd / Td, h Kq / Tq (and their halves)
const Zs = Float64           # z_d, z_q (ap_fixed<48, 3>)
const Gt = Float64           # Ts (its own grid: in Pf its rounding was ~2e-9 relative)
const Ai = Float64           # the phase increment (ap_fixed<48, -4>: with |phi| up to phi_max it reaches 7e-3 turn)
const GFL_I = Dict{Symbol,Int}(:Sig => 8, :St => 3, :Turn => 2, :Small => -7, :Ang => -6, :Coef => 10, :Rq => 4, :Rc => 2,
                               :Tq => -54, :Pf => 7, :Rm => -30, :Rn => 2, :Ry => 3, :Zc => 0, :Gl => -4, :Gp => -10, :Go => -3,
                               :Gc => -8, :Zs => 3, :Gt => -13, :Ai => -4)

gfl_data(i::Int; path = joinpath(GFL_DIR, "data", "ieee39_gfl.toml")) = TOML.parsefile(path)["converter"][i]
gfl_ninstances(; path = joinpath(GFL_DIR, "data", "ieee39_gfl.toml")) = length(TOML.parsefile(path)["converter"])

@hls struct GflCoefs
    kc::Small[1]     # c1 = Ts f_n (the phase increment's constant)
    kt::Tq[1]        # c1 - its Small rounding
    kp::Gp[2]        # c2 = Ts kp_pll / 2 pi, c3 = Ts ki_pll / 2 pi
    ko::Go[2]        # kp_pll / omega_n, ki_pll / omega_n (omega / omega_n - 1)
    kl::Gl[2]        # Ts / T_meas, Ts / T_cc
    kg::Pf[15]       # p_ref, q_ref, r_p Ts, r_q Ts, kp_P, ki_P, kp_Q, ki_Q, Kqf, k_droop_p, f_db, k_droop_q, v_db, i_max, phi_max
    kts::Gt[1]       # Ts
    kz::Zc[2]        # Kd, Kq (x1 below)
    kx::Zc[1]        # x1
end
@hls struct GflLoop
    kh::Gc[4]        # h Kd / Td, h Kq / Tq, their halves
    kz::Zc[3]        # Kd, Kq, x1
end

function gfl_coefs_big(d)
    b(k) = BigFloat(d[k])
    Ts = b("Ts"); fn = b("omega_n") / (2 * BigFloat(pi)); tp = 2 * BigFloat(pi)
    return (kc = [Ts * fn], kt = [Ts * fn - BigFloat(_q48(Ts * fn, GFL_I[:Small]))],
            kp = [Ts * b("kp_pll") / tp, Ts * b("ki_pll") / tp], ko = [b("kp_pll") / b("omega_n"), b("ki_pll") / b("omega_n")],
            kl = [Ts / b("T_meas"), Ts / b("T_cc")],
            kg = [b("p_ref"), b("q_ref"), b("r_p_max") * Ts, b("r_q_max") * Ts, b("kp_P"), b("ki_P"), b("kp_Q"), b("ki_Q"), b("Kqf"),
                  b("k_droop_p"), b("f_db"), b("k_droop_q"), b("v_db"), b("i_max"), b("phi_max")], kts = [Ts],
            kz = [b("Kd"), b("Kq")], kx = [b("x1")])
end
gfl_loop_big(d, h) = (b(k) = BigFloat(d[k]); H = BigFloat(h);
    (kh = [H * b("Kd") / b("Td"), H * b("Kq") / b("Tq"), H * b("Kd") / b("Td") / 2, H * b("Kq") / b("Tq") / 2],
     kz = [b("Kd"), b("Kq"), b("x1")]))

function gfl_q(v, T, name; check = true)
    I_ = GFL_I[T]
    m = Float64(maximum(abs, v))
    (!check || m < 2.0^(I_ - 1)) || error("$name: a coefficient (max |value| $m) does not fit $T = ap_fixed<48, $I_>")
    return [_q48(w, I_) for w in v]
end
function gfl_coefs(d; check = true)
    d["dc_mode"] == "continuous" || error("$(d["name"]): the build covers the continuous DC side")
    c = gfl_coefs_big(d); q(v, T) = gfl_q(v, T, d["name"]; check)
    return GflCoefs(q(c.kc, :Small), q(c.kt, :Tq), q(c.kp, :Gp), q(c.ko, :Go), q(c.kl, :Gl), q(c.kg, :Pf), q(c.kts, :Gt), q(c.kz, :Zc),
                    q(c.kx, :Zc))
end
gfl_loop(d, h; check = true) = (c = gfl_loop_big(d, h); GflLoop(gfl_q(c.kh, :Gc, d["name"]; check), gfl_q(c.kz, :Zc, d["name"]; check)))

function gfl_types!(h)
    mx = Dict(:Gl => 0.0, :Gp => 0.0, :Go => 0.0, :Gc => 0.0)
    for i in 1:gfl_ninstances()
        d = gfl_data(i); c = gfl_coefs_big(d); l = gfl_loop_big(d, h)
        for (T, v) in ((:Gl, c.kl), (:Gp, c.kp), (:Go, c.ko), (:Gc, l.kh))
            mx[T] = max(mx[T], Float64(maximum(abs, v)))
        end
    end
    for (T, m) in mx
        GFL_I[T] = ceil(Int, log2(m)) + 1
    end
    return mx
end

# ---------------------------------------------------------------------------------------------------------------------
# THE CALL (becomes the C++). ph = (psi) turns, pr = its remainder; xs = (phi, v_f, f_f, p_r, q_r, gP, gQ, id_l, iq_l);
# xr = remainders (v_f, f_f, gP, gQ, id_l, iq_l); lt = the latched words (ch re, ch im, ir_d, ir_q, omega_call); u = (v_re,
# v_im) in alpha-beta at the call; fl = (sat, gate < 1, gate = 0, phi clamped)
# ---------------------------------------------------------------------------------------------------------------------
@hls function gfl_call!(ph::Turn[1], pr::Ang[1], xs::Pf[9], xr::Rm[6], lt::Sig[5], u::Sig[2], fl::Int[4], k::GflCoefs)
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    @pragma "HLS ARRAY_PARTITION variable=xr complete"
    @pragma "HLS ARRAY_PARTITION variable=lt complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    @pragma "HLS ALLOCATION operation instances=mul limit=6"
    cs::Rc[1024] = rom(:VS_COS)
    sn::Rc[1024] = rom(:VS_SIN)
    p2::Coef[14] = rom(:DR_P2)
    pinv::Coef[14] = rom(:DR_PINV)
    one::Sig = 1
    zero::Sig = 0
    five::Sig = 5
    tenth::Sig = 0.1
    # --- e^{-j 2 pi psi}: the ROM at the bin centre and the remainder's rotation (as the VSM)
    phi0::Turn = ph[1]
    jb::Int = ifloor(scale2(phi0, 10))
    if jb > 1023
        jb = 1023
    end
    hf::Turn = 0.5
    ft::Turn = scale2(phi0, 10) - jb
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
    a1::Sig = u[1] * cph
    a2::Sig = u[2] * sph
    a3::Sig = u[2] * cph
    a4::Sig = u[1] * sph
    vd::Sig = a1 + a2                                # v = v_k e^{-j 2 pi psi}
    vq::Sig = a3 - a4
    # --- the old states
    phs::Pf = xs[1]
    vf0::Pf = xs[2]
    ff0::Pf = xs[3]
    pr0::Pf = xs[4]
    qr0::Pf = xs[5]
    gp0::Pf = xs[6]
    gq0::Pf = xs[7]
    id0::Pf = xs[8]
    iq0::Pf = xs[9]
    # --- omega / omega_n and the phase increment
    om1::Sig = k.ko[1] * vq
    om2::Sig = k.ko[2] * phs
    omn::Sig = one + om1 + om2                       # omega / omega_n
    # --- |v_k| (the v_f lag's input)
    nv::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=nv complete"
    yv::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yv complete"
    kv::Int[1]
    # |v_k| feeds the v_f lag directly, so it must hold below dr_mag!'s 2^-7 (a bolted fault): small vectors are shifted up
    # by 2^11 (2^22 below 2^-17, 2^33 below 2^-28) first and the magnitude shifted back (exact shifts; the shifted vector
    # stays below 32): every nonzero Sig value (>= 2^-40) lands in dr_mag!'s accurate range (review 66, 1: a held |v| =
    # 2^-30 biased the v_f lag by 2.9e-10 with two stages)
    va::Sig = u[1]
    if va < 0
        va = -va
    end
    vb::Sig = u[2]
    if vb < 0
        vb = -vb
    end
    vmx::Sig = va
    if vb > va
        vmx = vb
    end
    lo1::Sig = 0.015625
    lo2::Sig = 7.62939453125e-6
    lo3::Sig = 0.0000000037252902984619140625
    us1::Sig = u[1]
    us2::Sig = u[2]
    vsc::Int = 0
    if vmx < lo1
        us1 = scale2(u[1], 11)
        us2 = scale2(u[2], 11)
        vsc = 1
    end
    if vmx < lo2
        us1 = scale2(u[1], 22)
        us2 = scale2(u[2], 22)
        vsc = 2
    end
    if vmx < lo3
        us1 = scale2(u[1], 33)
        us2 = scale2(u[2], 33)
        vsc = 3
    end
    dr_mag!(us1, us2, nv, yv, kv)
    sv::Coef = p2[kv[1] + 7]
    vks::Sig = yv[2] * sv                            # the (shifted) magnitude
    vkm::Sig = vks
    if vsc == 1
        vkm = scale2(vks, -11)
    end
    if vsc == 2
        vkm = scale2(vks, -22)
    end
    if vsc == 3
        vkm = scale2(vks, -33)
    end
    # --- the droops (dead bands; zero gains in the case), the ramps
    ev::Pf = vf0 - one
    dbv::Pf = 0
    if ev > k.kg[13]
        dbv = ev - k.kg[13]
    end
    if ev < zero - k.kg[13]
        dbv = ev + k.kg[13]
    end
    dq0::Pf = zero - k.kg[12] * dbv
    ef::Pf = ff0 - one
    dbf::Pf = 0
    if ef > k.kg[11]
        dbf = ef - k.kg[11]
    end
    if ef < zero - k.kg[11]
        dbf = ef + k.kg[11]
    end
    dp0::Pf = zero - k.kg[10] * dbf
    rp::Pf = k.kg[1] - pr0
    if rp > k.kg[3]
        rp = k.kg[3]
    end
    if rp < zero - k.kg[3]
        rp = zero - k.kg[3]
    end
    rq::Pf = k.kg[2] - qr0
    if rq > k.kg[4]
        rq = k.kg[4]
    end
    if rq < zero - k.kg[4]
        rq = zero - k.kg[4]
    end
    # --- the outer PI, the circular limit, the gate
    ep1::Pf = vd * id0
    eP::Pf = pr0 + dp0 - ep1
    eq1::Pf = vd * iq0
    eQ::Pf = qr0 + dq0 + eq1
    idr::Sig = k.kg[5] * eP + k.kg[6] * gp0
    iqr::Sig = k.kg[9] * vq - (k.kg[7] * eQ + k.kg[8] * gq0)
    ni::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=ni complete"
    yi::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yi complete"
    ki::Int[1]
    dr_mag!(idr, iqr, ni, yi, ki)
    si::Coef = p2[ki[1] + 7]
    ima::Sig = yi[2] * si                            # |i_ref|
    sat::Int = 0
    idc::Sig = idr
    iqc::Sig = iqr
    if ima > k.kg[14]
        sat = 1
        pin::Coef = pinv[ki[1] + 7]
        fs::Rq = k.kg[14] * yi[1] * pin              # i_max / |i_ref|
        idc = idr * fs
        iqc = iqr * fs
    end
    gt::Sig = five * (vd - tenth)                    # (vd - 0.1) / 0.2
    gp::Int = 0
    if gt < one
        gp = 1
    end
    if gt > one
        gt = one
    end
    gz::Int = 0
    if gt < zero
        gt = zero
        gz = 1
    end
    aw::Sig = gt
    if sat == 1
        aw = zero
    end
    # --- latched: ch at psi + advance (the increment), ir = (id_l, iq_l) before the step, omega_call
    inc::Ai = k.kc[1] + k.kt[1] + k.kp[1] * vq + k.kp[2] * phs
    pha::Turn = phi0 + inc                           # advance = 1 (checked at export)
    if pha >= one
        pha = pha - one
    end
    if pha < zero                                    # the increment is negative for phi near -phi_max
        pha = pha + one
    end
    jh::Int = ifloor(scale2(pha, 10))
    if jh > 1023
        jh = 1023
    end
    fth::Turn = scale2(pha, 10) - jh
    fhh::Turn = fth - hf
    dlh::Turn = scale2(fhh, -10)
    drh::Ang = twopi * dlh
    d2h::Ang = drh * drh
    sah::Ang = d2h * s20
    sbh::Rq = one - sah
    sch::Ang = d2h * s6
    sdh::Ang = sch * sbh
    seh::Rq = one - sdh
    sdlh::Ang = drh * seh
    cah::Ang = d2h * s12
    cbh::Rq = one - cah
    cch::Ang = scale2(d2h, -1)
    cdh::Ang = cch * cbh
    cdlh::Rq = one - cdh
    c0h::Rc = cs[jh + 1]
    s0h::Rc = sn[jh + 1]
    u1::Sig = c0h * cdlh
    u2::Sig = s0h * sdlh
    u3::Sig = s0h * cdlh
    u4::Sig = c0h * sdlh
    lt[1] = u1 - u2
    lt[2] = u3 + u4
    lt[3] = id0
    lt[4] = iq0
    lt[5] = omn
    fl[1] = sat
    fl[2] = gp
    fl[3] = gz
    # --- the step: psi (compensated), phi (clamped), the lags and integrators (compensated, one expression each)
    inp::Ai = inc + pr[1]
    pn::Turn = phi0 + inp
    dd::Ai = pn - phi0
    pr[1] = inp - dd
    if pn >= one
        pn = pn - one
    end
    if pn < zero
        pn = pn + one
    end
    ph[1] = pn
    phn::Pf = phs + k.kts[1] * vq
    fpc::Int = 0
    if phn > k.kg[15]
        phn = k.kg[15]
        fpc = 1
    end
    if phn < zero - k.kg[15]
        phn = zero - k.kg[15]
        fpc = 1
    end
    xs[1] = phn
    fl[4] = fpc
    n2::Pf = vf0 + k.kl[1] * (vkm - vf0) + xr[1]
    xr[1] = vf0 - n2 + k.kl[1] * (vkm - vf0) + xr[1]
    xs[2] = n2
    n3::Pf = ff0 + k.kl[1] * (omn - ff0) + xr[2]
    xr[2] = ff0 - n3 + k.kl[1] * (omn - ff0) + xr[2]
    xs[3] = n3
    xs[4] = pr0 + rp
    xs[5] = qr0 + rq
    n6::Pf = gp0 + k.kts[1] * aw * eP + xr[3]
    xr[3] = gp0 - n6 + k.kts[1] * aw * eP + xr[3]
    xs[6] = n6
    n7::Pf = gq0 + k.kts[1] * aw * eQ + xr[4]
    xr[4] = gq0 - n7 + k.kts[1] * aw * eQ + xr[4]
    xs[7] = n7
    n8::Pf = id0 + k.kl[2] * (gt * idc - id0) + xr[5]
    xr[5] = id0 - n8 + k.kl[2] * (gt * idc - id0) + xr[5]
    xs[8] = n8
    n9::Pf = iq0 + k.kl[2] * (gt * iqc - iq0) + xr[6]
    xr[6] = iq0 - n9 + k.kl[2] * (gt * iqc - iq0) + xr[6]
    xs[9] = n9
end

# ---------------------------------------------------------------------------------------------------------------------
# THE CURRENT LOOP STEP (becomes the C++): z = (z_d, z_q), zr its remainders; lt the latched (ch, ir); i = the injected
# current at the step's midpoint (alpha-beta); y = u_c at the midpoint (alpha-beta, the DC side's command)
# ---------------------------------------------------------------------------------------------------------------------
@hls function gfl_loop!(z::Zs[2], zr::Rm[2], lt::Sig[5], i::Sig[2], y::Sig[2], kq::GflLoop)
    @pragma "HLS ARRAY_PARTITION variable=z complete"
    @pragma "HLS ARRAY_PARTITION variable=zr complete"
    @pragma "HLS ARRAY_PARTITION variable=lt complete"
    @pragma "HLS ARRAY_PARTITION variable=i complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    h1::Sig = i[1] * lt[1]
    h2::Sig = i[2] * lt[2]
    h3::Sig = i[2] * lt[1]
    h4::Sig = i[1] * lt[2]
    ihd::Sig = h1 + h2                               # ih = i conj(ch)
    ihq::Sig = h3 - h4
    ed::Sig = lt[3] - ihd
    eq::Sig = lt[4] - ihq
    zd0::Zs = z[1]
    zq0::Zs = z[2]
    ud::Sig = kq.kz[1] * ed + zd0 + kq.kh[3] * ed + zr[1] - kq.kz[3] * ihq    # u at the midpoint (frame of ch)
    uq::Sig = kq.kz[2] * eq + zq0 + kq.kh[4] * eq + zr[2] + kq.kz[3] * ihd
    w1::Sig = ud * lt[1]
    w2::Sig = uq * lt[2]
    w3::Sig = ud * lt[2]
    w4::Sig = uq * lt[1]
    y[1] = w1 - w2
    y[2] = w3 + w4
    nd::Zs = zd0 + kq.kh[1] * ed + zr[1]
    zr[1] = zd0 - nd + kq.kh[1] * ed + zr[1]
    z[1] = nd
    nq::Zs = zq0 + kq.kh[2] * eq + zr[2]
    zr[2] = zq0 - nq + kq.kh[2] * eq + zr[2]
    z[2] = nq
end

# ---------------------------------------------------------------------------------------------------------------------
# THE BOARD TOP (becomes the C++): one build, two modes (review 48, 1; 64): mode 0 a batch of up to GFL_NL current-loop
# steps (one pipelined loop, h = GFL_II x 8 ns), mode 1 a batch of up to GFL_NC calls (latency-bound); every state word in and
# out (the host carries them between batches and modes); probes: u_out (the loop's u_c alpha part), a strobe toggled per
# loop step or call, a sample pulse every decim loop steps
# ---------------------------------------------------------------------------------------------------------------------
const GFL_NL = 512
const GFL_NC = 64
const GFL_II = 4      # the loop at II 4 = 32 ns (II 2: 77.8 MHz est.; synth_gfl_3 / 4 sweeps)
@hls function gfl_run!(k::GflCoefs, kq::GflLoop, mode::Int, ph::Turn[1], pr::Ang[1], xs::Pf[9], xr::Rm[6], lt::Sig[5],
                       z::Zs[2], zr::Rm[2], uc::Sig[64, 2], fo::Int[64], il::Sig[512, 2], yl::Sig[512, 2], n::Int, decim::Int,
                       u_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=kq"
    @pragma "HLS INTERFACE mode=s_axilite port=mode"
    @pragma "HLS INTERFACE mode=s_axilite port=ph"
    @pragma "HLS INTERFACE mode=s_axilite port=pr"
    @pragma "HLS INTERFACE mode=s_axilite port=xs"
    @pragma "HLS INTERFACE mode=s_axilite port=xr"
    @pragma "HLS INTERFACE mode=s_axilite port=lt"
    @pragma "HLS INTERFACE mode=s_axilite port=z"
    @pragma "HLS INTERFACE mode=s_axilite port=zr"
    @pragma "HLS INTERFACE mode=s_axilite port=uc"
    @pragma "HLS INTERFACE mode=s_axilite port=fo"
    @pragma "HLS INTERFACE mode=s_axilite port=il"
    @pragma "HLS INTERFACE mode=s_axilite port=yl"
    @pragma "HLS INTERFACE mode=s_axilite port=n"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=u_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    phs::Turn[1]
    prs::Ang[1]
    xss::Pf[9]
    @pragma "HLS ARRAY_PARTITION variable=xss complete"
    xrs::Rm[6]
    @pragma "HLS ARRAY_PARTITION variable=xrs complete"
    lts::Sig[5]
    @pragma "HLS ARRAY_PARTITION variable=lts complete"
    zs::Zs[2]
    @pragma "HLS ARRAY_PARTITION variable=zs complete"
    zrs::Rm[2]
    @pragma "HLS ARRAY_PARTITION variable=zrs complete"
    phs[1] = ph[1]
    prs[1] = pr[1]
    for i in 1:9
        @pragma "HLS UNROLL"
        xss[i] = xs[i]
    end
    for i in 1:6
        @pragma "HLS UNROLL"
        xrs[i] = xr[i]
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        lts[i] = lt[i]
    end
    for i in 1:2
        @pragma "HLS UNROLL"
        zs[i] = z[i]
        zrs[i] = zr[i]
    end
    st::Bool = false
    cnt::Int = 0
    ii::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=ii complete"
    yy::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    uu::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=uu complete"
    ff::Int[4]
    @pragma "HLS ARRAY_PARTITION variable=ff complete"
    if mode == 0
        for s in 1:n
            @pragma "HLS PIPELINE II=4"
            @pragma "HLS LOOP_TRIPCOUNT min=1 max=512"
            ii[1] = il[s, 1]
            ii[2] = il[s, 2]
            gfl_loop!(zs, zrs, lts, ii, yy, kq)
            yl[s, 1] = yy[1]
            yl[s, 2] = yy[2]
            u_out[] = yy[1]
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
    else
        for c in 1:n
            @pragma "HLS LOOP_TRIPCOUNT min=1 max=64"
            uu[1] = uc[c, 1]
            uu[2] = uc[c, 2]
            gfl_call!(phs, prs, xss, xrs, lts, uu, ff, k)
            fo[c] = ff[1] + 2 * ff[2] + 4 * ff[3] + 8 * ff[4]
            st = !st
            strobe[] = st
        end
    end
    ph[1] = phs[1]
    pr[1] = prs[1]
    for i in 1:9
        @pragma "HLS UNROLL"
        xs[i] = xss[i]
    end
    for i in 1:6
        @pragma "HLS UNROLL"
        xr[i] = xrs[i]
    end
    for i in 1:5
        @pragma "HLS UNROLL"
        lt[i] = lts[i]
    end
    for i in 1:2
        @pragma "HLS UNROLL"
        z[i] = zs[i]
        zr[i] = zrs[i]
    end
end

"    gfl_state(xc, clk) -> (ph, pr, xs, xr, lt, z, zr): the build's states from the package's at the frame angle clk."
function gfl_state(xc, clk)
    ch = complex(xc[11], xc[12]) * cis(clk)          # the held frame in alpha-beta
    return ([mod((xc[1] + clk) / (2pi), 1.0)], [0.0], Float64.(xc[2:10]), zeros(6), [real(ch), imag(ch), xc[13], xc[14], xc[17]],
            [xc[15], xc[16]], zeros(2))
end

gfl_types() = [T => (T === :Sd ? "ap_ufixed<18, 2>" : "ap_fixed<48, $(GFL_I[T]), AP_RND_CONV>")
               for T in (:Sig, :St, :Turn, :Small, :Ang, :Coef, :Rq, :Rc, :Tq, :Pf, :Rm, :Rn, :Ry, :Zc, :Gl, :Gp, :Go, :Gc, :Zs, :Gt, :Ai, :Sd)]

"Writes src/rt/converters/CPP/gfl/ at h = GFL_II x 8 ns (the call, the loop, the units, the board top, the ROMs)."
function gfl_export(; ii = GFL_II, dir = joinpath(GFL_DIR, "CPP", "gfl"))
    occursin("II=$(GFL_II)\"", string(HLS_DEFS[:gfl_run!])) || error("the PIPELINE II in gfl_run! differs from GFL_II")
    h = setprecision(() -> big(ii) * 8 / big(10)^9, BigFloat, 256)
    setprecision(() -> gfl_types!(h), BigFloat, 256)
    all(gfl_data(j)["advance"] == 1.0 for j in 1:gfl_ninstances()) || error("the build latches ch at psi + 1 x the increment (advance = 1)")
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "gfl_rom.h"), [:VS_COS, :VS_SIN, :VS_RSQ, :DR_PINV, :DR_P2]; header = "vsm.jl and droop.jl ROMs")
    hls_write(joinpath(dir, "gfl.cpp"), [:GflCoefs, :GflLoop, :dr_mag!, :gfl_call!, :gfl_loop!, :gfl_run!];
              types = gfl_types(), local_includes = ["rt_util.h", "gfl_rom.h"],
              header = "the GFL converter (PHGfl): the sampled call and the h-paced current loop, h = $(Float64(h)) s; from src/rt/converters/gfl.jl")
    ks = [setprecision(() -> gfl_coefs(gfl_data(j)), BigFloat, 256) for j in 1:gfl_ninstances()]
    kl = [setprecision(() -> gfl_loop(gfl_data(j), h), BigFloat, 256) for j in 1:gfl_ninstances()]
    hls_write_const(joinpath(dir, "gfl_coefs.h"), "GFL_COEFS", ks[1])
    println("written $dir at h = $(Float64(h)) s (II $ii); coefficients of all $(length(ks)) GFL converters fit; ",
            join(("$T I=$(GFL_I[T])" for T in (:Gl, :Gp, :Go, :Gc)), ", "))
    return ks, kl, h
end

"The flow's export (stage 1); the board program and flow.cfg are added by gfl_board_export (the board stage)."
gfl_rt_export() = (gfl_export(); isdefined(@__MODULE__, :gfl_board_export) ? gfl_board_export() : nothing)

"flow.cfg and the board program (gfl_board.py): the event sequences of gfl_board_data.bin, the six converters interleaved per call period (each a mode-0 batch of its steps, then its call), every u_c and every call's 26 words and flags against the C++ (bit for bit) and Julia (5e-8); --timing: one mode-0 batch of 512 steps for the every-clock capture."
function gfl_board_export(; dir = joinpath(GFL_DIR, "CPP", "gfl"))
    h = setprecision(() -> big(GFL_II) * 8 / big(10)^9, BigFloat, 256)
    F(T) = 48 - GFL_I[T]
    flow_config(joinpath(dir, "flow.cfg"); model = "gfl", top = "gfl_run", clock_ns = 8, ip_version = "1.0",
                ip_name = "GFL converter (call and current loop)", probes = ["u_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["u_out" => (F(:Sig), "u_c alpha")], sample_dt = 32 * Float64(h),
                board_args = "--no-download", hls_extra = String[], impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = 0, ila_ref = "", ila_check = String[], strict = true)
    ks = [setprecision(() -> gfl_coefs(gfl_data(j)), BigFloat, 256) for j in 1:gfl_ninstances()]
    kl = [setprecision(() -> gfl_loop(gfl_data(j), h), BigFloat, 256) for j in 1:gfl_ninstances()]
    cf(k) = [(collect(getfield(k, f)), F(T)) for (f, T) in ((:kc, :Small), (:kt, :Tq), (:kp, :Gp), (:ko, :Go), (:kl, :Gl),
                                                            (:kg, :Pf), (:kts, :Gt), (:kz, :Zc), (:kx, :Zc))]
    lf(k) = [(collect(getfield(k, f)), F(T)) for (f, T) in ((:kh, :Gc), (:kz, :Zc))]
    py(fs) = "[" * join(("(" * repr(v) * ", $f)" for (v, f) in fs), ", ") * "]"
    sw = vcat([("PH", 0, F(:Turn)), ("PR", 0, F(:Ang))], [("XS", i, F(:Pf)) for i in 0:8], [("XR", i, F(:Rm)) for i in 0:5],
              [("LT", i, F(:Sig)) for i in 0:4], [("Z", i, F(:Zs)) for i in 0:1], [("ZR", i, F(:Rm)) for i in 0:1])
    open(joinpath(dir, "gfl_board.py"), "w") do io
        print(io, """
# gfl_board.py: written by PHPS.RT.gfl_board_export (src/rt/converters/gfl.jl); do not edit.
# Runs the GFL IP (one build, two modes) on the PYNQ-Z2: the 0.3 s event sequences of the six converters (gfl_board_data.bin),
# interleaved per call period (each converter: a mode-0 batch of its current-loop steps, then its call), coefficients and the 26
# state words reloaded every batch. Every u_c and every call's state words and flags against the C++ (bit for bit) and Julia
# (5e-8). --timing: one mode-0 batch of 512 steps (the every-clock capture).
#   python3 gfl_board.py [--no-download] [--timing]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
LIM, FSIG = 5e-8, $(F(:Sig))
SETS = [$(join((py(cf(k)) for k in ks), ",\n        "))]
LOOPS = [$(join((py(lf(k)) for k in kl), ",\n         "))]
SW = $(repr(sw))
A = {}
for m in re.finditer(r"#define XGFL_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xgfl_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
def reg(nm, io):
    return A[nm + "_" + io] if nm + "_" + io in A else A[nm]
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("gfl.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.gfl_run_0
mem = ip.mmio.array
def raw(v, F):
    return np.round(np.asarray(v, dtype=np.float64) * 2.0**F).astype(np.int64)
def wvec(base, vals, F):
    r = raw(vals, F) & ((1 << 48) - 1); o = base // 4
    mem[o:o + 2 * len(r):2] = (r & 0xFFFFFFFF).astype(np.uint32); mem[o + 1:o + 2 * len(r):2] = (r >> 32).astype(np.uint32)
def rvec(base, n):
    o = base // 4
    lo = mem[o:o + 2 * n:2].astype(np.int64); hi = mem[o + 1:o + 2 * n:2].astype(np.int64) & 0xFFFF
    r = lo | (hi << 32)
    return np.where(r >> 47, r - (1 << 48), r)
def wfields(base, fields):
    for vals, F in fields:
        wvec(base, vals, F); base += 8 * len(vals)
FS = np.array([F for (_, _, F) in SW], dtype=np.float64)
def wstate(v):
    for (nm, i, F), x in zip(SW, v):
        wvec((reg(nm, "I") if nm in ("PH", "PR") else A[nm]) + 8 * i, [x], F)
def rstate():
    return np.array([rvec((reg(nm, "O") if nm in ("PH", "PR") else A[nm]) + 8 * i, 1)[0] for (nm, i, F) in SW])
def run(mode, n):
    ip.write(A["MODE"], mode); ip.write(A["N"], n); ip.write(A["DECIM"], 32)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass

D = np.fromfile("gfl_board_data.bin", dtype=np.float64)
nconv = int(D[0]); p = 1; conv = []
for j in range(nconv):
    s, nev = int(D[p]), int(D[p + 1]); init = D[p + 2:p + 28].copy(); p += 28
    steps, calls, order = [], [], []
    for e in range(nev):
        if D[p] == 0:
            steps.append(D[p + 1:p + 7]); order.append(0); p += 7
        else:
            calls.append(D[p + 1:p + 57]); order.append(1); p += 57   # v (2), C++ state (26), C++ flags, Julia state (26), Julia flags
    conv.append(dict(s=s, st=init, steps=np.array(steps), calls=np.array(calls), order=order, ps=0, pc=0, dC=0, dS=0, dJ=0.0, nf=0))
if "--timing" in sys.argv:
    c = conv[0]; wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]]); wstate(c["st"])
    wvec(A["IL"], c["steps"][0:512, 0:2].reshape(-1), FSIG); run(0, 512)
    print("timing batch: 512 current-loop steps of converter 1"); sys.exit(0)
if "--timing-call" in sys.argv:   # a mode-1 batch of 64 calls of converter 1 (the inputs of its first 64 calls; timing only)
    c = conv[0]; wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]]); wstate(c["st"])
    wvec(A["UC"], c["calls"][0:64, 0:2].reshape(-1), FSIG); run(1, 64)
    print("timing batch: 64 calls of converter 1"); sys.exit(0)
t_all = 0.0; nb = 0
for period in range(max(len(c["calls"]) for c in conv) + 1):
    for c in conv:
        # the steps up to the next call (or the end), then that call
        o = c["order"]; i0 = c["ps"] + c["pc"]
        if i0 >= len(o):
            continue
        ta = time.perf_counter()
        wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]])
        n = 0
        while i0 + n < len(o) and o[i0 + n] == 0:
            n += 1
        k = 0
        while k < n:
            m = min(512, n - k); S = c["steps"][c["ps"]:c["ps"] + m]
            wstate(c["st"]); wvec(A["IL"], S[:, 0:2].reshape(-1), FSIG); run(0, m)
            Y = rvec(A["YL"], 2 * m).reshape(m, 2); c["st"] = rstate() / 2.0**FS; nb += 1
            c["dC"] = max(c["dC"], int(np.max(np.abs(Y - raw(S[:, 2:4], FSIG)))))
            c["dJ"] = max(c["dJ"], float(np.max(np.abs(Y / 2.0**FSIG - S[:, 4:6]) / np.maximum(1.0, np.abs(S[:, 4:6])))))
            c["ps"] += m; k += m
        if i0 + n < len(o):
            C = c["calls"][c["pc"]]
            wstate(c["st"]); wvec(A["UC"], C[0:2], FSIG); run(1, 1)
            rs = rstate(); fo = int(mem[A["FO"] // 4]); nb += 1
            c["dS"] = max(c["dS"], int(np.max(np.abs(rs - raw(C[2:28], FS)))))
            c["nf"] += int(fo != int(C[28])) + int(fo != int(C[55]))
            c["st"] = rs / 2.0**FS; c["pc"] += 1
        t_all += time.perf_counter() - ta
ok = True; wC = wS = 0; wJ = 0.0; nfa = 0
for j, c in enumerate(conv):
    pj = c["dC"] == 0 and c["dS"] == 0 and c["dJ"] <= LIM and c["nf"] == 0
    print("converter %d: %d steps, %d calls: u_c board - C++ %d LSB, call states (26 words) board - C++ %d LSB, board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, c["ps"], c["pc"], c["dC"], c["dS"], c["dJ"], LIM, c["nf"], "PASS" if pj else "FAIL"))
    ok &= pj; wC = max(wC, c["dC"]); wS = max(wS, c["dS"]); wJ = max(wJ, c["dJ"]); nfa += c["nf"]
print("interleaved: %d batches (mode 0 and mode 1), the six converters in turn per call period; host time %.1f s" % (nb, t_all))
print("board check: max |board - Julia| %.3e (limit %.0e), u_c board - C++ %d LSB, call states board - C++ %d LSB, flag differences %d: %s"
      % (wJ, LIM, wC, wS, nfa, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
""")
    end
    println("written $dir/gfl_board.py and flow.cfg")
end
