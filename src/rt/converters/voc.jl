# =====================================================================================================================
# The grid-forming virtual-oscillator converter's controller call (PHVoc, sampled, DC rejection; REALTIME_TODO item 3)
# =====================================================================================================================
# Source: src/phsim/converter.jl, `voc_sample!` (641-655) with `voc_rates` (620-639), `vsm_divider` (149-158, the VSM's
# predictive virtual impedance: the DLL gfm_voc_pf uses the same divider, record section 28), `_held_from` / `_radial_clip`;
# record docs/active/PH_CONVERTER_MODEL.md sections 28-32. Reference configuration: Ts = 50 us, dc_reject (gfm_voc_pf_dcr),
# the physical DC side held (the call's output is the modulation m).
#
# The package's call (simulator frame, turning at omega_b): u_c = the divider with E = v at the measured v_k; then
#   iref = (p_set - j q_set) / conj(v_k) (0 if |v_k|^2 < 1e-6);  rot = v / m (m = |v| of the previous call, a state);
#   Z = i conj(rot);  i_f = dcc (Z - dg V) rot;  dv = xi (V_nom^2 - |v|^2) v + eta J (iref - i_f)
#   v <- e^{j (omega_n - omega_b) Ts} (v + Ts dv);  V <- lam V + Z;  m <- |v| (before the step)
# This file's map holds the oscillator in the frame turning at omega_n: w = v_ab e^{-j 2 pi phi}, phi += Ts f_n per call
# (turns, compensated as the VSM's phase), so the step is w <- w + Ts dv(w) with no rotation (a rounded per-call rotation
# by omega_n Ts would drift the phase ~3e-14 rad per call); the inputs are turned into that frame and the output out of it
# once. Every term of dv is equivariant (a rotation of v, v_k, i together rotates dv; Z, V are in the frame of v), so the
# map is the package's exactly. With phi_1 = clock_1 / 2 pi, w = v_pkg at the first call.
# Hardware form: the VSM's divider (engagement by squares, chord steps certified per instance, ip without a reciprocal);
# 1 / |v_k| and 1 / m by the droop's normalising unit dr_mag!; the DC-rejection filter in increment form V + ((lam - 1) V + Z)
# with lam - 1 on a fine grid; w and V summed with compensation; no divider.
# =====================================================================================================================

const VOC_DIR = @__DIR__
const Dw = Float64           # the oscillator's increment Ts dv (ap_fixed<48, 0>)
const Gx = Float64           # Ts xi   (each gain on its own grid: droop.jl, the gains)
const Ge = Float64           # Ts eta J
const Ir = Float64           # iref and the current terms of dv (ap_fixed<48, 11>)
const Wv = Float64           # a compensated sum's increment, formed once (ap_fixed<112, 16>: wiring)
const Pl = Float64           # the set-points' second words (ap_fixed<48, -41>): a rounded p_set biases the oscillator's
                             # tangential term, i.e. its frequency (open loop: a phase drift)
const VOC_I = Dict{Symbol,Int}(:Sig => 8, :St => 3, :Turn => 2, :Small => -7, :Ang => -6, :Coef => 10, :Zc => 0, :Rq => 4,
                               :Rc => 2, :Tq => -54, :Pf => 6, :Vb => 13, :Gn => -4, :Dcf => 1, :Df => -2, :Dg => 8, :Rm => -30,
                               :Rn => 2, :Ry => 3, :Dw => 0, :Gx => -9, :Ge => -11, :Ir => 11, :Pl => -41, :Wv => 16)
const VOC_VDC = (0.5, 1.99)
const VOC_VKMIN = 0.025      # the call's domain |v_k| >= 0.025 (the replay's minimum is 0.136); below, flagged: in
                             # [2^-7, 0.025) iref along v_k's direction at 0.025, below 2^-7 (zero voltage) iref = 0
const VOC_VK0 = 0.0078125    # 2^-7: below it the magnitude unit gives no direction (review 60, 3)
const VOC_MMIN = 0.125       # m = |v| of the last call >= 1/8 (1 / m <= 8; below: rot = 0, flagged; the replay's minimum 0.33)

voc_data(i::Int; path = joinpath(VOC_DIR, "data", "ieee39_voc.toml")) = TOML.parsefile(path)["converter"][i]
voc_ninstances(; path = joinpath(VOC_DIR, "data", "ieee39_voc.toml")) = length(TOML.parsefile(path)["converter"])

@hls struct VocCoefs
    kz::Zc[10]       # the VSM's divider: r_vi, b = Zs + x_vi, Zs, x_vi, kpr, kpx, c1, A2, ab, unused
    kg::Sig[12]      # the VSM's: i_lim, thr2, y0 (re, im), unused x 4, m_max, m_max^2, v_min, v_max
    kc::Coef[5]      # the VSM's chord start: beta, beta^2, 4 c1, gam0, 1 / (2 c1)
    kl::Gn[4]        # lam - 1 (re, im), dg (re, im)
    kx::Gx[1]        # Ts xi
    ke::Ge[2]        # Ts eta J (re, im)
    kd::Dcf[2]       # dcc (re, im)
    kp::Pf[3]        # p_set, q_set, V_nom^2
    kq::Pl[2]        # p_set, q_set: what their Pf rounding drops
    kv::Sig[3]       # |v_k| floor 0.025, m floor 1/8, zero-voltage floor 2^-7
    ks::Small[1]     # Ts f_n
    kt::Tq[1]        # Ts f_n - its Small rounding
end

"The divider's coefficients as the VSM prepares them (vsm_coefs_big), from a VOC instance."
function voc_coefs_big(d)
    b(k) = BigFloat(d[k])
    bc(k) = complex(BigFloat(d[k][1]), BigFloat(d[k][2]))
    dv = Dict{String,Any}("r_vi" => d["r_vi"], "x_vi" => d["x_vi"], "Zs" => d["Zs"], "kpr" => d["kpr"], "kpx" => d["kpx"],
                          "i_lim" => d["i_lim"], "omega_n" => d["omega_n"], "Ts" => d["Ts"], "u_max" => -1.0, "p_set" => 0.0,
                          "u_set" => 0.0, "m_max" => d["m_max"], "Ta" => 1.0, "Dp" => 0.0, "omega_c" => 0.0, "Tv" => 1.0,
                          "dc_mode" => "held", "name" => d["name"])
    c = vsm_coefs_big(dv)
    Ts = b("Ts"); lam, dg, dcc, J = bc("dlam"), bc("dg"), bc("dcc"), bc("J")
    fn = b("omega_n") / (2 * BigFloat(pi))
    ej = Ts * b("eta") * J
    kg = copy(c.kg); kg[5:8] .= 0
    return (kz = c.kz, kg, kc = c.kc,
            kl = [real(lam) - 1, imag(lam), real(dg), imag(dg)], kx = [Ts * b("xi")], ke = [real(ej), imag(ej)],
            kd = [real(dcc), imag(dcc)], kp = [b("p_set"), b("q_set"), b("V_nom")^2],
            kq = [b("p_set") - BigFloat(_q48(b("p_set"), VOC_I[:Pf])), b("q_set") - BigFloat(_q48(b("q_set"), VOC_I[:Pf]))],
            kv = [BigFloat(VOC_VKMIN), BigFloat(VOC_MMIN), BigFloat(VOC_VK0)], ks = [Ts * fn],
            kt = [Ts * fn - BigFloat(_q48(Ts * fn, VOC_I[:Small]))], c1 = c.kz[7])
end

function voc_coefs(d; check = true)
    (d["dcr"] && d["dc_mode"] == "held") || error("$(d["name"]): the build covers dc_reject and the held modulation")
    c = voc_coefs_big(d)
    c.c1 > 0 || error("$(d["name"]): the virtual impedance does not grow (kpr = kpx = 0): not built")
    Float64(c.kg[2]) >= 2.0^-8 || error("$(d["name"]): the engagement threshold is below the rsqrt range")
    q(v, T) = begin
        I_ = VOC_I[T]
        m = Float64(maximum(abs, v))
        (!check || m < 2.0^(I_ - 1)) || error("$(d["name"]): a coefficient (max |value| $m) does not fit $T = ap_fixed<48, $I_>")
        [_q48(w, I_) for w in v]
    end
    return VocCoefs(q(c.kz, :Zc), q(c.kg, :Sig), q(c.kc, :Coef), q(c.kl, :Gn), q(c.kx, :Gx), q(c.ke, :Ge), q(c.kd, :Dcf),
                    q(c.kp, :Pf), q(c.kq, :Pl), q(c.kv, :Sig), q(c.ks, :Small), q(c.kt, :Tq))
end

function voc_types!()
    mx = Dict(:Gn => 0.0, :Gx => 0.0, :Ge => 0.0)
    for i in 1:voc_ninstances()
        c = voc_coefs_big(voc_data(i))
        for (T, v) in ((:Gn, c.kl), (:Gx, c.kx), (:Ge, c.ke))
            mx[T] = max(mx[T], Float64(maximum(abs, v)))
        end
    end
    for (T, m) in mx
        VOC_I[T] = ceil(Int, log2(m)) + 1
    end
    return mx
end

# ---------------------------------------------------------------------------------------------------------------------
# THE CALL (becomes the C++). ph = (phi) turns, pr = its remainder; xw = (w re, im) the oscillator in the omega_n frame;
# xv = (V re, im) the DC-rejection filter (frame of v); xm = (m = |v| of the last call); xr = remainders (w 2, V 2);
# u = (v_re, v_im, i_re, i_im, v_dc) in alpha-beta; y = the held modulation (alpha-beta); fl = (VI engaged, m clip,
# v_dc outside its domain, |v_k| below its floor, m below its floor)
# ---------------------------------------------------------------------------------------------------------------------
@hls function voc_call!(ph::Turn[1], pr::Ang[1], xw::Sig[2], xv::Vb[2], xm::St[1], xr::Rm[4], u::Sig[5], y::Sig[2],
                        fl::Int[5], k::VocCoefs)
    @pragma "HLS ARRAY_PARTITION variable=xw complete"
    @pragma "HLS ARRAY_PARTITION variable=xv complete"
    @pragma "HLS ARRAY_PARTITION variable=xr complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    @pragma "HLS ARRAY_PARTITION variable=fl complete"
    @pragma "HLS ALLOCATION operation instances=mul limit=6"
    cs::Rc[1024] = rom(:VS_COS)
    sn::Rc[1024] = rom(:VS_SIN)
    pinv::Coef[14] = rom(:DR_PINV)
    p2::Coef[14] = rom(:DR_P2)
    one::Sig = 1
    c40::Coef = 40                                   # 1 / 0.025
    # --- rot = e^{j 2 pi phi} (the omega_n frame), as the VSM
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
    # --- the inputs in the omega_n frame: v_n = v conj(rot), i_n = i conj(rot)
    a1::Sig = u[1] * cph
    a2::Sig = u[2] * sph
    a3::Sig = u[2] * cph
    a4::Sig = u[1] * sph
    vnr::Sig = a1 + a2
    vni::Sig = a3 - a4
    b1::Sig = u[3] * cph
    b2::Sig = u[4] * sph
    b3::Sig = u[4] * cph
    b4::Sig = u[3] * sph
    inr::Sig = b1 + b2
    ini::Sig = b3 - b4
    er::Sig = xw[1]                                  # E = w (the oscillator's voltage)
    ei::Sig = xw[2]
    # --- the VSM's virtual-impedance divider (vsm.jl vsm_call!, unchanged)
    dre::Sig = er - vnr
    dim::Sig = ei - vni
    dr2::Sig = dre * dre
    di2::Sig = dim * dim
    dd2::Sig = dr2 + di2
    on::Int = 0
    if dd2 > k.kg[2]
        on = 1
    end
    zr::Sig = k.kz[1]
    zx::Sig = k.kz[2]
    ipr::Sig = 0
    ipi::Sig = 0
    if on == 1
        rd::Sig = vs_rsqrt(dd2)
        dm::Sig = dd2 * rd
        gm::Sig = dm - k.kc[4]
        q4::Sig = k.kc[3] * gm
        ds::Sig = k.kc[2] + q4
        rs::Sig = vs_rsqrt(ds)
        sq::Sig = ds * rs
        sbt::Sig = sq - k.kc[1]
        ov::Sig = sbt * k.kc[5]
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
        for it in 1:9
            @pragma "HLS PIPELINE off"                   # one shared datapath for the 9 steps (unrolled, it congested the routing)
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
        ir::Sig = k.kg[1] + ov
        qr::Sig = k.kz[5] * ov
        qx::Sig = k.kz[6] * ov
        zr = k.kz[1] + qr
        zx = k.kz[2] + qx
        dhr::Sig = dre * rd
        dhi::Sig = dim * rd
        h2::Sig = dhr * ir
        h3::Sig = dhi * ir
        zir0::Sig = zr * ir
        zix0::Sig = zx * ir
        g1::Sig = zir0 * rd
        g2::Sig = zix0 * rd
        m1::Sig = h2 * g1
        m2::Sig = h3 * g2
        m3::Sig = h3 * g1
        m4::Sig = h2 * g2
        ipr = m1 + m2
        ipi = m3 - m4
    else
        n1::Sig = dre * k.kg[3]
        n2::Sig = dim * k.kg[4]
        n3::Sig = dim * k.kg[3]
        n4::Sig = dre * k.kg[4]
        ipr = n1 - n2
        ipi = n3 + n4
    end
    zex::Sig = zx - k.kz[3]
    w1::Sig = zr * ipr
    w2::Sig = zex * ipi
    w3::Sig = zr * ipi
    w4::Sig = zex * ipr
    zir::Sig = w1 - w2
    zii::Sig = w3 + w4
    ucr::Sig = er - zir                              # u_c in the omega_n frame
    uci::Sig = ei - zii
    # --- the held modulation: m = rot u_c / v_dc, clipped to m_max (as the VSM)
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
    vd2::Sig = vd * vd
    lim2::Sig = k.kg[10] * vd2
    cm::Int = 0
    if uu2 > lim2
        cm = 1
    end
    ru::Sig = vs_rsqrt(uu2)
    gv::Sig = vs_recipv(vd)
    fac::Sig = gv
    if cm == 1
        fac = k.kg[9] * ru
    end
    o1::Sig = cph * ucr
    o2::Sig = sph * uci
    o3::Sig = sph * ucr
    o4::Sig = cph * uci
    uar::Sig = o1 - o2
    uai::Sig = o3 + o4
    y[1] = uar * fac
    y[2] = uai * fac
    # --- iref = (p_set - j q_set) / conj(v_n) = (p_set - j q_set) v_n / |v_n|^2 (the floor |v_k| >= 0.025, flagged)
    nv::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=nv complete"
    yv::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yv complete"
    kv::Int[1]
    dr_mag!(vnr, vni, nv, yv, kv)
    sk::Coef = p2[kv[1] + 7]
    vkm::Sig = yv[2] * sk                            # |v_n| = |n| 2^k
    fvk::Int = 0
    yy2::Rq = yv[1] * yv[1]                          # 1 / |n|^2
    pk::Coef = pinv[kv[1] + 7]
    gk::Coef = yy2 * pk                              # v_n / |v_n|^2 = n 2^k / (|n|^2 4^k) = n y^2 2^-k
    if vkm < k.kv[1]                                 # below the floor: |v_k| taken as 0.025 along its direction (flagged)
        fvk = 1
        gk = yv[1] * c40
        if vkm < k.kv[3]                             # zero voltage (no direction below 2^-7): iref = 0 (the package: 0 below 1e-3)
            gk = 0
        end
    end
    ux::Coef = gk * nv[1]
    uy::Coef = gk * nv[2]
    fr1::Ir = k.kp[1] * ux + k.kq[1] * ux   # two 48 x 48 products (exact sum before the one rounding)
    fr2::Ir = k.kp[2] * uy + k.kq[2] * uy
    fr3::Ir = k.kp[1] * uy + k.kq[1] * uy
    fr4::Ir = k.kp[2] * ux + k.kq[2] * ux
    irr::Ir = fr1 + fr2                              # (p - j q)(ux + j uy)
    iri::Ir = fr3 - fr4
    # --- rot_v = w / m (m = |v| of the last call), Z = i_n conj(rot_v), i_f = dcc (Z - dg V) rot_v
    nm::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=nm complete"
    ym::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=ym complete"
    km::Int[1]
    mz::Sig = 0
    dr_mag!(xm[1], mz, nm, ym, km)
    pm::Coef = pinv[km[1] + 7]
    rm::Sig = 0                                      # 1 / m, formed only in its domain (60b, 2): below the floor rot = 0
    fmm::Int = 1                                     # (the package: rot = 0 for m <= 1e-6; here below 1/8, flagged)
    if xm[1] >= k.kv[2]
        rm = ym[1] * pm
        fmm = 0
    end
    rvr::Sig = er * rm
    rvi::Sig = ei * rm
    z1::Sig = inr * rvr
    z2::Sig = ini * rvi
    z3::Sig = ini * rvr
    z4::Sig = inr * rvi
    zzr::Sig = z1 + z2                               # Z = i_n conj(rot_v)
    zzi::Sig = z3 - z4
    gv1::Sig = k.kl[3] * xv[1]
    gv2::Sig = k.kl[4] * xv[2]
    gv3::Sig = k.kl[3] * xv[2]
    gv4::Sig = k.kl[4] * xv[1]
    fa1::Sig = zzr - (gv1 - gv2)
    fa2::Sig = zzi - (gv3 + gv4)
    hc1::Sig = k.kd[1] * fa1
    hc2::Sig = k.kd[2] * fa2
    hc3::Sig = k.kd[1] * fa2
    hc4::Sig = k.kd[2] * fa1
    hfr::Sig = hc1 - hc2
    hfi::Sig = hc3 + hc4
    j1::Ir = hfr * rvr
    j2::Ir = hfi * rvi
    j3::Ir = hfi * rvr
    j4::Ir = hfr * rvi
    ifr::Ir = j1 - j2                                # i_f
    ifi::Ir = j3 + j4
    # --- dv Ts = Ts xi (V_nom^2 - |w|^2) w + Ts eta J (iref - i_f)
    wqr::Sig = er * er
    wqi::Sig = ei * ei
    wq::Sig = wqr + wqi
    am::Sig = k.kp[3] - wq
    xa::Gn = k.kx[1] * am                            # Ts xi (V_nom^2 - |w|^2), up to 5e-4: on the wider Gn grid (2^-52)
    # --- m <- |w| (before the step)
    nw::Rn[2]
    @pragma "HLS ARRAY_PARTITION variable=nw complete"
    yw::Ry[2]
    @pragma "HLS ARRAY_PARTITION variable=yw complete"
    kw::Int[1]
    dr_mag!(er, ei, nw, yw, kw)
    p2w::Coef = p2[kw[1] + 7]
    xm[1] = yw[2] * p2w                              # |w| = |n| 2^k
    fl[1] = on
    fl[2] = cm
    fl[3] = fdc
    fl[4] = fvk
    fl[5] = fmm
    # --- the phase: Ts f_n, two words, compensated
    inc::Ang = k.ks[1] + k.kt[1] + pr[1]
    pn::Turn = phi + inc
    dd::Ang = pn - phi
    pr[1] = inc - dd
    if pn >= one
        pn = pn - one
    end
    ph[1] = pn
    # --- the compensated sums, each in one expression (as droop.jl): w += Ts dv, V += (lam - 1) V + Z; in C++ exact up to
    # the one rounding of the new word, which the remainder recovers
    v0::Vb = xv[1]
    v1::Vb = xv[2]
    # compensated sums: the increment (with the remainder) formed once on a wide word, then x_new = x + d,
    # e_new = (x - x_new) + d: each wide sum built once (formed twice, as one expression each, they congested the routing)
    sd1::Wv = xa * er + k.ke[1] * (irr - ifr) - k.ke[2] * (iri - ifi) + xr[1]
    n1w::Sig = er + sd1
    xr[1] = (er - n1w) + sd1
    xw[1] = n1w
    sd2::Wv = xa * ei + k.ke[1] * (iri - ifi) + k.ke[2] * (irr - ifr) + xr[2]
    n2w::Sig = ei + sd2
    xr[2] = (ei - n2w) + sd2
    xw[2] = n2w
    sd3::Wv = k.kl[1] * v0 - k.kl[2] * v1 + zzr + xr[3]
    n3v::Vb = v0 + sd3
    xr[3] = (v0 - n3v) + sd3
    xv[1] = n3v
    sd4::Wv = k.kl[1] * v1 + k.kl[2] * v0 + zzi + xr[4]
    n4v::Vb = v1 + sd4
    xr[4] = (v1 - n4v) + sd4
    xv[2] = n4v
end

# ---------------------------------------------------------------------------------------------------------------------
# THE BOARD TOP (becomes the C++): a batch of up to VOC_NB calls of one instance; the eleven state words in and out; per
# call the held output and the packed flags (VI + 2 m clip + 4 v_dc + 8 |v_k| floor + 16 m floor); probes as the VSM's
# ---------------------------------------------------------------------------------------------------------------------
const VOC_NB = 256
@hls function voc_run!(k::VocCoefs, ph::Turn[1], pr::Ang[1], xw::Sig[2], xv::Vb[2], xm::St[1], xr::Rm[4], u::Sig[256, 5],
                       y::Sig[256, 2], fo::Int[256], ncall::Int, m_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=ph"
    @pragma "HLS INTERFACE mode=s_axilite port=pr"
    @pragma "HLS INTERFACE mode=s_axilite port=xw"
    @pragma "HLS INTERFACE mode=s_axilite port=xv"
    @pragma "HLS INTERFACE mode=s_axilite port=xm"
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
    xws::Sig[2]
    @pragma "HLS ARRAY_PARTITION variable=xws complete"
    xvs::Vb[2]
    @pragma "HLS ARRAY_PARTITION variable=xvs complete"
    xms::St[1]
    xrs::Rm[4]
    @pragma "HLS ARRAY_PARTITION variable=xrs complete"
    phs[1] = ph[1]
    prs[1] = pr[1]
    xms[1] = xm[1]
    for i in 1:2
        @pragma "HLS UNROLL"
        xws[i] = xw[i]
        xvs[i] = xv[i]
    end
    for i in 1:4
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
        @pragma "HLS PIPELINE off"   # a call per Ts: an auto-pipelined batch loop unrolls the call (no sharing; 95 % LUT)
        sample[] = false
        for i in 1:5
            uu[i] = u[c, i]
        end
        voc_call!(phs, prs, xws, xvs, xms, xrs, uu, yy, ff, k)
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
    xm[1] = xms[1]
    for i in 1:2
        @pragma "HLS UNROLL"
        xw[i] = xws[i]
        xv[i] = xvs[i]
    end
    for i in 1:4
        @pragma "HLS UNROLL"
        xr[i] = xrs[i]
    end
end

"    voc_state(xc, clk) -> (ph, pr, xw, xv, xm, xr): the call unit's states from the package's at the frame angle clk (phi = clk / 2 pi, w = v)."
voc_state(xc, clk) = ([mod(clk / (2pi), 1.0)], [0.0], [xc[1], xc[2]], [xc[5], xc[6]], [xc[7]], zeros(4))

voc_types() = [T => (T === :Sd ? "ap_ufixed<18, 2>" : T === :Wv ? "ap_fixed<112, 16, AP_RND_CONV>" : "ap_fixed<48, $(VOC_I[T]), AP_RND_CONV>")
               for T in (:Sig, :St, :Turn, :Small, :Ang, :Coef, :Zc, :Rq, :Rc, :Tq, :Pf, :Pl, :Vb, :Gn, :Gx, :Ge, :Ir, :Dcf, :Df, :Dg,
                         :Rm, :Rn, :Ry, :Dw, :Sd, :Wv)]

"Writes src/rt/converters/CPP/voc/ (the call, its units, the board top, the ROMs, the first instance's coefficients)."
function voc_export(; dir = joinpath(VOC_DIR, "CPP", "voc"))
    setprecision(() -> voc_types!(), BigFloat, 256)
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "voc_rom.h"), [:VS_COS, :VS_SIN, :VS_RSQ, :VS_REC, :DR_PINV, :DR_P2]; header = "vsm.jl and droop.jl ROMs")
    hls_write(joinpath(dir, "voc.cpp"), [:VocCoefs, :vs_rsqrt, :vs_recip0, :vs_recipv, :dr_mag!, :voc_call!, :voc_run!];
              types = voc_types(), local_includes = ["rt_util.h", "voc_rom.h"],
              header = "the VOC converter's controller call (PHVoc, sampled, DC rejection); from src/rt/converters/voc.jl")
    ks = [setprecision(() -> voc_coefs(voc_data(j)), BigFloat, 256) for j in 1:voc_ninstances()]
    hls_write_const(joinpath(dir, "voc_coefs.h"), "VOC_COEFS", ks[1])
    println("written $dir (coefficients of all $(length(ks)) VOC converters fit; Gn I=$(VOC_I[:Gn]))")
    return ks
end

"The flow's export (stage 1): the C++, flow.cfg and the board program."
function voc_rt_export(; dir = joinpath(VOC_DIR, "CPP", "voc"))
    ks = voc_export(; dir)
    F(T) = 48 - VOC_I[T]
    flow_config(joinpath(dir, "flow.cfg"); model = "voc", top = "voc_run", clock_ns = 8, ip_version = "1.0",
                ip_name = "VOC converter controller call", probes = ["m_out" => 48, "strobe" => 1, "sample" => 1],
                sample_probe = "sample", plot = ["m_out" => (F(:Sig), "m alpha (held)")], sample_dt = 5e-5,
                board_args = "--no-download", hls_extra = String[], impl_strategy = "Performance_ExplorePostRoutePhysOpt",
                step_cycles = 0, ila_ref = "", ila_check = String[], strict = true)
    fields(k) = [(collect(getfield(k, f)), F(T)) for (f, T) in ((:kz, :Zc), (:kg, :Sig), (:kc, :Coef), (:kl, :Gn), (:kx, :Gx),
                                                               (:ke, :Ge), (:kd, :Dcf), (:kp, :Pf), (:kq, :Pl), (:kv, :Sig), (:ks, :Small), (:kt, :Tq))]
    sw = vcat([("PH", 0, F(:Turn)), ("PR", 0, F(:Ang))], [("XW", i, F(:Sig)) for i in 0:1], [("XV", i, F(:Vb)) for i in 0:1],
              [("XM", 0, F(:St))], [("XR", i, F(:Rm)) for i in 0:3])
    conv_board_script(joinpath(dir, "voc_board.py"); tag = "voc", nb = VOC_NB, sets = [fields(k) for k in ks], sw,
                      names = [voc_data(j)["name"] for j in 1:length(ks)], scalars = ["PH", "PR", "XM"], fsig = F(:Sig))
end
