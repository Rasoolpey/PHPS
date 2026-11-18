# =====================================================================================================================
# PHMachine (full-winding synchronous machine of the 464-state EMT model) in the real-time style, for the FPGA
# =====================================================================================================================
#
# Started from src/rt/TEMPLATE.jl and src/rt/machines/genrou.jl (2026-10-27). Plan: docs/realtime/docs/PHMACHINE_PLAN.md
# (draft 2, design review 28). Loaded by src/rt/RT.jl; `julia --project=. scripts/realtime/export_cpp.jl phmachine`
# writes the C++, the coefficients, the testbench, the board program and flow.cfg to src/rt/machines/CPP/phmachine/.
# =====================================================================================================================

import TOML
using LinearAlgebra: Diagonal, I

# ---------------------------------------------------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------------------------------------------------
# Source:      PHMachine (src/phsim/system.jl, header lines 21-24; WindingMachine in src/emdq/pilot.jl), the same
#              equations; data and operating point of the 464-state IEEE-39 case (src/rt/machines/data/
#              ieee39_machines.toml, written by scripts/realtime/phmachine_extract.jl).
# Frame:       the port in the stationary network frame (alpha, beta); the machine in its rotor frame at the absolute
#              angle theta = delta + theta_f, carried as (cos theta, sin theta).
# States x:    psi = omega_b lam for (d, q, fd, 1d, 1q, 2q) (pu), omega (pu), (cos, sin) of the step's MIDPOINT angle
#              theta^_n = theta_n + h omega_b a_n / 2; the predictor's history psi_d, psi_q of the previous step;
#              a_n (omega_n in a hold step, else the previous midpoint speed); the test bench's source phasor
#              (cos, sin) at the next step's midpoint                                                  (14 entries)
# Inputs u:    v_alpha, v_beta (network voltage at the step's midpoint), v_f, T_m                       (4 inputs)
# Outputs y:   i_alpha, i_beta: the current injected into the network at the step's midpoint (i_net / conj(a))
# Equations:   (lam = psi / omega_b, i = K psi, K = blockdiag(Kd, Kq), motor convention)
#              lam_d' = v_d - Ra i_d + omega_b omega lam_q,   lam_q' = v_q - Ra i_q - omega_b omega lam_d
#              lam_fd' = v_f - Rfd i_fd, lam_1d' = -R1d i_1d, lam_1q' = -R1q i_1q, lam_2q' = -R2q i_2q
#              2H omega' = T_m + omega_b (lam_d i_q - lam_q i_d),   theta' = omega_b omega
#              v_dq = j e^{-j theta} v / a,   i_net = -j e^{j theta} (i_d + j i_q)
# Step:        the frozen-skew linearly implicit midpoint (FPGA_SOLVER_METHOD.md section 3; plan section 2):
#              the stator coupling frozen at the predicted midpoint fluxes psi^ = (3 psi_n - psi_{n-1}) / 2, the speed
#              implicit; solved with prepared per-axis matrices and one reciprocal 1/s = 1 - (s - 1). The Park
#              transform at theta^ = theta_n + h omega_b a_n / 2. After an input edge the predictor is held
#              for one step (psi^ = psi_n, a_n = omega_n).
# Nonlinear:   no sin/cos unit (the midpoint and source phasors are rotated by small angles, series to d^3), no divider.
# Speed:       restructured for a short loop-carried chain (docs/realtime/models/PHMACHINE.md section 7, reviews 32-32c):
#              the midpoint phasor carried as a state, prepared current/torque products, 1/s folded into q, the
#              rotation's source renormalised at the start of the step. Longest loop: 5 products in a row (was 14).

const PHMACHINE_RT_NX = 14


# ---------------------------------------------------------------------------------------------------------------------
# 2. Parameters (the case's data)
# ---------------------------------------------------------------------------------------------------------------------
"""
    phmachine_rt_data(name = "GENROU_3"; path) -> Dict

One machine of the 464-state IEEE-39 model from `src/rt/machines/data/ieee39_machines.toml`, with `omega_b`.
"""
function phmachine_rt_data(name = "GENROU_3"; path = joinpath(@__DIR__, "data", "ieee39_machines.toml"))
    t = TOML.parsefile(path)
    i = findfirst(m -> m["name"] == name, t["machine"])
    i === nothing && error("no machine $name in $path")
    d = t["machine"][i]
    d["cut"] == 0 || error("$name sits at a reduced node (cut > 0): not in this build (plan section 1)")
    d["a_im"] == 0 || error("$name: complex ratio a not in this build")
    return merge(d, Dict("omega_b" => t["omega_b"]))
end


# ---------------------------------------------------------------------------------------------------------------------
# 3. Discretisation (preparation, BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""
    phmachine_rt_coefs(d, h) -> NamedTuple of BigFloat

Per axis (d group: d, fd, 1d; q group: q, 1q, 2q), with K the inverse inductance matrix, R the resistances and
A = -omega_b R K: M = I - (h/2) A, P = h M^-1 A, the voltage columns c = h omega_b M^-1 e_1 (and wf = h omega_b M^-1 e_2
for v_f on the d axis); the stator rows of K; 1/a; h/2H and h/8H; h omega_b and its half. A machine with one q damper
has its q group padded with zeros.
"""
function phmachine_rt_coefs(d, h)
    B(k) = BigFloat(d[k])
    wb = BigFloat(d["omega_b"])
    Ll, Lad, Laq = B("Ll"), B("Lad"), B("Laq")
    Ld = [Ll+Lad Lad Lad; Lad B("Lfd")+Lad Lad; Lad Lad B("L1d")+Lad]
    Kd = inv(Ld)
    if d["nq"] == 2
        Kq = inv([Ll+Laq Laq Laq; Laq B("L1q")+Laq Laq; Laq Laq B("L2q")+Laq])
    else
        Kq = zeros(BigFloat, 3, 3)
        Kq[1:2, 1:2] = inv([Ll+Laq Laq; Laq B("L1q")+Laq])
    end
    Rd = Diagonal([B("Ra"), B("Rfd"), B("R1d")])
    Rq = Diagonal([B("Ra"), B("R1q"), B("R2q")])
    axis(R, K) = (A = -wb * R * K; Mi = inv(Matrix{BigFloat}(I, 3, 3) - (h / 2) * A); (h * Mi * A, h * wb * Mi))
    Pd, Nd = axis(Rd, Kd)
    Pq, Nq = axis(Rq, Kq)
    hq = h / (2 * B("H"))
    return (; Pd, Pq, cd = Nd[:, 1], cq = Nq[:, 1], wf = Nd[:, 2], kd = Kd[1, :], kq = Kq[1, :],
            ainv = 1 / B("a_re"), hq, hq4 = hq / 4, wbh = h * wb, wbh2 = h * wb / 2)
end


# ---------------------------------------------------------------------------------------------------------------------
# 4. The coefficients of the step (the C++ struct PHMachineCoefs)
# ---------------------------------------------------------------------------------------------------------------------
# number types of this model besides Sig, Small, Coef (in Julia Float64, as the others; C++ formats in
# PHMACHINE_RT_TYPES; sized over all ten IEEE-39 machines at h = 1 us, PHMACHINE.md section 7.4)
const St = Float64       # the states x (+-4, 32 x finer than Sig: Sig's grid on omega failed C sim run 3, review 33c)
const Ang = Float64      # angles of one step: kappa, b, Delta
const Mid = Float64      # mid-size coefficients: alpha, bf, K c
const Rate = Float64     # h / 2H and q
const Beta = Float64     # beta = (h / 2H)(h / 8H) K c
const Beta2 = Float64    # beta psi^2

@hls struct PHMachineCoefs
    Pd::Small[3, 3]     # d group step matrix (d, fd, 1d)
    Pq::Small[3, 3]     # q group step matrix (q, 1q, 2q)
    cd::Small[3]        # d group: column of the stator voltage v_d (and of the speed coupling)
    cq::Small[3]        # q group: column of the stator voltage v_q (and of the speed coupling)
    cdv::Small[3]       # cd / a: the column of the network voltage's d component (the stator sees v / a)
    cqv::Small[3]       # cq / a
    wf::Small[3]        # d group: column of the field voltage v_f
    ad::Coef[3]         # kd + Pd' kd / 2: i_d of (psi + y / 2) from psi
    aq::Coef[3]         # kq + Pq' kq / 2
    alphad::Mid         # kd . cdv / 2: i_d of (psi + y / 2) per unit v_d
    alphaq::Mid         # kq . cqv / 2
    bf::Mid             # kd . wf / 2: i_d of (psi + y / 2) per unit v_f
    kcd::Mid            # kd . cd: stator current of the speed coupling per unit psi_q^
    kcq::Mid            # kq . cq
    kcd2::Mid           # kd . cd / 2
    kcq2::Mid           # kq . cq / 2
    ainv::Coef          # 1 / a (the injected current is i_net / a)
    hq::Rate            # h / 2H
    betad::Beta         # (h / 2H)(h / 8H) kd . cd: q = hq - betad psi_q^2 - betaq psi_d^2
    betaq::Beta         # (h / 2H)(h / 8H) kq . cq
    kap1::Ang           # 1.5 h omega_b: the midpoint angle's increment per unit omega_m (normal next step)
    kap2::Ang           # 2 h omega_b (the next step holds)
    wbh2::Ang           # h omega_b / 2
end

# the C++ formats: ap_fixed<48, I>, range +-2^(I-1), grid 2^-(48 - I); every coefficient and input is rounded to its
# type's grid here, so the Julia run and the C++ use the same values (coefficient rounding is then no difference)
const PHMACHINE_RT_I = (Sig = 8, St = 3, Small = -10, Coef = 8, Ang = -9, Mid = -2, Rate = -24, Beta = -59, Beta2 = -55)
const PHMACHINE_RT_F = map(I -> 48 - I, PHMACHINE_RT_I)
const PHMACHINE_RT_FIELDS = [:Pd => :Small, :Pq => :Small, :cd => :Small, :cq => :Small, :cdv => :Small,
                             :cqv => :Small, :wf => :Small, :ad => :Coef, :aq => :Coef, :alphad => :Mid,
                             :alphaq => :Mid, :bf => :Mid, :kcd => :Mid, :kcq => :Mid, :kcd2 => :Mid, :kcq2 => :Mid,
                             :ainv => :Coef, :hq => :Rate, :betad => :Beta, :betaq => :Beta, :kap1 => :Ang,
                             :kap2 => :Ang, :wbh2 => :Ang]       # the struct's order (the board program writes it so)
_q(v, F) = round(Float64(v) * 2.0^F) / 2.0^F

"""
    phmachine_rt_coefstruct(d, h; grid = true) -> PHMachineCoefs

The step's coefficients from the BigFloat preparation; each checked against its type's range and, with `grid`,
rounded to its grid (`grid = false`: plain Float64, for comparing formulations).
"""
function phmachine_rt_coefstruct(d, h; grid = true)
    c = phmachine_rt_coefs(d, h)
    cdv, cqv = c.cd * c.ainv, c.cq * c.ainv
    kcd, kcq = sum(c.kd .* c.cd), sum(c.kq .* c.cq)
    v = Dict(:Pd => c.Pd, :Pq => c.Pq, :cd => c.cd, :cq => c.cq, :cdv => cdv, :cqv => cqv, :wf => c.wf,
             :ad => c.kd + transpose(c.Pd) * c.kd / 2, :aq => c.kq + transpose(c.Pq) * c.kq / 2,
             :alphad => sum(c.kd .* cdv) / 2, :alphaq => sum(c.kq .* cqv) / 2, :bf => sum(c.kd .* c.wf) / 2,
             :kcd => kcd, :kcq => kcq, :kcd2 => kcd / 2, :kcq2 => kcq / 2, :ainv => c.ainv, :hq => c.hq,
             :betad => c.hq * c.hq4 * kcd, :betaq => c.hq * c.hq4 * kcq, :kap1 => 3 * c.wbh / 2, :kap2 => 2 * c.wbh,
             :wbh2 => c.wbh2)
    args = map(PHMACHINE_RT_FIELDS) do (f, T)
        I = getfield(PHMACHINE_RT_I, T)
        m = Float64(maximum(abs, v[f]))
        m < 2.0^(I - 1) || error("coefficient $f (max |value| $m) does not fit $T = ap_fixed<48, $I>")
        grid ? _q.(v[f], 48 - I) : Float64.(v[f])
    end
    return PHMachineCoefs(args...)
end


# ---------------------------------------------------------------------------------------------------------------------
# 5. THE STEP FUNCTION (this function becomes the C++ / the hardware)
# ---------------------------------------------------------------------------------------------------------------------
@hls function phmachine_step!(x::St[14], u::Sig[4], y::Sig[2], hold::Bool, hold_next::Bool, k::PHMachineCoefs)
    @pragma "HLS ARRAY_PARTITION variable=x complete"
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    @pragma "HLS ARRAY_PARTITION variable=y complete"
    ph_d::Sig = (3 * x[1] - x[10]) / 2               # predicted stator fluxes at the step's midpoint
    ph_q::Sig = (3 * x[2] - x[11]) / 2
    if hold                                          # the step after an input edge: no extrapolation
        ph_d = x[1]
        ph_q = x[2]
    end
    mph_d::Sig = -ph_d
    c::St = x[8]                                     # (cos, sin) of the step's midpoint angle theta^_n (on the St
    s::St = x[9]                                     # grid, as the state)
    gn::St = (3 - (c * c + s * s)) / 2               # the rotation's source renormalised (off the speed path)
    cn::St = c * gn
    sn::St = s * gn
    sixth::Sig = 0.16666666666666666
    cn6::Sig = cn * sixth
    sn6::Sig = sn * sixth
    kap::Ang = k.kap1                                # next midpoint: Delta = kap omega_m - b (PHMACHINE.md 7.2)
    asum::Sig = x[12]                                # b = h omega_b a_n / 2
    if hold_next                                     # the next step holds: a_{n+1} = omega_{n+1} = 2 omega_m - omega_n
        kap = k.kap2
        asum = x[7] + x[12]                          # b = h omega_b (omega_n + a_n) / 2
    end
    b::Ang = k.wbh2 * asum

    vd::Sig = s * u[1] - c * u[2]                    # Park transform at the midpoint: j e^{-j theta} v (the 1/a
    vq::Sig = c * u[1] + s * u[2]                    # is in the coefficients cdv, cqv)

    xd::Sig[3] = (x[1], x[3], x[4])                  # d group (d, fd, 1d), q group (q, 1q, 2q)
    xq::Sig[3] = (x[2], x[5], x[6])
    yd::Ang[3]                                       # the step's increment at frozen speed: y = P psi + W u, summed
    yq::Ang[3]                                       # in the fine Ang (in Sig the sums' rounding biased psi_fd by
    for i in 1:3                                     # one Sig grid per step: C sim run 1, review 33b)
        sd::Ang = k.cdv[i] * vd + k.wf[i] * u[3]
        sq::Ang = k.cqv[i] * vq
        for j in 1:3
            sd += k.Pd[i, j] * xd[j]
            sq += k.Pq[i, j] * xq[j]
        end
        yd[i] = sd
        yq[i] = sq
    end

    D::Sig = k.bf * u[3]                             # stator currents of (psi + y/2) without the v_dq terms
    Q::Sig = 0
    for j in 1:3
        D += k.ad[j] * xd[j]
        Q += k.aq[j] * xq[j]
    end
    ibd::Sig = D + k.alphad * vd                     # stator currents of (psi + y/2)
    ibq::Sig = Q + k.alphaq * vq
    a1d::Sig = k.alphad * u[1]                       # the torque with the midpoint phasor entering once:
    a2d::Sig = k.alphad * u[2]                       # T1 = ph_q D - ph_d Q + s X + c Y
    a1q::Sig = k.alphaq * u[1]
    a2q::Sig = k.alphaq * u[2]
    X::Sig = ph_q * a1d - ph_d * a2q
    Y::Sig = mph_d * a1q - ph_q * a2d
    phq2::Sig = ph_q * ph_q
    phd2::Sig = ph_d * ph_d
    wkd::Mid = x[7] * k.kcd2                         # omega_n / 2 times K c
    wkq::Mid = x[7] * k.kcq2
    T1::Sig = ph_q * D - ph_d * Q + s * X + c * Y    # a' K (psi + y/2)
    t3::Sig = wkd * phq2 + wkq * phd2                # omega_n / 2 a' K r
    f::Sig = u[4] - T1 - t3                          # T_m - a' K (psi + y/2 + omega_0 r / 2)
    bs::Beta2 = k.betad * phq2 + k.betaq * phd2
    q::Rate = k.hq - bs                              # h / 2H / s with 1/s = 1 - (s - 1)
    dw::Small = q * f                                # speed increment
    wm::Sig = x[7] + dw / 2                          # midpoint speed

    ird::Sig = k.kcd * ph_q                          # midpoint stator currents, to the network frame
    irq::Sig = k.kcq * mph_d
    hwm::Sig = wm / 2
    idm::Sig = ibd + hwm * ird
    iqm::Sig = ibq + hwm * irq
    t4::Sig = s * idm + c * iqm                      # i_net = -j e^{j theta} (i_d + j i_q), over a
    t5::Sig = s * iqm - c * idm
    y[1] = t4 * k.ainv
    y[2] = t5 * k.ainv

    dl::Ang = kap * wm - b                           # rotate the midpoint phasor to the next midpoint
    dl2::Small = dl * dl
    hdl2::Small = dl2 / 2
    sdl::Small = sn6 * dl
    cdl::Small = cn6 * dl
    c1::St = cn - cn * hdl2 - sn * dl + sdl * dl2    # series of cos to Delta^2, sin to Delta^3 (St: review 33c, 4)
    s1::St = sn - sn * hdl2 + cn * dl - cdl * dl2

    ecd::Mid[3]                                      # the speed coupling's columns times psi^, before omega_m
    ecq::Mid[3]
    for i in 1:3
        ecd[i] = k.cd[i] * ph_q
        ecq[i] = k.cq[i] * mph_d
    end
    x[10] = x[1]                                     # predictor history
    x[11] = x[2]
    x[1] = x[1] + yd[1] + ecd[1] * wm                # new fluxes: psi + y + omega_m r
    x[3] = x[3] + yd[2] + ecd[2] * wm
    x[4] = x[4] + yd[3] + ecd[3] * wm
    x[2] = x[2] + yq[1] + ecq[1] * wm
    x[5] = x[5] + yq[2] + ecq[2] * wm
    x[6] = x[6] + yq[3] + ecq[3] * wm
    x[7] = x[7] + dw
    x[8] = c1
    x[9] = s1
    x[12] = wm                                       # a_{n+1}
    if hold_next
        x[12] = x[7]
    end
end


# ---------------------------------------------------------------------------------------------------------------------
# 5b. THE BOARD TOP FUNCTION (the IP that runs on the FPGA)
# ---------------------------------------------------------------------------------------------------------------------
# Test bench: an infinite bus in the stationary frame, v = vmag e^{j (omega_b t + phase)}, its phasor at the next step's
# midpoint carried in x[13], x[14] and rotated each step by e^{j omega_b h} = 1 - cm + j srot (cm = 1 - cos omega_b h,
# srot = sin omega_b h, both in the fine type Small, so the source frequency is exact to ~1e-13); the magnitude is
# x vdip during steps n_on+1 .. n_off (global step numbers n = n0 + i, so a run can be continued in segments).
@hls function phmachine_run!(k::PHMachineCoefs, x::St[14], vmag::Sig, cm::Small, srot::Small, vf::Sig, tm::Sig,
                             vdip::Sig, n0::Int, n_on::Int, n_off::Int, nsteps::Int, decim::Int, omega_out::Out{Sig},
                             ia_out::Out{Sig}, ib_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
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
    xs::St[14]                                       # the states, kept in registers during the run
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    for i in 1:14
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
        @pragma "HLS PIPELINE II=40"                 # one step per 40 cycles (PHMACHINE_RT_STEP_CYCLES; sweep in
        @pragma "HLS ALLOCATION operation instances=mul limit=16"   # PHMACHINE.md 7.6: 16 multipliers, 153 DSP;
                                                     # 22 (207 DSP, 94 %) failed routing at 125 MHz, WNS -0.8 ns)
        n::Int = n0 + i
        vs::Sig = vmag
        if n > n_on && n <= n_off
            vs = vmag * vdip
        end
        hold::Bool = false                           # the step after an edge of the dip holds the predictor
        if n == n_on + 1
            hold = true
        end
        if n == n_off + 1
            hold = true
        end
        hold_next::Bool = false                      # and the step before it prepares the held midpoint angle
        if n == n_on
            hold_next = true
        end
        if n == n_off
            hold_next = true
        end
        u[1] = vs * xs[13]
        u[2] = vs * xs[14]
        u[3] = vf
        u[4] = tm
        phmachine_step!(xs, u, y, hold, hold_next, k)
        sc1::Sig = xs[13] - xs[13] * cm - xs[14] * srot   # the source phasor to the next step's midpoint
        ss1::Sig = xs[14] - xs[14] * cm + xs[13] * srot
        gs::Sig = (3 - (sc1 * sc1 + ss1 * ss1)) / 2
        xs[13] = sc1 * gs
        xs[14] = ss1 * gs
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
        omega_out[] = xs[7]                          # to the ILA
        ia_out[] = y[1]
        ib_out[] = y[2]
        strobe[] = st
    end
    for i in 1:14
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
end


# ---------------------------------------------------------------------------------------------------------------------
# 6. Exporting the C++
# ---------------------------------------------------------------------------------------------------------------------
# Real time as GENROU (option A): h = PHMACHINE_RT_STEP_CYCLES x 8 ns, the II of the C synthesis; as fast as possible
# (the user, 2026-10-27): h follows the hardware, 1 us is only the upper limit. The PIPELINE II in phmachine_run! must
# equal PHMACHINE_RT_STEP_CYCLES (checked at export); the coefficients are exported for that h.
const PHMACHINE_RT_CLOCK_NS = 8
const PHMACHINE_RT_STEP_CYCLES = 40
const PHMACHINE_RT_H = "$(PHMACHINE_RT_STEP_CYCLES * PHMACHINE_RT_CLOCK_NS)e-9"

const PHMACHINE_RT_TYPES = [:Sig => "ap_fixed<48, 8, AP_RND_CONV>",       # signals: +-128 pu, grid 9.1e-13
                            :St => "ap_fixed<48, 3, AP_RND_CONV>",        # the states: +-4, grid 2.8e-14
                            :Small => "ap_fixed<48, -10, AP_RND_CONV>",   # h-scaled coefficients: +-4.9e-4, 3.5e-18
                            :Coef => "ap_fixed<48, 8, AP_RND_CONV>",      # signal-size coefficients: as Sig
                            # (the user, 2026-10-27: 48 bits everywhere; PHMACHINE.md 7.4 for the formats below)
                            :Ang => "ap_fixed<48, -9, AP_RND_CONV>",      # +-9.8e-4, 6.9e-18
                            :Mid => "ap_fixed<48, -2, AP_RND_CONV>",      # +-0.125, 8.9e-16
                            :Rate => "ap_fixed<48, -24, AP_RND_CONV>",    # +-3.0e-8, 2.1e-22
                            :Beta => "ap_fixed<48, -59, AP_RND_CONV>",    # +-8.7e-19, 6.2e-33
                            :Beta2 => "ap_fixed<48, -55, AP_RND_CONV>"]   # +-1.4e-17, 9.9e-32

"""
    phmachine_rt_initial(d, h) -> (x0, bench)

The initial state at the case's operating point (synchronous frame at t = 0, so theta_0 = delta_0; the midpoint angle
theta^_0 = delta_0 + h omega_b omega_0 / 2, a_0 = omega_0) and the test bench's source: |v|, the rotation
e^{j omega_b h}, v_f, T_m. The source phasor starts at the first step's midpoint.
"""
function phmachine_rt_initial(d, h)
    wb = d["omega_b"]
    lam = d["lam"]
    psi = wb .* (d["nq"] == 2 ? lam : vcat(lam, 0.0))
    om = d["p"] / (2 * d["H"])
    v = complex(d["v_re"], d["v_im"])
    ph = angle(v) + wb * h / 2
    th = d["delta"] + wb * h * om / 2
    x0 = vcat(psi, om, cos(th), sin(th), psi[1], psi[2], om, cos(ph), sin(ph))
    F = PHMACHINE_RT_F                                 # on the C++ grids (section 4)
    return _q.(x0, F.St), (; vmag = _q(abs(v), F.Sig), cm = _q(2 * sin(wb * h / 2)^2, F.Small),
                             srot = _q(sin(wb * h), F.Small), vf = _q(d["vf"], F.Sig), tm = _q(d["Tm"], F.Sig))
end

# the checkpoints of the dip run: every 0.01 s and around both edges (the step before, the hold_next step, the
# hold step, the step after; PHMACHINE.md 7.5)
# (every 0.01 s rounded on its own, so the last is exactly 0.3 s: review 33, 4)
_phmachine_rt_cuts(h, nseg, non, noff) =
    sort(unique(vcat([round(Int, g * 0.01 / h) for g in 1:nseg], non-1:non+2, noff-1:noff+2)))

# the dip case of the testbench and of the board program: the Julia run of phmachine_run! (Float64), cut at `cuts`;
# the states after each cut and the outputs (i_alpha, i_beta) of the cut's last step
function _phmachine_rt_julia(k, x0, b, cuts; vdip = 0.5, non, noff)
    x = copy(x0); rec = [copy(x)]; outs = NTuple{2,Float64}[]
    w, ia, ib, st, sa = Ref(0.0), Ref(0.0), Ref(0.0), Ref(false), Ref(false)
    n = 0
    for c in cuts
        phmachine_run!(k, x, b.vmag, b.cm, b.srot, b.vf, b.tm, vdip, n, non, noff, c - n, 32, w, ia, ib, st, sa)
        push!(rec, copy(x)); push!(outs, (ia[], ib[])); n = c
    end
    return rec, outs
end

function phmachine_rt_testbench(path, k, x0, b; h, cuts, non, noff, vdip = 0.5)
    rec, outs = _phmachine_rt_julia(k, x0, b, cuts; vdip, non, noff)
    f(v) = repr(Float64(v))
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.phmachine_rt_testbench (src/rt/machines/phmachine.jl); do not edit
        // C simulation: the machine at its operating point on an infinite bus (stationary frame), the bus voltage x $vdip
        // for steps $(non + 1)-$noff, $(cuts[end]) steps of h = $h s through the board top function, cut at $(length(cuts))
        // checkpoints (every 0.01 s and around both edges); at each, the C++ states and outputs next to the same @hls
        // code run in Julia (Float64). Limits (PHMACHINE.md 7.5, fixed before any result): states and outputs 5e-8,
        // phasor lengths 1e-10, a continuous run equal to the segmented one bit for bit. Returns 1 if any fails.
        #include <cstdio>
        #include <cmath>
        #include "phmachine.h"
        #include "phmachine_coefs.h"

        const int NCUT = $(length(cuts)), N_ON = $non, N_OFF = $noff;
        const int CUTS[NCUT] = {$(join(cuts, ", "))};
        const double H = $(f(h)), VMAG = $(f(b.vmag)), CM = $(f(b.cm)), SROT = $(f(b.srot)), VF = $(f(b.vf)), TM = $(f(b.tm)), VDIP = $(f(vdip));
        const double LIM_X = 5e-8, LIM_NORM = 1e-10;
        const double X0[14] = {$(join(map(f, x0), ", "))};
        const double JULIA[NCUT + 1][9] = {   // psi_d, psi_q, psi_fd, psi_1d, psi_1q, psi_2q, omega, cos, sin (midpoint)
        $(join(("    {" * join(map(f, r[1:9]), ", ") * "}" for r in rec), ",\n"))
        };
        const double JOUT[NCUT][2] = {        // i_alpha, i_beta of each segment's last step
        $(join(("    {" * f(o[1]) * ", " * f(o[2]) * "}" for o in outs), ",\n"))
        };

        int main() {
            St x[14], xc[14];
            for (int i = 0; i < 14; i++) { x[i] = X0[i]; xc[i] = X0[i]; }
            Sig w, ia, ib; bool st, sa;
            double wx = 0, wo = 0, wn = 0;
            printf("%8s %8s %14s %14s %12s %12s %12s\\n", "step", "t [s]", "omega C++", "omega Julia", "max|dx|", "max|dy|", "max|norm-1|");
            int prev = 0;
            for (int g = 0; g < NCUT; g++) {
                phmachine_run(PHMACHINE_COEFS, x, VMAG, CM, SROT, VF, TM, VDIP, prev, N_ON, N_OFF, CUTS[g] - prev, 32, w, ia, ib, st, sa);
                prev = CUTS[g];
                const double *J = JULIA[g + 1];
                double dx = 0, dy, dn;
                for (int i = 0; i < 9; i++) dx = fmax(dx, fabs(x[i].to_double() - J[i]));
                dy = fmax(fabs(ia.to_double() - JOUT[g][0]), fabs(ib.to_double() - JOUT[g][1]));
                double c = x[7].to_double(), s = x[8].to_double(), sc = x[12].to_double(), ss = x[13].to_double();
                dn = fmax(fabs(c * c + s * s - 1), fabs(sc * sc + ss * ss - 1));
                printf("%8d %8.5f %14.10f %14.10f %12.3e %12.3e %12.3e\\n", CUTS[g], CUTS[g] * H, x[6].to_double(), J[6], dx, dy, dn);
                wx = fmax(wx, dx); wo = fmax(wo, dy); wn = fmax(wn, dn);
            }
            phmachine_run(PHMACHINE_COEFS, xc, VMAG, CM, SROT, VF, TM, VDIP, 0, N_ON, N_OFF, CUTS[NCUT - 1], 32, w, ia, ib, st, sa);
            int same = 1;
            for (int i = 0; i < 14; i++) same = same && (xc[i] == x[i]);
            int fail = !(wx <= LIM_X && wo <= LIM_X && wn <= LIM_NORM && same);
            printf("C sim check: max|dx| %.3e, max|dy| %.3e (limit %.0e), max|norm-1| %.3e (limit %.0e), continuous = segmented %s: %s\\n",
                   wx, wo, LIM_X, wn, LIM_NORM, same ? "yes" : "NO", fail ? "FAIL" : "PASS");
            return fail;
        }
        """)
    end
    println("written $path")
    return path
end

function phmachine_rt_board_script(path, k, x0, b; h, non, noff, vdip = 0.5, decim = 32)
    cuts = [non, round(Int, 0.15 / h), round(Int, 0.3 / h)]         # compare at 0.1 s, 0.15 s (in the dip), 0.3 s
    rec, _ = _phmachine_rt_julia(k, x0, b, cuts; vdip, non, noff)
    refs = [r[1:9] for r in rec[2:end]]
    coefs = Tuple{Float64,Int,Int}[]                                 # struct order, one 64-bit word each
    for (fld, T) in PHMACHINE_RT_FIELDS
        v = getfield(k, fld); F = getfield(PHMACHINE_RT_F, T)
        vals = v isa AbstractMatrix ? [v[i, j] for i in 1:size(v, 1) for j in 1:size(v, 2)] : collect(v)
        foreach(e -> push!(coefs, (e, F, 48)), vals)
    end
    f(v) = repr(Float64(v))
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.phmachine_rt_board_script (src/rt/machines/phmachine.jl); do not edit.
        # Runs the PHMachine IP on the PYNQ-Z2. Next to it: phmachine.bit, phmachine.hwh and the IP's register map
        # xphmachine_run_hw.h (from the HLS driver; the flow copies it). Run as root on the board.
        #   python3 phmachine_board.py [--no-download]   (--no-download keeps the bitstream already loaded, for the ILA)
        import sys, time, re, math
        from pynq import Overlay, Clocks

        H = $(f(h))
        COEFS = [$(join(("($(f(v)), $F, $W)" for (v, F, W) in coefs), ", "))]   # (value, fraction bits, width)
        X0 = [$(join(map(f, x0), ", "))]
        VMAG, CM, SROT, VF, TM, VDIP = $(f(b.vmag)), $(f(b.cm)), $(f(b.srot)), $(f(b.vf)), $(f(b.tm)), $(f(vdip))
        N_ON, N_OFF, DECIM, CUTS = $non, $noff, $decim, $cuts
        X_JULIA = [$(join(("[" * join(map(f, r), ", ") * "]" for r in refs), ", "))]
        NAMES = ["psi_d", "psi_q", "psi_fd", "psi_1d", "psi_1q", "psi_2q", "omega", "cos th^", "sin th^"]

        A = {}                                 # register offsets from the HLS driver header
        for m in re.finditer(r"#define XPHMACHINE_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xphmachine_run_hw.h").read()):
            A[m.group(1)] = int(m.group(3), 16)

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("phmachine.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / PHMACHINE_RT_CLOCK_NS)) > 0.01:
            Clocks.fclk0_mhz = $(1000 / PHMACHINE_RT_CLOCK_NS)
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.phmachine_run_0

        def raw(v, F, W):
            return int(round(v * 2**F)) & ((1 << W) - 1)
        def w48(addr, v, F=40, W=48):         # a W-bit value with F fraction bits in a 64-bit slot
            r = raw(v, F, W)
            ip.write(addr, r & 0xFFFFFFFF)
            ip.write(addr + 4, r >> 32)
        def r48(addr, F=40):
            r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
            if r >> 47:
                r -= 1 << 48
            return r / 2**F

        for i, (v, F, W) in enumerate(COEFS):
            w48(A["K"] + 8 * i, v, F, W)
        for name, v in (("VMAG", VMAG), ("VF", VF), ("TM", TM), ("VDIP", VDIP)):
            w48(A[name], v)
        w48(A["CM"], CM, $(PHMACHINE_RT_F.Small)); w48(A["SROT"], SROT, $(PHMACHINE_RT_F.Small))

        def run(x, n0, nsteps, n_on, n_off):
            for i, v in enumerate(x):
                w48(A["X"] + 8 * i, v, $(PHMACHINE_RT_F.St))
            ip.write(A["N0"], n0); ip.write(A["N_ON"], n_on); ip.write(A["N_OFF"], n_off)
            ip.write(A["NSTEPS"], nsteps); ip.write(A["DECIM"], DECIM)
            t0 = time.perf_counter()
            ip.write(0x00, 1)
            while not (ip.read(0x00) & 0x2):
                time.sleep(0.001)
            t = time.perf_counter() - t0
            return t, [r48(A["X"] + 8 * i, $(PHMACHINE_RT_F.St)) for i in range(14)]

        print("run 1: voltage dip to", VDIP, "pu for steps", N_ON + 1, "-", N_OFF, "(0.1 s to 0.2 s), compared at", [round(c * H, 3) for c in CUTS], "s")
        x, n, ttot = list(X0), 0, 0.0
        for g, c in enumerate(CUTS):
            t, x = run(x, n, c - n, N_ON, N_OFF)
            ttot += t
            print("  t = %.3f s: max |board - Julia| over the 9 states %.3e" % (c * H, max(abs(x[i] - X_JULIA[g][i]) for i in range(9))))
            n = c
        print("  wall-clock time %.4f s (simulated %.4f s)" % (ttot, CUTS[-1] * H))
        print("  %-7s %16s %16s %12s" % ("state", "board", "Julia", "difference"))
        for i in range(9):
            print("  %-7s %16.10f %16.10f %12.3e" % (NAMES[i], x[i], X_JULIA[-1][i], x[i] - X_JULIA[-1][i]))

        N2 = 10_000_000
        print("run 2: operating point,", N2, "steps (%.3f s of simulated time)" % (N2 * H))
        t, x = run(X0, 0, N2, -10, -10)
        print("  wall-clock time %.4f s, %.4f us per step (h = %.4f us)" % (t, t / N2 * 1e6, H * 1e6))
        rel = lambda z: math.atan2(z[8], z[7]) - math.atan2(z[13], z[12])    # midpoint rotor angle relative to the source
        dang = (rel(x) - rel(X0) + math.pi) % (2 * math.pi) - math.pi
        print("  max |x - x0| over the 7 rotor-frame states: %.3e; relative angle change %.3e rad (the operating point must hold)"
              % (max(abs(x[i] - X0[i]) for i in range(7)), dang))
        print("  phasor lengths after %d steps: |c^2 + s^2 - 1| machine %.3e, source %.3e (limit 1e-10)"
              % (N2, abs(x[7]**2 + x[8]**2 - 1), abs(x[12]**2 + x[13]**2 - 1)))
        """)
    end
    println("written $path")
    return path
end

"""
    phmachine_rt_export(name = "GENROU_3"; h = PHMACHINE_RT_H, dir = src/rt/machines/CPP/phmachine)

Writes phmachine.h/.cpp (translated from this file), phmachine_coefs.h (the machine `name` of the 464-state case),
the C simulation testbench, the board program and flow.cfg. Stops if a coefficient or a range of the dip run does not
fit its type, if the PIPELINE II differs from PHMACHINE_RT_STEP_CYCLES, or if a continuous Julia run differs from the
segmented one.
"""
function phmachine_rt_export(name = "GENROU_3"; h = PHMACHINE_RT_H, dir = joinpath(@__DIR__, "CPP", "phmachine"))
    occursin("II=$(PHMACHINE_RT_STEP_CYCLES)\"", string(HLS_DEFS[:phmachine_run!])) ||
        error("the PIPELINE II in phmachine_run! differs from PHMACHINE_RT_STEP_CYCLES = $PHMACHINE_RT_STEP_CYCLES")
    d = phmachine_rt_data(name)
    hb = setprecision(() -> parse(BigFloat, h), BigFloat, 256)
    hh = Float64(hb)
    k = setprecision(() -> phmachine_rt_coefstruct(d, hb), BigFloat, 256)
    x0, b = phmachine_rt_initial(d, hh)
    non = round(Int, 0.1 / hh); noff = round(Int, 0.2 / hh)
    cuts = _phmachine_rt_cuts(hh, 30, non, noff)
    # ranges on the dip run (PHMACHINE.md 7.4): the states against Sig, Delta against Ang, beta psi^2 against Beta2
    rec, _ = _phmachine_rt_julia(k, x0, b, cuts; non, noff)
    xm = maximum(r -> maximum(abs, r), _phmachine_rt_julia(k, x0, b, 1:cuts[end]; non, noff)[1])   # every step
    xm < 2.0^(PHMACHINE_RT_I.St - 2) || error("a state reaches $xm: too close to the St range")
    wmax = maximum(r -> abs(r[7]), rec)
    k.kap2 * wmax * 1.1 < 2.0^(PHMACHINE_RT_I.Ang - 1) || error("Delta up to $(k.kap2 * wmax) does not fit Ang")
    bsmax = maximum(r -> 4 * (abs(k.betad) * r[2]^2 + abs(k.betaq) * r[1]^2), rec)   # psi^ <= 2 |psi| (margin)
    bsmax < 2.0^(PHMACHINE_RT_I.Beta2 - 1) || error("beta psi^2 up to $bsmax does not fit Beta2")
    ymax = maximum(r -> maximum(i -> sum(abs, k.cdv[i]) * 2 * b.vmag + abs(k.wf[i] * b.vf) +   # the increments yd, yq
                                     sum(j -> abs(k.Pd[i, j]) * abs(r[(1, 3, 4)[j]]), 1:3) +     # (|v| <= 2 vmag:
                                     sum(abs, k.cqv[i]) * 2 * b.vmag +                            # margin)
                                     sum(j -> abs(k.Pq[i, j]) * abs(r[(2, 5, 6)[j]]), 1:3), 1:3), rec)
    ymax < 2.0^(PHMACHINE_RT_I.Ang - 1) || error("the flux increments up to $ymax do not fit Ang")
    xc = copy(x0)                                       # a continuous run equals the segmented one
    phmachine_run!(k, xc, b.vmag, b.cm, b.srot, b.vf, b.tm, 0.5, 0, non, noff, cuts[end], 32, Ref(0.0), Ref(0.0),
                   Ref(0.0), Ref(false), Ref(false))
    xc == rec[end] || error("the continuous Julia run differs from the segmented one")
    println("ranges on the dip run: max |state| $(round(xm; sigdigits = 3)), max |Delta| $(round(k.kap2 * wmax; sigdigits = 3)), ",
            "max beta psi^2 $(round(bsmax; sigdigits = 3)); $(length(cuts)) checkpoints")
    hls_write(joinpath(dir, "phmachine.cpp"), [:PHMachineCoefs, :phmachine_step!, :phmachine_run!];
              types = PHMACHINE_RT_TYPES, header = "PHMachine ($name), one step of h; from src/rt/machines/phmachine.jl")
    hls_write_const(joinpath(dir, "phmachine_coefs.h"), "PHMACHINE_COEFS", k;
                    header = "machine $name of the 464-state IEEE-39 case (src/rt/machines/data/ieee39_machines.toml)\nstep h = $hh s")
    phmachine_rt_testbench(joinpath(dir, "phmachine_tb.cpp"), k, x0, b; h = hh, cuts, non, noff)
    phmachine_rt_board_script(joinpath(dir, "phmachine_board.py"), k, x0, b; h = hh, non, noff)
    flow_config(joinpath(dir, "flow.cfg"); model = "phmachine", top = "phmachine_run", clock_ns = PHMACHINE_RT_CLOCK_NS,
                ip_version = "1.1", ip_name = "PHMachine real-time step",
                probes = ["omega_out" => 48, "ia_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                plot = ["omega_out" => (40, "omega [pu]"), "ia_out" => (40, "i_alpha [pu]")],
                sample_dt = 32 * hh, board_args = "--no-download")
end
