# =====================================================================================================================
# The brushless exciter (PHBrushlessExciter, unsaturated) on the FPGA: one build for the type, three instances
# =====================================================================================================================
# Record: docs/realtime/models/BRUSHLESS.md; plan and reviews: docs/realtime/reviews/fpga/38*_brushless_idea_*.md
# (idea gate closed at 38d: go).

# ---------------------------------------------------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------------------------------------------------
# Source: PHBrushlessExciter, src/phsim/exciter.jl lines 172-330 (`_exc_rhs!`, `_exc_powers`), with A = 0 (no
# saturation: every instance of the case; a parameter set with A != 0 is refused here). States x = (x_E, xa, xf);
# inputs u = (vterm, omega, i_fd), held over the step (sampled at the step's midpoint time); IFD = Lad i_fd:
#   VFE = KE x_E + KD IFD,   VR = clamp(xa, VRMIN, VRMAX)
#   TE x_E' = VR - VFE;   TA xa' = KA (Vref - vterm - KF (VFE - xf) / TF) - xa  (non-windup);   TF xf' = VFE - xf
#   outputs at the step's midpoint: v_f = (Rfd/Lad) EFD, EFD = FEX(KC IFD / x_E) omega x_E (no divider: the rectifier
#   modes by comparison), P = v_f i_fd - g KD IFD (VR - VFE) + g VR VFE (the power taken from the machine's shaft)
# The step: implicit midpoint, prepared per mode (BigFloat, per instance). Decision N (non-windup) once at x_n with
# this step's inputs, held for the step (xa' = 0 when frozen), as the package's `phs_modes`; the clamp of VR solved at
# the midpoint by consistent candidates L (VR = xa_mid), H (VRMAX), W (VRMIN), L preferred at both boundaries, as the
# package solves clamp(xa_mid). Banks: 1 (N free, L), 2 (N free, sat), 3 (N frozen, L), 4 (N frozen, sat); in the sat
# banks VR is an input: x+ = x + P x + W u + c + w VR, with w VRMAX and w VRMIN prepared.

const BRUSHLESS_RT_DIR = joinpath(@__DIR__)
const Inc = Float64          # the step's increments P x + W u + c (C++ type at export)
const Rq = Float64           # the rsqrt unit's internal values (m, y, y^2, ...)
const Pw = Float64           # the outputs v_f and P
const Wd = Float64           # the power terms VR KE x_E + KD IFD VFE (up to ~200)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Data and preparation (BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""    brushless_rt_data(i) -> Dict: instance i (1, 2, 3) of src/rt/exciters/data/ieee39_brushless.toml."""
function brushless_rt_data(i::Int = 1; path = joinpath(BRUSHLESS_RT_DIR, "data", "ieee39_brushless.toml"))
    d = TOML.parsefile(path)["exciter"][i]
    d["A"] == 0 || error("brushless exciter $(d["name"]): saturation A = $(d["A"]) != 0 is outside the type built " *
                         "(the unsaturated PHBrushlessExciter; BRUSHLESS.md 2)")
    return d
end
brushless_rt_ninstances(; path = joinpath(BRUSHLESS_RT_DIR, "data", "ieee39_brushless.toml")) =
    length(TOML.parsefile(path)["exciter"])

# the affine rows x' = A x + B (vterm, IFD) + c + e VR of one mode (BigFloat)
function _bx_affine(d, frozen::Bool, sat::Bool)
    b(k) = BigFloat(d[k])
    KA, TA, KE, TE, KF, TF, KD, Vref = b("KA"), b("TA"), b("KE"), b("TE"), b("KF"), b("TF"), b("KD"), b("Vref")
    A = zeros(BigFloat, 3, 3); B = zeros(BigFloat, 3, 2); c = zeros(BigFloat, 3); e = zeros(BigFloat, 3)
    A[1, 1] = -KE / TE; B[1, 2] = -KD / TE
    sat ? (e[1] = 1 / TE) : (A[1, 2] = 1 / TE)
    if !frozen
        A[2, 1] = -KA * KF * KE / (TF * TA); A[2, 2] = -1 / TA; A[2, 3] = KA * KF / (TF * TA)
        B[2, 1] = -KA / TA; B[2, 2] = -KA * KF * KD / (TF * TA); c[2] = KA * Vref / TA
    end
    A[3, 1] = KE / TF; A[3, 3] = -1 / TF; B[3, 2] = KD / TF
    return A, B, c, e
end

# the midpoint step of one mode: x+ = x + P x + W u + cc + w VR, M = (I - h A / 2)^-1 (BigFloat)
function _bx_midpoint(d, h::BigFloat, frozen::Bool, sat::Bool)
    A, B, c, e = _bx_affine(d, frozen, sat)
    M = inv(Matrix{BigFloat}(I, 3, 3) - h / 2 * A)
    return M * h * A, M * h * B, M * h * c, M * h * e
end

# ---------------------------------------------------------------------------------------------------------------------
# 3. The coefficients (the C++ struct BrushlessCoefs) and the seed ROM of the square root
# ---------------------------------------------------------------------------------------------------------------------
@hls struct BrushlessCoefs
    P::Small[4, 9]      # banks 1 free-L, 2 free-sat, 3 frozen-L, 4 frozen-sat: (I - hA/2)^-1 h A, row by row
    W::Small[4, 6]      # inputs (vterm, IFD), row by row
    c::Small[4, 3]      # the constant (Vref) term
    wmax::Small[2, 3]   # sat banks (free, frozen): w VRMAX
    wmin::Small[2, 3]   # and w VRMIN
    kc::Coef[15]        # 1 KA, 2 c1 = -KA KF KE / TF, 3 c3 = KA KF / TF, 4 KA KF KD / TF, 5 KE, 6 KD, 7 Vref, 8 VRMAX,
                        # 9 VRMIN, 10 Lad, 11 KC, 12 1/sqrt 3, 13 sqrt 3, 14 sqrt 3 / 4, 15 3/4
    kt::Tiny[3]         # 1 Rfd / Lad, 2 g, 3 g KD
end

const BRUSHLESS_RT_I = (St = 6, Sig = 8, Small = -7, Inc = -4, Coef = 8, Tiny = -8, Rq = 4, Pw = 2, Wd = 10)
const BRUSHLESS_RT_FIELDS = [:P => :Small, :W => :Small, :c => :Small, :wmax => :Small, :wmin => :Small,
                             :kc => :Coef, :kt => :Tiny]

"""
    brushless_rt_coefstruct(d, h; grid = true) -> BrushlessCoefs

The coefficients of instance `d` for the step h (BigFloat), rounded to their C++ types (grid = false: unrounded, for
the check of the rounding).
"""
function brushless_rt_coefstruct(d, h::BigFloat; grid = true)
    I_ = BRUSHLESS_RT_I
    r(v, T) = grid ? _q48(v, getfield(I_, T)) : Float64(v)
    P = zeros(4, 9); W = zeros(4, 6); c = zeros(4, 3); wmax = zeros(2, 3); wmin = zeros(2, 3)
    for (bk, (fz, sat)) in enumerate(((false, false), (false, true), (true, false), (true, true)))
        Pm, Wm, cm, wm = _bx_midpoint(d, h, fz, sat)
        for i in 1:3
            for j in 1:3; P[bk, 3(i-1)+j] = r(Pm[i, j], :Small); end
            for j in 1:2; W[bk, 2(i-1)+j] = r(Wm[i, j], :Small); end
            c[bk, i] = r(cm[i], :Small)
            if sat
                s = fz ? 2 : 1
                wmax[s, i] = r(wm[i] * BigFloat(d["VRMAX"]), :Small); wmin[s, i] = r(wm[i] * BigFloat(d["VRMIN"]), :Small)
            end
        end
    end
    b(k) = BigFloat(d[k])
    s3 = sqrt(BigFloat(3))
    kf = b("KA") * b("KF") / b("TF")
    kc = [r(v, :Coef) for v in (b("KA"), -kf * b("KE"), kf, kf * b("KD"), b("KE"), b("KD"), b("Vref"), b("VRMAX"),
                                b("VRMIN"), b("Lad"), b("KC"), 1 / s3, s3, s3 / 4, BigFloat(3) / 4)]
    kt = [r(v, :Tiny) for v in (b("gain"), b("g"), b("g") * b("KD"))]
    return BrushlessCoefs(P, W, c, wmax, wmin, kc, kt)
end

# the seeds of 1/sqrt(m), m in [1/4, 1): octave 0 = [1/4, 1/2), octave 1 = [1/2, 1), 128 bins each (top 7 bits below
# the leading one); seed 2 / (sqrt m_lo + sqrt m_hi) in 18 bits (16 fraction bits); relative error <= 2^-9 (38b merge)
function brushless_rt_seeds()
    s = zeros(256)
    for oct in 0:1, j in 0:127
        lo = BigFloat(2)^(oct - 2) * (1 + BigFloat(j) / 128); hi = BigFloat(2)^(oct - 2) * (1 + BigFloat(j + 1) / 128)
        g = BigFloat(2)^-16
        s[128oct+j+1] = Float64(round(2 / (sqrt(lo) + sqrt(hi)) / g) * g)
    end
    return s
end
hls_rom!(:BX_SEED, brushless_rt_seeds())

# ---------------------------------------------------------------------------------------------------------------------
# 4. THE SQUARE ROOT AND THE STEP (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
# sqrt(a) for 2^-12 <= a < 64 (the rectifier's mode II: a = (3/4) x_E^2 - (KC IFD)^2 > 0): a = m 4^e with m in [1/4, 1);
# the seed from the ROM; two Newton steps y <- y (3 - m y^2) / 2 (no divider); sqrt(a) = m y 2^e
@hls function bx_sqrt(a::Sig)::Sig
    @pragma "HLS INLINE off"                         # its own module: its seven products are one dependent chain,
    @pragma "HLS ALLOCATION operation instances=mul limit=1"   # so one shared multiplier costs no latency
    seed::Rq[256] = rom(:BX_SEED)
    quarter::Rq = 0.25
    half::Rq = 0.5
    three::Rq = 3
    four::Sig = 4
    sixteen::Sig = 16
    onesig::Sig = 1
    m::Rq = 0                                        # a >= 1 is scaled down exactly before it enters Rq (+-8)
    e::Int = 0
    if a >= sixteen                                  # a < 64: a / 4^3
        m = scale2(a, -6)
        e = 3
    elseif a >= four
        m = scale2(a, -4)
        e = 2
    elseif a >= onesig
        m = scale2(a, -2)
        e = 1
    else
        m = a
    end
    for i in 1:6                                     # a >= 2^-12: at most six multiplications by 4
        if m < quarter
            m = scale2(m, 2)
            e -= 1
        end
    end
    if m < quarter                                   # below the range (a < 2^-12; flagged by the export's range check)
        m = quarter
    end
    ix::Int = 0                                      # the seed's index: octave and the top 7 bits below the leading one
    if m >= half
        ix = ifloor(scale2(m, 8))                    # m in [1/2, 1): 128 .. 255
    else
        ix = ifloor(scale2(m, 9)) - 128              # m in [1/4, 1/2): 0 .. 127
    end
    y::Rq = seed[ix + 1]
    for it in 1:2
        y2::Rq = y * y
        my2::Rq = m * y2
        t::Rq = three - my2
        yt::Rq = y * t
        y = scale2(yt, -1)
    end
    r::Rq = m * y                                    # sqrt(m) in [1/2, 1)
    for i in 1:3
        if e > 0
            r = scale2(r, 1)
            e -= 1
        end
    end
    for i in 1:6
        if e < 0
            r = scale2(r, -1)
            e += 1
        end
    end
    return r
end

# one step of h: x = (x_E, xa, xf) changed in place; u = (vterm, omega, i_fd); y = (v_f, P) at the step's midpoint;
# md = (N: 0 free, 1 frozen high, 2 frozen low; C: 0 L, 1 VRMAX, 2 VRMIN, 3 none consistent; rectifier: 0 x_E <= 0,
# 1 mode I, 2 mode II, 3 mode III, 4 IN >= 1)
@hls function brushless_step!(x::St[3], u::Sig[3], y::Pw[2], md::Int[3], kP::Small[4, 9], kW::Small[4, 6],
                              kk::Small[4, 3], kwx::Small[2, 3], kwn::Small[2, 3], kc::Coef[15], kt::Tiny[3])
    vt::Sig = u[1]
    om::Sig = u[2]
    ifd::Sig = u[3]
    IFD::Sig = kc[10] * ifd
    kdI::Sig = kc[6] * IFD
    # the decision N at x_n with this step's inputs (held for the step)
    ve::Sig = kc[7] - vt
    r1::Sig = kc[1] * ve                             # KA (Vref - vterm)
    r2::Sig = kc[4] * IFD                            # KA KF KD / TF IFD
    r3::Sig = kc[2] * x[1]                           # -KA KF KE / TF x_E: one level of products on the loop path
    r4::Sig = kc[3] * x[3]                           # KA KF / TF xf
    ra::Sig = r1 - r2
    rb::Sig = r3 + r4
    r::Sig = ra + rb - x[2]
    nb::Int = 0
    if x[2] >= kc[8] && r > 0
        nb = 1
    elseif x[2] <= kc[9] && r < 0
        nb = 2
    end
    fz::Int = 0
    if nb > 0
        fz = 1
    end
    bL::Int = 1 + 2 * fz                             # the bank of candidate L, and of the sat candidates
    bS::Int = 2 + 2 * fz
    # the candidates: dL (VR = xa_mid), dS (VR an input; + w VRMAX or + w VRMIN)
    uu::Sig[2] = (vt, IFD)
    @pragma "HLS ARRAY_PARTITION variable=uu complete"
    dL::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=dL complete"
    dS::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=dS complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        # each product rounded on its own, then a balanced tree (not a sequential accumulation: synth 3)
        l1::Inc = kP[bL, 3 * i - 2] * x[1]
        l2::Inc = kP[bL, 3 * i - 1] * x[2]
        l3::Inc = kP[bL, 3 * i] * x[3]
        l4::Inc = kW[bL, 2 * i - 1] * uu[1]
        l5::Inc = kW[bL, 2 * i] * uu[2]
        s1::Inc = kP[bS, 3 * i - 2] * x[1]
        s2::Inc = kP[bS, 3 * i - 1] * x[2]
        s3::Inc = kP[bS, 3 * i] * x[3]
        s4::Inc = kW[bS, 2 * i - 1] * uu[1]
        s5::Inc = kW[bS, 2 * i] * uu[2]
        la::Inc = l1 + l2
        lb::Inc = l3 + kk[bL, i]
        lc::Inc = l4 + l5
        ld::Inc = la + lb
        dL[i] = ld + lc
        sa::Inc = s1 + s2
        sb::Inc = s3 + kk[bS, i]
        sc::Inc = s4 + s5
        sd::Inc = sa + sb
        dS[i] = sd + sc
    end
    # the clamp at the midpoint: the consistent candidate, L preferred at both boundaries
    xamL::Sig = x[2] + scale2(dL[2], -1)
    dH2::Inc = dS[2] + kwx[fz + 1, 2]
    dW2::Inc = dS[2] + kwn[fz + 1, 2]
    xamH::Sig = x[2] + scale2(dH2, -1)
    xamW::Sig = x[2] + scale2(dW2, -1)
    d::Inc[3]
    @pragma "HLS ARRAY_PARTITION variable=d complete"
    vrm::Sig = xamL
    cs::Int = 0
    if xamL >= kc[9] && xamL <= kc[8]
        cs = 0
    elseif xamH >= kc[8]
        cs = 1
    elseif xamW <= kc[9]
        cs = 2
    else
        cs = 3
    end
    for i in 1:3
        @pragma "HLS UNROLL"
        if cs == 1
            d[i] = dS[i] + kwx[fz + 1, i]
        elseif cs == 2
            d[i] = dS[i] + kwn[fz + 1, i]
        else
            d[i] = dL[i]
        end
    end
    if cs == 1
        vrm = kc[8]
    elseif cs == 2
        vrm = kc[9]
    end
    # the midpoint values and the outputs (not fed back: outside the recurrence)
    xEm::Sig = x[1] + scale2(d[1], -1)
    kem::Sig = kc[5] * xEm
    vfem::Sig = kem + kdI
    kcI::Sig = 0
    if IFD > 0
        kcI = kc[11] * IFD
    end
    t1::Sig = kc[14] * xEm
    t2::Sig = kc[15] * xEm
    rm::Int = 0
    if xEm <= 0
        rm = 0
    elseif kcI <= t1
        rm = 1
    elseif kcI < t2
        rm = 2
    elseif kcI < xEm
        rm = 3
    else
        rm = 4
    end
    # modes I and III are one linear form al x_E - be KC IFD: (1, 1/sqrt 3) and (sqrt 3, sqrt 3)
    al::Coef = 1
    be::Coef = kc[12]
    if rm == 3
        al = kc[13]
        be = kc[13]
    end
    lx::Sig = al * xEm
    lk::Sig = be * kcI
    lin::Sig = lx - lk
    xe2::Sig = xEm * xEm                             # mode II: sqrt((3/4) x_E^2 - (KC IFD)^2)
    kc2::Sig = kcI * kcI
    q2::Sig = kc[15] * xe2
    ar::Sig = q2 - kc2
    rad::Sig = 0
    if rm == 2
        rad = bx_sqrt(ar)
    end
    ef::Sig = 0
    if rm == 1 || rm == 3
        ef = lin
    elseif rm == 2
        ef = rad
    end
    efd::Sig = om * ef
    vf::Pw = kt[1] * efd
    # P = v_f i_fd - g KD IFD (VR - VFE) + g VR VFE = v_f i_fd + g (VR KE x_E + KD IFD VFE)
    p1::Pw = vf * ifd
    pa::Wd = vrm * kem
    pb::Wd = kdI * vfem
    pc::Wd = pa + pb
    p2::Pw = kt[2] * pc
    y[1] = vf
    y[2] = p1 + p2
    for i in 1:3
        @pragma "HLS UNROLL"
        x[i] = x[i] + d[i]
    end
    md[1] = nb
    md[2] = cs
    md[3] = rm
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. THE BOARD TOP FUNCTION (the IP on the FPGA): the step in a loop, the bench's inputs from a segment table
# ---------------------------------------------------------------------------------------------------------------------
# Inputs (vterm, omega, i_fd) in 4 segments starting at steps 0, n_on, n_mid, n_off; in segment s the step n (global,
# n = n0 + i, from t_(n-1) to t_n) gets u = us[s] + du[s] (n - 1 - start_s): us is the value at the first step's
# midpoint, du = h u' (midpoint sampling of the ramps, review 38c, 2). A run can be continued in segments (n0).
@hls function brushless_run!(k::BrushlessCoefs, x::St[3], us::Sig[4, 3], du::Small[4, 3], n_on::Int, n_mid::Int,
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
    xs::St[3]                                        # the states, in registers during the run
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    for i in 1:3
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    u::Sig[3]
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    yy::Pw[2]
    @pragma "HLS ARRAY_PARTITION variable=yy complete"
    md::Int[3]
    @pragma "HLS ARRAY_PARTITION variable=md complete"
    kP::Small[4, 9]                                  # the coefficients in registers for the run (read once over
    @pragma "HLS ARRAY_PARTITION variable=kP complete dim=0"   # AXI-Lite: the step reads them all in parallel)
    kW::Small[4, 6]
    @pragma "HLS ARRAY_PARTITION variable=kW complete dim=0"
    kk::Small[4, 3]
    @pragma "HLS ARRAY_PARTITION variable=kk complete dim=0"
    kwx::Small[2, 3]
    @pragma "HLS ARRAY_PARTITION variable=kwx complete dim=0"
    kwn::Small[2, 3]
    @pragma "HLS ARRAY_PARTITION variable=kwn complete dim=0"
    kc::Coef[15]
    @pragma "HLS ARRAY_PARTITION variable=kc complete"
    kt::Tiny[3]
    @pragma "HLS ARRAY_PARTITION variable=kt complete"
    for b in 1:4
        for j in 1:9
            kP[b, j] = k.P[b, j]
        end
        for j in 1:6
            kW[b, j] = k.W[b, j]
        end
        for j in 1:3
            kk[b, j] = k.c[b, j]
        end
    end
    for b in 1:2
        for j in 1:3
            kwx[b, j] = k.wmax[b, j]
            kwn[b, j] = k.wmin[b, j]
        end
    end
    for j in 1:15
        kc[j] = k.kc[j]
    end
    for j in 1:3
        kt[j] = k.kt[j]
    end
    st::Bool = false
    cnt::Int = 0
    pxa::St = xs[2]                                  # the probes' registers: the previous step's values, written to
    pvf::Pw = 0                                      # all four ports together at the start of each iteration, so
    psa::Bool = false                                # the ILA sees one step in every probe (review 39, 1)
    pst::Bool = false
    for i in 1:nsteps
        @pragma "HLS PIPELINE II=24"                 # BRUSHLESS_RT_STEP_CYCLES (checked at export)
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
        brushless_step!(xs, u, yy, md, kP, kW, kk, kwx, kwn, kc, kt)
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            psa = true
        else
            psa = false
        end
        pxa = xs[2]                                  # to the ILA in the next iteration
        pvf = yy[1]
        pst = st
    end
    for i in 1:3
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
    y[1] = yy[1]
    y[2] = yy[2]
end

# ---------------------------------------------------------------------------------------------------------------------
# 6. Export: bench cases, checks, C++, testbench, board program
# ---------------------------------------------------------------------------------------------------------------------
const BRUSHLESS_RT_CLOCK_NS = 8
const BRUSHLESS_RT_STEP_CYCLES = 24
const BRUSHLESS_RT_H = "$(BRUSHLESS_RT_STEP_CYCLES * BRUSHLESS_RT_CLOCK_NS)e-9"
const BRUSHLESS_RT_TYPES = [:St => "ap_fixed<48, 6, AP_RND_CONV>",       # states: +-32, grid 2.3e-13
                            :Sig => "ap_fixed<48, 8, AP_RND_CONV>",      # inputs, VR, VFE, EFD, radicand: +-128
                            :Small => "ap_fixed<48, -7, AP_RND_CONV>",   # prepared step entries: +-3.9e-3, 2.8e-17
                            :Inc => "ap_fixed<48, -4, AP_RND_CONV>",     # increments: +-0.031, 2.2e-16
                            :Coef => "ap_fixed<48, 8, AP_RND_CONV>",     # gains (KA = 40): +-128
                            :Tiny => "ap_fixed<48, -8, AP_RND_CONV>",    # g, Rfd/Lad: +-1.95e-3
                            :Rq => "ap_fixed<48, 4, AP_RND_CONV>",       # the square root's m, y: +-8, 5.7e-14
                            :Pw => "ap_fixed<48, 2, AP_RND_CONV>",       # v_f, P: +-2, 3.6e-15
                            :Wd => "ap_fixed<48, 10, AP_RND_CONV>"]      # power terms: +-512, 3.6e-12
const BRUSHLESS_RT_SCALE = (x = 10.0, vf = 1e-3, P = 1e-2)          # the scales of the relative limits (38b, 5)
const BRUSHLESS_RT_LIM = 5e-8                                        # C sim and board: relative to the scales

# the bench's segment table of case :A, :B or :C for instance d (midpoint-sampled; rounded to the C++ grids)
function brushless_rt_bench(d, case::Symbol, h::Float64)
    non, nmid, noff = round(Int, 0.1 / h), round(Int, 0.15 / h), round(Int, 0.2 / h)
    v0, w0, i0 = d["vterm"], d["omega"], d["ifd"]
    # segment values at the segment's start and end (time), per input (vterm, omega, i_fd)
    if case === :A
        a = [(v0, v0), (0.5, 0.5), (0.5, 0.5), (v0, v0)]
        o = [(w0, w0), (w0, 0.97), (0.97, w0), (w0, w0)]
        f = [(i0, i0), (i0, 5.25 * i0), (5.25 * i0, i0), (i0, i0)]
    elseif case === :B
        a = [(v0, v0), (1.4, 1.4), (1.4, 1.4), (v0, v0)]
        o = [(w0, w0) for _ in 1:4]; f = [(i0, i0) for _ in 1:4]
    else
        a = [(v0, v0) for _ in 1:4]; o = [(w0, w0) for _ in 1:4]; f = [(i0, i0) for _ in 1:4]
        non = nmid = noff = 1 << 30
    end
    st = [0, non, nmid, noff]; len = [non, nmid - non, noff - nmid, 1]
    us = zeros(4, 3); du = zeros(4, 3)
    for s in 1:4, (j, v) in enumerate((a, o, f))
        rate = case === :C ? 0.0 : (v[s][2] - v[s][1]) / len[s]          # per step
        du[s, j] = _q48(rate, BRUSHLESS_RT_I.Small)
        us[s, j] = _q48(v[s][1] + rate / 2, BRUSHLESS_RT_I.Sig)          # the first step's midpoint value
    end
    return (; non, nmid, noff, us, du)
end

# the Julia run of brushless_run! (Float64) cut at `cuts`: the states after each cut and the outputs of its last step
function _brushless_rt_julia(k, x0, b, cuts)
    x = copy(x0); rec = [copy(x)]; outs = [zeros(2)]
    y = zeros(2); r = (Ref(0.0), Ref(0.0), Ref(false), Ref(false)); n = 0
    for c in cuts
        brushless_run!(k, x, b.us, b.du, b.non, b.nmid, b.noff, n, c - n, 32, y, r...)
        push!(rec, copy(x)); push!(outs, copy(y)); n = c
    end
    return rec, outs
end

# every step of a case: the step function itself, with the inputs as brushless_run! forms them; returns the per-step
# records (x before, x after, u, y, md) for the ranges, occupancy and energy checks
function _brushless_rt_trace(k, x0, b, nsteps)
    x = copy(x0); y = zeros(2); md = zeros(Int, 3); u = zeros(3)
    X = Vector{Vector{Float64}}(); Xn = similar(X); U = similar(X); Y = similar(X); M = Vector{Vector{Int}}()
    for n in 1:nsteps
        m1 = n - 1
        sg, s0 = m1 >= b.noff ? (4, b.noff) : m1 >= b.nmid ? (3, b.nmid) : m1 >= b.non ? (2, b.non) : (1, 0)
        for j in 1:3; u[j] = b.us[sg, j] + b.du[sg, j] * (m1 - s0); end
        push!(X, copy(x)); push!(U, copy(u))
        brushless_step!(x, u, y, md, k.P, k.W, k.c, k.wmax, k.wmin, k.kc, k.kt)
        push!(Xn, copy(x)); push!(Y, copy(y)); push!(M, copy(md))
    end
    return (; X, Xn, U, Y, M)
end

_brushless_rt_cuts(h, b) = sort(unique(filter(c -> 0 < c <= round(Int, 0.3 / h),
    vcat([round(Int, g * 0.01 / h) for g in 1:30], b.non-1:b.non+2, b.nmid-1:b.nmid+2, b.noff-1:b.noff+2))))

# every intermediate of brushless_step! and bx_sqrt at (x, u), with the C++ type it is stored in (the range check of
# the export, review 39, 3; mirrors the @hls code statement by statement)
function _bx_intermediates!(mx::Dict{Symbol,Float64}, k, x, u)
    P, W, cc, wx, wn, kc, kt = k.P, k.W, k.c, k.wmax, k.wmin, k.kc, k.kt
    put(T, a) = (mx[T] = max(get(mx, T, 0.0), abs(a)); a)
    vt, om, ifd = u
    IFD = put(:Sig, kc[10] * ifd); kdI = put(:Sig, kc[6] * IFD)
    ve = put(:Sig, kc[7] - vt); r1 = put(:Sig, kc[1] * ve); r2 = put(:Sig, kc[4] * IFD)
    r3 = put(:Sig, kc[2] * x[1]); r4 = put(:Sig, kc[3] * x[3]); ra = put(:Sig, r1 - r2); rb = put(:Sig, r3 + r4)
    r = put(:Sig, ra + rb - x[2])
    fz = (x[2] >= kc[8] && r > 0) || (x[2] <= kc[9] && r < 0) ? 1 : 0
    bL, bS = 1 + 2fz, 2 + 2fz
    dL = zeros(3); dS = zeros(3)
    for i in 1:3
        l = [put(:Inc, P[bL, 3i-3+j] * x[j]) for j in 1:3]; lw = [put(:Inc, W[bL, 2i-2+j] * (vt, IFD)[j]) for j in 1:2]
        sv = [put(:Inc, P[bS, 3i-3+j] * x[j]) for j in 1:3]; sw = [put(:Inc, W[bS, 2i-2+j] * (vt, IFD)[j]) for j in 1:2]
        la = put(:Inc, l[1] + l[2]); lb = put(:Inc, l[3] + cc[bL, i]); lc = put(:Inc, lw[1] + lw[2]); ld = put(:Inc, la + lb)
        dL[i] = put(:Inc, ld + lc)
        sa = put(:Inc, sv[1] + sv[2]); sb = put(:Inc, sv[3] + cc[bS, i]); sc = put(:Inc, sw[1] + sw[2]); sd = put(:Inc, sa + sb)
        dS[i] = put(:Inc, sd + sc)
    end
    xamL = put(:Sig, x[2] + dL[2] / 2)
    put(:Inc, dS[2] + wx[fz+1, 2]); put(:Inc, dS[2] + wn[fz+1, 2])
    put(:Sig, x[2] + (dS[2] + wx[fz+1, 2]) / 2); put(:Sig, x[2] + (dS[2] + wn[fz+1, 2]) / 2)
    cs = kc[9] <= xamL <= kc[8] ? 0 : xamL >= kc[8] ? 1 : 2
    d = cs == 1 ? dS .+ wx[fz+1, :] : cs == 2 ? dS .+ wn[fz+1, :] : dL
    foreach(e -> put(:Inc, e), d)
    vrm = cs == 1 ? kc[8] : cs == 2 ? kc[9] : xamL
    xEm = put(:Sig, x[1] + d[1] / 2); kem = put(:Sig, kc[5] * xEm); vfem = put(:Sig, kem + kdI)
    kcI = put(:Sig, IFD > 0 ? kc[11] * IFD : 0.0); put(:Sig, kc[14] * xEm); put(:Sig, kc[15] * xEm)
    put(:Sig, kc[13] * xEm); put(:Sig, kc[13] * kcI); put(:Sig, kc[12] * kcI); put(:Sig, xEm - kc[12] * kcI)
    put(:Sig, kc[13] * xEm - kc[13] * kcI)
    xe2 = put(:Sig, xEm * xEm); kc2 = put(:Sig, kcI * kcI); q2 = put(:Sig, kc[15] * xe2); ar = put(:Sig, q2 - kc2)
    rm = xEm <= 0 ? 0 : kcI <= kc[14] * xEm ? 1 : kcI < kc[15] * xEm ? 2 : kcI < xEm ? 3 : 4
    if rm == 2                                       # the square root's internals
        m = ar; e = 0
        while m >= 1; m /= 4; e += 1; end
        while m < 0.25; m *= 4; e -= 1; end
        put(:Rq, m); y = HLS_ROMS[:BX_SEED][(m >= 0.5 ? ifloor(m * 256) : ifloor(m * 512) - 128) + 1]
        for _ in 1:2
            y2 = put(:Rq, y * y); my2 = put(:Rq, m * y2); t = put(:Rq, 3 - my2); yt = put(:Rq, y * t); y = put(:Rq, yt / 2)
        end
        put(:Rq, m * y)
    end
    ef = rm == 1 ? xEm - kc[12] * kcI : rm == 3 ? kc[13] * (xEm - kcI) : rm == 2 ? sqrt(ar) : 0.0
    efd = put(:Sig, om * ef); vf = put(:Pw, kt[1] * efd); p1 = put(:Pw, vf * ifd)
    pa = put(:Wd, vrm * kem); pb = put(:Wd, kdI * vfem); pc = put(:Wd, pa + pb); p2 = put(:Pw, kt[2] * pc); put(:Pw, p1 + p2)
    return (; rm, ar, xEm, IFD, kcI, om)
end

"""
    brushless_rt_export(; h = BRUSHLESS_RT_H, dir = src/rt/exciters/CPP/brushless)

Prepares the three instances for h (BigFloat), runs the Julia checks fixed in the idea gate (38b, 38c, 38d: ranges,
mode occupancy, candidate consistency, rounding, energy, square root, continuous = segmented) and writes the C++, the
seed ROM, the coefficients of all instances, the C simulation testbench, the board program and flow.cfg. The check
against the package's rows and the refinement: scripts/realtime/brushless_checks.jl.
"""
function brushless_rt_export(; h = BRUSHLESS_RT_H, dir = joinpath(BRUSHLESS_RT_DIR, "CPP", "brushless"))
    occursin("II=$(BRUSHLESS_RT_STEP_CYCLES)\"", string(HLS_DEFS[:brushless_run!])) ||
        error("the PIPELINE II in brushless_run! differs from BRUSHLESS_RT_STEP_CYCLES = $BRUSHLESS_RT_STEP_CYCLES")
    hb = setprecision(() -> parse(BigFloat, h), BigFloat, 256); hh = Float64(hb)
    ni = brushless_rt_ninstances()
    ds = [brushless_rt_data(i) for i in 1:ni]
    ks = [setprecision(() -> brushless_rt_coefstruct(d, hb), BigFloat, 256) for d in ds]
    ku = [setprecision(() -> brushless_rt_coefstruct(d, hb; grid = false), BigFloat, 256) for d in ds]
    x0s = [map(v -> _q48(v, BRUSHLESS_RT_I.St), d["x"]) for d in ds]
    nend = round(Int, 0.3 / hh)
    S = BRUSHLESS_RT_SCALE
    println("h = $hh s; $ni instances: ", join((d["machine"] for d in ds), ", "))
    I_ = BRUSHLESS_RT_I
    half(T) = 2.0^(getfield(I_, T) - 2)                              # half of the type's range
    for (i, k) in enumerate(ks), (fld, T) in BRUSHLESS_RT_FIELDS     # the coefficients (review 39, 3)
        m = maximum(abs, getfield(k, fld))
        m < half(T) || error("instance $i: coefficient $fld reaches $m (half range of $T: $(half(T)))")
    end
    for (ii, (d, k, ku_, x0)) in enumerate(zip(ds, ks, ku, x0s)), case in (:A, :B)
        b = brushless_rt_bench(d, case, hh)
        tr = _brushless_rt_trace(k, x0, b, nend)
        # ranges (38c, 4): every value against half its type's range
        mx = Dict{Symbol,Float64}()
        up!(s, v) = (mx[s] = max(get(mx, s, 0.0), abs(v)))
        Hm = d["g"] * d["TE"] * d["KE"] / 2; H0 = Hm * d["x"][1]^2
        Dmax = 0.0; Dsum = 0.0; amin = Inf; occ = zeros(Int, 3, 5)
        for n in 1:nend
            x, xn, u, y, md = tr.X[n], tr.Xn[n], tr.U[n], tr.Y[n], tr.M[n]
            foreach(v -> up!(:St, v), xn); foreach(v -> up!(:Inc, v), xn - x); foreach(v -> up!(:Sig, v), u)
            up!(:Pw, y[1]); up!(:Pw, y[2])
            IFD = d["Lad"] * u[3]; xEm = (x[1] + xn[1]) / 2
            md[3] == 2 && (a = 0.75 * xEm^2 - (d["KC"] * IFD)^2; amin = min(amin, a); up!(:Sig, a))
            vrm = md[2] == 1 ? d["VRMAX"] : md[2] == 2 ? d["VRMIN"] : (x[2] + xn[2]) / 2
            kem = d["KE"] * xEm; vf_ = kem + d["KD"] * IFD
            up!(:Sig, vf_); up!(:Sig, sqrt(3) * xEm); up!(:Sig, xEm^2); up!(:Wd, vrm * kem); up!(:Wd, d["KD"] * IFD * vf_)
            up!(:Wd, vrm * kem + d["KD"] * IFD * vf_)
            for j in 1:3; occ[j, md[j]+1] += 1; end
            iv = _bx_intermediates!(mx, k, x, u)                     # every intermediate against its type
            # the physical domain and the root's supported domain (review 39, 3): failures, not prints
            iv.xEm > 0 || error("$(d["machine"]) case $case step $n: x_E <= 0 (outside the model)")
            iv.IFD >= 0 || error("$(d["machine"]) case $case step $n: IFD < 0 (outside the model)")
            iv.kcI < iv.xEm || error("$(d["machine"]) case $case step $n: IN >= 1 (outside the model)")
            iv.om >= 0.5 || error("$(d["machine"]) case $case step $n: omega < 0.5")
            md[3] == 2 && !(1e-3 <= iv.ar < 64) && error("$(d["machine"]) case $case step $n: radicand $(iv.ar) outside [1e-3, 64)")
            # energy (38b, 5c): D = H(x+) - H(x) - h (P - v_f i_fd - g VFE^2), in BigFloat from the Float64 values
            vfem = big(d["KE"]) * (big(x[1]) + big(xn[1])) / 2 + big(d["KD"]) * IFD
            D = big(Hm) * (big(xn[1])^2 - big(x[1])^2) - big(hh) * (big(y[2]) - big(y[1]) * big(u[3]) - big(d["g"]) * vfem^2)
            Dmax = max(Dmax, Float64(abs(D))); Dsum += Float64(abs(D))
        end
        for (T, v) in mx
            v < half(T) || error("$(d["machine"]) case $case: a $T value reaches $v (half range $(half(T)))")
        end
        any(m -> m[2] == 3, tr.M) && error("$(d["machine"]) case $case: a step without a consistent clamp candidate")
        # occupancy (38c, 2)
        need = case === :A ? [(1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)] : [(1, 0), (2, 0), (1, 2), (2, 2)]
        for (j, v) in need
            occ[j, v+1] > 0 || error("$(d["machine"]) case $case: mode $(("N", "C", "rectifier")[j]) = $v never occurs")
        end
        # rounding (b): the same case with unrounded coefficients
        tu = _brushless_rt_trace(ku_, x0, b, nend)
        db = maximum(n -> max(maximum(abs, tr.Xn[n] - tu.Xn[n]) / S.x, abs(tr.Y[n][1] - tu.Y[n][1]) / S.vf,
                              abs(tr.Y[n][2] - tu.Y[n][2]) / S.P), 1:nend)
        db <= 5e-8 || error("$(d["machine"]) case $case: rounding moves the run by $db (relative)")
        (Dmax <= 1e-12 * H0 && Dsum <= 1e-9 * H0) || error("$(d["machine"]) case $case: energy defect max $Dmax, sum $Dsum (H0 $H0)")
        xs = reduce(hcat, tr.Xn)
        println("$(d["machine"]) case $case: occupancy N $(occ[1, 1:3]), C $(occ[2, 1:4]), rectifier $(occ[3, :]); ",
                "min x_E $(round(minimum(xs[1, :]); sigdigits = 5)), max IN $(round(maximum(n -> d["KC"] * max(d["Lad"] * tr.U[n][3], 0) / ((tr.X[n][1] + tr.Xn[n][1]) / 2), 1:nend); sigdigits = 5)), ",
                "min IFD $(round(minimum(n -> d["Lad"] * tr.U[n][3], 1:nend); sigdigits = 5)), min omega $(round(minimum(n -> tr.U[n][2], 1:nend); sigdigits = 5)); ",
                "rounding $(round(db; sigdigits = 3)) (limit 5e-8); energy max |D| $(round(Dmax / H0; sigdigits = 3)) H0, ",
                "sum $(round(Dsum / H0; sigdigits = 3)) H0 (limits 1e-12, 1e-9); min radicand $(amin == Inf ? "-" : round(amin; sigdigits = 4))")
    end
    # the square root (38b, 5e): Julia's bx_sqrt against sqrt over a in [1e-3, 60]
    es = maximum(a -> abs(bx_sqrt(a) / sqrt(a) - 1), exp10.(range(-3, log10(60), length = 100_000)))
    println("bx_sqrt (Julia, Float64): max relative error $es over a in [1e-3, 60] (limit 1e-10)")
    es <= 1e-10 || error("bx_sqrt error $es")
    # continuous = segmented, instance 1, case A
    b1 = brushless_rt_bench(ds[1], :A, hh); cuts = _brushless_rt_cuts(hh, b1)
    rec, _ = _brushless_rt_julia(ks[1], x0s[1], b1, cuts)
    xc = copy(x0s[1])
    brushless_run!(ks[1], xc, b1.us, b1.du, b1.non, b1.nmid, b1.noff, 0, cuts[end], 32, zeros(2), Ref(0.0), Ref(0.0), Ref(false), Ref(false))
    xc == rec[end] || error("the continuous Julia run differs from the segmented one")
    # the C++
    mkpath(dir)
    hls_write_util(dir)
    hls_write_roms(joinpath(dir, "brushless_rom.h"), [:BX_SEED]; header = "the seeds of 1/sqrt(m) (brushless.jl section 3)")
    hls_write(joinpath(dir, "brushless.cpp"), [:BrushlessCoefs, :bx_sqrt, :brushless_step!, :brushless_run!];
              types = BRUSHLESS_RT_TYPES, local_includes = ["rt_util.h", "brushless_rom.h"],
              header = "the brushless exciter (unsaturated PHBrushlessExciter), one run of steps of h = $hh s; from src/rt/exciters/brushless.jl")
    p = joinpath(dir, "brushless_coefs.h")
    open(p, "w") do io
        println(io, "// brushless_coefs.h: written by PHPS.RT.brushless_rt_export; do not edit\n// h = $hh s; instances of src/rt/exciters/data/ieee39_brushless.toml")
        for (i, (d, k)) in enumerate(zip(ds, ks))
            println(io, "// $i: $(d["name"]) on $(d["machine"])")
            println(io, "const BrushlessCoefs BRUSHLESS_COEFS_$i = ", _cinit(k), ";")
        end
    end
    println("written $p")
    brushless_rt_testbench(joinpath(dir, "brushless_tb.cpp"), ds, ks, x0s; h = hh)
    brushless_rt_board_script(joinpath(dir, "brushless_board.py"), ds, ks, x0s; h = hh)
    brushless_rt_ilaref(joinpath(dir, "brushless_ila_ref.csv"), ds, ks, x0s; h = hh)
    flow_config(joinpath(dir, "flow.cfg"); model = "brushless", top = "brushless_run", clock_ns = BRUSHLESS_RT_CLOCK_NS,
                ip_version = "1.0", ip_name = "Brushless exciter real-time step",
                probes = ["xa_out" => 48, "vf_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                plot = ["xa_out" => (48 - I_.St, "xa [pu]"), "vf_out" => (48 - I_.Pw, "v_f [pu]")],
                sample_dt = BRUSHLESS_RT_DECIM * hh, board_args = "--no-download",
                step_cycles = BRUSHLESS_RT_STEP_CYCLES, ila_ref = "brushless_ila_ref.csv",
                ila_check = ["xa_out:$(48 - I_.St):$(BRUSHLESS_RT_LIM * BRUSHLESS_RT_SCALE.x)",
                             "vf_out:$(48 - I_.Pw):$(BRUSHLESS_RT_LIM * BRUSHLESS_RT_SCALE.vf)"], strict = true)
end

# the board program's run order (brushless_rt_board_script): (instance, case) with its calls, then case C
const BRUSHLESS_RT_BOARD_RUNS = ((1, :A), (1, :B), (2, :A), (3, :A))
const BRUSHLESS_RT_DECIM = 96                        # ILA: one sample per 96 steps (16,384 samples cover 0.302 s)

# what the ILA records: brushless_run! statement by statement, with the probe registers; a sample is (xa, v_f) after
# the step at which the counter reached decim, seen at the next iteration's start (review 39, 1-2)
# The ports are written in different pipeline stages (probe_lags.py): for each sample the values written at the start
# of this iteration and of the 7 before it (lags 0..7) are kept; the flow picks each probe's lag from the schedule.
function _bx_ila_call!(rec, k, x, b, n0, nsteps, decim, nmax)
    pxa = x[2]; pvf = 0.0; psa = false; cnt = 0; y = zeros(2); md = zeros(Int, 3); u = zeros(3)
    ha = fill(NaN, 8); hv = fill(NaN, 8)                            # written values, newest first
    for i in 1:nsteps
        ha = vcat(pxa, ha[1:7]); hv = vcat(pvf, hv[1:7])
        psa && length(rec) < nmax && push!(rec, (copy(ha), copy(hv), n0 + i - 1))
        m1 = n0 + i - 1
        sg, s0 = m1 >= b.noff ? (4, b.noff) : m1 >= b.nmid ? (3, b.nmid) : m1 >= b.non ? (2, b.non) : (1, 0)
        for j in 1:3; u[j] = b.us[sg, j] + b.du[sg, j] * (m1 - s0); end
        brushless_step!(x, u, y, md, k.P, k.W, k.c, k.wmax, k.wmin, k.kc, k.kt)
        cnt += 1
        psa = cnt >= decim; psa && (cnt = 0)
        pxa = x[2]; pvf = y[1]
    end
    return x
end

# the ILA reference of the board program's first 16,384 samples; each call checked against brushless_run! itself
function brushless_rt_ilaref(path, ds, ks, x0s; h, decim = BRUSHLESS_RT_DECIM, nmax = 16384)
    rec = Tuple{Vector{Float64},Vector{Float64},Int}[]; tag = String[]
    for (i, case) in BRUSHLESS_RT_BOARD_RUNS
        b = brushless_rt_bench(ds[i], case, h); cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        x = copy(x0s[i]); n = 0
        for c in cuts
            l0 = length(rec)
            xr = copy(x)
            brushless_run!(ks[i], xr, b.us, b.du, b.non, b.nmid, b.noff, n, c - n, decim, zeros(2), Ref(0.0), Ref(0.0), Ref(false), Ref(false))
            _bx_ila_call!(rec, ks[i], x, b, n, c - n, decim, nmax)
            x == xr || error("the ILA mirror differs from brushless_run! (instance $i, case $case)")
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

_bx_f(v) = repr(Float64(v))
_bx_arr(a::AbstractMatrix) = "{" * join(("{" * join(map(_bx_f, a[i, :]), ", ") * "}" for i in 1:size(a, 1)), ", ") * "}"

# the C simulation: every instance in cases A and B through brushless_run!, cut at every 0.01 s and around the
# events; states and outputs against the same @hls code in Julia; limit 5e-8 relative to the scales; nonzero on failure
function brushless_rt_testbench(path, ds, ks, x0s; h)
    S = BRUSHLESS_RT_SCALE
    blocks = String[]; calls = String[]
    for (i, (d, k, x0)) in enumerate(zip(ds, ks, x0s)), case in (:A, :B)
        b = brushless_rt_bench(d, case, h); cuts = _brushless_rt_cuts(h, b)
        rec, outs = _brushless_rt_julia(k, x0, b, cuts)
        tag = "I$(i)$(case)"
        push!(blocks, """
        // instance $i ($(d["machine"])), case $case
        const int NC_$tag = $(length(cuts)), ON_$tag = $(b.non), MID_$tag = $(b.nmid), OFF_$tag = $(b.noff);
        const int CUTS_$tag[NC_$tag] = {$(join(cuts, ", "))};
        const double US_$tag[4][3] = $(_bx_arr(b.us));
        const double DU_$tag[4][3] = $(_bx_arr(b.du));
        const double X0_$tag[3] = {$(join(map(_bx_f, x0), ", "))};
        const double J_$tag[NC_$tag][5] = {$(join(("{" * join(map(_bx_f, vcat(rec[g+1], outs[g+1])), ", ") * "}" for g in eachindex(cuts)), ",\n    "))};
        """)
        push!(calls, "    w = fmax(w, run_case(\"$(d["machine"]) $case\", BRUSHLESS_COEFS_$i, X0_$tag, US_$tag, DU_$tag, ON_$tag, MID_$tag, OFF_$tag, NC_$tag, CUTS_$tag, J_$tag));")
    end
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.brushless_rt_testbench (src/rt/exciters/brushless.jl); do not edit
        // C simulation: every instance in bench cases A (vterm 0.5 pu, i_fd ramp x 5.25, omega ramp to 0.97, 0.1-0.2 s)
        // and B (vterm 1.4 pu, 0.1-0.2 s), h = $h s, cut every 0.01 s and around the events; the C++ states and outputs
        // next to the same @hls code in Julia (Float64). Limit (fixed before results, 38b): $(BRUSHLESS_RT_LIM) relative to
        // the scales x $(S.x), v_f $(S.vf), P $(S.P). Continuous = segmented (instance 1, case A). Returns 1 on failure.
        #include <cstdio>
        #include <cmath>
        #include "brushless.h"
        #include "brushless_coefs.h"

        $(join(blocks, "\n"))
        double run_case(const char *name, const BrushlessCoefs &k, const double *x0, const double us0[4][3], const double du0[4][3],
                        int on, int mid, int off, int nc, const int *cuts, const double J[][5]) {
            St x[3]; Sig us[4][3]; Small du[4][3]; Pw y[2]; St xa; Pw vf; bool st, sa;
            for (int i = 0; i < 3; i++) x[i] = x0[i];
            for (int s = 0; s < 4; s++) for (int j = 0; j < 3; j++) { us[s][j] = us0[s][j]; du[s][j] = du0[s][j]; }
            double w = 0; int prev = 0;
            for (int g = 0; g < nc; g++) {
                brushless_run(k, x, us, du, on, mid, off, prev, cuts[g] - prev, 32, y, xa, vf, st, sa);
                prev = cuts[g];
                double e = 0;
                for (int i = 0; i < 3; i++) e = fmax(e, fabs(x[i].to_double() - J[g][i]) / $(S.x));
                e = fmax(e, fabs(y[0].to_double() - J[g][3]) / $(S.vf));
                e = fmax(e, fabs(y[1].to_double() - J[g][4]) / $(S.P));
                w = fmax(w, e);
            }
            printf("%-22s %4d checkpoints: x_E %.10f xa %.10f xf %.10f v_f %.6e P %.6e, max relative difference %.3e\\n", name, nc,
                   x[0].to_double(), x[1].to_double(), x[2].to_double(), y[0].to_double(), y[1].to_double(), w);
            return w;
        }

        // the square root in the C++ fixed-point types (review 39, 4): bx_sqrt(a) against sqrt of the quantized input a
        // (the root's own error; the input's quantization is printed apart), over 200,000 values of a in [1e-3, 64) and
        // at every ROM-bin and normalisation boundary (and one grid step either side); limit 1e-10 relative
        double sqrt_sweep() {
            double w = 0, wq = 0, wa = 0;
            const double ulp = ldexp(1.0, -40);
            auto one = [&](double av) {
                if (av < 1e-3 || av >= 64) return;
                Sig a = av; double aq = a.to_double();
                double e = fabs(bx_sqrt(a).to_double() / sqrt(aq) - 1);
                if (e > w) { w = e; wa = aq; }
                wq = fmax(wq, fabs(sqrt(aq) / sqrt(av) - 1));
            };
            for (int i = 0; i < 200000; i++) one(1e-3 * pow(64e3, i / 199999.0));
            for (int e = -5; e <= 2; e++)
                for (int j = 0; j <= 256; j++) {
                    double m = j < 128 ? 0.25 * (1 + j / 128.0) : (j < 256 ? 0.5 * (1 + (j - 128) / 128.0) : 1.0);
                    double av = m * ldexp(1.0, 2 * e);
                    one(av - ulp); one(av); one(av + ulp);
                }
            printf("bx_sqrt sweep: max relative error %.3e at a = %.6e (limit 1e-10); input quantization alone %.3e\\n", w, wa, wq);
            return w;
        }

        int main() {
            double w = 0;
            int fail_sqrt = !(sqrt_sweep() <= 1e-10);
        $(join(calls, "\n"))
            St x[3], xc[3]; Sig us[4][3]; Small du[4][3]; Pw y[2]; St xa; Pw vf; bool st, sa;
            for (int i = 0; i < 3; i++) { x[i] = X0_I1A[i]; xc[i] = X0_I1A[i]; }
            for (int s = 0; s < 4; s++) for (int j = 0; j < 3; j++) { us[s][j] = US_I1A[s][j]; du[s][j] = DU_I1A[s][j]; }
            int prev = 0;
            for (int g = 0; g < NC_I1A; g++) { brushless_run(BRUSHLESS_COEFS_1, x, us, du, ON_I1A, MID_I1A, OFF_I1A, prev, CUTS_I1A[g] - prev, 32, y, xa, vf, st, sa); prev = CUTS_I1A[g]; }
            brushless_run(BRUSHLESS_COEFS_1, xc, us, du, ON_I1A, MID_I1A, OFF_I1A, 0, CUTS_I1A[NC_I1A - 1], 32, y, xa, vf, st, sa);
            int same = 1;
            for (int i = 0; i < 3; i++) same = same && (x[i] == xc[i]);
            int fail = !(w <= $(BRUSHLESS_RT_LIM) && same) || fail_sqrt;
            printf("C sim check: max relative difference %.3e (limit %.0e), continuous = segmented %s, sqrt sweep %s: %s\\n", w,
                   $(BRUSHLESS_RT_LIM), same ? "yes" : "NO", fail_sqrt ? "FAIL" : "ok", fail ? "FAIL" : "PASS");
            return fail;
        }
        """)
    end
    println("written $path")
    return path
end

# the board program: instance 1 in cases A and B, instances 2 and 3 in case A (coefficients reloaded on the same
# bitstream, 38, 5), then case C (10^7 steps at the operating point); against the embedded Julia values
function brushless_rt_board_script(path, ds, ks, x0s; h, decim = BRUSHLESS_RT_DECIM)
    S = BRUSHLESS_RT_SCALE; F = map(I -> 48 - I, BRUSHLESS_RT_I)
    runs = String[]
    for (i, case) in BRUSHLESS_RT_BOARD_RUNS
        d, k, x0 = ds[i], ks[i], x0s[i]
        b = brushless_rt_bench(d, case, h)
        cuts = [b.non, b.nmid, round(Int, 0.3 / h)]
        rec, outs = _brushless_rt_julia(k, x0, b, cuts)
        push!(runs, "($i, \"$case\", \"$(d["machine"])\", $(repr(vec(permutedims(b.us)))), $(repr(vec(permutedims(b.du)))), " *
                    "$(b.non), $(b.nmid), $(b.noff), $(repr(x0)), $cuts, " *
                    "$(repr([vcat(rec[g+1], outs[g+1]) for g in eachindex(cuts)])))")
    end
    bC = brushless_rt_bench(ds[1], :C, h)
    coefs = map(ks) do k
        c = Tuple{Float64,Int}[]
        for (fld, T) in BRUSHLESS_RT_FIELDS
            v = getfield(k, fld); f = getfield(F, T)
            vals = v isa AbstractMatrix ? [v[i, j] for i in 1:size(v, 1) for j in 1:size(v, 2)] : v isa AbstractVector ? collect(v) : [v]
            foreach(e -> push!(c, (e, f)), vals)
        end
        "[" * join(("($(_bx_f(v)), $f)" for (v, f) in c), ", ") * "]"
    end
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.brushless_rt_board_script (src/rt/exciters/brushless.jl); do not edit.
        # Runs the brushless exciter IP on the PYNQ-Z2 (next to it: brushless.bit, brushless.hwh, xbrushless_run_hw.h).
        #   python3 brushless_board.py [--no-download]
        import sys, time, re
        from pynq import Overlay, Clocks

        H, DECIM, LIM = $(_bx_f(h)), $decim, $(BRUSHLESS_RT_LIM)
        SX, SVF, SP = $(S.x), $(S.vf), $(S.P)
        FST, FSIG, FSMALL, FPW = $(F.St), $(F.Sig), $(F.Small), $(F.Pw)
        COEFS = [$(join(coefs, ",\n         "))]
        RUNS = [$(join(runs, ",\n        "))]
        US_C, DU_C, X0_C = $(repr(vec(permutedims(bC.us)))), $(repr(vec(permutedims(bC.du)))), $(repr(x0s[1]))

        A = {}
        for m in re.finditer(r"#define XBRUSHLESS_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xbrushless_run_hw.h").read()):
            A[m.group(1)] = int(m.group(3), 16)

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("brushless.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / BRUSHLESS_RT_CLOCK_NS)) > 0.01:
            Clocks.fclk0_mhz = $(1000 / BRUSHLESS_RT_CLOCK_NS)
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.brushless_run_0

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
            return t, [r48(A["X"] + 8 * j, FST) for j in range(3)], [r48(A["Y"] + 8 * j, FPW) for j in range(2)]

        worst = 0.0
        for (i, case, mach, us, du, on, mid, off, x0, cuts, ref) in RUNS:
            load(i)
            print("instance %d (%s), case %s: compared at %s s" % (i, mach, case, [round(c * H, 3) for c in cuts]))
            x, n, tt = list(x0), 0, 0.0
            for g, c in enumerate(cuts):
                t, x, y = run(x, us, du, on, mid, off, n, c - n)
                tt += t; n = c
                e = max(max(abs(x[j] - ref[g][j]) for j in range(3)) / SX, abs(y[0] - ref[g][3]) / SVF, abs(y[1] - ref[g][4]) / SP)
                worst = max(worst, e)
                print("  t = %.3f s: x_E %.10f xa %.10f xf %.10f v_f %.6e P %.6e; max relative |board - Julia| %.3e"
                      % (c * H, x[0], x[1], x[2], y[0], y[1], e))
            print("  host elapsed time %.4f s (simulated %.4f s)" % (tt, cuts[-1] * H))

        load(1)
        N2 = 10_000_000
        print("case C: instance 1 at its operating point,", N2, "steps (%.3f s simulated)" % (N2 * H))
        t, x, y = run(X0_C, US_C, DU_C, 1 << 30, 1 << 30, 1 << 30, 0, N2)
        dx = max(abs(x[j] - X0_C[j]) for j in range(3))
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
