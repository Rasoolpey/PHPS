# The switching SiC bridge over one step in port-Hamiltonian form (PH_TODO.md step 2, B3; review 88 rows 11-13).
# `bridge_structure` builds Rac = (2/3) sum_k w_k u_k u_k' from the legs' time fractions and the device's data,
# independently of the drop's accumulation in `bridge_average`; this checks, over random windows (both frames, with and
# without dead time inside, current reversals within the window, zero current), that udrop = -Rac i and pcond = i' Rac i
# exactly, that Rac is symmetric positive semidefinite with w_k >= 0, that the fractions add up to at most one, that the
# switching sink is non-negative, and that the lossless part is lossless: Re((v m) conj(i)) = v Re(m conj(i)).

using Test, LinearAlgebra, Random
using PHPS
const _BP = PHPS

const _BWB = 2pi * 60
_bnew(; ieps = 0.01) = _BP.SwitchingBridge(_BP.CAB760M12HM3_125C; Sr = 7.0, omega_b = _BWB, ieps_frac = ieps)

function _b_one(br, i, v)
    r = _BP.bridge_average(br, i, v)
    S = _BP.bridge_structure(br, r, i, v)
    iv = [real(i), imag(i)]
    ud = [real(r.udrop), imag(r.udrop)]
    sc = max(abs(r.udrop), opnorm(S.Rac) * abs(i), 1e-300)
    return (; r, S, drop = maximum(abs, ud .+ S.Rac * iv) / sc,
            pc = abs(r.pcond - dot(iv, S.Rac * iv)) / max(abs(r.pcond), opnorm(S.Rac) * abs2(i), 1e-300),
            asym = maximum(abs, S.Rac - S.Rac') / max(opnorm(S.Rac), 1e-300),
            emin = eigmin(Symmetric(S.Rac)) / max(opnorm(S.Rac), 1e-300),
            frac = maximum(r.fU .+ r.fL .+ r.fC), wmin = minimum(S.w), psw = r.psw, Gsw = S.Gsw)
end

@testset "switching bridge over a step: port-Hamiltonian form (B3)" begin
    rng = MersenneTwister(3)
    worst = (drop = 0.0, pc = 0.0, asym = 0.0, emin = 0.0, frac = 0.0, wmin = Inf, psw = Inf, Gsw = Inf)
    ndead = 0; ntiny = 0; nsmallv = 0
    for n in 1:5000
        br = _bnew()
        _BP.bridge_start!(br, rand(rng) * 1.1 * cis(2pi * rand(rng)), 0.0; stationary = rand(rng) < 0.5,
                          theta0 = 2pi * rand(rng))
        a = rand(rng) * 40e-6
        _BP.bridge_window!(br, a, a + rand(rng) * 2e-6 + 1e-8)
        br.islope = 1e5 * randn(rng, ComplexF64)          # large: the current may reverse within the window
        # every 50th zero current; every 7th tiny (1e-12: the diode's limit while clamped); every 11th a small DC
        # voltage (1e-2 or 1e-6 pu: Gsw large)
        i = n % 50 == 0 ? 0.0im : n % 7 == 0 ? 1e-12 * randn(rng, ComplexF64) :
            1e-3 * randn(rng, ComplexF64) * 10^(3rand(rng))
        v = n % 11 == 0 ? (n % 22 == 0 ? 1e-6 : 1e-2) : 1.0 + 0.1rand(rng)
        q = _b_one(br, i, v)
        ndead += any(q.r.fC .> 0)
        ntiny += any(q.r.fC .> 0) && 0 < abs(i) < 1e-10
        nsmallv += v < 0.1
        worst = (drop = max(worst.drop, q.drop), pc = max(worst.pc, q.pc), asym = max(worst.asym, q.asym),
                 emin = min(worst.emin, q.emin), frac = max(worst.frac, q.frac), wmin = min(worst.wmin, q.wmin),
                 psw = min(worst.psw, q.psw), Gsw = min(worst.Gsw, q.Gsw))
    end
    @info "bridge_structure over 5000 random windows" worst ndead
    @test worst.drop <= 1e-13                  # udrop = -Rac i
    @test worst.pc <= 1e-13                    # pcond = i' Rac i
    @test worst.asym == 0                      # Rac symmetric by construction
    @test worst.emin >= -1e-15                 # and positive semidefinite
    @test worst.wmin >= 0
    @test worst.frac <= 1 + 1e-12              # upper + lower + clamped <= 1 (the rest: the dead time's swing)
    @test worst.psw >= 0 && worst.Gsw >= 0
    @test ndead > 100                          # the windows did cover dead times (non-vacuous)
    @test ntiny > 10                           # clamped windows at tiny current (the diode's v_SD / abs(i) near 0)
    @test nsmallv > 100
    # zero current: no drop
    br = _bnew(); _BP.bridge_start!(br, 0.7 + 0.2im, 0.0; stationary = false, theta0 = 0.3)
    _BP.bridge_window!(br, 12e-6, 13e-6)
    q = _b_one(br, 0.0im, 1.05)
    @test q.r.udrop == 0 && q.r.pcond == 0
    # the lossless part: the AC power of v m equals the DC draw v Re(m conj(i)) (the modulation is a transformer)
    i = 0.3 - 0.4im; v = 1.07
    q = _b_one(br, i, v)
    @test real(v * q.r.m * conj(i)) ≈ v * real(q.S.m * conj(i)) rtol = 1e-15
    # outside the domain; the diagnostic drop mode refused
    @test_throws DomainError _BP.bridge_structure(br, q.r, i, 0.0)
    _BP.BRIDGE_DROP_PIECE[] = true
    try
        @test_throws ArgumentError _BP.bridge_structure(br, q.r, i, v)
    finally
        _BP.BRIDGE_DROP_PIECE[] = false
    end
end

@testset "switching bridge: a current reversal inside a dead time, its clamp direction, and event ownership (B3)" begin
    # stationary frame, leg 1 (a_1 = 0: its current is Re(i)); a dead time of leg 1 and a window around it
    br = _bnew(); _BP.bridge_start!(br, 0.5 + 0.1im, 0.0; stationary = true, theta0 = 0.0)
    ts, ss = br.tchg[1], br.schg[1]
    j = findfirst(k -> ss[k] === :D && k < length(ts) && ts[k] > 1e-6, eachindex(ss))
    @test j !== nothing
    tD, tE = ts[j], ts[j+1]                     # the dead time [tD, tE)
    a, b = tD - 0.25e-6, tE + 0.25e-6; tm = (a + b) / 2
    _BP.bridge_window!(br, a, b)
    ts_ = tD + (tE - tD) / 4                    # the leg current crosses zero a quarter into the dead time
    S0 = 2e5
    res = map((+1, -1)) do sgn
        br.islope = sgn * S0 + 0im
        i = complex(-(ts_ - tm) * sgn * S0, 0.0)
        il(t) = real(i + (t - tm) * br.islope)
        @test sign(il(tD)) == -sign(il(tE))     # the reversal is inside the dead time (by construction)
        q = _b_one(br, i, 1.0)
        @test q.drop <= 1e-13 && q.pc <= 1e-13
        (; m = q.r.m, sgn)
    end
    # the clamp follows the piece's own current: with the current positive over most of the dead time (slope +) the
    # pole is driven to the lower rail, so leg 1's average pole position, and Re(m-bar), is lower than with slope -
    @test real(res[1].m) < real(res[2].m)
    # event ownership: a change exactly at a window's start belongs to that window (hard: the current chosen so)
    k = findfirst(t -> t > 2e-6, ts)
    tk = ts[k]; st, sn = ss[k-1], ss[k]
    hard = (st === :U || sn === :U) ? 0.5 : -0.5        # U -> D, D -> U hard for i > 0; L -> D, D -> L for i < 0
    h = 1e-8
    others = [t for l in 2:3 for t in br.tchg[l] if tk - h <= t < tk + h]
    br.islope = 0.0im
    _BP.bridge_window!(br, tk - h, tk); r0 = _BP.bridge_average(br, complex(hard, 0.0), 1.0)
    _BP.bridge_window!(br, tk, tk + h); r1 = _BP.bridge_average(br, complex(hard, 0.0), 1.0)
    @test isempty(others)                       # no other leg's change near tk (else the window choice is wrong)
    @test r0.psw == 0
    @test r1.psw > 0
end
