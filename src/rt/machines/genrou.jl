# =====================================================================================================================
# GENROU (6th-order round-rotor machine) in the real-time style, for export to C++ and the FPGA
# =====================================================================================================================
#
# Started from src/rt/TEMPLATE.jl (2026-10-24); the template's instructions apply (see that file). Plan, decisions and
# checks: docs/realtime/docs/GENROU_PLAN.md (draft 3, design reviews 27, 27b). Sections 2-9 follow the plan.
#
# Loaded by src/rt/RT.jl. The step function (section 5) is written with @hls: it runs in Julia as written and
# genrou_rt_export() translates it line by line to src/rt/machines/CPP/genrou/genrou.cpp, with the coefficients in
# genrou_coefs.h (`julia --project=. scripts/realtime/export_cpp.jl genrou`).
# =====================================================================================================================


# ---------------------------------------------------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------------------------------------------------
# Source:      src/components/machines/genrou.jl (GENROU_PHTRUE), the same equations; user choices 2026-10-24:
#              the 6-state phasor (RMS) model, split step, infinite-bus test bench.
# Frame:       phasor model; the bus voltage in the network RI frame; the machine's own dq frame at angle delta.
# States x:    delta (rad, kept in [-pi, pi)), omega (pu), E_q' (pu), psi_d (pu),
#              E_d' (pu), psi_q (pu), and cos delta, sin delta carried along      (6 states + 2)
#              rotor part z = (E_q', psi_d, E_d', psi_q)
# Inputs u:    V_Re, V_Im (bus voltage, scheduled per step on the test bench), Tm, Efd (scheduled)   (4 inputs)
# Outputs y:   id, iq, It_Re, It_Im, omega, theta, Pe (from the helper genrou_rt_stator; the kernel returns states)
# Equations:   vd = V_Re sin d - V_Im cos d,  vq = V_Re cos d + V_Im sin d
#              psi_d'' = kd E_q' + (1 - kd) psi_d,  psi_q'' = -kq E_d' + (1 - kq) psi_q
#              kd = (xd'' - xl)/(xd' - xl),  kq = (xq'' - xl)/(xq' - xl)
#              (id, iq) = S (vd + omega psi_q'', vq - omega psi_d''),  S = [-ra -xq''; xd'' -ra] / (ra^2 + xd'' xq'')
#              Pe = vd id + vq iq
#              delta' = omega_b (omega - 1)
#              omega' = (Tm/omega - Pe/omega - D (omega - 1)) / 2H
#              z' = A z + Bf Efd + Bs (id, iq)    (A, Bf, Bs constant: rotor_structure in genrou.jl)
# pH form:     rotor magnetic storage H_mag = z' Q z / 2 plus H (omega - 1)^2 (genrou.jl hamiltonian)
# Events:      none in the machine (one bank); the dip is in the scheduled bus voltage
# Modes:       none
# Step:        h = 1 us; split (plan section 2): rotor by implicit midpoint with currents held at the start of the
#              step (P, W prepared); speed with midpoint damping; angle by trapezoid, kept in [-pi, pi)
# Nonlinear:   cos delta, sin delta carried as states and rotated by the step's angle increment; 1/omega by
#              Newton steps (seed 2 - omega). No sin/cos unit and no divider in the hardware (C synthesis 2026-10-25:
#              hls::sin took ~100 cycles and the divider ~140, against a budget of 100 cycles per 1 us step).

const GENROU_RT_NX = 8          # entries of the state array (6 states + cos delta, sin delta)
const GENROU_RT_NU = 4          # number of inputs
const GENROU_RT_NB = 1          # number of coefficient banks (no events in the machine)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Parameters and continuous-time matrices (preparation; BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""
    genrou_rt_params(; H, D, ra, xd, xq, xd_prime, xq_prime, xd_double_prime, xq_double_prime, Td0_prime, Tq0_prime,
                     Td0_double_prime, Tq0_double_prime, xl, fb = 60) -> NamedTuple

The machine parameters (names as in src/components/machines/genrou.jl) in the current BigFloat precision; pass decimal
strings (`xd = "1.8"`) so they are parsed exactly. `omega_b = 2 pi fb`.
"""
function genrou_rt_params(; H, D, ra, xd, xq, xd_prime, xq_prime, xd_double_prime, xq_double_prime, Td0_prime,
                          Tq0_prime, Td0_double_prime, Tq0_double_prime, xl, fb = 60)
    p(v) = v isa AbstractString ? parse(BigFloat, v) : BigFloat(v)
    return (; H = p(H), D = p(D), ra = p(ra), xd = p(xd), xq = p(xq), xdp = p(xd_prime), xqp = p(xq_prime),
            xdpp = p(xd_double_prime), xqpp = p(xq_double_prime), Td0p = p(Td0_prime), Tq0p = p(Tq0_prime),
            Td0pp = p(Td0_double_prime), Tq0pp = p(Tq0_double_prime), xl = p(xl), wb = 2 * BigFloat(pi) * p(fb))
end

"""
    genrou_rt_AB(par) -> (A, B)

The rotor z = (E_q', psi_d, E_d', psi_q): z' = A z + B (Efd, id, iq), B = [Bf Bs] (as `rotor_structure` in genrou.jl).
"""
function genrou_rt_AB(par)
    z = zero(par.H)
    A = [-1/par.Td0p z z z;
         1/par.Td0pp -1/par.Td0pp z z;
         z z -1/par.Tq0p z;
         z z -1/par.Tq0pp -1/par.Tq0pp]
    B = [1/par.Td0p -(par.xd - par.xdp)/par.Td0p z;
         z -(par.xdp - par.xl)/par.Td0pp z;
         z z (par.xq - par.xqp)/par.Tq0p;
         z z -(par.xqp - par.xl)/par.Tq0pp]
    return A, B
end


# ---------------------------------------------------------------------------------------------------------------------
# 3. Discretisation (preparation)
# ---------------------------------------------------------------------------------------------------------------------
"""
    genrou_rt_PW(par, h) -> (P, W)

Rotor by implicit midpoint with the inputs held over the step: z_n = z_{n-1} + P z_{n-1} + W (Efd, id, iq),
P = h M^-1 A, W = h M^-1 B, M = I - (h/2) A.
"""
function genrou_rt_PW(par, h)
    A, B = genrou_rt_AB(par)
    M = Matrix{BigFloat}(I, 4, 4) - (h / 2) * A
    return h * (M \ A), h * (M \ B)
end

"""
    genrou_rt_coefs(par, h) -> NamedTuple of BigFloat

Every coefficient of the step (plan section 2): kd, 1 - kd, -kq, 1 - kq; the stator matrix S (row-major); P, W; the
speed coefficients a1, a2; the angle coefficient kth = h omega_b / 2 pi (turns per pu speed deviation).
"""
function genrou_rt_coefs(par, h)
    kd = (par.xdpp - par.xl) / (par.xdp - par.xl)
    kq = (par.xqpp - par.xl) / (par.xqp - par.xl)
    det = par.ra^2 + par.xdpp * par.xqpp
    P, W = genrou_rt_PW(par, h)
    c = h * par.D / (2 * par.H)
    return (; kd, kd1 = 1 - kd, kqn = -kq, kq1 = 1 - kq,
            S = (-par.ra / det, -par.xqpp / det, par.xdpp / det, -par.ra / det), P, W,
            a1 = (1 - c / 2) / (1 + c / 2), a2 = (h / (2 * par.H)) / (1 + c / 2), kth = h * par.wb / (2 * BigFloat(pi)))
end


# ---------------------------------------------------------------------------------------------------------------------
# 4. The coefficients of the step (the C++ struct GenrouCoefs)
# ---------------------------------------------------------------------------------------------------------------------
@hls struct GenrouCoefs
    P::Small[4, 4]      # rotor step matrix
    W::Small[4, 3]      # rotor input matrix (inputs Efd, id, iq)
    kd::Sig             # subtransient weight, d axis
    kq::Sig             # subtransient weight, q axis
    S::Sig[4]           # stator matrix, row-major
    a1::Sig             # speed damping coefficient
    a2::Small           # speed forcing coefficient h / 2H
    wb_h::Small         # omega_b h
end

"""
    genrou_rt_coefstruct(par, h) -> GenrouCoefs

The coefficients for the machine `par` and step `h`, computed in BigFloat by `genrou_rt_coefs`, stored as Float64.
"""
function genrou_rt_coefstruct(par, h)
    c = genrou_rt_coefs(par, h)
    return GenrouCoefs(Float64.(c.P), Float64.(c.W), Float64(c.kd), Float64(-c.kqn), Float64.(collect(c.S)),
                       Float64(c.a1), Float64(c.a2), Float64(2 * BigFloat(pi) * c.kth))
end


# ---------------------------------------------------------------------------------------------------------------------
# 5. THE STEP FUNCTION (this function becomes the C++ / the hardware)
# ---------------------------------------------------------------------------------------------------------------------
# x = (delta [rad], omega [pu], Eq', psi_d, Ed', psi_q, cos delta, sin delta), u = (V_Re, V_Im, Tm, Efd);
# one step of h, x changed in place. cos delta and sin delta are carried as states and rotated each step by the small
# angle increment (no sin/cos unit: about 10 clock cycles instead of 100); 1/omega by Newton steps (no divider).
@hls function genrou_step!(x::Sig[8], u::Sig[4], k::GenrouCoefs)
    @pragma "HLS ARRAY_PARTITION variable=x complete"     # states and inputs as separate registers:
    @pragma "HLS ARRAY_PARTITION variable=u complete"     # all read and written in parallel
    c::Sig = x[7]
    s::Sig = x[8]
    vd::Sig = u[1] * s - u[2] * c                    # Park transform
    vq::Sig = u[1] * c + u[2] * s
    pdd::Sig = k.kd * x[3] + (1 - k.kd) * x[4]       # subtransient fluxes
    pqq::Sig = -k.kq * x[5] + (1 - k.kq) * x[6]
    rd::Sig = vd + x[2] * pqq
    rq::Sig = vq - x[2] * pdd
    id::Sig = k.S[1] * rd + k.S[2] * rq             # stator currents
    iq::Sig = k.S[3] * rd + k.S[4] * rq
    Pe::Sig = vd * id + vq * iq                      # terminal power

    v::Sig[3] = (u[4], id, iq)                       # rotor fluxes: implicit midpoint, currents held
    z::Sig[4]
    for i in 1:4
        d::Sig = 0
        for j in 1:4
            d += k.P[i, j] * x[2 + j]
        end
        for j in 1:3
            d += k.W[i, j] * v[j]
        end
        z[i] = x[2 + i] + d
    end

    r0::Sig = 2 - x[2]                               # 1/omega: seed 2 - omega, then three Newton steps
    t1::Sig = x[2] * r0                              # r = r (2 - omega r); every product rounded to Sig, so the
    r1::Sig = r0 * (2 - t1)                          # multipliers stay 48 x 48 bits; straight lines (no loop), so
    t2::Sig = x[2] * r1                              # HLS can run them alongside the stator and rotor
    r2::Sig = r1 * (2 - t2)
    t3::Sig = x[2] * r2
    r::Sig = r2 * (2 - t3)
    f::Sig = (u[3] - Pe) * r                         # (Tm - Pe) / omega
    w1::Sig = 1 + k.a1 * (x[2] - 1) + k.a2 * f       # speed

    dd::Small = k.wb_h * ((x[2] + w1) / 2 - 1)       # angle increment of the step (trapezoid)
    cd::Sig = 1 - dd * dd / 2                        # cos and sin of the increment (series; error below the last bit)
    c1::Sig = c * cd - s * dd                        # rotate (cos delta, sin delta)
    s1::Sig = s * cd + c * dd
    g::Sig = (3 - (c1 * c1 + s1 * s1)) / 2           # keep cos^2 + sin^2 = 1
    d1::Sig = x[1] + dd                              # delta, kept in [-pi, pi)
    piv::Sig = pi
    twopi::Sig = 2pi
    if d1 >= piv
        d1 -= twopi
    end
    if d1 < -piv
        d1 += twopi
    end

    x[1] = d1
    x[2] = w1
    for i in 1:4
        @pragma "HLS UNROLL"                         # the four copies at once (registers)
        x[2 + i] = z[i]
    end
    x[7] = c1 * g
    x[8] = s1 * g
end


# ---------------------------------------------------------------------------------------------------------------------
# 5b. THE BOARD TOP FUNCTION (the IP that runs on the FPGA)
# ---------------------------------------------------------------------------------------------------------------------
# The processor writes the coefficients, the initial state, the inputs and the voltage dip over AXI-Lite and starts the
# run; the IP then performs nsteps steps, one every 118 clock cycles (0.944 us at 125 MHz), with the
# states kept inside the FPGA. After every step delta, omega, E_q', psi_d and a toggling strobe are on output ports
# (for the ILA in Vivado), and `sample` is 1 during one step in every `decim` steps (the ILA records only then, so its
# memory covers a longer time); at the end the final state is written back to x.
@hls function genrou_run!(k::GenrouCoefs, x::Sig[8], vre::Sig, vim::Sig, tm::Sig, efd::Sig, vdip::Sig, n_on::Int,
                          n_off::Int, nsteps::Int, decim::Int, delta_out::Out{Sig}, omega_out::Out{Sig},
                          eq_out::Out{Sig}, psid_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=k"
    @pragma "HLS INTERFACE mode=s_axilite port=x"
    @pragma "HLS INTERFACE mode=s_axilite port=vre"
    @pragma "HLS INTERFACE mode=s_axilite port=vim"
    @pragma "HLS INTERFACE mode=s_axilite port=tm"
    @pragma "HLS INTERFACE mode=s_axilite port=efd"
    @pragma "HLS INTERFACE mode=s_axilite port=vdip"
    @pragma "HLS INTERFACE mode=s_axilite port=n_on"
    @pragma "HLS INTERFACE mode=s_axilite port=n_off"
    @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=delta_out"
    @pragma "HLS INTERFACE mode=ap_none port=omega_out"
    @pragma "HLS INTERFACE mode=ap_none port=eq_out"
    @pragma "HLS INTERFACE mode=ap_none port=psid_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xs::Sig[8]                                       # the states, kept in registers during the run
    @pragma "HLS ARRAY_PARTITION variable=xs complete"
    for i in 1:8
        @pragma "HLS UNROLL"
        xs[i] = x[i]
    end
    u::Sig[4]
    @pragma "HLS ARRAY_PARTITION variable=u complete"
    st::Bool = false
    cnt::Int = 0                                     # steps since the last ILA sample
    for n in 1:nsteps
        @pragma "HLS PIPELINE II=118"                # one step every 118 clock cycles (GENROU_RT_STEP_CYCLES)
        vs::Sig = 1                                  # bus voltage factor: vdip during steps n_on+1 .. n_off
        if n > n_on && n <= n_off
            vs = vdip
        end
        u[1] = vre * vs
        u[2] = vim * vs
        u[3] = tm
        u[4] = efd
        genrou_step!(xs, u, k)
        st = !st
        cnt += 1                                     # sample = 1 for one step every decim steps (the ILA records
        if cnt >= decim                              # only then: with decim = 32, 16k samples cover 0.5 s)
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
        delta_out[] = xs[1]                          # to the ILA
        omega_out[] = xs[2]
        eq_out[] = xs[3]
        psid_out[] = xs[4]
        strobe[] = st
    end
    for i in 1:8
        @pragma "HLS UNROLL"
        x[i] = xs[i]
    end
end


# ---------------------------------------------------------------------------------------------------------------------
# 6. Exporting the C++
# ---------------------------------------------------------------------------------------------------------------------
# AP_RND_CONV: round to nearest (ties to even) instead of AMD's default truncation, which biases every result
# downwards and made the state drift by about 1e-12 per step (first C simulation, 2026-10-25)
# Real-time pacing (user decision 2026-10-25, option A): the step size h is what the hardware does. C synthesis of
# genrou_run at 8 ns (125 MHz) gives one step every 118 clock cycles (Final II = 118, the length of one step), so
# h = 118 x 8 ns = 0.944 us and simulated time advances exactly as fast as wall-clock time. If a later synthesis
# reports a different II, update GENROU_RT_STEP_CYCLES (and the PIPELINE II in genrou_run!) and export again.
const GENROU_RT_CLOCK_NS = 8
const GENROU_RT_STEP_CYCLES = 118
const GENROU_RT_H = "$(GENROU_RT_STEP_CYCLES * GENROU_RT_CLOCK_NS)e-9"     # "944e-9" s

const GENROU_RT_TYPES = [:Sig => "ap_fixed<48, 8, AP_RND_CONV>",       # signals: range +-128 pu, resolution 1e-12
                         :Small => "ap_fixed<48, -10, AP_RND_CONV>"]   # small coefficients: range +-4.9e-4

"""
    genrou_rt_testbench(path, par, k; V = 1.0 + 0im, S = 0.8 + 0.2im, vdip = 0.5, N = 300_000, every = 10_000)

Writes the C simulation testbench `path` (genrou_tb.cpp): the machine `par` starts at its operating point
(`init_from_phasor` of src/components at bus voltage V and power S, so Tm and Efd are those of the operating point);
the bus voltage drops to `vdip` x V from 0.1 s to 0.2 s and returns; 0.3 s in steps of `h`. Every 10 ms it prints the C++ states next to the same @hls step run in Julia (Float64) on the same schedule.
"""
function genrou_rt_testbench(path, par, k::GenrouCoefs; h = 1.2e-6, V = 1.0 + 0im, S = 0.8 + 0.2im, vdip = 0.5,
                             N = round(Int, 0.3 / h), every = round(Int, 0.01 / h),
                             ndip = (round(Int, 0.1 / h), round(Int, 0.2 / h)))
    P = parentmodule(@__MODULE__)                      # PHPS: the original GENROU and its initialisation
    names = Dict(:H => "H", :D => "D", :ra => "ra", :xd => "xd", :xq => "xq", :xdp => "xd_prime", :xqp => "xq_prime",
                 :xdpp => "xd_double_prime", :xqpp => "xq_double_prime", :Td0p => "Td0_prime", :Tq0p => "Tq0_prime",
                 :Td0pp => "Td0_double_prime", :Tq0pp => "Tq0_double_prime", :xl => "xl", :wb => "omega_b")
    d = P.ParamDict(v => Float64(getfield(par, f)) for (f, v) in names); d["bus"] = 1
    x0, tg = P.init_from_phasor(P.GENROU_PHTRUE("G", d), ComplexF64(V), ComplexF64(conj(S / V)))
    x = vcat(x0, cos(x0[1]), sin(x0[1]))
    X0 = copy(x)
    u = zeros(4); rec = Vector{Vector{Float64}}()
    for n in 0:N                                       # the Julia run of the same @hls step
        n % every == 0 && push!(rec, x[1:6])
        n == N && break
        vs = ndip[1] <= n < ndip[2] ? vdip : 1.0
        u .= (real(V) * vs, imag(V) * vs, tg.Tm, tg.Efd)
        genrou_step!(x, u, k)
    end
    f(v) = string(v)
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.genrou_rt_testbench (src/rt/machines/genrou.jl); do not edit
        // C simulation: GENROU at its operating point (V = $V, S = $S), bus voltage x $vdip for steps $(ndip[1])-$(ndip[2] - 1),
        // $N steps of h = $h s. Prints the C++ states next to the Julia @hls step (Float64) on the same schedule.
        #include <cstdio>
        #include <cmath>
        #include "genrou.h"
        #include "genrou_coefs.h"

        const int N = $N, EVERY = $every, NDIP_ON = $(ndip[1]), NDIP_OFF = $(ndip[2]);
        const double H = $h;
        const double V_RE = $(f(real(V))), V_IM = $(f(imag(V))), TM = $(f(tg.Tm)), EFD = $(f(tg.Efd)), VDIP = $(f(vdip));
        const double X0[8] = {$(join(map(f, X0), ", "))};
        const double JULIA[$(length(rec))][6] = {   // delta, omega, Eq', psi_d, Ed', psi_q from the Julia run
        $(join(("    {" * join(map(f, r), ", ") * "}" for r in rec), ",\n"))
        };

        int main() {
            Sig x[8], u[4];
            for (int i = 0; i < 8; i++) x[i] = X0[i];
            printf("%8s %14s %14s %14s %14s %12s\\n", "t [s]", "delta C++", "delta Julia", "omega C++", "omega Julia", "max |diff|");
            for (int n = 0; n <= N; n++) {
                if (n % EVERY == 0) {
                    const double *J = JULIA[n / EVERY];
                    double dmax = 0;
                    for (int i = 0; i < 6; i++) dmax = fmax(dmax, fabs(x[i].to_double() - J[i]));
                    printf("%8.3f %14.9f %14.9f %14.9f %14.9f %12.3e\\n", n * H, x[0].to_double(), J[0],
                           x[1].to_double(), J[1], dmax);
                }
                if (n == N) break;
                double vs = (n >= NDIP_ON && n < NDIP_OFF) ? VDIP : 1.0;
                u[0] = V_RE * vs; u[1] = V_IM * vs; u[2] = TM; u[3] = EFD;
                genrou_step(x, u, GENROU_COEFS);
            }
            return 0;
        }
        """)
    end
    println("written $path")
    return path
end

"""
    genrou_rt_board_script(path, par, k; h, V = 1.0 + 0im, S = 0.8 + 0.2im, vdip = 0.5, decim = 32)

Writes `path` (genrou_board.py), the Python program that runs the IP on the PYNQ-Z2: it loads the overlay
(genrou.bit + genrou.hwh), writes the coefficients, the initial state (the operating point from `init_from_phasor`),
the inputs and the dip over AXI-Lite (register map of the packaged IP: x at 0x40, k at 0x200, one 64-bit word per
value), then makes two runs: (1) the dip case of the C simulation testbench, 0.3 s, and compares the final state with
the same @hls step run in Julia (Float64); (2) 10^7 steps at the operating point, timed: real time means
10^7 x h seconds of wall-clock time, and the state must stay at the operating point.
"""
function genrou_rt_board_script(path, par, k::GenrouCoefs; h, V = 1.0 + 0im, S = 0.8 + 0.2im, vdip = 0.5, decim = 32)
    P = parentmodule(@__MODULE__)
    names = Dict(:H => "H", :D => "D", :ra => "ra", :xd => "xd", :xq => "xq", :xdp => "xd_prime", :xqp => "xq_prime",
                 :xdpp => "xd_double_prime", :xqpp => "xq_double_prime", :Td0p => "Td0_prime", :Tq0p => "Tq0_prime",
                 :Td0pp => "Td0_double_prime", :Tq0pp => "Tq0_double_prime", :xl => "xl", :wb => "omega_b")
    d = P.ParamDict(v => Float64(getfield(par, f)) for (f, v) in names); d["bus"] = 1
    x0, tg = P.init_from_phasor(P.GENROU_PHTRUE("G", d), ComplexF64(V), ComplexF64(conj(S / V)))
    X0 = vcat(x0, cos(x0[1]), sin(x0[1]))
    N1 = round(Int, 0.3 / h); non = round(Int, 0.1 / h); noff = round(Int, 0.2 / h)
    x = copy(X0); u = zeros(4)                         # the Julia run of the dip case (same rule as genrou_run!)
    for n in 1:N1
        vs = non < n <= noff ? vdip : 1.0
        u .= (real(V) * vs, imag(V) * vs, tg.Tm, tg.Efd)
        genrou_step!(x, u, k)
    end
    # the coefficients in the order of the struct (one 64-bit word each) with their fraction bits (Sig 40, Small 58)
    coefs = Tuple{Float64,Int}[]
    for i in 1:4, j in 1:4; push!(coefs, (k.P[i, j], 58)); end
    for i in 1:4, j in 1:3; push!(coefs, (k.W[i, j], 58)); end
    push!(coefs, (k.kd, 40), (k.kq, 40)); foreach(v -> push!(coefs, (v, 40)), k.S)
    push!(coefs, (k.a1, 40), (k.a2, 58), (k.wb_h, 58))
    f(v) = repr(Float64(v))
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.genrou_rt_board_script (src/rt/machines/genrou.jl); do not edit.
        # Runs the GENROU IP on the PYNQ-Z2. Copy next to it: genrou.bit and genrou.hwh (same base name), from the
        # Vivado project (system_wrapper.bit and system.hwh). Run as root on the board, or in a Jupyter notebook.
        #   python3 genrou_board.py                 loads the bitstream (PYNQ also sets the 125 MHz clock), then runs
        #   python3 genrou_board.py --no-download   keeps the bitstream already loaded (for the ILA) and
        #                                           sets the 125 MHz clock itself, then runs
        import sys, time
        from pynq import Overlay, Clocks

        H = $(f(h))                     # step size [s]: one step every $(GENROU_RT_STEP_CYCLES) cycles at $(1000 ÷ GENROU_RT_CLOCK_NS) MHz
        COEFS = [$(join(("($(f(v)), $F)" for (v, F) in coefs), ", "))]
        X0 = [$(join(map(f, X0), ", "))]
        V_RE, V_IM, TM, EFD = $(f(real(V))), $(f(imag(V))), $(f(tg.Tm)), $(f(tg.Efd))
        VDIP, N1, N_ON, N_OFF, DECIM = $(f(vdip)), $N1, $non, $noff, $decim
        X_JULIA = [$(join(map(f, x[1:6]), ", "))]   # final state of the dip case, Julia @hls step (Float64)
        NAMES = ["delta", "omega", "Eq'", "psi_d", "Ed'", "psi_q"]

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("genrou.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / GENROU_RT_CLOCK_NS)) > 0.01:   # set the IP's clock (PL clock 0) only when
            Clocks.fclk0_mhz = $(1000 / GENROU_RT_CLOCK_NS)              # needed: reprogramming it can glitch the clock
                                                              # and stall the ILA, which runs on the same clock
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.genrou_run_0                  # the IP's AXI-Lite registers (offsets from the packaged IP's driver)

        def raw(v, F):                        # a value as the raw bits of ap_fixed with F fraction bits (48 bits)
            return int(round(v * 2**F)) & ((1 << 48) - 1)
        def w48(addr, v, F):                  # one 48-bit value in a 64-bit slot: low 32 bits, then the high 16
            r = raw(v, F)
            ip.write(addr, r & 0xFFFFFFFF)
            ip.write(addr + 4, r >> 32)
        def r48(addr, F=40):
            r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
            if r >> 47:
                r -= 1 << 48
            return r / 2**F

        for i, (v, F) in enumerate(COEFS):    # k: 37 words from 0x200
            w48(0x200 + 8 * i, v, F)

        def run(nsteps, vdip, n_on, n_off):
            for i, v in enumerate(X0):        # x: 8 words from 0x40 (initial state; final state after the run)
                w48(0x40 + 8 * i, v, 40)
            w48(0x18, V_RE, 40); w48(0x24, V_IM, 40); w48(0x30, TM, 40); w48(0x80, EFD, 40); w48(0x8C, vdip, 40)
            ip.write(0x98, n_on); ip.write(0xA0, n_off); ip.write(0xA8, nsteps); ip.write(0xB0, DECIM)
            t0 = time.perf_counter()
            ip.write(0x00, 1)                 # ap_start
            while not (ip.read(0x00) & 0x2):  # ap_done
                time.sleep(0.001)
            t = time.perf_counter() - t0
            return t, [r48(0x40 + 8 * i) for i in range(8)]

        print("run 1: voltage dip to", VDIP, "pu from 0.1 s to 0.2 s,", N1, "steps (0.3 s of simulated time)")
        t, x = run(N1, VDIP, N_ON, N_OFF)
        print("  wall-clock time %.4f s (simulated %.4f s)" % (t, N1 * H))
        print("  %-6s %16s %16s %12s" % ("state", "board", "Julia", "difference"))
        for i in range(6):
            print("  %-6s %16.10f %16.10f %12.3e" % (NAMES[i], x[i], X_JULIA[i], x[i] - X_JULIA[i]))

        N2 = 10_000_000
        print("run 2: operating point,", N2, "steps (%.3f s of simulated time)" % (N2 * H))
        t, x = run(N2, 1.0, 0, 0)
        print("  wall-clock time %.4f s, %.4f us per step (h = %.4f us)" % (t, t / N2 * 1e6, H * 1e6))
        print("  max |x - x0| over the 6 states: %.3e (the operating point must hold)" % max(abs(x[i] - X0[i]) for i in range(6)))
        """)
    end
    println("written $path")
    return path
end

"""
    genrou_rt_export(par = <the test machine of the plan>; h = "1e-6", dir = src/rt/machines/CPP/genrou)

Writes, translated from this file, `dir/genrou.h` (types, the struct GenrouCoefs, the prototype) and `dir/genrou.cpp`
(the function genrou_step); `dir/genrou_coefs.h` (the coefficients of the machine `par`, const GenrouCoefs
GENROU_COEFS); and `dir/genrou_tb.cpp`, the C simulation testbench (`genrou_rt_testbench`).
"""
function genrou_rt_export(par = nothing; h = GENROU_RT_H, dir = joinpath(@__DIR__, "CPP", "genrou"))
    p = par === nothing ? setprecision(BigFloat, 256) do
        genrou_rt_params(; H = "3", D = "0", ra = "0", xd = "1.8", xq = "1.7", xd_prime = "0.3", xq_prime = "0.55",
                         xd_double_prime = "0.25", xq_double_prime = "0.25", Td0_prime = "8", Tq0_prime = "0.4",
                         Td0_double_prime = "0.03", Tq0_double_prime = "0.05", xl = "0.15")
    end : par
    hh = setprecision(() -> parse(BigFloat, h), BigFloat, 256)
    hls_write(joinpath(dir, "genrou.cpp"), [:GenrouCoefs, :genrou_step!, :genrou_run!]; types = GENROU_RT_TYPES,
              header = "GENROU, one step of h; from src/rt/machines/genrou.jl")
    k = setprecision(() -> genrou_rt_coefstruct(p, hh), BigFloat, 256)
    hls_write_const(joinpath(dir, "genrou_coefs.h"), "GENROU_COEFS", k;
                    header = "machine: " * join(("$n = $(Float64(v))" for (n, v) in pairs(p)), ", ") * "\nstep h = $(Float64(hh)) s")
    genrou_rt_testbench(joinpath(dir, "genrou_tb.cpp"), p, k; h = Float64(hh))
    genrou_rt_board_script(joinpath(dir, "genrou_board.py"), p, k; h = Float64(hh))
    # the model's settings for the automated flow (scripts/realtime/flow/flow.sh)
    flow_config(joinpath(dir, "flow.cfg"); model = "genrou", top = "genrou_run", clock_ns = GENROU_RT_CLOCK_NS,
                ip_version = "1.0", ip_name = "GENROU real-time step",
                probes = ["delta_out" => 48, "omega_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                plot = ["delta_out" => (40, "delta [rad]"), "omega_out" => (40, "omega [pu]")],
                sample_dt = 32 * Float64(hh), board_args = "--no-download")
end
