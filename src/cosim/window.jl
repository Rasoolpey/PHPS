# Window-held port coupling (COSIM_DESIGN.md section 3 and 8.1-8.3; review 102) on a small coupled PH pair: experiment
# group A. A slow side (a source E behind R_s into a node of storage H_s(q), or with no storage: an algebraic node,
# feed-through) and a fast side (R_f in series with an inductor of storage H_f(lambda), with an optional port
# conductance G_p: feed-through), joined at one power port (the node voltage v, the current i drawn by the fast side).
# Storages H(z) = z^2 / (2 k) + c4 z^4 / 4 (c4 > 0: nonlinear), discrete gradients in closed form.
#
# The window: the slow side's discrete output a (its discrete-gradient voltage) held for the fast side's steps h_j
# (any mesh summing to H); the fast side's h-weighted average current b enters the slow step. The window's equations are
# solved on the interface variable a (secant iteration with a safeguard), stopping on the vector residuals (8.3).

"""A storage H(z) = z^2 / (2 k) + c4 z^4 / 4 with its discrete gradient."""
struct QuarticStorage
    k::Float64
    c4::Float64
end
energy(s::QuarticStorage, z) = z^2 / (2s.k) + s.c4 * z^4 / 4
# (H(z1) - H(z0)) / (z1 - z0) in closed form (exact; no division by z1 - z0)
dgrad(s::QuarticStorage, z0, z1) = (z0 + z1) / (2s.k) + s.c4 * (z1^3 + z1^2 * z0 + z1 * z0^2 + z0^3) / 4

"""The slow side: E behind R_s into a node of storage S (charge q), or (S === nothing) an algebraic node: v = E - R_s i."""
struct SlowRC
    E::Float64
    Rs::Float64
    S::Union{Nothing,QuarticStorage}
end

"""The fast side: R_f in series with an inductor of storage S (flux lambda), and a port conductance G_p."""
struct FastRL
    Rf::Float64
    S::QuarticStorage
    Gp::Float64
end

"""
    slow_step(s, q0, H, b) -> (q1, a, ledger)

One slow step of size H with the fast side's average current b drawn from the node: the discrete-gradient voltage a is
the port output. Ledger: (dH, supplied by E, dissipated in R_s, port energy H a b out).
"""
function slow_step(s::SlowRC, q0, H, b)
    if s.S === nothing                       # algebraic node: v = E - R_s b
        a = s.E - s.Rs * b
        i_s = b
        return q0, a, (dH = 0.0, Esup = H * s.E * i_s, D = H * s.Rs * i_s^2, Eport = H * a * b)
    end
    S = s.S
    q1 = q0
    conv = false
    for _ in 1:60                           # Newton on (q1 - q0)/H = (E - vbar)/R_s - b
        v = dgrad(S, q0, q1)
        r = (q1 - q0) / H - (s.E - v) / s.Rs + b
        dv = (1 / (2S.k) + S.c4 * (3q1^2 + 2q1 * q0 + q0^2) / 4)
        dq = r / (1 / H + dv / s.Rs)
        q1 -= dq
        abs(dq) <= 1e-15 * max(1.0, abs(q1)) && (conv = true; break)
    end
    conv || error("slow_step: Newton did not converge")
    a = dgrad(S, q0, q1)
    i_s = (s.E - a) / s.Rs
    return q1, a, (dH = energy(S, q1) - energy(S, q0), Esup = H * s.E * i_s, D = H * s.Rs * i_s^2, Eport = H * a * b)
end

"""
    fast_window(f, l0, a, hs) -> (l1, b, ledger)

The fast side over the steps hs with the held voltage a: per step (l1 - l0)/h = a - R_f ibar, ibar the discrete gradient;
the port current ibar + G_p a; b its h-weighted average. Ledger: (dH, dissipated, port energy in = sum h a i_port).
"""
function fast_window(f::FastRL, l0, a, hs)
    S = f.S
    l = l0; Ib = 0.0; D = 0.0; H = 0.0
    for h in hs
        l1 = l
        conv = false
        for _ in 1:60
            ib = dgrad(S, l, l1)
            r = (l1 - l) / h - a + f.Rf * ib
            di = (1 / (2S.k) + S.c4 * (3l1^2 + 2l1 * l + l^2) / 4)
            dl = r / (1 / h + f.Rf * di)
            l1 -= dl
            abs(dl) <= 1e-15 * max(1.0, abs(l1)) && (conv = true; break)
        end
        conv || error("fast_window: Newton did not converge")
        ib = dgrad(S, l, l1)
        ip = ib + f.Gp * a
        Ib += h * ip
        D += h * (f.Rf * ib^2 + f.Gp * a^2)
        H += h
        l = l1
    end
    b = Ib / H
    return l, b, (dH = energy(S, l) - energy(S, l0), D, Eport = H * a * b)
end

"""
    cosim_window(slow, fast, q0, l0, H, hs; a0, tol = 1e-13, maxit = 50, fastfun = fast_window)
        -> (q1, l1, a, b, ledger, iters)

One window of the window-held coupling, solved on the interface variable a: g(a) = a_slow(b(a)) - a = 0 by a secant
iteration (a Newton step from a finite-difference derivative first), stopping when both |g| and the change of a are
below tol relative to the port's scale (8.3). The ledger: the two sides' dH, dissipation, the source's supply, and the
interconnection's energy defect H (a_slow - a) b (zero at the solution).
"""
function cosim_window(slow::SlowRC, fast, q0, l0, H, hs; a0, tol = 1e-13, maxit = 50, fastfun = fast_window)
    g(a) = (fr = fastfun(fast, l0, a, hs); sr = slow_step(slow, q0, H, fr[2]); (sr[2] - a, fr, sr))
    a = a0
    ga, fr, sr = g(a)
    scale = max(abs(a), abs(sr[2]), 1.0)
    da = 1e-7 * scale
    a1 = a + da
    g1, fr1, sr1 = g(a1)
    it = 1
    while abs(g1) > tol * scale || abs(a1 - a) > tol * scale
        it += 1
        it > maxit && error("cosim_window: the interface iteration did not converge (|g| = $(abs(g1)))")
        sl = (g1 - ga) / (a1 - a)
        sl == 0 && error("cosim_window: a zero secant slope before convergence (|g| = $(abs(g1)))")
        a, ga = a1, g1
        a1 = a1 - g1 / sl
        g1, fr1, sr1 = g(a1)
    end
    l1, b, fl = fr1
    q1, as, sl_ = sr1
    defect = H * (as - a1) * b
    led = (dH = sl_.dH + fl.dH, Esup = sl_.Esup, D = sl_.D + fl.D, defect,
           closure = sl_.dH + fl.dH - (sl_.Esup - sl_.D - fl.D))
    return q1, l1, a1, b, led, it
end

"""
    monolithic_step(slow, fast, q0, l0, H) -> (q1, l1, a, b)

The monolithic discrete-gradient step of the coupled pair (one step H): Newton on (q1, l1) jointly; the reference for P3.
"""
function monolithic_step(slow::SlowRC, fast::FastRL, q0, l0, H)
    slow.S === nothing && throw(ArgumentError("monolithic_step: the algebraic node is not in this reference"))
    S, F = slow.S, fast.S
    q1, l1 = q0, l0
    conv = false
    for _ in 1:60
        v = dgrad(S, q0, q1); ib = dgrad(F, l0, l1); ip = ib + fast.Gp * v
        r1 = (q1 - q0) / H - (slow.E - v) / slow.Rs + ip
        r2 = (l1 - l0) / H - v + fast.Rf * ib
        dv = 1 / (2S.k) + S.c4 * (3q1^2 + 2q1 * q0 + q0^2) / 4
        di = 1 / (2F.k) + F.c4 * (3l1^2 + 2l1 * l0 + l0^2) / 4
        J = [1 / H + dv / slow.Rs + fast.Gp * dv  di; -dv  1 / H + fast.Rf * di]
        d = J \ [r1, r2]
        q1 -= d[1]; l1 -= d[2]
        maximum(abs, d) <= 1e-15 * max(1.0, abs(q1), abs(l1)) && (conv = true; break)
    end
    conv || error("monolithic_step: Newton did not converge")
    a = dgrad(S, q0, q1)
    return q1, l1, a, dgrad(F, l0, l1) + fast.Gp * a
end

"""
    explicit_vi_window(slow, fast, q0, l0, H, hs, a_prev, b_prev) -> (q1, l1, a, b, defect)

The baseline: explicit voltage-current exchange with zero-order hold (Jacobi): the fast side runs with the previous
window's slow output, the slow side with the previous window's fast average. `defect` = the energy the slow side books
as leaving the node minus the energy the fast side books as arriving, H (a b_prev - a_prev b) per window (Sadjina's
residual with this sign: positive = energy lost at the interface). The simultaneous exchange's per-window amplification
is not simply the feed-through loop gain -R_s G_p (review 103, finding 19).
"""
function explicit_vi_window(slow::SlowRC, fast::FastRL, q0, l0, H, hs, a_prev, b_prev)
    l1, b, fl = fast_window(fast, l0, a_prev, hs)
    q1, a, sl = slow_step(slow, q0, H, b_prev)
    defect = H * a * b_prev - H * a_prev * b          # slow books out a b_prev; fast books in a_prev b
    return q1, l1, a, b, defect
end

# ---------------------------------------------------------------------------------------------------------------------
# the fast side in C (the same arithmetic): a shared library called through the window interface (any language)
# ---------------------------------------------------------------------------------------------------------------------

const _C_FAST_SRC = joinpath(@__DIR__, "c", "fast_rl_window.c")

"""Compile the C fast side into dir (gcc) and return the library path."""
function build_c_fast(dir::AbstractString = mktempdir())
    lib = joinpath(dir, "libfastrl.so")
    run(`gcc -O2 -fPIC -shared -o $lib $(_C_FAST_SRC)`)
    return lib
end

"""A fast side implemented in C (`fast_rl_window.c`); `fast_window_c` has `fast_window`'s signature."""
struct FastRLc
    f::FastRL
    lib::String
end

function fast_window_c(fc::FastRLc, l0, a, hs)
    out = zeros(5)          # l1, b, dH, D, Eport
    hv = collect(Float64, hs)
    fn = Libc.Libdl.dlsym(Libc.Libdl.dlopen(fc.lib), :fast_rl_window)
    rc = ccall(fn, Cint, (Float64, Float64, Float64, Float64, Float64, Float64, Ptr{Float64}, Cint, Ptr{Float64}),
               fc.f.Rf, fc.f.S.k, fc.f.S.c4, fc.f.Gp, l0, a, hv, length(hv), out)
    rc == 0 || error("fast_window_c: Newton did not converge")
    return out[1], out[2], (dH = out[3], D = out[4], Eport = out[5])
end
