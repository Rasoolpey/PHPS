# Travelling-wave transmission line (upgrade 1 of 3, REALTIME_TODO section 3; reviews 71-72b).
#
# A constant-parameter line as its physical circuit: n lumped-loss Bergeron sections in cascade (EMTP's lossy line,
# Dommel 1969, per section: R_s/4 - lossless half - R_s/2 - lossless half - R_s/4, R_s = R / n); adjacent sections' end
# resistors merge, so the line is m = 2n lossless half-sections of delay tau / (2n) joined by R / (2n), with R / (4n) at
# each end (n = 4 by default: review 72b, the model error against the exact distributed line falls from 0.50 to 5.7e-4
# over |f - 60| <= 2 kHz). The network is a space vector in the frame turning at omega_b: a wave crossing a half-section
# arrives tau / (2n) later, turned by rho = e^{-j omega_b tau / (2n)}. Constants from the case's nominal pi (length 1):
# L = x / omega_b, C = b / omega_b, so Zc = sqrt(x / b), tau = sqrt(x b) / omega_b; R = r.
#
# Scattering variables of a lossless half-section at one of its ends, u the voltage at that end, i the current into it:
#     a = (u + Zc i) / (2 sqrt Zc)  (leaving the end into the half-section),   b = (u - Zc i) / (2 sqrt Zc)  (arriving),
#     |a|^2 - |b|^2 = Re(u conj i).
# Channels: half-section s has 2s - 1 (leaving its left end, travelling right) and 2s (leaving its right end, travelling
# left). Each stores its leaving waves at the step midpoints t_{n+1/2} (one fixed step h per run, events on the grid);
# the wave arriving at the far end at t_{n+1/2} is
#     b = rho w,   w = (1 - alpha) a_{n-d} + alpha a_{n-d-1},   d = floor(tau / (2 n h)) >= 2,  alpha = tau / (2 n h) - d.
# Storage of a channel S = h [sum_{j=1}^{d} |a_{n-j}|^2 + alpha |a_{n-d-1}|^2]; per step, exactly (71 merge, item 3)
#     S_{n+1} - S_n = h |a_n|^2 - h |w_n|^2 - h alpha (1 - alpha) |a_{n-d} - a_{n-d-1}|^2,
# so for the line: h Re(v1 conj i1 + v2 conj i2) = dS + h (resistor losses) + D_interp.
# At an end the line is a Norton source at the bus, i_k = (v_k - 2 sqrt(Zc) b_k) / Z, Z = Zc + R / (4n); a junction
# between two half-sections (resistor Rj = R / (2n), current ir from left to right) is solved from its two arriving
# waves: ir = 2 sqrt(Zc) (bL - bR) / (2 Zc + Rj).

"""Constants of a travelling-wave line built from a nominal pi (r, x, b in pu) at a fixed step h (s), n sections."""
struct TLine
    Zc::Float64
    R::Float64
    tau::Float64
    h::Float64
    d::Int              # whole steps in a half-section's delay tau / (2n)
    alpha::Float64      # the fraction
    rho::ComplexF64     # e^{-j omega_b tau / (2n)}
    n::Int              # sections (2n half-sections)
end

function TLine(r, x, b; omega_b, h, nsec::Int = 4, rot::Bool = true)
    Zc = sqrt(x / b); tau = sqrt(x * b) / omega_b
    q = tau / (2nsec * h); d = floor(Int, q)
    d >= 2 || throw(ArgumentError("TLine: h = $h s too long for tau = $tau s with $nsec sections (need h <= tau / (4 n))"))
    # (rot = false: a quantity without frame rotation, the zero (ground) mode of a transposed line: rho = 1; upgrade 4)
    return TLine(Zc, r, tau, h, d, q - d, rot ? cis(-omega_b * tau / (2nsec)) : 1.0 + 0im, nsec)
end

"""History of a line: 4n channels' leaving waves at the midpoints, newest at `k`, ring length d + 2."""
mutable struct TLineState
    buf::Matrix{ComplexF64}   # (d + 2) x 4n
    k::Int                    # index of the newest stored sample
end

# the wave arriving at the far end of channel c at this step: rho w, w from the samples d and d + 1 back
function _tl_arrive(L::TLine, s::TLineState, c::Int)
    nb = size(s.buf, 1)
    return L.rho * ((1 - L.alpha) * s.buf[mod1(s.k - L.d + 1, nb), c] + L.alpha * s.buf[mod1(s.k - L.d, nb), c])
end

"""
    tline_norton(L, s) -> (G, J1, J2)

The line at this step as seen from its buses: i_k = G v_k - J_k (current into the line), G = 1 / (Zc + R / (4n)).
"""
function tline_norton(L::TLine, s::TLineState)
    Z = L.Zc + L.R / (4L.n); g = 2 * sqrt(L.Zc) / Z
    return 1 / Z, g * _tl_arrive(L, s, 2), g * _tl_arrive(L, s, 4L.n - 1)
end

# the leaving waves, the end and junction currents from the arriving waves `arr` (by channel) and the bus voltages
function _tl_solve(L::TLine, arr, v1, v2)
    sz = sqrt(L.Zc); m = 2L.n; Re_ = L.R / (4L.n); Rj = L.R / (2L.n); Z = L.Zc + Re_
    a = Vector{ComplexF64}(undef, 2m); ir = Vector{ComplexF64}(undef, m - 1)
    i1 = (v1 - 2sz * arr[2]) / Z; i2 = (v2 - 2sz * arr[2m - 1]) / Z
    a[1] = (v1 - Re_ * i1 + L.Zc * i1) / (2sz)
    a[2m] = (v2 - Re_ * i2 + L.Zc * i2) / (2sz)
    for j in 1:m - 1                                    # junction between half-sections j and j + 1
        bL = arr[2j - 1]; bR = arr[2j + 2]
        r = 2sz * (bL - bR) / (2L.Zc + Rj); ir[j] = r
        a[2j] = (2sz * bL - 2L.Zc * r) / (2sz)          # into half j at its right end: u = -Zc r + 2 sqrt(Zc) bL, i = -r
        a[2j + 1] = (2sz * bR + 2L.Zc * r) / (2sz)      # into half j + 1 at its left end: u = Zc r + 2 sqrt(Zc) bR, i = r
    end
    return a, i1, i2, ir
end

"""
    tline_step!(L, s, v1, v2; full = true) -> (; i1, i2, ir, loss, dinterp, dS, flow, full)

Commit one step with the midpoint bus voltages v1, v2: the end currents, the junction currents, the resistor losses and
the interpolation dissipation of the step (energies: power x h), the storage change and the step's gross wave energy
(a scale); pushes the leaving waves. With `full` the storage change is the difference of the two full storage sums (an
independent check of the identity, O(d) per step); otherwise it is the identity's right side, h sum (|a|^2 - |w|^2) less
the interpolation dissipation (O(1)); a run checks the latter against the full storage at its ends.
"""
function tline_step!(L::TLine, s::TLineState, v1, v2; full::Bool = true)
    h = L.h; nb = size(s.buf, 1); nc = 4L.n
    arr = [_tl_arrive(L, s, c) for c in 1:nc]
    a, i1, i2, ir = _tl_solve(L, arr, v1, v2)
    S0 = full ? tline_storage(L, s) : 0.0
    di = 0.0
    for c in 1:nc
        di += h * L.alpha * (1 - L.alpha) * abs2(s.buf[mod1(s.k - L.d + 1, nb), c] - s.buf[mod1(s.k - L.d, nb), c])
    end
    s.k = mod1(s.k + 1, nb)
    for c in 1:nc
        s.buf[s.k, c] = a[c]
    end
    loss = h * (L.R / (4L.n) * (abs2(i1) + abs2(i2)) + L.R / (2L.n) * sum(abs2, ir; init = 0.0))
    flow = h * (sum(abs2, a) + sum(abs2, arr))
    dS = full ? tline_storage(L, s) - S0 : h * (sum(abs2, a) - sum(abs2, arr)) - di   # |b| = |w| (|rho| = 1)
    return (; i1, i2, ir, loss, dinterp = di, dS, flow, full)
end

"""Physical in-flight energy of the channels (pu*s); summed with compensation (its per-step change is a difference of
two sums of ~4 n d squares)."""
function tline_storage(L::TLine, s::TLineState)
    nb = size(s.buf, 1); S = 0.0; e = 0.0
    for c in 1:4L.n, j in 0:L.d
        t = j < L.d ? abs2(s.buf[mod1(s.k - j, nb), c]) : L.alpha * abs2(s.buf[mod1(s.k - L.d, nb), c])
        y = t - e; u = S + y; e = (u - S) - y; S = u
    end
    return L.h * S
end

# the line's constant-phasor solution for bus voltages v1, v2: every channel's (constant) leaving wave; the arriving
# wave of a channel is rho times its own leaving wave (a constant delayed)
function _tl_steady_waves(L::TLine, v1, v2)
    nc = 4L.n
    F(a) = _tl_solve(L, L.rho .* a, v1, v2)[1]          # affine in a
    c0 = F(zeros(ComplexF64, nc))
    A = hcat([F([i == j ? 1.0 + 0im : 0.0im for i in 1:nc]) - c0 for j in 1:nc]...)
    return ([i == j ? 1.0 + 0im : 0.0im for i in 1:nc, j in 1:nc] - A) \ c0
end

"""
    tline_steady(L, v1, v2) -> (i1, i2)

The line's sinusoidal steady state (constant phasors in the rotating frame) for bus voltages v1, v2: the end currents.
"""
function tline_steady(L::TLine, v1, v2)
    a = _tl_steady_waves(L, v1, v2)
    _, i1, i2, _ = _tl_solve(L, L.rho .* a, v1, v2)
    return i1, i2
end

"""A line's history filled with its steady state at bus voltages v1, v2 (all negative time)."""
function tline_state(L::TLine, v1 = 0.0im, v2 = 0.0im)
    a = _tl_steady_waves(L, v1, v2)
    return TLineState(repeat(transpose(a), L.d + 2, 1), L.d + 2)
end

# ------------------------------------------------------------------------------------------------
# The line as an explicit discrete-time system (PH_TODO.md step 2, B2). Built from the circuit relations, independently
# of `_tl_solve`'s closed forms: at an end of a half-section u = sqrt(Zc) (a + b), Zc i = sqrt(Zc) (a - b); each junction
# (KVL across Rj) and each end (the end resistor R / (4n) and the bus voltage) is solved numerically.

"""
    tline_history(L, s) -> X

The line's state as a vector: for each channel c = 1, ..., 4n its d + 1 newest leaving waves, newest first (the ring
buffer unrolled; entry (c - 1)(d + 1) + j + 1 is the wave left j steps ago).
"""
function tline_history(L::TLine, s::TLineState)
    nb = size(s.buf, 1); m = L.d + 1
    X = Vector{ComplexF64}(undef, 4L.n * m)
    for c in 1:4L.n, j in 0:L.d
        X[(c - 1) * m + j + 1] = s.buf[mod1(s.k - j, nb), c]
    end
    return X
end

"""
    tline_realisation(L) -> (; A, B, C, D, P, Lr, Li, Qr, Qi, Sj, N)

The line as a discrete-time linear system on its history X (`tline_history`, length N = 4n (d + 1)), with the midpoint
bus voltages v = (v1, v2) as input and the end currents i = (i1, i2) (into the line) as output:

    X+ = A X + B v,   i = C X + D v,   storage S(X) = h X' P X,   P = I_4n (x) diag(1, ..., 1, alpha) >= 0.

With z = (X, v), the loss ports are lr = Lr z (sqrt(h R/(4n)) i1, sqrt(h R/(4n)) i2 and sqrt(h R/(2n)) times each
junction current) and li = Li z (sqrt(h alpha (1 - alpha)) (x_d - x_{d+1}) per channel, the interpolation); the step's
resistor losses are |lr|^2 = z' Qr z and its interpolation dissipation |li|^2 = z' Qi z (energies, power x h), Qr = Lr' Lr,
Qi = Li' Li. The realisation satisfies, for every z,

    S(X+) - S(X) = h Re(v' i) - |lr|^2 - |li|^2,

i.e. the Hermitian matrix h ([A B]' P [A B] - diag(P, 0)) - (h / 2) (E' [C D] + [C D]' E) + Qr + Qi is zero
(E = [0 I] picks v): an impedance-passive realisation at the ports, built from scattering delay channels and resistive
junctions. `Sj` are the junctions' 2 x 2 scattering matrices (leaving = Sj x arriving), each a contraction.

Domain (checked): Zc > 0, R >= 0, h > 0, n >= 1, d >= 2, 0 <= alpha < 1, |rho| = 1. At alpha = 0 the oldest coordinate
of every channel is read by nothing and carries no storage (P semidefinite); it can be removed for a minimal realisation.
"""
function tline_realisation(L::TLine)
    (L.Zc > 0 && L.R >= 0 && L.h > 0 && L.n >= 1 && L.d >= 2 && 0 <= L.alpha < 1 && abs(abs(L.rho) - 1) <= 1e-14) ||
        throw(ArgumentError("tline_realisation: outside the domain Zc > 0, R >= 0, h > 0, n >= 1, d >= 2, 0 <= alpha < 1, |rho| = 1"))
    nc = 4L.n; m = L.d + 1; N = nc * m; nz = N + 2; h = L.h
    sz = sqrt(L.Zc); Re_ = L.R / (4L.n); Rj = L.R / (2L.n); hs = 2L.n
    ix(c, j) = (c - 1) * m + j + 1                       # wave left j steps ago on channel c
    # the arriving wave of channel c as a row over z: rho ((1 - alpha) x_{d-1} + alpha x_d) (j counted from 0)
    function arrive(c)
        r = zeros(ComplexF64, nz)
        r[ix(c, L.d - 1)] += L.rho * (1 - L.alpha); r[ix(c, L.d)] += L.rho * L.alpha
        return r
    end
    ev(k) = (r = zeros(ComplexF64, nz); r[N + k] = 1; r)
    a = Vector{Vector{ComplexF64}}(undef, nc)             # leaving waves (rows over z)
    # ends: unknowns (a, i), a - sqrt(Zc) i = b, sqrt(Zc) a + R/(4n) i = v - sqrt(Zc) b
    Me = [1.0 -sz; sz Re_]
    i_end = Vector{Vector{ComplexF64}}(undef, 2)
    for (k, cout, cin) in ((1, 1, 2), (2, nc, nc - 1))   # end 1: channel 1 leaves, 2 arrives; end 2: 4n leaves, 4n - 1 arrives
        b = arrive(cin)
        sol = Me \ permutedims(hcat(b, ev(k) .- sz .* b))
        a[cout] = sol[1, :]; i_end[k] = sol[2, :]
    end
    # junctions between half-sections q and q + 1: unknowns (aL, aR, r); half q's right end carries -r, half q + 1's
    # left end +r: aL - bL = -sqrt(Zc) r, aR - bR = sqrt(Zc) r, sqrt(Zc) (aL + bL) - sqrt(Zc) (aR + bR) = Rj r
    Mj = [1.0 0.0 sz; 0.0 1.0 -sz; sz -sz -Rj]
    ir = Vector{Vector{ComplexF64}}(undef, hs - 1)
    Sj = Vector{Matrix{Float64}}(undef, hs - 1)
    for q in 1:hs - 1
        bL = arrive(2q - 1); bR = arrive(2q + 2)
        sol = Mj \ permutedims(hcat(bL, bR, sz .* (bR .- bL)))
        a[2q] = sol[1, :]; a[2q + 1] = sol[2, :]; ir[q] = sol[3, :]
        # the junction alone: leaving = Sj arriving, from the same relations with unit arriving waves
        Sj[q] = (Mj \ [1.0 0.0; 0.0 1.0; -sz sz])[1:2, :]
    end
    # the shift: X+ holds each channel's new wave first, then its d older ones
    T = zeros(ComplexF64, N, nz)
    for c in 1:nc
        T[ix(c, 0), :] = a[c]
        for j in 1:L.d
            T[ix(c, j), ix(c, j - 1)] = 1
        end
    end
    K = permutedims(hcat(i_end[1], i_end[2]))
    w = vcat(ones(L.d), L.alpha)
    Pd = repeat(w, nc)
    P = Diagonal(Pd)
    # loss ports as rows over z (quantity = transpose(row) z): the resistors' and the interpolation's
    Lr = permutedims(hcat(sqrt(h * Re_) .* i_end[1], sqrt(h * Re_) .* i_end[2], (sqrt(h * Rj) .* r for r in ir)...))
    Li = zeros(ComplexF64, nc, nz)
    for c in 1:nc
        Li[c, ix(c, L.d - 1)] = sqrt(h * L.alpha * (1 - L.alpha)); Li[c, ix(c, L.d)] = -sqrt(h * L.alpha * (1 - L.alpha))
    end
    Qr = Lr' * Lr; Qi = Li' * Li                         # |L z|^2 = z' L' L z
    return (; A = T[:, 1:N], B = T[:, N+1:end], C = K[:, 1:N], D = K[:, N+1:end], P, Lr, Li,
            Qr = Hermitian(Qr), Qi = Hermitian(Qi), Sj, N)
end
