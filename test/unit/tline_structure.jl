# The travelling-wave line as an explicit discrete-time passive system (PH_TODO.md step 2, B2; review 88 rows 8-10).
# `tline_realisation` builds (A, B, C, D, P, Qr, Qi) from the circuit relations, independently of `tline_step!`; this
# checks (1) the exact dissipation equality as a matrix identity, (2) Qr, Qi >= 0 and the junctions' contraction,
# (3) the realisation against `tline_step!` at arbitrary histories over several ring wrap-arounds, (4) the edge cases:
# integer delay (alpha = 0, the unused history coordinate), zero resistance, the zero mode without rotation, alpha near
# 1, (5) the steady state as a fixed point and the Norton form.

using Test, LinearAlgebra, Random
using PHPS
using PHPS: TLine, TLineState, tline_state, tline_step!, tline_storage, tline_norton, tline_steady, tline_history,
               tline_realisation

const _WB = 2pi * 60

# a line of the IEEE-39 kind (pu on 100 MVA; length 1) and variants
_tl_base() = (r = 0.0035, x = 0.0411, b = 0.6987)
_tl_tau(p) = sqrt(p.x * p.b) / _WB

function _tl_cases()
    p = _tl_base(); tau = _tl_tau(p); n = 4
    L1 = TLine(p.r, p.x, p.b; omega_b = _WB, h = 1e-6)                          # realistic: d about 56
    L2 = TLine(p.r, p.x, p.b; omega_b = _WB, h = tau / (2n * 2.37))              # d = 2, alpha = 0.37
    L3 = TLine(L2.Zc, L2.R, L2.tau, L2.h, 3, 0.0, L2.rho, n)                     # integer delay: alpha = 0
    L4 = TLine(0.0, p.x, p.b; omega_b = _WB, h = tau / (2n * 3.6))               # no resistance
    L5 = TLine(3p.r, p.x, 0.6p.b; omega_b = _WB, h = tau / (2n * 4.2), rot = false)   # zero mode: rho = 1
    L6 = TLine(L2.Zc, L2.R, L2.tau, L2.h, 2, 0.999, L2.rho, n)                   # alpha near 1
    # an integer delay from the physical constructor (review 90, 9): tau / (2 n h) = 3 up to rounding, so alpha is
    # within rounding of 0 (d = 3) or of 1 (d = 2)
    L7 = TLine(p.r, p.x, p.b; omega_b = _WB, h = tau / (2n * 3))
    return ["IEEE-39 line, h = 1 us" => L1, "d = 2, alpha = 0.37" => L2, "alpha = 0 (fields set)" => L3, "R = 0" => L4,
            "zero mode (rho = 1)" => L5, "alpha = 0.999 (fields set)" => L6, "integer delay (constructor)" => L7]
end

_tl_S(L, P, X) = L.h * real(dot(X, P * X))

@testset "travelling-wave line: explicit discrete-time realisation (B2)" begin
    rng = MersenneTwister(2)
    for (name, L) in _tl_cases()
        @testset "$name" begin
            M = tline_realisation(L)
            N = M.N; nz = N + 2; h = L.h
            T = hcat(M.A, M.B); K = hcat(M.C, M.D)
            E = hcat(zeros(2, N), Matrix(1.0I, 2, 2))
            Pz = cat(Matrix(M.P), zeros(2, 2); dims = (1, 2))
            # (1) the dissipation equality as a Hermitian matrix identity
            G = h .* (T' * M.P * T .- Pz) .- (h / 2) .* (E' * K .+ K' * E) .+ M.Qr .+ M.Qi
            scale = max(opnorm(h .* (T' * M.P * T)), opnorm(Matrix(M.Qr)), 1e-300)
            @test opnorm(G) <= 1e-12 * scale
            # (2) both dissipations positive semidefinite; P >= 0; the junctions contract
            @test eigmin(M.Qr) >= -1e-13 * max(opnorm(Matrix(M.Qr)), scale)
            @test eigmin(M.Qi) >= -1e-13 * max(opnorm(Matrix(M.Qi)), scale)
            @test minimum(diag(M.P)) >= 0
            for S in M.Sj
                @test opnorm(S) <= 1 + 1e-14
                @test eigmin(Symmetric(I - S' * S)) >= -1e-14
                L.R == 0 && @test S' * S ≈ I atol = 1e-14
            end
            L.R == 0 && @test opnorm(Matrix(M.Qr)) <= 1e-14 * scale
            L.alpha == 0 && @test opnorm(Matrix(M.Qi)) == 0
            # (4) alpha = 0: the oldest coordinate of each channel is read by nothing (semidefinite storage)
            if L.alpha == 0
                old = [(c - 1) * (L.d + 1) + L.d + 1 for c in 1:4L.n]
                @test all(iszero, M.A[:, old]) && all(iszero, M.C[:, old])
                @test all(iszero, diag(M.P)[old])
            end
            # (3) against tline_step! at arbitrary histories, through several wrap-arounds of the ring buffer
            nb = L.d + 2
            s = TLineState(randn(rng, ComplexF64, nb, 4L.n), rand(rng, 1:nb))
            worst = (i = 0.0, X = 0.0, loss = 0.0, di = 0.0, dS = 0.0, bal = 0.0)
            for step in 1:3nb
                X = tline_history(L, s)
                v = randn(rng, ComplexF64, 2)
                z = vcat(X, v)
                r = tline_step!(L, s, v[1], v[2]; full = true)
                Xn = tline_history(L, s)
                sc = r.flow                  # the step's gross wave energy h (sum |a|^2 + sum |b|^2) (review 90, 10)
                worst = (i = max(worst.i, maximum(abs, [r.i1, r.i2] .- K * z) / maximum(abs, K * z)),
                         X = max(worst.X, maximum(abs, Xn .- T * z) / maximum(abs, Xn)),
                         loss = max(worst.loss, abs(r.loss - sum(abs2, M.Lr * z)) / sc),   # the loss ports
                         di = max(worst.di, abs(r.dinterp - sum(abs2, M.Li * z)) / sc),
                         dS = max(worst.dS, abs(r.dS - (_tl_S(L, M.P, T * z) - _tl_S(L, M.P, X))) / sc),
                         bal = max(worst.bal, abs(h * real(dot(v, K * z)) - r.dS - r.loss - r.dinterp) / sc))
            end
            @test worst.i <= 1e-13
            @test worst.X <= 1e-13
            @test worst.loss <= 1e-13
            @test worst.di <= 1e-13
            @test worst.dS <= 1e-13
            @test worst.bal <= 1e-13
            # (5) the steady state is a fixed point; the Norton form is D = G I and C X = -J
            v1, v2 = 1.02 * cis(0.1), 0.98 * cis(-0.2)
            s0 = tline_state(L, v1, v2)
            X0 = tline_history(L, s0); z0 = vcat(X0, [v1, v2])
            @test maximum(abs, T * z0 .- X0) <= 1e-12 * maximum(abs, X0)
            i1, i2 = tline_steady(L, v1, v2)
            @test maximum(abs, K * z0 .- [i1, i2]) <= 1e-12 * max(abs(i1), abs(i2))
            Gn, J1, J2 = tline_norton(L, s0)
            @test maximum(abs, M.D - Gn * I) <= 1e-14 * Gn
            @test maximum(abs, M.C * X0 .+ [J1, J2]) <= 1e-12 * max(abs(J1), abs(J2))
            # the loss ports reproduce the dissipation forms
            @test opnorm(Matrix(M.Qr) - M.Lr' * M.Lr) <= 1e-14 * max(opnorm(Matrix(M.Qr)), 1e-300)
            @test opnorm(Matrix(M.Qi) - M.Li' * M.Li) <= 1e-14 * max(opnorm(Matrix(M.Qi)), 1e-300)
        end
    end
    @testset "integer delay from the constructor" begin
        L7 = last(_tl_cases())[2]
        @test min(L7.alpha, 1 - L7.alpha) <= 1e-12
    end
    @testset "domain" begin
        L2 = _tl_cases()[2][2]
        for bad in (TLine(L2.Zc, L2.R, L2.tau, L2.h, 1, 0.3, L2.rho, L2.n),          # d < 2
                    TLine(L2.Zc, L2.R, L2.tau, L2.h, 2, 1.0, L2.rho, L2.n),          # alpha = 1
                    TLine(L2.Zc, -1e-3, L2.tau, L2.h, 2, 0.3, L2.rho, L2.n),         # R < 0
                    TLine(L2.Zc, L2.R, L2.tau, L2.h, 2, 0.3, 1.01 * L2.rho, L2.n))   # |rho| != 1
            @test_throws ArgumentError tline_realisation(bad)
        end
    end
    @testset "zero mode with real histories and inputs, the O(1) path (full = false), 10^4 steps" begin
        # the real zero-sequence lines of midpoint.jl (rot = false, real voltages): the realisation is real, a real
        # history stays real, and the per-step identity of full = false closes against the storage at the ends
        L = _tl_cases()[5][2]
        M = tline_realisation(L)
        @test all(iszero, imag.(M.A)) && all(iszero, imag.(M.B)) && all(iszero, imag.(M.C)) && all(iszero, imag.(M.D))
        nb = L.d + 2
        s = TLineState(complex.(randn(rng, nb, 4L.n)), rand(rng, 1:nb))
        S0 = tline_storage(L, s)
        sup = 0.0; dSsum = 0.0; loss = 0.0; flow = 0.0; maximag = 0.0
        for k in 1:10_000
            v1, v2 = randn(rng), randn(rng)
            r = tline_step!(L, s, complex(v1), complex(v2); full = false)
            sup += L.h * real(v1 * conj(r.i1) + v2 * conj(r.i2))
            dSsum += r.dS; loss += r.loss + r.dinterp; flow += r.flow
            maximag = max(maximag, maximum(abs, imag.(s.buf)), abs(imag(r.i1)), abs(imag(r.i2)))
        end
        @test maximag == 0                                            # a real history stays real
        @test abs(sup - dSsum - loss) <= 1e-13 * flow                 # the balance, step by step accumulated
        @test abs((tline_storage(L, s) - S0) - dSsum) <= 1e-13 * flow # the endpoint storage against the summed dS
    end
end
