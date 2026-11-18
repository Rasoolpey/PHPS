# One half-bridge leg of SiC MOSFETs: the step (7 unknowns), its exact ledger, and the solver (COSIM_DESIGN.md 8.7).
#
# Unknowns y = (v_gs,u, v_ds,u, v_ch,u, v_gs,l, v_ds,l, v_ch,l, i_L) at the step's end (v_ch: the channel voltage, an
# algebraic unknown; i_L the commutation loop's current, flux L i_L). Held over the step: V_dc, the phase current i_k
# (out of the pole), the drivers' voltages V_u (referenced to the pole) and V_l (to DC-).

"""The leg's circuit data."""
struct SiCLeg
    dev::SiCDevice
    RG::Float64          # total gate resistance per device (external + internal)
    L::Float64           # commutation loop inductance (H)
    Ru::Float64          # package resistance, upper and lower position (Ohm)
    Rl::Float64
end

"""The leg's state at a step boundary."""
mutable struct SiCLegState
    vgs_u::Float64
    vds_u::Float64
    vgs_l::Float64
    vds_l::Float64
    iL::Float64
    vch_u::Float64       # last channel voltages (the Newton guess)
    vch_l::Float64
end
Base.copy(s::SiCLegState) = SiCLegState(s.vgs_u, s.vds_u, s.vgs_l, s.vds_l, s.iL, s.vch_u, s.vch_l)
function Base.copyto!(a::SiCLegState, b::SiCLegState)
    a.vgs_u, a.vds_u, a.vgs_l, a.vds_l, a.iL, a.vch_u, a.vch_l = b.vgs_u, b.vds_u, b.vgs_l, b.vds_l, b.iL, b.vch_u, b.vch_l
    return a
end

"""One step's ledger (J) and port averages: Delta H = E_dc - E_ac + E_gate - D (exactly, to the Newton residual)."""
struct SiCStepLedger
    dH::Float64
    E_dc::Float64        # h V_dc ibar_L (into the leg)
    E_ac::Float64        # h phibar_p i_k (out of the pole)
    E_gate::Float64      # h (V_u i_Gu + V_l i_Gl)
    D_ch::Float64        # channel
    D_rd::Float64        # drift resistance
    D_bd::Float64        # body diodes
    D_rg::Float64        # gate resistances
    D_pkg::Float64       # package
    phip::Float64        # discrete pole effort (V, from DC-)
    iLm::Float64         # discrete loop current (A)
    iGu::Float64
    iGl::Float64
    phiDu::Float64       # discrete drain efforts (V)
    phiDl::Float64
    idu::Float64         # drain-terminal currents (A): upper = ibar_L, lower = ibar_L - i_k
    idl::Float64
end
dissipation(l::SiCStepLedger) = l.D_ch + l.D_rd + l.D_bd + l.D_rg + l.D_pkg
defect(l::SiCStepLedger) = l.dH - (l.E_dc - l.E_ac + l.E_gate - dissipation(l))

# the residual (scaled: currents in A, the loop row in V)
function _leg_residual!(r, y, leg::SiCLeg, s::SiCLegState, h, Vdc, ik, Vu, Vl)
    d = leg.dev
    dqGu, dqDu, phGu, phDu = dev_discrete(d, s.vgs_u, s.vds_u, y[1], y[2])
    dqGl, dqDl, phGl, phDl = dev_discrete(d, s.vgs_l, s.vds_l, y[4], y[5])
    iLm = (s.iL + y[7]) / 2
    iGu = (Vu - phGu) / leg.RG
    iGl = (Vl - phGl) / leg.RG
    Iu = chan_current(d, phGu, y[3])
    Il = chan_current(d, phGl, y[6])
    bu = body_current(d, -phDu)
    bl = body_current(d, -phDl)
    ilow = iLm - ik
    php = phDl + leg.Rl * ilow
    r[1] = dqGu / h - iGu
    r[2] = dqDu / h - (iLm - Iu + bu)
    r[3] = Iu - (phDu - y[3]) / d.Rd
    r[4] = dqGl / h - iGl
    r[5] = dqDl / h - (ilow - Il + bl)
    r[6] = Il - (phDl - y[6]) / d.Rd
    r[7] = leg.L * (y[7] - s.iL) / h - (Vdc - leg.Ru * iLm - phDu - php)
    return r
end

"""A leg solver's workspace (allocation-free steps)."""
mutable struct SiCLegSolver{C}
    leg::SiCLeg
    y::Vector{Float64}
    r::Vector{Float64}
    J::Matrix{Float64}
    dy::Vector{Float64}
    piv::Vector{Int}
    cfg::C
    s::SiCLegState                         # the state the residual closes over (start of the step)
    h::Float64
    Vdc::Float64
    ik::Float64
    Vu::Float64
    Vl::Float64
    iters::Int                             # Newton iterations of the last step
    maxres::Float64                        # its final scaled residual
    nretry::Int                            # steps that succeeded only after the channel guesses were reset
end

function SiCLegSolver(leg::SiCLeg)
    y = zeros(7); r = zeros(7)
    s = SiCLegState(0, 0, 0, 0, 0, 0, 0)
    cfg = ForwardDiff.JacobianConfig(nothing, r, y, ForwardDiff.Chunk{7}())   # untagged: the closure is rebuilt per step
    return SiCLegSolver{typeof(cfg)}(leg, y, r, zeros(7, 7), zeros(7), zeros(Int, 7), cfg, s, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0, 0.0, 0)
end

# Gaussian elimination with partial pivoting in place (7 x 7): solves J dy = r
function _lu_solve!(dy, J, r, piv)
    n = length(r)
    @inbounds for i in 1:n
        dy[i] = r[i]
    end
    @inbounds for k in 1:n
        p = k; mx = abs(J[k, k])
        for i in k+1:n
            a = abs(J[i, k])
            a > mx && (mx = a; p = i)
        end
        mx == 0 && return false
        if p != k
            for j in 1:n
                J[k, j], J[p, j] = J[p, j], J[k, j]
            end
            dy[k], dy[p] = dy[p], dy[k]
        end
        for i in k+1:n
            f = J[i, k] / J[k, k]
            f == 0 && continue
            for j in k+1:n
                J[i, j] -= f * J[k, j]
            end
            dy[i] -= f * dy[k]
        end
    end
    @inbounds for k in n:-1:1
        acc = dy[k]
        for j in k+1:n
            acc -= J[k, j] * dy[j]
        end
        dy[k] = acc / J[k, k]
    end
    return true
end

const LEG_TOL = 1e-9        # residual tolerance (A; V on the loop row)

"""
    leg_step!(sol, s, h, Vdc, ik, Vu, Vl) -> (ok, ledger)

One step from state s (updated in place on success): Newton on the 7 unknowns from the state as the guess, the
Jacobian by forward differentiation. `ok` false if Newton did not converge in 40 iterations (s unchanged).
"""
function leg_step!(sol::SiCLegSolver, s::SiCLegState, h, Vdc, ik, Vu, Vl; retry::Bool = false)
    copyto!(sol.s, s)
    sol.h, sol.Vdc, sol.ik, sol.Vu, sol.Vl = h, Vdc, ik, Vu, Vl
    y = sol.y
    y[1], y[2], y[3], y[4], y[5], y[6], y[7] = s.vgs_u, s.vds_u, s.vch_u, s.vgs_l, s.vds_l, s.vch_l, s.iL
    f! = (rr, yy) -> _leg_residual!(rr, yy, sol.leg, sol.s, sol.h, sol.Vdc, sol.ik, sol.Vu, sol.Vl)
    ok = false
    it = 0
    res = Inf
    while it < 40
        ForwardDiff.jacobian!(sol.J, f!, sol.r, y, sol.cfg, Val{false}())   # also fills r at y
        res = maximum(abs, sol.r)
        if res <= LEG_TOL
            ok = true
            break
        end
        _lu_solve!(sol.dy, sol.J, sol.r, sol.piv) || break
        @inbounds for i in 1:7
            y[i] -= sol.dy[i]
        end
        it += 1
    end
    if !ok && !retry
        # once more from the channel guesses reset to the drain voltages (a stale guess far from the solution)
        s2 = copy(s); s2.vch_u, s2.vch_l = s.vds_u, s.vds_l
        ok2, led2 = leg_step!(sol, s2, h, Vdc, ik, Vu, Vl; retry = true)
        if ok2
            copyto!(s, s2)
            sol.nretry += 1
            return true, led2
        end
    end
    sol.iters, sol.maxres = it, res
    ok || return false, nothing
    led = _leg_ledger(sol.leg, sol.s, y, h, Vdc, ik, Vu, Vl)
    s.vgs_u, s.vds_u, s.vch_u, s.vgs_l, s.vds_l, s.vch_l, s.iL = y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    return true, led
end

function _leg_ledger(leg::SiCLeg, s::SiCLegState, y, h, Vdc, ik, Vu, Vl)
    d = leg.dev
    _, _, phGu, phDu = dev_discrete(d, s.vgs_u, s.vds_u, y[1], y[2])
    _, _, phGl, phDl = dev_discrete(d, s.vgs_l, s.vds_l, y[4], y[5])
    iLm = (s.iL + y[7]) / 2
    iGu = (Vu - phGu) / leg.RG
    iGl = (Vl - phGl) / leg.RG
    Iu = chan_current(d, phGu, y[3])
    Il = chan_current(d, phGl, y[6])
    bu = body_current(d, -phDu)
    bl = body_current(d, -phDl)
    ilow = iLm - ik
    php = phDl + leg.Rl * ilow
    dH = (dev_energy(d, y[1], y[2]) - dev_energy(d, s.vgs_u, s.vds_u)) +
         (dev_energy(d, y[4], y[5]) - dev_energy(d, s.vgs_l, s.vds_l)) + leg.L * (y[7]^2 - s.iL^2) / 2
    return SiCStepLedger(dH, h * Vdc * iLm, h * php * ik, h * (Vu * iGu + Vl * iGl),
                         h * (y[3] * Iu + y[6] * Il), h * d.Rd * (Iu^2 + Il^2), h * (-phDu * bu - phDl * bl),
                         h * leg.RG * (iGu^2 + iGl^2), h * (leg.Ru * iLm^2 + leg.Rl * ilow^2),
                         php, iLm, iGu, iGl, phDu, phDl, iLm, ilow)
end

"""
    leg_equilibrium(leg, Vdc, ik, upper_on) -> SiCLegState

A consistent conduction state: the conducting device's gate at 15 V, the other's at -4 V, reached by steps growing
geometrically from 0.1 ns to 1 us with the inputs held (the growth damps the stiff modes that coarse midpoint steps
would leave alternating), then held at 1 us until steady.
"""
function leg_equilibrium(leg::SiCLeg, Vdc, ik, upper_on::Bool; Von = 15.0, Voff = -4.0)
    s = upper_on ? SiCLegState(Von, 0.0, Voff, Vdc, ik, 0.0, 0.0) : SiCLegState(Voff, Vdc, Von, 0.0, 0.0, 0.0, 0.0)
    sol = SiCLegSolver(leg)
    Vu, Vl = upper_on ? (Von, Voff) : (Voff, Von)
    h = 1e-10
    for _ in 1:2000
        ok, _ = leg_step!(sol, s, h, Vdc, ik, Vu, Vl)
        ok || error("leg_equilibrium: no convergence at h = $h")
        h = min(1.05h, 1e-6)
    end
    return s
end
