# GFL_PHTRUE: the grid-following converter, minimal RMS form (docs/active/PH_CONVERTER_MODEL.md section 38; the user's
# decision of 2026-10-16: a minimal component, the full RMS path of PowerFactory's gfl_pf DLL later if an RMS comparison is
# wanted).
#
# It holds the converter's injected current as two states with zero rate: a constant-current source at the operating point.
# The equilibrium closes its two rows with the first pass's current (the load flow's, `converter_init`), as it closes the
# grid-forming converters' set-point rows. It exists so that a case with GFL converters has an RMS equilibrium, from which
# the EMT simulator starts (`phs_equilibrium`: the EMF `E0 = V + j Zseries I` of the static generator's voltage source behind
# its reactance) and builds the EMT controller `PHGfl` (`src/phsim/converter.jl`) from the parameters kept here. It is not
# Python PHPS's `GflPHTrue` (which has the PLL, the outer loops and a frame feed-forward) and not PowerFactory's RMS model; it has
# no energy (H = 0).
#
# Parameters: the DLL gfl_pf v3.1's (section 36; on the system base for one unit, as the case files hold every converter
# value), the static generator's series reactor `Zseries` (= uk/100 on the system base) and its built-in current controller
# `Kd, Td, Kq, Tq` (TechRef StaticGenerator 7.1; `Kd, Kq` on the system base for one unit), and the dialog set-points
# `p_ref, q_ref`. Python PHPS's `T_gs` (its stand-in for the static generator's lag) and `pf_frame` are refused.
#
# States  [I_re, I_im]   (the injected current, network frame)
# Inputs  [Vd, Vq]
# Outputs [Id, Iq]

struct GflParams
    kp_pll::Float64
    ki_pll::Float64
    omega_n::Float64
    phi_max::Float64
    i_max::Float64
    T_cc::Float64
    kp_P::Float64
    ki_P::Float64
    kp_Q::Float64
    ki_Q::Float64
    k_droop_p::Float64
    f_db::Float64
    k_droop_q::Float64
    v_db::Float64
    r_p_max::Float64
    r_q_max::Float64
    T_meas::Float64
    Kqf::Float64
    p_ref::Float64
    q_ref::Float64
    Zseries::Float64
    Kd::Float64
    Td::Float64
    Kq::Float64
    Tq::Float64
    ra::Float64
end

"""
Grid-following converter, minimal RMS form (RMS stack).
"""
struct GFL_PHTRUE <: AbstractComponent
    name::String
    bus::Int
    p::GflParams
    params::ParamDict
end

const _GFL_STATES = ["I_re", "I_im"]
const _GFL_INPUTS = ["Vd", "Vq"]
const _GFL_OUTPUTS = ["Id", "Iq"]

model_type(::GFL_PHTRUE) = "GFL_PHTRUE"
component_role(::GFL_PHTRUE) = :generator
state_names(::GFL_PHTRUE) = _GFL_STATES
input_names(::GFL_PHTRUE) = _GFL_INPUTS
output_names(::GFL_PHTRUE) = _GFL_OUTPUTS
bus(c::GFL_PHTRUE) = c.bus

# gfl_pf.cpp v3.1's defaults (ParamIdx), and the static generator's
const _GFL_DEFAULTS = ("kp_pll" => 14.0, "ki_pll" => 100.0, "omega_n" => 376.991, "phi_max" => 5.0, "T_cc" => 0.002,
                       "kp_P" => 0.5, "ki_P" => 10.0, "kp_Q" => 0.5, "ki_Q" => 10.0, "k_droop_p" => 0.0, "f_db" => 0.001,
                       "k_droop_q" => 0.0, "v_db" => 0.02, "r_p_max" => 10.0, "r_q_max" => 10.0, "T_meas" => 0.05,
                       "Kqf" => 0.0, "pf_frame" => 0.0, "T_gs" => 0.0)

function type_defaults!(::Val{:GFL_PHTRUE}, p::ParamDict)
    for (k, v) in _GFL_DEFAULTS
        get!(p, k, v)
    end
    return p
end

function GFL_PHTRUE(name::String, d::ParamDict)
    f(k) = _p(d, k, name)
    (f("pf_frame") <= 0.5 && f("T_gs") == 0) ||
        throw(ArgumentError("$name: GFL_PHTRUE with Python PHPS's pf_frame or T_gs is not supported (record section 38)"))
    p = GflParams(f("kp_pll"), f("ki_pll"), f("omega_n"), f("phi_max"), f("i_max"), f("T_cc"), f("kp_P"), f("ki_P"),
                  f("kp_Q"), f("ki_Q"), f("k_droop_p"), f("f_db"), f("k_droop_q"), f("v_db"), f("r_p_max"), f("r_q_max"),
                  f("T_meas"), f("Kqf"), f("p_ref"), f("q_ref"), f("Zseries"), f("Kd"), f("Td"), f("Kq"), f("Tq"), f("ra"))
    return GFL_PHTRUE(name, Int(f("bus")), p, d)
end
COMPONENT_CONSTRUCTORS["GFL_PHTRUE"] = GFL_PHTRUE

function _outputs!(y, c::GFL_PHTRUE, x, u, p::GflParams, rec::ModeRecorder)
    y[1] = x[1]
    y[2] = x[2]
    return y
end

function _step!(dx, y, c::GFL_PHTRUE, x, u, p::GflParams, rec::ModeRecorder)
    y === nothing || _outputs!(y, c, x, u, p, rec)
    dx[1] = zero(eltype(dx))
    dx[2] = zero(eltype(dx))
    return dx
end

hamiltonian(c::GFL_PHTRUE, x, p::GflParams) = zero(eltype(x))
grad_hamiltonian!(g, c::GFL_PHTRUE, x, p::GflParams) = fill!(g, 0.0)

"""The injected current (network frame): the states, independent of the bus voltage; no Norton admittance."""
injection(c::GFL_PHTRUE, x, V, p::GflParams = c.p) = (x[1], x[2])
