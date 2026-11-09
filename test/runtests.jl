using PHPS
using Test
using SparseArrays
using LinearAlgebra
using ForwardDiff
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))

case_path(rel) = joinpath(ROOT, "cases", rel)

@testset "PHPS" begin
    @testset "unit" begin
        include("unit/expr.jl")
        include("unit/io.jl")
        include("unit/network.jl")
        include("unit/powerflow.jl")
        include("unit/components.jl")
        include("unit/assembly.jl")
        include("unit/ph.jl")
        include("unit/emdq.jl")
        include("unit/emdq_network.jl")
        include("unit/emdq_pilot2.jl")
        include("unit/phsim.jl")
        include("unit/phsim_transformer.jl")
        include("unit/phsim_voc.jl")
        include("unit/phsim_gfl.jl")
        include("unit/phsim_droop_dc.jl")
        include("unit/phsim_fixed_units.jl")
        include("unit/hydro_closed_gate.jl")
        include("unit/port_balance.jl")
        include("unit/load_passivity.jl")
        include("unit/component_ports.jl")
        include("unit/steam_resource.jl")
        include("unit/steam_turbine_shaft.jl")
        include("unit/iapws95.jl")
        include("unit/iapws95_provider.jl")
        include("unit/steam_composition.jl")
        include("unit/steam_machine_shaft.jl")
        include("unit/steam_ad_widths.jl")
        include("unit/steam_grid_unit.jl")
        include("unit/steam_shifted_balance.jl")
        include("unit/roa.jl")
    end
end
