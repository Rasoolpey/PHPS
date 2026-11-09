# The IEEE 39-bus full model as a case (cases/IEEE39Bus_FullModel; docs/active/FULL_MODEL_DIGITAL_TWIN_PLAN.md): the
# all-physical controlled model built from the case folder alone (the system file, PowerFactory's winding data written
# into the folder, and full_model.json with every physical option).

"""
    phs_full_model(dir; scenario = "fault_bus16_3ph.json") -> (; cs, x0, u, eq, sc, case, faults, spec)

The full model of the case folder `dir`: `phs_controlled` with the physical exciters (DC shunt where the IEEET1 has
`KE < 0`, brushless where `KE > 0`), the physical steam units and the physical hydro unit as `full_model.json`
declares them, the loads as declared, PowerFactory's machine winding data from `winding_circuits.json`. `x0` is the
operating point, `faults` the scenario's bus faults as `PHFault`s.
"""
function phs_full_model(dir::AbstractString; scenario::AbstractString = "fault_bus16_3ph.json")
    spec = JSON3.read(read(joinpath(dir, "full_model.json"), String))
    sc = load_scenario(joinpath(dir, scenario))
    case = load_case(sc.system_path)
    wc = JSON3.read(read(joinpath(dir, String(spec.winding_circuits)), String))
    fl(v) = v === nothing ? nothing : Float64(v)
    circuits = Dict{Int,NamedTuple}()
    for (b, v) in pairs(wc.machines)
        circuits[parse(Int, String(b))] = (; Ll = fl(v.Ll), Lad = fl(v.Lad), Laq = fl(v.Laq), Lfd = fl(v.Lfd),
                                           L1d = fl(v.L1d), L1q = fl(v.L1q), L2q = fl(v.L2q), Ra = fl(v.Ra),
                                           Rfd = fl(v.Rfd), R1d = fl(v.R1d), R1q = fl(v.R1q), R2q = fl(v.R2q), H = fl(v.H))
    end
    eq = solve_equilibrium(case, sc)
    kw = (; loads = Symbol(spec.network.loads), circuits)
    cs0, _, _, _ = phs_controlled(eq; kw...)
    shunt = [e.name for e in cs0.exc if e !== nothing && e.p.KE < 0]
    dc, bl, st, hy = spec.exciters.dc_shunt, spec.exciters.brushless, spec.steam, spec.hydro
    steam = (; paths = Dict(String(k) => Symbol(v) for (k, v) in pairs(st.paths)), r_v = Float64(st.r_v),
             rh_dT = Float64(st.rh_dT), tau_n = Float64(st.tau_n), lowflow = Float64(st.lowflow),
             cold_reheat = Float64(st.cold_reheat), volumes = Symbol(st.volumes), valve = Symbol(st.valve),
             tables = steam_tables())
    hydro = (; Tw = hy.Tw isa AbstractString ? Symbol(hy.Tw) : Float64(hy.Tw), Te = Float64(hy.Te), N = Int(hy.N),
             fp = Float64(hy.fp))
    cs, x0, u, _ = phs_controlled(eq; kw...,
                                  dc_exciter = (; n = Dict(k => Float64(dc.n) for k in shunt), field_ratio = Float64(dc.field_ratio)),
                                  brushless = (; KC = Float64(bl.KC), KD = Float64(bl.KD), sat = (Float64(bl.sat[1]), Float64(bl.sat[2])),
                                               field_share = Float64(bl.field_share)),
                                  steam, hydro)
    faults = [PHFault(e.bus, e.r, e.t_start, e.t_end) for e in sc.events if e isa BusFault]
    return (; cs, x0 = collect(x0), u, eq, sc, case, faults, spec)
end
