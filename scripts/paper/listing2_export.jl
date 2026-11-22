# Listing 2 of the paper, first part (exactly as printed, the model built as in Listing 1).
# Run from the PHPS folder: julia --project=. scripts/paper/listing2_export.jl
using PHPS
fm = phs_full_model("cases/IEEE39Bus_FullModel"; scenario = "fault_bus16_3ph.json")
am = phs_analysis_model(fm.cs, fm.x0, fm.u)
g = ["GENROU_4", "IEEET1_4", "IEEEG1_4"]
unit = phs_subsystem(am, g)      # unit 4 as a system
println(unit.ports)              # ["q_19_Re", "q_19_Im"]
println(unit.checks.shared_loss_entries)          # 0.0
phs_export("unit4", unit)   # CSV files, equations.md
phs_export("model.jld2", am)
