# Listing 1 of the paper (exactly as printed). Run from the PHPS folder: julia --project=. scripts/paper/listing1_fault.jl
using PHPS
fm = phs_full_model("cases/IEEE39Bus_FullModel";
                    scenario = "fault_bus16_3ph.json")
r = phs_simulate(fm.cs, fm.x0, fm.u; tstop = 0.3,
                 faults = fm.faults)
for k in unique(r.kind)  # relative to the throughput
    c = findall(==(k), r.kind); T = r.throughput[c]
    println(k, " ", maximum(abs.(r.scheme[c]) ./ T),
               " ", maximum(abs.(r.closure[c]) ./ T))
end
# rotor 1.2e-12 4.1e-15; exciter 2.6e-14 8.9e-17
