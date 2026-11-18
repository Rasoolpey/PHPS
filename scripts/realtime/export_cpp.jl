# Writes the C++ of one real-time model (and its coefficients). The model name is the start of its export function in src/rt/:
#   julia --project=. scripts/realtime/export_cpp.jl genrou    # calls genrou_rt_export(): writes src/rt/machines/CPP/genrou/
using PHPS
length(ARGS) == 1 || error("give one model name, e.g.: julia --project=. scripts/realtime/export_cpp.jl genrou")
f = Symbol(ARGS[1], "_rt_export")
isdefined(PHPS.RT, f) || error("no function $f in src/rt/ (is the model's file included in src/rt/RT.jl?)")
getfield(PHPS.RT, f)()
