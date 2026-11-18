"""
    PHPS.RT

Real-time component models for the FPGA. Each model is written in Julia with `@hls` (src/rt/translator/julia_to_cpp.jl):
the function runs in Julia as written and is translated line by line to C++ for Vitis HLS. Additive: nothing outside
this module uses it.

Layout (the component folders follow `src/components/`; how to write a model: `TEMPLATE.jl`):

    translator/julia_to_cpp.jl   the @hls macro and the C++ writer
    machines/ exciters/ governors/ loads/ converters/   one file per model; its C++ in <class>/CPP/<model>/
"""
module RT

using LinearAlgebra: I

include("translator/julia_to_cpp.jl")
include("translator/hls_graph.jl")
include("translator/engine_sched.jl")

# component models (same classification as src/components/); one include per model
include("machines/genrou.jl")
include("machines/phmachine.jl")
include("machines/phmachine4.jl")
include("network/network.jl")
include("exciters/brushless.jl")
include("exciters/shunt.jl")
include("governors/hydro.jl")
include("governors/steam.jl")
include("converters/vsm.jl")
include("converters/dcside.jl")
include("converters/droop.jl")
include("converters/voc.jl")
include("converters/gfl.jl")
include("converters/core.jl")
# governors/
# loads/
# converters/

end
