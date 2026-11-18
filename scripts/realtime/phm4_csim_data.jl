# PHMachine4's C simulation data (REALTIME_TODO START HERE item 6; review 47d's next step): all ten machine sets through the
# one build (phm4_run!), each on the dip bench (the port source at its operating voltage, x 0.5 for 0.1-0.2 s, 0.3 s), cut at
# checkpoints (every 0.01 s and around both edges): the C++ states (18) and outputs (2) against the same @hls code in Julia
# (Float64), and a continuous run equal to the segmented one bit for bit. Writes the coefficient header of all ten sets
# and the testbench src/rt/machines/CPP/phmachine4/phmachine4_tb.cpp (the Julia references inline).
#   julia --project=. scripts/realtime/phm4_csim_data.jl      (after PHPS.RT.phm4_export(): the same types and h)
using PHPS, Printf
const RT = PHPS.RT
root = pwd()
cpp = joinpath(root, "src", "rt", "machines", "CPP", "phmachine4")
h = setprecision(() -> BigFloat(RT.PHM4_STEP_CYCLES) * RT.PHM4_CLOCK_NS / 10^9, BigFloat, 256)
hh = Float64(h)
names = haskey(ENV, "PHM4_ONLY") ? String.(split(ENV["PHM4_ONLY"], ",")) : RT.phm4_names()   # PHM4_ONLY=GENROU_1,GENROU_11
nst = round(Int, 0.3 / hh); non = round(Int, 0.1 / hh); noff = round(Int, 0.2 / hh)
cuts = sort(unique(vcat([round(Int, t / hh) for t in 0.01:0.01:0.3], non - 1, non, non + 1, non + 2, noff - 1, noff, noff + 1, noff + 2, nst)))
f(v) = repr(Float64(v))
sets = Any[]
for nm in names
    d = RT.phm4_data(nm)
    k = setprecision(() -> RT.phm4_coefstruct(d, h), BigFloat, 256)
    x0, b = RT.phm4_initial(d, hh)
    x = copy(x0); recs = Vector{Vector{Float64}}(); outs = Vector{Vector{Float64}}(); n = 0
    ylast = zeros(2)
    for c in cuts
        RT.phm4_bench!(x, k, b, n, c - n; non, noff, out = (i, xx, y) -> (ylast .= y))
        push!(recs, copy(x)); push!(outs, copy(ylast)); n = c
    end
    push!(sets, (; nm, k, x0, b, recs, outs))
    @printf("%-12s: %d steps, %d checkpoints; omega at 0.3 s %.10f\n", nm, nst, length(cuts), x[9])
end
open(joinpath(cpp, "phmachine4_coefs_all.h"), "w") do io
    println(io, "// phmachine4_coefs_all.h: written by scripts/realtime/phm4_csim_data.jl; do not edit (h = $hh s)")
    for (i, s) in enumerate(sets)
        println(io, "const PHM4Coefs PHM4_COEFS_$i = ", RT._cinit(s.k), ";  // ", s.nm)
    end
    println(io, "const PHM4Coefs *PHM4_SETS[$(length(sets))] = {", join(("&PHM4_COEFS_$i" for i in eachindex(sets)), ", "), "};")
end
open(joinpath(cpp, "phmachine4_tb.cpp"), "w") do io
    print(io, """
// phmachine4_tb.cpp: written by scripts/realtime/phm4_csim_data.jl; do not edit
// C simulation of PHMachine4 (src/rt/machines/phmachine4.jl): all ten machine sets through the one build, each on the dip
// bench (x 0.5 for steps $(non + 1)-$noff, $nst steps of h = $hh s), cut at $(length(cuts)) checkpoints; at each, the C++
// states (18) and outputs (2) against the same @hls code in Julia (Float64). Limits (fixed before results, as PHMachine):
// states and outputs 5e-8 absolute + relative, phasor lengths 1e-10, a continuous run equal to the segmented one bit for bit.
#include <cstdio>
#include <cmath>
#include "phmachine4.h"
#include "phmachine4_coefs_all.h"
const int NSET = $(length(sets)), NCUT = $(length(cuts)), N_ON = $non, N_OFF = $noff;
const int CUTS[NCUT] = {$(join(cuts, ", "))};
const char *NAMES[NSET] = {$(join(("\"" * s.nm * "\"" for s in sets), ", "))};
const double LIM = 5e-8, LIM_NORM = 1e-10;
const double X0[NSET][18] = {$(join(("{" * join(map(f, s.x0), ", ") * "}" for s in sets), ",\n    "))};
const double BEN[NSET][5] = {$(join(("{" * join(map(f, (s.b.vmag, s.b.cm, s.b.srot, s.b.vf, s.b.tm)), ", ") * "}" for s in sets), ",\n    "))};
const double JX[NSET][NCUT][18] = {
$(join(("  {" * join(("{" * join(map(f, r), ", ") * "}" for r in s.recs), ", ") * "}" for s in sets), ",\n"))
};
const double JY[NSET][NCUT][2] = {
$(join(("  {" * join(("{" * join(map(f, o), ", ") * "}" for o in s.outs), ", ") * "}" for s in sets), ",\n"))
};
static double rd(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }
int main() {
    bool ok = true; double wall = 0;
    FILE *dump = std::fopen("$(joinpath(root, "outputs", "realtime", "flow", "phmachine4", "phm4_cpp_states.bin"))", "wb");   // the C++ checkpoints (raw words) for the board
    for (int m = 0; m < NSET; m++) {
        const PHM4Coefs &k = *PHM4_SETS[m];
        St x[18], xc[18];
        for (int i = 0; i < 18; i++) { x[i] = X0[m][i]; xc[i] = X0[m][i]; }
        Sig om, ia, ib; bool st, sa; double wx = 0, wn = 0; int prev = 0, wi = -1, wg = -1; double ww[20] = {0}, vv[20] = {0};
        for (int g = 0; g < NCUT; g++) {
            phm4_run(k, x, BEN[m][0], BEN[m][1], BEN[m][2], BEN[m][3], BEN[m][4], 0.5, prev, N_ON, N_OFF, CUTS[g] - prev, 32, om, ia, ib, st, sa);
            prev = CUTS[g];
            for (int i = 0; i < 18; i++) { long long r = x[i].range(47, 0).to_int64(); std::fwrite(&r, 8, 1, dump); }
            { long long r = ia.range(47, 0).to_int64(); std::fwrite(&r, 8, 1, dump); r = ib.range(47, 0).to_int64(); std::fwrite(&r, 8, 1, dump); }
            for (int i = 0; i < 18; i++) { double e = rd(x[i].to_double(), JX[m][g][i]); if (e > ww[i]) { ww[i] = e; vv[i] = JX[m][g][i]; } if (e > wx) { wx = e; wi = i; wg = g; } }
            for (int i = 0; i < 2; i++) { double e = rd((i ? ib : ia).to_double(), JY[m][g][i]); if (e > ww[18 + i]) { ww[18 + i] = e; vv[18 + i] = JY[m][g][i]; } if (e > wx) { wx = e; wi = 18 + i; wg = g; } }
            double c = x[9].to_double(), s = x[10].to_double(), sc = x[16].to_double(), ss = x[17].to_double();
            wn = std::fmax(wn, std::fmax(std::fabs(c * c + s * s - 1), std::fabs(sc * sc + ss * ss - 1)));
        }
        phm4_run(k, xc, BEN[m][0], BEN[m][1], BEN[m][2], BEN[m][3], BEN[m][4], 0.5, 0, N_ON, N_OFF, CUTS[NCUT - 1], 32, om, ia, ib, st, sa);
        bool same = true;
        for (int i = 0; i < 18; i++) same = same && (xc[i] == x[i]);
        bool p = wx <= LIM && wn <= LIM_NORM && same;
        std::printf("C sim check: %-12s %d checkpoints: max |C++ - Julia| %.3e (limit %.0e; word %d at step %d), max |norm - 1| %.3e (limit %.0e), continuous = segmented %s: %s\\n",
                    NAMES[m], NCUT, wx, LIM, wi + 1, wg >= 0 ? CUTS[wg] : -1, wn, LIM_NORM, same ? "bit for bit" : "NO", p ? "PASS" : "FAIL");
        ok = ok && p; wall = std::fmax(wall, wx);
        std::printf("   per word (max relative diff @ the Julia value):");
        for (int i = 0; i < 20; i++) std::printf(" %d:%.1e@%.3g", i + 1, ww[i], vv[i]);
        std::printf("\\n");
    }
    std::fclose(dump);
    std::printf("C sim check: %s (max %.3e)\\n", ok ? "ALL PASS" : "FAIL", wall);
    return ok ? 0 : 1;
}
""")
end
println("written ", joinpath(cpp, "phmachine4_tb.cpp"), " and phmachine4_coefs_all.h")
