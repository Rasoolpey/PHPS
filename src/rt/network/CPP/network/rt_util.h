// rt_util.h: written by PHPS.RT (src/rt/translator/julia_to_cpp.jl); do not edit
#pragma once
#include <ap_fixed.h>
// x 2^K exactly: the format is widened by |K| bits first, so the shift loses nothing (no multiplier: wiring)
template<int K, int W, int I, ap_q_mode Q, ap_o_mode O, int N>
ap_fixed<W + (K < 0 ? -K : K), I + (K > 0 ? K : 0), Q, O, N> scale2(ap_fixed<W, I, Q, O, N> x) {
    ap_fixed<W + (K < 0 ? -K : K), I + (K > 0 ? K : 0), Q, O, N> y = x;
    if (K > 0) y <<= K; else if (K < 0) y >>= -K;
    return y;
}
