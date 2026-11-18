// network.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the IEEE-39 network block, one run of steps of h = 8.64e-7 s; from src/rt/network/network_gen.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, -5, AP_RND_CONV> X;
typedef ap_fixed<48, 1, AP_RND_CONV> C;
typedef ap_fixed<77, 1> P;
typedef ap_fixed<77, 1> Acc;
typedef ap_uint<6> Ib;

#include "rt_util.h"
#include "network_rom.h"

void network_mac(C c[8], X oa[8], X ob[8], P pa[8], P pb[8]);
void network_run(X x[158], X ph[2], int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &v1_out, volatile Sig &v2_out, volatile bool &strobe, volatile bool &sample);
