// genrou.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// GENROU, one step of h; from src/rt/machines/genrou.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, -10, AP_RND_CONV> Small;

struct GenrouCoefs {
    Small P[4][4];  // rotor step matrix
    Small W[4][3];  // rotor input matrix (inputs Efd, id, iq)
    Sig kd;  // subtransient weight, d axis
    Sig kq;  // subtransient weight, q axis
    Sig S[4];  // stator matrix, row-major
    Sig a1;  // speed damping coefficient
    Small a2;  // speed forcing coefficient h / 2H
    Small wb_h;  // omega_b h
};

void genrou_step(Sig x[8], Sig u[4], const GenrouCoefs &k);
void genrou_run(const GenrouCoefs &k, Sig x[8], Sig vre, Sig vim, Sig tm, Sig efd, Sig vdip, int n_on, int n_off, int nsteps, int decim, volatile Sig &delta_out, volatile Sig &omega_out, volatile Sig &eq_out, volatile Sig &psid_out, volatile bool &strobe, volatile bool &sample);
