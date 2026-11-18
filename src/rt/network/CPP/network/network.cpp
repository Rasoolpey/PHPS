// network.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the IEEE-39 network block, one run of steps of h = 8.64e-7 s; from src/rt/network/network_gen.jl
#include "network.h"

void network_mac(C c[8], X oa[8], X ob[8], P pa[8], P pb[8]) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    for (int l = 0; l < 8; l++) {
        #pragma HLS UNROLL
        pa[l] = c[l] * oa[l];
        pb[l] = c[l] * ob[l];
    }
}

void network_run(X x[158], X ph[2], int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &v1_out, volatile Sig &v2_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=x storage_impl=bram
    #pragma HLS INTERFACE mode=s_axilite port=ph
    #pragma HLS INTERFACE mode=s_axilite port=n0
    #pragma HLS INTERFACE mode=s_axilite port=n_on
    #pragma HLS INTERFACE mode=s_axilite port=n_off
    #pragma HLS INTERFACE mode=s_axilite port=nsteps
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=v1_out
    #pragma HLS INTERFACE mode=ap_none port=v2_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    X xa[9][12];  // x operand banks, alpha (bank = writing lane; 9: the phasor)
    #pragma HLS ARRAY_PARTITION variable=xa dim=1 complete
    #pragma HLS BIND_STORAGE variable=xa type=ram_1wnr impl=lutram
    X xb[9][12];  // beta
    #pragma HLS ARRAY_PARTITION variable=xb dim=1 complete
    #pragma HLS BIND_STORAGE variable=xb type=ram_1wnr impl=lutram
    X xo[8][12];  // the owner's copy of its x slots (read-modify-write), so
    #pragma HLS ARRAY_PARTITION variable=xo dim=1 complete
    #pragma HLS BIND_STORAGE variable=xo type=ram_1wnr impl=lutram
    X yo[8][12];
    #pragma HLS ARRAY_PARTITION variable=yo dim=1 complete
    #pragma HLS BIND_STORAGE variable=yo type=ram_1wnr impl=lutram
    X qa[8][5];  // rq banks (written in P1, read in P2)
    #pragma HLS ARRAY_PARTITION variable=qa dim=1 complete
    #pragma HLS BIND_STORAGE variable=qa type=ram_1wnr impl=lutram
    X qb[8][5];
    #pragma HLS ARRAY_PARTITION variable=qb dim=1 complete
    #pragma HLS BIND_STORAGE variable=qb type=ram_1wnr impl=lutram
    X da[8][5];  // dq banks (written in P2, read in P3)
    #pragma HLS ARRAY_PARTITION variable=da dim=1 complete
    #pragma HLS BIND_STORAGE variable=da type=ram_1wnr impl=lutram
    X db[8][5];
    #pragma HLS ARRAY_PARTITION variable=db dim=1 complete
    #pragma HLS BIND_STORAGE variable=db type=ram_1wnr impl=lutram
    X pa[8][8];  // rp, lane-local (written in P1, the init of P3)
    #pragma HLS ARRAY_PARTITION variable=pa dim=1 complete
    #pragma HLS BIND_STORAGE variable=pa type=ram_1wnr impl=lutram
    X pb[8][8];
    #pragma HLS ARRAY_PARTITION variable=pb dim=1 complete
    #pragma HLS BIND_STORAGE variable=pb type=ram_1wnr impl=lutram
    X pn[8][2];  // the phasor rows' increments, lane-local
    #pragma HLS ARRAY_PARTITION variable=pn dim=0 complete
    X pm[8][2];
    #pragma HLS ARRAY_PARTITION variable=pm dim=0 complete
    C cc[8];  // one cycle's coefficients, operands and products
    #pragma HLS ARRAY_PARTITION variable=cc complete
    X va[8];
    #pragma HLS ARRAY_PARTITION variable=va complete
    X vb[8];
    #pragma HLS ARRAY_PARTITION variable=vb complete
    P pa_[8];
    #pragma HLS ARRAY_PARTITION variable=pa_ complete
    P pb_[8];
    #pragma HLS ARRAY_PARTITION variable=pb_ complete
    Acc aa[8];  // accumulators (>= 16 guard bits below X's grid)
    #pragma HLS ARRAY_PARTITION variable=aa complete
    Acc ab[8];
    #pragma HLS ARRAY_PARTITION variable=ab complete
    Acc zero = 0;
    X oa1[25];  // the 25 bank outputs (1-8 x, 9 phasor, 10-17 rq, 18-25 dq)
    #pragma HLS ARRAY_PARTITION variable=oa1 complete
    X ob1[25];
    #pragma HLS ARRAY_PARTITION variable=ob1 complete
    X oa2[25];
    #pragma HLS ARRAY_PARTITION variable=oa2 complete
    X ob2[25];
    #pragma HLS ARRAY_PARTITION variable=ob2 complete
    static const C rcp[8][89] = NET_CP;  // the lanes' program, one entry per lane and cycle: the
    static const C rcf[8][89] = NET_CF;  // coefficient (pre-fault, fault), the source bank (1-25),
    static const Ib rs[8][89] = NET_RS;  // first / last entry of a row, the init from rp, the write
    static const Ib rb1[25][89] = NET_BA1;  // kind (1 rq, 2 rp, 3 phasor, 4 dq and x_q, 5 x_p) and its
    static const Ib rb2[25][89] = NET_BA2;  // slots; per bank and group one read address (rb1, rb2;
    static const bool rf[8][89] = NET_F;  // unused: the parking slot)
    static const bool re[8][89] = NET_E;
    static const bool rik[8][89] = NET_IK;
    static const Ib rwk[8][89] = NET_WK;
    static const Ib rdq[8][89] = NET_DQ;
    static const Ib rdp[8][89] = NET_DP;
    static const Ib rdd[8][89] = NET_DD;
    static const Ib rdn[8][89] = NET_DN;
    static const Ib rw[8][89] = NET_W;
    static const Ib ri[8][89] = NET_I;
    static const Ib cbk[79] = NET_CB;  // bank and slot of each state (copy in and out)
    static const Ib csl[79] = NET_CS;
    #pragma HLS ARRAY_PARTITION variable=rcp dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rcf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rs dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=re dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rik dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rwk dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rdq dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rdp dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rdd dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rdn dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rw dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=ri dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rb1 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=rb2 dim=1 complete
    X z0 = 0;  // the parking slots hold zero (read, never written)
    for (int l = 0; l < 9; l++) {
        xa[l][11] = z0;
        xb[l][11] = z0;
    }
    for (int l = 0; l < 8; l++) {
        xo[l][11] = z0;
        yo[l][11] = z0;
        qa[l][4] = z0;
        qb[l][4] = z0;
        da[l][4] = z0;
        db[l][4] = z0;
        pa[l][7] = z0;
        pb[l][7] = z0;
    }
    for (int r = 0; r < 79; r++) {  // copy in: the states to their banks (and owner copies)
        xa[cbk[r] - 1][csl[r] - 1] = x[r];
        xb[cbk[r] - 1][csl[r] - 1] = x[r + 79];
        xo[cbk[r] - 1][csl[r] - 1] = x[r];
        yo[cbk[r] - 1][csl[r] - 1] = x[r + 79];
    }
    xa[8][0] = ph[0];  // the phasor bank: slot 1 (c, s), slot 2 (-s, c), x 2^sig
    xa[8][1] = -ph[1];
    xb[8][0] = ph[1];
    xb[8][1] = ph[0];
    bool st = false;
    int cnt = 0;
    for (int i = 0; i < nsteps; i++) {
        #pragma HLS ALLOCATION function instances=network_mac limit=1
        int n = n0 + (i + 1);
        bool fault = false;  // the fault bank, latched for the whole step
        if (n > n_on && n <= n_off) {
            fault = true;
        }
        for (int t = 0; t < 89; t++) {  // the step: rq, rp, phasor, dq (x_q), x_p rows interleaved
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=xa type=inter dependent=false
            #pragma HLS DEPENDENCE variable=xb type=inter dependent=false
            #pragma HLS DEPENDENCE variable=xo type=inter dependent=false
            #pragma HLS DEPENDENCE variable=yo type=inter dependent=false
            #pragma HLS DEPENDENCE variable=qa type=inter dependent=false
            #pragma HLS DEPENDENCE variable=qb type=inter dependent=false
            #pragma HLS DEPENDENCE variable=da type=inter dependent=false
            #pragma HLS DEPENDENCE variable=db type=inter dependent=false
            #pragma HLS DEPENDENCE variable=pa type=inter dependent=false
            #pragma HLS DEPENDENCE variable=pb type=inter dependent=false
            for (int b = 0; b < 9; b++) {  // one read per bank and group (review 37)
                #pragma HLS UNROLL
                oa1[b] = xa[b][rb1[b][t] - 1];
                ob1[b] = xb[b][rb1[b][t] - 1];
                oa2[b] = xa[b][rb2[b][t] - 1];
                ob2[b] = xb[b][rb2[b][t] - 1];
            }
            for (int b = 0; b < 8; b++) {
                #pragma HLS UNROLL
                oa1[b + 9] = qa[b][rb1[b + 9][t] - 1];
                ob1[b + 9] = qb[b][rb1[b + 9][t] - 1];
                oa2[b + 9] = qa[b][rb2[b + 9][t] - 1];
                ob2[b + 9] = qb[b][rb2[b + 9][t] - 1];
                oa1[b + 17] = da[b][rb1[b + 17][t] - 1];
                ob1[b + 17] = db[b][rb1[b + 17][t] - 1];
                oa2[b + 17] = da[b][rb2[b + 17][t] - 1];
                ob2[b + 17] = db[b][rb2[b + 17][t] - 1];
            }
            for (int l = 0; l < 8; l++) {
                #pragma HLS UNROLL
                cc[l] = rcp[l][t];
                if (fault) {
                    cc[l] = rcf[l][t];
                }
                if ((l + 1) <= 4) {
                    va[l] = oa1[rs[l][t] - 1];
                    vb[l] = ob1[rs[l][t] - 1];
                } else {
                    va[l] = oa2[rs[l][t] - 1];
                    vb[l] = ob2[rs[l][t] - 1];
                }
            }
            network_mac(cc, va, vb, pa_, pb_);
            for (int l = 0; l < 8; l++) {
                #pragma HLS UNROLL
                if (rf[l][t]) {
                    if (rik[l][t]) {
                        aa[l] = scale2<0>(pa[l][ri[l][t] - 1]) + pa_[l];
                        ab[l] = scale2<0>(pb[l][ri[l][t] - 1]) + pb_[l];
                    } else {
                        aa[l] = zero + pa_[l];
                        ab[l] = zero + pb_[l];
                    }
                } else {
                    aa[l] = aa[l] + pa_[l];
                    ab[l] = ab[l] + pb_[l];
                }
                if (rwk[l][t] == 1) {
                    qa[l][rdq[l][t] - 1] = scale2<2>(aa[l]);
                    qb[l][rdq[l][t] - 1] = scale2<2>(ab[l]);
                } else if (rwk[l][t] == 2) {
                    pa[l][rdp[l][t] - 1] = scale2<7>(aa[l]);
                    pb[l][rdp[l][t] - 1] = scale2<7>(ab[l]);
                } else if (rwk[l][t] == 3) {
                    pn[l][rdn[l][t] - 1] = scale2<-11>(aa[l]);
                    pm[l][rdn[l][t] - 1] = scale2<-11>(ab[l]);
                } else if (rwk[l][t] == 4) {
                    da[l][rdd[l][t] - 1] = scale2<1>(aa[l]);
                    db[l][rdd[l][t] - 1] = scale2<1>(ab[l]);
                    X ya2 = xo[l][rw[l][t] - 1] + scale2<-2>(aa[l]);
                    X yb2 = yo[l][rw[l][t] - 1] + scale2<-2>(ab[l]);
                    xo[l][rw[l][t] - 1] = ya2;
                    yo[l][rw[l][t] - 1] = yb2;
                    xa[l][rw[l][t] - 1] = ya2;
                    xb[l][rw[l][t] - 1] = yb2;
                } else if (rwk[l][t] == 5) {
                    X ya3 = xo[l][rw[l][t] - 1] + scale2<-11>(aa[l]);
                    X yb3 = yo[l][rw[l][t] - 1] + scale2<-11>(ab[l]);
                    xo[l][rw[l][t] - 1] = ya3;
                    yo[l][rw[l][t] - 1] = yb3;
                    xa[l][rw[l][t] - 1] = ya3;
                    xb[l][rw[l][t] - 1] = yb3;
                }
            }
        }
        xa[8][0] = xa[8][0] + pn[7][0];  // the phasor for the next step (P2, P3 do not read it)
        xa[8][1] = xa[8][1] + pn[0][1];
        xb[8][0] = xb[8][0] + pm[7][0];
        xb[8][1] = xb[8][1] + pm[0][1];
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
        v1_out = xa[3][1];  // alpha charge of node bus 16 (v = Q x), to the ILA
        v2_out = xa[0][3];  // alpha charge of node bus 39
        strobe = st;
    }
    for (int r = 0; r < 79; r++) {  // copy out
        x[r] = xa[cbk[r] - 1][csl[r] - 1];
        x[r + 79] = xb[cbk[r] - 1][csl[r] - 1];
    }
    ph[0] = xa[8][0];
    ph[1] = xb[8][0];
}

