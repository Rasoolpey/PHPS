// steam_a.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body A of the steam unit (12 dry storages at II 9); src/rt/governors/steam.jl
#include "steam_a.h"

int sta_norm(Sig a, Rq mo[1]) {
    #pragma HLS INLINE
    int k = 99;
    Rq m = 0.75;
    Sig c8 = 8;
    Sig c4 = 4;
    Sig c2 = 2;
    Sig c1 = 1;
    Sig c0 = 0.5;
    Sig cm1 = 0.25;
    Sig cm2 = 0.125;
    Sig cm3 = 0.0625;
    Sig cm4 = 0.03125;
    Sig cm5 = 0.015625;
    Sig cm6 = 0.0078125;
    Sig cm7 = 0.00390625;
    Sig cm8 = 0.001953125;
    Sig cm9 = 0.0009765625;
    if (a >= c8) {
        k = 99;
    } else if (a >= c4) {
        m = scale2<-3>(a);
        k = 3;
    } else if (a >= c2) {
        m = scale2<-2>(a);
        k = 2;
    } else if (a >= c1) {
        m = scale2<-1>(a);
        k = 1;
    } else if (a >= c0) {
        m = a;
        k = 0;
    } else if (a >= cm1) {
        m = scale2<1>(a);
        k = -1;
    } else if (a >= cm2) {
        m = scale2<2>(a);
        k = -2;
    } else if (a >= cm3) {
        m = scale2<3>(a);
        k = -3;
    } else if (a >= cm4) {
        m = scale2<4>(a);
        k = -4;
    } else if (a >= cm5) {
        m = scale2<5>(a);
        k = -5;
    } else if (a >= cm6) {
        m = scale2<6>(a);
        k = -6;
    } else if (a >= cm7) {
        m = scale2<7>(a);
        k = -7;
    } else if (a >= cm8) {
        m = scale2<8>(a);
        k = -8;
    } else if (a >= cm9) {
        m = scale2<9>(a);
        k = -9;
    }
    mo[0] = m;
    return k;
}

Sig sta_sh(Sig x, int k) {
    #pragma HLS INLINE
    Sig r = x;
    if (k == 3) {
        r = scale2<-3>(x);
    } else if (k == 2) {
        r = scale2<-2>(x);
    } else if (k == 1) {
        r = scale2<-1>(x);
    } else if (k == -1) {
        r = scale2<1>(x);
    } else if (k == -2) {
        r = scale2<2>(x);
    } else if (k == -3) {
        r = scale2<3>(x);
    } else if (k == -4) {
        r = scale2<4>(x);
    } else if (k == -5) {
        r = scale2<5>(x);
    } else if (k == -6) {
        r = scale2<6>(x);
    } else if (k == -7) {
        r = scale2<7>(x);
    } else if (k == -8) {
        r = scale2<8>(x);
    } else if (k == -9) {
        r = scale2<9>(x);
    }
    return r;
}

int sta_j(Rq m) {
    #pragma HLS INLINE
    int j = ifloor(scale2<11>(m)) - 1024;
    if (j < 0) {
        j = 0;
    }
    if (j > 1023) {
        j = 1023;
    }
    return j;
}

Rq sta_rcp(Rq m, Rq w, Rq e) {
    #pragma HLS INLINE
    Rq two = 2;
    Rq t1 = two - e;
    Rq y1 = w * t1;
    Rq e2 = m * y1;
    Rq t2 = two - e2;
    Rq y = y1 * t2;
    return y;
}

bool sta_cell(Tc tt, int lo, int hi, Cf flo, Cf fhi, Cf tlo, Cf thi, Wt cl, Wt ch, int ci[1], Wt ct[1]) {
    #pragma HLS INLINE
    bool ok = tt >= tlo && tt <= thi;
    int i = lo;
    if (tt < flo) {
        i = lo;
    } else if (tt >= fhi) {
        i = hi;
    } else {
        i = ifloor(tt - flo) + lo;
    }
    Tc tf = tt - i;
    Wt t = 0;
    if (ok) {
        t = tf;
    } else if (tf < cl) {
        t = cl;
    } else if (tf > ch) {
        t = ch;
    } else {
        t = tf;
    }
    ci[0] = i - lo;
    ct[0] = t;
    return ok;
}

void sta_basis(Wt t, Wt w[4]) {
    #pragma HLS INLINE
    Wt one = 1;
    Wt t2 = t * t;
    Wt t3 = t2 * t;
    Wt a = t3 - t2;
    Wt h01 = t2 + t2 + t2 - t3 - t3;
    w[0] = one - h01;
    w[1] = h01;
    w[2] = a - t2 + t;
    w[3] = a;
}

Tw sta_v2(Tw F[6048], Tw X[6048], Tw Y[6048], Tw Z[6048], int b, Wt a0, Wt a1, Wt c0, Wt c1, Wt b0, Wt b1, Wt d0, Wt d1) {
    #pragma HLS INLINE
    Tw s1 = a0 * F[b] + a1 * F[b + 18] + c0 * X[b] + c1 * X[b + 18];
    Tw s2 = a0 * Y[b] + a1 * Y[b + 18] + c0 * Z[b] + c1 * Z[b + 18];
    Tw t1 = a0 * F[b + 1] + a1 * F[b + 19] + c0 * X[b + 1] + c1 * X[b + 19];
    Tw t2 = a0 * Y[b + 1] + a1 * Y[b + 19] + c0 * Z[b + 1] + c1 * Z[b + 19];
    Tw r = b0 * s1 + d0 * s2 + b1 * t1 + d1 * t2;
    return r;
}

void steam_a_run(const SteamACoefs &k, int mode, int la, Tw lw, St M[12], St U[12], Sig p[12], Sig hh[12], Sig v[12], Sig s[12], int fl[12], Wt wo[12][8], int co[12][2]) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=mode
    #pragma HLS INTERFACE mode=s_axilite port=la
    #pragma HLS INTERFACE mode=s_axilite port=lw
    #pragma HLS INTERFACE mode=s_axilite port=M
    #pragma HLS INTERFACE mode=s_axilite port=U
    #pragma HLS INTERFACE mode=s_axilite port=p
    #pragma HLS INTERFACE mode=s_axilite port=hh
    #pragma HLS INTERFACE mode=s_axilite port=v
    #pragma HLS INTERFACE mode=s_axilite port=s
    #pragma HLS INTERFACE mode=s_axilite port=fl
    #pragma HLS INTERFACE mode=s_axilite port=wo
    #pragma HLS INTERFACE mode=s_axilite port=co
    static Tw VF[6048];
    static Tw VX[6048];
    static Tw VY[6048];
    static Tw VZ[6048];
    #pragma HLS BIND_STORAGE variable=VF type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VX type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VY type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VZ type=ram_t2p impl=bram
    static const Rq LGI[1024] = STA_LGI;
    static const Lw LGL[1024] = STA_LGL;
    static const Cf KL[13] = STA_KLN2;
    static const Tw G[4][91] = STA_GAS;
    #pragma HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=G type=rom_2p impl=bram
    if (mode == 1) {
        if (la < 6048) {
            VF[la] = lw;
        } else if (la < 12096) {
            VX[la - 6048] = lw;
        } else if (la < 18144) {
            VY[la - 12096] = lw;
        } else {
            VZ[la - 18144] = lw;
        }
    } else {
        Cf cvt[12];
        Cf ccx[12];
        int cxlo[12];
        int cxhi[12];
        Cf cxflo[12];
        Cf cxfhi[12];
        Cf cxtlo[12];
        Cf cxthi[12];
        Wt cxcl[12];
        Wt cxch[12];
        int cylo[12];
        int cyhi[12];
        Cf cyflo[12];
        Cf cyfhi[12];
        Cf cytlo[12];
        Cf cythi[12];
        Wt cycl[12];
        Wt cych[12];
        for (int i = 0; i < 12; i++) {
            cvt[i] = k.vt[i];
            ccx[i] = k.cx[i];
            cxlo[i] = k.xlo[i];
            cxhi[i] = k.xhi[i];
            cxflo[i] = k.xflo[i];
            cxfhi[i] = k.xfhi[i];
            cxtlo[i] = k.xtlo[i];
            cxthi[i] = k.xthi[i];
            cxcl[i] = k.xcl[i];
            cxch[i] = k.xch[i];
            cylo[i] = k.ylo[i];
            cyhi[i] = k.yhi[i];
            cyflo[i] = k.yflo[i];
            cyfhi[i] = k.yfhi[i];
            cytlo[i] = k.ytlo[i];
            cythi[i] = k.ythi[i];
            cycl[i] = k.ycl[i];
            cych[i] = k.ych[i];
        }
        Rq one = 1;
        Rq third = 0.3333333333333333;
        Rq half = 0.5;
        for (int it = 0; it < 12; it++) {
            #pragma HLS PIPELINE II=9
            #pragma HLS ALLOCATION operation instances=mul limit=8
            St Mi = M[it];
            St Ui = U[it];
            Rq mo[1];
            int km = sta_norm(Mi, mo);
            Rq m = mo[0];
            int j = sta_j(m);
            Rq w = LGI[j];
            Rq e = m * w;
            Rs r = e - one;
            Rq y = sta_rcp(m, w, e);
            Rq q1 = third - scale2<-2>(r);
            Rq q2 = r * q1 - half;
            Rq q3 = r * q2 + one;
            Rs lg = r * q3;
            int kl = km;
            if (km == 99) {
                kl = 0;
            }
            Sig X = lg + LGL[j] + KL[kl + 9] + ccx[it];
            Sig uy = Ui * y;
            Sig u = sta_sh(uy, km);
            Sig vy = cvt[it] * y;
            Sig vv = sta_sh(vy, km);
            Tc tx = (X - k.gx0) * k.gidx;
            int ci[1];
            Wt ct[1];
            bool okx = sta_cell(tx, cxlo[it], cxhi[it], cxflo[it], cxfhi[it], cxtlo[it], cxthi[it], cxcl[it], cxch[it], ci, ct);
            int lx = ci[0];
            int ix = lx + cxlo[it];
            Wt wx[4];
            sta_basis(ct[0], wx);
            Sig ug = wx[0] * G[0][ix] + wx[1] * G[0][ix + 1] + wx[2] * G[2][ix] + wx[3] * G[2][ix + 1];
            Sig ut = wx[0] * G[1][ix] + wx[1] * G[1][ix + 1] + wx[2] * G[3][ix] + wx[3] * G[3][ix + 1];
            Sig uge = ug * k.eps1;
            bool clear = u >= uge;
            Sig dd = ut - ug;
            Rq mo2[1];
            int kd = sta_norm(dd, mo2);
            Rq m2 = mo2[0];
            int j2 = sta_j(m2);
            Rq w2 = LGI[j2];
            Rq e2 = m2 * w2;
            Rq y2 = sta_rcp(m2, w2, e2);
            Sig du = u - ug;
            Tc xy = du * y2;
            Tc xi = sta_sh(xy, kd);
            Tc ty = (xi - k.vy0) * k.vidy;
            int cj[1];
            Wt cu[1];
            bool oky = sta_cell(ty, cylo[it], cyhi[it], cyflo[it], cyfhi[it], cytlo[it], cythi[it], cycl[it], cych[it], cj, cu);
            int ly = cj[0];
            Wt wy[4];
            sta_basis(cu[0], wy);
            int bp = ((it + 1) - 1) * 504 + lx * 18 + ly;
            Sig pp = sta_v2(VF, VX, VY, VZ, bp, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3]);
            Sig ss = sta_v2(VF, VX, VY, VZ, bp + 252, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3]);
            Sig pv = pp * vv;
            p[it] = pp;
            hh[it] = u + pv;
            v[it] = vv;
            s[it] = ss;
            int f = 0;
            if (!clear || (!okx || (!oky || (km == 99 || kd == 99)))) {
                f = 1;
            }
            fl[it] = f;
            for (int q = 0; q < 4; q++) {
                #pragma HLS UNROLL
                wo[it][q] = wx[q];
                wo[it][q + 4] = wy[q];
            }
            co[it][0] = lx;
            co[it][1] = ly;
        }
    }
}

