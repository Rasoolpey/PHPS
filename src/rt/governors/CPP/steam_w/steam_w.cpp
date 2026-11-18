// steam_w.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body W of the steam unit (the wet-capable slot, II 110 on 3 multipliers); src/rt/governors/steam.jl
#include "steam_w.h"

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

int stb_norm(Sig a, Rq mo[1]) {
    #pragma HLS INLINE
    Sig m = a;
    int k = 0;
    Sig c8 = 8;
    Sig c2 = 2;
    Sig c1 = 1;
    Sig e16 = 1.52587890625e-5;
    Sig e8 = 0.00390625;
    Sig e4 = 0.0625;
    Sig e2 = 0.25;
    Sig e1 = 0.5;
    if (m >= c8) {
        m = scale2<-4>(m);
        k = 4;
    }
    if (m >= c2) {
        m = scale2<-2>(m);
        k = k + 2;
    }
    if (m >= c1) {
        m = scale2<-1>(m);
        k = k + 1;
    }
    if (m < e16) {
        m = scale2<16>(m);
        k = k - 16;
    }
    if (m < e8) {
        m = scale2<8>(m);
        k = k - 8;
    }
    if (m < e4) {
        m = scale2<4>(m);
        k = k - 4;
    }
    if (m < e2) {
        m = scale2<2>(m);
        k = k - 2;
    }
    if (m < e1) {
        m = scale2<1>(m);
        k = k - 1;
    }
    Rq r = m;
    Rq tq = 0.75;
    if (m < e1) {
        k = 99;
        r = tq;
    }
    mo[0] = r;
    return k;
}

Sig stb_sh(Sig x, int n) {
    #pragma HLS INLINE
    Sig r = x;
    int s = n;
    if (n < 0) {
        s = 0 - n;
        if (s >= 8) {
            r = scale2<-8>(r);
            s = s - 8;
        }
        if (s >= 4) {
            r = scale2<-4>(r);
            s = s - 4;
        }
        if (s >= 2) {
            r = scale2<-2>(r);
            s = s - 2;
        }
        if (s >= 1) {
            r = scale2<-1>(r);
        }
    } else {
        if (s >= 8) {
            r = scale2<8>(r);
            s = s - 8;
        }
        if (s >= 4) {
            r = scale2<4>(r);
            s = s - 4;
        }
        if (s >= 2) {
            r = scale2<2>(r);
            s = s - 2;
        }
        if (s >= 1) {
            r = scale2<1>(r);
        }
    }
    return r;
}

bool stw_cell(Tc tt, Cf tlo, Cf thi, int ci[1], Wt ct[1]) {
    #pragma HLS INLINE
    bool ok = tt >= tlo && tt <= thi;
    Tc zero = 0;
    Tc top = 119;
    int i = 0;
    if (tt < zero) {
        i = 0;
    } else if (tt >= top) {
        i = 119;
    } else {
        i = ifloor(tt);
    }
    Tc tf = tt - i;
    ci[0] = i;
    ct[0] = tf;
    return ok;
}

void stw_dbasis(Wt t, Wt w[3]) {
    #pragma HLS INLINE
    Wt one = 1;
    Wt t2 = t * t;
    w[0] = t2 - t;
    w[1] = one - t - t - t - t + t2 + t2 + t2;
    w[2] = t2 + t2 + t2 - t - t;
}

Sig stw_srcp(Sig a, int nk[1]) {
    #pragma HLS INLINE
    static const Rq LGI[1024] = STA_LGI;
    Sig zs = 0;
    Sig b = a;
    if (a < zs) {
        b = zs - a;
    }
    Rq mo[1];
    int k = stb_norm(b, mo);
    Rq m = mo[0];
    int j = sta_j(m);
    Rq w = LGI[j];
    Rq e = m * w;
    Rq y = sta_rcp(m, w, e);
    int kk = 0 - k;
    if (k == 99) {
        kk = 0;
    }
    Sig r = stb_sh(y, kk);
    if (a < zs) {
        r = zs - r;
    }
    nk[0] = k;
    return r;
}

Sig stw_exp(Sig l, Cf l2e, Cf l2c) {
    #pragma HLS INLINE
    static const Rq EX2[256] = STW_EX2;
    static const Rq EXC[3] = STW_EXC;
    Rq one = 1;
    Tx zero = 0;
    Tx top = 15.999999;
    Tx t1 = l * l2e;
    Tx t = t1 + l2c;
    if (t < zero) {
        t = zero;
    }
    if (t > top) {
        t = top;
    }
    int k = ifloor(t);
    Tx f = t - k;
    Tc f8 = scale2<8>(f);
    int j = ifloor(f8);
    Tc r8 = f8 - j;
    Wt r = scale2<-8>(r8);
    Rq r2 = r * r;
    Rq q1 = EXC[0] * r;
    Rq q3 = EXC[2] * r;
    Rq lo = one + q1;
    Rq hi = EXC[1] + q3;
    Rq q2 = r2 * hi;
    Rq pl = lo + q2;
    Sig mm = EX2[j] * pl;
    Sig p = stb_sh(mm, k - 11);
    return p;
}

void steam_w_eval(const SteamWCoefs &k, Tw VF[504], Tw VX[504], Tw VY[504], Tw VZ[504], St Mi, St Ui, Sig lg0, Sig l0, Sig o[8], int fo[2]) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=110
    #pragma HLS ALLOCATION operation instances=mul limit=3
    static const Rq LGI[1024] = STA_LGI;
    static const Lw LGL[1024] = STA_LGL;
    static const Cf KL[13] = STA_KLN2;
    static const Tw G[4][91] = STA_GAS;
    static const Tw SV[6][121] = STW_V;
    static const Tw SX[6][121] = STW_X;
    static const Tw SE[6][120] = STW_E;
    static const Tw SD[6][121] = STW_D;
    #pragma HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=G type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SV type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SX type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SE type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SD type=rom_2p impl=bram
    Rq one = 1;
    Rq third = 0.3333333333333333;
    Rq half = 0.5;
    Sig zs = 0;
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
    Rs lgm = r * q3;
    int kl = km;
    if (km == 99) {
        kl = 0;
    }
    Sig X = lgm + LGL[j] + KL[kl + 9] + k.cx;
    Sig uy = Ui * y;
    Sig u = sta_sh(uy, km);
    Sig vy = k.vt * y;
    Sig vv = sta_sh(vy, km);
    Sig vs = scale2<-4>(vv);
    Tc tx = (X - k.gx0) * k.gidx;
    int ci[1];
    Wt ct[1];
    bool okx = sta_cell(tx, k.xlo, k.xhi, k.xflo, k.xfhi, k.xtlo, k.xthi, k.xcl, k.xch, ci, ct);
    int lx = ci[0];
    int ix = lx + k.xlo;
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
    bool oky = sta_cell(ty, k.ylo, k.yhi, k.yflo, k.yfhi, k.ytlo, k.ythi, k.ycl, k.ych, cj, cu);
    int ly = cj[0];
    Wt wy[4];
    sta_basis(cu[0], wy);
    int bp = lx * 18 + ly;
    Sig pd = sta_v2(VF, VX, VY, VZ, bp, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3]);
    Sig sd = sta_v2(VF, VX, VY, VZ, bp + 252, wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3]);
    Sig pvd = pd * vv;
    Sig hd = u + pvd;
    bool okv = okx && (oky && (km != 99 && kd != 99));
    bool inr = vs >= k.vlo && vs <= k.vhi;
    Sig lg = lg0;
    int nk[1];
    bool nbad = false;
    int cc[1];
    Wt tc[1];
    Wt wb[4];
    Wt wd[3];
    for (int it = 0; it < 2; it++) {
        Tc tg = (lg - k.sx0) * k.sidx;
        bool okg = stw_cell(tg, k.stlo, k.sthi, cc, tc);
        int ig = cc[0];
        sta_basis(tc[0], wb);
        stw_dbasis(tc[0], wd);
        Sig fg = wb[0] * SV[1][ig] + wb[1] * SV[1][ig + 1] + wb[2] * SX[1][ig] + wb[3] * SX[1][ig + 1] - vs;
        Sig dg = wd[0] * SE[1][ig] + wd[1] * SD[1][ig] + wd[2] * SD[1][ig + 1];
        Sig rg1 = stw_srcp(dg, nk);
        if (nk[0] == 99) {
            nbad = true;
        }
        lg = lg - fg * rg1;
    }
    Tc tg2 = (lg - k.sx0) * k.sidx;
    bool okg2 = stw_cell(tg2, k.stlo, k.sthi, cc, tc);
    int ig2 = cc[0];
    sta_basis(tc[0], wb);
    Sig ugs = wb[0] * SV[3][ig2] + wb[1] * SV[3][ig2 + 1] + wb[2] * SX[3][ig2] + wb[3] * SX[3][ig2 + 1];
    Sig rg = wb[0] * SV[1][ig2] + wb[1] * SV[1][ig2 + 1] + wb[2] * SX[1][ig2] + wb[3] * SX[1][ig2 + 1] - vs;
    bool wet = !clear && (inr && !(u >= ugs));
    Sig l = l0;
    Sig sv[4];
    Sig ev[4];
    for (int it = 0; it < 2; it++) {
        Tc tw = (l - k.sx0) * k.sidx;
        bool okw = stw_cell(tw, k.stlo, k.sthi, cc, tc);
        int iw = cc[0];
        sta_basis(tc[0], wb);
        stw_dbasis(tc[0], wd);
        for (int q = 0; q < 4; q++) {
            sv[q] = wb[0] * SV[q][iw] + wb[1] * SV[q][iw + 1] + wb[2] * SX[q][iw] + wb[3] * SX[q][iw + 1];
            ev[q] = wd[0] * SE[q][iw] + wd[1] * SD[q][iw] + wd[2] * SD[q][iw + 1];
        }
        Sig rdv = stw_srcp(sv[1] - sv[0], nk);
        if (nk[0] == 99) {
            nbad = true;
        }
        Sig xw = (vs - sv[0]) * rdv;
        Sig dxa = ev[0] + xw * (ev[1] - ev[0]);
        Sig dxl = zs - dxa * rdv;
        Sig duw = sv[3] - sv[2];
        Sig R = u - sv[2] - xw * duw;
        Sig dR = zs - ev[2] - dxl * duw - xw * (ev[3] - ev[2]);
        Sig rR = stw_srcp(dR, nk);
        if (nk[0] == 99) {
            nbad = true;
        }
        l = l - R * rR;
    }
    Tc tf = (l - k.sx0) * k.sidx;
    bool okl = stw_cell(tf, k.stlo, k.sthi, cc, tc);
    int il = cc[0];
    sta_basis(tc[0], wb);
    for (int q = 0; q < 4; q++) {
        sv[q] = wb[0] * SV[q][il] + wb[1] * SV[q][il + 1] + wb[2] * SX[q][il] + wb[3] * SX[q][il + 1];
    }
    Sig rxq = stw_srcp(sv[1] - sv[0], nk);
    if (nk[0] == 99) {
        nbad = true;
    }
    Sig xq = (vs - sv[0]) * rxq;
    Sig rl = u - sv[2] - xq * (sv[3] - sv[2]);
    Sig s8 = wb[0] * SV[4][il] + wb[1] * SV[4][il + 1] + wb[2] * SX[4][il] + wb[3] * SX[4][il + 1];
    Sig s9 = wb[0] * SV[5][il] + wb[1] * SV[5][il + 1] + wb[2] * SX[5][il] + wb[3] * SX[5][il + 1];
    Sig sw = s8 + xq * (s9 - s8);
    Sig pw = stw_exp(l, k.l2e, k.l2c);
    Sig pvw = pw * vv;
    Sig hw = u + pvw;
    Sig arl = rl;
    if (rl < zs) {
        arl = zs - rl;
    }
    Sig arg = rg;
    if (rg < zs) {
        arg = zs - rg;
    }
    Sig tolr = scale2<-34>(u);
    Sig tolg = scale2<-30>(vs);
    bool rgok = arg <= tolg;
    Sig xlo = -1.0e-9;
    Sig xhi = 1.000001;
    int f = 0;
    if (wet) {
        if (!(xq >= xlo && (xq <= xhi && (okl && (arl <= tolr && rgok)))) || nbad) {
            f = 1;
        }
    } else {
        if (!okv || inr && (!clear && (!rgok || nbad))) {
            f = 1;
        }
    }
    Sig pp = pd;
    Sig hh = hd;
    Sig ss = sd;
    int iwet = 0;
    if (wet) {
        pp = pw;
        hh = hw;
        ss = sw;
        iwet = 1;
    }
    o[0] = pp;
    o[1] = hh;
    o[2] = vv;
    o[3] = ss;
    o[4] = lg;
    o[5] = l;
    o[6] = rl;
    o[7] = rg;
    fo[0] = f;
    fo[1] = iwet;
}

void steam_w_run(const SteamWCoefs &k, int mode, int la, Tw lw, St Mi, St Ui, Sig lg0, Sig l0, Sig o[8], int fo[2]) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=mode
    #pragma HLS INTERFACE mode=s_axilite port=la
    #pragma HLS INTERFACE mode=s_axilite port=lw
    #pragma HLS INTERFACE mode=s_axilite port=Mi
    #pragma HLS INTERFACE mode=s_axilite port=Ui
    #pragma HLS INTERFACE mode=s_axilite port=lg0
    #pragma HLS INTERFACE mode=s_axilite port=l0
    #pragma HLS INTERFACE mode=s_axilite port=o
    #pragma HLS INTERFACE mode=s_axilite port=fo
    static Tw VF[504];
    static Tw VX[504];
    static Tw VY[504];
    static Tw VZ[504];
    #pragma HLS BIND_STORAGE variable=VF type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VX type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VY type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=VZ type=ram_t2p impl=bram
    if (mode == 1) {
        if (la < 504) {
            VF[la] = lw;
        } else if (la < 1008) {
            VX[la - 504] = lw;
        } else if (la < 1512) {
            VY[la - 1008] = lw;
        } else {
            VZ[la - 1512] = lw;
        }
    } else {
        steam_w_eval(k, VF, VX, VY, VZ, Mi, Ui, lg0, l0, o, fo);
    }
}

