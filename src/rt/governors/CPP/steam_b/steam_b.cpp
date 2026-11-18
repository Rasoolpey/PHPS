// steam_b.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body B of the steam unit (11 sections at II 10); src/rt/governors/steam.jl
#include "steam_b.h"

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

int stb_nq(Dq z, Rq mo[1]) {
    #pragma HLS INLINE
    Dq m = z;
    int n = 0;
    Dq one = 1;
    Dq e16 = 1.52587890625e-5;
    Dq e8 = 0.00390625;
    Dq e4 = 0.0625;
    if (m >= one) {
        m = scale2<-4>(m);
        n = -1;
    }
    if (m < e16) {
        m = scale2<16>(m);
        n = n + 4;
    }
    if (m < e8) {
        m = scale2<8>(m);
        n = n + 2;
    }
    if (m < e4) {
        m = scale2<4>(m);
        n = n + 1;
    }
    Rq r = m;
    Rq hq = 0.5;
    if (m < e4) {
        n = 99;
        r = hq;
    }
    mo[0] = r;
    return n;
}

int stb_nr(Dq x, Rq mo[1]) {
    #pragma HLS INLINE
    Dq m = x;
    int n = 0;
    Dq one = 1;
    Dq e8 = 0.00390625;
    Dq e4 = 0.0625;
    Dq e2 = 0.25;
    if (m >= one) {
        m = scale2<-2>(m);
        n = -1;
    }
    if (m < e8) {
        m = scale2<8>(m);
        n = n + 4;
    }
    if (m < e4) {
        m = scale2<4>(m);
        n = n + 2;
    }
    if (m < e2) {
        m = scale2<2>(m);
        n = n + 1;
    }
    Rq r = m;
    Rq hq = 0.5;
    if (m < e2) {
        n = 99;
        r = hq;
    }
    mo[0] = r;
    return n;
}

int stb_jq(Rq m) {
    #pragma HLS INLINE
    Rq half = 0.5;
    Rq quarter = 0.25;
    Rq eighth = 0.125;
    int j = 0;
    if (m >= half) {
        j = ifloor(scale2<11>(m)) + 2048;
    } else if (m >= quarter) {
        j = ifloor(scale2<12>(m)) + 1024;
    } else if (m >= eighth) {
        j = ifloor(scale2<13>(m));
    } else {
        j = ifloor(scale2<14>(m)) - 1024;
    }
    if (j < 0) {
        j = 0;
    }
    if (j > 4095) {
        j = 4095;
    }
    return j;
}

int stb_jr(Rq m) {
    #pragma HLS INLINE
    Rq half = 0.5;
    int j = 0;
    if (m >= half) {
        j = ifloor(scale2<11>(m));
    } else {
        j = ifloor(scale2<12>(m)) - 1024;
    }
    if (j < 0) {
        j = 0;
    }
    if (j > 2047) {
        j = 2047;
    }
    return j;
}

Rq stb_qrt(Rq m, Rq y0) {
    #pragma HLS INLINE
    Rq five = 5;
    Rq y = y0;
    for (int i = 0; i < 2; i++) {
        Rq y2 = y * y;
        Rq a = m * y2;
        Rq b = a * y2;
        Rq t = five - b;
        Rq yt = y * t;
        y = scale2<-2>(yt);
    }
    return y;
}

Rq stb_rsq(Rq m, Rq y0) {
    #pragma HLS INLINE
    Rq three = 3;
    Rq y = y0;
    for (int i = 0; i < 2; i++) {
        Rq y2 = y * y;
        Rq a = m * y2;
        Rq t = three - a;
        Rq yt = y * t;
        y = scale2<-1>(yt);
    }
    return y;
}

Dq stb_ps(Sig x, int e) {
    #pragma HLS INLINE
    Dq r = x;
    if (e == 6) {
        r = scale2<-6>(x);
    } else if (e == 5) {
        r = scale2<-5>(x);
    } else if (e == 4) {
        r = scale2<-4>(x);
    } else if (e == 3) {
        r = scale2<-3>(x);
    } else if (e == 2) {
        r = scale2<-2>(x);
    } else if (e == 1) {
        r = scale2<-1>(x);
    } else if (e == -1) {
        r = scale2<1>(x);
    } else if (e == -2) {
        r = scale2<2>(x);
    } else if (e == -3) {
        r = scale2<3>(x);
    } else if (e == -4) {
        r = scale2<4>(x);
    } else if (e == -5) {
        r = scale2<5>(x);
    } else if (e == -6) {
        r = scale2<6>(x);
    } else if (e == -7) {
        r = scale2<7>(x);
    }
    return r;
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

Tw stb_v2(Tw F[2717], Tw X[2717], Tw Y[2717], Tw Z[2717], int b, Wt a0, Wt a1, Wt c0, Wt c1, Wt b0, Wt b1, Wt d0, Wt d1) {
    #pragma HLS INLINE
    Tw s1 = a0 * F[b] + a1 * F[b + 13] + c0 * X[b] + c1 * X[b + 13];
    Tw s2 = a0 * Y[b] + a1 * Y[b + 13] + c0 * Z[b] + c1 * Z[b + 13];
    Tw t1 = a0 * F[b + 1] + a1 * F[b + 14] + c0 * X[b + 1] + c1 * X[b + 14];
    Tw t2 = a0 * Y[b + 1] + a1 * Y[b + 14] + c0 * Z[b + 1] + c1 * Z[b + 14];
    Tw r = b0 * s1 + d0 * s2 + b1 * t1 + d1 * t2;
    return r;
}

void steam_b_run(const SteamBCoefs &k, int mode, int la, Tw lw, Sig p[13], Sig hh[13], Sig v[13], Sig s[13], Sig q[11], Sig e1[11], Sig e2[11], int fw[11], int fl[11]) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=mode
    #pragma HLS INTERFACE mode=s_axilite port=la
    #pragma HLS INTERFACE mode=s_axilite port=lw
    #pragma HLS INTERFACE mode=s_axilite port=p
    #pragma HLS INTERFACE mode=s_axilite port=hh
    #pragma HLS INTERFACE mode=s_axilite port=v
    #pragma HLS INTERFACE mode=s_axilite port=s
    #pragma HLS INTERFACE mode=s_axilite port=q
    #pragma HLS INTERFACE mode=s_axilite port=e1
    #pragma HLS INTERFACE mode=s_axilite port=e2
    #pragma HLS INTERFACE mode=s_axilite port=fw
    #pragma HLS INTERFACE mode=s_axilite port=fl
    static Tw IF[2717];
    static Tw IX[2717];
    static Tw IY[2717];
    static Tw IZ[2717];
    #pragma HLS BIND_STORAGE variable=IF type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=IX type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=IY type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=IZ type=ram_t2p impl=bram
    static int cia[11];
    static int cib[11];
    static int ctm[11];
    static int cwo[11];
    static int cep[11];
    static Sig cpm[11];
    static Cf ckq[11];
    static Dq cd2[11];
    static Wt cet[11];
    static int cllo[11];
    static int clhi[11];
    static Cf clflo[11];
    static Cf clfhi[11];
    static Cf cltlo[11];
    static Cf clthi[11];
    static Wt clcl[11];
    static Wt clch[11];
    static int celo[11];
    static int cehi[11];
    static Cf ceflo[11];
    static Cf cefhi[11];
    static Cf cetlo[11];
    static Cf cethi[11];
    static Wt cecl[11];
    static Wt cech[11];
    static const Rq LGI[1024] = STA_LGI;
    static const Lw LGL[1024] = STA_LGL;
    static const Cf KLB[39] = STB_KLN;
    static const Sd QS[4096] = STB_QS;
    static const Sd RS[2048] = STB_RS;
    static const Tw SV[5][121] = STB_SV;
    static const Tw SD[5][121] = STB_SD;
    #pragma HLS BIND_STORAGE variable=LGI type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=LGL type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=QS type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=RS type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SV type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SD type=rom_2p impl=bram
    if (mode == 1) {
        if (la < 2717) {
            IF[la] = lw;
        } else if (la < 5434) {
            IX[la - 2717] = lw;
        } else if (la < 8151) {
            IY[la - 5434] = lw;
        } else {
            IZ[la - 8151] = lw;
        }
    } else if (mode == 2) {
        for (int i = 0; i < 11; i++) {
            cia[i] = k.ia[i];
            cib[i] = k.ib[i];
            ctm[i] = k.tm[i];
            cwo[i] = k.wo[i];
            cep[i] = k.ep[i];
            cpm[i] = k.pm[i];
            ckq[i] = k.kq[i];
            cd2[i] = k.d2[i];
            cet[i] = k.et[i];
            cllo[i] = k.llo[i];
            clhi[i] = k.lhi[i];
            clflo[i] = k.lflo[i];
            clfhi[i] = k.lfhi[i];
            cltlo[i] = k.ltlo[i];
            clthi[i] = k.lthi[i];
            clcl[i] = k.lcl[i];
            clch[i] = k.lch[i];
            celo[i] = k.elo[i];
            cehi[i] = k.ehi[i];
            ceflo[i] = k.eflo[i];
            cefhi[i] = k.efhi[i];
            cetlo[i] = k.etlo[i];
            cethi[i] = k.ethi[i];
            cecl[i] = k.ecl[i];
            cech[i] = k.ech[i];
        }
    } else {
        Rq one = 1;
        Rq third = 0.3333333333333333;
        Rq half = 0.5;
        Dq zq = 0;
        Sig zs = 0;
        Sig pc = k.pc;
        Sig lpc = k.lpc;
        Cf sx0 = k.sx0;
        Cf sidx = k.sidx;
        Cf ey0 = k.ey0;
        Cf eidy = k.eidy;
        for (int it = 0; it < 11; it++) {
            #pragma HLS PIPELINE II=10
            #pragma HLS ALLOCATION operation instances=mul limit=8
            int ia = cia[it];
            int ib = cib[it];
            bool tm = ctm[it] == 1;
            Sig pin = p[ia];
            Sig pb = p[ib];
            Sig va = v[ia];
            Sig vb = v[ib];
            Sig ha = hh[ia];
            Sig hbv = hh[ib];
            Sig sa = s[ia];
            Sig pob = pb;
            if (tm) {
                pob = pc;
            }
            Dq pti = stb_ps(pin, cep[it]);
            Dq pto = stb_ps(pob, cep[it]);
            Dq dd = pti * pti - pto * pto;
            bool fwd = dd >= zq;
            Dq z = dd * dd + cd2[it];
            Rq mz[1];
            int nz = stb_nq(z, mz);
            int jz = stb_jq(mz[0]);
            Rq sz = QS[jz];
            Rq y4 = stb_qrt(mz[0], sz);
            Sig pd = pin;
            Sig vd = va;
            if (!fwd && !tm) {
                pd = pb;
                vd = vb;
            }
            Dq x = pd * vd;
            Rq mx[1];
            int nx = stb_nr(x, mx);
            int jx = stb_jr(mx[0]);
            Rq sx = RS[jx];
            Rq y2 = stb_rsq(mx[0], sx);
            Dq t1 = ckq[it] * dd;
            Dq t2 = t1 * y4;
            Dq t3 = t2 * y2;
            Sig qq = stb_sh(t3, nz + nx);
            if (!fwd && tm) {
                qq = zs;
            }
            Rq ml[1];
            int kp = stb_norm(pob, ml);
            Rq m = ml[0];
            int j = sta_j(m);
            Rq w = LGI[j];
            Rq e = m * w;
            Rs r = e - one;
            Rq q1 = third - scale2<-2>(r);
            Rq q2 = r * q1 - half;
            Rq q3 = r * q2 + one;
            Rs lg = r * q3;
            int kl = kp;
            if (kp == 99) {
                kl = 0;
            }
            Sig l = lg + LGL[j] + KLB[kl + 31];
            if (tm) {
                l = lpc;
            }
            Tc tl = (l - sx0) * sidx;
            int ci[1];
            Wt ct[1];
            bool okl = sta_cell(tl, cllo[it], clhi[it], clflo[it], clfhi[it], cltlo[it], clthi[it], clcl[it], clch[it], ci, ct);
            int li = ci[0];
            int il = li + cllo[it];
            Wt wl[4];
            sta_basis(ct[0], wl);
            Sig tt = wl[0] * SV[0][il] + wl[1] * SV[0][il + 1] + wl[2] * SD[0][il] + wl[3] * SD[0][il + 1];
            Sig hf = wl[0] * SV[1][il] + wl[1] * SV[1][il + 1] + wl[2] * SD[1][il] + wl[3] * SD[1][il + 1];
            Sig sf = wl[0] * SV[2][il] + wl[1] * SV[2][il + 1] + wl[2] * SD[2][il] + wl[3] * SD[2][il + 1];
            Sig sg = wl[0] * SV[3][il] + wl[1] * SV[3][il + 1] + wl[2] * SD[3][il] + wl[3] * SD[3][il + 1];
            Sig stp = wl[0] * SV[4][il] + wl[1] * SV[4][il + 1] + wl[2] * SD[4][il] + wl[3] * SD[4][il + 1];
            bool dry = sa >= sg;
            Sig ds = stp - sg;
            Rq md[1];
            int kd = stb_norm(ds, md);
            Rq m2 = md[0];
            int j2 = sta_j(m2);
            Rq w2 = LGI[j2];
            Rq ee2 = m2 * w2;
            Rq yd = sta_rcp(m2, w2, ee2);
            Sig dsa = sa - sg;
            Sig ey = dsa * yd;
            Sig et = stb_sh(ey, 0 - kd);
            Tc te = (et - ey0) * eidy;
            int cj[1];
            Wt cu[1];
            bool oke = sta_cell(te, celo[it], cehi[it], ceflo[it], cefhi[it], cetlo[it], cethi[it], cecl[it], cech[it], cj, cu);
            int le = cj[0];
            Wt we[4];
            sta_basis(cu[0], we);
            int bi = ((it + 1) - 1) * 247 + li * 13 + le;
            Sig hsd = stb_v2(IF, IX, IY, IZ, bi, wl[0], wl[1], wl[2], wl[3], we[0], we[1], we[2], we[3]);
            Sig hsw = hf + tt * (sa - sf);
            Sig hs = hsd;
            if (!dry && cwo[it] == 1) {
                hs = hsw;
            }
            Sig hout = ha - cet[it] * (ha - hs);
            Sig hb = hbv;
            if (tm) {
                hb = ha;
            }
            Sig h1 = hb;
            Sig h2 = hb;
            if (fwd) {
                h1 = ha;
                h2 = hout;
            }
            q[it] = qq;
            e1[it] = qq * h1;
            e2[it] = qq * h2;
            int f = 0;
            if (fwd) {
                fw[it] = 1;
            } else {
                fw[it] = 0;
            }
            if (nz == 99 || (nx == 99 || (pin >= cpm[it] || pob >= cpm[it]))) {
                f = 1;
            }
            if (fwd && qq != zs) {
                if (!okl || kp == 99) {
                    f = 1;
                }
                if (dry && (!oke || kd == 99)) {
                    f = 1;
                }
                if (!dry && (cwo[it] == 0 || sa < sf)) {
                    f = 1;
                }
            }
            fl[it] = f;
        }
    }
}

