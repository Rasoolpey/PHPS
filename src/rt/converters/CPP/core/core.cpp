// core.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the saturating transformer core (PHCore): the discrete-gradient currents; from src/rt/converters/core.jl
#include "core.h"

Wx cr_shift(Wx x, int m) {
    #pragma HLS INLINE
    Wx v = x;
    int r = m;
    if (r >= 16) {
        v = scale2<16>(v);
        r -= 16;
    }
    if (r >= 8) {
        v = scale2<8>(v);
        r -= 8;
    }
    if (r >= 4) {
        v = scale2<4>(v);
        r -= 4;
    }
    if (r >= 2) {
        v = scale2<2>(v);
        r -= 2;
    }
    if (r >= 1) {
        v = scale2<1>(v);
        r -= 1;
    }
    if (r <= -16) {
        v = scale2<-16>(v);
        r += 16;
    }
    if (r <= -8) {
        v = scale2<-8>(v);
        r += 8;
    }
    if (r <= -4) {
        v = scale2<-4>(v);
        r += 4;
    }
    if (r <= -2) {
        v = scale2<-2>(v);
        r += 2;
    }
    if (r <= -1) {
        v = scale2<-1>(v);
        r += 1;
    }
    return v;
}

void core_solve(Fx psi[4], Rc ce[6], Fx lo[4], Fx am[3], int sg[3], const CoreCoefs &k) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=4
    static const Rq seed[1024] = VS_REC;
    Rq two = 2;
    Rq twothirds = 0.6666666666666666;
    Fx zero = 0;
    Fx llr = k.kc[0] * psi[0] + k.kc[1] * psi[2];  // lambda_l
    Fx lli = k.kc[0] * psi[1] + k.kc[1] * psi[3];
    Sig br = k.kl[0] * psi[0] + k.kl[1] * psi[2];  // b = psi_H / L2 + psi_S / L1
    Sig bi = k.kl[0] * psi[1] + k.kl[1] * psi[3];
    Fx lr = llr;
    Fx li = lli;
    for (int it = 0; it < 6; it++) {  // CORE_NIT Newton steps
        Sig imr = 0;
        Sig imi = 0;
        Coef k11 = k.kl[2];
        Coef k22 = k.kl[2];
        Coef k12 = 0;
        for (int p = 0; p < 3; p++) {
            #pragma HLS UNROLL
            Fx lk = lr * ce[2 * p] - li * ce[2 * p + 1];
            Fx ak = lk;
            if (lk < zero) {
                ak = zero - lk;
            }
            int s = 1;
            for (int j = 1; j < 9; j++) {
                #pragma HLS UNROLL
                if (ak >= k.lam[j]) {
                    s = (j + 1);
                }
            }
            Sig fk = k.cur[s - 1] + k.sl[s - 1] * (ak - k.lam[s - 1]);
            if (lk < zero) {
                fk = zero - fk;
            }
            imr += twothirds * fk * ce[2 * p];
            imi -= twothirds * fk * ce[2 * p + 1];
            Sig ssl = twothirds * k.sl[s - 1];
            k11 += ssl * ce[2 * p] * ce[2 * p];
            k22 += ssl * ce[2 * p + 1] * ce[2 * p + 1];
            k12 -= ssl * ce[2 * p] * ce[2 * p + 1];
            sg[p] = s;
            am[p] = ak;
        }
        Sig rr = k.kl[2] * lr - br + imr;
        Sig ri = k.kl[2] * li - bi + imi;
        Kd det = k11 * k22 - k12 * k12;
        int q = 10;
        Kd t = 1024;
        for (int j = 0; j < 8; j++) {
            #pragma HLS UNROLL
            if (det >= t) {
                q += 1;
            }
            t = scale2<1>(t);
        }
        Wx dw = det;
        Rq mn = cr_shift(dw, -q);
        int ix = ifloor(scale2<11>(mn)) - 1024;
        if (ix > 1023) {
            ix = 1023;
        }
        if (ix < 0) {
            ix = 0;
        }
        Rq y = seed[ix];
        for (int nt = 0; nt < 2; nt++) {
            Rq ty = mn * y;
            Rq uy = two - ty;
            y = y * uy;
        }
        Sig ar = k22 * rr - k12 * ri;  // adj(K) r
        Sig ai = k11 * ri - k12 * rr;
        Wx ary = ar * y;
        Wx aiy = ai * y;
        Fx dr = cr_shift(ary, -q);
        Fx di = cr_shift(aiy, -q);
        lr = lr - dr;
        li = li - di;
    }
    lo[0] = lr;
    lo[1] = li;
    lo[2] = lr - llr;
    lo[3] = li - lli;
}

void core_unit(Fx z[8], Turn th[1], Sig y[4], int fl[2], const CoreCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=z complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    #pragma HLS ALLOCATION operation instances=mul limit=6
    #pragma HLS ALLOCATION function instances=core_solve limit=1
    static const Rc cs[1024] = VS_COS;
    static const Rc sn[1024] = VS_SIN;
    static const Rq seed[1024] = VS_REC;
    Sig one = 1;
    Rq two = 2;
    Fx zero = 0;
    Rq half = 0.5;
    Rq r3 = 0.8660254037844386;
    Turn phi = th[0];
    int jb = ifloor(scale2<10>(phi));
    if (jb > 1023) {
        jb = 1023;
    }
    Turn hf = 0.5;
    Turn ft = scale2<10>(phi) - jb;
    Turn fh = ft - hf;
    Turn dl = scale2<-10>(fh);
    Coef twopi = 6.283185307179586;
    Ang dra = twopi * dl;
    Ang d2 = dra * dra;
    Coef s6 = 0.16666666666666666;
    Coef s20 = 0.05;
    Coef s12 = 0.08333333333333333;
    Ang sa = d2 * s20;
    Rq sb = one - sa;
    Ang sc = d2 * s6;
    Ang sd = sc * sb;
    Rq se = one - sd;
    Ang sdl = dra * se;
    Ang ca = d2 * s12;
    Rq cbb = one - ca;
    Ang cc = scale2<-1>(d2);
    Ang cd = cc * cbb;
    Rq cdl = one - cd;
    Rc c0 = cs[jb];
    Rc s0 = sn[jb];
    Rc ce[6];
    #pragma HLS ARRAY_PARTITION variable=ce complete
    ce[0] = c0 * cdl - s0 * sdl;  // cos theta_m
    ce[1] = s0 * cdl + c0 * sdl;  // sin theta_m
    ce[2] = r3 * ce[1] - half * ce[0];  // cos(theta_m - 2 pi / 3)
    ce[3] = zero - half * ce[1] - r3 * ce[0];  // sin(theta_m - 2 pi / 3)
    ce[4] = zero - half * ce[0] - r3 * ce[1];  // cos(theta_m + 2 pi / 3)
    ce[5] = r3 * ce[0] - half * ce[1];  // sin(theta_m + 2 pi / 3)
    Fx pm[4];
    #pragma HLS ARRAY_PARTITION variable=pm complete
    Fx p0[4];
    #pragma HLS ARRAY_PARTITION variable=p0 complete
    Fx p1[4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        p0[i] = z[i];
        p1[i] = z[i + 4];
        pm[i] = scale2<-1>(z[i] + z[i + 4]);
    }
    Fx om[4];
    #pragma HLS ARRAY_PARTITION variable=om complete
    Fx o0[4];
    #pragma HLS ARRAY_PARTITION variable=o0 complete
    Fx o1[4];
    #pragma HLS ARRAY_PARTITION variable=o1 complete
    Fx am[3];
    Fx a0[3];
    Fx a1[3];
    int sm[3];
    int s0g[3];
    int s1g[3];
    core_solve(pm, ce, om, am, sm, k);
    core_solve(p0, ce, o0, a0, s0g, k);
    core_solve(p1, ce, o1, a1, s1g, k);
    Sig ihr = k.kl[0] * (pm[0] - om[0]);
    Sig ihi = k.kl[0] * (pm[1] - om[1]);
    Sig isr = k.kl[1] * (om[0] - pm[2]);
    Sig isi = k.kl[1] * (om[1] - pm[3]);
    int diff = 0;
    int multi = 0;
    for (int p = 0; p < 3; p++) {
        #pragma HLS UNROLL
        if (s0g[p] != s1g[p] || s0g[p] != sm[p]) {
            diff = 1;
        }
        int dsg = s1g[p] - s0g[p];
        if (dsg > 1 || dsg < -1) {
            multi = 1;
        }
    }
    fl[0] = diff;
    fl[1] = multi;
    if (diff == 1) {
        Fx dz[4];
        #pragma HLS ARRAY_PARTITION variable=dz complete
        Fx mx = 0;
        for (int i = 0; i < 4; i++) {
            #pragma HLS UNROLL
            dz[i] = z[i + 4] - z[i];
            Fx ad = dz[i];
            if (ad < zero) {
                ad = zero - ad;
            }
            if (ad > mx) {
                mx = ad;
            }
        }
        int kd = -23;  // |dz| in [2^-24, 1): kd in -23 .. 0
        Coef t = 1.1920928955078125e-7;  // thresholds 2^-23 .. 2^-1
        for (int j = 0; j < 23; j++) {
            #pragma HLS UNROLL
            if (mx >= t) {
                kd += 1;
            }
            t = scale2<1>(t);
        }
        Rn nz[4];
        #pragma HLS ARRAY_PARTITION variable=nz complete
        for (int i = 0; i < 4; i++) {
            #pragma HLS UNROLL
            Wx dzx = dz[i];
            nz[i] = cr_shift(dzx, 0 - kd);  // dz 2^-kd (exact)
        }
        Ry nn = nz[0] * nz[0] + nz[1] * nz[1] + nz[2] * nz[2] + nz[3] * nz[3];  // in [1/4, 4)
        int e4 = 0;
        Rq mm = scale2<-2>(nn);
        for (int j = 0; j < 4; j++) {
            #pragma HLS UNROLL
            if (mm < half) {
                mm = scale2<1>(mm);
                e4 += 1;
            }
        }
        int jx = ifloor(scale2<11>(mm)) - 1024;
        if (jx > 1023) {
            jx = 1023;
        }
        Rq yr = seed[jx];
        for (int nt = 0; nt < 2; nt++) {
            Rq tr = mm * yr;
            Rq ur = two - tr;
            yr = yr * ur;
        }
        Wx yrx = yr;
        Coef rn = cr_shift(yrx, e4 - 2);  // 1 / nn = y 2^e4 / 4 (about 4: Coef)
        Wx pkx = (o1[2] - o0[2]) * (o1[2] + o0[2]) + (o1[3] - o0[3]) * (o1[3] + o0[3]);  // exact in Wx
        Wd pks = cr_shift(pkx, 0 - kd);
        Wd nk = k.kl[4] * pks;
        Wx gax = k.kl[0] * dz[0] + k.kl[1] * dz[2];
        Wx gbx = k.kl[0] * dz[1] + k.kl[1] * dz[3];
        Wd gas = cr_shift(gax, 0 - kd);
        Wd gbs = cr_shift(gbx, 0 - kd);
        Wd gt = om[2] * gas + om[3] * gbs;
        Wx nwx = 0;
        for (int p = 0; p < 3; p++) {
            #pragma HLS UNROLL
            Fx lo_ = a0[p];
            Fx hi_ = a1[p];
            int sgn = 1;
            if (a1[p] < a0[p]) {
                lo_ = a1[p];
                hi_ = a0[p];
                sgn = -1;
            }
            Wx wp = 0;
            for (int j = 1; j < 9; j++) {  // segment 1 has W_nl = 0
                #pragma HLS UNROLL
                Fx ja = lo_;
                if (ja < k.lam[j]) {
                    ja = k.lam[j];
                }
                if (ja > k.lam[j + 1]) {
                    ja = k.lam[j + 1];
                }
                Fx jbb = hi_;
                if (jbb < k.lam[j]) {
                    jbb = k.lam[j];
                }
                if (jbb > k.lam[j + 1]) {
                    jbb = k.lam[j + 1];
                }
                wp += (jbb - ja) * (k.g1[j] + k.g2[j] * (ja - k.lam[j] + (jbb - k.lam[j])));  // exact in Wx
            }
            if (sgn < 0) {
                wp = zero - wp;
            }
            nwx += wp;
        }
        Wd nw = cr_shift(nwx, 0 - kd);
        Rq twothirds = 0.6666666666666666;
        Wd num = nk + twothirds * nw + gt;  // [g(z1) - g(z0) - grad g(z_m) . dz] 2^-kd (gt: - grad_nl . dz)
        Sig cg = num * rn;  // c dz = cg n
        ihr = ihr + cg * nz[0];
        ihi = ihi + cg * nz[1];
        isr = isr - cg * nz[2];
        isi = isi - cg * nz[3];
    }
    y[0] = ihr;
    y[1] = ihi;
    y[2] = isr;
    y[3] = isi;
}

void core_run(const CoreCoefs &k, Fx z[256][8], Turn th[256], Sig y[256][4], int fo[256], int n, int decim, volatile Sig &i_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=z
    #pragma HLS INTERFACE mode=s_axilite port=th
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=s_axilite port=fo
    #pragma HLS INTERFACE mode=s_axilite port=n
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=i_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    Fx zz[8];
    #pragma HLS ARRAY_PARTITION variable=zz complete
    Turn tt[1];
    Sig yy[4];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int ff[2];
    #pragma HLS ARRAY_PARTITION variable=ff complete
    bool st = false;
    int cnt = 0;
    for (int s = 0; s < n; s++) {
        #pragma HLS PIPELINE II=64
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        for (int i = 0; i < 8; i++) {
            zz[i] = z[s][i];
        }
        tt[0] = th[s];
        core_unit(zz, tt, yy, ff, k);
        for (int i = 0; i < 4; i++) {
            y[s][i] = yy[i];
        }
        fo[s] = ff[0] + 2 * ff[1];
        i_out = yy[0];
        st = !st;
        strobe = st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
    }
}

