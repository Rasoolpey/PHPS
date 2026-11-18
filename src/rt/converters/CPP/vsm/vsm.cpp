// vsm.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the VSM converter's controller call (PHVsm, sampled); from src/rt/converters/vsm.jl
#include "vsm.h"

Sig vs_rsqrt(Sig a) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=1
    static const Sd seed[4096] = VS_RSQ;
    Rq quarter = 0.25;
    Rq half = 0.5;
    Rq three = 3;
    Sig four = 4;
    Sig sixteen = 16;
    Sig onesig = 1;
    Sig lo = 0.00390625;
    Sig aa = a;
    if (aa < lo) {  // outside the range (flagged by the range check): clamped
        aa = lo;
    }
    Rq m = 0;
    int e = 0;
    if (aa >= sixteen) {
        m = scale2<-6>(aa);
        e = 3;
    } else if (aa >= four) {
        m = scale2<-4>(aa);
        e = 2;
    } else if (aa >= onesig) {
        m = scale2<-2>(aa);
        e = 1;
    } else {
        m = aa;
    }
    for (int i = 0; i < 4; i++) {  // a >= 2^-8: at most four multiplications by 4
        #pragma HLS UNROLL
        if (m < quarter) {
            m = scale2<2>(m);
            e -= 1;
        }
    }
    int ix = 0;
    if (m >= half) {
        ix = ifloor(scale2<12>(m));  // [1/2, 1): 2048 .. 4095
    } else {
        ix = ifloor(scale2<13>(m)) - 2048;  // [1/4, 1/2): 0 .. 2047
    }
    if (ix > 4095) {  // a at the range's top edge (m rounded to 1)
        ix = 4095;
    }
    Rq y = seed[ix];
    for (int it = 0; it < 2; it++) {
        Rq y2 = y * y;
        Rq my2 = m * y2;
        Rq t = three - my2;
        Rq yt = y * t;
        y = scale2<-1>(yt);
    }
    Sig r = y;  // 1 / sqrt(m) in (1, 2]; then 2^-e, shifted from the finer Rq word in
    if (e == 3) {  // one step, so the result is rounded to Sig once (49c)
        r = scale2<-3>(y);
    } else if (e == 2) {
        r = scale2<-2>(y);
    } else if (e == 1) {
        r = scale2<-1>(y);
    } else if (e == -1) {
        r = scale2<1>(y);
    } else if (e == -2) {
        r = scale2<2>(y);
    } else if (e == -3) {
        r = scale2<3>(y);
    } else if (e == -4) {
        r = scale2<4>(y);
    }
    return r;
}

Sig vs_recip0(Sig a) {
    static const Rq seed[1024] = VS_REC;
    Rq half = 0.5;
    Sig onesig = 1;
    Sig lo = 0.03125;
    Sig hi = 63;
    Sig aa = a;
    if (aa < lo) {
        aa = lo;
    }
    if (aa > hi) {
        aa = hi;
    }
    Rq m = 0;
    int k = 0;
    if (aa >= onesig) {  // a < 64: a / 2^6, then up
        m = scale2<-6>(aa);
        k = 6;
    } else {
        m = aa;
    }
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        if (m < half) {
            m = scale2<1>(m);
            k -= 1;
        }
    }
    int j = ifloor(scale2<11>(m)) - 1024;
    Rq y = seed[j];
    Sig r = y;  // 1 / m in (1, 2]; 1 / a = y 2^-k
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        if (k > 0) {
            r = scale2<-1>(r);
            k -= 1;
        }
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        if (k < 0) {
            r = scale2<1>(r);
            k += 1;
        }
    }
    return r;
}

Sig vs_recipv(Sig v) {
    static const Rq seed[1024] = VS_REC;
    Rq two = 2;
    Sig onesig = 1;
    Rq m = v;
    int k = 0;
    if (v >= onesig) {
        m = scale2<-1>(v);
        k = 1;
    }
    int j = ifloor(scale2<11>(m)) - 1024;
    Rq y = seed[j];
    for (int it = 0; it < 2; it++) {
        Rq t = m * y;
        Rq u = two - t;
        y = y * u;
    }
    Sig r = y;
    if (k == 1) {
        r = scale2<-1>(y);
    }
    return r;
}

void vsm_call(St x[3], Turn ph[1], Ang pr[1], Sig u[5], Sig y[2], int fl[4], const VsmCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=x complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    #pragma HLS ALLOCATION operation instances=mul limit=6
    static const Rc cs[1024] = VS_COS;
    static const Rc sn[1024] = VS_SIN;
    Sig one = 1;
    Sig zero = 0;
    Sig half = 0.5;
    Turn phi = ph[0];
    int jb = ifloor(scale2<10>(phi));  // the bin (phi in [0, 1))
    if (jb > 1023) {
        jb = 1023;
    }
    Turn hf = 0.5;
    Turn ft = scale2<10>(phi) - jb;  // the fraction within the bin, [0, 1) (the index never enters Turn)
    Turn fh = ft - hf;
    Turn dl = scale2<-10>(fh);  // the remainder from the bin centre, |dl| <= 1/2048 turn (exact)
    Coef twopi = 6.283185307179586;
    Ang dr = twopi * dl;
    Ang d2 = dr * dr;
    Coef s6 = 0.16666666666666666;
    Coef s20 = 0.05;
    Coef s12 = 0.08333333333333333;
    Ang sa = d2 * s20;
    Rq sb = one - sa;
    Ang sc = d2 * s6;
    Ang sd = sc * sb;
    Rq se = one - sd;
    Ang sdl = dr * se;  // sin(dr) = dr (1 - d2/6 (1 - d2/20))
    Ang ca = d2 * s12;
    Rq cbb = one - ca;
    Ang cc = scale2<-1>(d2);
    Ang cd = cc * cbb;
    Rq cdl = one - cd;  // cos(dr) = 1 - d2/2 (1 - d2/12)
    Rc c0 = cs[jb];
    Rc s0 = sn[jb];
    Sig t1 = c0 * cdl;
    Sig t2 = s0 * sdl;
    Sig t3 = s0 * cdl;
    Sig t4 = c0 * sdl;
    Sig cph = t1 - t2;
    Sig sph = t3 + t4;
    St um = x[2];
    Sig er = um * cph;
    Sig ei = um * sph;
    Sig dre = er - u[0];
    Sig dim = ei - u[1];
    Sig dr2 = dre * dre;
    Sig di2 = dim * dim;
    Sig dd2 = dr2 + di2;
    int on = 0;
    if (k.kf[2] > half && dd2 > k.kg[1]) {
        on = 1;
    }
    Sig zr = k.kz[0];  // Z = (zr, zx) at the root (engaged) or Z0
    Sig zx = k.kz[1];
    Sig ipr = 0;
    Sig ipi = 0;
    if (on == 1) {
        Sig rd = vs_rsqrt(dd2);  // 1 / |D|
        Sig dm = dd2 * rd;  // |D|
        Sig gm = dm - k.kc[3];  // |D| - c0' i_lim > 0
        Sig q4 = k.kc[2] * gm;
        Sig ds = k.kc[1] + q4;  // beta^2 + 4 c1 gamma
        Sig rs = vs_rsqrt(ds);
        Sig sq = ds * rs;
        Sig sbt = sq - k.kc[0];
        Sig ov = sbt * k.kc[4];  // ov0 >= ov*
        Sig ii = k.kg[0] + ov;
        Sig pkr = k.kz[4] * ov;
        Sig px = k.kz[5] * ov;
        Sig re = k.kz[0] + pkr;
        Sig xe = k.kz[1] + px;
        Sig aa = ii * re;
        Sig ww = ii * xe;
        Sig ikr = ii * k.kz[4];
        Sig ikx = ii * k.kz[5];
        Sig gr = re + ikr;
        Sig gx = xe + ikx;
        Sig f1 = aa * gr;
        Sig f2 = ww * gx;
        Sig fs = f1 + f2;
        Sig sl = scale2<1>(fs);
        Sig g = vs_recip0(sl);
        for (int it = 0; it < 9; it++) {  // VSM_NCHORD chord steps
            Sig ic = k.kg[0] + ov;
            Sig cr = k.kz[4] * ov;
            Sig cx = k.kz[5] * ov;
            Sig rr = k.kz[0] + cr;
            Sig xx = k.kz[1] + cx;
            Sig ua = ic * rr;
            Sig uw = ic * xx;
            Sig ua2 = ua * ua;
            Sig uw2 = uw * uw;
            Sig fa = ua2 + uw2;
            Sig fv = fa - dd2;
            Sig st = g * fv;
            ov = ov - st;
        }
        Sig ir = k.kg[0] + ov;  // the root i, and Z(i)
        Sig qr = k.kz[4] * ov;
        Sig qx = k.kz[5] * ov;
        zr = k.kz[0] + qr;
        zx = k.kz[1] + qx;
        Sig dhr = dre * rd;  // D / |D|
        Sig dhi = dim * rd;
        Sig a2 = dhr * ir;  // D i / |D| ...
        Sig a3 = dhi * ir;
        Sig zir0 = zr * ir;  // ... times conj(Z) i / |D| = conj(Z) / |Z| at the root
        Sig zix0 = zx * ir;
        Sig b1 = zir0 * rd;
        Sig b2 = zix0 * rd;
        Sig m1 = a2 * b1;
        Sig m2 = a3 * b2;
        Sig m3 = a3 * b1;
        Sig m4 = a2 * b2;
        ipr = m1 + m2;
        ipi = m3 - m4;
    } else {
        Sig n1 = dre * k.kg[2];  // ip = D / Z0 = D y0
        Sig n2 = dim * k.kg[3];
        Sig n3 = dim * k.kg[2];
        Sig n4 = dre * k.kg[3];
        ipr = n1 - n2;
        ipi = n3 + n4;
    }
    Sig zex = zx - k.kz[2];  // ze = Z - j Zs
    Sig w1 = zr * ipr;
    Sig w2 = zex * ipi;
    Sig w3 = zr * ipi;
    Sig w4 = zex * ipr;
    Sig zir = w1 - w2;
    Sig zii = w3 + w4;
    Sig ucr = er - zir;
    Sig uci = ei - zii;
    Sig ur2 = ucr * ucr;
    Sig ui2 = uci * uci;
    Sig uu2 = ur2 + ui2;
    Sig vd = u[4];
    int fdc = 0;
    if (vd < k.kg[10]) {
        vd = k.kg[10];
        fdc = 1;
    }
    if (vd > k.kg[11]) {
        vd = k.kg[11];
        fdc = 1;
    }
    int cu = 0;
    if (k.kf[0] > half && uu2 > k.kg[7]) {
        cu = 1;
    }
    Sig up2 = uu2;  // |u_c'|^2 after the u_max clip
    if (cu == 1) {
        up2 = k.kg[7];
    }
    Sig vd2 = vd * vd;
    Sig lim2 = k.kg[9] * vd2;  // (m_max v_dc)^2
    int cm = 0;
    if (k.kf[1] > half && up2 > lim2) {
        cm = 1;
    }
    Sig ru = vs_rsqrt(uu2);  // 1 / |u_c| (used only when a clip acts)
    Sig gv = vs_recipv(vd);  // 1 / v_dc
    Sig fac = one;  // the held quantity = u_c fac
    if (k.kf[1] > half) {
        if (cm == 1) {
            fac = k.kg[8] * ru;  // m = m_max u_c / |u_c|
        } else if (cu == 1) {
            Sig fu = k.kg[6] * ru;
            fac = fu * gv;  // m = u_max (u_c / |u_c|) / v_dc
        } else {
            fac = gv;  // m = u_c / v_dc
        }
    } else {
        if (cu == 1) {
            fac = k.kg[6] * ru;
        }
    }
    y[0] = ucr * fac;
    y[1] = uci * fac;
    fl[0] = on;
    fl[1] = cu;
    fl[2] = cm;
    fl[3] = fdc;
    Sig vr = u[0] * u[2];
    Sig vi = u[1] * u[3];
    Sig p = vr + vi;
    St om = x[0];
    St of = x[1];
    Sig e1 = k.kx[0] - p;
    St e2 = of - one;
    St e3 = om - of;
    St e4 = k.kx[1] - um;
    St g1 = k.ka[0] * e1;
    St g2 = k.ks[1] * e2;
    St g3 = k.ks[2] * e3;
    St g4 = k.ks[3] * e4;
    x[0] = om + g1 - g2;
    x[1] = of + g3;
    x[2] = um + g4;
    Ang inc = k.ks[4] * om + k.kt[0] * om + pr[0];
    Turn pn = phi + inc;
    Ang dd = pn - phi;
    pr[0] = inc - dd;
    if (pn >= one) {
        pn = pn - one;
    }
    if (pn < zero) {
        pn = pn + one;
    }
    ph[0] = pn;
}

void vsm_run(const VsmCoefs &k, St x[3], Turn ph[1], Ang pr[1], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=ph
    #pragma HLS INTERFACE mode=s_axilite port=pr
    #pragma HLS INTERFACE mode=s_axilite port=u
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=s_axilite port=fo
    #pragma HLS INTERFACE mode=s_axilite port=ncall
    #pragma HLS INTERFACE mode=ap_none port=m_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    St xs[3];
    #pragma HLS ARRAY_PARTITION variable=xs complete
    Turn phs[1];
    Ang prs[1];
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    phs[0] = ph[0];
    prs[0] = pr[0];
    Sig uu[5];
    #pragma HLS ARRAY_PARTITION variable=uu complete
    Sig yy[2];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int ff[4];
    #pragma HLS ARRAY_PARTITION variable=ff complete
    bool st = false;
    for (int c = 0; c < ncall; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        sample = false;
        for (int i = 0; i < 5; i++) {
            uu[i] = u[c][i];
        }
        vsm_call(xs, phs, prs, uu, yy, ff, k);
        y[c][0] = yy[0];
        y[c][1] = yy[1];
        fo[c] = ff[0] + 2 * ff[1] + 4 * ff[2] + 8 * ff[3];
        m_out = yy[0];
        st = !st;
        strobe = st;
        sample = true;
    }
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
    ph[0] = phs[0];
    pr[0] = prs[0];
}

