// voc.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the VOC converter's controller call (PHVoc, sampled, DC rejection); from src/rt/converters/voc.jl
#include "voc.h"

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

void dr_mag(Sig xr, Sig xi, Rn on[2], Ry oy[2], int kk[1]) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=1
    static const Sd seed[4096] = VS_RSQ;
    static const Coef pinv[14] = DR_PINV;
    Ry quarter = 0.25;
    Ry half = 0.5;
    Ry onen = 1;
    Rq three = 3;
    Sig ar = xr;
    if (ar < 0) {
        ar = -ar;
    }
    Sig ai = xi;
    if (ai < 0) {
        ai = -ai;
    }
    Sig mx = ar;
    if (ai > ar) {
        mx = ai;
    }
    int k = -6;
    Coef t = 0.015625;  // thresholds 2^-6 .. 2^6 (Coef: the last doubling reaches 128)
    for (int i = 0; i < 13; i++) {
        #pragma HLS UNROLL
        if (mx >= t) {
            k += 1;
        }
        t = scale2<1>(t);
    }
    Coef ps = pinv[k + 6];
    Rn nr = xr * ps;
    Rn ni = xi * ps;
    Ry q1 = nr * nr;  // q in Ry <48,3>: |n| components < 1, but n may round to 1 at |x| ~ 128
    Ry q2 = ni * ni;
    Ry q0 = q1 + q2;
    Ry qv = q0;
    if (qv < quarter) {  // only when |x| < 2^-7 (clamped for the seed; the callers' decisions are
        qv = quarter;  // off there: thresholds >= 0.15)
    }
    int ix = 0;
    int hi = 0;
    if (qv >= onen) {  // [1, 2): the seed of q / 4, halved
        ix = ifloor(scale2<11>(qv)) - 2048;
        hi = 1;
    } else if (qv >= half) {
        ix = ifloor(scale2<12>(qv));
    } else {
        ix = ifloor(scale2<13>(qv)) - 2048;
    }
    if (ix > 4095) {
        ix = 4095;
    }
    Sd s0 = seed[ix];
    Ry y = s0;
    if (hi == 1) {
        y = scale2<-1>(y);
    }
    for (int it = 0; it < 2; it++) {
        Rq y2 = y * y;  // up to 4 (q = 1/4): Rq <48,4>
        Ry my2 = qv * y2;
        Rq tt = three - my2;
        Rq yt = y * tt;  // up to 4 (q = 1/4 exactly: n on an axis at 1/2): Rq <48,4>
        y = scale2<-1>(yt);
    }
    Ry nm = q0 * y;  // |n| = q / sqrt(q); x = 0 gives 0 (the zero bypass, 56, 3)
    on[0] = nr;
    on[1] = ni;
    oy[0] = y;
    oy[1] = nm;
    kk[0] = k;
}

void voc_call(Turn ph[1], Ang pr[1], Sig xw[2], Vb xv[2], St xm[1], Rm xr[4], Sig u[5], Sig y[2], int fl[5], const VocCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=xw complete
    #pragma HLS ARRAY_PARTITION variable=xv complete
    #pragma HLS ARRAY_PARTITION variable=xr complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    #pragma HLS ALLOCATION operation instances=mul limit=6
    static const Rc cs[1024] = VS_COS;
    static const Rc sn[1024] = VS_SIN;
    static const Coef pinv[14] = DR_PINV;
    static const Coef p2[14] = DR_P2;
    Sig one = 1;
    Coef c40 = 40;  // 1 / 0.025
    Turn phi = ph[0];
    int jb = ifloor(scale2<10>(phi));
    if (jb > 1023) {
        jb = 1023;
    }
    Turn hf = 0.5;
    Turn ft = scale2<10>(phi) - jb;
    Turn fh = ft - hf;
    Turn dl = scale2<-10>(fh);
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
    Ang sdl = dr * se;
    Ang ca = d2 * s12;
    Rq cbb = one - ca;
    Ang cc = scale2<-1>(d2);
    Ang cd = cc * cbb;
    Rq cdl = one - cd;
    Rc c0 = cs[jb];
    Rc s0 = sn[jb];
    Sig t1 = c0 * cdl;
    Sig t2 = s0 * sdl;
    Sig t3 = s0 * cdl;
    Sig t4 = c0 * sdl;
    Sig cph = t1 - t2;
    Sig sph = t3 + t4;
    Sig a1 = u[0] * cph;
    Sig a2 = u[1] * sph;
    Sig a3 = u[1] * cph;
    Sig a4 = u[0] * sph;
    Sig vnr = a1 + a2;
    Sig vni = a3 - a4;
    Sig b1 = u[2] * cph;
    Sig b2 = u[3] * sph;
    Sig b3 = u[3] * cph;
    Sig b4 = u[2] * sph;
    Sig inr = b1 + b2;
    Sig ini = b3 - b4;
    Sig er = xw[0];  // E = w (the oscillator's voltage)
    Sig ei = xw[1];
    Sig dre = er - vnr;
    Sig dim = ei - vni;
    Sig dr2 = dre * dre;
    Sig di2 = dim * dim;
    Sig dd2 = dr2 + di2;
    int on = 0;
    if (dd2 > k.kg[1]) {
        on = 1;
    }
    Sig zr = k.kz[0];
    Sig zx = k.kz[1];
    Sig ipr = 0;
    Sig ipi = 0;
    if (on == 1) {
        Sig rd = vs_rsqrt(dd2);
        Sig dm = dd2 * rd;
        Sig gm = dm - k.kc[3];
        Sig q4 = k.kc[2] * gm;
        Sig ds = k.kc[1] + q4;
        Sig rs = vs_rsqrt(ds);
        Sig sq = ds * rs;
        Sig sbt = sq - k.kc[0];
        Sig ov = sbt * k.kc[4];
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
        for (int it = 0; it < 9; it++) {
            #pragma HLS PIPELINE off
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
        Sig ir = k.kg[0] + ov;
        Sig qr = k.kz[4] * ov;
        Sig qx = k.kz[5] * ov;
        zr = k.kz[0] + qr;
        zx = k.kz[1] + qx;
        Sig dhr = dre * rd;
        Sig dhi = dim * rd;
        Sig h2 = dhr * ir;
        Sig h3 = dhi * ir;
        Sig zir0 = zr * ir;
        Sig zix0 = zx * ir;
        Sig g1 = zir0 * rd;
        Sig g2 = zix0 * rd;
        Sig m1 = h2 * g1;
        Sig m2 = h3 * g2;
        Sig m3 = h3 * g1;
        Sig m4 = h2 * g2;
        ipr = m1 + m2;
        ipi = m3 - m4;
    } else {
        Sig n1 = dre * k.kg[2];
        Sig n2 = dim * k.kg[3];
        Sig n3 = dim * k.kg[2];
        Sig n4 = dre * k.kg[3];
        ipr = n1 - n2;
        ipi = n3 + n4;
    }
    Sig zex = zx - k.kz[2];
    Sig w1 = zr * ipr;
    Sig w2 = zex * ipi;
    Sig w3 = zr * ipi;
    Sig w4 = zex * ipr;
    Sig zir = w1 - w2;
    Sig zii = w3 + w4;
    Sig ucr = er - zir;  // u_c in the omega_n frame
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
    Sig vd2 = vd * vd;
    Sig lim2 = k.kg[9] * vd2;
    int cm = 0;
    if (uu2 > lim2) {
        cm = 1;
    }
    Sig ru = vs_rsqrt(uu2);
    Sig gv = vs_recipv(vd);
    Sig fac = gv;
    if (cm == 1) {
        fac = k.kg[8] * ru;
    }
    Sig o1 = cph * ucr;
    Sig o2 = sph * uci;
    Sig o3 = sph * ucr;
    Sig o4 = cph * uci;
    Sig uar = o1 - o2;
    Sig uai = o3 + o4;
    y[0] = uar * fac;
    y[1] = uai * fac;
    Rn nv[2];
    #pragma HLS ARRAY_PARTITION variable=nv complete
    Ry yv[2];
    #pragma HLS ARRAY_PARTITION variable=yv complete
    int kv[1];
    dr_mag(vnr, vni, nv, yv, kv);
    Coef sk = p2[kv[0] + 7 - 1];
    Sig vkm = yv[1] * sk;  // |v_n| = |n| 2^k
    int fvk = 0;
    Rq yy2 = yv[0] * yv[0];  // 1 / |n|^2
    Coef pk = pinv[kv[0] + 7 - 1];
    Coef gk = yy2 * pk;  // v_n / |v_n|^2 = n 2^k / (|n|^2 4^k) = n y^2 2^-k
    if (vkm < k.kv[0]) {  // below the floor: |v_k| taken as 0.025 along its direction (flagged)
        fvk = 1;
        gk = yv[0] * c40;
        if (vkm < k.kv[2]) {  // zero voltage (no direction below 2^-7): iref = 0 (the package: 0 below 1e-3)
            gk = 0;
        }
    }
    Coef ux = gk * nv[0];
    Coef uy = gk * nv[1];
    Ir fr1 = k.kp[0] * ux + k.kq[0] * ux;  // two 48 x 48 products (exact sum before the one rounding)
    Ir fr2 = k.kp[1] * uy + k.kq[1] * uy;
    Ir fr3 = k.kp[0] * uy + k.kq[0] * uy;
    Ir fr4 = k.kp[1] * ux + k.kq[1] * ux;
    Ir irr = fr1 + fr2;  // (p - j q)(ux + j uy)
    Ir iri = fr3 - fr4;
    Rn nm[2];
    #pragma HLS ARRAY_PARTITION variable=nm complete
    Ry ym[2];
    #pragma HLS ARRAY_PARTITION variable=ym complete
    int km[1];
    Sig mz = 0;
    dr_mag(xm[0], mz, nm, ym, km);
    Coef pm = pinv[km[0] + 7 - 1];
    Sig rm = 0;  // 1 / m, formed only in its domain (60b, 2): below the floor rot = 0
    int fmm = 1;  // (the package: rot = 0 for m <= 1e-6; here below 1/8, flagged)
    if (xm[0] >= k.kv[1]) {
        rm = ym[0] * pm;
        fmm = 0;
    }
    Sig rvr = er * rm;
    Sig rvi = ei * rm;
    Sig z1 = inr * rvr;
    Sig z2 = ini * rvi;
    Sig z3 = ini * rvr;
    Sig z4 = inr * rvi;
    Sig zzr = z1 + z2;  // Z = i_n conj(rot_v)
    Sig zzi = z3 - z4;
    Sig gv1 = k.kl[2] * xv[0];
    Sig gv2 = k.kl[3] * xv[1];
    Sig gv3 = k.kl[2] * xv[1];
    Sig gv4 = k.kl[3] * xv[0];
    Sig fa1 = zzr - (gv1 - gv2);
    Sig fa2 = zzi - (gv3 + gv4);
    Sig hc1 = k.kd[0] * fa1;
    Sig hc2 = k.kd[1] * fa2;
    Sig hc3 = k.kd[0] * fa2;
    Sig hc4 = k.kd[1] * fa1;
    Sig hfr = hc1 - hc2;
    Sig hfi = hc3 + hc4;
    Ir j1 = hfr * rvr;
    Ir j2 = hfi * rvi;
    Ir j3 = hfi * rvr;
    Ir j4 = hfr * rvi;
    Ir ifr = j1 - j2;  // i_f
    Ir ifi = j3 + j4;
    Sig wqr = er * er;
    Sig wqi = ei * ei;
    Sig wq = wqr + wqi;
    Sig am = k.kp[2] - wq;
    Gn xa = k.kx[0] * am;  // Ts xi (V_nom^2 - |w|^2), up to 5e-4: on the wider Gn grid (2^-52)
    Rn nw[2];
    #pragma HLS ARRAY_PARTITION variable=nw complete
    Ry yw[2];
    #pragma HLS ARRAY_PARTITION variable=yw complete
    int kw[1];
    dr_mag(er, ei, nw, yw, kw);
    Coef p2w = p2[kw[0] + 7 - 1];
    xm[0] = yw[1] * p2w;  // |w| = |n| 2^k
    fl[0] = on;
    fl[1] = cm;
    fl[2] = fdc;
    fl[3] = fvk;
    fl[4] = fmm;
    Ang inc = k.ks[0] + k.kt[0] + pr[0];
    Turn pn = phi + inc;
    Ang dd = pn - phi;
    pr[0] = inc - dd;
    if (pn >= one) {
        pn = pn - one;
    }
    ph[0] = pn;
    Vb v0 = xv[0];
    Vb v1 = xv[1];
    Wv sd1 = xa * er + k.ke[0] * (irr - ifr) - k.ke[1] * (iri - ifi) + xr[0];
    Sig n1w = er + sd1;
    xr[0] = er - n1w + sd1;
    xw[0] = n1w;
    Wv sd2 = xa * ei + k.ke[0] * (iri - ifi) + k.ke[1] * (irr - ifr) + xr[1];
    Sig n2w = ei + sd2;
    xr[1] = ei - n2w + sd2;
    xw[1] = n2w;
    Wv sd3 = k.kl[0] * v0 - k.kl[1] * v1 + zzr + xr[2];
    Vb n3v = v0 + sd3;
    xr[2] = v0 - n3v + sd3;
    xv[0] = n3v;
    Wv sd4 = k.kl[0] * v1 + k.kl[1] * v0 + zzi + xr[3];
    Vb n4v = v1 + sd4;
    xr[3] = v1 - n4v + sd4;
    xv[1] = n4v;
}

void voc_run(const VocCoefs &k, Turn ph[1], Ang pr[1], Sig xw[2], Vb xv[2], St xm[1], Rm xr[4], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=ph
    #pragma HLS INTERFACE mode=s_axilite port=pr
    #pragma HLS INTERFACE mode=s_axilite port=xw
    #pragma HLS INTERFACE mode=s_axilite port=xv
    #pragma HLS INTERFACE mode=s_axilite port=xm
    #pragma HLS INTERFACE mode=s_axilite port=xr
    #pragma HLS INTERFACE mode=s_axilite port=u
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=s_axilite port=fo
    #pragma HLS INTERFACE mode=s_axilite port=ncall
    #pragma HLS INTERFACE mode=ap_none port=m_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    Turn phs[1];
    Ang prs[1];
    Sig xws[2];
    #pragma HLS ARRAY_PARTITION variable=xws complete
    Vb xvs[2];
    #pragma HLS ARRAY_PARTITION variable=xvs complete
    St xms[1];
    Rm xrs[4];
    #pragma HLS ARRAY_PARTITION variable=xrs complete
    phs[0] = ph[0];
    prs[0] = pr[0];
    xms[0] = xm[0];
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        xws[i] = xw[i];
        xvs[i] = xv[i];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        xrs[i] = xr[i];
    }
    Sig uu[5];
    #pragma HLS ARRAY_PARTITION variable=uu complete
    Sig yy[2];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int ff[5];
    #pragma HLS ARRAY_PARTITION variable=ff complete
    bool st = false;
    for (int c = 0; c < ncall; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        #pragma HLS PIPELINE off
        sample = false;
        for (int i = 0; i < 5; i++) {
            uu[i] = u[c][i];
        }
        voc_call(phs, prs, xws, xvs, xms, xrs, uu, yy, ff, k);
        y[c][0] = yy[0];
        y[c][1] = yy[1];
        fo[c] = ff[0] + 2 * ff[1] + 4 * ff[2] + 8 * ff[3] + 16 * ff[4];
        m_out = yy[0];
        st = !st;
        strobe = st;
        sample = true;
    }
    ph[0] = phs[0];
    pr[0] = prs[0];
    xm[0] = xms[0];
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        xw[i] = xws[i];
        xv[i] = xvs[i];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        xr[i] = xrs[i];
    }
}

