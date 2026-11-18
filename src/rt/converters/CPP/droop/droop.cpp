// droop.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the droop converter's controller call (PHDroop, sampled, DC rejection); from src/rt/converters/droop.jl
#include "droop.h"

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

void droop_call(Turn ph[1], Ang pr[1], Pf xp[2], St xu[1], Sig xs[6], Vb xv[2], Rm xr[11], Sig u[5], Sig y[2], int fl[5], const DroopCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=xp complete
    #pragma HLS ARRAY_PARTITION variable=xs complete
    #pragma HLS ARRAY_PARTITION variable=xv complete
    #pragma HLS ARRAY_PARTITION variable=xr complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    #pragma HLS ALLOCATION operation instances=mul limit=6
    static const Rc cs[1024] = VS_COS;
    static const Rc sn[1024] = VS_SIN;
    static const Coef p2[14] = DR_P2;
    static const Coef pinv[14] = DR_PINV;
    Sig one = 1;
    Sig zero = 0;
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
    Sig gv1 = k.kl[2] * xs[4];
    Sig gv2 = k.kl[3] * xs[5];
    Sig gv3 = k.kl[2] * xs[5];
    Sig gv4 = k.kl[3] * xs[4];
    Sig av = xs[0] - (gv1 - gv2);
    Sig bv = xs[1] - (gv3 + gv4);
    Sig gi1 = k.kl[2] * xv[0];
    Sig gi2 = k.kl[3] * xv[1];
    Sig gi3 = k.kl[2] * xv[1];
    Sig gi4 = k.kl[3] * xv[0];
    Sig ai = xs[2] - (gi1 - gi2);
    Sig bi = xs[3] - (gi3 + gi4);
    Sig hv1 = k.kd[0] * av;
    Sig hv2 = k.kd[1] * bv;
    Sig hv3 = k.kd[0] * bv;
    Sig hv4 = k.kd[1] * av;
    Sig vfr = hv1 - hv2;
    Sig vfi = hv3 + hv4;
    Sig hi1 = k.kd[0] * ai;
    Sig hi2 = k.kd[1] * bi;
    Sig hi3 = k.kd[0] * bi;
    Sig hi4 = k.kd[1] * ai;
    Sig ifr = hi1 - hi2;
    Sig ifi = hi3 + hi4;
    Rn ni[2];
    #pragma HLS ARRAY_PARTITION variable=ni complete
    Ry yi[2];
    #pragma HLS ARRAY_PARTITION variable=yi complete
    int ki[1];
    dr_mag(ifr, ifi, ni, yi, ki);
    Coef sci = p2[ki[0] + 7 - 1];
    Sig ia = yi[1] * sci;  // |imf|
    int bo = 0;
    Sig ur = xu[0];  // U = u_mag (+ j 0)
    Sig ui = 0;
    if (ia > k.kg[0]) {
        bo = 1;
        Sig ov = ia - k.kg[0];
        Sig kr = k.kk[0] * ov;
        Sig kx = k.kk[1] * ov;
        Sig du1 = kr * ifr;
        Sig du2 = kx * ifi;
        Sig du3 = kr * ifi;
        Sig du4 = kx * ifr;
        ur = ur - (du1 - du2);
        ui = zero - (du3 + du4);
    }
    Sig dre = ur - vfr;
    Sig dim = ui - vfi;
    Rn nd[2];
    #pragma HLS ARRAY_PARTITION variable=nd complete
    Ry yd[2];
    #pragma HLS ARRAY_PARTITION variable=yd complete
    int kd[1];
    dr_mag(dre, dim, nd, yd, kd);
    Coef scd = p2[kd[0] + 7 - 1];
    Sig dmg = yd[1] * scd;  // |d|
    int s1 = 0;
    if (dmg > k.kg[2]) {
        s1 = 1;
        Ry f1 = k.kg[2] * yd[0];  // dmax / |n_d|: U = vmf + n_d dmax / |n_d|
        Sig e1 = f1 * nd[0];
        Sig e2 = f1 * nd[1];
        ur = vfr + e1;
        ui = vfi + e2;
    }
    int s2 = 0;
    if (ia > k.kg[1]) {
        s2 = 1;
        Sig g0 = k.kg[1] * yi[0];
        Coef pin = pinv[ki[0] + 7 - 1];
        Fr fc = g0 * pin;  // i_con_lim / |imf|
        Sig wr = ur - vfr;
        Sig wi = ui - vfi;
        Sig w1 = wr * fc;
        Sig w2 = wi * fc;
        ur = vfr + w1;
        ui = vfi + w2;
    }
    Sig ur2 = ur * ur;
    Sig ui2 = ui * ui;
    Sig uu2 = ur2 + ui2;
    Sig vd = u[4];
    int fdc = 0;
    if (vd < k.kg[5]) {
        vd = k.kg[5];
        fdc = 1;
    }
    if (vd > k.kg[6]) {
        vd = k.kg[6];
        fdc = 1;
    }
    Sig vd2 = vd * vd;
    Sig lim2 = k.kg[4] * vd2;
    int cm = 0;
    if (uu2 > lim2) {
        cm = 1;
    }
    Sig ru = vs_rsqrt(uu2);
    Sig gv = vs_recipv(vd);
    Sig fac = gv;
    if (cm == 1) {
        fac = k.kg[3] * ru;
    }
    Sig r1 = cph * ur;
    Sig r2 = sph * ui;
    Sig r3 = sph * ur;
    Sig r4 = cph * ui;
    Sig uar = r1 - r2;
    Sig uai = r3 + r4;
    y[0] = uar * fac;
    y[1] = uai * fac;
    fl[0] = bo;
    fl[1] = s1;
    fl[2] = s2;
    fl[3] = cm;
    fl[4] = fdc;
    Sig pv1 = u[0] * u[2];
    Sig pv2 = u[1] * u[3];
    Sig pp = pv1 + pv2;  // P = Re(v conj i)
    Sig qv1 = u[1] * u[2];
    Sig qv2 = u[0] * u[3];
    Sig qq = qv1 - qv2;  // Q = Im(v conj i)
    Sig vc1 = u[0] * cph;
    Sig vc2 = u[1] * sph;
    Sig vc3 = u[1] * cph;
    Sig vc4 = u[0] * sph;
    Sig vcr = vc1 + vc2;  // v conj(rot)
    Sig vci = vc3 - vc4;
    Sig ic1 = u[2] * cph;
    Sig ic2 = u[3] * sph;
    Sig ic3 = u[3] * cph;
    Sig ic4 = u[2] * sph;
    Sig icr = ic1 + ic2;  // i conj(rot)
    Sig ici = ic3 - ic4;
    Pf pf0 = xp[0];
    Pf qf0 = xp[1];
    St um0 = xu[0];
    Sig vr0 = xs[0];
    Sig vi0 = xs[1];
    Sig ir0 = xs[2];
    Sig ii0 = xs[3];
    Sig wv0 = xs[4];
    Sig wv1 = xs[5];
    Vb zi0 = xv[0];
    Vb zi1 = xv[1];
    Ang inc = k.ks[0] + k.kt[0] - k.kpm[0] * (pf0 + xr[0]) + pr[0];
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
    Pf qe = qf0 - k.kp[1];
    Sig mqe = k.km[0] * qe;
    Sig ue0 = k.kp[2] - mqe;  // u_ref (the target of u_mag)
    Pf n01 = pf0 + k.ka[0] * (pp - pf0) + xr[0];
    xr[0] = pf0 - n01 + k.ka[0] * (pp - pf0) + xr[0];
    xp[0] = n01;
    Pf n02 = qf0 + k.ka[1] * (qq - qf0) + xr[1];
    xr[1] = qf0 - n02 + k.ka[1] * (qq - qf0) + xr[1];
    xp[1] = n02;
    St n03 = um0 + k.ku[0] * (ue0 - um0) + xr[2];
    xr[2] = um0 - n03 + k.ku[0] * (ue0 - um0) + xr[2];
    xu[0] = n03;
    Sig n04 = vr0 + k.kv[0] * (vcr - vr0) + xr[3];
    xr[3] = vr0 - n04 + k.kv[0] * (vcr - vr0) + xr[3];
    xs[0] = n04;
    Sig n05 = vi0 + k.kv[0] * (vci - vi0) + xr[4];
    xr[4] = vi0 - n05 + k.kv[0] * (vci - vi0) + xr[4];
    xs[1] = n05;
    Sig n06 = ir0 + k.kv[1] * (icr - ir0) + xr[5];
    xr[5] = ir0 - n06 + k.kv[1] * (icr - ir0) + xr[5];
    xs[2] = n06;
    Sig n07 = ii0 + k.kv[1] * (ici - ii0) + xr[6];
    xr[6] = ii0 - n07 + k.kv[1] * (ici - ii0) + xr[6];
    xs[3] = n07;
    Sig n08 = wv0 + k.kl[0] * wv0 - k.kl[1] * wv1 + vr0 + xr[7];
    xr[7] = wv0 - n08 + k.kl[0] * wv0 - k.kl[1] * wv1 + vr0 + xr[7];
    xs[4] = n08;
    Sig n09 = wv1 + k.kl[0] * wv1 + k.kl[1] * wv0 + vi0 + xr[8];
    xr[8] = wv1 - n09 + k.kl[0] * wv1 + k.kl[1] * wv0 + vi0 + xr[8];
    xs[5] = n09;
    Vb n10 = zi0 + k.kl[0] * zi0 - k.kl[1] * zi1 + ir0 + xr[9];
    xr[9] = zi0 - n10 + k.kl[0] * zi0 - k.kl[1] * zi1 + ir0 + xr[9];
    xv[0] = n10;
    Vb n11 = zi1 + k.kl[0] * zi1 + k.kl[1] * zi0 + ii0 + xr[10];
    xr[10] = zi1 - n11 + k.kl[0] * zi1 + k.kl[1] * zi0 + ii0 + xr[10];
    xv[1] = n11;
}

void droop_run(const DroopCoefs &k, Turn ph[1], Ang pr[1], Pf xp[2], St xu[1], Sig xs[6], Vb xv[2], Rm xr[11], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=ph
    #pragma HLS INTERFACE mode=s_axilite port=pr
    #pragma HLS INTERFACE mode=s_axilite port=xp
    #pragma HLS INTERFACE mode=s_axilite port=xu
    #pragma HLS INTERFACE mode=s_axilite port=xs
    #pragma HLS INTERFACE mode=s_axilite port=xv
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
    Pf xps[2];
    #pragma HLS ARRAY_PARTITION variable=xps complete
    St xus[1];
    Sig xss[6];
    #pragma HLS ARRAY_PARTITION variable=xss complete
    Vb xvs[2];
    #pragma HLS ARRAY_PARTITION variable=xvs complete
    Rm xrs[11];
    #pragma HLS ARRAY_PARTITION variable=xrs complete
    phs[0] = ph[0];
    prs[0] = pr[0];
    xus[0] = xu[0];
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        xps[i] = xp[i];
        xvs[i] = xv[i];
    }
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        xss[i] = xs[i];
    }
    for (int i = 0; i < 11; i++) {
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
        sample = false;
        for (int i = 0; i < 5; i++) {
            uu[i] = u[c][i];
        }
        droop_call(phs, prs, xps, xus, xss, xvs, xrs, uu, yy, ff, k);
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
    xu[0] = xus[0];
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        xp[i] = xps[i];
        xv[i] = xvs[i];
    }
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        xs[i] = xss[i];
    }
    for (int i = 0; i < 11; i++) {
        #pragma HLS UNROLL
        xr[i] = xrs[i];
    }
}

