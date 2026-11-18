// gfl.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the GFL converter (PHGfl): the sampled call and the h-paced current loop, h = 3.2e-8 s; from src/rt/converters/gfl.jl
#include "gfl.h"

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

void gfl_call(Turn ph[1], Ang pr[1], Pf xs[9], Rm xr[6], Sig lt[5], Sig u[2], int fl[4], const GflCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=xs complete
    #pragma HLS ARRAY_PARTITION variable=xr complete
    #pragma HLS ARRAY_PARTITION variable=lt complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    #pragma HLS ALLOCATION operation instances=mul limit=6
    static const Rc cs[1024] = VS_COS;
    static const Rc sn[1024] = VS_SIN;
    static const Coef p2[14] = DR_P2;
    static const Coef pinv[14] = DR_PINV;
    Sig one = 1;
    Sig zero = 0;
    Sig five = 5;
    Sig tenth = 0.1;
    Turn phi0 = ph[0];
    int jb = ifloor(scale2<10>(phi0));
    if (jb > 1023) {
        jb = 1023;
    }
    Turn hf = 0.5;
    Turn ft = scale2<10>(phi0) - jb;
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
    Sig vd = a1 + a2;  // v = v_k e^{-j 2 pi psi}
    Sig vq = a3 - a4;
    Pf phs = xs[0];
    Pf vf0 = xs[1];
    Pf ff0 = xs[2];
    Pf pr0 = xs[3];
    Pf qr0 = xs[4];
    Pf gp0 = xs[5];
    Pf gq0 = xs[6];
    Pf id0 = xs[7];
    Pf iq0 = xs[8];
    Sig om1 = k.ko[0] * vq;
    Sig om2 = k.ko[1] * phs;
    Sig omn = one + om1 + om2;  // omega / omega_n
    Rn nv[2];
    #pragma HLS ARRAY_PARTITION variable=nv complete
    Ry yv[2];
    #pragma HLS ARRAY_PARTITION variable=yv complete
    int kv[1];
    Sig va = u[0];
    if (va < 0) {
        va = -va;
    }
    Sig vb = u[1];
    if (vb < 0) {
        vb = -vb;
    }
    Sig vmx = va;
    if (vb > va) {
        vmx = vb;
    }
    Sig lo1 = 0.015625;
    Sig lo2 = 7.62939453125e-6;
    Sig lo3 = 3.725290298461914e-9;
    Sig us1 = u[0];
    Sig us2 = u[1];
    int vsc = 0;
    if (vmx < lo1) {
        us1 = scale2<11>(u[0]);
        us2 = scale2<11>(u[1]);
        vsc = 1;
    }
    if (vmx < lo2) {
        us1 = scale2<22>(u[0]);
        us2 = scale2<22>(u[1]);
        vsc = 2;
    }
    if (vmx < lo3) {
        us1 = scale2<33>(u[0]);
        us2 = scale2<33>(u[1]);
        vsc = 3;
    }
    dr_mag(us1, us2, nv, yv, kv);
    Coef sv = p2[kv[0] + 7 - 1];
    Sig vks = yv[1] * sv;  // the (shifted) magnitude
    Sig vkm = vks;
    if (vsc == 1) {
        vkm = scale2<-11>(vks);
    }
    if (vsc == 2) {
        vkm = scale2<-22>(vks);
    }
    if (vsc == 3) {
        vkm = scale2<-33>(vks);
    }
    Pf ev = vf0 - one;
    Pf dbv = 0;
    if (ev > k.kg[12]) {
        dbv = ev - k.kg[12];
    }
    if (ev < zero - k.kg[12]) {
        dbv = ev + k.kg[12];
    }
    Pf dq0 = zero - k.kg[11] * dbv;
    Pf ef = ff0 - one;
    Pf dbf = 0;
    if (ef > k.kg[10]) {
        dbf = ef - k.kg[10];
    }
    if (ef < zero - k.kg[10]) {
        dbf = ef + k.kg[10];
    }
    Pf dp0 = zero - k.kg[9] * dbf;
    Pf rp = k.kg[0] - pr0;
    if (rp > k.kg[2]) {
        rp = k.kg[2];
    }
    if (rp < zero - k.kg[2]) {
        rp = zero - k.kg[2];
    }
    Pf rq = k.kg[1] - qr0;
    if (rq > k.kg[3]) {
        rq = k.kg[3];
    }
    if (rq < zero - k.kg[3]) {
        rq = zero - k.kg[3];
    }
    Pf ep1 = vd * id0;
    Pf eP = pr0 + dp0 - ep1;
    Pf eq1 = vd * iq0;
    Pf eQ = qr0 + dq0 + eq1;
    Sig idr = k.kg[4] * eP + k.kg[5] * gp0;
    Sig iqr = k.kg[8] * vq - (k.kg[6] * eQ + k.kg[7] * gq0);
    Rn ni[2];
    #pragma HLS ARRAY_PARTITION variable=ni complete
    Ry yi[2];
    #pragma HLS ARRAY_PARTITION variable=yi complete
    int ki[1];
    dr_mag(idr, iqr, ni, yi, ki);
    Coef si = p2[ki[0] + 7 - 1];
    Sig ima = yi[1] * si;  // |i_ref|
    int sat = 0;
    Sig idc = idr;
    Sig iqc = iqr;
    if (ima > k.kg[13]) {
        sat = 1;
        Coef pin = pinv[ki[0] + 7 - 1];
        Rq fs = k.kg[13] * yi[0] * pin;  // i_max / |i_ref|
        idc = idr * fs;
        iqc = iqr * fs;
    }
    Sig gt = five * (vd - tenth);  // (vd - 0.1) / 0.2
    int gp = 0;
    if (gt < one) {
        gp = 1;
    }
    if (gt > one) {
        gt = one;
    }
    int gz = 0;
    if (gt < zero) {
        gt = zero;
        gz = 1;
    }
    Sig aw = gt;
    if (sat == 1) {
        aw = zero;
    }
    Ai inc = k.kc[0] + k.kt[0] + k.kp[0] * vq + k.kp[1] * phs;
    Turn pha = phi0 + inc;  // advance = 1 (checked at export)
    if (pha >= one) {
        pha = pha - one;
    }
    if (pha < zero) {  // the increment is negative for phi near -phi_max
        pha = pha + one;
    }
    int jh = ifloor(scale2<10>(pha));
    if (jh > 1023) {
        jh = 1023;
    }
    Turn fth = scale2<10>(pha) - jh;
    Turn fhh = fth - hf;
    Turn dlh = scale2<-10>(fhh);
    Ang drh = twopi * dlh;
    Ang d2h = drh * drh;
    Ang sah = d2h * s20;
    Rq sbh = one - sah;
    Ang sch = d2h * s6;
    Ang sdh = sch * sbh;
    Rq seh = one - sdh;
    Ang sdlh = drh * seh;
    Ang cah = d2h * s12;
    Rq cbh = one - cah;
    Ang cch = scale2<-1>(d2h);
    Ang cdh = cch * cbh;
    Rq cdlh = one - cdh;
    Rc c0h = cs[jh];
    Rc s0h = sn[jh];
    Sig u1 = c0h * cdlh;
    Sig u2 = s0h * sdlh;
    Sig u3 = s0h * cdlh;
    Sig u4 = c0h * sdlh;
    lt[0] = u1 - u2;
    lt[1] = u3 + u4;
    lt[2] = id0;
    lt[3] = iq0;
    lt[4] = omn;
    fl[0] = sat;
    fl[1] = gp;
    fl[2] = gz;
    Ai inp = inc + pr[0];
    Turn pn = phi0 + inp;
    Ai dd = pn - phi0;
    pr[0] = inp - dd;
    if (pn >= one) {
        pn = pn - one;
    }
    if (pn < zero) {
        pn = pn + one;
    }
    ph[0] = pn;
    Pf phn = phs + k.kts[0] * vq;
    int fpc = 0;
    if (phn > k.kg[14]) {
        phn = k.kg[14];
        fpc = 1;
    }
    if (phn < zero - k.kg[14]) {
        phn = zero - k.kg[14];
        fpc = 1;
    }
    xs[0] = phn;
    fl[3] = fpc;
    Pf n2 = vf0 + k.kl[0] * (vkm - vf0) + xr[0];
    xr[0] = vf0 - n2 + k.kl[0] * (vkm - vf0) + xr[0];
    xs[1] = n2;
    Pf n3 = ff0 + k.kl[0] * (omn - ff0) + xr[1];
    xr[1] = ff0 - n3 + k.kl[0] * (omn - ff0) + xr[1];
    xs[2] = n3;
    xs[3] = pr0 + rp;
    xs[4] = qr0 + rq;
    Pf n6 = gp0 + k.kts[0] * aw * eP + xr[2];
    xr[2] = gp0 - n6 + k.kts[0] * aw * eP + xr[2];
    xs[5] = n6;
    Pf n7 = gq0 + k.kts[0] * aw * eQ + xr[3];
    xr[3] = gq0 - n7 + k.kts[0] * aw * eQ + xr[3];
    xs[6] = n7;
    Pf n8 = id0 + k.kl[1] * (gt * idc - id0) + xr[4];
    xr[4] = id0 - n8 + k.kl[1] * (gt * idc - id0) + xr[4];
    xs[7] = n8;
    Pf n9 = iq0 + k.kl[1] * (gt * iqc - iq0) + xr[5];
    xr[5] = iq0 - n9 + k.kl[1] * (gt * iqc - iq0) + xr[5];
    xs[8] = n9;
}

void gfl_loop(Zs z[2], Rm zr[2], Sig lt[5], Sig i[2], Sig y[2], const GflLoop &kq) {
    #pragma HLS ARRAY_PARTITION variable=z complete
    #pragma HLS ARRAY_PARTITION variable=zr complete
    #pragma HLS ARRAY_PARTITION variable=lt complete
    #pragma HLS ARRAY_PARTITION variable=i complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    Sig h1 = i[0] * lt[0];
    Sig h2 = i[1] * lt[1];
    Sig h3 = i[1] * lt[0];
    Sig h4 = i[0] * lt[1];
    Sig ihd = h1 + h2;  // ih = i conj(ch)
    Sig ihq = h3 - h4;
    Sig ed = lt[2] - ihd;
    Sig eq = lt[3] - ihq;
    Zs zd0 = z[0];
    Zs zq0 = z[1];
    Sig ud = kq.kz[0] * ed + zd0 + kq.kh[2] * ed + zr[0] - kq.kz[2] * ihq;  // u at the midpoint (frame of ch)
    Sig uq = kq.kz[1] * eq + zq0 + kq.kh[3] * eq + zr[1] + kq.kz[2] * ihd;
    Sig w1 = ud * lt[0];
    Sig w2 = uq * lt[1];
    Sig w3 = ud * lt[1];
    Sig w4 = uq * lt[0];
    y[0] = w1 - w2;
    y[1] = w3 + w4;
    Zs nd = zd0 + kq.kh[0] * ed + zr[0];
    zr[0] = zd0 - nd + kq.kh[0] * ed + zr[0];
    z[0] = nd;
    Zs nq = zq0 + kq.kh[1] * eq + zr[1];
    zr[1] = zq0 - nq + kq.kh[1] * eq + zr[1];
    z[1] = nq;
}

void gfl_run(const GflCoefs &k, const GflLoop &kq, int mode, Turn ph[1], Ang pr[1], Pf xs[9], Rm xr[6], Sig lt[5], Zs z[2], Rm zr[2], Sig uc[64][2], int fo[64], Sig il[512][2], Sig yl[512][2], int n, int decim, volatile Sig &u_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=kq
    #pragma HLS INTERFACE mode=s_axilite port=mode
    #pragma HLS INTERFACE mode=s_axilite port=ph
    #pragma HLS INTERFACE mode=s_axilite port=pr
    #pragma HLS INTERFACE mode=s_axilite port=xs
    #pragma HLS INTERFACE mode=s_axilite port=xr
    #pragma HLS INTERFACE mode=s_axilite port=lt
    #pragma HLS INTERFACE mode=s_axilite port=z
    #pragma HLS INTERFACE mode=s_axilite port=zr
    #pragma HLS INTERFACE mode=s_axilite port=uc
    #pragma HLS INTERFACE mode=s_axilite port=fo
    #pragma HLS INTERFACE mode=s_axilite port=il
    #pragma HLS INTERFACE mode=s_axilite port=yl
    #pragma HLS INTERFACE mode=s_axilite port=n
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=u_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    Turn phs[1];
    Ang prs[1];
    Pf xss[9];
    #pragma HLS ARRAY_PARTITION variable=xss complete
    Rm xrs[6];
    #pragma HLS ARRAY_PARTITION variable=xrs complete
    Sig lts[5];
    #pragma HLS ARRAY_PARTITION variable=lts complete
    Zs zs[2];
    #pragma HLS ARRAY_PARTITION variable=zs complete
    Rm zrs[2];
    #pragma HLS ARRAY_PARTITION variable=zrs complete
    phs[0] = ph[0];
    prs[0] = pr[0];
    for (int i = 0; i < 9; i++) {
        #pragma HLS UNROLL
        xss[i] = xs[i];
    }
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        xrs[i] = xr[i];
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        lts[i] = lt[i];
    }
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        zs[i] = z[i];
        zrs[i] = zr[i];
    }
    bool st = false;
    int cnt = 0;
    Sig ii[2];
    #pragma HLS ARRAY_PARTITION variable=ii complete
    Sig yy[2];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    Sig uu[2];
    #pragma HLS ARRAY_PARTITION variable=uu complete
    int ff[4];
    #pragma HLS ARRAY_PARTITION variable=ff complete
    if (mode == 0) {
        for (int s = 0; s < n; s++) {
            #pragma HLS PIPELINE II=4
            #pragma HLS LOOP_TRIPCOUNT min=1 max=512
            ii[0] = il[s][0];
            ii[1] = il[s][1];
            gfl_loop(zs, zrs, lts, ii, yy, kq);
            yl[s][0] = yy[0];
            yl[s][1] = yy[1];
            u_out = yy[0];
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
    } else {
        for (int c = 0; c < n; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            uu[0] = uc[c][0];
            uu[1] = uc[c][1];
            gfl_call(phs, prs, xss, xrs, lts, uu, ff, k);
            fo[c] = ff[0] + 2 * ff[1] + 4 * ff[2] + 8 * ff[3];
            st = !st;
            strobe = st;
        }
    }
    ph[0] = phs[0];
    pr[0] = prs[0];
    for (int i = 0; i < 9; i++) {
        #pragma HLS UNROLL
        xs[i] = xss[i];
    }
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        xr[i] = xrs[i];
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        lt[i] = lts[i];
    }
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        z[i] = zs[i];
        zr[i] = zrs[i];
    }
}

