/* The fast side of the window-held coupling benchmark (COSIM_DESIGN.md 8.6, group A), in C: R_f in series with an
 * inductor of storage H(l) = l^2 / (2 k) + c4 l^4 / 4, a port conductance gp, the held port voltage a, steps hs[0..n-1].
 * The same arithmetic as `fast_window` in window.jl. out = {l1, b, dH, D, Eport}. Returns 0. */
#include <math.h>

static double energy(double k, double c4, double z) { return z * z / (2.0 * k) + c4 * z * z * z * z / 4.0; }
static double dgrad(double k, double c4, double z0, double z1) {
    return (z0 + z1) / (2.0 * k) + c4 * (z1 * z1 * z1 + z1 * z1 * z0 + z1 * z0 * z0 + z0 * z0 * z0) / 4.0;
}

int fast_rl_window(double rf, double k, double c4, double gp, double l0, double a, const double *hs, int n,
                   double *out) {
    double l = l0, Ib = 0.0, D = 0.0, H = 0.0;
    for (int j = 0; j < n; j++) {
        double h = hs[j], l1 = l;
        for (int it = 0; it < 60; it++) {
            double ib = dgrad(k, c4, l, l1);
            double r = (l1 - l) / h - a + rf * ib;
            double di = 1.0 / (2.0 * k) + c4 * (3.0 * l1 * l1 + 2.0 * l1 * l + l * l) / 4.0;
            double dl = r / (1.0 / h + rf * di);
            l1 -= dl;
            if (fabs(dl) <= 1e-15 * fmax(1.0, fabs(l1))) { it = -1; break; }
            if (it == 59) return 1;                       /* no convergence */
        }
        double ib = dgrad(k, c4, l, l1);
        double ip = ib + gp * a;
        Ib += h * ip;
        D += h * (rf * ib * ib + gp * a * a);
        H += h;
        l = l1;
    }
    out[0] = l;
    out[1] = Ib / H;
    out[2] = energy(k, c4, l) - energy(k, c4, l0);
    out[3] = D;
    out[4] = H * a * (Ib / H);
    return 0;
}
