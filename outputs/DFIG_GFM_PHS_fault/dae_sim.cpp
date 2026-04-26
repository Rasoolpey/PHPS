#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

const int N_DIFF  = 86;
const int N_BUS   = 14;
const int N_ALG   = 28;
const int N_TOTAL = 114;

const double LOAD_G[14] = { 0.0000000000e+00, 2.0454331228e-01, 9.2343887854e-01, 4.6426502478e-01, 7.3028446688e-02, 1.0557074182e-01, 0.0000000000e+00, 0.0000000000e+00, 2.9921224160e-01, 9.1559021030e-02, 3.4542339238e-02, 5.9446793454e-02, 1.3336849209e-01, 1.5527022516e-01 };
const double LOAD_B[14] = { 0.0000000000e+00, -1.1970968046e-01, -1.8625624939e-01, 3.7879363947e-02, -1.5374409829e-02, -7.0694693185e-02, 0.0000000000e+00, 0.0000000000e+00, -1.6837027832e-01, -5.9004702442e-02, -1.7764631608e-02, -1.5592601562e-02, -5.7299055860e-02, -5.2104102403e-02 };
const double LOAD_KPF[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };
const double LOAD_KQF[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };

// Full Y-bus (14x14) — NO Kron reduction
const double Y_real[196] = {
    6.0730221448e+00, -4.9991316008e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -4.9991316008e+00, 9.9825054817e+00, -1.1350191923e+00, -1.6860331506e+00, -1.7011396671e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1350191923e+00, 4.3010723394e+00, -1.9859757099e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.6860331506e+00, -1.9859757099e+00, 1.0977254547e+01, -6.8409806615e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, -1.7011396671e+00, 0.0000000000e+00, -6.8409806615e+00, 9.6410462302e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.9421327079e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, -1.5259674405e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.5663855863e-01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.6252672811e+00, -3.9020495524e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.9020495524e+00, 5.8744933272e+00, -1.8808847537e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.8808847537e+00, 3.8704556561e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.5259674405e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0744388207e+00, -2.4890245868e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -2.4890245868e+00, 6.8583146406e+00, -1.1369941578e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1369941578e+00, 2.7162698700e+00
};
const double Y_imag[196] = {
    -2.3446494288e+01, 1.5263086523e+01, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.5263086523e+01, -3.5451270949e+01, 4.7818631518e+00, 5.1158383259e+00, 5.1939273980e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7818631518e+00, -1.5068082249e+01, 5.0688169776e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.1158383259e+00, 5.0688169776e+00, -3.8324565507e+01, 2.1578553982e+01, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 5.1939273980e+00, 0.0000000000e+00, 2.1578553982e+01, -3.4948878524e+01, 3.9807970269e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.9807970269e+00, -2.2496630988e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 3.1759639650e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9549005948e+01, 5.6953759109e+00, 9.0900827198e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.6953759109e+00, -1.0773277457e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 9.0900827198e+00, 0.0000000000e+00, -2.4450876654e+01, 1.0365394127e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0365394127e+01, -1.4827342579e+01, 4.4029437495e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.4029437495e+00, -8.5147827253e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.1759639650e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -5.4435311928e+00, 2.2519746262e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.2519746262e+00, -1.0726992605e+01, 2.3149634751e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.3149634751e+00, -5.3961180344e+00
};

// Slack bus configuration
const int IS_SLACK[14] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const double Vd_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };
const double Vq_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };

// Bus fault events
const int N_FAULTS = 1;
const int FAULT_BUS[1] = { 3 };
const double FAULT_T_START[1] = { 2.0000000000e+01 };
const double FAULT_T_END[1] = { 2.0240000000e+01 };
const double FAULT_G[1] = { 5.0000000000e+03 };
const double FAULT_B[1] = { -5.0000000000e+03 };
static int fault_active[1] = {0};

double outputs_GENCLS_1[3];
double inputs_GENCLS_1[3];
double outputs_TGOV1_1[1];
double inputs_TGOV1_1[3];
double outputs_DFIG_PHS_1[17];
double inputs_DFIG_PHS_1[5];
double outputs_DFIG_RSC_GFM_PHS_1[4];
double inputs_DFIG_RSC_GFM_PHS_1[11];
double outputs_DFIG_DCLink_PHS_1[2];
double inputs_DFIG_DCLink_PHS_1[2];
double outputs_DFIG_GSC_GFM_PHS_1[5];
double inputs_DFIG_GSC_GFM_PHS_1[6];
double outputs_DFIG_DT_PHS_1[4];
double inputs_DFIG_DT_PHS_1[3];
double outputs_DFIG_PHS_2[17];
double inputs_DFIG_PHS_2[5];
double outputs_DFIG_RSC_GFM_PHS_2[4];
double inputs_DFIG_RSC_GFM_PHS_2[11];
double outputs_DFIG_DCLink_PHS_2[2];
double inputs_DFIG_DCLink_PHS_2[2];
double outputs_DFIG_GSC_GFM_PHS_2[5];
double inputs_DFIG_GSC_GFM_PHS_2[6];
double outputs_DFIG_DT_PHS_2[4];
double inputs_DFIG_DT_PHS_2[3];
double outputs_DFIG_PHS_3[17];
double inputs_DFIG_PHS_3[5];
double outputs_DFIG_RSC_GFM_PHS_3[4];
double inputs_DFIG_RSC_GFM_PHS_3[11];
double outputs_DFIG_DCLink_PHS_3[2];
double inputs_DFIG_DCLink_PHS_3[2];
double outputs_DFIG_GSC_GFM_PHS_3[5];
double inputs_DFIG_GSC_GFM_PHS_3[6];
double outputs_DFIG_DT_PHS_3[4];
double inputs_DFIG_DT_PHS_3[3];
double outputs_DFIG_PHS_4[17];
double inputs_DFIG_PHS_4[5];
double outputs_DFIG_RSC_GFM_PHS_4[4];
double inputs_DFIG_RSC_GFM_PHS_4[11];
double outputs_DFIG_DCLink_PHS_4[2];
double inputs_DFIG_DCLink_PHS_4[2];
double outputs_DFIG_GSC_GFM_PHS_4[5];
double inputs_DFIG_GSC_GFM_PHS_4[6];
double outputs_DFIG_DT_PHS_4[4];
double inputs_DFIG_DT_PHS_4[3];

void step_GENCLS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 2.0;
    const double ra = 0.003;
    const double xd1 = 0.25;
    const double omega_b = 2.0 * M_PI * 60.0;
    const double H = 5.0;
    const double E_p = 0.9957646256848883;
    // --- Kernel ---

            double omega = x[1];
            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm = inputs[2];
            
            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;
            
            // Solve Terminal Current
            // ed=0, eq=E_p
            // vd = -ra id + xd1 iq + ed -> vd = -ra id + xd1 iq
            // vq = -xd1 id - ra iq + eq -> vq - E_p = -xd1 id - ra iq
            
            // [vd; vq-E_p] = [-ra, xd1; -xd1, -ra] [id; iq]
            // inv = 1/det [-ra, -xd1; xd1, -ra]
            
            double det = ra*ra + xd1*xd1;
            double rhs_d = vd;
            double rhs_q = vq - E_p;
            
            double id = (-ra * rhs_d - xd1 * rhs_q) / det;
            double iq = (xd1 * rhs_d - ra * rhs_q) / det;
            
            double Te = vd*id + vq*iq;

            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = (Tm - Te - D*(omega - 1.0)) / (2.0 * H);
        
}
void step_GENCLS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 2.0;
    const double ra = 0.003;
    const double xd1 = 0.25;
    const double omega_b = 2.0 * M_PI * 60.0;
    const double H = 5.0;
    const double E_p = 0.9957646256848883;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);
            
            // Norton Current (Independent of V)
            // I_no = E / Z
            // E = j E_p (Rotor Frame) => ed=0, eq=E_p
            // Z = ra + j xd1
            // i_d = E_p * xd1 / (ra^2 + xd1^2)
            // i_q = E_p * ra / (ra^2 + xd1^2)
            
            double det = ra*ra + xd1*xd1;
            double id_no = E_p * xd1 / det;
            double iq_no = E_p * ra / det;
            
            // Rotate to Network Frame
            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;
            
            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1]; 
        
}
void step_TGOV1_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 0.05;
    const double Ki = 1.0;
    const double VMAX = 1.2;
    const double VMIN = 0.0;
    const double T1 = 0.5;
    const double T2 = 2.0;
    const double T3 = 5.0;
    const double Dt = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
    // --- Kernel ---

            double omega  = inputs[0];
            double Pref   = inputs[1];
            double u_agc  = inputs[2];   // AGC Pref correction (0.0 when AGC not wired)

            double x1 = x[0];   // valve position
            double x2 = x[1];   // reheater output
            double xi = x[2];   // integral correction

            // ---- Droop path (standard TGOV1 + AGC correction) ----
            double speed_error = (Pref + u_agc - omega) / R;
            double dx1 = (speed_error - x1) / T1;
            if (x1 >= VMAX && dx1 > 0) dx1 = 0;
            if (x1 <= VMIN && dx1 < 0) dx1 = 0;
            dxdt[0] = dx1;
            dxdt[1] = (x1 - x2) / T3;

            // ---- Integral correction with anti-windup ----
            double dxi = Ki * (wref0 - omega);
            // Anti-windup: freeze integrator when at limit and error drives it further
            if (xi >= xi_max && dxi > 0) dxi = 0;
            if (xi <= xi_min && dxi < 0) dxi = 0;
            dxdt[2] = dxi;
        
}
void step_TGOV1_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 0.05;
    const double Ki = 1.0;
    const double VMAX = 1.2;
    const double VMIN = 0.0;
    const double T1 = 0.5;
    const double T2 = 2.0;
    const double T3 = 5.0;
    const double Dt = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            // Clamp total mechanical torque to valve limits
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_DFIG_PHS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Inputs ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];
            double T_shaft = inputs[2];
            double Vrd = inputs[3];
            double Vrq = inputs[4];

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_g    = x[4];

            // ---- Hamiltonian gradient: ∇H₃ = M⁻¹ x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_g = p_g / j_inertia;

            // ---- J₃(x)·∇H₃ (skew-symmetric structure) ----
            // Synchronous-frame rotation + electromechanical gyrator
            double J0 =  omega_s * Ls * i_sq  + omega_s * Lm * i_rq;
            double J1 = -omega_s * Ls * i_sd  - omega_s * Lm * i_rd;
            double J2 =  omega_s * Lm * i_sq  + omega_s * Lr * i_rq - np * phi_rq * omega_g;
            double J3 = -omega_s * Lm * i_sd  - omega_s * Lr * i_rd + np * phi_rd * omega_g;
            double J4 =  np * (phi_rq * i_rd - phi_rd * i_rq);  // = -Te (skew row)

            // ---- R₃·∇H₃ (dissipation) ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * omega_g;

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PHS dynamics: ẋ = (1/ωb)(J₃ − R₃)∇H₃ + (1/ωb)g₃·u₃ ----
            // Flux eqs scaled by ωb; mechanical eq in seconds directly
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_shaft;  // = -Te - f_damp*ω + T_shaft

            // ---- Outputs (power, monitoring) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = Vrd * i_rd + Vrq * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
        
}
void step_DFIG_PHS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Currents from flux states: ∇H₃ = M⁻¹·x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * x[0] - Lm * x[2]) / sigma_LS;
            double i_sq =  (Lr * x[1] - Lm * x[3]) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;
            double omega_g = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Voltages ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Power (generator convention: positive = injected) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // ---- Norton equivalent: I_N = I_gen + Y_N × V ----
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            outputs[0] = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            outputs[1] = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[2] = omega_g;
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = x[0];
            outputs[12] = x[1];
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);
        
}
void step_DFIG_RSC_GFM_PHS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.193743427755563;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //   J·ω̇s = P*s − Ps − Dp·(ωs − ωN)
            //   θ̇s   = ωs
            // ==========================================================
            double omega_vs = x[0];
            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)) / J;
            dxdt[1] = omega_vs - omega_N;  // virtual angle deviation [rad]

            // ==========================================================
            // Frame rotation: RI → virtual dq (theta_s frame)
            //   The voltage controller must see voltages/currents in the
            //   theta_s frame so that vqs → 0 and vds → |V|.
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 2: RPC — Reactive Power Controller
            //   Vref = V_nom + DQ·(Q*−Qs) + ϕQs (spec §4)
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            // Clamp phi_Qs: RPC is a trim around V_nom, ±0.1 pu is sufficient
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3: Virtual impedance (theta_s frame)
            // ==========================================================
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // ==========================================================
            // STEP 4: Outer voltage loop
            //   Virtual EMF = V_term + Zvir·Ir → imag part of drop → q-axis, real → d-axis
            //   e_vqs = −(vqs_m + vaux_qs)    drive q-axis to 0
            //   e_vds =  (vds_m − Vref + vaux_ds)  drive d-axis to Vref
            // ==========================================================
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;
            dxdt[3] = e_vqs;
            dxdt[4] = e_vds;

            // ==========================================================
            // STEP 5: Current references from voltage loop
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            // ==========================================================
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // ==========================================================
            // STEP 6: Current saturation (circular limiter)
            // ==========================================================
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // ==========================================================
            // STEP 7: Inner current loop errors (theta_s frame)
            // ==========================================================
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // ==========================================================
            // STEP 8: Decoupling (theta_s frame)
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            // ==========================================================
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_slip = omega_vs - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // ==========================================================
            // STEP 9: Rotor voltage commands (theta_s frame) + saturation
            // ==========================================================
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // ==========================================================
            // STEP 10: Inner PI integrators with anti-windup
            // ==========================================================
            bool Vrd_sat = (fabs(Vrd_raw) > Vrd_max);
            bool Vrq_sat = (fabs(Vrq_raw) > Vrd_max);

            if (Vrd_sat && ((Vrd_raw > 0.0 && e_ird > 0.0) || (Vrd_raw < 0.0 && e_ird < 0.0)))
                dxdt[5] = 0.0;
            else
                dxdt[5] = e_ird;

            if (Vrq_sat && ((Vrq_raw > 0.0 && e_iqr > 0.0) || (Vrq_raw < 0.0 && e_iqr < 0.0)))
                dxdt[6] = 0.0;
            else
                dxdt[6] = e_iqr;

            if (cur_sat) {
                if ((x[3] > 0.0 && e_vqs > 0.0) || (x[3] < 0.0 && e_vqs < 0.0))
                    dxdt[3] = 0.0;
                if ((x[4] > 0.0 && e_vds > 0.0) || (x[4] < 0.0 && e_vds < 0.0))
                    dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 11: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];  // theta_s → to GSC
        
}
void step_DFIG_RSC_GFM_PHS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.193743427755563;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- STEP 2: RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];

            // --- STEP 3: Virtual impedance (theta_s frame) ---
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // --- STEP 4: Voltage loop errors ---
            // Virtual EMF = V_term + Zvir·Ir → real part of drop → d-axis, imag → q-axis
            // e_vqs = -(vqs_m + vaux_qs)   drive q-axis voltage to 0
            // e_vds =  (vds_m - Vref + vaux_ds)  drive d-axis to Vref
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;

            // --- STEP 5: Current references from voltage loop ---
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // --- STEP 6: Current saturation (circular limiter) ---
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // --- STEP 7: Inner current loop errors (theta_s frame) ---
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // --- STEP 8: Decoupling (theta_s frame) ---
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // --- STEP 9: Rotor voltage commands (theta_s frame) ---
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        
}
void step_DFIG_DCLink_PHS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc   = fmax(x[0], 0.01);
            double P_rsc = inputs[0];
            double P_gsc = inputs[1];

            // DC-link current
            double P_net = P_rsc - P_gsc;
            double i_dc  = P_net / Vdc;

            // ESR power loss
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper brake (overvoltage protection)
            double P_brake = 0.0;
            if (Vdc > 1.15 * Vdc_nom) {
                P_brake = (Vdc - 1.15 * Vdc_nom) * 10.0;
            }

            // C_dc · dVdc/dt = (P_rsc - P_gsc - P_loss - P_brake) / Vdc
            dxdt[0] = (P_net - P_loss - P_brake) / (C_dc * Vdc);
        
}
void step_DFIG_DCLink_PHS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc = fmax(x[0], 0.01);
            outputs[0] = Vdc;
            outputs[1] = inputs[0] - inputs[1];  // P_net = P_rsc - P_gsc
        
}
void step_DFIG_GSC_GFM_PHS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];
            // Note: inputs[5] = theta_s from RSC APC, used by system layer only

            // ---- PI controller: Vdc loop → active power command ----
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref
            double P_cmd = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → reactive power command ----
            double Q_meas = Vq * i_fd - Vd * i_fq;
            double e_Q = inputs[2] - Q_meas;  // Qref - Q_meas
            double Q_neg_cmd = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Power-projection: decouple P/Q control from voltage angle ----
            // From S = V·conj(I): Id=(Vd·P+Vq·Q)/V², Iq=(Vq·P−Vd·Q)/V²
            // Q_neg_cmd tracks (Qref−Q_meas) > 0 when Q below ref → more injection
            // Injection requires Iq < 0, so Iq = −Q_neg_cmd at Vd≈1,Vq≈0: sign is −Vd·Q
            double V_sq = fmax(Vd * Vd + Vq * Vq, 0.01);
            double Id_ref = (Vd * P_cmd + Vq * Q_neg_cmd) / V_sq;
            double Iq_ref = (Vq * P_cmd - Vd * Q_neg_cmd) / V_sq;

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((P_cmd > 0.0 && e_Vdc < 0.0) || (P_cmd < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Q_neg_cmd > 0.0 && e_Q < 0.0) || (Q_neg_cmd < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        
}
void step_DFIG_GSC_GFM_PHS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        
}
void step_DFIG_DT_PHS_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.7529991089181838;

            // ---- States ----
            double omega_t  = x[0] / fmax(H_t, 1e-6);
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Inputs ----
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double vw_pu   = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Shaft torque ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // ---- Aerodynamic torque (per-unit, uses dynamic Cp_eff) ----
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Empirical Cp (quasi-static target, per-unit lambda) ----
            double lambda_pu = vw_pu / omega_t_safe;
            double lambda_i = 1.0 / (1.0 / (lambda_pu + 0.08 * beta) - 0.035 / (beta * beta + 1.0));
            double Cp_raw = 0.5176 * (116.0 * lambda_i - 0.4 * beta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * lambda_pu;

            // Base Cp at rated (vw=1, omega=1, beta=0)
            double lambda_i_0 = 1.0 / (1.0 - 0.035);
            double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
            double Cp_emp = fmax(0.0, Cp_raw / fmax(Cp_0, 1e-6));
            if (Cp_emp > 2.0) Cp_emp = 2.0;

            // ---- Dynamic inflow time constant ----
            double tau_i = 1.0 / fmax(vw_pu, 0.1);

            // ---- PHS dynamics ----
            dxdt[0] = T_aero - T_shaft - D_t * omega_t;
            dxdt[1] = (Cp_emp - Cp_eff_val) / fmax(tau_i, 0.01);
            dxdt[2] = omega_t - omega_g;
        
}
void step_DFIG_DT_PHS_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.7529991089181838;

            double omega_t = x[0] / fmax(H_t, 1e-6);
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Per-unit wind speed ----
            double vw_pu = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Aerodynamic torque from dynamic Cp_eff (per-unit) ----
            // P_aero = Tm0 * vw_pu³ * Cp_eff_val,  T_aero = P_aero / omega_t
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Shaft torque: spring + damping ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            outputs[0] = T_shaft;
            outputs[1] = omega_t;
            outputs[2] = T_aero;
            outputs[3] = theta_tw;
        
}
void step_DFIG_PHS_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Inputs ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];
            double T_shaft = inputs[2];
            double Vrd = inputs[3];
            double Vrq = inputs[4];

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_g    = x[4];

            // ---- Hamiltonian gradient: ∇H₃ = M⁻¹ x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_g = p_g / j_inertia;

            // ---- J₃(x)·∇H₃ (skew-symmetric structure) ----
            // Synchronous-frame rotation + electromechanical gyrator
            double J0 =  omega_s * Ls * i_sq  + omega_s * Lm * i_rq;
            double J1 = -omega_s * Ls * i_sd  - omega_s * Lm * i_rd;
            double J2 =  omega_s * Lm * i_sq  + omega_s * Lr * i_rq - np * phi_rq * omega_g;
            double J3 = -omega_s * Lm * i_sd  - omega_s * Lr * i_rd + np * phi_rd * omega_g;
            double J4 =  np * (phi_rq * i_rd - phi_rd * i_rq);  // = -Te (skew row)

            // ---- R₃·∇H₃ (dissipation) ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * omega_g;

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PHS dynamics: ẋ = (1/ωb)(J₃ − R₃)∇H₃ + (1/ωb)g₃·u₃ ----
            // Flux eqs scaled by ωb; mechanical eq in seconds directly
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_shaft;  // = -Te - f_damp*ω + T_shaft

            // ---- Outputs (power, monitoring) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = Vrd * i_rd + Vrq * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
        
}
void step_DFIG_PHS_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Currents from flux states: ∇H₃ = M⁻¹·x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * x[0] - Lm * x[2]) / sigma_LS;
            double i_sq =  (Lr * x[1] - Lm * x[3]) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;
            double omega_g = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Voltages ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Power (generator convention: positive = injected) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // ---- Norton equivalent: I_N = I_gen + Y_N × V ----
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            outputs[0] = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            outputs[1] = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[2] = omega_g;
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = x[0];
            outputs[12] = x[1];
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);
        
}
void step_DFIG_RSC_GFM_PHS_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.1537832542129938;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //   J·ω̇s = P*s − Ps − Dp·(ωs − ωN)
            //   θ̇s   = ωs
            // ==========================================================
            double omega_vs = x[0];
            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)) / J;
            dxdt[1] = omega_vs - omega_N;  // virtual angle deviation [rad]

            // ==========================================================
            // Frame rotation: RI → virtual dq (theta_s frame)
            //   The voltage controller must see voltages/currents in the
            //   theta_s frame so that vqs → 0 and vds → |V|.
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 2: RPC — Reactive Power Controller
            //   Vref = V_nom + DQ·(Q*−Qs) + ϕQs (spec §4)
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            // Clamp phi_Qs: RPC is a trim around V_nom, ±0.1 pu is sufficient
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3: Virtual impedance (theta_s frame)
            // ==========================================================
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // ==========================================================
            // STEP 4: Outer voltage loop
            //   Virtual EMF = V_term + Zvir·Ir → imag part of drop → q-axis, real → d-axis
            //   e_vqs = −(vqs_m + vaux_qs)    drive q-axis to 0
            //   e_vds =  (vds_m − Vref + vaux_ds)  drive d-axis to Vref
            // ==========================================================
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;
            dxdt[3] = e_vqs;
            dxdt[4] = e_vds;

            // ==========================================================
            // STEP 5: Current references from voltage loop
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            // ==========================================================
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // ==========================================================
            // STEP 6: Current saturation (circular limiter)
            // ==========================================================
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // ==========================================================
            // STEP 7: Inner current loop errors (theta_s frame)
            // ==========================================================
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // ==========================================================
            // STEP 8: Decoupling (theta_s frame)
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            // ==========================================================
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_slip = omega_vs - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // ==========================================================
            // STEP 9: Rotor voltage commands (theta_s frame) + saturation
            // ==========================================================
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // ==========================================================
            // STEP 10: Inner PI integrators with anti-windup
            // ==========================================================
            bool Vrd_sat = (fabs(Vrd_raw) > Vrd_max);
            bool Vrq_sat = (fabs(Vrq_raw) > Vrd_max);

            if (Vrd_sat && ((Vrd_raw > 0.0 && e_ird > 0.0) || (Vrd_raw < 0.0 && e_ird < 0.0)))
                dxdt[5] = 0.0;
            else
                dxdt[5] = e_ird;

            if (Vrq_sat && ((Vrq_raw > 0.0 && e_iqr > 0.0) || (Vrq_raw < 0.0 && e_iqr < 0.0)))
                dxdt[6] = 0.0;
            else
                dxdt[6] = e_iqr;

            if (cur_sat) {
                if ((x[3] > 0.0 && e_vqs > 0.0) || (x[3] < 0.0 && e_vqs < 0.0))
                    dxdt[3] = 0.0;
                if ((x[4] > 0.0 && e_vds > 0.0) || (x[4] < 0.0 && e_vds < 0.0))
                    dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 11: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];  // theta_s → to GSC
        
}
void step_DFIG_RSC_GFM_PHS_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.1537832542129938;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- STEP 2: RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];

            // --- STEP 3: Virtual impedance (theta_s frame) ---
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // --- STEP 4: Voltage loop errors ---
            // Virtual EMF = V_term + Zvir·Ir → real part of drop → d-axis, imag → q-axis
            // e_vqs = -(vqs_m + vaux_qs)   drive q-axis voltage to 0
            // e_vds =  (vds_m - Vref + vaux_ds)  drive d-axis to Vref
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;

            // --- STEP 5: Current references from voltage loop ---
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // --- STEP 6: Current saturation (circular limiter) ---
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // --- STEP 7: Inner current loop errors (theta_s frame) ---
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // --- STEP 8: Decoupling (theta_s frame) ---
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // --- STEP 9: Rotor voltage commands (theta_s frame) ---
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        
}
void step_DFIG_DCLink_PHS_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc   = fmax(x[0], 0.01);
            double P_rsc = inputs[0];
            double P_gsc = inputs[1];

            // DC-link current
            double P_net = P_rsc - P_gsc;
            double i_dc  = P_net / Vdc;

            // ESR power loss
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper brake (overvoltage protection)
            double P_brake = 0.0;
            if (Vdc > 1.15 * Vdc_nom) {
                P_brake = (Vdc - 1.15 * Vdc_nom) * 10.0;
            }

            // C_dc · dVdc/dt = (P_rsc - P_gsc - P_loss - P_brake) / Vdc
            dxdt[0] = (P_net - P_loss - P_brake) / (C_dc * Vdc);
        
}
void step_DFIG_DCLink_PHS_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc = fmax(x[0], 0.01);
            outputs[0] = Vdc;
            outputs[1] = inputs[0] - inputs[1];  // P_net = P_rsc - P_gsc
        
}
void step_DFIG_GSC_GFM_PHS_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];
            // Note: inputs[5] = theta_s from RSC APC, used by system layer only

            // ---- PI controller: Vdc loop → active power command ----
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref
            double P_cmd = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → reactive power command ----
            double Q_meas = Vq * i_fd - Vd * i_fq;
            double e_Q = inputs[2] - Q_meas;  // Qref - Q_meas
            double Q_neg_cmd = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Power-projection: decouple P/Q control from voltage angle ----
            // From S = V·conj(I): Id=(Vd·P+Vq·Q)/V², Iq=(Vq·P−Vd·Q)/V²
            // Q_neg_cmd tracks (Qref−Q_meas) > 0 when Q below ref → more injection
            // Injection requires Iq < 0, so Iq = −Q_neg_cmd at Vd≈1,Vq≈0: sign is −Vd·Q
            double V_sq = fmax(Vd * Vd + Vq * Vq, 0.01);
            double Id_ref = (Vd * P_cmd + Vq * Q_neg_cmd) / V_sq;
            double Iq_ref = (Vq * P_cmd - Vd * Q_neg_cmd) / V_sq;

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((P_cmd > 0.0 && e_Vdc < 0.0) || (P_cmd < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Q_neg_cmd > 0.0 && e_Q < 0.0) || (Q_neg_cmd < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        
}
void step_DFIG_GSC_GFM_PHS_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        
}
void step_DFIG_DT_PHS_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.752713202793482;

            // ---- States ----
            double omega_t  = x[0] / fmax(H_t, 1e-6);
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Inputs ----
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double vw_pu   = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Shaft torque ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // ---- Aerodynamic torque (per-unit, uses dynamic Cp_eff) ----
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Empirical Cp (quasi-static target, per-unit lambda) ----
            double lambda_pu = vw_pu / omega_t_safe;
            double lambda_i = 1.0 / (1.0 / (lambda_pu + 0.08 * beta) - 0.035 / (beta * beta + 1.0));
            double Cp_raw = 0.5176 * (116.0 * lambda_i - 0.4 * beta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * lambda_pu;

            // Base Cp at rated (vw=1, omega=1, beta=0)
            double lambda_i_0 = 1.0 / (1.0 - 0.035);
            double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
            double Cp_emp = fmax(0.0, Cp_raw / fmax(Cp_0, 1e-6));
            if (Cp_emp > 2.0) Cp_emp = 2.0;

            // ---- Dynamic inflow time constant ----
            double tau_i = 1.0 / fmax(vw_pu, 0.1);

            // ---- PHS dynamics ----
            dxdt[0] = T_aero - T_shaft - D_t * omega_t;
            dxdt[1] = (Cp_emp - Cp_eff_val) / fmax(tau_i, 0.01);
            dxdt[2] = omega_t - omega_g;
        
}
void step_DFIG_DT_PHS_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.752713202793482;

            double omega_t = x[0] / fmax(H_t, 1e-6);
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Per-unit wind speed ----
            double vw_pu = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Aerodynamic torque from dynamic Cp_eff (per-unit) ----
            // P_aero = Tm0 * vw_pu³ * Cp_eff_val,  T_aero = P_aero / omega_t
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Shaft torque: spring + damping ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            outputs[0] = T_shaft;
            outputs[1] = omega_t;
            outputs[2] = T_aero;
            outputs[3] = theta_tw;
        
}
void step_DFIG_PHS_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Inputs ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];
            double T_shaft = inputs[2];
            double Vrd = inputs[3];
            double Vrq = inputs[4];

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_g    = x[4];

            // ---- Hamiltonian gradient: ∇H₃ = M⁻¹ x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_g = p_g / j_inertia;

            // ---- J₃(x)·∇H₃ (skew-symmetric structure) ----
            // Synchronous-frame rotation + electromechanical gyrator
            double J0 =  omega_s * Ls * i_sq  + omega_s * Lm * i_rq;
            double J1 = -omega_s * Ls * i_sd  - omega_s * Lm * i_rd;
            double J2 =  omega_s * Lm * i_sq  + omega_s * Lr * i_rq - np * phi_rq * omega_g;
            double J3 = -omega_s * Lm * i_sd  - omega_s * Lr * i_rd + np * phi_rd * omega_g;
            double J4 =  np * (phi_rq * i_rd - phi_rd * i_rq);  // = -Te (skew row)

            // ---- R₃·∇H₃ (dissipation) ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * omega_g;

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PHS dynamics: ẋ = (1/ωb)(J₃ − R₃)∇H₃ + (1/ωb)g₃·u₃ ----
            // Flux eqs scaled by ωb; mechanical eq in seconds directly
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_shaft;  // = -Te - f_damp*ω + T_shaft

            // ---- Outputs (power, monitoring) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = Vrd * i_rd + Vrq * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
        
}
void step_DFIG_PHS_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Currents from flux states: ∇H₃ = M⁻¹·x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * x[0] - Lm * x[2]) / sigma_LS;
            double i_sq =  (Lr * x[1] - Lm * x[3]) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;
            double omega_g = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Voltages ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Power (generator convention: positive = injected) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // ---- Norton equivalent: I_N = I_gen + Y_N × V ----
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            outputs[0] = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            outputs[1] = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[2] = omega_g;
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = x[0];
            outputs[12] = x[1];
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);
        
}
void step_DFIG_RSC_GFM_PHS_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.191457260775298;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //   J·ω̇s = P*s − Ps − Dp·(ωs − ωN)
            //   θ̇s   = ωs
            // ==========================================================
            double omega_vs = x[0];
            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)) / J;
            dxdt[1] = omega_vs - omega_N;  // virtual angle deviation [rad]

            // ==========================================================
            // Frame rotation: RI → virtual dq (theta_s frame)
            //   The voltage controller must see voltages/currents in the
            //   theta_s frame so that vqs → 0 and vds → |V|.
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 2: RPC — Reactive Power Controller
            //   Vref = V_nom + DQ·(Q*−Qs) + ϕQs (spec §4)
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            // Clamp phi_Qs: RPC is a trim around V_nom, ±0.1 pu is sufficient
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3: Virtual impedance (theta_s frame)
            // ==========================================================
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // ==========================================================
            // STEP 4: Outer voltage loop
            //   Virtual EMF = V_term + Zvir·Ir → imag part of drop → q-axis, real → d-axis
            //   e_vqs = −(vqs_m + vaux_qs)    drive q-axis to 0
            //   e_vds =  (vds_m − Vref + vaux_ds)  drive d-axis to Vref
            // ==========================================================
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;
            dxdt[3] = e_vqs;
            dxdt[4] = e_vds;

            // ==========================================================
            // STEP 5: Current references from voltage loop
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            // ==========================================================
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // ==========================================================
            // STEP 6: Current saturation (circular limiter)
            // ==========================================================
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // ==========================================================
            // STEP 7: Inner current loop errors (theta_s frame)
            // ==========================================================
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // ==========================================================
            // STEP 8: Decoupling (theta_s frame)
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            // ==========================================================
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_slip = omega_vs - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // ==========================================================
            // STEP 9: Rotor voltage commands (theta_s frame) + saturation
            // ==========================================================
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // ==========================================================
            // STEP 10: Inner PI integrators with anti-windup
            // ==========================================================
            bool Vrd_sat = (fabs(Vrd_raw) > Vrd_max);
            bool Vrq_sat = (fabs(Vrq_raw) > Vrd_max);

            if (Vrd_sat && ((Vrd_raw > 0.0 && e_ird > 0.0) || (Vrd_raw < 0.0 && e_ird < 0.0)))
                dxdt[5] = 0.0;
            else
                dxdt[5] = e_ird;

            if (Vrq_sat && ((Vrq_raw > 0.0 && e_iqr > 0.0) || (Vrq_raw < 0.0 && e_iqr < 0.0)))
                dxdt[6] = 0.0;
            else
                dxdt[6] = e_iqr;

            if (cur_sat) {
                if ((x[3] > 0.0 && e_vqs > 0.0) || (x[3] < 0.0 && e_vqs < 0.0))
                    dxdt[3] = 0.0;
                if ((x[4] > 0.0 && e_vds > 0.0) || (x[4] < 0.0 && e_vds < 0.0))
                    dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 11: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];  // theta_s → to GSC
        
}
void step_DFIG_RSC_GFM_PHS_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.191457260775298;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- STEP 2: RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];

            // --- STEP 3: Virtual impedance (theta_s frame) ---
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // --- STEP 4: Voltage loop errors ---
            // Virtual EMF = V_term + Zvir·Ir → real part of drop → d-axis, imag → q-axis
            // e_vqs = -(vqs_m + vaux_qs)   drive q-axis voltage to 0
            // e_vds =  (vds_m - Vref + vaux_ds)  drive d-axis to Vref
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;

            // --- STEP 5: Current references from voltage loop ---
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // --- STEP 6: Current saturation (circular limiter) ---
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // --- STEP 7: Inner current loop errors (theta_s frame) ---
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // --- STEP 8: Decoupling (theta_s frame) ---
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // --- STEP 9: Rotor voltage commands (theta_s frame) ---
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        
}
void step_DFIG_DCLink_PHS_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc   = fmax(x[0], 0.01);
            double P_rsc = inputs[0];
            double P_gsc = inputs[1];

            // DC-link current
            double P_net = P_rsc - P_gsc;
            double i_dc  = P_net / Vdc;

            // ESR power loss
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper brake (overvoltage protection)
            double P_brake = 0.0;
            if (Vdc > 1.15 * Vdc_nom) {
                P_brake = (Vdc - 1.15 * Vdc_nom) * 10.0;
            }

            // C_dc · dVdc/dt = (P_rsc - P_gsc - P_loss - P_brake) / Vdc
            dxdt[0] = (P_net - P_loss - P_brake) / (C_dc * Vdc);
        
}
void step_DFIG_DCLink_PHS_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc = fmax(x[0], 0.01);
            outputs[0] = Vdc;
            outputs[1] = inputs[0] - inputs[1];  // P_net = P_rsc - P_gsc
        
}
void step_DFIG_GSC_GFM_PHS_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];
            // Note: inputs[5] = theta_s from RSC APC, used by system layer only

            // ---- PI controller: Vdc loop → active power command ----
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref
            double P_cmd = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → reactive power command ----
            double Q_meas = Vq * i_fd - Vd * i_fq;
            double e_Q = inputs[2] - Q_meas;  // Qref - Q_meas
            double Q_neg_cmd = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Power-projection: decouple P/Q control from voltage angle ----
            // From S = V·conj(I): Id=(Vd·P+Vq·Q)/V², Iq=(Vq·P−Vd·Q)/V²
            // Q_neg_cmd tracks (Qref−Q_meas) > 0 when Q below ref → more injection
            // Injection requires Iq < 0, so Iq = −Q_neg_cmd at Vd≈1,Vq≈0: sign is −Vd·Q
            double V_sq = fmax(Vd * Vd + Vq * Vq, 0.01);
            double Id_ref = (Vd * P_cmd + Vq * Q_neg_cmd) / V_sq;
            double Iq_ref = (Vq * P_cmd - Vd * Q_neg_cmd) / V_sq;

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((P_cmd > 0.0 && e_Vdc < 0.0) || (P_cmd < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Q_neg_cmd > 0.0 && e_Q < 0.0) || (Q_neg_cmd < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        
}
void step_DFIG_GSC_GFM_PHS_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        
}
void step_DFIG_DT_PHS_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.652479985843298;

            // ---- States ----
            double omega_t  = x[0] / fmax(H_t, 1e-6);
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Inputs ----
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double vw_pu   = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Shaft torque ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // ---- Aerodynamic torque (per-unit, uses dynamic Cp_eff) ----
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Empirical Cp (quasi-static target, per-unit lambda) ----
            double lambda_pu = vw_pu / omega_t_safe;
            double lambda_i = 1.0 / (1.0 / (lambda_pu + 0.08 * beta) - 0.035 / (beta * beta + 1.0));
            double Cp_raw = 0.5176 * (116.0 * lambda_i - 0.4 * beta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * lambda_pu;

            // Base Cp at rated (vw=1, omega=1, beta=0)
            double lambda_i_0 = 1.0 / (1.0 - 0.035);
            double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
            double Cp_emp = fmax(0.0, Cp_raw / fmax(Cp_0, 1e-6));
            if (Cp_emp > 2.0) Cp_emp = 2.0;

            // ---- Dynamic inflow time constant ----
            double tau_i = 1.0 / fmax(vw_pu, 0.1);

            // ---- PHS dynamics ----
            dxdt[0] = T_aero - T_shaft - D_t * omega_t;
            dxdt[1] = (Cp_emp - Cp_eff_val) / fmax(tau_i, 0.01);
            dxdt[2] = omega_t - omega_g;
        
}
void step_DFIG_DT_PHS_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.652479985843298;

            double omega_t = x[0] / fmax(H_t, 1e-6);
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Per-unit wind speed ----
            double vw_pu = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Aerodynamic torque from dynamic Cp_eff (per-unit) ----
            // P_aero = Tm0 * vw_pu³ * Cp_eff_val,  T_aero = P_aero / omega_t
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Shaft torque: spring + damping ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            outputs[0] = T_shaft;
            outputs[1] = omega_t;
            outputs[2] = T_aero;
            outputs[3] = theta_tw;
        
}
void step_DFIG_PHS_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Inputs ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];
            double T_shaft = inputs[2];
            double Vrd = inputs[3];
            double Vrq = inputs[4];

            // ---- States ----
            double phi_sd = x[0];
            double phi_sq = x[1];
            double phi_rd = x[2];
            double phi_rq = x[3];
            double p_g    = x[4];

            // ---- Hamiltonian gradient: ∇H₃ = M⁻¹ x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * phi_sd - Lm * phi_rd) / sigma_LS;
            double i_sq =  (Lr * phi_sq - Lm * phi_rq) / sigma_LS;
            double i_rd = (-Lm * phi_sd + Ls * phi_rd) / sigma_LS;
            double i_rq = (-Lm * phi_sq + Ls * phi_rq) / sigma_LS;
            double omega_g = p_g / j_inertia;

            // ---- J₃(x)·∇H₃ (skew-symmetric structure) ----
            // Synchronous-frame rotation + electromechanical gyrator
            double J0 =  omega_s * Ls * i_sq  + omega_s * Lm * i_rq;
            double J1 = -omega_s * Ls * i_sd  - omega_s * Lm * i_rd;
            double J2 =  omega_s * Lm * i_sq  + omega_s * Lr * i_rq - np * phi_rq * omega_g;
            double J3 = -omega_s * Lm * i_sd  - omega_s * Lr * i_rd + np * phi_rd * omega_g;
            double J4 =  np * (phi_rq * i_rd - phi_rd * i_rq);  // = -Te (skew row)

            // ---- R₃·∇H₃ (dissipation) ----
            double R0 = Rs * i_sd;
            double R1 = Rs * i_sq;
            double R2 = Rr * i_rd;
            double R3 = Rr * i_rq;
            double R4 = f_damp * omega_g;

            // ---- Electromagnetic torque ----
            double Te = np * (phi_rd * i_rq - phi_rq * i_rd);

            // ---- PHS dynamics: ẋ = (1/ωb)(J₃ − R₃)∇H₃ + (1/ωb)g₃·u₃ ----
            // Flux eqs scaled by ωb; mechanical eq in seconds directly
            dxdt[0] = omega_b * (J0 - R0 + Vsd);
            dxdt[1] = omega_b * (J1 - R1 + Vsq);
            dxdt[2] = omega_b * (J2 - R2 + Vrd);
            dxdt[3] = omega_b * (J3 - R3 + Vrq);
            dxdt[4] = J4 - R4 + T_shaft;  // = -Te - f_damp*ω + T_shaft

            // ---- Outputs (power, monitoring) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = phi_sd;
            outputs[12] = phi_sq;
            outputs[13] = Vrd * i_rd + Vrq * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
        
}
void step_DFIG_PHS_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double Lm = 3.4;
    const double Rs = 0.01;
    const double Rr = 0.01;
    const double j_inertia = 1.0;
    const double f_damp = 0.2;
    const double np = 1;
    const double omega_b = 376.991;
    const double omega_s = 1.0;
    // --- Kernel ---

            // ---- Currents from flux states: ∇H₃ = M⁻¹·x₃ ----
            double sigma_LS = Ls * Lr - Lm * Lm;
            double i_sd =  (Lr * x[0] - Lm * x[2]) / sigma_LS;
            double i_sq =  (Lr * x[1] - Lm * x[3]) / sigma_LS;
            double i_rd = (-Lm * x[0] + Ls * x[2]) / sigma_LS;
            double i_rq = (-Lm * x[1] + Ls * x[3]) / sigma_LS;
            double omega_g = x[4] / j_inertia;

            // ---- Electromagnetic torque ----
            double Te = np * (x[2] * i_rq - x[3] * i_rd);

            // ---- Voltages ----
            double Vsd = inputs[0];
            double Vsq = inputs[1];

            // ---- Power (generator convention: positive = injected) ----
            double Pe = -(Vsd * i_sd + Vsq * i_sq);
            double Qe = -(Vsq * i_sd - Vsd * i_sq);

            // ---- Norton equivalent: I_N = I_gen + Y_N × V ----
            double Igen_re = -i_sd;
            double Igen_im = -i_sq;
            double Xs_sig = Ls - Lm * Lm / Lr;
            double Zmag2  = Rs * Rs + Xs_sig * Xs_sig;
            outputs[0] = Igen_re + (Rs * Vsd + Xs_sig * Vsq) / Zmag2;
            outputs[1] = Igen_im + (Rs * Vsq - Xs_sig * Vsd) / Zmag2;
            outputs[2] = omega_g;
            outputs[3] = Pe;
            outputs[4] = Qe;
            outputs[5] = i_sd;
            outputs[6] = i_sq;
            outputs[7] = -i_sd;
            outputs[8] = -i_sq;
            outputs[9]  = i_rd;
            outputs[10] = i_rq;
            outputs[11] = x[0];
            outputs[12] = x[1];
            outputs[13] = inputs[3] * i_rd + inputs[4] * i_rq;
            outputs[14] = (omega_s - omega_g) / omega_s;
            outputs[15] = Te;
            outputs[16] = sqrt(Vsd * Vsd + Vsq * Vsq);
        
}
void step_DFIG_RSC_GFM_PHS_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.155459906573626;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //   J·ω̇s = P*s − Ps − Dp·(ωs − ωN)
            //   θ̇s   = ωs
            // ==========================================================
            double omega_vs = x[0];
            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)) / J;
            dxdt[1] = omega_vs - omega_N;  // virtual angle deviation [rad]

            // ==========================================================
            // Frame rotation: RI → virtual dq (theta_s frame)
            //   The voltage controller must see voltages/currents in the
            //   theta_s frame so that vqs → 0 and vds → |V|.
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 2: RPC — Reactive Power Controller
            //   Vref = V_nom + DQ·(Q*−Qs) + ϕQs (spec §4)
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            // Clamp phi_Qs: RPC is a trim around V_nom, ±0.1 pu is sufficient
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3: Virtual impedance (theta_s frame)
            // ==========================================================
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // ==========================================================
            // STEP 4: Outer voltage loop
            //   Virtual EMF = V_term + Zvir·Ir → imag part of drop → q-axis, real → d-axis
            //   e_vqs = −(vqs_m + vaux_qs)    drive q-axis to 0
            //   e_vds =  (vds_m − Vref + vaux_ds)  drive d-axis to Vref
            // ==========================================================
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;
            dxdt[3] = e_vqs;
            dxdt[4] = e_vds;

            // ==========================================================
            // STEP 5: Current references from voltage loop
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            // ==========================================================
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // ==========================================================
            // STEP 6: Current saturation (circular limiter)
            // ==========================================================
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // ==========================================================
            // STEP 7: Inner current loop errors (theta_s frame)
            // ==========================================================
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // ==========================================================
            // STEP 8: Decoupling (theta_s frame)
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            // ==========================================================
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_slip = omega_vs - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // ==========================================================
            // STEP 9: Rotor voltage commands (theta_s frame) + saturation
            // ==========================================================
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // ==========================================================
            // STEP 10: Inner PI integrators with anti-windup
            // ==========================================================
            bool Vrd_sat = (fabs(Vrd_raw) > Vrd_max);
            bool Vrq_sat = (fabs(Vrq_raw) > Vrd_max);

            if (Vrd_sat && ((Vrd_raw > 0.0 && e_ird > 0.0) || (Vrd_raw < 0.0 && e_ird < 0.0)))
                dxdt[5] = 0.0;
            else
                dxdt[5] = e_ird;

            if (Vrq_sat && ((Vrq_raw > 0.0 && e_iqr > 0.0) || (Vrq_raw < 0.0 && e_iqr < 0.0)))
                dxdt[6] = 0.0;
            else
                dxdt[6] = e_iqr;

            if (cur_sat) {
                if ((x[3] > 0.0 && e_vqs > 0.0) || (x[3] < 0.0 && e_vqs < 0.0))
                    dxdt[3] = 0.0;
                if ((x[4] > 0.0 && e_vds > 0.0) || (x[4] < 0.0 && e_vds < 0.0))
                    dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 11: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];  // theta_s → to GSC
        
}
void step_DFIG_RSC_GFM_PHS_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double J = 5.0;
    const double Dp = 20.0;
    const double omega_N = 1.0;
    const double DQ = 0.05;
    const double kQs = 20.0;
    const double Rvir = 0.1;
    const double Xvir = 0.2;
    const double KpPs = 1.0;
    const double KiPs = 2.0;
    const double KpQs = 0.5;
    const double KiQs = 10.0;
    const double Kp_i = 5.0;
    const double Ki_i = 50.0;
    const double Lm = 3.4;
    const double Ls = 3.5;
    const double Lr = 3.5;
    const double omega_s = 1.0;
    const double Vrd_max = 0.5;
    const double I_max = 1.2;
    const double omega_b = 376.991;
    const double V_nom = 1.155459906573626;
    // --- Kernel ---

            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- STEP 2: RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];

            // --- STEP 3: Virtual impedance (theta_s frame) ---
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // --- STEP 4: Voltage loop errors ---
            // Virtual EMF = V_term + Zvir·Ir → real part of drop → d-axis, imag → q-axis
            // e_vqs = -(vqs_m + vaux_qs)   drive q-axis voltage to 0
            // e_vds =  (vds_m - Vref + vaux_ds)  drive d-axis to Vref
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;

            // --- STEP 5: Current references from voltage loop ---
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // --- STEP 6: Current saturation (circular limiter) ---
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // --- STEP 7: Inner current loop errors (theta_s frame) ---
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // --- STEP 8: Decoupling (theta_s frame) ---
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // --- STEP 9: Rotor voltage commands (theta_s frame) ---
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        
}
void step_DFIG_DCLink_PHS_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc   = fmax(x[0], 0.01);
            double P_rsc = inputs[0];
            double P_gsc = inputs[1];

            // DC-link current
            double P_net = P_rsc - P_gsc;
            double i_dc  = P_net / Vdc;

            // ESR power loss
            double P_loss = R_esr * i_dc * i_dc;

            // DC chopper brake (overvoltage protection)
            double P_brake = 0.0;
            if (Vdc > 1.15 * Vdc_nom) {
                P_brake = (Vdc - 1.15 * Vdc_nom) * 10.0;
            }

            // C_dc · dVdc/dt = (P_rsc - P_gsc - P_loss - P_brake) / Vdc
            dxdt[0] = (P_net - P_loss - P_brake) / (C_dc * Vdc);
        
}
void step_DFIG_DCLink_PHS_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double C_dc = 1.0;
    const double Vdc_nom = 1.0;
    const double R_esr = 0.001;
    // --- Kernel ---

            double Vdc = fmax(x[0], 0.01);
            outputs[0] = Vdc;
            outputs[1] = inputs[0] - inputs[1];  // P_net = P_rsc - P_gsc
        
}
void step_DFIG_GSC_GFM_PHS_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];
            // Note: inputs[5] = theta_s from RSC APC, used by system layer only

            // ---- PI controller: Vdc loop → active power command ----
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref
            double P_cmd = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → reactive power command ----
            double Q_meas = Vq * i_fd - Vd * i_fq;
            double e_Q = inputs[2] - Q_meas;  // Qref - Q_meas
            double Q_neg_cmd = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Power-projection: decouple P/Q control from voltage angle ----
            // From S = V·conj(I): Id=(Vd·P+Vq·Q)/V², Iq=(Vq·P−Vd·Q)/V²
            // Q_neg_cmd tracks (Qref−Q_meas) > 0 when Q below ref → more injection
            // Injection requires Iq < 0, so Iq = −Q_neg_cmd at Vd≈1,Vq≈0: sign is −Vd·Q
            double V_sq = fmax(Vd * Vd + Vq * Vq, 0.01);
            double Id_ref = (Vd * P_cmd + Vq * Q_neg_cmd) / V_sq;
            double Iq_ref = (Vq * P_cmd - Vd * Q_neg_cmd) / V_sq;

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((P_cmd > 0.0 && e_Vdc < 0.0) || (P_cmd < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Q_neg_cmd > 0.0 && e_Q < 0.0) || (Q_neg_cmd < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        
}
void step_DFIG_GSC_GFM_PHS_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double L_f = 0.15;
    const double R_f = 0.003;
    const double omega_s = 1.0;
    const double omega_b = 376.991;
    const double Kp_dc = 1.0;
    const double Ki_dc = 8.0;
    const double Kp_Qg = 0.5;
    const double Ki_Qg = 5.0;
    const double I_max = 0.8;
    // --- Kernel ---

            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        
}
void step_DFIG_DT_PHS_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.6013596305674611;

            // ---- States ----
            double omega_t  = x[0] / fmax(H_t, 1e-6);
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Inputs ----
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double vw_pu   = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Shaft torque ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            // ---- Aerodynamic torque (per-unit, uses dynamic Cp_eff) ----
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Empirical Cp (quasi-static target, per-unit lambda) ----
            double lambda_pu = vw_pu / omega_t_safe;
            double lambda_i = 1.0 / (1.0 / (lambda_pu + 0.08 * beta) - 0.035 / (beta * beta + 1.0));
            double Cp_raw = 0.5176 * (116.0 * lambda_i - 0.4 * beta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * lambda_pu;

            // Base Cp at rated (vw=1, omega=1, beta=0)
            double lambda_i_0 = 1.0 / (1.0 - 0.035);
            double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
            double Cp_emp = fmax(0.0, Cp_raw / fmax(Cp_0, 1e-6));
            if (Cp_emp > 2.0) Cp_emp = 2.0;

            // ---- Dynamic inflow time constant ----
            double tau_i = 1.0 / fmax(vw_pu, 0.1);

            // ---- PHS dynamics ----
            dxdt[0] = T_aero - T_shaft - D_t * omega_t;
            dxdt[1] = (Cp_emp - Cp_eff_val) / fmax(tau_i, 0.01);
            dxdt[2] = omega_t - omega_g;
        
}
void step_DFIG_DT_PHS_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double H_t = 4.0;
    const double K_shaft = 0.5;
    const double D_shaft = 1.5;
    const double D_t = 0.05;
    const double k_cp = 2.0;
    const double gear_ratio = 1.0;
    const double vw_nom = 12.0;
    // --- Kernel ---
double Tm0 = 0.6013596305674611;

            double omega_t = x[0] / fmax(H_t, 1e-6);
            double omega_g = inputs[0];
            double V_w     = fmax(inputs[1], 0.1);
            double beta    = inputs[2];
            double Cp_eff_val = x[1];
            double theta_tw = x[2];

            // ---- Per-unit wind speed ----
            double vw_pu = V_w / fmax(vw_nom, 0.1);
            double omega_t_safe = fmax(fabs(omega_t), 0.01);

            // ---- Aerodynamic torque from dynamic Cp_eff (per-unit) ----
            // P_aero = Tm0 * vw_pu³ * Cp_eff_val,  T_aero = P_aero / omega_t
            double P_aero = Tm0 * vw_pu * vw_pu * vw_pu * Cp_eff_val;
            double T_aero = P_aero / omega_t_safe;

            // ---- Shaft torque: spring + damping ----
            double T_shaft = K_shaft * theta_tw + D_shaft * (omega_t - omega_g);

            outputs[0] = T_shaft;
            outputs[1] = omega_t;
            outputs[2] = T_aero;
            outputs[3] = theta_tw;
        
}
void dae_residual(const double* y, const double* ydot, double* res, double t) {
    // --- Extract state partitions ---
    const double* x = y;
    double Vd_net[14], Vq_net[14], Vterm_net[14];
    for (int i = 0; i < N_BUS; ++i) {
        Vd_net[i]   = y[N_DIFF + 2*i];
        Vq_net[i]   = y[N_DIFF + 2*i + 1];
        Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
    }

    double Id_inj[14] = {0};
    double Iq_inj[14] = {0};

    // --- 1. Compute Outputs & Gather Injections ---
    { // GENCLS_1
        inputs_GENCLS_1[0] = Vd_net[0]; // Vd
        inputs_GENCLS_1[1] = Vq_net[0]; // Vq
        inputs_GENCLS_1[2] = outputs_TGOV1_1[0]; // Tm
        step_GENCLS_1_out(&x[0], inputs_GENCLS_1, outputs_GENCLS_1, t);
        Id_inj[0] += outputs_GENCLS_1[0];
        Iq_inj[0] += outputs_GENCLS_1[1];
    }
    { // TGOV1_1
        inputs_TGOV1_1[0] = outputs_GENCLS_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0433170946165862; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1_out(&x[2], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // DFIG_PHS_1
        inputs_DFIG_PHS_1[0] = Vd_net[1]; // Vd
        inputs_DFIG_PHS_1[1] = Vq_net[1]; // Vq
        inputs_DFIG_PHS_1[2] = outputs_DFIG_DT_PHS_1[0]; // Tm
        inputs_DFIG_PHS_1[3] = outputs_DFIG_RSC_GFM_PHS_1[0]; // Vrd
        inputs_DFIG_PHS_1[4] = outputs_DFIG_RSC_GFM_PHS_1[1]; // Vrq
        step_DFIG_PHS_1_out(&x[5], inputs_DFIG_PHS_1, outputs_DFIG_PHS_1, t);
        Id_inj[1] += outputs_DFIG_PHS_1[0];
        Iq_inj[1] += outputs_DFIG_PHS_1[1];
    }
    { // DFIG_RSC_GFM_PHS_1
        inputs_DFIG_RSC_GFM_PHS_1[0] = 0.5; // P_star
        inputs_DFIG_RSC_GFM_PHS_1[1] = 0.26; // Q_star
        inputs_DFIG_RSC_GFM_PHS_1[2] = outputs_DFIG_PHS_1[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_1[3] = outputs_DFIG_PHS_1[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_1[4] = Vd_net[1]; // vds
        inputs_DFIG_RSC_GFM_PHS_1[5] = Vq_net[1]; // vqs
        inputs_DFIG_RSC_GFM_PHS_1[6] = outputs_DFIG_PHS_1[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_1[7] = outputs_DFIG_PHS_1[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_1[8] = outputs_DFIG_PHS_1[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_1[9] = outputs_DFIG_PHS_1[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_1[10] = outputs_DFIG_PHS_1[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_1_out(&x[10], inputs_DFIG_RSC_GFM_PHS_1, outputs_DFIG_RSC_GFM_PHS_1, t);
    }
    { // DFIG_DCLink_PHS_1
        inputs_DFIG_DCLink_PHS_1[0] = outputs_DFIG_RSC_GFM_PHS_1[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_1[1] = outputs_DFIG_GSC_GFM_PHS_1[2]; // P_gsc
        step_DFIG_DCLink_PHS_1_out(&x[17], inputs_DFIG_DCLink_PHS_1, outputs_DFIG_DCLink_PHS_1, t);
    }
    { // DFIG_GSC_GFM_PHS_1
        inputs_DFIG_GSC_GFM_PHS_1[0] = outputs_DFIG_DCLink_PHS_1[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_1[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_1[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_1[3] = Vd_net[1]; // Vd
        inputs_DFIG_GSC_GFM_PHS_1[4] = Vq_net[1]; // Vq
        inputs_DFIG_GSC_GFM_PHS_1[5] = outputs_DFIG_RSC_GFM_PHS_1[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_1_out(&x[18], inputs_DFIG_GSC_GFM_PHS_1, outputs_DFIG_GSC_GFM_PHS_1, t);
        Id_inj[1] += outputs_DFIG_GSC_GFM_PHS_1[0];
        Iq_inj[1] += outputs_DFIG_GSC_GFM_PHS_1[1];
    }
    { // DFIG_DT_PHS_1
        inputs_DFIG_DT_PHS_1[0] = outputs_DFIG_PHS_1[2]; // omega_g
        inputs_DFIG_DT_PHS_1[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_1[2] = 0.0; // beta
        step_DFIG_DT_PHS_1_out(&x[22], inputs_DFIG_DT_PHS_1, outputs_DFIG_DT_PHS_1, t);
    }
    { // DFIG_PHS_2
        inputs_DFIG_PHS_2[0] = Vd_net[2]; // Vd
        inputs_DFIG_PHS_2[1] = Vq_net[2]; // Vq
        inputs_DFIG_PHS_2[2] = outputs_DFIG_DT_PHS_2[0]; // Tm
        inputs_DFIG_PHS_2[3] = outputs_DFIG_RSC_GFM_PHS_2[0]; // Vrd
        inputs_DFIG_PHS_2[4] = outputs_DFIG_RSC_GFM_PHS_2[1]; // Vrq
        step_DFIG_PHS_2_out(&x[25], inputs_DFIG_PHS_2, outputs_DFIG_PHS_2, t);
        Id_inj[2] += outputs_DFIG_PHS_2[0];
        Iq_inj[2] += outputs_DFIG_PHS_2[1];
    }
    { // DFIG_RSC_GFM_PHS_2
        inputs_DFIG_RSC_GFM_PHS_2[0] = 0.5; // P_star
        inputs_DFIG_RSC_GFM_PHS_2[1] = 0.16; // Q_star
        inputs_DFIG_RSC_GFM_PHS_2[2] = outputs_DFIG_PHS_2[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_2[3] = outputs_DFIG_PHS_2[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_2[4] = Vd_net[2]; // vds
        inputs_DFIG_RSC_GFM_PHS_2[5] = Vq_net[2]; // vqs
        inputs_DFIG_RSC_GFM_PHS_2[6] = outputs_DFIG_PHS_2[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_2[7] = outputs_DFIG_PHS_2[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_2[8] = outputs_DFIG_PHS_2[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_2[9] = outputs_DFIG_PHS_2[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_2[10] = outputs_DFIG_PHS_2[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_2_out(&x[30], inputs_DFIG_RSC_GFM_PHS_2, outputs_DFIG_RSC_GFM_PHS_2, t);
    }
    { // DFIG_DCLink_PHS_2
        inputs_DFIG_DCLink_PHS_2[0] = outputs_DFIG_RSC_GFM_PHS_2[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_2[1] = outputs_DFIG_GSC_GFM_PHS_2[2]; // P_gsc
        step_DFIG_DCLink_PHS_2_out(&x[37], inputs_DFIG_DCLink_PHS_2, outputs_DFIG_DCLink_PHS_2, t);
    }
    { // DFIG_GSC_GFM_PHS_2
        inputs_DFIG_GSC_GFM_PHS_2[0] = outputs_DFIG_DCLink_PHS_2[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_2[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_2[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_2[3] = Vd_net[2]; // Vd
        inputs_DFIG_GSC_GFM_PHS_2[4] = Vq_net[2]; // Vq
        inputs_DFIG_GSC_GFM_PHS_2[5] = outputs_DFIG_RSC_GFM_PHS_2[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_2_out(&x[38], inputs_DFIG_GSC_GFM_PHS_2, outputs_DFIG_GSC_GFM_PHS_2, t);
        Id_inj[2] += outputs_DFIG_GSC_GFM_PHS_2[0];
        Iq_inj[2] += outputs_DFIG_GSC_GFM_PHS_2[1];
    }
    { // DFIG_DT_PHS_2
        inputs_DFIG_DT_PHS_2[0] = outputs_DFIG_PHS_2[2]; // omega_g
        inputs_DFIG_DT_PHS_2[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_2[2] = 0.0; // beta
        step_DFIG_DT_PHS_2_out(&x[42], inputs_DFIG_DT_PHS_2, outputs_DFIG_DT_PHS_2, t);
    }
    { // DFIG_PHS_3
        inputs_DFIG_PHS_3[0] = Vd_net[5]; // Vd
        inputs_DFIG_PHS_3[1] = Vq_net[5]; // Vq
        inputs_DFIG_PHS_3[2] = outputs_DFIG_DT_PHS_3[0]; // Tm
        inputs_DFIG_PHS_3[3] = outputs_DFIG_RSC_GFM_PHS_3[0]; // Vrd
        inputs_DFIG_PHS_3[4] = outputs_DFIG_RSC_GFM_PHS_3[1]; // Vrq
        step_DFIG_PHS_3_out(&x[45], inputs_DFIG_PHS_3, outputs_DFIG_PHS_3, t);
        Id_inj[5] += outputs_DFIG_PHS_3[0];
        Iq_inj[5] += outputs_DFIG_PHS_3[1];
    }
    { // DFIG_RSC_GFM_PHS_3
        inputs_DFIG_RSC_GFM_PHS_3[0] = 0.4; // P_star
        inputs_DFIG_RSC_GFM_PHS_3[1] = 0.32; // Q_star
        inputs_DFIG_RSC_GFM_PHS_3[2] = outputs_DFIG_PHS_3[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_3[3] = outputs_DFIG_PHS_3[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_3[4] = Vd_net[5]; // vds
        inputs_DFIG_RSC_GFM_PHS_3[5] = Vq_net[5]; // vqs
        inputs_DFIG_RSC_GFM_PHS_3[6] = outputs_DFIG_PHS_3[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_3[7] = outputs_DFIG_PHS_3[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_3[8] = outputs_DFIG_PHS_3[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_3[9] = outputs_DFIG_PHS_3[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_3[10] = outputs_DFIG_PHS_3[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_3_out(&x[50], inputs_DFIG_RSC_GFM_PHS_3, outputs_DFIG_RSC_GFM_PHS_3, t);
    }
    { // DFIG_DCLink_PHS_3
        inputs_DFIG_DCLink_PHS_3[0] = outputs_DFIG_RSC_GFM_PHS_3[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_3[1] = outputs_DFIG_GSC_GFM_PHS_3[2]; // P_gsc
        step_DFIG_DCLink_PHS_3_out(&x[57], inputs_DFIG_DCLink_PHS_3, outputs_DFIG_DCLink_PHS_3, t);
    }
    { // DFIG_GSC_GFM_PHS_3
        inputs_DFIG_GSC_GFM_PHS_3[0] = outputs_DFIG_DCLink_PHS_3[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_3[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_3[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_3[3] = Vd_net[5]; // Vd
        inputs_DFIG_GSC_GFM_PHS_3[4] = Vq_net[5]; // Vq
        inputs_DFIG_GSC_GFM_PHS_3[5] = outputs_DFIG_RSC_GFM_PHS_3[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_3_out(&x[58], inputs_DFIG_GSC_GFM_PHS_3, outputs_DFIG_GSC_GFM_PHS_3, t);
        Id_inj[5] += outputs_DFIG_GSC_GFM_PHS_3[0];
        Iq_inj[5] += outputs_DFIG_GSC_GFM_PHS_3[1];
    }
    { // DFIG_DT_PHS_3
        inputs_DFIG_DT_PHS_3[0] = outputs_DFIG_PHS_3[2]; // omega_g
        inputs_DFIG_DT_PHS_3[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_3[2] = 0.0; // beta
        step_DFIG_DT_PHS_3_out(&x[62], inputs_DFIG_DT_PHS_3, outputs_DFIG_DT_PHS_3, t);
    }
    { // DFIG_PHS_4
        inputs_DFIG_PHS_4[0] = Vd_net[7]; // Vd
        inputs_DFIG_PHS_4[1] = Vq_net[7]; // Vq
        inputs_DFIG_PHS_4[2] = outputs_DFIG_DT_PHS_4[0]; // Tm
        inputs_DFIG_PHS_4[3] = outputs_DFIG_RSC_GFM_PHS_4[0]; // Vrd
        inputs_DFIG_PHS_4[4] = outputs_DFIG_RSC_GFM_PHS_4[1]; // Vrq
        step_DFIG_PHS_4_out(&x[65], inputs_DFIG_PHS_4, outputs_DFIG_PHS_4, t);
        Id_inj[7] += outputs_DFIG_PHS_4[0];
        Iq_inj[7] += outputs_DFIG_PHS_4[1];
    }
    { // DFIG_RSC_GFM_PHS_4
        inputs_DFIG_RSC_GFM_PHS_4[0] = 0.35; // P_star
        inputs_DFIG_RSC_GFM_PHS_4[1] = 0.15; // Q_star
        inputs_DFIG_RSC_GFM_PHS_4[2] = outputs_DFIG_PHS_4[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_4[3] = outputs_DFIG_PHS_4[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_4[4] = Vd_net[7]; // vds
        inputs_DFIG_RSC_GFM_PHS_4[5] = Vq_net[7]; // vqs
        inputs_DFIG_RSC_GFM_PHS_4[6] = outputs_DFIG_PHS_4[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_4[7] = outputs_DFIG_PHS_4[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_4[8] = outputs_DFIG_PHS_4[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_4[9] = outputs_DFIG_PHS_4[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_4[10] = outputs_DFIG_PHS_4[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_4_out(&x[70], inputs_DFIG_RSC_GFM_PHS_4, outputs_DFIG_RSC_GFM_PHS_4, t);
    }
    { // DFIG_DCLink_PHS_4
        inputs_DFIG_DCLink_PHS_4[0] = outputs_DFIG_RSC_GFM_PHS_4[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_4[1] = outputs_DFIG_GSC_GFM_PHS_4[2]; // P_gsc
        step_DFIG_DCLink_PHS_4_out(&x[77], inputs_DFIG_DCLink_PHS_4, outputs_DFIG_DCLink_PHS_4, t);
    }
    { // DFIG_GSC_GFM_PHS_4
        inputs_DFIG_GSC_GFM_PHS_4[0] = outputs_DFIG_DCLink_PHS_4[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_4[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_4[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_4[3] = Vd_net[7]; // Vd
        inputs_DFIG_GSC_GFM_PHS_4[4] = Vq_net[7]; // Vq
        inputs_DFIG_GSC_GFM_PHS_4[5] = outputs_DFIG_RSC_GFM_PHS_4[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_4_out(&x[78], inputs_DFIG_GSC_GFM_PHS_4, outputs_DFIG_GSC_GFM_PHS_4, t);
        Id_inj[7] += outputs_DFIG_GSC_GFM_PHS_4[0];
        Iq_inj[7] += outputs_DFIG_GSC_GFM_PHS_4[1];
    }
    { // DFIG_DT_PHS_4
        inputs_DFIG_DT_PHS_4[0] = outputs_DFIG_PHS_4[2]; // omega_g
        inputs_DFIG_DT_PHS_4[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_4[2] = 0.0; // beta
        step_DFIG_DT_PHS_4_out(&x[82], inputs_DFIG_DT_PHS_4, outputs_DFIG_DT_PHS_4, t);
    }

    // --- 2. Compute Dynamics (dxdt) ---
    double dxdt[86];
    { // GENCLS_1 dynamics
        inputs_GENCLS_1[0] = Vd_net[0]; // Vd
        inputs_GENCLS_1[1] = Vq_net[0]; // Vq
        inputs_GENCLS_1[2] = outputs_TGOV1_1[0]; // Tm
        step_GENCLS_1(&x[0], &dxdt[0], inputs_GENCLS_1, outputs_GENCLS_1, t);
    }
    { // TGOV1_1 dynamics
        inputs_TGOV1_1[0] = outputs_GENCLS_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0433170946165862; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1(&x[2], &dxdt[2], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // DFIG_PHS_1 dynamics
        inputs_DFIG_PHS_1[0] = Vd_net[1]; // Vd
        inputs_DFIG_PHS_1[1] = Vq_net[1]; // Vq
        inputs_DFIG_PHS_1[2] = outputs_DFIG_DT_PHS_1[0]; // Tm
        inputs_DFIG_PHS_1[3] = outputs_DFIG_RSC_GFM_PHS_1[0]; // Vrd
        inputs_DFIG_PHS_1[4] = outputs_DFIG_RSC_GFM_PHS_1[1]; // Vrq
        step_DFIG_PHS_1(&x[5], &dxdt[5], inputs_DFIG_PHS_1, outputs_DFIG_PHS_1, t);
    }
    { // DFIG_RSC_GFM_PHS_1 dynamics
        inputs_DFIG_RSC_GFM_PHS_1[0] = 0.5; // P_star
        inputs_DFIG_RSC_GFM_PHS_1[1] = 0.26; // Q_star
        inputs_DFIG_RSC_GFM_PHS_1[2] = outputs_DFIG_PHS_1[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_1[3] = outputs_DFIG_PHS_1[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_1[4] = Vd_net[1]; // vds
        inputs_DFIG_RSC_GFM_PHS_1[5] = Vq_net[1]; // vqs
        inputs_DFIG_RSC_GFM_PHS_1[6] = outputs_DFIG_PHS_1[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_1[7] = outputs_DFIG_PHS_1[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_1[8] = outputs_DFIG_PHS_1[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_1[9] = outputs_DFIG_PHS_1[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_1[10] = outputs_DFIG_PHS_1[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_1(&x[10], &dxdt[10], inputs_DFIG_RSC_GFM_PHS_1, outputs_DFIG_RSC_GFM_PHS_1, t);
    }
    { // DFIG_DCLink_PHS_1 dynamics
        inputs_DFIG_DCLink_PHS_1[0] = outputs_DFIG_RSC_GFM_PHS_1[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_1[1] = outputs_DFIG_GSC_GFM_PHS_1[2]; // P_gsc
        step_DFIG_DCLink_PHS_1(&x[17], &dxdt[17], inputs_DFIG_DCLink_PHS_1, outputs_DFIG_DCLink_PHS_1, t);
    }
    { // DFIG_GSC_GFM_PHS_1 dynamics
        inputs_DFIG_GSC_GFM_PHS_1[0] = outputs_DFIG_DCLink_PHS_1[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_1[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_1[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_1[3] = Vd_net[1]; // Vd
        inputs_DFIG_GSC_GFM_PHS_1[4] = Vq_net[1]; // Vq
        inputs_DFIG_GSC_GFM_PHS_1[5] = outputs_DFIG_RSC_GFM_PHS_1[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_1(&x[18], &dxdt[18], inputs_DFIG_GSC_GFM_PHS_1, outputs_DFIG_GSC_GFM_PHS_1, t);
    }
    { // DFIG_DT_PHS_1 dynamics
        inputs_DFIG_DT_PHS_1[0] = outputs_DFIG_PHS_1[2]; // omega_g
        inputs_DFIG_DT_PHS_1[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_1[2] = 0.0; // beta
        step_DFIG_DT_PHS_1(&x[22], &dxdt[22], inputs_DFIG_DT_PHS_1, outputs_DFIG_DT_PHS_1, t);
    }
    { // DFIG_PHS_2 dynamics
        inputs_DFIG_PHS_2[0] = Vd_net[2]; // Vd
        inputs_DFIG_PHS_2[1] = Vq_net[2]; // Vq
        inputs_DFIG_PHS_2[2] = outputs_DFIG_DT_PHS_2[0]; // Tm
        inputs_DFIG_PHS_2[3] = outputs_DFIG_RSC_GFM_PHS_2[0]; // Vrd
        inputs_DFIG_PHS_2[4] = outputs_DFIG_RSC_GFM_PHS_2[1]; // Vrq
        step_DFIG_PHS_2(&x[25], &dxdt[25], inputs_DFIG_PHS_2, outputs_DFIG_PHS_2, t);
    }
    { // DFIG_RSC_GFM_PHS_2 dynamics
        inputs_DFIG_RSC_GFM_PHS_2[0] = 0.5; // P_star
        inputs_DFIG_RSC_GFM_PHS_2[1] = 0.16; // Q_star
        inputs_DFIG_RSC_GFM_PHS_2[2] = outputs_DFIG_PHS_2[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_2[3] = outputs_DFIG_PHS_2[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_2[4] = Vd_net[2]; // vds
        inputs_DFIG_RSC_GFM_PHS_2[5] = Vq_net[2]; // vqs
        inputs_DFIG_RSC_GFM_PHS_2[6] = outputs_DFIG_PHS_2[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_2[7] = outputs_DFIG_PHS_2[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_2[8] = outputs_DFIG_PHS_2[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_2[9] = outputs_DFIG_PHS_2[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_2[10] = outputs_DFIG_PHS_2[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_2(&x[30], &dxdt[30], inputs_DFIG_RSC_GFM_PHS_2, outputs_DFIG_RSC_GFM_PHS_2, t);
    }
    { // DFIG_DCLink_PHS_2 dynamics
        inputs_DFIG_DCLink_PHS_2[0] = outputs_DFIG_RSC_GFM_PHS_2[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_2[1] = outputs_DFIG_GSC_GFM_PHS_2[2]; // P_gsc
        step_DFIG_DCLink_PHS_2(&x[37], &dxdt[37], inputs_DFIG_DCLink_PHS_2, outputs_DFIG_DCLink_PHS_2, t);
    }
    { // DFIG_GSC_GFM_PHS_2 dynamics
        inputs_DFIG_GSC_GFM_PHS_2[0] = outputs_DFIG_DCLink_PHS_2[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_2[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_2[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_2[3] = Vd_net[2]; // Vd
        inputs_DFIG_GSC_GFM_PHS_2[4] = Vq_net[2]; // Vq
        inputs_DFIG_GSC_GFM_PHS_2[5] = outputs_DFIG_RSC_GFM_PHS_2[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_2(&x[38], &dxdt[38], inputs_DFIG_GSC_GFM_PHS_2, outputs_DFIG_GSC_GFM_PHS_2, t);
    }
    { // DFIG_DT_PHS_2 dynamics
        inputs_DFIG_DT_PHS_2[0] = outputs_DFIG_PHS_2[2]; // omega_g
        inputs_DFIG_DT_PHS_2[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_2[2] = 0.0; // beta
        step_DFIG_DT_PHS_2(&x[42], &dxdt[42], inputs_DFIG_DT_PHS_2, outputs_DFIG_DT_PHS_2, t);
    }
    { // DFIG_PHS_3 dynamics
        inputs_DFIG_PHS_3[0] = Vd_net[5]; // Vd
        inputs_DFIG_PHS_3[1] = Vq_net[5]; // Vq
        inputs_DFIG_PHS_3[2] = outputs_DFIG_DT_PHS_3[0]; // Tm
        inputs_DFIG_PHS_3[3] = outputs_DFIG_RSC_GFM_PHS_3[0]; // Vrd
        inputs_DFIG_PHS_3[4] = outputs_DFIG_RSC_GFM_PHS_3[1]; // Vrq
        step_DFIG_PHS_3(&x[45], &dxdt[45], inputs_DFIG_PHS_3, outputs_DFIG_PHS_3, t);
    }
    { // DFIG_RSC_GFM_PHS_3 dynamics
        inputs_DFIG_RSC_GFM_PHS_3[0] = 0.4; // P_star
        inputs_DFIG_RSC_GFM_PHS_3[1] = 0.32; // Q_star
        inputs_DFIG_RSC_GFM_PHS_3[2] = outputs_DFIG_PHS_3[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_3[3] = outputs_DFIG_PHS_3[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_3[4] = Vd_net[5]; // vds
        inputs_DFIG_RSC_GFM_PHS_3[5] = Vq_net[5]; // vqs
        inputs_DFIG_RSC_GFM_PHS_3[6] = outputs_DFIG_PHS_3[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_3[7] = outputs_DFIG_PHS_3[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_3[8] = outputs_DFIG_PHS_3[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_3[9] = outputs_DFIG_PHS_3[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_3[10] = outputs_DFIG_PHS_3[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_3(&x[50], &dxdt[50], inputs_DFIG_RSC_GFM_PHS_3, outputs_DFIG_RSC_GFM_PHS_3, t);
    }
    { // DFIG_DCLink_PHS_3 dynamics
        inputs_DFIG_DCLink_PHS_3[0] = outputs_DFIG_RSC_GFM_PHS_3[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_3[1] = outputs_DFIG_GSC_GFM_PHS_3[2]; // P_gsc
        step_DFIG_DCLink_PHS_3(&x[57], &dxdt[57], inputs_DFIG_DCLink_PHS_3, outputs_DFIG_DCLink_PHS_3, t);
    }
    { // DFIG_GSC_GFM_PHS_3 dynamics
        inputs_DFIG_GSC_GFM_PHS_3[0] = outputs_DFIG_DCLink_PHS_3[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_3[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_3[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_3[3] = Vd_net[5]; // Vd
        inputs_DFIG_GSC_GFM_PHS_3[4] = Vq_net[5]; // Vq
        inputs_DFIG_GSC_GFM_PHS_3[5] = outputs_DFIG_RSC_GFM_PHS_3[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_3(&x[58], &dxdt[58], inputs_DFIG_GSC_GFM_PHS_3, outputs_DFIG_GSC_GFM_PHS_3, t);
    }
    { // DFIG_DT_PHS_3 dynamics
        inputs_DFIG_DT_PHS_3[0] = outputs_DFIG_PHS_3[2]; // omega_g
        inputs_DFIG_DT_PHS_3[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_3[2] = 0.0; // beta
        step_DFIG_DT_PHS_3(&x[62], &dxdt[62], inputs_DFIG_DT_PHS_3, outputs_DFIG_DT_PHS_3, t);
    }
    { // DFIG_PHS_4 dynamics
        inputs_DFIG_PHS_4[0] = Vd_net[7]; // Vd
        inputs_DFIG_PHS_4[1] = Vq_net[7]; // Vq
        inputs_DFIG_PHS_4[2] = outputs_DFIG_DT_PHS_4[0]; // Tm
        inputs_DFIG_PHS_4[3] = outputs_DFIG_RSC_GFM_PHS_4[0]; // Vrd
        inputs_DFIG_PHS_4[4] = outputs_DFIG_RSC_GFM_PHS_4[1]; // Vrq
        step_DFIG_PHS_4(&x[65], &dxdt[65], inputs_DFIG_PHS_4, outputs_DFIG_PHS_4, t);
    }
    { // DFIG_RSC_GFM_PHS_4 dynamics
        inputs_DFIG_RSC_GFM_PHS_4[0] = 0.35; // P_star
        inputs_DFIG_RSC_GFM_PHS_4[1] = 0.15; // Q_star
        inputs_DFIG_RSC_GFM_PHS_4[2] = outputs_DFIG_PHS_4[3]; // Ps
        inputs_DFIG_RSC_GFM_PHS_4[3] = outputs_DFIG_PHS_4[4]; // Qs
        inputs_DFIG_RSC_GFM_PHS_4[4] = Vd_net[7]; // vds
        inputs_DFIG_RSC_GFM_PHS_4[5] = Vq_net[7]; // vqs
        inputs_DFIG_RSC_GFM_PHS_4[6] = outputs_DFIG_PHS_4[9]; // ird
        inputs_DFIG_RSC_GFM_PHS_4[7] = outputs_DFIG_PHS_4[10]; // iqr
        inputs_DFIG_RSC_GFM_PHS_4[8] = outputs_DFIG_PHS_4[11]; // phi_sd
        inputs_DFIG_RSC_GFM_PHS_4[9] = outputs_DFIG_PHS_4[12]; // phi_sq
        inputs_DFIG_RSC_GFM_PHS_4[10] = outputs_DFIG_PHS_4[2]; // omega_r
        step_DFIG_RSC_GFM_PHS_4(&x[70], &dxdt[70], inputs_DFIG_RSC_GFM_PHS_4, outputs_DFIG_RSC_GFM_PHS_4, t);
    }
    { // DFIG_DCLink_PHS_4 dynamics
        inputs_DFIG_DCLink_PHS_4[0] = outputs_DFIG_RSC_GFM_PHS_4[2]; // P_rsc
        inputs_DFIG_DCLink_PHS_4[1] = outputs_DFIG_GSC_GFM_PHS_4[2]; // P_gsc
        step_DFIG_DCLink_PHS_4(&x[77], &dxdt[77], inputs_DFIG_DCLink_PHS_4, outputs_DFIG_DCLink_PHS_4, t);
    }
    { // DFIG_GSC_GFM_PHS_4 dynamics
        inputs_DFIG_GSC_GFM_PHS_4[0] = outputs_DFIG_DCLink_PHS_4[0]; // Vdc
        inputs_DFIG_GSC_GFM_PHS_4[1] = 1.0; // Vdc_ref
        inputs_DFIG_GSC_GFM_PHS_4[2] = 0.0; // Qref
        inputs_DFIG_GSC_GFM_PHS_4[3] = Vd_net[7]; // Vd
        inputs_DFIG_GSC_GFM_PHS_4[4] = Vq_net[7]; // Vq
        inputs_DFIG_GSC_GFM_PHS_4[5] = outputs_DFIG_RSC_GFM_PHS_4[3]; // theta_s
        step_DFIG_GSC_GFM_PHS_4(&x[78], &dxdt[78], inputs_DFIG_GSC_GFM_PHS_4, outputs_DFIG_GSC_GFM_PHS_4, t);
    }
    { // DFIG_DT_PHS_4 dynamics
        inputs_DFIG_DT_PHS_4[0] = outputs_DFIG_PHS_4[2]; // omega_g
        inputs_DFIG_DT_PHS_4[1] = 12.0; // V_w
        inputs_DFIG_DT_PHS_4[2] = 0.0; // beta
        step_DFIG_DT_PHS_4(&x[82], &dxdt[82], inputs_DFIG_DT_PHS_4, outputs_DFIG_DT_PHS_4, t);
    }
    double coi_omega = x[1];
    dxdt[85] = 0.0;

    // --- 3. Differential Residuals ---
    for (int i = 0; i < N_DIFF; ++i)
        res[i] = ydot[i] - dxdt[i];

    // --- Fault shunt admittance (segment-based flag) ---
    double Yf_g[14] = {0};
    double Yf_b[14] = {0};
    for (int f = 0; f < N_FAULTS; ++f) {
        if (fault_active[f]) {
            Yf_g[FAULT_BUS[f]] += FAULT_G[f];
            Yf_b[FAULT_BUS[f]] += FAULT_B[f];
        }
    }

    // --- 5. Algebraic Residuals (KCL / Slack) ---
    for (int i = 0; i < N_BUS; ++i) {
        if (IS_SLACK[i]) {
            // Slack bus: V = V_ref (voltage source)
            res[N_DIFF + 2*i]     = Vd_net[i] - Vd_slack_ref[i];
            res[N_DIFF + 2*i + 1] = Vq_net[i] - Vq_slack_ref[i];
        } else {
            // KCL: I_inj − (Y_bus + Y_fault) · V = 0
            double Id_ybus = 0.0, Iq_ybus = 0.0;
            for (int j = 0; j < N_BUS; ++j) {
                double G = Y_real[i*N_BUS + j];
                double B = Y_imag[i*N_BUS + j];
                Id_ybus += G*Vd_net[j] - B*Vq_net[j];
                Iq_ybus += G*Vq_net[j] + B*Vd_net[j];
            }
            // Add fault shunt current: I_fault = Y_fault · V_local
            Id_ybus += Yf_g[i]*Vd_net[i] - Yf_b[i]*Vq_net[i];
            Iq_ybus += Yf_g[i]*Vq_net[i] + Yf_b[i]*Vd_net[i];
            // Frequency-dependent load: extra current = kpf*dw*G_load*V (P) + kqf*dw*B_load*V (Q)
            if (LOAD_KPF[i] != 0.0 || LOAD_KQF[i] != 0.0) {
                double dw = coi_omega - 1.0;
                double dP = LOAD_KPF[i] * dw * LOAD_G[i];
                double dQ = LOAD_KQF[i] * dw * LOAD_B[i];
                Id_ybus += dP*Vd_net[i] - dQ*Vq_net[i];
                Iq_ybus += dP*Vq_net[i] + dQ*Vd_net[i];
            }
            res[N_DIFF + 2*i]     = Id_inj[i] - Id_ybus;
            res[N_DIFF + 2*i + 1] = Iq_inj[i] - Iq_ybus;
        }
    }
}


// =================================================================
// Dense LU solver for Ax = b (in-place, small N)
// =================================================================
void lu_solve(double* A, double* b, int n) {
    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int pivot = k;
        double pmax = fabs(A[k*n+k]);
        for (int i = k+1; i < n; ++i) {
            if (fabs(A[i*n+k]) > pmax) { pmax = fabs(A[i*n+k]); pivot = i; }
        }
        // Swap rows in A and b
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                double tmp = A[k*n+j]; A[k*n+j] = A[pivot*n+j]; A[pivot*n+j] = tmp;
            }
            double tmp = b[k]; b[k] = b[pivot]; b[pivot] = tmp;
        }
        // Eliminate below
        double akk = A[k*n+k];
        if (fabs(akk) < 1e-30) akk = 1e-30;
        for (int i = k+1; i < n; ++i) {
            double factor = A[i*n+k] / akk;
            for (int j = k+1; j < n; ++j)
                A[i*n+j] -= factor * A[k*n+j];
            b[i] -= factor * b[k];
        }
    }
    // Back substitution
    for (int i = n-1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i+1; j < n; ++j)
            sum -= A[i*n+j] * b[j];
        double aii = A[i*n+i];
        if (fabs(aii) < 1e-30) aii = 1e-30;
        b[i] = sum / aii;
    }
}

// =================================================================
// BDF-1 (Backward Euler) DAE Solver with Full Newton Iteration
// =================================================================

void solve_bdf1(double* y, double dt, int n_steps) {
    double y_old[N_TOTAL];
    double ydot[N_TOTAL];
    double res[N_TOTAL];
    double res_pert[N_TOTAL];
    double y_pert[N_TOTAL];
    double dy[N_TOTAL];
    double J[N_TOTAL * N_TOTAL];  // Dense Jacobian (column-major: J[i*N+j])

    const double newton_tol = 1e-8;
    const int max_newton = 20;
    const double eps_fd = 1e-7;

    // Output File
    std::ofstream outfile("simulation_results.csv");
    outfile << "t,GENCLS_1.delta,GENCLS_1.omega,GENCLS_1.delta_deg,GENCLS_1.Te,TGOV1_1.x1,TGOV1_1.x2,TGOV1_1.xi,TGOV1_1.Tm,TGOV1_1.Valve,TGOV1_1.xi,DFIG_PHS_1.phi_sd,DFIG_PHS_1.phi_sq,DFIG_PHS_1.phi_rd,DFIG_PHS_1.phi_rq,DFIG_PHS_1.p_g,DFIG_PHS_1.Pe,DFIG_PHS_1.Qe,DFIG_PHS_1.omega_pu,DFIG_PHS_1.slip,DFIG_PHS_1.phi_sd,DFIG_PHS_1.phi_sq,DFIG_PHS_1.phi_rd,DFIG_PHS_1.phi_rq,DFIG_PHS_1.Te_elec,DFIG_PHS_1.V_term,DFIG_PHS_1.i_rd,DFIG_PHS_1.i_rq,DFIG_PHS_1.Ir_mag,DFIG_PHS_1.i_sd,DFIG_PHS_1.i_sq,DFIG_PHS_1.Is_mag,DFIG_PHS_1.H_total,DFIG_RSC_GFM_PHS_1.omega_vs,DFIG_RSC_GFM_PHS_1.theta_s,DFIG_RSC_GFM_PHS_1.phi_Qs,DFIG_RSC_GFM_PHS_1.phi_vqs,DFIG_RSC_GFM_PHS_1.phi_vds,DFIG_RSC_GFM_PHS_1.phi_ird,DFIG_RSC_GFM_PHS_1.phi_iqr,DFIG_RSC_GFM_PHS_1.Vrd_cmd,DFIG_RSC_GFM_PHS_1.Vrq_cmd,DFIG_RSC_GFM_PHS_1.P_rotor,DFIG_RSC_GFM_PHS_1.theta_s,DFIG_RSC_GFM_PHS_1.omega_vs,DFIG_DCLink_PHS_1.V_dc,DFIG_DCLink_PHS_1.Vdc,DFIG_DCLink_PHS_1.P_net,DFIG_DCLink_PHS_1.H_dc,DFIG_GSC_GFM_PHS_1.phi_fd,DFIG_GSC_GFM_PHS_1.phi_fq,DFIG_GSC_GFM_PHS_1.x_Vdc,DFIG_GSC_GFM_PHS_1.x_Q,DFIG_GSC_GFM_PHS_1.i_fd,DFIG_GSC_GFM_PHS_1.i_fq,DFIG_GSC_GFM_PHS_1.P_gsc,DFIG_GSC_GFM_PHS_1.Q_gsc,DFIG_GSC_GFM_PHS_1.H_filter,DFIG_DT_PHS_1.p_t,DFIG_DT_PHS_1.Cp_eff,DFIG_DT_PHS_1.theta_tw,DFIG_DT_PHS_1.omega_t,DFIG_DT_PHS_1.theta_tw,DFIG_DT_PHS_1.Cp_eff,DFIG_DT_PHS_1.T_shaft,DFIG_DT_PHS_1.T_aero,DFIG_DT_PHS_1.twist_rate,DFIG_PHS_2.phi_sd,DFIG_PHS_2.phi_sq,DFIG_PHS_2.phi_rd,DFIG_PHS_2.phi_rq,DFIG_PHS_2.p_g,DFIG_PHS_2.Pe,DFIG_PHS_2.Qe,DFIG_PHS_2.omega_pu,DFIG_PHS_2.slip,DFIG_PHS_2.phi_sd,DFIG_PHS_2.phi_sq,DFIG_PHS_2.phi_rd,DFIG_PHS_2.phi_rq,DFIG_PHS_2.Te_elec,DFIG_PHS_2.V_term,DFIG_PHS_2.i_rd,DFIG_PHS_2.i_rq,DFIG_PHS_2.Ir_mag,DFIG_PHS_2.i_sd,DFIG_PHS_2.i_sq,DFIG_PHS_2.Is_mag,DFIG_PHS_2.H_total,DFIG_RSC_GFM_PHS_2.omega_vs,DFIG_RSC_GFM_PHS_2.theta_s,DFIG_RSC_GFM_PHS_2.phi_Qs,DFIG_RSC_GFM_PHS_2.phi_vqs,DFIG_RSC_GFM_PHS_2.phi_vds,DFIG_RSC_GFM_PHS_2.phi_ird,DFIG_RSC_GFM_PHS_2.phi_iqr,DFIG_RSC_GFM_PHS_2.Vrd_cmd,DFIG_RSC_GFM_PHS_2.Vrq_cmd,DFIG_RSC_GFM_PHS_2.P_rotor,DFIG_RSC_GFM_PHS_2.theta_s,DFIG_RSC_GFM_PHS_2.omega_vs,DFIG_DCLink_PHS_2.V_dc,DFIG_DCLink_PHS_2.Vdc,DFIG_DCLink_PHS_2.P_net,DFIG_DCLink_PHS_2.H_dc,DFIG_GSC_GFM_PHS_2.phi_fd,DFIG_GSC_GFM_PHS_2.phi_fq,DFIG_GSC_GFM_PHS_2.x_Vdc,DFIG_GSC_GFM_PHS_2.x_Q,DFIG_GSC_GFM_PHS_2.i_fd,DFIG_GSC_GFM_PHS_2.i_fq,DFIG_GSC_GFM_PHS_2.P_gsc,DFIG_GSC_GFM_PHS_2.Q_gsc,DFIG_GSC_GFM_PHS_2.H_filter,DFIG_DT_PHS_2.p_t,DFIG_DT_PHS_2.Cp_eff,DFIG_DT_PHS_2.theta_tw,DFIG_DT_PHS_2.omega_t,DFIG_DT_PHS_2.theta_tw,DFIG_DT_PHS_2.Cp_eff,DFIG_DT_PHS_2.T_shaft,DFIG_DT_PHS_2.T_aero,DFIG_DT_PHS_2.twist_rate,DFIG_PHS_3.phi_sd,DFIG_PHS_3.phi_sq,DFIG_PHS_3.phi_rd,DFIG_PHS_3.phi_rq,DFIG_PHS_3.p_g,DFIG_PHS_3.Pe,DFIG_PHS_3.Qe,DFIG_PHS_3.omega_pu,DFIG_PHS_3.slip,DFIG_PHS_3.phi_sd,DFIG_PHS_3.phi_sq,DFIG_PHS_3.phi_rd,DFIG_PHS_3.phi_rq,DFIG_PHS_3.Te_elec,DFIG_PHS_3.V_term,DFIG_PHS_3.i_rd,DFIG_PHS_3.i_rq,DFIG_PHS_3.Ir_mag,DFIG_PHS_3.i_sd,DFIG_PHS_3.i_sq,DFIG_PHS_3.Is_mag,DFIG_PHS_3.H_total,DFIG_RSC_GFM_PHS_3.omega_vs,DFIG_RSC_GFM_PHS_3.theta_s,DFIG_RSC_GFM_PHS_3.phi_Qs,DFIG_RSC_GFM_PHS_3.phi_vqs,DFIG_RSC_GFM_PHS_3.phi_vds,DFIG_RSC_GFM_PHS_3.phi_ird,DFIG_RSC_GFM_PHS_3.phi_iqr,DFIG_RSC_GFM_PHS_3.Vrd_cmd,DFIG_RSC_GFM_PHS_3.Vrq_cmd,DFIG_RSC_GFM_PHS_3.P_rotor,DFIG_RSC_GFM_PHS_3.theta_s,DFIG_RSC_GFM_PHS_3.omega_vs,DFIG_DCLink_PHS_3.V_dc,DFIG_DCLink_PHS_3.Vdc,DFIG_DCLink_PHS_3.P_net,DFIG_DCLink_PHS_3.H_dc,DFIG_GSC_GFM_PHS_3.phi_fd,DFIG_GSC_GFM_PHS_3.phi_fq,DFIG_GSC_GFM_PHS_3.x_Vdc,DFIG_GSC_GFM_PHS_3.x_Q,DFIG_GSC_GFM_PHS_3.i_fd,DFIG_GSC_GFM_PHS_3.i_fq,DFIG_GSC_GFM_PHS_3.P_gsc,DFIG_GSC_GFM_PHS_3.Q_gsc,DFIG_GSC_GFM_PHS_3.H_filter,DFIG_DT_PHS_3.p_t,DFIG_DT_PHS_3.Cp_eff,DFIG_DT_PHS_3.theta_tw,DFIG_DT_PHS_3.omega_t,DFIG_DT_PHS_3.theta_tw,DFIG_DT_PHS_3.Cp_eff,DFIG_DT_PHS_3.T_shaft,DFIG_DT_PHS_3.T_aero,DFIG_DT_PHS_3.twist_rate,DFIG_PHS_4.phi_sd,DFIG_PHS_4.phi_sq,DFIG_PHS_4.phi_rd,DFIG_PHS_4.phi_rq,DFIG_PHS_4.p_g,DFIG_PHS_4.Pe,DFIG_PHS_4.Qe,DFIG_PHS_4.omega_pu,DFIG_PHS_4.slip,DFIG_PHS_4.phi_sd,DFIG_PHS_4.phi_sq,DFIG_PHS_4.phi_rd,DFIG_PHS_4.phi_rq,DFIG_PHS_4.Te_elec,DFIG_PHS_4.V_term,DFIG_PHS_4.i_rd,DFIG_PHS_4.i_rq,DFIG_PHS_4.Ir_mag,DFIG_PHS_4.i_sd,DFIG_PHS_4.i_sq,DFIG_PHS_4.Is_mag,DFIG_PHS_4.H_total,DFIG_RSC_GFM_PHS_4.omega_vs,DFIG_RSC_GFM_PHS_4.theta_s,DFIG_RSC_GFM_PHS_4.phi_Qs,DFIG_RSC_GFM_PHS_4.phi_vqs,DFIG_RSC_GFM_PHS_4.phi_vds,DFIG_RSC_GFM_PHS_4.phi_ird,DFIG_RSC_GFM_PHS_4.phi_iqr,DFIG_RSC_GFM_PHS_4.Vrd_cmd,DFIG_RSC_GFM_PHS_4.Vrq_cmd,DFIG_RSC_GFM_PHS_4.P_rotor,DFIG_RSC_GFM_PHS_4.theta_s,DFIG_RSC_GFM_PHS_4.omega_vs,DFIG_DCLink_PHS_4.V_dc,DFIG_DCLink_PHS_4.Vdc,DFIG_DCLink_PHS_4.P_net,DFIG_DCLink_PHS_4.H_dc,DFIG_GSC_GFM_PHS_4.phi_fd,DFIG_GSC_GFM_PHS_4.phi_fq,DFIG_GSC_GFM_PHS_4.x_Vdc,DFIG_GSC_GFM_PHS_4.x_Q,DFIG_GSC_GFM_PHS_4.i_fd,DFIG_GSC_GFM_PHS_4.i_fq,DFIG_GSC_GFM_PHS_4.P_gsc,DFIG_GSC_GFM_PHS_4.Q_gsc,DFIG_GSC_GFM_PHS_4.H_filter,DFIG_DT_PHS_4.p_t,DFIG_DT_PHS_4.Cp_eff,DFIG_DT_PHS_4.theta_tw,DFIG_DT_PHS_4.omega_t,DFIG_DT_PHS_4.theta_tw,DFIG_DT_PHS_4.Cp_eff,DFIG_DT_PHS_4.T_shaft,DFIG_DT_PHS_4.T_aero,DFIG_DT_PHS_4.twist_rate,Vd_Bus1,Vq_Bus1,Vterm_Bus1,Vd_Bus2,Vq_Bus2,Vterm_Bus2,Vd_Bus3,Vq_Bus3,Vterm_Bus3,Vd_Bus4,Vq_Bus4,Vterm_Bus4,Vd_Bus5,Vq_Bus5,Vterm_Bus5,Vd_Bus6,Vq_Bus6,Vterm_Bus6,Vd_Bus7,Vq_Bus7,Vterm_Bus7,Vd_Bus8,Vq_Bus8,Vterm_Bus8,Vd_Bus9,Vq_Bus9,Vterm_Bus9,Vd_Bus10,Vq_Bus10,Vterm_Bus10,Vd_Bus11,Vq_Bus11,Vterm_Bus11,Vd_Bus12,Vq_Bus12,Vterm_Bus12,Vd_Bus13,Vq_Bus13,Vterm_Bus13,Vd_Bus14,Vq_Bus14,Vterm_Bus14" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double t = 0.0;
    const int log_every = 20;

    // Extract Vd/Vq arrays for logging
    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Initial diagnostics
    for (int i = 0; i < N_TOTAL; ++i) ydot[i] = 0.0;
    dae_residual(y, ydot, res, 0.0);
    double max_res = 0.0;
    int max_res_idx = -1;
    for (int i = 0; i < N_TOTAL; ++i) {
        if (fabs(res[i]) > max_res) { max_res = fabs(res[i]); max_res_idx = i; }
    }
    std::cout << "[DAE] Initial max |residual| = " << max_res
              << " at index " << max_res_idx << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        // Update Vd/Vq for logging
        for (int i = 0; i < N_BUS; ++i) {
            Vd_net[i]   = y[N_DIFF + 2*i];
            Vq_net[i]   = y[N_DIFF + 2*i + 1];
            Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
        }

        // Log
        if (step % log_every == 0) {
            const double* x = y;
            outfile << (t) << "," << (x[0]) << "," << (x[1]) << "," << (y[0] * 180.0 / 3.14159265359) << "," << ((inputs_GENCLS_1[0] * outputs_GENCLS_1[0] + inputs_GENCLS_1[1] * outputs_GENCLS_1[1])) << "," << (x[2]) << "," << (x[3]) << "," << (x[4]) << "," << (outputs_TGOV1_1[0]) << "," << (y[2]) << "," << (y[4]) << "," << (x[5]) << "," << (x[6]) << "," << (x[7]) << "," << (x[8]) << "," << (x[9]) << "," << (outputs_DFIG_PHS_1[3]) << "," << (outputs_DFIG_PHS_1[4]) << "," << (outputs_DFIG_PHS_1[2]) << "," << ((1 - outputs_DFIG_PHS_1[2]) / 1) << "," << (y[5]) << "," << (y[6]) << "," << (y[7]) << "," << (y[8]) << "," << (outputs_DFIG_PHS_1[15]) << "," << (sqrt(inputs_DFIG_PHS_1[0]*inputs_DFIG_PHS_1[0] + inputs_DFIG_PHS_1[1]*inputs_DFIG_PHS_1[1])) << "," << (outputs_DFIG_PHS_1[9]) << "," << (outputs_DFIG_PHS_1[10]) << "," << (sqrt(outputs_DFIG_PHS_1[9]*outputs_DFIG_PHS_1[9] + outputs_DFIG_PHS_1[10]*outputs_DFIG_PHS_1[10])) << "," << (outputs_DFIG_PHS_1[5]) << "," << (outputs_DFIG_PHS_1[6]) << "," << (sqrt(outputs_DFIG_PHS_1[5]*outputs_DFIG_PHS_1[5] + outputs_DFIG_PHS_1[6]*outputs_DFIG_PHS_1[6])) << "," << ((2.5362318840579663*((y[5])*(y[5])) - 4.9275362318840488*y[5]*y[7] + 2.5362318840579663*((y[6])*(y[6])) - 4.9275362318840488*y[6]*y[8] + 2.5362318840579663*((y[7])*(y[7])) + 2.5362318840579663*((y[8])*(y[8])) + 0.5*((y[9])*(y[9])))) << "," << (x[10]) << "," << (x[11]) << "," << (x[12]) << "," << (x[13]) << "," << (x[14]) << "," << (x[15]) << "," << (x[16]) << "," << (outputs_DFIG_RSC_GFM_PHS_1[0]) << "," << (outputs_DFIG_RSC_GFM_PHS_1[1]) << "," << (outputs_DFIG_RSC_GFM_PHS_1[2]) << "," << (y[11]) << "," << (y[10]) << "," << (x[17]) << "," << (y[17]) << "," << (outputs_DFIG_DCLink_PHS_1[1]) << "," << (0.5 * 1 * y[17] * y[17]) << "," << (x[18]) << "," << (x[19]) << "," << (x[20]) << "," << (x[21]) << "," << (y[18] / 0.15) << "," << (y[19] / 0.15) << "," << (outputs_DFIG_GSC_GFM_PHS_1[2]) << "," << (outputs_DFIG_GSC_GFM_PHS_1[3]) << "," << ((y[18]*y[18] + y[19]*y[19]) / (2.0 * 0.15)) << "," << (x[22]) << "," << (x[23]) << "," << (x[24]) << "," << (outputs_DFIG_DT_PHS_1[1]) << "," << (y[24]) << "," << (y[23]) << "," << (outputs_DFIG_DT_PHS_1[0]) << "," << (outputs_DFIG_DT_PHS_1[2]) << "," << (outputs_DFIG_DT_PHS_1[1] - inputs_DFIG_DT_PHS_1[0]) << "," << (x[25]) << "," << (x[26]) << "," << (x[27]) << "," << (x[28]) << "," << (x[29]) << "," << (outputs_DFIG_PHS_2[3]) << "," << (outputs_DFIG_PHS_2[4]) << "," << (outputs_DFIG_PHS_2[2]) << "," << ((1 - outputs_DFIG_PHS_2[2]) / 1) << "," << (y[25]) << "," << (y[26]) << "," << (y[27]) << "," << (y[28]) << "," << (outputs_DFIG_PHS_2[15]) << "," << (sqrt(inputs_DFIG_PHS_2[0]*inputs_DFIG_PHS_2[0] + inputs_DFIG_PHS_2[1]*inputs_DFIG_PHS_2[1])) << "," << (outputs_DFIG_PHS_2[9]) << "," << (outputs_DFIG_PHS_2[10]) << "," << (sqrt(outputs_DFIG_PHS_2[9]*outputs_DFIG_PHS_2[9] + outputs_DFIG_PHS_2[10]*outputs_DFIG_PHS_2[10])) << "," << (outputs_DFIG_PHS_2[5]) << "," << (outputs_DFIG_PHS_2[6]) << "," << (sqrt(outputs_DFIG_PHS_2[5]*outputs_DFIG_PHS_2[5] + outputs_DFIG_PHS_2[6]*outputs_DFIG_PHS_2[6])) << "," << ((2.5362318840579663*((y[25])*(y[25])) - 4.9275362318840488*y[25]*y[27] + 2.5362318840579663*((y[26])*(y[26])) - 4.9275362318840488*y[26]*y[28] + 2.5362318840579663*((y[27])*(y[27])) + 2.5362318840579663*((y[28])*(y[28])) + 0.5*((y[29])*(y[29])))) << "," << (x[30]) << "," << (x[31]) << "," << (x[32]) << "," << (x[33]) << "," << (x[34]) << "," << (x[35]) << "," << (x[36]) << "," << (outputs_DFIG_RSC_GFM_PHS_2[0]) << "," << (outputs_DFIG_RSC_GFM_PHS_2[1]) << "," << (outputs_DFIG_RSC_GFM_PHS_2[2]) << "," << (y[31]) << "," << (y[30]) << "," << (x[37]) << "," << (y[37]) << "," << (outputs_DFIG_DCLink_PHS_2[1]) << "," << (0.5 * 1 * y[37] * y[37]) << "," << (x[38]) << "," << (x[39]) << "," << (x[40]) << "," << (x[41]) << "," << (y[38] / 0.15) << "," << (y[39] / 0.15) << "," << (outputs_DFIG_GSC_GFM_PHS_2[2]) << "," << (outputs_DFIG_GSC_GFM_PHS_2[3]) << "," << ((y[38]*y[38] + y[39]*y[39]) / (2.0 * 0.15)) << "," << (x[42]) << "," << (x[43]) << "," << (x[44]) << "," << (outputs_DFIG_DT_PHS_2[1]) << "," << (y[44]) << "," << (y[43]) << "," << (outputs_DFIG_DT_PHS_2[0]) << "," << (outputs_DFIG_DT_PHS_2[2]) << "," << (outputs_DFIG_DT_PHS_2[1] - inputs_DFIG_DT_PHS_2[0]) << "," << (x[45]) << "," << (x[46]) << "," << (x[47]) << "," << (x[48]) << "," << (x[49]) << "," << (outputs_DFIG_PHS_3[3]) << "," << (outputs_DFIG_PHS_3[4]) << "," << (outputs_DFIG_PHS_3[2]) << "," << ((1 - outputs_DFIG_PHS_3[2]) / 1) << "," << (y[45]) << "," << (y[46]) << "," << (y[47]) << "," << (y[48]) << "," << (outputs_DFIG_PHS_3[15]) << "," << (sqrt(inputs_DFIG_PHS_3[0]*inputs_DFIG_PHS_3[0] + inputs_DFIG_PHS_3[1]*inputs_DFIG_PHS_3[1])) << "," << (outputs_DFIG_PHS_3[9]) << "," << (outputs_DFIG_PHS_3[10]) << "," << (sqrt(outputs_DFIG_PHS_3[9]*outputs_DFIG_PHS_3[9] + outputs_DFIG_PHS_3[10]*outputs_DFIG_PHS_3[10])) << "," << (outputs_DFIG_PHS_3[5]) << "," << (outputs_DFIG_PHS_3[6]) << "," << (sqrt(outputs_DFIG_PHS_3[5]*outputs_DFIG_PHS_3[5] + outputs_DFIG_PHS_3[6]*outputs_DFIG_PHS_3[6])) << "," << ((2.5362318840579663*((y[45])*(y[45])) - 4.9275362318840488*y[45]*y[47] + 2.5362318840579663*((y[46])*(y[46])) - 4.9275362318840488*y[46]*y[48] + 2.5362318840579663*((y[47])*(y[47])) + 2.5362318840579663*((y[48])*(y[48])) + 0.5*((y[49])*(y[49])))) << "," << (x[50]) << "," << (x[51]) << "," << (x[52]) << "," << (x[53]) << "," << (x[54]) << "," << (x[55]) << "," << (x[56]) << "," << (outputs_DFIG_RSC_GFM_PHS_3[0]) << "," << (outputs_DFIG_RSC_GFM_PHS_3[1]) << "," << (outputs_DFIG_RSC_GFM_PHS_3[2]) << "," << (y[51]) << "," << (y[50]) << "," << (x[57]) << "," << (y[57]) << "," << (outputs_DFIG_DCLink_PHS_3[1]) << "," << (0.5 * 1 * y[57] * y[57]) << "," << (x[58]) << "," << (x[59]) << "," << (x[60]) << "," << (x[61]) << "," << (y[58] / 0.15) << "," << (y[59] / 0.15) << "," << (outputs_DFIG_GSC_GFM_PHS_3[2]) << "," << (outputs_DFIG_GSC_GFM_PHS_3[3]) << "," << ((y[58]*y[58] + y[59]*y[59]) / (2.0 * 0.15)) << "," << (x[62]) << "," << (x[63]) << "," << (x[64]) << "," << (outputs_DFIG_DT_PHS_3[1]) << "," << (y[64]) << "," << (y[63]) << "," << (outputs_DFIG_DT_PHS_3[0]) << "," << (outputs_DFIG_DT_PHS_3[2]) << "," << (outputs_DFIG_DT_PHS_3[1] - inputs_DFIG_DT_PHS_3[0]) << "," << (x[65]) << "," << (x[66]) << "," << (x[67]) << "," << (x[68]) << "," << (x[69]) << "," << (outputs_DFIG_PHS_4[3]) << "," << (outputs_DFIG_PHS_4[4]) << "," << (outputs_DFIG_PHS_4[2]) << "," << ((1 - outputs_DFIG_PHS_4[2]) / 1) << "," << (y[65]) << "," << (y[66]) << "," << (y[67]) << "," << (y[68]) << "," << (outputs_DFIG_PHS_4[15]) << "," << (sqrt(inputs_DFIG_PHS_4[0]*inputs_DFIG_PHS_4[0] + inputs_DFIG_PHS_4[1]*inputs_DFIG_PHS_4[1])) << "," << (outputs_DFIG_PHS_4[9]) << "," << (outputs_DFIG_PHS_4[10]) << "," << (sqrt(outputs_DFIG_PHS_4[9]*outputs_DFIG_PHS_4[9] + outputs_DFIG_PHS_4[10]*outputs_DFIG_PHS_4[10])) << "," << (outputs_DFIG_PHS_4[5]) << "," << (outputs_DFIG_PHS_4[6]) << "," << (sqrt(outputs_DFIG_PHS_4[5]*outputs_DFIG_PHS_4[5] + outputs_DFIG_PHS_4[6]*outputs_DFIG_PHS_4[6])) << "," << ((2.5362318840579663*((y[65])*(y[65])) - 4.9275362318840488*y[65]*y[67] + 2.5362318840579663*((y[66])*(y[66])) - 4.9275362318840488*y[66]*y[68] + 2.5362318840579663*((y[67])*(y[67])) + 2.5362318840579663*((y[68])*(y[68])) + 0.5*((y[69])*(y[69])))) << "," << (x[70]) << "," << (x[71]) << "," << (x[72]) << "," << (x[73]) << "," << (x[74]) << "," << (x[75]) << "," << (x[76]) << "," << (outputs_DFIG_RSC_GFM_PHS_4[0]) << "," << (outputs_DFIG_RSC_GFM_PHS_4[1]) << "," << (outputs_DFIG_RSC_GFM_PHS_4[2]) << "," << (y[71]) << "," << (y[70]) << "," << (x[77]) << "," << (y[77]) << "," << (outputs_DFIG_DCLink_PHS_4[1]) << "," << (0.5 * 1 * y[77] * y[77]) << "," << (x[78]) << "," << (x[79]) << "," << (x[80]) << "," << (x[81]) << "," << (y[78] / 0.15) << "," << (y[79] / 0.15) << "," << (outputs_DFIG_GSC_GFM_PHS_4[2]) << "," << (outputs_DFIG_GSC_GFM_PHS_4[3]) << "," << ((y[78]*y[78] + y[79]*y[79]) / (2.0 * 0.15)) << "," << (x[82]) << "," << (x[83]) << "," << (x[84]) << "," << (outputs_DFIG_DT_PHS_4[1]) << "," << (y[84]) << "," << (y[83]) << "," << (outputs_DFIG_DT_PHS_4[0]) << "," << (outputs_DFIG_DT_PHS_4[2]) << "," << (outputs_DFIG_DT_PHS_4[1] - inputs_DFIG_DT_PHS_4[0]) << "," << (Vd_net[0]) << "," << (Vq_net[0]) << "," << (Vterm_net[0]) << "," << (Vd_net[1]) << "," << (Vq_net[1]) << "," << (Vterm_net[1]) << "," << (Vd_net[2]) << "," << (Vq_net[2]) << "," << (Vterm_net[2]) << "," << (Vd_net[3]) << "," << (Vq_net[3]) << "," << (Vterm_net[3]) << "," << (Vd_net[4]) << "," << (Vq_net[4]) << "," << (Vterm_net[4]) << "," << (Vd_net[5]) << "," << (Vq_net[5]) << "," << (Vterm_net[5]) << "," << (Vd_net[6]) << "," << (Vq_net[6]) << "," << (Vterm_net[6]) << "," << (Vd_net[7]) << "," << (Vq_net[7]) << "," << (Vterm_net[7]) << "," << (Vd_net[8]) << "," << (Vq_net[8]) << "," << (Vterm_net[8]) << "," << (Vd_net[9]) << "," << (Vq_net[9]) << "," << (Vterm_net[9]) << "," << (Vd_net[10]) << "," << (Vq_net[10]) << "," << (Vterm_net[10]) << "," << (Vd_net[11]) << "," << (Vq_net[11]) << "," << (Vterm_net[11]) << "," << (Vd_net[12]) << "," << (Vq_net[12]) << "," << (Vterm_net[12]) << "," << (Vd_net[13]) << "," << (Vq_net[13]) << "," << (Vterm_net[13]) << std::endl;
        }

        // Save current state
        for (int i = 0; i < N_TOTAL; ++i) y_old[i] = y[i];

        double t_new = t + dt;

        // Update fault flags based on current time
        for (int f = 0; f < N_FAULTS; ++f) {
            fault_active[f] = (t_new >= FAULT_T_START[f]
                            && t_new <  FAULT_T_END[f]) ? 1 : 0;
        }

        // Newton iteration to solve: G(y) = F(t_new, y, (y-y_old)/dt) = 0
        for (int nit = 0; nit < max_newton; ++nit) {
            // Compute ydot = (y - y_old) / dt
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = (y[i] - y_old[i]) / dt;

            // Compute residual
            dae_residual(y, ydot, res, t_new);

            // Check convergence
            double res_norm = 0.0;
            for (int i = 0; i < N_TOTAL; ++i)
                res_norm += res[i] * res[i];
            res_norm = sqrt(res_norm);
            if (res_norm < newton_tol) break;

            // Build full Jacobian via finite differences: J[i][j] = dG_i/dy_j
            // G(y) = F(t_new, y, (y-y_old)/dt)
            // dG/dy_j = dF/dy_j + (1/dt) * dF/dydot_j  (for variable j)
            // We compute this numerically by perturbing y[j] and recomputing G.
            for (int j = 0; j < N_TOTAL; ++j) {
                for (int k = 0; k < N_TOTAL; ++k) y_pert[k] = y[k];
                double h = eps_fd * (1.0 + fabs(y[j]));
                y_pert[j] += h;

                double ydot_pert[N_TOTAL];
                for (int k = 0; k < N_TOTAL; ++k)
                    ydot_pert[k] = (y_pert[k] - y_old[k]) / dt;

                dae_residual(y_pert, ydot_pert, res_pert, t_new);

                for (int i = 0; i < N_TOTAL; ++i)
                    J[i*N_TOTAL + j] = (res_pert[i] - res[i]) / h;
            }

            // Solve J * dy = -res  →  dy = J^{-1} * (-res)
            for (int i = 0; i < N_TOTAL; ++i) dy[i] = -res[i];
            lu_solve(J, dy, N_TOTAL);

            // Line search with backtracking for robustness
            double alpha = 1.0;
            for (int ls = 0; ls < 5; ++ls) {
                for (int i = 0; i < N_TOTAL; ++i)
                    y_pert[i] = y[i] + alpha * dy[i];
                double ydot_ls[N_TOTAL];
                for (int i = 0; i < N_TOTAL; ++i)
                    ydot_ls[i] = (y_pert[i] - y_old[i]) / dt;
                dae_residual(y_pert, ydot_ls, res_pert, t_new);
                double new_norm = 0.0;
                for (int i = 0; i < N_TOTAL; ++i)
                    new_norm += res_pert[i]*res_pert[i];
                new_norm = sqrt(new_norm);
                if (new_norm < res_norm) break;
                alpha *= 0.5;
            }

            for (int i = 0; i < N_TOTAL; ++i)
                y[i] += alpha * dy[i];
        }

        t = t_new;

        // Progress check every ~1 second of simulation time
        if (step % (int)(1.0 / dt) == 0) {
            for (int i = 0; i < N_BUS; ++i) {
                Vd_net[i]   = y[N_DIFF + 2*i];
                Vq_net[i]   = y[N_DIFF + 2*i + 1];
                Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
            }

            // Compute residual norm for monitoring
            for (int i = 0; i < N_TOTAL; ++i)
                ydot[i] = (y[i] - y_old[i]) / dt;
            dae_residual(y, ydot, res, t);
            double rn = 0.0;
            for (int i = 0; i < N_TOTAL; ++i) rn += res[i]*res[i];
            std::cout << "t=" << t << " |res|=" << sqrt(rn)
                      << " Vterm[0]=" << Vterm_net[0] << std::endl;

            if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {
                std::cout << "Stability limit reached. Stopping." << std::endl;
                break;
            }
        }
    }

    outfile.close();
    std::cout << "Done. Results in simulation_results.csv" << std::endl;
}

int main() {
    double y[N_TOTAL] = { 0.215293901061, 1.000000000000, 0.866341892332, 0.866341892332, 0.000000000000, -0.035412451667, -1.035808820484, 0.060096035604, -1.120964803462, 1.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.240871848705, -0.058206785865, 0.000095866220, -0.000116413572, 1.000000000000, 0.000827516942, -0.000028334417, 0.000689597452, 0.000000000000, 4.000000000000, 1.000000000000, 1.405998217836, -0.099304893387, -1.011814064553, -0.005657735394, -1.084108458998, 1.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.200775117544, -0.051335041596, 0.000092125960, -0.000102670083, 1.000000000000, 0.000703035315, -0.000069859674, 0.000585862762, 0.000000000000, 4.000000000000, 1.000000000000, 1.405426405587, -0.094547745548, -1.031296830810, -0.024811843510, -1.131718672406, 1.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.123981018553, -0.065884946184, 0.000068006053, -0.000131769892, 1.000000000000, 0.000797184408, -0.000073206020, 0.000664320340, 0.000000000000, 4.000000000000, 1.000000000000, 1.204959971687, -0.017148929301, -1.035020392781, 0.050574016079, -1.095670538750, 1.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.172287379596, -0.045764862344, 0.000068207367, -0.000091529725, 1.000000000000, 0.000474310942, -0.000008706951, 0.000395259118, 0.000000000000, 4.000000000000, 1.000000000000, 1.102719261135, 0.000000000000, 1.031314605408, 0.002207674903, 1.031042792039, -0.032711864743, 1.007046634238, -0.097202035835, 1.013162263177, -0.080170130131, 1.019569483884, -0.066950100733, 1.027712690708, -0.091087189933, 1.008829378827, -0.075284786130, 1.031644748854, -0.015654465069, 0.988486341324, -0.109787048328, 0.986859475327, -0.110998474471, 1.002989766146, -0.103148677100, 1.009057404414, -0.106766851720, 1.002004538512, -0.107828226705, 0.973132555772, -0.125752674692 };

    std::cout << "Dirac DAE Simulation" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  dt = 0.0005,  T = 60.0 s" << std::endl;
    std::cout << std::endl;

    solve_bdf1(y, 0.0005, 120000);

    return 0;
}
