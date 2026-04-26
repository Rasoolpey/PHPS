#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

// SUNDIALS IDA
#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>

const int N_DIFF  = 66;
const int N_BUS   = 14;
const int N_ALG   = 28;
const int N_TOTAL = 94;

const double LOAD_G[14] = { 0.0000000000e+00, 1.9871339942e-01, 9.2343887854e-01, 4.5231570501e-01, 7.1637289094e-02, 9.7825137567e-02, 0.0000000000e+00, 0.0000000000e+00, 2.6254894980e-01, 8.1014226098e-02, 3.1267755618e-02, 5.4701848026e-02, 1.2221607621e-01, 1.3829024989e-01 };
const double LOAD_B[14] = { 0.0000000000e+00, -1.1629770381e-01, -1.8625624939e-01, 3.6904419446e-02, -1.5081534546e-02, -6.5507904620e-02, 0.0000000000e+00, 0.0000000000e+00, -1.4773940904e-01, -5.2209167930e-02, -1.6080560032e-02, -1.4348025712e-02, -5.2507647558e-02, -4.6406124123e-02 };
const double LOAD_KPF[14] = { 0.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00 };
const double LOAD_KQF[14] = { 0.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00, 1.0000000000e+00 };

// Full Y-bus (14x14) — NO Kron reduction
const double Y_real[196] = {
    6.0250290558e+00, -4.9991316008e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -4.9991316008e+00, 9.7200370102e+00, -1.1350191923e+00, -1.6860331506e+00, -1.7011396671e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1350191923e+00, 4.0444337808e+00, -1.9859757099e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.6860331506e+00, -1.9859757099e+00, 1.0965147750e+01, -6.8409806615e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, -1.7011396671e+00, 0.0000000000e+00, -6.8409806615e+00, 9.6396808134e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.6777485450e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, -1.5259674405e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.5886788069e+00, -3.9020495524e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.9020495524e+00, 5.8639225586e+00, -1.8808847537e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.8808847537e+00, 3.8671536057e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.5259674405e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0697426008e+00, -2.4890245868e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -2.4890245868e+00, 6.8471859504e+00, -1.1369941578e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1369941578e+00, 2.6992928258e+00
};
const double Y_imag[196] = {
    -3.9447070206e+01, 1.5263086523e+01, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.5263086523e+01, -3.5388413103e+01, 4.7818631518e+00, 5.1158383259e+00, 5.1939273980e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7818631518e+00, -1.5008636379e+01, 5.0688169776e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.1158383259e+00, 5.0688169776e+00, -3.8325553300e+01, 2.1578553982e+01, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 5.1939273980e+00, 0.0000000000e+00, 2.1578553982e+01, -3.4948591068e+01, 3.9807970269e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.9807970269e+00, -2.2431998329e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 3.1759639650e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9549005948e+01, 5.6953759109e+00, 9.0900827198e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.6953759109e+00, -1.0713831587e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 9.0900827198e+00, 0.0000000000e+00, -2.4240287885e+01, 1.0365394127e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0365394127e+01, -1.4820530306e+01, 4.4029437495e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.4029437495e+00, -8.5130845280e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.1759639650e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -5.4422993974e+00, 2.2519746262e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.2519746262e+00, -1.0722211390e+01, 2.3149634751e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.3149634751e+00, -5.2404210397e+00
};

// Slack bus configuration
const int IS_SLACK[14] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const double Vd_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };
const double Vq_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };

// Bus fault events
const int N_FAULTS = 1;
const int FAULT_BUS[1] = { 3 };
const double FAULT_T_START[1] = { 2.0000000000e+00 };
const double FAULT_T_END[1] = { 2.0500000000e+00 };
const double FAULT_G[1] = { 5.0000000000e+03 };
const double FAULT_B[1] = { -5.0000000000e+03 };
static int fault_active[1] = {0};

double outputs_GENROU_1[9];
double inputs_GENROU_1[4];
double outputs_GENROU_2[9];
double inputs_GENROU_2[4];
double outputs_GENROU_3[9];
double inputs_GENROU_3[4];
double outputs_GENROU_4[9];
double inputs_GENROU_4[4];
double outputs_GENROU_5[9];
double inputs_GENROU_5[4];
double outputs_EXST1_1[1];
double inputs_EXST1_1[2];
double outputs_EXST1_2[1];
double inputs_EXST1_2[2];
double outputs_EXST1_3[1];
double inputs_EXST1_3[2];
double outputs_EXST1_4[1];
double inputs_EXST1_4[2];
double outputs_EXST1_5[1];
double inputs_EXST1_5[2];
double outputs_TGOV1_1[1];
double inputs_TGOV1_1[3];
double outputs_TGOV1_2[1];
double inputs_TGOV1_2[3];
double outputs_TGOV1_3[1];
double inputs_TGOV1_3[3];
double outputs_TGOV1_4[1];
double inputs_TGOV1_4[3];
double outputs_TGOV1_5[1];
double inputs_TGOV1_5[3];
double outputs_CLOAD_2[4];
double inputs_CLOAD_2[2];
double outputs_CLOAD_3[4];
double inputs_CLOAD_3[2];
double outputs_CLOAD_4[4];
double inputs_CLOAD_4[2];
double outputs_CLOAD_5[4];
double inputs_CLOAD_5[2];
double outputs_CLOAD_6[4];
double inputs_CLOAD_6[2];
double outputs_CLOAD_9[4];
double inputs_CLOAD_9[2];
double outputs_CLOAD_10[4];
double inputs_CLOAD_10[2];
double outputs_CLOAD_11[4];
double inputs_CLOAD_11[2];
double outputs_CLOAD_12[4];
double inputs_CLOAD_12[2];
double outputs_CLOAD_13[4];
double inputs_CLOAD_13[2];
double outputs_CLOAD_14[4];
double inputs_CLOAD_14[2];

void step_GENROU_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.025;
    const double xd = 0.5;
    const double xq = 0.5;
    const double xd_prime = 0.075;
    const double xq_prime = 0.075;
    const double xd_double_prime = 0.05;
    const double xq_double_prime = 0.05;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 16.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            double J01 = omega_b / (2.0 * H);

            // d-axis inter-winding coupling (Eq' ↔ ψd)
            // From the standard eqs:
            //   dEq'/dt = ... − Eq'/Td0' + ...
            //   dψd/dt  = Eq'/Td0'' − ψd/Td0'' + ...
            //
            // The coupling dψd/dt ∝ Eq' and dEq'/dt ∝ ψd can be split into
            // skew-symmetric (J) and symmetric (R) parts.
            //
            // dψd/dt = Eq'/Td0'' → acts on ∂H/∂Eq' = Eq'/(xd−xl)
            //   coefficient of ∂H/∂Eq': (xd−xl)/Td0'' → this splits:
            //   J[3,2] = +β_d,  J[2,3] = −β_d for the skew part
            //
            // Similarly for q-axis (Ed' ↔ ψq)
            //
            // However, the cleanest decomposition keeps the standard form
            // and identifies the natural J, R from the physical structure.
            //
            // Using the "straightforward" decomposition:
            //   dxdt_i = Σ_j (J_ij − R_ij) · dH/dx_j + g_ik · u_k
            //
            // We match each GENROU equation term-by-term.

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- δ equation ---
            // dδ/dt = ωb·(ω − 1) = ωb/(2H) · [2H(ω−1)] = J[0,1]·∂H/∂ω
            dxdt[0] = J01 * dH_domega;

            // --- ω equation ---
            // Standard swing equation: dω/dt = (ωb/(2H)) · (Tm − Te − D(ω−1))
            //
            // In PHS form: ẋ₁ = (J − R)∇H + g·u
            //   J[1,0] = −ωb/(2H) = −J01,  J[0,1] = J01  (skew-symmetric)
            //   R[1,1] = D·ωb / (4H²)  →  −R[1,1]·∂H/∂ω = −D·ωb·(ω−1)/(2H) ✓
            //   g_Tm = ωb/(2H) = J01
            //   Te port: −J01 · Te  (network coupling)
            //
            // J01 = omega_b / (2H) is already defined above.
            dxdt[1] = -J01 * dH_ddelta                        // = 0 (∂H/∂δ = 0)
                      - (D * omega_b / (4.0*H*H)) * dH_domega // damping: −D·ωb·(ω−1)/(2H)
                      + J01 * Tm                               // mechanical input: ωb·Tm/(2H)
                      - J01 * Te;                              // electrical port:  ωb·Te/(2H)

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        
}
void step_GENROU_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.025;
    const double xd = 0.5;
    const double xq = 0.5;
    const double xd_prime = 0.075;
    const double xq_prime = 0.075;
    const double xd_double_prime = 0.05;
    const double xq_double_prime = 0.05;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 16.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        
}
void step_GENROU_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 4.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            double J01 = omega_b / (2.0 * H);

            // d-axis inter-winding coupling (Eq' ↔ ψd)
            // From the standard eqs:
            //   dEq'/dt = ... − Eq'/Td0' + ...
            //   dψd/dt  = Eq'/Td0'' − ψd/Td0'' + ...
            //
            // The coupling dψd/dt ∝ Eq' and dEq'/dt ∝ ψd can be split into
            // skew-symmetric (J) and symmetric (R) parts.
            //
            // dψd/dt = Eq'/Td0'' → acts on ∂H/∂Eq' = Eq'/(xd−xl)
            //   coefficient of ∂H/∂Eq': (xd−xl)/Td0'' → this splits:
            //   J[3,2] = +β_d,  J[2,3] = −β_d for the skew part
            //
            // Similarly for q-axis (Ed' ↔ ψq)
            //
            // However, the cleanest decomposition keeps the standard form
            // and identifies the natural J, R from the physical structure.
            //
            // Using the "straightforward" decomposition:
            //   dxdt_i = Σ_j (J_ij − R_ij) · dH/dx_j + g_ik · u_k
            //
            // We match each GENROU equation term-by-term.

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- δ equation ---
            // dδ/dt = ωb·(ω − 1) = ωb/(2H) · [2H(ω−1)] = J[0,1]·∂H/∂ω
            dxdt[0] = J01 * dH_domega;

            // --- ω equation ---
            // Standard swing equation: dω/dt = (ωb/(2H)) · (Tm − Te − D(ω−1))
            //
            // In PHS form: ẋ₁ = (J − R)∇H + g·u
            //   J[1,0] = −ωb/(2H) = −J01,  J[0,1] = J01  (skew-symmetric)
            //   R[1,1] = D·ωb / (4H²)  →  −R[1,1]·∂H/∂ω = −D·ωb·(ω−1)/(2H) ✓
            //   g_Tm = ωb/(2H) = J01
            //   Te port: −J01 · Te  (network coupling)
            //
            // J01 = omega_b / (2H) is already defined above.
            dxdt[1] = -J01 * dH_ddelta                        // = 0 (∂H/∂δ = 0)
                      - (D * omega_b / (4.0*H*H)) * dH_domega // damping: −D·ωb·(ω−1)/(2H)
                      + J01 * Tm                               // mechanical input: ωb·Tm/(2H)
                      - J01 * Te;                              // electrical port:  ωb·Te/(2H)

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        
}
void step_GENROU_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 4.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        
}
void step_GENROU_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            double J01 = omega_b / (2.0 * H);

            // d-axis inter-winding coupling (Eq' ↔ ψd)
            // From the standard eqs:
            //   dEq'/dt = ... − Eq'/Td0' + ...
            //   dψd/dt  = Eq'/Td0'' − ψd/Td0'' + ...
            //
            // The coupling dψd/dt ∝ Eq' and dEq'/dt ∝ ψd can be split into
            // skew-symmetric (J) and symmetric (R) parts.
            //
            // dψd/dt = Eq'/Td0'' → acts on ∂H/∂Eq' = Eq'/(xd−xl)
            //   coefficient of ∂H/∂Eq': (xd−xl)/Td0'' → this splits:
            //   J[3,2] = +β_d,  J[2,3] = −β_d for the skew part
            //
            // Similarly for q-axis (Ed' ↔ ψq)
            //
            // However, the cleanest decomposition keeps the standard form
            // and identifies the natural J, R from the physical structure.
            //
            // Using the "straightforward" decomposition:
            //   dxdt_i = Σ_j (J_ij − R_ij) · dH/dx_j + g_ik · u_k
            //
            // We match each GENROU equation term-by-term.

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- δ equation ---
            // dδ/dt = ωb·(ω − 1) = ωb/(2H) · [2H(ω−1)] = J[0,1]·∂H/∂ω
            dxdt[0] = J01 * dH_domega;

            // --- ω equation ---
            // Standard swing equation: dω/dt = (ωb/(2H)) · (Tm − Te − D(ω−1))
            //
            // In PHS form: ẋ₁ = (J − R)∇H + g·u
            //   J[1,0] = −ωb/(2H) = −J01,  J[0,1] = J01  (skew-symmetric)
            //   R[1,1] = D·ωb / (4H²)  →  −R[1,1]·∂H/∂ω = −D·ωb·(ω−1)/(2H) ✓
            //   g_Tm = ωb/(2H) = J01
            //   Te port: −J01 · Te  (network coupling)
            //
            // J01 = omega_b / (2H) is already defined above.
            dxdt[1] = -J01 * dH_ddelta                        // = 0 (∂H/∂δ = 0)
                      - (D * omega_b / (4.0*H*H)) * dH_domega // damping: −D·ωb·(ω−1)/(2H)
                      + J01 * Tm                               // mechanical input: ωb·Tm/(2H)
                      - J01 * Te;                              // electrical port:  ωb·Te/(2H)

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        
}
void step_GENROU_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        
}
void step_GENROU_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            double J01 = omega_b / (2.0 * H);

            // d-axis inter-winding coupling (Eq' ↔ ψd)
            // From the standard eqs:
            //   dEq'/dt = ... − Eq'/Td0' + ...
            //   dψd/dt  = Eq'/Td0'' − ψd/Td0'' + ...
            //
            // The coupling dψd/dt ∝ Eq' and dEq'/dt ∝ ψd can be split into
            // skew-symmetric (J) and symmetric (R) parts.
            //
            // dψd/dt = Eq'/Td0'' → acts on ∂H/∂Eq' = Eq'/(xd−xl)
            //   coefficient of ∂H/∂Eq': (xd−xl)/Td0'' → this splits:
            //   J[3,2] = +β_d,  J[2,3] = −β_d for the skew part
            //
            // Similarly for q-axis (Ed' ↔ ψq)
            //
            // However, the cleanest decomposition keeps the standard form
            // and identifies the natural J, R from the physical structure.
            //
            // Using the "straightforward" decomposition:
            //   dxdt_i = Σ_j (J_ij − R_ij) · dH/dx_j + g_ik · u_k
            //
            // We match each GENROU equation term-by-term.

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- δ equation ---
            // dδ/dt = ωb·(ω − 1) = ωb/(2H) · [2H(ω−1)] = J[0,1]·∂H/∂ω
            dxdt[0] = J01 * dH_domega;

            // --- ω equation ---
            // Standard swing equation: dω/dt = (ωb/(2H)) · (Tm − Te − D(ω−1))
            //
            // In PHS form: ẋ₁ = (J − R)∇H + g·u
            //   J[1,0] = −ωb/(2H) = −J01,  J[0,1] = J01  (skew-symmetric)
            //   R[1,1] = D·ωb / (4H²)  →  −R[1,1]·∂H/∂ω = −D·ωb·(ω−1)/(2H) ✓
            //   g_Tm = ωb/(2H) = J01
            //   Te port: −J01 · Te  (network coupling)
            //
            // J01 = omega_b / (2H) is already defined above.
            dxdt[1] = -J01 * dH_ddelta                        // = 0 (∂H/∂δ = 0)
                      - (D * omega_b / (4.0*H*H)) * dH_domega // damping: −D·ωb·(ω−1)/(2H)
                      + J01 * Tm                               // mechanical input: ωb·Tm/(2H)
                      - J01 * Te;                              // electrical port:  ωb·Te/(2H)

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        
}
void step_GENROU_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        
}
void step_GENROU_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            // ============================================================
            // GENROU Port-Hamiltonian Dynamics
            // ẋ = (J − R) ∇H + g(x,V) · u
            // ============================================================

            double V_Re = inputs[0];
            double V_Im = inputs[1];
            double Tm    = inputs[2];
            double Efd   = inputs[3];

            double delta = x[0];
            double omega = x[1];
            double Eq_p  = x[2];
            double psi_d = x[3];
            double Ed_p  = x[4];
            double psi_q = x[5];

            double sin_d = sin(delta);
            double cos_d = cos(delta);

            // --- Park transform: network → dq frame ---
            double vd = V_Re * sin_d - V_Im * cos_d;
            double vq = V_Re * cos_d + V_Im * sin_d;

            // --- Sub-transient flux linkages ---
            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            // --- Stator algebraic: solve for (id, iq) ---
            // Partially-neglected speed variation: omega multiplies subtransient fluxes
            // (PF Eq. 73-74: u''_d = -omega*psi''_q, u''_q = omega*psi''_d, n=1 in impedance)
            double rhs_d = vd + omega * psi_q_pp;
            double rhs_q = vq - omega * psi_d_pp;
            double det_s = ra*ra + xd_double_prime * xq_double_prime;
            double id = (-ra * rhs_d - xq_double_prime * rhs_q) / det_s;
            double iq = (xd_double_prime * rhs_d - ra * rhs_q) / det_s;

            // --- Electrical torque ---
            // Divide by omega: (vd*id + vq*iq) is air-gap power in pu;
            // torque in pu = power / omega (standard per-unit convention).
            double Te = (vd * id + vq * iq) / omega;

            // ============================================================
            // Hamiltonian gradient  ∂H/∂x
            // ============================================================
            // H = H·(ω−1)² + Eq'²/[2(xd−xl)] + ψd²/[2(xd'−xd'')]
            //   + Ed'²/[2(xq−xl)] + ψq²/[2(xq'−xq'')]
            //
            // ∂H/∂δ    = 0   (cyclic coordinate)
            // ∂H/∂ω    = 2H·(ω − 1)
            // ∂H/∂Eq'  = Eq' / (xd − xl)
            // ∂H/∂ψd   = ψd / (xd' − xd'')
            // ∂H/∂Ed'  = Ed' / (xq − xl)
            // ∂H/∂ψq   = ψq / (xq' − xq'')

            double dH_ddelta = 0.0;
            double dH_domega = 2.0 * H * (omega - 1.0);
            double dH_dEqp   = Eq_p / (xd - xl);
            double dH_dpsid  = psi_d / (xd_prime - xd_double_prime);
            double dH_dEdp   = Ed_p / (xq - xl);
            double dH_dpsiq  = psi_q / (xq_prime - xq_double_prime);

            // ============================================================
            // J matrix (skew-symmetric interconnection)
            // ============================================================
            // J couples:
            //   δ ↔ ω :  J[0,1] = ωb/(2H),  J[1,0] = −ωb/(2H)
            //   Eq' ↔ ψd:  J[2,3] = −αd,    J[3,2] =  αd
            //   Ed' ↔ ψq:  J[4,5] =  αq,    J[5,4] = −αq
            //
            // where αd = 1/(Td0'·(xd'−xd'')), αq = 1/(Tq0'·(xq'−xq''))
            // These inter-winding couplings arise from:
            //   dEq'/dt has +ψd/(Td0'·(xd'−xd'')) via the flux coupling
            //   dψd/dt  has −Eq'/(Td0''·(xd−xl)) via the back-EMF coupling

            double J01 = omega_b / (2.0 * H);

            // d-axis inter-winding coupling (Eq' ↔ ψd)
            // From the standard eqs:
            //   dEq'/dt = ... − Eq'/Td0' + ...
            //   dψd/dt  = Eq'/Td0'' − ψd/Td0'' + ...
            //
            // The coupling dψd/dt ∝ Eq' and dEq'/dt ∝ ψd can be split into
            // skew-symmetric (J) and symmetric (R) parts.
            //
            // dψd/dt = Eq'/Td0'' → acts on ∂H/∂Eq' = Eq'/(xd−xl)
            //   coefficient of ∂H/∂Eq': (xd−xl)/Td0'' → this splits:
            //   J[3,2] = +β_d,  J[2,3] = −β_d for the skew part
            //
            // Similarly for q-axis (Ed' ↔ ψq)
            //
            // However, the cleanest decomposition keeps the standard form
            // and identifies the natural J, R from the physical structure.
            //
            // Using the "straightforward" decomposition:
            //   dxdt_i = Σ_j (J_ij − R_ij) · dH/dx_j + g_ik · u_k
            //
            // We match each GENROU equation term-by-term.

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- δ equation ---
            // dδ/dt = ωb·(ω − 1) = ωb/(2H) · [2H(ω−1)] = J[0,1]·∂H/∂ω
            dxdt[0] = J01 * dH_domega;

            // --- ω equation ---
            // Standard swing equation: dω/dt = (ωb/(2H)) · (Tm − Te − D(ω−1))
            //
            // In PHS form: ẋ₁ = (J − R)∇H + g·u
            //   J[1,0] = −ωb/(2H) = −J01,  J[0,1] = J01  (skew-symmetric)
            //   R[1,1] = D·ωb / (4H²)  →  −R[1,1]·∂H/∂ω = −D·ωb·(ω−1)/(2H) ✓
            //   g_Tm = ωb/(2H) = J01
            //   Te port: −J01 · Te  (network coupling)
            //
            // J01 = omega_b / (2H) is already defined above.
            dxdt[1] = -J01 * dH_ddelta                        // = 0 (∂H/∂δ = 0)
                      - (D * omega_b / (4.0*H*H)) * dH_domega // damping: −D·ωb·(ω−1)/(2H)
                      + J01 * Tm                               // mechanical input: ωb·Tm/(2H)
                      - J01 * Te;                              // electrical port:  ωb·Te/(2H)

            // --- Eq' equation (d-axis field winding) ---
            // dEq'/dt = [Efd − Eq' − (xd−xd')·id] / Td0'
            //
            // PHS decomposition:
            //   −Eq'/Td0' = −R[2,2]·∂H/∂Eq'  where R[2,2] = (xd−xl)/(Td0'·(xd−xl)) = 1/Td0'
            //                                  and ∂H/∂Eq' = Eq'/(xd−xl)
            //   Wait: −R[2,2]·∂H/∂Eq' = −(1/Td0')·Eq'/(xd−xl)·(xd−xl) = −Eq'/Td0' ✓
            //   ... but we need R[2,2]·(xd−xl) = (xd−xl)/Td0'? No.
            //
            //   Let's be precise. We want:
            //     −R22 · dH_dEqp = −Eq'/Td0'
            //     −R22 · Eq'/(xd−xl) = −Eq'/Td0'
            //     R22 = (xd−xl)/Td0'
            //
            //   +Efd/Td0' = g[2,Efd] · Efd   → g[2,Efd] = 1/Td0'
            //   −(xd−xd')·id/Td0' : this is the stator-coupled port term
            double R22 = (xd - xl) / Td0_prime;
            dxdt[2] = -R22 * dH_dEqp                   // dissipation: −Eq'/Td0'
                      + Efd / Td0_prime                 // field voltage input
                      - (xd - xd_prime) * id / Td0_prime;   // stator coupling

            // --- ψd equation (d-axis damper winding) ---
            // dψd/dt = [Eq' − ψd − (xd'−xd'')·id] / Td0''
            //
            // −ψd/Td0'' = −R33·∂H/∂ψd  where R33 = (xd'−xd'')/Td0''
            //   Check: −R33·ψd/(xd'−xd'') = −ψd/Td0'' ✓
            //
            // +Eq'/Td0'' : This is the inter-winding coupling.
            //   = Eq'/(Td0''·1) — we need to express via ∂H/∂Eq' = Eq'/(xd−xl)
            //   Eq'/Td0'' = [(xd−xl)/Td0''] · [Eq'/(xd−xl)] = J32 · ∂H/∂Eq'
            //   So J[3,2] = (xd−xl)/Td0'' and J[2,3] = −(xd−xl)/Td0'' (skew)
            double J32 = (xd - xl) / Td0_double_prime;
            double R33 = (xd_prime - xd_double_prime) / Td0_double_prime;
            dxdt[3] = J32 * dH_dEqp                    // inter-winding coupling: +Eq'/Td0''
                      - R33 * dH_dpsid                  // dissipation: −ψd/Td0''
                      - (xd_prime - xd_double_prime) * id / Td0_double_prime;  // stator coupling

            // --- Ed' equation (q-axis field/transient winding) ---
            // dEd'/dt = [−Ed' + (xq−xq')·iq] / Tq0'
            //
            // −Ed'/Tq0' = −R44·∂H/∂Ed'  where R44 = (xq−xl)/Tq0'
            //   Check: −R44·Ed'/(xq−xl) = −Ed'/Tq0' ✓
            //
            // +(xq−xq')·iq/Tq0' : stator coupling
            double R44 = (xq - xl) / Tq0_prime;
            dxdt[4] = -R44 * dH_dEdp                   // dissipation: −Ed'/Tq0'
                      + (xq - xq_prime) * iq / Tq0_prime;    // stator coupling

            // --- ψq equation (q-axis damper winding) ---
            // dψq/dt = [−Ed' − ψq − (xq'−xq'')·iq] / Tq0''
            //
            // This matches the standard GENROU formulation (Sauer-Pai, PSS/E).
            // At steady-state: ψq = −Ed' − (xq'−xq'')·iq
            //
            // −ψq/Tq0'' = −R55·∂H/∂ψq  where R55 = (xq'−xq'')/Tq0''
            //   Check: −R55·ψq/(xq'−xq'') = −ψq/Tq0'' ✓
            //
            // −Ed'/Tq0'': inter-winding coupling
            //   = −[(xq−xl)/Tq0''] · [Ed'/(xq−xl)] = J54 · ∂H/∂Ed'
            //   So J[5,4] = −(xq−xl)/Tq0'' and J[4,5] = +(xq−xl)/Tq0'' (skew)
            double J54 = -(xq - xl) / Tq0_double_prime;
            double R55 = (xq_prime - xq_double_prime) / Tq0_double_prime;
            dxdt[5] = J54 * dH_dEdp                    // inter-winding coupling: −Ed'/Tq0''
                      - R55 * dH_dpsiq                  // dissipation: −ψq/Tq0''
                      - (xq_prime - xq_double_prime) * iq / Tq0_double_prime;  // stator coupling (MINUS per standard GENROU)

            // --- Update outputs ---
            outputs[3] = Te;              // Pe = Te at ω ≈ 1
            outputs[4] = vq*id - vd*iq;   // Qe
            outputs[5] = id;
            outputs[6] = iq;
            outputs[7] = id * sin_d + iq * cos_d;   // It_Re
            outputs[8] = -id * cos_d + iq * sin_d;  // It_Im
        
}
void step_GENROU_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double D = 0.0;
    const double ra = 0.0;
    const double xl = 0.1;
    const double xd = 2.0;
    const double xq = 2.0;
    const double xd_prime = 0.3;
    const double xq_prime = 0.3;
    const double xd_double_prime = 0.2;
    const double xq_double_prime = 0.2;
    const double Td0_prime = 6.666667;
    const double Tq0_prime = 6.666667;
    const double Td0_double_prime = 0.075;
    const double Tq0_double_prime = 0.075;
    const double H = 5.0;
    const double omega_b = 2.0 * M_PI * 60.0;
    // --- Kernel ---

            double delta = x[0];
            double sin_d = sin(delta);
            double cos_d = cos(delta);

            double Eq_p = x[2];
            double psi_d = x[3];
            double Ed_p = x[4];
            double psi_q = x[5];

            double k_d = (xd_double_prime - xl) / (xd_prime - xl);
            double k_q = (xq_double_prime - xl) / (xq_prime - xl);
            double psi_d_pp = Eq_p * k_d + psi_d * (1.0 - k_d);
            double psi_q_pp = -Ed_p * k_q + psi_q * (1.0 - k_q);

            double det = ra*ra + xd_double_prime * xq_double_prime;
            // Partially-neglected speed variation: scale subtransient EMF by omega
            // (PF Technical Reference Eq. 74: u''_d = -n*psi''_q, u''_q = n*psi''_d)
            double omega = x[1];
            double id_no = omega * (-ra * psi_q_pp + xq_double_prime * psi_d_pp) / det;
            double iq_no = omega * ( xd_double_prime * psi_q_pp + ra * psi_d_pp) / det;

            double I_Re = id_no * sin_d + iq_no * cos_d;
            double I_Im = -id_no * cos_d + iq_no * sin_d;

            outputs[0] = I_Re;
            outputs[1] = I_Im;
            outputs[2] = x[1];
            outputs[3] = 0.0;
            outputs[4] = 0.0;
            outputs[5] = id_no;
            outputs[6] = iq_no;
            outputs[7] = I_Re;
            outputs[8] = I_Im;
        
}
void step_EXST1_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        
}
void step_EXST1_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        
}
void step_EXST1_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        
}
void step_EXST1_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        
}
void step_EXST1_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        
}
void step_EXST1_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        
}
void step_EXST1_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        
}
void step_EXST1_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        
}
void step_EXST1_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            // ============================================================
            // EXST1 Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → dissipation via R.
            // The KF/TF washout provides derivative feedback on Efd.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double Vf  = x[3];

            double Vterm = inputs[0];
            double Vref  = inputs[1];

            // 1. Voltage transducer lag
            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Washout rate feedback: Vf_out = KF*(Vr - Vf)/TF
            double TF_eff = (TF > 1e-4) ? TF : 1e-4;
            double Vf_out = KF * (Vr - Vf) / TF_eff;

            // 3. Voltage error + input limiter
            double Verr = Vref - Vm - Vf_out;
            if (Verr > VIMAX) Verr = VIMAX;
            if (Verr < VIMIN) Verr = VIMIN;

            // 4. Lead-lag compensator (TC/TB)
            double vll_out;
            if (TB > 1e-4) {
                dxdt[1] = (Verr - LLx) / TB;
                vll_out = LLx + (TC / TB) * (Verr - LLx);
            } else {
                dxdt[1] = 0.0;
                vll_out = Verr;
            }

            // 5. Voltage regulator (KA/TA) with dynamic ceiling anti-windup
            double XadIfd_eff = (Vr > 0.0) ? Vr : 0.0;
            double VRMAX_dyn = VRMAX - KC * XadIfd_eff;
            double VRMIN_dyn = VRMIN + KC * XadIfd_eff;
            double TA_eff = (TA > 1e-4) ? TA : 1e-4;
            double dVr = (KA * vll_out - Vr) / TA_eff;
            if (Vr >= VRMAX_dyn && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN_dyn && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 6. Washout filter state: dVf/dt = (Vr - Vf) / TF
            dxdt[3] = (Vr - Vf) / TF_eff;
        
}
void step_EXST1_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 1000.0;
    const double VIMAX = 99.0;
    const double VIMIN = -99.0;
    const double TC = 0.0;
    const double TB = 1000.0;
    const double KA = 10.0;
    const double TA = 1000.0;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KC = 0.0;
    const double KF = 0.0;
    const double TF = 1000.0;
    // --- Kernel ---

            double Vr_o  = x[2];
            double Xad_o = (Vr_o > 0.0) ? Vr_o : 0.0;
            double Efd_o = Vr_o;
            double ceil_o = VRMAX - KC * Xad_o;
            double flr_o  = VRMIN + KC * Xad_o;
            if (Efd_o > ceil_o) Efd_o = ceil_o;
            if (Efd_o < flr_o)  Efd_o = flr_o;
            outputs[0] = Efd_o;
        
}
void step_TGOV1_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'TGOV1_PHS'

double x1 = x[0];
double x2 = x[1];
double xi = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double R_eff = (R > 1e-4) ? R : 1e-4;
double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
double _dx0_raw = Pref/(R_eff*T1_eff) - x1/T1_eff - omega/(R_eff*T1_eff) + u_agc/(R_eff*T1_eff);
if (x1 >= VMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= VMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= VMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= VMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_TGOV1_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'TGOV1_PHS'

double x1 = x[0];
double x2 = x[1];
double xi = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double R_eff = (R > 1e-4) ? R : 1e-4;
double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
double _dx0_raw = Pref/(R_eff*T1_eff) - x1/T1_eff - omega/(R_eff*T1_eff) + u_agc/(R_eff*T1_eff);
if (x1 >= VMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= VMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= VMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= VMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_TGOV1_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'TGOV1_PHS'

double x1 = x[0];
double x2 = x[1];
double xi = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double R_eff = (R > 1e-4) ? R : 1e-4;
double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
double _dx0_raw = Pref/(R_eff*T1_eff) - x1/T1_eff - omega/(R_eff*T1_eff) + u_agc/(R_eff*T1_eff);
if (x1 >= VMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= VMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= VMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= VMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_TGOV1_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'TGOV1_PHS'

double x1 = x[0];
double x2 = x[1];
double xi = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double R_eff = (R > 1e-4) ? R : 1e-4;
double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
double _dx0_raw = Pref/(R_eff*T1_eff) - x1/T1_eff - omega/(R_eff*T1_eff) + u_agc/(R_eff*T1_eff);
if (x1 >= VMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= VMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= VMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= VMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_TGOV1_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'TGOV1_PHS'

double x1 = x[0];
double x2 = x[1];
double xi = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double R_eff = (R > 1e-4) ? R : 1e-4;
double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
double _dx0_raw = Pref/(R_eff*T1_eff) - x1/T1_eff - omega/(R_eff*T1_eff) + u_agc/(R_eff*T1_eff);
if (x1 >= VMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= VMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= VMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= VMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 999.0;
    const double VMAX = 5.0;
    const double VMIN = 0.0;
    const double T1 = 1000.0;
    const double T2 = 1.0;
    const double T3 = 1000.0;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 0.5;
    const double xi_min = -0.5;
    // --- Kernel ---

            double x1 = x[0];
            double x2 = x[1];
            double xi = x[2];
            double Tm_droop = x2 + (T2/T3) * (x1 - x2);
            double Tm_total = Tm_droop + xi;
            if (Tm_total > VMAX) Tm_total = VMAX;
            if (Tm_total < VMIN) Tm_total = VMIN;
            outputs[0] = Tm_total;
        
}
void step_CLOAD_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 2;
    const double P0 = 0.217;
    const double Q0 = 0.127;
    const double V0 = 1.045;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 2;
    const double P0 = 0.217;
    const double Q0 = 0.127;
    const double V0 = 1.045;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 1.987133994185e-01;
        double B_load = -1.162977038071e-01;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 3;
    const double P0 = 0.942;
    const double Q0 = 0.19;
    const double V0 = 1.01;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 3;
    const double P0 = 0.942;
    const double Q0 = 0.19;
    const double V0 = 1.01;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 9.234388785413e-01;
        double B_load = -1.862562493873e-01;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 4;
    const double P0 = 0.478;
    const double Q0 = -0.039;
    const double V0 = 1.028;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 4;
    const double P0 = 0.478;
    const double Q0 = -0.039;
    const double V0 = 1.028;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 4.523157050069e-01;
        double B_load = 3.690441944617e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 5;
    const double P0 = 0.076;
    const double Q0 = 0.016;
    const double V0 = 1.03;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 5;
    const double P0 = 0.076;
    const double Q0 = 0.016;
    const double V0 = 1.03;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 7.163728909417e-02;
        double B_load = -1.508153454614e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_6(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 6;
    const double P0 = 0.112;
    const double Q0 = 0.075;
    const double V0 = 1.07;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_6_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 6;
    const double P0 = 0.112;
    const double Q0 = 0.075;
    const double V0 = 1.07;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 9.782513756660e-02;
        double B_load = -6.550790462049e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_9(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 9;
    const double P0 = 0.295;
    const double Q0 = 0.166;
    const double V0 = 1.06;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_9_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 9;
    const double P0 = 0.295;
    const double Q0 = 0.166;
    const double V0 = 1.06;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 2.625489498042e-01;
        double B_load = -1.477394090424e-01;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_10(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 10;
    const double P0 = 0.09;
    const double Q0 = 0.058;
    const double V0 = 1.054;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_10_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 10;
    const double P0 = 0.09;
    const double Q0 = 0.058;
    const double V0 = 1.054;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 8.101422609810e-02;
        double B_load = -5.220916792989e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_11(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 11;
    const double P0 = 0.035;
    const double Q0 = 0.018;
    const double V0 = 1.058;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_11_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 11;
    const double P0 = 0.035;
    const double Q0 = 0.018;
    const double V0 = 1.058;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 3.126775561837e-02;
        double B_load = -1.608056003230e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_12(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 12;
    const double P0 = 0.061;
    const double Q0 = 0.016;
    const double V0 = 1.056;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_12_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 12;
    const double P0 = 0.061;
    const double Q0 = 0.016;
    const double V0 = 1.056;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 5.470184802571e-02;
        double B_load = -1.434802571166e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_13(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 13;
    const double P0 = 0.135;
    const double Q0 = 0.058;
    const double V0 = 1.051;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_13_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 13;
    const double P0 = 0.135;
    const double Q0 = 0.058;
    const double V0 = 1.051;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 1.222160762121e-01;
        double B_load = -5.250764755781e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
}
void step_CLOAD_14(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double bus = 14;
    const double P0 = 0.149;
    const double Q0 = 0.05;
    const double V0 = 1.038;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

}
void step_CLOAD_14_out(const double* x, const double* inputs, double* outputs, double t) {
    const double bus = 14;
    const double P0 = 0.149;
    const double Q0 = 0.05;
    const double V0 = 1.038;
    const double kpf = 1.0;
    const double kqf = 1.0;
    // --- Kernel ---

        double Vd = inputs[0];
        double Vq = inputs[1];
        double Vmag2 = Vd*Vd + Vq*Vq;

        // Constant-impedance load current: I = Y_load * V
        // Y_load = G_load + j*B_load where G = P0/V0^2, B = -Q0/V0^2
        double G_load = 1.382902498877e-01;
        double B_load = -4.640612412339e-02;

        double Id_load = G_load*Vd - B_load*Vq;
        double Iq_load = G_load*Vq + B_load*Vd;

        // P = Vd*Id + Vq*Iq,  Q = Vq*Id - Vd*Iq (consumed)
        double Pload = Vd*Id_load + Vq*Iq_load;
        double Qload = Vq*Id_load - Vd*Iq_load;

        // Output: Id, Iq are zero here (base load handled in Y-bus,
        // freq correction handled in DAE compiler KCL)
        outputs[0] = 0.0;  // Id (not injected from component)
        outputs[1] = 0.0;  // Iq (not injected from component)
        outputs[2] = Pload; // diagnostic: P consumed
        outputs[3] = Qload; // diagnostic: Q consumed
        
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

    // dq-frame voltages for exciter inputs
    double vd_dq_GENROU_1 = Vd_net[0]*sin(x[0]) - Vq_net[0]*cos(x[0]);
    double vq_dq_GENROU_1 = Vd_net[0]*cos(x[0]) + Vq_net[0]*sin(x[0]);
    double vd_dq_GENROU_2 = Vd_net[1]*sin(x[6]) - Vq_net[1]*cos(x[6]);
    double vq_dq_GENROU_2 = Vd_net[1]*cos(x[6]) + Vq_net[1]*sin(x[6]);
    double vd_dq_GENROU_3 = Vd_net[2]*sin(x[12]) - Vq_net[2]*cos(x[12]);
    double vq_dq_GENROU_3 = Vd_net[2]*cos(x[12]) + Vq_net[2]*sin(x[12]);
    double vd_dq_GENROU_4 = Vd_net[5]*sin(x[18]) - Vq_net[5]*cos(x[18]);
    double vq_dq_GENROU_4 = Vd_net[5]*cos(x[18]) + Vq_net[5]*sin(x[18]);
    double vd_dq_GENROU_5 = Vd_net[7]*sin(x[24]) - Vq_net[7]*cos(x[24]);
    double vq_dq_GENROU_5 = Vd_net[7]*cos(x[24]) + Vq_net[7]*sin(x[24]);

    double Id_inj[14] = {0};
    double Iq_inj[14] = {0};

    // --- 1. Compute Outputs & Gather Injections ---
    { // GENROU_1
        inputs_GENROU_1[0] = Vd_net[0]; // Vd
        inputs_GENROU_1[1] = Vq_net[0]; // Vq
        inputs_GENROU_1[2] = outputs_TGOV1_1[0]; // Tm
        inputs_GENROU_1[3] = outputs_EXST1_1[0]; // Efd
        step_GENROU_1_out(&x[0], inputs_GENROU_1, outputs_GENROU_1, t);
        Id_inj[0] += outputs_GENROU_1[0];
        Iq_inj[0] += outputs_GENROU_1[1];
    }
    { // GENROU_2
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_2[0]; // Efd
        step_GENROU_2_out(&x[6], inputs_GENROU_2, outputs_GENROU_2, t);
        Id_inj[1] += outputs_GENROU_2[0];
        Iq_inj[1] += outputs_GENROU_2[1];
    }
    { // GENROU_3
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_3[3] = outputs_EXST1_3[0]; // Efd
        step_GENROU_3_out(&x[12], inputs_GENROU_3, outputs_GENROU_3, t);
        Id_inj[2] += outputs_GENROU_3[0];
        Iq_inj[2] += outputs_GENROU_3[1];
    }
    { // GENROU_4
        inputs_GENROU_4[0] = Vd_net[5]; // Vd
        inputs_GENROU_4[1] = Vq_net[5]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_4[0]; // Tm
        inputs_GENROU_4[3] = outputs_EXST1_4[0]; // Efd
        step_GENROU_4_out(&x[18], inputs_GENROU_4, outputs_GENROU_4, t);
        Id_inj[5] += outputs_GENROU_4[0];
        Iq_inj[5] += outputs_GENROU_4[1];
    }
    { // GENROU_5
        inputs_GENROU_5[0] = Vd_net[7]; // Vd
        inputs_GENROU_5[1] = Vq_net[7]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_5[0]; // Tm
        inputs_GENROU_5[3] = outputs_EXST1_5[0]; // Efd
        step_GENROU_5_out(&x[24], inputs_GENROU_5, outputs_GENROU_5, t);
        Id_inj[7] += outputs_GENROU_5[0];
        Iq_inj[7] += outputs_GENROU_5[1];
    }
    { // EXST1_1
        inputs_EXST1_1[0] = Vterm_net[0]; // Vterm
        inputs_EXST1_1[1] = 1.177164293192513; // Vref
        step_EXST1_1_out(&x[30], inputs_EXST1_1, outputs_EXST1_1, t);
    }
    { // EXST1_2
        inputs_EXST1_2[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_2[1] = 1.1676313662205968; // Vref
        step_EXST1_2_out(&x[34], inputs_EXST1_2, outputs_EXST1_2, t);
    }
    { // EXST1_3
        inputs_EXST1_3[0] = Vterm_net[2]; // Vterm
        inputs_EXST1_3[1] = 1.1300276634403366; // Vref
        step_EXST1_3_out(&x[38], inputs_EXST1_3, outputs_EXST1_3, t);
    }
    { // EXST1_4
        inputs_EXST1_4[0] = Vterm_net[5]; // Vterm
        inputs_EXST1_4[1] = 1.2314682829419707; // Vref
        step_EXST1_4_out(&x[42], inputs_EXST1_4, outputs_EXST1_4, t);
    }
    { // EXST1_5
        inputs_EXST1_5[0] = Vterm_net[7]; // Vterm
        inputs_EXST1_5[1] = 1.2442089403946075; // Vref
        step_EXST1_5_out(&x[46], inputs_EXST1_5, outputs_EXST1_5, t);
    }
    { // TGOV1_1
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1177.140384165324; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1_out(&x[50], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // TGOV1_2
        inputs_TGOV1_2[0] = outputs_GENROU_2[2]; // omega
        inputs_TGOV1_2[1] = 330.83038143298836; // Pref
        inputs_TGOV1_2[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_2_out(&x[53], inputs_TGOV1_2, outputs_TGOV1_2, t);
    }
    { // TGOV1_3
        inputs_TGOV1_3[0] = outputs_GENROU_3[2]; // omega
        inputs_TGOV1_3[1] = 379.2881366194895; // Pref
        inputs_TGOV1_3[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_3_out(&x[56], inputs_TGOV1_3, outputs_TGOV1_3, t);
    }
    { // TGOV1_4
        inputs_TGOV1_4[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_4[1] = 388.02516013150716; // Pref
        inputs_TGOV1_4[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_4_out(&x[59], inputs_TGOV1_4, outputs_TGOV1_4, t);
    }
    { // TGOV1_5
        inputs_TGOV1_5[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_5[1] = 367.84989405934886; // Pref
        inputs_TGOV1_5[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_5_out(&x[62], inputs_TGOV1_5, outputs_TGOV1_5, t);
    }
    { // CLOAD_2
        inputs_CLOAD_2[0] = Vd_net[1]; // Vd
        inputs_CLOAD_2[1] = Vq_net[1]; // Vq
        step_CLOAD_2_out(&x[65], inputs_CLOAD_2, outputs_CLOAD_2, t);
        Id_inj[1] += outputs_CLOAD_2[0];
        Iq_inj[1] += outputs_CLOAD_2[1];
    }
    { // CLOAD_3
        inputs_CLOAD_3[0] = Vd_net[2]; // Vd
        inputs_CLOAD_3[1] = Vq_net[2]; // Vq
        step_CLOAD_3_out(&x[65], inputs_CLOAD_3, outputs_CLOAD_3, t);
        Id_inj[2] += outputs_CLOAD_3[0];
        Iq_inj[2] += outputs_CLOAD_3[1];
    }
    { // CLOAD_4
        inputs_CLOAD_4[0] = Vd_net[3]; // Vd
        inputs_CLOAD_4[1] = Vq_net[3]; // Vq
        step_CLOAD_4_out(&x[65], inputs_CLOAD_4, outputs_CLOAD_4, t);
        Id_inj[3] += outputs_CLOAD_4[0];
        Iq_inj[3] += outputs_CLOAD_4[1];
    }
    { // CLOAD_5
        inputs_CLOAD_5[0] = Vd_net[4]; // Vd
        inputs_CLOAD_5[1] = Vq_net[4]; // Vq
        step_CLOAD_5_out(&x[65], inputs_CLOAD_5, outputs_CLOAD_5, t);
        Id_inj[4] += outputs_CLOAD_5[0];
        Iq_inj[4] += outputs_CLOAD_5[1];
    }
    { // CLOAD_6
        inputs_CLOAD_6[0] = Vd_net[5]; // Vd
        inputs_CLOAD_6[1] = Vq_net[5]; // Vq
        step_CLOAD_6_out(&x[65], inputs_CLOAD_6, outputs_CLOAD_6, t);
        Id_inj[5] += outputs_CLOAD_6[0];
        Iq_inj[5] += outputs_CLOAD_6[1];
    }
    { // CLOAD_9
        inputs_CLOAD_9[0] = Vd_net[8]; // Vd
        inputs_CLOAD_9[1] = Vq_net[8]; // Vq
        step_CLOAD_9_out(&x[65], inputs_CLOAD_9, outputs_CLOAD_9, t);
        Id_inj[8] += outputs_CLOAD_9[0];
        Iq_inj[8] += outputs_CLOAD_9[1];
    }
    { // CLOAD_10
        inputs_CLOAD_10[0] = Vd_net[9]; // Vd
        inputs_CLOAD_10[1] = Vq_net[9]; // Vq
        step_CLOAD_10_out(&x[65], inputs_CLOAD_10, outputs_CLOAD_10, t);
        Id_inj[9] += outputs_CLOAD_10[0];
        Iq_inj[9] += outputs_CLOAD_10[1];
    }
    { // CLOAD_11
        inputs_CLOAD_11[0] = Vd_net[10]; // Vd
        inputs_CLOAD_11[1] = Vq_net[10]; // Vq
        step_CLOAD_11_out(&x[65], inputs_CLOAD_11, outputs_CLOAD_11, t);
        Id_inj[10] += outputs_CLOAD_11[0];
        Iq_inj[10] += outputs_CLOAD_11[1];
    }
    { // CLOAD_12
        inputs_CLOAD_12[0] = Vd_net[11]; // Vd
        inputs_CLOAD_12[1] = Vq_net[11]; // Vq
        step_CLOAD_12_out(&x[65], inputs_CLOAD_12, outputs_CLOAD_12, t);
        Id_inj[11] += outputs_CLOAD_12[0];
        Iq_inj[11] += outputs_CLOAD_12[1];
    }
    { // CLOAD_13
        inputs_CLOAD_13[0] = Vd_net[12]; // Vd
        inputs_CLOAD_13[1] = Vq_net[12]; // Vq
        step_CLOAD_13_out(&x[65], inputs_CLOAD_13, outputs_CLOAD_13, t);
        Id_inj[12] += outputs_CLOAD_13[0];
        Iq_inj[12] += outputs_CLOAD_13[1];
    }
    { // CLOAD_14
        inputs_CLOAD_14[0] = Vd_net[13]; // Vd
        inputs_CLOAD_14[1] = Vq_net[13]; // Vq
        step_CLOAD_14_out(&x[65], inputs_CLOAD_14, outputs_CLOAD_14, t);
        Id_inj[13] += outputs_CLOAD_14[0];
        Iq_inj[13] += outputs_CLOAD_14[1];
    }

    // Refresh actual dq-frame stator currents
    {
        double psi_d_pp = x[2]*0.500000 + x[3]*(1.0-0.500000);
        double psi_q_pp = -x[4]*0.500000 + x[5]*(1.0-0.500000);
        double rhs_d = vd_dq_GENROU_1 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_1 - psi_d_pp;
        outputs_GENROU_1[5] = (-0.000000*rhs_d - 0.050000*rhs_q) / 0.002500;
        outputs_GENROU_1[6] = (0.050000*rhs_d + -0.000000*rhs_q) / 0.002500;
    }
    {
        double psi_d_pp = x[8]*0.500000 + x[9]*(1.0-0.500000);
        double psi_q_pp = -x[10]*0.500000 + x[11]*(1.0-0.500000);
        double rhs_d = vd_dq_GENROU_2 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_2 - psi_d_pp;
        outputs_GENROU_2[5] = (-0.000000*rhs_d - 0.200000*rhs_q) / 0.040000;
        outputs_GENROU_2[6] = (0.200000*rhs_d + -0.000000*rhs_q) / 0.040000;
    }
    {
        double psi_d_pp = x[14]*0.500000 + x[15]*(1.0-0.500000);
        double psi_q_pp = -x[16]*0.500000 + x[17]*(1.0-0.500000);
        double rhs_d = vd_dq_GENROU_3 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_3 - psi_d_pp;
        outputs_GENROU_3[5] = (-0.000000*rhs_d - 0.200000*rhs_q) / 0.040000;
        outputs_GENROU_3[6] = (0.200000*rhs_d + -0.000000*rhs_q) / 0.040000;
    }
    {
        double psi_d_pp = x[20]*0.500000 + x[21]*(1.0-0.500000);
        double psi_q_pp = -x[22]*0.500000 + x[23]*(1.0-0.500000);
        double rhs_d = vd_dq_GENROU_4 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_4 - psi_d_pp;
        outputs_GENROU_4[5] = (-0.000000*rhs_d - 0.200000*rhs_q) / 0.040000;
        outputs_GENROU_4[6] = (0.200000*rhs_d + -0.000000*rhs_q) / 0.040000;
    }
    {
        double psi_d_pp = x[26]*0.500000 + x[27]*(1.0-0.500000);
        double psi_q_pp = -x[28]*0.500000 + x[29]*(1.0-0.500000);
        double rhs_d = vd_dq_GENROU_5 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_5 - psi_d_pp;
        outputs_GENROU_5[5] = (-0.000000*rhs_d - 0.200000*rhs_q) / 0.040000;
        outputs_GENROU_5[6] = (0.200000*rhs_d + -0.000000*rhs_q) / 0.040000;
    }

    // --- 2. Compute Dynamics (dxdt) ---
    double dxdt[66];
    { // GENROU_1 dynamics
        inputs_GENROU_1[0] = Vd_net[0]; // Vd
        inputs_GENROU_1[1] = Vq_net[0]; // Vq
        inputs_GENROU_1[2] = outputs_TGOV1_1[0]; // Tm
        inputs_GENROU_1[3] = outputs_EXST1_1[0]; // Efd
        step_GENROU_1(&x[0], &dxdt[0], inputs_GENROU_1, outputs_GENROU_1, t);
    }
    { // GENROU_2 dynamics
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_2[0]; // Efd
        step_GENROU_2(&x[6], &dxdt[6], inputs_GENROU_2, outputs_GENROU_2, t);
    }
    { // GENROU_3 dynamics
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_3[3] = outputs_EXST1_3[0]; // Efd
        step_GENROU_3(&x[12], &dxdt[12], inputs_GENROU_3, outputs_GENROU_3, t);
    }
    { // GENROU_4 dynamics
        inputs_GENROU_4[0] = Vd_net[5]; // Vd
        inputs_GENROU_4[1] = Vq_net[5]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_4[0]; // Tm
        inputs_GENROU_4[3] = outputs_EXST1_4[0]; // Efd
        step_GENROU_4(&x[18], &dxdt[18], inputs_GENROU_4, outputs_GENROU_4, t);
    }
    { // GENROU_5 dynamics
        inputs_GENROU_5[0] = Vd_net[7]; // Vd
        inputs_GENROU_5[1] = Vq_net[7]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_5[0]; // Tm
        inputs_GENROU_5[3] = outputs_EXST1_5[0]; // Efd
        step_GENROU_5(&x[24], &dxdt[24], inputs_GENROU_5, outputs_GENROU_5, t);
    }
    { // EXST1_1 dynamics
        inputs_EXST1_1[0] = Vterm_net[0]; // Vterm
        inputs_EXST1_1[1] = 1.177164293192513; // Vref
        step_EXST1_1(&x[30], &dxdt[30], inputs_EXST1_1, outputs_EXST1_1, t);
    }
    { // EXST1_2 dynamics
        inputs_EXST1_2[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_2[1] = 1.1676313662205968; // Vref
        step_EXST1_2(&x[34], &dxdt[34], inputs_EXST1_2, outputs_EXST1_2, t);
    }
    { // EXST1_3 dynamics
        inputs_EXST1_3[0] = Vterm_net[2]; // Vterm
        inputs_EXST1_3[1] = 1.1300276634403366; // Vref
        step_EXST1_3(&x[38], &dxdt[38], inputs_EXST1_3, outputs_EXST1_3, t);
    }
    { // EXST1_4 dynamics
        inputs_EXST1_4[0] = Vterm_net[5]; // Vterm
        inputs_EXST1_4[1] = 1.2314682829419707; // Vref
        step_EXST1_4(&x[42], &dxdt[42], inputs_EXST1_4, outputs_EXST1_4, t);
    }
    { // EXST1_5 dynamics
        inputs_EXST1_5[0] = Vterm_net[7]; // Vterm
        inputs_EXST1_5[1] = 1.2442089403946075; // Vref
        step_EXST1_5(&x[46], &dxdt[46], inputs_EXST1_5, outputs_EXST1_5, t);
    }
    { // TGOV1_1 dynamics
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1177.140384165324; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1(&x[50], &dxdt[50], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // TGOV1_2 dynamics
        inputs_TGOV1_2[0] = outputs_GENROU_2[2]; // omega
        inputs_TGOV1_2[1] = 330.83038143298836; // Pref
        inputs_TGOV1_2[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_2(&x[53], &dxdt[53], inputs_TGOV1_2, outputs_TGOV1_2, t);
    }
    { // TGOV1_3 dynamics
        inputs_TGOV1_3[0] = outputs_GENROU_3[2]; // omega
        inputs_TGOV1_3[1] = 379.2881366194895; // Pref
        inputs_TGOV1_3[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_3(&x[56], &dxdt[56], inputs_TGOV1_3, outputs_TGOV1_3, t);
    }
    { // TGOV1_4 dynamics
        inputs_TGOV1_4[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_4[1] = 388.02516013150716; // Pref
        inputs_TGOV1_4[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_4(&x[59], &dxdt[59], inputs_TGOV1_4, outputs_TGOV1_4, t);
    }
    { // TGOV1_5 dynamics
        inputs_TGOV1_5[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_5[1] = 367.84989405934886; // Pref
        inputs_TGOV1_5[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_5(&x[62], &dxdt[62], inputs_TGOV1_5, outputs_TGOV1_5, t);
    }
    { // CLOAD_2 dynamics
        inputs_CLOAD_2[0] = Vd_net[1]; // Vd
        inputs_CLOAD_2[1] = Vq_net[1]; // Vq
        step_CLOAD_2(&x[65], &dxdt[65], inputs_CLOAD_2, outputs_CLOAD_2, t);
    }
    { // CLOAD_3 dynamics
        inputs_CLOAD_3[0] = Vd_net[2]; // Vd
        inputs_CLOAD_3[1] = Vq_net[2]; // Vq
        step_CLOAD_3(&x[65], &dxdt[65], inputs_CLOAD_3, outputs_CLOAD_3, t);
    }
    { // CLOAD_4 dynamics
        inputs_CLOAD_4[0] = Vd_net[3]; // Vd
        inputs_CLOAD_4[1] = Vq_net[3]; // Vq
        step_CLOAD_4(&x[65], &dxdt[65], inputs_CLOAD_4, outputs_CLOAD_4, t);
    }
    { // CLOAD_5 dynamics
        inputs_CLOAD_5[0] = Vd_net[4]; // Vd
        inputs_CLOAD_5[1] = Vq_net[4]; // Vq
        step_CLOAD_5(&x[65], &dxdt[65], inputs_CLOAD_5, outputs_CLOAD_5, t);
    }
    { // CLOAD_6 dynamics
        inputs_CLOAD_6[0] = Vd_net[5]; // Vd
        inputs_CLOAD_6[1] = Vq_net[5]; // Vq
        step_CLOAD_6(&x[65], &dxdt[65], inputs_CLOAD_6, outputs_CLOAD_6, t);
    }
    { // CLOAD_9 dynamics
        inputs_CLOAD_9[0] = Vd_net[8]; // Vd
        inputs_CLOAD_9[1] = Vq_net[8]; // Vq
        step_CLOAD_9(&x[65], &dxdt[65], inputs_CLOAD_9, outputs_CLOAD_9, t);
    }
    { // CLOAD_10 dynamics
        inputs_CLOAD_10[0] = Vd_net[9]; // Vd
        inputs_CLOAD_10[1] = Vq_net[9]; // Vq
        step_CLOAD_10(&x[65], &dxdt[65], inputs_CLOAD_10, outputs_CLOAD_10, t);
    }
    { // CLOAD_11 dynamics
        inputs_CLOAD_11[0] = Vd_net[10]; // Vd
        inputs_CLOAD_11[1] = Vq_net[10]; // Vq
        step_CLOAD_11(&x[65], &dxdt[65], inputs_CLOAD_11, outputs_CLOAD_11, t);
    }
    { // CLOAD_12 dynamics
        inputs_CLOAD_12[0] = Vd_net[11]; // Vd
        inputs_CLOAD_12[1] = Vq_net[11]; // Vq
        step_CLOAD_12(&x[65], &dxdt[65], inputs_CLOAD_12, outputs_CLOAD_12, t);
    }
    { // CLOAD_13 dynamics
        inputs_CLOAD_13[0] = Vd_net[12]; // Vd
        inputs_CLOAD_13[1] = Vq_net[12]; // Vq
        step_CLOAD_13(&x[65], &dxdt[65], inputs_CLOAD_13, outputs_CLOAD_13, t);
    }
    { // CLOAD_14 dynamics
        inputs_CLOAD_14[0] = Vd_net[13]; // Vd
        inputs_CLOAD_14[1] = Vq_net[13]; // Vq
        step_CLOAD_14(&x[65], &dxdt[65], inputs_CLOAD_14, outputs_CLOAD_14, t);
    }

    // COI Reference Frame Correction
    double coi_total_2H = 32.000000 + 8.000000 + 10.000000 + 10.000000 + 10.000000;
    double coi_omega = (32.000000 * x[1] + 8.000000 * x[7] + 10.000000 * x[13] + 10.000000 * x[19] + 10.000000 * x[25]) / coi_total_2H;
    double omega_b_sys = 2.0 * M_PI * 60.0;
    dxdt[0] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[6] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[12] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[18] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[24] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[65] = omega_b_sys * (coi_omega - 1.0);

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
// IDA residual wrapper — adapts our dae_residual to SUNDIALS signature
// =================================================================
int ida_residual(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr,
                 void* /*user_data*/) {
    const double* y_data    = N_VGetArrayPointer(yy);
    const double* ydot_data = N_VGetArrayPointer(yp);
    double*       res_data  = N_VGetArrayPointer(rr);
    dae_residual(y_data, ydot_data, res_data, (double)t);
    return 0;
}

// Helper: log one data point
static inline void log_state(std::ofstream& outfile, double t_val,
                              const double* yy_data,
                              double* Vd_net, double* Vq_net, double* Vterm_net) {
    double t = t_val;
    const double* y = yy_data;
    const double* x = y;
    for (int i = 0; i < N_BUS; ++i) {
        Vd_net[i]   = y[N_DIFF + 2*i];
        Vq_net[i]   = y[N_DIFF + 2*i + 1];
        Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
    }
    outfile << (t) << "," << (x[0]) << "," << (x[1]) << "," << (x[2]) << "," << (x[3]) << "," << (x[4]) << "," << (x[5]) << "," << (y[0] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_1[3]) << "," << (inputs_GENROU_1[2]) << "," << (outputs_GENROU_1[3]) << "," << (outputs_GENROU_1[4]) << "," << (sqrt(inputs_GENROU_1[0]*inputs_GENROU_1[0] + inputs_GENROU_1[1]*inputs_GENROU_1[1])) << "," << (y[2]) << "," << (y[1]) << "," << ((16.0*((y[1])*(y[1])) - 32.0*y[1] + 1.0526315789473684*((y[2])*(y[2])) + 20.000000000000004*((y[3])*(y[3])) + 1.0526315789473684*((y[4])*(y[4])) + 20.000000000000004*((y[5])*(y[5])) + 16.0)) << "," << (x[6]) << "," << (x[7]) << "," << (x[8]) << "," << (x[9]) << "," << (x[10]) << "," << (x[11]) << "," << (y[6] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_2[3]) << "," << (inputs_GENROU_2[2]) << "," << (outputs_GENROU_2[3]) << "," << (outputs_GENROU_2[4]) << "," << (sqrt(inputs_GENROU_2[0]*inputs_GENROU_2[0] + inputs_GENROU_2[1]*inputs_GENROU_2[1])) << "," << (y[8]) << "," << (y[7]) << "," << ((4.0*((y[7])*(y[7])) - 8.0*y[7] + 0.26315789473684209*((y[8])*(y[8])) + 5.0000000000000009*((y[9])*(y[9])) + 0.26315789473684209*((y[10])*(y[10])) + 5.0000000000000009*((y[11])*(y[11])) + 4.0)) << "," << (x[12]) << "," << (x[13]) << "," << (x[14]) << "," << (x[15]) << "," << (x[16]) << "," << (x[17]) << "," << (y[12] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_3[3]) << "," << (inputs_GENROU_3[2]) << "," << (outputs_GENROU_3[3]) << "," << (outputs_GENROU_3[4]) << "," << (sqrt(inputs_GENROU_3[0]*inputs_GENROU_3[0] + inputs_GENROU_3[1]*inputs_GENROU_3[1])) << "," << (y[14]) << "," << (y[13]) << "," << ((5.0*((y[13])*(y[13])) - 10.0*y[13] + 0.26315789473684209*((y[14])*(y[14])) + 5.0000000000000009*((y[15])*(y[15])) + 0.26315789473684209*((y[16])*(y[16])) + 5.0000000000000009*((y[17])*(y[17])) + 5.0)) << "," << (x[18]) << "," << (x[19]) << "," << (x[20]) << "," << (x[21]) << "," << (x[22]) << "," << (x[23]) << "," << (y[18] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_4[3]) << "," << (inputs_GENROU_4[2]) << "," << (outputs_GENROU_4[3]) << "," << (outputs_GENROU_4[4]) << "," << (sqrt(inputs_GENROU_4[0]*inputs_GENROU_4[0] + inputs_GENROU_4[1]*inputs_GENROU_4[1])) << "," << (y[20]) << "," << (y[19]) << "," << ((5.0*((y[19])*(y[19])) - 10.0*y[19] + 0.26315789473684209*((y[20])*(y[20])) + 5.0000000000000009*((y[21])*(y[21])) + 0.26315789473684209*((y[22])*(y[22])) + 5.0000000000000009*((y[23])*(y[23])) + 5.0)) << "," << (x[24]) << "," << (x[25]) << "," << (x[26]) << "," << (x[27]) << "," << (x[28]) << "," << (x[29]) << "," << (y[24] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_5[3]) << "," << (inputs_GENROU_5[2]) << "," << (outputs_GENROU_5[3]) << "," << (outputs_GENROU_5[4]) << "," << (sqrt(inputs_GENROU_5[0]*inputs_GENROU_5[0] + inputs_GENROU_5[1]*inputs_GENROU_5[1])) << "," << (y[26]) << "," << (y[25]) << "," << ((5.0*((y[25])*(y[25])) - 10.0*y[25] + 0.26315789473684209*((y[26])*(y[26])) + 5.0000000000000009*((y[27])*(y[27])) + 0.26315789473684209*((y[28])*(y[28])) + 5.0000000000000009*((y[29])*(y[29])) + 5.0)) << "," << (x[30]) << "," << (x[31]) << "," << (x[32]) << "," << (x[33]) << "," << (y[32]) << "," << ((0.5*(y[30]*y[30]+y[31]*y[31]+y[32]*y[32]+y[33]*y[33]))) << "," << (x[34]) << "," << (x[35]) << "," << (x[36]) << "," << (x[37]) << "," << (y[36]) << "," << ((0.5*(y[34]*y[34]+y[35]*y[35]+y[36]*y[36]+y[37]*y[37]))) << "," << (x[38]) << "," << (x[39]) << "," << (x[40]) << "," << (x[41]) << "," << (y[40]) << "," << ((0.5*(y[38]*y[38]+y[39]*y[39]+y[40]*y[40]+y[41]*y[41]))) << "," << (x[42]) << "," << (x[43]) << "," << (x[44]) << "," << (x[45]) << "," << (y[44]) << "," << ((0.5*(y[42]*y[42]+y[43]*y[43]+y[44]*y[44]+y[45]*y[45]))) << "," << (x[46]) << "," << (x[47]) << "," << (x[48]) << "," << (x[49]) << "," << (y[48]) << "," << ((0.5*(y[46]*y[46]+y[47]*y[47]+y[48]*y[48]+y[49]*y[49]))) << "," << (x[50]) << "," << (x[51]) << "," << (x[52]) << "," << (outputs_TGOV1_1[0]) << "," << (y[50]) << "," << (y[52]) << "," << (x[53]) << "," << (x[54]) << "," << (x[55]) << "," << (outputs_TGOV1_2[0]) << "," << (y[53]) << "," << (y[55]) << "," << (x[56]) << "," << (x[57]) << "," << (x[58]) << "," << (outputs_TGOV1_3[0]) << "," << (y[56]) << "," << (y[58]) << "," << (x[59]) << "," << (x[60]) << "," << (x[61]) << "," << (outputs_TGOV1_4[0]) << "," << (y[59]) << "," << (y[61]) << "," << (x[62]) << "," << (x[63]) << "," << (x[64]) << "," << (outputs_TGOV1_5[0]) << "," << (y[62]) << "," << (y[64]) << "," << (outputs_CLOAD_2[2]) << "," << (outputs_CLOAD_2[3]) << "," << (outputs_CLOAD_3[2]) << "," << (outputs_CLOAD_3[3]) << "," << (outputs_CLOAD_4[2]) << "," << (outputs_CLOAD_4[3]) << "," << (outputs_CLOAD_5[2]) << "," << (outputs_CLOAD_5[3]) << "," << (outputs_CLOAD_6[2]) << "," << (outputs_CLOAD_6[3]) << "," << (outputs_CLOAD_9[2]) << "," << (outputs_CLOAD_9[3]) << "," << (outputs_CLOAD_10[2]) << "," << (outputs_CLOAD_10[3]) << "," << (outputs_CLOAD_11[2]) << "," << (outputs_CLOAD_11[3]) << "," << (outputs_CLOAD_12[2]) << "," << (outputs_CLOAD_12[3]) << "," << (outputs_CLOAD_13[2]) << "," << (outputs_CLOAD_13[3]) << "," << (outputs_CLOAD_14[2]) << "," << (outputs_CLOAD_14[3]) << "," << (Vd_net[0]) << "," << (Vq_net[0]) << "," << (Vterm_net[0]) << "," << (Vd_net[1]) << "," << (Vq_net[1]) << "," << (Vterm_net[1]) << "," << (Vd_net[2]) << "," << (Vq_net[2]) << "," << (Vterm_net[2]) << "," << (Vd_net[3]) << "," << (Vq_net[3]) << "," << (Vterm_net[3]) << "," << (Vd_net[4]) << "," << (Vq_net[4]) << "," << (Vterm_net[4]) << "," << (Vd_net[5]) << "," << (Vq_net[5]) << "," << (Vterm_net[5]) << "," << (Vd_net[6]) << "," << (Vq_net[6]) << "," << (Vterm_net[6]) << "," << (Vd_net[7]) << "," << (Vq_net[7]) << "," << (Vterm_net[7]) << "," << (Vd_net[8]) << "," << (Vq_net[8]) << "," << (Vterm_net[8]) << "," << (Vd_net[9]) << "," << (Vq_net[9]) << "," << (Vterm_net[9]) << "," << (Vd_net[10]) << "," << (Vq_net[10]) << "," << (Vterm_net[10]) << "," << (Vd_net[11]) << "," << (Vq_net[11]) << "," << (Vterm_net[11]) << "," << (Vd_net[12]) << "," << (Vq_net[12]) << "," << (Vterm_net[12]) << "," << (Vd_net[13]) << "," << (Vq_net[13]) << "," << (Vterm_net[13]) << std::endl;
}

// =================================================================
// SUNDIALS IDA solver (variable-order BDF, adaptive step)
//   with event-driven reinitialization at fault boundaries
// =================================================================
int main() {
    // --- Context ---
    SUNContext sunctx = NULL;
    // SUN_COMM_NULL was introduced in SUNDIALS 7.0; on 6.x SUNContext_Create
    // takes void* and accepts NULL. Define it ourselves when missing so the
    // same generated code compiles against either ABI.
#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif
    int ierr = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (ierr) { std::cerr << "SUNContext_Create failed" << std::endl; return 1; }

    // --- Initial state y0 ---
    N_Vector yy = N_VNew_Serial(N_TOTAL, sunctx);
    N_Vector yp = N_VNew_Serial(N_TOTAL, sunctx);
    sunrealtype* yy_data = N_VGetArrayPointer(yy);
    sunrealtype* yp_data = N_VGetArrayPointer(yp);

    double y0_arr[N_TOTAL] = { 0.473518178721, 1.000000000000, 0.969661463533, 0.957794393347, 0.427146419365, -0.452272679327, 0.477228234970, 1.000000000000, 0.944690175570, 0.928181634848, 0.458056574620, -0.485001079009, 0.534327060192, 1.000000000000, 0.852058125399, 0.831599396743, 0.536508790083, -0.568068130676, 0.346032050182, 1.000000000000, 1.045994451040, 1.012517579835, 0.407777140224, -0.431764030826, 0.396530122657, 1.000000000000, 1.057430088179, 1.028915969642, 0.404799112402, -0.428610824896, 1.060024127523, 0.117140165669, 1.171401656691, 1.171401656691, 1.045097829436, 0.122533536785, 1.225335367849, 1.225335367849, 1.010042012186, 0.119985651254, 1.199856512545, 1.199856512545, 1.069958156790, 0.161510126152, 1.615101261524, 1.615101261524, 1.089991930064, 0.154217010330, 1.542170103302, 1.542170103302, 1.177317701867, 1.177317701867, 0.000000000000, 0.330160541975, 0.330160541975, 0.000000000000, 0.378666803423, 0.378666803423, 0.000000000000, 0.387412572704, 0.387412572704, 0.000000000000, 0.367217111171, 0.367217111171, 0.000000000000, 0.000000000000, 1.060301422710, -0.008066208887, 1.044084612827, -0.052383612576, 1.002907069624, -0.122334406164, 1.036637085784, -0.106575212637, 1.043586070054, -0.090545992662, 1.064069109355, -0.114085412228, 1.062777374050, -0.106328591508, 1.089130854150, -0.047878940986, 1.056099110873, -0.142423523236, 1.049316637494, -0.141813437585, 1.052794112702, -0.130253391233, 1.051169271601, -0.130769498647, 1.048762185210, -0.135657354527, 1.053983553862, -0.170622301414 };
    for (int i = 0; i < N_TOTAL; ++i) yy_data[i] = y0_arr[i];

    // --- Compute consistent initial yp from residual ---
    double res0[N_TOTAL];
    for (int i = 0; i < N_TOTAL; ++i) yp_data[i] = 0.0;
    dae_residual(yy_data, yp_data, res0, 0.0);
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
    for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;

    // --- Variable type ID (1 = differential, 0 = algebraic) ---
    N_Vector var_id = N_VNew_Serial(N_TOTAL, sunctx);
    sunrealtype* var_id_data = N_VGetArrayPointer(var_id);
    double var_id_arr[N_TOTAL] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < N_TOTAL; ++i) var_id_data[i] = var_id_arr[i];

    // --- Dense linear solver (shared across reinits) ---
    SUNMatrix A = SUNDenseMatrix(N_TOTAL, N_TOTAL, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(yy, A, sunctx);

    // --- Create IDA solver ---
    void* ida_mem = IDACreate(sunctx);
    if (!ida_mem) { std::cerr << "IDACreate failed" << std::endl; return 1; }

    ierr = IDAInit(ida_mem, ida_residual, 0.0, yy, yp);
    if (ierr < 0) { std::cerr << "IDAInit failed: " << ierr << std::endl; return 1; }

    // Tolerances: relax for larger systems to avoid excessive Jacobian updates
    double ida_rtol = (N_TOTAL > 30) ? 1e-4 : 1e-6;
    double ida_atol = (N_TOTAL > 30) ? 1e-6 : 1e-8;
    ierr = IDASStolerances(ida_mem, ida_rtol, ida_atol);
    if (ierr < 0) { std::cerr << "IDASStolerances failed" << std::endl; return 1; }

    ierr = IDASetLinearSolver(ida_mem, LS, A);
    if (ierr < 0) { std::cerr << "IDASetLinearSolver failed" << std::endl; return 1; }

    ierr = IDASetId(ida_mem, var_id);
    if (ierr < 0) { std::cerr << "IDASetId failed" << std::endl; return 1; }

    // Max internal step: allow IDA freedom, capped at output interval
    double ida_max_step = (N_TOTAL > 30) ? fmax(0.0005, 0.005) : 0.0005;
    IDASetMaxStep(ida_mem, ida_max_step);
    IDASetMaxNumSteps(ida_mem, 5000000);
    // Limit BDF order to 3 for large stiff systems (more stable)
    if (N_TOTAL > 30) IDASetMaxOrd(ida_mem, 3);

    // --- Correct initial conditions ---
    ierr = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, 0.005);
    if (ierr < 0) {
        std::cerr << "WARNING: IDACalcIC returned " << ierr
                  << " (proceeding anyway)" << std::endl;
    }
    IDAGetConsistentIC(ida_mem, yy, yp);

    // === IC torque-balance diagnostics (all governors with xi state) ===
    // After _sync_voltages_to_states, the KCL residual should be ~0.
    // These diagnostics verify that yp[omega] (= dω/dt at t=0) is negligible.
    // IC-diag: TGOV1_1 residual torque mismatch of GENROU_1
    std::printf("[IC-diag] TGOV1_1 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[1] * (2.0 * 16.0), yp_data[1]);
    // IC-diag: TGOV1_2 residual torque mismatch of GENROU_2
    std::printf("[IC-diag] TGOV1_2 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[7] * (2.0 * 4.0), yp_data[7]);
    // IC-diag: TGOV1_3 residual torque mismatch of GENROU_3
    std::printf("[IC-diag] TGOV1_3 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[13] * (2.0 * 5.0), yp_data[13]);
    // IC-diag: TGOV1_4 residual torque mismatch of GENROU_4
    std::printf("[IC-diag] TGOV1_4 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[19] * (2.0 * 5.0), yp_data[19]);
    // IC-diag: TGOV1_5 residual torque mismatch of GENROU_5
    std::printf("[IC-diag] TGOV1_5 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[25] * (2.0 * 5.0), yp_data[25]);

    // === IC exciter Vm diagnostic ===
    // After _sync_voltages_to_states, Vm should match Vterm_IDA.
    // These diagnostics verify the residual is negligible.
    // IC-diag: EXST1_1.Vm residual at bus 1
    {
        double Vd_ic = yy_data[N_DIFF + 0];
        double Vq_ic = yy_data[N_DIFF + 1];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_1.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[30], Vt_ic, Vt_ic - yy_data[30]);
    }
    // IC-diag: EXST1_2.Vm residual at bus 2
    {
        double Vd_ic = yy_data[N_DIFF + 2];
        double Vq_ic = yy_data[N_DIFF + 3];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_2.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[34], Vt_ic, Vt_ic - yy_data[34]);
    }
    // IC-diag: EXST1_3.Vm residual at bus 3
    {
        double Vd_ic = yy_data[N_DIFF + 4];
        double Vq_ic = yy_data[N_DIFF + 5];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_3.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[38], Vt_ic, Vt_ic - yy_data[38]);
    }
    // IC-diag: EXST1_4.Vm residual at bus 6
    {
        double Vd_ic = yy_data[N_DIFF + 10];
        double Vq_ic = yy_data[N_DIFF + 11];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_4.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[42], Vt_ic, Vt_ic - yy_data[42]);
    }
    // IC-diag: EXST1_5.Vm residual at bus 8
    {
        double Vd_ic = yy_data[N_DIFF + 14];
        double Vq_ic = yy_data[N_DIFF + 15];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_5.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[46], Vt_ic, Vt_ic - yy_data[46]);
    }

    // Re-sync yp from the (unmodified) state, then tell IDA about the IC.
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = 0.0;
    dae_residual(yy_data, yp_data, res0, 0.0);
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
    for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;
    IDAReInit(ida_mem, 0.0, yy, yp);

    // --- Output setup ---
    std::ofstream outfile("simulation_results.csv");
    outfile << "t,GENROU_1.delta,GENROU_1.omega,GENROU_1.E_q_prime,GENROU_1.psi_d,GENROU_1.E_d_prime,GENROU_1.psi_q,GENROU_1.delta_deg,GENROU_1.Te,GENROU_1.Tm_in,GENROU_1.Pe,GENROU_1.Qe,GENROU_1.V_term,GENROU_1.Eq_p,GENROU_1.omega,GENROU_1.H_total,GENROU_2.delta,GENROU_2.omega,GENROU_2.E_q_prime,GENROU_2.psi_d,GENROU_2.E_d_prime,GENROU_2.psi_q,GENROU_2.delta_deg,GENROU_2.Te,GENROU_2.Tm_in,GENROU_2.Pe,GENROU_2.Qe,GENROU_2.V_term,GENROU_2.Eq_p,GENROU_2.omega,GENROU_2.H_total,GENROU_3.delta,GENROU_3.omega,GENROU_3.E_q_prime,GENROU_3.psi_d,GENROU_3.E_d_prime,GENROU_3.psi_q,GENROU_3.delta_deg,GENROU_3.Te,GENROU_3.Tm_in,GENROU_3.Pe,GENROU_3.Qe,GENROU_3.V_term,GENROU_3.Eq_p,GENROU_3.omega,GENROU_3.H_total,GENROU_4.delta,GENROU_4.omega,GENROU_4.E_q_prime,GENROU_4.psi_d,GENROU_4.E_d_prime,GENROU_4.psi_q,GENROU_4.delta_deg,GENROU_4.Te,GENROU_4.Tm_in,GENROU_4.Pe,GENROU_4.Qe,GENROU_4.V_term,GENROU_4.Eq_p,GENROU_4.omega,GENROU_4.H_total,GENROU_5.delta,GENROU_5.omega,GENROU_5.E_q_prime,GENROU_5.psi_d,GENROU_5.E_d_prime,GENROU_5.psi_q,GENROU_5.delta_deg,GENROU_5.Te,GENROU_5.Tm_in,GENROU_5.Pe,GENROU_5.Qe,GENROU_5.V_term,GENROU_5.Eq_p,GENROU_5.omega,GENROU_5.H_total,EXST1_1.Vm,EXST1_1.LLx,EXST1_1.Vr,EXST1_1.Vf,EXST1_1.Efd,EXST1_1.H_exc,EXST1_2.Vm,EXST1_2.LLx,EXST1_2.Vr,EXST1_2.Vf,EXST1_2.Efd,EXST1_2.H_exc,EXST1_3.Vm,EXST1_3.LLx,EXST1_3.Vr,EXST1_3.Vf,EXST1_3.Efd,EXST1_3.H_exc,EXST1_4.Vm,EXST1_4.LLx,EXST1_4.Vr,EXST1_4.Vf,EXST1_4.Efd,EXST1_4.H_exc,EXST1_5.Vm,EXST1_5.LLx,EXST1_5.Vr,EXST1_5.Vf,EXST1_5.Efd,EXST1_5.H_exc,TGOV1_1.x1,TGOV1_1.x2,TGOV1_1.xi,TGOV1_1.Tm,TGOV1_1.Valve,TGOV1_1.xi,TGOV1_2.x1,TGOV1_2.x2,TGOV1_2.xi,TGOV1_2.Tm,TGOV1_2.Valve,TGOV1_2.xi,TGOV1_3.x1,TGOV1_3.x2,TGOV1_3.xi,TGOV1_3.Tm,TGOV1_3.Valve,TGOV1_3.xi,TGOV1_4.x1,TGOV1_4.x2,TGOV1_4.xi,TGOV1_4.Tm,TGOV1_4.Valve,TGOV1_4.xi,TGOV1_5.x1,TGOV1_5.x2,TGOV1_5.xi,TGOV1_5.Tm,TGOV1_5.Valve,TGOV1_5.xi,CLOAD_2.Pload,CLOAD_2.Qload,CLOAD_3.Pload,CLOAD_3.Qload,CLOAD_4.Pload,CLOAD_4.Qload,CLOAD_5.Pload,CLOAD_5.Qload,CLOAD_6.Pload,CLOAD_6.Qload,CLOAD_9.Pload,CLOAD_9.Qload,CLOAD_10.Pload,CLOAD_10.Qload,CLOAD_11.Pload,CLOAD_11.Qload,CLOAD_12.Pload,CLOAD_12.Qload,CLOAD_13.Pload,CLOAD_13.Qload,CLOAD_14.Pload,CLOAD_14.Qload,Vd_Bus1,Vq_Bus1,Vterm_Bus1,Vd_Bus2,Vq_Bus2,Vterm_Bus2,Vd_Bus3,Vq_Bus3,Vterm_Bus3,Vd_Bus4,Vq_Bus4,Vterm_Bus4,Vd_Bus5,Vq_Bus5,Vterm_Bus5,Vd_Bus6,Vq_Bus6,Vterm_Bus6,Vd_Bus7,Vq_Bus7,Vterm_Bus7,Vd_Bus8,Vq_Bus8,Vterm_Bus8,Vd_Bus9,Vq_Bus9,Vterm_Bus9,Vd_Bus10,Vq_Bus10,Vterm_Bus10,Vd_Bus11,Vq_Bus11,Vterm_Bus11,Vd_Bus12,Vq_Bus12,Vterm_Bus12,Vd_Bus13,Vq_Bus13,Vterm_Bus13,Vd_Bus14,Vq_Bus14,Vterm_Bus14" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Print header
    std::cout << "Dirac DAE Simulation (SUNDIALS IDA)" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  max_dt = 0.0005,  T = 15.0 s" << std::endl;
    std::cout << "  Solver: SUNDIALS IDA (variable-order BDF, adaptive step)"
              << std::endl;
    std::cout << "  rtol = " << ida_rtol << ",  atol = " << ida_atol
              << ",  max_step = " << ida_max_step << std::endl;
    std::cout << std::endl;

    // Initial diagnostics
    dae_residual(yy_data, yp_data, res0, 0.0);
    double max_res = 0.0;
    int max_res_idx = -1;
    for (int i = 0; i < N_TOTAL; ++i) {
        if (fabs(res0[i]) > max_res) { max_res = fabs(res0[i]); max_res_idx = i; }
    }
    std::cout << "[IDA] Initial max |residual| = " << max_res
              << " at index " << max_res_idx << std::endl;

    // Log initial state
    log_state(outfile, 0.0, yy_data, Vd_net, Vq_net, Vterm_net);
    long n_logged = 1;

    // --- Event-driven integration segments ---
    // Fault boundaries create discontinuities; IDA must be restarted
    // at each boundary to avoid step-size collapse.
    const double seg_ends[3] = { 2.000000000000, 2.050000000000, 15.000000000000 };
    const int n_segments = 3;
    double t_current = 0.0;
    double t_next_log = 0.01;
    int last_sec = -1;
    bool aborted = false;

    for (int seg = 0; seg < n_segments && !aborted; ++seg) {
        double t_seg_end = seg_ends[seg];

        if (t_seg_end <= t_current + 1e-12) continue;

        // Update fault flags for this segment (must happen BEFORE residual
        // evaluations so IDACalcIC / IDAReInit see the correct topology)
        for (int f = 0; f < N_FAULTS; ++f) {
            fault_active[f] = (t_current >= FAULT_T_START[f]
                               && t_current < FAULT_T_END[f]) ? 1 : 0;
        }

        // Reinitialize IDA at segment boundary (except first segment)
        if (seg > 0) {
            // Recompute yp for the new regime (fault topology changed)
            for (int i = 0; i < N_TOTAL; ++i) yp_data[i] = 0.0;
            dae_residual(yy_data, yp_data, res0, t_current);
            for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
            for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;

            ierr = IDAReInit(ida_mem, t_current, yy, yp);
            if (ierr < 0) {
                std::cerr << "IDAReInit failed at t=" << t_current
                          << " flag=" << ierr << std::endl;
                break;
            }

            // Fix algebraic variables for new regime
            ierr = IDACalcIC(ida_mem, IDA_YA_YDP_INIT,
                             t_current + 0.005);
            if (ierr < 0) {
                std::cerr << "WARNING: IDACalcIC at t=" << t_current
                          << " returned " << ierr << std::endl;
            }
            IDAGetConsistentIC(ida_mem, yy, yp);

            std::cout << "[IDA] Reinit at t=" << t_current
                      << " (segment " << seg << "/" << n_segments << ")"
                      << std::endl;
        }

        // Integrate within this segment
        sunrealtype t_ret = t_current;
        while (t_ret < t_seg_end - 1e-12) {
            double t_out = t_ret + 0.0005;
            if (t_out > t_seg_end) t_out = t_seg_end;

            ierr = IDASolve(ida_mem, t_out, &t_ret, yy, yp, IDA_NORMAL);
            if (ierr < 0) {
                std::cerr << "IDASolve failed at t=" << t_ret
                          << " flag=" << ierr << std::endl;
                aborted = true;
                break;
            }

            // Log at regular intervals
            if (t_ret >= t_next_log - 1e-12 || t_ret >= 15.0 - 1e-12) {
                log_state(outfile, t_ret, yy_data, Vd_net, Vq_net, Vterm_net);
                n_logged++;
                while (t_next_log <= t_ret + 1e-12)
                    t_next_log += 0.01;
            }

            // Progress check every ~1 second
            int sec = (int)t_ret;
            if (sec > last_sec) {
                last_sec = sec;
                for (int i = 0; i < N_BUS; ++i) {
                    Vd_net[i]   = yy_data[N_DIFF + 2*i];
                    Vq_net[i]   = yy_data[N_DIFF + 2*i + 1];
                    Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i]
                                        + Vq_net[i]*Vq_net[i]);
                }
                std::cout << "t=" << t_ret
                          << " Vterm[0]=" << Vterm_net[0] << std::endl;

                if (Vterm_net[0] > 5.0 || std::isnan(Vterm_net[0])) {
                    std::cout << "Stability limit reached. Stopping." << std::endl;
                    goto ida_done;
                }
            }
        }

        t_current = t_seg_end;
    }

ida_done:
    outfile.close();

    // --- Statistics ---
    long nst, nre, nje, nni, ncfn;
    IDAGetNumSteps(ida_mem, &nst);
    IDAGetNumResEvals(ida_mem, &nre);
    IDAGetNumJacEvals(ida_mem, &nje);
    IDAGetNumNonlinSolvIters(ida_mem, &nni);
    IDAGetNumNonlinSolvConvFails(ida_mem, &ncfn);
    std::cout << std::endl;
    std::cout << "[IDA] Steps: " << nst << "  Residual evals: " << nre
              << "  Jacobian evals: " << nje << std::endl;
    std::cout << "[IDA] Newton iters: " << nni
              << "  Conv fails: " << ncfn << std::endl;
    std::cout << "[IDA] Logged " << n_logged << " data points" << std::endl;
    std::cout << "Done. Results in simulation_results.csv" << std::endl;

    // --- Cleanup ---
    IDAFree(&ida_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(var_id);
    SUNContext_Free(&sunctx);

    return 0;
}
