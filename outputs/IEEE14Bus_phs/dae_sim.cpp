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

const int N_DIFF  = 81;
const int N_BUS   = 14;
const int N_ALG   = 28;
const int N_TOTAL = 109;

const double LOAD_G[14] = { 0.0000000000e+00, 1.9871339942e-01, 4.9014802470e-01, 4.4715858923e-01, 7.0068133717e-02, 1.3101580924e-01, 0.0000000000e+00, 0.0000000000e+00, 2.6121291579e-01, 8.0613588047e-02, 3.1161544110e-02, 5.4472939656e-02, 1.2114170494e-01, 1.7819113881e-01 };
const double LOAD_B[14] = { 0.0000000000e+00, -1.1629770381e-01, -2.4507401235e-01, -9.3547822014e-02, -1.4751186046e-02, -6.5507904620e-02, 0.0000000000e+00, 0.0000000000e+00, -1.4698760685e-01, -5.1950978964e-02, -1.6025936971e-02, -1.4287984172e-02, -5.2046065824e-02, -6.2366898583e-02 };
const double LOAD_KPF[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };
const double LOAD_KQF[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };

// Full Y-bus (14x14) — NO Kron reduction
const double Y_real[196] = {
    6.0250290558e+00, -4.9991316008e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -4.9991316008e+00, 9.7200370102e+00, -1.1350191923e+00, -1.6860331506e+00, -1.7011396671e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1350191923e+00, 3.6111429269e+00, -1.9859757099e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.6860331506e+00, -1.9859757099e+00, 1.0960148111e+01, -6.8409806615e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.0258974550e+00, -1.7011396671e+00, 0.0000000000e+00, -6.8409806615e+00, 9.6380859173e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.7109392167e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, -1.5259674405e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.5872679553e+00, -3.9020495524e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.9020495524e+00, 5.8635478942e+00, -1.8808847537e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9550285632e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.8808847537e+00, 3.8670748610e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.5259674405e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0694649669e+00, -2.4890245868e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -3.0989274038e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -2.4890245868e+00, 6.8460878534e+00, -1.1369941578e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.4240054870e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -1.1369941578e+00, 2.7391907836e+00
};
const double Y_imag[196] = {
    -2.3794896292e+01, 1.5263086523e+01, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.5263086523e+01, -3.3959841674e+01, 4.7818631518e+00, 5.1158383259e+00, 5.1939273980e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7818631518e+00, -1.3008630612e+01, 5.0688169776e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.1158383259e+00, 5.0688169776e+00, -3.8455992693e+01, 2.1578553982e+01, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.2349836823e+00, 5.1939273980e+00, 0.0000000000e+00, 2.1578553982e+01, -3.4948255300e+01, 3.9807970269e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.9807970269e+00, -2.1003426901e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 3.1759639650e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.7974391101e+00, 0.0000000000e+00, 0.0000000000e+00, -1.9549005948e+01, 5.6953759109e+00, 9.0900827198e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 5.6953759109e+00, -8.6550080575e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.8038053628e+00, 0.0000000000e+00, 0.0000000000e+00, 9.0900827198e+00, 0.0000000000e+00, -2.4239493982e+01, 1.0365394127e+01, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0365394127e+01, -1.4820288855e+01, 4.4029437495e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.0940743442e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 4.4029437495e+00, -8.5130440307e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.1759639650e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, -5.4422265754e+00, 2.2519746262e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 6.1027554482e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.2519746262e+00, -1.0721739615e+01, 2.3149634751e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 3.0290504569e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 2.3149634751e+00, -5.2563808306e+00
};

// Slack bus configuration
const int IS_SLACK[14] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const double Vd_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };
const double Vq_slack_ref[14] = { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 };

// Bus fault events
const int N_FAULTS = 0;
const int FAULT_BUS[1] = {0};
const double FAULT_T_START[1] = {0.0};
const double FAULT_T_END[1] = {0.0};
const double FAULT_G[1] = {0.0};
const double FAULT_B[1] = {0.0};
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
double outputs_ESST3A_2[1];
double inputs_ESST3A_2[6];
double outputs_ESST3A_3[1];
double inputs_ESST3A_3[6];
double outputs_ESST3A_4[1];
double inputs_ESST3A_4[6];
double outputs_ESST3A_5[1];
double inputs_ESST3A_5[6];
double outputs_EXST1_1[1];
double inputs_EXST1_1[2];
double outputs_TGOV1_1[1];
double inputs_TGOV1_1[3];
double outputs_TGOV1_2[1];
double inputs_TGOV1_2[3];
double outputs_TGOV1_3[1];
double inputs_TGOV1_3[3];
double outputs_IEEEG1_4[1];
double inputs_IEEEG1_4[3];
double outputs_IEEEG1_5[1];
double inputs_IEEEG1_5[3];
double outputs_ST2CUT_3[1];
double inputs_ST2CUT_3[3];
double outputs_IEEEST_1[1];
double inputs_IEEEST_1[3];

void step_GENROU_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.23;
    const double xq_double_prime = 0.23;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
            double Te = vd * id + vq * iq;

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
            // dω/dt = (Tm − Te − D(ω−1)) / (2H)
            //       = −J[1,0]·∂H/∂δ − R[1,1]·∂H/∂ω + g_Tm·Tm − Te/(2H)
            //
            // Since ∂H/∂δ = 0, the J[1,0] term vanishes.
            // R[1,1] = D/(2H) → −R[1,1]·∂H/∂ω = −D(ω−1) ✓
            // g_Tm = 1/(2H)
            // Te term: Te/(2H) is the network port interaction
            // Standard swing equation: dω/dt = (Tm − Te − D(ω−1)) / (2H)
            // PHS: −R[1,1]·∂H/∂ω where R[1,1] = D/(4H²), ∂H/∂ω = 2H(ω−1)
            //   gives −D(ω−1)/(2H) ✓
            dxdt[1] = -J01 * dH_ddelta          // = 0 (skew-sym completion)
                      - (D / (4.0*H*H)) * dH_domega   // damping: −D(ω−1)/(2H)
                      + Tm / (2.0*H)               // mechanical input port
                      - Te / (2.0*H);              // electrical port (from stator)

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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.23;
    const double xq_double_prime = 0.23;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
void step_GENROU_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.28;
    const double xq_double_prime = 0.28;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
    const double H = 6.5;
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
            double Te = vd * id + vq * iq;

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
            // dω/dt = (Tm − Te − D(ω−1)) / (2H)
            //       = −J[1,0]·∂H/∂δ − R[1,1]·∂H/∂ω + g_Tm·Tm − Te/(2H)
            //
            // Since ∂H/∂δ = 0, the J[1,0] term vanishes.
            // R[1,1] = D/(2H) → −R[1,1]·∂H/∂ω = −D(ω−1) ✓
            // g_Tm = 1/(2H)
            // Te term: Te/(2H) is the network port interaction
            // Standard swing equation: dω/dt = (Tm − Te − D(ω−1)) / (2H)
            // PHS: −R[1,1]·∂H/∂ω where R[1,1] = D/(4H²), ∂H/∂ω = 2H(ω−1)
            //   gives −D(ω−1)/(2H) ✓
            dxdt[1] = -J01 * dH_ddelta          // = 0 (skew-sym completion)
                      - (D / (4.0*H*H)) * dH_domega   // damping: −D(ω−1)/(2H)
                      + Tm / (2.0*H)               // mechanical input port
                      - Te / (2.0*H);              // electrical port (from stator)

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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.28;
    const double xq_double_prime = 0.28;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
    const double H = 6.5;
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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.34;
    const double xq_double_prime = 0.34;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
            double Te = vd * id + vq * iq;

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
            // dω/dt = (Tm − Te − D(ω−1)) / (2H)
            //       = −J[1,0]·∂H/∂δ − R[1,1]·∂H/∂ω + g_Tm·Tm − Te/(2H)
            //
            // Since ∂H/∂δ = 0, the J[1,0] term vanishes.
            // R[1,1] = D/(2H) → −R[1,1]·∂H/∂ω = −D(ω−1) ✓
            // g_Tm = 1/(2H)
            // Te term: Te/(2H) is the network port interaction
            // Standard swing equation: dω/dt = (Tm − Te − D(ω−1)) / (2H)
            // PHS: −R[1,1]·∂H/∂ω where R[1,1] = D/(4H²), ∂H/∂ω = 2H(ω−1)
            //   gives −D(ω−1)/(2H) ✓
            dxdt[1] = -J01 * dH_ddelta          // = 0 (skew-sym completion)
                      - (D / (4.0*H*H)) * dH_domega   // damping: −D(ω−1)/(2H)
                      + Tm / (2.0*H)               // mechanical input port
                      - Te / (2.0*H);              // electrical port (from stator)

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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.34;
    const double xq_double_prime = 0.34;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.28;
    const double xq_double_prime = 0.28;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
            double Te = vd * id + vq * iq;

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
            // dω/dt = (Tm − Te − D(ω−1)) / (2H)
            //       = −J[1,0]·∂H/∂δ − R[1,1]·∂H/∂ω + g_Tm·Tm − Te/(2H)
            //
            // Since ∂H/∂δ = 0, the J[1,0] term vanishes.
            // R[1,1] = D/(2H) → −R[1,1]·∂H/∂ω = −D(ω−1) ✓
            // g_Tm = 1/(2H)
            // Te term: Te/(2H) is the network port interaction
            // Standard swing equation: dω/dt = (Tm − Te − D(ω−1)) / (2H)
            // PHS: −R[1,1]·∂H/∂ω where R[1,1] = D/(4H²), ∂H/∂ω = 2H(ω−1)
            //   gives −D(ω−1)/(2H) ✓
            dxdt[1] = -J01 * dH_ddelta          // = 0 (skew-sym completion)
                      - (D / (4.0*H*H)) * dH_domega   // damping: −D(ω−1)/(2H)
                      + Tm / (2.0*H)               // mechanical input port
                      - Te / (2.0*H);              // electrical port (from stator)

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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.28;
    const double xq_double_prime = 0.28;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.34;
    const double xq_double_prime = 0.34;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
            double Te = vd * id + vq * iq;

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
            // dω/dt = (Tm − Te − D(ω−1)) / (2H)
            //       = −J[1,0]·∂H/∂δ − R[1,1]·∂H/∂ω + g_Tm·Tm − Te/(2H)
            //
            // Since ∂H/∂δ = 0, the J[1,0] term vanishes.
            // R[1,1] = D/(2H) → −R[1,1]·∂H/∂ω = −D(ω−1) ✓
            // g_Tm = 1/(2H)
            // Te term: Te/(2H) is the network port interaction
            // Standard swing equation: dω/dt = (Tm − Te − D(ω−1)) / (2H)
            // PHS: −R[1,1]·∂H/∂ω where R[1,1] = D/(4H²), ∂H/∂ω = 2H(ω−1)
            //   gives −D(ω−1)/(2H) ✓
            dxdt[1] = -J01 * dH_ddelta          // = 0 (skew-sym completion)
                      - (D / (4.0*H*H)) * dH_domega   // damping: −D(ω−1)/(2H)
                      + Tm / (2.0*H)               // mechanical input port
                      - Te / (2.0*H);              // electrical port (from stator)

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
    const double D = 2.0;
    const double ra = 0.0;
    const double xl = 0.15;
    const double xd = 1.8;
    const double xq = 1.75;
    const double xd_prime = 0.6;
    const double xq_prime = 0.8;
    const double xd_double_prime = 0.34;
    const double xq_double_prime = 0.34;
    const double Td0_prime = 6.5;
    const double Tq0_prime = 0.2;
    const double Td0_double_prime = 0.06;
    const double Tq0_double_prime = 0.05;
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
void step_ESST3A_2(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.5;
    const double VIMIN = -0.5;
    const double KM = 5.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 10.0;
    const double TA = 0.02;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KG = 1.0;
    const double KP = 3.0;
    const double KI = 0.3;
    const double VBMAX = 4.5;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.0;
    const double THETAP = 3.33;
    const double TM = 0.5;
    const double VMMAX = 8.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            // ============================================================
            // ESST3A Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → natural dissipation via R.
            // The signal path (Vm→Verr→LL→Vr→VG→VM→VB→Efd) provides
            // the interconnection structure.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double VM  = x[3];
            double VB  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];
            double iq    = inputs[3];
            double Vd    = inputs[4];
            double Vq    = inputs[5];

            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;

            // 1. Voltage transducer: dVm/dt = (Vterm − Vm) / TR
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA/TA) with anti-windup
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN_fex = KC * XadIfd / VE;
            double FEX;
            if      (IN_fex <= 0.0)    FEX = 1.0;
            else if (IN_fex <= 0.433)  FEX = 1.0 - 0.577 * IN_fex;
            else if (IN_fex <= 0.75)   FEX = sqrt(0.75 - IN_fex * IN_fex);
            else if (IN_fex <= 1.0)    FEX = 1.732 * (1.0 - IN_fex);
            else                       FEX = 0.0;

            // 7. Rectifier voltage VB — fast lag
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;

            // 8. Field voltage feedback
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM/TM) with anti-windup
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        
}
void step_ESST3A_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.5;
    const double VIMIN = -0.5;
    const double KM = 5.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 10.0;
    const double TA = 0.02;
    const double VRMAX = 10.0;
    const double VRMIN = -10.0;
    const double KG = 1.0;
    const double KP = 3.0;
    const double KI = 0.3;
    const double VBMAX = 4.5;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.0;
    const double THETAP = 3.33;
    const double TM = 0.5;
    const double VMMAX = 8.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            double VB_o = x[4];
            double VM_o = x[3];
            double Efd_o = VB_o * VM_o;
            if (Efd_o > Efd_max) Efd_o = Efd_max;
            if (Efd_o < Efd_min) Efd_o = Efd_min;
            outputs[0] = Efd_o;
        
}
void step_ESST3A_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            // ============================================================
            // ESST3A Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → natural dissipation via R.
            // The signal path (Vm→Verr→LL→Vr→VG→VM→VB→Efd) provides
            // the interconnection structure.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double VM  = x[3];
            double VB  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];
            double iq    = inputs[3];
            double Vd    = inputs[4];
            double Vq    = inputs[5];

            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;

            // 1. Voltage transducer: dVm/dt = (Vterm − Vm) / TR
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA/TA) with anti-windup
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN_fex = KC * XadIfd / VE;
            double FEX;
            if      (IN_fex <= 0.0)    FEX = 1.0;
            else if (IN_fex <= 0.433)  FEX = 1.0 - 0.577 * IN_fex;
            else if (IN_fex <= 0.75)   FEX = sqrt(0.75 - IN_fex * IN_fex);
            else if (IN_fex <= 1.0)    FEX = 1.732 * (1.0 - IN_fex);
            else                       FEX = 0.0;

            // 7. Rectifier voltage VB — fast lag
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;

            // 8. Field voltage feedback
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM/TM) with anti-windup
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        
}
void step_ESST3A_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            double VB_o = x[4];
            double VM_o = x[3];
            double Efd_o = VB_o * VM_o;
            if (Efd_o > Efd_max) Efd_o = Efd_max;
            if (Efd_o < Efd_min) Efd_o = Efd_min;
            outputs[0] = Efd_o;
        
}
void step_ESST3A_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            // ============================================================
            // ESST3A Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → natural dissipation via R.
            // The signal path (Vm→Verr→LL→Vr→VG→VM→VB→Efd) provides
            // the interconnection structure.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double VM  = x[3];
            double VB  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];
            double iq    = inputs[3];
            double Vd    = inputs[4];
            double Vq    = inputs[5];

            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;

            // 1. Voltage transducer: dVm/dt = (Vterm − Vm) / TR
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA/TA) with anti-windup
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN_fex = KC * XadIfd / VE;
            double FEX;
            if      (IN_fex <= 0.0)    FEX = 1.0;
            else if (IN_fex <= 0.433)  FEX = 1.0 - 0.577 * IN_fex;
            else if (IN_fex <= 0.75)   FEX = sqrt(0.75 - IN_fex * IN_fex);
            else if (IN_fex <= 1.0)    FEX = 1.732 * (1.0 - IN_fex);
            else                       FEX = 0.0;

            // 7. Rectifier voltage VB — fast lag
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;

            // 8. Field voltage feedback
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM/TM) with anti-windup
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        
}
void step_ESST3A_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            double VB_o = x[4];
            double VM_o = x[3];
            double Efd_o = VB_o * VM_o;
            if (Efd_o > Efd_max) Efd_o = Efd_max;
            if (Efd_o < Efd_min) Efd_o = Efd_min;
            outputs[0] = Efd_o;
        
}
void step_ESST3A_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            // ============================================================
            // ESST3A Port-Hamiltonian Dynamics
            //
            // Storage: H = ½||x||²  →  ∇H = x
            //
            // Each state is a first-order lag → natural dissipation via R.
            // The signal path (Vm→Verr→LL→Vr→VG→VM→VB→Efd) provides
            // the interconnection structure.
            // ============================================================

            double Vm  = x[0];
            double LLx = x[1];
            double Vr  = x[2];
            double VM  = x[3];
            double VB  = x[4];

            double Vterm = inputs[0];
            double Vref  = inputs[1];
            double id    = inputs[2];
            double iq    = inputs[3];
            double Vd    = inputs[4];
            double Vq    = inputs[5];

            double TR_eff = (TR > 1e-4) ? TR : 1e-4;
            double TA_eff = (TA > 5e-3) ? TA : 5e-3;
            double TM_eff = (TM > 1e-4) ? TM : 1e-4;

            // 1. Voltage transducer: dVm/dt = (Vterm − Vm) / TR
            dxdt[0] = (Vterm - Vm) / TR_eff;

            // 2. Voltage error + input limiter
            double vi = Vref - Vm;
            if (vi > VIMAX) vi = VIMAX;
            if (vi < VIMIN) vi = VIMIN;

            // 3. Lead-lag compensator (TC/TB)
            double LL_y;
            if (TB > 1e-4) {
                LL_y = (TC / TB) * (vi - LLx) + LLx;
                dxdt[1] = (vi - LLx) / TB;
            } else {
                LL_y = vi;
                dxdt[1] = 0.0;
            }

            // 4. Voltage regulator (KA/TA) with anti-windup
            double Vr_ref = KA * LL_y;
            if (Vr_ref > VRMAX) Vr_ref = VRMAX;
            if (Vr_ref < VRMIN) Vr_ref = VRMIN;
            double dVr = (Vr_ref - Vr) / TA_eff;
            const double dVr_max = 2e3;
            if (dVr >  dVr_max) dVr =  dVr_max;
            if (dVr < -dVr_max) dVr = -dVr_max;
            if (Vr >= VRMAX && dVr > 0.0) dVr = 0.0;
            if (Vr <= VRMIN && dVr < 0.0) dVr = 0.0;
            dxdt[2] = dVr;

            // 5. VE computation (IEEE ESST3A load compensator)
            double theta_rad = THETAP * 3.14159265358979323846 / 180.0;
            double KPC_r = KP * cos(theta_rad);
            double KPC_i = KP * sin(theta_rad);
            double jc_r = -(KPC_i * XL);
            double jc_i =   KI + KPC_r * XL;
            double v_r = KPC_r * Vd - KPC_i * Vq + jc_r * id - jc_i * iq;
            double v_i = KPC_r * Vq + KPC_i * Vd + jc_r * iq + jc_i * id;
            double VE = sqrt(v_r * v_r + v_i * v_i);
            if (VE < 1e-6) VE = 1e-6;

            // 6. FEX rectifier loading function
            double XadIfd = VB * VM;
            if (XadIfd < 0.0) XadIfd = 0.0;
            double IN_fex = KC * XadIfd / VE;
            double FEX;
            if      (IN_fex <= 0.0)    FEX = 1.0;
            else if (IN_fex <= 0.433)  FEX = 1.0 - 0.577 * IN_fex;
            else if (IN_fex <= 0.75)   FEX = sqrt(0.75 - IN_fex * IN_fex);
            else if (IN_fex <= 1.0)    FEX = 1.732 * (1.0 - IN_fex);
            else                       FEX = 0.0;

            // 7. Rectifier voltage VB — fast lag
            double VB_ceil = VE * FEX;
            if (VB_ceil > VBMAX) VB_ceil = VBMAX;
            if (VB_ceil < 0.0)   VB_ceil = 0.0;
            double dVB = (VB_ceil - VB) / 0.01;
            if (VB >= VBMAX && dVB > 0.0) dVB = 0.0;
            if (VB <= 0.0   && dVB < 0.0) dVB = 0.0;
            dxdt[4] = dVB;

            // 8. Field voltage feedback
            double Efd_c = VB * VM;
            if (Efd_c > Efd_max) Efd_c = Efd_max;
            if (Efd_c < Efd_min) Efd_c = Efd_min;
            double VG = KG * Efd_c;
            if (VG > VGMAX) VG = VGMAX;
            if (VG < 0.0)   VG = 0.0;

            // 9. Inner field voltage regulator (KM/TM) with anti-windup
            double vrs = Vr - VG;
            double VM_ref = KM * vrs;
            double VMMAX_eff = (VMMAX > 90.0) ? 10.0 : VMMAX;
            if (VM_ref > VMMAX_eff) VM_ref = VMMAX_eff;
            if (VM_ref < VMMIN)     VM_ref = VMMIN;
            double dVM = (VM_ref - VM) / TM_eff;
            if (VM >= VMMAX_eff && dVM > 0.0) dVM = 0.0;
            if (VM <= VMMIN     && dVM < 0.0) dVM = 0.0;
            dxdt[3] = dVM;
        
}
void step_ESST3A_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.2;
    const double VIMIN = -0.2;
    const double KM = 8.0;
    const double TC = 1.0;
    const double TB = 5.0;
    const double KA = 20.0;
    const double TA = 0.0;
    const double VRMAX = 99.0;
    const double VRMIN = -99.0;
    const double KG = 1.0;
    const double KP = 3.67;
    const double KI = 0.435;
    const double VBMAX = 5.48;
    const double KC = 0.01;
    const double XL = 0.0098;
    const double VGMAX = 3.86;
    const double THETAP = 3.33;
    const double TM = 0.4;
    const double VMMAX = 99.0;
    const double VMMIN = 0.0;
    const double Efd_max = 5.0;
    const double Efd_min = 0.0;
    // --- Kernel ---

            double VB_o = x[4];
            double VM_o = x[3];
            double Efd_o = VB_o * VM_o;
            if (Efd_o > Efd_max) Efd_o = Efd_max;
            if (Efd_o < Efd_min) Efd_o = Efd_min;
            outputs[0] = Efd_o;
        
}
void step_EXST1_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double TR = 0.02;
    const double VIMAX = 0.5;
    const double VIMIN = -0.5;
    const double TC = 0.0;
    const double TB = 0.02;
    const double KA = 10.0;
    const double TA = 0.05;
    const double VRMAX = 5.0;
    const double VRMIN = -5.0;
    const double KC = 0.2;
    const double KF = 0.1;
    const double TF = 1.0;
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
    const double TR = 0.02;
    const double VIMAX = 0.5;
    const double VIMIN = -0.5;
    const double TC = 0.0;
    const double TB = 0.02;
    const double KA = 10.0;
    const double TA = 0.05;
    const double VRMAX = 5.0;
    const double VRMIN = -5.0;
    const double KC = 0.2;
    const double KF = 0.1;
    const double TF = 1.0;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 5.0;
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
dxdt[1] = x1/T3_eff - x2/T3_eff;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 5.0;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
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
dxdt[1] = x1/T3_eff - x2/T3_eff;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_2_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
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
dxdt[1] = x1/T3_eff - x2/T3_eff;
// dxi/dt
double _dx2_raw = -Ki*omega + Ki*wref0;
if (xi >= xi_max && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (xi <= xi_min && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
}
void step_TGOV1_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double wref0 = 1.0;
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 1.0;
    const double T3 = 2.1;
    const double Dt = 0.0;
    const double Ki = 0.0;
    const double xi_max = 1.0;
    const double xi_min = -1.0;
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
void step_IEEEG1_4(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double K = 20.0;
    const double T1 = 0.1;
    const double T2 = 0.0;
    const double T3 = 0.2;
    const double UO = 1.0;
    const double UC = -1.0;
    const double PMAX = 0.95;
    const double PMIN = 0.0;
    const double T4 = 0.1;
    const double K1 = 0.0;
    const double K2 = 0.0;
    const double T5 = 0.0;
    const double K3 = 0.0;
    const double K4 = 0.0;
    const double T6 = 0.0;
    const double K5 = 0.3;
    const double K6 = 0.0;
    const double T7 = 8.72;
    const double K7 = 0.7;
    const double K8 = 0.0;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'IEEEG1_PHS'

double x1 = x[0];
double x2 = x[1];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
dxdt[0] = K*Pref/T1_eff - K*omega/T1_eff + K*u_agc/T1_eff - x1/T1_eff;
// dx2/dt
dxdt[1] = x1/T3_eff - x2/T3_eff;
}
void step_IEEEG1_4_out(const double* x, const double* inputs, double* outputs, double t) {
    const double K = 20.0;
    const double T1 = 0.1;
    const double T2 = 0.0;
    const double T3 = 0.2;
    const double UO = 1.0;
    const double UC = -1.0;
    const double PMAX = 0.95;
    const double PMIN = 0.0;
    const double T4 = 0.1;
    const double K1 = 0.0;
    const double K2 = 0.0;
    const double T5 = 0.0;
    const double K3 = 0.0;
    const double K4 = 0.0;
    const double T6 = 0.0;
    const double K5 = 0.3;
    const double K6 = 0.0;
    const double T7 = 8.72;
    const double K7 = 0.7;
    const double K8 = 0.0;
    // --- Kernel ---
// Output map — auto-generated from SymbolicPHS 'IEEEG1_PHS'
double x1 = x[0];
double x2 = x[1];
outputs[0] = x2;
}
void step_IEEEG1_5(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double K = 20.0;
    const double T1 = 0.1;
    const double T2 = 0.0;
    const double T3 = 0.2;
    const double UO = 1.0;
    const double UC = -1.0;
    const double PMAX = 0.95;
    const double PMIN = 0.0;
    const double T4 = 0.1;
    const double K1 = 0.0;
    const double K2 = 0.0;
    const double T5 = 0.0;
    const double K3 = 0.0;
    const double K4 = 0.0;
    const double T6 = 0.0;
    const double K5 = 0.3;
    const double K6 = 0.0;
    const double T7 = 8.72;
    const double K7 = 0.7;
    const double K8 = 0.0;
    // --- Kernel ---
// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u
// Auto-generated from SymbolicPHS 'IEEEG1_PHS'

double x1 = x[0];
double x2 = x[1];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;

// dx1/dt
dxdt[0] = K*Pref/T1_eff - K*omega/T1_eff + K*u_agc/T1_eff - x1/T1_eff;
// dx2/dt
dxdt[1] = x1/T3_eff - x2/T3_eff;
}
void step_IEEEG1_5_out(const double* x, const double* inputs, double* outputs, double t) {
    const double K = 20.0;
    const double T1 = 0.1;
    const double T2 = 0.0;
    const double T3 = 0.2;
    const double UO = 1.0;
    const double UC = -1.0;
    const double PMAX = 0.95;
    const double PMIN = 0.0;
    const double T4 = 0.1;
    const double K1 = 0.0;
    const double K2 = 0.0;
    const double T5 = 0.0;
    const double K3 = 0.0;
    const double K4 = 0.0;
    const double T6 = 0.0;
    const double K5 = 0.3;
    const double K6 = 0.0;
    const double T7 = 8.72;
    const double K7 = 0.7;
    const double K8 = 0.0;
    // --- Kernel ---
// Output map — auto-generated from SymbolicPHS 'IEEEG1_PHS'
double x1 = x[0];
double x2 = x[1];
outputs[0] = x2;
}
void step_ST2CUT_3(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double MODE = 1.0;
    const double K1 = 10.0;
    const double K2 = 0.0;
    const double T1 = 0.0;
    const double T2 = 0.0;
    const double T3 = 3.0;
    const double T4 = 3.0;
    const double T5 = 0.15;
    const double T6 = 0.05;
    const double T7 = 0.15;
    const double T8 = 0.05;
    const double T9 = 0.15;
    const double T10 = 0.05;
    const double LSMAX = 0.05;
    const double LSMIN = -0.05;
    // --- Kernel ---
// Signal-flow code — auto-generated from 'ST2CUT_PHS'

double xl1 = x[0];
double xl2 = x[1];
double xwo = x[2];
double xll1 = x[3];
double xll2 = x[4];
double xll3 = x[5];

double omega = inputs[0];
double Pe = inputs[1];
double Tm = inputs[2];

double sig1_in;
int sig1_in_mode = (int)MODE;
if (sig1_in_mode == 1) {
    sig1_in = (omega - 1.0);
}
else if (sig1_in_mode == 3) {
    sig1_in = Pe;
}
else if (sig1_in_mode == 4) {
    sig1_in = (Tm - Pe);
}
else {
    sig1_in = 0.0;
}
double l1_track;
if (T1 > 1e-06) {
    dxdt[0] = (sig1_in - xl1) / T1;
    l1_track = xl1;
} else {
    dxdt[0] = 0.0;
    l1_track = sig1_in;
}
double L1_out = (K1 * l1_track);
double l2_track;
if (T2 > 1e-06) {
    dxdt[1] = (0.0 - xl2) / T2;
    l2_track = xl2;
} else {
    dxdt[1] = 0.0;
    l2_track = 0.0;
}
double L2_out = (K2 * l2_track);
double IN = ((0.0 + L1_out) + L2_out);
double wo_out;
if (T4 > 1e-06) {
    dxdt[2] = (IN - xwo) / T4;
    wo_out = T3 * (IN - xwo) / T4;
} else {
    dxdt[2] = 0.0;
    wo_out = T3 * (IN);
}
double ll1_out;
if (T6 > 1e-06) {
    dxdt[3] = (wo_out - xll1) / T6;
    ll1_out = xll1 + (T5 / T6) * (wo_out - xll1);
} else {
    dxdt[3] = 0.0;
    ll1_out = wo_out;
}
double ll2_out;
if (T8 > 1e-06) {
    dxdt[4] = (ll1_out - xll2) / T8;
    ll2_out = xll2 + (T7 / T8) * (ll1_out - xll2);
} else {
    dxdt[4] = 0.0;
    ll2_out = ll1_out;
}
double ll3_out;
if (T10 > 1e-06) {
    dxdt[5] = (ll2_out - xll3) / T10;
    ll3_out = xll3 + (T9 / T10) * (ll2_out - xll3);
} else {
    dxdt[5] = 0.0;
    ll3_out = ll2_out;
}
double Vss = ll3_out;
if (Vss > LSMAX) Vss = LSMAX;
if (Vss < LSMIN) Vss = LSMIN;
}
void step_ST2CUT_3_out(const double* x, const double* inputs, double* outputs, double t) {
    const double MODE = 1.0;
    const double K1 = 10.0;
    const double K2 = 0.0;
    const double T1 = 0.0;
    const double T2 = 0.0;
    const double T3 = 3.0;
    const double T4 = 3.0;
    const double T5 = 0.15;
    const double T6 = 0.05;
    const double T7 = 0.15;
    const double T8 = 0.05;
    const double T9 = 0.15;
    const double T10 = 0.05;
    const double LSMAX = 0.05;
    const double LSMIN = -0.05;
    // --- Kernel ---
// Signal-flow code — auto-generated from 'ST2CUT_PHS'

double xl1 = x[0];
double xl2 = x[1];
double xwo = x[2];
double xll1 = x[3];
double xll2 = x[4];
double xll3 = x[5];

double omega = inputs[0];
double Pe = inputs[1];
double Tm = inputs[2];

double sig1_in;
int sig1_in_mode = (int)MODE;
if (sig1_in_mode == 1) {
    sig1_in = (omega - 1.0);
}
else if (sig1_in_mode == 3) {
    sig1_in = Pe;
}
else if (sig1_in_mode == 4) {
    sig1_in = (Tm - Pe);
}
else {
    sig1_in = 0.0;
}
double l1_track;
if (T1 > 1e-06) {
    l1_track = xl1;
} else {
    l1_track = sig1_in;
}
double L1_out = (K1 * l1_track);
double l2_track;
if (T2 > 1e-06) {
    l2_track = xl2;
} else {
    l2_track = 0.0;
}
double L2_out = (K2 * l2_track);
double IN = ((0.0 + L1_out) + L2_out);
double wo_out;
if (T4 > 1e-06) {
    wo_out = T3 * (IN - xwo) / T4;
} else {
    wo_out = T3 * (IN);
}
double ll1_out;
if (T6 > 1e-06) {
    ll1_out = xll1 + (T5 / T6) * (wo_out - xll1);
} else {
    ll1_out = wo_out;
}
double ll2_out;
if (T8 > 1e-06) {
    ll2_out = xll2 + (T7 / T8) * (ll1_out - xll2);
} else {
    ll2_out = ll1_out;
}
double ll3_out;
if (T10 > 1e-06) {
    ll3_out = xll3 + (T9 / T10) * (ll2_out - xll3);
} else {
    ll3_out = ll2_out;
}
double Vss = ll3_out;
if (Vss > LSMAX) Vss = LSMAX;
if (Vss < LSMIN) Vss = LSMIN;

outputs[0] = Vss;
}
void step_IEEEST_1(const double* x, double* dxdt, const double* inputs, double* outputs, double t) {
    const double MODE = 3.0;
    const double A1 = 0.0;
    const double A2 = 0.0;
    const double A3 = 0.0;
    const double A4 = 0.0;
    const double A5 = 0.0;
    const double A6 = 0.0;
    const double T1 = 0.0;
    const double T2 = 0.0;
    const double T3 = 0.0;
    const double T4 = 0.75;
    const double T5 = 1.0;
    const double T6 = 4.2;
    const double KS = -2.0;
    const double LSMAX = 0.1;
    const double LSMIN = -0.1;
    // --- Kernel ---
// Signal-flow code — auto-generated from 'IEEEST_PHS'

double xf1 = x[0];
double xf2 = x[1];
double xll1 = x[2];
double xll2 = x[3];
double xll3 = x[4];
double xll4 = x[5];
double xwo = x[6];

double omega = inputs[0];
double Pe = inputs[1];
double Tm = inputs[2];

double sig_in;
int sig_in_mode = (int)MODE;
if (sig_in_mode == 1) {
    sig_in = (omega - 1.0);
}
else if (sig_in_mode == 3) {
    sig_in = Pe;
}
else if (sig_in_mode == 4) {
    sig_in = (Tm - Pe);
}
else {
    sig_in = 0.0;
}
double f1_out;
if (A1 > 1e-06) {
    dxdt[0] = (sig_in - xf1) / A1;
    f1_out = xf1;
} else {
    dxdt[0] = 0.0;
    f1_out = sig_in;
}
double f2_out;
if (A2 > 1e-06) {
    dxdt[1] = (f1_out - xf2) / A2;
    f2_out = xf2;
} else {
    dxdt[1] = 0.0;
    f2_out = f1_out;
}
double ll1_out;
if (A4 > 1e-06) {
    dxdt[2] = (f2_out - xll1) / A4;
    ll1_out = xll1 + (A3 / A4) * (f2_out - xll1);
} else {
    dxdt[2] = 0.0;
    ll1_out = f2_out;
}
double ll2_out;
if (A6 > 1e-06) {
    dxdt[3] = (ll1_out - xll2) / A6;
    ll2_out = xll2 + (A5 / A6) * (ll1_out - xll2);
} else {
    dxdt[3] = 0.0;
    ll2_out = ll1_out;
}
double ll3_out;
if (T2 > 1e-06) {
    dxdt[4] = (ll2_out - xll3) / T2;
    ll3_out = xll3 + (T1 / T2) * (ll2_out - xll3);
} else {
    dxdt[4] = 0.0;
    ll3_out = ll2_out;
}
double ll4_out;
if (T4 > 1e-06) {
    dxdt[5] = (ll3_out - xll4) / T4;
    ll4_out = xll4 + (T3 / T4) * (ll3_out - xll4);
} else {
    dxdt[5] = 0.0;
    ll4_out = ll3_out;
}
double vks = (KS * ll4_out);
double wo_out;
if (T6 > 1e-06) {
    dxdt[6] = (vks - xwo) / T6;
    wo_out = T5 * (vks - xwo) / T6;
} else {
    dxdt[6] = 0.0;
    wo_out = T5 * (vks);
}
double Vss = wo_out;
if (Vss > LSMAX) Vss = LSMAX;
if (Vss < LSMIN) Vss = LSMIN;
}
void step_IEEEST_1_out(const double* x, const double* inputs, double* outputs, double t) {
    const double MODE = 3.0;
    const double A1 = 0.0;
    const double A2 = 0.0;
    const double A3 = 0.0;
    const double A4 = 0.0;
    const double A5 = 0.0;
    const double A6 = 0.0;
    const double T1 = 0.0;
    const double T2 = 0.0;
    const double T3 = 0.0;
    const double T4 = 0.75;
    const double T5 = 1.0;
    const double T6 = 4.2;
    const double KS = -2.0;
    const double LSMAX = 0.1;
    const double LSMIN = -0.1;
    // --- Kernel ---
// Signal-flow code — auto-generated from 'IEEEST_PHS'

double xf1 = x[0];
double xf2 = x[1];
double xll1 = x[2];
double xll2 = x[3];
double xll3 = x[4];
double xll4 = x[5];
double xwo = x[6];

double omega = inputs[0];
double Pe = inputs[1];
double Tm = inputs[2];

double sig_in;
int sig_in_mode = (int)MODE;
if (sig_in_mode == 1) {
    sig_in = (omega - 1.0);
}
else if (sig_in_mode == 3) {
    sig_in = Pe;
}
else if (sig_in_mode == 4) {
    sig_in = (Tm - Pe);
}
else {
    sig_in = 0.0;
}
double f1_out;
if (A1 > 1e-06) {
    f1_out = xf1;
} else {
    f1_out = sig_in;
}
double f2_out;
if (A2 > 1e-06) {
    f2_out = xf2;
} else {
    f2_out = f1_out;
}
double ll1_out;
if (A4 > 1e-06) {
    ll1_out = xll1 + (A3 / A4) * (f2_out - xll1);
} else {
    ll1_out = f2_out;
}
double ll2_out;
if (A6 > 1e-06) {
    ll2_out = xll2 + (A5 / A6) * (ll1_out - xll2);
} else {
    ll2_out = ll1_out;
}
double ll3_out;
if (T2 > 1e-06) {
    ll3_out = xll3 + (T1 / T2) * (ll2_out - xll3);
} else {
    ll3_out = ll2_out;
}
double ll4_out;
if (T4 > 1e-06) {
    ll4_out = xll4 + (T3 / T4) * (ll3_out - xll4);
} else {
    ll4_out = ll3_out;
}
double vks = (KS * ll4_out);
double wo_out;
if (T6 > 1e-06) {
    wo_out = T5 * (vks - xwo) / T6;
} else {
    wo_out = T5 * (vks);
}
double Vss = wo_out;
if (Vss > LSMAX) Vss = LSMAX;
if (Vss < LSMIN) Vss = LSMIN;

outputs[0] = Vss;
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
        inputs_GENROU_1[3] = outputs_ESST3A_2[0]; // Efd
        step_GENROU_1_out(&x[0], inputs_GENROU_1, outputs_GENROU_1, t);
        Id_inj[0] += outputs_GENROU_1[0];
        Iq_inj[0] += outputs_GENROU_1[1];
    }
    { // GENROU_2
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_IEEEG1_4[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_1[0]; // Efd
        step_GENROU_2_out(&x[6], inputs_GENROU_2, outputs_GENROU_2, t);
        Id_inj[1] += outputs_GENROU_2[0];
        Iq_inj[1] += outputs_GENROU_2[1];
    }
    { // GENROU_3
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_IEEEG1_5[0]; // Tm
        inputs_GENROU_3[3] = outputs_ESST3A_3[0]; // Efd
        step_GENROU_3_out(&x[12], inputs_GENROU_3, outputs_GENROU_3, t);
        Id_inj[2] += outputs_GENROU_3[0];
        Iq_inj[2] += outputs_GENROU_3[1];
    }
    { // GENROU_4
        inputs_GENROU_4[0] = Vd_net[5]; // Vd
        inputs_GENROU_4[1] = Vq_net[5]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_4[3] = outputs_ESST3A_4[0]; // Efd
        step_GENROU_4_out(&x[18], inputs_GENROU_4, outputs_GENROU_4, t);
        Id_inj[5] += outputs_GENROU_4[0];
        Iq_inj[5] += outputs_GENROU_4[1];
    }
    { // GENROU_5
        inputs_GENROU_5[0] = Vd_net[7]; // Vd
        inputs_GENROU_5[1] = Vq_net[7]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_5[3] = outputs_ESST3A_5[0]; // Efd
        step_GENROU_5_out(&x[24], inputs_GENROU_5, outputs_GENROU_5, t);
        Id_inj[7] += outputs_GENROU_5[0];
        Iq_inj[7] += outputs_GENROU_5[1];
    }
    { // ESST3A_2
        inputs_ESST3A_2[0] = Vterm_net[0]; // Vterm
        inputs_ESST3A_2[1] = 1.2491057516128623; // Vref
        inputs_ESST3A_2[2] = outputs_GENROU_1[5]; // id_dq
        inputs_ESST3A_2[3] = outputs_GENROU_1[6]; // iq_dq
        inputs_ESST3A_2[4] = vd_dq_GENROU_1; // Vd
        inputs_ESST3A_2[5] = vq_dq_GENROU_1; // Vq
        step_ESST3A_2_out(&x[30], inputs_ESST3A_2, outputs_ESST3A_2, t);
    }
    { // ESST3A_3
        inputs_ESST3A_3[0] = Vterm_net[2]; // Vterm
        inputs_ESST3A_3[1] = 1.0672880008722112; // Vref
        inputs_ESST3A_3[2] = outputs_GENROU_3[5]; // id_dq
        inputs_ESST3A_3[3] = outputs_GENROU_3[6]; // iq_dq
        inputs_ESST3A_3[4] = vd_dq_GENROU_3; // Vd
        inputs_ESST3A_3[5] = vq_dq_GENROU_3; // Vq
        step_ESST3A_3_out(&x[35], inputs_ESST3A_3, outputs_ESST3A_3, t);
    }
    { // ESST3A_4
        inputs_ESST3A_4[0] = Vterm_net[5]; // Vterm
        inputs_ESST3A_4[1] = 1.151244611057205; // Vref
        inputs_ESST3A_4[2] = outputs_GENROU_4[5]; // id_dq
        inputs_ESST3A_4[3] = outputs_GENROU_4[6]; // iq_dq
        inputs_ESST3A_4[4] = vd_dq_GENROU_4; // Vd
        inputs_ESST3A_4[5] = vq_dq_GENROU_4; // Vq
        step_ESST3A_4_out(&x[40], inputs_ESST3A_4, outputs_ESST3A_4, t);
    }
    { // ESST3A_5
        inputs_ESST3A_5[0] = Vterm_net[7]; // Vterm
        inputs_ESST3A_5[1] = 1.1664395347233716; // Vref
        inputs_ESST3A_5[2] = outputs_GENROU_5[5]; // id_dq
        inputs_ESST3A_5[3] = outputs_GENROU_5[6]; // iq_dq
        inputs_ESST3A_5[4] = vd_dq_GENROU_5; // Vd
        inputs_ESST3A_5[5] = vq_dq_GENROU_5; // Vq
        step_ESST3A_5_out(&x[45], inputs_ESST3A_5, outputs_ESST3A_5, t);
    }
    { // EXST1_1
        inputs_EXST1_1[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_1[1] = 1.17104862089841; // Vref
        step_EXST1_1_out(&x[50], inputs_EXST1_1, outputs_EXST1_1, t);
    }
    { // TGOV1_1
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0404749870427543; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1_out(&x[54], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // TGOV1_2
        inputs_TGOV1_2[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_2[1] = 1.0144700805637359; // Pref
        inputs_TGOV1_2[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_2_out(&x[57], inputs_TGOV1_2, outputs_TGOV1_2, t);
    }
    { // TGOV1_3
        inputs_TGOV1_3[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_3[1] = 1.0173706310161303; // Pref
        inputs_TGOV1_3[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_3_out(&x[60], inputs_TGOV1_3, outputs_TGOV1_3, t);
    }
    { // IEEEG1_4
        inputs_IEEEG1_4[0] = outputs_GENROU_2[2]; // omega
        inputs_IEEEG1_4[1] = 1.0201002233014962; // Pref
        inputs_IEEEG1_4[2] = 0.0; // UNWIRED u_agc
        step_IEEEG1_4_out(&x[63], inputs_IEEEG1_4, outputs_IEEEG1_4, t);
    }
    { // IEEEG1_5
        inputs_IEEEG1_5[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEG1_5[1] = 1.0206493476102367; // Pref
        inputs_IEEEG1_5[2] = 0.0; // UNWIRED u_agc
        step_IEEEG1_5_out(&x[65], inputs_IEEEG1_5, outputs_IEEEG1_5, t);
    }
    { // ST2CUT_3
        inputs_ST2CUT_3[0] = outputs_GENROU_2[2]; // omega
        inputs_ST2CUT_3[1] = outputs_GENROU_2[3]; // Pe
        inputs_ST2CUT_3[2] = outputs_IEEEG1_4[0]; // Tm
        step_ST2CUT_3_out(&x[67], inputs_ST2CUT_3, outputs_ST2CUT_3, t);
    }
    { // IEEEST_1
        inputs_IEEEST_1[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEST_1[1] = outputs_GENROU_3[3]; // Pe
        inputs_IEEEST_1[2] = outputs_IEEEG1_5[0]; // Tm
        step_IEEEST_1_out(&x[73], inputs_IEEEST_1, outputs_IEEEST_1, t);
    }

    // Refresh actual dq-frame stator currents
    {
        double psi_d_pp = x[2]*0.177778 + x[3]*(1.0-0.177778);
        double psi_q_pp = -x[4]*0.123077 + x[5]*(1.0-0.123077);
        double rhs_d = vd_dq_GENROU_1 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_1 - psi_d_pp;
        outputs_GENROU_1[5] = (-0.000000*rhs_d - 0.230000*rhs_q) / 0.052900;
        outputs_GENROU_1[6] = (0.230000*rhs_d + -0.000000*rhs_q) / 0.052900;
    }
    {
        double psi_d_pp = x[8]*0.288889 + x[9]*(1.0-0.288889);
        double psi_q_pp = -x[10]*0.200000 + x[11]*(1.0-0.200000);
        double rhs_d = vd_dq_GENROU_2 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_2 - psi_d_pp;
        outputs_GENROU_2[5] = (-0.000000*rhs_d - 0.280000*rhs_q) / 0.078400;
        outputs_GENROU_2[6] = (0.280000*rhs_d + -0.000000*rhs_q) / 0.078400;
    }
    {
        double psi_d_pp = x[14]*0.422222 + x[15]*(1.0-0.422222);
        double psi_q_pp = -x[16]*0.292308 + x[17]*(1.0-0.292308);
        double rhs_d = vd_dq_GENROU_3 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_3 - psi_d_pp;
        outputs_GENROU_3[5] = (-0.000000*rhs_d - 0.340000*rhs_q) / 0.115600;
        outputs_GENROU_3[6] = (0.340000*rhs_d + -0.000000*rhs_q) / 0.115600;
    }
    {
        double psi_d_pp = x[20]*0.288889 + x[21]*(1.0-0.288889);
        double psi_q_pp = -x[22]*0.200000 + x[23]*(1.0-0.200000);
        double rhs_d = vd_dq_GENROU_4 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_4 - psi_d_pp;
        outputs_GENROU_4[5] = (-0.000000*rhs_d - 0.280000*rhs_q) / 0.078400;
        outputs_GENROU_4[6] = (0.280000*rhs_d + -0.000000*rhs_q) / 0.078400;
    }
    {
        double psi_d_pp = x[26]*0.422222 + x[27]*(1.0-0.422222);
        double psi_q_pp = -x[28]*0.292308 + x[29]*(1.0-0.292308);
        double rhs_d = vd_dq_GENROU_5 + psi_q_pp;
        double rhs_q = vq_dq_GENROU_5 - psi_d_pp;
        outputs_GENROU_5[5] = (-0.000000*rhs_d - 0.340000*rhs_q) / 0.115600;
        outputs_GENROU_5[6] = (0.340000*rhs_d + -0.000000*rhs_q) / 0.115600;
    }

    // --- 2. Compute Dynamics (dxdt) ---
    double dxdt[81];
    { // GENROU_1 dynamics
        inputs_GENROU_1[0] = Vd_net[0]; // Vd
        inputs_GENROU_1[1] = Vq_net[0]; // Vq
        inputs_GENROU_1[2] = outputs_TGOV1_1[0]; // Tm
        inputs_GENROU_1[3] = outputs_ESST3A_2[0]; // Efd
        step_GENROU_1(&x[0], &dxdt[0], inputs_GENROU_1, outputs_GENROU_1, t);
    }
    { // GENROU_2 dynamics
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_IEEEG1_4[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_1[0]; // Efd
        step_GENROU_2(&x[6], &dxdt[6], inputs_GENROU_2, outputs_GENROU_2, t);
    }
    { // GENROU_3 dynamics
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_IEEEG1_5[0]; // Tm
        inputs_GENROU_3[3] = outputs_ESST3A_3[0]; // Efd
        step_GENROU_3(&x[12], &dxdt[12], inputs_GENROU_3, outputs_GENROU_3, t);
    }
    { // GENROU_4 dynamics
        inputs_GENROU_4[0] = Vd_net[5]; // Vd
        inputs_GENROU_4[1] = Vq_net[5]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_4[3] = outputs_ESST3A_4[0]; // Efd
        step_GENROU_4(&x[18], &dxdt[18], inputs_GENROU_4, outputs_GENROU_4, t);
    }
    { // GENROU_5 dynamics
        inputs_GENROU_5[0] = Vd_net[7]; // Vd
        inputs_GENROU_5[1] = Vq_net[7]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_5[3] = outputs_ESST3A_5[0]; // Efd
        step_GENROU_5(&x[24], &dxdt[24], inputs_GENROU_5, outputs_GENROU_5, t);
    }
    { // ESST3A_2 dynamics
        inputs_ESST3A_2[0] = Vterm_net[0]; // Vterm
        inputs_ESST3A_2[1] = 1.2491057516128623; // Vref
        inputs_ESST3A_2[2] = outputs_GENROU_1[5]; // id_dq
        inputs_ESST3A_2[3] = outputs_GENROU_1[6]; // iq_dq
        inputs_ESST3A_2[4] = vd_dq_GENROU_1; // Vd
        inputs_ESST3A_2[5] = vq_dq_GENROU_1; // Vq
        step_ESST3A_2(&x[30], &dxdt[30], inputs_ESST3A_2, outputs_ESST3A_2, t);
    }
    { // ESST3A_3 dynamics
        inputs_ESST3A_3[0] = Vterm_net[2]; // Vterm
        inputs_ESST3A_3[1] = 1.0672880008722112; // Vref
        inputs_ESST3A_3[2] = outputs_GENROU_3[5]; // id_dq
        inputs_ESST3A_3[3] = outputs_GENROU_3[6]; // iq_dq
        inputs_ESST3A_3[4] = vd_dq_GENROU_3; // Vd
        inputs_ESST3A_3[5] = vq_dq_GENROU_3; // Vq
        step_ESST3A_3(&x[35], &dxdt[35], inputs_ESST3A_3, outputs_ESST3A_3, t);
    }
    { // ESST3A_4 dynamics
        inputs_ESST3A_4[0] = Vterm_net[5]; // Vterm
        inputs_ESST3A_4[1] = 1.151244611057205; // Vref
        inputs_ESST3A_4[2] = outputs_GENROU_4[5]; // id_dq
        inputs_ESST3A_4[3] = outputs_GENROU_4[6]; // iq_dq
        inputs_ESST3A_4[4] = vd_dq_GENROU_4; // Vd
        inputs_ESST3A_4[5] = vq_dq_GENROU_4; // Vq
        step_ESST3A_4(&x[40], &dxdt[40], inputs_ESST3A_4, outputs_ESST3A_4, t);
    }
    { // ESST3A_5 dynamics
        inputs_ESST3A_5[0] = Vterm_net[7]; // Vterm
        inputs_ESST3A_5[1] = 1.1664395347233716; // Vref
        inputs_ESST3A_5[2] = outputs_GENROU_5[5]; // id_dq
        inputs_ESST3A_5[3] = outputs_GENROU_5[6]; // iq_dq
        inputs_ESST3A_5[4] = vd_dq_GENROU_5; // Vd
        inputs_ESST3A_5[5] = vq_dq_GENROU_5; // Vq
        step_ESST3A_5(&x[45], &dxdt[45], inputs_ESST3A_5, outputs_ESST3A_5, t);
    }
    { // EXST1_1 dynamics
        inputs_EXST1_1[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_1[1] = 1.17104862089841; // Vref
        step_EXST1_1(&x[50], &dxdt[50], inputs_EXST1_1, outputs_EXST1_1, t);
    }
    { // TGOV1_1 dynamics
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0404749870427543; // Pref
        inputs_TGOV1_1[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_1(&x[54], &dxdt[54], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    { // TGOV1_2 dynamics
        inputs_TGOV1_2[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_2[1] = 1.0144700805637359; // Pref
        inputs_TGOV1_2[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_2(&x[57], &dxdt[57], inputs_TGOV1_2, outputs_TGOV1_2, t);
    }
    { // TGOV1_3 dynamics
        inputs_TGOV1_3[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_3[1] = 1.0173706310161303; // Pref
        inputs_TGOV1_3[2] = 0.0; // UNWIRED u_agc
        step_TGOV1_3(&x[60], &dxdt[60], inputs_TGOV1_3, outputs_TGOV1_3, t);
    }
    { // IEEEG1_4 dynamics
        inputs_IEEEG1_4[0] = outputs_GENROU_2[2]; // omega
        inputs_IEEEG1_4[1] = 1.0201002233014962; // Pref
        inputs_IEEEG1_4[2] = 0.0; // UNWIRED u_agc
        step_IEEEG1_4(&x[63], &dxdt[63], inputs_IEEEG1_4, outputs_IEEEG1_4, t);
    }
    { // IEEEG1_5 dynamics
        inputs_IEEEG1_5[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEG1_5[1] = 1.0206493476102367; // Pref
        inputs_IEEEG1_5[2] = 0.0; // UNWIRED u_agc
        step_IEEEG1_5(&x[65], &dxdt[65], inputs_IEEEG1_5, outputs_IEEEG1_5, t);
    }
    { // ST2CUT_3 dynamics
        inputs_ST2CUT_3[0] = outputs_GENROU_2[2]; // omega
        inputs_ST2CUT_3[1] = outputs_GENROU_2[3]; // Pe
        inputs_ST2CUT_3[2] = outputs_IEEEG1_4[0]; // Tm
        step_ST2CUT_3(&x[67], &dxdt[67], inputs_ST2CUT_3, outputs_ST2CUT_3, t);
    }
    { // IEEEST_1 dynamics
        inputs_IEEEST_1[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEST_1[1] = outputs_GENROU_3[3]; // Pe
        inputs_IEEEST_1[2] = outputs_IEEEG1_5[0]; // Tm
        step_IEEEST_1(&x[73], &dxdt[73], inputs_IEEEST_1, outputs_IEEEST_1, t);
    }

    // COI Reference Frame Correction
    double coi_total_2H = 8.000000 + 13.000000 + 10.000000 + 10.000000 + 10.000000;
    double coi_omega = (8.000000 * x[1] + 13.000000 * x[7] + 10.000000 * x[13] + 10.000000 * x[19] + 10.000000 * x[25]) / coi_total_2H;
    double omega_b_sys = 2.0 * M_PI * 60.0;
    dxdt[0] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[6] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[12] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[18] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[24] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[80] = omega_b_sys * (coi_omega - 1.0);

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
    outfile << (t) << "," << (x[0]) << "," << (x[1]) << "," << (x[2]) << "," << (x[3]) << "," << (x[4]) << "," << (x[5]) << "," << (y[0] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_1[3]) << "," << (inputs_GENROU_1[2]) << "," << (outputs_GENROU_1[3]) << "," << (outputs_GENROU_1[4]) << "," << (sqrt(inputs_GENROU_1[0]*inputs_GENROU_1[0] + inputs_GENROU_1[1]*inputs_GENROU_1[1])) << "," << (y[2]) << "," << (y[1]) << "," << ((4.0*((y[1])*(y[1])) - 8.0*y[1] + 0.30303030303030298*((y[2])*(y[2])) + 1.3513513513513513*((y[3])*(y[3])) + 0.3125*((y[4])*(y[4])) + 0.8771929824561403*((y[5])*(y[5])) + 4.0)) << "," << (x[6]) << "," << (x[7]) << "," << (x[8]) << "," << (x[9]) << "," << (x[10]) << "," << (x[11]) << "," << (y[6] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_2[3]) << "," << (inputs_GENROU_2[2]) << "," << (outputs_GENROU_2[3]) << "," << (outputs_GENROU_2[4]) << "," << (sqrt(inputs_GENROU_2[0]*inputs_GENROU_2[0] + inputs_GENROU_2[1]*inputs_GENROU_2[1])) << "," << (y[8]) << "," << (y[7]) << "," << ((6.5*((y[7])*(y[7])) - 13.0*y[7] + 0.30303030303030298*((y[8])*(y[8])) + 1.5625000000000002*((y[9])*(y[9])) + 0.3125*((y[10])*(y[10])) + 0.96153846153846145*((y[11])*(y[11])) + 6.5)) << "," << (x[12]) << "," << (x[13]) << "," << (x[14]) << "," << (x[15]) << "," << (x[16]) << "," << (x[17]) << "," << (y[12] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_3[3]) << "," << (inputs_GENROU_3[2]) << "," << (outputs_GENROU_3[3]) << "," << (outputs_GENROU_3[4]) << "," << (sqrt(inputs_GENROU_3[0]*inputs_GENROU_3[0] + inputs_GENROU_3[1]*inputs_GENROU_3[1])) << "," << (y[14]) << "," << (y[13]) << "," << ((5.0*((y[13])*(y[13])) - 10.0*y[13] + 0.30303030303030298*((y[14])*(y[14])) + 1.9230769230769234*((y[15])*(y[15])) + 0.3125*((y[16])*(y[16])) + 1.0869565217391304*((y[17])*(y[17])) + 5.0)) << "," << (x[18]) << "," << (x[19]) << "," << (x[20]) << "," << (x[21]) << "," << (x[22]) << "," << (x[23]) << "," << (y[18] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_4[3]) << "," << (inputs_GENROU_4[2]) << "," << (outputs_GENROU_4[3]) << "," << (outputs_GENROU_4[4]) << "," << (sqrt(inputs_GENROU_4[0]*inputs_GENROU_4[0] + inputs_GENROU_4[1]*inputs_GENROU_4[1])) << "," << (y[20]) << "," << (y[19]) << "," << ((5.0*((y[19])*(y[19])) - 10.0*y[19] + 0.30303030303030298*((y[20])*(y[20])) + 1.5625000000000002*((y[21])*(y[21])) + 0.3125*((y[22])*(y[22])) + 0.96153846153846145*((y[23])*(y[23])) + 5.0)) << "," << (x[24]) << "," << (x[25]) << "," << (x[26]) << "," << (x[27]) << "," << (x[28]) << "," << (x[29]) << "," << (y[24] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_5[3]) << "," << (inputs_GENROU_5[2]) << "," << (outputs_GENROU_5[3]) << "," << (outputs_GENROU_5[4]) << "," << (sqrt(inputs_GENROU_5[0]*inputs_GENROU_5[0] + inputs_GENROU_5[1]*inputs_GENROU_5[1])) << "," << (y[26]) << "," << (y[25]) << "," << ((5.0*((y[25])*(y[25])) - 10.0*y[25] + 0.30303030303030298*((y[26])*(y[26])) + 1.9230769230769234*((y[27])*(y[27])) + 0.3125*((y[28])*(y[28])) + 1.0869565217391304*((y[29])*(y[29])) + 5.0)) << "," << (x[30]) << "," << (x[31]) << "," << (x[32]) << "," << (x[33]) << "," << (x[34]) << "," << (y[34]*y[33]) << "," << ((0.5*(y[30]*y[30]+y[31]*y[31]+y[32]*y[32]+y[33]*y[33]+y[34]*y[34]))) << "," << (x[35]) << "," << (x[36]) << "," << (x[37]) << "," << (x[38]) << "," << (x[39]) << "," << (y[39]*y[38]) << "," << ((0.5*(y[35]*y[35]+y[36]*y[36]+y[37]*y[37]+y[38]*y[38]+y[39]*y[39]))) << "," << (x[40]) << "," << (x[41]) << "," << (x[42]) << "," << (x[43]) << "," << (x[44]) << "," << (y[44]*y[43]) << "," << ((0.5*(y[40]*y[40]+y[41]*y[41]+y[42]*y[42]+y[43]*y[43]+y[44]*y[44]))) << "," << (x[45]) << "," << (x[46]) << "," << (x[47]) << "," << (x[48]) << "," << (x[49]) << "," << (y[49]*y[48]) << "," << ((0.5*(y[45]*y[45]+y[46]*y[46]+y[47]*y[47]+y[48]*y[48]+y[49]*y[49]))) << "," << (x[50]) << "," << (x[51]) << "," << (x[52]) << "," << (x[53]) << "," << (y[52]) << "," << ((0.5*(y[50]*y[50]+y[51]*y[51]+y[52]*y[52]+y[53]*y[53]))) << "," << (x[54]) << "," << (x[55]) << "," << (x[56]) << "," << (outputs_TGOV1_1[0]) << "," << (y[54]) << "," << (y[56]) << "," << (x[57]) << "," << (x[58]) << "," << (x[59]) << "," << (outputs_TGOV1_2[0]) << "," << (y[57]) << "," << (y[59]) << "," << (x[60]) << "," << (x[61]) << "," << (x[62]) << "," << (outputs_TGOV1_3[0]) << "," << (y[60]) << "," << (y[62]) << "," << (x[63]) << "," << (x[64]) << "," << (y[64]) << "," << ((0.5*(y[63]*y[63]+y[64]*y[64]))) << "," << (x[65]) << "," << (x[66]) << "," << (y[66]) << "," << ((0.5*(y[65]*y[65]+y[66]*y[66]))) << "," << (x[67]) << "," << (x[68]) << "," << (x[69]) << "," << (x[70]) << "," << (x[71]) << "," << (x[72]) << "," << (outputs_ST2CUT_3[0]) << "," << (x[73]) << "," << (x[74]) << "," << (x[75]) << "," << (x[76]) << "," << (x[77]) << "," << (x[78]) << "," << (x[79]) << "," << (outputs_IEEEST_1[0]) << "," << (0.5*(y[73]*y[73]+y[74]*y[74]+y[75]*y[75]+y[76]*y[76]+y[77]*y[77]+y[78]*y[78]+y[79]*y[79])) << "," << (Vd_net[0]) << "," << (Vq_net[0]) << "," << (Vterm_net[0]) << "," << (Vd_net[1]) << "," << (Vq_net[1]) << "," << (Vterm_net[1]) << "," << (Vd_net[2]) << "," << (Vq_net[2]) << "," << (Vterm_net[2]) << "," << (Vd_net[3]) << "," << (Vq_net[3]) << "," << (Vterm_net[3]) << "," << (Vd_net[4]) << "," << (Vq_net[4]) << "," << (Vterm_net[4]) << "," << (Vd_net[5]) << "," << (Vq_net[5]) << "," << (Vterm_net[5]) << "," << (Vd_net[6]) << "," << (Vq_net[6]) << "," << (Vterm_net[6]) << "," << (Vd_net[7]) << "," << (Vq_net[7]) << "," << (Vterm_net[7]) << "," << (Vd_net[8]) << "," << (Vq_net[8]) << "," << (Vterm_net[8]) << "," << (Vd_net[9]) << "," << (Vq_net[9]) << "," << (Vterm_net[9]) << "," << (Vd_net[10]) << "," << (Vq_net[10]) << "," << (Vterm_net[10]) << "," << (Vd_net[11]) << "," << (Vq_net[11]) << "," << (Vterm_net[11]) << "," << (Vd_net[12]) << "," << (Vq_net[12]) << "," << (Vterm_net[12]) << "," << (Vd_net[13]) << "," << (Vq_net[13]) << "," << (Vterm_net[13]) << std::endl;
}

// =================================================================
// SUNDIALS IDA solver (variable-order BDF, adaptive step)
//   with event-driven reinitialization at fault boundaries
// =================================================================
int main() {
    // --- Context ---
    SUNContext sunctx = NULL;
    int ierr = SUNContext_Create(NULL, &sunctx);
    if (ierr) { std::cerr << "SUNContext_Create failed" << std::endl; return 1; }

    // --- Initial state y0 ---
    N_Vector yy = N_VNew_Serial(N_TOTAL, sunctx);
    N_Vector yp = N_VNew_Serial(N_TOTAL, sunctx);
    sunrealtype* yy_data = N_VGetArrayPointer(yy);
    sunrealtype* yp_data = N_VGetArrayPointer(yp);

    double y0_arr[N_TOTAL] = { 0.849671477136, 1.000000000000, 1.045877921581, 0.819689199036, 0.440393528067, -0.704629644907, 0.532921601633, 1.000000000000, 1.007109331657, 0.939754724156, 0.306324075656, -0.473996201278, 0.623888080090, 1.000000000000, 0.893057746143, 0.846516820658, 0.358465039106, -0.532037584357, 0.207638886037, 1.000000000000, 1.189177759643, 1.085851727741, 0.176603314218, -0.273270391475, 0.356223332897, 1.000000000000, 1.154728025922, 1.083414260829, 0.225533422937, -0.334739080359, 1.060098103141, 0.189007648472, 1.890076484719, 0.553067585244, 3.217442162853, 1.010011211537, 0.057276789336, 1.145535786711, 0.301390152799, 3.675840127236, 1.069977092532, 0.081267518525, 1.625350370504, 0.389599929820, 4.046844618282, 1.089975106794, 0.076464427929, 1.529288558579, 0.363360627065, 4.083734916970, 1.045079709920, 0.125968910979, 1.259689109786, 1.259689109786, 0.809499740855, 0.809499740855, 0.000000000000, 0.289401611275, 0.289401611275, 0.000000000000, 0.347412620323, 0.347412620323, 0.000000000000, 0.402004466030, 0.402004466030, 0.412986952205, 0.412986952205, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.400000000829, 0.400000000829, 0.400000000829, 0.400000000829, 0.400000000829, 0.400000000829, -0.800000001659, 0.000000000000, 1.059910875872, 0.025862419993, 1.045197259924, -0.000000468608, 1.009827185513, -0.024535386080, 1.032750276609, -0.049169238031, 1.040702966574, -0.040143700045, 1.066415949965, -0.088299444618, 1.063027260060, -0.059872851025, 1.090051426480, -0.003880377784, 1.058111056037, -0.100380614045, 1.051747888587, -0.102842680356, 1.055398807615, -0.097794750142, 1.053088712857, -0.104699248599, 1.050121068304, -0.108769756016, 1.050359097841, -0.139307343170 };
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
    double var_id_arr[N_TOTAL] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
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
    std::printf("[IC-diag] TGOV1_1 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[1] * (2.0 * 4.0), yp_data[1]);
    // IC-diag: TGOV1_2 residual torque mismatch of GENROU_4
    std::printf("[IC-diag] TGOV1_2 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[19] * (2.0 * 5.0), yp_data[19]);
    // IC-diag: TGOV1_3 residual torque mismatch of GENROU_5
    std::printf("[IC-diag] TGOV1_3 torque_residual=%.6e  (omega_dot=%.6e)\n", yp_data[25] * (2.0 * 5.0), yp_data[25]);

    // === IC exciter Vm diagnostic ===
    // After _sync_voltages_to_states, Vm should match Vterm_IDA.
    // These diagnostics verify the residual is negligible.
    // IC-diag: ESST3A_2.Vm residual at bus 1
    {
        double Vd_ic = yy_data[N_DIFF + 0];
        double Vq_ic = yy_data[N_DIFF + 1];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] ESST3A_2.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[30], Vt_ic, Vt_ic - yy_data[30]);
    }
    // IC-diag: ESST3A_3.Vm residual at bus 3
    {
        double Vd_ic = yy_data[N_DIFF + 4];
        double Vq_ic = yy_data[N_DIFF + 5];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] ESST3A_3.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[35], Vt_ic, Vt_ic - yy_data[35]);
    }
    // IC-diag: ESST3A_4.Vm residual at bus 6
    {
        double Vd_ic = yy_data[N_DIFF + 10];
        double Vq_ic = yy_data[N_DIFF + 11];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] ESST3A_4.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[40], Vt_ic, Vt_ic - yy_data[40]);
    }
    // IC-diag: ESST3A_5.Vm residual at bus 8
    {
        double Vd_ic = yy_data[N_DIFF + 14];
        double Vq_ic = yy_data[N_DIFF + 15];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] ESST3A_5.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[45], Vt_ic, Vt_ic - yy_data[45]);
    }
    // IC-diag: EXST1_1.Vm residual at bus 2
    {
        double Vd_ic = yy_data[N_DIFF + 2];
        double Vq_ic = yy_data[N_DIFF + 3];
        double Vt_ic = sqrt(Vd_ic*Vd_ic + Vq_ic*Vq_ic);
        std::printf("[IC-diag] EXST1_1.Vm=%.6f  Vterm_IDA=%.6f  (delta=%.2e)\n", yy_data[50], Vt_ic, Vt_ic - yy_data[50]);
    }

    // Re-sync yp from the (unmodified) state, then tell IDA about the IC.
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = 0.0;
    dae_residual(yy_data, yp_data, res0, 0.0);
    for (int i = 0; i < N_DIFF; ++i) yp_data[i] = -res0[i];
    for (int i = N_DIFF; i < N_TOTAL; ++i) yp_data[i] = 0.0;
    IDAReInit(ida_mem, 0.0, yy, yp);

    // --- Output setup ---
    std::ofstream outfile("simulation_results.csv");
    outfile << "t,GENROU_1.delta,GENROU_1.omega,GENROU_1.E_q_prime,GENROU_1.psi_d,GENROU_1.E_d_prime,GENROU_1.psi_q,GENROU_1.delta_deg,GENROU_1.Te,GENROU_1.Tm_in,GENROU_1.Pe,GENROU_1.Qe,GENROU_1.V_term,GENROU_1.Eq_p,GENROU_1.omega,GENROU_1.H_total,GENROU_2.delta,GENROU_2.omega,GENROU_2.E_q_prime,GENROU_2.psi_d,GENROU_2.E_d_prime,GENROU_2.psi_q,GENROU_2.delta_deg,GENROU_2.Te,GENROU_2.Tm_in,GENROU_2.Pe,GENROU_2.Qe,GENROU_2.V_term,GENROU_2.Eq_p,GENROU_2.omega,GENROU_2.H_total,GENROU_3.delta,GENROU_3.omega,GENROU_3.E_q_prime,GENROU_3.psi_d,GENROU_3.E_d_prime,GENROU_3.psi_q,GENROU_3.delta_deg,GENROU_3.Te,GENROU_3.Tm_in,GENROU_3.Pe,GENROU_3.Qe,GENROU_3.V_term,GENROU_3.Eq_p,GENROU_3.omega,GENROU_3.H_total,GENROU_4.delta,GENROU_4.omega,GENROU_4.E_q_prime,GENROU_4.psi_d,GENROU_4.E_d_prime,GENROU_4.psi_q,GENROU_4.delta_deg,GENROU_4.Te,GENROU_4.Tm_in,GENROU_4.Pe,GENROU_4.Qe,GENROU_4.V_term,GENROU_4.Eq_p,GENROU_4.omega,GENROU_4.H_total,GENROU_5.delta,GENROU_5.omega,GENROU_5.E_q_prime,GENROU_5.psi_d,GENROU_5.E_d_prime,GENROU_5.psi_q,GENROU_5.delta_deg,GENROU_5.Te,GENROU_5.Tm_in,GENROU_5.Pe,GENROU_5.Qe,GENROU_5.V_term,GENROU_5.Eq_p,GENROU_5.omega,GENROU_5.H_total,ESST3A_2.Vm,ESST3A_2.LLx,ESST3A_2.Vr,ESST3A_2.VM,ESST3A_2.VB,ESST3A_2.Efd,ESST3A_2.H_exc,ESST3A_3.Vm,ESST3A_3.LLx,ESST3A_3.Vr,ESST3A_3.VM,ESST3A_3.VB,ESST3A_3.Efd,ESST3A_3.H_exc,ESST3A_4.Vm,ESST3A_4.LLx,ESST3A_4.Vr,ESST3A_4.VM,ESST3A_4.VB,ESST3A_4.Efd,ESST3A_4.H_exc,ESST3A_5.Vm,ESST3A_5.LLx,ESST3A_5.Vr,ESST3A_5.VM,ESST3A_5.VB,ESST3A_5.Efd,ESST3A_5.H_exc,EXST1_1.Vm,EXST1_1.LLx,EXST1_1.Vr,EXST1_1.Vf,EXST1_1.Efd,EXST1_1.H_exc,TGOV1_1.x1,TGOV1_1.x2,TGOV1_1.xi,TGOV1_1.Tm,TGOV1_1.Valve,TGOV1_1.xi,TGOV1_2.x1,TGOV1_2.x2,TGOV1_2.xi,TGOV1_2.Tm,TGOV1_2.Valve,TGOV1_2.xi,TGOV1_3.x1,TGOV1_3.x2,TGOV1_3.xi,TGOV1_3.Tm,TGOV1_3.Valve,TGOV1_3.xi,IEEEG1_4.x1,IEEEG1_4.x2,IEEEG1_4.Tm,IEEEG1_4.H_gov,IEEEG1_5.x1,IEEEG1_5.x2,IEEEG1_5.Tm,IEEEG1_5.H_gov,ST2CUT_3.xl1,ST2CUT_3.xl2,ST2CUT_3.xwo,ST2CUT_3.xll1,ST2CUT_3.xll2,ST2CUT_3.xll3,ST2CUT_3.Vss,IEEEST_1.xf1,IEEEST_1.xf2,IEEEST_1.xll1,IEEEST_1.xll2,IEEEST_1.xll3,IEEEST_1.xll4,IEEEST_1.xwo,IEEEST_1.Vss,IEEEST_1.H_pss,Vd_Bus1,Vq_Bus1,Vterm_Bus1,Vd_Bus2,Vq_Bus2,Vterm_Bus2,Vd_Bus3,Vq_Bus3,Vterm_Bus3,Vd_Bus4,Vq_Bus4,Vterm_Bus4,Vd_Bus5,Vq_Bus5,Vterm_Bus5,Vd_Bus6,Vq_Bus6,Vterm_Bus6,Vd_Bus7,Vq_Bus7,Vterm_Bus7,Vd_Bus8,Vq_Bus8,Vterm_Bus8,Vd_Bus9,Vq_Bus9,Vterm_Bus9,Vd_Bus10,Vq_Bus10,Vterm_Bus10,Vd_Bus11,Vq_Bus11,Vterm_Bus11,Vd_Bus12,Vq_Bus12,Vterm_Bus12,Vd_Bus13,Vq_Bus13,Vterm_Bus13,Vd_Bus14,Vq_Bus14,Vterm_Bus14" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double Vd_net[N_BUS], Vq_net[N_BUS], Vterm_net[N_BUS];

    // Print header
    std::cout << "Dirac DAE Simulation (SUNDIALS IDA)" << std::endl;
    std::cout << "  Differential states: " << N_DIFF << std::endl;
    std::cout << "  Algebraic states:    " << N_ALG << std::endl;
    std::cout << "  Total DAE dimension: " << N_TOTAL << std::endl;
    std::cout << "  Buses:               " << N_BUS << std::endl;
    std::cout << "  max_dt = 0.0005,  T = 60.0 s" << std::endl;
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
    const double seg_ends[1] = { 60.000000000000 };
    const int n_segments = 1;
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
            if (t_ret >= t_next_log - 1e-12 || t_ret >= 60.0 - 1e-12) {
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
                    std::cout << "Stability limit reached. Stopping."
                              << std::endl;
                    aborted = true;
                    break;
                }
            }
        }

        t_current = t_seg_end;
    }

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
