#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <iostream>

const int N_STATES = 83;
const int N_COMPONENTS = 17;
const int N_BUS = 5;
const int N_TOPOS = 3;

// Topology 0: Pre-fault Kron-reduced Z-Bus (5x5)
const double Z_real_all[3][25] = {
    { // topo 0: pre-fault
        1.103793e-02, 4.717886e-03, 2.116449e-03, 4.109009e-03, 3.493721e-03, 4.717886e-03, 1.114774e-02, 5.389711e-03, 5.169519e-03, 4.758685e-03, 2.116449e-03, 5.389711e-03, 2.296988e-02, 4.706936e-03, 5.088203e-03, 4.109009e-03, 5.169519e-03, 4.706936e-03, 1.871980e-02, 5.024837e-03, 3.493721e-03, 4.758685e-03, 5.088203e-03, 5.024837e-03, 9.368294e-03
    },
    { // topo 1: fault@bus4
        1.253436e-02, 5.362397e-03, 2.416345e-03, 1.640505e-03, 9.210612e-05, 5.362397e-03, 1.071210e-02, 4.551480e-03, 1.449864e-03, 8.855494e-05, 2.416345e-03, 4.551480e-03, 2.171786e-02, 6.091472e-04, 5.934083e-05, 1.640505e-03, 1.449864e-03, 6.091472e-04, 1.289775e-02, -1.301182e-03, 9.210612e-05, 8.855494e-05, 5.934083e-05, -1.301182e-03, 2.698210e-03
    },
    { // topo 2: restored
        1.103793e-02, 4.717886e-03, 2.116449e-03, 4.109009e-03, 3.493721e-03, 4.717886e-03, 1.114774e-02, 5.389711e-03, 5.169519e-03, 4.758685e-03, 2.116449e-03, 5.389711e-03, 2.296988e-02, 4.706936e-03, 5.088203e-03, 4.109009e-03, 5.169519e-03, 4.706936e-03, 1.871980e-02, 5.024837e-03, 3.493721e-03, 4.758685e-03, 5.088203e-03, 5.024837e-03, 9.368294e-03
    }
};
const double Z_imag_all[3][25] = {
    { // topo 0: pre-fault
        9.379607e-02, 6.680482e-02, 4.644153e-02, 3.336845e-02, 2.609415e-02, 6.680482e-02, 8.448115e-02, 5.547101e-02, 3.600728e-02, 2.893198e-02, 4.644153e-02, 5.547101e-02, 1.167176e-01, 3.349057e-02, 2.886659e-02, 3.336845e-02, 3.600728e-02, 3.349057e-02, 1.331726e-01, 3.762993e-02, 2.609415e-02, 2.893198e-02, 2.886659e-02, 3.762993e-02, 1.814712e-01
    },
    { // topo 1: fault@bus4
        6.339592e-02, 3.269841e-02, 1.184817e-02, 5.252625e-03, 7.723581e-04, 3.269841e-02, 4.625174e-02, 1.670779e-02, 4.594453e-03, 6.808184e-04, 1.184817e-02, 1.670779e-02, 7.741726e-02, 1.673977e-03, 2.660408e-04, 5.252625e-03, 4.594453e-03, 1.673977e-03, 1.076565e-01, 1.479847e-02, 7.723581e-04, 6.808184e-04, 2.660408e-04, 1.479847e-02, 1.610884e-01
    },
    { // topo 2: restored
        9.379607e-02, 6.680482e-02, 4.644153e-02, 3.336845e-02, 2.609415e-02, 6.680482e-02, 8.448115e-02, 5.547101e-02, 3.600728e-02, 2.893198e-02, 4.644153e-02, 5.547101e-02, 1.167176e-01, 3.349057e-02, 2.886659e-02, 3.336845e-02, 3.600728e-02, 3.349057e-02, 1.331726e-01, 3.762993e-02, 2.609415e-02, 2.893198e-02, 2.886659e-02, 3.762993e-02, 1.814712e-01
    }
};
// Slack-bus Norton current bias per topology (network Re/Im frame)
const double I_slack_d_all[3][5] = {
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 },
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 },
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 }
};
const double I_slack_q_all[3][5] = {
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 },
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 },
    { 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00 }
};

// Fault event times: topology switches from 0->1->2->... at these times
const double FAULT_TIMES[2] = {
    2.000000, 2.150000
};
const int N_FAULT_EVENTS = 2;

const int N_OBS = 9;
const double OBS_M_real[3][45] = {
    {1.1866565940e-01, 3.8256670529e-01, 2.3577780711e-01, 1.6573134520e-01, 1.0443973967e-01, 1.9431399207e-01, 3.8895045740e-01, 1.4658956970e-01, 2.1006018825e-01, 6.6096820835e-02, 5.2653556946e-02, 1.7035328102e-01, 1.0526083527e-01, 2.2514291509e-01, 4.6652941481e-01, 5.0608276813e-02, 1.6445360025e-01, 1.0193691852e-01, 3.9672181870e-01, 3.2164361898e-01, 4.1761783963e-02, 1.3579311594e-01, 8.4210261094e-02, 5.0216884924e-01, 2.6522994831e-01, 2.1382482099e-02, 6.9607496306e-02, 4.3201792175e-02, 7.4509773327e-01, 1.3562567194e-01, 3.7793291753e-03, 1.2187111604e-02, 7.5122903550e-03, 9.5812683021e-01, 2.4225278697e-02, 7.4686132652e-03, 2.4353584225e-02, 1.5133105282e-02, 9.1682141384e-01, 4.7283363300e-02, 3.2500883731e-02, 1.0575091246e-01, 6.5611491027e-02, 6.3839773811e-01, 2.0625923752e-01},
    {3.3584944395e-04, 1.1171713144e-03, 7.0399662053e-04, 4.2858582959e-04, 2.3615106380e-04, 1.2046667754e-01, 1.5142310146e-01, 4.4723155205e-04, 1.0627747588e-01, 1.5410260912e-04, 1.4361596536e-04, 4.7995710100e-04, 3.0342247337e-04, 1.5249551652e-01, 4.2132677691e-01, 1.3160849481e-04, 4.4258374680e-04, 2.8099049220e-04, 3.2772855171e-01, 2.7942659217e-01, 1.0782749616e-04, 3.6295959039e-04, 2.3058840842e-04, 4.4533620696e-01, 2.3054092088e-01, 5.4492413072e-05, 1.8375197327e-04, 1.1687738400e-04, 7.1609155032e-01, 1.1800155993e-01, 1.0670184547e-05, 3.5504149498e-05, 2.2377996286e-05, 9.5286555996e-01, 2.0911544461e-02, 1.8669507495e-05, 6.3121728769e-05, 4.0221000824e-05, 9.0673705871e-01, 4.1197118905e-02, 8.3283172309e-05, 2.8062716908e-04, 1.7840598036e-04, 5.9424998094e-01, 1.7938377275e-01},
    {1.1866565940e-01, 3.8256670529e-01, 2.3577780711e-01, 1.6573134520e-01, 1.0443973967e-01, 1.9431399207e-01, 3.8895045740e-01, 1.4658956970e-01, 2.1006018825e-01, 6.6096820835e-02, 5.2653556946e-02, 1.7035328102e-01, 1.0526083527e-01, 2.2514291509e-01, 4.6652941481e-01, 5.0608276813e-02, 1.6445360025e-01, 1.0193691852e-01, 3.9672181870e-01, 3.2164361898e-01, 4.1761783963e-02, 1.3579311594e-01, 8.4210261094e-02, 5.0216884924e-01, 2.6522994831e-01, 2.1382482099e-02, 6.9607496306e-02, 4.3201792175e-02, 7.4509773327e-01, 1.3562567194e-01, 3.7793291753e-03, 1.2187111604e-02, 7.5122903550e-03, 9.5812683021e-01, 2.4225278697e-02, 7.4686132652e-03, 2.4353584225e-02, 1.5133105282e-02, 9.1682141384e-01, 4.7283363300e-02, 3.2500883731e-02, 1.0575091246e-01, 6.5611491027e-02, 6.3839773811e-01, 2.0625923752e-01}
};
const double OBS_M_imag[3][45] = {
    {1.7793535683e-03, 2.7091705638e-02, 2.6298576049e-02, -2.2619969453e-02, -3.5304611037e-02, 4.8329189606e-04, 3.3818137202e-02, 2.2267020452e-02, -3.9051362235e-02, -1.9458906021e-02, -2.5634109894e-03, 1.2023718683e-03, 4.9974180400e-03, 2.9814968223e-02, -3.7143118679e-02, -6.4519189054e-03, -1.1712284141e-02, -3.1321234488e-03, 7.6057714855e-02, -6.1246892767e-02, -5.8051171015e-03, -1.1216981707e-02, -3.5417336354e-03, 6.8495516904e-02, -5.3621984987e-02, -3.4167103142e-03, -7.1772073593e-03, -2.6977214678e-03, 4.0367323557e-02, -3.0301917362e-02, 4.0458277497e-05, 8.1052330908e-04, 8.0531401852e-04, -1.5321716042e-03, -1.2282474772e-03, -1.4191902603e-03, -3.2354019596e-03, -1.3915299441e-03, 1.3586556013e-02, -1.2030332979e-02, -4.9104758586e-03, -9.9965606315e-03, -3.5376730734e-03, 4.5100627551e-02, -4.4246388809e-02},
    {-1.8629759378e-04, -5.4068182380e-04, -3.0628182636e-04, -3.3063365520e-04, -2.6742448135e-04, -3.6770531814e-03, 7.1076795666e-03, -1.7382198719e-04, -2.9228886566e-02, -1.6115146176e-04, -9.2071242465e-05, -2.7123772764e-04, -1.5565881666e-04, 4.4365236079e-02, -1.8667844469e-02, -9.9685550792e-05, -2.9796646544e-04, -1.7311247496e-04, 9.5585759463e-02, -4.0005669574e-02, -8.3609938251e-05, -2.5037556285e-04, -1.4568626408e-04, 8.5278549810e-02, -3.5673629032e-02, -4.4056271371e-05, -1.3234772368e-04, -7.7211759668e-05, 4.9578119901e-02, -2.0723956707e-02, -5.9787985505e-06, -1.7371413177e-05, -9.8501598951e-06, -7.9975640840e-04, -9.8203038247e-05, -1.6021809548e-05, -4.8336966166e-05, -2.8299451221e-05, 1.7117570213e-02, -8.4876610960e-03, -6.6170847329e-05, -1.9852259646e-04, -1.1569341817e-04, 5.8707704763e-02, -2.9935174466e-02},
    {1.7793535683e-03, 2.7091705638e-02, 2.6298576049e-02, -2.2619969453e-02, -3.5304611037e-02, 4.8329189606e-04, 3.3818137202e-02, 2.2267020452e-02, -3.9051362235e-02, -1.9458906021e-02, -2.5634109894e-03, 1.2023718683e-03, 4.9974180400e-03, 2.9814968223e-02, -3.7143118679e-02, -6.4519189054e-03, -1.1712284141e-02, -3.1321234488e-03, 7.6057714855e-02, -6.1246892767e-02, -5.8051171015e-03, -1.1216981707e-02, -3.5417336354e-03, 6.8495516904e-02, -5.3621984987e-02, -3.4167103142e-03, -7.1772073593e-03, -2.6977214678e-03, 4.0367323557e-02, -3.0301917362e-02, 4.0458277497e-05, 8.1052330908e-04, 8.0531401852e-04, -1.5321716042e-03, -1.2282474772e-03, -1.4191902603e-03, -3.2354019596e-03, -1.3915299441e-03, 1.3586556013e-02, -1.2030332979e-02, -4.9104758586e-03, -9.9965606315e-03, -3.5376730734e-03, 4.5100627551e-02, -4.4246388809e-02}
};
double Vd_obs[9];
double Vq_obs[9];
double Vterm_obs[9];

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

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

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

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

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

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

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

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

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

            // ============================================================
            // Assemble dynamics ẋ = (J − R)∇H + g·u
            // ============================================================

            // --- Swing equation (textbook form, ω in pu, t in seconds) ---
            //
            //   dδ/dt = ωb · (ω − 1)
            //   dω/dt = (Tm − Te − D·(ω − 1)) / (2H)
            //
            // The earlier (J − R)∇H + g·u expansion that lived here
            // multiplied the Tm, Te and damping terms by an extra ωb,
            // collapsing the swing time constant from O(seconds) to
            // O(milliseconds) and turning the rotor into a heavily
            // overdamped first-order lag — first-swing instability could
            // not manifest no matter how long the fault. Match gencls.
            double inv_2H = 1.0 / (2.0 * H);
            dxdt[0] = omega_b * (omega - 1.0);
            dxdt[1] = inv_2H * (Tm - Te - D * (omega - 1.0));
            (void)dH_ddelta; (void)dH_domega;  // gradients unused by swing block

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
    const double T2 = 0.0;
    const double T3 = 2.1;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 0.0;
    const double T3 = 2.1;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 0.0;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 0.0;
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
    const double T2 = 0.0;
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
    const double R = 0.05;
    const double VMAX = 1.05;
    const double VMIN = 0.0;
    const double T1 = 0.05;
    const double T2 = 0.0;
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
double x3 = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;
double T7_eff = (T7 > 1e-4) ? T7 : 1e-4;

// dx1/dt
double _dx0_raw = K*Pref/T1_eff - K*omega/T1_eff + K*u_agc/T1_eff - x1/T1_eff;
if (x1 >= PMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= PMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= PMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= PMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dx3/dt
double _dx2_raw = x2/T7_eff - x3/T7_eff;
if (x3 >= PMAX && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (x3 <= PMIN && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
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
double x3 = x[2];
outputs[0] = K5*x2 + K7*x3;
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
double x3 = x[2];

double omega = inputs[0];
double Pref = inputs[1];
double u_agc = inputs[2];

double T1_eff = (T1 > 1e-4) ? T1 : 1e-4;
double T3_eff = (T3 > 1e-4) ? T3 : 1e-4;
double T7_eff = (T7 > 1e-4) ? T7 : 1e-4;

// dx1/dt
double _dx0_raw = K*Pref/T1_eff - K*omega/T1_eff + K*u_agc/T1_eff - x1/T1_eff;
if (x1 >= PMAX && _dx0_raw > 0.0) _dx0_raw = 0.0;
if (x1 <= PMIN && _dx0_raw < 0.0) _dx0_raw = 0.0;
dxdt[0] = _dx0_raw;
// dx2/dt
double _dx1_raw = x1/T3_eff - x2/T3_eff;
if (x2 >= PMAX && _dx1_raw > 0.0) _dx1_raw = 0.0;
if (x2 <= PMIN && _dx1_raw < 0.0) _dx1_raw = 0.0;
dxdt[1] = _dx1_raw;
// dx3/dt
double _dx2_raw = x2/T7_eff - x3/T7_eff;
if (x3 >= PMAX && _dx2_raw > 0.0) _dx2_raw = 0.0;
if (x3 <= PMIN && _dx2_raw < 0.0) _dx2_raw = 0.0;
dxdt[2] = _dx2_raw;
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
double x3 = x[2];
outputs[0] = K5*x2 + K7*x3;
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
void system_step(double* x, double* dxdt, double t, double* Vd_net, double* Vq_net, double* Vterm_net) {
    int topo_idx = 0;
    // --- Topology Selection (Fault Events) ---
    for(int e = 0; e < N_FAULT_EVENTS; ++e) {
        if(t >= FAULT_TIMES[e]) topo_idx = e + 1;
    }
    const double* Z_real = Z_real_all[topo_idx];
    const double* Z_imag = Z_imag_all[topo_idx];
    const double* I_slack_d = I_slack_d_all[topo_idx];
    const double* I_slack_q = I_slack_q_all[topo_idx];
    double Id_inj[5];
    double Iq_inj[5];

    // dq-frame voltages for exciter inputs (updated inside loop)
    double vd_dq_GENROU_1 = Vd_net[0]*sin(x[0]) - Vq_net[0]*cos(x[0]);
    double vq_dq_GENROU_1 = Vd_net[0]*cos(x[0]) + Vq_net[0]*sin(x[0]);
    double vd_dq_GENROU_2 = Vd_net[1]*sin(x[6]) - Vq_net[1]*cos(x[6]);
    double vq_dq_GENROU_2 = Vd_net[1]*cos(x[6]) + Vq_net[1]*sin(x[6]);
    double vd_dq_GENROU_3 = Vd_net[2]*sin(x[12]) - Vq_net[2]*cos(x[12]);
    double vq_dq_GENROU_3 = Vd_net[2]*cos(x[12]) + Vq_net[2]*sin(x[12]);
    double vd_dq_GENROU_4 = Vd_net[3]*sin(x[18]) - Vq_net[3]*cos(x[18]);
    double vq_dq_GENROU_4 = Vd_net[3]*cos(x[18]) + Vq_net[3]*sin(x[18]);
    double vd_dq_GENROU_5 = Vd_net[4]*sin(x[24]) - Vq_net[4]*cos(x[24]);
    double vq_dq_GENROU_5 = Vd_net[4]*cos(x[24]) + Vq_net[4]*sin(x[24]);

    // --- Iterative Network Solution (Algebraic Loop) ---
    const double ALG_RELAX = 0.30; // under-relaxation for nonlinear current injections
    for(int iter=0; iter<200; ++iter) {
        // Zero Accumulators then add constant slack-bus Norton bias
        for(int i=0; i<N_BUS; ++i) { Id_inj[i]=I_slack_d[i]; Iq_inj[i]=I_slack_q[i]; }

        // --- 1. Compute Outputs & Gather Injections ---
        // GENROU_1 Outputs
        {
        inputs_GENROU_1[0] = Vd_net[0]; // Vd
        inputs_GENROU_1[1] = Vq_net[0]; // Vq
        inputs_GENROU_1[2] = outputs_TGOV1_1[0]; // Tm
        inputs_GENROU_1[3] = outputs_ESST3A_2[0]; // Efd
            step_GENROU_1_out(&x[0], inputs_GENROU_1, outputs_GENROU_1, t);
            Id_inj[0] += outputs_GENROU_1[0];
            Iq_inj[0] += outputs_GENROU_1[1];
        }
        // GENROU_2 Outputs
        {
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_IEEEG1_4[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_1[0]; // Efd
            step_GENROU_2_out(&x[6], inputs_GENROU_2, outputs_GENROU_2, t);
            Id_inj[1] += outputs_GENROU_2[0];
            Iq_inj[1] += outputs_GENROU_2[1];
        }
        // GENROU_3 Outputs
        {
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_IEEEG1_5[0]; // Tm
        inputs_GENROU_3[3] = outputs_ESST3A_3[0]; // Efd
            step_GENROU_3_out(&x[12], inputs_GENROU_3, outputs_GENROU_3, t);
            Id_inj[2] += outputs_GENROU_3[0];
            Iq_inj[2] += outputs_GENROU_3[1];
        }
        // GENROU_4 Outputs
        {
        inputs_GENROU_4[0] = Vd_net[3]; // Vd
        inputs_GENROU_4[1] = Vq_net[3]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_4[3] = outputs_ESST3A_4[0]; // Efd
            step_GENROU_4_out(&x[18], inputs_GENROU_4, outputs_GENROU_4, t);
            Id_inj[3] += outputs_GENROU_4[0];
            Iq_inj[3] += outputs_GENROU_4[1];
        }
        // GENROU_5 Outputs
        {
        inputs_GENROU_5[0] = Vd_net[4]; // Vd
        inputs_GENROU_5[1] = Vq_net[4]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_5[3] = outputs_ESST3A_5[0]; // Efd
            step_GENROU_5_out(&x[24], inputs_GENROU_5, outputs_GENROU_5, t);
            Id_inj[4] += outputs_GENROU_5[0];
            Iq_inj[4] += outputs_GENROU_5[1];
        }
        // ESST3A_2 Outputs
        {
        inputs_ESST3A_2[0] = Vterm_net[0]; // Vterm
        inputs_ESST3A_2[1] = 1.2488587905371222; // Vref
        inputs_ESST3A_2[2] = outputs_GENROU_1[5]; // id_dq
        inputs_ESST3A_2[3] = outputs_GENROU_1[6]; // iq_dq
        inputs_ESST3A_2[4] = vd_dq_GENROU_1; // Vd
        inputs_ESST3A_2[5] = vq_dq_GENROU_1; // Vq
            step_ESST3A_2_out(&x[30], inputs_ESST3A_2, outputs_ESST3A_2, t);
        }
        // ESST3A_3 Outputs
        {
        inputs_ESST3A_3[0] = Vterm_net[2]; // Vterm
        inputs_ESST3A_3[1] = 1.0657855574308523 + outputs_IEEEST_1[0]; // Vref
        inputs_ESST3A_3[2] = outputs_GENROU_3[5]; // id_dq
        inputs_ESST3A_3[3] = outputs_GENROU_3[6]; // iq_dq
        inputs_ESST3A_3[4] = vd_dq_GENROU_3; // Vd
        inputs_ESST3A_3[5] = vq_dq_GENROU_3; // Vq
            step_ESST3A_3_out(&x[35], inputs_ESST3A_3, outputs_ESST3A_3, t);
        }
        // ESST3A_4 Outputs
        {
        inputs_ESST3A_4[0] = Vterm_net[3]; // Vterm
        inputs_ESST3A_4[1] = 1.1509907289647061; // Vref
        inputs_ESST3A_4[2] = outputs_GENROU_4[5]; // id_dq
        inputs_ESST3A_4[3] = outputs_GENROU_4[6]; // iq_dq
        inputs_ESST3A_4[4] = vd_dq_GENROU_4; // Vd
        inputs_ESST3A_4[5] = vq_dq_GENROU_4; // Vq
            step_ESST3A_4_out(&x[40], inputs_ESST3A_4, outputs_ESST3A_4, t);
        }
        // ESST3A_5 Outputs
        {
        inputs_ESST3A_5[0] = Vterm_net[4]; // Vterm
        inputs_ESST3A_5[1] = 1.1661851742181313; // Vref
        inputs_ESST3A_5[2] = outputs_GENROU_5[5]; // id_dq
        inputs_ESST3A_5[3] = outputs_GENROU_5[6]; // iq_dq
        inputs_ESST3A_5[4] = vd_dq_GENROU_5; // Vd
        inputs_ESST3A_5[5] = vq_dq_GENROU_5; // Vq
            step_ESST3A_5_out(&x[45], inputs_ESST3A_5, outputs_ESST3A_5, t);
        }
        // EXST1_1 Outputs
        {
        inputs_EXST1_1[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_1[1] = 1.1700020597521597 + outputs_ST2CUT_3[0]; // Vref
            step_EXST1_1_out(&x[50], inputs_EXST1_1, outputs_EXST1_1, t);
        }
        // TGOV1_1 Outputs
        {
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0405775964342303; // Pref
        inputs_TGOV1_1[2] = 0.0; // u_agc
            step_TGOV1_1_out(&x[54], inputs_TGOV1_1, outputs_TGOV1_1, t);
        }
        // TGOV1_2 Outputs
        {
        inputs_TGOV1_2[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_2[1] = 1.0145386318256848; // Pref
        inputs_TGOV1_2[2] = 0.0; // u_agc
            step_TGOV1_2_out(&x[57], inputs_TGOV1_2, outputs_TGOV1_2, t);
        }
        // TGOV1_3 Outputs
        {
        inputs_TGOV1_3[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_3[1] = 1.0174277669952128; // Pref
        inputs_TGOV1_3[2] = 0.0; // u_agc
            step_TGOV1_3_out(&x[60], inputs_TGOV1_3, outputs_TGOV1_3, t);
        }
        // IEEEG1_4 Outputs
        {
        inputs_IEEEG1_4[0] = outputs_GENROU_2[2]; // omega
        inputs_IEEEG1_4[1] = 1.0200811127647176; // Pref
        inputs_IEEEG1_4[2] = 0.0; // u_agc
            step_IEEEG1_4_out(&x[63], inputs_IEEEG1_4, outputs_IEEEG1_4, t);
        }
        // IEEEG1_5 Outputs
        {
        inputs_IEEEG1_5[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEG1_5[1] = 1.0205131068455713; // Pref
        inputs_IEEEG1_5[2] = 0.0; // u_agc
            step_IEEEG1_5_out(&x[66], inputs_IEEEG1_5, outputs_IEEEG1_5, t);
        }
        // ST2CUT_3 Outputs
        {
        inputs_ST2CUT_3[0] = outputs_GENROU_2[2]; // omega
        inputs_ST2CUT_3[1] = outputs_GENROU_2[3]; // Pe
        inputs_ST2CUT_3[2] = outputs_IEEEG1_4[0]; // Tm
            step_ST2CUT_3_out(&x[69], inputs_ST2CUT_3, outputs_ST2CUT_3, t);
        }
        // IEEEST_1 Outputs
        {
        inputs_IEEEST_1[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEST_1[1] = outputs_GENROU_3[3]; // Pe
        inputs_IEEEST_1[2] = outputs_IEEEG1_5[0]; // Tm
            step_IEEEST_1_out(&x[75], inputs_IEEEST_1, outputs_IEEEST_1, t);
        }

        // --- 2. Solve Network (V = Z * I) ---
        double max_err = 0.0;
        for(int i=0; i<N_BUS; ++i) {
            double Vd_raw = 0.0; double Vq_raw = 0.0;
            for(int j=0; j<N_BUS; ++j) {
                double R = Z_real[i*N_BUS + j];
                double X = Z_imag[i*N_BUS + j];
                double Id = Id_inj[j];
                double Iq = Iq_inj[j];
                Vd_raw += R*Id - X*Iq;
                Vq_raw += R*Iq + X*Id;
            }
            double Vd_new = (1.0 - ALG_RELAX) * Vd_net[i] + ALG_RELAX * Vd_raw;
            double Vq_new = (1.0 - ALG_RELAX) * Vq_net[i] + ALG_RELAX * Vq_raw;
            double err = fabs(Vd_new - Vd_net[i]) + fabs(Vq_new - Vq_net[i]);
            if(err > max_err) max_err = err;
            Vd_net[i] = Vd_new;
            Vq_net[i] = Vq_new;
            Vterm_net[i] = sqrt(Vd_net[i]*Vd_net[i] + Vq_net[i]*Vq_net[i]);
        }
        // 2b: refresh dq voltages/currents from updated V
        vd_dq_GENROU_1 = Vd_net[0]*sin(x[0]) - Vq_net[0]*cos(x[0]);
        vq_dq_GENROU_1 = Vd_net[0]*cos(x[0]) + Vq_net[0]*sin(x[0]);
        {
            double psi_d_pp_x = x[2]*0.177778 + x[3]*(1.0-0.177778);
            double psi_q_pp_x = -x[5]*0.123077 + x[4]*(1.0-0.123077);
            double rhs_d_x = vd_dq_GENROU_1 + psi_q_pp_x;
            double rhs_q_x = vq_dq_GENROU_1 - psi_d_pp_x;
            outputs_GENROU_1[5] = (-0.000000*rhs_d_x - 0.230000*rhs_q_x) / 0.052900;
            outputs_GENROU_1[6] = (0.230000*rhs_d_x + -0.000000*rhs_q_x) / 0.052900;
        }
        vd_dq_GENROU_2 = Vd_net[1]*sin(x[6]) - Vq_net[1]*cos(x[6]);
        vq_dq_GENROU_2 = Vd_net[1]*cos(x[6]) + Vq_net[1]*sin(x[6]);
        {
            double psi_d_pp_x = x[8]*0.288889 + x[9]*(1.0-0.288889);
            double psi_q_pp_x = -x[11]*0.200000 + x[10]*(1.0-0.200000);
            double rhs_d_x = vd_dq_GENROU_2 + psi_q_pp_x;
            double rhs_q_x = vq_dq_GENROU_2 - psi_d_pp_x;
            outputs_GENROU_2[5] = (-0.000000*rhs_d_x - 0.280000*rhs_q_x) / 0.078400;
            outputs_GENROU_2[6] = (0.280000*rhs_d_x + -0.000000*rhs_q_x) / 0.078400;
        }
        vd_dq_GENROU_3 = Vd_net[2]*sin(x[12]) - Vq_net[2]*cos(x[12]);
        vq_dq_GENROU_3 = Vd_net[2]*cos(x[12]) + Vq_net[2]*sin(x[12]);
        {
            double psi_d_pp_x = x[14]*0.422222 + x[15]*(1.0-0.422222);
            double psi_q_pp_x = -x[17]*0.292308 + x[16]*(1.0-0.292308);
            double rhs_d_x = vd_dq_GENROU_3 + psi_q_pp_x;
            double rhs_q_x = vq_dq_GENROU_3 - psi_d_pp_x;
            outputs_GENROU_3[5] = (-0.000000*rhs_d_x - 0.340000*rhs_q_x) / 0.115600;
            outputs_GENROU_3[6] = (0.340000*rhs_d_x + -0.000000*rhs_q_x) / 0.115600;
        }
        vd_dq_GENROU_4 = Vd_net[3]*sin(x[18]) - Vq_net[3]*cos(x[18]);
        vq_dq_GENROU_4 = Vd_net[3]*cos(x[18]) + Vq_net[3]*sin(x[18]);
        {
            double psi_d_pp_x = x[20]*0.288889 + x[21]*(1.0-0.288889);
            double psi_q_pp_x = -x[23]*0.200000 + x[22]*(1.0-0.200000);
            double rhs_d_x = vd_dq_GENROU_4 + psi_q_pp_x;
            double rhs_q_x = vq_dq_GENROU_4 - psi_d_pp_x;
            outputs_GENROU_4[5] = (-0.000000*rhs_d_x - 0.280000*rhs_q_x) / 0.078400;
            outputs_GENROU_4[6] = (0.280000*rhs_d_x + -0.000000*rhs_q_x) / 0.078400;
        }
        vd_dq_GENROU_5 = Vd_net[4]*sin(x[24]) - Vq_net[4]*cos(x[24]);
        vq_dq_GENROU_5 = Vd_net[4]*cos(x[24]) + Vq_net[4]*sin(x[24]);
        {
            double psi_d_pp_x = x[26]*0.422222 + x[27]*(1.0-0.422222);
            double psi_q_pp_x = -x[29]*0.292308 + x[28]*(1.0-0.292308);
            double rhs_d_x = vd_dq_GENROU_5 + psi_q_pp_x;
            double rhs_q_x = vq_dq_GENROU_5 - psi_d_pp_x;
            outputs_GENROU_5[5] = (-0.000000*rhs_d_x - 0.340000*rhs_q_x) / 0.115600;
            outputs_GENROU_5[6] = (0.340000*rhs_d_x + -0.000000*rhs_q_x) / 0.115600;
        }
        if(max_err < 1e-6) break;
    }

    // --- Observer Bus Voltage Recovery ---
    const double* obs_mr = OBS_M_real[topo_idx];
    const double* obs_mi = OBS_M_imag[topo_idx];
    for(int i = 0; i < N_OBS; ++i) {
        double vr = 0.0, vi = 0.0;
        for(int j = 0; j < N_BUS; ++j) {
            double mr = obs_mr[i*N_BUS + j];
            double mi = obs_mi[i*N_BUS + j];
            vr += mr * Vd_net[j] - mi * Vq_net[j];
            vi += mr * Vq_net[j] + mi * Vd_net[j];
        }
        Vd_obs[i] = vr; Vq_obs[i] = vi;
        Vterm_obs[i] = sqrt(vr*vr + vi*vi);
    }

    // --- 3. Compute Dynamics (dxdt) using Solved V ---
    // GENROU_1 Dynamics
    {
        inputs_GENROU_1[0] = Vd_net[0]; // Vd
        inputs_GENROU_1[1] = Vq_net[0]; // Vq
        inputs_GENROU_1[2] = outputs_TGOV1_1[0]; // Tm
        inputs_GENROU_1[3] = outputs_ESST3A_2[0]; // Efd
        step_GENROU_1(&x[0], &dxdt[0], inputs_GENROU_1, outputs_GENROU_1, t);
    }
    // GENROU_2 Dynamics
    {
        inputs_GENROU_2[0] = Vd_net[1]; // Vd
        inputs_GENROU_2[1] = Vq_net[1]; // Vq
        inputs_GENROU_2[2] = outputs_IEEEG1_4[0]; // Tm
        inputs_GENROU_2[3] = outputs_EXST1_1[0]; // Efd
        step_GENROU_2(&x[6], &dxdt[6], inputs_GENROU_2, outputs_GENROU_2, t);
    }
    // GENROU_3 Dynamics
    {
        inputs_GENROU_3[0] = Vd_net[2]; // Vd
        inputs_GENROU_3[1] = Vq_net[2]; // Vq
        inputs_GENROU_3[2] = outputs_IEEEG1_5[0]; // Tm
        inputs_GENROU_3[3] = outputs_ESST3A_3[0]; // Efd
        step_GENROU_3(&x[12], &dxdt[12], inputs_GENROU_3, outputs_GENROU_3, t);
    }
    // GENROU_4 Dynamics
    {
        inputs_GENROU_4[0] = Vd_net[3]; // Vd
        inputs_GENROU_4[1] = Vq_net[3]; // Vq
        inputs_GENROU_4[2] = outputs_TGOV1_2[0]; // Tm
        inputs_GENROU_4[3] = outputs_ESST3A_4[0]; // Efd
        step_GENROU_4(&x[18], &dxdt[18], inputs_GENROU_4, outputs_GENROU_4, t);
    }
    // GENROU_5 Dynamics
    {
        inputs_GENROU_5[0] = Vd_net[4]; // Vd
        inputs_GENROU_5[1] = Vq_net[4]; // Vq
        inputs_GENROU_5[2] = outputs_TGOV1_3[0]; // Tm
        inputs_GENROU_5[3] = outputs_ESST3A_5[0]; // Efd
        step_GENROU_5(&x[24], &dxdt[24], inputs_GENROU_5, outputs_GENROU_5, t);
    }
    // ESST3A_2 Dynamics
    {
        inputs_ESST3A_2[0] = Vterm_net[0]; // Vterm
        inputs_ESST3A_2[1] = 1.2488587905371222; // Vref
        inputs_ESST3A_2[2] = outputs_GENROU_1[5]; // id_dq
        inputs_ESST3A_2[3] = outputs_GENROU_1[6]; // iq_dq
        inputs_ESST3A_2[4] = vd_dq_GENROU_1; // Vd
        inputs_ESST3A_2[5] = vq_dq_GENROU_1; // Vq
        step_ESST3A_2(&x[30], &dxdt[30], inputs_ESST3A_2, outputs_ESST3A_2, t);
    }
    // ESST3A_3 Dynamics
    {
        inputs_ESST3A_3[0] = Vterm_net[2]; // Vterm
        inputs_ESST3A_3[1] = 1.0657855574308523 + outputs_IEEEST_1[0]; // Vref
        inputs_ESST3A_3[2] = outputs_GENROU_3[5]; // id_dq
        inputs_ESST3A_3[3] = outputs_GENROU_3[6]; // iq_dq
        inputs_ESST3A_3[4] = vd_dq_GENROU_3; // Vd
        inputs_ESST3A_3[5] = vq_dq_GENROU_3; // Vq
        step_ESST3A_3(&x[35], &dxdt[35], inputs_ESST3A_3, outputs_ESST3A_3, t);
    }
    // ESST3A_4 Dynamics
    {
        inputs_ESST3A_4[0] = Vterm_net[3]; // Vterm
        inputs_ESST3A_4[1] = 1.1509907289647061; // Vref
        inputs_ESST3A_4[2] = outputs_GENROU_4[5]; // id_dq
        inputs_ESST3A_4[3] = outputs_GENROU_4[6]; // iq_dq
        inputs_ESST3A_4[4] = vd_dq_GENROU_4; // Vd
        inputs_ESST3A_4[5] = vq_dq_GENROU_4; // Vq
        step_ESST3A_4(&x[40], &dxdt[40], inputs_ESST3A_4, outputs_ESST3A_4, t);
    }
    // ESST3A_5 Dynamics
    {
        inputs_ESST3A_5[0] = Vterm_net[4]; // Vterm
        inputs_ESST3A_5[1] = 1.1661851742181313; // Vref
        inputs_ESST3A_5[2] = outputs_GENROU_5[5]; // id_dq
        inputs_ESST3A_5[3] = outputs_GENROU_5[6]; // iq_dq
        inputs_ESST3A_5[4] = vd_dq_GENROU_5; // Vd
        inputs_ESST3A_5[5] = vq_dq_GENROU_5; // Vq
        step_ESST3A_5(&x[45], &dxdt[45], inputs_ESST3A_5, outputs_ESST3A_5, t);
    }
    // EXST1_1 Dynamics
    {
        inputs_EXST1_1[0] = Vterm_net[1]; // Vterm
        inputs_EXST1_1[1] = 1.1700020597521597 + outputs_ST2CUT_3[0]; // Vref
        step_EXST1_1(&x[50], &dxdt[50], inputs_EXST1_1, outputs_EXST1_1, t);
    }
    // TGOV1_1 Dynamics
    {
        inputs_TGOV1_1[0] = outputs_GENROU_1[2]; // omega
        inputs_TGOV1_1[1] = 1.0405775964342303; // Pref
        inputs_TGOV1_1[2] = 0.0; // u_agc
        step_TGOV1_1(&x[54], &dxdt[54], inputs_TGOV1_1, outputs_TGOV1_1, t);
    }
    // TGOV1_2 Dynamics
    {
        inputs_TGOV1_2[0] = outputs_GENROU_4[2]; // omega
        inputs_TGOV1_2[1] = 1.0145386318256848; // Pref
        inputs_TGOV1_2[2] = 0.0; // u_agc
        step_TGOV1_2(&x[57], &dxdt[57], inputs_TGOV1_2, outputs_TGOV1_2, t);
    }
    // TGOV1_3 Dynamics
    {
        inputs_TGOV1_3[0] = outputs_GENROU_5[2]; // omega
        inputs_TGOV1_3[1] = 1.0174277669952128; // Pref
        inputs_TGOV1_3[2] = 0.0; // u_agc
        step_TGOV1_3(&x[60], &dxdt[60], inputs_TGOV1_3, outputs_TGOV1_3, t);
    }
    // IEEEG1_4 Dynamics
    {
        inputs_IEEEG1_4[0] = outputs_GENROU_2[2]; // omega
        inputs_IEEEG1_4[1] = 1.0200811127647176; // Pref
        inputs_IEEEG1_4[2] = 0.0; // u_agc
        step_IEEEG1_4(&x[63], &dxdt[63], inputs_IEEEG1_4, outputs_IEEEG1_4, t);
    }
    // IEEEG1_5 Dynamics
    {
        inputs_IEEEG1_5[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEG1_5[1] = 1.0205131068455713; // Pref
        inputs_IEEEG1_5[2] = 0.0; // u_agc
        step_IEEEG1_5(&x[66], &dxdt[66], inputs_IEEEG1_5, outputs_IEEEG1_5, t);
    }
    // ST2CUT_3 Dynamics
    {
        inputs_ST2CUT_3[0] = outputs_GENROU_2[2]; // omega
        inputs_ST2CUT_3[1] = outputs_GENROU_2[3]; // Pe
        inputs_ST2CUT_3[2] = outputs_IEEEG1_4[0]; // Tm
        step_ST2CUT_3(&x[69], &dxdt[69], inputs_ST2CUT_3, outputs_ST2CUT_3, t);
    }
    // IEEEST_1 Dynamics
    {
        inputs_IEEEST_1[0] = outputs_GENROU_3[2]; // omega
        inputs_IEEEST_1[1] = outputs_GENROU_3[3]; // Pe
        inputs_IEEEST_1[2] = outputs_IEEEG1_5[0]; // Tm
        step_IEEEST_1(&x[75], &dxdt[75], inputs_IEEEST_1, outputs_IEEEST_1, t);
    }

    // --- 4. COI Reference Frame Correction ---
    double coi_total_2H = 8.000000 + 13.000000 + 10.000000 + 10.000000 + 10.000000;
    double coi_omega = (8.000000 * x[1] + 13.000000 * x[7] + 10.000000 * x[13] + 10.000000 * x[19] + 10.000000 * x[25]) / coi_total_2H;
    double omega_b_sys = 2.0 * M_PI * 60.0;
    dxdt[0] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[6] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[12] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[18] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[24] -= omega_b_sys * (coi_omega - 1.0);
    dxdt[82] = omega_b_sys * (coi_omega - 1.0);
}

#include <fstream>
#include <iomanip>
#include <cmath>

int main() {
    double x[N_STATES] = { 0.849671477136, 1.000000000000, 1.046113341196, 0.818867158530, 0.440603341312, -0.704965346100, 0.532921601633, 1.000000000000, 1.005342945299, 0.938298364321, 0.306734422322, -0.474631158751, 0.623888080090, 1.000000000000, 0.889466307237, 0.843950322547, 0.358721399482, -0.532418077126, 0.207638886037, 1.000000000000, 1.189327821301, 1.085466645609, 0.177206888609, -0.274204343427, 0.356223332897, 1.000000000000, 1.154669012135, 1.083021276684, 0.226060349609, -0.335521150473, 1.059485529897, 0.189373260640, 1.893732606400, 0.554380759076, 3.215999538439, 1.008945851178, 0.056839706252, 1.136794125049, 0.299549408587, 3.670076344584, 1.069608327838, 0.081382401127, 1.627648022532, 0.390228812683, 4.046054376742, 1.089644754008, 0.076540420210, 1.530808404193, 0.363798934889, 4.082888168248, 1.044340015876, 0.125662043876, 1.256620438757, 1.256620438757, 0.811551928685, 0.811551928685, 0.000000000000, 0.290772636514, 0.290772636514, 0.000000000000, 0.348555339904, 0.348555339904, 0.000000000000, 0.401622255294, 0.401622255294, 0.401622255294, 0.410262136911, 0.410262136911, 0.410262136911, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.410261297800, 0.410261297800, 0.410261297800, 0.410261297800, 0.410261297800, 0.410261297800, -0.820522595600, 0.000000000000 };
    double dxdt[N_STATES];
    
    // Solver stage buffers (used by RK2, RK4, and SDIRK-2)
    double k1[N_STATES], k2[N_STATES], k3[N_STATES], k4[N_STATES];
    double x_temp[N_STATES];
    double res[N_STATES];   // Newton residual (SDIRK-2)
    
    // Initial network voltages from power flow (Vd = V*cos(theta), Vq = V*sin(theta))
    double Vd[N_BUS] = { 1.059193970620, 1.044339904412, 1.008620736022, 1.065881106036, 1.089634766797 };
    double Vq[N_BUS] = { 0.024871773131, -0.001020322704, -0.025625929439, -0.089219189552, -0.004731662604 };
    double Vt[N_BUS] = {0};
    
    // Output File
    std::ofstream outfile("simulation_results.csv");
    outfile << "t,Vterm_GENROU_1 (pu),Vterm_GENROU_2 (pu),Vterm_GENROU_3 (pu),Vterm_GENROU_4 (pu),Vterm_GENROU_5 (pu),Vterm_Bus4 (pu),V_Re_Bus4 (pu),V_Im_Bus4 (pu),Vterm_Bus5 (pu),V_Re_Bus5 (pu),V_Im_Bus5 (pu),Vterm_Bus7 (pu),V_Re_Bus7 (pu),V_Im_Bus7 (pu),Vterm_Bus9 (pu),V_Re_Bus9 (pu),V_Im_Bus9 (pu),Vterm_Bus10 (pu),V_Re_Bus10 (pu),V_Im_Bus10 (pu),Vterm_Bus11 (pu),V_Re_Bus11 (pu),V_Im_Bus11 (pu),Vterm_Bus12 (pu),V_Re_Bus12 (pu),V_Im_Bus12 (pu),Vterm_Bus13 (pu),V_Re_Bus13 (pu),V_Im_Bus13 (pu),Vterm_Bus14 (pu),V_Re_Bus14 (pu),V_Im_Bus14 (pu),It_Re_GENROU_1,It_Im_GENROU_1,Pe_GENROU_1,Qe_GENROU_1,It_Re_GENROU_2,It_Im_GENROU_2,Pe_GENROU_2,Qe_GENROU_2,It_Re_GENROU_3,It_Im_GENROU_3,Pe_GENROU_3,Qe_GENROU_3,It_Re_GENROU_4,It_Im_GENROU_4,Pe_GENROU_4,Qe_GENROU_4,It_Re_GENROU_5,It_Im_GENROU_5,Pe_GENROU_5,Qe_GENROU_5,GENROU_1_delta,GENROU_1_omega,GENROU_1_E_q_prime,GENROU_1_psi_d,GENROU_1_E_d_prime,GENROU_1_psi_q,GENROU_1_delta_deg (deg),GENROU_1_Te (pu),GENROU_1_Tm_in (pu),GENROU_1_Pe (pu),GENROU_1_Qe (pu),GENROU_1_V_term (pu),GENROU_1_Eq_p (pu),GENROU_1_omega (pu),GENROU_1_H_total (pu),GENROU_2_delta,GENROU_2_omega,GENROU_2_E_q_prime,GENROU_2_psi_d,GENROU_2_E_d_prime,GENROU_2_psi_q,GENROU_2_delta_deg (deg),GENROU_2_Te (pu),GENROU_2_Tm_in (pu),GENROU_2_Pe (pu),GENROU_2_Qe (pu),GENROU_2_V_term (pu),GENROU_2_Eq_p (pu),GENROU_2_omega (pu),GENROU_2_H_total (pu),GENROU_3_delta,GENROU_3_omega,GENROU_3_E_q_prime,GENROU_3_psi_d,GENROU_3_E_d_prime,GENROU_3_psi_q,GENROU_3_delta_deg (deg),GENROU_3_Te (pu),GENROU_3_Tm_in (pu),GENROU_3_Pe (pu),GENROU_3_Qe (pu),GENROU_3_V_term (pu),GENROU_3_Eq_p (pu),GENROU_3_omega (pu),GENROU_3_H_total (pu),GENROU_4_delta,GENROU_4_omega,GENROU_4_E_q_prime,GENROU_4_psi_d,GENROU_4_E_d_prime,GENROU_4_psi_q,GENROU_4_delta_deg (deg),GENROU_4_Te (pu),GENROU_4_Tm_in (pu),GENROU_4_Pe (pu),GENROU_4_Qe (pu),GENROU_4_V_term (pu),GENROU_4_Eq_p (pu),GENROU_4_omega (pu),GENROU_4_H_total (pu),GENROU_5_delta,GENROU_5_omega,GENROU_5_E_q_prime,GENROU_5_psi_d,GENROU_5_E_d_prime,GENROU_5_psi_q,GENROU_5_delta_deg (deg),GENROU_5_Te (pu),GENROU_5_Tm_in (pu),GENROU_5_Pe (pu),GENROU_5_Qe (pu),GENROU_5_V_term (pu),GENROU_5_Eq_p (pu),GENROU_5_omega (pu),GENROU_5_H_total (pu),ESST3A_2_Vm,ESST3A_2_LLx,ESST3A_2_Vr,ESST3A_2_VM,ESST3A_2_VB,ESST3A_2_Efd (pu),ESST3A_2_H_exc (pu),ESST3A_2_Efd_out,ESST3A_3_Vm,ESST3A_3_LLx,ESST3A_3_Vr,ESST3A_3_VM,ESST3A_3_VB,ESST3A_3_Efd (pu),ESST3A_3_H_exc (pu),ESST3A_3_Efd_out,ESST3A_4_Vm,ESST3A_4_LLx,ESST3A_4_Vr,ESST3A_4_VM,ESST3A_4_VB,ESST3A_4_Efd (pu),ESST3A_4_H_exc (pu),ESST3A_4_Efd_out,ESST3A_5_Vm,ESST3A_5_LLx,ESST3A_5_Vr,ESST3A_5_VM,ESST3A_5_VB,ESST3A_5_Efd (pu),ESST3A_5_H_exc (pu),ESST3A_5_Efd_out,EXST1_1_Vm,EXST1_1_LLx,EXST1_1_Vr,EXST1_1_Vf,EXST1_1_Efd (pu),EXST1_1_H_exc (pu),EXST1_1_Efd_out,TGOV1_1_x1,TGOV1_1_x2,TGOV1_1_xi,TGOV1_1_Tm (pu),TGOV1_1_Valve (pu),TGOV1_1_xi (pu),TGOV1_2_x1,TGOV1_2_x2,TGOV1_2_xi,TGOV1_2_Tm (pu),TGOV1_2_Valve (pu),TGOV1_2_xi (pu),TGOV1_3_x1,TGOV1_3_x2,TGOV1_3_xi,TGOV1_3_Tm (pu),TGOV1_3_Valve (pu),TGOV1_3_xi (pu),IEEEG1_4_x1,IEEEG1_4_x2,IEEEG1_4_x3,IEEEG1_4_Tm (pu),IEEEG1_4_Valve (pu),IEEEG1_4_H_gov (pu),IEEEG1_5_x1,IEEEG1_5_x2,IEEEG1_5_x3,IEEEG1_5_Tm (pu),IEEEG1_5_Valve (pu),IEEEG1_5_H_gov (pu),ST2CUT_3_xl1,ST2CUT_3_xl2,ST2CUT_3_xwo,ST2CUT_3_xll1,ST2CUT_3_xll2,ST2CUT_3_xll3,ST2CUT_3_Vss (pu),IEEEST_1_xf1,IEEEST_1_xf2,IEEEST_1_xll1,IEEEST_1_xll2,IEEEST_1_xll3,IEEEST_1_xll4,IEEEST_1_xwo,IEEEST_1_Vss (pu),IEEEST_1_H_pss (pu),delta_COI (deg),GENROU_1_delta_abs_deg (deg),GENROU_2_delta_abs_deg (deg),GENROU_3_delta_abs_deg (deg),GENROU_4_delta_abs_deg (deg),GENROU_5_delta_abs_deg (deg)" << std::endl;
    outfile << std::scientific << std::setprecision(8);

    double t = 0.0;
    const double dt = 0.0005;
    const int steps = 30000;
    const int log_every = 10; // log every 10 steps = every 0.0050 s
    
    
    // --- DIAGNOSTICS: Check Initial Derivatives ---
    std::cout << "\n[Diagnostics] Checking Equilibrium Quality..." << std::endl;
    system_step(x, dxdt, 0.0, Vd, Vq, Vt);
    
    bool stable = true;
    double max_dxdt_init = 0.0;
    int max_idx = -1;
    for(int i=0; i<N_STATES; ++i) {
        if(fabs(dxdt[i]) > 0.05) {
            std::cout << "  WARNING: State[" << i << "] large dxdt=" << dxdt[i] << std::endl;
            stable = false;
        }
        if(fabs(dxdt[i]) > max_dxdt_init) { max_dxdt_init = fabs(dxdt[i]); max_idx = i; }
    }
    std::cout << "  Max |dxdt| = " << max_dxdt_init << " at state[" << max_idx << "]" << std::endl;
    if(stable) std::cout << "  Initial equilibrium looks good (all |dx/dt| < 0.05)." << std::endl;
    else std::cout << "  SYSTEM NOT IN EQUILIBRIUM. Expect transients." << std::endl;
    std::cout << "  GENROU_1: dw/dt=" << dxdt[1] << " Tm-Te=" << dxdt[1]*8.0 << std::endl;
    std::cout << "  GENROU_2: dw/dt=" << dxdt[7] << " Tm-Te=" << dxdt[7]*13.0 << std::endl;
    std::cout << "  GENROU_3: dw/dt=" << dxdt[13] << " Tm-Te=" << dxdt[13]*10.0 << std::endl;
    std::cout << "  GENROU_4: dw/dt=" << dxdt[19] << " Tm-Te=" << dxdt[19]*10.0 << std::endl;
    std::cout << "  GENROU_5: dw/dt=" << dxdt[25] << " Tm-Te=" << dxdt[25]*10.0 << std::endl;
    std::cout << "  V after algebraic solve:" << std::endl;
    for(int b=0; b<N_BUS; ++b)
        std::cout << "    Bus[" << b << "]: Vd=" << Vd[b] << " Vq=" << Vq[b]
                  << " |V|=" << Vt[b] << std::endl;
    std::cout << "------------------------------------------\n" << std::endl;

    
    std::cout << "Starting Simulation (JIT, T=15.0s, dt=0.0005)..." << std::endl;
    
    for(int i=0; i<steps; ++i) {
        // Log (Decimated)
        if (i % log_every == 0) {
            outfile << (t) << "," << (Vt[0]) << "," << (Vt[1]) << "," << (Vt[2]) << "," << (Vt[3]) << "," << (Vt[4]) << "," << (Vterm_obs[0]) << "," << (Vd_obs[0]) << "," << (Vq_obs[0]) << "," << (Vterm_obs[1]) << "," << (Vd_obs[1]) << "," << (Vq_obs[1]) << "," << (Vterm_obs[2]) << "," << (Vd_obs[2]) << "," << (Vq_obs[2]) << "," << (Vterm_obs[3]) << "," << (Vd_obs[3]) << "," << (Vq_obs[3]) << "," << (Vterm_obs[4]) << "," << (Vd_obs[4]) << "," << (Vq_obs[4]) << "," << (Vterm_obs[5]) << "," << (Vd_obs[5]) << "," << (Vq_obs[5]) << "," << (Vterm_obs[6]) << "," << (Vd_obs[6]) << "," << (Vq_obs[6]) << "," << (Vterm_obs[7]) << "," << (Vd_obs[7]) << "," << (Vq_obs[7]) << "," << (Vterm_obs[8]) << "," << (Vd_obs[8]) << "," << (Vq_obs[8]) << "," << (outputs_GENROU_1[7]) << "," << (outputs_GENROU_1[8]) << "," << (outputs_GENROU_1[3]) << "," << (outputs_GENROU_1[4]) << "," << (outputs_GENROU_2[7]) << "," << (outputs_GENROU_2[8]) << "," << (outputs_GENROU_2[3]) << "," << (outputs_GENROU_2[4]) << "," << (outputs_GENROU_3[7]) << "," << (outputs_GENROU_3[8]) << "," << (outputs_GENROU_3[3]) << "," << (outputs_GENROU_3[4]) << "," << (outputs_GENROU_4[7]) << "," << (outputs_GENROU_4[8]) << "," << (outputs_GENROU_4[3]) << "," << (outputs_GENROU_4[4]) << "," << (outputs_GENROU_5[7]) << "," << (outputs_GENROU_5[8]) << "," << (outputs_GENROU_5[3]) << "," << (outputs_GENROU_5[4]) << "," << (x[0]) << "," << (x[1]) << "," << (x[2]) << "," << (x[3]) << "," << (x[4]) << "," << (x[5]) << "," << (x[0] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_1[3]) << "," << (inputs_GENROU_1[2]) << "," << (outputs_GENROU_1[3]) << "," << (outputs_GENROU_1[4]) << "," << (sqrt(inputs_GENROU_1[0]*inputs_GENROU_1[0] + inputs_GENROU_1[1]*inputs_GENROU_1[1])) << "," << (x[2]) << "," << (x[1]) << "," << ((4.0*((x[1])*(x[1])) - 8.0*x[1] + 0.30303030303030298*((x[2])*(x[2])) + 1.3513513513513513*((x[3])*(x[3])) + 0.3125*((x[4])*(x[4])) + 0.8771929824561403*((x[5])*(x[5])) + 4.0)) << "," << (x[6]) << "," << (x[7]) << "," << (x[8]) << "," << (x[9]) << "," << (x[10]) << "," << (x[11]) << "," << (x[6] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_2[3]) << "," << (inputs_GENROU_2[2]) << "," << (outputs_GENROU_2[3]) << "," << (outputs_GENROU_2[4]) << "," << (sqrt(inputs_GENROU_2[0]*inputs_GENROU_2[0] + inputs_GENROU_2[1]*inputs_GENROU_2[1])) << "," << (x[8]) << "," << (x[7]) << "," << ((6.5*((x[7])*(x[7])) - 13.0*x[7] + 0.30303030303030298*((x[8])*(x[8])) + 1.5625000000000002*((x[9])*(x[9])) + 0.3125*((x[10])*(x[10])) + 0.96153846153846145*((x[11])*(x[11])) + 6.5)) << "," << (x[12]) << "," << (x[13]) << "," << (x[14]) << "," << (x[15]) << "," << (x[16]) << "," << (x[17]) << "," << (x[12] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_3[3]) << "," << (inputs_GENROU_3[2]) << "," << (outputs_GENROU_3[3]) << "," << (outputs_GENROU_3[4]) << "," << (sqrt(inputs_GENROU_3[0]*inputs_GENROU_3[0] + inputs_GENROU_3[1]*inputs_GENROU_3[1])) << "," << (x[14]) << "," << (x[13]) << "," << ((5.0*((x[13])*(x[13])) - 10.0*x[13] + 0.30303030303030298*((x[14])*(x[14])) + 1.9230769230769234*((x[15])*(x[15])) + 0.3125*((x[16])*(x[16])) + 1.0869565217391304*((x[17])*(x[17])) + 5.0)) << "," << (x[18]) << "," << (x[19]) << "," << (x[20]) << "," << (x[21]) << "," << (x[22]) << "," << (x[23]) << "," << (x[18] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_4[3]) << "," << (inputs_GENROU_4[2]) << "," << (outputs_GENROU_4[3]) << "," << (outputs_GENROU_4[4]) << "," << (sqrt(inputs_GENROU_4[0]*inputs_GENROU_4[0] + inputs_GENROU_4[1]*inputs_GENROU_4[1])) << "," << (x[20]) << "," << (x[19]) << "," << ((5.0*((x[19])*(x[19])) - 10.0*x[19] + 0.30303030303030298*((x[20])*(x[20])) + 1.5625000000000002*((x[21])*(x[21])) + 0.3125*((x[22])*(x[22])) + 0.96153846153846145*((x[23])*(x[23])) + 5.0)) << "," << (x[24]) << "," << (x[25]) << "," << (x[26]) << "," << (x[27]) << "," << (x[28]) << "," << (x[29]) << "," << (x[24] * 180.0 / 3.14159265359) << "," << (outputs_GENROU_5[3]) << "," << (inputs_GENROU_5[2]) << "," << (outputs_GENROU_5[3]) << "," << (outputs_GENROU_5[4]) << "," << (sqrt(inputs_GENROU_5[0]*inputs_GENROU_5[0] + inputs_GENROU_5[1]*inputs_GENROU_5[1])) << "," << (x[26]) << "," << (x[25]) << "," << ((5.0*((x[25])*(x[25])) - 10.0*x[25] + 0.30303030303030298*((x[26])*(x[26])) + 1.9230769230769234*((x[27])*(x[27])) + 0.3125*((x[28])*(x[28])) + 1.0869565217391304*((x[29])*(x[29])) + 5.0)) << "," << (x[30]) << "," << (x[31]) << "," << (x[32]) << "," << (x[33]) << "," << (x[34]) << "," << (x[34]*x[33]) << "," << ((0.5*(x[30]*x[30]+x[31]*x[31]+x[32]*x[32]+x[33]*x[33]+x[34]*x[34]))) << "," << (fmax(0.000000, fmin(5.000000, (x[34]*x[33])))) << "," << (x[35]) << "," << (x[36]) << "," << (x[37]) << "," << (x[38]) << "," << (x[39]) << "," << (x[39]*x[38]) << "," << ((0.5*(x[35]*x[35]+x[36]*x[36]+x[37]*x[37]+x[38]*x[38]+x[39]*x[39]))) << "," << (fmax(0.000000, fmin(5.000000, (x[39]*x[38])))) << "," << (x[40]) << "," << (x[41]) << "," << (x[42]) << "," << (x[43]) << "," << (x[44]) << "," << (x[44]*x[43]) << "," << ((0.5*(x[40]*x[40]+x[41]*x[41]+x[42]*x[42]+x[43]*x[43]+x[44]*x[44]))) << "," << (fmax(0.000000, fmin(5.000000, (x[44]*x[43])))) << "," << (x[45]) << "," << (x[46]) << "," << (x[47]) << "," << (x[48]) << "," << (x[49]) << "," << (x[49]*x[48]) << "," << ((0.5*(x[45]*x[45]+x[46]*x[46]+x[47]*x[47]+x[48]*x[48]+x[49]*x[49]))) << "," << (fmax(0.000000, fmin(5.000000, (x[49]*x[48])))) << "," << (x[50]) << "," << (x[51]) << "," << (x[52]) << "," << (x[53]) << "," << (x[52]) << "," << ((0.5*(x[50]*x[50]+x[51]*x[51]+x[52]*x[52]+x[53]*x[53]))) << "," << (x[52]) << "," << (x[54]) << "," << (x[55]) << "," << (x[56]) << "," << (outputs_TGOV1_1[0]) << "," << (x[54]) << "," << (x[56]) << "," << (x[57]) << "," << (x[58]) << "," << (x[59]) << "," << (outputs_TGOV1_2[0]) << "," << (x[57]) << "," << (x[59]) << "," << (x[60]) << "," << (x[61]) << "," << (x[62]) << "," << (outputs_TGOV1_3[0]) << "," << (x[60]) << "," << (x[62]) << "," << (x[63]) << "," << (x[64]) << "," << (x[65]) << "," << (0.3*x[64] + 0.7*x[65]) << "," << (x[63]) << "," << ((0.5*(x[63]*x[63]+x[64]*x[64]+x[65]*x[65]))) << "," << (x[66]) << "," << (x[67]) << "," << (x[68]) << "," << (0.3*x[67] + 0.7*x[68]) << "," << (x[66]) << "," << ((0.5*(x[66]*x[66]+x[67]*x[67]+x[68]*x[68]))) << "," << (x[69]) << "," << (x[70]) << "," << (x[71]) << "," << (x[72]) << "," << (x[73]) << "," << (x[74]) << "," << (outputs_ST2CUT_3[0]) << "," << (x[75]) << "," << (x[76]) << "," << (x[77]) << "," << (x[78]) << "," << (x[79]) << "," << (x[80]) << "," << (x[81]) << "," << (outputs_IEEEST_1[0]) << "," << (0.5*(x[75]*x[75]+x[76]*x[76]+x[77]*x[77]+x[78]*x[78]+x[79]*x[79]+x[80]*x[80]+x[81]*x[81])) << "," << (x[82] * 180.0 / M_PI) << "," << ((x[0] + x[82]) * 180.0 / M_PI) << "," << ((x[6] + x[82]) * 180.0 / M_PI) << "," << ((x[12] + x[82]) * 180.0 / M_PI) << "," << ((x[18] + x[82]) * 180.0 / M_PI) << "," << ((x[24] + x[82]) * 180.0 / M_PI) << std::endl;
        }

        
        // RK4 Stage 1
        system_step(x, k1, t, Vd, Vq, Vt);
        
        // RK4 Stage 2
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + 0.5 * dt * k1[j];
        system_step(x_temp, k2, t + 0.5*dt, Vd, Vq, Vt);
        
        // RK4 Stage 3
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + 0.5 * dt * k2[j];
        system_step(x_temp, k3, t + 0.5*dt, Vd, Vq, Vt);
        
        // RK4 Stage 4
        for(int j=0; j<N_STATES; ++j) x_temp[j] = x[j] + dt * k3[j];
        system_step(x_temp, k4, t + dt, Vd, Vq, Vt);
        
        // Final Update
        for(int j=0; j<N_STATES; ++j) x[j] += (dt/6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);

        
        t += dt;
        
        // Progress & safety check every 10 seconds
        if (i % (int)(10.0 / dt) == 0) {
             // Recompute derivative at current state (dxdt not updated by RK4/RK2)
             system_step(x, dxdt, t, Vd, Vq, Vt);
             double max_d = 0.0;
             int max_d_idx = -1;
             for(int k=0; k<N_STATES; ++k) {
                 if(fabs(dxdt[k]) > max_d) {
                     max_d = fabs(dxdt[k]);
                     max_d_idx = k;
                 }
             }
             std::cout << "t=" << t << " max_d=" << max_d << " at state[" << max_d_idx << "] Vterm[0]=" << Vt[0] << std::endl;
             
             if(Vt[0] > 5.0 || std::isnan(Vt[0])) {
                 std::cout << "Stability Limit Reached. Stopping." << std::endl;
                 break;
             }
        }
    }
    
    outfile.close();
    std::cout << "Done. Results in simulation_results.csv" << std::endl;
    return 0;
}
