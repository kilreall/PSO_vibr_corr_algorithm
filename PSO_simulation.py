import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter

# accelerometer sensetivity function
def fa(t):
    if 0 < t <= T+2*ty:
        return t/(T+2*ty)**2
    elif T+2*ty < t <= 2*T+4*ty:
        return (2*(T+2*ty)-t)/(T+2*ty)**2
    else:
        return 0
vfunc = np.vectorize(fa)

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)


def init_values(alp, P_exp, poi):

    alp_init = alp.copy()[:poi]
    P_init = P_exp.copy()[:poi]
    # idx = np.argsort(alp_init)
    # alp_init = alp_init[idx]
    # P_init  = P_init[idx]


    p0 = [ (np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2, (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]
    popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt
    sigma_A = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0)) # for real data

    # test init fit
    # plt.figure()
    # plt.scatter(alp_init, P_init)
    # plt.scatter(np.sort(alp_init), model(np.sort(alp_init), A0, B0, ph0))


    return A0, B0, ph0, pcov, sigma_A

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A - B*np.cos(Phi)

    return np.array([P])

def HJacobian(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    H = np.array([
        [1.0,
         -np.cos(Phi),
         -B*np.sin(Phi)]
    ])

    return H


def kalmanFit_EKF(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0):

    N = len(alp)

    # --------------------------------------------------
    # Output arrays
    # --------------------------------------------------

    A = np.zeros(N)
    B = np.zeros(N)
    ph = np.zeros(N)
    P_m = np.zeros(N)
    e = np.zeros(N)
    en = np.zeros(N)
    P_cov = np.zeros((N, 3, 3))

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    # create EKF
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)

    A[0], B[0], ph[0], ekf.P = A0, B0, ph0, P_cov0

    # initial covariance
    P_cov[0] = ekf.P

    P_m[0] = model(
        alp[0],
        A[0],
        B[0],
        ph[0]
    )

    # state = [A, B, phi]
    ekf.x = np.array([A[0], B[0], ph[0]])

    # process noise
    ekf.Q = Q


    # --------------------------------------------------
    # Main cycle
    # --------------------------------------------------

    for i in range(1, N):

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        ekf.predict()

        # predicted measurement
        P_m[i] = Hx(
            ekf.x,
            alp[i],
            T
        )[0]

        # innovation
        e[i] = P_exp[i] - P_m[i]

        # ----------------------------------------------
        # Measurement update
        # ----------------------------------------------

        # measurement noise
        ekf.R = np.array([
            [sigma_A**2 + B[i-1]**2*np.sin(2*np.pi*alp[i]*T*T - ph[i-1])**2*sigma_ph**2]
        ])

        ekf.update(
            np.array([P_exp[i]]),
            HJacobian,
            Hx,
            args=(alp[i], T),
            hx_args=(alp[i], T)
        )

        # normalized innovation
        S_i = ekf.S[0, 0]
        en[i]= e[i]/np.sqrt(S_i)

        # ----------------------------------------------
        # Save state
        # ----------------------------------------------

        A[i] = ekf.x[0]
        B[i] = ekf.x[1]
        ph[i] = ekf.x[2]
    
        # covariance
        P_cov[i] = ekf.P

    return P_m, A, B, ph, P_cov, e, en

# constants
lm = 780e-9
keff = 4*np.pi/lm

# physical params
T = 10e-3
ty = 20e-6


# smimulation data
N_sim = 10000

g0 = 9.8101507
Dg = 300*1e-8
g_sim = np.zeros(N_sim)
g_sim[-1] = g0
drift_corr = 3000
Dg_drift = 30e-8
theta_drift   = 1 - np.exp(-1.0/drift_corr)
sigma_g_drift = Dg_drift * np.sqrt(theta_drift*(2 - theta_drift))
print(f"sigma_g_drift = {sigma_g_drift}")

alp_min = keff*g0/2/np.pi - 1/5/T/T
alp_max = keff*g0/2/np.pi + 1/5/T/T
alp_amount = 201
alp_start = np.linspace(alp_min, alp_max, alp_amount)
alp = np.zeros(N_sim)


Ph = np.zeros(N_sim)
F_vib = np.zeros(N_sim)
sigma_ph_vibr = 1e-4

A_sim = np.zeros(N_sim)
A0_sim = 0.15
A_sim[-1] = A0_sim
dA_sim = 1e-4


B_sim = np.zeros(N_sim)
B0_sim = 0.21
B_sim[-1] = B0_sim
dB_sim = 1e-4

ph_sim = np.zeros(N_sim)
dph_sim = 1e-6


P_sim = np.zeros(N_sim)
P_sim_noise = np.zeros(N_sim)
sigma_A_sim = 3e-3

for i in range(N_sim):
    g_sim[i] = g0 + (g_sim[i-1] - g0)*(1 - theta_drift) + sigma_g_drift*np.random.normal()
    F_vib[i] = np.random.uniform(-np.pi/12, np.pi/12)
    alp[i] = alp_start[i%alp_amount] - F_vib[i]/2/np.pi/T/T
    A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
    B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
    ph_sim[i] = keff*g_sim[i]*T*T
    Ph[i] = 2*np.pi*alp[i]*T*T - ph_sim[i]
    P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])
    F_vib[i] += np.random.normal(0, sigma_ph_vibr)
    alp[i] = alp_start[i%alp_amount] - F_vib[i]/2/np.pi/T/T
    P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)
    
# sim show
# plt.scatter(alp, P_sim_noise)



# kalman fit



dA_model = np.sqrt(DA_sim**2 + dA_sim**2)
dB_model = dB_sim
dph_model = sigma_g_drift*keff*T*T #np.std(ph_sim[1:] - np.roll(ph_sim, 1)[1:]) * 2# np.sqrt(dph_sim**2 + Dph_sim**2)
Q = np.diag([dA_model**2, dB_model**2, dph_model**2])

# initial values
poi = alp_amount
A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
sigma_ph = np.sqrt(sigma_ph_vibr**2 + dph_sim**2*0)
sigma_A = sigma_A_sim # only for sim
P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(alp, P_sim_noise, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0)


