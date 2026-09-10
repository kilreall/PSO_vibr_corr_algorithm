# func to test kalman filter without vibrations on ceration data with comfortable initializtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import savgol_filter

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A + B*np.cos(Phi)

    return np.array([P])


def init_values(alp, P_exp, method, poi, T, sigma_P, pcov):

    alp_init = alp.copy()
    P_init = P_exp.copy()
    idx = np.argsort(alp_init)
    alp_init = alp_init[idx]
    P_init  = P_init[idx]

    if method == "fit":
        p0 = [ (np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2, (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
        lb = [-1.1, 0, 0]
        ub = [1.1, 1.1, 2*np.pi]
        popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
        A0, B0, ph0 = popt
        sigma_P = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0)) # for real data
        print(A0, B0, ph0)

        # # test init fit
        # plt.figure()
        # plt.plot(alp_init, P_init)
        # plt.plot(alp_init, model(alp_init, A0, B0, ph0))


        return A0, B0, ph0, pcov, sigma_P

    elif method == "eval":
        A0 = (np.max(P_init[:poi]) + np.min(P_init[:poi]))/2
        B0 = (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2
        lm = 780e-9
        keff = 4*np.pi/lm
        g = 9.8125
        ph0 = keff*g*T**2 % (2*np.pi)
        return A0, B0, ph0, pcov, sigma_P

def HJacobian(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    H = np.array([
        [1.0,
         np.cos(Phi),
         B*np.sin(Phi)]
    ])

    return H


def kalmanFit_EKF(alp, P_exp, T, init_method, poi, Q, P_cov0, sigma_P):

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

    A[0], B[0], ph[0], ekf.P, sigma_P = init_values(alp, P_exp, init_method, poi, T, sigma_P, P_cov0)

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

    # measurement noise
    ekf.R = np.array([
        [sigma_P**2]
    ])

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

def kalman_graphs(alp, P_m, A, B, ph, P_cov, e, en):

    # main graphic
    plt.figure()
    plt.plot(P_exp, label="data")#, marker="o")
    plt.plot(model(alp, A, B, ph), label="kalman")
    #plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
    plt.legend()

    # additional graphics
    plt.title("koef dynamics")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(A)
    axs[0].set_title('A')

    # Второй график
    axs[1].plot(B)
    axs[1].set_title('B')

    # Третий график
    lm = 780e-9
    keff = 4*np.pi/lm
    alp_min = alp[np.argmin(P_exp)]
    ph_target = 2 * np.pi * alp_min * T**2 - np.pi   # cos = -1 при B > 0 pi возникает из-за полож B, лучше не менять
    M = np.round( (ph_target - ph) / (2 * np.pi) )
    ph = ph + 2 * np.pi * M
    #axs[2].plot(ph) 
    axs[2].plot(ph/keff/T/T*1e5, label="kalman")

    # for comparison
    g_comp = np.load("g_slide_delay800_w5.npy")
    dots = np.arange(0, len(alp), 5)
    axs[2].plot(dots, g_comp*1e5*-1, label="comparison")
    # --- Savitzky-Golay для сравнения ---
    # окно должно быть нечётным и меньше длины массива
    window_length = 21          # можно менять (11, 21, 51, 101...)
    polyorder = 3
    g_savgol = savgol_filter(g_comp*-1, window_length=window_length, polyorder=polyorder)
    axs[2].plot(dots, g_savgol*1e5, label="savgol")

    axs[2].set_title(r'$g$')
    plt.legend()

    
    # uncerteinty 
    fig1, axss = plt.subplots(1, 3, figsize=(14, 4))
    plt.title("Standart deviations")
    axss[0].plot(np.sqrt(P_cov[:, 0, 0]))
    axss[0].set_title('dA')

    # Второй график
    axss[1].plot(np.sqrt(P_cov[:, 1, 1]))
    axss[1].set_title('dB')

    # Третий график
    lm = 780e-9
    keff = 4*np.pi/lm
    axss[2].plot(np.sqrt(P_cov[:, 2, 2])/keff/T/T*1e5)  # или axs[1, 0]
    axss[2].set_title('$dg$')

    # Q finder
    # plt.figure()
    # plt.title("Q find params")
    # plt.plot(en, label="normalized innovation")
    print(f"mean norm e ={np.mean(en)}")
    print(f"std norm e ={np.std(en)}")


data = np.load("data_delay_800.npy")

T = 10e-3
# kalman fit
alp = data[0]*1e6 # 1e6 из-за особенности data
P_exp = data[1]
init_method = "fit"
poi = len(alp)
P_cov0 = np.diag([1,1,1])*1e-3
sigma_P = 1e-4

dg_model = 0.1 # mGal
dA_model = 1e-3
dB_model = 1e-3

lm = 780e-9
keff = 4*np.pi/lm
dph_model = dg_model*keff*T*T/1e5
print(f"dph_model = {dph_model}")
Q = np.diag([dA_model**2, dB_model**2, dph_model**2])
P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(alp, P_exp, T, init_method, poi, Q, P_cov0, sigma_P)

kalman_graphs(alp, P_m, A, B, ph, P_cov, e, en)


plt.show()


