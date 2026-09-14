# func to test kalman filter without vibrations on ceration data with comfortable initializtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import savgol_filter

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A - B*np.cos(Phi)

    return np.array([P])


def init_values(alp, P_exp, method, poi, T, sigma_P, pcov):

    alp_init = alp.copy()
    P_init = P_exp.copy()
    # idx = np.argsort(alp_init)
    # alp_init = alp_init[idx]
    # P_init  = P_init[idx]

    if method == "fit":
        p0 = [ (np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2, (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
        lb = [-1.1, 0, 0]
        ub = [1.1, 1.1, 2*np.pi]
        popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
        A0, B0, ph0 = popt
        sigma_P = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0)) # for real data

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
         -np.cos(Phi),
         -B*np.sin(Phi)]
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

    sigma_P = dP_sim # only for simulation


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
            [sigma_P**2 + B[i-1]**2*np.sin(2*np.pi*alp[i]*T*T - ph[i-1])**2*sigma_ph_vibr**2]
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

def windowFit(alp, P_exp, T, window):

    lm=780e-9
    N = len(alp)

    keff = 4*np.pi/lm

    # результаты
    g = np.full(N, np.nan)
    A_fit = np.full(N, np.nan)
    B_fit = np.full(N, np.nan)
    ph_fit = np.full(N, np.nan)

    # половина окна
    half = window // 2

    for i in range(half, N-half):

        # -----------------------------------------
        # текущее окно
        # -----------------------------------------

        sl = slice(i-half, i+half+1)

        x = alp[sl]
        y = P_exp[sl]

        # -----------------------------------------
        # начальные значения
        # -----------------------------------------

        A0 = (np.max(y) + np.min(y)) / 2
        B0 = (np.max(y) - np.min(y)) / 2

        # начальная оценка фазы
        idx_min = np.argmin(y)

        phi0 = 2*np.pi*x[idx_min]*T**2

        # -----------------------------------------
        # fit
        # -----------------------------------------

        try:

            popt, pcov = curve_fit(
                model,
                x,
                y,
                p0=[A0, B0, phi0],
                bounds=(
                    [-1.1, 0, -np.inf],
                    [1.1, 1.1, np.inf]
                ),
                maxfev=10000
            )

            Ai, Bi, phi = popt

            # -----------------------------------------
            # выбор правильной ветви фазы
            # -----------------------------------------

            phi_target = 2*np.pi*x[idx_min]*T**2

            M = np.round(
                (phi_target - phi)/(2*np.pi)
            )

            phi = phi + 2*np.pi*M

            # -----------------------------------------
            # вычисление g
            # -----------------------------------------

            gi = phi/(keff*T**2)

            # -----------------------------------------
            # сохранение результата
            # -----------------------------------------

            g[i] = gi
            A_fit[i] = Ai
            B_fit[i] = Bi
            ph_fit[i] = phi

        except (RuntimeError, ValueError):
            pass

    return g



def kalmanGraphs(alp, P_exp, A, B, ph, P_cov, e, en):

    # main graphic
    plt.figure()
    plt.plot(P_exp, label="data")#, marker="o")
    plt.plot(model(alp, A, B, ph), label="kalman")
    plt.plot(P_sim, label="true data")
    #plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
    plt.legend()

    # additional graphics
    plt.title("koef dynamics")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(A, label="kalman")
    axs[0].plot(A_sim, label="simulation")
    axs[0].set_title('A')
    plt.legend()

    # Второй график
    axs[1].plot(B, label="kalman")
    axs[1].plot(B_sim, label="simulation")
    axs[1].set_title('B')
    plt.legend()

    # Третий график
    lm = 780e-9
    keff = 4*np.pi/lm
    alp_min = alp[np.argmin(P_exp)]
    ph_target = 2 * np.pi * alp_min * T**2
    M = np.round( (ph_target - ph) / (2 * np.pi) )
    ph = ph + 2 * np.pi * M
    #axs[2].plot(ph) 
    g_kalman = ph/keff/T/T
    print(f"Dg = {2*np.pi/keff/T/T*1e5}")
    axs[2].plot(g_kalman*1e5, label="kalman")

    # comparison
    g_sim = np.ones(len(alp)) * g
    axs[2].plot(g_sim*1e5, label="simulation")
    g_window = windowFit(alp, P_exp, T, 20)
    axs[2].plot(g_window*1e5, label="window")
    # --- Savitzky-Golay для сравнения ---
    # окно должно быть нечётным и меньше длины массива
    window_length = 21          # можно менять (11, 21, 51, 101...)
    polyorder = 3

    # для исправления nan
    valid = np.isfinite(g_window)
    g_savgol = np.full_like(g_window, np.nan)
    if np.sum(valid) >= window_length:
        g_savgol[valid] = savgol_filter(g_window[valid], window_length=window_length, polyorder=polyorder)


    #g_savgol = savgol_filter(g_window, window_length=window_length, polyorder=polyorder) # работает в институте



    axs[2].plot(g_savgol*1e5, label="savgol")

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
    axss[2].plot(np.sqrt(P_cov[:, 2, 2])/keff/T/T*1e5, label="kalman eval")  # или axs[1, 0]
    axss[2].plot((g_kalman - g_sim)*1e5, label='kalman diff')
    axss[2].plot((g_window - g_sim)*1e5, label='window diff')
    axss[2].plot((g_savgol - g_sim)*1e5, label='savgol_diff')
    axss[2].set_title('$dg$')
    plt.legend()

    # Q finder
    # plt.figure()
    # plt.title("Q find params")
    # plt.plot(en, label="normalized innovation")
    print(f"mean norm e ={np.mean(en)}")
    print(f"std norm e ={np.std(en)}")




T = 10e-3


# experimental data
data = np.load("data_delay_800.npy")
alp = data[0]*1e6 # 1e6 из-за особенности data
P_exp = data[1]

# smimulation data
g0 = 9.8101507
Dg = 300*1e-8
lm = 780e-9
keff = 4*np.pi/lm
Dph = Dg*keff*T*T/15
alp_min = keff*g0/2/np.pi - 1/5/T/T
alp_max = keff*g0/2/np.pi + 1/5/T/T
print((alp_min + alp_max)/2)
alp_amount = 20
alp_start = np.linspace(alp_min, alp_max, alp_amount)
alp = np.zeros(1000)
g = np.zeros(len(alp))
Ph = np.zeros(len(alp))
F_vib = np.zeros(len(alp))
sigma_ph_vibr = 1e-4
A_sim = np.zeros(len(alp))
A0 = 0.15
dA_sim = 1e-3*0
DA_sim = 3e-5
B_sim = np.zeros(len(alp))
B0 = 0.21
dB_sim = 1e-3*0
ph_sim = np.zeros(len(alp))
dph_sim = 1e-4*0
P_sim = np.zeros(len(alp))
P_sim_noise = np.zeros(len(alp))
dP_sim = 1e-3
for i in range(len(alp)):
    g[i] = g0 + Dg*np.sin(2*np.pi/100*i)
    F_vib[i] = np.random.uniform(-np.pi/12, np.pi/12)
    alp[i] = alp_start[i%alp_amount] - F_vib[i]/2/np.pi/T/T
    A_sim[i] = A0 + DA_sim*i + np.random.normal(0, dA_sim)
    B_sim[i] = B0 + np.random.normal(0, dB_sim)
    ph_sim[i] = keff*g[i]*T*T + np.random.normal(0, dph_sim)
    Ph[i] = 2*np.pi*alp[i]*T*T - ph_sim[i]
    P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])
    F_vib[i] += np.random.normal(0, sigma_ph_vibr)
    alp[i] = alp_start[i%alp_amount] - F_vib[i]/2/np.pi/T/T
    P_sim_noise[i] = P_sim[i] + np.random.normal(0, dP_sim)
    



# kalman fit
init_method = "fit"
poi = 20 #len(alp)
P_cov0 = np.diag([1,1,1])*1e-3
sigma_P = 1e-4

dg_model = 0.1 # mGal
dA_model = 1e-3
dB_model = 1e-3

dA_model = np.sqrt(DA_sim**2 + dA_sim**2)
dB_model = dB_sim

lm = 780e-9
keff = 4*np.pi/lm
dph_model = dg_model*keff*T*T/1e5
print(f"dph_model = {dph_model}")
dph_model = np.sqrt(dph_sim**2*0 + Dph**2) # for simulation
Q = np.diag([dA_model**2, dB_model**2, dph_model**2])
P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(alp, P_sim_noise, T, init_method, poi, Q, P_cov0, sigma_P)

kalmanGraphs(alp, P_sim_noise, A, B, ph, P_cov, e, en)


plt.show()


