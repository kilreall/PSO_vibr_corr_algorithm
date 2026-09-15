# func to test kalman filter without vibrations on ceration data with comfortable initializtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import savgol_filter

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

def init_values_linear(alp, P_exp, poi, T):
    """
    Линейный аналог init_values. Модель A - B*cos(Phi-ph) раскладывается
    как A - C*cos(Phi) - D*sin(Phi), где C=B*cos(ph), D=B*sin(ph) --
    линейные параметры, решаем обычной МНК вместо nonlinear curve_fit.
    Возвращает то же самое: A0, B0, ph0, pcov (3x3, в параметризации
    [A,B,ph], перенесённой Якобианом из [A,C,D]), sigma_A.
    """
 
    x = alp[:poi]
    y = P_exp[:poi]
 
    Phi = 2*np.pi*x*T**2
    c = np.cos(Phi)
    s = np.sin(Phi)
    M = np.stack([np.ones_like(c), -c, -s], axis=-1)     # (poi, 3)
 
    params, *_ = np.linalg.lstsq(M, y, rcond=None)
    A0, C0, D0 = params
 
    B0 = np.sqrt(C0**2 + D0**2)
    ph0 = np.arctan2(D0, C0)
 
    resid = y - (A0 - C0*c - D0*s)
    sigma_A = np.std(resid)
 
    dof = max(len(y) - 3, 1)
    var_resid = np.sum(resid**2) / dof
    MtM_inv = np.linalg.inv(M.T @ M)
    cov_ACD = MtM_inv * var_resid
 
    if B0 > 1e-12:
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, C0/B0, D0/B0],
            [0.0, -D0/B0**2, C0/B0**2],
        ])
    else:
        J = np.eye(3)
 
    pcov = J @ cov_ACD @ J.T
 
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

def windowFit_linear(alp, P_exp, T, window, step=1):
    """Быстрый аналог windowFit: линейный (квадратурный) фит внутри
    окна, все окна решаются одним батчем np.linalg.solve."""
 
    lm = 780e-9
    keff = 4*np.pi/lm
    N = len(alp)
    half = window // 2
 
    g = np.full(N, np.nan)
 
    centers = np.arange(half, N - half, step)
    if len(centers) == 0:
        return g
 
    idx = centers[:, None] + np.arange(-half, half + 1)[None, :]
    x = alp[idx]
    y = P_exp[idx]
 
    Phi = 2*np.pi*x*T**2
    c = np.cos(Phi)
    s = np.sin(Phi)
    ones = np.ones_like(c)
 
    Mrows = np.stack([ones, -c, -s], axis=-1)           # (K, window, 3)
 
    MtM = np.einsum('kwi,kwj->kij', Mrows, Mrows)         # (K, 3, 3)
    Mty = np.einsum('kwi,kw->ki', Mrows, y)[..., None]     # (K, 3, 1)
 
    params = np.full((len(centers), 3), np.nan)
    try:
        params = np.linalg.solve(MtM, Mty)[..., 0]         # (K, 3) -> [A, C, D]
    except np.linalg.LinAlgError:
        Mty2 = Mty[..., 0]
        for k in range(len(centers)):
            try:
                params[k] = np.linalg.solve(MtM[k], Mty2[k])
            except np.linalg.LinAlgError:
                try:
                    params[k] = np.linalg.lstsq(Mrows[k], y[k], rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
 
    A_k = params[:, 0]
    C_k = params[:, 1]
    D_k = params[:, 2]
 
    B_k = np.sqrt(C_k**2 + D_k**2)
    phi_k = np.arctan2(D_k, C_k)
 
    idx_min = np.argmin(y, axis=1)
    x_min = x[np.arange(len(centers)), idx_min]
    phi_target = 2*np.pi*x_min*T**2
    Mbr = np.round((phi_target - phi_k)/(2*np.pi))
    phi_k = phi_k + 2*np.pi*Mbr
 
    g_k = phi_k/(keff*T**2)
 
    g[centers] = g_k
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
    #axss[2].plot(np.sqrt(P_cov[:, 2, 2])/keff/T/T*1e5, label="kalman eval")  # или axs[1, 0]
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


# constants
lm = 780e-9
keff = 4*np.pi/lm


T = 10e-3


# experimental data
data = np.load("data_delay_800.npy")
alp = data[0]*1e6 # 1e6 из-за особенности data
P_exp = data[1]

# smimulation data
N_sim = 1000
f = 1/200

g0 = 9.8101507
Dg = 300*1e-8
g_sim = np.zeros(N_sim)


alp_min = keff*g0/2/np.pi - 1/5/T/T
alp_max = keff*g0/2/np.pi + 1/5/T/T
alp_amount = 20
alp_start = np.linspace(alp_min, alp_max, alp_amount)
alp = np.zeros(1000)





Ph = np.zeros(len(alp))
F_vib = np.zeros(len(alp))
sigma_ph_vibr = 1e-4

A_sim = np.zeros(len(alp))
A0_sim = 0.15
dA_sim = 1e-3*0
DA_sim = 3e-5

B_sim = np.zeros(len(alp))
B0_sim = 0.21
dB_sim = 1e-3*0

ph_sim = np.zeros(len(alp))
dph_sim = 1e-4*0
Dph_sim = Dg*2*np.pi*f*(1 - np.cos(2*np.pi*f))*keff*T*T

P_sim = np.zeros(len(alp))
P_sim_noise = np.zeros(len(alp))
sigma_A_sim = 1e-3

for i in range(len(alp)):
    g_sim[i] = g0 + Dg*np.sin(2*np.pi*f*i)
    F_vib[i] = np.random.uniform(-np.pi/12, np.pi/12)
    alp[i] = alp_start[i%alp_amount] - F_vib[i]/2/np.pi/T/T
    A_sim[i] = A0_sim + DA_sim*i + np.random.normal(0, dA_sim)
    B_sim[i] = B0_sim + np.random.normal(0, dB_sim)
    ph_sim[i] = keff*g_sim[i]*T*T + np.random.normal(0, dph_sim)
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
dph_model = np.std(ph_sim[1:] - np.roll(ph_sim, 1)[1:]) * 2# np.sqrt(dph_sim**2 + Dph_sim**2)
Q = np.diag([dA_model**2, dB_model**2, dph_model**2])

# initial values
poi = 20
A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
sigma_ph = sigma_ph_vibr
sigma_A = sigma_A_sim # only for sim
P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(alp, P_sim_noise, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0)


#kalmanGraphs(alp, P_sim_noise, A, B, ph, P_cov, e, en)



### временная ФЧХ

# Вот нормальный блок АЧХ/ФЧХ для 3-состоянийного Kalman (A, B, φ) в точности в том же стиле, что и для четырёх — с windowFit_linear и savgol.
# Python### ФЧХ / АЧХ (Kalman 3 states + window + savgol)

sim_params = dict(
    g0=g0, Dg=Dg, lm=lm, keff=keff, T=T,
    alp_min=alp_min, alp_max=alp_max, alp_amount=alp_amount, alp_start=alp_start,
    sigma_ph_vibr=sigma_ph_vibr,
    A0_sim=A0_sim, dA_sim=dA_sim, DA_sim=DA_sim,
    B0_sim=B0_sim, dB_sim=dB_sim,
    dph_sim=dph_sim,
    sigma_A_sim=sigma_A_sim,
)


def savgolFilter(g_in, window_length, polyorder):
    g_out = np.full_like(g_in, np.nan)
    valid = np.isfinite(g_in)
    if np.sum(valid) >= window_length:
        g_out[valid] = savgol_filter(g_in[valid],
                                     window_length=window_length,
                                     polyorder=polyorder)
    return g_out


def simulate_data(f_mod, N, params, seed=None):
    """Генерирует alp, g_sim, ph_sim, P_sim_noise."""
    if seed is not None:
        np.random.seed(seed)

    g0 = params['g0']
    Dg = params['Dg']
    T = params['T']
    keff = params['keff']
    alp_amount = params['alp_amount']
    alp_start = params['alp_start']
    sigma_ph_vibr = params['sigma_ph_vibr']
    A0_sim = params['A0_sim']
    dA_sim = params['dA_sim']
    DA_sim = params['DA_sim']
    B0_sim = params['B0_sim']
    dB_sim = params['dB_sim']
    dph_sim = params['dph_sim']
    sigma_A_sim = params['sigma_A_sim']

    alp = np.zeros(N)
    g_sim = np.zeros(N)
    ph_sim = np.zeros(N)
    P_sim_noise = np.zeros(N)

    for i in range(N):
        g_sim[i] = g0 + Dg * np.sin(2 * np.pi * f_mod * i)

        F_vib = np.random.uniform(-np.pi/12, np.pi/12)
        alp[i] = alp_start[i % alp_amount] - F_vib / (2 * np.pi * T**2)

        A = A0_sim + DA_sim * i + np.random.normal(0, dA_sim)
        B = B0_sim + np.random.normal(0, dB_sim)
        ph_sim[i] = keff * g_sim[i] * T**2 + np.random.normal(0, dph_sim)

        Ph = 2 * np.pi * alp[i] * T**2 - ph_sim[i]
        P = A - B * np.cos(Ph)

        F_vib += np.random.normal(0, sigma_ph_vibr)
        alp[i] = alp_start[i % alp_amount] - F_vib / (2 * np.pi * T**2)
        P_sim_noise[i] = P + np.random.normal(0, sigma_A_sim)

    return alp, g_sim, ph_sim, P_sim_noise


def _interp_nans(x):
    x = x.copy()
    idx = np.arange(len(x))
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return x
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


def kalman_tracker(f_mod, sim, params, poi=20):
    """Оценка g через 3-состоянийный EKF (A, B, φ)."""
    alp, g_sim, ph_sim, P_sim_noise = sim
    T = params['T']
    keff = params['keff']

    # --- Q (как в основном блоке для 3 состояний) ---
    dA_model = np.sqrt(params['DA_sim']**2 + params['dA_sim']**2)
    dB_model = params['dB_sim']
    dph_model = np.std(ph_sim[1:] - np.roll(ph_sim, 1)[1:]) * 2
    Q = np.diag([dA_model**2, dB_model**2, dph_model**2])

    # --- инициализация ---
    A0, B0, ph0, P_cov0, sigma_A = init_values_linear(alp, P_sim_noise, poi, T)
    # можно заменить на обычный:
    # A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)

    sigma_A = params['sigma_A_sim']
    sigma_ph = params['sigma_ph_vibr']

    _, A, B, ph, _, _, _ = kalmanFit_EKF(
        alp, P_sim_noise, T, Q, P_cov0,
        sigma_A, sigma_ph, A0, B0, ph0
    )

    return ph / (keff * T**2)


def make_multi_tracker(poi, window, step, savgol_window_length, savgol_polyorder):
    """Kalman + window + savgol за один проход."""
    def tracker(f_mod, sim, params):
        alp, g_sim, ph_sim, P_sim_noise = sim

        g_kalman = kalman_tracker(f_mod, sim, params, poi=poi)
        g_window = windowFit_linear(alp, P_sim_noise, params['T'],
                                    window=window, step=step)
        g_savgol = savgolFilter(g_window, savgol_window_length, savgol_polyorder)

        return {'kalman': g_kalman, 'window': g_window, 'savgol': g_savgol}
    return tracker


def freq_point_multi(f_mod, tracker, channels,
                     N=2000, n_skip=200, n_avg=3, params=sim_params):
    """H(f) сразу для нескольких каналов."""
    t = np.arange(N)[n_skip:]
    win = np.hanning(len(t))

    Hs = {ch: [] for ch in channels}
    for k in range(n_avg):
        sim = simulate_data(f_mod, N, params, seed=k)
        g_sim = sim[1]

        g_ests = tracker(f_mod, sim, params)

        x_in = (g_sim[n_skip:] - np.mean(g_sim[n_skip:])) * win
        c_in = np.sum(x_in * np.exp(-1j * 2 * np.pi * f_mod * t))

        for ch in channels:
            g_est = _interp_nans(g_ests[ch])
            x_out = (g_est[n_skip:] - np.mean(g_est[n_skip:])) * win
            c_out = np.sum(x_out * np.exp(-1j * 2 * np.pi * f_mod * t))
            Hs[ch].append(c_out / c_in)

    return {ch: np.mean(vals) for ch, vals in Hs.items()}


# -------------------- свип по частоте --------------------

WINDOW = 20
WINDOW_STEP = 1          # можно 2–5 для ускорения
SAVGOL_WINDOW_LENGTH = 21
SAVGOL_POLYORDER = 3

multi_tracker = make_multi_tracker(poi, WINDOW, WINDOW_STEP,
                                   SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER)

freqs = np.logspace(-5, np.log10(0.3), 150)

channels = ['kalman', 'window', 'savgol']
H_by_channel = {ch: np.zeros(len(freqs), dtype=complex) for ch in channels}

for idx, f_mod in enumerate(freqs):
    Hs = freq_point_multi(f_mod, multi_tracker, channels)
    for ch in channels:
        H_by_channel[ch][idx] = Hs[ch]

H_kalman = H_by_channel['kalman']
H_window = H_by_channel['window']
H_savgol = H_by_channel['savgol']

# -------------------- графики --------------------

plt.figure(figsize=(9, 5))
plt.semilogx(freqs, 20*np.log10(np.abs(H_kalman)), label="kalman")
plt.semilogx(freqs, 20*np.log10(np.abs(H_window)), label="window")
plt.semilogx(freqs, 20*np.log10(np.abs(H_savgol)), label="savgol")
plt.xlabel("частота модуляции g, циклы/отсчёт")
plt.ylabel("АЧХ, дБ")
plt.title("Амплитудно-частотная характеристика (Kalman 3 states)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("amplitude_kalman3params.png", dpi=150)

plt.figure(figsize=(9, 5))
plt.semilogx(freqs, np.unwrap(np.angle(H_kalman))*180/np.pi, label="kalman")
plt.semilogx(freqs, np.unwrap(np.angle(H_window))*180/np.pi, label="window")
plt.semilogx(freqs, np.unwrap(np.angle(H_savgol))*180/np.pi, label="savgol")
plt.xlabel("частота модуляции g, циклы/отсчёт")
plt.ylabel("ФЧХ, град")
plt.title("Фазо-частотная характеристика (Kalman 3 states)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("phase_kalman3params.png", dpi=150)

plt.show()


plt.show()


