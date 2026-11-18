# func to test kalman filter without vibrations on ceration data with comfortable initializtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import savgol_filter

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)


def init_values(alp, P_exp, poi):

    alp_init = alp.copy()
    P_init = P_exp.copy()
    # idx = np.argsort(alp_init)
    # alp_init = alp_init[idx]
    # P_init  = P_init[idx]


    p0 = [ (np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2, (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]
    popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt
    sigma_A = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0)) # for real data

    # # test init fit
    # plt.figure()
    # plt.plot(alp_init, P_init)
    # plt.plot(alp_init, model(alp_init, A0, B0, ph0))


    return A0, B0, ph0, pcov, sigma_A

def init_values_linear(alp, P_exp, poi, T):
    """
    Линейный аналог init_values. Возвращает то же самое: A0, B0, ph0,
    приближённую ковариацию pcov (3x3, в параметризации [A, B, ph],
    получена через линейную МНК + перенос ошибок Якобианом (C,D)->(B,ph)),
    и sigma_A.
    """
 
    x = alp[:poi]
    y = P_exp[:poi]
 
    Phi = 2 * np.pi * x * T**2
    c = np.cos(Phi)
    s = np.sin(Phi)
    M = np.stack([np.ones_like(c), -c, -s], axis=-1)     # (poi, 3)
 
    # линейная МНК
    params, *_ = np.linalg.lstsq(M, y, rcond=None)
    A0, C0, D0 = params
 
    B0 = np.sqrt(C0**2 + D0**2)
    ph0 = np.arctan2(D0, C0)
 
    resid = y - (A0 - C0 * c - D0 * s)
    sigma_A = np.std(resid)
 
    # ковариация [A, C, D] -> перенос в [A, B, ph] через Якобиан
    dof = max(len(y) - 3, 1)
    var_resid = np.sum(resid**2) / dof
    MtM_inv = np.linalg.inv(M.T @ M)
    cov_ACD = MtM_inv * var_resid
 
    if B0 > 1e-12:
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, C0 / B0, D0 / B0],
            [0.0, -D0 / B0**2, C0 / B0**2],
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
         -B*np.sin(Phi), 0, 0]
    ])

    return H


def kalmanFit_EKF(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0, v_ph0, a_ph0):

    N = len(alp)

    # --------------------------------------------------
    # Output arrays
    # --------------------------------------------------

    A = np.zeros(N)
    B = np.zeros(N)
    ph = np.zeros(N)
    v_ph = np.zeros(N)
    a_ph = np.zeros(N)
    P_m = np.zeros(N)
    e = np.zeros(N)
    en = np.zeros(N)
    P_cov = np.zeros((N, 5, 5))

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    # create EKF
    ekf = ExtendedKalmanFilter(dim_x=5, dim_z=1)

    A[0], B[0], ph[0], v_ph[0], a_ph[0], ekf.P = A0, B0, ph0, v_ph0, a_ph0, P_cov0

    # initial covariance
    P_cov[0] = ekf.P

    P_m[0] = model(
        alp[0],
        A[0],
        B[0],
        ph[0]
    )

    # state = [A, B, phi]
    ekf.x = np.array([A[0], B[0], ph[0], v_ph[0], a_ph[0]])

    # process noise
    ekf.Q = Q

    ekf.F = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.5],
    [0.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
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
        v_ph[i] = ekf.x[3]
        a_ph[i] = ekf.x[4]
    
        # covariance
        P_cov[i] = ekf.P

    return P_m, A, B, ph, v_ph, a_ph, P_cov, e, en

def windowFit(alp, P_exp, T, window, step):
 
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
 
    # step — на сколько отсчётов сдвигается центр окна между соседними
    # фитами (т.е. сколько точек "отрезается"/пропускается на каждом шаге).
    # step=1 воспроизводит прежнее поведение (фит на каждом отсчёте).
    for i in range(half, N-half, step):
 
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
    """
    Быстрый аналог windowFit: та же логика (скользящее окно + оценка g),
    но фит внутри каждого окна линейный (квадратурное разложение), а
    все окна считаются одним батчем через np.linalg.solve, без Python
    цикла по окнам и без curve_fit.
    """
 
    lm = 780e-9
    keff = 4 * np.pi / lm
    N = len(alp)
    half = window // 2
 
    g = np.full(N, np.nan)
 
    centers = np.arange(half, N - half, step)
    if len(centers) == 0:
        return g
 
    # индексы всех окон сразу: (K, window)
    idx = centers[:, None] + np.arange(-half, half + 1)[None, :]
    x = alp[idx]          # (K, window)
    y = P_exp[idx]         # (K, window)
 
    Phi = 2 * np.pi * x * T**2
    c = np.cos(Phi)
    s = np.sin(Phi)
    ones = np.ones_like(c)
 
    # design matrix: model = A - C*cos(Phi) - D*sin(Phi)
    Mrows = np.stack([ones, -c, -s], axis=-1)          # (K, window, 3)
 
    MtM = np.einsum('kwi,kwj->kij', Mrows, Mrows)        # (K, 3, 3)
    Mty = np.einsum('kwi,kw->ki', Mrows, y)[..., None]    # (K, 3, 1) -- нужна доп. ось,
                                                           # иначе np.linalg.solve батчево
                                                           # трактует (K,3) как core-размерности,
                                                           # а не (batch, m)
 
    params = np.full((len(centers), 3), np.nan)
    try:
        params = np.linalg.solve(MtM, Mty)[..., 0]        # (K, 3) -> [A, C, D]
    except np.linalg.LinAlgError:
        # если какие-то окна вырождены (крайне маловероятно) -- добираем
        # их поштучно через lstsq, остальное остаётся векторизованным
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
 
    # выбор правильной ветви фазы -- как в оригинале
    idx_min = np.argmin(y, axis=1)
    x_min = x[np.arange(len(centers)), idx_min]
    phi_target = 2 * np.pi * x_min * T**2
    Mbr = np.round((phi_target - phi_k) / (2 * np.pi))
    phi_k = phi_k + 2 * np.pi * Mbr
 
    g_k = phi_k / (keff * T**2)
 
    g[centers] = g_k
    return g


def kalmanGraphs(alp, P_exp, A, B, ph, v_ph, P_cov, e, en):

    # main graphic
    plt.figure()
    plt.plot(P_exp, label="data")#, marker="o")
    plt.plot(model(alp, A, B, ph), label="kalman")
    plt.plot(P_sim, label="true data")
    #plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
    plt.legend()

    # additional graphics
    fig, axs = plt.subplots(1, 5, figsize=(14, 4))
    plt.title("koef dynamics")
    axs[0].plot(A, label="kalman")
    axs[0].plot(A_sim, label="simulation")
    axs[0].set_title('A')
    axs[0].legend()

    # Второй график
    axs[1].plot(B, label="kalman")
    axs[1].plot(B_sim, label="simulation")
    axs[1].set_title('B')
    axs[1].legend()

    # Третий график
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
    g_window = windowFit(alp, P_exp, T, 20, 1)
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
    axs[2].legend()

    # первая производная
    axs[3].plot(v_ph, label="kalman")
    axs[3].plot(v_ph_sim, label="simulation")
    axs[3].set_title('v_ph')
    axs[3].legend()

    # вторая производная 
    axs[4].plot(a_ph, label="kalman")
    axs[4].plot(a_ph_sim, label="simulation")
    axs[4].set_title('a_ph')
    axs[4].legend()

    
    # uncerteinty 
    fig1, axss = plt.subplots(1, 3, figsize=(14, 4))
    plt.title("Standart deviations")
    axss[0].plot(np.sqrt(P_cov[:, 0, 0]))
    axss[0].set_title('dA')

    # Второй график
    axss[1].plot(np.sqrt(P_cov[:, 1, 1]))
    axss[1].set_title('dB')

    # Третий график
    #axss[2].plot(np.sqrt(P_cov[:, 2, 2])/keff/T/T*1e5, label="kalman eval")
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
f = 1/100

g0 = 9.8101507
Dg = 300*1e-8
g_sim = np.zeros(N_sim)

alp_min = keff*g0/2/np.pi - 1/1.6/T/T
alp_max = keff*g0/2/np.pi + 1/1.6/T/T
alp_amount = 20
alp_start = np.linspace(alp_min, alp_max, alp_amount)
alp = np.zeros(N_sim)


Ph = np.zeros(N_sim)
F_vib = np.zeros(N_sim)
sigma_ph_vibr = 1e-4

A_sim = np.zeros(N_sim)
A0_sim = 0.15
dA_sim = 1e-3*0
DA_sim = 3e-5

B_sim = np.zeros(N_sim)
B0_sim = 0.21
dB_sim = 1e-3*0

ph_sim = np.zeros(N_sim)
dph_sim = 1e-4*0

v_ph_sim = np.zeros(N_sim)
dv_ph_sim = 0

a_ph_sim = np.zeros(N_sim)


P_sim = np.zeros(N_sim)
P_sim_noise = np.zeros(N_sim)
sigma_A_sim = 1e-3


for i in range(len(alp)):
    g_sim[i] = g0 + Dg*np.sin(2*np.pi*f*i)
    v_ph_sim[i] = Dg*2*np.pi*f*np.cos(2*np.pi*f*i)*keff*T**2
    a_ph_sim[i] = -Dg*(2*np.pi*f)**2*np.sin(2*np.pi*f*i)*keff*T**2
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
    


# kalman fit
dA_model = np.sqrt(DA_sim**2 + dA_sim**2)
dB_model = dB_sim
dph_model = np.sqrt(dph_sim**2)
dv_ph_model = dv_ph_sim
da_ph_model = np.std(a_ph_sim[1:] - np.roll(a_ph_sim, 1)[1:]) * 1.9
Q = np.diag([dA_model**2, dB_model**2, dph_model**2, dv_ph_model**2, da_ph_model**2])

# initial values
poi = 20
A0, B0, ph0, P_cov_3d, sigma_A = init_values(alp, P_sim_noise, poi)
v_ph0 = Dg*2*np.pi*f*keff*T*T # incorrect?
a_ph0 = 0

P_cov0 = np.zeros((5,5))
for i in range(3):
    for j in range(3):
        P_cov0[i,j] = P_cov_3d[i,j]
P_cov0[3,3] = (Dg*2*np.pi*f*keff*T*T/3)**2
P_cov0[4,4] = (Dg*(2*np.pi*f)**2*keff*T*T/3)**2

sigma_A = sigma_A_sim
sigma_ph = sigma_ph_vibr

P_m, A, B, ph, v_ph, a_ph, P_cov, e, en = kalmanFit_EKF(alp, P_sim_noise, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0, v_ph0, a_ph0)


#kalmanGraphs(alp, P_sim_noise, A, B, ph, v_ph, P_cov, e, en)

 

### ФЧХ

sim_params = dict(
    g0=g0, Dg=Dg, lm=lm, keff=keff, T=T,
    alp_min=alp_min, alp_max=alp_max, alp_amount=alp_amount, alp_start=alp_start,
    sigma_ph_vibr=sigma_ph_vibr,
    A0_sim=A0_sim, dA_sim=dA_sim, DA_sim=DA_sim,
    B0_sim=B0_sim, dB_sim=dB_sim,
    dph_sim=dph_sim,
    dv_ph_sim=dv_ph_sim,
    sigma_A_sim=sigma_A_sim,
)
 
 
def simulate_data(f_mod, N, params, seed=None):
    """
    Генерирует alp / g_sim / ph_sim / v_ph_sim / a_ph_sim / P_sim_noise по
    ТОЙ ЖЕ модели и С ТЕМИ ЖЕ параметрами, что заданы в блоке симуляции
    выше (params — это sim_params). Меняется только частота модуляции
    f_mod и длина N.
    """
    if seed is not None:
        np.random.seed(seed)
 
    g0 = params['g0']; Dg = params['Dg']; T = params['T']; keff = params['keff']
    alp_amount = params['alp_amount']; alp_start = params['alp_start']
    sigma_ph_vibr = params['sigma_ph_vibr']
    A0_sim = params['A0_sim']; dA_sim = params['dA_sim']; DA_sim = params['DA_sim']
    B0_sim = params['B0_sim']; dB_sim = params['dB_sim']
    dph_sim = params['dph_sim']
    sigma_A_sim = params['sigma_A_sim']
 
    alp = np.zeros(N)
    g_sim = np.zeros(N)
    v_ph_sim = np.zeros(N)
    a_ph_sim = np.zeros(N)
    Ph = np.zeros(N)
    F_vib = np.zeros(N)
    A_sim = np.zeros(N)
    B_sim = np.zeros(N)
    ph_sim = np.zeros(N)
    P_sim = np.zeros(N)
    P_sim_noise = np.zeros(N)
 
    for i in range(N):
        g_sim[i] = g0 + Dg*np.sin(2*np.pi*f_mod*i)
        v_ph_sim[i] = Dg*2*np.pi*f_mod*np.cos(2*np.pi*f_mod*i)*keff*T**2
        a_ph_sim[i] = -Dg*(2*np.pi*f_mod)**2*np.sin(2*np.pi*f_mod*i)*keff*T**2
        F_vib[i] = np.random.uniform(-np.pi/12, np.pi/12)
        alp[i] = alp_start[i % alp_amount] - F_vib[i]/2/np.pi/T/T
        A_sim[i] = A0_sim + DA_sim*i + np.random.normal(0, dA_sim)
        B_sim[i] = B0_sim + np.random.normal(0, dB_sim)
        ph_sim[i] = keff*g_sim[i]*T*T + np.random.normal(0, dph_sim)
        Ph[i] = 2*np.pi*alp[i]*T*T - ph_sim[i]
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])
        F_vib[i] += np.random.normal(0, sigma_ph_vibr)
        alp[i] = alp_start[i % alp_amount] - F_vib[i]/2/np.pi/T/T
        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)
 
    return alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise
 
 
# --------------------------------------------------------------------
# Универсальный расчёт АЧХ/ФЧХ для произвольного метода оценки g.
#
# "Трекер" — это функция вида
#
#     tracker(f_mod, sim, params) -> g_est
#
# где sim = (alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise) — то,
# что вернул simulate_data для этой реализации шума. g_est может
# содержать NaN там, где метод не даёт оценку (например, края окна в
# windowFit) — они линейно интерполируются перед расчётом Фурье-
# коэффициента, чтобы это не мешало усреднению по частоте.
#
# Чтобы добавить ФЧХ калмана без производной (3 состояния) или с одной
# производной (4 состояния) — не нужно трогать freq_point_generic,
# windowFit, savgolFilter или графики: достаточно написать свой tracker
# с такой же сигнатурой (взяв за образец kalman_tracker ниже, но со
# своими Q/P_cov0/kalmanFit_EKF) и передать его в freq_point_generic.
# --------------------------------------------------------------------
 
def _interp_nans(x):
    """Линейно интерполирует NaN внутри массива (нужно для методов вроде
    windowFit, которые не дают оценку на каждом отсчёте)."""
    x = x.copy()
    idx = np.arange(len(x))
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return x
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x
 
 
def freq_point_generic(f_mod, tracker, N=2000, n_skip=200, n_avg=3, params=sim_params):
    """Комплексный коэффициент передачи H(f) = out/in на частоте f_mod
    для произвольного tracker(f_mod, sim, params) -> g_est."""
    t = np.arange(N)[n_skip:]
    window = np.hanning(len(t))
 
    Hs = []
    for k in range(n_avg):
        sim = simulate_data(f_mod, N, params, seed=k)
        alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
 
        g_est = _interp_nans(tracker(f_mod, sim, params))
 
        x_in  = (g_sim[n_skip:] - np.mean(g_sim[n_skip:])) * window
        x_out = (g_est[n_skip:] - np.mean(g_est[n_skip:])) * window
 
        c_in  = np.sum(x_in  * np.exp(-1j*2*np.pi*f_mod*t))
        c_out = np.sum(x_out * np.exp(-1j*2*np.pi*f_mod*t))
 
        Hs.append(c_out/c_in)
 
    return np.mean(Hs)
 
 
# ---------------------------- трекеры ----------------------------------
 
def kalman_tracker(f_mod, sim, params, poi=poi):
    """Оценка g через EKF (A, B, ph, v_ph, a_ph) — та же логика Q/P_cov0,
    что и в основном блоке скрипта выше."""
    alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
    T = params['T']
    keff = params['keff']
    Dg = params['Dg']
 
    dA_model = np.sqrt(params['DA_sim']**2 + params['dA_sim']**2)
    dB_model = params['dB_sim']
    dph_model = np.sqrt(params['dph_sim']**2)
    dv_ph_model = params['dv_ph_sim']
    da_ph_model = np.std(a_ph_sim[1:] - np.roll(a_ph_sim, 1)[1:]) * 2
    Q = np.diag([dA_model**2, dB_model**2, dph_model**2, dv_ph_model**2, da_ph_model**2])
 
    sigma_ph = params['sigma_ph_vibr']
    A0, B0, ph0, P_cov_3d, sigma_A = init_values_linear(alp, P_sim_noise, poi, T)
    v_ph0 = Dg*2*np.pi*f_mod*keff*T*T
    a_ph0 = 0
 
    P_cov0 = np.zeros((5, 5))
    for a in range(3):
        for b in range(3):
            P_cov0[a, b] = P_cov_3d[a, b]
    P_cov0[3, 3] = (Dg*keff*T*T/3)**2
    P_cov0[4, 4] = (Dg*keff*T*T/3)**2
 
    sigma_A = params['sigma_A_sim']  # only for sim
 
    _, A, B, ph, v_ph, a_ph, _, _, _ = kalmanFit_EKF(
        alp, P_sim_noise, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0, v_ph0, a_ph0
    )
 
    return ph/keff/T**2  # непрерывная величина, развёртка веток не нужна
 
 
def savgolFilter(g_in, window_length, polyorder):
    """Savitzky-Golay по валидным (не NaN) точкам входного сигнала — как в
    kalmanGraphs, но вынесено в функцию с настраиваемыми параметрами
    фильтра. Точки, где фильтр применить нельзя (NaN на входе, либо
    валидных точек меньше window_length), остаются NaN."""
    g_out = np.full_like(g_in, np.nan)
    valid = np.isfinite(g_in)
    if np.sum(valid) >= window_length:
        g_out[valid] = savgol_filter(g_in[valid], window_length=window_length, polyorder=polyorder)
    return g_out
 
 
def make_window_tracker(window, step):
    """window — размер окна windowFit, step — сколько отсчётов
    пропускается между соседними окнами (см. windowFit). Оставлено для
    случаев, когда нужен только один канал — если нужны сразу window И
    savgol, используйте multi_tracker ниже, чтобы не считать windowFit
    дважды."""
    def tracker(f_mod, sim, params):
        alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
        return windowFit_linear(alp, P_sim_noise, params['T'], window=window, step=step)
    return tracker
 
 
def make_savgol_tracker(window, step, savgol_window_length, savgol_polyorder):
    """Сначала windowFit(window, step), затем Savitzky-Golay поверх него.
    Как и make_window_tracker — самостоятельный трекер для одного канала;
    при совместном расчёте window+savgol используйте multi_tracker."""
    def tracker(f_mod, sim, params):
        alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
        g_window = windowFit_linear(alp, P_sim_noise, params['T'], window=window, step=step)
        return savgolFilter(g_window, savgol_window_length, savgol_polyorder)
    return tracker
 
 
def make_multi_tracker(poi, window, step, savgol_window_length, savgol_polyorder):
    """Считает kalman, window и savgol ЗА ОДИН ПРОХОД на каждый (f_mod, seed):
    - simulate_data вызывается один раз (а не по разу на каждый из трёх
      трекеров, как было бы при трёх отдельных freq_point_generic);
    - windowFit вызывается один раз, savgol строится поверх готового
      результата, а не пересчитывает окна заново.
    Это и есть основная причина ускорения — раньше окно фитировалось
    дважды (для window и для savgol), а данные симуляции генерировались
    трижды (по разу на каждый канал)."""
    def tracker(f_mod, sim, params):
        alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
 
        g_kalman = kalman_tracker(f_mod, sim, params, poi=poi)
        g_window = windowFit_linear(alp, P_sim_noise, params['T'], window=window, step=step)
        g_savgol = savgolFilter(g_window, savgol_window_length, savgol_polyorder)
 
        return {'kalman': g_kalman, 'window': g_window, 'savgol': g_savgol}
    return tracker
 
 
def freq_point_multi(f_mod, tracker, channels, N=2000, n_skip=200, n_avg=3, params=sim_params):
    """Как freq_point_generic, но tracker(f_mod, sim, params) возвращает
    словарь {имя_канала: g_est}, и H считается сразу для всех channels —
    simulate_data и любые общие промежуточные расчёты (например,
    windowFit внутри multi_tracker) выполняются один раз на seed, а не
    по разу на каждый канал."""
    t = np.arange(N)[n_skip:]
    window = np.hanning(len(t))
 
    Hs = {ch: [] for ch in channels}
    for k in range(n_avg):
        sim = simulate_data(f_mod, N, params, seed=k)
        alp, g_sim, ph_sim, v_ph_sim, a_ph_sim, P_sim_noise = sim
 
        g_ests = tracker(f_mod, sim, params)
 
        x_in = (g_sim[n_skip:] - np.mean(g_sim[n_skip:])) * window
        c_in = np.sum(x_in * np.exp(-1j*2*np.pi*f_mod*t))
 
        for ch in channels:
            g_est = _interp_nans(g_ests[ch])
            x_out = (g_est[n_skip:] - np.mean(g_est[n_skip:])) * window
            c_out = np.sum(x_out * np.exp(-1j*2*np.pi*f_mod*t))
            Hs[ch].append(c_out/c_in)
 
    return {ch: np.mean(vals) for ch, vals in Hs.items()}
 
 
# ---- свип по частоте и построение АЧХ/ФЧХ ----
 
# настройки фильтров windowFit / savgol — здесь и только здесь
WINDOW = 20              # размер окна windowFit
WINDOW_STEP = 1         # шаг между соседними окнами (сколько отсчётов "отрезается" за раз);
                         # увеличивайте, если расчёт всё ещё медленный — на ФЧХ почти не влияет,
                         # т.к. нас интересует только одна спектральная компонента на f_mod
SAVGOL_WINDOW_LENGTH = 21
SAVGOL_POLYORDER = 3
 
multi_tracker = make_multi_tracker(poi, WINDOW, WINDOW_STEP, SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER)
 
freqs = np.logspace(-5, np.log10(0.3), 200)   # циклы/отсчёт, до ~Найквиста (0.5)
 
channels = ['kalman', 'window', 'savgol']
H_by_channel = {ch: np.zeros(len(freqs), dtype=complex) for ch in channels}
for idx, f_mod in enumerate(freqs):
    Hs = freq_point_multi(f_mod, multi_tracker, channels)
    for ch in channels:
        H_by_channel[ch][idx] = Hs[ch]
 
H_kalman = H_by_channel['kalman']
H_window = H_by_channel['window']
H_savgol = H_by_channel['savgol']
 
T_rep = 200e-3
 
plt.figure()
plt.semilogx(freqs, 20*np.log10(np.abs(H_kalman)), label="kalman")
plt.semilogx(freqs, 20*np.log10(np.abs(H_window)), label="window")
plt.semilogx(freqs, 20*np.log10(np.abs(H_savgol)), label="savgol")
plt.xlabel("частота модуляции g, циклы/отсчёт")
plt.ylabel("АЧХ, дБ")
plt.title("Амплитудно-частотная характеристика")
plt.grid(True, which="both")
plt.legend()
plt.savefig("amplitude_kalman5params.png")
 
plt.figure()
plt.semilogx(freqs, np.unwrap(np.angle(H_kalman))*180/np.pi, label="kalman")
plt.semilogx(freqs, np.unwrap(np.angle(H_window))*180/np.pi, label="window")
plt.semilogx(freqs, np.unwrap(np.angle(H_savgol))*180/np.pi, label="savgol")
plt.xlabel("частота модуляции g, циклы/отсчёт")
plt.ylabel("ФЧХ, град")
plt.title("Фазо-частотная характеристика")
plt.grid(True, which="both")
plt.legend()
plt.savefig("phase_kalman5params.png")




plt.show()


