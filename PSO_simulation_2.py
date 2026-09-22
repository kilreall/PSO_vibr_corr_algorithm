import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import butter, filtfilt
 
 
# accelerometer sensetivity function
def fa(t):
    if 0 < t <= T+2*ty:
        return t/(T+2*ty)**2
    elif T+2*ty < t <= 2*T+4*ty:
        return (2*(T+2*ty)-t)/(T+2*ty)**2
    else:
        return 0
fat_v = np.vectorize(fa)
 
def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)
 
 
def init_values(alp, P_exp, poi):
 
    alp_init = alp.copy()[:poi]
    P_init = P_exp.copy()[:poi]
 
    p0 = [ (np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2, (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]
    popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt
 
    # --- разрешение порядка фринджа: возвращаем правильную "ветвь" 2π ---
    ph_expected = 2*np.pi*np.mean(alp_init)*T**2      # физическая привязка из самих alpha
    M = np.round((ph_expected - ph0) / (2*np.pi))
    ph0 = ph0 + 2*np.pi*M
 
    sigma_A = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0))
 
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
 
def gen_vibration_trace(N, dt, amp_seismic=1e-7, amp_res=1e-7,
                         f_res=1.5, Q_res=2):
    fs = 1.0 / dt
 
    # сейсмический фон ~ 1/f^0.8
    white = np.random.normal(0, 1, N)
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs[0] = freqs[1]
    shape = 1.0 / freqs**0.8
    seismic = np.fft.irfft(np.fft.rfft(white) * shape, n=N)
    seismic /= np.std(seismic)
 
    # резонанс изолирующей платформы
    low  = max(f_res - f_res/(2*Q_res), 0.05) / (fs/2)
    high = (f_res + f_res/(2*Q_res)) / (fs/2)
    b, a_f = butter(2, [low, high], btype='band')
    drive = np.random.normal(0, 1, N)
    resonance = filtfilt(b, a_f, drive)
    resonance /= np.std(resonance)
 
    return amp_seismic*seismic + amp_res*resonance
 
def sens_integral(delay, a):
    a = a[delay:delay + end]
    under_integral = fa_t*a
    integral = np.trapz(under_integral, dx=t_step)
    return keff*integral
 
 
# constants
lm = 780e-9
keff = 4*np.pi/lm
 
# physical params
T = 10e-3
ty = 20e-6
N_RP = 16384
T_RP = 33e-3
t_step = T_RP/N_RP
t_range = np.linspace(0, 2*T + 4*ty, round((2*T + 4*ty)/t_step))
end = len(t_range)
fa_t = fat_v(t_range)
 
# precomputed trapezoidal weights for vectorized sens_integral
_trapz_w = np.ones(end) * t_step
_trapz_w[0] *= 0.5
_trapz_w[-1] *= 0.5
weight_vec = fa_t * _trapz_w  # F_vib_batch = keff * K * (a_window @ weight_vec)
 
 
# simulation func
def simul_acc(N_sim, alp_amount, delay, K):
 
    g0 = 9.8101507
    g_sim = np.zeros(N_sim)
    g_sim[-1] = g0
    drift_corr = 3000
    Dg_drift = 30e-8
    theta_drift   = 1 - np.exp(-1.0/drift_corr)
    sigma_g_drift = Dg_drift * np.sqrt(theta_drift*(2 - theta_drift))
    print(f"sigma_g_drift = {sigma_g_drift}")
 
    alp_min = keff*g0/2/np.pi - 1/5/T/T
    alp_max = keff*g0/2/np.pi + 1/5/T/T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)
    alp = np.zeros(N_sim)
    alp_vib = np.zeros(N_sim)
 
    Ph = np.zeros(N_sim)
    F_vib = np.zeros(N_sim)
    a_m = np.zeros((N_sim, N_RP))
    sigma_a = 0.8e-8
    sigma_ph_vibr = K * keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2))
    print(f"sigma_ph_vibr = {sigma_ph_vibr/keff/T/T*1e8} uGal")
 
    A_sim = np.zeros(N_sim)
    A0_sim = 0.15
    A_sim[-1] = A0_sim
    dA_sim = 1e-4
 
    B_sim = np.zeros(N_sim)
    B0_sim = 0.21
    B_sim[-1] = B0_sim
    dB_sim = 1e-4
 
    ph_sim = np.zeros(N_sim)
 
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)
    sigma_A_sim = 3e-3
 
    for i in range(N_sim):
        g_sim[i] = g0 + (g_sim[i-1] - g0)*(1 - theta_drift) + sigma_g_drift*np.random.normal()
        ph_sim[i] = keff*g_sim[i]*T*T
 
        a = gen_vibration_trace(N_RP, t_step)
        F_vib[i] = sens_integral(delay, a)*K
        alp[i] = alp_start[i%alp_amount]
        alp_vib[i] = alp[i] - F_vib[i]/2/np.pi/T/T
        Ph[i] = 2*np.pi*alp_vib[i]*T*T - ph_sim[i]
 
 
        A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
 
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])
 
        a_m[i] = a + np.random.normal(0, sigma_a, N_RP)
        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)
 
    dph_sim = sigma_g_drift*keff*T*T
 
 
    return alp, P_sim_noise, P_sim,  A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim, sigma_g_drift, sigma_ph_vibr, sigma_A_sim, a_m
 
 
# sim params
N_sim = 1000
alp_amount = 201
delay = 500
K = 0.9
 
# sim data
alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim, sigma_g_drift, sigma_ph_vibr, sigma_A_sim, a_m = simul_acc(N_sim, alp_amount, delay, K)
 
 
# kalman fit
dA_model = dA_sim
dB_model = dB_sim
dph_model = dph_sim
Q = np.diag([dA_model**2, dB_model**2, dph_model**2])
 
# initial values
poi = alp_amount
A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
sigma_ph = np.sqrt(sigma_ph_vibr**2 + dph_sim**2*0)
sigma_A = sigma_A_sim # only for sim
 
#PSO params
tau_range = [0, 1000]
K_range = [0.7, 1.1]
N_particles = 20
M_iter = 20
 
# Номинальное K, используемое на первом шаге перебора τ (шаг 1 алгоритма из статьи -
# сначала ищем τ_opt при фиксированном K, затем при τ=τ_opt ищем K_opt)
K_nominal_for_tau_search = 1.0
 
def kalman_fitness(tau, K, alp, P_exp, a_m, warmup=50):
    """
    Compensate alp with candidate (tau, K), refit initial EKF state,
    run the EKF, and score by the std of post-warmup innovations.
    Lower is better (tighter, more consistent residuals).
    """
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    K = float(np.clip(K, K_range[0], K_range[1]))
 
    if tau + end > a_m.shape[1]:
        return 1e6  # out of bounds, heavy penalty
 
    a_window = a_m[:, tau:tau + end]           # (N_sim, end)
    F_vib = keff * K * (a_window @ weight_vec)          # (N_sim,)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    try:
        A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, P_exp, poi)
    except Exception:
        return 1e6
 
    try:
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, P_exp, T, Q, P_cov0p, sigma_A_sim, sigma_ph, A0p, B0p, ph0p
        )
    except Exception:
        return 1e6
 
    return np.std(e[warmup:])
 
def curvefit_fitness(tau, K, alp, P_exp, a_noise_all):
    """
    "Классический" вариант: одна глобальная подгонка A,B,ph curve_fit'ом
    по всему компенсированному датасету сразу (без слежения за дрейфом
    A,B,ph во времени, в отличие от Калмана). Это ровно та же схема,
    что и в статье: минимизация RMSE косинусной аппроксимации фринджа.
    Возвращает std остатков подгонки. Lower is better.
    """
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    K = float(np.clip(K, K_range[0], K_range[1]))
 
    if tau + end > a_noise_all.shape[1]:
        return 1e6
 
    a_window = a_noise_all[:, tau:tau + end]
    F_vib = keff * K * (a_window @ weight_vec)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    try:
        p0 = [np.mean(P_exp), (np.max(P_exp) - np.min(P_exp)) / 2, 0]
        lb_p = [-1.1, 0, 0]
        ub_p = [1.1, 1.1, 2*np.pi]
        popt, _ = curve_fit(model, alp_comp, P_exp, p0=p0, bounds=(lb_p, ub_p), maxfev=2000)
        P_fit = model(alp_comp, *popt)
        return np.std(P_exp - P_fit)
    except Exception:
        return 1e6
 
 
def PSO(fitness_func, N_particles, M_iter, tau_range, K_range, alp, P_exp, a_m):
 
    c1 = 2.0
    c2 = 2.0
    w = 0.9
 
    dim = 2  # tau, K
 
    lb = np.array([tau_range[0], K_range[0]])
    ub = np.array([tau_range[1], K_range[1]])
 
    pos = np.random.uniform(lb, ub, size=(N_particles, dim))
    vel = np.random.uniform(-0.1*(ub-lb), 0.1*(ub-lb), size=(N_particles, dim))
 
    pbest = pos.copy()
    pbest_fit = np.full(N_particles, np.inf)
    gbest = pos[0].copy()
    gbest_fit = np.inf
 
    history = []
    n_calls = 0
 
    for it in range(M_iter):
        for i in range(N_particles):
            fit = fitness_func(pos[i, 0], pos[i, 1], alp, P_exp, a_m)
            n_calls += 1
 
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = pos[i].copy()
 
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = pos[i].copy()
 
        history.append(gbest_fit)
 
        r1 = np.random.rand(N_particles, dim)
        r2 = np.random.rand(N_particles, dim)
 
        vel = (w * vel
               + c1 * r1 * (pbest - pos)
               + c2 * r2 * (gbest - pos))
 
        pos = pos + vel
        pos = np.clip(pos, lb, ub)
 
        print(f"Iter {it+1:02d}/{M_iter} | best sigma = {gbest_fit:.6e} | tau={gbest[0]:.1f}, K={gbest[1]:.4f}")
 
    best_tau = int(np.round(gbest[0]))
    best_K = gbest[1]
 
    return best_tau, best_K, gbest_fit, history, n_calls
 
 
def sequential_grid_search(fitness_func, tau_range, K_range, alp, P_exp, a_m,
                            K_nominal=K_nominal_for_tau_search,
                            n_tau=None, n_K=401):
    """
    Последовательный перебор (coefficient searching), как описан в статье:
    Шаг 1. При фиксированном (номинальном) K перебираем весь диапазон
            задержек tau с шагом 1 (tau используется как целочисленный
            индекс окна акселерометра) и ищем tau_opt, минимизирующий
            fitness_func (RMSE косинусной аппроксимации фринджа).
    Шаг 2. Зафиксировав tau = tau_opt, перебираем диапазон K и ищем K_opt,
            снова минимизируя fitness_func.
 
    В отличие от PSO это НЕ совместная 2D-оптимизация, а последовательный
    1D-поиск по каждой координате, как в оригинальном алгоритме.
    """
 
    # --- шаг 1: поиск tau_opt при K = K_nominal ---
    if n_tau is None:
        tau_vals = np.arange(tau_range[0], tau_range[1] + 1, 1)
    else:
        tau_vals = np.round(np.linspace(tau_range[0], tau_range[1], n_tau)).astype(int)
        tau_vals = np.unique(tau_vals)
 
    fitness_tau = np.array([
        fitness_func(tau, K_nominal, alp, P_exp, a_m) for tau in tau_vals
    ])
    tau_opt = int(tau_vals[np.argmin(fitness_tau)])
 
    # --- шаг 2: поиск K_opt при tau = tau_opt ---
    K_vals = np.linspace(K_range[0], K_range[1], n_K)
    fitness_K = np.array([
        fitness_func(tau_opt, K, alp, P_exp, a_m) for K in K_vals
    ])
    K_opt = float(K_vals[np.argmin(fitness_K)])
 
    best_fit = float(np.min(fitness_K))
    n_calls = len(tau_vals) + len(K_vals)
 
    history = {
        "tau_vals": tau_vals, "fitness_tau": fitness_tau,
        "K_vals": K_vals, "fitness_K": fitness_K,
    }
 
    return tau_opt, K_opt, best_fit, n_calls, history
 
 
# ============================================================
# ПОЛНАЯ 2D-СЕТКА, закрытая форма cos-подгонки (БЕЗ numba/prange --
# обычная векторизация numpy: внешний цикл по tau -- plain Python,
# все K для данного tau считаются одним численным выражением)
# ============================================================
 
def full_grid_search_curvefit(alp, P_exp, a_m, tau_range, K_range, n_K=401, tau_step=1):
    """
    Полная 2D-сетка (tau, K). Fitness для каждой ячейки -- замкнутая
    форма линейной cos-подгонки (тот же приём, что в init_values_linear/
    windowFit_linear): вместо нелинейного curve_fit(A,B,ph) решаем
    линейную МНК по регрессорам c=cos(Phi0-K*Fr), s=sin(Phi0-K*Fr),
    что математически эквивалентно curvefit_fitness, но без итеративной
    нелинейной оптимизации на каждую ячейку сетки.
 
    БЕЗ numba/prange/joblib: внешний цикл по tau -- обычный Python for,
    один поток; для фиксированного tau все n_K значений K считаются
    одним numpy-выражением (векторизация numpy, а не JIT/параллелизм).
    """
    tau_vals = np.arange(tau_range[0], tau_range[1] + 1, tau_step)
    K_vals = np.linspace(K_range[0], K_range[1], n_K)
 
    Phi0 = 2*np.pi*alp*T**2
    cosPhi0 = np.cos(Phi0)
    sinPhi0 = np.sin(Phi0)
    N = len(alp)
    syy = float(np.sum(P_exp**2))
    sy = float(np.sum(P_exp))
 
    fitness = np.full((len(tau_vals), len(K_vals)), 1e6)
    n_calls = 0
 
    for it, tau in enumerate(tau_vals):
 
        if tau + end > a_m.shape[1]:
            continue  # окно акселерометра выходит за пределы записи
 
        Fr = keff * (a_m[:, tau:tau + end] @ weight_vec)   # (N,) -- для этой tau, все K используют его
 
        # ang, c, s: (n_K, N) -- по всем K сразу, обычная numpy-векторизация
        ang = K_vals[:, None] * Fr[None, :]
        ca = np.cos(ang)
        sa = np.sin(ang)
        c = cosPhi0[None, :]*ca + sinPhi0[None, :]*sa      # cos(Phi0 - K*Fr)
        s = sinPhi0[None, :]*ca - cosPhi0[None, :]*sa      # sin(Phi0 - K*Fr)
 
        s1 = N
        sc = c.sum(axis=1); ss = s.sum(axis=1)
        scc = (c*c).sum(axis=1); sss = (s*s).sum(axis=1); scs = (c*s).sum(axis=1)
        scy = c @ P_exp; ssy = s @ P_exp
 
        n_calls += len(K_vals)
 
        det = (s1*(scc*sss - scs*scs)
               - sc*(sc*sss - scs*ss)
               + ss*(sc*scs - scc*ss))
 
        with np.errstate(invalid="ignore", divide="ignore"):
            detA = (sy*(scc*sss - scs*scs)
                    - sc*(scy*sss - scs*ssy)
                    + ss*(scy*scs - scc*ssy))
            detBc = (s1*(scy*sss - scs*ssy)
                     - sy*(sc*sss - scs*ss)
                     + ss*(sc*ssy - scy*ss))
            detBs = (s1*(scc*ssy - scy*scs)
                     - sc*(sc*ssy - scy*ss)
                     + sy*(sc*scs - scc*ss))
 
            A = detA/det
            Bc = detBc/det
            Bs = detBs/det
 
            RSS = syy - (A*sy + Bc*scy + Bs*ssy)
 
        bad = (np.abs(det) < 1e-12) | ~np.isfinite(RSS) | (RSS < 0)
        row_fitness = np.where(bad, 1e6, np.sqrt(np.maximum(RSS, 0.0)/N))
 
        fitness[it] = row_fitness
 
    idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    tau_opt = int(tau_vals[idx[0]])
    K_opt = float(K_vals[idx[1]])
    return tau_opt, K_opt, float(fitness[idx]), n_calls, {
        "tau_vals": tau_vals, "K_vals": K_vals, "fitness": fitness}
 
 
# ============================================================
# ПОЛНАЯ 2D-СЕТКА, fitness = EKF innovations (БЕЗ joblib --
# обычный последовательный двойной цикл, один поток)
# ============================================================
 
def full_grid_search_kalman(tau_range, K_range, alp, P_exp, a_m, n_tau=41, n_K=21):
    """Полная 2D-сетка (tau, K) с fitness = std innovations EKF.
    Обычный последовательный двойной цикл -- каждая ячейка считается
    по очереди, без joblib/многопроцессности. Медленнее ускоренной
    версии (нет распараллеливания по ядрам), но проще и предсказуемее."""
 
    tau_vals = np.unique(np.round(np.linspace(tau_range[0], tau_range[1], n_tau)).astype(int))
    K_vals = np.linspace(K_range[0], K_range[1], n_K)
 
    fitness = np.full((len(tau_vals), len(K_vals)), np.nan)
    n_calls = 0
 
    for it, tau in enumerate(tau_vals):
        for ik, Kv in enumerate(K_vals):
            fitness[it, ik] = kalman_fitness(int(tau), Kv, alp, P_exp, a_m)
            n_calls += 1
        print(f"  full_grid_search_kalman: строка {it+1}/{len(tau_vals)} (tau={tau}) готова")
 
    idx = np.unravel_index(np.nanargmin(fitness), fitness.shape)
    tau_opt = int(tau_vals[idx[0]])
    K_opt = float(K_vals[idx[1]])
    return tau_opt, K_opt, float(fitness[idx]), n_calls, {
        "tau_vals": tau_vals, "K_vals": K_vals, "fitness": fitness}
 
 
def evaluate_with_kalman(tau, K, alp, P_exp, a_noise_all):
    """Прогнать финальный EKF с заданными (tau, K) -- для честного сравнения
    методов по одной и той же метрике (std innovations + точность g)."""
    a_window = a_noise_all[:, tau:tau + end]
    F_vib = keff * K * (a_window @ weight_vec)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    A0e, B0e, ph0e, P_cov0e, _ = init_values(alp_comp, P_exp, poi)
 
    P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(
        alp_comp, P_exp, T, Q, P_cov0e, sigma_A_sim, sigma_ph, A0e, B0e, ph0e
    )
    return alp_comp, ph, e, en
 
def g_error_stats(ph, g_sim, warmup=50, bias_warn_threshold=1e-3):
    """RMS ошибки определения g относительно истины. Вычитаем медианный
    сдвиг перед RMS_debiased -- но также возвращаем RMS_total (без
    вычитания ничего) и явно предупреждаем, если bias подозрительно
    велик (похоже на ошибку разрешения порядка фринджа, а не на
    обычную ошибку инициализации)."""
    g_est = ph / keff / T**2
    diff = g_est[warmup:] - g_sim[warmup:]
 
    bias = np.median(diff)
    if abs(bias) > bias_warn_threshold:
        print(f"⚠ ВНИМАНИЕ: bias(g) = {bias*1e8:.3f} µGal подозрительно велик — "
              f"похоже на ошибку разрешения порядка фринджа")
 
    rms_debiased = np.sqrt(np.mean((diff - bias)**2))
    rms_total = np.sqrt(np.mean(diff**2))
 
    return rms_total, rms_debiased, bias, g_est
 
def report_case(label, tau, K, ph_est, e_arr, g_sim, elapsed_s, n_calls, warmup=50):
    std_e = np.std(e_arr[warmup:])
    rms_total, rms_debiased, bias_g, _ = g_error_stats(ph_est, g_sim, warmup=warmup)
    print(f"{label:26s}{tau:8d}{K:10.4f}{std_e:16.4e}{rms_total*1e8:16.3f}"
          f"{bias_g*1e8:14.3f}{rms_debiased*1e8:16.3f}{elapsed_s:12.2f}{n_calls:12d}")
    return std_e, rms_total, bias_g, rms_debiased
 
 
if __name__ == "__main__":
 
    warmup = 50
    results = {}  # label -> (tau, K, elapsed_s, n_calls)
 
    print("=== PSO with Kalman-filter fitness ===")
    t0 = time.perf_counter()
    tau_kf, K_kf, fit_kf, history_kf, calls_kf = PSO(
        kalman_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m
    )
    results["PSO (Kalman)"] = (tau_kf, K_kf, time.perf_counter() - t0, calls_kf)
 
    print("\n=== PSO with global curve_fit fitness ===")
    t0 = time.perf_counter()
    tau_cf, K_cf, fit_cf, history_cf, calls_cf = PSO(
        curvefit_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m
    )
    results["PSO (curve_fit)"] = (tau_cf, K_cf, time.perf_counter() - t0, calls_cf)
 
    print("\n=== Sequential grid search (coefficient searching, cos-fit RMSE) ===")
    t0 = time.perf_counter()
    tau_gs, K_gs, fit_gs, calls_gs, history_gs = sequential_grid_search(
        curvefit_fitness, tau_range, K_range, alp, P_sim_noise, a_m
    )
    results["Sequential (curve_fit)"] = (tau_gs, K_gs, time.perf_counter() - t0, calls_gs)
 
    print("\n=== FULL 2D grid, closed-form cos-fit (plain numpy, no numba) ===")
    t0 = time.perf_counter()
    tau_g1, K_g1, fit_g1, calls_g1, history_g1 = full_grid_search_curvefit(
        alp, P_sim_noise, a_m, tau_range, K_range, n_K=401, tau_step=1
    )
    results["Full grid (cos-fit)"] = (tau_g1, K_g1, time.perf_counter() - t0, calls_g1)
 
    print("\n=== FULL 2D grid, Kalman fitness (plain sequential loop, no joblib) ===")
    t0 = time.perf_counter()
    tau_g2, K_g2, fit_g2, calls_g2, history_g2 = full_grid_search_kalman(
        tau_range, K_range, alp, P_sim_noise, a_m, n_tau=41, n_K=21
    )
    results["Full grid (Kalman)"] = (tau_g2, K_g2, time.perf_counter() - t0, calls_g2)
 
    # --- единая метрика (EKF innovations + точность g) для всех методов ---
    print("\n============================== Сравнение ==============================")
    print(f"{'':26s}{'tau':>8s}{'K':>10s}{'std(e), EKF':>16s}{'RMS_tot(g), uGal':>16s}"
          f"{'bias(g), uGal':>14s}{'RMS_deb(g), uGal':>16s}{'time, s':>12s}{'fitness calls':>12s}")
    print(f"{'true':26s}{delay:8d}{K:10.4f}")
 
    for label, (tau_v, K_v, t_v, n_calls_v) in results.items():
        alp_v, ph_v, e_v, en_v = evaluate_with_kalman(tau_v, K_v, alp, P_sim_noise, a_m)
        report_case(label, tau_v, K_v, ph_v, e_v, g_sim, t_v, n_calls_v, warmup)
 
    # --- convergence curves (PSO) ---
    plt.figure()
    plt.plot(history_kf, label="Kalman fitness")
    plt.plot(history_cf, label="curve_fit fitness")
    plt.xlabel("PSO iteration")
    plt.ylabel("best fitness (своя шкала для каждого метода)")
    plt.title("PSO convergence: Kalman vs curve_fit")
    plt.legend()
 
    # --- ход перебора tau и K для sequential grid search ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_gs["tau_vals"], history_gs["fitness_tau"])
    axes[0].axvline(tau_gs, color="k", ls="--", label=f"tau_opt={tau_gs}")
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("fitness (std cos-fit residual)")
    axes[0].set_title("Шаг 1: поиск tau_opt (K фиксирован)")
    axes[0].legend()
 
    axes[1].plot(history_gs["K_vals"], history_gs["fitness_K"])
    axes[1].axvline(K_gs, color="k", ls="--", label=f"K_opt={K_gs:.3f}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("fitness (std cos-fit residual)")
    axes[1].set_title("Шаг 2: поиск K_opt (tau = tau_opt)")
    axes[1].legend()
    fig.tight_layout()
 
    # --- heatmap полной сетки (closed-form cos-fit) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(history_g1["K_vals"], history_g1["tau_vals"], history_g1["fitness"],
                        shading="auto")
    ax.plot(K_g1, tau_g1, 'r*', markersize=12, label=f"optimum (tau={tau_g1}, K={K_g1:.3f})")
    ax.set_xlabel("K")
    ax.set_ylabel("tau")
    ax.set_title("Полная сетка (closed-form cos-fit): fitness(tau, K)")
    fig.colorbar(im, ax=ax, label="fitness (std cos-fit residual)")
    ax.legend()
 
    # --- итоговые innovations ---
    plt.figure()
    for label, (tau_v, K_v, t_v, n_calls_v) in results.items():
        _, ph_v, e_v, _ = evaluate_with_kalman(tau_v, K_v, alp, P_sim_noise, a_m)
        plt.plot(e_v, label=f"{label} (tau={tau_v}, K={K_v:.3f})", alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel("EKF innovation e")
    plt.title("Итоговые невязки EKF при разных способах компенсации")
    plt.legend()
 
    # --- ошибка определения g в физических единицах ---
    plt.figure()
    for label, (tau_v, K_v, t_v, n_calls_v) in results.items():
        _, ph_v, _, _ = evaluate_with_kalman(tau_v, K_v, alp, P_sim_noise, a_m)
        plt.plot((ph_v/keff/T**2 - g_sim)*1e8, label=label, alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel(r"$g_{est} - g_{sim}$, µGal")
    plt.title("Ошибка определения g при разных способах компенсации вибрации")
    plt.legend()
 
    plt.show()