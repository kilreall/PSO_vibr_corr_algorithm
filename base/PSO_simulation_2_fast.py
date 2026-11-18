"""
Сравнение методов поиска (tau, K) для компенсации вибраций:
  1. PSO, fitness = std innovations EKF                      (распараллелен по частицам)
  2. PSO, fitness = std остатков глобальной cos-подгонки      (распараллелен по частицам)
  3. Последовательный перебор из статьи: tau при фикс. K -> K при tau_opt
  4. ПОЛНАЯ 2D-сетка, fitness = закрытая форма cos-подгонки, numba+prange
  5. ПОЛНАЯ 2D-сетка, fitness = std innovations EKF, joblib (multiprocessing)
 
Для каждого метода в итоговой таблице печатаются: tau, K, std(e) EKF,
RMS(g), bias(g), время выполнения и ЧИСЛО ВЫЗОВОВ fitness-функции
(строго по построению метода, а не через runtime-инструментацию --
т.к. PSO/сетки не делают ранней остановки, число вызовов детерминировано
конфигурацией метода).
 
Зависимости: numpy, scipy, matplotlib, filterpy, numba, joblib
    pip install numba joblib
"""
 
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from filterpy.kalman import ExtendedKalmanFilter
from numba import njit, prange
from joblib import Parallel, delayed
 
 
# ============================================================
# Модель фринджа и вспомогательные функции
# ============================================================
 
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
 
    p0 = [(np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2,
          (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]
    popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt
 
    ph_expected = 2*np.pi*np.mean(alp_init)*T**2
    M = np.round((ph_expected - ph0) / (2*np.pi))
    ph0 = ph0 + 2*np.pi*M
 
    sigma_A = np.std(P_init[:poi] - model(alp_init[:poi], A0, B0, ph0))
    return A0, B0, ph0, pcov, sigma_A
 
 
def Hx(x, alpha, T):
    A, B, ph = x
    Phi = 2*np.pi*alpha*T**2 - ph
    return np.array([A - B*np.cos(Phi)])
 
 
def HJacobian(x, alpha, T):
    A, B, ph = x
    Phi = 2*np.pi*alpha*T**2 - ph
    return np.array([[1.0, -np.cos(Phi), -B*np.sin(Phi)]])
 
 
def kalmanFit_EKF(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0):
    N = len(alp)
    A = np.zeros(N); B = np.zeros(N); ph = np.zeros(N)
    P_m = np.zeros(N); e = np.zeros(N); en = np.zeros(N)
    P_cov = np.zeros((N, 3, 3))
 
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    A[0], B[0], ph[0], ekf.P = A0, B0, ph0, P_cov0
    P_cov[0] = ekf.P
    P_m[0] = model(alp[0], A[0], B[0], ph[0])
    ekf.x = np.array([A[0], B[0], ph[0]])
    ekf.Q = Q
 
    for i in range(1, N):
        ekf.predict()
        P_m[i] = Hx(ekf.x, alp[i], T)[0]
        e[i] = P_exp[i] - P_m[i]
 
        ekf.R = np.array([[sigma_A**2 +
                            B[i-1]**2*np.sin(2*np.pi*alp[i]*T*T - ph[i-1])**2*sigma_ph**2]])
 
        ekf.update(np.array([P_exp[i]]), HJacobian, Hx,
                   args=(alp[i], T), hx_args=(alp[i], T))
 
        S_i = ekf.S[0, 0]
        en[i] = e[i]/np.sqrt(S_i)
 
        A[i], B[i], ph[i] = ekf.x
        P_cov[i] = ekf.P
 
    return P_m, A, B, ph, P_cov, e, en
 
 
def gen_vibration_trace(N, dt, amp_seismic=1e-7, amp_res=1e-7, f_res=1.5, Q_res=2):
    fs = 1.0 / dt
    white = np.random.normal(0, 1, N)
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs[0] = freqs[1]
    shape = 1.0 / freqs**0.8
    seismic = np.fft.irfft(np.fft.rfft(white) * shape, n=N)
    seismic /= np.std(seismic)
 
    low = max(f_res - f_res/(2*Q_res), 0.05) / (fs/2)
    high = (f_res + f_res/(2*Q_res)) / (fs/2)
    b, a_f = butter(2, [low, high], btype='band')
    drive = np.random.normal(0, 1, N)
    resonance = filtfilt(b, a_f, drive)
    resonance /= np.std(resonance)
 
    return amp_seismic*seismic + amp_res*resonance
 
 
def sens_integral(delay, a):
    a = a[delay:delay + end]
    return keff * np.trapz(fa_t*a, dx=t_step)
 
 
# ============================================================
# Константы и физические параметры
# ============================================================
 
lm = 780e-9
keff = 4*np.pi/lm
 
T = 10e-3
ty = 20e-6
N_RP = 16384
T_RP = 33e-3
t_step = T_RP/N_RP
t_range = np.linspace(0, 2*T + 4*ty, round((2*T + 4*ty)/t_step))
end = len(t_range)
fa_t = fat_v(t_range)
 
_trapz_w = np.ones(end) * t_step
_trapz_w[0] *= 0.5
_trapz_w[-1] *= 0.5
weight_vec = fa_t * _trapz_w  # F_vib_raw(tau) = keff * (a_window @ weight_vec)
 
 
def simul_acc(N_sim, alp_amount, delay, K):
    g0 = 9.8101507
    g_sim = np.zeros(N_sim); g_sim[-1] = g0
    drift_corr = 3000
    Dg_drift = 30e-8
    theta_drift = 1 - np.exp(-1.0/drift_corr)
    sigma_g_drift = Dg_drift * np.sqrt(theta_drift*(2 - theta_drift))
    print(f"sigma_g_drift = {sigma_g_drift}")
 
    alp_min = keff*g0/2/np.pi - 1/5/T/T
    alp_max = keff*g0/2/np.pi + 1/5/T/T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)
    alp = np.zeros(N_sim); alp_vib = np.zeros(N_sim)
 
    Ph = np.zeros(N_sim); F_vib = np.zeros(N_sim)
    a_m = np.zeros((N_sim, N_RP))
    sigma_a = 0.8e-8
    sigma_ph_vibr = K * keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2))
    print(f"sigma_ph_vibr = {sigma_ph_vibr/keff/T/T*1e8} uGal")
 
    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15; dA_sim = 1e-4
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21; dB_sim = 1e-4
    ph_sim = np.zeros(N_sim)
    P_sim = np.zeros(N_sim); P_sim_noise = np.zeros(N_sim)
    sigma_A_sim = 3e-3
 
    for i in range(N_sim):
        g_sim[i] = g0 + (g_sim[i-1] - g0)*(1 - theta_drift) + sigma_g_drift*np.random.normal()
        ph_sim[i] = keff*g_sim[i]*T*T
 
        a = gen_vibration_trace(N_RP, t_step)
        F_vib[i] = sens_integral(delay, a)*K
        alp[i] = alp_start[i % alp_amount]
        alp_vib[i] = alp[i] - F_vib[i]/2/np.pi/T/T
        Ph[i] = 2*np.pi*alp_vib[i]*T*T - ph_sim[i]
 
        A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])
 
        a_m[i] = a + np.random.normal(0, sigma_a, N_RP)
        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)
 
    dph_sim = sigma_g_drift*keff*T*T
    return (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim,
            dph_sim, sigma_g_drift, sigma_ph_vibr, sigma_A_sim, a_m)
 
 
N_sim = 1000
alp_amount = 201
delay = 500
K = 0.9
 
(alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
 sigma_g_drift, sigma_ph_vibr, sigma_A_sim, a_m) = simul_acc(N_sim, alp_amount, delay, K)
 
Q = np.diag([dA_sim**2, dB_sim**2, dph_sim**2])
poi = alp_amount
A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
sigma_ph = np.sqrt(sigma_ph_vibr**2 + dph_sim**2*0)
sigma_A = sigma_A_sim
 
tau_range = [0, 1000]
K_range = [0.7, 1.1]
N_particles = 20
M_iter = 20
K_nominal_for_tau_search = 1.0
 
 
# ============================================================
# Fitness-функции для одиночной точки
# ============================================================
 
def kalman_fitness(tau, K, alp, P_exp, a_m, warmup=50):
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    K = float(np.clip(K, K_range[0], K_range[1]))
    if tau + end > a_m.shape[1]:
        return 1e6
 
    a_window = a_m[:, tau:tau + end]
    F_vib = keff * K * (a_window @ weight_vec)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    try:
        A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, P_exp, poi)
    except Exception:
        return 1e6
    try:
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, P_exp, T, Q, P_cov0p, sigma_A_sim, sigma_ph, A0p, B0p, ph0p)
    except Exception:
        return 1e6
    return np.std(e[warmup:])
 
 
def curvefit_fitness(tau, K, alp, P_exp, a_noise_all):
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
 
 
# ============================================================
# PSO, распараллелен по частицам (joblib) -- на каждой итерации
# все N_particles fitness-вызовов независимы и считаются одновременно
# ============================================================
 
def PSO(fitness_func, N_particles, M_iter, tau_range, K_range, alp, P_exp, a_m, n_jobs=-1):
    c1, c2, w = 2.0, 2.0, 0.9
    dim = 2
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
 
    with Parallel(n_jobs=n_jobs) as parallel:
        for it in range(M_iter):
            fits = parallel(
                delayed(fitness_func)(pos[i, 0], pos[i, 1], alp, P_exp, a_m)
                for i in range(N_particles)
            )
            fits = np.array(fits)
            n_calls += N_particles
 
            improved = fits < pbest_fit
            pbest_fit[improved] = fits[improved]
            pbest[improved] = pos[improved]
 
            i_best = np.argmin(fits)
            if fits[i_best] < gbest_fit:
                gbest_fit = fits[i_best]
                gbest = pos[i_best].copy()
            history.append(gbest_fit)
 
            r1 = np.random.rand(N_particles, dim)
            r2 = np.random.rand(N_particles, dim)
            vel = w*vel + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
            pos = np.clip(pos + vel, lb, ub)
 
            print(f"Iter {it+1:02d}/{M_iter} | best sigma = {gbest_fit:.6e} | "
                  f"tau={gbest[0]:.1f}, K={gbest[1]:.4f}")
 
    return int(np.round(gbest[0])), gbest[1], gbest_fit, history, n_calls
 
 
def sequential_grid_search(fitness_func, tau_range, K_range, alp, P_exp, a_m,
                            K_nominal=K_nominal_for_tau_search, n_tau=None, n_K=401):
    """Последовательный перебор из статьи: tau при K=K_nominal, затем K при tau_opt."""
    if n_tau is None:
        tau_vals = np.arange(tau_range[0], tau_range[1] + 1, 1)
    else:
        tau_vals = np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int))
 
    fitness_tau = np.array([fitness_func(tau, K_nominal, alp, P_exp, a_m) for tau in tau_vals])
    tau_opt = int(tau_vals[np.argmin(fitness_tau)])
 
    K_vals = np.linspace(*K_range, n_K)
    fitness_K = np.array([fitness_func(tau_opt, K, alp, P_exp, a_m) for K in K_vals])
    K_opt = float(K_vals[np.argmin(fitness_K)])
 
    n_calls = len(tau_vals) + len(K_vals)
 
    best_fit = fitness_func(tau_opt, K_opt, alp, P_exp, a_m)
    n_calls += 1
 
    return tau_opt, K_opt, best_fit, n_calls, {
        "tau_vals": tau_vals, "fitness_tau": fitness_tau,
        "K_vals": K_vals, "fitness_K": fitness_K}
 
 
# ============================================================
# ПОЛНАЯ 2D-СЕТКА, ускоренная numba (закрытая форма cos-подгонки)
# ============================================================
 
def precompute_Fr_matrix(a_m, tau_min, n_tau, weight_vec, end, keff):
    N_sim = a_m.shape[0]
    Fr = np.empty((n_tau, N_sim), dtype=np.float64)
    for i in range(N_sim):
        segment = a_m[i, tau_min: tau_min + n_tau + end - 1]
        windows = sliding_window_view(segment, end)
        Fr[:, i] = windows @ weight_vec
    Fr *= keff
    return Fr
 
 
@njit(parallel=True, fastmath=True, cache=True)
def _full_grid_curvefit_kernel(P_exp, cosPhi0, sinPhi0, Fr, K_vals, syy):
    n_tau, N = Fr.shape
    n_K = K_vals.shape[0]
    fitness = np.empty((n_tau, n_K))
 
    for it in prange(n_tau):
        for ik in range(n_K):
            Kv = K_vals[ik]
 
            s1 = N
            sc = 0.0; ss = 0.0
            scc = 0.0; sss = 0.0; scs = 0.0
            sy = 0.0; scy = 0.0; ssy = 0.0
 
            for j in range(N):
                ang = Kv * Fr[it, j]
                ca = np.cos(ang)
                sa = np.sin(ang)
                c = cosPhi0[j]*ca + sinPhi0[j]*sa
                s = sinPhi0[j]*ca - cosPhi0[j]*sa
                y = P_exp[j]
 
                sc += c; ss += s
                scc += c*c; sss += s*s; scs += c*s
                sy += y; scy += c*y; ssy += s*y
 
            det = (s1*(scc*sss - scs*scs)
                   - sc*(sc*sss - scs*ss)
                   + ss*(sc*scs - scc*ss))
 
            if abs(det) < 1e-12:
                fitness[it, ik] = 1e6
                continue
 
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
            if RSS < 0.0:
                RSS = 0.0
            fitness[it, ik] = np.sqrt(RSS/N)
 
    return fitness
 
 
def full_grid_search_curvefit(alp, P_exp, a_m, tau_range, K_range, n_K=401, tau_step=1):
    tau_min, tau_max = tau_range
    tau_vals = np.arange(tau_min, tau_max + 1, tau_step)
 
    Fr = precompute_Fr_matrix(a_m, tau_min, len(np.arange(tau_min, tau_max + 1, 1)),
                               weight_vec, end, keff)
    if tau_step > 1:
        Fr = Fr[::tau_step]
 
    Phi0 = 2*np.pi*alp*T**2
    cosPhi0 = np.cos(Phi0)
    sinPhi0 = np.sin(Phi0)
    syy = float(np.sum(P_exp**2))
    K_vals = np.linspace(*K_range, n_K)
 
    fitness = _full_grid_curvefit_kernel(P_exp, cosPhi0, sinPhi0, Fr, K_vals, syy)
    n_calls = fitness.size  # каждая ячейка сетки = одна fitness-оценка (закрытая форма)
 
    idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    tau_opt = int(tau_vals[idx[0]])
    K_opt = float(K_vals[idx[1]])
    return tau_opt, K_opt, fitness[idx], n_calls, {
        "tau_vals": tau_vals, "K_vals": K_vals, "fitness": fitness}
 
 
# ============================================================
# ПОЛНАЯ 2D-СЕТКА с fitness = Калман (joblib, multiprocessing)
# ============================================================
 
def _kalman_row(tau, K_vals, alp, P_exp, a_m):
    row = np.empty(len(K_vals))
    for ik, Kv in enumerate(K_vals):
        row[ik] = kalman_fitness(tau, Kv, alp, P_exp, a_m)
    return row
 
 
def full_grid_search_kalman(tau_range, K_range, alp, P_exp, a_m, n_tau=41, n_K=21, n_jobs=-1):
    tau_vals = np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int))
    K_vals = np.linspace(*K_range, n_K)
 
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_kalman_row)(int(tau), K_vals, alp, P_exp, a_m) for tau in tau_vals
    )
    fitness = np.array(rows)
    n_calls = fitness.size
 
    idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    tau_opt = int(tau_vals[idx[0]])
    K_opt = float(K_vals[idx[1]])
    return tau_opt, K_opt, fitness[idx], n_calls, {
        "tau_vals": tau_vals, "K_vals": K_vals, "fitness": fitness}
 
 
# ============================================================
# Оценка результата и отчёт
# ============================================================
 
def evaluate_with_kalman(tau, K, alp, P_exp, a_noise_all):
    a_window = a_noise_all[:, tau:tau + end]
    F_vib = keff * K * (a_window @ weight_vec)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    A0e, B0e, ph0e, P_cov0e, _ = init_values(alp_comp, P_exp, poi)
    P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(
        alp_comp, P_exp, T, Q, P_cov0e, sigma_A_sim, sigma_ph, A0e, B0e, ph0e)
    return alp_comp, ph, e, en
 
 
def g_error_stats(ph, g_sim, warmup=50):
    g_est = ph / keff / T**2
    diff = g_est[warmup:] - g_sim[warmup:]
    bias = np.median(diff)
    rms = np.sqrt(np.mean((diff - bias)**2))
    return rms, bias, g_est
 
 
def report_case(label, tau, K, ph_est, e_arr, g_sim, elapsed_s, n_calls, warmup=50):
    std_e = np.std(e_arr[warmup:])
    rms_g, bias_g, _ = g_error_stats(ph_est, g_sim, warmup=warmup)
    print(f"{label:26s}{tau:8d}{K:10.4f}{std_e:16.4e}{rms_g*1e8:16.3f}"
          f"{bias_g*1e8:14.3f}{elapsed_s:12.2f}{n_calls:12d}")
    return std_e, rms_g, bias_g
 
 
if __name__ == "__main__":
 
    warmup = 50
    results = {}  # label -> (tau, K, elapsed_s, n_calls)
 
    print("=== PSO with Kalman-filter fitness (parallel) ===")
    t0 = time.perf_counter()
    tau_kf, K_kf, fit_kf, hist_kf, calls_kf = PSO(
        kalman_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m)
    results["PSO (Kalman, parallel)"] = (tau_kf, K_kf, time.perf_counter() - t0, calls_kf)
 
    print("\n=== PSO with curve_fit fitness (parallel) ===")
    t0 = time.perf_counter()
    tau_cf, K_cf, fit_cf, hist_cf, calls_cf = PSO(
        curvefit_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m)
    results["PSO (curve_fit, parallel)"] = (tau_cf, K_cf, time.perf_counter() - t0, calls_cf)
 
    print("\n=== Sequential search (article algorithm, curve_fit fitness) ===")
    t0 = time.perf_counter()
    tau_sq, K_sq, fit_sq, calls_sq, hist_sq = sequential_grid_search(
        curvefit_fitness, tau_range, K_range, alp, P_sim_noise, a_m)
    results["Sequential (curve_fit)"] = (tau_sq, K_sq, time.perf_counter() - t0, calls_sq)
 
    print("\n=== FULL 2D grid, closed-form cos-fit (numba+prange) ===")
    t0 = time.perf_counter()
    tau_g1, K_g1, fit_g1, calls_g1, hist_g1 = full_grid_search_curvefit(
        alp, P_sim_noise, a_m, tau_range, K_range, n_K=401, tau_step=1)
    results["Full grid (cos-fit, numba)"] = (tau_g1, K_g1, time.perf_counter() - t0, calls_g1)
 
    print("\n=== FULL 2D grid, Kalman fitness (joblib, coarser grid) ===")
    t0 = time.perf_counter()
    tau_g2, K_g2, fit_g2, calls_g2, hist_g2 = full_grid_search_kalman(
        tau_range, K_range, alp, P_sim_noise, a_m, n_tau=41, n_K=21, n_jobs=-1)
    results["Full grid (Kalman, joblib)"] = (tau_g2, K_g2, time.perf_counter() - t0, calls_g2)
 
    # --- единая метрика (EKF innovations + точность g) для всех методов ---
    print("\n============================== Сравнение ==============================")
    print(f"{'':26s}{'tau':>8s}{'K':>10s}{'std(e), EKF':>16s}{'RMS(g), uGal':>16s}"
          f"{'bias(g), uGal':>14s}{'time, s':>12s}{'fitness calls':>12s}")
    print(f"{'true':26s}{delay:8d}{K:10.4f}")
 
    for label, (tau_v, K_v, t_v, n_calls_v) in results.items():
        alp_v, ph_v, e_v, en_v = evaluate_with_kalman(tau_v, K_v, alp, P_sim_noise, a_m)
        report_case(label, tau_v, K_v, ph_v, e_v, g_sim, t_v, n_calls_v, warmup)
 
    # --- heatmap полной сетки (cos-fit) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(hist_g1["K_vals"], hist_g1["tau_vals"], hist_g1["fitness"],
                        shading="auto")
    ax.plot(K_g1, tau_g1, 'r*', markersize=12, label=f"optimum (tau={tau_g1}, K={K_g1:.3f})")
    ax.set_xlabel("K")
    ax.set_ylabel("tau")
    ax.set_title("Полная сетка (cos-fit, numba): fitness(tau, K)")
    fig.colorbar(im, ax=ax, label="fitness (std cos-fit residual)")
    ax.legend()
 
    plt.show()