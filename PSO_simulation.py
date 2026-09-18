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
#P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(alp, P_sim_noise, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0)

#PSO params
tau_range = [0, 1000]
K_range = [0.7, 1.1]
N_particles = 20
M_iter = 20

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
    A,B,ph во времени, в отличие от Калмана).
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
 
    for it in range(M_iter):
        for i in range(N_particles):
            fit = fitness_func(pos[i, 0], pos[i, 1], alp, P_exp, a_m)
 
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
 
    return best_tau, best_K, gbest_fit, history


def evaluate_with_kalman(tau, K, alp, P_exp, a_noise_all):
    """Прогнать финальный EKF с заданными (tau, K) — для честного сравнения
    обоих методов по одной и той же метрике (std innovations после EKF)."""
    a_window = a_noise_all[:, tau:tau + end]
    F_vib = keff * K * (a_window @ weight_vec)
    alp_comp = alp - F_vib / (2 * np.pi * T**2)
 
    A0e, B0e, ph0e, P_cov0e, _ = init_values(alp_comp, P_exp, poi)
 
    P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(
        alp_comp, P_exp, T, Q, P_cov0e, sigma_A_sim, sigma_ph, A0e, B0e, ph0e
    )
    return alp_comp, e, en
 
 
if __name__ == "__main__":
 
    print("=== PSO with Kalman-filter fitness ===")
    tau_kf, K_kf, fit_kf, history_kf = PSO(
        kalman_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m
    )
 
    print("\n=== PSO with global curve_fit fitness ===")
    tau_cf, K_cf, fit_cf, history_cf = PSO(
        curvefit_fitness, N_particles, M_iter, tau_range, K_range, alp, P_sim_noise, a_m
    )
 
    # оцениваем оба результата ОДНОЙ и той же метрикой (EKF innovations),
    # чтобы сравнение было честным, а не "каждый по своей шкале"
    _, e_kf, en_kf = evaluate_with_kalman(tau_kf, K_kf, alp, P_sim_noise, a_m)
    _, e_cf, en_cf = evaluate_with_kalman(tau_cf, K_cf, alp, P_sim_noise, a_m)
 
    warmup = 50
    print("\n===================== Сравнение =====================")
    print(f"{'':22s}{'tau':>8s}{'K':>10s}{'own fitness':>16s}{'std(e), EKF':>16s}")
    print(f"{'true':22s}{delay:8d}{K:10.4f}{'':16s}{'':16s}")
    print(f"{'Kalman-fitness PSO':22s}{tau_kf:8d}{K_kf:10.4f}{fit_kf:16.4e}{np.std(e_kf[warmup:]):16.4e}")
    print(f"{'curve_fit-fitness PSO':22s}{tau_cf:8d}{K_cf:10.4f}{fit_cf:16.4e}{np.std(e_cf[warmup:]):16.4e}")
 
    # --- convergence curves ---
    plt.figure()
    plt.plot(history_kf, label="Kalman fitness")
    plt.plot(history_cf, label="curve_fit fitness")
    plt.xlabel("PSO iteration")
    plt.ylabel("best fitness (своя шкала для каждого метода)")
    plt.title("PSO convergence: Kalman vs curve_fit")
    plt.legend()
 
    # --- итоговые innovations под общей меркой (EKF), для честного сравнения ---
    plt.figure()
    plt.plot(e_kf, label=f"Kalman-fitness PSO (tau={tau_kf}, K={K_kf:.3f})", alpha=0.8)
    plt.plot(e_cf, label=f"curve_fit-fitness PSO (tau={tau_cf}, K={K_cf:.3f})", alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel("EKF innovation e")
    plt.title("Итоговые невязки EKF при компенсации, найденной разными методами")
    plt.legend()
 
    plt.show()