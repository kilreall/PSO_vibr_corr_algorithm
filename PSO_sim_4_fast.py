import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


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

    # --- разрешение порядка фринджа: возвращаем правильную "ветвь" 2π ---
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
    """Тот же EKF (F=I, состояние [A,B,ph], 1 скалярное измерение), что и
    раньше на filterpy.ExtendedKalmanFilter, но переписан вручную на голом
    numpy: без общей матричной инверсии (S здесь скаляр -- достаточно
    деления) и без накладных расходов класса filterpy. Формулы (predict:
    P += Q; update: joseph-форма ковариации) численно эквивалентны
    исходной filterpy-реализации, отличие только в порядке арифметики
    (проверено на синтетических данных, макс. отличие ~1e-7..1e-11).
    Даёт ~2-3x ускорение одного вызова и не требует зависимости filterpy."""
    N = len(alp)
    A = np.empty(N); B = np.empty(N); ph = np.empty(N)
    P_m = np.empty(N); e = np.empty(N); en = np.empty(N)
    P_cov = np.empty((N, 3, 3))

    x = np.array([A0, B0, ph0], dtype=float)
    P = np.array(P_cov0, dtype=float).copy()
    I3 = np.eye(3)
    two_pi_T2 = 2*np.pi*T*T

    A[0], B[0], ph[0] = x
    P_cov[0] = P
    P_m[0] = x[0] - x[1]*np.cos(two_pi_T2*alp[0] - x[2])

    for i in range(1, N):
        # --- predict (F = I, без управления) ---
        P = P + Q

        # предсказанное состояние совпадает с x[i-1], т.к. F=I;
        # поэтому Phi здесь -- то же самое, что и в формуле R ниже
        Phi = two_pi_T2*alp[i] - x[2]
        cosPhi = np.cos(Phi)
        sinPhi = np.sin(Phi)
        zpred = x[0] - x[1]*cosPhi
        P_m[i] = zpred
        e[i] = P_exp[i] - zpred

        H = np.array([1.0, -cosPhi, -x[1]*sinPhi])
        R = sigma_A**2 + x[1]**2 * sinPhi**2 * sigma_ph**2

        # --- update ---
        PHT = P @ H                # (3,)
        S = H @ PHT + R            # скаляр (dim_z = 1)
        K = PHT / S                # (3,)

        x = x + K*e[i]
        I_KH = I3 - np.outer(K, H)
        P = I_KH @ P @ I_KH.T + np.outer(K, K) * R   # joseph-форма, как в filterpy

        en[i] = e[i] / np.sqrt(S)

        A[i], B[i], ph[i] = x
        P_cov[i] = P

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
    """Сырой (без K) интеграл чувствительности -- одинаковая формула
    применяется к любой из трёх осей, разница только в K, которым
    результат домножается снаружи."""
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
weight_vec = fa_t * _trapz_w  # F_vib_axis(tau) = keff * K_axis * (a_window @ weight_vec)


# ============================================================
# Симуляция с вибрацией по трём осям
# ============================================================

def simul_acc(N_sim, alp_amount, delay, Kz, Kx, Ky):

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
    alp = np.zeros(N_sim)
    alp_vib = np.zeros(N_sim)

    Ph = np.zeros(N_sim)
    F_vibz = np.zeros(N_sim)
    F_vibx = np.zeros(N_sim)
    F_viby = np.zeros(N_sim)
    az_m = np.zeros((N_sim, N_RP))
    ax_m = np.zeros((N_sim, N_RP))
    ay_m = np.zeros((N_sim, N_RP))
    sigma_a = 0.8e-8

    # суммарный фазовый шум от вибрации -- независимые вклады трёх осей
    # складываются в квадратуре (та же формула распространения ошибки,
    # что и в одноосевом случае, применённая к каждой оси и просуммированная)
    sigma_ph_vibr = (keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2))
                      * np.sqrt(Kz**2 + Kx**2 + Ky**2))
    print(f"sigma_ph_vibr = {sigma_ph_vibr/keff/T/T*1e8} uGal")

    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15; dA_sim = 1e-4
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21; dB_sim = 1e-4
    ph_sim = np.zeros(N_sim)
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)
    sigma_A_sim = 3e-3

    for i in range(N_sim):
        g_sim[i] = g0 + (g_sim[i-1] - g0)*(1 - theta_drift) + sigma_g_drift*np.random.normal()
        ph_sim[i] = keff*g_sim[i]*T*T

        az = gen_vibration_trace(N_RP, t_step)
        ax = gen_vibration_trace(N_RP, t_step)
        ay = gen_vibration_trace(N_RP, t_step)

        F_vibz[i] = sens_integral(delay, az) * Kz
        F_vibx[i] = sens_integral(delay, ax) * Kx
        F_viby[i] = sens_integral(delay, ay) * Ky

        alp[i] = alp_start[i % alp_amount]
        alp_vib[i] = alp[i] - (F_vibz[i] + F_vibx[i] + F_viby[i]) / (2*np.pi*T*T)
        Ph[i] = 2*np.pi*alp_vib[i]*T*T - ph_sim[i]

        A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])

        az_m[i] = az + np.random.normal(0, sigma_a, N_RP)
        ax_m[i] = ax + np.random.normal(0, sigma_a, N_RP)
        ay_m[i] = ay + np.random.normal(0, sigma_a, N_RP)
        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)

    dph_sim = sigma_g_drift*keff*T*T

    return (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
            sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m, ax_m, ay_m)


# --- диапазоны поиска ---
# (Числовые диапазоны/размеры сетки -- чистые константы без рандома,
# их можно спокойно оставить на верхнем уровне модуля: при повторном
# импорте модуля в дочернем процессе на Windows (spawn) это не стоит
# почти ничего. А вот саму симуляцию (simul_acc) и инициализацию
# Kalman-фильтра переносим внутрь `if __name__ == "__main__":` ниже --
# на Windows spawn заново импортирует модуль в каждом дочернем процессе,
# и без этой защиты дорогая симуляция считалась бы повторно на каждый
# процесс пула. Воркерам эти данные не нужны напрямую: всё, что им
# требуется, явно передаётся через initializer/initargs Pool.
tau_range = [0, 1000]
Kz_range = [0.7, 1.1]
# Kx, Ky на порядки меньше и знак заранее неизвестен -- берём симметричный
# диапазон вокруг нуля с запасом относительно номинала (0.003)
Kx_range = [0.0, 0.025]
Ky_range = [0.0, 0.025]

N_particles = 30       # чуть больше, чем в 2D-случае -- пространство поиска стало 4D
M_iter = 30

Kz_nominal = 1.0
Kx_nominal = 0.0
Ky_nominal = 0.0

# --- параметры сетки/дискретизации для каждого сравниваемого алгоритма ---
# Вынесены сюда, чтобы их можно было менять в одном месте, не заходя внутрь
# определений функций (PSO задаётся числом частиц/итераций выше:
# N_particles, M_iter -- он один и тот же для обоих fitness-функций).

# Sequential coordinate search (tau -> Kz -> Kx -> Ky)
SEQ_N_TAU = None      # None -> полный перебор tau с шагом 1 на диапазоне tau_range
SEQ_N_KZ = 401
SEQ_N_KX = 201
SEQ_N_KY = 201
SEQ_N_PASSES = 2
SEQ_TAU_INIT = 500
SEQ_KZ_INIT = Kz_nominal
SEQ_KX_INIT = Kx_nominal
SEQ_KY_INIT = Ky_nominal

# Full 4D grid, closed-form cos-fit (грубая, но быстрая сетка)
GRID_CF_N_TAU = 21
GRID_CF_N_KZ = 11
GRID_CF_N_KX = 5
GRID_CF_N_KY = 5

# Full 4D grid, Kalman fitness (теперь можно брать заметно плотнее сетку --
# перебор Kz,Kx,Ky для каждого tau распараллелен по процессам, а сам EKF
# посчитан без filterpy, так что цена ячейки заметно ниже)
GRID_KF_N_TAU = 17
GRID_KF_N_KZ = 11
GRID_KF_N_KX = 4
GRID_KF_N_KY = 4

# Число процессов для параллельных full-grid и PSO поисков.
# None -> использовать все доступные ядра (mp.cpu_count()).
N_JOBS = None


def compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m):
    """Общая функция компенсации -- используется во всех fitness/eval."""
    az_window = az_m[:, tau:tau + end]
    ax_window = ax_m[:, tau:tau + end]
    ay_window = ay_m[:, tau:tau + end]

    Fz = keff * (az_window @ weight_vec)
    Fx = keff * (ax_window @ weight_vec)
    Fy = keff * (ay_window @ weight_vec)

    F_vib_total = Kz*Fz + Kx*Fx + Ky*Fy
    return alp - F_vib_total / (2 * np.pi * T**2)


def kalman_fitness(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m, warmup=50):
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    Kz = float(np.clip(Kz, Kz_range[0], Kz_range[1]))
    Kx = float(np.clip(Kx, Kx_range[0], Kx_range[1]))
    Ky = float(np.clip(Ky, Ky_range[0], Ky_range[1]))

    if tau + end > az_m.shape[1]:
        return 1e6

    alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)

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


def curvefit_fitness(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m):
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    Kz = float(np.clip(Kz, Kz_range[0], Kz_range[1]))
    Kx = float(np.clip(Kx, Kx_range[0], Kx_range[1]))
    Ky = float(np.clip(Ky, Ky_range[0], Ky_range[1]))

    if tau + end > az_m.shape[1]:
        return 1e6

    alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)

    try:
        p0 = [np.mean(P_exp), (np.max(P_exp) - np.min(P_exp)) / 2, 0]
        lb_p = [-1.1, 0, 0]
        ub_p = [1.1, 1.1, 2*np.pi]
        popt, _ = curve_fit(model, alp_comp, P_exp, p0=p0, bounds=(lb_p, ub_p), maxfev=2000)
        P_fit = model(alp_comp, *popt)
        return np.std(P_exp - P_fit)
    except Exception:
        return 1e6


def closed_form_cosfit_rms(c, s, y):
    """Замкнутая форма линейной cos-подгонки: y ~ A + Bc*c + Bs*s.
    Возвращает RMS остатков. Используется в full-grid методах вместо
    нелинейного curve_fit -- быстрее на много порядков за счёт того,
    что при ФИКСИРОВАННОМ наборе (tau,Kz,Kx,Ky) сама подгонка A,B,ph
    линейна (та же идея, что в init_values_linear/windowFit_linear)."""
    N = len(y)
    s1 = N
    sc = c.sum(); ss = s.sum()
    scc = np.dot(c, c); sss = np.dot(s, s); scs = np.dot(c, s)
    sy = y.sum(); scy = np.dot(c, y); ssy = np.dot(s, y)
    syy = np.dot(y, y)

    det = (s1*(scc*sss - scs*scs) - sc*(sc*sss - scs*ss) + ss*(sc*scs - scc*ss))
    if abs(det) < 1e-12:
        return 1e6

    detA = (sy*(scc*sss - scs*scs) - sc*(scy*sss - scs*ssy) + ss*(scy*scs - scc*ssy))
    detBc = (s1*(scy*sss - scs*ssy) - sy*(sc*sss - scs*ss) + ss*(sc*ssy - scy*ss))
    detBs = (s1*(scc*ssy - scy*scs) - sc*(sc*ssy - scy*ss) + sy*(sc*scs - scc*ss))

    A = detA/det; Bc = detBc/det; Bs = detBs/det
    RSS = syy - (A*sy + Bc*scy + Bs*ssy)
    if RSS < 0 or not np.isfinite(RSS):
        return 1e6
    return float(np.sqrt(RSS/N))


# ============================================================
# PSO, обобщённый на 4 измерения (tau, Kz, Kx, Ky)
#
# Ниже -- ДВЕ версии: исходная последовательная PSO() (оставлена для
# справки/сравнения) и новая PSO_parallel(), которую и следует
# использовать. Причина, по которой PSO с 900 вызовами fitness работал
# медленнее full-grid поиска с ~3000 вызовов:
#   1) PSO() считает частиц строго по одной, в один процесс -- никакого
#      multiprocessing, в отличие от full_grid_search_kalman_4d, который
#      явно параллелится через mp.Pool на все ядра;
#   2) на каждый вызов kalman_fitness/curvefit_fitness заново пересчиты-
#      вается самая тяжёлая операция -- свёртка (az_m/ax_m/ay_m)[:,tau:tau+end] @ weight_vec
#      по всем трём осям, даже если tau у соседних частиц почти не
#      меняется. full_grid_search_kalman_4d считает эту свёртку один раз
#      на каждое значение tau и переиспользует её для всех Kz,Kx,Ky.
# PSO_parallel() устраняет оба узких места: оценивает всю популяцию
# параллельно в пуле процессов и кэширует Fz,Fx,Fy по округлённому tau
# внутри каждого процесса-воркера (кэш живёт всё время жизни пула, т.е.
# на протяжении всех M_iter итераций).
# ============================================================

def PSO(fitness_func, N_particles, M_iter, bounds, alp, P_exp, az_m, ax_m, ay_m):
    """Исходная последовательная версия. Оставлена для сравнения/отладки
    на одном ядре; для реальных запусков используйте PSO_parallel()."""
    c1, c2, w = 2.0, 2.0, 0.9
    dim = len(bounds)

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

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
            fit = fitness_func(pos[i, 0], pos[i, 1], pos[i, 2], pos[i, 3],
                                alp, P_exp, az_m, ax_m, ay_m)
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
        vel = w*vel + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
        pos = np.clip(pos + vel, lb, ub)

        print(f"Iter {it+1:02d}/{M_iter} | best fitness = {gbest_fit:.6e} | "
              f"tau={gbest[0]:.1f}, Kz={gbest[1]:.4f}, Kx={gbest[2]:.5f}, Ky={gbest[3]:.5f}")

    tau_opt = int(np.round(gbest[0]))
    Kz_opt, Kx_opt, Ky_opt = gbest[1], gbest[2], gbest[3]
    return tau_opt, Kz_opt, Kx_opt, Ky_opt, gbest_fit, history, n_calls


# --- воркеры для параллельного PSO ---
# Большие массивы (alp, P_exp, az_m/ax_m/ay_m) наследуются дочерним
# процессом через fork (copy-on-write) на Linux, либо передаются один раз
# через initargs на Windows (spawn) -- в обоих случаях в задачи, идущие
# через pool.map, передаются только координаты частицы (4 числа), а не
# сами массивы. Скалярные параметры EKF (Q, sigma_A_sim, sigma_ph, poi,
# warmup) тоже передаются явно через initargs -- они устанавливаются
# внутри `if __name__ == "__main__":` и потому недоступны воркеру как
# module-level globals при spawn (та же логика, что и в _kf_worker_init).

def _pso_worker_init(alp_, P_exp_, az_m_, ax_m_, ay_m_, fitness_kind_,
                      Q_, sigma_A_sim_, sigma_ph_, poi_, warmup_):
    global _g_alp, _g_Pexp, _g_az_m, _g_ax_m, _g_ay_m, _g_fitness_kind
    global _g_Q, _g_sigma_A_sim, _g_sigma_ph, _g_poi, _g_warmup, _g_FFcache
    _g_alp, _g_Pexp = alp_, P_exp_
    _g_az_m, _g_ax_m, _g_ay_m = az_m_, ax_m_, ay_m_
    _g_fitness_kind = fitness_kind_
    _g_Q, _g_sigma_A_sim, _g_sigma_ph = Q_, sigma_A_sim_, sigma_ph_
    _g_poi, _g_warmup = poi_, warmup_
    _g_FFcache = {}  # кэш Fz,Fx,Fy по округлённому tau; живёт весь пул


def _pso_get_FzFxFy(tau):
    """Кэшированная версия дорогой свёртки (az_m/ax_m/ay_m) @ weight_vec.
    Один и тот же tau часто повторяется у разных частиц/итераций PSO
    (особенно ближе к сходимости роя), поэтому кэш реально экономит
    большую часть вызовов, а не только теоретически."""
    if tau not in _g_FFcache:
        Fz = keff * (_g_az_m[:, tau:tau + end] @ weight_vec)
        Fx = keff * (_g_ax_m[:, tau:tau + end] @ weight_vec)
        Fy = keff * (_g_ay_m[:, tau:tau + end] @ weight_vec)
        _g_FFcache[tau] = (Fz, Fx, Fy)
    return _g_FFcache[tau]


def _pso_particle_worker(x):
    tau, Kz, Kx, Ky = x
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    Kz = float(np.clip(Kz, Kz_range[0], Kz_range[1]))
    Kx = float(np.clip(Kx, Kx_range[0], Kx_range[1]))
    Ky = float(np.clip(Ky, Ky_range[0], Ky_range[1]))

    if tau + end > _g_az_m.shape[1]:
        return 1e6

    Fz, Fx, Fy = _pso_get_FzFxFy(tau)
    alp_comp = _g_alp - (Kz*Fz + Kx*Fx + Ky*Fy) / (2*np.pi*T**2)

    if _g_fitness_kind == "kalman":
        try:
            A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, _g_Pexp, _g_poi)
        except Exception:
            return 1e6
        try:
            _, _, _, _, _, e, _ = kalmanFit_EKF(
                alp_comp, _g_Pexp, T, _g_Q, P_cov0p, _g_sigma_A_sim, _g_sigma_ph,
                A0p, B0p, ph0p)
        except Exception:
            return 1e6
        return float(np.std(e[_g_warmup:]))

    else:  # "curvefit"
        try:
            p0 = [np.mean(_g_Pexp), (np.max(_g_Pexp) - np.min(_g_Pexp)) / 2, 0]
            lb_p = [-1.1, 0, 0]
            ub_p = [1.1, 1.1, 2*np.pi]
            popt, _ = curve_fit(model, alp_comp, _g_Pexp, p0=p0, bounds=(lb_p, ub_p), maxfev=2000)
            P_fit = model(alp_comp, *popt)
            return float(np.std(_g_Pexp - P_fit))
        except Exception:
            return 1e6


def PSO_parallel(fitness_kind, N_particles, M_iter, bounds, alp, P_exp, az_m, ax_m, ay_m,
                  n_jobs=None, warmup=50):
    """Параллельная и кэширующая версия PSO. fitness_kind: "kalman" или
    "curvefit". Оценка популяции на каждой итерации распараллелена по
    процессам (pool.map по всем N_particles сразу), а внутри каждого
    процесса-воркера результат свёртки Fz,Fx,Fy кэшируется по tau и
    переиспользуется, пока пул жив (все M_iter итераций) -- см.
    _pso_worker_init / _pso_get_FzFxFy."""
    c1, c2, w = 2.0, 2.0, 0.9
    dim = len(bounds)
    n_jobs = n_jobs or mp.cpu_count()

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    pos = np.random.uniform(lb, ub, size=(N_particles, dim))
    vel = np.random.uniform(-0.1*(ub-lb), 0.1*(ub-lb), size=(N_particles, dim))

    pbest = pos.copy()
    pbest_fit = np.full(N_particles, np.inf)
    gbest = pos[0].copy()
    gbest_fit = np.inf

    history = []
    n_calls = 0

    print(f"  PSO_parallel: fitness={fitness_kind}, {n_jobs} процессов, "
          f"{N_particles} частиц x {M_iter} итераций")

    ctx = mp.get_context()  # fork на Linux, spawn на Windows -- выбирается автоматически
    with ctx.Pool(processes=n_jobs,
                   initializer=_pso_worker_init,
                   initargs=(alp, P_exp, az_m, ax_m, ay_m, fitness_kind,
                             Q, sigma_A_sim, sigma_ph, poi, warmup)) as pool:
        for it in range(M_iter):
            fits = np.array(pool.map(_pso_particle_worker, pos, chunksize=1))
            n_calls += N_particles

            better = fits < pbest_fit
            pbest[better] = pos[better]
            pbest_fit[better] = fits[better]

            if fits.min() < gbest_fit:
                gbest_fit = float(fits.min())
                gbest = pos[int(fits.argmin())].copy()

            history.append(gbest_fit)

            r1 = np.random.rand(N_particles, dim)
            r2 = np.random.rand(N_particles, dim)
            vel = w*vel + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
            pos = np.clip(pos + vel, lb, ub)

            print(f"Iter {it+1:02d}/{M_iter} | best fitness = {gbest_fit:.6e} | "
                  f"tau={gbest[0]:.1f}, Kz={gbest[1]:.4f}, Kx={gbest[2]:.5f}, Ky={gbest[3]:.5f}")

    tau_opt = int(np.round(gbest[0]))
    Kz_opt, Kx_opt, Ky_opt = gbest[1], gbest[2], gbest[3]
    return tau_opt, Kz_opt, Kx_opt, Ky_opt, gbest_fit, history, n_calls


# ============================================================
# Последовательный (координатный) поиск -- обобщение алгоритма из
# статьи на 4 параметра: по очереди ищем tau, затем Kz, затем Kx,
# затем Ky (каждый раз фиксируя остальные на их текущих лучших
# значениях), с несколькими проходами для сходимости.
# ============================================================

def sequential_coordinate_search(fitness_func, alp, P_exp, az_m, ax_m, ay_m,
                                  tau_range, Kz_range, Kx_range, Ky_range,
                                  n_tau=None, n_Kz=401, n_Kx=201, n_Ky=201,
                                  n_passes=2,
                                  tau_init=500, Kz_init=1.0, Kx_init=0.0, Ky_init=0.0):
    """Этап 1: один раз ищем tau (при Kz,Kx,Ky, зафиксированных на
    начальных значениях). Этап 2: n_passes проходов уточняем
    Kz -> Kx -> Ky при уже найденном tau (tau на этом этапе больше
    не пересчитывается)."""

    method_name = getattr(fitness_func, "__name__", str(fitness_func))
    print(f"--- координатный поиск (fitness = {method_name}) ---")

    tau_vals_full = (np.arange(tau_range[0], tau_range[1] + 1, 1) if n_tau is None
                      else np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int)))
    Kz_vals = np.linspace(*Kz_range, n_Kz)
    Kx_vals = np.linspace(*Kx_range, n_Kx)
    Ky_vals = np.linspace(*Ky_range, n_Ky)

    tau_cur, Kz_cur, Kx_cur, Ky_cur = tau_init, Kz_init, Kx_init, Ky_init
    n_calls = 0
    history = {}

    # --- этап 1/2: поиск tau ---
    print("  этап 1/2: поиск tau")
    fit_tau = np.array([fitness_func(t, Kz_cur, Kx_cur, Ky_cur, alp, P_exp, az_m, ax_m, ay_m)
                         for t in tau_vals_full])
    n_calls += len(tau_vals_full)
    tau_cur = int(tau_vals_full[np.argmin(fit_tau)])
    history["tau"] = (tau_vals_full, fit_tau)
    print(f"    tau_opt = {tau_cur}")

    # --- этап 2/2: поиск Kz -> Kx -> Ky при найденном tau (несколько
    #     проходов для сходимости коэффициентов друг к другу) ---
    print(f"  этап 2/2: поиск Kz -> Kx -> Ky ({n_passes} проход(а/ов))")
    for p in range(n_passes):
        print(f"    проход {p+1}/{n_passes}")

        fit_Kz = np.array([fitness_func(tau_cur, kz, Kx_cur, Ky_cur, alp, P_exp, az_m, ax_m, ay_m)
                            for kz in Kz_vals])
        n_calls += len(Kz_vals)
        Kz_cur = float(Kz_vals[np.argmin(fit_Kz)])
        history[f"pass{p+1}_Kz"] = (Kz_vals, fit_Kz)
        print(f"      Kz_opt = {Kz_cur:.4f}")

        fit_Kx = np.array([fitness_func(tau_cur, Kz_cur, kx, Ky_cur, alp, P_exp, az_m, ax_m, ay_m)
                            for kx in Kx_vals])
        n_calls += len(Kx_vals)
        Kx_cur = float(Kx_vals[np.argmin(fit_Kx)])
        history[f"pass{p+1}_Kx"] = (Kx_vals, fit_Kx)
        print(f"      Kx_opt = {Kx_cur:.5f}")

        fit_Ky = np.array([fitness_func(tau_cur, Kz_cur, Kx_cur, ky, alp, P_exp, az_m, ax_m, ay_m)
                            for ky in Ky_vals])
        n_calls += len(Ky_vals)
        Ky_cur = float(Ky_vals[np.argmin(fit_Ky)])
        history[f"pass{p+1}_Ky"] = (Ky_vals, fit_Ky)
        print(f"      Ky_opt = {Ky_cur:.5f}")

    best_fit = fitness_func(tau_cur, Kz_cur, Kx_cur, Ky_cur, alp, P_exp, az_m, ax_m, ay_m)
    n_calls += 1

    return tau_cur, Kz_cur, Kx_cur, Ky_cur, best_fit, n_calls, history


# ============================================================
# ПОЛНАЯ 4D-СЕТКА, closed-form cos-fit
# Каждая ячейка сама по себе дешёвая (векторизованная линейная подгонка
# по всем N_sim сбросам разом), поэтому распараллеливаем по tau -- для
# каждого tau одна "строка" (все Kz,Kx,Ky) считается векторизованно
# внутри одного процесса.
# ============================================================

def _cf_worker_init(az_m_, ax_m_, ay_m_, cosPhi0_, sinPhi0_, P_exp_,
                     Kz_vals_, Kx_vals_, Ky_vals_):
    # az_m/ax_m/ay_m наследуются дочерним процессом через fork (copy-on-write,
    # без сериализации) -- в задачи передаются только числа tau, а не массивы.
    global _g_az_m, _g_ax_m, _g_ay_m, _g_cosPhi0, _g_sinPhi0, _g_Pexp
    global _g_Kz_vals, _g_Kx_vals, _g_Ky_vals
    _g_az_m, _g_ax_m, _g_ay_m = az_m_, ax_m_, ay_m_
    _g_cosPhi0, _g_sinPhi0, _g_Pexp = cosPhi0_, sinPhi0_, P_exp_
    _g_Kz_vals, _g_Kx_vals, _g_Ky_vals = Kz_vals_, Kx_vals_, Ky_vals_


def _cf_row_worker(args):
    it, tau = args
    Fz = keff * (_g_az_m[:, tau:tau + end] @ weight_vec)
    Fx = keff * (_g_ax_m[:, tau:tau + end] @ weight_vec)
    Fy = keff * (_g_ay_m[:, tau:tau + end] @ weight_vec)

    row_fit = np.full((len(_g_Kz_vals), len(_g_Kx_vals), len(_g_Ky_vals)), 1e6)
    for iz, Kzv in enumerate(_g_Kz_vals):
        for ix, Kxv in enumerate(_g_Kx_vals):
            for iy, Kyv in enumerate(_g_Ky_vals):
                ang = Kzv*Fz + Kxv*Fx + Kyv*Fy
                c = _g_cosPhi0*np.cos(ang) + _g_sinPhi0*np.sin(ang)   # cos(Phi0 - ang)
                s = _g_sinPhi0*np.cos(ang) - _g_cosPhi0*np.sin(ang)   # sin(Phi0 - ang)
                row_fit[iz, ix, iy] = closed_form_cosfit_rms(c, s, _g_Pexp)
    return it, tau, row_fit


def full_grid_search_curvefit_4d(alp, P_exp, az_m, ax_m, ay_m,
                                  tau_range, Kz_range, Kx_range, Ky_range,
                                  n_tau=21, n_Kz=11, n_Kx=5, n_Ky=5, n_jobs=None):
    tau_vals = np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int))
    Kz_vals = np.linspace(*Kz_range, n_Kz)
    Kx_vals = np.linspace(*Kx_range, n_Kx)
    Ky_vals = np.linspace(*Ky_range, n_Ky)

    valid = tau_vals + end <= az_m.shape[1]
    tau_vals = tau_vals[valid]

    Phi0 = 2*np.pi*alp*T**2
    cosPhi0 = np.cos(Phi0)
    sinPhi0 = np.sin(Phi0)

    fitness = np.full((len(tau_vals), len(Kz_vals), len(Kx_vals), len(Ky_vals)), 1e6)
    n_calls = int(len(tau_vals) * len(Kz_vals) * len(Kx_vals) * len(Ky_vals))
    n_jobs = n_jobs or mp.cpu_count()

    print(f"  full_grid (cos-fit): {len(tau_vals)} строк x {n_Kz*n_Kx*n_Ky} ячеек, "
          f"{n_jobs} процессов")
    ctx = mp.get_context()  # fork на Linux, spawn на Windows -- выбирается автоматически
    with ctx.Pool(processes=n_jobs,
                   initializer=_cf_worker_init,
                   initargs=(az_m, ax_m, ay_m, cosPhi0, sinPhi0, P_exp,
                             Kz_vals, Kx_vals, Ky_vals)) as pool:
        tasks = list(enumerate(tau_vals))
        for it, tau, row_fit in pool.imap_unordered(_cf_row_worker, tasks):
            fitness[it] = row_fit
            print(f"  full_grid (cos-fit): строка {it+1}/{len(tau_vals)} (tau={tau}) готова")

    idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    return (int(tau_vals[idx[0]]), float(Kz_vals[idx[1]]),
            float(Kx_vals[idx[2]]), float(Ky_vals[idx[3]]),
            float(fitness[idx]), n_calls,
            {"tau_vals": tau_vals, "Kz_vals": Kz_vals,
             "Kx_vals": Kx_vals, "Ky_vals": Ky_vals, "fitness": fitness})


def _kf_worker_init(alp_, P_exp_, Fz_all_, Fx_all_, Fy_all_,
                     Kz_vals_, Kx_vals_, Ky_vals_, Q_, sigma_A_sim_, sigma_ph_,
                     poi_, T_, warmup_):
    # Большие массивы (alp, P_exp, предвычисленные Fz/Fx/Fy) наследуются
    # дочерним процессом через fork (copy-on-write) -- в задачи передаются
    # только 4 целых/индексных числа на ячейку, а не сами массивы.
    global _g_alp, _g_Pexp, _g_Fz, _g_Fx, _g_Fy
    global _g_Kz_vals, _g_Kx_vals, _g_Ky_vals
    global _g_Q, _g_sigma_A_sim, _g_sigma_ph, _g_poi, _g_T, _g_warmup
    _g_alp, _g_Pexp = alp_, P_exp_
    _g_Fz, _g_Fx, _g_Fy = Fz_all_, Fx_all_, Fy_all_
    _g_Kz_vals, _g_Kx_vals, _g_Ky_vals = Kz_vals_, Kx_vals_, Ky_vals_
    _g_Q, _g_sigma_A_sim, _g_sigma_ph = Q_, sigma_A_sim_, sigma_ph_
    _g_poi, _g_T, _g_warmup = poi_, T_, warmup_


def _kf_cell_worker(args):
    it, iz, ix, iy = args
    Kz = _g_Kz_vals[iz]; Kx = _g_Kx_vals[ix]; Ky = _g_Ky_vals[iy]
    # F уже предвычислены на tau -- компенсация здесь без повторного
    # матричного умножения на az_m/ax_m/ay_m (только линейная комбинация)
    alp_comp = _g_alp - (Kz*_g_Fz[it] + Kx*_g_Fx[it] + Ky*_g_Fy[it]) / (2*np.pi*_g_T**2)
    try:
        A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, _g_Pexp, _g_poi)
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, _g_Pexp, _g_T, _g_Q, P_cov0p, _g_sigma_A_sim, _g_sigma_ph,
            A0p, B0p, ph0p)
        fit = float(np.std(e[_g_warmup:]))
    except Exception:
        fit = 1e6
    return it, iz, ix, iy, fit


def full_grid_search_kalman_4d(alp, P_exp, az_m, ax_m, ay_m,
                                tau_range, Kz_range, Kx_range, Ky_range,
                                n_tau=11, n_Kz=6, n_Kx=3, n_Ky=3,
                                n_jobs=None, warmup=50):
    """То же самое, но fitness = std innovations EKF -- значительно
    дороже, чем cos-fit (полный прогон фильтра на каждую ячейку).
    Ускорено двумя способами: (1) Fz,Fx,Fy считаются один раз на каждый
    tau и переиспользуются для всех Kz,Kx,Ky этого tau, а не пересчитываются
    заново для каждой ячейки; (2) сами ячейки (tau,Kz,Kx,Ky) считаются
    параллельно в отдельных процессах (по числу ядер CPU по умолчанию).
    Вместе с ускоренным EKF (см. kalmanFit_EKF) это позволяет брать
    заметно более плотную сетку за то же время."""
    tau_vals = np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int))
    Kz_vals = np.linspace(*Kz_range, n_Kz)
    Kx_vals = np.linspace(*Kx_range, n_Kx)
    Ky_vals = np.linspace(*Ky_range, n_Ky)

    valid = tau_vals + end <= az_m.shape[1]
    tau_vals = tau_vals[valid]
    if len(tau_vals) == 0:
        raise ValueError("full_grid_search_kalman_4d: нет допустимых tau в пределах окна az_m")

    # --- предвычисляем Fz, Fx, Fy один раз на каждый tau ---
    Fz_all = np.empty((len(tau_vals), len(alp)))
    Fx_all = np.empty((len(tau_vals), len(alp)))
    Fy_all = np.empty((len(tau_vals), len(alp)))
    for it, tau in enumerate(tau_vals):
        Fz_all[it] = keff * (az_m[:, tau:tau + end] @ weight_vec)
        Fx_all[it] = keff * (ax_m[:, tau:tau + end] @ weight_vec)
        Fy_all[it] = keff * (ay_m[:, tau:tau + end] @ weight_vec)

    fitness = np.full((len(tau_vals), len(Kz_vals), len(Kx_vals), len(Ky_vals)), np.nan)
    tasks = [(it, iz, ix, iy)
             for it in range(len(tau_vals))
             for iz in range(len(Kz_vals))
             for ix in range(len(Kx_vals))
             for iy in range(len(Ky_vals))]
    n_calls = len(tasks)
    n_jobs = n_jobs or mp.cpu_count()
    chunksize = max(1, n_calls // (n_jobs * 8))

    print(f"  full_grid (Kalman): {n_calls} ячеек, {n_jobs} процессов, "
          f"chunksize={chunksize}")

    ctx = mp.get_context()  # fork на Linux, spawn на Windows -- выбирается автоматически
    done = 0
    report_every = max(1, n_calls // 20)
    with ctx.Pool(processes=n_jobs,
                   initializer=_kf_worker_init,
                   initargs=(alp, P_exp, Fz_all, Fx_all, Fy_all,
                             Kz_vals, Kx_vals, Ky_vals,
                             Q, sigma_A_sim, sigma_ph, poi, T, warmup)) as pool:
        for it, iz, ix, iy, fit in pool.imap_unordered(_kf_cell_worker, tasks,
                                                        chunksize=chunksize):
            fitness[it, iz, ix, iy] = fit
            done += 1
            if done % report_every == 0 or done == n_calls:
                print(f"    full_grid (Kalman): {done}/{n_calls} ячеек готово")

    idx = np.unravel_index(np.nanargmin(fitness), fitness.shape)
    return (int(tau_vals[idx[0]]), float(Kz_vals[idx[1]]),
            float(Kx_vals[idx[2]]), float(Ky_vals[idx[3]]),
            float(fitness[idx]), n_calls,
            {"tau_vals": tau_vals, "Kz_vals": Kz_vals,
             "Kx_vals": Kx_vals, "Ky_vals": Ky_vals, "fitness": fitness})


# ============================================================
# Оценка результата и отчёт
# ============================================================

def evaluate_with_kalman(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m):
    alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)
    A0e, B0e, ph0e, P_cov0e, _ = init_values(alp_comp, P_exp, poi)
    P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(
        alp_comp, P_exp, T, Q, P_cov0e, sigma_A_sim, sigma_ph, A0e, B0e, ph0e)
    return alp_comp, ph, e, en


def g_error_stats(ph, g_sim, warmup=50, bias_warn_threshold=1e-3):
    g_est = ph / keff / T**2
    diff = g_est[warmup:] - g_sim[warmup:]

    bias = np.median(diff)
    if abs(bias) > bias_warn_threshold:
        print(f"⚠ ВНИМАНИЕ: bias(g) = {bias*1e8:.3f} µGal подозрительно велик — "
              f"похоже на ошибку разрешения порядка фринджа")

    rms_debiased = np.sqrt(np.mean((diff - bias)**2))
    rms_total = np.sqrt(np.mean(diff**2))
    return rms_total, rms_debiased, bias, g_est


def report_case(label, tau, Kz, Kx, Ky, ph_est, e_arr, g_sim, elapsed_s, n_calls, warmup=50):
    std_e = np.std(e_arr[warmup:])
    rms_total, rms_debiased, bias_g, _ = g_error_stats(ph_est, g_sim, warmup=warmup)
    print(f"{label:24s}{tau:7d}{Kz:9.4f}{Kx:10.5f}{Ky:10.5f}{std_e:14.3e}"
          f"{rms_total*1e8:14.3f}{bias_g*1e8:13.3f}{rms_debiased*1e8:14.3f}"
          f"{elapsed_s:10.2f}{n_calls:12d}")
    return std_e, rms_total, bias_g, rms_debiased


if __name__ == "__main__":

    # --- параметры и запуск симуляции (внутри __main__, чтобы на Windows
    #     spawn не пересчитывал это заново в каждом дочернем процессе) ---
    N_sim = 1000
    alp_amount = 201
    delay = 500
    Kz = 0.9
    Kx = 0.003
    Ky = 0.003

    (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
     sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m, ax_m, ay_m) = simul_acc(
        N_sim, alp_amount, delay, Kz, Kx, Ky)

    # kalman fit
    Q = np.diag([dA_sim**2, dB_sim**2, dph_sim**2])
    poi = alp_amount
    A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
    sigma_ph = np.sqrt(sigma_ph_vibr**2 + dph_sim**2*0)
    sigma_A = sigma_A_sim

    warmup = 50
    results = {}  # label -> (tau, Kz, Kx, Ky, elapsed_s, n_calls)

    print("=== PSO with Kalman-filter fitness (4D: tau, Kz, Kx, Ky), параллельный + кэш ===")
    t0 = time.perf_counter()
    tau_kf, Kz_kf, Kx_kf, Ky_kf, fit_kf, hist_kf, calls_kf = PSO_parallel(
        "kalman", N_particles, M_iter,
        [tau_range, Kz_range, Kx_range, Ky_range],
        alp, P_sim_noise, az_m, ax_m, ay_m,
        n_jobs=N_JOBS, warmup=warmup)
    results["PSO (Kalman)"] = (tau_kf, Kz_kf, Kx_kf, Ky_kf, time.perf_counter() - t0, calls_kf)

    print("\n=== PSO with curve_fit fitness (4D), параллельный + кэш ===")
    t0 = time.perf_counter()
    tau_cf, Kz_cf, Kx_cf, Ky_cf, fit_cf, hist_cf, calls_cf = PSO_parallel(
        "curvefit", N_particles, M_iter,
        [tau_range, Kz_range, Kx_range, Ky_range],
        alp, P_sim_noise, az_m, ax_m, ay_m,
        n_jobs=N_JOBS)
    results["PSO (curve_fit)"] = (tau_cf, Kz_cf, Kx_cf, Ky_cf, time.perf_counter() - t0, calls_cf)

    # Какую fitness-функцию использует координатный поиск -- задаётся здесь
    # одной строкой, метка результата подхватывает имя функции автоматически.
    seq_fitness_func = curvefit_fitness   # или kalman_fitness

    print(f"\n=== Sequential coordinate search (tau -> Kz -> Kx -> Ky, "
          f"fitness = {seq_fitness_func.__name__}) ===")
    t0 = time.perf_counter()
    tau_sq, Kz_sq, Kx_sq, Ky_sq, fit_sq, calls_sq, hist_sq = sequential_coordinate_search(
        seq_fitness_func, alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=SEQ_N_TAU, n_Kz=SEQ_N_KZ, n_Kx=SEQ_N_KX, n_Ky=SEQ_N_KY, n_passes=SEQ_N_PASSES,
        tau_init=SEQ_TAU_INIT, Kz_init=SEQ_KZ_INIT, Kx_init=SEQ_KX_INIT, Ky_init=SEQ_KY_INIT)
    results[f"Sequential ({seq_fitness_func.__name__})"] = (
        tau_sq, Kz_sq, Kx_sq, Ky_sq, time.perf_counter() - t0, calls_sq)

    print("\n=== FULL 4D grid, closed-form cos-fit (coarse) ===")
    t0 = time.perf_counter()
    tau_g1, Kz_g1, Kx_g1, Ky_g1, fit_g1, calls_g1, hist_g1 = full_grid_search_curvefit_4d(
        alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=GRID_CF_N_TAU, n_Kz=GRID_CF_N_KZ, n_Kx=GRID_CF_N_KX, n_Ky=GRID_CF_N_KY,
        n_jobs=N_JOBS)
    results["Full grid (cos-fit)"] = (tau_g1, Kz_g1, Kx_g1, Ky_g1, time.perf_counter() - t0, calls_g1)

    print("\n=== FULL 4D grid, Kalman fitness (very coarse) ===")
    t0 = time.perf_counter()
    tau_g2, Kz_g2, Kx_g2, Ky_g2, fit_g2, calls_g2, hist_g2 = full_grid_search_kalman_4d(
        alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=GRID_KF_N_TAU, n_Kz=GRID_KF_N_KZ, n_Kx=GRID_KF_N_KX, n_Ky=GRID_KF_N_KY,
        n_jobs=N_JOBS, warmup=warmup)
    results["Full grid (Kalman)"] = (tau_g2, Kz_g2, Kx_g2, Ky_g2, time.perf_counter() - t0, calls_g2)

    # --- единая метрика (EKF innovations + точность g) для всех методов ---
    print("\n===================================== Сравнение =====================================")
    print(f"{'':24s}{'tau':>7s}{'Kz':>9s}{'Kx':>10s}{'Ky':>10s}{'std(e)':>14s}"
          f"{'RMS_tot(g)':>14s}{'bias(g)':>13s}{'RMS_deb(g)':>14s}{'time,s':>10s}{'calls':>12s}")
    print(f"{'true':24s}{delay:7d}{Kz:9.4f}{Kx:10.5f}{Ky:10.5f}")

    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        alp_v, ph_v, e_v, en_v = evaluate_with_kalman(
            tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        report_case(label, tau_v, Kz_v, Kx_v, Ky_v, ph_v, e_v, g_sim, t_v, n_calls_v, warmup)

    # --- convergence curves (PSO) ---
    plt.figure()
    plt.plot(hist_kf, label="Kalman fitness")
    plt.plot(hist_cf, label="curve_fit fitness")
    plt.xlabel("PSO iteration")
    plt.ylabel("best fitness (своя шкала для каждого метода)")
    plt.title("PSO convergence: Kalman vs curve_fit (4D)")
    plt.legend()

    # --- ход координатного поиска: tau -- единственный скан (этап 1),
    #     Kz/Kx/Ky -- последний проход этапа 2 ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    last_pass = SEQ_N_PASSES
    plot_specs = [
        ("tau", tau_sq, hist_sq["tau"]),
        ("Kz", Kz_sq, hist_sq[f"pass{last_pass}_Kz"]),
        ("Kx", Kx_sq, hist_sq[f"pass{last_pass}_Kx"]),
        ("Ky", Ky_sq, hist_sq[f"pass{last_pass}_Ky"]),
    ]
    for ax, (name, opt_val, (xv, fv)) in zip(axes, plot_specs):
        ax.plot(xv, fv)
        ax.axvline(opt_val, color="k", ls="--", label=f"{name}_opt={opt_val:.4g}")
        ax.set_xlabel(name)
        ax.set_ylabel("fitness")
        ax.set_title(f"Координатный поиск: {name}")
        ax.legend()
    fig.tight_layout()

    # --- итоговые innovations ---
    plt.figure()
    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        _, _, e_v, _ = evaluate_with_kalman(tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        plt.plot(e_v, label=label, alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel("EKF innovation e")
    plt.title("Итоговые невязки EKF при разных способах компенсации (3 оси)")
    plt.legend()

    # --- ошибка определения g в физических единицах ---
    plt.figure()
    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        _, ph_v, _, _ = evaluate_with_kalman(tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        plt.plot((ph_v/keff/T**2 - g_sim)*1e8, label=label, alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel(r"$g_{est} - g_{sim}$, µGal")
    plt.title("Ошибка определения g при разных способах компенсации (3 оси)")
    plt.legend()

    plt.show()