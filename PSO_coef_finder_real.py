import time
import multiprocessing as mp
import numpy as np
from scipy.optimize import curve_fit


# ---- физические константы ----
lm = 780e-9
keff = 4 * np.pi / lm

T = 10e-3     # длительность одного плеча интерферометрической последовательности
ty = 20e-6    # длительность импульса

# Число отсчётов акселерометра на сброс -- фиксировано аппаратурой,
# не зависит от T_RP.
N_RP = 16384


def fa(t):
    """Весовая функция чувствительности интерферометра к ускорению."""
    if 0 < t <= T + 2 * ty:
        return t
    elif T + 2 * ty < t <= 2 * T + 4 * ty:
        return 2 * (T + 2 * ty) - t
    else:
        return 0.0


fa_v = np.vectorize(fa, otypes=[float])


def _build_weight_vec(t_step):
    """fa(t) и весовой вектор для трапециевидного интегрирования
    F_vib(tau) = keff * Kz * (az_window @ weight_vec)."""
    tr = np.linspace(0, 2 * T + 4 * ty, round((2 * T + 4 * ty) / t_step))
    n = len(tr)
    fa_t = fa_v(tr)
    w = np.full(n, t_step)
    w[0] *= 0.5
    w[-1] *= 0.5
    return fa_t, fa_t * w, n


# ---- модель фринджа ----

def model(alp, A, B, ph):
    return A - B * np.cos(2 * np.pi * alp * T ** 2 - ph)


def init_values(alp, P_exp, poi):
    """Начальный cos-fit по первым poi точкам."""
    alp0 = alp[:poi]
    P0 = P_exp[:poi]

    p0 = [(P0.max() + P0.min()) / 2, (P0.max() - P0.min()) / 2, 0.0]
    lb = [-1.1, 0.0, 0.0]
    ub = [1.1, 1.1, 2 * np.pi]
    popt, pcov = curve_fit(model, alp0, P0, p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt

    sigma_A = np.std(P0 - model(alp0, A0, B0, ph0))
    return A0, B0, ph0, pcov, sigma_A


def _sigma_from_pcov(pcov, poi):
    """
    Переводит ковариацию параметров начального cos-fit в оценку std шума
    на одну точку: pcov[i,i] -- дисперсия среднего по poi точкам, т.е.
    примерно в poi раз меньше дисперсии шума на одну точку:

        sigma_single_i = sqrt(pcov[i,i] * poi)

    Возвращает (sigma_A_auto, sigma_ph_auto) -- то, что sigma_mode="auto"
    кладёт в R матрицу EKF.
    """
    sigma_A_auto = float(np.sqrt(pcov[0, 0] * poi))
    sigma_ph_auto = float(np.sqrt(pcov[2, 2] * poi))
    return sigma_A_auto, sigma_ph_auto


# ---- EKF (состояние [A,B,ph], одно скалярное измерение) ----

def kalmanFit_EKF(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0):
    N = len(alp)
    A = np.empty(N); B = np.empty(N); ph = np.empty(N)
    P_m = np.empty(N); e = np.empty(N); en = np.empty(N)
    P_cov = np.empty((N, 3, 3))

    x = np.array([A0, B0, ph0], dtype=float)
    P = np.array(P_cov0, dtype=float).copy()
    I3 = np.eye(3)
    two_pi_T2 = 2 * np.pi * T * T

    A[0], B[0], ph[0] = x
    P_cov[0] = P
    P_m[0] = x[0] - x[1] * np.cos(two_pi_T2 * alp[0] - x[2])

    for i in range(1, N):
        P = P + Q

        Phi = two_pi_T2 * alp[i] - x[2]
        cosPhi = np.cos(Phi)
        sinPhi = np.sin(Phi)
        zpred = x[0] - x[1] * cosPhi
        P_m[i] = zpred
        e[i] = P_exp[i] - zpred

        H = np.array([1.0, -cosPhi, -x[1] * sinPhi])
        R = sigma_A ** 2 + x[1] ** 2 * sinPhi ** 2 * sigma_ph ** 2

        PHT = P @ H
        S = H @ PHT + R
        K = PHT / S

        x = x + K * e[i]
        I_KH = I3 - np.outer(K, H)
        P = I_KH @ P @ I_KH.T + np.outer(K, K) * R

        en[i] = e[i] / np.sqrt(S)
        A[i], B[i], ph[i] = x
        P_cov[i] = P

    return P_m, A, B, ph, P_cov, e, en


# ---- компенсация вибрации (ось z) и оконный cos-fit ----

def compensate_alp(tau, Kz, alp, az_m, weight_vec, win_len):
    """alp_comp = alp - Kz * Fz(tau) / (2*pi*T^2)."""
    az_window = az_m[:, tau:tau + win_len]
    Fz = keff * (az_window @ weight_vec)
    return alp - Kz * Fz / (2 * np.pi * T ** 2)


def sigma_ph_from_K(Kz, sigma_a_acc, fa_t, t_step):
    """Аналитическая оценка фазового шума от шума акселерометра (не путать
    с sigma_mode="auto"/"manual" -- используется только в compare_sigma_ph)."""
    return keff * t_step * sigma_a_acc * np.sqrt(np.sum(fa_t ** 2)) * abs(Kz)


def compare_sigma_ph(alp_comp, P_exp, poi, Kz, sigma_a_acc, fa_t, t_step):
    """
    Сравнение двух оценок фазового шума в точке (Kz, alp_comp):
      sigma_ph_accel -- вклад только от шума акселерометра;
      sigma_ph_auto  -- полный шум из ковариации начального cos-fit
                        (то же, что подставляется в EKF при sigma_mode="auto").
    """
    _, _, _, pcov, _ = init_values(alp_comp, P_exp, poi)
    sigma_ph_accel = sigma_ph_from_K(Kz, sigma_a_acc, fa_t, t_step)
    _, sigma_ph_auto = _sigma_from_pcov(pcov, poi)
    return sigma_ph_accel, sigma_ph_auto


def make_windows(N, win):
    """Непересекающиеся окна, покрывающие весь набор без остатка."""
    n_win = max(1, N // win)
    return np.linspace(0, N, n_win + 1).astype(int)


def windowed_cosfit(phase, y, edges):
    """
    Независимый cos-fit в каждом окне (векторизовано через lstsq).
    phase = 2*pi*alp_comp*T^2, модель: A + c1*cos(phase) + c2*sin(phase).
    Возвращает (общий RMS невязки по всем окнам, массив фаз ph по окнам).
    """
    c = np.cos(phase); s = np.sin(phase)
    rss = 0.0
    ph = np.empty(len(edges) - 1)
    for k in range(len(edges) - 1):
        sl = slice(edges[k], edges[k + 1])
        X = np.column_stack((np.ones(edges[k + 1] - edges[k]), c[sl], s[sl]))
        coef, *_ = np.linalg.lstsq(X, y[sl], rcond=None)
        r = y[sl] - X @ coef
        rss += float(r @ r)
        ph[k] = np.arctan2(-coef[2], -coef[1])
    return float(np.sqrt(rss / (edges[-1] - edges[0]))), ph


# ---- fitness-функции ----

def _kalman_fitness_from_Fz(Fz, Kz, alp, P_exp, Q, poi, warmup,
                             sigma_mode, sigma_A_manual, sigma_ph_manual):
    """
    sigma_mode = "manual" -> sigma_A, sigma_ph заданы вручную.
    sigma_mode = "auto"   -> sigma_A, sigma_ph берутся из ковариации
                              начального cos-fit (оценивается на каждом (tau, Kz)).
    """
    alp_comp = alp - Kz * Fz / (2 * np.pi * T ** 2)
    try:
        A0, B0, ph0, pcov, _ = init_values(alp_comp, P_exp, poi)
    except Exception:
        return 1e6

    if sigma_mode == "manual":
        sigma_A = sigma_A_manual
        sigma_ph = sigma_ph_manual
    elif sigma_mode == "auto":
        sigma_A, sigma_ph = _sigma_from_pcov(pcov, poi)
    else:
        raise ValueError("sigma_mode must be 'auto' or 'manual'")

    try:
        # pcov -- та же матрица, что используется как начальная ковариация
        # состояния EKF (P_cov0), независимо от sigma_mode
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, P_exp, T, Q, pcov, sigma_A, sigma_ph, A0, B0, ph0)
        return float(np.std(e[warmup:]))
    except Exception:
        return 1e6


def _windowed_fitness_from_Fz(Fz, Kz, alp, P_exp, win_edges):
    alp_comp = alp - Kz * Fz / (2 * np.pi * T ** 2)
    try:
        rms, _ = windowed_cosfit(2 * np.pi * alp_comp * T ** 2, P_exp, win_edges)
        return rms
    except Exception:
        return 1e6


def kalman_fitness(tau, Kz, alp, P_exp, az_m, Q, poi, warmup,
                    sigma_mode, sigma_A_manual, sigma_ph_manual,
                    tau_bounds, Kz_bounds, weight_vec, win_len):
    tau = int(np.clip(round(tau), *tau_bounds))
    Kz = float(np.clip(Kz, *Kz_bounds))
    if tau + win_len > az_m.shape[1]:
        return 1e6
    Fz = keff * (az_m[:, tau:tau + win_len] @ weight_vec)
    return _kalman_fitness_from_Fz(Fz, Kz, alp, P_exp, Q, poi, warmup,
                                    sigma_mode, sigma_A_manual, sigma_ph_manual)


def windowed_fitness(tau, Kz, alp, P_exp, az_m, win_edges,
                      tau_bounds, Kz_bounds, weight_vec, win_len):
    tau = int(np.clip(round(tau), *tau_bounds))
    Kz = float(np.clip(Kz, *Kz_bounds))
    if tau + win_len > az_m.shape[1]:
        return 1e6
    Fz = keff * (az_m[:, tau:tau + win_len] @ weight_vec)
    return _windowed_fitness_from_Fz(Fz, Kz, alp, P_exp, win_edges)


# ---- PSO (2D: tau, Kz), распараллелен по частицам, с кэшем Fz(tau) ----

def _pso_worker_init(alp_, P_exp_, az_m_, kind_, Q_, poi_, warmup_,
                      sigma_mode_, sigma_A_manual_, sigma_ph_manual_,
                      win_edges_, tau_bounds_, Kz_bounds_, weight_vec_, win_len_):
    global _W
    _W = dict(alp=alp_, P_exp=P_exp_, az_m=az_m_, kind=kind_, Q=Q_,
              poi=poi_, warmup=warmup_,
              sigma_mode=sigma_mode_, sigma_A_manual=sigma_A_manual_,
              sigma_ph_manual=sigma_ph_manual_,
              win_edges=win_edges_, tau_bounds=tau_bounds_, Kz_bounds=Kz_bounds_,
              weight_vec=weight_vec_, win_len=win_len_,
              Fz_cache={})


def _pso_particle_worker(x):
    tau, Kz = x
    tau = int(np.clip(round(tau), *_W['tau_bounds']))
    Kz = float(np.clip(Kz, *_W['Kz_bounds']))
    win_len = _W['win_len']
    if tau + win_len > _W['az_m'].shape[1]:
        return 1e6

    if tau not in _W['Fz_cache']:
        _W['Fz_cache'][tau] = keff * (_W['az_m'][:, tau:tau + win_len] @ _W['weight_vec'])
    Fz = _W['Fz_cache'][tau]

    if _W['kind'] == 'kalman':
        return _kalman_fitness_from_Fz(
            Fz, Kz, _W['alp'], _W['P_exp'], _W['Q'], _W['poi'], _W['warmup'],
            _W['sigma_mode'], _W['sigma_A_manual'], _W['sigma_ph_manual'])
    else:
        return _windowed_fitness_from_Fz(Fz, Kz, _W['alp'], _W['P_exp'], _W['win_edges'])


def PSO_parallel(kind, N_particles, M_iter, bounds, alp, P_exp, az_m,
                  Q, poi, warmup, sigma_mode, sigma_A_manual, sigma_ph_manual,
                  weight_vec, win_len, win_edges=None, n_jobs=None, verbose=True):
    """
    kind = "kalman"   -> fitness = std(EKF innovation)
    kind = "windowed" -> fitness = RMS оконного cos-fit (sigma_* не используются)
    bounds = [tau_bounds, Kz_bounds]
    """
    c1, c2, w = 2.0, 2.0, 0.9
    dim = 2
    n_jobs = n_jobs or mp.cpu_count()
    tau_bounds, Kz_bounds = bounds

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    pos = np.random.uniform(lb, ub, size=(N_particles, dim))
    vel = np.random.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=(N_particles, dim))

    pbest = pos.copy()
    pbest_fit = np.full(N_particles, np.inf)
    gbest = pos[0].copy()
    gbest_fit = np.inf
    history = []
    n_calls = 0

    ctx = mp.get_context()
    with ctx.Pool(processes=n_jobs, initializer=_pso_worker_init,
                   initargs=(alp, P_exp, az_m, kind, Q, poi, warmup,
                             sigma_mode, sigma_A_manual, sigma_ph_manual,
                             win_edges, tau_bounds, Kz_bounds,
                             weight_vec, win_len)) as pool:
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
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, lb, ub)

            if verbose:
                print(f"  iter {it + 1:02d}/{M_iter} | best fitness = {gbest_fit:.4e} "
                      f"| tau={gbest[0]:.1f}, Kz={gbest[1]:.4f}")

    tau_opt = int(np.round(gbest[0]))
    Kz_opt = float(gbest[1])
    return tau_opt, Kz_opt, gbest_fit, history, n_calls


# ---- верхнеуровневая функция: оба метода сразу, единый отчёт ----

def fit_vibration_compensation(alp, P_exp, az_m, *,
                                T_RP,
                                tau_bounds, Kz_bounds,
                                dA_model, dB_model, dph_model,
                                poi, warmup, win_size,
                                sigma_mode="auto",
                                sigma_A_manual=None, sigma_ph_manual=None,
                                sigma_a_acc=None,
                                N_particles=30, M_iter=30,
                                n_jobs=None):
    """
    Подбор (tau, Kz) двумя методами по некомпенсированным alp, P_exp, az_m.

    T_RP : длительность реализации акселерометра на сброс, с. Вместе с
        глобальным N_RP (число отсчётов на сброс, фиксировано аппаратурой)
        задаёт шаг дискретизации t_step = T_RP / N_RP, из которого строится
        весовой вектор интегрирования fa(t) (см. _build_weight_vec).

    dA_model, dB_model, dph_model : шаги блуждания A, B, ph между сбросами
        (диффузия модели фринджа). Из них один раз собирается матрица шумов
        процесса EKF: Q = diag(dA_model^2, dB_model^2, dph_model^2) -- одна
        и та же для всех кандидатов (tau, Kz) в PSO.

    poi      : число первых точек для начального cos-fit (init_values).
    win_size : размер окна для оконного cos-fit.

    sigma_mode : "auto" | "manual"
        "manual" -- sigma_A = sigma_A_manual, sigma_ph = sigma_ph_manual
                    (оба параметра обязательны; sigma_a_acc здесь не
                    используется, нужен только опционально для диагностики).
        "auto"   -- sigma_A, sigma_ph оцениваются из ковариации начального
                    cos-fit на каждом (tau, Kz) (весь реальный шум).

    sigma_a_acc : шум акселерометра, м/с^2 -- независимо от sigma_mode,
        если задан, в найденной точке (tau, Kz) дополнительно считается
        sigma_ph_accel (оценка фазового шума только от акселерометра) для
        сравнения с sigma_ph_auto / sigma_ph_manual.

    Возвращает
    ----------
    dict:
        {"kalman":   {"tau": int, "Kz": float, "std_e": float},
         "windowed": {"tau": int, "Kz": float, "std_e": float}}
    и, если sigma_a_acc задан, дополнительно "sigma_ph_accel", "sigma_ph_auto"
    для каждого метода.

    Для метода "windowed" std_e -- это std(EKF innovation) в НАЙДЕННОЙ этим
    методом точке (tau, Kz), для честного сравнения обоих методов по одной
    и той же метрике (с тем же sigma_mode).
    """
    if sigma_mode == "manual" and (sigma_A_manual is None or sigma_ph_manual is None):
        raise ValueError("sigma_mode='manual' требует и sigma_A_manual, и sigma_ph_manual")

    t_step = T_RP / N_RP
    fa_t, weight_vec, win_len = _build_weight_vec(t_step)

    N_sim = len(alp)
    win_edges = make_windows(N_sim, win_size)
    bounds = [tau_bounds, Kz_bounds]
    Q = np.diag([dA_model ** 2, dB_model ** 2, dph_model ** 2])

    def _sigma_compare(tau, Kz):
        """sigma_ph_accel / sigma_ph_auto в точке (tau, Kz); None, если
        sigma_a_acc не задан."""
        if sigma_a_acc is None:
            return None
        alp_comp = compensate_alp(tau, Kz, alp, az_m, weight_vec, win_len)
        return compare_sigma_ph(alp_comp, P_exp, poi, Kz, sigma_a_acc, fa_t, t_step)

    print(f"=== PSO, fitness = std(EKF innovation), sigma_mode='{sigma_mode}' ===")
    t0 = time.perf_counter()
    tau_k, Kz_k, std_e_k, hist_k, calls_k = PSO_parallel(
        "kalman", N_particles, M_iter, bounds, alp, P_exp, az_m,
        Q=Q, poi=poi, warmup=warmup,
        sigma_mode=sigma_mode, sigma_A_manual=sigma_A_manual, sigma_ph_manual=sigma_ph_manual,
        weight_vec=weight_vec, win_len=win_len, win_edges=None, n_jobs=n_jobs)
    print(f"  -> tau={tau_k}, Kz={Kz_k:.5f}, std(e)={std_e_k:.4e} "
          f"[{calls_k} вычислений, {time.perf_counter() - t0:.1f} с]")
    cmp_k = _sigma_compare(tau_k, Kz_k)
    if cmp_k is not None:
        print(f"  sigma_ph: accel_only={cmp_k[0]:.4e} рад, auto(pcov)={cmp_k[1]:.4e} рад")
    print()

    print("=== PSO, fitness = RMS оконного cos-fit ===")
    t0 = time.perf_counter()
    tau_w, Kz_w, rms_w, hist_w, calls_w = PSO_parallel(
        "windowed", N_particles, M_iter, bounds, alp, P_exp, az_m,
        Q=Q, poi=poi, warmup=warmup,
        sigma_mode=sigma_mode, sigma_A_manual=sigma_A_manual, sigma_ph_manual=sigma_ph_manual,
        weight_vec=weight_vec, win_len=win_len, win_edges=win_edges, n_jobs=n_jobs)
    std_e_w = kalman_fitness(tau_w, Kz_w, alp, P_exp, az_m, Q, poi, warmup,
                              sigma_mode, sigma_A_manual, sigma_ph_manual,
                              tau_bounds, Kz_bounds, weight_vec, win_len)
    print(f"  -> tau={tau_w}, Kz={Kz_w:.5f}, RMS(win)={rms_w:.4e}, std(e)={std_e_w:.4e} "
          f"[{calls_w} вычислений, {time.perf_counter() - t0:.1f} с]")
    cmp_w = _sigma_compare(tau_w, Kz_w)
    if cmp_w is not None:
        print(f"  sigma_ph: accel_only={cmp_w[0]:.4e} рад, auto(pcov)={cmp_w[1]:.4e} рад")
    print()

    results = {
        "kalman": {"tau": tau_k, "Kz": Kz_k, "std_e": std_e_k},
        "windowed": {"tau": tau_w, "Kz": Kz_w, "std_e": std_e_w},
    }
    if cmp_k is not None:
        results["kalman"]["sigma_ph_accel"], results["kalman"]["sigma_ph_auto"] = cmp_k
    if cmp_w is not None:
        results["windowed"]["sigma_ph_accel"], results["windowed"]["sigma_ph_auto"] = cmp_w
    return results


def main():
    # Реальные данные гравиметра:
    #   alp   : (N_sim,)        -- некомпенсированная частота АОМ на сброс
    #   P_exp : (N_sim,)        -- нормированный сигнал интерферометра [-1.1..1.1]
    #   az_m  : (N_sim, N_acc)  -- отсчёты вертикального акселерометра на сброс,
    #                               az_m.shape[1] >= tau_bounds[1] + win_len
    data = np.load("gravimeter_data.npz")
    alp, P_exp, az_m = data["alp"], data["P_exp"], data["az_m"]

    T_RP = 33e-3          # длительность реализации акселерометра на сброс, с
    tau_bounds = (0, 5000)  # диапазон поиска tau, отсчёты акселерометра
    Kz_bounds = (0.0, 1.5)  # диапазон поиска Kz (безразмерный)

    # Шаги блуждания модели фринджа между сбросами -- подбираются вручную.
    dA_model = 5e-3
    dB_model = 5e-3
    dph_model = 1e-3

    # Источник sigma_A, sigma_ph для R матрицы EKF:
    #   "auto"   -- рекомендуется, считаются из ковариации начального cos-fit
    #   "manual" -- задаются вручную через sigma_A_manual / sigma_ph_manual
    sigma_mode = "manual"
    sigma_A_manual = 7e-3
    sigma_ph_manual = 5e-3

    # Опционально, чисто диагностический параметр: если задан, в найденной
    # точке (tau, Kz) дополнительно печатается вклад шума акселерометра
    # (sigma_ph_accel) для сравнения с sigma_ph_auto / sigma_ph_manual.
    sigma_a_acc = 3e-5

    # poi и win_size выбираются под конкретную серию, значения ниже -- пример.
    poi = 200
    win_size = 20
    warmup = 120
    N_particles = 30
    M_iter = 30

    results = fit_vibration_compensation(
        alp, P_exp, az_m,
        T_RP=T_RP,
        tau_bounds=tau_bounds, Kz_bounds=Kz_bounds,
        dA_model=dA_model, dB_model=dB_model, dph_model=dph_model,
        poi=poi, warmup=warmup, win_size=win_size,
        sigma_mode=sigma_mode,
        sigma_A_manual=sigma_A_manual, sigma_ph_manual=sigma_ph_manual,
        sigma_a_acc=sigma_a_acc,
        N_particles=N_particles, M_iter=M_iter)

    print("=== Итог ===")
    for method, r in results.items():
        line = f"{method:10s}: tau={r['tau']:5d}  Kz={r['Kz']:.5f}  std(e)={r['std_e']:.4e}"
        if "sigma_ph_accel" in r:
            line += f"  | sigma_ph: accel_only={r['sigma_ph_accel']:.3e}, auto={r['sigma_ph_auto']:.3e}"
        print(line)


if __name__ == "__main__":
    main()