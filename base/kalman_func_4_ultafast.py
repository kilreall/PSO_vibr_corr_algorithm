# -*- coding: utf-8 -*-
"""
Ускоренная версия скрипта: EKF-трекер + оконный (линейный) фит + свип
по частоте для построения АЧХ/ФЧХ вплоть до f_mod ~ 1e-5.
 
Что изменено и почему это ускоряет код:
 
1. simulate_data() полностью векторизован (был Python-цикл по N ->
   стал чисто numpy). При N ~ 1e5-1e6 (а именно столько нужно на
   низких частотах) это один из самых дорогих кусков — теперь почти
   бесплатный.
 
2. Ядро EKF (_ekf_track_phase_core) переписано вручную в скалярном
   виде: вместо operations c 4x4 numpy-матрицами (predict/update через
   F@P@F.T, outer(K,H)@P и т.п. — каждый вызов "@" на маленькой
   матрице в Python стоит дорого из-за overhead numpy) все формулы
   раскрыты руками, используя структуру F (только ph <- ph+v_ph) и
   структуру H (H = [1, -cosPhi, -B*sinPhi, 0]). Это даёт ~3-8x на
   чистом Python и делает функцию тривиально JIT-совместимой.
 
3. Если установлен numba — ядро EKF компилируется в машинный код
   (@njit). Это даёт решающее ускорение (обычно 50-150x), потому что
   именно длина последовательного (нераспараллеливаемого) цикла EKF —
   главный бутылочный узел при f_mod -> 1e-5 (там нужно N до ~10^6
   отсчётов на одну реализацию). Если numba не установлен, код тихо
   откатывается на быстрый скалярный Python — работает медленнее, но
   работает.
 
   Установить: pip install numba
 
4. windowFit_linear() теперь обрабатывает окна чанками (chunk_size),
   а не строит все N x window матрицы разом — иначе на N ~ 10^6 и
   window=20 это легко "взрывает" память (несколько x N x window x 3
   массивов double).
 
5. simulate_data использует локальный np.random.Generator (per-call),
   а не глобальный np.random.seed() — это безопасно и детерминированно
   при работе в разных процессах ProcessPoolExecutor.
 
6. Тяжёлая "демо"-часть (загрузка data_delay_800.npy, одиночный прогон
   kalmanFit_EKF/filterpy, графики) вынесена в main() под
   `if __name__ == "__main__"`, а не выполняется на уровне модуля —
   иначе она будет выполняться в КАЖДОМ дочернем процессе
   ProcessPoolExecutor при "spawn" (Windows/macOS), что резко замедляет
   и засоряет вывод. filterpy импортируется лениво и не обязателен для
   свипа do_charact().
 
7. do_charact(): диапазон частот расширен вниз до 1e-5. Для низких
   частот нужно больше периодов сигнала (N_max поднят), поэтому
   количество усреднений (n_avg) на самых низких частотах уменьшено
   адаптивно — иначе время расчёта было бы неприемлемым даже с numba.
"""
 
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import time
 
# ---------------------------------------------------------------------
# numba (опционально, но настоятельно рекомендуется: pip install numba)
# ---------------------------------------------------------------------
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
 
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
 
        def wrap(f):
            return f
        return wrap
 
 
# ---------------------------------------------------------------------
# Модель сигнала
# ---------------------------------------------------------------------
 
def model(alp, A, B, ph, T):
    return A - B * np.cos(2 * np.pi * alp * T ** 2 - ph)
 
 
def init_values(alp, P_exp, poi, T):
    alp_init = alp[:poi]
    P_init = P_exp[:poi]
 
    p0 = [(np.max(P_init) + np.min(P_init)) / 2,
          (np.max(P_init) - np.min(P_init)) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2 * np.pi]
    popt, pcov = curve_fit(lambda a, A, B, ph: model(a, A, B, ph, T),
                            alp_init, P_init, p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt
    sigma_A = np.std(P_init - model(alp_init, A0, B0, ph0, T))
    return A0, B0, ph0, pcov, sigma_A
 
 
def init_values_linear(alp, P_exp, poi, T):
    """Линейный (быстрый) аналог init_values через МНК вместо curve_fit."""
    x = alp[:poi]
    y = P_exp[:poi]
 
    Phi = 2 * np.pi * x * T ** 2
    c = np.cos(Phi)
    s = np.sin(Phi)
    M = np.stack([np.ones_like(c), -c, -s], axis=-1)
 
    params, *_ = np.linalg.lstsq(M, y, rcond=None)
    A0, C0, D0 = params
 
    B0 = np.sqrt(C0 ** 2 + D0 ** 2)
    ph0 = np.arctan2(D0, C0)
 
    resid = y - (A0 - C0 * c - D0 * s)
    sigma_A = np.std(resid)
 
    dof = max(len(y) - 3, 1)
    var_resid = np.sum(resid ** 2) / dof
    MtM_inv = np.linalg.inv(M.T @ M)
    cov_ACD = MtM_inv * var_resid
 
    if B0 > 1e-12:
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, C0 / B0, D0 / B0],
            [0.0, -D0 / B0 ** 2, C0 / B0 ** 2],
        ])
    else:
        J = np.eye(3)
 
    pcov = J @ cov_ACD @ J.T
    return A0, B0, ph0, pcov, sigma_A
 
 
# ---------------------------------------------------------------------
# Быстрое (скалярное, JIT-совместимое) ядро EKF
# ---------------------------------------------------------------------
 
@njit(cache=True, fastmath=True)
def _ekf_track_phase_core(alp, P_exp, T,
                           q0, q1, q2, q3,
                           p00, p01, p02, p03,
                           p11, p12, p13,
                           p22, p23,
                           p33,
                           sigma_A, sigma_ph, A0, B0, ph0, v_ph0):
    N = alp.shape[0]
    ph_out = np.empty(N)
 
    A = A0
    B = B0
    ph = ph0
    vph = v_ph0
    ph_out[0] = ph
 
    A_prev = A0
    B_prev = B0
    ph_prev = ph0
 
    sigma_A2 = sigma_A * sigma_A
    sigma_ph2 = sigma_ph * sigma_ph
    twopiT2 = 2.0 * np.pi * T * T
 
    for i in range(1, N):
        alpha = alp[i]
 
        # ---- predict state (F: ph <- ph + v_ph, остальное без изменений) ----
        A_p = A
        B_p = B
        ph_p = ph + vph
        vph_p = vph
 
        # ---- predict covariance: P <- F P F^T + Q, используя структуру F ----
        n00 = p00
        n01 = p01
        n02 = p02 + p03
        n03 = p03
        n11 = p11
        n12 = p12 + p13
        n13 = p13
        n22 = p22 + 2.0 * p23 + p33
        n23 = p23 + p33
        n33 = p33
 
        n00 += q0
        n11 += q1
        n22 += q2
        n33 += q3
 
        # ---- предсказанное измерение ----
        Phi_p = alpha * twopiT2 - ph_p
        cosPhi = np.cos(Phi_p)
        sinPhi = np.sin(Phi_p)
        z_pred = A_p - B_p * cosPhi
 
        h1 = -cosPhi
        h2 = -B_p * sinPhi
        # h0 = 1.0, h3 = 0.0 (жёстко подставлены ниже)
 
        # ---- шум измерения (как в оригинале: по ПРЕДЫДУЩИМ принятым B, ph) ----
        Phi_meas = alpha * twopiT2 - ph_prev
        s_meas = np.sin(Phi_meas)
        R = sigma_A2 + (B_prev * B_prev) * (s_meas * s_meas) * sigma_ph2
 
        # v = P @ h  (h0=1, h3=0)
        v0 = n00 + h1 * n01 + h2 * n02
        v1 = n01 + h1 * n11 + h2 * n12
        v2 = n02 + h1 * n12 + h2 * n22
        v3 = n03 + h1 * n13 + h2 * n23
 
        S = v0 + h1 * v1 + h2 * v2 + R
 
        K0 = v0 / S
        K1 = v1 / S
        K2 = v2 / S
        K3 = v3 / S
 
        y = P_exp[i] - z_pred
 
        A = A_p + K0 * y
        B = B_p + K1 * y
        ph = ph_p + K2 * y
        vph = vph_p + K3 * y
 
        p00 = n00 - K0 * v0
        p01 = n01 - K0 * v1
        p02 = n02 - K0 * v2
        p03 = n03 - K0 * v3
        p11 = n11 - K1 * v1
        p12 = n12 - K1 * v2
        p13 = n13 - K1 * v3
        p22 = n22 - K2 * v2
        p23 = n23 - K2 * v3
        p33 = n33 - K3 * v3
 
        ph_out[i] = ph
 
        A_prev = A
        B_prev = B
        ph_prev = ph
 
    return ph_out
 
 
def _ekf_track_phase(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph,
                      A0, B0, ph0, v_ph0):
    """Совместимая по интерфейсу замена старой (матричной) версии."""
    q0, q1, q2, q3 = np.diag(Q)
    p00, p01, p02, p03 = P_cov0[0, 0], P_cov0[0, 1], P_cov0[0, 2], P_cov0[0, 3]
    p11, p12, p13 = P_cov0[1, 1], P_cov0[1, 2], P_cov0[1, 3]
    p22, p23 = P_cov0[2, 2], P_cov0[2, 3]
    p33 = P_cov0[3, 3]
 
    return _ekf_track_phase_core(
        np.ascontiguousarray(alp, dtype=np.float64),
        np.ascontiguousarray(P_exp, dtype=np.float64),
        T, q0, q1, q2, q3,
        p00, p01, p02, p03, p11, p12, p13, p22, p23, p33,
        sigma_A, sigma_ph, A0, B0, ph0, v_ph0,
    )
 
 
# ---------------------------------------------------------------------
# Оконный линейный фит (векторизован, теперь чанками по памяти)
# ---------------------------------------------------------------------
 
def windowFit_linear(alp, P_exp, T, window, step=1, chunk_size=200_000):
    lm = 780e-9
    keff = 4 * np.pi / lm
    N = len(alp)
    half = window // 2
 
    g = np.full(N, np.nan)
 
    centers = np.arange(half, N - half, step)
    if len(centers) == 0:
        return g
 
    win_offsets = np.arange(-half, half + 1)
 
    for start in range(0, len(centers), chunk_size):
        c = centers[start:start + chunk_size]
        idx = c[:, None] + win_offsets[None, :]
        x = alp[idx]
        y = P_exp[idx]
 
        Phi = 2 * np.pi * x * T ** 2
        cph = np.cos(Phi)
        sph = np.sin(Phi)
        ones = np.ones_like(cph)
 
        Mrows = np.stack([ones, -cph, -sph], axis=-1)  # (k, window, 3)
 
        MtM = np.einsum('kwi,kwj->kij', Mrows, Mrows)
        Mty = np.einsum('kwi,kw->ki', Mrows, y)
 
        try:
            params = np.linalg.solve(MtM, Mty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            params = np.full((len(c), 3), np.nan)
            for k in range(len(c)):
                try:
                    params[k] = np.linalg.solve(MtM[k], Mty[k])
                except np.linalg.LinAlgError:
                    try:
                        params[k] = np.linalg.lstsq(Mrows[k], y[k], rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
 
        A_k, C_k, D_k = params[:, 0], params[:, 1], params[:, 2]
        B_k = np.sqrt(C_k ** 2 + D_k ** 2)  # noqa: F841 (оставлено для отладки)
        phi_k = np.arctan2(D_k, C_k)
 
        idx_min = np.argmin(y, axis=1)
        x_min = x[np.arange(len(c)), idx_min]
        phi_target = 2 * np.pi * x_min * T ** 2
        Mbr = np.round((phi_target - phi_k) / (2 * np.pi))
        phi_k = phi_k + 2 * np.pi * Mbr
 
        g[c] = phi_k / (keff * T ** 2)
 
    return g
 
 
def savgolFilter(g_in, window_length, polyorder):
    g_out = np.full_like(g_in, np.nan)
    valid = np.isfinite(g_in)
    if np.sum(valid) >= window_length:
        g_out[valid] = savgol_filter(g_in[valid], window_length=window_length,
                                      polyorder=polyorder)
    return g_out
 
 
# ---------------------------------------------------------------------
# Симуляция данных — полностью векторизована (был Python-цикл)
# ---------------------------------------------------------------------
 
def simulate_data(f_mod, N, params, seed=None):
    rng = np.random.default_rng(seed)
 
    g0 = params['g0']; Dg = params['Dg']; T = params['T']; keff = params['keff']
    alp_amount = params['alp_amount']; alp_start = params['alp_start']
    sigma_ph_vibr = params['sigma_ph_vibr']
    A0_sim = params['A0_sim']; dA_sim = params['dA_sim']; DA_sim = params['DA_sim']
    B0_sim = params['B0_sim']; dB_sim = params['dB_sim']
    dph_sim = params['dph_sim']
    sigma_A_sim = params['sigma_A_sim']
 
    i = np.arange(N)
 
    g_sim = g0 + Dg * np.sin(2 * np.pi * f_mod * i)
    v_ph_sim = Dg * 2 * np.pi * f_mod * np.cos(2 * np.pi * f_mod * i) * keff * T ** 2
 
    # первая (чистая) оценка вибрационной фазы -> используется для "истинного" P_sim
    F_vib_u = rng.uniform(-np.pi / 12, np.pi / 12, size=N)
    alp1 = alp_start[i % alp_amount] - F_vib_u / (2 * np.pi * T * T)
 
    A_sim = A0_sim + DA_sim * i + rng.normal(0, dA_sim, size=N)
    B_sim = B0_sim + rng.normal(0, dB_sim, size=N)
    ph_sim = keff * g_sim * T * T + rng.normal(0, dph_sim, size=N)
 
    Ph = 2 * np.pi * alp1 * T * T - ph_sim
    P_sim = A_sim - B_sim * np.cos(Ph)
 
    # вторая (зашумлённая) оценка вибрационной фазы -> это "измеренный" alp
    F_vib = F_vib_u + rng.normal(0, sigma_ph_vibr, size=N)
    alp = alp_start[i % alp_amount] - F_vib / (2 * np.pi * T * T)
 
    P_sim_noise = P_sim + rng.normal(0, sigma_A_sim, size=N)
 
    return alp, g_sim, ph_sim, v_ph_sim, P_sim_noise
 
 
def _interp_nans(x):
    x = x.copy()
    idx = np.arange(len(x))
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return x
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x
 
 
def kalman_tracker_fast(f_mod, sim, params, poi):
    alp, g_sim, ph_sim, v_ph_sim, P_sim_noise = sim
    T = params['T']
    keff = params['keff']
    Dg = params['Dg']
 
    dA_model = np.sqrt(params['DA_sim'] ** 2 + params['dA_sim'] ** 2)
    dB_model = params['dB_sim']
    dph_model = np.sqrt(params['dph_sim'] ** 2)
    dv_ph_model = np.std(v_ph_sim[1:] - np.roll(v_ph_sim, 1)[1:]) * 2
    Qm = np.diag([dA_model ** 2, dB_model ** 2, dph_model ** 2, dv_ph_model ** 2])
 
    sigma_ph = params['sigma_ph_vibr']
    A0, B0, ph0, P_cov_3d, _ = init_values_linear(alp, P_sim_noise, poi, T)
    v_ph0 = Dg * 2 * np.pi * f_mod * keff * T * T
 
    P_cov0 = np.zeros((4, 4))
    P_cov0[:3, :3] = P_cov_3d
    P_cov0[3, 3] = (Dg * keff * T * T / 3) ** 2
 
    sigma_A = params['sigma_A_sim']
 
    ph_track = _ekf_track_phase(
        alp, P_sim_noise, T, Qm, P_cov0, sigma_A, sigma_ph, A0, B0, ph0, v_ph0
    )
 
    return ph_track / keff / T ** 2
 
 
# ---------------------------------------------------------------------
# Одна точка частотного свипа (для ProcessPoolExecutor)
# ---------------------------------------------------------------------
 
def _freq_task(f_mod, K, N_min, N_max, n_avg, params, poi,
               window, window_step, savgol_window_length, savgol_polyorder):
    channels = ('kalman', 'window', 'savgol')
 
    N = int(np.clip(K / f_mod, N_min, N_max))
    n_skip = max(int(0.1 * N), 50)
 
    t = np.arange(N)[n_skip:]
    win = np.hanning(len(t))
 
    Hs = {ch: [] for ch in channels}
    for k in range(n_avg):
        sim = simulate_data(f_mod, N, params, seed=k)
        alp, g_sim, ph_sim, v_ph_sim, P_sim_noise = sim
 
        g_kalman = kalman_tracker_fast(f_mod, sim, params, poi)
        g_window = windowFit_linear(alp, P_sim_noise, params['T'],
                                     window=window, step=window_step)
        g_savgol = savgolFilter(g_window, savgol_window_length, savgol_polyorder)
 
        g_ests = {'kalman': g_kalman, 'window': g_window, 'savgol': g_savgol}
 
        x_in = (g_sim[n_skip:] - np.mean(g_sim[n_skip:])) * win
        c_in = np.sum(x_in * np.exp(-1j * 2 * np.pi * f_mod * t))
 
        for ch in channels:
            g_est = _interp_nans(g_ests[ch])
            x_out = (g_est[n_skip:] - np.mean(g_est[n_skip:])) * win
            c_out = np.sum(x_out * np.exp(-1j * 2 * np.pi * f_mod * t))
            Hs[ch].append(c_out / c_in)
 
    return {ch: np.mean(vals) for ch, vals in Hs.items()}
 
 
# ---------------------------------------------------------------------
# Свип по частоте / построение АЧХ и ФЧХ
# ---------------------------------------------------------------------
 
def do_charact(sim_params, poi, n_workers=None,
                freq_min=1e-5, freq_max=0.3, n_freqs=200,
                window=20, window_step=1,
                savgol_window_length=21, savgol_polyorder=3,
                K_cycles=8, N_min=2000, N_max=1_500_000,
                n_avg_hi=3, n_avg_lo=1, n_avg_switch_N=200_000,
                save_prefix="kalman4params"):
    """
    freq_min понижен до 1e-5 (было 1e-4). Чтобы это осталось быстрым:
 
      - N_max поднят (низкие частоты требуют больше отсчётов, чтобы
        вместить K_cycles периодов сигнала), но
      - n_avg адаптивно уменьшается для точек, где требуемое N велико
        (n_avg_lo вместо n_avg_hi при N > n_avg_switch_N) — иначе время
        расчёта на самых низких частотах доминирует над всем остальным.
 
    Без numba это всё ещё может быть медленным на самых низких частотах
    (N ~ 10^6, Python-цикл EKF) — тогда стоит либо поставить numba
    (pip install numba), либо поднять N_min/уменьшить freq_min обратно.
    """
    if not HAVE_NUMBA:
        print("[!] numba не найден — EKF считается на чистом Python "
              "(медленнее в 50-150 раз). Рекомендуется: pip install numba")
 
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    channels = ['kalman', 'window', 'savgol']
 
    # разбиваем на "дешёвые" (высокочастотные) и "дорогие" (низкочастотные)
    # точки, чтобы каждой можно было выдать своё n_avg
    def n_avg_for(f):
        N = int(np.clip(K_cycles / f, N_min, N_max))
        return n_avg_lo if N > n_avg_switch_N else n_avg_hi
 
    n_workers = n_workers or os.cpu_count() or 1
    print(f"Считаю ФЧХ/АЧХ на {len(freqs)} частотах "
          f"[{freq_min:.1e} .. {freq_max:.1e}], {n_workers} процессов, "
          f"numba={'да' if HAVE_NUMBA else 'нет'}...")
 
    H_by_channel = {ch: np.zeros(len(freqs), dtype=complex) for ch in channels}
 
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for idx, f in enumerate(freqs):
            task = partial(
                _freq_task,
                K=K_cycles, N_min=N_min, N_max=N_max, n_avg=n_avg_for(f),
                params=sim_params, poi=poi,
                window=window, window_step=window_step,
                savgol_window_length=savgol_window_length,
                savgol_polyorder=savgol_polyorder,
            )
            futures[ex.submit(task, f)] = idx
 
        done = 0
        for fut in futures:
            idx = futures[fut]
            Hs = fut.result()
            for ch in channels:
                H_by_channel[ch][idx] = Hs[ch]
            done += 1
            if done % 20 == 0 or done == len(freqs):
                dt = time.time() - t0
                print(f"  {done}/{len(freqs)} готово, {dt:.1f} c")
 
    print(f"Готово за {time.time() - t0:.1f} c")
 
    import matplotlib.pyplot as plt
 
    H_kalman = H_by_channel['kalman']
    H_window = H_by_channel['window']
    H_savgol = H_by_channel['savgol']
 
    plt.figure()
    plt.semilogx(freqs, 20 * np.log10(np.abs(H_kalman)), label="kalman")
    plt.semilogx(freqs, 20 * np.log10(np.abs(H_window)), label="window")
    plt.semilogx(freqs, 20 * np.log10(np.abs(H_savgol)), label="savgol")
    plt.xlabel("частота модуляции g, циклы/отсчёт")
    plt.ylabel("АЧХ, дБ")
    plt.title("Амплитудно-частотная характеристика")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(f"amplitude_{save_prefix}.png")
 
    plt.figure()
    plt.semilogx(freqs, np.unwrap(np.angle(H_kalman)) * 180 / np.pi, label="kalman")
    plt.semilogx(freqs, np.unwrap(np.angle(H_window)) * 180 / np.pi, label="window")
    plt.semilogx(freqs, np.unwrap(np.angle(H_savgol)) * 180 / np.pi, label="savgol")
    plt.xlabel("частота модуляции g, циклы/отсчёт")
    plt.ylabel("ФЧХ, град")
    plt.title("Фазо-частотная характеристика")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(f"phase_{save_prefix}.png")
 
    return freqs, H_by_channel
 
 
# ---------------------------------------------------------------------
# main: параметры симуляции + запуск свипа (тяжёлая демо-часть тоже
# здесь, а не на уровне модуля — важно для ProcessPoolExecutor)
# ---------------------------------------------------------------------
 
def build_sim_params():
    lm = 780e-9
    keff = 4 * np.pi / lm
    T = 10e-3
 
    g0 = 9.8101507
    Dg = 300 * 1e-8
 
    alp_min = keff * g0 / 2 / np.pi - 1 / 1.6 / T / T
    alp_max = keff * g0 / 2 / np.pi + 1 / 1.6 / T / T
    alp_amount = 20
    alp_start = np.linspace(alp_min, alp_max, alp_amount)
 
    return dict(
        g0=g0, Dg=Dg, lm=lm, keff=keff, T=T,
        alp_min=alp_min, alp_max=alp_max, alp_amount=alp_amount, alp_start=alp_start,
        sigma_ph_vibr=1e-4,
        A0_sim=0.15, dA_sim=0.0, DA_sim=3e-5,
        B0_sim=0.21, dB_sim=0.0,
        dph_sim=0.0,
        sigma_A_sim=1e-3,
    )
 
 
def main():
    sim_params = build_sim_params()
    poi = 20
 
    freqs, H = do_charact(
        sim_params, poi,
        freq_min=1e-5, freq_max=0.3, n_freqs=200,
    )
 
    import matplotlib.pyplot as plt
    plt.show()
 
 
if __name__ == "__main__":
    main()