# Один файл: (1) генерация данных симуляции фринджа с вибрационным ускорением
# ТОЛЬКО по оси z; (2) оценка sigma_A и sigma_ph по невязкам фита в бинах alp
# и сравнение с заложенными в генерацию sigma_A_sim и sigma_ph_vibr.
#
# Метод: фазу Phi = 2*pi*T^2*alp_fit - ph (alp_fit -- скомпенсированный alp)
# берём по модулю pi и делим на N_BINS бинов (это бины скомпенсированного alp
# по модулю периода фринджа). В каждом бине дисперсия невязки
#     sigma_P^2 = sigma_A^2 + sigma_ph^2 * <B^2 sin^2(Phi)>_bin
# -- линейная регрессия по бинам даёт sigma_A^2 (свободный член) и
# sigma_ph^2 (наклон).
#
# Фит всегда идёт по alp, скомпенсированному по ИЗМЕРЕННОМУ ускорению az_m
# (с шумом акселерометра), как и с реальными данными.
#
# Запуск:  python sim_z_generation_and_compare.py

import numpy as np
from scipy.optimize import curve_fit, nnls
from scipy.signal import butter, filtfilt

# ============================================================
# НАСТРОЙКИ ПОДГОНКИ (curve_fit) -- менять здесь.
# Параметры генерации и физики находятся ниже (константы физики) и в вызове
# simul_acc в блоке `if __name__ == "__main__"`.
# ============================================================

POI = 200                # по скольким ПЕРВЫМ точкам каждого фринджа делать подгонку (4 <= POI <= alp_amount)
N_RUNS = 20              # число независимых прогонов simul_acc
FRINGES_PER_RUN = 10     # число фринджей в одном прогоне (N_sim = alp_amount * FRINGES_PER_RUN)
SEED_MC = 123            # сид Монте-Карло (сиды вибрации и шумов выводятся из него)
N_BINS = 10              # число бинов по фазе Phi mod pi (для оценки sigma_A, sigma_ph)

FIT_LB = [-1.1, 0.0, 0.0]            # нижние границы [A, B, ph]
FIT_UB = [1.1, 1.1, 2*np.pi]         # верхние границы [A, B, ph]
FIT_G0_PRIOR = 9.8101507             # априорное g для выбора ветви 2π (может отличаться от G0 генерации)
FIT_P0_PH_FROM_PRIOR = True          # True: p0[ph] = keff*g0_prior*T^2 mod 2π, False: p0[ph] = 0
FIT_RESOLVE_BRANCH = True            # выбирать ветвь 2π по FIT_G0_PRIOR (как в init_values)
FIT_KZ = None                        # Kz, используемый при компенсации в фите (None -> Kz генерации)


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

G0 = 9.8101507
SIGMA_A_ACC = 3e-5   # шум акселерометра [м/с^2]


def fa(t):
    if 0 < t <= T + 2*ty:
        return t
    elif T + 2*ty < t <= 2*T + 4*ty:
        return 2*(T + 2*ty) - t
    else:
        return 0


fat_v = np.vectorize(fa, otypes=[float])

t_range = np.linspace(0, 2*T + 4*ty, round((2*T + 4*ty)/t_step))
end = len(t_range)
fa_t = fat_v(t_range)

# веса трапеций: интеграл = keff * (a_window @ weight_vec)
_trapz_w = np.ones(end) * t_step
_trapz_w[0] *= 0.5
_trapz_w[-1] *= 0.5
weight_vec = fa_t * _trapz_w

F_LOW_PHYS = 1.0 / (2*T)  # справочно


# ============================================================
# Целевые ASD [м/с^2/sqrt(Гц)] -- только ось z
# ============================================================

_ASD_TABLES = {
    'mooring': {
        'f_nodes': np.array([1e-3, 3e-2, 0.1, 0.3, 1.0, 2.0, 4.0, 6.0,
                             10.0, 20.0, 40.0, 100.0, 300.0, 1000.0]),
        'z': np.array([5.5e-5, 5.5e-5, 5.0e-5, 3.0e-5, 8.0e-5, 3.0e-4,
                       8.0e-4, 2.8e-3, 1.1e-3, 3.2e-4, 1.3e-4, 3.2e-5,
                       4.2e-6, 3.0e-7]),
        'res_lines': [],
    },
    'sailing': {
        'f_nodes': np.array([1e-3, 1e-2, 3e-2, 0.1, 0.15, 0.3, 0.6, 1.0,
                             2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 70.0, 100.0,
                             200.0, 400.0, 1000.0]),
        'z': np.array([1.2e-4, 1.3e-4, 5.0e-4, 1.0e-1, 1.9e-1, 2.7e-2,
                       3.0e-3, 6.0e-4, 1.2e-4, 6.0e-5, 4.5e-4, 1.2e-3,
                       1.0e-4, 8.0e-5, 8.0e-5, 2.0e-5, 8.0e-6, 1.0e-5,
                       5.0e-7]),
        # (f0, добротность, относительная амплитуда горба)
        'res_lines': [(6.0, 10, 1.2), (12.0, 12, 0.8), (24.0, 15, 0.5),
                      (45.0, 15, 0.4)],
    },
}


# ============================================================
# Генерация вибрационного ускорения по оси z
# ============================================================

def _synthesize_from_asd(freqs, target_asd, N, fs, rng):
    """Реализация временного ряда с заданным ОДНОСТОРОННИМ ASD
    методом случайных фаз (Timmer & Koenig)."""
    n_freq = len(freqs)
    amp = target_asd * np.sqrt(N * fs / 2.0)

    phase = rng.uniform(0, 2*np.pi, n_freq)
    Xf = amp * np.exp(1j*phase)

    Xf[0] = amp[0] * rng.normal() * np.sqrt(2.0)
    if N % 2 == 0:
        Xf[-1] = amp[-1] * rng.normal() * np.sqrt(2.0)

    return np.fft.irfft(Xf, n=N)


def gen_vibration_trace_z(N, dt, state='mooring', seed=None,
                          hp_cutoff=0.01, hp_order=2,
                          aa_cutoff_factor=1.0,
                          platform_atten_db=-80.0,
                          return_components=False):
    """
    Одна реализация вибрационного ускорения по оси z.

    N, dt             : длина реализации и шаг по времени
    state             : 'mooring' | 'sailing'
    seed              : сид ГПСЧ (None -- без фиксации)
    hp_cutoff         : срез аппаратного ВЧ-фильтра акселерометра [Гц]
                        (держать <= 0.01-0.05 Гц)
    hp_order          : порядок Баттерворта
    aa_cutoff_factor  : множитель к верхнему узлу ASD для anti-alias спада
    platform_atten_db : ослабление [дБ] сырой вибрации корпуса до оси
                        интерферометра (0 дБ -- сырая вибрация)
    return_components : вернуть также {'raw', 'target_asd', 'freqs'}

    Возвращает a(t) [м/с^2] (или (a(t), components)).
    """
    if state not in _ASD_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs_safe = freqs.copy()
    freqs_safe[0] = freqs_safe[1] * 0.5  # избегаем log(0)

    table = _ASD_TABLES[state]
    f_nodes = table['f_nodes']
    asd_nodes = table['z']
    res_lines = table['res_lines']

    if hp_cutoff > 0.05:
        print(f"[gen_vibration_trace_z] ВНИМАНИЕ: hp_cutoff={hp_cutoff} Гц "
              f"может резать реальный НЧ вибрационный сигнал "
              f"(пик ASD ~0.1-1 Гц)")

    # плавная форма ASD: интерполяция в log-log
    log_target = np.interp(np.log10(freqs_safe),
                           np.log10(f_nodes), np.log10(asd_nodes))
    target_asd = 10 ** log_target

    # узкополосные резонансные горбы
    for f0, q, rel_amp in res_lines:
        target_asd *= 1.0 + rel_amp * np.exp(-0.5*((freqs_safe - f0)/(f0/q))**2)

    # изоляция платформы
    target_asd = target_asd * (10 ** (platform_atten_db / 20.0))

    # мягкий anti-alias спад у верхнего узла
    f_aa = f_nodes[-1] * aa_cutoff_factor
    target_asd *= 1.0 / (1.0 + (freqs_safe / f_aa)**6)

    raw = _synthesize_from_asd(freqs_safe, target_asd, N, fs, rng)

    # ВЧ-фильтр акселерометра, нулевая фаза
    wn = hp_cutoff / (fs/2)
    if 0 < wn < 1:
        b, a_f = butter(hp_order, wn, btype='high')
        a_trace = filtfilt(b, a_f, raw)
    else:
        a_trace = raw

    if return_components:
        return a_trace, {'raw': raw, 'target_asd': target_asd,
                         'freqs': freqs_safe}
    return a_trace


def sens_integral(delay, a):
    """Сырой (без Kz) интеграл чувствительности по окну интерферометра."""
    seg = a[delay:delay + end]
    if len(seg) != end:
        raise ValueError(f"delay={delay}: окно [{delay}, {delay+end}) выходит "
                         f"за пределы реализации длиной {len(a)}")
    return keff * float(seg @ weight_vec)


# ============================================================
# Симуляция данных (ось z)
# ============================================================

def simul_acc(N_sim, alp_amount, delay, Kz,
              vib_state='mooring', hp_cutoff=0.01, platform_atten_db=-80.0,
              seed_vib=None, seed_noise=None, verbose=True,
              return_extra=False, dA_step=5e-3, dB_step=5e-3, DA_extra=0.0):
    """
    Генерирует данные для симуляции с вибрацией только по оси z.

    N_sim             : число сбросов
    alp_amount        : число точек развёртки чирпа alp в одном фрингe
    delay             : смещение окна интерферометра в записи ускорения [отсчёты]
    Kz                : коэффициент связи вибрации z с фазой
    vib_state         : 'mooring' | 'sailing'
    hp_cutoff         : срез ВЧ-фильтра акселерометра [Гц]
    platform_atten_db : ослабление сырой вибрации [дБ]
    seed_vib          : сид генератора вибрации (None -- без фиксации)
    seed_noise        : сид остальных шумов/дрейфов (None -- без фиксации)
    dA_step, dB_step  : шаг блуждания A и B за сброс (std); 0 -- A и B постоянны
    DA_extra          : добавка к возвращаемому dA_sim (sqrt(dA^2 + DA^2)),
                        на генерацию A_sim не влияет; для редких тестов, обычно 0

    return_extra      : если True, дополнительно вернуть словарь с "истинными"
                        внутренними величинами: F_vibz (фазовый вклад вибрации
                        [рад], без шума акселерометра), ph_sim, Ph

    Возвращает кортеж:
      alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
      sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m  [, extra]
    """
    rng_noise = np.random.default_rng(seed_noise)
    rng_vib = np.random.default_rng(seed_vib)

    # --- диагностика масштаба: несжатая (K=1) фаза от вибрации на сброс ---
    if verbose:
        def _raw_phase_estimate(atten_db):
            probe = gen_vibration_trace_z(N_RP, t_step, state=vib_state,
                                          hp_cutoff=hp_cutoff,
                                          platform_atten_db=atten_db)
            return keff * float(np.sum(weight_vec * probe[:end]))

        ph_raw_used = _raw_phase_estimate(platform_atten_db)
        ph_raw_hull = _raw_phase_estimate(0.0)
        print(f"[simul_acc] vib_state={vib_state}, "
              f"platform_atten_db={platform_atten_db} dB")
        print(f"[simul_acc] несжатая (K=1) фаза от вибрации за 1 сброс: "
              f"~{ph_raw_used:.2f} рад (выбранное ослабление); "
              f"~{ph_raw_hull:.2f} рад при 0 дБ")
        if abs(ph_raw_used) > 5:
            print("[simul_acc] ВНИМАНИЕ: несжатая фаза > 5 рад -- велика "
                  "вероятность промаха по порядку фринджа; сделайте "
                  "platform_atten_db более отрицательным")

    # --- дрейф g (процесс Орнштейна-Уленбека) ---
    g0 = G0
    g_sim = np.zeros(N_sim); g_sim[-1] = g0
    drift_corr = 200
    Dg_drift = 500e-8
    theta_drift = 1 - np.exp(-1.0/drift_corr)
    sigma_g_drift = Dg_drift * np.sqrt(theta_drift*(2 - theta_drift))

    # --- сетка alp ---
    alp_min = keff*g0/2/np.pi - 1/5/T/T
    alp_max = keff*g0/2/np.pi + 1/5/T/T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)

    alp = np.zeros(N_sim)
    alp_vib = np.zeros(N_sim)
    Ph = np.zeros(N_sim)
    F_vibz = np.zeros(N_sim)
    az_m = np.zeros((N_sim, N_RP))
    sigma_a = SIGMA_A_ACC

    # фазовый шум от шума акселерометра
    sigma_ph_vibr = keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2)) * abs(Kz)
    if verbose:
        print(f"sigma_g_drift = {sigma_g_drift}")
        print(f"vib_state = {vib_state}, hp_cutoff = {hp_cutoff} Гц, "
              f"f_low_phys (1/2T) = {F_LOW_PHYS:.3f} Гц")
        print(f"sigma_ph_vibr = {sigma_ph_vibr/keff/T/T*1e8} uGal")

    # --- параметры фринджа ---
    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15; dA_sim = dA_step; DA_sim = DA_extra
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21; dB_sim = dB_step
    ph_sim = np.zeros(N_sim)
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)
    sigma_A_sim = 7e-3

    for i in range(N_sim):
        g_sim[i] = (g0 + (g_sim[i-1] - g0)*(1 - theta_drift)
                    + sigma_g_drift*rng_noise.normal())
        ph_sim[i] = keff*g_sim[i]*T*T

        seed_z = None if seed_vib is None else int(rng_vib.integers(0, 2**31 - 1))
        az = gen_vibration_trace_z(N_RP, t_step, state=vib_state,
                                   hp_cutoff=hp_cutoff,
                                   platform_atten_db=platform_atten_db,
                                   seed=seed_z)

        F_vibz[i] = sens_integral(delay, az) * Kz

        alp[i] = alp_start[i % alp_amount]
        alp_vib[i] = alp[i] - F_vibz[i] / (2*np.pi*T*T)
        Ph[i] = 2*np.pi*alp_vib[i]*T*T - ph_sim[i]

        A_sim[i] = A_sim[i-1] + rng_noise.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + rng_noise.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])

        az_m[i] = az + rng_noise.normal(0, sigma_a, N_RP)  # измеренное ускорение
        P_sim_noise[i] = P_sim[i] + rng_noise.normal(0, sigma_A_sim)

    dph_sim = sigma_g_drift*keff*T*T
    dA_sim = np.sqrt(dA_sim**2 + DA_sim**2)

    out = (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim,
           dph_sim, sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m)
    if return_extra:
        out += ({'F_vibz': F_vibz, 'ph_sim': ph_sim, 'Ph': Ph},)
    return out

# ============================================================
# Сравнение pcov из curve_fit с заложенными в генерацию dA_sim и sigma_ph_vibr
# ============================================================

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)


def fit_fringe(alp, P):
    """curve_fit одного фринджа. Параметры подгонки -- из блока НАСТРОЙКИ
    ПОДГОНКИ вверху файла. Возвращает (popt, pcov)."""
    ph_expected = keff * FIT_G0_PRIOR * T**2
    ph_p0 = ph_expected % (2*np.pi) if FIT_P0_PH_FROM_PRIOR else 0.0
    p0 = [(P.max() + P.min())/2, (P.max() - P.min())/2, ph_p0]

    popt, pcov = curve_fit(model, alp, P, p0=p0, bounds=(FIT_LB, FIT_UB))
    popt = popt.copy()

    if FIT_RESOLVE_BRANCH:
        M = np.round((ph_expected - popt[2]) / (2*np.pi))
        popt[2] += 2*np.pi*M
    return popt, pcov


def estimate_sigmas(phi, x, res, poi, n_bins):
    """
    Оценка sigma_A и sigma_ph по невязкам фита в бинах фазы.

    phi  : Phi mod pi для каждой точки, [0, pi)  (Phi = 2*pi*T^2*alp_fit - ph)
    x    : B^2 * sin^2(Phi) в этой точке (из параметров фита)
    res  : невязка фита в этой точке
    poi  : число точек фита на один фрингe (для поправки на 3 параметра)
    n_bins : число бинов

    Бины по Phi mod pi (а не по номеру точки): вибрация сдвигает фазу каждого
    сброса на несколько радиан, поэтому номинальный alp не определяет Phi.

    В каждом бине: v_b = средний квадрат невязки (с поправкой poi/(poi-3)),
    x_b = средний x. Модель v_b = sigma_A^2 + sigma_ph^2 * x_b решается
    неотрицательным МНК с весами sqrt(n_b)/v_model (дисперсия оценки дисперсии
    ~ 2 v^2 / n), несколько итераций перевзвешивания.

    Возвращает (sigma_A, sigma_ph, (n_b, x_b, v_b, v_model)).
    """
    bins = np.minimum((phi / np.pi * n_bins).astype(int), n_bins - 1)
    corr = poi / (poi - 3)

    n_b = np.array([np.sum(bins == b) for b in range(n_bins)], dtype=float)
    x_b = np.array([x[bins == b].mean() for b in range(n_bins)])
    v_b = np.array([corr * np.mean(res[bins == b]**2) for b in range(n_bins)])

    X = np.column_stack((np.ones(n_bins), x_b))
    w = np.sqrt(n_b)
    for _ in range(5):
        coef, _ = nnls(X * w[:, None], v_b * w)
        v_model = X @ coef
        w = np.sqrt(n_b) / np.maximum(v_model, 1e-30)
    return np.sqrt(coef[0]), np.sqrt(coef[1]), (n_b, x_b, v_b, X @ coef)


if __name__ == "__main__":

    # ---------- параметры ГЕНЕРАЦИИ (физика / симуляция) ----------
    alp_amount = 200               # число точек развёртки alp в одном фрингe
    delay = 0                      # смещение окна интерферометра в записи ускорения
    Kz = 1.0                       # коэффициент связи вибрации z с фазой (генерация)
    dA_step = 5e-3                 # шаг блуждания A за сброс (0 -- A постоянна)
    dB_step = 5e-3                 # шаг блуждания B за сброс (0 -- B постоянна)
    DA_extra = 0.0                 # добавка к возвращаемому dA_sim, обычно 0
    # остальные -- в вызове simul_acc ниже: vib_state, platform_atten_db, hp_cutoff

    if not (4 <= POI <= alp_amount):
        raise ValueError(f"POI должно быть в диапазоне [4, {alp_amount}], получено {POI}")

    rng = np.random.default_rng(SEED_MC)
    Kz_fit = Kz if FIT_KZ is None else FIT_KZ

    pooled_phi, pooled_x, pooled_res = [], [], []   # по всем прогонам
    sA_runs, sph_runs = [], []                      # оценки по отдельным прогонам
    n_fail = 0

    for r in range(N_RUNS):
        out = simul_acc(N_sim=alp_amount*FRINGES_PER_RUN,
                        alp_amount=alp_amount,
                        delay=delay,
                        Kz=Kz,
                        vib_state='mooring',
                        hp_cutoff=0.01,
                        platform_atten_db=0.0,
                        dA_step=dA_step, dB_step=dB_step, DA_extra=DA_extra,
                        seed_vib=int(rng.integers(0, 2**31 - 1)),
                        seed_noise=int(rng.integers(0, 2**31 - 1)),
                        verbose=(r == 0))
        (alp, P_noise, P_clean, A_s, B_s, g_s, dA_sim, dB_sim, dph_sim,
         sig_g, sigma_ph_vibr, sigma_A_sim, az_m) = out

        # вибрационная фаза, ОЦЕНЁННАЯ по измеренному ускорению (с шумом акселерометра)
        F_meas = keff * Kz_fit * (az_m[:, delay:delay + end] @ weight_vec)
        alp_fit = alp - F_meas / (2*np.pi*T**2)
        del az_m  # экономим память

        run_phi, run_x, run_res = [], [], []
        for k in range(FRINGES_PER_RUN):
            sl = slice(k*alp_amount, k*alp_amount + POI)   # первые POI точек фринджа
            try:
                popt, _ = fit_fringe(alp_fit[sl], P_noise[sl])
            except Exception:
                n_fail += 1
                continue
            A_f, B_f, ph_f = popt
            Phi = 2*np.pi*T**2*alp_fit[sl] - ph_f
            run_phi.append(np.mod(Phi, np.pi))
            run_x.append(B_f**2 * np.sin(Phi)**2)
            run_res.append(P_noise[sl] - model(alp_fit[sl], *popt))

        if not run_phi:
            continue
        run_phi = np.concatenate(run_phi)
        run_x = np.concatenate(run_x)
        run_res = np.concatenate(run_res)

        sA_r, sph_r, _ = estimate_sigmas(run_phi, run_x, run_res, POI, N_BINS)
        sA_runs.append(sA_r)
        sph_runs.append(sph_r)
        pooled_phi.append(run_phi); pooled_x.append(run_x); pooled_res.append(run_res)

    sA_pool, sph_pool, (n_b, x_b, v_b, v_mod) = estimate_sigmas(
        np.concatenate(pooled_phi), np.concatenate(pooled_x),
        np.concatenate(pooled_res), POI, N_BINS)

    print("\n" + "="*64)
    print(f"прогонов: {len(sA_runs)}, фринджей на прогон: {FRINGES_PER_RUN} "
          f"(сбоев фита {n_fail}), POI = {POI}, бинов: {N_BINS}")
    print("="*64)
    print("бины (по всем данным):")
    print(f"{'бин':>4s}{'точек':>9s}{'<B^2 sin^2>':>14s}{'sigma_P^2':>14s}{'модель':>14s}")
    for b in range(N_BINS):
        print(f"{b:4d}{int(n_b[b]):9d}{x_b[b]:14.3e}{v_b[b]:14.3e}{v_mod[b]:14.3e}")

    print()
    print(f"{'':28s}{'заложено':>11s}{'все данные':>12s}{'по прогонам (среднее±std)':>28s}")
    print(f"{'sigma_A':28s}{sigma_A_sim:11.3e}{sA_pool:12.3e}"
          f"{np.mean(sA_runs):16.3e} ± {np.std(sA_runs):.1e}")
    print(f"{'sigma_ph, рад':28s}{sigma_ph_vibr:11.3e}{sph_pool:12.3e}"
          f"{np.mean(sph_runs):16.3e} ± {np.std(sph_runs):.1e}")
    print(f"{'оценка/заложено (sigma_A)':28s}{'':11s}{sA_pool/sigma_A_sim:12.2f}")
    print(f"{'оценка/заложено (sigma_ph)':28s}{'':11s}{sph_pool/sigma_ph_vibr:12.2f}")