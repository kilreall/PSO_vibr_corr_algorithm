"""
verify_sigma_full_mode.py
==========================

Задача: проверить, насколько хорошо sigma_A и sigma_ph, которые в "full"
режиме обработки реального гравиметра оцениваются из ковариации
начального cos-fit (см. _sigma_from_pcov() во втором скрипте), совпадают
с "истинными" значениями, заложенными в генератор данных (первый
скрипт).

Что этот скрипт НЕ делает:
    - не подбирает tau и Kz (PSO / координатный / grid поиск отсутствуют
      полностью) -- используются заведомо ИЗВЕСТНЫЕ истинные tau_true,
      Kz_true;
    - не упрощает модель генерации вибрации: ASD-профиль (mooring/
      sailing), генерация методом случайных фаз (Timmer & Koenig),
      резонансные линии, ВЧ-фильтр акселерометра (Butterworth+filtfilt),
      ослабление платформы, anti-alias спад -- всё перенесено из первого
      скрипта без изменений;
    - вибрация генерируется ТОЛЬКО по оси Z (Kx = Ky = 0 по постановке
      задачи), поэтому оси x, y из генератора исключены.

Что скрипт делает:
    1) Генерирует N_REAL независимых реализаций серии из N_SIM сбросов
       (полная модель, как в simul_acc из первого скрипта, но только с
       вибрацией по оси Z).
    2) Компенсирует alp заведомо верным Kz_true по ЗАШУМЛЁННЫМ показаниям
       акселерометра az_m (так же, как это делал бы реальный анализ).
    3) На скомпенсированных данных вызывает init_values() (тот же код,
       что и во втором скрипте) и берёт из её pcov оценки sigma_A_full,
       sigma_ph_full -- ТЕ САМЫЕ величины, которые в "full"-режиме идут в
       R-матрицу EKF.
    4) Сравнивает их с "истинными" значениями:
         sigma_A_true  = SIGMA_A_SIM  -- шум амплитуды, буквально
                         добавленный в генератор к P_sim при создании
                         P_sim_noise;
         sigma_ph_true = sigma_ph_from_K(Kz_true, SIGMA_A_ACC) --
                         аналитическая оценка фазового шума ОТ ОСТАТКА
                         вибрации после идеальной (по известному Kz)
                         компенсации; она зависит только от шума
                         акселерометра SIGMA_A_ACC, "просвеченного" через
                         ту же весовую функцию fa(t), что и в генераторе/
                         компенсации -- см. вывод в комментарии к
                         sigma_ph_from_K().
    5) Печатает mean ± std оценок full-режима по реализациям и их
       относительное отклонение от истины, для нескольких размеров окна
       POI (и, при желании, для обоих состояний вибрации).

    ДОБАВЛЕНО по запросу:
      - в вывод добавлена sigma_A_resid -- второй, независимый от pcov
        способ оценить sigma_A на том же fit: std невязки (P0 - model)
        после cos-fit по poi точкам (то самое значение, которое
        init_values() уже считает и раньше отбрасывалось через `_`).
        Сравнивается и с sigma_A_full (из pcov*poi), и с sigma_A_true.
      - добавлен вариант БЕЗ умножения на poi: sqrt(pcov[i,i]) "как есть"
        (это то, что curve_fit возвращает как дисперсию оценки параметра
        по всем poi точкам сразу, без домножения на poi) -- для sigma_A
        и sigma_ph, чтобы можно было сравнить, какой из двух вариантов
        (с "*poi" или без) ближе к истинным значениям.

Запуск: просто python3 verify_sigma_full_mode.py
Параметры для правки -- в блоке if __name__ == "__main__".
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


# ============================================================
# Физические константы и весовая функция интерферометра
# (идентично первому/второму скрипту)
# ============================================================

lm = 780e-9
keff = 4 * np.pi / lm

T = 10e-3      # длительность плеча интерферометрической последовательности
ty = 20e-6     # длительность импульса

N_RP = 16384   # число отсчётов акселерометра на сброс
T_RP = 33e-3   # длительность реализации акселерометра на сброс, с
t_step = T_RP / N_RP

G0_PRIOR = 9.8101507   # опорное g -- только для выбора ветви 2π


def fa(t):
    if 0 < t <= T + 2 * ty:
        return t
    elif T + 2 * ty < t <= 2 * T + 4 * ty:
        return 2 * (T + 2 * ty) - t
    else:
        return 0.0


fa_v = np.vectorize(fa, otypes=[float])

t_range = np.linspace(0, 2 * T + 4 * ty, round((2 * T + 4 * ty) / t_step))
end = len(t_range)
fa_t = fa_v(t_range)

_trapz_w = np.full(end, t_step)
_trapz_w[0] *= 0.5
_trapz_w[-1] *= 0.5
weight_vec = fa_t * _trapz_w   # F_vib(tau) = keff * Kz * (az_window @ weight_vec)


def model(alp, A, B, ph):
    return A - B * np.cos(2 * np.pi * alp * T ** 2 - ph)


# ============================================================
# Генератор вибрации по ASD -- ТОЛЬКО ось Z, без упрощений
# (таблицы и алгоритм -- дословно из первого скрипта, оси x,y убраны)
# ============================================================

_ASD_Z_TABLES = {
    'mooring': {
        'f_nodes': np.array([1e-3, 3e-2, 0.1, 0.3, 1.0, 2.0, 4.0, 6.0,
                              10.0, 20.0, 40.0, 100.0, 300.0, 1000.0]),
        'asd': np.array([5.5e-5, 5.5e-5, 5.0e-5, 3.0e-5, 8.0e-5, 3.0e-4,
                          8.0e-4, 2.8e-3, 1.1e-3, 3.2e-4, 1.3e-4, 3.2e-5,
                          4.2e-6, 3.0e-7]),
        'res_lines': [],
    },
    'sailing': {
        'f_nodes': np.array([1e-3, 1e-2, 3e-2, 0.1, 0.15, 0.3, 0.6, 1.0,
                              2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 70.0, 100.0,
                              200.0, 400.0, 1000.0]),
        'asd': np.array([1.2e-4, 1.3e-4, 5.0e-4, 1.0e-1, 1.9e-1, 2.7e-2,
                          3.0e-3, 6.0e-4, 1.2e-4, 6.0e-5, 4.5e-4, 1.2e-3,
                          1.0e-4, 8.0e-5, 8.0e-5, 2.0e-5, 8.0e-6, 1.0e-5,
                          5.0e-7]),
        'res_lines': [(6.0, 10, 1.2), (12.0, 12, 0.8), (24.0, 15, 0.5),
                      (45.0, 15, 0.4)],
    },
}


def _synthesize_from_asd(freqs, target_asd, N, fs, rng):
    """Timmer & Koenig: реализация временного ряда с заданной ОДНОСТОРОННЕЙ
    амплитудной спектральной плотностью target_asd(freqs) [ед/sqrt(Гц)]."""
    n_freq = len(freqs)
    amp = target_asd * np.sqrt(N * fs / 2.0)

    phase = rng.uniform(0, 2 * np.pi, n_freq)
    Xf = amp * np.exp(1j * phase)

    Xf[0] = amp[0] * rng.normal() * np.sqrt(2.0)
    if N % 2 == 0:
        Xf[-1] = amp[-1] * rng.normal() * np.sqrt(2.0)

    return np.fft.irfft(Xf, n=N)


def gen_vibration_trace_z(N, dt, state='mooring', seed=None,
                           hp_cutoff=0.01, hp_order=2,
                           aa_cutoff_factor=1.0, platform_atten_db=-80.0):
    """
    Одноосевая (Z) реализация вибрационного ускорения платформы, приближённая
    к измеренным ASD (Qiao 2025). Логика идентична gen_vibration_trace() из
    первого скрипта (ось 'z'): кусочно-линейная ASD в log-log, случайные
    фазы, резонансные линии (для sailing), ослабление платформы, мягкий
    anti-alias спад, ВЧ-фильтр акселерометра (Butterworth + filtfilt).
    """
    if state not in _ASD_Z_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs_safe = freqs.copy()
    freqs_safe[0] = freqs_safe[1] * 0.5

    table = _ASD_Z_TABLES[state]
    f_nodes = table['f_nodes']
    asd_nodes = table['asd']
    res_lines = table['res_lines']

    log_target = np.interp(np.log10(freqs_safe),
                            np.log10(f_nodes), np.log10(asd_nodes))
    target_asd = 10 ** log_target

    for f0, q, rel_amp in res_lines:
        target_asd *= 1.0 + rel_amp * np.exp(-0.5 * ((freqs_safe - f0) / (f0 / q)) ** 2)

    target_asd = target_asd * (10 ** (platform_atten_db / 20.0))

    f_aa = f_nodes[-1] * aa_cutoff_factor
    target_asd *= 1.0 / (1.0 + (freqs_safe / f_aa) ** 6)

    raw = _synthesize_from_asd(freqs_safe, target_asd, N, fs, rng)

    wn = hp_cutoff / (fs / 2)
    if 0 < wn < 1:
        b, a_f = butter(hp_order, wn, btype='high')
        a_trace = filtfilt(b, a_f, raw)
    else:
        a_trace = raw

    return a_trace


# ============================================================
# init_values / sigma_from_pcov -- дословно из второго скрипта
# (это и есть "full"-режим оценки sigma_A, sigma_ph)
# ============================================================

def init_values(alp, P_exp, poi, g0_prior=G0_PRIOR):
    alp0 = alp[:poi]
    P0 = P_exp[:poi]

    p0 = [(P0.max() + P0.min()) / 2, (P0.max() - P0.min()) / 2, 0.0]
    lb = [-1.1, 0.0, 0.0]
    ub = [1.1, 1.1, 2 * np.pi]
    popt, pcov = curve_fit(model, alp0, P0, p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt

    ph_expected = keff * g0_prior * T ** 2
    M = np.round((ph_expected - ph0) / (2 * np.pi))
    ph0 = ph0 + 2 * np.pi * M

    sigma_A = np.std(P0 - model(alp0, A0, B0, ph0))
    return A0, B0, ph0, pcov, sigma_A


def sigma_from_pcov(pcov, poi):
    """pcov[i,i] -- дисперсия ОЦЕНКИ параметра по poi точкам, т.е. примерно
    в poi раз меньше дисперсии шума на одну точку -> sigma_single = sqrt(pcov*poi).
    Возвращает (sigma_A_full, sigma_ph_full) -- то, что "full"-режим кладёт в R EKF.

    Дополнительно (по запросу) возвращает ещё вариант БЕЗ умножения на poi:
    sqrt(pcov[i,i]) "как есть" -- это стандартная ошибка ОЦЕНКИ параметра
    (т.е. насколько неопределённо мы знаем A0/ph0 по всем poi точкам сразу),
    а не оценка std шума на одну точку. Возвращаем оба варианта, чтобы
    сравнить, какой ближе к истинным sigma_A_true / sigma_ph_true.
    """
    sigma_A_full = float(np.sqrt(pcov[0, 0] * poi))
    sigma_ph_full = float(np.sqrt(pcov[2, 2] * poi))
    sigma_A_full_nopoi = float(np.sqrt(pcov[0, 0]))
    sigma_ph_full_nopoi = float(np.sqrt(pcov[2, 2]))
    return sigma_A_full, sigma_ph_full, sigma_A_full_nopoi, sigma_ph_full_nopoi


def sigma_ph_from_K(Kz, sigma_a_acc):
    """
    Аналитическая "истина": фазовый шум, вносимый ТОЛЬКО измерительным
    шумом акселерометра sigma_a_acc (белый шум с этим std добавляется к
    каждому отсчёту az при формировании az_m), после идеальной компенсации
    заведомо верным Kz. Ошибка компенсации на один сброс:

        d(alp_comp) = -Kz * keff * sum(noise * weight_vec) / (2*pi*T^2)
        d(Ph)       = 2*pi*T^2 * d(alp_comp)
                    = -Kz * keff * sum(noise * weight_vec)

    noise -- iid N(0, sigma_a_acc) по отсчётам, weight_vec = fa_t * t_step
    (с половинными весами на краях), поэтому

        std(d(Ph)) = |Kz| * keff * sigma_a_acc * t_step * sqrt(sum(fa_t^2))

    что и есть sigma_ph_from_K() из второго скрипта (Kx = Ky = 0).
    """
    return keff * t_step * sigma_a_acc * np.sqrt(np.sum(fa_t ** 2)) * abs(Kz)


# ============================================================
# Генерация серии сбросов -- аналог simul_acc() из первого скрипта,
# ТОЛЬКО ось Z, БЕЗ подбора (tau, Kz) -- они известны заранее
# ============================================================

def simul_acc_z(N_sim, alp_amount, tau_true, Kz_true,
                 vib_state='mooring', hp_cutoff=0.01, platform_atten_db=-80.0,
                 sigma_a_acc=3e-5, sigma_A_sim=7e-3,
                 dA_sim=5e-3, DA_sim=1.5e-4, dB_sim=5e-3,
                 Dg_drift=500e-8, drift_corr=200, g0=G0_PRIOR,
                 seed_vib=None, seed_noise=None):
    """
    Возвращает словарь с:
      alp            -- несжатая (заданная) частота АОМ на сброс
      P_sim_noise    -- нормированный сигнал интерферометра (с шумом sigma_A_sim)
      az_m           -- (N_sim, N_RP) показания акселерометра с измерительным
                        шумом sigma_a_acc (то, что "видит" анализ)
      g_sim          -- истинное g на сброс (для справки, здесь не используется)
      sigma_ph_vibr_true -- аналитическая "истинная" sigma_ph (см. sigma_ph_from_K)
      sigma_A_sim    -- истинная sigma_A (буквально то, что добавлено к P_sim)
    """
    if tau_true + end > N_RP:
        raise ValueError("tau_true + end превышает длину реализации акселерометра N_RP")

    rng_noise = np.random.default_rng(seed_noise)
    rng_vib = np.random.default_rng(seed_vib)

    g_sim = np.zeros(N_sim)
    g_sim[-1] = g0
    theta_drift = 1 - np.exp(-1.0 / drift_corr)
    sigma_g_drift = Dg_drift * np.sqrt(theta_drift * (2 - theta_drift))

    alp_min = keff * g0 / 2 / np.pi - 1 / 5 / T / T
    alp_max = keff * g0 / 2 / np.pi + 1 / 5 / T / T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)
    alp = np.zeros(N_sim)

    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21
    ph_sim = np.zeros(N_sim)
    Ph = np.zeros(N_sim)
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)

    az_m = np.zeros((N_sim, N_RP))
    F_vibz = np.zeros(N_sim)
    ph_raw_abs_max = 0.0

    for i in range(N_sim):
        g_sim[i] = g0 + (g_sim[i - 1] - g0) * (1 - theta_drift)*0 + sigma_g_drift * rng_noise.normal()*0
        ph_sim[i] = keff * g_sim[i] * T * T

        seed_z = None if seed_vib is None else int(rng_vib.integers(0, 2 ** 31 - 1))
        az = gen_vibration_trace_z(N_RP, t_step, state=vib_state,
                                    hp_cutoff=hp_cutoff,
                                    platform_atten_db=platform_atten_db,
                                    seed=seed_z)

        # ИСТИННЫЙ фазовый вклад вибрации (по НЕЗАШУМЛЁННОМУ az) -- то, что
        # реально сдвигает Ph, как в simul_acc() первого скрипта
        F_vibz[i] = keff * float(np.sum(weight_vec * az[tau_true:tau_true + end])) * Kz_true
        ph_raw_abs_max = max(ph_raw_abs_max, abs(F_vibz[i]))

        alp[i] = alp_start[i % alp_amount]
        alp_vib_i = alp[i] - F_vibz[i] / (2 * np.pi * T ** 2)
        Ph[i] = 2 * np.pi * alp_vib_i * T ** 2 - ph_sim[i]

        A_sim[i] = A_sim[i - 1] + rng_noise.normal(0, dA_sim)
        B_sim[i] = B_sim[i - 1] + rng_noise.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i] * np.cos(Ph[i])

        # то, что "видит" анализ -- ИСТИННЫЙ az + измерительный шум акселерометра
        az_m[i] = az + rng_noise.normal(0, sigma_a_acc, N_RP)
        P_sim_noise[i] = P_sim[i] + rng_noise.normal(0, sigma_A_sim)

    sigma_ph_vibr_true = sigma_ph_from_K(Kz_true, sigma_a_acc)

    return dict(alp=alp, P_sim_noise=P_sim_noise, az_m=az_m, g_sim=g_sim,
                A_sim=A_sim, B_sim=B_sim,
                sigma_ph_vibr_true=sigma_ph_vibr_true, sigma_A_sim=sigma_A_sim,
                ph_raw_abs_max=ph_raw_abs_max)


def compensate_alp_z(tau, Kz, alp, az_m):
    """Компенсация по ЗАШУМЛЁННЫМ показаниям акселерометра az_m -- так же,
    как это делал бы реальный анализ (compensate_alp с Kx=Ky=0)."""
    az_window = az_m[:, tau:tau + end]
    Fz = keff * (az_window @ weight_vec)
    return alp - Kz * Fz / (2 * np.pi * T ** 2)


# ============================================================
# Проверка: сравнение sigma_A_full / sigma_ph_full (из pcov, "full"-режим)
# с истинными значениями генератора, БЕЗ поиска (tau, Kz)
# ============================================================

def _robust_stats(arr):
    """mean/std искажаются единичными выбросами (плохо обусловленный
    curve_fit на узком -- 0.4 цикла по построению -- диапазоне фазы иногда
    даёт близкую к вырожденной pcov). Поэтому дополнительно считаем
    медиану и 16-84% интервал (устойчивы к выбросам)."""
    finite = arr[np.isfinite(arr)]
    n_bad = len(arr) - len(finite)
    if len(finite) == 0:
        return dict(n_bad=n_bad, mean=np.nan, std=np.nan,
                    median=np.nan, p16=np.nan, p84=np.nan)
    return dict(n_bad=n_bad, mean=finite.mean(), std=finite.std(),
                median=np.median(finite),
                p16=np.percentile(finite, 16), p84=np.percentile(finite, 84))


def run_check(vib_state, Kz_true, tau_true, poi_list, N_sim, alp_amount,
              hp_cutoff, platform_atten_db, sigma_a_acc, sigma_A_sim,
              N_real, seed0=0):
    print(f"\n=== vib_state='{vib_state}'  Kz_true={Kz_true}  tau_true={tau_true}  "
          f"platform_atten_db={platform_atten_db}  hp_cutoff={hp_cutoff} Гц ===")

    sigma_A_true = sigma_A_sim
    sigma_ph_true = sigma_ph_from_K(Kz_true, sigma_a_acc)
    print("Истинные значения (из параметров генератора, не зависят от poi):")
    print(f"  sigma_A_true  = {sigma_A_true:.4e}")
    print(f"  sigma_ph_true = {sigma_ph_true:.4e} рад "
          f"({sigma_ph_true/keff/T**2*1e8:.2f} мкГал-экв.)")

    max_poi = max(poi_list)
    if max_poi > N_sim:
        raise ValueError("max(poi_list) не должен превышать N_sim")

    def _err(x):
        return (x - 1.0) * 100

    for poi in poi_list:
        sA_list, sph_list = [], []
        sA_nopoi_list, sph_nopoi_list = [], []
        sA_resid_list = []
        ph_raw_list = []
        n_fit_fail = 0
        for r in range(N_real):
            data = simul_acc_z(
                N_sim, alp_amount, tau_true, Kz_true,
                vib_state=vib_state, hp_cutoff=hp_cutoff,
                platform_atten_db=platform_atten_db,
                sigma_a_acc=sigma_a_acc, sigma_A_sim=sigma_A_sim,
                seed_vib=seed0 + 1000 * r + 1, seed_noise=seed0 + 1000 * r + 2)

            alp_comp = compensate_alp_z(tau_true, Kz_true, data['alp'], data['az_m'])
            try:
                _, _, _, pcov, sigma_A_resid = init_values(alp_comp, data['P_sim_noise'], poi)
                sA_full, sph_full, sA_full_nopoi, sph_full_nopoi = sigma_from_pcov(pcov, poi)
            except Exception:
                n_fit_fail += 1
                continue

            sA_list.append(sA_full)
            sph_list.append(sph_full)
            sA_nopoi_list.append(sA_full_nopoi)
            sph_nopoi_list.append(sph_full_nopoi)
            sA_resid_list.append(sigma_A_resid)
            ph_raw_list.append(data['ph_raw_abs_max'])

        sA_stats = _robust_stats(np.array(sA_list))
        sph_stats = _robust_stats(np.array(sph_list))
        sA_nopoi_stats = _robust_stats(np.array(sA_nopoi_list))
        sph_nopoi_stats = _robust_stats(np.array(sph_nopoi_list))
        sA_resid_stats = _robust_stats(np.array(sA_resid_list))
        n_ok = N_real - n_fit_fail

        print(f"\n--- poi = {poi}  ({n_ok}/{N_real} реализаций дали успешный fit, "
              f"|Ph_raw|max ~ {np.mean(ph_raw_list) if ph_raw_list else float('nan'):.2f} рад) ---")
        if n_ok == 0:
            print("    все реализации дали сбой curve_fit -- пропущено")
            continue

        print("    -- sigma_A: три способа оценки --")
        print(f"    sigma_A_full   (sqrt(pcov*poi))  : mean={sA_stats['mean']:.3e}±{sA_stats['std']:.1e}"
              f"  median={sA_stats['median']:.3e} "
              f"[{sA_stats['p16']:.3e}, {sA_stats['p84']:.3e}]"
              f"  (истина {sigma_A_true:.3e}, median/true={_err(sA_stats['median']/sigma_A_true):+.1f}%)")
        print(f"    sigma_A_nopoi  (sqrt(pcov))      : mean={sA_nopoi_stats['mean']:.3e}±{sA_nopoi_stats['std']:.1e}"
              f"  median={sA_nopoi_stats['median']:.3e} "
              f"[{sA_nopoi_stats['p16']:.3e}, {sA_nopoi_stats['p84']:.3e}]"
              f"  (истина {sigma_A_true:.3e}, median/true={_err(sA_nopoi_stats['median']/sigma_A_true):+.1f}%)")
        print(f"    sigma_A_resid  (std невязки fit) : mean={sA_resid_stats['mean']:.3e}±{sA_resid_stats['std']:.1e}"
              f"  median={sA_resid_stats['median']:.3e} "
              f"[{sA_resid_stats['p16']:.3e}, {sA_resid_stats['p84']:.3e}]"
              f"  (истина {sigma_A_true:.3e}, median/true={_err(sA_resid_stats['median']/sigma_A_true):+.1f}%)")

        print("    -- sigma_ph: два способа оценки --")
        print(f"    sigma_ph_full  (sqrt(pcov*poi))  : mean={sph_stats['mean']:.3e}±{sph_stats['std']:.1e}"
              f"  median={sph_stats['median']:.3e} "
              f"[{sph_stats['p16']:.3e}, {sph_stats['p84']:.3e}]"
              f"  (истина {sigma_ph_true:.3e}, median/true={_err(sph_stats['median']/sigma_ph_true):+.1f}%)")
        print(f"    sigma_ph_nopoi (sqrt(pcov))      : mean={sph_nopoi_stats['mean']:.3e}±{sph_nopoi_stats['std']:.1e}"
              f"  median={sph_nopoi_stats['median']:.3e} "
              f"[{sph_nopoi_stats['p16']:.3e}, {sph_nopoi_stats['p84']:.3e}]"
              f"  (истина {sigma_ph_true:.3e}, median/true={_err(sph_nopoi_stats['median']/sigma_ph_true):+.1f}%)")

        if sA_stats['n_bad'] or sph_stats['n_bad']:
            print(f"    (не-конечные значения: sigma_A {sA_stats['n_bad']}, "
                  f"sigma_ph {sph_stats['n_bad']} из {n_ok})")


if __name__ == "__main__":

    # ---- параметры генерации (можно менять) ----
    N_SIM = 1000          # число сбросов в серии (как в первом скрипте)
    ALP_AMOUNT = 200      # длина одного скана по alp
    TAU_TRUE = 700         # истинное окно акселерометра (как delay в первом скрипте)
    KZ_TRUE = 0.9          # истинный коэффициент компенсации по Z

    HP_CUTOFF = 0.01                 # Гц -- ВЧ-фильтр акселерометра
    PLATFORM_ATTEN_DB = -0.0        # дБ -- ослабление сырой вибрации корпуса
                                      # до уровня на оси интерферометра
                                      # (0 дБ -- "сырая" вибрация, как в
                                      # __main__ первого скрипта; тогда
                                      # несжатая фаза может быть очень большой)

    SIGMA_A_ACC = 3e-5     # шум акселерометра, м/с^2 (как SIGMA_A_ACC в первом скрипте)
    SIGMA_A_SIM = 7e-3     # шум амплитуды сигнала интерферометра (как sigma_A_sim)

    POI_LIST = [50, 100, 200, 400]   # размеры окна начального cos-fit для проверки
    N_REAL = 20             # число независимых реализаций (Монте-Карло) на каждый poi

    for state in ('mooring', 'sailing'):
        run_check(state, KZ_TRUE, TAU_TRUE, POI_LIST, N_SIM, ALP_AMOUNT,
                  HP_CUTOFF, PLATFORM_ATTEN_DB, SIGMA_A_ACC, SIGMA_A_SIM,
                  N_REAL, seed0=0 if state == 'mooring' else 10_000)