# Один файл: (1) генерация данных симуляции фринджа с вибрационным ускорением
# ТОЛЬКО по оси z; (2) сравнение pcov / sigma_A / sigma_ph из curve_fit с тем,
# что заложено в генерацию.
#
# Запуск:  python sim_z_generation_and_compare.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


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
              return_extra=False):
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
    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15; dA_sim = 5e-3; DA_sim = 1.5e-4
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21; dB_sim = 5e-3
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
# Сравнение curve_fit (pcov, sigma_A, sigma_ph) с заложенным в генерацию
# ============================================================

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)


def fit_fringe(alp, P, sigma_abs=None, g0_prior=G0):
    """curve_fit одного фринджа. Возвращает (popt, pcov_default, pcov_abs)."""
    ph_expected = keff * g0_prior * T**2
    p0 = [(P.max() + P.min())/2, (P.max() - P.min())/2,
          ph_expected % (2*np.pi)]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]

    popt, pcov = curve_fit(model, alp, P, p0=p0, bounds=(lb, ub))

    # разрешение ветви 2π по известному g0 (как в init_values)
    M = np.round((ph_expected - popt[2]) / (2*np.pi))
    popt = popt.copy()
    popt[2] += 2*np.pi*M

    pcov_abs = None
    if sigma_abs is not None:
        _, pcov_abs = curve_fit(model, alp, P, p0=popt[:3] * [1, 1, 1],
                                sigma=np.full_like(P, sigma_abs),
                                absolute_sigma=True,
                                bounds=([-1.1, 0, -np.inf], [1.1, 1.1, np.inf]))
    return popt, pcov, pcov_abs


def run_comparison(n_runs=20, fringes_per_run=10, alp_amount=20,
                   delay=0, Kz=1.0, vib_state='mooring',
                   platform_atten_db=-80.0, hp_cutoff=0.01,
                   seed=123, make_plot=True):

    rng = np.random.default_rng(seed)

    est = []          # оценки [A, B, ph]
    truth_mean = []   # среднее по фринджу [A_sim, B_sim, ph_sim]
    truth_eff = []    # среднее ph_sim + F_vibz (эффективная фаза в фринджe)
    sig_pcov = []     # sqrt(diag(pcov)) по умолчанию
    sig_pcov_abs = [] # sqrt(diag(pcov)) с sigma = sigma_A_sim
    sigA_resid = []   # std остатков (как в init_values)
    sigA_resid_dof = []
    n_fail = 0

    F_all = []
    B2sin2 = []
    sigma_A_sim = dA_step = dph_sim = sigma_ph_vibr = None

    for r in range(n_runs):
        out = simul_acc(N_sim=alp_amount*fringes_per_run,
                           alp_amount=alp_amount, delay=delay, Kz=Kz,
                           vib_state=vib_state, hp_cutoff=hp_cutoff,
                           platform_atten_db=platform_atten_db,
                           seed_vib=int(rng.integers(0, 2**31 - 1)),
                           seed_noise=int(rng.integers(0, 2**31 - 1)),
                           verbose=(r == 0), return_extra=True)
        (alp, P_noise, P_clean, A_s, B_s, g_s, dA_step, dB_step, dph_sim,
         sig_g, sigma_ph_vibr, sigma_A_sim, az_m, extra) = out
        del az_m  # не нужен -- экономим память

        F = extra['F_vibz']
        ph_sim = extra['ph_sim']
        Ph = extra['Ph']
        F_all.append(F)

        for k in range(fringes_per_run):
            sl = slice(k*alp_amount, (k+1)*alp_amount)
            try:
                popt, pcov, pcov_abs = fit_fringe(alp[sl], P_noise[sl],
                                                  sigma_abs=sigma_A_sim)
            except Exception as ex:
                n_fail += 1
                continue

            res = P_noise[sl] - model(alp[sl], *popt)
            est.append(popt)
            truth_mean.append([A_s[sl].mean(), B_s[sl].mean(), ph_sim[sl].mean()])
            truth_eff.append((ph_sim[sl] + F[sl]).mean())
            sig_pcov.append(np.sqrt(np.diag(pcov)))
            sig_pcov_abs.append(np.sqrt(np.diag(pcov_abs)))
            sigA_resid.append(np.std(res))
            sigA_resid_dof.append(np.sqrt(np.sum(res**2)/(alp_amount - 3)))
            B2sin2.append(np.mean(B_s[sl]**2 * np.sin(Ph[sl])**2))

    est = np.array(est); truth_mean = np.array(truth_mean)
    truth_eff = np.array(truth_eff)
    sig_pcov = np.array(sig_pcov); sig_pcov_abs = np.array(sig_pcov_abs)
    F_all = np.concatenate(F_all)

    err = est - truth_mean                 # ошибка относительно "физической" истины
    err_ph_eff = est[:, 2] - truth_eff     # относительно эффективной фазы
    n_ok = len(est)

    # ---------- эффективный белый шум, эквивалентный заложенному ----------
    sigma_F = np.std(F_all)                # вибрационная фаза, рад/сброс
    sigma_A_eff = np.sqrt(sigma_A_sim**2 + np.mean(B2sin2)*sigma_F**2)

    to_ugal = 1e8 / (keff*T**2)            # рад -> мкГал

    print("\n" + "="*72)
    print(f"Фринджей успешно: {n_ok}  (сбоев curve_fit: {n_fail}), "
          f"точек на фрингe: {alp_amount}")
    print("="*72)

    print("\n--- ЗАЛОЖЕНО В ГЕНЕРАЦИЮ ---")
    print(f"sigma_A_sim (белый шум на P)                : {sigma_A_sim:.3e}")
    print(f"dA_step (случайное блуждание A за сброс)    : {dA_step:.3e}")
    print(f"вибрац. фаза, std(F_vibz) за сброс          : {sigma_F:.3e} рад "
          f"({sigma_F*to_ugal:.1f} мкГал)")
    print(f"дрейф g, dph_sim (инновация за сброс)       : {dph_sim:.3e} рад")
    print(f"sigma_ph_vibr (шум акселерометра, в P НЕ идёт): {sigma_ph_vibr:.3e} рад")
    print(f"sigma_A_eff = sqrt(sA^2 + <B^2 sin^2>*sF^2) : {sigma_A_eff:.3e}")

    print("\n--- sigma_A ИЗ ФИТА (std остатков) ---")
    print(f"среднее, как в init_values (без dof)        : {np.mean(sigA_resid):.3e}")
    print(f"среднее, с поправкой N-3                    : {np.mean(sigA_resid_dof):.3e}")
    print(f"отношение (dof) / sigma_A_sim               : "
          f"{np.mean(sigA_resid_dof)/sigma_A_sim:.2f}")
    print(f"отношение (dof) / sigma_A_eff               : "
          f"{np.mean(sigA_resid_dof)/sigma_A_eff:.2f}")

    names = ['A', 'B', 'ph']
    print("\n--- ПАРАМЕТРЫ: sqrt(diag(pcov)) vs реальный разброс ---")
    print(f"{'':4s}{'pcov (по остаткам)':>22s}{'pcov (sigma=sA_sim)':>22s}"
          f"{'эмпир. std ошибки':>20s}{'эмп./pcov':>11s}{'эмп./pcov_abs':>15s}")
    for j, n in enumerate(names):
        s_def = np.median(sig_pcov[:, j])
        s_abs = np.median(sig_pcov_abs[:, j])
        s_emp = np.std(err[:, j])
        print(f"{n:4s}{s_def:22.3e}{s_abs:22.3e}{s_emp:20.3e}"
              f"{s_emp/s_def:11.2f}{s_emp/s_abs:15.2f}")

    print("\nph (в единицах g):")
    print(f"  sqrt(pcov[ph,ph]) по остаткам   : {np.median(sig_pcov[:, 2])*to_ugal:9.1f} мкГал")
    print(f"  sqrt(pcov[ph,ph]) c sigma_A_sim : {np.median(sig_pcov_abs[:, 2])*to_ugal:9.1f} мкГал")
    print(f"  эмпир. std (оценка - ph_sim)    : {np.std(err[:, 2])*to_ugal:9.1f} мкГал")
    print(f"  эмпир. std (оценка - ph_sim-F)  : {np.std(err_ph_eff)*to_ugal:9.1f} мкГал")
    print(f"  bias ph (оценка - ph_sim)       : {np.mean(err[:, 2])*to_ugal:9.1f} мкГал")
    print(f"  вибрация усреднённая по фрингу  : ~{sigma_F/np.sqrt(alp_amount)*to_ugal:9.1f} мкГал "
          f"(sigma_F/sqrt(N))")

    # кол-во фринджей, у которых оценка попала в ±1σ pcov
    for lab, sg in (('pcov по остаткам', sig_pcov), ('pcov c sigma_A_sim', sig_pcov_abs)):
        cover = np.mean(np.abs(err) < sg, axis=0)
        print(f"покрытие ±1σ [{lab}]: "
              + ", ".join(f"{n}={c*100:.0f}%" for n, c in zip(names, cover))
              + "  (идеал ~68%)")

    if make_plot:
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
        for j, n in enumerate(names):
            ax[j].hist(err[:, j], bins=30, color='C0', alpha=0.7)
            s1 = np.median(sig_pcov[:, j]); s2 = np.median(sig_pcov_abs[:, j])
            for s, c, lb in ((s1, 'r', 'pcov (остатки)'), (s2, 'g', 'pcov (sA_sim)')):
                ax[j].axvline(s, color=c, ls='--', label=lb)
                ax[j].axvline(-s, color=c, ls='--')
            ax[j].set_title(f"ошибка {n}"); ax[j].legend(fontsize=7)
        ax[3].hist(np.array(sigA_resid_dof), bins=30, color='C1', alpha=0.7)
        ax[3].axvline(sigma_A_sim, color='k', label='sigma_A_sim')
        ax[3].axvline(sigma_A_eff, color='m', ls='--', label='sigma_A_eff')
        ax[3].set_title("sigma_A из остатков (N-3)"); ax[3].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig("compare_fit_vs_generation.png", dpi=130)
        print("\nГрафик: compare_fit_vs_generation.png")

    return dict(est=est, truth=truth_mean, err=err, sig_pcov=sig_pcov,
                sig_pcov_abs=sig_pcov_abs, sigA_resid=np.array(sigA_resid_dof))


if __name__ == "__main__":
    run_comparison(n_runs=20, fringes_per_run=10, alp_amount=20,
                   delay=0, Kz=1.0, vib_state='mooring',
                   platform_atten_db=-80.0)