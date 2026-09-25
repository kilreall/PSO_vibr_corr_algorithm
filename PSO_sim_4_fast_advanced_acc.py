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
        return t
    elif T+2*ty < t <= 2*T+4*ty:
        return 2*(T+2*ty)-t
    else:
        return 0
fat_v = np.vectorize(fa, otypes=[float])   # <-- КРИТИЧНО: явно задать float

def model(alp, A, B, ph):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)


G0_PRIOR = 9.8101507  # известное априори значение g (не "истина" для оценки,
                       # а грубая опорная точка -- используется ТОЛЬКО
                       # для выбора правильной ветви 2π)

def init_values(alp, P_exp, poi, g0_prior=G0_PRIOR):

    alp_init = alp.copy()[:poi]
    P_init = P_exp.copy()[:poi]

    p0 = [(np.max(P_init[:poi]) + np.min(P_init[:poi])) / 2,
          (np.max(P_init[:poi]) - np.min(P_init[:poi])) / 2, 0]
    lb = [-1.1, 0, 0]
    ub = [1.1, 1.1, 2*np.pi]
    popt, pcov = curve_fit(model, alp_init[:poi], P_init[:poi], p0=p0, bounds=(lb, ub))
    A0, B0, ph0 = popt

    # --- разрешение порядка фринджа: g_eval должен быть ближайшим к
    #     известному g0_prior, а не зависеть от alp/вибрации ---
    ph_expected = keff * g0_prior * T**2
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


# ============================================================
# ГЕНЕРАЦИЯ ВИБРАЦИОННОГО УСКОРЕНИЯ -- mooring / sailing, по осям,
# с учётом ВЧ-фильтра акселерометра и связки с 1/(2T)
# ============================================================
#
#  1) ЦЕЛЕВАЯ одностороння ASD (амплитудная спектральная плотность)
#     задаётся кусочно-линейно в log-log координатах, отдельно для каждой
#     оси (x,y,z) и для двух состояний: 'mooring' (стоянка) и 'sailing'
#     (ход, сильные вибрации).
#  2) Реализация временного ряда строится методом случайных фаз
#     (Timmer & Koenig) -- корректный масштаб именно как ASD [м/с^2/√Гц].
#  3) Поверх плавной формы накладываются узкополосные "линии" (sailing).
#  4) На выходе -- ВЧ-фильтр (Butterworth + filtfilt, нулевая фаза),
#     имитирующий аппаратный HPF акселерометра.
#  5) f_low_phys (обычно 1/(2T)) -- только справочно.
# ============================================================

# --- таблицы ASD [м/с^2 / sqrt(Гц)], узлы по осям и состояниям -----------

_ASD_TABLES = {
    'mooring': {
        # общие частотные узлы для всех трёх осей в состоянии mooring
        'f_nodes': np.array([1e-3, 3e-2, 0.1, 0.3, 1.0, 2.0, 4.0, 6.0,
                              10.0, 20.0, 40.0, 100.0, 300.0, 1000.0]),
        'axis': {
            'x': np.array([2.3e-4, 2.3e-4, 1.8e-4, 6.0e-5, 1.5e-4, 5.0e-4,
                           1.2e-3, 3.0e-3, 1.0e-3, 3.0e-4, 1.2e-4, 3.0e-5,
                           4.0e-6, 3.0e-7]),
            'y': np.array([2.0e-4, 2.0e-4, 1.6e-4, 5.0e-5, 1.3e-4, 4.5e-4,
                           1.0e-3, 2.5e-3, 9.0e-4, 2.5e-4, 1.0e-4, 2.5e-5,
                           3.5e-6, 2.5e-7]),
            'z': np.array([5.5e-5, 5.5e-5, 5.0e-5, 3.0e-5, 8.0e-5, 3.0e-4,
                           8.0e-4, 2.8e-3, 1.1e-3, 3.2e-4, 1.3e-4, 3.2e-5,
                           4.2e-6, 3.0e-7]),
        },
        # для mooring выраженных узкополосных линий на графике нет
        'res_lines': [],
    },
    # Значения откалиброваны по оцифровке рисунка (PSD для sailing).
    # Это БАЗОВАЯ (сглаженная) кривая ДО умножения на узкополосные
    # резонансные горбы res_lines (они добавляются кодом поверх).
    'sailing': {
        'f_nodes': np.array([1e-3, 1e-2, 3e-2, 0.1, 0.15, 0.3, 0.6, 1.0,
                              2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 70.0, 100.0,
                              200.0, 400.0, 1000.0]),
        'axis': {
            'x': np.array([2.0e-3, 3.5e-3, 5.5e-3, 9.0e-2, 1.0e-1, 4.0e-2,
                           4.5e-3, 1.1e-3, 2.8e-4, 1.0e-4, 1.5e-4, 1.2e-4,
                           6.0e-5, 5.0e-5, 4.0e-5, 3.0e-5, 4.0e-6, 4.0e-6,
                           3.0e-7]),
            'y': np.array([4.0e-4, 7.5e-4, 1.3e-3, 3.3e-2, 3.6e-2, 2.8e-2,
                           3.2e-3, 9.5e-4, 2.2e-4, 1.2e-4, 1.6e-4, 1.0e-4,
                           9.0e-5, 3.0e-5, 3.0e-5, 3.0e-5, 5.0e-6, 1.5e-5,
                           1.5e-6]),
            'z': np.array([1.2e-4, 1.3e-4, 5.0e-4, 1.0e-1, 1.9e-1, 2.7e-2,
                           3.0e-3, 6.0e-4, 1.2e-4, 6.0e-5, 4.5e-4, 1.2e-3,
                           1.0e-4, 8.0e-5, 8.0e-5, 2.0e-5, 8.0e-6, 1.0e-5,
                           5.0e-7]),
        },
        'res_lines': [(6.0, 10, 1.2), (12.0, 12, 0.8), (24.0, 15, 0.5),
                      (45.0, 15, 0.4)],
    },
}


def _synthesize_from_asd(freqs, target_asd, N, fs, rng):
    """
    Строит реализацию временного ряда с заданной ОДНОСТОРОННЕЙ
    амплитудной спектральной плотностью target_asd(freqs) [ед/√Гц],
    методом случайных фаз (Timmer & Koenig).

    Для одностороннего ASD S(f) [x/sqrt(Hz)] и N точек с шагом dt=1/fs:
        |X_k| ~ S(f_k) * sqrt(N*fs/2)   (k = 1..N/2-1, случайная фаза)
        |X_0|, |X_{N/2}| -- действительные компоненты (DC и Найквист)
    """
    n_freq = len(freqs)
    amp = target_asd * np.sqrt(N * fs / 2.0)

    phase = rng.uniform(0, 2*np.pi, n_freq)
    Xf = amp * np.exp(1j*phase)

    Xf[0] = amp[0] * rng.normal() * np.sqrt(2.0)
    if N % 2 == 0:
        Xf[-1] = amp[-1] * rng.normal() * np.sqrt(2.0)

    return np.fft.irfft(Xf, n=N)


def gen_vibration_trace(N, dt, axis='z', state='mooring', seed=None,
                         hp_cutoff=0.01, hp_order=2,
                         f_low_phys=None, aa_cutoff_factor=1.0,
                         platform_atten_db=-80.0,
                         return_components=False):
    """
    Генерация одноосевой реализации вибрационного ускорения платформы,
    приближённой к измеренным ASD (Qiao 2025) для двух состояний.

    Физика: весовая функция интерферометра fa(t) ведёт себя как
    НИЗКОЧАСТОТНЫЙ фильтр, поэтому именно МЕДЛЕННЫЕ вибрации (доли Гц --
    единицы Гц) дают наибольший вклад в ошибку g и требуют компенсации
    через Kz,Kx,Ky. hp_cutoff -- это имитация АППАРАТНОГО HPF самого
    акселерометра (убирает дрейф нуля/1/f-шум датчика), поэтому он должен
    быть очень низким (0.01 Гц или ниже), заведомо ниже пика 0.1-1 Гц.

    Параметры
    ---------
    N, dt        : длина реализации и шаг по времени
    axis         : 'x' | 'y' | 'z'
    state        : 'mooring' | 'sailing'
    seed         : сид ГПСЧ (None -- без фиксации)
    hp_cutoff    : частота среза ВЧ-фильтра акселерометра [Гц]
    hp_order     : порядок Баттерворта для HPF
    f_low_phys   : контрольная частота (обычно 1/(2T)), справочно
    aa_cutoff_factor : множитель к верхнему узлу ASD для anti-alias спада
    platform_atten_db : ослабление [дБ] между "сырым" ускорением корпуса и
                   ускорением на чувствительной оси интерферометра
                   (изоляция/подвес/сервоплатформа). При 0 дБ -- "сырая"
                   вибрация, фаза порядка тысяч радиан, порядок фринджа
                   ожидаемо не разрешается.
    return_components : если True, вернуть ещё словарь
                   {'raw', 'target_asd', 'freqs'}

    Возвращает
    ----------
    a(t) [м/с^2] после ослабления и HPF (или (a(t), components))
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if state not in _ASD_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs_safe = freqs.copy()
    freqs_safe[0] = freqs_safe[1] * 0.5  # избегаем log(0)

    table = _ASD_TABLES[state]
    f_nodes = table['f_nodes']
    asd_nodes = table['axis'][axis]
    res_lines = table['res_lines']

    _danger_f = 0.05
    if hp_cutoff > _danger_f:
        print(f"[gen_vibration_trace] ВНИМАНИЕ: hp_cutoff={hp_cutoff} Гц "
              f"может начать резать реальный низкочастотный вибрационный "
              f"сигнал (пик ASD лежит в районе 0.1-1 Гц) -- держите "
              f"cutoff в районе 0.01 Гц или ниже")
    if f_low_phys is not None:
        pass  # справочно

    log_target = np.interp(np.log10(freqs_safe),
                            np.log10(f_nodes), np.log10(asd_nodes))
    target_asd = 10 ** log_target

    for f0, q, rel_amp in res_lines:
        target_asd *= 1.0 + rel_amp * np.exp(-0.5*((freqs_safe - f0)/(f0/q))**2)

    # --- ослабление "сырой" вибрации корпуса до уровня на оси интерферометра ---
    target_asd = target_asd * (10 ** (platform_atten_db / 20.0))

    # мягкий anti-alias спад у верхнего узла
    f_aa = f_nodes[-1] * aa_cutoff_factor
    target_asd *= 1.0 / (1.0 + (freqs_safe / f_aa)**6)

    raw = _synthesize_from_asd(freqs_safe, target_asd, N, fs, rng)

    # --- ВЧ-фильтр акселерометра, zero-phase (filtfilt) ---
    wn = hp_cutoff / (fs/2)
    if 0 < wn < 1:
        b, a_f = butter(hp_order, wn, btype='high')
        a_trace = filtfilt(b, a_f, raw)
    else:
        a_trace = raw  # cutoff вне допустимого диапазона -- фильтр пропускаем

    if return_components:
        return a_trace, {'raw': raw, 'target_asd': target_asd, 'freqs': freqs_safe}
    return a_trace


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

# контрольная частота (справочно, см. gen_vibration_trace)
F_LOW_PHYS = 1.0 / (2*T)


# ============================================================
# Симуляция с вибрацией по трём осям
# ============================================================

SIGMA_A_ACC = 3e-5   # шум акселерометра -- должно совпадать с sigma_a в simul_acc

def simul_acc(N_sim, alp_amount, delay, Kz, Kx, Ky,
              vib_state='mooring', hp_cutoff=0.01, platform_atten_db=-80.0,
              seed_vib=None):
    """
    vib_state : 'mooring' | 'sailing' -- профиль ASD вибрации
    hp_cutoff : частота среза ВЧ-фильтра акселерометра [Гц]
    platform_atten_db : ослабление [дБ] "сырой" вибрации корпуса до
                уровня на чувствительной оси интерферометра
    seed_vib  : базовый сид для генератора вибрации (None -- без фиксации)
    """

    # --- диагностика масштаба: несжатый (K=1) фазовый вклад вибрации на сброс ---
    def _raw_phase_estimate(atten_db):
        probe = gen_vibration_trace(N_RP, t_step, axis='z', state=vib_state,
                                     hp_cutoff=hp_cutoff, platform_atten_db=atten_db)
        return keff * float(np.sum(weight_vec * probe[:end]))

    ph_raw_used = _raw_phase_estimate(platform_atten_db)
    ph_raw_hull = _raw_phase_estimate(0.0)
    print(f"[simul_acc] vib_state={vib_state}, platform_atten_db={platform_atten_db} dB")
    print(f"[simul_acc] оценка несжатой (K=1) фазы от вибрации на 1 сброс: "
          f"~{ph_raw_used:.2f} рад (при выбранном ослаблении); "
          f"~{ph_raw_hull:.2f} рад при 0 дБ (сырая вибрация корпуса)")
    if abs(ph_raw_used) > 5:
        print("[simul_acc] ВНИМАНИЕ: несжатая фаза > 5 рад -- велика "
              "вероятность промаха по порядку фринджа при разрешении сетки "
              "Kz,Kx,Ky, используемом в этом скрипте; уменьшите "
              "platform_atten_db (сделайте более отрицательным)")

    g0 = 9.8101507
    g_sim = np.zeros(N_sim); g_sim[-1] = g0
    drift_corr = 200
    Dg_drift = 500e-8
    theta_drift = 1 - np.exp(-1.0/drift_corr)
    sigma_g_drift = Dg_drift * np.sqrt(theta_drift*(2 - theta_drift))
    print(f"sigma_g_drift = {sigma_g_drift}")
    print(f"vib_state = {vib_state}, hp_cutoff = {hp_cutoff} Гц, "
          f"f_low_phys (1/2T) = {F_LOW_PHYS:.3f} Гц")

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
    sigma_a = SIGMA_A_ACC

    # суммарный фазовый шум от вибрации -- независимые вклады трёх осей
    # складываются в квадратуре, каждый со своим K
    sigma_ph_vibr = (keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2))
                      * np.sqrt(Kz**2 + Kx**2 + Ky**2))
    print(f"sigma_ph_vibr = {sigma_ph_vibr/keff/T/T*1e8} uGal")

    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15; dA_sim = 5e-3; DA_sim = 1.5e-4
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21; dB_sim = 5e-3
    ph_sim = np.zeros(N_sim)
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)
    sigma_A_sim = 7e-3

    rng_vib = np.random.default_rng(seed_vib)

    for i in range(N_sim):
        g_sim[i] = g0 + (g_sim[i-1] - g0)*(1 - theta_drift) + sigma_g_drift*np.random.normal()
        ph_sim[i] = keff*g_sim[i]*T*T

        # три независимых сида на сброс (если seed_vib задан)
        seed_z = None if seed_vib is None else int(rng_vib.integers(0, 2**31 - 1))
        seed_x = None if seed_vib is None else int(rng_vib.integers(0, 2**31 - 1))
        seed_y = None if seed_vib is None else int(rng_vib.integers(0, 2**31 - 1))

        az = gen_vibration_trace(N_RP, t_step, axis='z', state=vib_state,
                                  hp_cutoff=hp_cutoff, f_low_phys=F_LOW_PHYS,
                                  platform_atten_db=platform_atten_db, seed=seed_z)
        ax = gen_vibration_trace(N_RP, t_step, axis='x', state=vib_state,
                                  hp_cutoff=hp_cutoff, f_low_phys=F_LOW_PHYS,
                                  platform_atten_db=platform_atten_db, seed=seed_x)
        ay = gen_vibration_trace(N_RP, t_step, axis='y', state=vib_state,
                                  hp_cutoff=hp_cutoff, f_low_phys=F_LOW_PHYS,
                                  platform_atten_db=platform_atten_db, seed=seed_y)

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
    dA_sim = np.sqrt(dA_sim**2 + DA_sim**2)

    return (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
            sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m, ax_m, ay_m)


# --- диапазоны поиска ---
tau_range = [0, 5000]
Kz_range = [0.7, 1.1]
Kx_range = [0.0, 0.025]
Ky_range = [0.0, 0.025]

N_particles = 30
M_iter = 30

Kz_nominal = 1.0
Kx_nominal = 0.0
Ky_nominal = 0.0

SEQ_N_TAU = None
SEQ_N_KZ = 401
SEQ_N_KX = 201
SEQ_N_KY = 201
SEQ_N_PASSES = 1
SEQ_TAU_INIT = 500
SEQ_KZ_INIT = Kz_nominal
SEQ_KX_INIT = Kx_nominal
SEQ_KY_INIT = Ky_nominal

GRID_CF_N_TAU = 21
GRID_CF_N_KZ = 11
GRID_CF_N_KX = 5
GRID_CF_N_KY = 5

GRID_KF_N_TAU = 21
GRID_KF_N_KZ = 11
GRID_KF_N_KX = 5
GRID_KF_N_KY = 5

N_JOBS = None

# --- оконный cos-fit ---
WIN_SIZE = 20      # None -> alp_amount (одно полное сканирование alp на окно)
WIN_EDGES = None     # границы окон, задаются в __main__


def compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m):
    az_window = az_m[:, tau:tau + end]
    ax_window = ax_m[:, tau:tau + end]
    ay_window = ay_m[:, tau:tau + end]

    Fz = keff * (az_window @ weight_vec)
    Fx = keff * (ax_window @ weight_vec)
    Fy = keff * (ay_window @ weight_vec)

    F_vib_total = Kz*Fz + Kx*Fx + Ky*Fy
    return alp - F_vib_total / (2 * np.pi * T**2)

def sigma_ph_from_K(Kz, Kx, Ky, sigma_a=SIGMA_A_ACC):
    return keff * t_step * sigma_a * np.sqrt(np.sum(fa_t**2)) * np.sqrt(Kz**2 + Kx**2 + Ky**2)

def kalman_fitness(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m, warmup=50):
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    Kz = float(np.clip(Kz, Kz_range[0], Kz_range[1]))
    Kx = float(np.clip(Kx, Kx_range[0], Kx_range[1]))
    Ky = float(np.clip(Ky, Ky_range[0], Ky_range[1]))

    if tau + end > az_m.shape[1]:
        return 1e6

    alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)
    sigma_ph_cand = sigma_ph_from_K(Kz, Kx, Ky)

    try:
        A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, P_exp, poi)
    except Exception:
        return 1e6
    try:
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, P_exp, T, Q, P_cov0p, sigma_A_sim, sigma_ph_cand, A0p, B0p, ph0p)
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
# Оконный cos-fit (компенсация дрейфа A, B, g внутри серии)
# ============================================================

def make_windows(N, win):
    """Непересекающиеся окна, покрывающие весь набор (остаток размазывается
    по окнам, ничего не выбрасывается). Возвращает массив границ."""
    n_win = max(1, N // win)
    return np.linspace(0, N, n_win + 1).astype(int)


def windowed_cosfit(phase, y, edges):
    """
    Независимый cos-fit в каждом окне. phase = 2*pi*alp_comp*T^2.
    Модель A - B*cos(phase - ph) = A + c1*cos(phase) + c2*sin(phase),
    решается через lstsq; ph = atan2(-c2, -c1).
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


def windowed_fitness(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m):
    """Fitness для sequential-поиска: оконный cos-fit (сигнатура как у curvefit_fitness)."""
    tau = int(np.clip(round(tau), tau_range[0], tau_range[1]))
    Kz = float(np.clip(Kz, Kz_range[0], Kz_range[1]))
    Kx = float(np.clip(Kx, Kx_range[0], Kx_range[1]))
    Ky = float(np.clip(Ky, Ky_range[0], Ky_range[1]))
    if tau + end > az_m.shape[1]:
        return 1e6
    try:
        alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)
        rms, _ = windowed_cosfit(2*np.pi*alp_comp*T**2, P_exp, WIN_EDGES)
        return rms
    except Exception:
        return 1e6


def windowed_g_error(alp_used, P_exp, g_sim, edges):
    """
    Оценка g по окнам (cos-fit в каждом окне) и её отклонение от g_sim.
    Опорное значение окна -- среднее g_sim в этом окне (фит даёт усреднённую
    по окну оценку). Порядок фринджа снимается через resolve_g_fringe_order.
    Возвращает (g_win, dg_win, rms_dg, mean_dg).
    """
    _, ph = windowed_cosfit(2*np.pi*alp_used*T**2, P_exp, edges)
    g_win = resolve_g_fringe_order(ph / keff / T**2)
    g_ref = np.array([g_sim[edges[k]:edges[k + 1]].mean() for k in range(len(edges) - 1)])
    dg = g_win - g_ref
    return g_win, dg, float(np.sqrt(np.mean(dg**2))), float(np.mean(dg))


# ============================================================
# PSO
# ============================================================

def PSO(fitness_func, N_particles, M_iter, bounds, alp, P_exp, az_m, ax_m, ay_m):
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


def _pso_worker_init(alp_, P_exp_, az_m_, ax_m_, ay_m_, fitness_kind_,
                      Q_, sigma_A_sim_, sigma_ph_, poi_, warmup_, win_edges_=None):
    global _g_alp, _g_Pexp, _g_az_m, _g_ax_m, _g_ay_m, _g_fitness_kind
    global _g_Q, _g_sigma_A_sim, _g_sigma_ph, _g_poi, _g_warmup, _g_FFcache
    global _g_winedges
    _g_alp, _g_Pexp = alp_, P_exp_
    _g_az_m, _g_ax_m, _g_ay_m = az_m_, ax_m_, ay_m_
    _g_fitness_kind = fitness_kind_
    _g_Q, _g_sigma_A_sim, _g_sigma_ph = Q_, sigma_A_sim_, sigma_ph_
    _g_poi, _g_warmup = poi_, warmup_
    _g_FFcache = {}
    _g_winedges = win_edges_


def _pso_get_FzFxFy(tau):
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
        sigma_ph_cand = sigma_ph_from_K(Kz, Kx, Ky)
        try:
            A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, _g_Pexp, _g_poi)
        except Exception:
            return 1e6
        try:
            _, _, _, _, _, e, _ = kalmanFit_EKF(
                alp_comp, _g_Pexp, T, _g_Q, P_cov0p, _g_sigma_A_sim, sigma_ph_cand,
                A0p, B0p, ph0p)
        except Exception:
            return 1e6
        return float(np.std(e[_g_warmup:]))

    elif _g_fitness_kind == "windowed":
        try:
            rms, _ = windowed_cosfit(2*np.pi*alp_comp*T**2, _g_Pexp, _g_winedges)
            return rms
        except Exception:
            return 1e6

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
                  n_jobs=None, warmup=50, win_edges=None):
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

    ctx = mp.get_context()
    with ctx.Pool(processes=n_jobs,
                   initializer=_pso_worker_init,
                   initargs=(alp, P_exp, az_m, ax_m, ay_m, fitness_kind,
                             Q, sigma_A_sim, sigma_ph, poi, warmup, win_edges)) as pool:
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
# Координатный поиск
# ============================================================

def sequential_coordinate_search(fitness_func, alp, P_exp, az_m, ax_m, ay_m,
                                  tau_range, Kz_range, Kx_range, Ky_range,
                                  n_tau=None, n_Kz=401, n_Kx=201, n_Ky=201,
                                  n_passes=2,
                                  tau_init=500, Kz_init=1.0, Kx_init=0.0, Ky_init=0.0):
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

    print("  этап 1/2: поиск tau")
    fit_tau = np.array([fitness_func(t, Kz_cur, Kx_cur, Ky_cur, alp, P_exp, az_m, ax_m, ay_m)
                         for t in tau_vals_full])
    n_calls += len(tau_vals_full)
    tau_cur = int(tau_vals_full[np.argmin(fit_tau)])
    history["tau"] = (tau_vals_full, fit_tau)
    print(f"    tau_opt = {tau_cur}")

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
# Полная 4D-сетка: cos-fit (глобальный или оконный)
# ============================================================

def _cf_worker_init(az_m_, ax_m_, ay_m_, cosPhi0_, sinPhi0_, P_exp_,
                     Kz_vals_, Kx_vals_, Ky_vals_, win_edges_=None):
    global _g_az_m, _g_ax_m, _g_ay_m, _g_cosPhi0, _g_sinPhi0, _g_Pexp
    global _g_Kz_vals, _g_Kx_vals, _g_Ky_vals, _g_winedges
    _g_az_m, _g_ax_m, _g_ay_m = az_m_, ax_m_, ay_m_
    _g_cosPhi0, _g_sinPhi0, _g_Pexp = cosPhi0_, sinPhi0_, P_exp_
    _g_Kz_vals, _g_Kx_vals, _g_Ky_vals = Kz_vals_, Kx_vals_, Ky_vals_
    _g_winedges = win_edges_


def _cf_row_worker(args):
    it, tau = args
    Fz = keff * (_g_az_m[:, tau:tau + end] @ weight_vec)
    Fx = keff * (_g_ax_m[:, tau:tau + end] @ weight_vec)
    Fy = keff * (_g_ay_m[:, tau:tau + end] @ weight_vec)

    Phi0 = np.arctan2(_g_sinPhi0, _g_cosPhi0)   # нужен только для оконного режима

    row_fit = np.full((len(_g_Kz_vals), len(_g_Kx_vals), len(_g_Ky_vals)), 1e6)
    for iz, Kzv in enumerate(_g_Kz_vals):
        for ix, Kxv in enumerate(_g_Kx_vals):
            for iy, Kyv in enumerate(_g_Ky_vals):
                ang = Kzv*Fz + Kxv*Fx + Kyv*Fy
                if _g_winedges is None:
                    c = _g_cosPhi0*np.cos(ang) + _g_sinPhi0*np.sin(ang)
                    s = _g_sinPhi0*np.cos(ang) - _g_cosPhi0*np.sin(ang)
                    row_fit[iz, ix, iy] = closed_form_cosfit_rms(c, s, _g_Pexp)
                else:
                    row_fit[iz, ix, iy] = windowed_cosfit(Phi0 - ang, _g_Pexp, _g_winedges)[0]
    return it, tau, row_fit


def full_grid_search_curvefit_4d(alp, P_exp, az_m, ax_m, ay_m,
                                  tau_range, Kz_range, Kx_range, Ky_range,
                                  n_tau=21, n_Kz=11, n_Kx=5, n_Ky=5, n_jobs=None,
                                  win_edges=None):
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

    kind = "windowed cos-fit" if win_edges is not None else "cos-fit"
    print(f"  full_grid ({kind}): {len(tau_vals)} строк x {n_Kz*n_Kx*n_Ky} ячеек, "
          f"{n_jobs} процессов")
    ctx = mp.get_context()
    with ctx.Pool(processes=n_jobs,
                   initializer=_cf_worker_init,
                   initargs=(az_m, ax_m, ay_m, cosPhi0, sinPhi0, P_exp,
                             Kz_vals, Kx_vals, Ky_vals, win_edges)) as pool:
        tasks = list(enumerate(tau_vals))
        for it, tau, row_fit in pool.imap_unordered(_cf_row_worker, tasks):
            fitness[it] = row_fit
            print(f"  full_grid ({kind}): строка {it+1}/{len(tau_vals)} (tau={tau}) готова")

    idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    return (int(tau_vals[idx[0]]), float(Kz_vals[idx[1]]),
            float(Kx_vals[idx[2]]), float(Ky_vals[idx[3]]),
            float(fitness[idx]), n_calls,
            {"tau_vals": tau_vals, "Kz_vals": Kz_vals,
             "Kx_vals": Kx_vals, "Ky_vals": Ky_vals, "fitness": fitness})


# ============================================================
# Полная 4D-сетка: Kalman
# ============================================================

def _kf_worker_init(alp_, P_exp_, Fz_all_, Fx_all_, Fy_all_,
                     Kz_vals_, Kx_vals_, Ky_vals_, Q_, sigma_A_sim_, sigma_ph_,
                     poi_, T_, warmup_):
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
    alp_comp = _g_alp - (Kz*_g_Fz[it] + Kx*_g_Fx[it] + Ky*_g_Fy[it]) / (2*np.pi*_g_T**2)
    sigma_ph_cand = sigma_ph_from_K(Kz, Kx, Ky)
    try:
        A0p, B0p, ph0p, P_cov0p, _ = init_values(alp_comp, _g_Pexp, _g_poi)
        _, _, _, _, _, e, _ = kalmanFit_EKF(
            alp_comp, _g_Pexp, _g_T, _g_Q, P_cov0p, _g_sigma_A_sim, sigma_ph_cand,
            A0p, B0p, ph0p)
        fit = float(np.std(e[_g_warmup:]))
    except Exception:
        fit = 1e6
    return it, iz, ix, iy, fit


def full_grid_search_kalman_4d(alp, P_exp, az_m, ax_m, ay_m,
                                tau_range, Kz_range, Kx_range, Ky_range,
                                n_tau=11, n_Kz=6, n_Kx=3, n_Ky=3,
                                n_jobs=None, warmup=50):
    tau_vals = np.unique(np.round(np.linspace(*tau_range, n_tau)).astype(int))
    Kz_vals = np.linspace(*Kz_range, n_Kz)
    Kx_vals = np.linspace(*Kx_range, n_Kx)
    Ky_vals = np.linspace(*Ky_range, n_Ky)

    valid = tau_vals + end <= az_m.shape[1]
    tau_vals = tau_vals[valid]
    if len(tau_vals) == 0:
        raise ValueError("full_grid_search_kalman_4d: нет допустимых tau в пределах окна az_m")

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

    ctx = mp.get_context()
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
# Оценка результата
# ============================================================

def evaluate_with_kalman(tau, Kz, Kx, Ky, alp, P_exp, az_m, ax_m, ay_m):
    alp_comp = compensate_alp(tau, Kz, Kx, Ky, alp, az_m, ax_m, ay_m)
    sigma_ph_cand = sigma_ph_from_K(Kz, Kx, Ky)
    A0e, B0e, ph0e, P_cov0e, _ = init_values(alp_comp, P_exp, poi)
    P_m, A, B, ph, P_cov, e, en = kalmanFit_EKF(
        alp_comp, P_exp, T, Q, P_cov0e, sigma_A_sim, sigma_ph_cand, A0e, B0e, ph0e)
    return alp_comp, ph, e, en

def resolve_g_fringe_order(g_raw, g0_prior=G0_PRIOR):
    """
    Постобработка: убирает 2π-неоднозначность в готовой последовательности
    g_est, никак не трогая внутренности EKF. Анкер для первой точки --
    g0_prior, для всех последующих -- предыдущая УЖЕ скорректированная
    точка (предполагается, что истинный g меняется от сброса к сбросу
    много медленнее, чем dg_quantum).
    """
    dg_quantum = 2*np.pi / (keff * T**2)
    g_corr = np.empty_like(g_raw)

    k0 = np.round((g0_prior - g_raw[0]) / dg_quantum)
    g_corr[0] = g_raw[0] + k0*dg_quantum

    for i in range(1, len(g_raw)):
        k = np.round((g_corr[i-1] - g_raw[i]) / dg_quantum)
        g_corr[i] = g_raw[i] + k*dg_quantum

    return g_corr


def g_error_stats(ph, g_sim, warmup=50, n_tail=100, bias_warn_threshold=1e-3):
    """
    RMS/bias считаются ТОЛЬКО по последним n_tail точкам серии (а не по всему
    хвосту после warmup). warmup по-прежнему используется как индекс, С КОТОРОГО
    запускается цепочка resolve_g_fringe_order -- чтобы плохо сошедшиеся первые
    точки фильтра не портили привязку 2π-неоднозначности для всех последующих
    точек, включая финальные n_tail.
    """
    g_raw = ph / keff / T**2
    g_est_from_warmup = resolve_g_fringe_order(g_raw[warmup:], g0_prior=G0_PRIOR)

    n_tail = min(n_tail, len(g_est_from_warmup))
    g_est_tail = g_est_from_warmup[-n_tail:]
    g_sim_tail = g_sim[-n_tail:]

    diff = g_est_tail - g_sim_tail

    bias = np.median(diff)
    if abs(bias) > bias_warn_threshold:
        print(f"⚠ ВНИМАНИЕ: bias(g) = {bias*1e8:.3f} µGal подозрительно велик — "
              f"похоже на ошибку разрешения порядка фринджа")

    rms_debiased = np.sqrt(np.mean((diff - bias)**2))
    rms_total = np.sqrt(np.mean(diff**2))
    return rms_total, rms_debiased, bias, g_est_tail


def report_case(label, tau, Kz, Kx, Ky, ph_est, e_arr, g_sim, elapsed_s, n_calls,
                 warmup=50, n_tail=100):
    std_e = np.std(e_arr[warmup:])
    rms_total, rms_debiased, bias_g, _ = g_error_stats(ph_est, g_sim, warmup=warmup, n_tail=n_tail)
    print(f"{label:24s}{tau:7d}{Kz:9.4f}{Kx:10.5f}{Ky:10.5f}{std_e:14.3e}"
          f"{rms_total*1e8:14.0f}{bias_g*1e8:13.0f}{rms_debiased*1e8:14.0f}"
          f"{elapsed_s:10.2f}{n_calls:12d}")
    return std_e, rms_total, bias_g, rms_debiased

def nocomp_cosfit_g(alp, P_exp, g_sim):
    """
    Оценка g БЕЗ компенсации вибрации: один cos-fit (curve_fit) по всей
    серии, без Калмана. Начальное приближение и порядок фринджа берутся
    из init_values (анкер G0_PRIOR); фаза ищется в окне ±π вокруг ph0,
    чтобы fit не ушёл на соседнюю ветвь 2π.
    Возвращает (g_cf, g_cf - g_sim[-1], popt).
    """
    A0, B0, ph0, _, _ = init_values(alp, P_exp, poi)
    lb_p = [-1.1, 0.0, ph0 - np.pi]
    ub_p = [1.1, 1.1, ph0 + np.pi]
    popt, _ = curve_fit(model, alp, P_exp, p0=[A0, B0, ph0],
                        bounds=(lb_p, ub_p), maxfev=5000)
    g_raw = popt[2] / keff / T**2
    g_cf = resolve_g_fringe_order(np.array([g_raw]))[0]
    return g_cf, g_cf - g_sim[-1], popt


if __name__ == "__main__":

    # --- выбор состояния вибрации и параметров ВЧ-фильтра акселерометра ---
    VIB_STATE = 'mooring'         # 'mooring' или 'sailing'
    HP_CUTOFF = 0.01              # Гц -- убирает только дрейф самого датчика
    PLATFORM_ATTEN_DB = -0.0      # дБ -- ослабление "сырой" вибрации корпуса
                                  # до уровня на оси интерферометра;
                                  # 0.0 -- честная "сырая" вибрация с графика

    N_sim = 1000
    alp_amount = 200
    delay = 700
    Kz = 0.9
    Kx = 0.003
    Ky = 0.003

    (alp, P_sim_noise, P_sim, A_sim, B_sim, g_sim, dA_sim, dB_sim, dph_sim,
     sigma_g_drift, sigma_ph_vibr, sigma_A_sim, az_m, ax_m, ay_m) = simul_acc(
        N_sim, alp_amount, delay, Kz, Kx, Ky,
        vib_state=VIB_STATE, hp_cutoff=HP_CUTOFF,
        platform_atten_db=PLATFORM_ATTEN_DB)

    # --- контроль сгенерированного профиля ASD (одна реализация на ось) ---
    a_check, dbg = gen_vibration_trace(N_RP, t_step, axis='z', state=VIB_STATE,
                                        hp_cutoff=HP_CUTOFF, f_low_phys=F_LOW_PHYS,
                                        platform_atten_db=PLATFORM_ATTEN_DB,
                                        return_components=True)
    plt.figure()
    plt.loglog(dbg['freqs'], dbg['target_asd'], 'k--', label=f'target ASD, z, {VIB_STATE}')
    plt.xlabel('Frequency, Hz'); plt.ylabel(r'ASD, m/s$^2$/$\sqrt{Hz}$')
    plt.title('Проверка целевого профиля ASD (генератор вибрации)')
    plt.legend(); plt.grid(True, which='both')

    # kalman fit
    Q = np.diag([dA_sim**2, dB_sim**2, dph_sim**2])
    poi = alp_amount
    poi = 201
    A0, B0, ph0, P_cov0, sigma_A = init_values(alp, P_sim_noise, poi)
    sigma_ph = np.sqrt(sigma_ph_vibr**2 + dph_sim**2*0)
    sigma_A = sigma_A_sim

    warmup = 120
    results = {}

    # --- окна для оконного cos-fit ---
    WIN_SIZE = alp_amount if WIN_SIZE is None else WIN_SIZE
    WIN_EDGES = make_windows(N_sim, WIN_SIZE)
    print(f"Окна для оконного cos-fit: {len(WIN_EDGES)-1} шт., границы = {WIN_EDGES.tolist()}")

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

    print("\n=== PSO with WINDOWED cos-fit fitness (4D) ===")
    t0 = time.perf_counter()
    tau_w, Kz_w, Kx_w, Ky_w, fit_w, hist_w, calls_w = PSO_parallel(
        "windowed", N_particles, M_iter,
        [tau_range, Kz_range, Kx_range, Ky_range],
        alp, P_sim_noise, az_m, ax_m, ay_m,
        n_jobs=N_JOBS, win_edges=WIN_EDGES)
    results["PSO (windowed cos-fit)"] = (tau_w, Kz_w, Kx_w, Ky_w, time.perf_counter() - t0, calls_w)

    seq_fitness_func = curvefit_fitness

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

    print("\n=== Sequential coordinate search, windowed cos-fit ===")
    t0 = time.perf_counter()
    tau_sw, Kz_sw, Kx_sw, Ky_sw, fit_sw, calls_sw, hist_sw = sequential_coordinate_search(
        windowed_fitness, alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=SEQ_N_TAU, n_Kz=SEQ_N_KZ, n_Kx=SEQ_N_KX, n_Ky=SEQ_N_KY, n_passes=SEQ_N_PASSES,
        tau_init=SEQ_TAU_INIT, Kz_init=SEQ_KZ_INIT, Kx_init=SEQ_KX_INIT, Ky_init=SEQ_KY_INIT)
    results["Sequential (windowed)"] = (tau_sw, Kz_sw, Kx_sw, Ky_sw, time.perf_counter() - t0, calls_sw)

    print("\n=== FULL 4D grid, closed-form cos-fit (coarse) ===")
    t0 = time.perf_counter()
    tau_g1, Kz_g1, Kx_g1, Ky_g1, fit_g1, calls_g1, hist_g1 = full_grid_search_curvefit_4d(
        alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=GRID_CF_N_TAU, n_Kz=GRID_CF_N_KZ, n_Kx=GRID_CF_N_KX, n_Ky=GRID_CF_N_KY,
        n_jobs=N_JOBS)
    results["Full grid (cos-fit)"] = (tau_g1, Kz_g1, Kx_g1, Ky_g1, time.perf_counter() - t0, calls_g1)

    print("\n=== FULL 4D grid, windowed cos-fit ===")
    t0 = time.perf_counter()
    tau_gw, Kz_gw, Kx_gw, Ky_gw, fit_gw, calls_gw, hist_gw = full_grid_search_curvefit_4d(
        alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=GRID_CF_N_TAU, n_Kz=GRID_CF_N_KZ, n_Kx=GRID_CF_N_KX, n_Ky=GRID_CF_N_KY,
        n_jobs=N_JOBS, win_edges=WIN_EDGES)
    results["Full grid (windowed)"] = (tau_gw, Kz_gw, Kx_gw, Ky_gw, time.perf_counter() - t0, calls_gw)

    print("\n=== FULL 4D grid, Kalman fitness (very coarse) ===")
    t0 = time.perf_counter()
    tau_g2, Kz_g2, Kx_g2, Ky_g2, fit_g2, calls_g2, hist_g2 = full_grid_search_kalman_4d(
        alp, P_sim_noise, az_m, ax_m, ay_m,
        tau_range, Kz_range, Kx_range, Ky_range,
        n_tau=GRID_KF_N_TAU, n_Kz=GRID_KF_N_KZ, n_Kx=GRID_KF_N_KX, n_Ky=GRID_KF_N_KY,
        n_jobs=N_JOBS, warmup=warmup)
    results["Full grid (Kalman)"] = (tau_g2, Kz_g2, Kx_g2, Ky_g2, time.perf_counter() - t0, calls_g2)

    # --- baseline: БЕЗ компенсации, через cos-fit (не через EKF: для
    #     Калмана вибрация без коррекции -- неучтённый шум) ---
    g_cf_nc, dg_cf_nc, _ = nocomp_cosfit_g(alp, P_sim_noise, g_sim)

    print("\n===================================== Сравнение =====================================")
    print(f"{'':24s}{'tau':>7s}{'Kz':>9s}{'Kx':>10s}{'Ky':>10s}{'std(e)':>14s}"
          f"{'RMS_tot(g)':>14s}{'bias(g)':>13s}{'RMS_deb(g)':>14s}{'time,s':>10s}{'calls':>12s}")
    print(f"{'true':24s}{delay:7d}{Kz:9.4f}{Kx:10.5f}{Ky:10.5f}")

    N_TAIL = 800   # RMS_tot/bias/RMS_deb считаются только по последним N_TAIL точкам

    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        alp_v, ph_v, e_v, en_v = evaluate_with_kalman(
            tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        report_case(label, tau_v, Kz_v, Kx_v, Ky_v, ph_v, e_v, g_sim, t_v, n_calls_v,
                    warmup=warmup, n_tail=N_TAIL)
    
    print("-"*140)
    print(f"No compensation (cos-fit по всей серии, без Калмана):")
    print(f"    g_cf          = {g_cf_nc:.9f} м/с^2")
    print(f"    g_sim[-1]     = {g_sim[-1]:.9f} м/с^2")
    print(f"    g_cf - g_sim[-1] = {dg_cf_nc*1e8:.3f} µGal")

    # --- честное сравнение: и baseline, и все методы -- одним оконным cos-fit ---
    def _fmt_g(val_uGal):
        """Компактный формат: переключается на mGal, если величина большая."""
        if abs(val_uGal) > 1e5:
            return f"{val_uGal/1e3:.1f} mGal"
        return f"{val_uGal:.0f} µGal"

    def _print_win_row(label, dg):
        rms = np.sqrt(np.mean(dg**2)) * 1e8
        mean = np.mean(dg) * 1e8
        median = np.median(dg) * 1e8
        amax = np.max(np.abs(dg)) * 1e8
        print(f"{label:28s}{_fmt_g(rms):>12s}{_fmt_g(mean):>12s}"
              f"{_fmt_g(median):>12s}{_fmt_g(amax):>12s}")

    def print_windows_detail(label, dg, per_line=10):
        """Опционально: подробный список по окнам (µGal, целые), в несколько строк."""
        vals = [f"{v*1e8:>7.0f}" for v in dg]
        print(f"  {label}:")
        for i in range(0, len(vals), per_line):
            print("    " + " ".join(vals[i:i + per_line]))

    print("\n---- Оконный cos-fit: g_win - <g_sim>_окна ----")
    print(f"{'':28s}{'RMS':>12s}{'mean':>12s}{'median':>12s}{'|max|':>12s}")

    _, dg0, rms0, m0 = windowed_g_error(alp, P_sim_noise, g_sim, WIN_EDGES)
    _print_win_row("No compensation", dg0)

    windowed_errors = {"No compensation": dg0}
    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        alp_c = compensate_alp(tau_v, Kz_v, Kx_v, Ky_v, alp, az_m, ax_m, ay_m)
        _, dgc, rmsc, mc = windowed_g_error(alp_c, P_sim_noise, g_sim, WIN_EDGES)
        _print_win_row(label, dgc)
        windowed_errors[label] = dgc

    plt.figure()
    plt.plot(hist_kf, label="Kalman fitness")
    plt.plot(hist_cf, label="curve_fit fitness")
    plt.plot(hist_w, label="windowed cos-fit fitness")
    plt.xlabel("PSO iteration")
    plt.ylabel("best fitness (своя шкала для каждого метода)")
    plt.title(f"PSO convergence: Kalman vs curve_fit vs windowed (4D), vib_state={VIB_STATE}")
    plt.legend()

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

    plt.figure()
    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        _, _, e_v, _ = evaluate_with_kalman(tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        plt.plot(e_v, label=label, alpha=0.8)
    plt.xlabel("shot #")
    plt.ylabel("EKF innovation e")
    plt.title(f"Итоговые невязки EKF при разных способах компенсации, vib_state={VIB_STATE}")
    plt.legend()

    plt.figure()
    for label, (tau_v, Kz_v, Kx_v, Ky_v, t_v, n_calls_v) in results.items():
        _, ph_v, _, _ = evaluate_with_kalman(tau_v, Kz_v, Kx_v, Ky_v, alp, P_sim_noise, az_m, ax_m, ay_m)
        plt.plot((ph_v/keff/T**2 - g_sim)*1e8, label=label, alpha=0.8)
    plt.axhline(dg_cf_nc*1e8, color='k', ls='--',
                label=f"No compensation, cos-fit: {dg_cf_nc*1e8:.1f} µGal")
    plt.xlabel("shot #")
    plt.ylabel(r"$g_{est} - g_{sim}$, µGal")
    plt.title(f"Ошибка определения g при разных способах компенсации, vib_state={VIB_STATE}")
    plt.legend()

    plt.show()