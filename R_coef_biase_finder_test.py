# Проверка автоподбора параметров EKF (sigma_A, sigma_ph, Q) на синтетических
# данных с ИЗВЕСТНОЙ истиной.
#
# Структура файла:
#   1. КОНСТАНТЫ            -- физические/аппаратные константы и таблицы ASD (сверху);
#   2. Классы параметров    -- SimCfg (симуляция) и FitCfg (обработка);
#   3. Симуляция            -- simul_acc и её вспомогательные функции;
#   4. Обработка и печать   -- запуск EKF-автоподбора, сравнение с истиной;
#   5. main()               -- ВСЕ параметры задаются здесь, внизу файла:
#                              (А) параметры симуляции, (Б) параметры обработки
#                              (начальные Q и R, poi/warmup, настройка EKF).
#
# Что делает тест:
#   1. Генерирует данные (вибрация по оси z, дрейф g, блуждание A и B, шум
#      детектирования, белый фазовый шум).
#   2. Компенсирует alp по ускорению az_m (как на реальных данных).
#   3. Запускает ту же настройку, что в kalman_pso_fit.py (tune_and_score):
#      стартовые sigma -> EKF -> оценка R (Фишер / байесовская сетка) -> подбор Q.
#   4. Сравнивает результат с заложенным по n_runs независимым прогонам и с
#      «эталонным» EKF, которому отдают заложенные значения.
#
# Заложено в генерацию:
#   sigma_A   -- белый шум детектирования P (sigma_A_sim);
#   sigma_ph  -- белый фазовый шум на сброс, рад (sigma_ph_sim);
#   dA, dB    -- шаги блуждания A и B за сброс (dA_step, dB_step);
#   dph       -- std шага дрейфа g (процесс Орнштейна-Уленбека), пересчитанного в
#                фазу. EKF моделирует ph как случайное блуждание, так что dph --
#                эффективный шаг, а не точное соответствие процессу.
# Шума акселерометра нет (az_m = az): при kz_fit = Kz и tau_fit = delay
# компенсация вибрации точная.
#
# Файл EKF_MODULE должен лежать рядом с этим скриптом.
# Запуск:  python test_ekf_vs_truth.py

import importlib
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

# =====================================================================
# 1. КОНСТАНТЫ
# =====================================================================

EKF_MODULE = "PSO_coef_finder_real_v2"   # файл с EKF и автоподбором (без .py)

LM = 780e-9                  # длина волны, м
KEFF = 4 * np.pi / LM        # эффективный волновой вектор
N_RP = 16384                 # отсчётов акселерометра на сброс (аппаратура, как в EKF-модуле)
G0 = 9.8101507               # опорное g, м/с^2

# Целевые ASD вибрации [м/с^2/sqrt(Гц)] -- только ось z
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

ekf = importlib.import_module(EKF_MODULE)

PARAM_ORDER = ["sigma_A", "sigma_ph", "dA", "dB", "dph"]
PARAM_LABEL = {"sigma_A": "sigma_A", "sigma_ph": "sigma_ph, рад", "dA": "dA (шаг A)",
               "dB": "dB (шаг B)", "dph": "dph, рад"}
Q_NAME = {"A": "dA", "B": "dB", "ph": "dph"}


# =====================================================================
# 2. КЛАССЫ ПАРАМЕТРОВ (значения задаются в main)
# =====================================================================

@dataclass
class SimCfg:
    """Параметры симуляции."""
    # --- интерферометр ---
    T: float                    # длительность плеча, с
    ty: float                   # длительность импульса, с
    T_RP: float                 # запись акселерометра на сброс, с
    # --- набор данных ---
    n_runs: int                 # число независимых прогонов
    n_sim: int                  # сбросов в одном прогоне
    alp_amount: int             # точек развёртки alp в одном фринджe
    seed: int                   # сид Монте-Карло (сиды вибрации и шумов выводятся из него)
    # --- вибрация ---
    vib_state: str              # 'mooring' | 'sailing'
    hp_cutoff: float            # срез ВЧ-фильтра акселерометра, Гц
    platform_atten_db: float    # ослабление сырой вибрации, дБ
    Kz: float                   # коэффициент связи вибрации z с фазой (генерация)
    delay: int                  # смещение окна интерферометра в записи ускорения, отсчёты
    # --- фринж ---
    A0: float                   # начальное A
    B0: float                   # начальное B
    dA_step: float              # шаг блуждания A за сброс (std)
    dB_step: float              # шаг блуждания B за сброс (std)
    # --- шумы ---
    sigma_A_sim: float          # белый шум детектирования P
    sigma_ph_sim: float         # белый фазовый шум на сброс, рад
    # --- дрейф g (процесс Орнштейна-Уленбека) ---
    drift_corr: float           # время корреляции, сбросов
    Dg_drift: float             # амплитуда дрейфа
    # --- вывод ---
    verbose_first_run: bool     # печатать диагностику масштаба вибрации в 1-м прогоне

    @property
    def t_step(self):
        return self.T_RP / N_RP


@dataclass
class FitCfg:
    """Параметры обработки: компенсация, начальные Q и R, настройка EKF."""
    # --- компенсация вибрации при обработке ---
    kz_fit: Optional[float]     # None -> Kz генерации; иначе проверка при неточном Kz
    tau_fit: Optional[int]      # None -> delay генерации; иначе проверка при неточном tau
    # --- начальный фит и фильтр ---
    poi: int                    # точек для начального линейного фита
    warmup: int                 # инноваций после poi, не входящих в fitness
    # --- начальные Q ---
    q_init_std: Tuple[float, float, float]   # std шага (dA, dB, dph); в EkfParams уходят квадраты
    # --- начальные R ---
    sigma_init_mode: str                     # "bins" | "split" | "manual"
    sigma_init: Optional[Tuple[float, float]]  # (sigma_A, sigma_ph) для "manual"
    # --- настройка EKF ---
    tune: dict                  # общие поля ekf.TuneCfg
    methods: dict               # название -> поля ekf.TuneCfg, различающие методы
    # --- вывод ---
    verbose_runs: bool          # печатать оценки каждого прогона


# =====================================================================
# 3. СИМУЛЯЦИЯ
# =====================================================================

def fa(t, T, ty):
    """Функция чувствительности интерферометра к ускорению (скаляр)."""
    if 0 < t <= T + 2 * ty:
        return t
    elif T + 2 * ty < t <= 2 * T + 4 * ty:
        return 2 * (T + 2 * ty) - t
    else:
        return 0


def sim_weights(T, ty, t_step):
    """Веса интеграла чувствительности: F = KEFF * (a_window @ weight_vec).
    Независимая от EKF-модуля копия (генерация не должна зависеть от обработки)."""
    duration = 2 * T + 4 * ty
    t_grid = np.linspace(0, duration, round(duration / t_step))
    fa_t = np.array([fa(t, T, ty) for t in t_grid], dtype=float)
    trapz = np.full(len(t_grid), t_step)
    trapz[0] *= 0.5
    trapz[-1] *= 0.5
    return fa_t * trapz


def _synthesize_from_asd(freqs, target_asd, N, fs, rng):
    """Реализация временного ряда с заданным ОДНОСТОРОННИМ ASD
    методом случайных фаз (Timmer & Koenig)."""
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
        target_asd *= 1.0 + rel_amp * np.exp(-0.5 * ((freqs_safe - f0) / (f0 / q)) ** 2)

    # изоляция платформы
    target_asd = target_asd * (10 ** (platform_atten_db / 20.0))

    # мягкий anti-alias спад у верхнего узла
    f_aa = f_nodes[-1] * aa_cutoff_factor
    target_asd *= 1.0 / (1.0 + (freqs_safe / f_aa) ** 6)

    raw = _synthesize_from_asd(freqs_safe, target_asd, N, fs, rng)

    # ВЧ-фильтр акселерометра, нулевая фаза
    wn = hp_cutoff / (fs / 2)
    if 0 < wn < 1:
        b, a_f = butter(hp_order, wn, btype='high')
        a_trace = filtfilt(b, a_f, raw)
    else:
        a_trace = raw

    if return_components:
        return a_trace, {'raw': raw, 'target_asd': target_asd,
                         'freqs': freqs_safe}
    return a_trace


def sens_integral(delay, a, weight_vec):
    """Сырой (без Kz) интеграл чувствительности по окну интерферометра."""
    win = len(weight_vec)
    seg = a[delay:delay + win]
    if len(seg) != win:
        raise ValueError(f"delay={delay}: окно [{delay}, {delay + win}) выходит "
                         f"за пределы реализации длиной {len(a)}")
    return KEFF * float(seg @ weight_vec)


def simul_acc(cfg: SimCfg, weight_vec, seed_vib=None, seed_noise=None, verbose=False):
    """
    Генерирует данные одного прогона с вибрацией только по оси z.
    Шума акселерометра нет: az_m -- это вибрационное ускорение az как есть.

    cfg        : SimCfg
    weight_vec : веса интеграла чувствительности (sim_weights)
    seed_vib   : сид генератора вибрации (None -- без фиксации)
    seed_noise : сид остальных шумов/дрейфов (None -- без фиксации)

    Возвращает словарь:
      alp, P_noise, P_clean, A, B, g, ph, F_vibz, Ph, az_m  -- массивы по сбросам
      dA, dB, dph, sigma_A, sigma_ph, sigma_g_drift         -- заложенные значения
    """
    T, t_step, win = cfg.T, cfg.t_step, len(weight_vec)
    N_sim = cfg.n_sim
    rng_noise = np.random.default_rng(seed_noise)
    rng_vib = np.random.default_rng(seed_vib)

    # --- диагностика масштаба: несжатая (K=1) фаза от вибрации на сброс ---
    if verbose:
        def _raw_phase_estimate(atten_db):
            probe = gen_vibration_trace_z(N_RP, t_step, state=cfg.vib_state,
                                          hp_cutoff=cfg.hp_cutoff,
                                          platform_atten_db=atten_db)
            return KEFF * float(np.sum(weight_vec * probe[:win]))

        ph_raw_used = _raw_phase_estimate(cfg.platform_atten_db)
        ph_raw_hull = _raw_phase_estimate(0.0)
        print(f"[simul_acc] vib_state={cfg.vib_state}, "
              f"platform_atten_db={cfg.platform_atten_db} dB")
        print(f"[simul_acc] несжатая (K=1) фаза от вибрации за 1 сброс: "
              f"~{ph_raw_used:.2f} рад (выбранное ослабление); "
              f"~{ph_raw_hull:.2f} рад при 0 дБ")
        if abs(ph_raw_used) > 5:
            print("[simul_acc] ВНИМАНИЕ: несжатая фаза > 5 рад -- велика "
                  "вероятность промаха по порядку фринджа; сделайте "
                  "platform_atten_db более отрицательным")

    # --- дрейф g (процесс Орнштейна-Уленбека) ---
    g0 = G0
    g_sim = np.zeros(N_sim)
    g_sim[-1] = g0
    theta_drift = 1 - np.exp(-1.0 / cfg.drift_corr)
    sigma_g_drift = cfg.Dg_drift * np.sqrt(theta_drift * (2 - theta_drift))

    # --- сетка alp ---
    alp_min = KEFF * g0 / 2 / np.pi - 1 / 5 / T / T
    alp_max = KEFF * g0 / 2 / np.pi + 1 / 5 / T / T
    alp_start = np.linspace(alp_min, alp_max, cfg.alp_amount)

    alp = np.zeros(N_sim)
    alp_vib = np.zeros(N_sim)
    Ph = np.zeros(N_sim)
    F_vibz = np.zeros(N_sim)
    az_m = np.zeros((N_sim, N_RP))

    if verbose:
        print(f"sigma_g_drift = {sigma_g_drift}")

    # --- параметры фринджа ---
    A_sim = np.zeros(N_sim)
    A_sim[-1] = cfg.A0
    B_sim = np.zeros(N_sim)
    B_sim[-1] = cfg.B0
    ph_sim = np.zeros(N_sim)
    P_sim = np.zeros(N_sim)
    P_noise = np.zeros(N_sim)

    for i in range(N_sim):
        g_sim[i] = (g0 + (g_sim[i - 1] - g0) * (1 - theta_drift)
                    + sigma_g_drift * rng_noise.normal())
        ph_sim[i] = KEFF * g_sim[i] * T * T

        seed_z = None if seed_vib is None else int(rng_vib.integers(0, 2 ** 31 - 1))
        az = gen_vibration_trace_z(N_RP, t_step, state=cfg.vib_state,
                                   hp_cutoff=cfg.hp_cutoff,
                                   platform_atten_db=cfg.platform_atten_db,
                                   seed=seed_z)

        F_vibz[i] = sens_integral(cfg.delay, az, weight_vec) * cfg.Kz

        alp[i] = alp_start[i % cfg.alp_amount]
        alp_vib[i] = alp[i] - F_vibz[i] / (2 * np.pi * T * T)
        Ph[i] = (2 * np.pi * alp_vib[i] * T * T - ph_sim[i]
                 + rng_noise.normal(0, cfg.sigma_ph_sim))

        A_sim[i] = A_sim[i - 1] + rng_noise.normal(0, cfg.dA_step)
        B_sim[i] = B_sim[i - 1] + rng_noise.normal(0, cfg.dB_step)
        P_sim[i] = A_sim[i] - B_sim[i] * np.cos(Ph[i])

        az_m[i] = az                                   # измеренное ускорение (без шума)
        P_noise[i] = P_sim[i] + rng_noise.normal(0, cfg.sigma_A_sim)

    return {"alp": alp, "P_noise": P_noise, "P_clean": P_sim,
            "A": A_sim, "B": B_sim, "g": g_sim, "ph": ph_sim,
            "F_vibz": F_vibz, "Ph": Ph, "az_m": az_m,
            "dA": float(cfg.dA_step), "dB": float(cfg.dB_step),
            "dph": float(sigma_g_drift * KEFF * T * T),
            "sigma_A": float(cfg.sigma_A_sim), "sigma_ph": float(cfg.sigma_ph_sim),
            "sigma_g_drift": float(sigma_g_drift)}


# =====================================================================
# 4. ОБРАБОТКА, СРАВНЕНИЕ С ИСТИНОЙ, ПЕЧАТЬ
# =====================================================================

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def safe_nanmean(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def ratio_text(value, truth, width):
    """Отношение оценки к заложенному; при заложенном 0 -- прочерк."""
    if truth > 0:
        return f"{value / truth:{width}.2f}"
    return f"{'н/д':>{width}}"


def q_start_text(fit: FitCfg):
    return ", ".join(f"{v:.1e}" for v in fit.q_init_std)


def state_rms(x_est, data, fit: FitCfg):
    """
    RMS ошибки оценок состояния [A, B, ph] относительно истины.
    x_est[i] -- оценка после сброса poi + i; берутся точки после warmup.
    """
    w = fit.warmup
    A_e, B_e, ph_e = x_est[w:, 0], x_est[w:, 1], x_est[w:, 2]
    s = fit.poi + w
    A_t, B_t, ph_t = data["A"][s:], data["B"][s:], data["ph"][s:]
    if np.sum(B_e * B_t) < 0:        # (B, ph) и (-B, ph + pi) -- одна и та же кривая
        B_e, ph_e = -B_e, ph_e + np.pi
    dph = (ph_e - ph_t + np.pi) % (2 * np.pi) - np.pi     # ph определена по модулю 2*pi
    return np.array([rms(A_e - A_t), rms(B_e - B_t), rms(dph)])


def sigma_at_bound(est, tcfg):
    """Имена sigma, оценка которых упёрлась в границу допуска TuneCfg."""
    hits = []
    if est["sigma_A"] <= 1.02 * tcfg.sA_lo or est["sigma_A"] >= tcfg.sA_hi / 1.02:
        hits.append("sigma_A")
    if est["sigma_ph"] <= 1.02 * tcfg.sph_lo or est["sigma_ph"] >= tcfg.sph_hi / 1.02:
        hits.append("sigma_ph")
    return hits


def generate(sim: SimCfg, weight_sim, weight_ekf, win_len, tau_fit, seed_vib, seed_noise,
             verbose=False):
    """Данные одного прогона + истинные значения. az_m не хранится (только Fz)."""
    d = simul_acc(sim, weight_sim, seed_vib=seed_vib, seed_noise=seed_noise,
                  verbose=verbose)

    # вибрационная фаза по измеренному ускорению (без Kz; Kz применяется в compensate_alp)
    Fz = ekf.vibration_phase(d["az_m"], tau_fit, weight_ekf, win_len)

    truth = {p: d[p] for p in PARAM_ORDER}
    return {"alp": d["alp"], "y": d["P_noise"], "Fz": Fz, "A": d["A"], "B": d["B"],
            "ph": d["ph"], "truth": truth}


def run_oracle(data, sim: SimCfg, fit: FitCfg, kz_fit):
    """Эталон: тот же EKF и тот же начальный фит, но с ЗАЛОЖЕННЫМИ sigma_A, sigma_ph, Q."""
    poi, w = fit.poi, fit.warmup
    alp_comp = ekf.compensate_alp(data["alp"], data["Fz"], kz_fit, sim.T)
    x0, P0, _ = ekf.initial_fit(alp_comp[:poi], data["y"][:poi], sim.T)
    tr = data["truth"]
    q = np.array([tr["dA"], tr["dB"], tr["dph"]], dtype=float) ** 2
    r = ekf.run_ekf(alp_comp[poi:], data["y"][poi:], x0, P0, q,
                    tr["sigma_A"], tr["sigma_ph"], sim.T, w)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return None
    return {"J": r["nll"] / r["cnt"], "n": r["cnt"], "rms": state_rms(r["x"], data, fit)}


def run_method(data, sim: SimCfg, fit: FitCfg, tcfg, kz_fit):
    """Автоподбор из kalman_pso_fit.tune_and_score при заданном старте Q."""
    prm = ekf.EkfParams(T=sim.T,
                        q_init=np.array(fit.q_init_std, dtype=float) ** 2,
                        poi=fit.poi, warmup=fit.warmup)
    try:
        J, info = ekf.tune_and_score(data["Fz"], kz_fit, data["alp"], data["y"],
                                     prm, tcfg, diagnostics=True)
    except Exception as exc:
        print(f"    [tune_and_score упал: {exc!r}]")
        return None
    if not info:
        return None

    q = np.asarray(info["q"], dtype=float)
    est = {"sigma_A": float(info["sigma_A"]), "sigma_ph": float(info["sigma_ph"]),
           "dA": float(np.sqrt(q[0])), "dB": float(np.sqrt(q[1])),
           "dph": float(np.sqrt(q[2]))}
    nu = info["nu_stats"]
    return {"J": float(J), "est": est,
            "sd": (info["sd_sigma_A"], info["sd_sigma_ph"]),
            "rms": state_rms(info["states"], data, fit),
            "at_bound": list(info["q_at_bound"]),
            "nu_std": nu["std"], "lb_p": nu["lb_pvalue"]}


def print_method(name, fit, runs, truth, tcfg, n_runs):
    print(f"\n--- {name}; старт Q: std = ({q_start_text(fit)}); "
          f"успешных прогонов {len(runs)}/{n_runs} ---")
    if not runs:
        print("  нет успешных прогонов")
        return

    est = np.array([[r["est"][p] for p in PARAM_ORDER] for r in runs])
    sd_f = {"sigma_A": safe_nanmean([r["sd"][0] for r in runs]),
            "sigma_ph": safe_nanmean([r["sd"][1] for r in runs])}

    print(f"  {'параметр':<15}{'заложено':>11}{'оценка, ср.':>13}{'std прогонов':>14}"
          f"{'оценка/заложено':>17}{'SD Фишера':>12}")
    for j, p in enumerate(PARAM_ORDER):
        col = est[:, j]
        std = np.std(col, ddof=1) if len(col) > 1 else float("nan")
        sd = f"{sd_f[p]:12.2e}" if p in sd_f else f"{'':>12}"
        print(f"  {PARAM_LABEL[p]:<15}{truth[p]:11.3e}{np.mean(col):13.3e}{std:14.2e}"
              f"{ratio_text(np.median(col), truth[p], 17)}{sd}")
    print("  (оценка/заложено -- медиана по прогонам; SD Фишера -- средняя неопределённость "
          "одной оценки,\n   сравнивать со столбцом «std прогонов»; н/д -- заложено 0)")

    rms_m = np.mean([r["rms"] for r in runs], axis=0)
    print(f"  RMS ошибки состояния (A, B, ph[рад]): "
          f"{rms_m[0]:.2e}, {rms_m[1]:.2e}, {rms_m[2]:.2e}")

    dJ = [r["dJ"] for r in runs if r.get("dJ") is not None]
    if dJ:
        print(f"  Δ(-2lnL) = (J - J_эталон)*N: медиана {np.median(dJ):+.2f} "
              f"[мин {min(dJ):+.2f}, макс {max(dJ):+.2f}]  "
              f"(ориентир при согласованной модели ~ -5, хи-квадрат с 5 параметрами)")
    print(f"  инновации: std(nu) в среднем {np.mean([r['nu_std'] for r in runs]):.3f} "
          f"(ожидается 1), Льюнг-Бокс p, медиана {np.median([r['lb_p'] for r in runs]):.3f}")

    cnt = Counter()
    for r in runs:
        for nm in r["at_bound"]:
            cnt[Q_NAME.get(nm, nm)] += 1
        for nm in sigma_at_bound(r["est"], tcfg):
            cnt[nm] += 1
    bound = ", ".join(f"{k}: {v}/{len(runs)}" for k, v in cnt.items()) or "нет"
    print(f"  у границы допуска: {bound}")


def print_report(sim, fit, truth, results, tcfgs, orc, dt):
    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТ   [{dt:.0f} с]")
    print("=" * 78)
    print(f"N_sim = {sim.n_sim}, прогонов = {sim.n_runs}, poi = {fit.poi}, "
          f"warmup = {fit.warmup}")
    print("ЗАЛОЖЕНО: " + ", ".join(f"{PARAM_LABEL[p].split(',')[0]} = {truth[p]:.3e}"
                                   for p in PARAM_ORDER))

    orc_ok = [o for o in orc if o]
    if orc_ok:
        rms_o = np.mean([o["rms"] for o in orc_ok], axis=0)
        print(f"ЭТАЛОН (EKF с заложенными параметрами, {len(orc_ok)}/{len(orc)} прогонов): "
              f"J = {np.mean([o['J'] for o in orc_ok]):.5f}, RMS состояния (A, B, ph[рад]) = "
              f"{rms_o[0]:.2e}, {rms_o[1]:.2e}, {rms_o[2]:.2e}")

    for m in fit.methods:
        print_method(m, fit, results[m], truth, tcfgs[m], sim.n_runs)


def run_test(sim: SimCfg, fit: FitCfg):
    """Сгенерировать данные, прогнать автоподбор, сравнить с заложенным."""
    if sim.n_sim - fit.poi <= fit.warmup + 10:
        raise ValueError("n_sim слишком мал для poi + warmup")
    if fit.sigma_init_mode == "manual" and fit.sigma_init is None:
        raise ValueError("для sigma_init_mode='manual' задайте sigma_init=(sA, sph)")
    if ekf.N_RP != N_RP:
        raise ValueError(f"N_RP различается: {ekf.N_RP} и {N_RP}")

    kz_fit = sim.Kz if fit.kz_fit is None else fit.kz_fit
    tau_fit = sim.delay if fit.tau_fit is None else fit.tau_fit

    weight_sim = sim_weights(sim.T, sim.ty, sim.t_step)
    weight_ekf, win_len = ekf.build_weight_vec(sim.T, sim.ty, sim.t_step)
    if win_len != len(weight_sim):
        raise ValueError(f"длина окна различается: {win_len} и {len(weight_sim)}")
    for name, tau in (("delay", sim.delay), ("tau_fit", tau_fit)):
        if tau < 0 or tau + win_len > N_RP:
            raise ValueError(f"{name} + win_len = {tau + win_len} > N_RP = {N_RP}")

    ekf._jit_warmup()

    tcfgs = {m: ekf.TuneCfg(**{**fit.tune,
                               "sigma_init_mode": fit.sigma_init_mode,
                               "sigma_init": fit.sigma_init,
                               **mcfg})
             for m, mcfg in fit.methods.items()}
    results = {m: [] for m in fit.methods}
    orc = []
    truth = None
    rng = np.random.default_rng(sim.seed)
    t_all = time.perf_counter()

    for r in range(sim.n_runs):
        t_run = time.perf_counter()
        seed_vib = int(rng.integers(0, 2 ** 31 - 1))
        seed_noise = int(rng.integers(0, 2 ** 31 - 1))
        data = generate(sim, weight_sim, weight_ekf, win_len, tau_fit,
                        seed_vib, seed_noise,
                        verbose=(sim.verbose_first_run and r == 0))
        truth = data["truth"]

        o = run_oracle(data, sim, fit, kz_fit)
        orc.append(o)

        for m in fit.methods:
            out = run_method(data, sim, fit, tcfgs[m], kz_fit)
            if out is None:
                continue
            out["dJ"] = (out["J"] - o["J"]) * o["n"] if o else None
            results[m].append(out)
            if fit.verbose_runs:
                e = out["est"]
                print(f"    {m}: sigma_A={e['sigma_A']:.3e} sigma_ph={e['sigma_ph']:.3e} "
                      f"dA={e['dA']:.2e} dB={e['dB']:.2e} dph={e['dph']:.2e}")
        print(f"  прогон {r + 1}/{sim.n_runs} готов, {time.perf_counter() - t_run:.0f} с")

    print_report(sim, fit, truth, results, tcfgs, orc, time.perf_counter() - t_all)


# =====================================================================
# 5. main: ВСЕ ПАРАМЕТРЫ ЗАДАЮТСЯ ЗДЕСЬ
# =====================================================================

def main():
    # ==================================================================
    # (А) ПАРАМЕТРЫ СИМУЛЯЦИИ
    # ==================================================================
    sim = SimCfg(
        # --- интерферометр ---
        T=10e-3,                    # длительность плеча, с
        ty=20e-6,                   # длительность импульса, с
        T_RP=33e-3,                 # запись акселерометра на сброс, с
        # --- набор данных ---
        n_runs=10,                  # число независимых прогонов
        n_sim=2000,                 # сбросов в одном прогоне
        alp_amount=200,             # точек развёртки alp в одном фринджe
        seed=123,                   # сид Монте-Карло
        # --- вибрация ---
        vib_state="mooring",        # 'mooring' | 'sailing'
        hp_cutoff=0.01,             # срез ВЧ-фильтра акселерометра, Гц
        platform_atten_db=0.0,      # ослабление сырой вибрации, дБ
        Kz=1.0,                     # коэффициент связи вибрации z с фазой
        delay=0,                    # смещение окна интерферометра, отсчёты
        # --- фринж ---
        A0=0.15,                    # начальное A
        B0=0.21,                    # начальное B
        dA_step=5e-3,               # шаг блуждания A за сброс
        dB_step=5e-3,               # шаг блуждания B за сброс
        # --- шумы ---
        sigma_A_sim=7e-3,           # белый шум детектирования P
        sigma_ph_sim=5.6e-3,        # белый фазовый шум на сброс, рад (0 -- нет)
        # --- дрейф g ---
        drift_corr=1000,             # время корреляции, сбросов
        Dg_drift=500e-8,            # амплитуда дрейфа
        # --- вывод ---
        verbose_first_run=False)    # диагностика масштаба вибрации в 1-м прогоне

    # ==================================================================
    # (Б) ПАРАМЕТРЫ ОБРАБОТКИ: компенсация, начальные Q и R, настройка EKF
    # ==================================================================
    fit = FitCfg(
        # --- компенсация вибрации при обработке ---
        kz_fit=None,                # None -> Kz генерации; иначе проверка при неточном Kz
        tau_fit=None,               # None -> delay генерации; иначе проверка при неточном tau
        # --- начальный фит и фильтр ---
        poi=200,                    # точек для начального линейного фита
        warmup=200,                 # инноваций после poi, не входящих в fitness
        # --- начальные Q ---
        q_init_std=(1e-3, 1e-3, 1e-3),      # std шага (dA, dB, dph)
        # --- начальные R (только для первого прогона EKF) ---
        sigma_init_mode="bins",     # "bins" | "split" | "manual"
        sigma_init=None,            # (sigma_A, sigma_ph) при "manual"
        # --- настройка EKF (поля ekf.TuneCfg) ---
        # сюда же можно добавить: границы sigma (sA_lo, sA_hi, sph_lo, sph_hi),
        # q_decades, q_log_bounds, n_grid, n_zoom, use_state_cov и т.д.
        tune=dict(tune_mode="alternate",    # "alternate" | "joint"
                  q_optimizer="lbfgs",      # "lbfgs" | "nm"
                  n_outer=7,
                  q_free=(True, True, True)),
        methods={                   # методы оценки R (отличаются полями TuneCfg)
            "ML (Фишер)": dict(r_method="fisher"),
            "Байес (сетка)": dict(r_method="grid"),
            # "joint ML": dict(tune_mode="joint"),
        },
        # --- вывод ---
        verbose_runs=False)         # печатать оценки каждого прогона

    run_test(sim, fit)


if __name__ == "__main__":
    main()