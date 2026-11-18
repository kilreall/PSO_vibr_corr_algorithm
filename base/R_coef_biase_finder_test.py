# Проверка автоподбора параметров EKF (sigma_A, sigma_ph, Q) на синтетических
# данных с ИЗВЕСТНОЙ истиной.
#
# Данные генерирует gen_waveform.simul_wavefrom (импортируется, не дублируется):
# вибрация по x, y, z, дрейф ph (процесс Орнштейна-Уленбека), блуждание A и B,
# белый фазовый шум и шум детектирования P. Генератор принимает параметры
# по именам и отдаёт кортеж из 15 элементов:
#   (alp, P_noise, P_clean, ph_total, ph_sim, A, B, dA, dB, dph,
#    sigma_A, sigma_ph, az_m, ax_m, ay_m)
# где ph_sim -- чистый OU-дрейф (то, что оценивает EKF), а
# ph_total = ph_sim + белый фазовый шум (то, что реально шло в фринж).
# Для сравнения с оценкой состояния EKF истиной служит ph_sim.
#
# Файл добавляет только обвязку:
#   * SimCfg      -- контейнер параметров генерации (в генератор уходят по именам);
#   * sim_weights -- веса интеграла чувствительности (в генераторе они считаются
#                    внутри simul_wavefrom и наружу не отдаются);
#   * обработку одной осью z (x и y в данных есть через Kx, Ky, но при
#     обработке их никто не измеряет и не компенсирует).
#
# Структура файла:
#   1. КОНСТАНТЫ            -- имя EKF-модуля, порядок параметров;
#   2. Классы параметров    -- SimCfg (генерация), McCfg (Монте-Карло) и FitCfg
#                              (обработка);
#   3. Генерация            -- generate(): simul_wavefrom + Fz по измеренному az;
#   4. Обработка и печать   -- запуск EKF-автоподбора, сравнение с истиной;
#   5. main()               -- ВСЕ параметры задаются здесь, внизу файла.
#
# Что делает тест:
#   1. Генерирует данные (simul_wavefrom: вибрация по x, y, z, дрейф ph, блуждание
#      A и B, белый фазовый шум, шум детектирования sigma_A_sim в P).
#   2. Компенсирует alp по ускорению ТОЛЬКО по оси z (как на реальных данных).
#   3. Запускает ту же настройку, что в kalman_pso_fit.py (tune_and_score):
#      стартовые sigma -> EKF -> оценка R (Фишер / байесовская сетка) -> подбор Q.
#   4. Сравнивает результат с заложенным по n_runs независимым прогонам и с
#      «эталонным» EKF, которому отдают заложенные значения.
#
# Заложено в генерацию:
#   sigma_A   -- белый шум детектирования P (sigma_A_sim);
#   sigma_ph  -- белый фазовый шум на сброс, рад (sigma_ph_sim);
#   dA, dB    -- шаги блуждания A и B за сброс (dA_step, dB_step);
#   dph       -- std шага дрейфа ph (процесс Орнштейна-Уленбека):
#                dph = Dph_drift*sqrt(theta*(2 - theta)), theta = 1 - exp(-1/drift_corr).
#                EKF моделирует ph как случайное блуждание, так что dph -- эффективный
#                шаг; чтобы модель EKF была адекватна, drift_corr должно заметно
#                превышать n_sim (иначе OU «возвращается к среднему» внутри прогона).
#
# РАЗВЁРТКА alp в генераторе: alp пробегает ±1/(5 T²) вокруг keff*g0/(2π), то есть
# фаза Ph = 2π T² alp - ph меняется на ±2π/5 ≈ ±1.26 рад около экстремума фринджа
# (охвачено ~40% периода). Дрейф ph должен быть заметно меньше 1.26 рад, иначе
# развёртка «уйдёт» с экстремума.
#
# ВАЖНО: вибрация по x, y (и остаток по z при неточных kz_fit / tau_fit, шум
# акселерометра) не компенсируется и попадает в фазу как дополнительный шум.
# Поэтому оценка sigma_ph будет больше заложенного sigma_ph_sim; в отчёте
# печатается эффективное значение, с которым её и надо сравнивать.
#
# СИД ВИБРАЦИИ: в текущем gen_waveform generate_vibration вызывается с seed=42
# жёстко, поэтому вибрация во всех прогонах Монте-Карло ОДНА И ТА ЖЕ (меняются
# только шумы A, B, ph, детектирования и акселерометра). Чтобы прогоны были
# независимы, добавьте в simul_wavefrom параметр vib_seed=42 и передайте его в
# generate_vibration(seed=vib_seed): этот файл подхватит его автоматически.
#
# ПАМЯТЬ: генератор держит записи ускорений по трём осям, это
# 3 * n_sim * N_RP * 8 байт (при n_sim = 2000 около 0.8 ГБ) на каждый прогон;
# прогоны идут последовательно, память между ними освобождается.
#
# Файлы gen_waveform.py (и gen_acc.py, который он импортирует) и EKF_MODULE
# должны быть доступны для импорта.
# Запуск:  python test_ekf_vs_truth.py

import importlib
import inspect
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from main.gen_waveform import simul_wavefrom, plot_waveform, fa, keff
    from main.gen_waveform import N_RP as GEN_N_RP
except ImportError:                      # gen_waveform.py лежит рядом со скриптом
    from gen_waveform import simul_wavefrom, plot_waveform, fa, keff
    from gen_waveform import N_RP as GEN_N_RP

# =====================================================================
# 1. КОНСТАНТЫ
# =====================================================================

EKF_MODULE = "PSO_coef_finder_real_v2"   # файл с EKF и автоподбором (без .py)

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
    """Параметры simul_wavefrom. N_RP в генераторе -- константа модуля."""
    # --- интерферометр ---
    T: float                    # длительность плеча, с
    ty: float                   # длительность импульса, с (tau в генераторе)
    T_RP: float                 # запись акселерометра на сброс, с
    delay: int                  # смещение окна интерферометра, отсчёты
    # --- набор данных ---
    n_sim: int                  # сбросов в одном прогоне
    alp_amount: int             # точек развёртки alp в одном фринджe
    # --- вибрация (генерируются все три оси, обработка видит только z) ---
    vib_state: str              # 'mooring' | 'sailing'
    Kz: float                   # коэффициент связи по z
    Kx: float                   # по x (не компенсируется)
    Ky: float                   # по y (не компенсируется)
    sigma_a_acc: float          # шум акселерометра, м/с^2
    # --- фринж и шумы ---
    dA_step: float              # шаг блуждания A за сброс
    dB_step: float              # шаг блуждания B за сброс
    sigma_A_sim: float          # белый шум детектирования P
    sigma_ph_sim: float         # белый фазовый шум на сброс, рад
    # --- дрейф фазы ph ---
    drift_corr: float           # время корреляции OU, сбросов
    Dph_drift: float            # стационарная std дрейфа ph (OU), рад

    @property
    def N_RP(self) -> int:
        return GEN_N_RP

    @property
    def t_step(self) -> float:
        return self.T_RP / GEN_N_RP


@dataclass
class McCfg:
    """Монте-Карло."""
    n_runs: int                 # число независимых прогонов
    seed: int                   # сид Монте-Карло (сиды прогонов выводятся из него)
    show_first_run: bool        # сводка и графики генератора (plot_waveform) для 1-го прогона


@dataclass
class FitCfg:
    """Параметры обработки: компенсация, начальные Q и R, настройка EKF."""
    # --- компенсация вибрации при обработке (только ось z) ---
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
# 3. ГЕНЕРАЦИЯ ОДНОГО ПРОГОНА
# =====================================================================

def sim_weights(T, ty, t_step):
    """
    Веса интеграла чувствительности: keff * (a[окно] @ w) -- интеграл fa(t)*a(t)
    по окну интерферометра. Построены ровно так же, как внутри simul_wavefrom
    (там они наружу не отдаются), поэтому при правке генератора правьте и здесь.
    """
    t_range = np.linspace(0, 2 * T + 4 * ty, round((2 * T + 4 * ty) / t_step))
    fa_t = np.vectorize(fa, otypes=[float])(t_range, T, ty)
    w = np.full(len(t_range), t_step)
    w[0] *= 0.5
    w[-1] *= 0.5
    return fa_t * w


def generate(sim: SimCfg, mc: McCfg, weight_ekf, win_len, w_gen, kz_fit, tau_fit,
             seed_gen, show_summary=False):
    """
    Данные одного прогона + истинные значения.

    simul_wavefrom отдаёт P уже с шумом детектирования и измеренные (с шумом
    акселерометра) ускорения по трём осям. Обработка видит только ось z:
    Fz -- интеграл чувствительности по измеренному az на окне tau_fit (без Kz;
    Kz применяется в compensate_alp). Оси x, y входят только в истинную фазу.

    Истинное состояние для сравнения с EKF: A, B и ph_sim (чистый OU-дрейф без
    белого фазового шума; белый шум EKF учитывает через sigma_ph в R).
    """
    # генератор берёт шум (A, B, ph, детектирование, акселерометр) из глобального
    # np.random
    np.random.seed(seed_gen)
    kwargs = dict(
        N_sim=sim.n_sim, alp_amount=sim.alp_amount, T_RP=sim.T_RP, T=sim.T,
        tau=sim.ty, delay=sim.delay, Kz=sim.Kz, Kx=sim.Kx, Ky=sim.Ky,
        vib_state=sim.vib_state, sigma_a=sim.sigma_a_acc,
        Dph_drift=sim.Dph_drift, drift_corr=sim.drift_corr,
        sigma_A_sim=sim.sigma_A_sim, sigma_ph_sim=sim.sigma_ph_sim,
        dA_sim=sim.dA_step, dB_sim=sim.dB_step,
        DA_sim=0.0)             # DA_sim лишь добавляется к возвращаемому dA
    # сид вибрации: если в simul_wavefrom есть параметр vib_seed -- меняем его
    # от прогона к прогону, иначе вибрация во всех прогонах одна и та же
    if "vib_seed" in inspect.signature(simul_wavefrom).parameters:
        kwargs["vib_seed"] = seed_gen
    out = simul_wavefrom(**kwargs)
    (alp, P_noise, _P_clean, ph_total, ph_sim, A, B, dA, dB, dph, sigma_A,
     sigma_ph, az_m, ax_m, ay_m) = out

    if show_summary:
        plot_waveform(out, T=sim.T, tau=sim.ty, T_RP=sim.T_RP, delay=sim.delay,
                      Kz=sim.Kz, Kx=sim.Kx, Ky=sim.Ky, sigma_a=sim.sigma_a_acc,
                      Dph_drift=sim.Dph_drift, title="прогон 1")

    Fz = ekf.vibration_phase(az_m, tau_fit, weight_ekf, win_len)

    # веса EKF и генератора должны давать один и тот же интеграл
    Fz_ref = keff * (az_m[:, tau_fit:tau_fit + win_len] @ w_gen)
    if not np.allclose(Fz, Fz_ref, rtol=1e-6, atol=1e-9 * np.max(np.abs(Fz_ref))):
        raise ValueError("ekf.vibration_phase не совпадает с интегралом по весам "
                         "генератора: проверьте веса, keff и окно")

    # фаза, не убранная компенсацией: x, y, ошибки kz/tau (по измеренным a, как в
    # plot_waveform) + шум акселерометра, попадающий в компенсацию через az.
    # При tau_fit != delay и sigma_a_acc > 0 оценка слегка завышена шумом.
    sl = slice(sim.delay, sim.delay + win_len)
    Fx = keff * (ax_m[:, sl] @ w_gen)
    Fy = keff * (ay_m[:, sl] @ w_gen)
    Fz_d = keff * (az_m[:, sl] @ w_gen)
    r_det = rms(sim.Kx * Fx + sim.Ky * Fy + sim.Kz * Fz_d - kz_fit * Fz)
    r_noise = kz_fit * keff * sim.sigma_a_acc * float(np.linalg.norm(w_gen))
    resid_rms = float(np.hypot(r_det, r_noise))

    truth = {"sigma_A": float(sigma_A), "sigma_ph": float(sigma_ph),
             "dA": float(dA), "dB": float(dB), "dph": float(dph)}
    return {"alp": alp, "y": P_noise, "Fz": Fz, "A": A, "B": B,
            "ph": ph_sim, "ph_total": ph_total, "truth": truth,
            "resid_rms": resid_rms}


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
    RMS ошибки оценок состояния [A, B, ph] относительно истины (ph -- ph_sim,
    чистый OU-дрейф). x_est[i] -- оценка после сброса poi + i; берутся точки
    после warmup.
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


def print_report(sim, mc, fit, truth, results, tcfgs, orc, resid, dt):
    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТ   [{dt:.0f} с]")
    print("=" * 78)
    print(f"N_sim = {sim.n_sim}, прогонов = {mc.n_runs}, poi = {fit.poi}, "
          f"warmup = {fit.warmup}")
    print("ЗАЛОЖЕНО: " + ", ".join(f"{PARAM_LABEL[p].split(',')[0]} = {truth[p]:.3e}"
                                   for p in PARAM_ORDER))

    r_rms = float(np.mean(resid))
    s_eff = float(np.hypot(truth["sigma_ph"], r_rms))
    print(f"НЕКОМПЕНСИРОВАННАЯ вибрационная фаза (x, y, шум акселерометра, kz/tau): "
          f"RMS = {r_rms:.3e} рад")
    print(f"  => эффективный sigma_ph ≈ {s_eff:.3e} рад "
          f"(оценку sigma_ph сравнивать с ним, а не с заложенным {truth['sigma_ph']:.3e})")

    orc_ok = [o for o in orc if o]
    if orc_ok:
        rms_o = np.mean([o["rms"] for o in orc_ok], axis=0)
        print(f"ЭТАЛОН (EKF с заложенными параметрами, {len(orc_ok)}/{len(orc)} прогонов): "
              f"J = {np.mean([o['J'] for o in orc_ok]):.5f}, RMS состояния (A, B, ph[рад]) = "
              f"{rms_o[0]:.2e}, {rms_o[1]:.2e}, {rms_o[2]:.2e}")

    for m in fit.methods:
        print_method(m, fit, results[m], truth, tcfgs[m], mc.n_runs)


def run_test(sim: SimCfg, mc: McCfg, fit: FitCfg):
    """Сгенерировать данные, прогнать автоподбор, сравнить с заложенным."""
    if sim.n_sim - fit.poi <= fit.warmup + 10:
        raise ValueError("n_sim слишком мал для poi + warmup")
    if fit.sigma_init_mode == "manual" and fit.sigma_init is None:
        raise ValueError("для sigma_init_mode='manual' задайте sigma_init=(sA, sph)")
    if ekf.N_RP != sim.N_RP:
        raise ValueError(f"N_RP различается: EKF {ekf.N_RP} и генератор {sim.N_RP}")

    kz_fit = sim.Kz if fit.kz_fit is None else fit.kz_fit
    tau_fit = sim.delay if fit.tau_fit is None else fit.tau_fit

    weight_ekf, win_len = ekf.build_weight_vec(sim.T, sim.ty, sim.t_step)
    w_gen = sim_weights(sim.T, sim.ty, sim.t_step)
    if win_len != len(w_gen):
        raise ValueError(f"длина окна различается: EKF {win_len} и генератор {len(w_gen)}")
    for name, tau in (("delay", sim.delay), ("tau_fit", tau_fit)):
        if tau < 0 or tau + win_len > sim.N_RP:
            raise ValueError(f"{name} + win_len = {tau + win_len} > N_RP = {sim.N_RP}")

    # консистентность дрейфа с моделью EKF (случайное блуждание)
    if sim.drift_corr < 5 * sim.n_sim:
        print(f"ВНИМАНИЕ: drift_corr = {sim.drift_corr:g} < 5*n_sim = {5 * sim.n_sim}: "
              f"OU заметно возвращается к среднему внутри прогона, EKF (случайное "
              f"блуждание ph) модель дрейфа описывает хуже.")
    if "vib_seed" not in inspect.signature(simul_wavefrom).parameters:
        print("ВНИМАНИЕ: в simul_wavefrom нет vib_seed -- вибрация во всех прогонах "
              "одна и та же (seed=42 внутри генератора).")

    ekf._jit_warmup()

    tcfgs = {m: ekf.TuneCfg(**{**fit.tune,
                               "sigma_init_mode": fit.sigma_init_mode,
                               "sigma_init": fit.sigma_init,
                               **mcfg})
             for m, mcfg in fit.methods.items()}
    results = {m: [] for m in fit.methods}
    orc = []
    resid = []
    truth = None
    rng = np.random.default_rng(mc.seed)
    t_all = time.perf_counter()

    for r in range(mc.n_runs):
        t_run = time.perf_counter()
        seed_gen = int(rng.integers(0, 2 ** 31 - 1))
        data = generate(sim, mc, weight_ekf, win_len, w_gen, kz_fit, tau_fit,
                        seed_gen, show_summary=(mc.show_first_run and r == 0))
        truth = data["truth"]
        resid.append(data["resid_rms"])

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
        print(f"  прогон {r + 1}/{mc.n_runs} готов, {time.perf_counter() - t_run:.0f} с")

    print_report(sim, mc, fit, truth, results, tcfgs, orc, resid,
                 time.perf_counter() - t_all)
    if mc.show_first_run:
        plt.show()              # графики 1-го прогона показываются после отчёта


# =====================================================================
# 5. main: ВСЕ ПАРАМЕТРЫ ЗАДАЮТСЯ ЗДЕСЬ
# =====================================================================

def main():
    # Значения подобраны так, чтобы все пять параметров (sigma_A, sigma_ph, dA, dB,
    # dph) были различимы на n_sim = 2000 сбросах:
    #   * sigma_ph заметно выше порога определимости (см. «Порог определимости» в
    #     сводке plot_waveform, ~2.4e-3 рад для этих sigma_A и B) -- вклад
    #     B²sin²Ph·sigma_ph² в дисперсию P сравним с sigma_A², и их можно разделить
    #     по зависимости дисперсии невязки от sin²Ph (развёртка охватывает ±1.26 рад);
    #   * шаги dA, dB, dph достаточно велики, чтобы блуждание было видно на фоне шума
    #     за 2000 сбросов, но A, B и ph не уходят далеко (A, B -- на ~0.04, ph -- на
    #     ~0.35 рад, развёртка ±1.26 рад остаётся вокруг экстремума);
    #   * drift_corr = 10*n_sim, поэтому OU ≈ случайное блуждание, как в модели EKF.
    # Ожидаемая относительная точность одной оценки (асимптотика максимального
    # правдоподобия, реплика фильтра, 12 реализаций): sigma_A ~5%, sigma_ph ~5%,
    # dA ~25%, dB ~25%, dph ~13%. dA и dB различимы хуже всего: A, B и ph
    # коррелированы на неполном периоде фринджа.

    # ==================================================================
    # (А) ПАРАМЕТРЫ ГЕНЕРАЦИИ (gen_waveform.simul_wavefrom) И МОНТЕ-КАРЛО
    # ==================================================================
    sim = SimCfg(
        # --- интерферометр ---
        T=10e-3,                    # длительность плеча, с
        ty=20e-6,                   # длительность импульса, с
        T_RP=33.556e-3,             # запись акселерометра на сброс, с
        delay=700,                  # смещение окна интерферометра, отсчёты
        # --- набор данных ---
        n_sim=2000,                 # сбросов в одном прогоне (10 фринджей)
        alp_amount=200,             # точек развёртки alp в одном фринджe
        # --- вибрация (генерируются все три оси, обработка видит только z) ---
        vib_state="mooring",        # 'mooring' | 'sailing'
        Kz=0.9,                     # коэффициент связи по z
        Kx=0.003,                   # коэффициент связи по x (не компенсируется)
        Ky=0.003,                   # коэффициент связи по y (не компенсируется)
        sigma_a_acc=3e-5,           # шум акселерометра, м/с^2 (даёт ~5e-4 рад в компенсации)
        # --- фринж и шумы ---
        dA_step=8e-4,               # шаг блуждания A за сброс
        dB_step=8e-4,               # шаг блуждания B за сброс
        sigma_A_sim=2e-3,           # белый шум детектирования P
        sigma_ph_sim=2e-2,          # белый фазовый шум на сброс, рад
        # --- дрейф фазы ph ---
        drift_corr=20000,           # время корреляции OU, сбросов (>> n_sim)
        Dph_drift=0.8)              # стационарная std дрейфа ph, рад; шаг dph ≈ 8e-3 рад

    mc = McCfg(
        n_runs=10,                  # число независимых прогонов
        seed=123,                   # сид Монте-Карло
        show_first_run=False)       # сводка и графики генератора для 1-го прогона

    # ==================================================================
    # (Б) ПАРАМЕТРЫ ОБРАБОТКИ: компенсация, начальные Q и R, настройка EKF
    # ==================================================================
    fit = FitCfg(
        # --- компенсация вибрации при обработке (только z) ---
        kz_fit=None,                # None -> Kz генерации; иначе проверка при неточном Kz
        tau_fit=None,               # None -> delay генерации; иначе проверка при неточном tau
        # --- начальный фит и фильтр ---
        poi=200,                    # точек для начального линейного фита (один фринж)
        warmup=200,                 # инноваций после poi, не входящих в fitness
        # --- начальные Q ---
        # близко к истине (в пределах ~2 раз), но не равно ей: dA, dB = 8e-4, dph ≈ 8e-3
        q_init_std=(5e-4, 5e-4, 4e-3),      # std шага (dA, dB, dph)
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

    run_test(sim, mc, fit)


if __name__ == "__main__":
    main()