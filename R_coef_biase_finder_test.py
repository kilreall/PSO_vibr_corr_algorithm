# Проверка автоподбора параметров EKF (sigma_A, sigma_ph, Q) на синтетических
# данных с ИЗВЕСТНОЙ истиной.
#
# Генерация фринджа НЕ дублируется здесь: она импортируется из gen_fringe.py
# (SimCfg, simulate_fringe). Этот файл добавляет к ней только то, чего в
# генераторе быть не должно:
#   * белый шум детектирования P (sigma_A_sim) -- добавляется в generate();
#   * обработку одной осью z (x и y в данных есть через Kx, Ky, но при
#     обработке их никто не измеряет и не компенсирует).
#
# Структура файла:
#   1. КОНСТАНТЫ            -- имя EKF-модуля, порядок параметров;
#   2. Классы параметров    -- McCfg (Монте-Карло + шум детектирования) и FitCfg
#                              (обработка); параметры симуляции -- gen_fringe.SimCfg;
#   3. Генерация            -- generate(): simulate_fringe + шум детектирования + Fz;
#   4. Обработка и печать   -- запуск EKF-автоподбора, сравнение с истиной;
#   5. main()               -- ВСЕ параметры задаются здесь, внизу файла.
#
# Что делает тест:
#   1. Генерирует данные (gen_fringe: вибрация по x, y, z, дрейф ph, блуждание
#      A и B, белый фазовый шум) и добавляет шум детектирования sigma_A_sim к P.
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
#   dph       -- std шага дрейфа ph (процесс Орнштейна-Уленбека). EKF моделирует
#                ph как случайное блуждание, так что dph -- эффективный шаг.
#
# ВАЖНО: вибрация по x, y (и остаток по z при неточных kz_fit / tau_fit, шум
# акселерометра) не компенсируется и попадает в фазу как дополнительный шум.
# Поэтому оценка sigma_ph будет больше заложенного sigma_ph_sim; в отчёте
# печатается эффективное значение, с которым её и надо сравнивать.
#
# Файлы gen_fringe.py (и gen_acc.py, который он импортирует) и EKF_MODULE
# должны лежать рядом с этим скриптом.
# Запуск:  python test_ekf_vs_truth.py

import importlib
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

from main.gen_waveform import SimCfg, simulate_fringe, sim_weights, print_summary

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
class McCfg:
    """Монте-Карло и то, что добавляется к данным генератора."""
    n_runs: int                 # число независимых прогонов
    seed: int                   # сид Монте-Карло (сиды прогонов выводятся из него)
    sigma_A_sim: float          # белый шум детектирования P (в генераторе его нет)
    verbose_first_run: bool     # печатать сводку генератора для 1-го прогона


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

def generate(sim: SimCfg, mc: McCfg, weight_ekf, win_len, kz_fit, tau_fit,
             seed_gen, seed_det, show_summary=False):
    """
    Данные одного прогона + истинные значения.

    simulate_fringe даёт P с белым фазовым шумом; шум детектирования sigma_A
    добавляется здесь. Обработка видит только ось z: Fz -- интеграл
    чувствительности по измеренному az на окне tau_fit (без Kz; Kz применяется
    в compensate_alp). Оси x, y входят только в истинную фазу.
    """
    # записи ускорений нужны только если окно обработки не совпадает с окном генерации
    need_acc = tau_fit != sim.delay
    cfg = replace(sim, seed=seed_gen, keep_acc=need_acc)
    d = simulate_fringe(cfg, verbose=False)
    if show_summary:
        print_summary(d, cfg)

    if need_acc:
        Fz = ekf.vibration_phase(d["az_m"], tau_fit, weight_ekf, win_len)
    else:
        Fz = d["Fz_raw"]

    y = d["P_noise"] + np.random.default_rng(seed_det).normal(
        0, mc.sigma_A_sim, cfg.n_sim)

    # фаза, не убранная компенсацией: x, y, шум акселерометра, ошибки kz/tau
    resid_rms = rms(d["F_vib"] - kz_fit * Fz)

    truth = {"sigma_A": float(mc.sigma_A_sim), "sigma_ph": d["sigma_ph"],
             "dA": d["dA"], "dB": d["dB"], "dph": d["dph"]}
    return {"alp": d["alp"], "y": y, "Fz": Fz, "A": d["A"], "B": d["B"],
            "ph": d["ph"], "truth": truth, "resid_rms": resid_rms}


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
    n_gen = len(sim_weights(sim.T, sim.ty, sim.t_step))
    if win_len != n_gen:
        raise ValueError(f"длина окна различается: EKF {win_len} и генератор {n_gen}")
    for name, tau in (("delay", sim.delay), ("tau_fit", tau_fit)):
        if tau < 0 or tau + win_len > sim.N_RP:
            raise ValueError(f"{name} + win_len = {tau + win_len} > N_RP = {sim.N_RP}")

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
        seed_det = int(rng.integers(0, 2 ** 31 - 1))
        data = generate(sim, mc, weight_ekf, win_len, kz_fit, tau_fit,
                        seed_gen, seed_det,
                        show_summary=(mc.verbose_first_run and r == 0))
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


# =====================================================================
# 5. main: ВСЕ ПАРАМЕТРЫ ЗАДАЮТСЯ ЗДЕСЬ
# =====================================================================

def main():
    # ==================================================================
    # (А) ПАРАМЕТРЫ СИМУЛЯЦИИ (gen_fringe.SimCfg) И МОНТЕ-КАРЛО
    # ==================================================================
    sim = SimCfg(
        # --- интерферометр ---
        T=10e-3,                    # длительность плеча, с
        ty=20e-6,                   # длительность импульса, с
        T_RP=33.556e-3,             # запись акселерометра на сброс, с
        N_RP=16384,                 # отсчётов акселерометра на сброс
        # --- набор данных ---
        n_sim=2000,                 # сбросов в одном прогоне
        alp_amount=200,             # точек развёртки alp в одном фринджe
        seed=0,                     # не используется: сиды прогонов задаёт McCfg.seed
        # --- вибрация (генерируются все три оси, обработка видит только z) ---
        vib_state="mooring",        # 'mooring' | 'sailing'
        hp_cutoff=0.01,             # срез ВЧ-фильтра акселерометра, Гц
        Kz=0.9,                     # коэффициент связи по z
        Kx=0.003,                   # коэффициент связи по x (не компенсируется)
        Ky=0.003,                   # коэффициент связи по y (не компенсируется)
        delay=700,                  # смещение окна интерферометра, отсчёты
        sigma_a_acc=0.0,            # шум акселерометра, м/с^2 (в gen_fringe по умолчанию 3e-5)
        # --- фринж ---
        A0=0.15,                    # начальное A
        B0=0.21,                    # начальное B
        dA_step=5e-3,               # шаг блуждания A за сброс
        dB_step=5e-3,               # шаг блуждания B за сброс
        # --- шум ---
        sigma_ph_sim=5.6e-3,        # белый фазовый шум на сброс, рад
        # --- дрейф фазы ph ---
        drift_corr=1000,            # время корреляции, сбросов
        Dph_drift=0.01,             # амплитуда дрейфа ph, рад
        # --- вывод ---
        keep_acc=False)             # переопределяется в generate() по необходимости

    mc = McCfg(
        n_runs=10,                  # число независимых прогонов
        seed=123,                   # сид Монте-Карло
        sigma_A_sim=7e-3,           # белый шум детектирования P (добавляется в тесте)
        verbose_first_run=False)    # сводка генератора для 1-го прогона

    # ==================================================================
    # (Б) ПАРАМЕТРЫ ОБРАБОТКИ: компенсация, начальные Q и R, настройка EKF
    # ==================================================================
    fit = FitCfg(
        # --- компенсация вибрации при обработке (только z) ---
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

    run_test(sim, mc, fit)


if __name__ == "__main__":
    main()