# Проверка автоподбора параметров EKF (sigma_A, sigma_ph, Q) на синтетических
# данных с ИЗВЕСТНОЙ истиной.
#
# Что делает:
#   1. Генерирует данные функцией simul_acc из файла генерации (вибрация по оси z,
#      дрейф g, блуждание A и B, шум детектирования, шум акселерометра).
#   2. Компенсирует alp по ИЗМЕРЕННОМУ ускорению az_m (как на реальных данных).
#   3. Запускает ту же настройку, что в kalman_pso_fit.py (tune_and_score):
#      стартовые sigma -> EKF -> оценка R (Фишер / байесовская сетка) -> подбор Q (L-BFGS).
#   4. Сравнивает результат с тем, что заложено в генерацию, по N_RUNS независимым
#      прогонам, и дополнительно с «эталонным» EKF, которому отдают заложенные
#      значения.
#
# Заложено в генерацию:
#   sigma_A   -- белый шум детектирования P (sigma_A_sim из simul_acc);
#   sigma_ph  -- фазовый шум от шума акселерометра (sigma_ph_vibr из simul_acc);
#   dA, dB    -- шаги блуждания A и B за сброс (dA_step, dB_step);
#   dph       -- std шага дрейфа g (процесс Орнштейна-Уленбека), пересчитанного в
#                фазу (dph_sim). EKF моделирует ph как случайное блуждание, так что
#                dph -- эффективный шаг, а не точное соответствие процессу.
#
# Файлы SIM_MODULE и EKF_MODULE должны лежать рядом с этим скриптом.
# Запуск:  python test_ekf_vs_truth.py
 
import importlib
import time
from collections import Counter
 
import numpy as np
 
# =====================================================================
# НАСТРОЙКИ
# =====================================================================
 
SIM_MODULE = "R_coef_bin_finder_test"   # файл с simul_acc (без .py)
EKF_MODULE = "PSO_coef_finder_real_v2"                 # файл с EKF и автоподбором (без .py)
 
# ---------- генерация (параметры simul_acc) ----------
N_RUNS = 10                # число независимых прогонов
N_SIM = 2000               # сбросов в одном прогоне
ALP_AMOUNT = 200           # точек развёртки alp в одном фринджe
DELAY = 0                  # смещение окна интерферометра в записи ускорения
KZ = 1.0                   # коэффициент связи вибрации z с фазой (генерация)
DA_STEP = 5e-3             # шаг блуждания A за сброс
DB_STEP = 5e-3             # шаг блуждания B за сброс
VIB_STATE = "mooring"      # 'mooring' | 'sailing'
HP_CUTOFF = 0.01
PLATFORM_ATTEN_DB = 0.0
SEED_MC = 123              # из него выводятся сиды вибрации и шумов (одинаковые в сценариях)
 
# Сценарии отличаются шумом акселерометра (SIGMA_A_ACC в файле генерации).
# При 3e-5 фазовый шум ~6e-4 рад и вносит в дисперсию сигнала на ~4 порядка меньше,
# чем sigma_A = 7e-3, поэтому sigma_ph там принципиально плохо определяется.
# sigma_A в simul_acc зашита (7e-3) и из этого скрипта не меняется.
SCENARIOS = [
    dict(name="как в файле генерации (шум акселерометра 3e-5 м/с^2)", sigma_acc=3e-5),
    dict(name="шум акселерометра x100 (3e-3 м/с^2)", sigma_acc=3e-3),
]
 
# ---------- компенсация вибрации при обработке ----------
KZ_FIT = None              # None -> KZ генерации; иначе проверка при неточном Kz
TAU_FIT = None             # None -> DELAY генерации; иначе проверка при неточном tau
                           # (при KZ_FIT != KZ «заложенный» sigma_ph пересчитывается
                           # только по шуму акселерометра, ошибка компенсации в него не входит)
 
# ---------- EKF и автоподбор (как в kalman_pso_fit.py) ----------
POI = 200                  # точек для начального линейного фита
WARMUP = 200               # инноваций после POI, не входящих в fitness
Q_INIT_STD = (5e-3, 5e-3, 1e-3)    # стартовые std шага (dA, dB, dph); в EkfParams уходят квадраты
START_FACTORS = (0.1, 1.0, 10.0)   # множители к Q_INIT_STD: проверка зависимости от старта;
                                   # обязательно должен быть 1.0 (основная таблица)
 
TUNE_COMMON = dict(sigma_init_mode="bins", tune_mode="alternate",
                   q_optimizer="lbfgs", n_outer=3, q_free=(True, True, True))
METHODS = {
    "ML (Фишер)": dict(r_method="fisher"),
    "Байес (сетка)": dict(r_method="grid"),
    # "joint ML": dict(tune_mode="joint"),
}
VERBOSE_RUNS = False       # печатать оценки каждого прогона (старт x1.0)
 
# =====================================================================
 
sim = importlib.import_module(SIM_MODULE)
ekf = importlib.import_module(EKF_MODULE)
 
PARAM_ORDER = ["sigma_A", "sigma_ph", "dA", "dB", "dph"]
PARAM_LABEL = {"sigma_A": "sigma_A", "sigma_ph": "sigma_ph, рад", "dA": "dA (шаг A)",
               "dB": "dB (шаг B)", "dph": "dph, рад"}
Q_NAME = {"A": "dA", "B": "dB", "ph": "dph"}
 
 
# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------
 
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))
 
 
def safe_nanmean(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")
 
 
def q_start_text(factor):
    return ", ".join(f"{v * factor:.1e}" for v in Q_INIT_STD)
 
 
def state_rms(x_est, data):
    """
    RMS ошибки оценок состояния [A, B, ph] относительно истины.
    x_est[i] -- оценка после сброса POI + i; берутся точки после WARMUP.
    """
    A_e, B_e, ph_e = x_est[WARMUP:, 0], x_est[WARMUP:, 1], x_est[WARMUP:, 2]
    s = POI + WARMUP
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
 
 
# ---------------------------------------------------------------------
# Генерация и обработка одного прогона
# ---------------------------------------------------------------------
 
def generate(sc, seed_vib, seed_noise, kz_fit, tau_fit, weight_vec, win_len):
    """Данные одного прогона + истинные значения. az_m не хранится (только Fz)."""
    sim.SIGMA_A_ACC = sc["sigma_acc"]      # simul_acc читает эту константу при вызове
    (alp, P_noise, _P_clean, A_s, B_s, _g, dA_sim, dB_sim, dph_sim, _sig_g,
     sigma_ph_vibr, sigma_A_sim, az_m, extra) = sim.simul_acc(
        N_SIM, ALP_AMOUNT, DELAY, KZ,
        vib_state=VIB_STATE, hp_cutoff=HP_CUTOFF,
        platform_atten_db=PLATFORM_ATTEN_DB,
        seed_vib=seed_vib, seed_noise=seed_noise,
        verbose=False, return_extra=True,
        dA_step=DA_STEP, dB_step=DB_STEP)
 
    # вибрационная фаза по ИЗМЕРЕННОМУ ускорению (без Kz; Kz применяется в compensate_alp)
    Fz = ekf.vibration_phase(az_m, tau_fit, weight_vec, win_len)
    del az_m
 
    truth = {"sigma_A": float(sigma_A_sim),
             "sigma_ph": float(sigma_ph_vibr) * abs(kz_fit) / abs(KZ),
             "dA": float(dA_sim), "dB": float(dB_sim), "dph": float(dph_sim)}
    return {"alp": alp, "y": P_noise, "Fz": Fz, "A": A_s, "B": B_s,
            "ph": extra["ph_sim"], "truth": truth}
 
 
def run_oracle(data, kz_fit):
    """Эталон: тот же EKF и тот же начальный фит, но с ЗАЛОЖЕННЫМИ sigma_A, sigma_ph, Q."""
    alp_comp = ekf.compensate_alp(data["alp"], data["Fz"], kz_fit, sim.T)
    x0, P0, _ = ekf.initial_fit(alp_comp[:POI], data["y"][:POI], sim.T)
    tr = data["truth"]
    q = np.array([tr["dA"], tr["dB"], tr["dph"]], dtype=float) ** 2
    r = ekf.run_ekf(alp_comp[POI:], data["y"][POI:], x0, P0, q,
                    tr["sigma_A"], tr["sigma_ph"], sim.T, WARMUP)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return None
    return {"J": r["nll"] / r["cnt"], "n": r["cnt"], "rms": state_rms(r["x"], data)}
 
 
def run_method(data, tcfg, factor, kz_fit):
    """Автоподбор из kalman_pso_fit.tune_and_score при заданном старте Q."""
    prm = ekf.EkfParams(T=sim.T,
                        q_init=(np.array(Q_INIT_STD, dtype=float) * factor) ** 2,
                        poi=POI, warmup=WARMUP)
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
            "rms": state_rms(info["states"], data),
            "at_bound": list(info["q_at_bound"]),
            "nu_std": nu["std"], "lb_p": nu["lb_pvalue"]}
 
 
# ---------------------------------------------------------------------
# Печать
# ---------------------------------------------------------------------
 
def print_method(name, factor, runs, truth, tcfg, n_runs):
    print(f"\n--- {name}; старт Q: std = ({q_start_text(factor)}); "
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
              f"{np.median(col) / truth[p]:17.2f}{sd}")
    print("  (оценка/заложено -- медиана по прогонам; SD Фишера -- средняя неопределённость "
          "одной оценки,\n   сравнивать со столбцом «std прогонов»)")
 
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
 
 
def print_start_dependence(name, by_factor, truth):
    print(f"\n  зависимость от старта Q, {name} (медиана оценка/заложено):")
    print(f"    {'старт':<8}" + "".join(f"{p:>10}" for p in PARAM_ORDER) + f"{'Δ(-2lnL)':>10}")
    for f, runs in by_factor.items():
        if not runs:
            print(f"    x{f:<7g}  нет успешных прогонов")
            continue
        cells = "".join(
            f"{np.median([r['est'][p] for r in runs]) / truth[p]:10.2f}" for p in PARAM_ORDER)
        dJ = [r["dJ"] for r in runs if r.get("dJ") is not None]
        print(f"    x{f:<7g}{cells}{(np.median(dJ) if dJ else float('nan')):+10.2f}")
 
 
def print_scenario(sc, truth, results, tcfgs, orc, dt):
    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТ: {sc['name']}   [{dt:.0f} с]")
    print("=" * 78)
    print(f"N_sim = {N_SIM}, прогонов = {N_RUNS}, POI = {POI}, WARMUP = {WARMUP}")
    print("ЗАЛОЖЕНО: " + ", ".join(f"{PARAM_LABEL[p].split(',')[0]} = {truth[p]:.3e}"
                                   for p in PARAM_ORDER))
 
    orc_ok = [o for o in orc if o]
    if orc_ok:
        rms_o = np.mean([o["rms"] for o in orc_ok], axis=0)
        print(f"ЭТАЛОН (EKF с заложенными параметрами, {len(orc_ok)}/{len(orc)} прогонов): "
              f"J = {np.mean([o['J'] for o in orc_ok]):.5f}, RMS состояния (A, B, ph[рад]) = "
              f"{rms_o[0]:.2e}, {rms_o[1]:.2e}, {rms_o[2]:.2e}")
 
    for m in METHODS:
        print_method(m, 1.0, results[(m, 1.0)], truth, tcfgs[m], N_RUNS)
    if len(START_FACTORS) > 1:
        for m in METHODS:
            print_start_dependence(m, {f: results[(m, f)] for f in START_FACTORS}, truth)
 
 
# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
 
def main():
    if 1.0 not in START_FACTORS:
        raise ValueError("START_FACTORS должен содержать 1.0")
    if N_SIM - POI <= WARMUP + 10:
        raise ValueError("N_SIM слишком мал для POI + WARMUP")
    if ekf.N_RP != sim.N_RP:
        raise ValueError(f"N_RP различается: {ekf.N_RP} и {sim.N_RP}")
 
    kz_fit = KZ if KZ_FIT is None else KZ_FIT
    tau_fit = DELAY if TAU_FIT is None else TAU_FIT
 
    weight_vec, win_len = ekf.build_weight_vec(sim.T, sim.ty, sim.t_step)
    if win_len != sim.end:
        raise ValueError(f"длина окна различается: {win_len} и {sim.end}")
    if tau_fit < 0 or tau_fit + win_len > ekf.N_RP:
        raise ValueError(f"tau_fit + win_len = {tau_fit + win_len} > N_RP = {ekf.N_RP}")
 
    ekf._jit_warmup()
 
    for sc in SCENARIOS:
        print("\n" + "#" * 78)
        print(f"# сценарий: {sc['name']}")
        print("#" * 78)
 
        tcfgs = {m: ekf.TuneCfg(**{**TUNE_COMMON, **mcfg}) for m, mcfg in METHODS.items()}
        results = {(m, f): [] for m in METHODS for f in START_FACTORS}
        orc = []
        truth = None
        rng = np.random.default_rng(SEED_MC)
        t_sc = time.perf_counter()
 
        for r in range(N_RUNS):
            t_run = time.perf_counter()
            seed_vib = int(rng.integers(0, 2 ** 31 - 1))
            seed_noise = int(rng.integers(0, 2 ** 31 - 1))
            data = generate(sc, seed_vib, seed_noise, kz_fit, tau_fit, weight_vec, win_len)
            truth = data["truth"]
 
            o = run_oracle(data, kz_fit)
            orc.append(o)
 
            for m in METHODS:
                for f in START_FACTORS:
                    out = run_method(data, tcfgs[m], f, kz_fit)
                    if out is None:
                        continue
                    out["dJ"] = (out["J"] - o["J"]) * o["n"] if o else None
                    results[(m, f)].append(out)
                    if VERBOSE_RUNS and f == 1.0:
                        e = out["est"]
                        print(f"    {m}: sigma_A={e['sigma_A']:.3e} sigma_ph={e['sigma_ph']:.3e} "
                              f"dA={e['dA']:.2e} dB={e['dB']:.2e} dph={e['dph']:.2e}")
            print(f"  прогон {r + 1}/{N_RUNS} готов, {time.perf_counter() - t_run:.0f} с")
 
        print_scenario(sc, truth, results, tcfgs, orc, time.perf_counter() - t_sc)
 
 
if __name__ == "__main__":
    main()