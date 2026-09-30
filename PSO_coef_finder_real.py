"""
Подбор параметров компенсации вибрации (tau, Kz) для атомного гравиметра.

Два критерия для PSO:
  * "kalman"   -- std инноваций EKF;
  * "windowed" -- RMS невязки независимых cos-фитов в окнах.

Параметры интерферометра T и ty передаются в fit_vibration_compensation.
sigma_A и sigma_ph (шумы измерения в матрице R) задаются ВРУЧНУЮ.
"""

import time
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


# =====================================================================
# Константы, не зависящие от режима работы
# =====================================================================

LM = 780e-9
KEFF = 4 * np.pi / LM
N_RP = 16384    # отсчётов акселерометра на сброс (фиксировано аппаратурой)

FIT_LB = [-1.1, 0.0, 0.0]          # границы [A, B, ph] для curve_fit
FIT_UB = [1.1, 1.1, 2 * np.pi]

BAD_FITNESS = 1e6


# =====================================================================
# Весовая функция интерферометра
# =====================================================================

def fa(t, T, ty):
    """Функция чувствительности интерферометра к ускорению (векторизована)."""
    t = np.asarray(t, dtype=float)
    rising = (t > 0) & (t <= T + 2 * ty)
    falling = (t > T + 2 * ty) & (t <= 2 * T + 4 * ty)
    return np.where(rising, t, np.where(falling, 2 * (T + 2 * ty) - t, 0.0))


def build_weight_vec(T, ty, t_step):
    """
    Возвращает (fa_t, weight_vec, win_len).
    Вибрационная фаза за сброс: F = KEFF * (az_window @ weight_vec).
    """
    duration = 2 * T + 4 * ty
    t_grid = np.linspace(0, duration, round(duration / t_step))
    fa_t = fa(t_grid, T, ty)

    trapz = np.full(len(t_grid), t_step)   # веса трапеций
    trapz[0] *= 0.5
    trapz[-1] *= 0.5
    return fa_t, fa_t * trapz, len(t_grid)


def vibration_phase(az_m, tau, weight_vec, win_len):
    """Вибрационная фаза F(tau) для каждого сброса, рад. Форма (N_sim,)."""
    return KEFF * (az_m[:, tau:tau + win_len] @ weight_vec)


def compensate_alp(alp, Fz, Kz, T):
    """alp_comp = alp - Kz * Fz / (2*pi*T^2)."""
    return alp - Kz * Fz / (2 * np.pi * T ** 2)


def sigma_ph_from_accel(Kz, sigma_a_acc, fa_t, t_step):
    """Фазовый шум (на сброс) только от шума акселерометра -- диагностика."""
    return KEFF * t_step * sigma_a_acc * np.sqrt(np.sum(fa_t ** 2)) * abs(Kz)


# =====================================================================
# Модель фринджа и начальный cos-fit
# =====================================================================

def model(alp, A, B, ph, T):
    return A - B * np.cos(2 * np.pi * T ** 2 * alp - ph)


def initial_fit(alp, P_exp, poi, T):
    """
    cos-fit по первым poi точкам.
    Возвращает (x0 = [A0, B0, ph0], pcov).
    pcov используется как начальная ковариация состояния EKF.
    """
    alp0, P0 = alp[:poi], P_exp[:poi]
    p0 = [(P0.max() + P0.min()) / 2, (P0.max() - P0.min()) / 2, 0.0]

    def f(a, A, B, ph):
        return model(a, A, B, ph, T)

    popt, pcov = curve_fit(f, alp0, P0, p0=p0, bounds=(FIT_LB, FIT_UB))
    return popt, pcov


# =====================================================================
# EKF: состояние [A, B, ph], одно скалярное измерение
# =====================================================================

@dataclass
class EkfParams:
    T: float             # длительность плеча интерферометра, с
    Q: np.ndarray        # ковариация шума процесса (3x3)
    sigma_A: float       # шум детектора (std), ручной
    sigma_ph: float      # шум фазы (std), ручной
    poi: int             # точек для начального cos-fit
    warmup: int          # сколько первых инноваций не учитывать в fitness


def kalman_fit_ekf(alp, P_exp, x0, P0, prm: EkfParams):
    """
    EKF по ряду (alp, P_exp).
    R_i = sigma_A^2 + B^2 * sin^2(Phi_i) * sigma_ph^2.

    Возвращает dict: P_m (предсказание), A, B, ph, P_cov, e (инновации),
    en (нормированные инновации e/sqrt(S)).
    """
    N = len(alp)
    two_pi_T2 = 2 * np.pi * prm.T ** 2
    out = {
        "P_m": np.zeros(N), "A": np.zeros(N), "B": np.zeros(N),
        "ph": np.zeros(N), "e": np.zeros(N), "en": np.zeros(N),
        "P_cov": np.zeros((N, 3, 3)),
    }

    x = np.asarray(x0, dtype=float).copy()
    P = np.asarray(P0, dtype=float).copy()
    I3 = np.eye(3)

    out["A"][0], out["B"][0], out["ph"][0] = x
    out["P_cov"][0] = P
    out["P_m"][0] = x[0] - x[1] * np.cos(two_pi_T2 * alp[0] - x[2])

    for i in range(1, N):
        P = P + prm.Q                                    # предсказание

        Phi = two_pi_T2 * alp[i] - x[2]
        cosPhi, sinPhi = np.cos(Phi), np.sin(Phi)
        z_pred = x[0] - x[1] * cosPhi
        e = P_exp[i] - z_pred

        H = np.array([1.0, -cosPhi, -x[1] * sinPhi])
        R = prm.sigma_A ** 2 + (x[1] * sinPhi * prm.sigma_ph) ** 2

        PHt = P @ H
        S = H @ PHt + R
        K = PHt / S

        x = x + K * e                                    # обновление
        I_KH = I3 - np.outer(K, H)
        P = I_KH @ P @ I_KH.T + np.outer(K, K) * R       # форма Джозефа

        out["P_m"][i], out["e"][i], out["en"][i] = z_pred, e, e / np.sqrt(S)
        out["A"][i], out["B"][i], out["ph"][i] = x
        out["P_cov"][i] = P

    return out


# =====================================================================
# Оконный cos-fit
# =====================================================================

def make_windows(N, win):
    """Границы непересекающихся окон, покрывающих весь набор."""
    n_win = max(1, N // win)
    return np.linspace(0, N, n_win + 1).astype(int)


def windowed_cosfit(phase, y, edges):
    """
    Независимый линейный fit A + c1*cos(phase) + c2*sin(phase) в каждом окне.
    Возвращает (общий RMS невязки, массив фаз по окнам).
    """
    c, s = np.cos(phase), np.sin(phase)
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


# =====================================================================
# Fitness-функции (принимают уже посчитанную Fz)
# =====================================================================

def ekf_fitness(Fz, Kz, alp, P_exp, prm: EkfParams):
    """std инноваций EKF после warmup."""
    alp_comp = compensate_alp(alp, Fz, Kz, prm.T)
    try:
        x0, pcov = initial_fit(alp_comp, P_exp, prm.poi, prm.T)
        res = kalman_fit_ekf(alp_comp, P_exp, x0, pcov, prm)
        return float(np.std(res["e"][prm.warmup:]))
    except Exception:
        return BAD_FITNESS


def windowed_fitness(Fz, Kz, alp, P_exp, win_edges, T):
    """RMS оконного cos-fit."""
    alp_comp = compensate_alp(alp, Fz, Kz, T)
    try:
        rms, _ = windowed_cosfit(2 * np.pi * T ** 2 * alp_comp, P_exp, win_edges)
        return rms
    except Exception:
        return BAD_FITNESS


# =====================================================================
# PSO (2D: tau, Kz), параллельно по частицам, с кэшем Fz(tau)
# =====================================================================

_W = {}   # контекст воркера (заполняется в _worker_init)


def _worker_init(ctx):
    global _W
    _W = dict(ctx, Fz_cache={})


def _evaluate_particle(x):
    tau, Kz = x
    tau = int(np.clip(round(tau), *_W["tau_bounds"]))
    Kz = float(np.clip(Kz, *_W["Kz_bounds"]))
    win_len = _W["win_len"]
    if tau + win_len > _W["az_m"].shape[1]:
        return BAD_FITNESS

    cache = _W["Fz_cache"]
    if tau not in cache:
        cache[tau] = vibration_phase(_W["az_m"], tau, _W["weight_vec"], win_len)
    Fz = cache[tau]

    if _W["kind"] == "kalman":
        return ekf_fitness(Fz, Kz, _W["alp"], _W["P_exp"], _W["prm"])
    return windowed_fitness(Fz, Kz, _W["alp"], _W["P_exp"],
                            _W["win_edges"], _W["prm"].T)


def pso_parallel(kind, ctx, n_particles, n_iter, n_jobs=None, verbose=True):
    """
    kind = "kalman" | "windowed".
    ctx  -- словарь с alp, P_exp, az_m, prm, win_edges, tau_bounds,
            Kz_bounds, weight_vec, win_len.
    Возвращает (tau_opt, Kz_opt, best_fitness, history, n_calls).
    """
    c1, c2, w = 2.0, 2.0, 0.9
    n_jobs = n_jobs or mp.cpu_count()
    bounds = [ctx["tau_bounds"], ctx["Kz_bounds"]]
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)

    pos = np.random.uniform(lb, ub, size=(n_particles, dim))
    vel = np.random.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=(n_particles, dim))
    pbest, pbest_fit = pos.copy(), np.full(n_particles, np.inf)
    gbest, gbest_fit = pos[0].copy(), np.inf
    history, n_calls = [], 0

    worker_ctx = dict(ctx, kind=kind)
    with mp.get_context().Pool(processes=n_jobs, initializer=_worker_init,
                               initargs=(worker_ctx,)) as pool:
        for it in range(n_iter):
            fits = np.array(pool.map(_evaluate_particle, pos, chunksize=1))
            n_calls += n_particles

            better = fits < pbest_fit
            pbest[better], pbest_fit[better] = pos[better], fits[better]
            if fits.min() < gbest_fit:
                gbest_fit = float(fits.min())
                gbest = pos[int(fits.argmin())].copy()
            history.append(gbest_fit)

            r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, lb, ub)

            if verbose:
                print(f"  iter {it + 1:02d}/{n_iter} | best = {gbest_fit:.4e} "
                      f"| tau={gbest[0]:.1f}, Kz={gbest[1]:.4f}")

    return int(np.round(gbest[0])), float(gbest[1]), gbest_fit, history, n_calls


# =====================================================================
# Верхнеуровневая функция
# =====================================================================

def fit_vibration_compensation(alp, P_exp, az_m, *,
                               T, ty, T_RP,
                               tau_bounds, Kz_bounds,
                               dA_model, dB_model, dph_model,
                               sigma_A, sigma_ph,
                               poi, warmup, win_size,
                               sigma_a_acc=None,
                               n_particles=30, n_iter=30, n_jobs=None):
    """
    Подбор (tau, Kz) двумя критериями (std инноваций EKF и RMS оконного fit).

    T, ty            : длительность плеча и импульса интерферометра, с.
    T_RP             : длительность записи акселерометра на сброс, с
                       (t_step = T_RP / N_RP).
    dA/dB/dph_model  : шаги блуждания A, B, ph между сбросами
                       -> Q = diag(dA^2, dB^2, dph^2).
    sigma_A, sigma_ph: шум детектора и фазы (std) для R матрицы EKF, вручную.
    poi              : точек для начального cos-fit.
    win_size         : размер окна для оконного fit.
    sigma_a_acc      : шум акселерометра, м/с^2 (необязательно) -- только для
                       печати sigma_ph_accel в найденной точке.

    Возвращает {"kalman": {...}, "windowed": {...}}, где в каждом
    tau, Kz, std_e (std инноваций EKF в найденной точке, единая метрика
    для обоих методов) и, если задан sigma_a_acc, sigma_ph_accel.
    """
    t_step = T_RP / N_RP
    fa_t, weight_vec, win_len = build_weight_vec(T, ty, t_step)

    prm = EkfParams(
        T=T,
        Q=np.diag([dA_model ** 2, dB_model ** 2, dph_model ** 2]),
        sigma_A=sigma_A, sigma_ph=sigma_ph, poi=poi, warmup=warmup)

    ctx = dict(alp=alp, P_exp=P_exp, az_m=az_m, prm=prm,
               win_edges=make_windows(len(alp), win_size),
               tau_bounds=tau_bounds, Kz_bounds=Kz_bounds,
               weight_vec=weight_vec, win_len=win_len)

    def std_e_at(tau, Kz):
        Fz = vibration_phase(az_m, tau, weight_vec, win_len)
        return ekf_fitness(Fz, Kz, alp, P_exp, prm)

    results = {}
    for kind, title in (("kalman", "fitness = std(EKF innovation)"),
                        ("windowed", "fitness = RMS оконного cos-fit")):
        print(f"=== PSO, {title}, sigma_A={sigma_A:.2e}, sigma_ph={sigma_ph:.2e} ===")
        t0 = time.perf_counter()
        tau, Kz, best, _, n_calls = pso_parallel(kind, ctx, n_particles, n_iter, n_jobs)
        dt = time.perf_counter() - t0

        std_e = best if kind == "kalman" else std_e_at(tau, Kz)
        extra = "" if kind == "kalman" else f", RMS(win)={best:.4e}"
        print(f"  -> tau={tau}, Kz={Kz:.5f}{extra}, std(e)={std_e:.4e} "
              f"[{n_calls} вычислений, {dt:.1f} с]")

        results[kind] = {"tau": tau, "Kz": Kz, "std_e": std_e}
        if sigma_a_acc is not None:
            s_acc = sigma_ph_from_accel(Kz, sigma_a_acc, fa_t, t_step)
            results[kind]["sigma_ph_accel"] = s_acc
            print(f"  sigma_ph: accel_only={s_acc:.4e} рад, manual={sigma_ph:.4e} рад")
        print()
    return results


# =====================================================================
# main
# =====================================================================

def main():
    # Реальные данные:
    #   alp   : (N_sim,)       -- некомпенсированный чирп на сброс
    #   P_exp : (N_sim,)       -- нормированный сигнал интерферометра
    #   az_m  : (N_sim, N_acc) -- отсчёты акселерометра,
    #                             N_acc >= tau_bounds[1] + win_len
    data = np.load("gravimeter_data.npz")
    alp, P_exp, az_m = data["alp"], data["P_exp"], data["az_m"]

    results = fit_vibration_compensation(
        alp, P_exp, az_m,
        T=10e-3,                     # длительность плеча, с
        ty=20e-6,                    # длительность импульса, с
        T_RP=33e-3,                  # запись акселерометра на сброс, с
        tau_bounds=(0, 5000),        # отсчёты акселерометра
        Kz_bounds=(0.0, 1.5),
        # шаги блуждания модели фринджа (вручную)
        dA_model=5e-3, dB_model=5e-3, dph_model=1e-3,
        # шумы измерения в R (вручную)
        sigma_A=7e-3, sigma_ph=5e-3,
        poi=200, warmup=120, win_size=20,
        sigma_a_acc=3e-5,            # None -> не печатать sigma_ph_accel
        n_particles=30, n_iter=30)

    print("=== Итог ===")
    for method, r in results.items():
        line = (f"{method:9s}: tau={r['tau']:5d}  Kz={r['Kz']:.5f}  "
                f"std(e)={r['std_e']:.4e}")
        if "sigma_ph_accel" in r:
            line += f"  | sigma_ph_accel={r['sigma_ph_accel']:.3e}"
        print(line)


if __name__ == "__main__":
    main()