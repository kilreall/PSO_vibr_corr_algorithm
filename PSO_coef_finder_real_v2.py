"""
Подбор параметров компенсации вибрации (tau, Kz) для атомного гравиметра.

Внешний цикл: PSO по (tau, Kz). Таблица F_z(tau) считается один раз
(FFT-корреляция) и лежит в shared memory; между узлами tau -- линейная
интерполяция (tau вещественное).

Fitness для каждой пары (tau, Kz):
  1. компенсация чирпа; линейный cos-fit первых poi точек -> x0, P0 для EKF;
  2. стартовые sigma_A, sigma_ph (только для первого прогона EKF):
       "bins" | "split" | "manual";
  3. настройка EKF:
       tune_mode="alternate" (Cheiney et al. 2018, Appendix A):
         a) прогон EKF; b) оценка R ("fisher" | "grid");
         c) подбор диагонали Q при фиксированном R;
         d) повтор до сходимости или n_outer раз;
       tune_mode="joint": совместно (sigma_A, sigma_ph, q);
     оптимизатор: q_optimizer="lbfgs" (аналитический градиент через
     уравнения чувствительности EKF) | "nm" (Nelder-Mead);
  4. fitness = J = mean(ln S_i + e_i^2 / S_i) по точкам после warmup.

Для итоговой точки: диагностика нормированных инноваций и проверка
аналитического градиента конечными разностями.
"""

import time
import multiprocessing as mp
from multiprocessing import shared_memory
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize, nnls
from scipy.signal import fftconvolve
from scipy.stats import chi2 as chi2_dist


# =====================================================================
# Константы
# =====================================================================

LM = 780e-9
KEFF = 4 * np.pi / LM
N_RP = 16384          # отсчётов акселерометра на сброс (фиксировано аппаратурой)

BAD_FITNESS = 1e6
BIG_OBJ = 1e10        # значение цели при сбое EKF внутри оптимизатора
LN10 = np.log(10.0)
Q_NAMES = np.array(["A", "B", "ph"])
PAR_NAMES = ["sigma_A", "sigma_ph", "q_A", "q_B", "q_ph"]


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
    Возвращает (weight_vec, win_len).
    Вибрационная фаза за сброс: F = KEFF * (az_window @ weight_vec).
    """
    duration = 2 * T + 4 * ty
    t_grid = np.linspace(0, duration, round(duration / t_step))
    fa_t = fa(t_grid, T, ty)

    trapz = np.full(len(t_grid), t_step)   # веса трапеций
    trapz[0] *= 0.5
    trapz[-1] *= 0.5
    return fa_t * trapz, len(t_grid)


def vibration_phase(az_m, tau, weight_vec, win_len):
    """Вибрационная фаза F(tau) прямым расчётом (для проверки таблицы)."""
    return KEFF * (az_m[:, tau:tau + win_len] @ weight_vec)


def compensate_alp(alp, Fz, Kz, T):
    """alp_comp = alp - Kz * Fz / (2*pi*T^2)."""
    return alp - Kz * Fz / (2 * np.pi * T ** 2)


# =====================================================================
# Таблица F_z(tau) в shared memory
# =====================================================================

@dataclass
class FzTableMeta:
    shm_name: str
    shape: Tuple[int, int]     # (n_tau, N_sim)
    dtype: str
    tau_lo: int
    tau_step: int


def build_fz_table(az_m, weight_vec, win_len, tau_bounds, tau_step, dtype, chunk_rows):
    """
    Таблица F_z[k, n] = KEFF * sum_j az_m[n, tau_k + j] * weight_vec[j],
    tau_k = tau_lo + k * tau_step, в shared memory.
    Считается FFT-корреляцией кусками по chunk_rows строк.
    Возвращает (shm, table, meta).
    """
    tau_lo, tau_hi = int(tau_bounds[0]), int(tau_bounds[1])
    if tau_step < 1:
        raise ValueError("tau_step должен быть >= 1")
    taus = np.arange(tau_lo, tau_hi + 1, tau_step)
    tau_last = int(taus[-1])
    n_rows, n_acc = az_m.shape
    if tau_last + win_len > n_acc:
        raise ValueError(f"tau_bounds[1] + win_len = {tau_last + win_len} > N_acc = {n_acc}")

    dtype = np.dtype(dtype)
    shape = (len(taus), n_rows)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    table = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    wrev = weight_vec[::-1][None, :]
    try:
        for r0 in range(0, n_rows, chunk_rows):
            r1 = min(r0 + chunk_rows, n_rows)
            seg = np.asarray(az_m[r0:r1, tau_lo:tau_last + win_len], dtype=float)
            corr = fftconvolve(seg, wrev, mode="valid", axes=1)   # tau = tau_lo + t
            table[:, r0:r1] = (KEFF * corr[:, ::tau_step]).T
    except Exception:
        table = None
        shm.close()
        shm.unlink()
        raise

    meta = FzTableMeta(shm.name, shape, dtype.str, tau_lo, int(tau_step))
    return shm, table, meta


def _attach_shm(name):
    """Подключение к существующему блоку (без регистрации в трекере на 3.13+)."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:          # Python < 3.13
        return shared_memory.SharedMemory(name=name)


def fz_lookup(table, tau, tau_lo, tau_step, n_cols):
    """F_z для вещественного tau: линейная интерполяция между узлами таблицы."""
    n_tau = table.shape[0]
    if n_tau == 1:
        return table[0, :n_cols].astype(np.float64)
    u = (tau - tau_lo) / tau_step
    i0 = int(min(max(np.floor(u), 0), n_tau - 2))
    f = min(max(u - i0, 0.0), 1.0)
    r0 = table[i0, :n_cols].astype(np.float64)
    r1 = table[i0 + 1, :n_cols].astype(np.float64)
    return r0 + f * (r1 - r0)


# =====================================================================
# Модель фринджа и начальный фит
# =====================================================================

def model(alp, A, B, ph, T):
    return A - B * np.cos(2 * np.pi * T ** 2 * alp - ph)


def initial_fit(alp, y, T):
    """
    Линейный МНК: y = A + c1*cos(Phi0) + c2*sin(Phi0), Phi0 = 2*pi*T^2*alp.
    Эквивалентен model() при B = sqrt(c1^2 + c2^2), ph = atan2(-c2, -c1).
    Возвращает (x0 = [A, B, ph], P0 (3x3), невязки фита).
    """
    Phi0 = 2 * np.pi * T ** 2 * alp
    X = np.column_stack((np.ones_like(Phi0), np.cos(Phi0), np.sin(Phi0)))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef

    s2 = float(res @ res) / (len(y) - 3)
    cov_lin = s2 * np.linalg.inv(X.T @ X)

    A, c1, c2 = coef
    B2 = c1 * c1 + c2 * c2
    B = np.sqrt(B2)
    ph = np.arctan2(-c2, -c1)

    Jac = np.array([[1.0, 0.0, 0.0],
                    [0.0, c1 / B, c2 / B],
                    [0.0, -c2 / B2, c1 / B2]])
    P0 = Jac @ cov_lin @ Jac.T
    return np.array([A, B, ph]), P0, res


# =====================================================================
# Стартовые sigma_A, sigma_ph (только для первого прогона EKF)
# =====================================================================

def _phase_sensitivity(alp, x0, T):
    """B^2 sin^2(Phi) для каждой точки и сама фаза Phi."""
    A, B, ph = x0
    Phi = 2 * np.pi * T ** 2 * alp - ph
    return (B * np.sin(Phi)) ** 2, Phi


def sigmas_split(res, x0, alp, T):
    """Дисперсия невязки делится пополам между детектированием и фазой."""
    n = len(res)
    v = n / (n - 3) * float(np.mean(res ** 2))
    xs, _ = _phase_sensitivity(alp, x0, T)
    m = float(np.mean(xs))
    return np.sqrt(v / 2), np.sqrt(v / (2 * max(m, 1e-12)))


def sigmas_bins(res, x0, alp, T, n_bins, min_per_bin, floor_frac):
    """
    Метод моментов по бинам Phi mod pi:
        v_b = sigma_A^2 + sigma_ph^2 * <B^2 sin^2(Phi)>_b,
    взвешенный NNLS с перевзвешиванием. None, если шумы неразделимы.
    """
    n = len(res)
    corr = n / (n - 3)
    v_tot = corr * float(np.mean(res ** 2))
    xs, Phi = _phase_sensitivity(alp, x0, T)
    m_x = float(np.mean(xs))

    bins = np.minimum((np.mod(Phi, np.pi) / np.pi * n_bins).astype(int), n_bins - 1)
    used = [b for b in range(n_bins) if np.sum(bins == b) >= min_per_bin]
    if len(used) < 2:
        return None

    n_b = np.array([np.sum(bins == b) for b in used], dtype=float)
    x_b = np.array([xs[bins == b].mean() for b in used])
    v_b = np.array([corr * np.mean(res[bins == b] ** 2) for b in used])
    if np.ptp(x_b) < 0.1 * max(m_x, 1e-12):
        return None

    X = np.column_stack((np.ones(len(used)), x_b))
    w = np.sqrt(n_b) / max(v_tot, 1e-30)
    for _ in range(5):
        coef, _ = nnls(X * w[:, None], v_b * w)
        w = np.sqrt(n_b) / np.maximum(X @ coef, 1e-3 * v_tot)

    sA2 = max(coef[0], floor_frac * v_tot)
    sph2 = max(coef[1], floor_frac * v_tot / max(m_x, 1e-12))
    return np.sqrt(sA2), np.sqrt(sph2)


def start_sigmas(res, x0, alp, T, cfg):
    """Стартовые sigma_A, sigma_ph и источник."""
    if cfg.sigma_init_mode == "manual":
        sA, sph = cfg.sigma_init
        src = "manual"
    elif cfg.sigma_init_mode == "bins":
        est = sigmas_bins(res, x0, alp, T, cfg.n_bins, cfg.min_per_bin,
                          cfg.sigma_floor_frac)
        if est is None:
            sA, sph = sigmas_split(res, x0, alp, T)
            src = "split (bins: шумы неразделимы)"
        else:
            sA, sph = est
            src = "bins"
    else:
        sA, sph = sigmas_split(res, x0, alp, T)
        src = "split"

    sA = float(np.clip(sA, cfg.sA_lo, cfg.sA_hi))
    sph = float(np.clip(sph, cfg.sph_lo, cfg.sph_hi))
    return sA, sph, src


# =====================================================================
# EKF (numba): состояние [A, B, ph], F = I, диагональная Q
# =====================================================================

@njit(cache=True)
def _ekf_core(alp, y, x0, P0, qd, sA2, sph2, two_pi_T2, start,
              e_out, h_out, b_out, x_out):
    """
    Прямой прогон EKF с сохранением траектории.
      e_out -- инновации; h_out -- H P^- H^T; b_out -- B^2 sin^2(Phi);
      x_out -- апостериорные оценки состояния.
    Возвращает (sum_{i>=start} [ln S_i + e_i^2/S_i], число таких точек).
    """
    n = alp.shape[0]
    x = x0.copy()
    P = P0.copy()
    H = np.empty(3)
    PHt = np.empty(3)
    K = np.empty(3)
    nll = 0.0
    cnt = 0

    for i in range(n):
        for j in range(3):
            P[j, j] += qd[j]

        Phi = two_pi_T2 * alp[i] - x[2]
        c = np.cos(Phi)
        s = np.sin(Phi)
        e = y[i] - (x[0] - x[1] * c)
        H[0] = 1.0
        H[1] = -c
        H[2] = -x[1] * s

        hh = 0.0
        for j in range(3):
            PHt[j] = P[j, 0] * H[0] + P[j, 1] * H[1] + P[j, 2] * H[2]
            hh += H[j] * PHt[j]
        bb = x[1] * x[1] * s * s
        S = hh + sA2 + bb * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return np.inf, 0

        for j in range(3):
            K[j] = PHt[j] / S
            x[j] += K[j] * e
        for j in range(3):
            for k in range(3):
                P[j, k] -= K[j] * K[k] * S
        for j in range(3):
            for k in range(j + 1, 3):
                m = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = m
                P[k, j] = m

        e_out[i] = e
        h_out[i] = hh
        b_out[i] = bb
        x_out[i, 0] = x[0]
        x_out[i, 1] = x[1]
        x_out[i, 2] = x[2]

        if i >= start:
            nll += np.log(S) + e * e / S
            cnt += 1

    return nll, cnt


@njit(cache=True)
def _ekf_nll_grad(alp, y, x0, P0, qd, sA2, sph2, two_pi_T2, start,
                  dq, dsA2, dsph2, grad):
    """
    EKF + точный градиент NLL по параметрам theta_m (уравнения чувствительности).

    dq    : (M, 3) -- d q_j / d theta_m (диагональ Q);
    dsA2  : (M,)   -- d sigma_A^2 / d theta_m;
    dsph2 : (M,)   -- d sigma_ph^2 / d theta_m;
    grad  : (M,)   -- выход, d/d theta_m sum_{i>=start} [ln S_i + e_i^2/S_i].
    x0, P0 от theta не зависят: dx = 0, dP = 0 на старте.
    Возвращает (nll, cnt).
    """
    n = alp.shape[0]
    M = dq.shape[0]
    x = x0.copy()
    P = P0.copy()
    dx = np.zeros((M, 3))
    dP = np.zeros((M, 3, 3))
    H = np.empty(3)
    PHt = np.empty(3)
    K = np.empty(3)
    dPHt = np.empty((M, 3))
    dS = np.empty(M)
    de = np.empty(M)
    for m in range(M):
        grad[m] = 0.0
    nll = 0.0
    cnt = 0

    for i in range(n):
        # --- предсказание: P^- = P + diag(q) ---
        for j in range(3):
            P[j, j] += qd[j]
            for m in range(M):
                dP[m, j, j] += dq[m, j]

        # --- измерение ---
        Phi = two_pi_T2 * alp[i] - x[2]
        c = np.cos(Phi)
        s = np.sin(Phi)
        Bx = x[1]
        e = y[i] - (x[0] - Bx * c)
        H[0] = 1.0
        H[1] = -c
        H[2] = -Bx * s

        hh = 0.0
        for j in range(3):
            PHt[j] = P[j, 0] * H[0] + P[j, 1] * H[1] + P[j, 2] * H[2]
            hh += H[j] * PHt[j]
        bb = Bx * Bx * s * s
        S = hh + sA2 + bb * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return np.inf, 0
        iS = 1.0 / S
        for j in range(3):
            K[j] = PHt[j] * iS

        # --- производные величин шага (по x^-, P^-) ---
        for m in range(M):
            dB = dx[m, 1]
            dph = dx[m, 2]
            dH1 = -s * dph                      # d(-cos Phi),   dPhi = -dph
            dH2 = -s * dB + Bx * c * dph        # d(-B sin Phi)
            de[m] = -(dx[m, 0] + H[1] * dB + H[2] * dph)   # de = -H dx
            dhh = dH1 * PHt[1] + dH2 * PHt[2]
            for j in range(3):
                v = (dP[m, j, 0] * H[0] + dP[m, j, 1] * H[1] + dP[m, j, 2] * H[2]
                     + P[j, 1] * dH1 + P[j, 2] * dH2)
                dPHt[m, j] = v
                dhh += H[j] * v
            dbb = 2.0 * Bx * s * s * dB - 2.0 * Bx * Bx * s * c * dph
            dS[m] = dhh + dsA2[m] + dbb * sph2 + bb * dsph2[m]

        # --- вклад в NLL и градиент ---
        if i >= start:
            nll += np.log(S) + e * e * iS
            cS = iS - e * e * iS * iS
            for m in range(M):
                grad[m] += dS[m] * cS + 2.0 * e * de[m] * iS
            cnt += 1

        # --- производные обновления (по старым K, PHt, S) ---
        for m in range(M):
            for j in range(3):
                dKj = (dPHt[m, j] - K[j] * dS[m]) * iS
                dx[m, j] += dKj * e + K[j] * de[m]
            for j in range(3):
                for k in range(3):
                    dP[m, j, k] += (K[j] * K[k] * dS[m]
                                    - (dPHt[m, j] * PHt[k] + PHt[j] * dPHt[m, k]) * iS)
            for j in range(3):
                for k in range(j + 1, 3):
                    v = 0.5 * (dP[m, j, k] + dP[m, k, j])
                    dP[m, j, k] = v
                    dP[m, k, j] = v

        # --- обновление состояния ---
        for j in range(3):
            x[j] += K[j] * e
        for j in range(3):
            for k in range(3):
                P[j, k] -= K[j] * K[k] * S
        for j in range(3):
            for k in range(j + 1, 3):
                v = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = v
                P[k, j] = v

    return nll, cnt


def run_ekf(alp, y, x0, P0, qd, sA, sph, T, start):
    """Обёртка над _ekf_core. Возвращает dict с nll, cnt, e, h, b, x."""
    n = len(alp)
    e = np.empty(n)
    h = np.empty(n)
    b = np.empty(n)
    x = np.empty((n, 3))
    nll, cnt = _ekf_core(alp, y, x0, P0, qd, sA * sA, sph * sph,
                         2 * np.pi * T * T, start, e, h, b, x)
    return {"nll": nll, "cnt": cnt, "e": e, "h": h, "b": b, "x": x}


# =====================================================================
# Оценка R: метод Фишера (ML) и байесовская сетка
# =====================================================================

@njit(cache=True)
def _fisher_core(e2, h, b, a, p, a_lo, p_lo, a_hi, p_hi, n_iter, tol):
    """
    Метод Фишера для (a, p) = (sigma_A^2, sigma_ph^2), S_k = h_k + a + b_k p.
    Возвращает (a, p, I00, I01, I11) -- информация в итоговой точке.
    При n_iter = 0 только считает информацию.
    """
    for _ in range(n_iter):
        g0 = 0.0
        g1 = 0.0
        I00 = 0.0
        I01 = 0.0
        I11 = 0.0
        for k in range(e2.shape[0]):
            iS = 1.0 / (h[k] + a + b[k] * p)
            r = (e2[k] * iS - 1.0) * iS
            g0 += r
            g1 += b[k] * r
            w = iS * iS
            I00 += w
            I01 += b[k] * w
            I11 += b[k] * b[k] * w
        det = I00 * I11 - I01 * I01
        if not (det > 0.0):
            break
        da = (I11 * g0 - I01 * g1) / det
        dp = (-I01 * g0 + I00 * g1) / det

        step = 1.0
        while step > 1e-3 and (a + step * da < a_lo or p + step * dp < p_lo):
            step *= 0.5
        a_new = min(max(a + step * da, a_lo), a_hi)
        p_new = min(max(p + step * dp, p_lo), p_hi)
        conv = abs(a_new - a) <= tol * a and abs(p_new - p) <= tol * p
        a = a_new
        p = p_new
        if conv:
            break

    I00 = 0.0
    I01 = 0.0
    I11 = 0.0
    for k in range(e2.shape[0]):
        iS = 1.0 / (h[k] + a + b[k] * p)
        w = 0.5 * iS * iS
        I00 += w
        I01 += b[k] * w
        I11 += b[k] * b[k] * w
    return a, p, I00, I01, I11


def fisher_R(e, h, b, sA, sph, cfg, n_iter):
    """ML-оценка sigma_A, sigma_ph и их SD из информационной матрицы."""
    hh = h if cfg.use_state_cov else np.zeros_like(h)
    a, p, I00, I01, I11 = _fisher_core(
        e * e, hh, b, sA * sA, sph * sph,
        cfg.sA_lo ** 2, cfg.sph_lo ** 2, cfg.sA_hi ** 2, cfg.sph_hi ** 2,
        n_iter, cfg.fisher_tol)
    sA_n, sph_n = np.sqrt(a), np.sqrt(p)
    det = I00 * I11 - I01 * I01
    if det > 0:
        sdA = np.sqrt(I11 / det) / (2 * sA_n)
        sdP = np.sqrt(I00 / det) / (2 * sph_n)
    else:
        sdA = sdP = np.nan
    return sA_n, sph_n, sdA, sdP


@njit(cache=True)
def _grid_loglik(e2, h, b, sa, sp):
    """Лог-правдоподобие инноваций на сетке (sigma_A, sigma_ph)."""
    na = sa.shape[0]
    npp = sp.shape[0]
    out = np.empty((na, npp))
    for i in range(na):
        a2 = sa[i] * sa[i]
        for j in range(npp):
            p2 = sp[j] * sp[j]
            acc = 0.0
            for k in range(e2.shape[0]):
                S = h[k] + a2 + b[k] * p2
                acc += np.log(S) + e2[k] / S
            out[i, j] = -0.5 * acc
    return out


def _posterior_moments(L, sa, sp, cell_w):
    """Апостериорные среднее и SD по сетке (cell_w -- площади ячеек)."""
    w = np.exp(L - L.max()) * cell_w
    w /= w.sum()
    pa, pp = w.sum(axis=1), w.sum(axis=0)
    mA, mP = float(pa @ sa), float(pp @ sp)
    sdA = np.sqrt(max(float(pa @ sa ** 2) - mA ** 2, 0.0))
    sdP = np.sqrt(max(float(pp @ sp ** 2) - mP ** 2, 0.0))
    return mA, mP, sdA, sdP


def grid_R(e, h, b, cfg):
    """Апостериорное среднее sigma_A, sigma_ph при равномерном prior."""
    e2 = e * e
    hh = h if cfg.use_state_cov else np.zeros_like(h)

    sa = np.geomspace(cfg.sA_lo, cfg.sA_hi, cfg.n_grid)
    sp = np.geomspace(cfg.sph_lo, cfg.sph_hi, cfg.n_grid)
    L = _grid_loglik(e2, hh, b, sa, sp)
    mA, mP, sdA, sdP = _posterior_moments(L, sa, sp, np.outer(sa, sp))
    stepA = mA * (sa[1] / sa[0] - 1)
    stepP = mP * (sp[1] / sp[0] - 1)

    for _ in range(cfg.n_zoom):
        hwA = max(cfg.zoom_k * sdA, 2 * stepA)
        hwP = max(cfg.zoom_k * sdP, 2 * stepP)
        sa = np.linspace(max(cfg.sA_lo, mA - hwA), min(cfg.sA_hi, mA + hwA), cfg.n_grid)
        sp = np.linspace(max(cfg.sph_lo, mP - hwP), min(cfg.sph_hi, mP + hwP), cfg.n_grid)
        L = _grid_loglik(e2, hh, b, sa, sp)
        mA, mP, sdA, sdP = _posterior_moments(L, sa, sp, 1.0)
        stepA, stepP = sa[1] - sa[0], sp[1] - sp[0]

    return mA, mP


def estimate_R(e, h, b, sA, sph, cfg):
    """Новые sigma_A, sigma_ph по инновациям выбранным методом."""
    if cfg.r_method == "grid":
        return grid_R(e, h, b, cfg)
    sA_n, sph_n, _, _ = fisher_R(e, h, b, sA, sph, cfg, cfg.fisher_iter)
    return sA_n, sph_n


# =====================================================================
# Целевая функция и оптимизация Q (и R в режиме joint)
# =====================================================================

def _q_log_bounds(q_init, idx, cfg):
    """Границы log10 q: абсолютные (q_log_bounds) или +-q_decades от q_init."""
    if cfg.q_log_bounds is not None:
        lo = np.full(idx.size, float(cfg.q_log_bounds[0]))
        hi = np.full(idx.size, float(cfg.q_log_bounds[1]))
    else:
        z = np.log10(q_init[idx])
        lo, hi = z - cfg.q_decades, z + cfg.q_decades
    return lo, hi


def _unpacker(q, idx, sA, sph, joint):
    """z -> (sigma_A, sigma_ph, q). При joint z = [log10 sA, log10 sph, log10 q[idx]]."""
    off = 2 if joint else 0

    def unpack(z):
        qq = q.copy()
        qq[idx] = 10.0 ** z[off:]
        if joint:
            return 10.0 ** z[0], 10.0 ** z[1], qq
        return sA, sph, qq
    return unpack, off


def make_grad_objective(a, yy, x0, P0, T, start, q, idx, sA, sph, joint):
    """
    Возвращает fun(z) -> (J, dJ/dz), J -- средний NLL инноваций,
    z -- log10 параметров (см. _unpacker).
    """
    unpack, off = _unpacker(q, idx, sA, sph, joint)
    M = off + idx.size
    two_pi_T2 = 2 * np.pi * T * T
    grad = np.empty(M)

    def fun(z):
        sa, sp, qq = unpack(z)
        sa2, sp2 = sa * sa, sp * sp
        dq = np.zeros((M, 3))
        dsA2 = np.zeros(M)
        dsph2 = np.zeros(M)
        if joint:
            dsA2[0] = 2 * LN10 * sa2
            dsph2[1] = 2 * LN10 * sp2
        for m, j in enumerate(idx):
            dq[off + m, j] = LN10 * qq[j]
        nll, cnt = _ekf_nll_grad(a, yy, x0, P0, qq, sa2, sp2, two_pi_T2, start,
                                 dq, dsA2, dsph2, grad)
        if not np.isfinite(nll) or cnt == 0:
            return BIG_OBJ, np.zeros(M)
        return nll / cnt, grad / cnt
    return fun


def _simplex(z0, lo, hi, d):
    """Стартовый симплекс Nelder-Mead: шаг d по каждой оси внутрь области."""
    step = np.where(z0 + d <= hi, d, -d)
    return np.clip(np.vstack([z0, z0 + np.diag(step)]), lo, hi)


def optimize_params(a, yy, x0, P0, T, start, q, q_init, idx, sA, sph, joint, cfg):
    """
    Минимизация среднего NLL по log10 свободных q (и log10 sigma при joint).
    Возвращает (sigma_A, sigma_ph, q, число вычислений цели).
    """
    qlo, qhi = _q_log_bounds(q_init, idx, cfg)
    if joint:
        lo = np.concatenate(([np.log10(cfg.sA_lo), np.log10(cfg.sph_lo)], qlo))
        hi = np.concatenate(([np.log10(cfg.sA_hi), np.log10(cfg.sph_hi)], qhi))
        z0 = np.concatenate(([np.log10(sA), np.log10(sph)], np.log10(q[idx])))
    else:
        lo, hi = qlo, qhi
        z0 = np.log10(q[idx])
    if z0.size == 0:
        return sA, sph, q.copy(), 0
    z0 = np.clip(z0, lo, hi)
    unpack, _ = _unpacker(q, idx, sA, sph, joint)
    bounds = list(zip(lo, hi))

    if cfg.q_optimizer == "lbfgs":
        fun = make_grad_objective(a, yy, x0, P0, T, start, q, idx, sA, sph, joint)
        r = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(ftol=cfg.lbfgs_ftol, gtol=cfg.lbfgs_gtol,
                                  maxfun=cfg.lbfgs_maxfun))
    else:
        def fun(z):
            sa, sp, qq = unpack(z)
            res = run_ekf(a, yy, x0, P0, qq, sa, sp, T, start)
            if not np.isfinite(res["nll"]) or res["cnt"] == 0:
                return BIG_OBJ
            return res["nll"] / res["cnt"]

        r = minimize(fun, z0, method="Nelder-Mead", bounds=bounds,
                     options=dict(initial_simplex=_simplex(z0, lo, hi, 0.3 if joint else 0.5),
                                  xatol=cfg.q_xatol, fatol=cfg.nm_fatol,
                                  maxfev=cfg.joint_maxfev if joint else cfg.q_maxfev))

    sa, sp, qq = unpack(r.x)
    return sa, sp, qq, int(r.nfev)


def gradient_check(a, yy, x0, P0, T, start, q, sA, sph, h=1e-4):
    """
    Аналитический градиент среднего NLL по log10 (sA, sph, qA, qB, qph)
    против центральных разностей. Возвращает (g_analytic, g_fd).
    """
    idx = np.arange(3)
    fun = make_grad_objective(a, yy, x0, P0, T, start, q, idx, sA, sph, True)
    z = np.concatenate(([np.log10(sA), np.log10(sph)], np.log10(q)))
    _, g = fun(z)
    g_fd = np.empty_like(g)
    for m in range(z.size):
        zp, zm = z.copy(), z.copy()
        zp[m] += h
        zm[m] -= h
        g_fd[m] = (fun(zp)[0] - fun(zm)[0]) / (2 * h)
    return g, g_fd


def q_bound_hits(q, q_init, idx, cfg, margin=0.05):
    """Имена компонент q у границы (в пределах margin декад)."""
    if idx.size == 0:
        return []
    lo, hi = _q_log_bounds(q_init, idx, cfg)
    z = np.log10(q[idx])
    return [str(n) for n, zz, l, h in zip(Q_NAMES[idx], z, lo, hi)
            if zz - l < margin or h - zz < margin]


# =====================================================================
# Диагностика нормированных инноваций
# =====================================================================

def innovation_stats(nu, n_lags):
    """
    nu = e_i / sqrt(S_i). Для согласованного фильтра: mean ~ 0, std ~ 1,
    автокорреляция на лагах >= 1 в полосе +-1.96/sqrt(N), Льюнг-Бокс ~ chi2(n_lags).
    """
    n = len(nu)
    m = float(np.mean(nu))
    c = nu - m
    c0 = float(c @ c) / n
    lags = np.arange(1, n_lags + 1)
    rho = np.array([float(c[:-k] @ c[k:]) / n / c0 for k in lags])
    band = 1.96 / np.sqrt(n)
    lb = n * (n + 2) * float(np.sum(rho ** 2 / (n - lags)))
    return {
        "n": n, "mean": m, "mean_se": 1 / np.sqrt(n),
        "std": float(np.std(nu, ddof=1)),
        "rho": rho, "band": band,
        "n_out_band": int(np.sum(np.abs(rho) > band)),
        "ljung_box": lb, "lb_pvalue": float(chi2_dist.sf(lb, n_lags)),
        "kurtosis": float(np.mean(c ** 4) / c0 ** 2 - 3),
    }


# =====================================================================
# Конфигурация
# =====================================================================

@dataclass
class EkfParams:
    T: float               # длительность плеча интерферометра, с
    q_init: np.ndarray     # начальная диагональ Q: [dA^2, dB^2, dph^2]
    poi: int               # точек для начального фита
    warmup: int            # инноваций после poi, не входящих в fitness


@dataclass
class TuneCfg:
    # --- стартовые sigma ---
    sigma_init_mode: str = "bins"      # "bins" | "split" | "manual"
    sigma_init: Optional[Tuple[float, float]] = None
    n_bins: int = 10
    min_per_bin: int = 3
    sigma_floor_frac: float = 1e-2
    # --- режим настройки ---
    tune_mode: str = "alternate"       # "alternate" | "joint"
    r_method: str = "fisher"           # "fisher" | "grid" (для alternate)
    q_optimizer: str = "lbfgs"         # "lbfgs" (аналит. градиент) | "nm"
    # --- внешние итерации R -> Q (alternate) ---
    n_outer: int = 3
    tol_sigma: float = 0.01
    tol_logq: float = 0.02
    # --- границы Q ---
    q_free: Tuple[bool, bool, bool] = (True, True, True)
    q_decades: float = 3.0             # +-декад вокруг q_init
    q_log_bounds: Optional[Tuple[float, float]] = None  # абсолютные границы log10 q
    # --- L-BFGS-B (цель -- средний NLL) ---
    lbfgs_ftol: float = 1e-10
    lbfgs_gtol: float = 1e-6
    lbfgs_maxfun: int = 100
    # --- Nelder-Mead ---
    q_xatol: float = 0.02
    nm_fatol: float = 5e-5             # в единицах среднего NLL
    q_maxfev: int = 80
    joint_maxfev: int = 300
    # --- оценка R ---
    sA_lo: float = 1e-5
    sA_hi: float = 0.5
    sph_lo: float = 1e-4
    sph_hi: float = 3.0
    fisher_iter: int = 30
    fisher_tol: float = 1e-4
    n_grid: int = 21
    n_zoom: int = 3
    zoom_k: float = 6.0
    use_state_cov: bool = True
    # --- диагностика ---
    n_lags: int = 20


# =====================================================================
# Fitness: настройка EKF и NLL инноваций
# =====================================================================

def tune_and_score(Fz, Kz, alp, y, prm: EkfParams, tcfg: TuneCfg,
                   diagnostics=False, grad_check=False):
    """
    Полная обработка одной точки (tau, Kz). Возвращает (J, info).
    """
    alp_comp = compensate_alp(alp, Fz, Kz, prm.T)
    alp0, y0 = alp_comp[:prm.poi], y[:prm.poi]
    x0, P0, res0 = initial_fit(alp0, y0, prm.T)
    a, yy = alp_comp[prm.poi:], y[prm.poi:]
    w = prm.warmup

    sA, sph, src = start_sigmas(res0, x0, alp0, prm.T, tcfg)
    sA_start, sph_start = sA, sph

    q = prm.q_init.astype(float).copy()
    idx = np.flatnonzero(np.asarray(tcfg.q_free, dtype=bool))
    nfev = 0

    if tcfg.tune_mode == "joint":
        sA, sph, q, nfev = optimize_params(a, yy, x0, P0, prm.T, w, q, prm.q_init,
                                           idx, sA, sph, True, tcfg)
        n_it = 1
    else:
        n_it = 0
        for n_it in range(1, tcfg.n_outer + 1):
            r = run_ekf(a, yy, x0, P0, q, sA, sph, prm.T, w)
            if not np.isfinite(r["nll"]):
                return BAD_FITNESS, {}

            sA_n, sph_n = estimate_R(r["e"][w:], r["h"][w:], r["b"][w:], sA, sph, tcfg)
            _, _, q_n, nf = optimize_params(a, yy, x0, P0, prm.T, w, q, prm.q_init,
                                            idx, sA_n, sph_n, False, tcfg)
            nfev += nf

            converged = (abs(sA_n / sA - 1) < tcfg.tol_sigma
                         and abs(sph_n / sph - 1) < tcfg.tol_sigma
                         and np.max(np.abs(np.log10(q_n / q))) < tcfg.tol_logq)
            sA, sph, q = sA_n, sph_n, q_n
            if converged:
                break

    r = run_ekf(a, yy, x0, P0, q, sA, sph, prm.T, w)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return BAD_FITNESS, {}
    J = r["nll"] / r["cnt"]
    if not diagnostics:
        return J, {"sigma_A": sA, "sigma_ph": sph, "q": q}

    e, h, b = r["e"][w:], r["h"][w:], r["b"][w:]
    _, _, sdA, sdP = fisher_R(e, h, b, sA, sph, tcfg, 0)
    S = h + sA ** 2 + b * sph ** 2
    info = {
        "sigma_A": sA, "sigma_ph": sph, "sd_sigma_A": sdA, "sd_sigma_ph": sdP,
        "sigma_A_start": sA_start, "sigma_ph_start": sph_start,
        "sigma_start_src": src,
        "q": q, "n_it": n_it, "nfev": nfev,
        "q_at_bound": q_bound_hits(q, prm.q_init, idx, tcfg),
        "nu_stats": innovation_stats(e / np.sqrt(S), tcfg.n_lags),
        "states": r["x"],
    }
    if grad_check:
        info["grad_check"] = gradient_check(a, yy, x0, P0, prm.T, w, q, sA, sph)
    return J, info


def ekf_fitness(Fz, Kz, alp, y, prm, tcfg):
    try:
        return tune_and_score(Fz, Kz, alp, y, prm, tcfg)[0]
    except Exception:
        return BAD_FITNESS


# =====================================================================
# Оконный cos-fit (критерий для сравнения)
# =====================================================================

def make_windows(N, win):
    """Границы непересекающихся окон, покрывающих весь набор."""
    n_win = max(1, N // win)
    return np.linspace(0, N, n_win + 1).astype(int)


def windowed_cosfit(phase, y, edges):
    """Линейный fit A + c1*cos + c2*sin в каждом окне. (RMS, фазы по окнам)."""
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


def windowed_fitness(Fz, Kz, alp, y, win_edges, T):
    alp_comp = compensate_alp(alp, Fz, Kz, T)
    try:
        rms, _ = windowed_cosfit(2 * np.pi * T ** 2 * alp_comp, y, win_edges)
        return rms
    except Exception:
        return BAD_FITNESS


# =====================================================================
# PSO (2D: tau, Kz), параллельно по частицам; F_z из shared memory
# =====================================================================

_W = {}       # контекст воркера
_SHM = None   # держим ссылку, чтобы блок не закрылся сборщиком мусора


def _worker_init(ctx, meta: FzTableMeta):
    global _W, _SHM
    _SHM = _attach_shm(meta.shm_name)
    table = np.ndarray(meta.shape, dtype=np.dtype(meta.dtype), buffer=_SHM.buf)
    _W = dict(ctx, table=table)


def _evaluate_particle(x):
    tau, Kz = x
    tau = float(np.clip(tau, *_W["tau_bounds"]))
    Kz = float(np.clip(Kz, *_W["Kz_bounds"]))
    Fz = fz_lookup(_W["table"], tau, _W["tau_lo"], _W["tau_step"], _W["n_cols"])

    if _W["kind"] == "kalman":
        return ekf_fitness(Fz, Kz, _W["alp"], _W["P_exp"], _W["prm"], _W["tcfg"])
    return windowed_fitness(Fz, Kz, _W["alp"], _W["P_exp"],
                            _W["win_edges"], _W["prm"].T)


def pso_parallel(kind, ctx, meta, n_particles, n_iter, n_jobs=None, seed=None,
                 verbose=True):
    """
    kind = "kalman" | "windowed".
    Возвращает (tau_opt (float), Kz_opt, best_fitness, history, n_calls).
    """
    rng = np.random.default_rng(seed)
    c1, c2, w = 2.0, 2.0, 0.9
    n_jobs = n_jobs or mp.cpu_count()
    bounds = [ctx["tau_bounds"], ctx["Kz_bounds"]]
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)

    pos = rng.uniform(lb, ub, size=(n_particles, dim))
    vel = rng.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=(n_particles, dim))
    pbest, pbest_fit = pos.copy(), np.full(n_particles, np.inf)
    gbest, gbest_fit = pos[0].copy(), np.inf
    history, n_calls = [], 0

    worker_ctx = dict(ctx, kind=kind)
    with mp.get_context().Pool(processes=n_jobs, initializer=_worker_init,
                               initargs=(worker_ctx, meta)) as pool:
        for it in range(n_iter):
            fits = np.array(pool.map(_evaluate_particle, pos, chunksize=1))
            n_calls += n_particles

            better = fits < pbest_fit
            pbest[better], pbest_fit[better] = pos[better], fits[better]
            if fits.min() < gbest_fit:
                gbest_fit = float(fits.min())
                gbest = pos[int(fits.argmin())].copy()
            history.append(gbest_fit)

            r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, lb, ub)

            if verbose:
                print(f"  iter {it + 1:02d}/{n_iter} | best = {gbest_fit:.6e} "
                      f"| tau={gbest[0]:.2f}, Kz={gbest[1]:.4f}")

    return float(gbest[0]), float(gbest[1]), gbest_fit, history, n_calls


# =====================================================================
# Печать
# =====================================================================

def print_point(J, info, n_show_lags=5):
    """Подробная печать результата настройки в одной точке."""
    q_std = np.sqrt(info["q"])
    st = info["nu_stats"]
    print(f"  J={J:.6e}")
    print(f"  старт ({info['sigma_start_src']}): "
          f"sigma_A={info['sigma_A_start']:.3e}, sigma_ph={info['sigma_ph_start']:.3e} рад")
    print(f"  итог: sigma_A={info['sigma_A']:.3e}+-{info['sd_sigma_A']:.1e}, "
          f"sigma_ph={info['sigma_ph']:.3e}+-{info['sd_sigma_ph']:.1e} рад")
    bound = ", ".join(info["q_at_bound"]) or "нет"
    print(f"  Q (std шага): dA={q_std[0]:.2e}, dB={q_std[1]:.2e}, dph={q_std[2]:.2e}; "
          f"итераций={info['n_it']}, вычислений цели={info['nfev']}; у границы: {bound}")
    rho_s = " ".join(f"{v:+.3f}" for v in st["rho"][:n_show_lags])
    print(f"  nu: mean={st['mean']:+.4f} (se {st['mean_se']:.4f}), "
          f"std={st['std']:.4f}, эксцесс={st['kurtosis']:+.3f}")
    print(f"  nu: rho[1..{n_show_lags}] = {rho_s}  (полоса +-{st['band']:.3f})")
    print(f"  nu: вне полосы {st['n_out_band']}/{len(st['rho'])} лагов, "
          f"Ljung-Box={st['ljung_box']:.1f}, p={st['lb_pvalue']:.3f}")
    if "grad_check" in info:
        g, g_fd = info["grad_check"]
        print("  проверка градиента (dJ/dlog10, аналитический / разностный / отн. ошибка):")
        for name, ga, gf in zip(PAR_NAMES, g, g_fd):
            rel = abs(ga - gf) / max(abs(ga), abs(gf), 1e-12)
            print(f"    {name:9s}: {ga:+.6e}  {gf:+.6e}  {rel:.1e}")


# =====================================================================
# Верхнеуровневая функция
# =====================================================================

def _jit_warmup():
    """Компиляция numba-ядер в главном процессе (кэш на диске для воркеров)."""
    n = 8
    a = np.linspace(0.0, 1.0, n)
    x0 = np.array([0.5, 0.3, 0.0])
    P0 = 1e-4 * np.eye(3)
    run_ekf(a, np.zeros(n), x0, P0, np.full(3, 1e-6), 1e-2, 1e-1, 10e-3, 0)
    _ekf_nll_grad(a, np.zeros(n), x0, P0, np.full(3, 1e-6), 1e-4, 1e-2,
                  2 * np.pi * 1e-4, 0, np.zeros((2, 3)), np.zeros(2), np.zeros(2),
                  np.empty(2))
    _grid_loglik(np.ones(n), np.zeros(n), np.ones(n),
                 np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    _fisher_core(np.ones(n), np.zeros(n), np.ones(n), 0.1, 0.1,
                 1e-8, 1e-8, 1.0, 1.0, 2, 1e-4)


def _check_cfg(tcfg: TuneCfg):
    checks = (("sigma_init_mode", ("bins", "split", "manual")),
              ("tune_mode", ("alternate", "joint")),
              ("r_method", ("fisher", "grid")),
              ("q_optimizer", ("lbfgs", "nm")))
    for field, allowed in checks:
        v = getattr(tcfg, field)
        if v not in allowed:
            raise ValueError(f"{field}: {' | '.join(allowed)}, получено {v!r}")
    if tcfg.sigma_init_mode == "manual" and tcfg.sigma_init is None:
        raise ValueError("для sigma_init_mode='manual' задайте sigma_init=(sA, sph)")


def fit_vibration_compensation(alp, P_exp, az_m, *,
                               T, ty, T_RP,
                               tau_bounds, Kz_bounds,
                               dA_model, dB_model, dph_model,
                               poi, warmup, win_size,
                               tune_cfg: Optional[TuneCfg] = None,
                               tau_step=1, fz_dtype=np.float64, fz_chunk_rows=256,
                               n_pso=None, q_init_check=(), grad_check=True,
                               n_particles=30, n_iter=30, n_jobs=None, seed=None):
    """
    Подбор (tau, Kz): NLL инноваций EKF с внутренней настройкой Q, R
    и RMS оконного cos-fit (для сравнения).

    tau_step      : шаг узлов таблицы F_z(tau) в отсчётах акселерометра;
                    между узлами -- линейная интерполяция.
    fz_dtype      : тип таблицы (float32 вдвое меньше памяти).
    fz_chunk_rows : строк на один FFT-блок при построении таблицы.
    n_pso         : PSO на первых n_pso сбросах (None -- на всех).
    q_init_check  : множители q_init для проверки зависимости от старта.
    grad_check    : сравнить аналитический градиент с разностным в EKF-точке.
    """
    tcfg = tune_cfg or TuneCfg()
    _check_cfg(tcfg)
    if poi < 4:
        raise ValueError(f"poi должно быть >= 4, получено {poi}")
    if len(alp) - poi <= warmup + 10:
        raise ValueError("слишком мало точек после poi + warmup")
    if n_pso is not None and n_pso - poi <= warmup + 10:
        raise ValueError("n_pso слишком мало для poi + warmup")

    _jit_warmup()

    t_step = T_RP / N_RP
    weight_vec, win_len = build_weight_vec(T, ty, t_step)
    prm = EkfParams(T=T,
                    q_init=np.array([dA_model ** 2, dB_model ** 2, dph_model ** 2]),
                    poi=poi, warmup=warmup)

    t0 = time.perf_counter()
    shm, table, meta = build_fz_table(az_m, weight_vec, win_len, tau_bounds,
                                      tau_step, fz_dtype, fz_chunk_rows)
    results = {}
    try:
        ref = vibration_phase(az_m, meta.tau_lo, weight_vec, win_len)
        err = float(np.max(np.abs(table[0].astype(np.float64) - ref)))
        mb = table.nbytes / 2 ** 20
        print(f"Таблица F_z: {meta.shape[0]} x {meta.shape[1]} ({meta.dtype}), "
              f"{mb:.0f} МБ, {time.perf_counter() - t0:.1f} с; "
              f"max|F_table - F_direct| при tau={meta.tau_lo}: {err:.1e} рад\n")

        tau_hi_eff = meta.tau_lo + (meta.shape[0] - 1) * meta.tau_step
        n_cols = len(alp) if n_pso is None else n_pso
        ctx = dict(alp=alp[:n_cols], P_exp=P_exp[:n_cols], prm=prm, tcfg=tcfg,
                   win_edges=make_windows(n_cols, win_size),
                   tau_bounds=(float(meta.tau_lo), float(tau_hi_eff)),
                   Kz_bounds=Kz_bounds,
                   tau_lo=meta.tau_lo, tau_step=meta.tau_step, n_cols=n_cols)

        for kind, title in (("kalman", "fitness = NLL инноваций EKF"),
                            ("windowed", "fitness = RMS оконного cos-fit")):
            print(f"=== PSO, {title} (tune_mode={tcfg.tune_mode}, "
                  f"r_method={tcfg.r_method}, q_optimizer={tcfg.q_optimizer}, "
                  f"N={n_cols}) ===")
            t0 = time.perf_counter()
            tau, Kz, best, _, n_calls = pso_parallel(kind, ctx, meta, n_particles,
                                                     n_iter, n_jobs, seed)
            dt = time.perf_counter() - t0

            Fz = fz_lookup(table, tau, meta.tau_lo, meta.tau_step, len(alp))
            J, info = tune_and_score(Fz, Kz, alp, P_exp, prm, tcfg, diagnostics=True,
                                     grad_check=(grad_check and kind == "kalman"))
            extra = "" if kind == "kalman" else f", RMS(win)={best:.4e}"
            print(f"  -> tau={tau:.2f}, Kz={Kz:.5f}{extra} "
                  f"[{n_calls} вычислений, {dt:.1f} с]")
            if not info:
                print("  EKF не сошёлся в найденной точке\n")
                results[kind] = {"tau": tau, "Kz": Kz, "J": J}
                continue
            print(f"  итоговая оценка по всем данным (N={len(alp)}):")
            print_point(J, info)

            res = {"tau": tau, "Kz": Kz, "J": J,
                   **{k: v for k, v in info.items() if k != "states"}}
            res["Q_std"] = np.sqrt(info["q"])

            if kind == "kalman" and q_init_check:
                print("  проверка зависимости от q_init:")
                checks = []
                for f in q_init_check:
                    prm_f = replace(prm, q_init=prm.q_init * f)
                    Jf, inf_f = tune_and_score(Fz, Kz, alp, P_exp, prm_f, tcfg,
                                               diagnostics=True)
                    if not inf_f:
                        print(f"    q_init x{f:g}: EKF не сошёлся")
                        continue
                    qs = np.sqrt(inf_f["q"])
                    bound = ", ".join(inf_f["q_at_bound"]) or "нет"
                    print(f"    q_init x{f:g}: J={Jf:.6e} (dJ={Jf - J:+.1e}), "
                          f"sigma_A={inf_f['sigma_A']:.3e}, "
                          f"sigma_ph={inf_f['sigma_ph']:.3e}, "
                          f"dA={qs[0]:.2e}, dB={qs[1]:.2e}, dph={qs[2]:.2e}; "
                          f"у границы: {bound}")
                    checks.append({"factor": f, "J": Jf, "sigma_A": inf_f["sigma_A"],
                                   "sigma_ph": inf_f["sigma_ph"], "Q_std": qs})
                res["q_init_check"] = checks
            print()
            results[kind] = res
    finally:
        table = None          # снять ссылки на буфер до close()
        shm.close()
        shm.unlink()
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
        T=10e-3,                    # длительность плеча, с
        ty=20e-6,                   # длительность импульса, с
        T_RP=33e-3,                 # запись акселерометра на сброс, с
        tau_bounds=(0, 5000),       # отсчёты акселерометра
        Kz_bounds=(0.0, 1.5),
        dA_model=5e-3, dB_model=5e-3, dph_model=1e-3,   # начальная Q
        poi=200, warmup=200, win_size=20,
        tune_cfg=TuneCfg(sigma_init_mode="bins",
                         tune_mode="alternate", r_method="fisher",
                         q_optimizer="lbfgs",
                         n_outer=3, q_free=(True, True, True),
                         # q_log_bounds=(-12, -2),
                         ),
        tau_step=1,                 # 5 -> таблица в 5 раз меньше
        fz_dtype=np.float64,        # np.float32 -> вдвое меньше памяти
        n_pso=None,
        q_init_check=(0.1, 10.0),
        grad_check=True,
        n_particles=30, n_iter=30, seed=0)

    print("=== Итог ===")
    for method, r in results.items():
        if "sigma_A" not in r:
            print(f"{method:9s}: tau={r['tau']:8.2f}  Kz={r['Kz']:.5f}  EKF не сошёлся")
            continue
        st = r["nu_stats"]
        print(f"{method:9s}: tau={r['tau']:8.2f}  Kz={r['Kz']:.5f}  J={r['J']:.6e}  "
              f"sigma_A={r['sigma_A']:.3e}  sigma_ph={r['sigma_ph']:.3e}  "
              f"std(nu)={st['std']:.3f}  rho1={st['rho'][0]:+.3f}  "
              f"LB p={st['lb_pvalue']:.3f}")


if __name__ == "__main__":
    main()