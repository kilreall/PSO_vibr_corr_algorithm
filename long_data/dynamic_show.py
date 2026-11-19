"""
Динамика коэффициента компенсации вибрации Kz и ускорение g по ОБОИМ знакам чирпа
(атомный гравиметр).

Данные: текстовая таблица date time norm+ norm- scan+ scan- corr+ corr- offset.
Одна строка = один цикл; ветки "+" и "-" лежат в разных колонках одной строки и
обрабатываются вместе (строка, плохая хотя бы для одного знака, выбрасывается
целиком, чтобы ряды двух веток оставались согласованными по циклам).

Для КАЖДОЙ ветки независимо (alp = scan, P_exp = norm, F_z = FZ_SIGN * corr / CORR_PER_G):
  1. данные режутся на блоки по BLOCK_SIZE сбросов (хвост отбрасывается);
  2. в каждом блоке подбирается свой Kz двумя критериями (PSO по Kz, 1D):
       "kalman"   -- NLL инноваций EKF (с внутренней настройкой Q и R);
       "windowed" -- RMS остатков оконного cos-fit;
  3. по лучшему набору параметров EKF -- итоговый прогон по всему ряду, ряд g_±(цикл).

Среднее g по двум веткам:  g = (g_+ + g_-) / 2  (после приведения знака G_SIGN).
Фазовые сдвиги, не зависящие от направления k_eff, входят в g_+ и g_- с разными
знаками и в среднем сокращаются.

Сравнение с методом БЕЗ фильтра Калмана ("фринж по N точкам"): подряд идущие
сбросы режутся на непересекающиеся группы по FRINGE_NPTS точек; в каждой группе
линейный фит y = A + c1*cos(Phi0) + c2*sin(Phi0) даёт фазу фринджа и по ней g
(для каждого знака отдельно, затем среднее). Компенсация вибрации -- с тем же Kz.

Kz для компенсации (KZ_MODE):
  "global" -- один Kz на весь ряд (лучший по BEST_MODE);
  "blocks" -- кусочно-постоянный Kz(t): в каждом блоке свой Kz из подбора.
Фринж всегда считается в обоих режимах, чтобы получить эффект Kz(t) на g.

Чувствительность g к Kz: dg/dKz (мкГал на единицу Kz) считается численно
(пересчёт фазы при Kz +- dKz/2) для фринджа (по группам) и для EKF (по циклам).

Время: ось времени строится по реальным меткам оставшихся строк таблицы;
T_CYCLE_NOMINAL (если задан) -- номинальный период цикла для пересчёта
чувствительности в мкГал/sqrt(Гц).

Результат: сводки в консоль и графики Kz(t) (обе ветки, оба критерия), g(t)
(EKF и фринж на одном графике), девиация Аллана g по числу циклов, график
чувствительности и эффекта Kz(t). Файлы (CSV, PNG) не сохраняются.

Компенсация чирпа:  alp_comp = alp - Kz * F_z / (2*pi*T^2).  Kz ~ 9.8.

---------------------------------------------------------------------
Обозначения
---------------------------------------------------------------------
  alp, P_exp   чирп (по сбросам) и нормированный сигнал интерферометра;
  T            длительность плеча интерферометра, с;
  x = [A, B, ph]
               состояние EKF: смещение, амплитуда, фаза фринджа;
               модель измерения  y = A - B*cos(Phi0 - ph);
  e            инновация; H -- якобиан модели; h = H P^- H^T;
  b = B^2 sin^2(Phi) -- чувствительность сигнала к шуму фазы;
  sigma_A      шум детектирования; sigma_ph -- шум фазы, рад;
  S = h + sigma_A^2 + b*sigma_ph^2 -- дисперсия инновации;
  q            диагональ Q (дисперсии шагов блуждания состояния за сброс);
  J            средний NLL инноваций: J = mean(ln S_i + e_i^2 / S_i).
"""

import os

# один поток BLAS на процесс: параллелизм идёт по блокам (до импорта numpy)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize, minimize_scalar, nnls
from scipy.stats import chi2 as chi2_dist


# =====================================================================
# 1. Константы
# =====================================================================

# --- физика ---
LM = 780e-9                  # длина волны лазера, м
KEFF = 4 * np.pi / LM        # эффективный волновой вектор, 1/м
UGAL = 1e8                   # м/с^2 -> мкГал

# --- служебные значения ---
BAD_FITNESS = 1e6            # fitness при сбое обработки точки Kz
BIG_OBJ = 1e10               # значение цели при сбое EKF внутри оптимизатора
LN10 = np.log(10.0)          # для производных по log10-параметрам

Q_NAMES = np.array(["A", "B", "ph"])

# --- коэффициенты PSO ---
PSO_C1 = 2.0                 # когнитивная составляющая (к личному лучшему)
PSO_C2 = 2.0                 # социальная составляющая (к глобальному лучшему)
PSO_INERTIA = 0.9            # инерция в начале
PSO_INERTIA_END = 0.4        # инерция в конце (линейное убывание)
PSO_VMAX_FRAC = 0.25         # ограничение скорости, доля размера области


# =====================================================================
# 2. Компенсация чирпа
# =====================================================================

def compensate_alp(alp, Fz, Kz, T):
    """
    Вычитание вибрационной добавки из чирпа: alp_comp = alp - Kz * Fz / (2*pi*T^2).
    Kz -- скаляр или массив той же длины, что и alp (кусочно-постоянный Kz(t)).
    """
    return alp - Kz * Fz / (2 * np.pi * T ** 2)


# =====================================================================
# 3. Модель фринджа и начальный фит
# =====================================================================

def g_per_rad(T):
    """м/с^2 на 1 рад фазы фринджа: ph = KEFF * g * T^2."""
    return 1.0 / (KEFF * T ** 2)


def initial_fit(alp, y, T):
    """
    Линейный МНК: y = A + c1*cos(Phi0) + c2*sin(Phi0), Phi0 = 2*pi*T^2*alp.
    Эквивалентен модели A - B*cos(Phi0 - ph) при B = sqrt(c1^2 + c2^2),
    ph = atan2(-c2, -c1).
    Возвращает (x0 = [A, B, ph], P0 (3x3), невязки фита).

    P0 -- ковариация x0: ковариация линейного МНК, пересчитанная в (A, B, ph)
    через якобиан преобразования (c1, c2) -> (B, ph).
    """
    Phi0 = 2 * np.pi * T ** 2 * alp
    X = np.column_stack((np.ones_like(Phi0), np.cos(Phi0), np.sin(Phi0)))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef

    s2 = float(res @ res) / (len(y) - 3)           # несмещённая дисперсия остатка
    cov_lin = s2 * np.linalg.inv(X.T @ X)

    A, c1, c2 = coef
    B2 = c1 * c1 + c2 * c2
    B = np.sqrt(B2)
    ph = np.arctan2(-c2, -c1)

    # якобиан (A, c1, c2) -> (A, B, ph)
    Jac = np.array([[1.0, 0.0, 0.0],
                    [0.0, c1 / B, c2 / B],
                    [0.0, -c2 / B2, c1 / B2]])
    P0 = Jac @ cov_lin @ Jac.T
    return np.array([A, B, ph]), P0, res


# =====================================================================
# 4. Стартовые sigma_A, sigma_ph (только для первого прогона EKF)
# =====================================================================
#
# Дисперсия невязки в точке с фазой Phi:
#     var(res) = sigma_A^2 + sigma_ph^2 * B^2 sin^2(Phi)

def _phase_sensitivity(alp, x0, T):
    """B^2 sin^2(Phi) для каждой точки и сама фаза Phi."""
    A, B, ph = x0
    Phi = 2 * np.pi * T ** 2 * alp - ph
    return (B * np.sin(Phi)) ** 2, Phi


def sigmas_split(res, x0, alp, T):
    """Дисперсия невязки делится пополам между детектированием и фазой."""
    n = len(res)
    v = n / (n - 3) * float(np.mean(res ** 2))     # поправка на 3 подогнанных параметра
    xs, _ = _phase_sensitivity(alp, x0, T)
    m = float(np.mean(xs))
    return np.sqrt(v / 2), np.sqrt(v / (2 * max(m, 1e-12)))


def sigmas_bins(res, x0, alp, T, n_bins, min_per_bin, floor_frac):
    """
    Метод моментов по бинам Phi mod pi:
        v_b = sigma_A^2 + sigma_ph^2 * <B^2 sin^2(Phi)>_b,
    взвешенный NNLS с перевзвешиванием. None, если шумы неразделимы
    (мало заполненных бинов или чувствительность почти не меняется).
    """
    n = len(res)
    corr = n / (n - 3)
    v_tot = corr * float(np.mean(res ** 2))
    xs, Phi = _phase_sensitivity(alp, x0, T)
    m_x = float(np.mean(xs))

    # разбиение по фазе (sin^2 имеет период pi)
    bins = np.minimum((np.mod(Phi, np.pi) / np.pi * n_bins).astype(int), n_bins - 1)
    used = [b for b in range(n_bins) if np.sum(bins == b) >= min_per_bin]
    if len(used) < 2:
        return None

    n_b = np.array([np.sum(bins == b) for b in used], dtype=float)      # точек в бине
    x_b = np.array([xs[bins == b].mean() for b in used])                # <B^2 sin^2>
    v_b = np.array([corr * np.mean(res[bins == b] ** 2) for b in used])  # дисперсия невязки
    if np.ptp(x_b) < 0.1 * max(m_x, 1e-12):
        return None

    # v_b = coef[0] + coef[1] * x_b, coef >= 0
    X = np.column_stack((np.ones(len(used)), x_b))
    w = np.sqrt(n_b) / max(v_tot, 1e-30)
    for _ in range(5):                                  # перевзвешивание (IRLS)
        coef, _ = nnls(X * w[:, None], v_b * w)
        w = np.sqrt(n_b) / np.maximum(X @ coef, 1e-3 * v_tot)

    sA2 = max(coef[0], floor_frac * v_tot)
    sph2 = max(coef[1], floor_frac * v_tot / max(m_x, 1e-12))
    return np.sqrt(sA2), np.sqrt(sph2)


def start_sigmas(res, x0, alp, T, cfg):
    """Стартовые sigma_A, sigma_ph и описание источника (для печати)."""
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
# 5. EKF (numba): состояние [A, B, ph], F = I, диагональная Q
# =====================================================================
#
# Один шаг фильтра (по одному сбросу i):
#   1) предсказание:   x^- = x,  P^- = P + diag(q)           (F = I)
#   2) невязка:        e = y - (A - B cos Phi),  Phi = 2*pi*T^2*alp - ph
#   3) якобиан:        H = [1, -cos Phi, -B sin Phi]
#   4) дисперсия:      S = H P^- H^T + sigma_A^2 + B^2 sin^2(Phi) * sigma_ph^2
#   5) обновление:     K = P^- H^T / S,  x = x^- + K e,  P = P^- - K K^T S

@njit(cache=True)
def _ekf_core(alp, y, x0, P0, qd, sA2, sph2, two_pi_T2, start,
              e_out, h_out, b_out, x_out, p_out):
    """
    Прямой прогон EKF с сохранением траектории.

    Вход:
      alp, y     -- компенсированный чирп и сигнал (после начального фита);
      x0, P0     -- начальные состояние и ковариация;
      qd         -- диагональ Q (3,);  sA2, sph2 -- sigma_A^2, sigma_ph^2;
      two_pi_T2  -- 2*pi*T^2;
      start      -- с какого индекса точки входят в сумму NLL (warmup).
    Выход (заполняются на месте):
      e_out -- инновации; h_out -- H P^- H^T; b_out -- B^2 sin^2(Phi);
      x_out -- апостериорные оценки состояния; p_out -- P[2,2].
    Возвращает (sum_{i>=start} [ln S_i + e_i^2/S_i], число таких точек);
    (inf, 0) при сбое (S <= 0 или нечисловая невязка).
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
        # --- 1) предсказание ---
        for j in range(3):
            P[j, j] += qd[j]

        # --- 2-3) невязка и якобиан ---
        Phi = two_pi_T2 * alp[i] - x[2]
        c = np.cos(Phi)
        s = np.sin(Phi)
        e = y[i] - (x[0] - x[1] * c)
        H[0] = 1.0
        H[1] = -c
        H[2] = -x[1] * s

        # --- 4) дисперсия инновации ---
        hph = 0.0                                     # H P^- H^T
        for j in range(3):
            PHt[j] = P[j, 0] * H[0] + P[j, 1] * H[1] + P[j, 2] * H[2]
            hph += H[j] * PHt[j]
        bsin2 = x[1] * x[1] * s * s                   # B^2 sin^2(Phi)
        S = hph + sA2 + bsin2 * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return np.inf, 0

        # --- 5) обновление ---
        for j in range(3):
            K[j] = PHt[j] / S
            x[j] += K[j] * e
        for j in range(3):
            for k in range(3):
                P[j, k] -= K[j] * K[k] * S
        for j in range(3):                            # симметризация P
            for k in range(j + 1, 3):
                m = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = m
                P[k, j] = m

        # --- запись траектории и вклад в NLL ---
        e_out[i] = e
        h_out[i] = hph
        b_out[i] = bsin2
        x_out[i, 0] = x[0]
        x_out[i, 1] = x[1]
        x_out[i, 2] = x[2]
        p_out[i] = P[2, 2]

        if i >= start:
            nll += np.log(S) + e * e / S
            cnt += 1

    return nll, cnt


@njit(cache=True)
def _ekf_nll_grad(alp, y, x0, P0, qd, sA2, sph2, two_pi_T2, start,
                  dq, dsA2, dsph2, grad):
    """
    EKF + точный градиент NLL по параметрам theta_m (уравнения чувствительности).

    Параллельно с x и P в фильтре распространяются их производные
    dx[m] = dx/dtheta_m и dP[m] = dP/dtheta_m. Каждая формула шага EKF
    дифференцируется по цепному правилу.

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
    dPHt = np.empty((M, 3))       # d(P^- H^T)/dtheta_m
    dS = np.empty(M)              # dS/dtheta_m
    de = np.empty(M)              # de/dtheta_m
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
        B_hat = x[1]
        e = y[i] - (x[0] - B_hat * c)
        H[0] = 1.0
        H[1] = -c
        H[2] = -B_hat * s

        hph = 0.0                                     # H P^- H^T
        for j in range(3):
            PHt[j] = P[j, 0] * H[0] + P[j, 1] * H[1] + P[j, 2] * H[2]
            hph += H[j] * PHt[j]
        bsin2 = B_hat * B_hat * s * s                 # B^2 sin^2(Phi)
        S = hph + sA2 + bsin2 * sph2
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
            dH2 = -s * dB + B_hat * c * dph     # d(-B sin Phi)
            de[m] = -(dx[m, 0] + H[1] * dB + H[2] * dph)   # de = -H dx
            # d(H P H^T) = dH·(P H^T) + H·d(P H^T),
            # где d(P H^T) = dP H^T + P dH^T  (сохраняется в dPHt, нужен и для K).
            # dH[0] = 0, поэтому первый член -- только по компонентам 1 и 2.
            dhph = dH1 * PHt[1] + dH2 * PHt[2]
            for j in range(3):
                v = (dP[m, j, 0] * H[0] + dP[m, j, 1] * H[1] + dP[m, j, 2] * H[2]
                     + P[j, 1] * dH1 + P[j, 2] * dH2)
                dPHt[m, j] = v
                dhph += H[j] * v
            dbsin2 = 2.0 * B_hat * s * s * dB - 2.0 * B_hat * B_hat * s * c * dph
            dS[m] = dhph + dsA2[m] + dbsin2 * sph2 + bsin2 * dsph2[m]

        # --- вклад в NLL и градиент ---
        # d/dtheta [ln S + e^2/S] = dS * (1/S - e^2/S^2) + 2 e de / S
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
            for j in range(3):                  # симметризация dP
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
    """Обёртка над _ekf_core. Возвращает dict с nll, cnt, e, h, b, x, p22."""
    n = len(alp)
    e, h, b, p = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    x = np.empty((n, 3))
    nll, cnt = _ekf_core(alp, y, x0, P0, qd, sA * sA, sph * sph,
                         2 * np.pi * T * T, start, e, h, b, x, p)
    return {"nll": nll, "cnt": cnt, "e": e, "h": h, "b": b, "x": x, "p22": p}


# =====================================================================
# 6. Оценка R: метод Фишера (ML) и байесовская сетка
# =====================================================================
#
# При фиксированной траектории инноваций (e, h, b) дисперсия
#     S_k = h_k + a + b_k * p,     a = sigma_A^2,  p = sigma_ph^2,
# линейна по (a, p). Оба метода ищут (a, p) по правдоподобию инноваций.

@njit(cache=True)
def _fisher_core(e2, h, b, a, p, a_lo, p_lo, a_hi, p_hi, n_iter, tol):
    """
    Метод скоринга Фишера для (a, p) = (sigma_A^2, sigma_ph^2) с ограничениями.
    Возвращает (a, p, I00, I01, I11), где I -- информационная матрица
    в итоговой точке (при n_iter=0 -- в исходной).
    """
    for _ in range(n_iter):
        # градиент g и информационная матрица I (с точностью до множителя 1/2)
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

        # координата на границе с градиентом наружу -- заморожена
        a_frozen = (a <= a_lo * (1 + 1e-9) and g0 < 0.0) or (a >= a_hi * (1 - 1e-9) and g0 > 0.0)
        p_frozen = (p <= p_lo * (1 + 1e-9) and g1 < 0.0) or (p >= p_hi * (1 - 1e-9) and g1 > 0.0)

        if a_frozen and p_frozen:
            break
        elif p_frozen:
            da = g0 / I00
            dp = 0.0
        elif a_frozen:
            da = 0.0
            dp = g1 / I11
        else:
            det = I00 * I11 - I01 * I01
            if not (det > 1e-12 * I00 * I11):
                # почти вырожденная информация: идём по координатам по отдельности
                da = g0 / I00
                dp = g1 / I11
            else:
                # шаг Ньютона-Фишера: I^{-1} g
                da = (I11 * g0 - I01 * g1) / det
                dp = (-I01 * g0 + I00 * g1) / det

        a_new = min(max(a + da, a_lo), a_hi)
        p_new = min(max(p + dp, p_lo), p_hi)
        converged = abs(a_new - a) <= tol * a and abs(p_new - p) <= tol * p
        a = a_new
        p = p_new
        if converged:
            break

    # информация в итоговой точке (для стандартных ошибок)
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
    """
    ML-оценка sigma_A, sigma_ph и их SD из информационной матрицы.
    При n_iter=0 оценка не меняется (только SD в заданной точке).
    Возвращает (sigma_A, sigma_ph, sd_sigma_A, sd_sigma_ph).
    """
    hh = h if cfg.use_state_cov else np.zeros_like(h)
    a, p, I00, I01, I11 = _fisher_core(
        e * e, hh, b, sA * sA, sph * sph,
        cfg.sA_lo ** 2, cfg.sph_lo ** 2, cfg.sA_hi ** 2, cfg.sph_hi ** 2,
        n_iter, cfg.fisher_tol)
    sA_n, sph_n = np.sqrt(a), np.sqrt(p)
    det = I00 * I11 - I01 * I01
    if det > 0:
        # SD дисперсий -> SD sigma по дельта-методу: d sigma = d(var) / (2 sigma)
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
    """
    Апостериорные среднее и SD по сетке.
    L -- лог-правдоподобие на сетке; cell_w -- площади ячеек (или 1.0).
    Возвращает (mean_A, mean_ph, sd_A, sd_ph).
    """
    w = np.exp(L - L.max()) * cell_w
    w /= w.sum()
    pa, pp = w.sum(axis=1), w.sum(axis=0)           # маргинальные распределения
    mA, mP = float(pa @ sa), float(pp @ sp)
    sdA = np.sqrt(max(float(pa @ sa ** 2) - mA ** 2, 0.0))
    sdP = np.sqrt(max(float(pp @ sp ** 2) - mP ** 2, 0.0))
    return mA, mP, sdA, sdP


def grid_R(e, h, b, cfg):
    """
    Апостериорное среднее sigma_A, sigma_ph при равномерном prior.
    Сначала грубая логарифмическая сетка по всему диапазону, затем
    n_zoom раз сетка сужается вокруг среднего (+-zoom_k апостериорных SD).
    """
    e2 = e * e
    hh = h if cfg.use_state_cov else np.zeros_like(h)

    # грубая сетка по всему диапазону
    sa = np.geomspace(cfg.sA_lo, cfg.sA_hi, cfg.n_grid)
    sp = np.geomspace(cfg.sph_lo, cfg.sph_hi, cfg.n_grid)
    L = _grid_loglik(e2, hh, b, sa, sp)
    mA, mP, sdA, sdP = _posterior_moments(L, sa, sp, np.outer(sa, sp))
    stepA = mA * (sa[1] / sa[0] - 1)
    stepP = mP * (sp[1] / sp[0] - 1)

    # уточнение: линейные сетки вокруг текущего среднего
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
    """Новые (sigma_A, sigma_ph) по инновациям методом cfg.r_method."""
    if cfg.r_method == "grid":
        return grid_R(e, h, b, cfg)
    sA_n, sph_n, _, _ = fisher_R(e, h, b, sA, sph, cfg, cfg.fisher_iter)
    return sA_n, sph_n


# =====================================================================
# 7. Оптимизация Q (и R в режиме joint)
# =====================================================================
#
# Вектор оптимизируемых параметров z -- log10 величин:
#   alternate: z = log10 q[idx]
#   joint:     z = [log10 sigma_A, log10 sigma_ph, log10 q[idx]]
# idx -- индексы свободных компонент диагонали q.

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
    """
    Возвращает (unpack, off), где unpack(z) -> (sigma_A, sigma_ph, q).
    При joint z = [log10 sA, log10 sph, log10 q[idx]] (off = 2),
    иначе z = log10 q[idx] (off = 0), а sigma_A, sigma_ph фиксированы.
    Неоптимизируемые компоненты q берутся из переданного q.
    """
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
    Производные по log10: d(10^z)/dz = ln(10) * 10^z.
    """
    unpack, off = _unpacker(q, idx, sA, sph, joint)
    M = off + idx.size                       # число оптимизируемых параметров
    two_pi_T2 = 2 * np.pi * T * T
    grad = np.empty(M)

    def fun(z):
        sa, sp, qq = unpack(z)
        sa2, sp2 = sa * sa, sp * sp
        dq = np.zeros((M, 3))
        dsA2 = np.zeros(M)
        dsph2 = np.zeros(M)
        if joint:
            dsA2[0] = 2 * LN10 * sa2         # d sigma_A^2 / d log10 sigma_A
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
    # --- границы и стартовая точка ---
    qlo, qhi = _q_log_bounds(q_init, idx, cfg)
    if joint:
        lo = np.concatenate(([np.log10(cfg.sA_lo), np.log10(cfg.sph_lo)], qlo))
        hi = np.concatenate(([np.log10(cfg.sA_hi), np.log10(cfg.sph_hi)], qhi))
        z0 = np.concatenate(([np.log10(sA), np.log10(sph)], np.log10(q[idx])))
    else:
        lo, hi = qlo, qhi
        z0 = np.log10(q[idx])
    if z0.size == 0:                          # все компоненты q зафиксированы
        return sA, sph, q.copy(), 0
    z0 = np.clip(z0, lo, hi)
    unpack, _ = _unpacker(q, idx, sA, sph, joint)
    bounds = list(zip(lo, hi))

    # --- оптимизация ---
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


def q_bound_hits(q, q_init, idx, cfg, margin=0.05):
    """Имена компонент q у границы (в пределах margin декад)."""
    if idx.size == 0:
        return []
    lo, hi = _q_log_bounds(q_init, idx, cfg)
    z = np.log10(q[idx])
    return [str(n) for n, zz, l, h in zip(Q_NAMES[idx], z, lo, hi)
            if zz - l < margin or h - zz < margin]


# =====================================================================
# 8. Диагностика нормированных инноваций
# =====================================================================

def innovation_stats(nu, n_lags):
    """
    nu = e_i / sqrt(S_i). Для согласованного фильтра: mean ~ 0, std ~ 1,
    автокорреляция на лагах >= 1 в полосе +-1.96/sqrt(N), Льюнг-Бокс ~ chi2(n_lags).
    """
    n = len(nu)
    m = float(np.mean(nu))
    c = nu - m
    c0 = float(c @ c) / n                                   # дисперсия
    lags = np.arange(1, n_lags + 1)
    rho = np.array([float(c[:-k] @ c[k:]) / n / c0 for k in lags])   # автокорреляция
    band = 1.96 / np.sqrt(n)                                # 95%-полоса для белого шума
    lb = n * (n + 2) * float(np.sum(rho ** 2 / (n - lags)))          # Льюнг-Бокс
    return {
        "n": n, "mean": m, "mean_se": 1 / np.sqrt(n),
        "std": float(np.std(nu, ddof=1)),
        "rho": rho, "band": band,
        "n_out_band": int(np.sum(np.abs(rho) > band)),
        "ljung_box": lb, "lb_pvalue": float(chi2_dist.sf(lb, n_lags)),
        "kurtosis": float(np.mean(c ** 4) / c0 ** 2 - 3),   # эксцесс (0 для нормального)
    }


# =====================================================================
# 9. Конфигурация
# =====================================================================

@dataclass
class EkfParams:
    """Параметры EKF, не зависящие от настроек оптимизации."""
    T: float               # длительность плеча интерферометра, с
    q_init: np.ndarray     # начальная диагональ Q: [dA^2, dB^2, dph^2]
    poi: int               # точек для начального фита
    warmup: int            # инноваций после poi, не входящих в fitness


@dataclass
class TuneCfg:
    """Настройки автоподбора Q и R внутри fitness."""
    # --- стартовые sigma ---
    sigma_init_mode: str = "bins"      # "bins" | "split" | "manual"
    sigma_init: Optional[Tuple[float, float]] = None   # (sA, sph) для "manual"
    n_bins: int = 10                   # число бинов по фазе (для "bins")
    min_per_bin: int = 3               # минимум точек в бине
    sigma_floor_frac: float = 1e-2     # нижний предел вкладов в долях полной дисперсии
    # --- режим настройки ---
    tune_mode: str = "alternate"       # "alternate" | "joint"
    r_method: str = "fisher"           # "fisher" | "grid" (для alternate)
    q_optimizer: str = "lbfgs"         # "lbfgs" (аналит. градиент) | "nm"
    # --- внешние итерации R -> Q (alternate) ---
    n_outer: int = 3                   # максимум внешних итераций
    tol_sigma: float = 0.01            # сходимость по относительному изменению sigma
    tol_logq: float = 0.02             # (не используется: остановка идёт по tol_J)
    tol_J: float = 1e-4                # остановка, если J улучшился меньше чем на это
    # --- границы Q ---
    q_free: Tuple[bool, bool, bool] = (True, True, True)   # какие компоненты q подбирать
    q_decades: float = 3.0             # +-декад вокруг q_init
    q_log_bounds: Optional[Tuple[float, float]] = None  # абсолютные границы log10 q
    # --- L-BFGS-B (цель -- средний NLL) ---
    lbfgs_ftol: float = 1e-10
    lbfgs_gtol: float = 1e-6
    lbfgs_maxfun: int = 100
    # --- Nelder-Mead ---
    q_xatol: float = 0.02
    nm_fatol: float = 5e-5             # в единицах среднего NLL
    q_maxfev: int = 80                 # лимит вычислений цели (alternate)
    joint_maxfev: int = 300            # лимит вычислений цели (joint)
    # --- оценка R ---
    sA_lo: float = 1e-5
    sA_hi: float = 0.5
    sph_lo: float = 1e-4
    sph_hi: float = 3.0
    fisher_iter: int = 30
    fisher_tol: float = 1e-4
    n_grid: int = 21                   # узлов сетки по каждой оси
    n_zoom: int = 3                    # число сужений сетки
    zoom_k: float = 6.0                # полуширина сетки в апостериорных SD
    use_state_cov: bool = True         # учитывать h = H P^- H^T в S при оценке R
    # --- диагностика ---
    n_lags: int = 20


# =====================================================================
# 10. Fitness: настройка EKF и NLL инноваций
# =====================================================================

def _ekf_J(a, yy, x0, P0, q, sA, sph, prm: EkfParams):
    """Средний NLL инноваций EKF при заданных (sigma, q); (inf, None) при сбое."""
    r = run_ekf(a, yy, x0, P0, q, sA, sph, prm.T, prm.warmup)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return np.inf, None
    return r["nll"] / r["cnt"], r


def _tune_alternate(a, yy, x0, P0, q, sA, sph, prm: EkfParams, idx, tcfg: TuneCfg):
    """
    Режим "alternate": итерации «оценка R -> подбор Q при фиксированном R».

    Шаг по R считается по инновациям при ЗАМОРОЖЕННОЙ траектории (h, b), но при
    изменении R сам фильтр меняется, поэтому полный шаг может ухудшить настоящий
    NLL (и скатить sigma к нижним границам). Поэтому шаг по R проходит
    линейный поиск в лог-масштабе: пробуются доли шага 1, 1/2, 1/4 и 0 (остаться),
    выбирается лучший настоящий J. Шаг по Q принимается, только если J не вырос.
    Остановка: улучшение J за внешнюю итерацию меньше tcfg.tol_J.
    Возвращает (sigma_A, sigma_ph, q, nfev, n_it) или None, если EKF разошёлся.
    """
    w = prm.warmup
    nfev = 0
    n_it = 0
    J_cur, r = _ekf_J(a, yy, x0, P0, q, sA, sph, prm)
    if r is None:
        return None

    for n_it in range(1, tcfg.n_outer + 1):
        # a) новая оценка R по инновациям (без warmup) + линейный поиск по настоящему J
        sA_c, sph_c = estimate_R(r["e"][w:], r["h"][w:], r["b"][w:], sA, sph, tcfg)
        J_R, sA_n, sph_n = J_cur, sA, sph
        for t in (1.0, 0.5, 0.25):
            sA_t = sA * (sA_c / sA) ** t
            sph_t = sph * (sph_c / sph) ** t
            J_t, _ = _ekf_J(a, yy, x0, P0, q, sA_t, sph_t, prm)
            if J_t < J_R:
                J_R, sA_n, sph_n = J_t, sA_t, sph_t

        # b) подбор Q при фиксированном R; принимается, только если J не вырос
        _, _, q_n, nf = optimize_params(a, yy, x0, P0, prm.T, w, q, prm.q_init,
                                        idx, sA_n, sph_n, False, tcfg)
        nfev += nf
        J_Q, r_new = _ekf_J(a, yy, x0, P0, q_n, sA_n, sph_n, prm)
        if r_new is None or J_Q > J_R:
            q_n, J_Q = q, J_R
            _, r_new = _ekf_J(a, yy, x0, P0, q_n, sA_n, sph_n, prm)

        # c) проверка сходимости по улучшению настоящего J
        gain = J_cur - J_Q
        sA, sph, q, J_cur, r = sA_n, sph_n, q_n, J_Q, r_new
        if gain < tcfg.tol_J:
            break
    return sA, sph, q, nfev, n_it


def _collect_diagnostics(r, w, sA, sph, q, sA_start, sph_start, src,
                         n_it, nfev, prm: EkfParams, idx, tcfg: TuneCfg):
    """Словарь диагностики для итогового прогона EKF r (см. tune_and_score)."""
    e, h, b = r["e"][w:], r["h"][w:], r["b"][w:]
    # n_iter=0: оценка не меняется, нужны только стандартные ошибки
    _, _, sdA, sdP = fisher_R(e, h, b, sA, sph, tcfg, 0)
    S = h + sA ** 2 + b * sph ** 2

    return {
        "sigma_A": sA, "sigma_ph": sph, "sd_sigma_A": sdA, "sd_sigma_ph": sdP,
        "sigma_A_start": sA_start, "sigma_ph_start": sph_start,
        "sigma_start_src": src,
        "q": q, "n_it": n_it, "nfev": nfev,
        "q_at_bound": q_bound_hits(q, prm.q_init, idx, tcfg),
        "nu_stats": innovation_stats(e / np.sqrt(S), tcfg.n_lags),
        "dg_sigma_ph": sph * g_per_rad(prm.T),      # шум фазы на сброс, м/с^2
    }


def tune_and_score(Fz, Kz, alp, y, prm: EkfParams, tcfg: TuneCfg, diagnostics=False):
    """
    Полная обработка одной точки Kz. Возвращает (J, info).

    Fz -- вибрационная фаза F_z для всех сбросов блока; Kz -- коэффициент
    компенсации. При сбое EKF возвращается (BAD_FITNESS, {}).
    info: без diagnostics -- {sigma_A, sigma_ph, q}; с diagnostics --
    полная диагностика (см. _collect_diagnostics).
    """
    # 1) компенсация и начальный фит по первым poi точкам
    alp_comp = compensate_alp(alp, Fz, Kz, prm.T)
    alp0, y0 = alp_comp[:prm.poi], y[:prm.poi]
    x0, P0, res0 = initial_fit(alp0, y0, prm.T)
    a, yy = alp_comp[prm.poi:], y[prm.poi:]        # данные, на которых работает EKF
    w = prm.warmup

    # 2) стартовые sigma и q
    sA, sph, src = start_sigmas(res0, x0, alp0, prm.T, tcfg)
    sA_start, sph_start = sA, sph

    q = prm.q_init.astype(float).copy()
    idx = np.flatnonzero(np.asarray(tcfg.q_free, dtype=bool))

    # 3) настройка Q и R
    if tcfg.tune_mode == "joint":
        sA, sph, q, nfev = optimize_params(a, yy, x0, P0, prm.T, w, q, prm.q_init,
                                           idx, sA, sph, True, tcfg)
        n_it = 1
    else:
        tuned = _tune_alternate(a, yy, x0, P0, q, sA, sph, prm, idx, tcfg)
        if tuned is None:
            return BAD_FITNESS, {}
        sA, sph, q, nfev, n_it = tuned

    # 4) итоговый прогон и fitness
    r = run_ekf(a, yy, x0, P0, q, sA, sph, prm.T, w)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return BAD_FITNESS, {}
    J = r["nll"] / r["cnt"]
    if not diagnostics:
        return J, {"sigma_A": sA, "sigma_ph": sph, "q": q}

    info = _collect_diagnostics(r, w, sA, sph, q, sA_start, sph_start, src,
                                n_it, nfev, prm, idx, tcfg)
    return J, info


def ekf_fitness(Fz, Kz, alp, y, prm, tcfg):
    """Fitness для PSO: J, а при любом исключении -- BAD_FITNESS."""
    try:
        return tune_and_score(Fz, Kz, alp, y, prm, tcfg)[0]
    except Exception:
        return BAD_FITNESS


# =====================================================================
# 11. Оконный cos-fit (критерий для сравнения)
# =====================================================================

def make_windows(N, win):
    """Границы непересекающихся окон, покрывающих весь набор."""
    n_win = max(1, N // win)
    return np.linspace(0, N, n_win + 1).astype(int)


def windowed_cosfit(phase, y, edges):
    """
    Линейный fit A + c1*cos + c2*sin в каждом окне.
    Возвращает (RMS остатков по всем окнам, фазы фринджа по окнам).
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


def windowed_fitness(Fz, Kz, alp, y, win_edges, T):
    """Fitness для PSO: RMS оконного cos-fit; BAD_FITNESS при сбое."""
    alp_comp = compensate_alp(alp, Fz, Kz, T)
    try:
        rms, _ = windowed_cosfit(2 * np.pi * T ** 2 * alp_comp, y, win_edges)
        if not np.isfinite(rms):
            return BAD_FITNESS
        return rms
    except Exception:
        return BAD_FITNESS


# =====================================================================
# 12. PSO по Kz (1D) и обработка одного блока
# =====================================================================

def pso_1d(fun, lo, hi, n_particles, n_iter, rng, x_tol=0.0):
    """
    Глобальный PSO по одному параметру на отрезке [lo, hi] (последовательный).

    Начальные позиции -- расслоенная выборка (по одной частице на подотрезок),
    инерция линейно убывает от PSO_INERTIA до PSO_INERTIA_END, скорость
    ограничена PSO_VMAX_FRAC * (hi - lo). Остановка раньше срока, если
    разброс частиц меньше x_tol.
    Возвращает (x_best, f_best, history, n_calls).
    """
    span = hi - lo
    pos = lo + (np.arange(n_particles) + rng.random(n_particles)) / n_particles * span
    vel = rng.uniform(-0.1 * span, 0.1 * span, n_particles)
    vmax = PSO_VMAX_FRAC * span
    pbest, pbest_fit = pos.copy(), np.full(n_particles, np.inf)
    gbest, gbest_fit = float(pos[0]), np.inf
    history, n_calls = [], 0

    for it in range(n_iter):
        fits = np.array([fun(float(x)) for x in pos])
        n_calls += n_particles

        better = fits < pbest_fit
        pbest[better], pbest_fit[better] = pos[better], fits[better]
        j = int(np.argmin(fits))
        if fits[j] < gbest_fit:
            gbest_fit = float(fits[j])
            gbest = float(pos[j])
        history.append(gbest_fit)

        if x_tol > 0 and float(np.ptp(pos)) < x_tol:
            break

        w = PSO_INERTIA + (PSO_INERTIA_END - PSO_INERTIA) * it / max(n_iter - 1, 1)
        r1, r2 = rng.random(n_particles), rng.random(n_particles)
        vel = w * vel + PSO_C1 * r1 * (pbest - pos) + PSO_C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        new = pos + vel
        vel[(new < lo) | (new > hi)] = 0.0         # у стенки скорость гасим
        pos = np.clip(new, lo, hi)

    return gbest, gbest_fit, history, n_calls


def _process_block(job):
    """
    Воркер: один блок данных. Подбирает Kz критерием EKF и критерием оконного
    cos-fit, для найденного Kz (EKF) считает итоговые sigma, q и диагностику.
    """
    t0 = time.perf_counter()
    k, alp, y, Fz = job["k"], job["alp"], job["y"], job["Fz"]
    prm, tcfg, edges = job["prm"], job["tcfg"], job["edges"]
    lo, hi = job["Kz_bounds"]
    seed = job["seed"]
    rng_k = np.random.default_rng(None if seed is None else seed + 2 * k)
    rng_w = np.random.default_rng(None if seed is None else seed + 2 * k + 1)
    nan = float("nan")

    out = {"k": k, "Kz_kalman": nan, "J": nan, "sigma_A": nan, "sigma_ph": nan,
           "qA": nan, "qB": nan, "qph": nan,
           "dg_sigma_ph": nan, "nu_std": nan, "lb_p": nan, "q_at_bound": [],
           "Kz_win": nan, "rms": nan, "n_calls_k": 0, "n_calls_w": 0}

    # --- 1) критерий EKF ---
    Kz_k, J_pso, _, n_k = pso_1d(
        lambda Kz: ekf_fitness(Fz, Kz, alp, y, prm, tcfg),
        lo, hi, job["n_particles"], job["n_iter"], rng_k, job["pso_xtol"])
    out["n_calls_k"] = n_k
    if J_pso < BAD_FITNESS:
        out["Kz_kalman"] = Kz_k
        out["J"] = J_pso
        try:
            J, info = tune_and_score(Fz, Kz_k, alp, y, prm, tcfg, diagnostics=True)
        except Exception:
            J, info = BAD_FITNESS, {}
        if info:
            out.update(J=J, sigma_A=info["sigma_A"], sigma_ph=info["sigma_ph"],
                       dg_sigma_ph=info["dg_sigma_ph"],
                       nu_std=info["nu_stats"]["std"],
                       lb_p=info["nu_stats"]["lb_pvalue"],
                       q_at_bound=info["q_at_bound"])
            out["qA"], out["qB"], out["qph"] = (float(v) for v in info["q"])

    # --- 2) критерий оконного cos-fit ---
    Kz_w, rms, _, n_w = pso_1d(
        lambda Kz: windowed_fitness(Fz, Kz, alp, y, edges, prm.T),
        lo, hi, job["n_particles"], job["n_iter"], rng_w, job["pso_xtol"])
    out["n_calls_w"] = n_w
    if rms < BAD_FITNESS:
        out["Kz_win"] = Kz_w
        out["rms"] = rms

    out["dt"] = time.perf_counter() - t0
    return out


# =====================================================================
# 13. Печать и график Kz
# =====================================================================

def _print_block(r, n_blocks, i0, i1, t_c, tag=""):
    """Одна строка результата по блоку."""
    head = f"{tag}[{r['k'] + 1:3d}/{n_blocks}] сбросы {i0}..{i1 - 1}, t={t_c:9.1f} с | "
    if np.isfinite(r["Kz_kalman"]):
        bound = ",".join(r["q_at_bound"]) or "нет"
        ekf = (f"EKF: Kz={r['Kz_kalman']:.5f} J={r['J']:.5f} "
               f"sA={r['sigma_A']:.2e} sph={r['sigma_ph']:.2e} "
               f"({r['dg_sigma_ph'] * UGAL:.3g} мкГал) "
               f"std(nu)={r['nu_std']:.3f} LBp={r['lb_p']:.3f} q у границы: {bound}")
    else:
        ekf = "EKF: не сошёлся"
    win = (f"окно: Kz={r['Kz_win']:.5f} RMS={r['rms']:.4e}"
           if np.isfinite(r["Kz_win"]) else "окно: сбой")
    print(f"{head}{ekf} | {win} | {r['n_calls_k']}+{r['n_calls_w']} выч., {r['dt']:.1f} с",
          flush=True)


def _time_unit(tmax):
    """Делитель и подпись единицы времени для оси x графиков."""
    if tmax >= 2 * 3600:
        return 3600.0, "ч"
    if tmax >= 2 * 60:
        return 60.0, "мин"
    return 1.0, "с"


BRANCH_COLORS = {"+": "tab:blue", "-": "tab:red"}


def plot_kz_dynamics(res, branch, kz_ref=None):
    """График Kz(t) для выбранной ветки: EKF и оконный cos-fit."""
    import matplotlib.pyplot as plt

    t_s = np.asarray(res["t_s"], dtype=float)
    div, unit = _time_unit(float(np.nanmax(t_s)))
    t = t_s / div
    col = BRANCH_COLORS[branch]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, res["Kz_kalman"], "o-", ms=4, color=col,
            label=f"EKF (NLL инноваций), чирп {branch}")
    ax.plot(t, res["Kz_win"], "s--", ms=4, color=col, alpha=0.55,
            label=f"оконный cos-fit, чирп {branch}")
    if kz_ref is not None:
        ax.axhline(kz_ref, color="gray", ls=":", lw=1, label=f"Kz = {kz_ref:g}")
    ax.set_xlabel(f"Время, {unit} (центр блока)")
    ax.set_ylabel("Kz")
    ax.set_title(f"Динамика Kz по блокам ({res['block_size']} сбросов), чирп {branch}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()



def print_summary(res, label=""):
    """Сводка по всем блокам: среднее и разброс Kz."""
    print(f"\n=== Сводка по блокам {label}===")
    for name, key in (("EKF", "Kz_kalman"), ("окно", "Kz_win")):
        v = np.asarray(res[key], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            print(f"  {name:5s}: Kz mean={v.mean():.5f}, std={v.std(ddof=1) if v.size > 1 else 0.0:.2e}, "
                  f"min={v.min():.5f}, max={v.max():.5f} (блоков: {v.size})")
        else:
            print(f"  {name:5s}: нет валидных блоков")


# =====================================================================
# 14. Конфигурация эксперимента и верхнеуровневая функция (одна ветка)
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
    xf, xs = np.empty((n, 4)), np.empty((n, 4))
    Pf, Ps = np.empty((n, 4, 4)), np.empty((n, 4, 4))
    _ekf4_rts_core(a, np.zeros(n), np.zeros(n), np.array([0.5, 0.3, 0.0, 0.0]),
                   1e-4 * np.eye(4), np.full(4, 1e-6), 1e-4, 1e-2, xf, Pf, xs, Ps)


def _check_cfg(tcfg: TuneCfg):
    """Проверка строковых опций TuneCfg."""
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


def _check_inputs(block_size, poi, warmup, win_size):
    """Проверка размеров блока относительно poi, warmup и win_size."""
    if poi < 4:
        raise ValueError(f"poi должно быть >= 4, получено {poi}")
    if block_size - poi <= warmup + 10:
        raise ValueError("block_size слишком мал для poi + warmup")
    if block_size < win_size:
        raise ValueError("block_size меньше win_size")


def run_kz_dynamics(alp, P_exp, Fz, *,
                    T, Kz_bounds,
                    block_size, max_blocks=None, t_cycle, t_axis=None,
                    dA_model, dB_model, dph_model,
                    poi, warmup, win_size,
                    tune_cfg: Optional[TuneCfg] = None,
                    n_particles=10, n_iter=20, pso_xtol=1e-3,
                    n_jobs=None, seed=None, tag=""):
    """
    Подбор Kz в каждом блоке из block_size сбросов (EKF и оконный cos-fit)
    для ОДНОЙ ветки чирпа (вызывается отдельно для "+" и "-").

    Данные:
      alp, P_exp, Fz : (N,) чирп, сигнал и F_z (в единицах, для которых Kz ~ 9.8).
    Интерферометр:
      T             : длительность плеча, с.
    Блоки:
      block_size    : сбросов в блоке; остаток данных (< block_size) отбрасывается;
      max_blocks    : ограничить число блоков (None -- все);
      t_cycle       : период между сбросами, с -- для оси времени по умолчанию;
      t_axis        : (N,) реальное время каждого цикла, с (None -- индекс * t_cycle);
                      время блока = среднее t_axis по блоку.
    EKF / окно:
      dA_model, dB_model, dph_model : начальные std шагов состояния (q_init = квадраты);
      poi, warmup   : точек начального фита и инноваций, не входящих в fitness
                      (в пределах каждого блока);
      win_size      : размер окна оконного cos-fit.
    PSO (по Kz, 1D): Kz_bounds, n_particles, n_iter, pso_xtol, seed.
    Параллелизм: n_jobs процессов (по блокам).
    tag: метка ветки для печати (например "[+] ").
    """
    tcfg = tune_cfg or TuneCfg()
    _check_cfg(tcfg)
    _check_inputs(block_size, poi, warmup, win_size)

    alp = np.asarray(alp, dtype=np.float64).ravel()
    P_exp = np.asarray(P_exp, dtype=np.float64).ravel()
    Fz = np.asarray(Fz, dtype=np.float64).ravel()
    N = len(alp)
    if len(P_exp) != N or len(Fz) != N:
        raise ValueError(f"длины не совпадают: alp={N}, P_exp={len(P_exp)}, Fz={len(Fz)}")

    t_cycle = float(t_cycle)
    if t_axis is None:
        t_axis = np.arange(N) * t_cycle
    t_axis = np.asarray(t_axis, dtype=np.float64).ravel()
    if len(t_axis) != N:
        raise ValueError(f"длина t_axis ({len(t_axis)}) != N ({N})")

    n_blocks = N // block_size
    if max_blocks is not None:
        n_blocks = min(n_blocks, int(max_blocks))
    if n_blocks < 1:
        raise ValueError(f"данных ({N}) меньше одного блока ({block_size})")
    n_use = n_blocks * block_size
    Fz_all = np.ascontiguousarray(Fz[:n_use])

    print(f"Данных: {N} сбросов; блоков: {n_blocks} x {block_size} "
          f"(используется {n_use}, отброшено {N - n_use}); "
          f"блок ~ {block_size * t_cycle / 60:.1f} мин")
    qb = tcfg.q_log_bounds
    print(f"T={T * 1e3:g} мс, Kz_bounds={tuple(Kz_bounds)}, "
          f"PSO: {n_particles} частиц x {n_iter} итераций, "
          f"tune_mode={tcfg.tune_mode}, r_method={tcfg.r_method}, "
          f"q_optimizer={tcfg.q_optimizer}, "
          f"log10 q: {tuple(qb) if qb is not None else f'q_init ± {tcfg.q_decades:g} дек.'}\n")

    prm = EkfParams(T=T,
                    q_init=np.array([dA_model ** 2, dB_model ** 2, dph_model ** 2]),
                    poi=poi, warmup=warmup)
    edges = make_windows(block_size, win_size)

    jobs = []
    for k in range(n_blocks):
        sl = slice(k * block_size, (k + 1) * block_size)
        jobs.append(dict(k=k,
                         alp=np.ascontiguousarray(alp[sl]),
                         y=np.ascontiguousarray(P_exp[sl]),
                         Fz=np.ascontiguousarray(Fz_all[sl]),
                         prm=prm, tcfg=tcfg, edges=edges,
                         Kz_bounds=(float(Kz_bounds[0]), float(Kz_bounds[1])),
                         n_particles=n_particles, n_iter=n_iter,
                         pso_xtol=pso_xtol, seed=seed))

    _jit_warmup()
    n_jobs = max(1, min(n_jobs or mp.cpu_count(), n_blocks))

    results = []
    t_all = time.perf_counter()

    def _handle(r):
        k = r["k"]
        i0, i1 = k * block_size, (k + 1) * block_size
        t_c = float(np.mean(t_axis[i0:i1]))
        _print_block(r, n_blocks, i0, i1, t_c, tag)
        results.append(r)

    if n_jobs == 1:
        for job in jobs:
            _handle(_process_block(job))
    else:
        with mp.get_context().Pool(processes=n_jobs) as pool:
            for r in pool.imap(_process_block, jobs, chunksize=1):
                _handle(r)
    print(f"\nГотово: {n_blocks} блоков, {n_jobs} процессов, "
          f"{time.perf_counter() - t_all:.1f} с")

    # --- сборка массивов (в порядке блоков) ---
    results.sort(key=lambda r: r["k"])
    k_arr = np.arange(n_blocks)
    i_start = k_arr * block_size
    i_stop = i_start + block_size
    t_blocks = np.array([np.mean(t_axis[a:b]) for a, b in zip(i_start, i_stop)])

    def col(key):
        return np.array([r[key] for r in results], dtype=float)

    return {
        "block": k_arr, "i_start": i_start, "i_stop": i_stop,
        "t_s": t_blocks,
        "Kz_kalman": col("Kz_kalman"), "J": col("J"),
        "sigma_A": col("sigma_A"), "sigma_ph": col("sigma_ph"),
        "q_A": col("qA"), "q_B": col("qB"), "q_ph": col("qph"),
        "dg_sigma_ph_ugal": col("dg_sigma_ph") * UGAL,
        "nu_std": col("nu_std"), "lb_p": col("lb_p"),
        "Kz_win": col("Kz_win"), "rms": col("rms"),
        "block_size": block_size, "n_dropped": N - n_use, "blocks": results,
        "Fz_all": Fz_all, "n_use": n_use, "prm": prm,
    }


# =====================================================================
# 15. Загрузка таблицы (обе ветки), лучший прогон EKF
# =====================================================================

def _spike_mask(y, alp, k=8.0, win=41):
    """
    Выбросы сигнала y: для каждого значения чирпа (лестница scan) -- отклонение
    от скользящей медианы больше k робастных sigma (1.4826 * MAD).
    """
    from scipy.ndimage import median_filter
    out = np.zeros(len(y), dtype=bool)
    for v in np.unique(alp):
        idx = np.flatnonzero(alp == v)
        s = y[idx]
        r = s - median_filter(s, size=win, mode="nearest")
        # локальная робастная sigma: скользящая медиана |r| (шум может дрейфовать)
        sig = 1.4826 * median_filter(np.abs(r), size=501, mode="nearest")
        out[idx] = np.abs(r) > k * np.maximum(sig, 1e-12)
    return out


def load_table_pm(path, row_range=None, branch="+", corr_max=60.0, spike_k=8.0,
                  t_cycle_nominal=None):
    """
    Читает таблицу data.txt: date time norm+ norm- scan+ scan- corr+ corr- offset
    (одна строка = один цикл, плюс и минус лежат в РАЗНЫХ колонках одной строки).
    Возвращает ОДНУ выбранную ветку (branch = "+" или "-").

    row_range: (i0, i1) -- какие строки файла брать (None -- все).
    Роли: alp = scan, P_exp = norm, corr = ускорение в бинах (используется как F_z
    без масштаба и знака: они входят в Kz).
    Строка выбрасывается, если |corr| > corr_max или сигнал -- выброс (_spike_mask)
    у ВЫБРАННОЙ ветки.

    t_cycle_nominal: номинальный период цикла, с (None -- брать по файлу).

    Возвращает (data, t_cycle, info):
      data = (alp, P_exp, corr);
      t_cycle -- номинальный период (если задан) либо средний по файлу, с;
      info["t"] -- реальное время оставшихся строк, с (от первой строки диапазона);
      info["t_cycle_file"] -- средний период по оставшимся строкам.
    """
    if branch not in ("+", "-"):
        raise ValueError(f"branch: '+' | '-', получено {branch!r}")
    meta = np.loadtxt(path, dtype=str, skiprows=1, usecols=(0, 1))
    num = np.loadtxt(path, skiprows=1, usecols=range(2, 9))
    t = np.char.add(np.char.add(meta[:, 0], "T"), meta[:, 1]).astype("datetime64[ms]")
    t = (t - t[0]) / np.timedelta64(1, "s")
    if row_range is not None:
        sl = slice(int(row_range[0]), int(row_range[1]))
        num, t = num[sl], t[sl]
    t = t - t[0]

    y_p, y_m, a_p, a_m, c_p, c_m = (num[:, i] for i in range(6))
    y, a, c = (y_p, a_p, c_p) if branch == "+" else (y_m, a_m, c_m)

    bad = np.abs(c) > corr_max
    n_corr = int(bad.sum())
    bad |= _spike_mask(y, a, spike_k)
    keep = ~bad

    data = tuple(np.ascontiguousarray(v[keep]) for v in (a, y, c))
    t_keep = np.ascontiguousarray(t[keep])
    t_file = float((t_keep[-1] - t_keep[0]) / (keep.sum() - 1))
    t_cycle = float(t_cycle_nominal) if t_cycle_nominal else t_file
    info = {"n_total": len(bad), "n_bad": int(bad.sum()), "n_bad_corr": n_corr,
            "t": t_keep, "t_cycle_file": t_file}
    print(f"Таблица {path}, ветка {branch}: {len(bad)} строк, выброшено {info['n_bad']} "
          f"(из них |corr|>{corr_max:g}: {n_corr}); осталось {int(keep.sum())} циклов")
    print(f"Период цикла: по файлу {t_file * 1e3:.2f} мс (с учётом пропусков), "
          f"используется {t_cycle * 1e3:.2f} мс"
          + (" (номинал)" if t_cycle_nominal else ""))
    if t_cycle_nominal and abs(t_file / t_cycle_nominal - 1) > 0.05:
        print(f"  ВНИМАНИЕ: период по файлу отличается от номинала более чем на 5%")
    return data, t_cycle, info


def pick_best_kalman(res, mode="min_J"):
    """
    Параметры «наилучшего» EKF-прогона по результатам блоков.
      mode="min_J"   -- блок с минимальным средним NLL инноваций J;
      mode="median"  -- медианы Kz, sigma_A, sigma_ph и (в log10) q по блокам.
    Возвращает dict(block, Kz, sigma_A, sigma_ph, q[3]).
    """
    q_all = np.column_stack((res["q_A"], res["q_B"], res["q_ph"]))
    ok = (np.isfinite(res["Kz_kalman"]) & np.isfinite(res["J"])
          & np.isfinite(res["sigma_A"]) & np.isfinite(res["sigma_ph"])
          & np.all(np.isfinite(q_all), axis=1))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise RuntimeError("ни в одном блоке EKF не сошёлся -- лучшего прогона нет")

    if mode == "min_J":
        k = int(idx[np.argmin(res["J"][idx])])
        return {"block": k, "Kz": float(res["Kz_kalman"][k]),
                "sigma_A": float(res["sigma_A"][k]),
                "sigma_ph": float(res["sigma_ph"][k]),
                "q": q_all[k].astype(np.float64).copy()}
    if mode == "median":
        return {"block": None, "Kz": float(np.median(res["Kz_kalman"][idx])),
                "sigma_A": float(np.median(res["sigma_A"][idx])),
                "sigma_ph": float(np.median(res["sigma_ph"][idx])),
                "q": 10.0 ** np.median(np.log10(q_all[idx]), axis=0)}
    raise ValueError(f"BEST_MODE: 'min_J' | 'median', получено {mode!r}")


def print_best(best, label=""):
    src = f"блок {best['block'] + 1}" if best["block"] is not None else "медианы по блокам"
    q = best["q"]
    print(f"\n=== Параметры итогового прогона EKF {label}({src}) ===\n"
          f"  Kz={best['Kz']:.5f}  sigma_A={best['sigma_A']:.3e}  sigma_ph={best['sigma_ph']:.3e}\n"
          f"  q = [{q[0]:.3e}, {q[1]:.3e}, {q[2]:.3e}]  (A, B, ph)")


def kz_per_cycle(res, key, fallback, n_use):
    """
    Кусочно-постоянный Kz(t) по циклам: значение блока повторяется block_size раз.
    Блоки без валидного Kz (NaN) получают fallback. Длина результата == n_use.
    """
    kz = np.asarray(res[key], dtype=float).copy()
    kz[~np.isfinite(kz)] = fallback
    out = np.repeat(kz, res["block_size"])
    if len(out) != n_use:
        raise RuntimeError("kz_per_cycle: длина не совпала с n_use")
    return out


# =====================================================================
# 16. Ряд g: EKF (с фильтром) и фринж по N точкам (без фильтра)
# =====================================================================

def final_kalman_run(alp, y, Fz, best, prm, g_sign=1.0):
    """
    Один прогон EKF по всему ряду с ФИКСИРОВАННЫМИ параметрами best
    (без перенастройки). Начальный фит -- по первым poi точкам ряда.
    best["Kz"] -- скаляр или массив по циклам (кусочно-постоянный Kz(t)).

    Ряд g -- апостериорная фаза фринджа x[:, 2] после warmup, переведённая
    в ускорение: g = g_sign * ph / (KEFF*T^2); вычитается среднее, единицы -- мкГал.
    g_sign (+-1) приводит ветку к общему знаку g (см. G_SIGN в main).
    Это ОТФИЛЬТРОВАННАЯ оценка: на малом числе циклов она сглажена фильтром.
    """
    T, poi, w = prm.T, prm.poi, prm.warmup
    alp_comp = compensate_alp(alp, Fz, best["Kz"], T)
    x0, P0, _ = initial_fit(alp_comp[:poi], y[:poi], T)
    r = run_ekf(np.ascontiguousarray(alp_comp[poi:]), np.ascontiguousarray(y[poi:]),
                x0, P0, np.asarray(best["q"], dtype=np.float64),
                best["sigma_A"], best["sigma_ph"], T, w)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        raise RuntimeError("итоговый прогон EKF разошёлся")

    ph = r["x"][w:, 2]
    g_ugal = g_sign * (ph - ph.mean()) * g_per_rad(T) * UGAL
    S = r["h"][w:] + best["sigma_A"] ** 2 + r["b"][w:] * best["sigma_ph"] ** 2
    nu = innovation_stats(r["e"][w:] / np.sqrt(S), 20)
    return {"g_ugal": g_ugal, "ph": ph, "J": r["nll"] / r["cnt"],
            "nu_std": nu["std"], "lb_p": nu["lb_pvalue"], "i_first": poi + w, "e_rms": float(np.sqrt(np.mean(r["e"][w:] ** 2))),}


def fringe_npoint(alp, y, Fz, Kz, T, npts, i_first, i_stop, g_sign=1.0):
    """
    Оценка g по фринджу БЕЗ фильтра Калмана.

    Сбросы i_first ... i_stop-1 режутся на непересекающиеся группы по npts подряд
    идущих циклов (npts точек фринджа при разных значениях чирпа). В каждой группе
    линейный МНК  y = A + c1*cos(Phi0) + c2*sin(Phi0),  Phi0 = 2*pi*T^2*alp_comp
    (alp_comp -- с компенсацией вибрации тем же Kz; Kz -- скаляр или массив по
    циклам), фаза ph = atan2(-c2, -c1), g = g_sign * ph / (KEFF*T^2). Фазы
    приводятся к интервалу (-pi, pi] вокруг их кругового среднего значения,
    затем вычитается среднее (мкГал).

    Начало группировки (i_first) то же, что у ряда EKF, поэтому ряды сопоставимы.
    Возвращает dict(g_ugal, ph, amp, i_center, n_groups, npts, n_distinct);
    i_center -- индекс центра каждой группы (в тех же индексах циклов, что и у EKF).
    """
    n_g = (i_stop - i_first) // npts
    if n_g < 2:
        raise ValueError("слишком мало данных для групп фринджа")
    sl = slice(i_first, i_first + n_g * npts)
    Kz_s = Kz[sl] if np.ndim(Kz) else Kz
    alp_c = compensate_alp(alp[sl], Fz[sl], Kz_s, T)
    Phi0 = (2 * np.pi * T ** 2 * alp_c).reshape(n_g, npts)
    yy = y[sl].reshape(n_g, npts)

    X = np.stack((np.ones_like(Phi0), np.cos(Phi0), np.sin(Phi0)), axis=2)   # (n_g, npts, 3)
    coef = np.einsum("gij,gj->gi", np.linalg.pinv(X), yy)                    # МНК в каждой группе
    ph = np.arctan2(-coef[:, 2], -coef[:, 1])
    amp = np.hypot(coef[:, 1], coef[:, 2])

    ph_ref = np.angle(np.mean(np.exp(1j * ph)))          # круговое среднее
    d = np.angle(np.exp(1j * (ph - ph_ref)))             # отклонение в (-pi, pi]
    d = d - d.mean()
    g_ugal = g_sign * d * g_per_rad(T) * UGAL
    i_center = i_first + (np.arange(n_g) + 0.5) * npts - 0.5
    # сколько РАЗНЫХ значений чирпа попало в группу (при полной лестнице == npts)
    s_a = np.sort(alp[sl].reshape(n_g, npts), axis=1)
    n_distinct = 1 + np.sum(np.diff(s_a, axis=1) != 0, axis=1)
    return {"g_ugal": g_ugal, "ph": ph, "amp": amp, "i_center": i_center,
            "n_groups": n_g, "npts": npts, "n_distinct": n_distinct}


def branch_consistency(g_plus, g_minus, npts):
    """
    Проверка знака G_SIGN. После приведения знака общие для двух веток вариации
    (остаточная вибрация, дрейф g) должны давать ПОЛОЖИТЕЛЬНУЮ корреляцию g_+ и g_-
    (средние по группам из npts циклов). Отрицательная -- вероятно, знак одной
    из веток в G_SIGN нужно поменять. Возвращает коэффициент корреляции.
    """
    n = len(g_plus) // npts
    a = g_plus[:n * npts].reshape(n, npts).mean(axis=1)
    b = g_minus[:n * npts].reshape(n, npts).mean(axis=1)
    return float(np.corrcoef(a, b)[0, 1])


# =====================================================================
# 16b. Чувствительность g к Kz
# =====================================================================
#
# Компенсация меняет фазу Phi0 = 2*pi*T^2*alp - Kz*Fz, поэтому ошибка dKz даёт
# сдвиг фазы фринджа ~ dKz * Fz_eff и сдвиг g:
#     dg ~ g_sign * g_per_rad(T) * UGAL * dKz * Fz_eff      (мкГал)
# Производная dg/dKz считается численно: пересчёт фазы при Kz +- dKz/2
# (Kz -- скаляр или массив по циклам; сдвигаются все значения сразу).

def kz_sensitivity_fringe(alp, y, Fz, Kz, T, npts, i_first, i_stop, g_sign, dKz=0.02):
    """dg/dKz по группам фринджа, мкГал на единицу Kz (знак учтён)."""
    f0 = fringe_npoint(alp, y, Fz, Kz - dKz / 2, T, npts, i_first, i_stop, g_sign)
    f1 = fringe_npoint(alp, y, Fz, Kz + dKz / 2, T, npts, i_first, i_stop, g_sign)
    d = np.angle(np.exp(1j * (f1["ph"] - f0["ph"])))        # разность фаз с обёрткой
    return g_sign * d / dKz * g_per_rad(T) * UGAL


def kz_sensitivity_ekf(alp, y, Fz, best, prm, g_sign, dKz=0.02):
    """dg/dKz по циклам для EKF (q, sigma заморожены), мкГал на единицу Kz."""
    b0 = dict(best, Kz=best["Kz"] - dKz / 2)
    b1 = dict(best, Kz=best["Kz"] + dKz / 2)
    p0 = final_kalman_run(alp, y, Fz, b0, prm, g_sign)["ph"]
    p1 = final_kalman_run(alp, y, Fz, b1, prm, g_sign)["ph"]
    return g_sign * (p1 - p0) / dKz * g_per_rad(prm.T) * UGAL




# =====================================================================
# 17. Девиация Аллана и графики g
# =====================================================================

def overlapping_adev(g, n_points=40, min_groups=4):
    """
    Перекрывающаяся девиация Аллана ряда g (отсчёты -- по циклам).
    m = 1 ... N // min_groups (логарифмическая сетка целых m).
    Возвращает (m, adev, err); err -- грубая оценка SD: adev / sqrt(2 (N/m - 1)).
    """
    g = np.asarray(g, dtype=np.float64)
    g = g - g.mean()                           # для устойчивости cumsum
    N = len(g)
    m_max = N // min_groups
    if m_max < 1:
        raise ValueError(f"ряд из {N} точек слишком короток для девиации Аллана")
    ms = np.unique(np.round(np.logspace(0, np.log10(m_max), n_points)).astype(int))

    c = np.concatenate(([0.0], np.cumsum(g)))
    adev = np.empty(len(ms))
    for i, m in enumerate(ms):
        avg = (c[m:] - c[:-m]) / m             # средние по всем окнам из m точек
        d = avg[m:] - avg[:-m]                 # разности соседних (встык) окон
        adev[i] = np.sqrt(0.5 * np.mean(d * d))
    err = adev / np.sqrt(2.0 * (N / ms - 1.0))
    return ms, adev, err


def adev_in_cycles(g, cycles_per_sample, n_points=40, min_groups=4):
    """
    Девиация Аллана с осью в ЦИКЛАХ измерения. У ряда EKF один отсчёт = 1 цикл,
    у ряда фринджа один отсчёт = npts циклов (cycles_per_sample = npts).
    Возвращает (m_cycles, adev, err).
    """
    ms, adev, err = overlapping_adev(g, n_points, min_groups)
    return ms * cycles_per_sample, adev, err


def white_noise_sensitivity(ms, adev, m_lo, m_hi):
    """
    Чувствительность S, мкГал/sqrt(цикл): уровень белого шума sigma(m) = S / sqrt(m),
    среднее геометрическое adev*sqrt(m) по точкам m_lo <= m <= m_hi (m -- в циклах).
    """
    sel = (ms >= m_lo) & (ms <= m_hi)
    if not sel.any():
        sel = np.zeros(len(ms), dtype=bool)
        sel[:3] = True
    return float(np.exp(np.mean(np.log(adev[sel] * np.sqrt(ms[sel])))))


# =====================================================================
# 18. Kz как состояние фильтра Калмана: x = [A, B, ph, Kz]
# =====================================================================
#
# Phi = 2*pi*T^2*alp - Kz*Fz - ph  (alp -- СЫРОЙ чирп, компенсация внутри модели)
# H = [1, -cos Phi, -B sin Phi, -B sin Phi * Fz],  F = I,  Q = diag(qA, qB, qph, qKz)

@njit(cache=True)
def _ekf4_core(c, Fz, y, x0, P0, qd, sA2, sph2, start,
               e_out, h_out, b_out, x_out, p_out):
    """
    Прямой прогон EKF с 4 состояниями. c = 2*pi*T^2*alp.
    Выход: e_out, h_out, b_out как в _ekf_core; x_out (n,4) -- состояние;
    p_out (n,2) -- P[2,2] (ph) и P[3,3] (Kz).
    Возвращает (sum_{i>=start}[ln S + e^2/S], число точек); (inf, 0) при сбое.
    """
    n = c.shape[0]
    x = x0.copy()
    P = P0.copy()
    H = np.empty(4)
    PHt = np.empty(4)
    K = np.empty(4)
    nll = 0.0
    cnt = 0

    for i in range(n):
        for j in range(4):
            P[j, j] += qd[j]

        Phi = c[i] - x[3] * Fz[i] - x[2]
        cs = np.cos(Phi)
        sn = np.sin(Phi)
        e = y[i] - (x[0] - x[1] * cs)
        H[0] = 1.0
        H[1] = -cs
        H[2] = -x[1] * sn
        H[3] = H[2] * Fz[i]

        hph = 0.0
        for j in range(4):
            acc = 0.0
            for k in range(4):
                acc += P[j, k] * H[k]
            PHt[j] = acc
            hph += H[j] * acc
        bsin2 = x[1] * x[1] * sn * sn
        S = hph + sA2 + bsin2 * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return np.inf, 0

        for j in range(4):
            K[j] = PHt[j] / S
            x[j] += K[j] * e
        for j in range(4):
            for k in range(4):
                P[j, k] -= K[j] * K[k] * S
        for j in range(4):
            for k in range(j + 1, 4):
                m = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = m
                P[k, j] = m

        e_out[i] = e
        h_out[i] = hph
        b_out[i] = bsin2
        for j in range(4):
            x_out[i, j] = x[j]
        p_out[i, 0] = P[2, 2]
        p_out[i, 1] = P[3, 3]

        if i >= start:
            nll += np.log(S) + e * e / S
            cnt += 1

    return nll, cnt


def run_ekf_kz(alp, Fz, y, x0, P0, qd, sA, sph, T, start):
    """Обёртка над _ekf4_core. Возвращает dict с nll, cnt, e, h, b, x (n,4), p (n,2)."""
    n = len(alp)
    c = 2 * np.pi * T * T * np.asarray(alp, dtype=np.float64)
    e, h, b = np.empty(n), np.empty(n), np.empty(n)
    x = np.empty((n, 4))
    p = np.empty((n, 2))
    nll, cnt = _ekf4_core(c, np.asarray(Fz, dtype=np.float64),
                          np.asarray(y, dtype=np.float64), x0, P0, qd,
                          sA * sA, sph * sph, start, e, h, b, x, p)
    return {"nll": nll, "cnt": cnt, "e": e, "h": h, "b": b, "x": x, "p": p}


def find_initial_kz(alp, y, Fz, *, n_init, T, Kz_bounds,
                    dA_model, dB_model, dph_model, poi, warmup, win_size,
                    tune_cfg: TuneCfg, n_particles, n_iter, pso_xtol, seed):
    """
    Kz и параметры EKF (sigma_A, sigma_ph, q) по первым n_init точкам -- ОДИН блок,
    без разбиения ряда (прежняя процедура: PSO по Kz, критерии EKF и оконный cos-fit).
    Возвращает (результат блока, EkfParams).
    """
    _check_cfg(tune_cfg)
    _check_inputs(n_init, poi, warmup, win_size)
    prm = EkfParams(T=T,
                    q_init=np.array([dA_model ** 2, dB_model ** 2, dph_model ** 2]),
                    poi=poi, warmup=warmup)
    job = dict(k=0,
               alp=np.ascontiguousarray(alp[:n_init], dtype=np.float64),
               y=np.ascontiguousarray(y[:n_init], dtype=np.float64),
               Fz=np.ascontiguousarray(Fz[:n_init], dtype=np.float64),
               prm=prm, tcfg=tune_cfg, edges=make_windows(n_init, win_size),
               Kz_bounds=(float(Kz_bounds[0]), float(Kz_bounds[1])),
               n_particles=n_particles, n_iter=n_iter,
               pso_xtol=pso_xtol, seed=seed)
    _jit_warmup()
    return _process_block(job), prm


def kz_filter_start(alp, y, Fz, Kz0, prm, sigma_kz0):
    """Начальные x0 (4,) и P0 (4,4): фит по первым poi точкам при Kz0; sigma_kz0 -- std Kz."""
    poi = prm.poi
    alp_c = compensate_alp(alp[:poi], Fz[:poi], Kz0, prm.T)
    x3, P3, _ = initial_fit(alp_c, y[:poi], prm.T)
    x0 = np.append(x3, Kz0)
    P0 = np.zeros((4, 4))
    P0[:3, :3] = P3
    P0[3, 3] = sigma_kz0 ** 2
    return x0, P0


def _kz_J(alp, y, Fz, x0, P0, qd, sA, sph, prm):
    """Средний NLL инноваций фильтра с Kz; (inf, None) при сбое."""
    r = run_ekf_kz(alp[prm.poi:], Fz[prm.poi:], y[prm.poi:], x0, P0, qd,
                   sA, sph, prm.T, prm.warmup)
    if not np.isfinite(r["nll"]) or r["cnt"] == 0:
        return np.inf, None
    return r["nll"] / r["cnt"], r


def tune_q_kz(alp, y, Fz, x0, P0, q3, sA, sph, prm, log_bounds, n_grid=13):
    """
    Подбор q_Kz (дисперсия шага Kz за цикл) по NLL инноваций на всём ряду:
    грубая сетка по log10 q_Kz, затем уточнение вокруг лучшего узла.
    Нижний край сетки ~ «Kz постоянен» (но всё равно уточняется фильтром).
    Возвращает (log10 q_Kz, J_лучший, сетка, J на сетке).
    """
    def J_of(lq):
        J, _ = _kz_J(alp, y, Fz, x0, P0, np.append(q3, 10.0 ** lq), sA, sph, prm)
        return J if np.isfinite(J) else BIG_OBJ

    lqs = np.linspace(log_bounds[0], log_bounds[1], n_grid)
    Js = np.array([J_of(l) for l in lqs])
    i = int(np.argmin(Js))
    lq_best, J_best = float(lqs[i]), float(Js[i])
    r = minimize_scalar(J_of, bounds=(lqs[max(i - 1, 0)], lqs[min(i + 1, n_grid - 1)]),
                        method="bounded", options=dict(xatol=0.02))
    if r.fun < J_best:
        lq_best, J_best = float(r.x), float(r.fun)
    return lq_best, J_best, lqs, Js


def kz_filter_run(alp, y, Fz, x0, P0, qd, sA, sph, prm, g_sign=1.0):
    """
    Итоговый прогон фильтра с Kz. Ряды после warmup:
    g_ugal (из ph, мкГал, без среднего), Kz(t) и его sd (из P[3,3]).
    """
    T, poi, w = prm.T, prm.poi, prm.warmup
    J, r = _kz_J(alp, y, Fz, x0, P0, qd, sA, sph, prm)
    if r is None:
        raise RuntimeError("фильтр с Kz разошёлся")
    ph = r["x"][w:, 2]
    S = r["h"][w:] + sA ** 2 + r["b"][w:] * sph ** 2
    nu = innovation_stats(r["e"][w:] / np.sqrt(S), 20)
    return {"g_ugal": g_sign * (ph - ph.mean()) * g_per_rad(T) * UGAL,
            "ph": ph, "Kz": r["x"][w:, 3],
            "sd_Kz": np.sqrt(np.maximum(r["p"][w:, 1], 0.0)),
            "J": J, "nu_std": nu["std"], "lb_p": nu["lb_pvalue"], "i_first": poi + w, "e_rms": float(np.sqrt(np.mean(r["e"][w:] ** 2))),}

# =====================================================================
# 18b. Обратный проход (RTS-сглаживатель) для фильтра с Kz
# =====================================================================
#
# Прямой EKF использует только прошлые данные, поэтому оценки ph и Kz
# запаздывают. Обратный проход Рауха-Тунга-Штрибеля (F = I):
#   C_i   = P_i (P_i + Q)^-1
#   x^s_i = x_i + C_i (x^s_{i+1} - x_i)
#   P^s_i = P_i + C_i (P^s_{i+1} - (P_{i} + Q)) C_i^T
# Якобианы берутся из прямого прогона (стандартное EKF-RTS приближение).

@njit(cache=True)
def _ekf4_rts_core(c, Fz, y, x0, P0, qd, sA2, sph2, xf, Pf, xs, Ps):
    """
    Прямой прогон EKF (4 состояния) с сохранением x_i, P_i (апостериорных),
    затем обратный проход RTS.
    Выход (на месте): xf (n,4), Pf (n,4,4) -- фильтр; xs, Ps -- сглаженные.
    Возвращает True, False при сбое (S <= 0 или нечисловая невязка).
    """
    n = c.shape[0]
    x = x0.copy()
    P = P0.copy()
    H = np.empty(4)
    PHt = np.empty(4)
    K = np.empty(4)

    # ---------- прямой проход ----------
    for i in range(n):
        for j in range(4):
            P[j, j] += qd[j]

        Phi = c[i] - x[3] * Fz[i] - x[2]
        cs = np.cos(Phi)
        sn = np.sin(Phi)
        e = y[i] - (x[0] - x[1] * cs)
        H[0] = 1.0
        H[1] = -cs
        H[2] = -x[1] * sn
        H[3] = H[2] * Fz[i]

        hph = 0.0
        for j in range(4):
            acc = 0.0
            for k in range(4):
                acc += P[j, k] * H[k]
            PHt[j] = acc
            hph += H[j] * acc
        S = hph + sA2 + x[1] * x[1] * sn * sn * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return False

        for j in range(4):
            K[j] = PHt[j] / S
            x[j] += K[j] * e
        for j in range(4):
            for k in range(4):
                P[j, k] -= K[j] * K[k] * S
        for j in range(4):
            for k in range(j + 1, 4):
                m = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = m
                P[k, j] = m

        for j in range(4):
            xf[i, j] = x[j]
            for k in range(4):
                Pf[i, j, k] = P[j, k]

    # ---------- обратный проход ----------
    Pi = np.empty((4, 4))
    Pm = np.empty((4, 4))
    dP = np.empty((4, 4))
    tmp = np.empty((4, 4))
    dx = np.empty(4)
    for j in range(4):
        xs[n - 1, j] = xf[n - 1, j]
        for k in range(4):
            Ps[n - 1, j, k] = Pf[n - 1, j, k]

    for i in range(n - 2, -1, -1):
        for j in range(4):
            for k in range(4):
                Pi[j, k] = Pf[i, j, k]
                Pm[j, k] = Pi[j, k]
            Pm[j, j] += qd[j]                      # P^-_{i+1} = P_i + Q
        Ct = np.linalg.solve(Pm, Pi)               # C^T = (P_i + Q)^-1 P_i
        for j in range(4):
            dx[j] = xs[i + 1, j] - xf[i, j]        # x^-_{i+1} = x_i
            for k in range(4):
                dP[j, k] = Ps[i + 1, j, k] - Pm[j, k]

        for j in range(4):                         # x^s_i = x_i + C dx
            acc = 0.0
            for k in range(4):
                acc += Ct[k, j] * dx[k]
            xs[i, j] = xf[i, j] + acc

        for j in range(4):                         # tmp = dP Ct
            for k in range(4):
                acc = 0.0
                for m in range(4):
                    acc += dP[j, m] * Ct[m, k]
                tmp[j, k] = acc
        for j in range(4):                         # P^s_i = P_i + Ct^T tmp
            for k in range(4):
                acc = 0.0
                for m in range(4):
                    acc += Ct[m, j] * tmp[m, k]
                Ps[i, j, k] = Pi[j, k] + acc
        for j in range(4):
            for k in range(j + 1, 4):
                m = 0.5 * (Ps[i, j, k] + Ps[i, k, j])
                Ps[i, j, k] = m
                Ps[i, k, j] = m
    return True


def kz_filter_smooth(alp, y, Fz, x0, P0, qd, sA, sph, prm, g_sign=1.0):
    """
    Фильтр с Kz + обратный проход (RTS). Параметры те же, что у kz_filter_run.
    Ряды после warmup: g_ugal (из СГЛАЖЕННОЙ фазы, мкГал, без среднего),
    Kz (сглаженный), sd_Kz (из P^s[3,3]), а также Kz_fwd -- прямой (несглаженный).
    Сглаживание идёт по всему ряду (с начала данных EKF), warmup отрезается после.
    """
    T, poi, w = prm.T, prm.poi, prm.warmup
    a = np.ascontiguousarray(alp[poi:], dtype=np.float64)
    c = 2 * np.pi * T * T * a
    F = np.ascontiguousarray(Fz[poi:], dtype=np.float64)
    yy = np.ascontiguousarray(y[poi:], dtype=np.float64)
    n = len(a)
    xf, xs = np.empty((n, 4)), np.empty((n, 4))
    Pf, Ps = np.empty((n, 4, 4)), np.empty((n, 4, 4))
    ok = _ekf4_rts_core(c, F, yy, x0, P0, np.asarray(qd, dtype=np.float64),
                        sA * sA, sph * sph, xf, Pf, xs, Ps)
    if not ok:
        raise RuntimeError("фильтр с Kz (сглаживатель) разошёлся")
    ph = xs[w:, 2]
    return {"g_ugal": g_sign * (ph - ph.mean()) * g_per_rad(T) * UGAL,
            "ph": ph, "Kz": xs[w:, 3],
            "sd_Kz": np.sqrt(np.maximum(Ps[w:, 3, 3], 0.0)),
            "Kz_fwd": xf[w:, 3], "i_first": poi + w}


# =====================================================================
# 19. Фильтр с Kz и его скоростью: x = [A, B, ph, Kz, v],  v = dKz/цикл
# =====================================================================

@njit(cache=True)
def _ekf5_core(c, Fz, y, x0, P0, Q, sA2, sph2, start,
               e_out, h_out, b_out, x_out, p_out):
    """
    EKF с 5 состояниями. Предсказание: Kz += v, P = F P F^T + Q, F = I + e3 e4^T.
    Измерение как в _ekf4_core (скорость в H не входит).
    Выход: e_out, h_out, b_out; x_out (n,5); p_out (n,2) -- P[2,2] и P[3,3].
    Возвращает (sum_{i>=start}[ln S + e^2/S], число точек); (inf, 0) при сбое.
    """
    n = c.shape[0]
    x = x0.copy()
    P = P0.copy()
    H = np.zeros(5)
    PHt = np.empty(5)
    K = np.empty(5)
    nll = 0.0
    cnt = 0

    for i in range(n):
        x[3] += x[4]
        for k in range(5):
            P[3, k] += P[4, k]
        for j in range(5):
            P[j, 3] += P[j, 4]
        for j in range(5):
            for k in range(5):
                P[j, k] += Q[j, k]

        Phi = c[i] - x[3] * Fz[i] - x[2]
        cs = np.cos(Phi)
        sn = np.sin(Phi)
        e = y[i] - (x[0] - x[1] * cs)
        H[0] = 1.0
        H[1] = -cs
        H[2] = -x[1] * sn
        H[3] = H[2] * Fz[i]

        hph = 0.0
        for j in range(5):
            acc = 0.0
            for k in range(5):
                acc += P[j, k] * H[k]
            PHt[j] = acc
            hph += H[j] * acc
        bsin2 = x[1] * x[1] * sn * sn
        S = hph + sA2 + bsin2 * sph2
        if not (S > 0.0) or not np.isfinite(e):
            return np.inf, 0

        for j in range(5):
            K[j] = PHt[j] / S
            x[j] += K[j] * e
        for j in range(5):
            for k in range(5):
                P[j, k] -= K[j] * K[k] * S
        for j in range(5):
            for k in range(j + 1, 5):
                m = 0.5 * (P[j, k] + P[k, j])
                P[j, k] = m
                P[k, j] = m

        e_out[i] = e
        h_out[i] = hph
        b_out[i] = bsin2
        for j in range(5):
            x_out[i, j] = x[j]
        p_out[i, 0] = P[2, 2]
        p_out[i, 1] = P[3, 3]

        if i >= start:
            nll += np.log(S) + e * e / S
            cnt += 1

    return nll, cnt


def kzv_Q(q3, q_a):
    """
    Q (5x5): диагональ q3 для [A, B, ph]; для (Kz, v) -- дискретный белый шум
    ускорения Kz с интенсивностью q_a (интегрированное блуждание):
    [[q_a/3, q_a/2], [q_a/2, q_a]].
    """
    Q = np.zeros((5, 5))
    Q[0, 0], Q[1, 1], Q[2, 2] = q3
    Q[3, 3] = q_a / 3.0
    Q[3, 4] = Q[4, 3] = q_a / 2.0
    Q[4, 4] = q_a
    return Q


def kzv_start(x0, P0, sigma_v0):
    """Из начальных (x0 (4,), P0 (4,4)) фильтра с Kz: добавить скорость v = 0 со std sigma_v0."""
    x5 = np.append(x0, 0.0)
    P5 = np.zeros((5, 5))
    P5[:4, :4] = P0
    P5[4, 4] = sigma_v0 ** 2
    return x5, P5


def _kzv_J(alp, y, Fz, x0, P0, Q, sA, sph, prm):
    """Средний NLL инноваций фильтра Kz + скорость; (inf, None) при сбое."""
    a, yy, F = alp[prm.poi:], y[prm.poi:], Fz[prm.poi:]
    n = len(a)
    e, h, b = np.empty(n), np.empty(n), np.empty(n)
    x = np.empty((n, 5))
    p = np.empty((n, 2))
    nll, cnt = _ekf5_core(2 * np.pi * prm.T ** 2 * np.asarray(a, dtype=np.float64),
                          np.asarray(F, dtype=np.float64), np.asarray(yy, dtype=np.float64),
                          x0, P0, Q, sA * sA, sph * sph, prm.warmup, e, h, b, x, p)
    if not np.isfinite(nll) or cnt == 0:
        return np.inf, None
    return nll / cnt, {"e": e, "h": h, "b": b, "x": x, "p": p}


def tune_q_kzv(alp, y, Fz, x0, P0, q3, sA, sph, prm, log_bounds, n_grid=13):
    """Подбор q_a по NLL инноваций на всём ряду (сетка по log10 q_a + уточнение)."""
    def J_of(lq):
        J, _ = _kzv_J(alp, y, Fz, x0, P0, kzv_Q(q3, 10.0 ** lq), sA, sph, prm)
        return J if np.isfinite(J) else BIG_OBJ

    lqs = np.linspace(log_bounds[0], log_bounds[1], n_grid)
    Js = np.array([J_of(l) for l in lqs])
    i = int(np.argmin(Js))
    lq_best, J_best = float(lqs[i]), float(Js[i])
    r = minimize_scalar(J_of, bounds=(lqs[max(i - 1, 0)], lqs[min(i + 1, n_grid - 1)]),
                        method="bounded", options=dict(xatol=0.02))
    if r.fun < J_best:
        lq_best, J_best = float(r.x), float(r.fun)
    return lq_best, J_best, lqs, Js


def kzv_filter_run(alp, y, Fz, x0, P0, q3, q_a, sA, sph, prm, g_sign=1.0):
    """Итоговый прогон: ряды после warmup -- g_ugal, Kz(t), sd_Kz, J, диагностика."""
    T, poi, w = prm.T, prm.poi, prm.warmup
    J, r = _kzv_J(alp, y, Fz, x0, P0, kzv_Q(q3, q_a), sA, sph, prm)
    if r is None:
        raise RuntimeError("фильтр Kz + скорость разошёлся")
    ph = r["x"][w:, 2]
    S = r["h"][w:] + sA ** 2 + r["b"][w:] * sph ** 2
    nu = innovation_stats(r["e"][w:] / np.sqrt(S), 20)
    return {"g_ugal": g_sign * (ph - ph.mean()) * g_per_rad(T) * UGAL,
            "ph": ph, "Kz": r["x"][w:, 3],
            "sd_Kz": np.sqrt(np.maximum(r["p"][w:, 1], 0.0)),
            "J": J, "nu_std": nu["std"], "lb_p": nu["lb_pvalue"], "i_first": poi + w, "e_rms": float(np.sqrt(np.mean(r["e"][w:] ** 2))),}

def plot_kz_track(t_s, kz, sd, kz0, branch, kz0_label, kz_s=None):
    """Kz(t), найденный фильтром Калмана, с полосой ±1 sigma; Kz0 -- пунктиром."""
    import matplotlib.pyplot as plt

    div, unit = _time_unit(float(t_s[-1]))
    t = np.asarray(t_s) / div
    col = BRANCH_COLORS[branch]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t, kz - sd, kz + sd, color=col, alpha=0.2, lw=0, label="±1σ (фильтр)")
    ax.plot(t, kz, "-", lw=1.0, color=col, label=f"Kz(t), фильтр Калмана, чирп {branch}")
    if kz_s is not None:
        ax.plot(t, kz_s, "-", lw=1.2, color="red", zorder=5,
                label="Kz(t): фильтр + обратный проход (RTS)")
    ax.axhline(kz0, color="gray", ls="--", lw=1, label=kz0_label)
    ax.set_xlabel(f"Время, {unit}")
    ax.set_ylabel("Kz")
    ax.set_title(f"Kz как параметр фильтра Калмана, чирп {branch}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()


    

def plot_kz_compare(t_k, kz, sd, res_blocks, branch, kz0, t0, kz0_label,
                    kz_v=None, kz_s=None):
    """
    Kz(t): фильтр Калмана (линия, ±1σ), опционально фильтр со скоростью kz_v и
    сглаженный (RTS) kz_s, Kz по блокам (EKF и оконный cos-fit), Kz0 (звезда в t0).
    """
    import matplotlib.pyplot as plt

    tb_s = np.asarray(res_blocks["t_s"], dtype=float)
    div, unit = _time_unit(max(float(t_k[-1]), float(np.nanmax(tb_s))))
    col = BRANCH_COLORS[branch]
    bs = res_blocks["block_size"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(t_k / div, kz - sd, kz + sd, color=col, alpha=0.2, lw=0,
                    label="±1σ (фильтр)")
    ax.plot(t_k / div, kz, "-", lw=1.0, color=col,
            label=f"Kz(t): фильтр Калмана (Kz в состоянии), чирп {branch}")
    if kz_v is not None:
        ax.plot(t_k / div, kz_v, "-", lw=1.0, color="tab:purple",
                label="Kz(t): фильтр Калмана (Kz + скорость dKz/dt)")
    if kz_s is not None:
        ax.plot(t_k / div, kz_s, "-", lw=1.2, color="red", zorder=5,
                label="Kz(t): фильтр Калмана + обратный проход (RTS)")
    tb = tb_s / div
    ax.plot(tb, res_blocks["Kz_kalman"], "o-", ms=4, color="k",
            label=f"по блокам ({bs} сбросов): EKF (NLL инноваций)")
    ax.plot(tb, res_blocks["Kz_win"], "s--", ms=4, color="tab:green", alpha=0.7,
            label="по блокам: оконный cos-fit")
    ax.plot([t0 / div], [kz0], "*", ms=14, color="tab:orange", label=kz0_label)
    ax.set_xlabel(f"Время, {unit}")
    ax.set_ylabel("Kz")
    ax.set_title(f"Kz: фильтр Калмана и подбор по блокам, чирп {branch}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

def estimate_lag(fwd, smo, max_lag=20000, step=100):
    """
    Лаг прямого ряда относительно сглаженного (в циклах) по максимуму
    кросс-корреляции. Положительный лаг: прямой фильтр ЗАПАЗДЫВАЕТ.
    Возвращает (лаг, коэффициент корреляции в максимуме).
    """
    a = fwd - fwd.mean()
    b = smo - smo.mean()
    best_l, best_c = 0, -2.0
    for L in range(-max_lag, max_lag + 1, step):
        if L >= 0:
            x, y_ = a[L:], b[:len(b) - L]
        else:
            x, y_ = a[:L], b[-L:]
        c = float(x @ y_) / np.sqrt(float(x @ x) * float(y_ @ y_))
        if c > best_c:
            best_l, best_c = L, c
    return best_l, best_c


def kalman_report(fin3, fin4, fin4s, fin5, Fz, prm, t_k, win=1000, skip=1000):
    """
    Сводка «как отработал фильтр с Kz»:
      1) качество описания данных (J, RMS инноваций, белизна инноваций);
      2) значимость и форма Kz(t), запаздывание прямого фильтра;
      3) эффект Kz(t) на g относительно Kz = const;
      4) масштаб ошибки g от неопределённости Kz.
    win -- размер окна усреднения для оценки эффекта на g; skip -- сколько
    первых циклов исключить из статистик Kz (переходный процесс).
    """
    cnt = len(fin4["g_ugal"])
    i0 = fin4["i_first"]
    fz = np.asarray(Fz[i0:i0 + cnt], dtype=float)
    dt = float((t_k[-1] - t_k[0]) / (len(t_k) - 1))
    print(f"\n{'=' * 72}\nОтчёт по фильтру Калмана (циклов: {cnt}, шаг {dt:.3f} с)\n{'=' * 72}")

    # --- 1) качество ---
    print("1) Описание данных (чем меньше J и RMS(e), тем лучше; std(nu) ~ 1, LBp > 0.05):")
    rows = [("Kz фиксирован   ", fin3), ("Kz в состоянии  ", fin4)]
    if fin5 is not None:
        rows.append(("Kz + скорость   ", fin5))
    for name, f in rows:
        print(f"   {name}: J={f['J']:.5f}  Δ(-2lnL) к фикс. = {cnt * (fin3['J'] - f['J']):+8.1f}  "
              f"RMS(e)={f['e_rms']:.5f} ({f['e_rms'] / fin3['e_rms'] - 1:+.2%})  "
              f"std(nu)={f['nu_std']:.3f}  LBp={f['lb_p']:.3f}")

    # --- 2) Kz(t) ---
    kz, sd = fin4["Kz"][skip:], fin4["sd_Kz"][skip:]
    print(f"\n2) Kz(t) без первых {skip} циклов:")
    print(f"   прямой:    mean={kz.mean():.4e}  std={kz.std():.2e}  "
          f"p2p={np.ptp(kz):.2e}  средняя sd={sd.mean():.2e}")
    if fin4s is not None:
        ks, sds = fin4s["Kz"][skip:], fin4s["sd_Kz"][skip:]
        snr = ks.std() / sds.mean()
        print(f"   сглажен.:  mean={ks.mean():.4e}  std={ks.std():.2e}  "
              f"p2p={np.ptp(ks):.2e}  средняя sd={sds.mean():.2e}")
        print(f"   std(Kz сглаж.) / sd(Kz сглаж.) = {snr:.1f}  "
              f"(≲1 -- дрейф на уровне шума оценки; ≫1 -- дрейф реален)")
        lag, c = estimate_lag(fin4["Kz"][skip:], fin4s["Kz"][skip:])
        print(f"   запаздывание прямого фильтра относительно сглаженного: {lag} циклов "
              f"({lag * dt / 60:.1f} мин), корреляция {c:.3f}")

    # --- 3) эффект на g ---
    m = cnt // win
    gb3 = fin3["g_ugal"][:m * win].reshape(m, win).mean(axis=1)
    noise = float(np.std(np.diff(gb3)) / np.sqrt(2))   # шум средних по окну (без приливов)
    print(f"\n3) Эффект Kz(t) на g относительно Kz = const (окно {win} циклов, "
          f"шум средних по окну ~ {noise:.2f} мкГал):")
    th = np.asarray(t_k, dtype=float) / 3600.0
    variants = [("Kz в состоянии ", fin4)]
    if fin4s is not None:
        variants.append(("Kz + RTS        ", fin4s))
    if fin5 is not None:
        variants.append(("Kz + скорость  ", fin5))
    for name, f in variants:
        d = f["g_ugal"] - fin3["g_ugal"]
        db = d[:m * win].reshape(m, win).mean(axis=1)
        slope = np.polyfit(th, d, 1)[0]
        print(f"   {name}: std(Δg по циклам)={d.std():.3g}  std(Δg по окнам)={db.std():.3g}  "
              f"p2p(окна)={np.ptp(db):.3g} мкГал; тренд {slope:+.3g} мкГал/ч; "
              f"std(окна)/шум = {db.std() / noise:.1f}")

    # --- 4) масштаб ошибки от Kz ---
    k = g_per_rad(prm.T) * UGAL
    print(f"\n4) std(Fz) = {fz.std():.3g}, mean(Fz) = {fz.mean():.3g}; "
          f"sd(Kz) {sd.mean():.1e} × std(Fz) ≈ {sd.mean() * fz.std() * k:.3g} мкГал "
          f"(грубый масштаб по циклам; постоянный сдвиг Kz действует через mean(Fz))")

def _adev_at(m, adev, m_target):
    """Значение девиации Аллана в узле, ближайшем к m_target циклов."""
    i = int(np.argmin(np.abs(np.log(m) - np.log(m_target))))
    return int(m[i]), float(adev[i])

def _adev_interp(m, adev, m_t):
    """Девиация Аллана при m_t циклов (лог-лог интерполяция); NaN вне диапазона."""
    if m_t < m[0] or m_t > m[-1]:
        return np.nan
    return float(np.exp(np.interp(np.log(m_t), np.log(m), np.log(adev))))


def plot_allan(curves, t_cycle, title="Девиация Аллана g"):
    """
    curves: список (подпись, (m, adev, err), S [мкГал/sqrt(цикл)], цвет, маркер).
    Пунктир -- белый шум S/sqrt(m). Нижняя ось -- циклы, верхняя -- секунды.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for lab, (m, adev, err), S, col, mk in curves:
        ax.errorbar(m, adev, yerr=err, fmt=mk + "-", ms=4, lw=1, capsize=2, color=col,
                    label=f"{lab}: {S:.3g} мкГал/√цикл ({S * np.sqrt(t_cycle):.3g} мкГал/√Гц)")
        mm = np.array([m[0], m[-1]], dtype=float)
        ax.plot(mm, S / np.sqrt(mm), ":", color=col, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число усредняемых циклов m")
    ax.set_ylabel("Девиация Аллана g, мкГал")
    sec = ax.secondary_xaxis("top", functions=(lambda m: m * t_cycle, lambda s: s / t_cycle))
    sec.set_xlabel("τ, с")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()


def kz_segments(alp, y, Fz, t, init, prm, g_sign, n_seg,
                bounds=(-6.5e-3, -4.0e-3), h=5e-5, verbose=True):
    """
    Kz по n_seg сегментам при ФИКСИРОВАННЫХ sigma и q (init), без PSO.
    sd сегмента -- из кривизны J(Kz): var = 2 / (cnt * J''), т.к. -2lnL = cnt*J.
    Тесты: (1) χ² на постоянство Kz; (2) тренд (взвешенная линейная регрессия);
    (3) χ² по разностям СОСЕДНИХ сегментов -- калибровка реальной погрешности.
    """
    edges = np.linspace(0, len(alp), n_seg + 1).astype(int)
    ks, sds, th = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        def f(k, a=a, b=b):
            try:
                return final_kalman_run(alp[a:b], y[a:b], Fz[a:b],
                                        dict(init, Kz=k), prm, g_sign)["J"]
            except Exception:
                return BIG_OBJ
        k0 = float(minimize_scalar(f, bounds=bounds, method="bounded",
                                   options=dict(xatol=1e-8)).x)
        d2 = (f(k0 + h) - 2 * f(k0) + f(k0 - h)) / h ** 2
        cnt = (b - a) - prm.poi - prm.warmup
        ks.append(k0)
        sds.append(np.sqrt(2.0 / (cnt * d2)) if d2 > 0 else np.nan)
        th.append(float(np.mean(t[a:b])) / 3600.0)
    ks, sds, th = np.array(ks), np.array(sds), np.array(th)
    ok = np.isfinite(sds)
    ks, sds, th = ks[ok], sds[ok], th[ok]
    n = len(ks)
    dof = n - 1

    # (1) постоянство
    w = 1.0 / sds ** 2
    kbar = float(np.sum(w * ks) / np.sum(w))
    chi_c = float(np.sum(w * (ks - kbar) ** 2))
    # (2) тренд
    pc, cov = np.polyfit(th, ks, 1, w=1.0 / sds, cov="unscaled")
    chi_l = float(np.sum(((ks - np.polyval(pc, th)) / sds) ** 2))
    d_chi = chi_c - chi_l
    slope, slope_sd = pc[0], float(np.sqrt(cov[0, 0]))
    # (3) соседние сегменты
    dif = np.diff(ks)
    chi_a = float(np.sum(dif ** 2 / (sds[:-1] ** 2 + sds[1:] ** 2)))

    if verbose:
        print(f"\n--- Kz по {n} сегментам (≈{(edges[1] - edges[0])} циклов, "
              f"sigma и q фиксированы) ---")
        print("  t, ч :  " + " ".join(f"{v:9.2f}" for v in th))
        print("  Kz   :  " + " ".join(f"{v * 1e3:9.4f}" for v in ks) + "   (×1e-3)")
        print(f"  формальная sd сегмента: {sds.mean():.2e};  std(Kz сегментов) = {ks.std(ddof=1):.2e}")
        print(f"  (1) постоянство : χ² = {chi_c:.1f}, ст. своб. {dof}, χ²/dof = {chi_c / dof:.2f}, "
              f"p = {chi2_dist.sf(chi_c, dof):.3g}")
        print(f"  (2) тренд       : {slope * 1e6:+.2f} ± {slope_sd * 1e6:.2f} (×1e-6 /ч) "
              f"({slope / slope_sd:+.1f}σ); Δχ² = {d_chi:.1f} при 1 ст. своб. "
              f"(p = {chi2_dist.sf(max(d_chi, 0), 1):.3g}); "
              f"размах по записи {slope * (th[-1] - th[0]) * 1e5:+.1f}e-5")
        print(f"  (3) соседние    : χ²/dof = {chi_a / dof:.2f}  "
              f"-> реальная sd ≈ формальная × {np.sqrt(chi_a / dof):.2f}")
        print(f"  после поправки на (3): χ²(постоянство)/dof = "
              f"{chi_c / dof / (chi_a / dof):.2f}")
    return {"t_h": th, "Kz": ks, "sd": sds, "chi_const": chi_c, "dof": dof,
            "chi_adj": chi_a, "slope": slope, "slope_sd": slope_sd}


# =====================================================================
# main
# =====================================================================

def main():
    # ---------------- данные ----------------
    DATA_PATH = r"long_data\data.txt"   # date time norm+ norm- scan+ scan- corr+ corr- offset
    BRANCH = "+"               # какую ветку чирпа обрабатывать: "+" или "-"
    TABLE_ROWS = (0, 127000)
    T_CYCLE_NOMINAL = 0.175    # номинальный период цикла, с; None -- взять по файлу
    SHOW_PLOT = True
    G_SIGN = {"+": +1.0, "-": -1.0}   # только направление оси g

    T = 12e-3                  # длительность плеча, с

    # ---------------- начальный Kz: по первым N_INIT точкам (один блок) ----------------
    N_INIT = 1000
    KZ_BOUNDS = (-1.0, 1.0)
    KZ_INIT_SRC = "kalman"     # откуда Kz0: "kalman" (NLL инноваций) | "window" (оконный cos-fit)
    N_PARTICLES = 10
    N_ITER = 40
    PSO_XTOL = 1e-7
    SEED = 0

    # ---------------- EKF ----------------
    DA_MODEL = DB_MODEL = DPH_MODEL = 1e-3   # начальные std шагов [A, B, ph]
    POI = 200                  # точек начального фита
    WARMUP = 200               # инноваций, не входящих в fitness и в ряды g, Kz
    WIN_SIZE = 20

    # ---------------- Kz как состояние ----------------
    KZ_SIGMA0 = 3e-4           # начальная std Kz (неопределённость Kz0)
    Q_KZ_MODE = "tune"         # "tune" -- q_Kz по NLL на всём ряду; "manual" -- из DKZ_MODEL
    DKZ_MODEL = 1e-6           # std шага Kz за цикл (для "manual")
    Q_KZ_LOG_BOUNDS = (-18.0, -9.0)   # границы log10 q_Kz для "tune"
    DO_KZ_VEL = False                 # второй вариант фильтра: Kz + скорость dKz/цикл
    KZV_SIGMA0 = 1e-8                   # начальная std скорости, 1/цикл
    Q_KZV_LOG_BOUNDS = (-26.0, -14.0)   # границы log10 q_a (интенсивность «ускорения» Kz)
    # ---------------- Kz по блокам (прежний метод, для сравнения на графике) ----------------
    DO_BLOCKS = True           # False -- не считать блоки (график только с фильтром)
    BLOCK_SIZE = 9000          # сбросов в блоке; остаток отбрасывается
    MAX_BLOCKS = None          # None -- все блоки
    N_JOBS = None              # None -- все ядра
    DO_KZ_SMOOTH = True   
    RECALIB_GLOBAL = True      # sigma и q по всему ряду (а не по первым N_INIT точкам)

    # ---------------- сравнение с фринджем ----------------
    DO_ALLAN = True
    FRINGE_NPTS = 6
    FRINGE_KZ_MODE = "init"    # Kz для фринджа и для EKF без Kz в состоянии:
                               #   "init" -- Kz0; "median" -- медиана Kz(t) фильтра;
                               #   "track" -- Kz(t) фильтра по циклам (только для фринджа)
    ALLAN_FIT_M = (6, 60)

    if BRANCH not in ("+", "-"):
        raise ValueError(f"BRANCH: '+' | '-', получено {BRANCH!r}")
    if KZ_INIT_SRC not in ("kalman", "window"):
        raise ValueError(f"KZ_INIT_SRC: 'kalman' | 'window', получено {KZ_INIT_SRC!r}")
    if Q_KZ_MODE not in ("tune", "manual"):
        raise ValueError(f"Q_KZ_MODE: 'tune' | 'manual', получено {Q_KZ_MODE!r}")
    if FRINGE_KZ_MODE not in ("init", "median", "track"):
        raise ValueError(f"FRINGE_KZ_MODE: 'init' | 'median' | 'track', получено {FRINGE_KZ_MODE!r}")

    tune_cfg = TuneCfg(sigma_init_mode="bins",
                       tune_mode="alternate",
                       r_method="fisher",
                       q_optimizer="lbfgs",
                       n_outer=6,
                       q_free=(True, True, True),
                       q_log_bounds=(-13.0, -4.0),
                       lbfgs_maxfun=200,
                       )

    # ---------------- загрузка ----------------
    (alp, y, corr), t_cycle, tinfo = load_table_pm(
        DATA_PATH, TABLE_ROWS, branch=BRANCH, t_cycle_nominal=T_CYCLE_NOMINAL)
    t_all = tinfo["t"]
    Fz = corr                  # бины, как есть
    g_sign = G_SIGN[BRANCH]
    n = len(alp)
    npts = FRINGE_NPTS

    # ================= шаг 1: Kz0 и параметры EKF по первым N_INIT точкам =================
    print(f"\n{'=' * 72}\nШаг 1: Kz0 по первым {N_INIT} точкам (чирп {BRANCH})\n{'=' * 72}")
    init_res, prm = find_initial_kz(
        alp, y, Fz, n_init=N_INIT, T=T, Kz_bounds=KZ_BOUNDS,
        dA_model=DA_MODEL, dB_model=DB_MODEL, dph_model=DPH_MODEL,
        poi=POI, warmup=WARMUP, win_size=WIN_SIZE, tune_cfg=tune_cfg,
        n_particles=N_PARTICLES, n_iter=N_ITER, pso_xtol=PSO_XTOL, seed=SEED)
    _print_block(init_res, 1, 0, N_INIT, float(np.mean(t_all[:N_INIT])), f"[{BRANCH}] ")
    if not np.isfinite(init_res["Kz_kalman"]):
        raise RuntimeError("EKF по первым точкам не сошёлся: нет sigma и q для фильтра")
    kz_src = init_res["Kz_kalman"] if KZ_INIT_SRC == "kalman" else init_res["Kz_win"]
    if not np.isfinite(kz_src):
        raise RuntimeError(f"Kz0 ({KZ_INIT_SRC}) не найден")
    Kz0 = float(kz_src)
    init = {"Kz": Kz0, "sigma_A": init_res["sigma_A"], "sigma_ph": init_res["sigma_ph"],
            "q": np.array([init_res["qA"], init_res["qB"], init_res["qph"]])}
    print(f"  Kz0 = {Kz0:.4e} (источник: {KZ_INIT_SRC}); "
          f"Kz(EKF) = {init_res['Kz_kalman']:.4e}, Kz(окно) = {init_res['Kz_win']:.4e}")

    # ================= глобальная перекалибровка R и q по всему ряду =================
    if RECALIB_GLOBAL:
        print(f"\n{'-' * 72}\nПерекалибровка sigma и q по всему ряду ({n} циклов), Kz = Kz0\n{'-' * 72}")
        t0_rc = time.perf_counter()
        J_rc, info_rc = tune_and_score(Fz, Kz0, alp, y, prm, tune_cfg, diagnostics=True)
        if not info_rc:
            raise RuntimeError("перекалибровка по всему ряду не сошлась")
        q_old = init["q"].copy()
        print(f"  sigma_A : {init['sigma_A']:.3e} -> {info_rc['sigma_A']:.3e}")
        print(f"  sigma_ph: {init['sigma_ph']:.3e} -> {info_rc['sigma_ph']:.3e} "
              f"({info_rc['dg_sigma_ph'] * UGAL:.3g} мкГал на сброс)")
        lo_b, hi_b = tune_cfg.q_log_bounds
        for nm, qo, qn in zip(Q_NAMES, q_old, info_rc["q"]):
            print(f"  q_{nm:<2s}    : {qo:.3e} -> {qn:.3e}  "
                  f"(log10 = {np.log10(qn):.2f}, границы [{lo_b:g}, {hi_b:g}])")
        print(f"  J = {J_rc:.5f}, std(nu) = {info_rc['nu_stats']['std']:.3f}, "
              f"LBp = {info_rc['nu_stats']['lb_pvalue']:.3f}, "
              f"q у границы: {','.join(info_rc['q_at_bound']) or 'нет'}, "
              f"{info_rc['n_it']} внешних итераций, {time.perf_counter() - t0_rc:.1f} с")
        init["sigma_A"] = info_rc["sigma_A"]
        init["sigma_ph"] = info_rc["sigma_ph"]
        init["q"] = np.asarray(info_rc["q"], dtype=np.float64)

        # стационарность: std(nu) по четвертям ряда при ФИКСИРОВАННЫХ параметрах
        w_ = prm.warmup
        a_c = compensate_alp(alp, Fz, Kz0, T)
        x03, P03, _ = initial_fit(a_c[:prm.poi], y[:prm.poi], T)
        r_ = run_ekf(np.ascontiguousarray(a_c[prm.poi:]), np.ascontiguousarray(y[prm.poi:]),
                     x03, P03, init["q"], init["sigma_A"], init["sigma_ph"], T, w_)
        S_ = r_["h"][w_:] + init["sigma_A"] ** 2 + r_["b"][w_:] * init["sigma_ph"] ** 2
        nu_ = r_["e"][w_:] / np.sqrt(S_)
        parts = np.array_split(nu_, 4)
        print("  std(nu) по четвертям ряда: " + ", ".join(f"{p.std(ddof=1):.3f}" for p in parts)
              + "  (≈1 везде -- параметры стационарны; тренд -- шум меняется)")


    # ================= шаг 2: EKF с ФИКСИРОВАННЫМ Kz0 (для сравнения) =================
    fin3 = final_kalman_run(alp, y, Fz, init, prm, g_sign)
    print(f"\nEKF, Kz фиксирован = {Kz0:.4e}: J={fin3['J']:.5f}, "
          f"std(nu)={fin3['nu_std']:.3f}, LBp={fin3['lb_p']:.3f}")

        
    for n_seg in (7, 14, 25):
        kz_segments(alp, y, Fz, t_all, init, prm, g_sign, n_seg)

    # ================= шаг 3: EKF с Kz в состоянии =================
    print(f"\n{'=' * 72}\nШаг 3: фильтр Калмана с Kz в состоянии (весь ряд, {n} циклов)\n{'=' * 72}")
    x0, P0 = kz_filter_start(alp, y, Fz, Kz0, prm, KZ_SIGMA0)
    if Q_KZ_MODE == "tune":
        lq, J_best, lqs, Js = tune_q_kz(alp, y, Fz, x0, P0, init["q"],
                                        init["sigma_A"], init["sigma_ph"], prm,
                                        Q_KZ_LOG_BOUNDS)
        cnt = n - prm.poi - prm.warmup
        print(f"  q_Kz подобран: log10 q_Kz = {lq:.2f} (std шага {10 ** (lq / 2):.2e} за цикл), "
              f"J = {J_best:.5f}")
        print(f"  Kz почти постоянен (q_Kz на нижней границе): J = {Js[0]:.5f}; "
              f"выигрыш от дрейфа Δ(-2lnL) = {cnt * (Js[0] - J_best):.1f} "
              f"(≳10 -- дрейф Kz значим)")
    else:
        lq = 2 * np.log10(DKZ_MODEL)
        print(f"  q_Kz задан: std шага {DKZ_MODEL:.2e} за цикл")
    qd = np.append(init["q"], 10.0 ** lq)
    fin4 = kz_filter_run(alp, y, Fz, x0, P0, qd, init["sigma_A"], init["sigma_ph"],
                         prm, g_sign)
    kz_t = fin4["Kz"]
    Kz_glob = float(np.median(kz_t))
    fin3g = final_kalman_run(alp, y, Fz, dict(init, Kz=Kz_glob), prm, g_sign)
    print(f"\nEKF, Kz глобальный = {Kz_glob:.4e}: J={fin3g['J']:.5f}, "
          f"std g {fin3g['g_ugal'].std():.3g} мкГал (Kz0: {fin3['g_ugal'].std():.3g}, "
          f"Kz в состоянии: {fin4['g_ugal'].std():.3g})")
    print(f"  J={fin4['J']:.5f}, std(nu)={fin4['nu_std']:.3f}, LBp={fin4['lb_p']:.3f}")
    print(f"  Kz(t): начало {kz_t[0]:.4e}, конец {kz_t[-1]:.4e}, mean {kz_t.mean():.4e}, "
          f"std {kz_t.std():.2e}, min {kz_t.min():.4e}, max {kz_t.max():.4e}; "
          f"средняя sd фильтра {fin4['sd_Kz'].mean():.2e}")
    

    fin4s = None
    if DO_KZ_SMOOTH:
        fin4s = kz_filter_smooth(alp, y, Fz, x0, P0, qd, init["sigma_A"],
                                 init["sigma_ph"], prm, g_sign)
        d = fin4s["Kz"] - fin4s["Kz_fwd"]
        print(f"\n  Обратный проход (RTS): Kz сглаж. mean={fin4s['Kz'].mean():.4e}, "
              f"std={fin4s['Kz'].std():.2e}; средняя sd {fin4s['sd_Kz'].mean():.2e} "
              f"(прямой: {fin4['sd_Kz'].mean():.2e}); "
              f"std(сглаж. − прямой) = {d.std():.2e}")

    # ================= эффект кусочного Kz на g =================
    n_seg_g = 14
    seg = kz_segments(alp, y, Fz, t_all, init, prm, g_sign, n_seg_g, verbose=False)
    if len(seg["Kz"]) == n_seg_g:
        edges = np.linspace(0, n, n_seg_g + 1).astype(int)
        kz_pw = np.repeat(seg["Kz"], np.diff(edges))           # длина n
        fin_pw = final_kalman_run(alp, y, Fz, dict(init, Kz=kz_pw), prm, g_sign)
        fin_gl = final_kalman_run(alp, y, Fz, dict(init, Kz=Kz_glob), prm, g_sign)
        d = fin_pw["g_ugal"] - fin_gl["g_ugal"]                # индекс 0 <-> цикл i_first
        i0 = fin_gl["i_first"]

        db, gb = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            a_, b_ = max(a, i0) - i0, b - i0
            if b_ - a_ > 1000:
                db.append(d[a_:b_].mean())
                gb.append(fin_gl["g_ugal"][a_:b_].mean())
        db, gb = np.array(db), np.array(gb)
        noise_g = float(np.std(np.diff(gb)) / np.sqrt(2))      # шум средних по сегменту
        print(f"\nΔg (кусочный Kz − глобальный Kz = {Kz_glob:.4e}), {len(db)} сегментов:")
        print(f"  std = {db.std():.3g}, p2p = {np.ptp(db):.3g} мкГал; "
              f"шум g в тех же сегментах ≈ {noise_g:.3g} мкГал; "
              f"std(Δg)/шум = {db.std() / noise_g:.2f}")

    fin5 = None
    if DO_KZ_VEL:
        x0v, P0v = kzv_start(x0, P0, KZV_SIGMA0)
        lqv, Jv, lqsv, Jsv = tune_q_kzv(alp, y, Fz, x0v, P0v, init["q"],
                                        init["sigma_A"], init["sigma_ph"], prm,
                                        Q_KZV_LOG_BOUNDS)
        fin5 = kzv_filter_run(alp, y, Fz, x0v, P0v, init["q"], 10.0 ** lqv,
                              init["sigma_A"], init["sigma_ph"], prm, g_sign)
        cnt = n - prm.poi - prm.warmup
        print(f"\n  Kz + скорость: log10 q_a = {lqv:.2f}, J = {fin5['J']:.5f}, "
              f"std(nu)={fin5['nu_std']:.3f}; "
              f"Δ(-2lnL) относительно фильтра без скорости = {cnt * (fin4['J'] - fin5['J']):+.1f} "
              f"(скорость оправдана, если заметно больше ~5)")
        if min(lqv - Q_KZV_LOG_BOUNDS[0], Q_KZV_LOG_BOUNDS[1] - lqv) < 0.1:
            print("  ВНИМАНИЕ: log10 q_a у границы Q_KZV_LOG_BOUNDS")

    i_first = fin4["i_first"]
    t_k = t_all[i_first + np.arange(len(kz_t))]

    # ================= фринж по npts точкам (без фильтра) =================
    if FRINGE_KZ_MODE == "init":
        kz_f = Kz0
    elif FRINGE_KZ_MODE == "median":
        kz_f = float(np.median(kz_t))
    else:
        kz_f = np.full(n, Kz0)
        kz_f[i_first:] = kz_t
    fr = fringe_npoint(alp, y, Fz, kz_f, T, npts, i_first, n, g_sign)
    print(f"\nФринж по {npts} точкам (Kz: {FRINGE_KZ_MODE}): групп {fr['n_groups']}, "
          f"групп с полной лестницей: {np.mean(fr['n_distinct'] == npts):.1%}")

    g3, g4, gf = fin3["g_ugal"], fin4["g_ugal"], fr["g_ugal"]
    print(f"std g: EKF (Kz фикс.) {g3.std():.3g}, EKF (Kz в состоянии) {g4.std():.3g} мкГал (по циклам); "
          f"фринж {gf.std():.3g} мкГал ({npts}-точечные группы)")
    kalman_report(fin3, fin4, fin4s, fin5, Fz, prm, t_k)

    # ================= девиация Аллана: EKF (Kz = const) против фринджа =================
    allan = None
    if DO_ALLAN:
        Kz_mean = float(np.mean(kz_t[1000:]))        # средний Kz по треку фильтра
        fin_c = final_kalman_run(alp, y, Fz, dict(init, Kz=Kz_mean), prm, g_sign)
        fr_c = fringe_npoint(alp, y, Fz, Kz_mean, T, npts, fin_c["i_first"], n, g_sign)

        m_e, a_e, e_e = adev_in_cycles(fin_c["g_ugal"], 1)       # 1 отсчёт = 1 цикл
        m_f, a_f, e_f = adev_in_cycles(fr_c["g_ugal"], npts)     # 1 отсчёт = npts циклов
        S_e = white_noise_sensitivity(m_e, a_e, *ALLAN_FIT_M)
        S_f = white_noise_sensitivity(m_f, a_f, *ALLAN_FIT_M)

        print(f"\nДевиация Аллана, Kz = {Kz_mean:.4e} (среднее по треку фильтра)")
        print(f"  чувствительность (m = {ALLAN_FIT_M[0]}..{ALLAN_FIT_M[1]}): "
              f"EKF {S_e:.3g}, фринж {S_f:.3g} мкГал/√цикл "
              f"({S_e * np.sqrt(t_cycle):.3g} и {S_f * np.sqrt(t_cycle):.3g} мкГал/√Гц)")
        print("  m, циклов :   EKF, мкГал   фринж, мкГал   фринж/EKF")
        for mt in (npts, 60, 600, 6000, 20000):
            ae, af = _adev_interp(m_e, a_e, mt), _adev_interp(m_f, a_f, mt)
            print(f"  {mt:9d} : {ae:11.3g} {af:14.3g} {af / ae:11.2f}")
        allan = [("EKF (Kz = const)", (m_e, a_e, e_e), S_e, "tab:blue", "o"),
                 (f"фринж по {npts} точкам", (m_f, a_f, e_f), S_f, "tab:red", "s")]

    # ================= Kz по блокам (прежний метод) =================
    res_blocks = None
    if DO_BLOCKS:
        print(f"\n{'=' * 72}\nKz по блокам ({BLOCK_SIZE} сбросов), чирп {BRANCH}\n{'=' * 72}")
        res_blocks = run_kz_dynamics(
            alp, y, Fz, T=T, Kz_bounds=KZ_BOUNDS,
            block_size=BLOCK_SIZE, max_blocks=MAX_BLOCKS,
            t_cycle=t_cycle, t_axis=t_all,
            dA_model=DA_MODEL, dB_model=DB_MODEL, dph_model=DPH_MODEL,
            poi=POI, warmup=WARMUP, win_size=WIN_SIZE,
            tune_cfg=tune_cfg,
            n_particles=N_PARTICLES, n_iter=N_ITER, pso_xtol=PSO_XTOL,
            n_jobs=N_JOBS, seed=SEED, tag=f"[{BRANCH}] ")
        print_summary(res_blocks, f"(чирп {BRANCH}) ")

        # Kz фильтра, усреднённый по тем же блокам, против Kz блока
        filters = [("без скорости", kz_t)]
        if fin5 is not None:
            filters.append(("со скоростью", fin5["Kz"]))
        for flab, kzs in filters:
            kz_f_blk = np.full(len(res_blocks["Kz_kalman"]), np.nan)
            for k, (i0, i1) in enumerate(zip(res_blocks["i_start"], res_blocks["i_stop"])):
                lo = max(int(i0), i_first)
                if i1 > lo:
                    kz_f_blk[k] = kzs[lo - i_first:int(i1) - i_first].mean()
            for name, key in (("EKF", "Kz_kalman"), ("окно", "Kz_win")):
                d = kz_f_blk - res_blocks[key]
                d = d[np.isfinite(d)]
                if d.size > 1:
                    print(f"  Kz фильтра {flab} (среднее по блоку) − Kz блока [{name}]: "
                          f"mean={d.mean():+.2e}, std={d.std(ddof=1):.2e}")


    # ================= график: Kz(t) =================
    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        kz0_label = f"Kz0 по первым {N_INIT} точкам = {Kz0:.4e}"
        if res_blocks is not None:
            plot_kz_compare(t_k, kz_t, fin4["sd_Kz"], res_blocks, BRANCH, Kz0,
                            float(np.mean(t_all[:N_INIT])), kz0_label,
                            kz_v=None if fin5 is None else fin5["Kz"],
                            kz_s=None if fin4s is None else fin4s["Kz"])
        else:
            plot_kz_track(t_k, kz_t, fin4["sd_Kz"], Kz0, BRANCH, kz0_label,
                          kz_s=None if fin4s is None else fin4s["Kz"])
        
        if allan is not None:
            plot_allan(allan, t_cycle, f"Девиация Аллана g, чирп {BRANCH}, Kz = const")
        
        plt.show()

if __name__ == "__main__":
    main()
