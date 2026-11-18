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

Результат: графики Kz(t) (обе ветки, оба критерия), g(t) (EKF и фринж на одном
графике), девиация Аллана g по числу циклов, график чувствительности и эффекта
Kz(t), CSV с таблицами Kz по блокам и с рядами g.

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
from scipy.optimize import minimize, nnls
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
# 13. Печать, CSV, график Kz
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


CSV_COLUMNS = ["block", "i_start", "i_stop", "t_s", "Kz_kalman", "J", "sigma_A",
               "sigma_ph", "dg_sigma_ph_ugal", "nu_std", "lb_p", "Kz_win", "rms",
               "q_A", "q_B", "q_ph"]


def save_csv(res, path):
    """Таблица результатов по блокам (колонки CSV_COLUMNS)."""
    arr = np.column_stack([np.asarray(res[c], dtype=float) for c in CSV_COLUMNS])
    np.savetxt(path, arr, delimiter=",", header=",".join(CSV_COLUMNS),
               comments="", fmt="%.10g")


def _time_unit(tmax):
    """Делитель и подпись единицы времени для оси x графиков."""
    if tmax >= 2 * 3600:
        return 3600.0, "ч"
    if tmax >= 2 * 60:
        return 60.0, "мин"
    return 1.0, "с"


BRANCH_COLORS = {"+": "tab:blue", "-": "tab:red"}


def plot_kz_dynamics(branches, kz_ref=None, out_png=None, show=True):
    """
    График Kz(t) для обеих веток чирпа: EKF и оконный cos-fit.
    branches: {"+": res_plus, "-": res_minus} -- результаты run_kz_dynamics.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmax = max(float(np.nanmax(np.asarray(r["t_s"], dtype=float)))
               for r in branches.values() if len(r["t_s"]))
    div, unit = _time_unit(tmax)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sg, res in branches.items():
        t = np.asarray(res["t_s"], dtype=float) / div
        col = BRANCH_COLORS[sg]
        ax.plot(t, res["Kz_kalman"], "o-", ms=4, color=col,
                label=f"EKF (NLL инноваций), чирп {sg}")
        ax.plot(t, res["Kz_win"], "s--", ms=4, color=col, alpha=0.55,
                label=f"оконный cos-fit, чирп {sg}")
    if kz_ref is not None:
        ax.axhline(kz_ref, color="gray", ls=":", lw=1, label=f"Kz = {kz_ref:g}")
    bs = next(iter(branches.values()))["block_size"]
    ax.set_xlabel(f"Время, {unit} (центр блока)")
    ax.set_ylabel("Kz")
    ax.set_title(f"Динамика Kz по блокам ({bs} сбросов)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    if not show:
        plt.close(fig)          # при show=True окно открывается общим plt.show() в main


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
    print(f"T={T * 1e3:g} мс, Kz_bounds={tuple(Kz_bounds)}, "
          f"PSO: {n_particles} частиц x {n_iter} итераций, "
          f"tune_mode={tcfg.tune_mode}, r_method={tcfg.r_method}, "
          f"q_optimizer={tcfg.q_optimizer}\n")

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


def load_table_pm(path, row_range=None, corr_max=60.0, spike_k=8.0,
                  t_cycle_nominal=None):
    """
    Читает таблицу data.txt: date time norm+ norm- scan+ scan- corr+ corr- offset
    (одна строка = один цикл, плюс и минус лежат в РАЗНЫХ колонках одной строки).
    Возвращает ОБЕ ветки сразу.

    row_range: (i0, i1) -- какие строки файла брать (None -- все). Нужен, потому что
    после перехода режима (сигнал сменил уровень и стал на порядок шумнее) данные
    для EKF непригодны, а это не выброс, а другой режим.
    Роли: alp = scan, P_exp = norm, corr = ускорение в бинах (масштаб и знак
    приводятся к F_z в main).
    Строка выбрасывается ЦЕЛИКОМ (для обоих знаков, чтобы ряды двух веток оставались
    согласованными по циклам), если |corr| > corr_max у любого знака или сигнал
    любого знака -- выброс (_spike_mask).

    t_cycle_nominal: номинальный период цикла, с (None -- брать по файлу).

    Возвращает (data, t_cycle, info):
      data = {"+": (alp, P_exp, corr), "-": (alp, P_exp, corr)};
      t_cycle -- номинальный период (если задан) либо средний по файлу, с;
      info["t"] -- реальное время оставшихся строк, с (от первой строки диапазона);
      info["t_cycle_file"] -- средний период по оставшимся строкам (включает
      пропуски из-за выброшенных строк).
    """
    meta = np.loadtxt(path, dtype=str, skiprows=1, usecols=(0, 1))
    num = np.loadtxt(path, skiprows=1, usecols=range(2, 9))
    t = np.char.add(np.char.add(meta[:, 0], "T"), meta[:, 1]).astype("datetime64[ms]")
    t = (t - t[0]) / np.timedelta64(1, "s")
    if row_range is not None:
        sl = slice(int(row_range[0]), int(row_range[1]))
        num, t = num[sl], t[sl]
    t = t - t[0]

    y_p, y_m, a_p, a_m, c_p, c_m = (num[:, i] for i in range(6))
    bad = (np.abs(c_p) > corr_max) | (np.abs(c_m) > corr_max)
    n_corr = int(bad.sum())
    bad |= _spike_mask(y_p, a_p, spike_k) | _spike_mask(y_m, a_m, spike_k)
    keep = ~bad

    def pick(*arrs):
        return tuple(np.ascontiguousarray(a[keep]) for a in arrs)

    data = {"+": pick(a_p, y_p, c_p), "-": pick(a_m, y_m, c_m)}
    t_keep = np.ascontiguousarray(t[keep])
    t_file = float((t_keep[-1] - t_keep[0]) / (keep.sum() - 1))
    t_cycle = float(t_cycle_nominal) if t_cycle_nominal else t_file
    info = {"n_total": len(bad), "n_bad": int(bad.sum()), "n_bad_corr": n_corr,
            "t": t_keep, "t_cycle_file": t_file}
    print(f"Таблица {path}: {len(bad)} строк, выброшено {info['n_bad']} "
          f"(из них |corr|>{corr_max:g}: {n_corr}); осталось {int(keep.sum())} циклов "
          f"в каждой ветке")
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
            "nu_std": nu["std"], "lb_p": nu["lb_pvalue"], "i_first": poi + w}


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


def plot_kz_effect(t_f, sens, effect, out_png=None, show=True):
    """
    Две панели: (1) dg/dKz по группам фринджа для веток и среднего (мкГал на ед. Kz);
    (2) эффект Kz(t) на g: g(Kz по блокам) - g(глобальный Kz), среднее по веткам.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    div, unit = _time_unit(float(t_f[-1]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(t_f / div, sens["+"], "-", lw=0.7, color=BRANCH_COLORS["+"], alpha=0.7, label="чирп +")
    ax.plot(t_f / div, sens["-"], "-", lw=0.7, color=BRANCH_COLORS["-"], alpha=0.7, label="чирп −")
    ax.plot(t_f / div, sens["mean"], "-", lw=1.2, color="k", label="среднее")
    ax.set_ylabel("dg/dKz, мкГал / ед. Kz")
    ax.set_title("Чувствительность g к Kz (фринж)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(t_f / div, effect, "-", lw=0.8, color="tab:green")
    ax.set_ylabel("Δg, мкГал")
    ax.set_xlabel(f"Время, {unit}")
    ax.set_title("Эффект Kz(t): g(Kz по блокам) − g(глобальный Kz), среднее по веткам")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    if not show:
        plt.close(fig)


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


def plot_allan_compare(curves, t_cycle, fit_m, out_png=None, show=True, title=""):
    """
    Девиация Аллана g (log-log, ось x -- число циклов) для нескольких рядов на одном графике.
    curves: список dict(label, m, adev, err, sens, color, [fmt], [alpha]).
    В подписи -- чувствительность каждого ряда.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    notes = []
    for c in curves:
        ax.errorbar(c["m"], c["adev"], yerr=c["err"], fmt=c.get("fmt", "o-"), ms=4,
                    capsize=3, color=c["color"], alpha=c.get("alpha", 1.0),
                    label=c["label"])
        mm = np.array([c["m"][0], c["m"][-1]], dtype=float)
        ax.plot(mm, c["sens"] / np.sqrt(mm), "--", lw=1, color=c["color"], alpha=0.7)
        notes.append(f"{c['label']}: {c['sens']:.3g} мкГал/√цикл "
                     f"≈ {c['sens'] * np.sqrt(t_cycle):.3g} мкГал/√Гц")
    ax.plot([], [], "k--", lw=1, label=r"белый шум $S/\sqrt{m}$")
    ax.axvspan(fit_m[0], fit_m[1], color="gray", alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число усредняемых циклов m")
    ax.set_ylabel("Девиация Аллана g, мкГал")
    ax.set_title(title or "Девиация Аллана g")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    ax.text(0.97, 0.96,
            "Чувствительность (период %.4g мс):\n" % (t_cycle * 1e3) + "\n".join(notes),
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    if not show:
        plt.close(fig)          # при show=True окно открывается общим plt.show() в main


def plot_g_compare(t_k, g_k, t_f, g_f, npts, out_png=None, show=True, title=""):
    """
    g(t) для EKF и фринджа по npts точкам на одном графике.
    Три панели: среднее по ветвям, ветка "+", ветка "-".
    g_k, g_f: dict с ключами "mean", "+", "-" (мкГал, без среднего);
    t_k, t_f -- время, с, для EKF (каждый цикл) и фринджа (центр группы).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    div, unit = _time_unit(float(max(t_k[-1], t_f[-1])))
    panels = (("mean", "среднее g = (g₊ + g₋)/2"),
              ("+", "ветка чирпа «+»"),
              ("-", "ветка чирпа «−»"))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (key, name) in zip(axes, panels):
        ax.plot(t_k / div, g_k[key], "-", lw=0.6, color="tab:blue", alpha=0.8,
                label="EKF (по циклам)")
        ax.plot(t_f / div, g_f[key], "o", ms=2.5, color="tab:orange", alpha=0.7,
                label=f"фринж, {npts} точек (без фильтра)")
        ax.set_ylabel("g, мкГал")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(f"Время, {unit}")
    fig.suptitle(title or "Ряд g: EKF и фринж по точкам")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    if not show:
        plt.close(fig)          # при show=True окно открывается общим plt.show() в main


def save_g_series_csv(path, header, columns):
    """Сохраняет столбцы (список массивов одной длины) в CSV с заголовком header."""
    arr = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.10g")


# =====================================================================
# main
# =====================================================================

def main():
    # ---------------- данные ----------------
    DATA_PATH = r"long_data\data.txt"   # date time norm+ norm- scan+ scan- corr+ corr- offset
    CORR_PER_G = 2048.0        # бинов corr на 1 g (F_z = знак * corr / CORR_PER_G)
    TABLE_ROWS = (0, 127000)   # строки таблицы; дальше ~128000 сигнал меняет режим (непригоден)
    T_CYCLE_NOMINAL = 0.175    # номинальный период цикла (+ и -), с; None -- взять по файлу
    SHOW_PLOT = True

    # Знак F_z для каждой ветки: подобран эмпирически -- на сырых corr оптимум Kz
    # отрицателен для "+" и положителен для "-" (обращение k_eff меняет знак
    # вибрационной фазы).
    FZ_SIGN = {"+": -1.0, "-": +1.0}
    # Знак перевода фазы фринджа в g (g = G_SIGN * ph / (KEFF*T^2)). При обращении
    # k_eff фаза от g меняет знак, поэтому у веток он противоположный, и именно тогда
    # k-независимые фазовые сдвиги сокращаются в среднем (g_+ + g_-)/2. Общий знак
    # (какая ветка "+") задаёт направление оси g; ПРОВЕРЯЕТСЯ по корреляции g_+ и g_-
    # (должна быть положительной), при отрицательной -- поменяйте знак одной ветки.
    G_SIGN = {"+": +1.0, "-": -1.0}

    # ---------------- интерферометр ----------------
    T = 12e-3                    # длительность плеча, с

    # ---------------- блоки ----------------
    BLOCK_SIZE = 9000            # сбросов в блоке (~26 мин при 175 мс); остаток отбрасывается
    MAX_BLOCKS = None            # None -- все блоки; число -- только первые N

    # ---------------- поиск Kz (отдельно для каждой ветки) ----------------
    # F_z = знак * corr / CORR_PER_G, Kz ~ 9.8 (g -> м/с^2).
    # Пробный скан при CORR_PER_G = 2048 дал Kz ~ 10.7 для "+" и ~ 10.1 для "-".
    KZ_BOUNDS = (1.0, 20.0)
    KZ_REF = 9.80665             # линия на графике (None -- не рисовать)
    N_PARTICLES = 10
    N_ITER = 20
    PSO_XTOL = 1e-4              # остановка PSO, когда разброс частиц по Kz меньше
    N_JOBS = None                # None -- все ядра (параллельно по блокам)
    SEED = 0

    # ---------------- EKF / оконный фит ----------------
    DA_MODEL = DB_MODEL = DPH_MODEL = 1e-3   # начальные std шагов [A, B, ph]
    POI = 200                    # точек начального фита (в каждом блоке)
    WARMUP = 200                 # инноваций, не входящих в fitness
    WIN_SIZE = 20                # окно оконного cos-fit

    # ---------------- g: EKF против фринджа, девиация Аллана ----------------
    DO_ALLAN = True
    BEST_MODE = "min_J"          # "min_J" -- блок с наименьшим J; "median" -- медианы блоков
    KZ_MODE = "global"           # Kz при компенсации в итоговых рядах g:
                                 #   "global" -- один Kz на весь ряд (из BEST_MODE);
                                 #   "blocks" -- кусочно-постоянный Kz(t) по блокам
    FRINGE_NPTS = 6              # точек фринджа в группе = длина лестницы scan (метод без фильтра Калмана)
    FRINGE_KZ = "kalman"         # источник Kz для фринджа: "kalman" -- тот же, что в EKF;
                                 # "window" -- Kz оконного cos-fit (глобально: медиана по блокам)
    ALLAN_FIT_M = (6, 60)        # диапазон m (циклов), где аппроксимируется белый шум S/sqrt(m)
    ALLAN_SHOW_BRANCHES = False  # True -- добавить на график Аллана ряды веток "+" и "-"

    # ---------------- чувствительность g к Kz ----------------
    DO_KZ_SENS = True
    DKZ_STEP = 0.02              # шаг численной производной по Kz

    OUT_KZ_PNG = "kz_dynamics_pm.png"
    OUT_G_PNG = "g_compare_pm.png"
    OUT_ALLAN_PNG = "allan_g_pm.png"
    OUT_KZEFF_PNG = "kz_effect_pm.png"
    OUT_G_KALMAN_CSV = "g_kalman_pm.csv"
    OUT_G_FRINGE_CSV = "g_fringe_pm.csv"
    OUT_KZSENS_CSV = "kz_sensitivity_pm.csv"
    NAME = {"+": "plus", "-": "minus"}

    if KZ_MODE not in ("global", "blocks"):
        raise ValueError(f"KZ_MODE: 'global' | 'blocks', получено {KZ_MODE!r}")
    if FRINGE_KZ not in ("kalman", "window"):
        raise ValueError(f"FRINGE_KZ: 'kalman' | 'window', получено {FRINGE_KZ!r}")

    # "alternate": чередование R -> Q с линейным поиском по настоящему J (см. _tune_alternate);
    # "joint" -- подбор sigma и q вместе, даёт те же J, но медленнее.
    tune_cfg = TuneCfg(sigma_init_mode="bins",
                       tune_mode="alternate",
                       r_method="fisher",
                       q_optimizer="lbfgs",
                       n_outer=6,
                       q_free=(True, True, True),
                       # q_log_bounds=(-12, -2),
                       )

    # ---------------- загрузка: обе ветки из одной таблицы ----------------
    data, t_cycle, tinfo = load_table_pm(DATA_PATH, TABLE_ROWS,
                                         t_cycle_nominal=T_CYCLE_NOMINAL)
    t_all = tinfo["t"]           # реальное время каждого оставшегося цикла, с

    # ---------------- Kz по блокам: отдельно для "+" и "-" ----------------
    branches = {}
    for sg in ("+", "-"):
        alp, y, corr = data[sg]
        Fz = FZ_SIGN[sg] * corr / CORR_PER_G
        print(f"\n{'=' * 72}\nВетка чирпа {sg}\n{'=' * 72}")
        res = run_kz_dynamics(
            alp, y, Fz,
            T=T, Kz_bounds=KZ_BOUNDS,
            block_size=BLOCK_SIZE, max_blocks=MAX_BLOCKS,
            t_cycle=t_cycle, t_axis=t_all,
            dA_model=DA_MODEL, dB_model=DB_MODEL, dph_model=DPH_MODEL,
            poi=POI, warmup=WARMUP, win_size=WIN_SIZE,
            tune_cfg=tune_cfg,
            n_particles=N_PARTICLES, n_iter=N_ITER, pso_xtol=PSO_XTOL,
            n_jobs=N_JOBS, seed=SEED, tag=f"[{sg}] ")
        print_summary(res, f"(чирп {sg}) ")
        csv_path = f"kz_dynamics_{NAME[sg]}.csv"
        save_csv(res, csv_path)
        print(f"Таблица сохранена: {csv_path}")
        branches[sg] = {"alp": alp, "y": y, "res": res}

    plot_kz_dynamics({sg: b["res"] for sg, b in branches.items()},
                     kz_ref=KZ_REF, out_png=OUT_KZ_PNG, show=SHOW_PLOT)
    print(f"График сохранён: {OUT_KZ_PNG}")

    # ---------------- g: EKF и фринж, среднее по веткам, Аллан ----------------
    if DO_ALLAN:
        n_use = branches["+"]["res"]["n_use"]
        if branches["-"]["res"]["n_use"] != n_use:
            raise RuntimeError("ветки имеют разное число используемых циклов")
        prm = branches["+"]["res"]["prm"]
        print(f"\nРежим Kz в итоговых рядах g: {KZ_MODE}")

        fin, fr = {}, {}
        fr_glob, fr_blk = {}, {}       # фринж с глобальным Kz и с Kz(t) по блокам
        best_run, kz_f_used = {}, {}
        for sg in ("+", "-"):
            b = branches[sg]
            res = b["res"]
            best = pick_best_kalman(res, BEST_MODE)
            print_best(best, f"(чирп {sg}) ")

            # --- итоговый EKF: Kz скаляр или Kz(t) по блокам ---
            kz_e = (kz_per_cycle(res, "Kz_kalman", best["Kz"], n_use)
                    if KZ_MODE == "blocks" else best["Kz"])
            best_run[sg] = dict(best, Kz=kz_e)
            fin[sg] = final_kalman_run(b["alp"][:n_use], b["y"][:n_use], res["Fz_all"],
                                       best_run[sg], prm, G_SIGN[sg])
            print(f"  итоговый прогон EKF ({KZ_MODE}): J={fin[sg]['J']:.5f}, "
                  f"std(nu)={fin[sg]['nu_std']:.3f}, LBp={fin[sg]['lb_p']:.3f}, "
                  f"точек g: {len(fin[sg]['g_ugal'])}")

            # --- фринж: считаем и с глобальным Kz, и с Kz(t); выбираем по KZ_MODE ---
            if FRINGE_KZ == "window":
                kz_glob = float(np.nanmedian(res["Kz_win"]))
                kz_blk = kz_per_cycle(res, "Kz_win", kz_glob, n_use)
            else:
                kz_glob = best["Kz"]
                kz_blk = kz_per_cycle(res, "Kz_kalman", best["Kz"], n_use)
            i_first_b = fin[sg]["i_first"]
            fr_glob[sg] = fringe_npoint(b["alp"][:n_use], b["y"][:n_use], res["Fz_all"],
                                        kz_glob, T, FRINGE_NPTS, i_first_b, n_use, G_SIGN[sg])
            fr_blk[sg] = fringe_npoint(b["alp"][:n_use], b["y"][:n_use], res["Fz_all"],
                                       kz_blk, T, FRINGE_NPTS, i_first_b, n_use, G_SIGN[sg])
            fr[sg] = fr_blk[sg] if KZ_MODE == "blocks" else fr_glob[sg]
            kz_f_used[sg] = kz_blk if KZ_MODE == "blocks" else kz_glob
            kz_txt = (f"Kz(t) по блокам ({FRINGE_KZ}), mean={np.mean(kz_blk):.5f}"
                      if KZ_MODE == "blocks" else f"Kz={kz_glob:.5f}")
            print(f"  фринж по {FRINGE_NPTS} точкам: {kz_txt}, групп: {fr[sg]['n_groups']}, "
                  f"медиана амплитуды B={np.median(fr[sg]['amp']):.4g}; "
                  f"групп с полной лестницей из {FRINGE_NPTS} значений чирпа: "
                  f"{np.mean(fr[sg]['n_distinct'] == FRINGE_NPTS):.1%}")

        i_first = fin["+"]["i_first"]
        npts = FRINGE_NPTS

        # --- ряды g: ветки и среднее ---
        g_k = {"+": fin["+"]["g_ugal"], "-": fin["-"]["g_ugal"]}
        g_k["mean"] = 0.5 * (g_k["+"] + g_k["-"])
        g_f = {"+": fr["+"]["g_ugal"], "-": fr["-"]["g_ugal"]}
        g_f["mean"] = 0.5 * (g_f["+"] + g_f["-"])
        # реальное время: каждого цикла (EKF) и центра группы (фринж)
        t_k = t_all[i_first + np.arange(len(g_k["mean"]))]
        t_f = np.interp(fr["+"]["i_center"], np.arange(len(t_all)), t_all)

        # --- проверки ---
        r_k = branch_consistency(g_k["+"], g_k["-"], npts)
        r_f = float(np.corrcoef(g_f["+"], g_f["-"])[0, 1])
        print(f"\nКорреляция g_+ и g_- (должна быть > 0 при верных G_SIGN): "
              f"EKF {r_k:+.3f}, фринж {r_f:+.3f}")
        if r_k < 0 or r_f < 0:
            print("  ВНИМАНИЕ: корреляция отрицательна -- проверьте знак G_SIGN одной из веток")
        n_g = fr["+"]["n_groups"]
        k_avg = g_k["mean"][:n_g * npts].reshape(n_g, npts).mean(axis=1)
        print(f"Среднее g: std EKF = {g_k['mean'].std():.3g} мкГал (по циклам), "
              f"std фринж = {g_f['mean'].std():.3g} мкГал ({npts}-точечные группы); "
              f"std(EKF усредн. по группам - фринж) = {np.std(k_avg - g_f['mean']):.3g} мкГал")

        # --- CSV ---
        save_g_series_csv(OUT_G_KALMAN_CSV, "cycle,t_s,g_plus_ugal,g_minus_ugal,g_mean_ugal",
                          [i_first + np.arange(len(g_k["mean"])), t_k,
                           g_k["+"], g_k["-"], g_k["mean"]])
        save_g_series_csv(OUT_G_FRINGE_CSV, "cycle_center,t_s,g_plus_ugal,g_minus_ugal,g_mean_ugal",
                          [fr["+"]["i_center"], t_f, g_f["+"], g_f["-"], g_f["mean"]])
        print(f"Ряды g сохранены: {OUT_G_KALMAN_CSV}, {OUT_G_FRINGE_CSV}")

        # --- график g: EKF и фринж на одном графике ---
        plot_g_compare(t_k, g_k, t_f, g_f, npts, OUT_G_PNG, SHOW_PLOT,
                       f"Ряд g (среднее по ветвям ±): EKF и фринж по точкам, Kz: {KZ_MODE}")
        print(f"График сохранён: {OUT_G_PNG}")

        # --- девиация Аллана: EKF и фринж на одном графике ---
        series = [("mean", "среднее", "tab:blue", "tab:orange")]
        if ALLAN_SHOW_BRANCHES:
            series += [("+", "чирп +", "tab:cyan", "tab:olive"),
                       ("-", "чирп −", "tab:purple", "tab:brown")]
        curves = []
        for key, lab, col_k, col_f in series:
            for g, cps, name, col in ((g_k[key], 1, "EKF", col_k),
                                      (g_f[key], npts, f"фринж {npts} т.", col_f)):
                m, adev, err = adev_in_cycles(g, cps)
                sens = white_noise_sensitivity(m, adev, *ALLAN_FIT_M)
                main_curve = key == "mean"
                curves.append(dict(label=f"{name}, {lab}", m=m, adev=adev, err=err, sens=sens,
                                   color=col, fmt="o-" if main_curve else "-",
                                   alpha=1.0 if main_curve else 0.6))
                print(f"Аллан [{name}, {lab}]: S = {sens:.3g} мкГал/√цикл "
                      f"(≈ {sens * np.sqrt(t_cycle):.3g} мкГал/√Гц); "
                      f"ADEV({m[0]} ц.) = {adev[0]:.3g}, ADEV({m[-1]} ц.) = {adev[-1]:.3g} мкГал")
        plot_allan_compare(curves, t_cycle, ALLAN_FIT_M, OUT_ALLAN_PNG, SHOW_PLOT,
                           "Девиация Аллана g: EKF и фринж по точкам")
        print(f"График сохранён: {OUT_ALLAN_PNG}")

        # ---------------- чувствительность g к Kz ----------------
        if DO_KZ_SENS:
            print(f"\n{'=' * 72}\nЧувствительность g к Kz (шаг {DKZ_STEP:g})\n{'=' * 72}")
            sens_f, sens_k, kz_sd = {}, {}, {}
            for sg in ("+", "-"):
                b, res = branches[sg], branches[sg]["res"]
                # производная берётся при том Kz, которым считаются итоговые ряды
                sens_f[sg] = kz_sensitivity_fringe(
                    b["alp"][:n_use], b["y"][:n_use], res["Fz_all"], kz_f_used[sg], T,
                    npts, i_first, n_use, G_SIGN[sg], DKZ_STEP)
                sens_k[sg] = kz_sensitivity_ekf(
                    b["alp"][:n_use], b["y"][:n_use], res["Fz_all"], best_run[sg],
                    prm, G_SIGN[sg], DKZ_STEP)
                kz_sd[sg] = float(np.nanstd(res["Kz_kalman"], ddof=1))   # разброс Kz по блокам
                sf = sens_f[sg]
                print(f"[{sg}] dg/dKz, мкГал/ед.Kz: фринж mean={sf.mean():+.4g}, "
                      f"std по группам={sf.std():.3g}; EKF mean={sens_k[sg].mean():+.4g}, "
                      f"std по циклам={sens_k[sg].std():.3g}")
                print(f"     разброс Kz по блокам std={kz_sd[sg]:.3g} -> g: "
                      f"сдвиг ~ {abs(sf.mean()) * kz_sd[sg]:.3g} мкГал, "
                      f"шум по группам ~ {sf.std() * kz_sd[sg]:.3g} мкГал")

            s_mean = 0.5 * (sens_f["+"] + sens_f["-"])
            # ошибки Kz двух веток: коррелированы (одинаковый знак вклада в среднее)
            # или независимы -- вилка оценки сдвига среднего g
            shift_corr = 0.5 * (abs(sens_f["+"].mean()) * kz_sd["+"]
                                + abs(sens_f["-"].mean()) * kz_sd["-"])
            shift_ind = 0.5 * np.hypot(sens_f["+"].mean() * kz_sd["+"],
                                       sens_f["-"].mean() * kz_sd["-"])
            print(f"Среднее по ветвям: dg/dKz mean={s_mean.mean():+.4g}, "
                  f"std по группам={s_mean.std():.3g} мкГал/ед.Kz")
            print(f"Сдвиг среднего g из-за разброса Kz по блокам: "
                  f"независимые ошибки ~ {shift_ind:.3g} мкГал, "
                  f"общая ошибка (худший случай) ~ {shift_corr:.3g} мкГал")
            print(f"Разница Kz(+)-Kz(-) по лучшим прогонам: "
                  f"{best_run['+']['Kz'].mean() - best_run['-']['Kz'].mean() if KZ_MODE == 'blocks' else best_run['+']['Kz'] - best_run['-']['Kz']:+.4f}")

            # --- эффект Kz(t): фринж с Kz по блокам минус фринж с глобальным Kz ---
            g_glob_mean = 0.5 * (fr_glob["+"]["g_ugal"] + fr_glob["-"]["g_ugal"])
            g_blk_mean = 0.5 * (fr_blk["+"]["g_ugal"] + fr_blk["-"]["g_ugal"])
            effect = g_blk_mean - g_glob_mean
            print(f"Эффект Kz(t) на g (фринж, среднее по ветвям; блоки минус глобальный Kz): "
                  f"std={effect.std():.3g} мкГал, размах={np.ptp(effect):.3g} мкГал; "
                  f"для сравнения std g (глобальный Kz)={g_glob_mean.std():.3g} мкГал")

            save_g_series_csv(OUT_KZSENS_CSV,
                              "cycle_center,t_s,dgdKz_plus,dgdKz_minus,dgdKz_mean,"
                              "g_effect_Kz_t_ugal",
                              [fr["+"]["i_center"], t_f, sens_f["+"], sens_f["-"], s_mean,
                               effect])
            print(f"Чувствительность сохранена: {OUT_KZSENS_CSV}")
            plot_kz_effect(t_f, {"+": sens_f["+"], "-": sens_f["-"], "mean": s_mean},
                           effect, OUT_KZEFF_PNG, SHOW_PLOT)
            print(f"График сохранён: {OUT_KZEFF_PNG}")

    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        plt.show()              # все фигуры одновременно


if __name__ == "__main__":
    main()