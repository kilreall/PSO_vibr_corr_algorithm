import numpy as np
from numba import njit


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
      x_out -- апостериорные оценки состояния.
    Возвращает (sum_{i>=start} [ln S_i + e_i^2/S_i], число таких точек);
    (inf, 0) при сбое (S <= 0 или нечисловая невязка).
    """
    n = alp.shape[0]
    x = x0.copy()
    P = P0.copy()
    H = np.empty(3)
    PHt = np.empty(3)
    K = np.empty(3)
    nll = 0.0 # negative likelohood - мера по типу std
    count = 0 # счётчик

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

        # --- 5) обновление (форма Джозефа не нужна: P^- H^T / S) ---
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
            count += 1

    return nll, count