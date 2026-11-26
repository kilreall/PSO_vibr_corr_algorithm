# func to test kalman filter without vibrations on ceration data with comfortable initializtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter
from scipy.signal import savgol_filter
from scipy.signal import savgol_coeffs
from gen_waveform import simul_waveform


# constants
lm = 780e-9
keff = 4*np.pi/lm
g0 = 9.8125

def model(alp, A, B, ph, T):
    return A - B*np.cos(2*np.pi*alp*T**2 - ph)




def init_values_linear(alp, P_exp, poi, T):
    """
    Линейный аналог init_values. Модель A - B*cos(Phi-ph) раскладывается
    как A - C*cos(Phi) - D*sin(Phi), где C=B*cos(ph), D=B*sin(ph) --
    линейные параметры, решаем обычной МНК вместо nonlinear curve_fit.
    Возвращает то же самое: A0, B0, ph0, pcov (3x3, в параметризации
    [A,B,ph], перенесённой Якобианом из [A,C,D]), sigma_A.
    """
 
    x = alp[:poi]
    y = P_exp[:poi]
 
    Phi = 2*np.pi*x*T**2
    c = np.cos(Phi)
    s = np.sin(Phi)
    M = np.stack([np.ones_like(c), -c, -s], axis=-1)     # (poi, 3)
 
    params, *_ = np.linalg.lstsq(M, y, rcond=None)
    A0, C0, D0 = params
 
    B0 = np.sqrt(C0**2 + D0**2)
    ph0 = np.arctan2(D0, C0)
 
    resid = y - (A0 - C0*c - D0*s)
    sigma_A = np.std(resid)
 
    dof = max(len(y) - 3, 1)
    var_resid = np.sum(resid**2) / dof
    MtM_inv = np.linalg.inv(M.T @ M)
    cov_ACD = MtM_inv * var_resid
 
    if B0 > 1e-12:
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, C0/B0, D0/B0],
            [0.0, -D0/B0**2, C0/B0**2],
        ])
    else:
        J = np.eye(3)
 
    pcov = J @ cov_ACD @ J.T
 
    return A0, B0, ph0, pcov, sigma_A

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A - B*np.cos(Phi)

    return np.array([P])

def HJacobian(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    H = np.array([
        [1.0,
         -np.cos(Phi),
         -B*np.sin(Phi)]
    ])

    return H


def kalmanFit_EKF(alp, P_exp, T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0):

    N = len(alp)

    # --------------------------------------------------
    # Output arrays
    # --------------------------------------------------

    A = np.zeros(N)
    B = np.zeros(N)
    ph = np.zeros(N)
    P_m = np.zeros(N)
    e = np.zeros(N)
    en = np.zeros(N)
    P_cov = np.zeros((N, 3, 3))

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    # create EKF
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)

    A[0], B[0], ph[0], ekf.P = A0, B0, ph0, P_cov0

    # initial covariance
    P_cov[0] = ekf.P

    P_m[0] = model(
        alp[0],
        A[0],
        B[0],
        ph[0],
        T
    )

    # state = [A, B, phi]
    ekf.x = np.array([A[0], B[0], ph[0]])

    # process noise
    ekf.Q = Q

    # --------------------------------------------------
    # Main cycle
    # --------------------------------------------------

    for i in range(1, N):

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        ekf.predict()

        # predicted measurement
        P_m[i] = Hx(
            ekf.x,
            alp[i],
            T
        )[0]

        # innovation
        e[i] = P_exp[i] - P_m[i]

        # ----------------------------------------------
        # Measurement update
        # ----------------------------------------------

        # measurement noise
        ekf.R = np.array([
            [sigma_A**2 + B[i-1]**2*np.sin(2*np.pi*alp[i]*T*T - ph[i-1])**2*sigma_ph**2]
        ])

        ekf.update(
            np.array([P_exp[i]]),
            HJacobian,
            Hx,
            args=(alp[i], T),
            hx_args=(alp[i], T)
        )

        # normalized innovation
        S_i = ekf.S[0, 0]
        en[i]= e[i]/np.sqrt(S_i)

        # ----------------------------------------------
        # Save state
        # ----------------------------------------------

        A[i] = ekf.x[0]
        B[i] = ekf.x[1]
        ph[i] = ekf.x[2]
    
        # covariance
        P_cov[i] = ekf.P

    return P_m, A, B, ph, P_cov, e, en



def windowFit_linear(alp, P_exp, T, window, step=1):
    """Каузальный фит: оценка на индексе k использует только отсчёты k-window+1 ... k."""
    N = len(alp)
    phi = np.full(N, np.nan)

    ends = np.arange(window - 1, N, step)
    if len(ends) == 0:
        return phi

    idx = ends[:, None] + np.arange(-(window - 1), 1)[None, :]
    x = alp[idx]
    y = P_exp[idx]

    Phi = 2*np.pi*x*T**2
    c, s = np.cos(Phi), np.sin(Phi)
    Mrows = np.stack([np.ones_like(c), -c, -s], axis=-1)     # (K, window, 3)

    MtM = np.einsum('kwi,kwj->kij', Mrows, Mrows)
    Mty = np.einsum('kwi,kw->ki', Mrows, y)[..., None]

    params = np.full((len(ends), 3), np.nan)
    try:
        params = np.linalg.solve(MtM, Mty)[..., 0]           # [A, C, D]
    except np.linalg.LinAlgError:
        for k in range(len(ends)):
            params[k] = np.linalg.lstsq(Mrows[k], y[k], rcond=None)[0]

    phi[ends] = np.arctan2(params[:, 2], params[:, 1])       # фаза mod 2π, сравнивать через wrap
    return phi



def main():
    # ---------- интерферометр ----------
    T = 10e-3                  # длительность плеча, с
    tau = 10e-6                # длительность импульса, с
    # ---------- набор данных ----------
    alp_amount = 40           # точек alp в одном фринджe
    n_fringes = 1000             # число фринджей
    # ---------- фринж и шумы ----------
    dA_sim = 1e-5              # шаг блуждания A за сброс
    dB_sim = 2e-6              # шаг блуждания B за сброс
    DA_sim = 0.0               # добавка к возвращаемому dA
    sigma_A_sim = 1e-3         # шум детектирования P
    sigma_ph_sim = 3e-2        # белый фазовый шум на сброс
    # ---------- дрейф фазы ----------
    Dph_drift = 1e-3           # стационарная std дрейфа ph (OU), рад
    drift_corr = 50000          # время корреляции, сбросов

    # Gen data
    N_sim = alp_amount*n_fringes 
    out = simul_waveform(N_sim, alp_amount, T, Dph_drift, drift_corr,
                         sigma_A_sim, sigma_ph_sim, dA_sim, dB_sim, DA_sim)
    alp, P_sim_noise, P_sim, ph_sim_noise, ph_sim, A_sim, B_sim, dA_sim, dB_sim, dph_sim, sigma_A_sim, sigma_ph_sim = out
    print(f"dph_sim={dph_sim*1e3} mrad")

    # Kalman params
    alp = alp
    P_exp = P_sim_noise
    warmup = alp_amount
    A0, B0, ph0, P_cov0, sigma_A = init_values_linear(alp, P_exp, warmup, T)
    Q = np.diag([dA_sim**2, dB_sim**2, dph_sim**2])
    sigma_A = sigma_A_sim
    sigma_ph = sigma_ph_sim
    P_kalman, A, B, ph_kalman, P_cov, e, en = kalmanFit_EKF(alp[warmup-1:], P_exp[warmup-1:], T, Q, P_cov0, sigma_A, sigma_ph, A0, B0, ph0)

    #curve_fit
    window = alp_amount
    ph_cf_1 = windowFit_linear(alp, P_exp, T, window, step=1)
    ph_cf_a = windowFit_linear(alp, P_exp, T, window, step=window)


    #plot
    plot_result(warmup, ph_sim_noise % (2*np.pi), ph_sim % (2*np.pi), ph_kalman % (2*np.pi), P_cov, ph_cf_1 % (2*np.pi), ph_cf_a % (2*np.pi))



def plot_result(warmup, ph_sim_noise, ph_sim, ph_kalman, P_cov, ph_cf_1, ph_cf_a):

    razm = 1e3

    # # !Kalman evaluation vs true values!
    # plt.figure()
    # plt.title(r"Kalman evaluation vs true values")
    # plt.plot(abs(ph_sim[warmup-1:]-ph_kalman)*razm, label="True", color="blue")
    # plt.plot(abs(ph_sim_noise[warmup-1:]-ph_kalman)*razm, label = "Noisy", color="orange")
    # plt.xlabel(r"count")
    # plt.ylabel(r"|$\delta \phi$|, [mrad]")
    # plt.legend()

    # # !Kalman covariance for phase!
    # plt.figure()
    # plt.title(r"Kalman covariance for phase")
    # plt.plot(np.sqrt(P_cov[:,2,2])*razm)
    # plt.xlabel(r"count")
    # plt.ylabel(r"$\sigma_{\phi}$, [mrad]")

    # # !Comparing Kalman with continious curve-fit!
    # plt.figure()
    # plt.title(r"Comparing Kalman with continious curve-fit")
    # plt.plot((ph_sim[warmup-1:]-ph_kalman)*razm, label = "Kalman", color="blue")
    # plt.plot((ph_sim[warmup-1:]-ph_cf_1[warmup-1:])*razm, label = "curve-fit", color="orange")
    # plt.xlabel(r"count")
    # plt.ylabel(r"$\delta \phi$, [mrad]")
    # plt.legend()


    # !Comparing Kalman with averaging curve-fit!

    # число использованных точек: оценка на глобальном индексе k использует k+1 отсчётов
    x_k = np.arange(warmup, warmup + len(ph_kalman))

    idx_a = np.flatnonzero(~np.isnan(ph_cf_a))     # где у curve_fit есть оценка
    idx_a = idx_a[idx_a >= warmup-1]
    x_a = idx_a + 1

    # накопленное (круговое) среднее оценок curve_fit по непересекающимся окнам
    est_a = ph_cf_a[idx_a]
    cf_a_avg = np.angle(np.cumsum(np.exp(1j*est_a)) / np.arange(1, len(est_a)+1))

    plt.figure()
    plt.title(r"Comparing Kalman with averaging curve-fit")
    plt.plot(x_k, (ph_sim[warmup-1:]-ph_kalman)*razm, label = "Kalman", color="blue")
    plt.plot(x_a, (ph_sim[idx_a]-est_a)*razm, label = "curve-fit", color="orange")
    plt.plot(x_a, (ph_sim[idx_a]-cf_a_avg)*razm, label = "curve-fit, averaging", color="green")
    plt.xlabel(r"counts")
    plt.ylabel(r"$\delta \phi$, [mrad]")
    plt.legend()

    # !Mistake comparing!

    # среднеквадратичная ошибка среднего (SEM) для curve-fit по непересекающимся окнам:
    # разброс оценок вокруг их текущего среднего / sqrt(n), n -- число оценок в среднем
    n_est = np.arange(1, len(est_a)+1)
    sem_a = np.full(len(est_a), np.nan)
    for n in range(2, len(est_a)+1):
        dev = (est_a[:n] - cf_a_avg[n-1])            # отклонения от текущего среднего
        sem_a[n-1] = np.sqrt(np.sum(dev**2)/(n-1)) / np.sqrt(n)

    plt.figure()
    plt.title(r"Mistake comparing")
    plt.plot(x_a, sem_a*razm, label = r"curve-fit SEM = std/$\sqrt{n}$", color="green")
    plt.plot(x_k, np.sqrt(P_cov[:,2,2])*razm, label = r"Kalman $\sigma_\phi$", color="blue")
    plt.plot(x_a, np.abs(ph_sim[idx_a]-cf_a_avg)*razm, label = r"|true mistake of real|", color="orange")
    plt.plot(x_k, np.abs(ph_sim[warmup-1:]-ph_kalman)*razm, label = r"|true mistake of Kalman|", color="red")
    plt.xlabel(r"counts")
    plt.ylabel(r"$\sigma_\phi$, [mrad]")
    plt.legend()

    # !allan deviation!
    def allan_dev(x, ms):
        """sigma_y(m) = sqrt(0.5 * <(ybar_{k+m} - ybar_k)^2>), ybar -- среднее по m отсчётам."""
        x = np.asarray(x, dtype=float)
        c = np.concatenate(([0.0], np.cumsum(x)))
        m_out, s_out = [], []
        for m in ms:
            if 2*m > len(x):
                break
            yb = (c[m:] - c[:-m]) / m          # скользящие средние по m отсчётам
            d = yb[m:] - yb[:-m]               # разность соседних (непересекающихся) средних
            m_out.append(m)
            s_out.append(np.sqrt(0.5*np.mean(d**2)))
        return np.array(m_out), np.array(s_out)

    def log_ms(n):
        return np.unique(np.logspace(0, np.log10(max(n//2, 1)), 20).astype(int))

    # фазы приходят после % 2π, поэтому перед Алланом разворачиваем
    ph_k_u  = np.unwrap(ph_kalman)
    est_a_u = np.unwrap(est_a)
    avg_a_u = np.unwrap(cf_a_avg)
    ph_t_u  = np.unwrap(ph_sim[warmup-1:])      # истинный дрейф, для справки

    dt_a = idx_a[1] - idx_a[0]                  # шаг между оценками curve-fit (= window точек)

    m_k, s_k = allan_dev(ph_k_u,  log_ms(len(ph_k_u)))
    m_e, s_e = allan_dev(est_a_u, log_ms(len(est_a_u)))
    m_v, s_v = allan_dev(avg_a_u, log_ms(len(avg_a_u)))
    m_t, s_t = allan_dev(ph_t_u,  log_ms(len(ph_t_u)))
    ph_cf_1_u = np.unwrap(ph_cf_1[warmup-1:])      # скользящее окно, без NaN (warmup == window)
    m_s, s_s = allan_dev(ph_cf_1_u, log_ms(len(ph_cf_1_u)))

    plt.figure()
    plt.title(r"Allan deviation of $\phi$")
    plt.loglog(m_e*dt_a, s_e*razm, 'o-', label=r"curve-fit (non-overlap)", color="orange")
    plt.loglog(m_v*dt_a, s_v*razm, 's-', label=r"curve-fit, for averaging", color="green")
    plt.loglog(m_k,      s_k*razm, '-',  label=r"Kalman", color="blue")
    plt.loglog(m_s,      s_s*razm, '-',  label=r"curve-fit (sliding)", color="red")
    plt.loglog(m_t,      s_t*razm, '--', label=r"true $\phi$ (drift)", color="gray")
    plt.xlabel(r"counts")
    plt.ylabel(r"$\sigma_\phi(N)$, [mrad]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()


    return 1

if __name__ == "__main__": main()
plt.show()