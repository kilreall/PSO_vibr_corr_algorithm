import numpy as np
import matplotlib.pyplot as plt


# constants
lm = 780e-9
keff = 4*np.pi/lm
N_RP = 16384


def simul_waveform(N_sim, alp_amount, T, Dph_drift, drift_corr, sigma_A_sim, sigma_ph_sim, dA_sim, dB_sim, DA_sim):


    # --- дрейф фазы (процесс Орнштейна-Уленбека) ---
    g0 = 9.8125
    ph0 = g0*keff*T**2
    ph_sim = np.zeros(N_sim); ph_sim[-1] = ph0
    ph_sim_noise = np.zeros(N_sim)
    theta_drift = 1 - np.exp(-1.0/drift_corr)
    d_ph_drift = Dph_drift * np.sqrt(theta_drift*(2 - theta_drift))

    # --- сетка alp ---
    alp_min = keff*g0/2/np.pi - 2/T/T
    alp_max = keff*g0/2/np.pi + 2/T/T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)

    alp = np.zeros(N_sim)
    Ph = np.zeros(N_sim)


    # --- параметры фринджа ---
    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15 
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)

    for i in range(N_sim):

        ph_sim[i] = ph0 + (ph_sim[i-1] - ph0)*(1 - theta_drift) + d_ph_drift*np.random.normal(0, 1)
        ph_sim_noise[i] = ph_sim[i] + np.random.normal(0, sigma_ph_sim)

        alp[i] = alp_start[i % alp_amount]
        Ph[i] = 2*np.pi*alp[i]*T*T - ph_sim_noise[i]

        A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])

        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)

    dph_sim = d_ph_drift
    dA_sim = np.sqrt(dA_sim**2 + DA_sim**2)

    out = (alp, P_sim_noise, P_sim, ph_sim_noise, ph_sim, A_sim, B_sim, dA_sim, dB_sim,
           dph_sim, sigma_A_sim, sigma_ph_sim)

    return out


 
def main():
    # ---------- интерферометр ----------
    T = 10e-3                  # длительность плеча, с
    tau = 10e-6                # длительность импульса, с
    # ---------- набор данных ----------
    alp_amount = 200           # точек alp в одном фринджe
    n_fringes = 10             # число фринджей
    # ---------- фринж и шумы ----------
    dA_sim = 1e-4              # шаг блуждания A за сброс
    dB_sim = 4e-4              # шаг блуждания B за сброс
    DA_sim = 0.0               # добавка к возвращаемому dA
    sigma_A_sim = 5e-3         # шум детектирования P
    sigma_ph_sim = 1e-4        # белый фазовый шум на сброс, рад (0 -- выключить)
    # ---------- дрейф фазы ----------
    Dph_drift = 1e-3           # стационарная std дрейфа ph (OU), рад
    drift_corr = 5000          # время корреляции, сбросов
 
    N_sim = alp_amount*n_fringes

 
    out = simul_waveform(N_sim, alp_amount, T, Dph_drift, drift_corr,
                         sigma_A_sim, sigma_ph_sim, dA_sim, dB_sim, DA_sim)

 
    plot_waveform(out, T=T, Dph_drift=Dph_drift,
                  title=f"N_sim={N_sim}, sigma_A={sigma_A_sim:g}, sigma_ph={sigma_ph_sim:g}")
    plt.show()

    return out
 
 
def plot_waveform(out, *, T, g0=9.8125, Dph_drift=None, n_bins=10, title=""):
    """
    Отображение результата simul_wavefrom (кортеж out из 12 элементов, в его порядке).
    ph_total = ph_sim + белый фазовый шум -- фаза, которая реально идёт в фринж;
    ph_sim -- чистый OU-дрейф (то, что оценивает EKF).
    Рисует две фигуры: waveform/истина/фринж и разбор невязки по фазе.
    """
    (alp, P_noise, P_clean, ph_total, ph_sim, A_s, B_s, dA, dB, dph,
     sigma_A, sigma_ph) = out
    N_sim = len(alp)

    ph0 = g0*keff*T**2
    ph_noise = ph_total - ph_sim          # белый фазовый шум

    wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
    x_raw = wrap(2*np.pi*T**2*alp - ph0)       # фаза без учёта дрейфа
    x_drift = wrap(2*np.pi*T**2*alp - ph_sim)  # фаза с учётом истинного дрейфа
    xs = np.linspace(-np.pi, np.pi, 400)
    model = A_s.mean() - B_s.mean()*np.cos(xs)
    shots = np.arange(N_sim)

    # невязка относительно ИСТИННЫХ A, B и ЧИСТОГО дрейфа ph_sim (без белого фазового
    # шума): невязка содержит и шум детектирования, и белый фазовый шум
    Phi = 2*np.pi*T**2*alp - ph_sim
    res = P_noise - (A_s - B_s*np.cos(Phi))
    phi_m = np.mod(Phi, np.pi)
    xk = B_s**2*np.sin(Phi)**2
    bins = np.minimum((phi_m/np.pi*n_bins).astype(int), n_bins - 1)
    b_ok = [b for b in range(n_bins) if np.sum(bins == b) > 5]
    v_b = np.array([np.mean(res[bins == b]**2) for b in b_ok])
    x_b = np.array([np.mean(xk[bins == b]) for b in b_ok])
    v_th = sigma_A**2 + sigma_ph**2*x_b

    # ---------- сводка ----------
    print("Заложено / ожидается:")
    print(f"  dA = {dA:.2e}, dB = {dB:.2e}, dph (шаг OU) = {dph:.2e} рад")
    print(f"  sigma_A = {sigma_A:.2e}")
    print(f"  sigma_ph (белый, заложен) = {sigma_ph:.2e} рад; "
          f"в данных std(ph_total - ph_sim) = {np.std(ph_noise):.2e} рад")
    # насколько фазовый шум вообще виден в невязке (по сравнению с sigma_A^2)
    share = sigma_ph**2*x_b.max()/sigma_A**2
    print(f"Вклад фазового шума в дисперсию невязки: до {100*share:.2g}% от sigma_A² "
          f"(при max<B²sin²Ph> = {x_b.max():.2e})")
    # порог определимости sigma_ph по инновациям: вклад ~ B²/2*sigma_ph² сравним
    # с погрешностью дисперсии sqrt(2/N)*sigma_A²
    B_mean = float(np.mean(np.abs(B_s)))
    floor = sigma_A/B_mean*np.sqrt(2)*(2/N_sim)**0.25
    print(f"Порог определимости sigma_ph по {N_sim} сбросам (1 sigma) ≈ {floor:.2e} рад: "
          f"{'выше порога' if sigma_ph > floor else 'НИЖЕ порога, автоподбор не определит'}")
    print(f"std невязки = {np.std(res):.3e}, sqrt(sigma_A² + вклад) = "
          f"{np.sqrt(np.mean(v_th)):.3e}")

    # ---------- фигура 1: waveform, истина, фринж ----------
    fig1, axs = plt.subplots(2, 3, figsize=(16, 8))

    axs[0, 0].plot(shots, P_noise, '.', ms=2, alpha=0.6, label='P + шум детектирования')
    axs[0, 0].plot(shots, P_clean, lw=0.6, label='P (с фаз. шумом, без шума дет.)')
    axs[0, 0].set(xlabel='сброс #', ylabel='P', title='Waveform P')
    axs[0, 0].legend()

    axs[0, 1].plot(shots, A_s, label='A')
    axs[0, 1].plot(shots, B_s, label='B')
    axs[0, 1].set(xlabel='сброс #', title='Истина: A, B')
    axs[0, 1].legend()

    axs[0, 2].plot(shots, ph_sim - ph0, label='ph_sim (OU)')
    if Dph_drift is not None:
        axs[0, 2].axhline(Dph_drift, color='gray', ls='--', lw=0.8, label='±Dph_drift (std OU)')
        axs[0, 2].axhline(-Dph_drift, color='gray', ls='--', lw=0.8)
    axs[0, 2].set(xlabel='сброс #', ylabel='ph - ph0, рад', title='Истина: дрейф ph (OU)')
    axs[0, 2].legend()

    axs[1, 0].plot(shots, ph_noise, lw=0.5)
    axs[1, 0].set(xlabel='сброс #', ylabel='рад',
                  title='Истина: белый фазовый шум (ph_total − ph_sim)')

    axs[1, 1].plot(x_raw, P_noise, '.', ms=2, alpha=0.5)
    axs[1, 1].plot(xs, model, 'r-', lw=1)
    axs[1, 1].set(xlabel='Ph = 2πT²·alp − ph0 (mod 2π), рад', ylabel='P',
                  title='Фринж БЕЗ учёта дрейфа')

    axs[1, 2].plot(x_drift, P_noise, '.', ms=2, alpha=0.5)
    axs[1, 2].plot(xs, model, 'r-', lw=1)
    axs[1, 2].set(xlabel='Ph = 2πT²·alp − ph_sim (mod 2π), рад', ylabel='P',
                  title='Фринж С учётом истинного дрейфа')

    for a in axs.ravel():
        a.grid(True, alpha=0.3)
    fig1.suptitle(title)
    fig1.tight_layout()

    # ---------- фигура 2: невязка, её дисперсия по фазе, белый фазовый шум ----------
    fig2, axs2 = plt.subplots(1, 3, figsize=(17, 4.5))
    axs2[0].plot(np.mod(Phi + np.pi/2, np.pi) - np.pi/2, res, '.', ms=2, alpha=0.5)
    axs2[0].set(xlabel='Ph (mod π, сдвиг к [-π/2, π/2)), рад', ylabel='P − P_истина',
                title='Невязка относительно истинных A, B, ph_sim')
    axs2[1].plot(x_b, v_b, 'o', label='дисперсия невязки в бине')
    axs2[1].plot(x_b, v_th, 'r-', label='sigma_A² + sigma_ph²·<B²sin²Ph>')
    axs2[1].set(xlabel='<B² sin²Ph> в бине', ylabel='дисперсия',
                title='Проверка: sigma_A и sigma_ph')
    axs2[1].legend()
    axs2[2].hist(ph_noise, bins=50, density=True, alpha=0.6, label='ph_total − ph_sim')
    if sigma_ph > 0:
        xg = np.linspace(-4*sigma_ph, 4*sigma_ph, 200)
        axs2[2].plot(xg, np.exp(-xg**2/(2*sigma_ph**2))/(sigma_ph*np.sqrt(2*np.pi)),
                     'r-', label='N(0, sigma_ph)')
    axs2[2].set(xlabel='рад', ylabel='плотность', title='Белый фазовый шум')
    axs2[2].legend()
    for a in axs2:
        a.grid(True, alpha=0.3)
    fig2.tight_layout()
    return fig1, fig2

 


if __name__ == "__main__":
    main()