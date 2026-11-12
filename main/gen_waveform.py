import numpy as np
import matplotlib.pyplot as plt

from main.gen_acc import generate_vibration

# constants
lm = 780e-9
keff = 4*np.pi/lm
N_RP = 16384

def fa(t, T, tau):
    if 0 < t <= T + 2*tau:
        return t
    elif T + 2*tau < t <= 2*T + 4*tau:
        return 2*(T + 2*tau) - t
    else:
        return 0


def sens_integral(delay, a, end, weight_vec):
    """Сырой (без Kz) интеграл чувствительности по окну интерферометра."""
    seg = a[delay:delay + end]
    if len(seg) != end:
        raise ValueError(f"delay={delay}: окно [{delay}, {delay+end}) выходит "
                         f"за пределы реализации длиной {len(a)}")
    return keff * float(seg @ weight_vec)


def simul_wavefrom(N_sim, alp_amount, T_RP, T, tau, delay, Kz, Kx, Ky, vib_state, sigma_a, Dph_drift, drift_corr, sigma_A_sim, sigma_ph_sim, dA_sim, dB_sim, DA_sim, vib_seed=42):
    #params
    fat_v = np.vectorize(fa, otypes=[float])
    t_step = T_RP/N_RP
    t_range = np.linspace(0, 2*T + 4*tau, round((2*T + 4*tau)/t_step))
    end = len(t_range)
    fa_t = fat_v(t_range, T, tau)
    # веса трапеций: интеграл = keff * (a_window @ weight_vec)
    _trapz_w = np.ones(end) * t_step
    _trapz_w[0] *= 0.5
    _trapz_w[-1] *= 0.5
    weight_vec = fa_t * _trapz_w

    # --- дрейф фазы (процесс Орнштейна-Уленбека) ---
    g0 = 9.8125
    ph0 = g0*keff*T**2
    ph_sim = np.zeros(N_sim); ph_sim[-1] = ph0
    ph_sim_noise = np.zeros(N_sim)
    theta_drift = 1 - np.exp(-1.0/drift_corr)
    sigma_ph_drift = Dph_drift * np.sqrt(theta_drift*(2 - theta_drift))

    # --- сетка alp ---
    alp_min = keff*g0/2/np.pi - 1/5/T/T
    alp_max = keff*g0/2/np.pi + 1/5/T/T
    alp_start = np.linspace(alp_min, alp_max, alp_amount)

    alp = np.zeros(N_sim)
    alp_vib = np.zeros(N_sim)
    Ph = np.zeros(N_sim)
    F_vibz = np.zeros(N_sim)
    F_vibx = np.zeros(N_sim)
    F_viby = np.zeros(N_sim)

    ax_m, ay_m, az_m, info = generate_vibration(
        vib_state, N_RP, T_RP, seed=vib_seed,
        hp_cutoff=0.01, N_sim=N_sim)
    ax_m = ax_m.reshape(N_sim, N_RP)
    ay_m = ay_m.reshape(N_sim, N_RP)
    az_m = az_m.reshape(N_sim, N_RP)



    # --- параметры фринджа ---
    A_sim = np.zeros(N_sim); A_sim[-1] = 0.15 
    B_sim = np.zeros(N_sim); B_sim[-1] = 0.21
    P_sim = np.zeros(N_sim)
    P_sim_noise = np.zeros(N_sim)

    for i in range(N_sim):

        ph_sim[i] = ph0 + (ph_sim[i-1] - ph0)*(1 - theta_drift) + sigma_ph_drift*np.random.normal(0, 1)
        ph_sim_noise[i] = ph_sim[i] + np.random.normal(0, sigma_ph_sim)

        F_vibz[i] = sens_integral(delay, az_m[i], end, weight_vec) * Kz
        F_vibx[i] = sens_integral(delay, ax_m[i], end, weight_vec) * Kx
        F_viby[i] = sens_integral(delay, ay_m[i], end, weight_vec) * Ky

        alp[i] = alp_start[i % alp_amount]
        alp_vib[i] = alp[i] - F_vibz[i] / (2*np.pi*T*T) - F_vibx[i] / (2*np.pi*T*T) - F_viby[i] / (2*np.pi*T*T)
        Ph[i] = 2*np.pi*alp_vib[i]*T*T - ph_sim_noise[i]

        A_sim[i] = A_sim[i-1] + np.random.normal(0, dA_sim)
        B_sim[i] = B_sim[i-1] + np.random.normal(0, dB_sim)
        P_sim[i] = A_sim[i] - B_sim[i]*np.cos(Ph[i])

        az_m[i] = az_m[i] + np.random.normal(0, sigma_a, N_RP)  # измеренное ускорение
        ax_m[i] = ax_m[i] + np.random.normal(0, sigma_a, N_RP)  # измеренное ускорение
        ay_m[i] = ay_m[i] + np.random.normal(0, sigma_a, N_RP)  # измеренное ускорение
        P_sim_noise[i] = P_sim[i] + np.random.normal(0, sigma_A_sim)

    dph_sim = sigma_ph_drift
    dA_sim = np.sqrt(dA_sim**2 + DA_sim**2)

    out = (alp, P_sim_noise, P_sim, ph_sim_noise, ph_sim, A_sim, B_sim, dA_sim, dB_sim,
           dph_sim, sigma_A_sim, sigma_ph_sim, az_m, ax_m, ay_m)

    return out


 
def main():
    # ---------- интерферометр ----------
    T = 10e-3                  # длительность плеча, с
    tau = 20e-6                # длительность импульса, с
    T_RP = 33.556e-3           # запись акселерометра на сброс, с
    delay = 700                # смещение окна интерферометра, отсчёты
    # ---------- набор данных ----------
    alp_amount = 200           # точек alp в одном фринджe
    n_fringes = 10             # число фринджей
    # ---------- вибрация ----------
    vib_state = 'mooring'      # 'mooring' | 'sailing'
    Kz, Kx, Ky = 0.9, 0.003, 0.003
    sigma_a = 3e-5             # шум акселерометра, м/с^2
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
    print(f"N_sim = {N_sim}, память на ax/ay/az: {3*N_sim*N_RP*8/1e6:.0f} МБ")
 
    out = simul_wavefrom(N_sim, alp_amount, T_RP, T, tau, delay, Kz, Kx, Ky,
                         vib_state, sigma_a, Dph_drift, drift_corr,
                         sigma_A_sim, sigma_ph_sim, dA_sim, dB_sim, DA_sim)
 
    plot_waveform(out, T=T, tau=tau, T_RP=T_RP, delay=delay,
                  Kz=Kz, Kx=Kx, Ky=Ky, sigma_a=sigma_a, Dph_drift=Dph_drift,
                  title=f"{vib_state}, N_sim={N_sim}, sigma_a={sigma_a:g}, "
                        f"sigma_A={sigma_A_sim:g}, sigma_ph={sigma_ph_sim:g}")
    plt.show()
    return out
 
 
def plot_waveform(out, *, T, tau, T_RP, delay, Kz, Kx, Ky, sigma_a,
                  g0=9.8125, Dph_drift=None, n_bins=10, title=""):
    """
    Отображение результата simul_wavefrom (кортеж out из 15 элементов, в его порядке).
    ph_total = ph_sim + белый фазовый шум -- фаза, которая реально идёт в фринж;
    ph_sim -- чистый OU-дрейф (то, что оценивает EKF).
    Компенсация alp делается по ИЗМЕРЕННОМУ ускорению (a + шум), как с реальными данными.
    Рисует две фигуры: waveform/истина/фринж и разбор невязки по фазе.
    """
    (alp, P_noise, P_clean, ph_total, ph_sim, A_s, B_s, dA, dB, dph, sigma_A,
     sigma_ph, az_m, ax_m, ay_m) = out
    N_sim = len(alp)
 
    # веса интеграла чувствительности (как в simul_wavefrom)
    t_step = T_RP/N_RP
    t_range = np.linspace(0, 2*T + 4*tau, round((2*T + 4*tau)/t_step))
    end = len(t_range)
    fa_t = np.vectorize(fa, otypes=[float])(t_range, T, tau)
    trapz = np.full(end, t_step)
    trapz[0] *= 0.5
    trapz[-1] *= 0.5
    wv = fa_t*trapz
 
    # K*F по измеренным записям, рад
    Fx = Kx*keff*(ax_m[:, delay:delay + end] @ wv)
    Fy = Ky*keff*(ay_m[:, delay:delay + end] @ wv)
    Fz = Kz*keff*(az_m[:, delay:delay + end] @ wv)
    F_meas = Fx + Fy + Fz
 
    ph0 = g0*keff*T**2
    alp_comp = alp - F_meas/(2*np.pi*T**2)
 
    # ожидаемый фазовый шум компенсации от шума акселерометра и полный белый фазовый шум
    sigma_ph_vibr = (keff*t_step*sigma_a*np.sqrt(np.sum(fa_t**2))
                     *np.sqrt(Kx**2 + Ky**2 + Kz**2))
    sigma_ph_eff = np.hypot(sigma_ph, sigma_ph_vibr)
 
    wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi
    x_raw = wrap(2*np.pi*T**2*alp - ph0)
    x_comp = wrap(2*np.pi*T**2*alp_comp - ph0)
    xs = np.linspace(-np.pi, np.pi, 400)
    model = A_s.mean() - B_s.mean()*np.cos(xs)
    shots = np.arange(N_sim)
 
    # невязка относительно ИСТИННЫХ A, B и ЧИСТОГО дрейфа ph_sim (без белого фазового
    # шума): невязка содержит и шум детектирования, и белый фазовый шум
    Phi = 2*np.pi*T**2*alp_comp - ph_sim
    res = P_noise - (A_s - B_s*np.cos(Phi))
    phi_m = np.mod(Phi, np.pi)
    xk = B_s**2*np.sin(Phi)**2
    bins = np.minimum((phi_m/np.pi*n_bins).astype(int), n_bins - 1)
    b_ok = [b for b in range(n_bins) if np.sum(bins == b) > 5]
    v_b = np.array([np.mean(res[bins == b]**2) for b in b_ok])
    x_b = np.array([np.mean(xk[bins == b]) for b in b_ok])
    v_th = sigma_A**2 + sigma_ph_eff**2*x_b
 
    # ---------- сводка ----------
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    ph_noise = ph_total - ph_sim
    print("K*F по измеренному a, рад (RMS):  "
          f"x {rms(Fx):.2e}  y {rms(Fy):.2e}  z {rms(Fz):.2e}  сумма {rms(F_meas):.2e}")
    print("Заложено / ожидается:")
    print(f"  dA = {dA:.2e}, dB = {dB:.2e}, dph (шаг OU) = {dph:.2e} рад")
    print(f"  sigma_A = {sigma_A:.2e}")
    print(f"  sigma_ph (белый, заложен) = {sigma_ph:.2e} рад; "
          f"в данных std(ph_total - ph_sim) = {np.std(ph_noise):.2e} рад")
    print(f"  sigma_ph_vibr (от шума акселерометра) = {sigma_ph_vibr:.2e} рад")
    print(f"  sigma_ph эффективный = sqrt(sum квадратов) = {sigma_ph_eff:.2e} рад")
    # насколько фазовый шум вообще виден в невязке (по сравнению с sigma_A^2)
    share = sigma_ph_eff**2*x_b.max()/sigma_A**2
    print(f"Вклад фазового шума в дисперсию невязки: до {100*share:.2g}% от sigma_A² "
          f"(при max<B²sin²Ph> = {x_b.max():.2e})")
    # порог определимости sigma_ph по инновациям: вклад ~ B²/2*sigma_ph² сравним
    # с погрешностью дисперсии sqrt(2/N)*sigma_A²
    B_mean = float(np.mean(np.abs(B_s)))
    floor = sigma_A/B_mean*np.sqrt(2)*(2/N_sim)**0.25
    print(f"Порог определимости sigma_ph по {N_sim} сбросам (1 sigma) ≈ {floor:.2e} рад: "
          f"{'выше порога' if sigma_ph_eff > floor else 'НИЖЕ порога, автоподбор не определит'}")
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
 
    axs[1, 0].plot(shots, Fz, lw=0.6, label='Kz·Fz')
    axs[1, 0].plot(shots, Fx, lw=0.6, label='Kx·Fx')
    axs[1, 0].plot(shots, Fy, lw=0.6, label='Ky·Fy')
    axs[1, 0].set(xlabel='сброс #', ylabel='фаза, рад',
                  title='Вибрационная фаза (по измеренному a)')
    axs[1, 0].legend()
 
    axs[1, 1].plot(x_raw, P_noise, '.', ms=2, alpha=0.5)
    axs[1, 1].plot(xs, model, 'r-', lw=1)
    axs[1, 1].set(xlabel='Ph = 2πT²·alp − ph0 (mod 2π), рад', ylabel='P',
                  title='Фринж БЕЗ компенсации вибрации')
 
    axs[1, 2].plot(x_comp, P_noise, '.', ms=2, alpha=0.5)
    axs[1, 2].plot(xs, model, 'r-', lw=1)
    axs[1, 2].set(xlabel='Ph = 2πT²·alp_comp − ph0 (mod 2π), рад', ylabel='P',
                  title='Фринж С компенсацией по измеренному a')
 
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
    axs2[1].plot(x_b, v_th, 'r-', label='sigma_A² + sigma_ph_eff²·<B²sin²Ph>')
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