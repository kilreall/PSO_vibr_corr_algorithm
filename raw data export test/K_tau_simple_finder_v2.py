import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# sens func
def fa(t, T, tau):
    if 0 < t <= T+2*tau:
        return t
    elif T+2*tau < t <= 2*T+4*tau:
        return 2*(T+2*tau)-t
    else:
        return 0
fat_v = np.vectorize(fa, otypes=[float])   # <-- КРИТИЧНО: явно задать float


def simpson_weights(n, h):
    """Веса составной формулы Симпсона: h/3 * [1, 4, 2, 4, ..., 2, 4, 1]. n нечётное."""
    if n % 2 == 0:
        raise ValueError("для Симпсона нужно нечётное число узлов")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0

def build_weight_vec(T, tau, t_step):
    """
    Вектор w_j = fa(t_j) * simpson_w_j. Тогда
        интеграл a(t)*fa(t) dt  ~  az_window @ w   (Симпсон по узлам акселерометра).
    """
    duration = 2 * T + 4 * tau
    n = int(round(duration / t_step)) + 1
    if n % 2 == 0:
        n += 1                              # fa = 0 за пределами окна, лишний узел безвреден
    t = np.arange(n) * t_step
    return fat_v(t, T, tau) * simpson_weights(n, t_step), n

def vib_phase(az_m, delay, wvec, n):
    """Вибрационная фаза по сбросам, KEFF * integral(a*fa) dt (Симпсон). Среднее вычтено."""
    F = KEFF * (az_m[:, delay:delay + n] @ wvec)
    #F = F[np.random.permutation(len(F))] # случайное перемешивание
    return F - F.mean()


# ------------------------------------------------------------------
# Критерий: один cos-fit по ВСЕМ данным
# ------------------------------------------------------------------
def global_cosfit(phase, y):
    """
    y = A - B*cos(phase - ph)  <=>  y = A + c1*cos(phase) + c2*sin(phase).
    Линейный МНК по всем точкам. Возвращает (RMS остатков, A, B, ph).
    """
    X = np.column_stack((np.ones_like(phase), np.cos(phase), np.sin(phase)))
    coef = np.linalg.solve(X.T @ X, X.T @ y)            # нормальные уравнения 3x3
    r = y - X @ coef
    A, c1, c2 = coef
    return float(np.sqrt(np.mean(r ** 2))), A, np.hypot(c1, c2), np.arctan2(-c2, -c1)



# ------------------------------------------------------------------
# Поиск K при фиксированном delay
# ------------------------------------------------------------------
def find_K(Fz, alp, P_exp, kappa_max, T, n_grid=401):
    """
    Сетка по kappa = K*std(Fz) (рад) -> bounded-уточнение вокруг лучшего узла.
    Единицы акселерометра не важны. Возвращает (K_opt, rms_opt).
    """
    phase0 = 2 * np.pi * T ** 2 * alp

    def obj(K):
        return global_cosfit(phase0 - K * Fz, P_exp)[0]

    s = float(Fz.std())
    if not s > 0:
        return 0.0, obj(0.0)

    kap = np.linspace(-kappa_max, kappa_max, n_grid)
    vals = np.array([obj(k / s) for k in kap])
    i = int(np.argmin(vals))

    lo, hi = kap[max(i - 1, 0)] / s, kap[min(i + 1, n_grid - 1)] / s
    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-4 / s))        # точность 1e-4 рад по kappa
    if r.fun < vals[i]:
        return float(r.x), float(r.fun)
    return float(kap[i] / s), float(vals[i])

def K_tau_find(kappa_max, delay_bounds, alp, P_exp, az_m, T_RP, tau, T,
               step=5, verbose=True):
    wvec, n = build_weight_vec(T, tau, T_RP / N_RP)
    d_hi = min(delay_bounds[1], az_m.shape[1] - n + 1)       # окно не выходит за запись

    hist = []
    for delay_i in range(delay_bounds[0], d_hi, step):
        Fz = vib_phase(az_m, delay_i, wvec, n)
        K_i, rms_i = find_K(Fz, alp, P_exp, kappa_max, T)
        hist.append((delay_i, K_i, rms_i))
        if verbose and len(hist) % 10 == 1:
            print(f"delay={delay_i:5d}  K={K_i:+.4e}  RMS={rms_i:.5e}")

    hist = np.array(hist)
    j = int(np.argmin(hist[:, 2]))                           # итоговый минимум по всем delay
    return hist[j, 1], int(hist[j, 0]), hist



# Global constants
LM = 780e-9
KEFF = 4*np.pi/LM
N_RP = 16384

# data init
data = np.load(r"raw data export test\gravimeter_data.npz")
alp, P_exp, az_m = data["alp"], data["P_exp"], data["az_m"]
T_RP = 33.556e-3


# QG params
T = 10e-3
tau = 5.1e-6

# search param
delay_bounds = [0, N_RP - round((2*T+4*tau)/T_RP*N_RP) - 1]
kappa_max = 1.5


K_opt, delay_opt, hist = K_tau_find(kappa_max, delay_bounds, alp, P_exp, az_m, T_RP, tau, T, step=5)

d0 = max(delay_opt - 25, delay_bounds[0])
d1 = min(delay_opt + 26, delay_bounds[1])
K_opt, delay_opt, hist_f = K_tau_find(kappa_max, [d0, d1], alp, P_exp, az_m,
                                      T_RP, tau, T, step=1, verbose=False)

print(f"Optimal K     = {K_opt:.6e}")
print(f"Optimal delay = {delay_opt}")


# fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
# ax[0].plot(hist[:, 0], hist[:, 2])
# rms0 = global_cosfit(2*np.pi*T**2*alp, P_exp)[0]
# ax[0].axhline(rms0, color="gray", ls=":", label="K = 0")   # rms0 из diagnostics
# ax[0].axvline(delay_opt, color="r", ls="--")
# ax[0].set_ylabel("RMS при оптимальном K")
# ax[0].legend()
# ax[1].plot(hist[:, 0], hist[:, 1])
# ax[1].axvline(delay_opt, color="r", ls="--")
# ax[1].set_ylabel("K_opt(delay)")
# ax[1].set_xlabel("delay, отсчёты")
# plt.tight_layout()
# plt.show()


def diagnostics(delay, alp, P_exp, az_m, T_RP, tau, T):
    wvec, n = build_weight_vec(T, tau, T_RP / N_RP)
    Fz = vib_phase(az_m, delay, wvec, n)
    phase0 = 2 * np.pi * T ** 2 * alp
    rms0 = global_cosfit(phase0, P_exp)[0]
    print(f"std(P_exp)            = {P_exp.std():.4e}")
    print(f"RMS cos-fit при K = 0 = {rms0:.4e}  (отношение к std(P_exp): {rms0 / P_exp.std():.2f})")
    print(f"размах Φ0             = {np.ptp(phase0) / (2 * np.pi):.2f} оборотов")
    print(f"std(Fz)               = {Fz.std():.4e} рад на единицу K  (K ~ {1 / Fz.std():.2e} даёт 1 рад)")


def plot_K_scan(delay, alp, P_exp, az_m, T_RP, tau, T, kappa_max=20.0, n_pts=2001):
    """RMS cos-fit как функция kappa = K*std(Fz); K = kappa / std(Fz)."""
    wvec, n = build_weight_vec(T, tau, T_RP / N_RP)
    Fz = vib_phase(az_m, delay, wvec, n)
    s = Fz.std()
    phase0 = 2 * np.pi * T ** 2 * alp
    kap = np.linspace(-kappa_max, kappa_max, n_pts)
    rms = [global_cosfit(phase0 - (k / s) * Fz, P_exp)[0] for k in kap]
    plt.figure(figsize=(9, 3.5))
    plt.plot(kap, rms)
    plt.xlabel("κ = K·std(Fz), рад")
    plt.ylabel("RMS cos-fit")
    plt.title(f"delay = {delay}, std(Fz) = {s:.3e}")
    plt.tight_layout()
    plt.show()


def plot_fit(K, delay, alp, P_exp, az_m, T_RP, tau, T):
    """
    Рисует cos-fit до и после компенсации вибраций при заданных K и delay.
    По оси X -- фаза Phi mod 2*pi, по Y -- сигнал; линия -- A - B*cos(Phi - ph).
    """
    wvec, n = build_weight_vec(T, tau, T_RP / N_RP)
    if delay + n > az_m.shape[1]:
        raise ValueError(f"delay + n = {delay + n} > N_acc = {az_m.shape[1]}")

    Fz = vib_phase(az_m, delay, wvec, n)
    phase0 = 2 * np.pi * T ** 2 * alp                  # без компенсации
    phase_c = phase0 - K * Fz                          # с компенсацией

    phi_grid = np.linspace(0, 2 * np.pi, 400)
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cmp = fig.add_subplot(gs[0, 1], sharey=ax_raw)
    ax_res = fig.add_subplot(gs[1, :])

    rms_c = None
    for ax, phase, title in ((ax_raw, phase0, "без компенсации (K = 0)"),
                             (ax_cmp, phase_c, f"компенсация: K = {K:.4f}, delay = {delay}")):
        rms, A, B, ph = global_cosfit(phase, P_exp)
        ax.plot(np.mod(phase, 2 * np.pi), P_exp, ".", ms=3, alpha=0.5, label="данные")
        ax.plot(phi_grid, A - B * np.cos(phi_grid - ph), "r-", lw=2, label="fit")
        ax.set_title(f"{title}\nRMS = {rms:.4e}, A = {A:.3f}, B = {B:.3f}, ph = {ph:.3f}")
        ax.set_xlabel("Φ mod 2π, рад")
        ax.set_xlim(0, 2 * np.pi)
        ax.legend()
        if ax is ax_cmp:
            rms_c, A_c, B_c, ph_c = rms, A, B, ph
    ax_raw.set_ylabel("P_exp")

    # остатки скомпенсированного фита по сбросам
    resid = P_exp - (A_c - B_c * np.cos(phase_c - ph_c))
    ax_res.plot(resid, ".", ms=3)
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_xlabel("номер сброса")
    ax_res.set_ylabel("остаток")
    ax_res.set_title(f"остатки после компенсации (RMS = {rms_c:.4e})")

    plt.tight_layout()
    plt.show()

def plot_fit(K, delay, alp, P_exp, az_m, T_RP, tau, T, n_show=None):
    wvec, n = build_weight_vec(T, tau, T_RP / N_RP)
    if delay + n > az_m.shape[1]:
        raise ValueError(f"delay + n = {delay + n} > N_acc = {az_m.shape[1]}")
    Fz = vib_phase(az_m, delay, wvec, n)
    phase0 = 2 * np.pi * T ** 2 * alp

    cases = (("до компенсации (K = 0)", phase0, "tab:blue"),
             (f"после: K = {K:.4g}, delay = {delay}", phase0 - K * Fz, "tab:red"))

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_fit = [fig.add_subplot(gs[0, 0])]
    ax_fit.append(fig.add_subplot(gs[0, 1], sharey=ax_fit[0]))
    ax_res = fig.add_subplot(gs[1, :])

    phi = np.linspace(0, 2 * np.pi, 400)
    sl = slice(0, n_show)                                  # n_show=None -> все сбросы
    shots = np.arange(len(P_exp))[sl]

    for ax, (label, phase, color) in zip(ax_fit, cases):
        rms, A, B, ph = global_cosfit(phase, P_exp)
        ax.plot(np.mod(phase, 2 * np.pi), P_exp, ".", ms=3, alpha=0.5,
                color=color, label="данные")
        ax.plot(phi, A - B * np.cos(phi - ph), "k-", lw=2, label="fit")
        ax.set_title(f"{label}\nRMS = {rms:.4e}, A = {A:.3f}, B = {B:.3f}, ph = {ph:.3f}")
        ax.set_xlabel("Φ mod 2π, рад")
        ax.set_xlim(0, 2 * np.pi)
        ax.legend()

        resid = P_exp - (A - B * np.cos(phase - ph))
        ax_res.plot(shots, resid[sl], "-o", ms=3, lw=0.8, alpha=0.8,
                    color=color, label=f"{label}: RMS = {rms:.4e}")

    ax_fit[0].set_ylabel("P_exp")
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_xlabel("номер сброса")
    ax_res.set_ylabel("невязка")
    ax_res.legend()
    plt.tight_layout()
    plt.show()


diagnostics(delay_opt, alp, P_exp, az_m, T_RP, tau, T)
plot_K_scan(delay_opt, alp, P_exp, az_m, T_RP, tau, T)
plot_fit(K_opt, delay_opt, alp, P_exp, az_m, T_RP, tau, T, n_show=300)