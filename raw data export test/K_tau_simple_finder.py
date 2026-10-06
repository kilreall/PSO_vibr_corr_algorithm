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
    """Вибрационная фаза F(tau) для всех сбросов: KEFF * integral(a * fa) dt по Симпсону."""
    return KEFF * (az_m[:, delay:delay + n] @ wvec)


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
def find_K(Fz, alp, P_exp, K_bounds, T):
    """
    Грубая сетка по K (цель многоэкстремальна) -> уточнение bounded-минимизацией
    между соседями лучшего узла. Возвращает (K_opt, rms_opt).
    """
    phase0 = 2 * np.pi * T ** 2 * alp                   # фаза от чирпа без компенсации

    def obj(K):
        # alp_comp = alp - K*Fz/(2*pi*T^2)  =>  Phi = 2*pi*T^2*alp - K*Fz
        return global_cosfit(phase0 - K * Fz, P_exp)[0]

    n_grid = 201
    Ks = np.linspace(K_bounds[0], K_bounds[1], n_grid)
    vals = np.array([obj(K) for K in Ks])
    i = int(np.argmin(vals))

    lo = Ks[max(i - 1, 0)]
    hi = Ks[min(i + 1, n_grid - 1)]
    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-6))
    if r.fun < vals[i]:
        return float(r.x), float(r.fun)
    return float(Ks[i]), float(vals[i])

def K_tau_find(K_bounds, delay_bounds, alp, P_exp, az_m, T_RP, tau, T):
    t_step = T_RP / N_RP
    wvec, n = build_weight_vec(T, tau, t_step)

    hist = []  
    for delay_i in range(delay_bounds[0], delay_bounds[1], 5):
        Fz = vib_phase(az_m, delay_i, wvec, n)
        K_i, rms_i = find_K(Fz, alp, P_exp, K_bounds, T)
        hist.append((delay_i, K_i, rms_i))

    hist = np.array(hist)
    j = int(np.argmin(hist[:, 2]))                         # итоговый минимум по всем tau
    return hist[j, 1], int(hist[j, 0])



# Global constants
LM = 780e-9
KEFF = 4*np.pi/LM
N_RP = 16384

# data init

def load_q(path):
    z = np.load(path)
    q = np.concatenate([z["head"].astype(np.int64)[:, None],
                        z["d"].astype(np.int64)], axis=1).cumsum(axis=1)
    return z["alp"], z["P_exp"], q / float(z["scale"])

# data = np.load(r"raw data export test\gravimeter_data.npz")
# alp, P_exp, az_m = data["alp"], data["P_exp"], data["az_m"]
alp, P_exp, az_m = load_q(r"raw data export test\gravimeter_data_q0.npz")
T_RP = 33.556e-3


# QG params
T = 10e-3
tau = 5.1e-6

# search param
delay_bounds = [0, N_RP - round((2*T+4*tau)/T_RP*N_RP) - 1]
K_bounds = [-0.1, 0.1]


K_opt, delay_opt = K_tau_find(K_bounds, delay_bounds, alp, P_exp, az_m, T_RP, tau, T)

print(f"Optimal K ={K_opt}")
print(f"Optimal delay ={delay_opt}")

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


plot_fit(K_opt, delay_opt, alp, P_exp, az_m, T_RP, tau, T)



