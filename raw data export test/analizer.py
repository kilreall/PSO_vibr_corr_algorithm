import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import allantools
from matplotlib import rcParams


# plot settings
rcParams['font.size'] = 14
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams['savefig.dpi'] = 300
rcParams["legend.frameon"] = True

# data loading
data = np.loadtxt(f'raw data export test\data_T_10_ms_delay_200_ms.txt', delimiter=',', unpack=False, skiprows=0)
alpha = data[:, 0]
norm = data[:, 1]
accel = data[:, 2:]
#print(accel)

# plot
fig2 = plt.figure(figsize=(5, 5), layout='constrained')
axs = fig2.subplot_mosaic([['fringe'],
                           ['accel']])

axs['fringe'].plot(alpha, norm, 'o', ms=1)
draw_accel = np.round(accel[0], 0)
axs['accel'].plot(draw_accel, lw=1)

# сохранение в npz

SCALE = 1e2                      # 4 знака после запятой
path = r"raw data export test\gravimeter_data_q2.npz"

q = np.rint(accel * SCALE).astype(np.int64)      # целые кванты 1e-2
d = np.diff(q, axis=1)                           # разности соседних отсчётов
assert np.abs(d).max() < 2**15, "разности не влезают в int16"
assert np.abs(q[:, 0]).max() < 2**31

np.savez_compressed(path, alp=alpha, P_exp=norm,
                    d=d.astype(np.int16), head=q[:, 0].astype(np.int32),
                    scale=SCALE)

plt.tight_layout()
print(f"length alp = {len(alpha)}")
print(f"acc max = {np.max(accel)}")
# plt.savefig(f'fig_vibrational_noise_{delay}.png')

# check save comperssion


LM = 780e-9
KEFF = 4*np.pi/LM
N_RP = 16384

def load_q(path):
    z = np.load(path)
    q = np.concatenate([z["head"].astype(np.int64)[:, None],
                        z["d"].astype(np.int64)], axis=1).cumsum(axis=1)
    return z["alp"], z["P_exp"], q / float(z["scale"])

alp, P_exp, az_m = load_q(r"raw data export test\gravimeter_data_q2.npz")


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

def check_rounding(accel, alp, P_exp, K, delay, T, ty, T_RP, scale=1e4):
    wv, n = build_weight_vec(T, ty, T_RP / N_RP)
    acc_q = np.rint(accel * scale) / scale
    F0 = vib_phase(accel, delay, wv, n)
    F1 = vib_phase(acc_q, delay, wv, n)
    print(f"rms(K·ΔF) = {(K * (F1 - F0)).std():.2e} рад   (нужно << 0.07)")
    ph0 = 2 * np.pi * T**2 * alp
    for name, F in (("исходные", F0), ("округлённые", F1)):
        print(f"{name:12s} RMS cos-fit = {global_cosfit(ph0 - K * F, P_exp)[0]:.5e}")

check_rounding(accel, alpha, norm, K=-0.00031867653469475044, delay=4840,
               T=10e-3, ty=5.1e-6, T_RP=33.556e-3)


plt.show()
