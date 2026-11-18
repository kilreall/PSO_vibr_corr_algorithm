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

T = 2*np.pi*0.01**2*1e6          # коэффициент в их модели

def fit_func(t, A, B, phi):
    return A*np.cos(T*t + phi) + B


# ============================================================
# Скользящий фит по 5 точкам → g(t)
# ============================================================
def sliding_fit_g(corr, norm, T_coef, window=1, lm=780e-9, T_int=0.01):
    """
    corr  - ось (уже с коррекцией вибраций)
    norm  - нормализованный сигнал
    T_coef - тот самый T = 2π·T_int²·1e6 из файла
    """
    N = len(corr)
    keff = 4 * np.pi / lm

    g  = np.full(N, np.nan)
    A  = np.full(N, np.nan)
    B  = np.full(N, np.nan)
    ph = np.full(N, np.nan)
    ok = np.zeros(N, dtype=bool)

    half = window // 2

    for i in range(half, N - half, 1):
        sl = slice(i - half, i + half + 1)
        x = corr[sl]
        y = norm[sl]

        A0 = (np.max(y) - np.min(y)) / 2
        B0 = np.mean(y)
        # грубая оценка фазы по минимуму в окне
        idx_min = np.argmin(y)
        phi0 = -T_coef * x[idx_min] + np.pi          # чтобы cos ≈ -1

        try:
            popt, _ = curve_fit(
                fit_func,
                x, y,
                p0=[A0, B0, phi0],
                bounds=([0, -1.5, -np.inf], [1.5, 1.5, np.inf]),
                maxfev=8000
            )
            Ai, Bi, phi = popt

            # приводим фазу к правильной ветви
            phi_target = -T_coef * x[idx_min] + np.pi
            M = np.round((phi_target - phi) / (2 * np.pi))
            phi += 2 * np.pi * M

            A[i]  = Ai
            B[i]  = Bi
            ph[i] = phi
            g[i]  = phi / (keff * T_int**2)          # реальный g
            ok[i] = True

        except Exception:
            pass

    return g, A, B, ph, ok


# ============================================================
# Основной код обработки одного delay
# ============================================================
cut = 6500
delay = 800
sum1, sum2, norm, vibr, scan = np.loadtxt(
    f'data_delay_{delay}.csv', delimiter=',', unpack=True, skiprows=1, max_rows=1000
)

a = 9
scan *= 1e-6
vibr *= 1e-6

cycles = np.sum(scan == np.amax(scan))
points = np.where(scan == np.amax(scan))[0][0] + 1
print(cycles, points)

norm = norm[:int(cycles*points)]
scan = scan[:int(cycles*points)]
vibr = vibr[:int(cycles*points)]

norm = np.reshape(norm, (cycles, points))
scan = np.reshape(scan, (cycles, points))
vibr = np.reshape(vibr, (cycles, points))

corr = np.zeros((cycles, points))
for i in range(cycles):
    corr[i] = scan[i] + a * vibr[i]

    # нормализация каждого цикла
    guess = [abs(np.amax(norm[i])-np.amin(norm[i]))/2,
             np.mean(norm[i]),
             np.pi]
    popt, pcov = curve_fit(fit_func, corr[i], norm[i], p0=guess, maxfev=10000)
    norm[i] -= popt[1]
    norm[i] /= popt[0]

norm = np.ravel(norm)
scan = np.ravel(scan)
corr = np.ravel(corr)

# глобальный фит (как было)
guess = [abs(np.amax(norm)-np.amin(norm))/2,
         np.mean(norm),
         np.pi]
popt, pcov = curve_fit(fit_func, corr, norm, p0=guess, maxfev=10000)
fit_x = np.linspace(np.amin(corr), np.amax(corr), 1000)
fit_y = fit_func(fit_x, *popt)
perr = np.sqrt(np.diag(pcov))
print("Global phi_err =", perr[-1])

# --- скользящий фит по 5 точкам ---
window = 5
g_slide, A_s, B_s, ph_s, ok = sliding_fit_g(corr, norm, T, window=window, T_int=0.01)
g_valid = g_slide[ok]
# сохраняем
np.save(f"g_slide_delay{delay}_w{window}.npy", g_slide)
# np.save(f"ph_slide_delay{delay}_w{window}.npy", ph_s)
# np.save(f"ok_slide_delay{delay}_w{window}.npy", ok)

print(f"Сохранено g_slide_delay{delay}_w{window}.npy")
print(f"Успешных точек: {np.sum(ok)} / {len(ok)}")

# ============================================================
# графики
# ============================================================
fig2 = plt.figure(figsize=(5, 5), layout='constrained')
axs = fig2.subplot_mosaic([['before'],
                           ['after']])

axs['before'].plot(scan, norm, 'o', ms=1, lw=0.1)
axs['before'].set_xlim([fit_x[0], fit_x[-1]])

axs['after'].plot(corr, norm, 'o', ms=1)
axs['after'].plot(fit_x, fit_y, '-')
axs['after'].set_xlim([fit_x[0], fit_x[-1]])
axs['after'].set_title(
    rf"$\sigma_\phi={round(perr[-1], 5)}$ rad, "
    rf"$\sigma_g={int(perr[-1] / (2*np.pi) * 400*1e3)}$ $\mu$Gal, $N={len(norm)}$",
    fontsize=12
)

plt.tight_layout()
plt.savefig(f'fig_vibrational_noise_{delay}.png')
plt.show()

# дополнительный график g(t)
plt.figure(figsize=(10, 4))
plt.plot(g_slide * 1e5, 'o-', ms=3, label=f'sliding window={window}')
plt.ylabel(r'$g$ ($10^{-5}$ m/s²)')
plt.legend()
plt.grid(True)
plt.title(f"g(t) from {window}-point sliding fit, delay={delay}")
plt.tight_layout()
plt.show()