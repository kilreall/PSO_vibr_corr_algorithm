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
rcParams["legend.frameon"] = False


def fit_func(t, a):
    return a / np.sqrt(t)   # + b


# data loading
dt = 0.120 + 0.04965 + 0.004
dt = dt * 2
gp, gm = np.loadtxt('grav_data.txt', delimiter=' ', unpack=True, dtype=str)
start = 1
end = -1
gp = gp[start:end:1].astype(float)
gm = gm[start:end:1].astype(float)
grav = (gp + gm) / 2
points = 1000
cycles = len(grav) // points
grav = grav[:int(cycles*points)].astype(float)
grav = (grav - np.mean(grav)) * 1e5
grav = np.reshape(grav, (cycles, points))
grav = np.mean(grav, axis=1)
print(np.std(grav))

period = points * dt
print(f'period: {period}')
t = np.arange(period, (cycles + 1) * period, period) / 60 / 60

# plot
fig2, ax = plt.subplots(figsize=(8, 4))

ax.plot(t, grav, 'o-', ms=2)

ax.set_title('T = 12 ms, Rate = 8 Hz')
ax.set_xlabel('Integration Time, [min]')
ax.set_ylabel(r'$\Delta g$, [mGal]')
ax.grid(True)

plt.tight_layout()
plt.savefig('fig_g_longterm.png')
plt.show()
