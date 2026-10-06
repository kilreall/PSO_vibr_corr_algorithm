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
print(accel)

# plot
fig2 = plt.figure(figsize=(5, 5), layout='constrained')
axs = fig2.subplot_mosaic([['fringe'],
                           ['accel']])

axs['fringe'].plot(alpha, norm, 'o', ms=1)
axs['accel'].plot(accel[0], lw=1)

np.savez_compressed(r"raw data export test\gravimeter_data.npz", alp=alpha, P_exp=norm, az_m=accel)

plt.tight_layout()
print(f"length alp = {len(alpha)}")
# plt.savefig(f'fig_vibrational_noise_{delay}.png')
plt.show()
