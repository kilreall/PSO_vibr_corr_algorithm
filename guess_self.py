# kalman analog with coef guess with vibrations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def kalmanFit1(alp, P_exp, n_A, n_B, n_ph): # K guess alghorythm

    # param init
    A = np.zeros(len(alp))
    B = np.zeros(len(alp))
    ph = np.zeros(len(alp))
    e = np.zeros(len(alp))

    # init coef
    poi = 201
    p0 = [(np.max(P_exp)+np.min(P_exp))/2, (np.max(P_exp)-np.min(P_exp))/2, 0]
    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]
    popt, pcov = curve_fit(model, alp[:poi], P_exp[:poi], p0=p0, bounds=(lb, ub))
    A[0], B[0], ph[0] = popt


    for i in range(1, len(alp)):

        e[i] = P_exp[i] - model(alp[i], A[i-1], B[i-1])

        Ph_i = 2*np.pi*alp[i]*T**2 - ph[i-1]
        D_i = 1 + np.cos(Ph_i)**2 + B[i-1]**2*np.sin(Ph_i)**2

        A[i] = A[i-1] + n_A*e[i]/D_i
        B[i] = B[i-1] + n_B*np.cos(Ph_i)*e[i]/D_i
        ph[i] = ph[i-1] + n_ph*B[i-1]*np.sin(Ph_i)*e[i]/D_i



    return 1

plt.figure()

# exp
T = 5e-3
kp = r"test_no_vibr.csv"
poi = 201
sum1kp, sum2kp, normkp, scankp = np.loadtxt(kp, delimiter=',', skiprows=1, unpack=True)
# sum1kp = sum1kp[:len(sum1kp)//poi*poi]
# sum2kp = sum2kp[:len(sum2kp)//poi*poi]
# normkp = normkp[:len(normkp)//poi*poi]
# scankp = scankp[:len(scankp)//poi*poi]
# cycleskp = len(normkp) // poi
# scankp = scankp.reshape((cycleskp, poi))
# normkp = normkp.reshape((cycleskp, poi))
# scankp = np.mean(scankp, axis=0)
# normkp = np.mean(normkp, axis=0)

plt.plot(normkp, label="exp")

# kalman fit
alp = scankp
P_exp = normkp
P_m, A, B, ph = kalmanFit1(alp, P_exp)
plt.plot(model(alp, A, B, ph), label="kalman")
plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
plt.legend()

#plt.figure()
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(A)
axs[0].set_title('A')

# Второй график (верхний правый)
axs[1].plot(B)
axs[1].set_title('B')

# Третий график (нижний левый)
axs[2].plot(ph)  # или axs[1, 0]
axs[2].set_title('$\phi$')

plt.show()