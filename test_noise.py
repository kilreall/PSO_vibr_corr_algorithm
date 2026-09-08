import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)


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

plt.plot(scankp, normkp, label="exp")

# fit
P_exp = normkp
alp = scankp
poi = 201
p0 = [(np.max(P_exp)+np.min(P_exp))/2, (np.max(P_exp)-np.min(P_exp))/2, 0]
lb = [0, 0, 0]
ub = [1, 1, 2*np.pi]
popt, pcov = curve_fit(model, alp, P_exp, p0=p0, bounds=(lb, ub))
A, B, ph = popt
sigma_A, sigma_B, sigma_ph = np.sqrt(np.diag(pcov))
print(f"sigma avg A = {sigma_A}, sigma avg B = {sigma_B}, sigma avg ph = {sigma_ph}")

u = np.sqrt(len(alp))
print(f"sigma A = {sigma_A*u}, sigma B = {sigma_B*u}, sigma ph = {sigma_ph*u}")

plt.plot(alp, model(alp, A, B, ph), label="fit")
sigma_P = np.sqrt(np.std(P_exp-model(alp, A, B, ph)))
print(f"sigma P = {sigma_P}")
plt.legend()

plt.show()