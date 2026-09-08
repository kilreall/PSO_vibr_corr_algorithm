import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def kalmanFit3(alp, P_exp):

    # param init
    A = np.zeros(len(alp))
    B = np.zeros(len(alp))
    ph = np.zeros(len(alp))
    e = np.zeros(len(alp))
    P_m = np.zeros(len(alp))

    # matrix init
    P_cov = np.diag([0.0022, 0.003*2, 0.11**2])

    Q = np.diag([0, 0, 0])


    R = 0.047**2

    # init values
    poi = 201
    p0 = [(np.max(P_exp[:poi])+np.min(P_exp[:poi]))/2, (np.max(P_exp[:poi])-np.min(P_exp[:poi]))/2, 0]
    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]
    popt, pcov = curve_fit(model, alp[:poi], P_exp[:poi], p0=p0, bounds=(lb, ub))
    A[0], B[0], ph[0] = popt
    P_m[0] = model(alp[0], A[0], B[0], ph[0])


    for i in range(1, len(alp)):

        # Jacobian
        Ph_i = 2*np.pi*alp[i]*T**2 - ph[i-1]
        H = np.array([1, np.cos(Ph_i), B[i-1]*np.sin(Ph_i)])

        # prediction step
        P_cov = P_cov + Q 

        # model prediction
        P_m[i] = A[i-1] + B[i-1]*np.cos(Ph_i)

        # mistake
        e[i] = P_exp[i] - P_m[i]

        # measurement covariance/uncertinty
        S_i = H @ P_cov @ H.T + R

        
        # Kalman gain
        K_i = P_cov @ H.T / S_i    
        
        # update
        A[i] = A[i-1] + K_i[0]*e[i]
        B[i] = B[i-1] + K_i[1]*e[i]
        ph[i] = ph[i-1] + K_i[2]*e[i]

        P_cov = (np.eye(3) - np.outer(K_i, H)) @ P_cov

        # I = np.eye(3)
        # KH = np.outer(K_i, H)

        # P_cov = ((I - KH) @ P_cov @ (I - KH).T + np.outer(K_i, K_i) * R)


    return P_m, A, B, ph

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

plt.plot(scankp, normkp, label="exp")

# kalman fit
alp = scankp
P_exp = normkp
P_m, A, B, ph = kalmanFit3(alp, P_exp)
plt.plot(alp, model(alp, A, B, ph), label="kalman")
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