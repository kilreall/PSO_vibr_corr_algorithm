import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def kalmanFit2(alp, P_exp):

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
    p0 = [(np.max(P_exp)+np.min(P_exp))/2, (np.max(P_exp)-np.min(P_exp))/2, 0]
    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]
    popt, pcov = curve_fit(model, alp, P_exp, p0=p0, bounds=(lb, ub))
    A[0], B[0], ph[0] = popt
    P_m[0] = model(alp[0], A[0], B[0], ph[0])

    # sigma points
    n = 3
    alp_ukf = 1e-3
    Beta = 2
    kappa = 0
    lambd = alp_ukf**2*(kappa + n) - n
    prop = np.sqrt(n+lambd)

    for i in range(1, len(alp)):

        # cholskiy разложение P = L*L^T L = chol(P)
        L = np.linalg.cholesky(P_cov)
        L = L.T

        # определение точек около основной
        hi = np.zeros((2*n+1,3))
        hi[0]  = np.array([A[i-1], B[i-1], ph[i-1]])
        for j in range(1, 4):
            hi[j] = hi[0] + prop*L[j-1]
            hi[j+n] = hi[0] - prop*L[j-1]
        

        y = np.zeros(2*n+1)
        for j in range(2*n+1):
            y[j] = model(alp[i], hi[j,0], hi[j,1], hi[j,2])

        # определение предсказания экспериментального измерения
        Wm = np.ones(2*n + 1) / (2*(n + lambd))
        Wc = np.ones(2*n + 1) / (2*(n + lambd))

        Wm[0] = lambd / (n + lambd)
        Wc[0] = lambd / (n + lambd) + (1 - alp_ukf**2 + Beta)



        for j in range(2*n+1):
            P_m[i] += Wm[j]*y[j]

        # measurement covariance/uncertinty
        S_i = R
        for j in range(2*n+1):
            S_i += Wc[j]*(P_m[i] - y[j])**2

        # cross covariance
        C_i = np.zeros(3)
        
        for j in range(2*n+1):
            C_i += Wc[j]*(hi[j]-hi[0])*(y[j] - P_m[i]) # тут hi[0]?
        
        # Kalman gain
        K_i = C_i/S_i        
        
        e[i] = P_exp[i] - P_m[i]

        A[i] = A[i-1] + K_i[0]*e[i]
        B[i] = B[i-1] + K_i[1]*e[i]
        ph[i] = ph[i-1] + K_i[2]*e[i]

        P_cov = P_cov - S_i * np.outer(K_i, K_i)
        P_cov = P_cov + Q

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
P_m, A, B, ph = kalmanFit2(alp, P_exp)
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