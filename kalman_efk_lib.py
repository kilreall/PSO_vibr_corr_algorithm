# extended jacobin kalman filter without vibrations based on filterpy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter


def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A + B*np.cos(Phi)

    return np.array([P])

def HJacobian(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    H = np.array([
        [1.0,
         np.cos(Phi),
         B*np.sin(Phi)]
    ])

    return H

def kalmanFit3(alp, P_exp):

    N = len(alp)

    # --------------------------------------------------
    # Output arrays
    # --------------------------------------------------

    A = np.zeros(N)
    B = np.zeros(N)
    ph = np.zeros(N)
    P_m = np.zeros(N)
    e = np.zeros(N)
    P_cov = np.zeros((N, 3, 3))

    # --------------------------------------------------
    # Initial fit
    # --------------------------------------------------

    poi = 201

    p0 = [
        (np.max(P_exp[:poi]) + np.min(P_exp[:poi])) / 2,
        (np.max(P_exp[:poi]) - np.min(P_exp[:poi])) / 2,
        0
    ]

    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]

    popt, pcov = curve_fit(
        model,
        alp[:poi],
        P_exp[:poi],
        p0=p0,
        bounds=(lb, ub)
    )

    A[0], B[0], ph[0] = popt

    P_m[0] = model(
        alp[0],
        A[0],
        B[0],
        ph[0]
    )

    # uncertainties from initial curve fit

    sigma_P = np.std(P_exp[:poi] - model(alp[:poi], A[0], B[0], ph[0]))

    # --------------------------------------------------
    # Create EKF
    # --------------------------------------------------

    ekf = ExtendedKalmanFilter(
        dim_x=3,
        dim_z=1
    )

    # state = [A, B, phi]
    ekf.x = np.array([
        A[0],
        B[0],
        ph[0]
    ])

    # initial covariance
    ekf.P =  pcov.copy()

    # process noise
    ekf.Q = np.diag([
        1e-8,
        1e-8,
        1e-6
    ])

    # measurement noise
    ekf.R = np.array([
        [sigma_P**2]
    ])

    # --------------------------------------------------
    # Main cycle
    # --------------------------------------------------

    for i in range(1, N):

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        ekf.predict()

        # predicted measurement
        P_m[i] = Hx(
            ekf.x,
            alp[i],
            T
        )[0]

        # innovation
        e[i] = P_exp[i] - P_m[i]

        # ----------------------------------------------
        # Measurement update
        # ----------------------------------------------

        ekf.update(
            np.array([P_exp[i]]),
            HJacobian,
            Hx,
            args=(alp[i], T),
            hx_args=(alp[i], T)
        )


        # ----------------------------------------------
        # Save state
        # ----------------------------------------------

        A[i] = ekf.x[0]
        B[i] = ekf.x[1]
        ph[i] = ekf.x[2]


        # innovation
        e[i] = P_exp[i] - P_m[i]

        # covariance
        P_cov[i] = ekf.P

    return P_m, A, B, ph, P_cov

plt.figure()

# test experimental data
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
P_m, A, B, ph, P_cov = kalmanFit3(alp, P_exp)
plt.plot(model(alp, A, B, ph), label="kalman")
plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
plt.legend()

# additional graphics
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(A)
axs[0].set_title('A')
print(A)

# Второй график (верхний правый)
axs[1].plot(B)
axs[1].set_title('B')

# Третий график (нижний левый)
axs[2].plot(ph)  # или axs[1, 0]
axs[2].set_title('$\phi$')

fig1, axss = plt.subplots(1, 3, figsize=(14, 4))
axss[0].plot(np.sqrt(P_cov[:, 0, 0]))
axss[0].set_title('dA')

# Второй график (верхний правый)
axss[1].plot(np.sqrt(P_cov[:, 1, 1]))
axss[1].set_title('dB')

# Третий график (нижний левый)
lm = 780e-9
keff = 4*np.pi/lm
axss[2].plot(np.sqrt(P_cov[:, 2, 2]))  # или axs[1, 0]
axss[2].set_title('$d\phi$')

plt.show()