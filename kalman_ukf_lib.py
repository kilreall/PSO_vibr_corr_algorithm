# unscented kalman filter without vibrations based on lib filterpy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

def Hx(x, alpha, T):

    A = x[0]
    B = x[1]
    ph = x[2]

    Phi = 2*np.pi*alpha*T**2 - ph

    P = A + B*np.cos(Phi)

    return np.array([P])


def Fx(x, dt):

    # A, B, phi считаем постоянными
    return x


def kalmanFit_UKF(alp, P_exp):

    N = len(alp)

    # --------------------------------------------------
    # Output arrays
    # --------------------------------------------------

    A = np.zeros(N)
    B = np.zeros(N)
    ph = np.zeros(N)

    e = np.zeros(N)
    P_m = np.zeros(N)

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

    # uncertainty from initial fit

    sigma_P = np.std(P_exp[:poi] - model(alp[:poi], A[0], B[0], ph[0]))

    # --------------------------------------------------
    # UKF parameters
    # --------------------------------------------------

    n = 3

    alpha_ukf = 1e-3
    beta_ukf = 2
    kappa_ukf = 0

    points = MerweScaledSigmaPoints(
        n=n,
        alpha=alpha_ukf,
        beta=beta_ukf,
        kappa=kappa_ukf
    )

    ukf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=1,
        dt=1.0,
        fx=Fx,
        hx=Hx,
        points=points
    )

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    ukf.x = np.array([
        A[0],
        B[0],
        ph[0]
    ])

    # --------------------------------------------------
    # Initial covariance
    # --------------------------------------------------

    ukf.P = pcov.copy()

    # --------------------------------------------------
    # Process noise
    # --------------------------------------------------

    ukf.Q = np.diag([
        1e-8,
        1e-8,
        1e-6
    ])

    # --------------------------------------------------
    # Measurement noise
    # --------------------------------------------------

    ukf.R = np.array([
        [sigma_P**2]
    ])

    # --------------------------------------------------
    # Main cycle
    # --------------------------------------------------

    for i in range(1, N):

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        ukf.predict()

        # ----------------------------------------------
        # Measurement prediction
        # ----------------------------------------------

        P_m[i] = Hx(
            ukf.x,
            alp[i],
            T
        )[0]

        # ----------------------------------------------
        # Innovation
        # ----------------------------------------------

        e[i] = P_exp[i] - P_m[i]

        # ----------------------------------------------
        # Measurement update
        # ----------------------------------------------

        ukf.update(
            np.array([P_exp[i]]),
            alpha=alp[i],
            T=T
        )

        # ----------------------------------------------
        # Save state
        # ----------------------------------------------

        A[i] = ukf.x[0]
        B[i] = ukf.x[1]
        ph[i] = ukf.x[2]

        # ----------------------------------------------
        # Save covariance
        # ----------------------------------------------

        P_cov[i] = ukf.P

    return P_m, A, B, ph, P_cov, e

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


# kalman fit
alp = scankp
P_exp = normkp
P_m, A, B, ph, P_cov, e = kalmanFit_UKF(alp, P_exp)



# # main graphic
# plt.figure()
# plt.plot(normkp, label="exp")
# plt.plot(model(alp, A, B, ph), label="kalman")
# plt.plot(model(alp, A[0], B[0], ph[0]), label="sin")
# plt.legend()


# # additional graphics
# fig, axs = plt.subplots(1, 3, figsize=(14, 4))
# axs[0].plot(A)
# axs[0].set_title('A')

# # Второй график (верхний правый)
# axs[1].plot(B)
# axs[1].set_title('B')

# # Третий график (нижний левый)
# axs[2].plot(ph)  # или axs[1, 0]
# axs[2].set_title(r'$\phi$')

# fig1, axss = plt.subplots(1, 3, figsize=(14, 4))
# axss[0].plot(np.sqrt(P_cov[:, 0, 0]))
# axss[0].set_title('dA')

# # Второй график (верхний правый)
# axss[1].plot(np.sqrt(P_cov[:, 1, 1]))
# axss[1].set_title('dB')

# # Третий график (нижний левый)
# axss[2].plot(np.sqrt(P_cov[:, 2, 2]))  # или axs[1, 0]
# axss[2].set_title(r'$d\phi$')

# Q finder
plt.figure()
plt.title("Q find params")
plt.plot(e, label="innovation")


plt.show()