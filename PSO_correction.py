import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def sinf(alpha, A, B, ph):
    return A + B*np.cos(2*np.pi*alpha*T**2 - ph)

def model(alp, A, B, ph):
    return A + B*np.cos(2*np.pi*alp*T**2 - ph)

# accelerometer sensetivity function
def fa(t):
    if 0 < t <= T+2*ty:
        return t/(T+2*ty)**2
    elif T+2*ty < t <= 2*T+4*ty:
        return (2*(T+2*ty)-t)/(T+2*ty)**2
    else:
        return 0
vfunc = np.vectorize(fa)


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

def kalmanFit2(alp, P_exp): # nonlinear/unscented kalman filter

    # param init
    A = np.zeros(len(alp))
    B = np.zeros(len(alp))
    ph = np.zeros(len(alp))
    e = np.zeros(len(alp))
    P_m = np.zeros(len(alp))

    # matrix init
    P_cov = np.diag([0.01**2, 0.01**2, 0.01**2])

    Q = np.zeros((3,3))

    R = 1e-4

    # sgima points
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

    return P_m

def kalmanFilter3(alp, P_exp): # jacobian kalman
    return 1






def rules(f_vib_, dP):
    weight_1 = 0
    weight_2 = 0

    if abs(f_vib_) > 0.1: weight_1 = 1
    if abs(f_vib_) > 1: weight_1 = 2
    if abs(f_vib_) > 2: weight_1 = 3

    if abs(dP) > 0.01: weight_2 = 1
    if abs(dP) > 0.05: weight_2 = 2
    if abs(dP) > 0.1: weight_2 = 3

    K = np.zeros((7,7))
    K[0] = [1, ]

    return 1


def fitness_fuzzy(alpha, P_exp, acc_mz, kz, st_i): # StI - time delay in index

    # rules
    k_A = 0.01
    k_B = 1.1
    k_alp = 1.1
    k_others = 1.1

    # fit coef
    A_m = np.zeros(len(alpha))
    B_m = np.zeros(len(alpha))
    F_oth_m = np.zeros(len(alpha))
    P_eval_m = np.zeros(len(alpha))

    # initial coef
    poi = 201
    p0 = [(np.max(P_exp)+np.min(P_exp))/2, (np.max(P_exp)-np.min(P_exp))/2, 0]
    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]
    popt, pcov = curve_fit(sinf, alpha[:poi], P_exp[:poi], p0=p0, bound=(lb, ub))
    A_m[0], B_m[0], F_oth_m[0] = popt

    # Fvib
    intvib = acc_mz[:, st_i:st_i+iTAI+1]*fat
    fvib = keff*simpson(y=intvib, x=tan, axis=-1)*kz

    # Fvib* [0, 2pi)
    fvib_ = fvib % (2*np.pi) # тут не уверен, возможно один период внутри косинуса имели ввиду

    alpha_corr = alpha - fvib_/T**2/(2*np.pi) # + or -, 2pi? # тут можно сделать чтобы alpha corr попадала в тот же диапазон что и alpha

    # iteration process
    for i in range(len(chirp_rate)):

        P_eval_m[i] = A_m[i] + B_m[i]*np.cos(-2*np.pi*alpha_corr[i]*T**2 + F_oth_m[i])
        dP = P_exp[i] - P_eval_m[i]

        k_A, k_B, k_g, k_alp, k_others = rules(fvib_[i], dP)
        A_m[i+1] = A_m[i] + k_A
        B_m[i+1] = B_m[i] + k_B*np.cos(-2*np.pi*alpha_corr[i]*T**2 + F_oth_m[i])
        alpha_corr[i+1] = alpha_corr[i+1] + k_alp*np.sin( - 2*np.pi*alpha_corr[i]*T**2 + F_oth_m[i]) * 2*np.pi*T*T
        F_oth_m[i+1] = F_oth_m[i] - k_others*B_m[i]*np.sin( - 2*np.pi*alpha_corr[i]*T**2 + F_oth_m[i])

    sigma = np.std(P_eval_m - P_exp)


    return sigma, alpha_corr, P_eval_m


# constants
lam = 780e-9
keff = 4*np.pi/lam

# QG Params
T = 8200e-6
ty = 20e-6
TAI = 2*T+4*ty
TRP = 33.556e-3
Ampl = 20
dt = TRP/16383 # Red Pitaya time step
iTAI = int(np.floor(TAI/dt))
ta = np.arange(0, 16384)*dt
tan = ta[:iTAI+1]
fat = vfunc(tan)

### test data

file_path = r'v2\int_data.csv' 
data = np.genfromtxt(file_path, delimiter=',', dtype=None, skip_header=1)
data = np.array(data.tolist())
chirp_rate = data[:-1,0]
intensity = data[:-1,1]

acc_mz = np.load(r'v2\57201025193210504.npy')
acc_mz = (acc_mz.astype(np.float32) + 168)/ 8191.0 * 20 / 150 / Ampl
acc_mz = acc_mz - np.mean(acc_mz)
acc_mz = acc_mz[:]

plt.figure()
plt.title("Intrference signal")
plt.plot(chirp_rate, intensity, marker="o")

plt.figure()
plt.title("accelerometer signal")
plt.plot(acc_mz[6])

###

# kalman filter fitting


plt.show()