import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def sinf(alpha, A, B, ph):
    return A + B*np.cos(-2*np.pi*alpha*T**2 + ph)


# accelerometer sensetivity function
def fa(t):
    if 0 < t <= T+2*ty:
        return t/(T+2*ty)**2
    elif T+2*ty < t <= 2*T+4*ty:
        return (2*(T+2*ty)-t)/(T+2*ty)**2
    else:
        return 0
vfunc = np.vectorize(fa)



def kalman_fit():
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

kz = 1
StI = 0
fitness(acc_mz, kz, StI)
plt.show()