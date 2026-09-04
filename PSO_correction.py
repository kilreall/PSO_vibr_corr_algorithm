import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def sinf(alpha, A, B, ph):
    return B*np.cos(-2*np.pi*alpha*T**2 + ph) + A


# accelerometer sensetivity function
def fa(t):
    if 0 < t <= T+2*ty:
        return t/(T+2*ty)**2
    elif T+2*ty < t <= 2*T+4*ty:
        return (2*(T+2*ty)-t)/(T+2*ty)**2
    else:
        return 0
vfunc = np.vectorize(fa)

def fitness(chirp_rate, intensity, acc_mz, kz, StI): # StI - time delay in index


    # fit coef
    A_m = np.zeros(len(chirp_rate))
    B_m = np.zeros(len(chirp_rate))
    Foth_m = np.zeros(len(chirp_rate))
    P_eval_m = np.zeros(len(chirp_rate))
    g_ph_m = np.zeros(len(chirp_rate))

    # initial coef
    Foth_m[0] = 0
    poi = 201
    p0 = [(np.max(intensity)-np.min(intensity))/2, (np.max(intensity)+np.min(intensity))/2, 0]
    lb = [0, 0, 0]
    ub = [1, 1, 2*np.pi]
    popt, pcov = curve_fit(sinf, chirp_rate[:poi], intensity[:poi], p0=p0, bound=(lb, ub))
    A_m[0], B_m[0], g_ph_m[0] = popt

    # Fvib
    intvib = acc_mz[:, StI:StI+iTAI+1]*fat
    fvib = keff*simpson(y=intvib, x=tan, axis=-1)*kz

    # Fvib* [0, 2pi)
    fvib_ = fvib % (2*np.pi) # тут не уверен, возможно один период внутри косинуса имели ввиду

    chirp_corr = chirp_rate - fvib_/T**2/(2*np.pi) # + or -, 2pi?

    # iteration process
    for i in range(len(chirp_rate)):

        P_eval_m[i] = A_m[i] + B_m[i]*np.cos(g_ph_m[i] - 2*np.pi*chirp_corr[i]*T**2 + Foth_m[i])
        dP = intensity - P_eval_m[i]
        A_m[i+1] = dP/


    return 1
#
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