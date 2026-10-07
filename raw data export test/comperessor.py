import numpy as np

data = np.loadtxt(f'raw data export test\data_T_10_ms_delay_200_ms.txt', delimiter=',', unpack=False, skiprows=0)
alpha = data[:, 0]
norm = data[:, 1]
accel = data[:, 2:]




# сохранение в npz
SCALE = 1e2                      # 2 знака после запятой
save_path = r"raw data export test\gravimeter_data_q2.npz"

q = np.rint(accel * SCALE).astype(np.int64)      # целые кванты 1e-2
d = np.diff(q, axis=1)                           # разности соседних отсчётов
assert np.abs(d).max() < 2**15, "разности не влезают в int16"
assert np.abs(q[:, 0]).max() < 2**31

np.savez_compressed(save_path, alp=alpha, P_exp=norm,
                    d=d.astype(np.int16), head=q[:, 0].astype(np.int32),
                    scale=SCALE)

# load file

def load_q(path):
    z = np.load(path)
    q = np.concatenate([z["head"].astype(np.int64)[:, None],
                        z["d"].astype(np.int64)], axis=1).cumsum(axis=1)
    return z["alp"], z["P_exp"], q / float(z["scale"])

load_path = r"raw data export test\gravimeter_data_q2.npz"
alp, P_exp, az_m = load_q(load_path)