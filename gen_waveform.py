# gen_fringe.py
# =============================================================================
# Генерация данных фринджа P = A - B*cos(Ph) с ИЗВЕСТНОЙ истиной A, B, ph
# при учёте вибрации по ВСЕМ трём осям (ax, ay, az).
#
# Ускорения берутся из vibration_gen.generate_vibration() -- файл vibration_gen.py
# должен лежать рядом с этим скриптом.
#
# Модель:
#   ph[i]     -- дрейф фазы (процесс Орнштейна-Уленбека) вокруг ph_ref = KEFF*G0*T^2
#                (G0 используется ТОЛЬКО как опорное значение: центр сетки alp и ph_ref)
#   A[i], B[i]-- случайные блуждания (шаги dA_step, dB_step за сброс)
#   F_vib[i]  = Kz*Fz + Kx*Fx + Ky*Fy,  F_a = KEFF * (a[delay:delay+win] @ weight_vec)
#   alp_vib   = alp - F_vib / (2*pi*T^2)
#   Ph_clean  = 2*pi*alp_vib*T^2 - ph[i]
#   Ph        = Ph_clean + N(0, sigma_ph_sim)       <-- единственный шум фринджа
#   P_clean   = A - B*cos(Ph_clean)
#   P_noise   = A - B*cos(Ph)                        (шума детектирования нет)
#
# На каждом сбросе генерируются три независимые реализации ax, ay, az длиной
# n_rp отсчётов (как в исходном simul_acc: окна разных сбросов не коррелированы).
#
# Структура файла:
#   1. КОНСТАНТЫ      2. SimCfg      3. Симуляция      4. Вывод/графики
#   5. main()  -- ВСЕ параметры задаются здесь, внизу файла.
#
# Запуск:  python gen_fringe.py
# Использование из другого скрипта:
#   from gen_fringe import SimCfg, simulate_fringe
#   data = simulate_fringe(SimCfg(...))
#   data["A"], data["B"], data["ph"]  -- истина;  data["alp"], data["P_noise"] -- данные
# =============================================================================

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from gen_acc import generate_vibration

# =====================================================================
# 1. КОНСТАНТЫ
# =====================================================================

LM = 780e-9                  # длина волны, м
KEFF = 4 * np.pi / LM        # эффективный волновой вектор
G0 = 9.8101507               # ОПОРНОЕ g, м/с^2 (только reference: центр alp и ph_ref)


# =====================================================================
# 2. ПАРАМЕТРЫ СИМУЛЯЦИИ (значения задаются в main)
# =====================================================================

@dataclass
class SimCfg:
    # --- интерферометр ---
    T: float                    # длительность плеча, с
    ty: float                   # длительность импульса, с
    T_RP: float                 # запись акселерометра на сброс, с
    n_rp: int                   # отсчётов акселерометра на сброс
    # --- набор данных ---
    n_sim: int                  # число сбросов
    alp_amount: int             # точек развёртки alp в одном фринджe
    seed: int                   # сид (сиды вибрации и шумов выводятся из него)
    # --- вибрация ---
    vib_state: str              # 'mooring' | 'sailing'
    hp_cutoff: float            # срез ВЧ-фильтра акселерометра, Гц
    Kz: float                   # коэффициент связи вибрации z с фазой
    Kx: float                   # коэффициент связи вибрации x с фазой
    Ky: float                   # коэффициент связи вибрации y с фазой
    delay: int                  # смещение окна интерферометра в записи, отсчёты
    sigma_a_acc: float          # шум акселерометра, м/с^2 (добавляется ВСЕГДА; 3e-5 как в PSO-скрипте)
    # --- фринж ---
    A0: float                   # начальное A
    B0: float                   # начальное B
    dA_step: float              # шаг блуждания A за сброс (std)
    dB_step: float              # шаг блуждания B за сброс (std)
    # --- шум ---
    sigma_ph_sim: float         # белый фазовый шум на сброс, рад (единственный шум)
    # --- дрейф фазы ph (процесс Орнштейна-Уленбека) ---
    drift_corr: float           # время корреляции, сбросов
    Dph_drift: float            # амплитуда дрейфа ph, рад
    # --- вывод ---
    keep_acc: bool              # хранить записи ax_m, ay_m, az_m (n_sim x n_rp x 3 x 8 байт!)

    @property
    def t_step(self):
        return self.T_RP / self.n_rp

    @property
    def ph_ref(self):
        """Опорная фаза KEFF*G0*T^2 (центр дрейфа ph)."""
        return KEFF * G0 * self.T ** 2


# =====================================================================
# 3. СИМУЛЯЦИЯ
# =====================================================================

def fa(t, T, ty):
    """Функция чувствительности интерферометра к ускорению (скаляр)."""
    if 0 < t <= T + 2 * ty:
        return t
    elif T + 2 * ty < t <= 2 * T + 4 * ty:
        return 2 * (T + 2 * ty) - t
    else:
        return 0


def sim_weights(T, ty, t_step):
    """Веса интеграла чувствительности: F = KEFF * (a_window @ weight_vec)."""
    duration = 2 * T + 4 * ty
    t_grid = np.linspace(0, duration, round(duration / t_step))
    fa_t = np.array([fa(t, T, ty) for t in t_grid], dtype=float)
    trapz = np.full(len(t_grid), t_step)
    trapz[0] *= 0.5
    trapz[-1] *= 0.5
    return fa_t * trapz


def sens_integral(delay, a, weight_vec):
    """Сырой (без K) интеграл чувствительности по окну интерферометра."""
    win = len(weight_vec)
    seg = a[delay:delay + win]
    if len(seg) != win:
        raise ValueError(f"delay={delay}: окно [{delay}, {delay + win}) выходит "
                         f"за пределы записи длиной {len(a)}")
    return KEFF * float(seg @ weight_vec)


def simulate_fringe(cfg: SimCfg, verbose=True):
    """
    Генерирует один набор данных с вибрацией по трём осям.

    Возвращает словарь (массивы по сбросам, длина n_sim):
      истина фринджа : A, B, ph (рад, без свёртки по 2*pi), ph_ref
      данные         : alp, P_noise, P_clean, Ph (фаза в косинусе), Ph_clean, alp_vib
                       (P_clean -- без белого фазового шума, P_noise -- с ним)
      вибрация       : F_vib (суммарная, с K), F_vib_x/y/z (K*F по осям),
                       Fx_raw/Fy_raw/Fz_raw (интеграл по ИЗМЕРЕННОМУ ускорению
                       на окне delay, без K -- то, что видит обработка;
                       alp, скорректированный по измерениям:
                       alp - K*F_raw/(2*pi*T^2) -- зашумлён акселерометром)
      (опц.)         : ax_m, ay_m, az_m -- измеренные записи (n_sim, n_rp),
                       если cfg.keep_acc
      заложенные     : dA, dB, dph (= sigma_ph_drift), sigma_ph
    """
    if cfg.vib_state not in ('mooring', 'sailing'):
        raise ValueError("vib_state должен быть 'mooring' или 'sailing'")

    n = cfg.n_sim
    weight_vec = sim_weights(cfg.T, cfg.ty, cfg.t_step)
    win = len(weight_vec)
    if cfg.delay < 0 or cfg.delay + win > cfg.n_rp:
        raise ValueError(f"delay + win = {cfg.delay + win} > n_rp = {cfg.n_rp}")

    ss_vib, ss_noise = np.random.SeedSequence(cfg.seed).spawn(2)
    rng_vib = np.random.default_rng(ss_vib)
    rng_noise = np.random.default_rng(ss_noise)

    # --- опорная фаза и дрейф ph (процесс Орнштейна-Уленбека) ---
    ph_ref = cfg.ph_ref
    theta = 1 - np.exp(-1.0 / cfg.drift_corr)
    sigma_ph_drift = cfg.Dph_drift * np.sqrt(theta * (2 - theta))

    # --- сетка alp (центр -- по опорному G0) ---
    alp_c = KEFF * G0 / 2 / np.pi
    alp_min = alp_c - 1 / 5 / cfg.T / cfg.T
    alp_max = alp_c + 1 / 5 / cfg.T / cfg.T
    alp_start = np.linspace(alp_min, alp_max, cfg.alp_amount)

    alp = np.zeros(n)
    alp_vib = np.zeros(n)
    Ph_clean = np.zeros(n)
    Ph = np.zeros(n)
    ph = np.zeros(n)
    A = np.zeros(n)
    B = np.zeros(n)
    P_clean = np.zeros(n)
    P_noise = np.zeros(n)

    F_axis = {k: np.zeros(n) for k in 'xyz'}      # K * F (то, что вошло в фазу)
    F_raw = {k: np.zeros(n) for k in 'xyz'}       # F по ИЗМЕРЕННОМУ ускорению, без K
    K = {'x': cfg.Kx, 'y': cfg.Ky, 'z': cfg.Kz}

    acc_m = ({k: np.zeros((n, cfg.n_rp)) for k in 'xyz'} if cfg.keep_acc else None)

    ph_prev, A_prev, B_prev = ph_ref, cfg.A0, cfg.B0
    t0 = time.perf_counter()
    report_every = max(1, n // 10)

    for i in range(n):
        ph[i] = ph_ref + (ph_prev - ph_ref) * (1 - theta) + sigma_ph_drift * rng_noise.normal()
        ph_prev = ph[i]

        # --- вибрация: три независимые реализации ax, ay, az на этот сброс ---
        seed_i = int(rng_vib.integers(0, 2 ** 31 - 1))
        ax, ay, az, _ = generate_vibration(
            cfg.vib_state, cfg.n_rp, cfg.T_RP, seed=seed_i,
            hp_cutoff=cfg.hp_cutoff, N_sim=1)
        acc = {'x': ax, 'y': ay, 'z': az}

        for k in 'xyz':
            # истинная вибрация -> в фазу интерферометра
            F_axis[k][i] = sens_integral(cfg.delay, acc[k], weight_vec) * K[k]
            # то, что видит обработка: ускорение + шум акселерометра (всегда)
            a_meas = acc[k] + rng_noise.normal(0, cfg.sigma_a_acc, cfg.n_rp)
            F_raw[k][i] = sens_integral(cfg.delay, a_meas, weight_vec)
            if cfg.keep_acc:
                acc_m[k][i] = a_meas

        F_total = F_axis['x'][i] + F_axis['y'][i] + F_axis['z'][i]

        alp[i] = alp_start[i % cfg.alp_amount]
        alp_vib[i] = alp[i] - F_total / (2 * np.pi * cfg.T ** 2)
        Ph_clean[i] = 2 * np.pi * alp_vib[i] * cfg.T ** 2 - ph[i]
        Ph[i] = Ph_clean[i] + rng_noise.normal(0, cfg.sigma_ph_sim)

        A[i] = A_prev + rng_noise.normal(0, cfg.dA_step)
        B[i] = B_prev + rng_noise.normal(0, cfg.dB_step)
        A_prev, B_prev = A[i], B[i]

        P_clean[i] = A[i] - B[i] * np.cos(Ph_clean[i])
        P_noise[i] = A[i] - B[i] * np.cos(Ph[i])

        if verbose and ((i + 1) % report_every == 0 or i + 1 == n):
            print(f"  [simulate_fringe] сброс {i + 1}/{n}, "
                  f"{time.perf_counter() - t0:.1f} с")

    data = {
        "alp": alp, "alp_vib": alp_vib, "P_noise": P_noise, "P_clean": P_clean,
        "A": A, "B": B, "ph": ph, "ph_ref": float(ph_ref),
        "Ph": Ph, "Ph_clean": Ph_clean,
        "F_vib": F_axis['x'] + F_axis['y'] + F_axis['z'],
        "F_vib_x": F_axis['x'], "F_vib_y": F_axis['y'], "F_vib_z": F_axis['z'],
        "Fx_raw": F_raw['x'], "Fy_raw": F_raw['y'], "Fz_raw": F_raw['z'],
        "dA": float(cfg.dA_step), "dB": float(cfg.dB_step),
        "dph": float(sigma_ph_drift),
        "sigma_ph": float(cfg.sigma_ph_sim),
    }
    if cfg.keep_acc:
        data["ax_m"], data["ay_m"], data["az_m"] = acc_m['x'], acc_m['y'], acc_m['z']
    return data


# =====================================================================
# 4. ВЫВОД И ГРАФИКИ
# =====================================================================

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def print_summary(data, cfg: SimCfg):
    print("\n" + "=" * 70)
    print("ИТОГ ГЕНЕРАЦИИ")
    print("=" * 70)
    print(f"vib_state = {cfg.vib_state}, n_sim = {cfg.n_sim}, n_rp = {cfg.n_rp}, "
          f"T_RP = {cfg.T_RP:g} с, hp_cutoff = {cfg.hp_cutoff:g} Гц")
    print(f"Kz = {cfg.Kz:g}, Kx = {cfg.Kx:g}, Ky = {cfg.Ky:g}, delay = {cfg.delay}")
    print("\nЗаложено в шумы/шаги:")
    print(f"  sigma_ph = {data['sigma_ph']:.3e} рад, sigma_a_acc = {cfg.sigma_a_acc:.3e} м/с^2")
    print(f"  dA = {data['dA']:.3e}, dB = {data['dB']:.3e}, dph = {data['dph']:.3e} рад")
    print("\nФазовый вклад вибрации K*F, рад (RMS / max|.|):")
    for k, lab in (('x', 'ax'), ('y', 'ay'), ('z', 'az')):
        v = data[f"F_vib_{k}"]
        print(f"  по {lab}: {rms(v):.3e} / {np.max(np.abs(v)):.3e}")
    v = data["F_vib"]
    print(f"  сумма : {rms(v):.3e} / {np.max(np.abs(v)):.3e}")
    print("\nИстина по сбросам (начало -> конец):")
    print(f"  A  : {data['A'][0]:.4f} -> {data['A'][-1]:.4f}")
    print(f"  B  : {data['B'][0]:.4f} -> {data['B'][-1]:.4f}")
    dph = data['ph'] - data['ph_ref']
    print(f"  ph : {dph[0]:+.3f} -> {dph[-1]:+.3f} рад относительно ph_ref "
          f"(RMS {rms(dph):.3f} рад)")


def plot_fringe(data, cfg: SimCfg):
    shots = np.arange(cfg.n_sim)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    axs[0, 0].plot(shots, data["A"], label="A")
    axs[0, 0].plot(shots, data["B"], label="B")
    axs[0, 0].set_xlabel("сброс #")
    axs[0, 0].set_title("Истина: A, B")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(shots, data["ph"] - data["ph_ref"])
    axs[0, 1].set_xlabel("сброс #")
    axs[0, 1].set_ylabel("ph - ph_ref, рад")
    axs[0, 1].set_title("Истина: ph")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(shots, data["P_noise"], ".", ms=2, alpha=0.6, label="P с фазовым шумом")
    axs[1, 0].plot(shots, data["P_clean"], lw=0.6, label="P без фазового шума")
    axs[1, 0].set_xlabel("сброс #")
    axs[1, 0].set_ylabel("P")
    axs[1, 0].set_title("Сигнал фринджа")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(shots, data["F_vib_z"], lw=0.6, label="Kz·Fz")
    axs[1, 1].plot(shots, data["F_vib_x"], lw=0.6, label="Kx·Fx")
    axs[1, 1].plot(shots, data["F_vib_y"], lw=0.6, label="Ky·Fy")
    axs[1, 1].set_xlabel("сброс #")
    axs[1, 1].set_ylabel("фаза, рад")
    axs[1, 1].set_title("Вклад вибрации в фазу по осям")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"vib_state={cfg.vib_state}, n_sim={cfg.n_sim}")
    fig.tight_layout()
    return fig


# =====================================================================
# 5. main: ВСЕ ПАРАМЕТРЫ ЗАДАЮТСЯ ЗДЕСЬ
# =====================================================================

def main():
    cfg = SimCfg(
        # --- интерферометр ---
        T=10e-3,                    # длительность плеча, с
        ty=20e-6,                   # длительность импульса, с
        T_RP=33.556e-3,             # запись акселерометра на сброс, с
        N_rp=16384,                 # отсчётов акселерометра на сброс
        # --- набор данных ---
        n_sim=2000,                 # число сбросов
        alp_amount=200,             # точек развёртки alp в одном фринджe
        seed=123,                   # сид
        # --- вибрация (все три оси) ---
        vib_state="mooring",        # 'mooring' | 'sailing'
        hp_cutoff=0.01,             # срез ВЧ-фильтра акселерометра, Гц
        Kz=0.9,                     # коэффициент связи по z
        Kx=0.003,                   # коэффициент связи по x
        Ky=0.003,                   # коэффициент связи по y
        delay=700,                  # смещение окна интерферометра, отсчёты
        sigma_a_acc=3e-5,           # шум акселерометра, м/с^2 (как в PSO-скрипте)
        # --- фринж ---
        A0=0.15,                    # начальное A
        B0=0.21,                    # начальное B
        dA_step=5e-3,               # шаг блуждания A за сброс
        dB_step=5e-3,               # шаг блуждания B за сброс
        # --- шум ---
        sigma_ph_sim=5.6e-3,        # белый фазовый шум на сброс, рад
        # --- дрейф фазы ph ---
        drift_corr=1000,            # время корреляции, сбросов
        Dph_drift=0.01,              # амплитуда дрейфа ph, рад (~ прежние 500e-8 по g)
        # --- вывод ---
        keep_acc=False)             # хранить записи ускорений (большой объём памяти)

    SAVE_PATH = None                # например "fringe_data.npz"; None -- не сохранять
    DO_PLOT = True

    data = simulate_fringe(cfg)
    print_summary(data, cfg)

    if SAVE_PATH:
        np.savez_compressed(SAVE_PATH, **data)
        print(f"\nДанные сохранены: {SAVE_PATH}")

    if DO_PLOT:
        plot_fringe(data, cfg)
        plt.show()

    return data, cfg


if __name__ == "__main__":
    main()