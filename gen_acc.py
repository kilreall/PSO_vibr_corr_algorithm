"""
vibration_gen.py
=================

Основной скрипт генерации вибрационного ускорения платформы
(mooring / sailing) по трём осям.

Структура:

  gen_vibration_trace()      -- одна ось, метод случайных фаз
                                (Timmer & Koenig) по табличной ASD;
  generate_vibration()       -- ОСНОВНАЯ ГЕНЕРАЦИЯ: по флагу state
                                ('mooring' / 'sailing') возвращает
                                ax, ay, az (+ служебную информацию);
  plot_vibration_overview()  -- ОТРИСОВКА: принимает уже готовый a(t)
                                (ничего не генерирует сама);
  main                       -- вызывает generate_vibration(), затем
                                передаёт результат в отрисовку.

Чтобы использовать как библиотеку из другого скрипта:

    from vibration_gen import generate_vibration
    ax, ay, az, info = generate_vibration(state='sailing', N=..., dt=...)

Ослабление платформы (platform_atten_db) убрано: сигнал строится ровно
по табличной ASD (с учётом резонансных горбов res_lines и
антиалиасингового спада) и ВЧ-фильтра акселерометра.

Если N и dt не заданы, сетка подбирается автоматически функцией
suggest_grid() под диапазон f_nodes таблицы выбранного состояния
(N, dt из параметров короткого EKF-окна для просмотра спектра в диапазоне
1e-3..1e3 Гц непригодны: Df = 1/T ~ 30 Гц при T=33 мс).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, spectrogram


# ============================================================
# Таблицы ASD [м/с^2 / sqrt(Гц)], узлы по осям и состояниям
# ============================================================

_ASD_TABLES = {
    'mooring': {
        'f_nodes': np.array([1e-3, 3e-2, 0.1, 0.3, 1.0, 2.0, 4.0, 6.0,
                              10.0, 20.0, 40.0, 100.0, 300.0, 1000.0]),
        'axis': {
            'x': np.array([2.3e-4, 2.3e-4, 1.8e-4, 6.0e-5, 1.5e-4, 5.0e-4,
                           1.2e-3, 3.0e-3, 1.0e-3, 3.0e-4, 1.2e-4, 3.0e-5,
                           4.0e-6, 3.0e-7]),
            'y': np.array([2.0e-4, 2.0e-4, 1.6e-4, 5.0e-5, 1.3e-4, 4.5e-4,
                           1.0e-3, 2.5e-3, 9.0e-4, 2.5e-4, 1.0e-4, 2.5e-5,
                           3.5e-6, 2.5e-7]),
            'z': np.array([5.5e-5, 5.5e-5, 5.0e-5, 3.0e-5, 8.0e-5, 3.0e-4,
                           8.0e-4, 2.8e-3, 1.1e-3, 3.2e-4, 1.3e-4, 3.2e-5,
                           4.2e-6, 3.0e-7]),
        },
        'res_lines': [],
    },
    # Значения ниже откалиброваны по оцифровке рисунка Fig. (PSD для
    # sailing-состояния): пиксельные координаты кривых a_x/a_y/a_z были
    # сняты с картинки (граница осей на изображении: X от 1e-3 до 1e3 Гц,
    # Y от 1e0 до 1e-7 (м/с^2/√Гц), лог-лог) и пересчитаны в частоты/ASD.
    # Прежние узлы были систематически завышены: в 5-40 раз на низком
    # плато (1e-3..3e-2 Гц) и в "впадине" 1-40 Гц; у самого пика
    # (0.1-0.6 Гц) и выше 70 Гц совпадение и раньше было неплохим.
    # Значения ниже -- это БАЗОВАЯ (сглаженная) кривая ДО умножения на
    # узкополосные резонансные горбы res_lines (они по-прежнему
    # добавляются кодом поверх, поэтому в узлах 6/10/20/40 Гц заложены
    # величины немного ниже видимых на графике пиков -- сам пик даёт
    # множитель res_lines).
    'sailing': {
        'f_nodes': np.array([1e-3, 1e-2, 3e-2, 0.1, 0.15, 0.3, 0.6, 1.0,
                              2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 70.0, 100.0,
                              200.0, 400.0, 1000.0]),
        'axis': {
            'x': np.array([2.0e-3, 3.5e-3, 5.5e-3, 9.0e-2, 1.0e-1, 4.0e-2,
                           4.5e-3, 1.1e-3, 2.8e-4, 1.0e-4, 1.5e-4, 1.2e-4,
                           6.0e-5, 5.0e-5, 4.0e-5, 3.0e-5, 4.0e-6, 4.0e-6,
                           3.0e-7]),
            'y': np.array([4.0e-4, 7.5e-4, 1.3e-3, 3.3e-2, 3.6e-2, 2.8e-2,
                           3.2e-3, 9.5e-4, 2.2e-4, 1.2e-4, 1.6e-4, 1.0e-4,
                           9.0e-5, 3.0e-5, 3.0e-5, 3.0e-5, 5.0e-6, 1.5e-5,
                           1.5e-6]),
            'z': np.array([1.2e-4, 1.3e-4, 5.0e-4, 1.0e-1, 1.9e-1, 2.7e-2,
                           3.0e-3, 6.0e-4, 1.2e-4, 6.0e-5, 4.5e-4, 1.2e-3,
                           1.0e-4, 8.0e-5, 8.0e-5, 2.0e-5, 8.0e-6, 1.0e-5,
                           5.0e-7]),
        },
        'res_lines': [(6.0, 10, 1.2), (12.0, 12, 0.8), (24.0, 15, 0.5),
                      (45.0, 15, 0.4)],
    },
}


# ============================================================
# Генерация одной оси
# ============================================================

def _synthesize_from_asd(freqs, target_asd, N, fs, rng):
    """
    Строит реализацию временного ряда с заданной ОДНОСТОРОННЕЙ
    амплитудной спектральной плотностью target_asd(freqs) [ед/√Гц],
    методом случайных фаз (Timmer & Koenig).
    """
    n_freq = len(freqs)
    amp = target_asd * np.sqrt(N * fs / 2.0)

    phase = rng.uniform(0, 2*np.pi, n_freq)
    Xf = amp * np.exp(1j*phase)

    Xf[0] = amp[0] * rng.normal() * np.sqrt(2.0)
    if N % 2 == 0:
        Xf[-1] = amp[-1] * rng.normal() * np.sqrt(2.0)

    return np.fft.irfft(Xf, n=N)


def gen_vibration_trace(N, dt, axis='z', state='mooring', seed=None,
                         hp_cutoff=0.01, hp_order=2,
                         f_low_phys=None, aa_cutoff_factor=1.0,
                         return_components=False):
    """
    Генерация одноосевой реализации вибрационного ускорения платформы,
    приближённой к измеренным ASD (Qiao 2025) для состояний
    'mooring' и 'sailing'.

    seed -- int, None или np.random.SeedSequence (всё, что принимает
    np.random.default_rng).
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if state not in _ASD_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs_safe = freqs.copy()
    freqs_safe[0] = freqs_safe[1] * 0.5  # избегаем log(0)

    table = _ASD_TABLES[state]
    f_nodes = table['f_nodes']
    asd_nodes = table['axis'][axis]
    res_lines = table['res_lines']

    _danger_f = 0.05
    if hp_cutoff > _danger_f:
        print(f"[gen_vibration_trace] ВНИМАНИЕ: hp_cutoff={hp_cutoff} Гц "
              f"может начать резать реальный низкочастотный вибрационный "
              f"сигнал (пик ASD лежит в районе 0.1-1 Гц) -- держите "
              f"cutoff в районе 0.01 Гц или ниже")
    if f_low_phys is not None:
        pass  # справочно, не используется как порог отсечки

    log_target = np.interp(np.log10(freqs_safe),
                            np.log10(f_nodes), np.log10(asd_nodes))
    target_asd = 10 ** log_target

    for f0, q, rel_amp in res_lines:
        target_asd *= 1.0 + rel_amp * np.exp(-0.5*((freqs_safe - f0)/(f0/q))**2)

    f_aa = f_nodes[-1] * aa_cutoff_factor
    target_asd *= 1.0 / (1.0 + (freqs_safe / f_aa)**6)

    raw = _synthesize_from_asd(freqs_safe, target_asd, N, fs, rng)

    wn = hp_cutoff / (fs/2)
    if 0 < wn < 1:
        b, a_f = butter(hp_order, wn, btype='high')
        a_trace = filtfilt(b, a_f, raw)
    else:
        a_trace = raw

    if return_components:
        return a_trace, {'raw': raw, 'target_asd': target_asd, 'freqs': freqs_safe}
    return a_trace


# ============================================================
# Автоподбор сетки (N, dt) под диапазон f_nodes таблицы
# ============================================================

def suggest_grid(state, low_res_factor=3.0, high_margin=2.5,
                  max_N=4_000_000, verbose=True):
    """
    Подбирает (N, dt), самосогласованные с диапазоном частот таблицы
    ASD выбранного состояния (`state`).

    Требования:
      - Nyquist: fs = 1/dt должна быть заметно выше верхнего узла
        таблицы f_max. Берём fs = high_margin * f_max.
      - Частотное разрешение БПФ Df = 1/T = 1/(N*dt) должно быть
        заметно МЕНЬШЕ нижнего узла таблицы f_min. Берём
        Df = f_min / low_res_factor.

    Если "честное" N превышает `max_N`, N обрезается до max_N, fs
    сохраняется, а достигнутое разрешение Df оказывается хуже
    желаемого -- об этом печатается предупреждение.

    Возвращает dict с полями N, dt, fs, T, df, f_min, f_max, warning.
    """
    if state not in _ASD_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    f_nodes = _ASD_TABLES[state]['f_nodes']
    f_min, f_max = f_nodes[0], f_nodes[-1]

    fs = high_margin * f_max
    df_target = f_min / low_res_factor
    T_target = 1.0 / df_target
    N_target = int(np.ceil(T_target * fs))

    warning = None
    N = N_target
    if N_target > max_N:
        N = int(max_N)
        T_actual = N / fs
        df_actual = 1.0 / T_actual
        warning = (
            f"[suggest_grid:{state}] 'Честная' сетка требует N={N_target:,} "
            f"отсчётов (T={T_target:.1f} с при fs={fs:.1f} Гц), чтобы разрешить "
            f"f_min={f_min:g} Гц с запасом x{low_res_factor:g}. Это больше "
            f"max_N={max_N:,}, поэтому N ограничен до {max_N:,}. "
            f"Достигнутое разрешение Df={df_actual:.3g} Гц (T={T_actual:.1f} с) "
            f"вместо желаемых Df={df_target:.3g} Гц -- структура ASD в районе "
            f"{df_actual*3:.3g} Гц и ниже может быть недостоверна/усреднена. "
            f"Поднимите max_N, если нужна более честная картина у 1e-3..1e-2 Гц."
        )

    dt = 1.0 / fs
    T = N * dt
    df = 1.0 / T

    info = dict(N=N, dt=dt, fs=fs, T=T, df=df, f_min=f_min, f_max=f_max,
                warning=warning)

    if verbose:
        print(f"[suggest_grid:{state}] N={N:,}, dt={dt:.3e} с, fs={fs:.1f} Гц, "
              f"T={T:.1f} с, Df={df:.3g} Гц (диапазон таблицы "
              f"{f_min:g}..{f_max:g} Гц)")
        if warning:
            print(warning)

    return info


# ============================================================
# ОСНОВНАЯ ГЕНЕРАЦИЯ: по флагу state -> ax, ay, az
# ============================================================

def generate_vibration(state, N_RP, T_RP, seed=42,
                        hp_cutoff=0.01, hp_order=2,
                        aa_cutoff_factor=1.0, N_sim=1):
    """
    Генерирует вибрационное ускорение по трём осям для выбранного
    состояния платформы на сетке окна пайплайна.

    Параметры
    ---------
    state : 'mooring' | 'sailing'
        Флаг состояния -- определяет таблицу ASD и резонансные горбы.
    N_RP : int
        Число отсчётов в одном окне.
    T_RP : float
        Длительность одного окна, с. Шаг сетки dt = T_RP / N_RP.
    seed : int | None
        Базовый seed. Для осей x, y, z из него порождаются независимые
        дочерние потоки (SeedSequence.spawn), поэтому оси
        некоррелированы между собой, но результат воспроизводим.
    hp_cutoff, hp_order, aa_cutoff_factor
        Параметры ВЧ-фильтра акселерометра и антиалиасингового спада
        (пробрасываются в gen_vibration_trace).
    N_sim : int
        Число подряд идущих окон. Общая длина реализации
        N = N_sim * N_RP (по умолчанию одно окно).

    Возвращает
    ----------
    ax, ay, az : np.ndarray  -- ускорения по осям, м/с^2
    info : dict
        state, N, dt, fs, t (вектор времени), hp_cutoff и
        'dbg' = {'x': {...}, 'y': {...}, 'z': {...}} с целевой ASD
        (target_asd, freqs, raw) для каждой оси.
    """
    if state not in _ASD_TABLES:
        raise ValueError("state must be 'mooring' or 'sailing'")

    dt = T_RP / N_RP
    N = int(N_sim) * int(N_RP)
    fs = 1.0 / dt

    f_max = _ASD_TABLES[state]['f_nodes'][-1]
    if fs / 2.0 < f_max:
        print(f"[generate_vibration] ВНИМАНИЕ: fs/2={fs/2:.3g} Гц ниже "
              f"верхнего узла таблицы {f_max:g} Гц -- ВЧ-часть ASD "
              f"будет обрезана по Найквисту")

    child_seeds = np.random.SeedSequence(seed).spawn(3)

    traces, dbgs = {}, {}
    for ax_name, ss in zip(('x', 'y', 'z'), child_seeds):
        traces[ax_name], dbgs[ax_name] = gen_vibration_trace(
            N, dt, axis=ax_name, state=state, seed=ss,
            hp_cutoff=hp_cutoff, hp_order=hp_order,
            aa_cutoff_factor=aa_cutoff_factor,
            return_components=True)

    info = dict(state=state, N=N, dt=dt, fs=fs,
                t=np.arange(N) * dt, hp_cutoff=hp_cutoff, dbg=dbgs)
    return traces['x'], traces['y'], traces['z'], info


# ============================================================
# ОТРИСОВКА: принимает уже готовые данные, ничего не генерирует
# ============================================================

def plot_vibration_overview(a_t, info, axis, nperseg_frac=8):
    """
    Строит для готовой реализации a(t) одной оси:
      - временной ряд;
      - целевую ASD (табличную, по которой строился сигнал) и
        Welch-оценку ASD по самой реализации;
      - спектрограмму (time-frequency picture).

    Параметры
    ---------
    a_t  : np.ndarray -- ускорение одной оси (из generate_vibration).
    info : dict       -- словарь info, возвращённый generate_vibration.
    axis : 'x' | 'y' | 'z' -- какая это ось (для подписей и выбора
           целевой ASD из info['dbg']).
    """
    state = info['state']
    N, dt, fs, t = info['N'], info['dt'], info['fs'], info['t']
    dbg = info['dbg'][axis]

    # nperseg для Welch/спектрограммы: не больше N; не мельчим сегмент
    # сильнее, чем нужно для усреднения, и не теряем разрешение у
    # нижних узлов таблицы.
    nperseg = int(np.clip(N // nperseg_frac, 1024, N))
    f_welch, Pxx = welch(a_t, fs=fs, nperseg=nperseg)
    asd_welch = np.sqrt(Pxx)

    f_spec, t_spec, Sxx = spectrogram(a_t, fs=fs, nperseg=nperseg,
                                       noverlap=nperseg // 2)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].plot(t, a_t, lw=0.6)
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel(r'a, м/с$^2$')
    axs[0].set_title(f'Временной ряд: a{axis}, {state}')

    axs[1].loglog(dbg['freqs'], dbg['target_asd'], 'k--', lw=1.5, label='target ASD')
    axs[1].loglog(f_welch, asd_welch, alpha=0.75, label='Welch-оценка по a(t)')
    axs[1].set_xlabel('f, Гц')
    axs[1].set_ylabel(r'ASD, м/с$^2$/$\sqrt{Гц}$')
    axs[1].set_title('Спектр (ASD)')
    axs[1].legend()
    axs[1].grid(True, which='both', alpha=0.3)

    im = axs[2].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-30), shading='auto')
    axs[2].set_yscale('log')
    axs[2].set_ylim(max(f_spec[1], 1e-3), fs/2)
    axs[2].set_xlabel('t, с')
    axs[2].set_ylabel('f, Гц')
    axs[2].set_title('Спектрограмма')
    fig.colorbar(im, ax=axs[2], label='дБ')

    fig.suptitle(f'axis={axis}, state={state}, N={N}, dt={dt:.3e} с, '
                 f'hp_cutoff={info["hp_cutoff"]} Гц')
    fig.tight_layout()
    return fig


# ============================================================
# main
# ============================================================

def main():
    # ---------------- настройки ----------------
    STATE = 'sailing'      # флаг состояния: 'mooring' или 'sailing'
    SEED = 42
    HP_CUTOFF = 0.01

    # --- сетка ---
    # Сетка берётся из параметров окна пайплайна:
    #        N = N_SIM * N_RP,  dt = T_RP / N_RP.
    # USE_RP_GRID = True дополнительно нарезает результат на окна
    # по N_RP отсчётов (info['windows']).
    USE_RP_GRID = False
    N_RP = 16384           # отсчётов в одном окне
    T_RP = 33e-3           # длительность одного окна, с
    N_SIM = 10             # число подряд идущих окон по N_RP отсчётов

    DO_PLOT = True
    # -------------------------------------------

    # 1) ОСНОВНАЯ ГЕНЕРАЦИЯ
    ax, ay, az, info = generate_vibration(
        STATE, N_RP, T_RP, seed=SEED,
        hp_cutoff=HP_CUTOFF, N_sim=N_SIM)

    # Нарезка на окна по N_RP отсчётов: windows[axis][i] -- i-е окно
    if USE_RP_GRID:
        info['N_RP'], info['N_sim'] = N_RP, N_SIM
        info['windows'] = {k: a.reshape(N_SIM, N_RP)
                           for k, a in (('x', ax), ('y', ay), ('z', az))}

    print(f"Сгенерировано: state={info['state']}, N={info['N']:,}, "
          f"dt={info['dt']:.3e} с, T={info['N']*info['dt']:.3g} с"
          + (f", окон N_sim={N_SIM} x N_RP={N_RP}" if USE_RP_GRID else ""))
    print(f"  RMS ax={np.std(ax):.3e}, ay={np.std(ay):.3e}, "
          f"az={np.std(az):.3e} м/с^2")

    # 2) ОТРИСОВКА (получает уже готовые ax, ay, az)
    if DO_PLOT:
        for axis_name, a_t in (('x', ax), ('y', ay), ('z', az)):
            plot_vibration_overview(a_t, info, axis_name)
        plt.show()

    return ax, ay, az, info


if __name__ == "__main__":
    main()