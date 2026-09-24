"""
vibration_gen.py
=================
 
Функции генерации вибрационного ускорения платформы (mooring / sailing),
выделенные из основного скрипта симуляции гравиметра, ЧТОБЫ можно было
отдельно, без остального пайплайна (EKF, PSO, поиск Kz/Kx/Ky и т.д.),
посмотреть:
 
  1) как выглядит сама реализация a(t) во времени;
  2) её спектр (ASD, амплитудная спектральная плотность) и сравнение
     с целевой (табличной) ASD, из которой она строится методом
     случайных фаз (Timmer & Koenig);
  3) спектрограмму (time-frequency picture), чтобы увидеть, как ведёт
     себя сигнал во времени -- есть ли явно выраженные горбы/линии.
 
Логика самой генерации (таблицы ASD, метод случайных фаз, ВЧ-фильтр
акселерометра, ослабление платформы) скопирована 1:1 из основного
скрипта simul_acc.py -- здесь ничего не меняется по существу, только
убрано всё, что не нужно для просмотра a(t) и её спектра.
 
ВАЖНО (исправление): (N, dt), с которыми раньше вызывался генератор в
блоке __main__ (N=16384, T=33 мс), были взяты из основного пайплайна
(параметры одного короткого высокочастотного измерительного окна EKF)
и физически непригодны для того, чтобы увидеть спектр в диапазоне
1e-3..1e3 Гц, который описывает таблица ASD и который показан на
рисунке из статьи (Fig. 1): при T=33 мс частотное разрешение БПФ было
Df = 1/T ~ 30 Гц, то есть вся структура ASD ниже ~30 Гц (провал у
0.1-0.3 Гц, горб у 1-6 Гц) физически не могла присутствовать в
сгенерированном сигнале. Теперь (N, dt) для демонстрационного графика
подбираются автоматически функцией `suggest_grid()` исходя из
диапазона f_nodes таблицы состояния (mooring/sailing), а не
копируются из параметров EKF-окна.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, spectrogram
 
 
# ============================================================
# Таблицы ASD [м/с^2 / sqrt(Гц)], узлы по осям и состояниям
# (скопировано без изменений из основного скрипта)
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
                         platform_atten_db=-80.0,
                         return_components=False):
    """
    Генерация одноосевой реализации вибрационного ускорения платформы,
    приближённой к измеренным ASD (Qiao 2025) для состояний
    'mooring' и 'sailing'. Подробности физики/параметров -- см. docstring
    в основном скрипте симуляции (simul_acc.py); здесь код идентичен.
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
 
    target_asd = target_asd * (10 ** (platform_atten_db / 20.0))
 
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
    ASD выбранного состояния (`state`), а не взятые "снаружи" (как
    раньше -- из параметров совсем другого EKF-окна).
 
    Требования:
      - Nyquist: fs = 1/dt должна быть заметно выше верхнего узла
        таблицы f_max, иначе высокочастотная часть ASD (горб/спад
        в районе десятков-сотен Гц) будет замэплена / обрезана.
        Берём fs = high_margin * f_max (fs > 2*f_max гарантированно).
      - Частотное разрешение БПФ Df = 1/T = 1/(N*dt) должно быть
        заметно МЕНЬШЕ нижнего узла таблицы f_min, иначе низкочастотная
        структура (провал/горб у долей Гц, как на Fig. 1) просто не
        попадёт ни в одну частотную ячейку. Берём
        Df = f_min / low_res_factor.
 
    Так как таблица охватывает ~6 декад (1e-3..1e3 Гц), "честное" N
    получается очень большим (миллионы-десятки миллионов отсчётов).
    Если оно превышает `max_N`, N обрезается до max_N, fs сохраняется
    (чтобы не потерять высокочастотную часть), а достигнутое разрешение
    Df оказывается хуже желаемого -- об этом печатается предупреждение.
 
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
# Визуальный контроль: временной ряд + спектр (ASD) + спектрограмма
# ============================================================
 
def plot_vibration_overview(N=None, dt=None, axis='z', state='mooring', seed=42,
                             hp_cutoff=0.01, platform_atten_db=-80.0,
                             nperseg_frac=8, grid_kwargs=None):
    """
    Строит для одной реализации a(t):
      - временной ряд;
      - целевую ASD (табличную, по которой строился сигнал) и
        Welch-оценку ASD по самой реализации -- для проверки, что
        сгенерированный сигнал действительно соответствует таблице;
      - спектрограмму (time-frequency picture).
 
    Если N и dt не заданы явно, они подбираются автоматически функцией
    `suggest_grid(state, ...)` -- под диапазон f_nodes данного состояния,
    а не берутся "снаружи" из параметров другого окна/пайплайна.
    `grid_kwargs` -- доп. параметры, передаваемые в suggest_grid
    (low_res_factor, high_margin, max_N).
    """
    if N is None or dt is None:
        grid = suggest_grid(state, **(grid_kwargs or {}))
        N, dt = grid['N'], grid['dt']
 
    a_t, dbg = gen_vibration_trace(N, dt, axis=axis, state=state, seed=seed,
                                    hp_cutoff=hp_cutoff,
                                    platform_atten_db=platform_atten_db,
                                    return_components=True)
    fs = 1.0 / dt
    t = np.arange(N) * dt
 
    # nperseg для Welch/спектрограммы: не может быть больше N; стараемся
    # не мельчить сегмент сильнее, чем нужно для разумного усреднения,
    # но и не терять частотное разрешение у нижних узлов таблицы.
    nperseg = int(np.clip(N // nperseg_frac, 1024, N))
    f_welch, Pxx = welch(a_t, fs=fs, nperseg=nperseg)
    asd_welch = np.sqrt(Pxx)
 
    f_spec, t_spec, Sxx = spectrogram(a_t, fs=fs, nperseg=nperseg,
                                       noverlap=nperseg // 2)
 
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
 
    axs[0].plot(t, a_t, lw=0.6)
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel(r'a, м/с$^2$')
    axs[0].set_title(f'Временной ряд: {axis}, {state}')
 
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
                 f'hp_cutoff={hp_cutoff} Гц, platform_atten_db={platform_atten_db} дБ')
    fig.tight_layout()
    return fig, a_t, dbg
 
 
if __name__ == "__main__":
    # (N, dt) больше НЕ берутся из параметров EKF-окна основного скрипта
    # (N_RP=16384, T_RP=33e-3) -- та сетка была рассчитана для другой
    # задачи и физически не могла показать спектр ниже ~30 Гц.
    # Вместо этого сетка подбирается автоматически под таблицу ASD
    # каждого состояния функцией suggest_grid().
 
    HP_CUTOFF = 0.01
    PLATFORM_ATTEN_DB = 0.0   # поставьте -80.0, чтобы увидеть "ослабленный"
                              # сигнал, который реально используется
                              # в simul_acc.py по умолчанию
 
    # max_N=4_000_000 -- разумный компромисс между честным разрешением
    # у нижних узлов таблицы (~1e-3 Гц) и временем/памятью на irfft +
    # filtfilt. При необходимости увеличьте, если позволяют ресурсы.
    GRID_KWARGS = dict(low_res_factor=3.0, high_margin=2.5, max_N=4_000_000)
 
    for state in ('mooring', 'sailing'):
        grid = suggest_grid(state, **GRID_KWARGS)
        for axis in ('x', 'y', 'z'):
            plot_vibration_overview(N=grid['N'], dt=grid['dt'], axis=axis,
                                     state=state, seed=42,
                                     hp_cutoff=HP_CUTOFF,
                                     platform_atten_db=PLATFORM_ATTEN_DB)
 
    plt.show()