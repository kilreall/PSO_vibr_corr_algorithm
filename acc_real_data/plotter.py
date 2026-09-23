import numpy as np
import matplotlib.pyplot as plt

# --- Настройки ---
file_path = "data_file_192.168.55.72_2026-09-16_19-54-21.bin"
dtype = np.int16
raw_data = np.fromfile(file_path, dtype=dtype)
ch1 = raw_data
scale_factor = 20.0 / 8192.0
data = ch1.astype(np.float32) * scale_factor

idxs = np.where(data > 70)[0]
idxs = idxs[::-1]
for idx in idxs:
    data = np.delete(data, np.s_[idx-60:idx+1])
data = data[55:]

sample_rate = 488280
time = np.arange(len(data)) / sample_rate

# 2. Расчет FFT
n = len(data)
fft_result = np.fft.fft(data)             # Вычисление FFT
frequencies = np.fft.fftfreq(n, 1 / sample_rate) # Оси частот

# Берем только положительные частоты и нормируем амплитуду
positive_frequencies = frequencies[:n // 2]
amplitude = np.abs(fft_result[:n // 2]) * 2 / n

plt.figure(figsize=(12, 6))

plt.plot(positive_frequencies, amplitude, color='blue', linewidth=0.8)
plt.yscale('log')
plt.xlim([0, 3])
# plt.ylim([1e-4, 0])

plt.figure(figsize=(12, 6))
plt.plot(time, data, color='blue', linewidth=0.8)
plt.title("Red Pitaya Stream Data - Channel 1")
plt.ylabel("Voltage (V)")
# plt.xlim([0, 3])
plt.grid(True)

plt.tight_layout()
plt.show()