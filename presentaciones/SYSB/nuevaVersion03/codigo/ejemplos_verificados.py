"""Casos sintéticos reproducibles para SYSB. No son registros de pacientes.

Uso: python codigo/ejemplos_verificados.py
Genera figuras y verifica identidades matemáticas conocidas.
"""
from pathlib import Path
import json
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#17394D", "text.color": "#17394D",
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
BLUE, TEAL, RED = "#0B3954", "#168C91", "#B94C42"
results = {}

# Identidad RMS para una realización finita.
fs = 256.0
t = np.arange(2048) / fs
x = 0.2 + np.cos(2*np.pi*10*t) + 0.3*np.sin(2*np.pi*23*t)
assert np.isclose(np.mean(x*x), np.var(x, ddof=0) + np.mean(x)**2)
results["identidad_rms"] = True

# Amplitud unilateral para N par/impar, tonos alineados con bins.
for n_count in (999, 1000):
    z = 2*np.cos(2*np.pi*17*np.arange(n_count)/n_count)
    amplitude = np.abs(np.fft.rfft(z))/n_count
    amplitude[1:-1 if n_count % 2 == 0 else None] *= 2
    assert np.isclose(amplitude[17], 2.)
results["amplitud_par_impar"] = True

# Parseval con potencia ponderada por ventana.
w = signal.get_window("hann", len(x), fftbins=True)
x0 = x-x.mean()
f, psd = signal.periodogram(x0, fs=fs, window=w, detrend=False)
p_f = psd.sum()*(f[1]-f[0])
p_t = np.sum((x0*w)**2)/np.sum(w*w)
assert np.isclose(p_f, p_t)
results["parseval_error_absoluto"] = float(abs(p_f-p_t))

# Retardo conocido.
ref = np.array([0., 1., 2., 1., 0.])
delayed = np.r_[np.zeros(3), ref]
r = signal.correlate(delayed, ref)
lags = signal.correlation_lags(len(delayed), len(ref))
assert lags[r.argmax()] == 3
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
axs[0].plot(np.arange(len(ref)), ref, "o-", color=BLUE, label="Referencia x")
axs[0].plot(np.arange(len(delayed)), delayed, "s--", color=TEAL, label="y retrasada")
axs[0].set(xlabel="Índice n", ylabel="Amplitud (u. a.)")
axs[0].legend(fontsize=12)
axs[1].stem(lags, r, linefmt=BLUE, markerfmt="o", basefmt=" ")
axs[1].axvline(3, color=RED, ls="--", label="Máximo: 3 muestras")
axs[1].set(xlabel="Retardo ℓ (muestras)", ylabel="Correlación sin normalizar")
axs[1].legend(fontsize=12)
fig.savefig(FIG / "correlacion.png", dpi=180); plt.close(fig)

# Reconstrucción de un pulso periódico, D = 1/2.
t_f = np.linspace(-1, 1, 2400)
pulse = (np.abs((t_f+.5) % 1-.5) < .25).astype(float)
fig, ax = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
ax.plot(t_f, pulse, color="#8A959B", lw=2, label="Pulso ideal")
for K, color in [(3, BLUE), (15, TEAL)]:
    reconstruction = .5*np.ones_like(t_f)
    for k in range(1, K+1):
        reconstruction += np.sinc(k*.5)*np.cos(2*np.pi*k*t_f)
    ax.plot(t_f, reconstruction, color=color, label=f"Hasta armónico {K}")
ax.set(xlabel="Tiempo / periodo", ylabel="Amplitud (u. a.)", ylim=(-.17, 1.18))
ax.legend(fontsize=12, loc="upper right")
fig.savefig(FIG / "fourier-pulso.png", dpi=180); plt.close(fig)

# Convolución continua y aproximación de la integral.
dt = .001
t_c = np.arange(0, 5, dt)
x_c = (t_c < 1).astype(float)
h_c = np.exp(-t_c)
y_c = dt*signal.convolve(x_c, h_c)[:len(t_c)]
y_exact = np.where(t_c < 1, 1-np.exp(-t_c), (np.e-1)*np.exp(-t_c))
err = np.max(np.abs(y_c-y_exact))
assert err < .0011
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
axs[0].plot(t_c, x_c, color=BLUE, label="Pulso x")
axs[0].plot(t_c, h_c, color=TEAL, label="Respuesta h")
axs[1].plot(t_c, y_exact, color=BLUE, lw=3, label="Solución por intervalos")
axs[1].plot(t_c[::100], y_c[::100], ".", color=RED, label="Suma × Ts")
for ax in axs:
    ax.set(xlabel="Tiempo (s)", ylabel="Amplitud (u. a.)")
    ax.legend(fontsize=11)
fig.savefig(FIG / "convolucion.png", dpi=180); plt.close(fig)
results["convolucion_error_maximo"] = float(err)

# Bode continuo y verificación de la frecuencia de corte.
tau = .2
fc = 1/(2*np.pi*tau)
freq = np.geomspace(.01, 100, 1000)
H = 1/(1+1j*2*np.pi*freq*tau)
assert np.isclose(abs(1/(1+1j*2*np.pi*fc*tau)), 1/np.sqrt(2))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
axs[0].semilogx(freq, 20*np.log10(abs(H)), color=BLUE)
axs[1].semilogx(freq, np.angle(H, deg=True), color=TEAL)
for ax in axs:
    ax.axvline(fc, color=RED, ls="--", label=f"fc = {fc:.3f} Hz")
    ax.set_xlabel("Frecuencia (Hz)");ax.grid(alpha=.18);ax.legend(fontsize=12)
axs[0].set_ylabel("Magnitud (dB)");axs[1].set_ylabel("Fase (grados)")
fig.savefig(FIG / "bode-primer-orden.png", dpi=180);plt.close(fig)
results["corte_primer_orden_hz"] = float(fc)

# CWT: elección de escala desde frecuencias y DWT: reconstrucción.
freq_obj = np.geomspace(4, min(40, fs/4), 64)
scales = pywt.frequency2scale("cmor1.5-1.0", freq_obj/fs)
coef, freq_cwt = pywt.cwt(x, scales, "cmor1.5-1.0", sampling_period=1/fs)
assert np.allclose(freq_cwt, freq_obj)
wavelet = pywt.Wavelet("db4")
assert pywt.dwt_max_level(len(x), wavelet.dec_len) >= 5
coeffs = pywt.wavedec(x, wavelet, level=5, mode="symmetric")
x_rec = pywt.waverec(coeffs, wavelet, mode="symmetric")[:len(x)]
assert np.allclose(x_rec, x)
results["dwt_error_relativo"] = float(np.linalg.norm(x_rec-x)/np.linalg.norm(x))
print(json.dumps(results, ensure_ascii=False, indent=2))
