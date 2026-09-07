from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#123047"
TEAL = "#008C86"
CORAL = "#E76F51"
GOLD = "#D9A441"
SKY = "#4C93C3"
GRAY = "#6B7785"
LIGHT = "#EAF1F5"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.22,
        "grid.color": GRAY,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def synthetic_ecg(t, heart_rate=72.0):
    """Didactic morphology; it is not a patient model or diagnostic trace."""
    rr = 60.0 / heart_rate
    phase = np.mod(t, rr)

    def g(center, width, amp):
        return amp * np.exp(-0.5 * ((phase - center * rr) / (width * rr)) ** 2)

    return (
        g(0.18, 0.035, 0.12)
        + g(0.38, 0.012, -0.15)
        + g(0.40, 0.010, 1.00)
        + g(0.43, 0.014, -0.25)
        + g(0.68, 0.070, 0.30)
    )


rng = np.random.default_rng(20260907)

# 01. Four synthetic biosignal families
fs = 500
t = np.arange(0, 4, 1 / fs)
ecg = synthetic_ecg(t)
emg_raw = rng.normal(size=t.size)
sos = signal.butter(4, [20, 180], btype="bandpass", fs=fs, output="sos")
emg = signal.sosfiltfilt(sos, emg_raw) * (0.15 + 0.85 * ((t > 1.0) & (t < 2.7)))
eeg = 0.45 * np.sin(2 * np.pi * 10 * t) + 0.18 * np.sin(2 * np.pi * 22 * t) + 0.12 * rng.normal(size=t.size)
resp = np.sin(2 * np.pi * 0.28 * t) + 0.08 * np.sin(2 * np.pi * 0.56 * t)
fig, axs = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
for ax, y, title, unit, color in zip(
    axs,
    [ecg, emg, eeg, resp],
    ["ECG sintético", "EMG sintético", "EEG sintético", "Respiración sintética"],
    ["amplitud (u.a.)"] * 4,
    [TEAL, CORAL, SKY, GOLD],
):
    ax.plot(t, y, color=color, lw=1.6)
    ax.set_ylabel(unit)
    ax.set_title(title, loc="left")
    ax.grid(True)
axs[-1].set_xlabel("tiempo (s)")
save(fig, "01_biosignals_overview.png")

# 02. Measurement model: source + baseline + power line + transient artifact
t = np.arange(0, 5, 1 / fs)
clean = synthetic_ecg(t, 66)
baseline = 0.20 * np.sin(2 * np.pi * 0.25 * t)
line = 0.035 * np.sin(2 * np.pi * 50 * t)
artifact = 0.55 * np.exp(-((t - 3.25) / 0.07) ** 2)
observed = clean + baseline + line + artifact
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
axs[0].plot(t, clean, color=TEAL, lw=1.5); axs[0].set_title("Componente fisiológica idealizada", loc="left")
axs[1].plot(t, baseline + line + artifact, color=CORAL, lw=1.2); axs[1].set_title("Contaminación: deriva + red + artefacto transitorio", loc="left")
axs[2].plot(t, observed, color=NAVY, lw=1.2); axs[2].set_title("Señal observada", loc="left")
for ax in axs:
    ax.grid(True); ax.set_ylabel("u.a.")
axs[-1].set_xlabel("tiempo (s)")
save(fig, "02_measurement_noise.png")

# 03. Continuous-time concept and discrete samples
t = np.linspace(0, 1.2, 1200)
x = np.sin(2 * np.pi * 3 * t + 0.35)
ts = np.arange(0, 1.201, 0.08)
xs = np.sin(2 * np.pi * 3 * ts + 0.35)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
axs[0].plot(t, x, color=TEAL, lw=2.4); axs[0].set_title("Modelo continuo: $x(t)$")
axs[1].stem(ts, xs, linefmt=CORAL, markerfmt="o", basefmt=" "); axs[1].set_title("Secuencia discreta: $x[n]$")
for ax in axs:
    ax.set_xlabel("tiempo"); ax.set_ylabel("amplitud"); ax.grid(True)
save(fig, "03_continuous_discrete.png")

# 04. Multichannel representation
t = np.arange(0, 3, 1 / fs)
base = synthetic_ecg(t, 75)
channels = [base, 0.85 * base + 0.05 * np.sin(2 * np.pi * 0.3 * t), -0.45 * base + 0.04 * rng.normal(size=t.size)]
fig, ax = plt.subplots(figsize=(12, 4.6))
offsets = [2.2, 1.1, 0]
for y, off, name, color in zip(channels, offsets, ["canal 1", "canal 2", "canal 3"], [TEAL, SKY, CORAL]):
    ax.plot(t, y + off, color=color, lw=1.3, label=name)
ax.set_yticks(offsets, ["canal 1", "canal 2", "canal 3"]); ax.set_xlabel("tiempo (s)")
ax.set_title("Una observación multicanal es un vector que cambia con el tiempo", loc="left"); ax.grid(True, axis="x")
save(fig, "04_multichannel.png")

# 05. Temporal measures on two waveforms
t = np.linspace(0, 2, 1500)
x1 = 0.7 * np.sin(2 * np.pi * 2 * t)
x2 = 0.20 * np.sin(2 * np.pi * 2 * t) + 1.1 * np.exp(-0.5 * ((t - 1.0) / 0.055) ** 2)
fig, axs = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
for ax, y, title, color in zip(axs, [x1, x2], ["Señal A: oscilación sostenida", "Señal B: evento transitorio"], [TEAL, CORAL]):
    rms = np.sqrt(np.mean(y**2)); mean = np.mean(y)
    ax.plot(t, y, color=color, lw=2)
    ax.axhline(mean, color=GOLD, ls="--", lw=1.7, label=f"media = {mean:.2f}")
    ax.axhline(rms, color=NAVY, ls=":", lw=1.7, label=f"RMS = {rms:.2f}")
    ax.set_title(title, loc="left"); ax.legend(loc="upper right", ncols=2); ax.grid(True); ax.set_ylabel("u.a.")
axs[-1].set_xlabel("tiempo (s)")
save(fig, "05_temporal_measures.png")

# 06. RMS depends on the analysis window
fs = 1000
t = np.arange(0, 3, 1 / fs)
carrier = rng.normal(size=t.size)
env = 0.12 + 0.75 * ((t > 0.8) & (t < 1.45)) + 0.45 * ((t > 2.0) & (t < 2.55))
x = env * carrier
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(t, x, color="#B8C6D1", lw=0.7, label="EMG sintético")
for window_ms, color in [(50, CORAL), (200, TEAL)]:
    n = max(1, int(window_ms * fs / 1000))
    rms = np.sqrt(np.convolve(x**2, np.ones(n) / n, mode="same"))
    ax.plot(t, rms, color=color, lw=2.2, label=f"RMS móvil: {window_ms} ms")
ax.set_title("La ventana controla el compromiso entre detalle y suavizado", loc="left")
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud (u.a.)"); ax.legend(); ax.grid(True)
save(fig, "06_rms_windows.png")

# 07. Sinusoidal parameters
t = np.linspace(0, 2.2, 1600)
A, f, phi = 1.4, 1.0, np.pi / 6
x = A * np.cos(2 * np.pi * f * t + phi)
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(t, x, color=TEAL, lw=2.5)
ax.axhline(0, color=GRAY, lw=1)
ax.annotate("amplitud $A$", xy=(0.92, A), xytext=(1.12, 0.55), arrowprops=dict(arrowstyle="->", color=CORAL), color=CORAL, fontsize=13)
ax.annotate("periodo $T_0$", xy=(0.92, -1.05), xytext=(1.92, -1.05), ha="center", va="center", arrowprops=dict(arrowstyle="<->", color=NAVY), color=NAVY, fontsize=13)
ax.scatter([0], [A * np.cos(phi)], color=GOLD, s=70, zorder=3)
ax.text(0.04, 0.96, "fase inicial $\\phi$", transform=ax.transAxes, va="top", color=GOLD, fontsize=13)
ax.set_title("$x(t)=A\\cos(2\\pi f_0t+\\phi)$", loc="left")
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud"); ax.set_ylim(-1.8, 1.8); ax.grid(True)
save(fig, "07_sinusoid_parameters.png")

# 08. Same frequency, different phase
t = np.linspace(0, 1.5, 1200)
fig, ax = plt.subplots(figsize=(12, 4.5))
for phi, color, label in [(0, TEAL, "$\\phi=0$"), (np.pi/2, CORAL, "$\\phi=\\pi/2$"), (np.pi, SKY, "$\\phi=\\pi$")]:
    ax.plot(t, np.cos(2*np.pi*2*t + phi), lw=2, color=color, label=label)
ax.set_title("La fase cambia la alineación temporal, no la frecuencia", loc="left")
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud"); ax.legend(ncols=3); ax.grid(True)
save(fig, "08_phase_comparison.png")

# 09. Sum of commensurate periodic signals
t = np.linspace(0, 12, 3000)
x1 = np.sin(2*np.pi*t/2)
x2 = 0.6*np.cos(2*np.pi*t/3)
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for ax, y, ttl, color in zip(axs, [x1, x2, x1+x2], ["$T_1=2$ s", "$T_2=3$ s", "Suma: periodo común $T_0=6$ s"], [TEAL, SKY, CORAL]):
    ax.plot(t, y, color=color, lw=1.8); ax.set_title(ttl, loc="left"); ax.grid(True); ax.set_ylabel("u.a.")
axs[-1].axvspan(0, 6, color=GOLD, alpha=0.13, label="un periodo fundamental"); axs[-1].legend(); axs[-1].set_xlabel("tiempo (s)")
save(fig, "09_periodic_sum.png")

# 10. Signal transformations
t = np.linspace(-3, 3, 1500)
x = np.exp(-t**2) * (1 + 0.35*t)
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
items = [(x, "$x(t)$", NAVY), (2*x, "$2x(t)$", CORAL), (np.exp(-(t-1)**2)*(1+0.35*(t-1)), "$x(t-1)$", TEAL), (np.exp(-(-2*t)**2)*(1+0.35*(-2*t)), "$x(-2t)$", SKY)]
for ax, (y, ttl, color) in zip(axs.ravel(), items):
    ax.plot(t, y, color=color, lw=2.2); ax.axvline(0, color=GRAY, lw=0.8); ax.set_title(ttl); ax.grid(True)
for ax in axs[-1]: ax.set_xlabel("tiempo")
for ax in axs[:,0]: ax.set_ylabel("amplitud")
save(fig, "10_transformations.png")

# 11. Even/odd decomposition
t = np.linspace(-4, 4, 2000)
x = np.where(t >= 0, np.exp(-t), 0)
x_rev = np.where(-t >= 0, np.exp(t), 0)
xe = 0.5*(x+x_rev); xo = 0.5*(x-x_rev)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, y, ttl, color in zip(axs, [x, xe, xo], ["$x(t)$", "parte par $x_e(t)$", "parte impar $x_o(t)$"], [NAVY, TEAL, CORAL]):
    ax.plot(t, y, color=color, lw=2.2); ax.axvline(0, color=GRAY, lw=0.8); ax.set_title(ttl); ax.grid(True); ax.set_xlabel("tiempo")
axs[0].set_ylabel("amplitud")
save(fig, "11_even_odd.png")

# 12. Time-domain workflow as a compact process diagram
fig, ax = plt.subplots(figsize=(12, 3.7))
ax.axis("off")
labels = ["Inspeccionar", "Acondicionar", "Segmentar", "Medir", "Interpretar"]
colors = [NAVY, SKY, TEAL, GOLD, CORAL]
xs = np.linspace(0.08, 0.92, len(labels))
for i, (xpos, label, color) in enumerate(zip(xs, labels, colors)):
    ax.text(xpos, 0.55, label, ha="center", va="center", color="white", fontsize=14, fontweight="bold", bbox=dict(boxstyle="round,pad=0.65", fc=color, ec="none"), transform=ax.transAxes)
    if i < len(labels)-1:
        ax.annotate("", xy=(xs[i+1]-0.09, 0.55), xytext=(xpos+0.09, 0.55), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=2, color=GRAY))
ax.text(0.5, 0.16, "La interpretación exige unidades, contexto fisiológico y control de calidad", ha="center", color=NAVY, fontsize=14, transform=ax.transAxes)
save(fig, "12_time_workflow.png")

# 13. Peak detection on a synthetic ECG
fs = 500
t = np.arange(0, 7, 1/fs)
x = synthetic_ecg(t, 72) + 0.025*rng.normal(size=t.size)
peaks, props = signal.find_peaks(x, distance=int(0.6*fs), prominence=0.55)
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.plot(t, x, color=NAVY, lw=1.2, label="ECG sintético")
ax.scatter(t[peaks], x[peaks], color=CORAL, s=60, zorder=3, label="eventos detectados")
ax.set_title("Detectar un pico no equivale a interpretar un evento fisiológico", loc="left")
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud (u.a.)"); ax.legend(); ax.grid(True)
save(fig, "13_peak_detection.png")

# 14. Rectification and envelope for synthetic EMG
fs = 1000
t = np.arange(0, 3, 1/fs)
raw = rng.normal(size=t.size)
sos = signal.butter(4, [25, 250], btype="bandpass", fs=fs, output="sos")
emg = signal.sosfiltfilt(sos, raw)
true_env = 0.08 + 0.85*np.exp(-0.5*((t-1.45)/0.38)**2)
emg *= true_env
rectified = np.abs(emg)
sos_env = signal.butter(3, 5, btype="lowpass", fs=fs, output="sos")
env_est = signal.sosfiltfilt(sos_env, rectified)
fig, axs = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True)
axs[0].plot(t, emg, color=NAVY, lw=0.7); axs[0].set_title("EMG sintético", loc="left")
axs[1].plot(t, rectified, color="#C9D4DC", lw=0.6, label="rectificada"); axs[1].plot(t, env_est, color=CORAL, lw=2.4, label="envolvente")
axs[1].set_title("Rectificación + suavizado", loc="left"); axs[1].legend()
for ax in axs: ax.grid(True); ax.set_ylabel("u.a.")
axs[-1].set_xlabel("tiempo (s)")
save(fig, "14_envelope.png")

# 15. Sliding segmentation
t = np.linspace(0, 8, 2000)
x = 0.5*np.sin(2*np.pi*0.8*t) + 0.15*np.sin(2*np.pi*3.2*t)
fig, ax = plt.subplots(figsize=(12, 4.3))
ax.plot(t, x, color=NAVY, lw=1.5)
starts = [0.5, 2.0, 3.5, 5.0]
for i, start in enumerate(starts, 1):
    ax.axvspan(start, start+2.0, color=[TEAL, SKY, GOLD, CORAL][i-1], alpha=0.14)
    ax.text(start+1.0, 0.82, f"ventana {i}", ha="center", color=[TEAL, SKY, GOLD, CORAL][i-1], fontweight="bold")
ax.set_ylim(-1.0, 1.05); ax.set_title("Ventanas de 2 s con solapamiento de 0,5 s", loc="left")
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud"); ax.grid(True)
save(fig, "15_segmentation.png")

# 16. Normalization changes scale, not information quality
t = np.linspace(0, 4, 1200)
x = 2.4 + 0.65*np.sin(2*np.pi*1.1*t) + 0.25*np.sin(2*np.pi*2.3*t)
z = (x - np.mean(x))/np.std(x)
mm = (x - np.min(x))/(np.max(x)-np.min(x))
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for ax, y, ttl, color in zip(axs, [x, z, mm], ["Original: conserva unidades", "Estandarizada: media 0, desviación 1", "Min–max: intervalo [0,1]"], [NAVY, TEAL, CORAL]):
    ax.plot(t, y, color=color, lw=2); ax.set_title(ttl, loc="left"); ax.grid(True)
axs[-1].set_xlabel("tiempo (s)")
save(fig, "16_normalization.png")

# 17. Time and frequency views of the same signal
fs = 256
t = np.arange(0, 2, 1/fs)
x = 1.0*np.sin(2*np.pi*8*t) + 0.45*np.sin(2*np.pi*22*t + 0.6)
freq = np.fft.rfftfreq(t.size, 1/fs)
mag = 2*np.abs(np.fft.rfft(x))/t.size
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(t, x, color=NAVY, lw=1.7); axs[0].set_title("Dominio temporal"); axs[0].set_xlabel("tiempo (s)"); axs[0].set_ylabel("amplitud"); axs[0].grid(True)
axs[1].stem(freq, mag, linefmt=CORAL, markerfmt="o", basefmt=" "); axs[1].set_xlim(0, 40); axs[1].set_title("Contenido frecuencial"); axs[1].set_xlabel("frecuencia (Hz)"); axs[1].set_ylabel("magnitud"); axs[1].grid(True)
save(fig, "17_time_frequency.png")

# 18. Complex plane and Euler relation
theta = np.linspace(0, 2*np.pi, 500)
alpha = np.pi/3
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.plot(np.cos(theta), np.sin(theta), color="#C9D4DC", lw=2)
ax.arrow(0, 0, np.cos(alpha)*0.92, np.sin(alpha)*0.92, width=0.015, head_width=0.09, color=TEAL, length_includes_head=True)
ax.plot([np.cos(alpha), np.cos(alpha)], [0, np.sin(alpha)], ls="--", color=CORAL)
ax.plot([0, np.cos(alpha)], [np.sin(alpha), np.sin(alpha)], ls="--", color=SKY)
ax.text(0.20, 0.11, "$\\alpha$", fontsize=15, color=GOLD)
ax.text(np.cos(alpha)/2, -0.12, "$\\cos\\alpha$", ha="center", color=CORAL)
ax.text(np.cos(alpha)+0.08, np.sin(alpha)/2, "$\\sin\\alpha$", va="center", color=SKY)
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.15, 1.15); ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
ax.set_xlabel("parte real"); ax.set_ylabel("parte imaginaria"); ax.set_title("$e^{j\\alpha}=\\cos\\alpha+j\\sin\\alpha$")
save(fig, "18_complex_plane.png")

# 19. Orthogonality as a Gram matrix
t = np.linspace(0, 1, 4000, endpoint=False)
basis = [np.ones_like(t), np.sqrt(2)*np.cos(2*np.pi*t), np.sqrt(2)*np.sin(2*np.pi*t), np.sqrt(2)*np.cos(4*np.pi*t), np.sqrt(2)*np.sin(4*np.pi*t)]
G = np.array([[np.trapezoid(a*b, t) for b in basis] for a in basis])
labels = ["1", "cos 1", "sin 1", "cos 2", "sin 2"]
fig, ax = plt.subplots(figsize=(7.5, 5.3))
im = ax.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", color="white" if abs(G[i,j])>0.55 else NAVY)
ax.set_title("Productos internos: bases sinusoidales ortogonales")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "19_orthogonality.png")

# 20. Fourier partial sums for a square wave
t = np.linspace(-np.pi, np.pi, 2500)
target = np.where(np.sin(t) >= 0, 1.0, -1.0)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
for ax, n_terms, color in zip(axs, [1, 3, 15], [SKY, TEAL, CORAL]):
    y = np.zeros_like(t)
    for m in range(n_terms):
        k = 2*m+1
        y += 4/np.pi*np.sin(k*t)/k
    ax.plot(t, target, color="#C9D4DC", lw=2.0, label="objetivo")
    ax.plot(t, y, color=color, lw=2.0, label="suma parcial")
    ax.set_title(f"{n_terms} armónico(s) impar(es)"); ax.set_xlabel("tiempo"); ax.grid(True)
axs[0].set_ylabel("amplitud"); axs[-1].legend(loc="lower right")
save(fig, "20_fourier_square.png")

# 21. Gibbs effect near a jump
t = np.linspace(-0.8, 0.8, 2400)
fig, ax = plt.subplots(figsize=(12, 4.5))
for n_terms, color in [(5, SKY), (15, TEAL), (60, CORAL)]:
    y = np.zeros_like(t)
    for m in range(n_terms):
        k = 2*m+1
        y += 4/np.pi*np.sin(k*t)/k
    ax.plot(t, y, color=color, lw=2, label=f"{n_terms} armónicos")
ax.axhline(1, color=GRAY, ls="--", lw=1); ax.axhline(-1, color=GRAY, ls="--", lw=1); ax.axvline(0, color=NAVY, lw=1)
ax.set_title("Fenómeno de Gibbs: la oscilación se concentra cerca de la discontinuidad", loc="left")
ax.set_xlabel("tiempo"); ax.set_ylabel("amplitud"); ax.legend(ncols=3); ax.grid(True)
save(fig, "21_gibbs.png")

print(f"Generated {len(list(OUT.glob('*.png')))} figures in {OUT}")
