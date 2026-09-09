"""Genera figuras sintéticas y reproducibles para las semanas 6 a 10 de SYSB."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

NAVY = "#0B3558"
TEAL = "#007F82"
CORAL = "#E06B52"
GOLD = "#E8B44A"
GRAY = "#6B7785"
PAPER = "#F7F5F0"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
    }
)


def finish(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def synthetic_ecg(t: np.ndarray, hr_hz: float = 1.0) -> np.ndarray:
    phase = np.mod(t, 1 / hr_hz) * hr_hz
    g = lambda mu, sigma, amp: amp * np.exp(-0.5 * ((phase - mu) / sigma) ** 2)
    return (
        g(0.18, 0.025, 0.12)
        + g(0.36, 0.012, -0.16)
        + g(0.40, 0.010, 1.00)
        + g(0.43, 0.014, -0.28)
        + g(0.68, 0.060, 0.30)
    )


def aliasing() -> None:
    f0, fs = 70.0, 100.0
    t = np.linspace(0, 0.12, 1800)
    n = np.arange(0, 0.12, 1 / fs)
    xa = np.sin(2 * np.pi * f0 * t)
    xs = np.sin(2 * np.pi * f0 * n)
    falias = abs(f0 - round(f0 / fs) * fs)
    xalias = np.sin(2 * np.pi * (-falias) * t)
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.plot(t * 1000, xa, color=GRAY, lw=2.0, label="Señal analógica: 70 Hz")
    ax.plot(t * 1000, xalias, color=TEAL, lw=2.2, ls="--", label="Sinusoide indistinguible: 30 Hz")
    ax.stem(n * 1000, xs, linefmt=CORAL, markerfmt="o", basefmt=" ", label="Muestras a 100 Hz")
    ax.axhline(0, color=NAVY, lw=0.8)
    ax.set(xlabel="Tiempo [ms]", ylabel="Amplitud normalizada", title="Aliasing: las muestras no identifican de forma única 70 Hz")
    ax.legend(ncols=3, loc="upper center", frameon=False)
    ax.grid(alpha=0.18)
    finish(fig, "aliasing.png")


def quantization() -> None:
    fs = 500
    t = np.arange(0, 2, 1 / fs)
    x = synthetic_ecg(t) + 0.015 * np.sin(2 * np.pi * 0.4 * t)
    xmin, xmax = -0.4, 1.2
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.5), sharex=True)
    for bits, color in [(4, CORAL), (8, TEAL)]:
        levels = 2**bits
        delta = (xmax - xmin) / levels
        q = np.clip(np.round((x - xmin) / delta) * delta + xmin, xmin, xmax - delta)
        axes[0].plot(t, q, lw=1.2, color=color, label=f"{bits} bits")
        axes[1].plot(t, x - q, lw=1.0, color=color, label=f"error, {bits} bits")
    axes[0].plot(t, x, color=NAVY, lw=2.0, alpha=0.80, label="Señal sintética")
    axes[0].set(ylabel="Amplitud", title="La cuantización discretiza amplitud; no discretiza tiempo")
    axes[1].set(xlabel="Tiempo [s]", ylabel="Error")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(ncols=3, frameon=False, loc="upper right")
    finish(fig, "quantization.png")


def convolution() -> None:
    x = np.zeros(30)
    x[4:12] = 1.0
    h = np.ones(5) / 5
    y = np.convolve(x, h)
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 7.1), sharex=False)
    for ax, data, title, color in [
        (axes[0], x, r"Entrada $x[n]$: pulso", NAVY),
        (axes[1], h, r"Respuesta al impulso $h[n]$: promedio de 5 puntos", CORAL),
        (axes[2], y, r"Salida $y[n]=x[n]*h[n]$", TEAL),
    ]:
        markerline, stemlines, baseline = ax.stem(np.arange(len(data)), data, basefmt=" ")
        plt.setp(markerline, color=color)
        plt.setp(stemlines, color=color, linewidth=1.7)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.16)
    axes[-1].set_xlabel("Índice n")
    finish(fig, "convolution.png")


def fourier_reconstruction() -> None:
    t = np.linspace(-np.pi, np.pi, 2400)
    target = np.sign(np.sin(t))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.7), sharex=True, sharey=True)
    for ax, nmax, color in zip(axes.flat, [1, 3, 9, 39], [GOLD, CORAL, TEAL, NAVY]):
        y = np.zeros_like(t)
        for k in range(1, nmax + 1, 2):
            y += 4 / (np.pi * k) * np.sin(k * t)
        ax.plot(t, target, color=GRAY, lw=1.3, alpha=0.75, label="Objetivo")
        ax.plot(t, y, color=color, lw=2.2, label=f"Armónicos hasta {nmax}")
        ax.set_title(f"N = {nmax}")
        ax.grid(alpha=0.16)
        ax.set_ylim(-1.45, 1.45)
    axes[1, 0].set_xlabel("t [rad]")
    axes[1, 1].set_xlabel("t [rad]")
    axes[0, 0].set_ylabel("Amplitud")
    axes[1, 0].set_ylabel("Amplitud")
    fig.suptitle("Reconstrucción de una señal periódica con senos armónicos", color=NAVY, fontsize=18)
    finish(fig, "fourier_reconstruction.png")


def spectral_leakage() -> None:
    fs, n = 128.0, 128
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 10.5 * t)
    f = np.fft.rfftfreq(n, 1 / fs)
    spectra = []
    for w in [np.ones(n), np.hanning(n)]:
        spectra.append(np.abs(np.fft.rfft(x * w)) / np.sum(w) * 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), sharey=True)
    for ax, spec, label, color in zip(axes, spectra, ["Ventana rectangular", "Ventana Hann"], [CORAL, TEAL]):
        ax.plot(f, spec, color=color, lw=2.2)
        ax.axvline(10.5, color=NAVY, ls="--", lw=1.2)
        ax.set(xlim=(0, 35), xlabel="Frecuencia [Hz]", title=label)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Magnitud normalizada")
    fig.suptitle("Fuga espectral: observar un segmento equivale a multiplicar por una ventana", color=NAVY, fontsize=17)
    finish(fig, "spectral_leakage.png")


def pole_zero() -> None:
    b = np.array([1.0, -0.4])
    a = np.array([1.0, -0.82])
    z, p, _ = signal.tf2zpk(b, a)
    w, h = signal.freqz(b, a, worN=1024)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5))
    theta = np.linspace(0, 2 * np.pi, 500)
    axes[0].plot(np.cos(theta), np.sin(theta), color=GRAY, ls="--", lw=1.5)
    axes[0].scatter(z.real, z.imag, facecolors="none", edgecolors=TEAL, s=140, linewidths=2.5, label="Ceros")
    axes[0].scatter(p.real, p.imag, marker="x", color=CORAL, s=130, linewidths=3, label="Polos")
    axes[0].axhline(0, color=GRAY, lw=0.8)
    axes[0].axvline(0, color=GRAY, lw=0.8)
    axes[0].set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xlabel="Re{z}", ylabel="Im{z}", title="Plano z")
    axes[0].set_aspect("equal")
    axes[0].legend(frameon=False)
    axes[1].plot(w / np.pi, 20 * np.log10(np.maximum(np.abs(h), 1e-9)), color=NAVY, lw=2.3)
    axes[1].set(xlabel=r"Frecuencia normalizada $\omega/\pi$", ylabel="Magnitud [dB]", title="Respuesta sobre el círculo unidad")
    axes[1].grid(alpha=0.18)
    finish(fig, "pole_zero_response.png")


def fir_design() -> None:
    fs = 250.0
    taps = signal.firwin(81, 40, fs=fs, window="hamming")
    w, h = signal.freqz(taps, worN=2048, fs=fs)
    n = np.arange(len(taps))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5))
    axes[0].stem(n, taps, linefmt=TEAL, markerfmt="o", basefmt=" ")
    axes[0].axvline((len(taps) - 1) / 2, color=CORAL, ls="--")
    axes[0].set(xlabel="n", ylabel="h[n]", title="FIR simétrico: fase lineal")
    axes[0].grid(alpha=0.16)
    axes[1].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-9)), color=NAVY, lw=2.2)
    axes[1].axvline(40, color=CORAL, ls="--", label="40 Hz")
    axes[1].set(xlim=(0, fs / 2), ylim=(-100, 5), xlabel="Frecuencia [Hz]", ylabel="Magnitud [dB]", title="Pasa-bajas, fs = 250 Hz")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    finish(fig, "fir_design.png")


def iir_emg() -> None:
    rng = np.random.default_rng(31)
    fs = 2000.0
    t = np.arange(0, 3.0, 1 / fs)
    white = rng.standard_normal(t.size)
    muscle_sos = signal.butter(4, [25, 380], btype="bandpass", fs=fs, output="sos")
    muscle = signal.sosfiltfilt(muscle_sos, white)
    env = 0.18 + 0.82 * np.exp(-0.5 * ((t - 1.5) / 0.42) ** 2)
    clean = muscle * env
    contaminated = clean + 0.55 * np.sin(2 * np.pi * 5 * t) + 0.18 * np.sin(2 * np.pi * 720 * t)
    design_sos = signal.butter(6, [20, 450], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(design_sos, contaminated)
    f0, p0 = signal.welch(contaminated, fs, nperseg=1024)
    f1, p1 = signal.welch(filtered, fs, nperseg=1024)
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.7))
    sel = (t >= 1.0) & (t <= 2.0)
    axes[0].plot(t[sel], contaminated[sel], color=GRAY, lw=0.8, label="Antes")
    axes[0].plot(t[sel], filtered[sel], color=TEAL, lw=0.9, label="Después")
    axes[0].set(ylabel="Amplitud", title="Ejemplo EMG sintético: segmento temporal")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].semilogy(f0, p0, color=GRAY, lw=1.4, label="Antes")
    axes[1].semilogy(f1, p1, color=TEAL, lw=2.0, label="Después")
    axes[1].axvspan(20, 450, color=GOLD, alpha=0.18, label="Banda objetivo")
    axes[1].set(xlim=(0, 900), xlabel="Frecuencia [Hz]", ylabel="PSD", title="Densidad espectral de potencia")
    axes[1].legend(frameon=False, ncols=3)
    for ax in axes:
        ax.grid(alpha=0.16)
    finish(fig, "iir_emg.png")


def phase_comparison() -> None:
    fs = 500.0
    t = np.arange(0, 2.0, 1 / fs)
    x = synthetic_ecg(t) + 0.07 * np.sin(2 * np.pi * 70 * t)
    sos = signal.butter(4, 35, btype="low", fs=fs, output="sos")
    y_causal = signal.sosfilt(sos, x)
    y_zero = signal.sosfiltfilt(sos, x)
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.3), sharex=True)
    axes[0].plot(t, x, color=GRAY, lw=1.0, label="Entrada")
    axes[0].plot(t, y_causal, color=CORAL, lw=2.0, label="IIR causal")
    axes[1].plot(t, x, color=GRAY, lw=1.0, label="Entrada")
    axes[1].plot(t, y_zero, color=TEAL, lw=2.0, label="Filtrado hacia adelante y atrás")
    axes[0].set(title="El filtrado causal introduce retardo de fase", ylabel="Amplitud")
    axes[1].set(title="El filtrado fuera de línea puede anular fase, pero no es causal", xlabel="Tiempo [s]", ylabel="Amplitud")
    for ax in axes:
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, ncols=2)
    finish(fig, "phase_comparison.png")


if __name__ == "__main__":
    aliasing()
    quantization()
    convolution()
    fourier_reconstruction()
    spectral_leakage()
    pole_zero()
    fir_design()
    iir_emg()
    phase_comparison()
