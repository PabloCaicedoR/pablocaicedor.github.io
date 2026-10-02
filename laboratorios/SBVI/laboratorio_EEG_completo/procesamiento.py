from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt, welch

EDF = Path("data/raw/chb01_03.edf")
SUMMARY = Path("data/raw/chb01-summary.txt")
raw = mne.io.read_raw_edf(EDF, preload=False, verbose=False)
fs = float(raw.info["sfreq"])
channel = "FP1-F7"
if channel not in raw.ch_names:
    raise ValueError("Revisar nombres reales y montaje")
text = SUMMARY.read_text()
blocks = re.split(r"File Name:\s*", text)[1:]
block = next(b for b in blocks if b.splitlines()[0].strip() == EDF.name)
starts = [int(v) for v in re.findall(
    r"Seizure(?:\s+\d+)? Start Time:\s*(\d+) seconds", block)]
ends = [int(v) for v in re.findall(
    r"Seizure(?:\s+\d+)? End Time:\s*(\d+) seconds", block)]
expected = int(re.search(r"Number of Seizures in File:\s*(\d+)", block)[1])
assert len(starts) == len(ends) == expected
intervals = list(zip(starts, ends))
assert all(0 <= a < b <= raw.n_times/fs for a, b in intervals)
# Contexto amplio; las métricas se limitarán a 2876--3156 s.
t0, t1 = 2800., 3232.
i0, i1 = round(t0*fs), round(t1*fs)
x_uv = raw.get_data(picks=[channel], start=i0, stop=i1)[0] * 1e6
sos = butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos")
xf = sosfiltfilt(sos, x_uv)
print(raw.ch_names, fs, intervals)
print(hashlib.sha256(EDF.read_bytes()).hexdigest())

def bandpower(f, p, a, b):
    inside = (f > a) & (f < b)
    grid = np.r_[a, f[inside], b]
    values = np.interp(grid, f, p)
    return np.trapezoid(values, grid)


def features(epoch, fs):
    y = epoch - epoch.mean()
    f, p = welch(y, fs=fs, window="hann",
                 nperseg=round(2*fs), noverlap=round(fs),
                 detrend="constant", scaling="density")
    total = bandpower(f, p, 0.5, 40)
    sel = (f >= 0.5) & (f <= 40)
    mass = p[sel].sum()
    h = np.nan
    if mass > 0:
        q = p[sel] / mass
        positive = q > 0
        h = -np.sum(q[positive]*np.log(q[positive])) / np.log(len(q))
    out = {"rms_uv": np.sqrt(np.mean(y*y)),
           "ll_uv": np.mean(np.abs(np.diff(epoch))),
           "entropy": h, "power_uv2": total}
    bands = {"delta": (0.5, 4), "theta": (4, 8),
             "alpha": (8, 13), "beta": (13, 30)}
    for name, (a, b) in bands.items():
        val = bandpower(f, p, a, b)
        out[name+"_uv2"] = val
        out[name+"_rel"] = val/total if total > 0 else np.nan
    return out


def ictal_overlap(a, b, intervals):
    # Intervalos oficiales no solapados, verificados al importar.
    return sum(max(0., min(b, v)-max(a, u)) for u, v in intervals)

rows = []
n, step = round(4*fs), round(2*fs)
for j in range(0, len(xf)-n+1, step):
    a, b = t0+j/fs, t0+(j+n)/fs
    if a < 2876 or b > 3156:
        continue
    e, original = xf[j:j+n], x_uv[j:j+n]
    overlap = ictal_overlap(a, b, intervals)
    state = "ictal" if overlap == 4 else (
        "sin_crisis_anotada" if overlap == 0 else "transicion")
    row = features(e, fs)
    row.update(start_s=a, end_s=b, channel=channel, state=state,
               overlap_s=overlap,
               peak_to_peak_raw_uv=np.ptp(original),
               flat_raw=bool(np.std(original) < 0.1))
    rows.append(row)
df = pd.DataFrame(rows)
Path("data/derived").mkdir(parents=True, exist_ok=True)
df.to_csv("data/derived/metricas.csv", index=False)
