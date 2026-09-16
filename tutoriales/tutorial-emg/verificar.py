import numpy as np
from datos import cargar, ETIQUETAS
from motor import preparar, medir, ventanas

esperadas = dict(emg_healthy=50860, emg_myopathy=110337,
                 emg_neuropathy=147858)
for registro in ETIQUETAS:
    t, x, meta = cargar(registro)  # Incluye SHA-256 de DAT y HEA.
    fs = meta['fs_hz']
    assert len(x) == esperadas[registro]
    assert np.allclose(np.diff(t), 1/fs)
    y = preparar(x, fs)
    tabla = ventanas(y, fs, 1, 3, 1)
    assert len(tabla) == 3
    m, _, _ = medir(y[4000:8000], fs)
    doble, _, _ = medir(2*y[4000:8000], fs)
    assert np.isclose(doble['rms_uv'], 2*m['rms_uv'])
    assert np.isclose(doble['pico_pico_uv'], 2*m['pico_pico_uv'])
    assert np.isclose(doble['cresta'], m['cresta'])
    assert np.isclose(doble['fmed_20_1000_hz'], m['fmed_20_1000_hz'])
    print(registro, 'OK', meta['n'], meta['duracion_s'])
# Calibración independiente de la primera muestra del registro L5.
_, x, _ = cargar('emg_neuropathy')
assert np.isclose(x[0], 90.0)
x[10] = np.nan
try:
    preparar(x, 4000)
except ValueError:
    pass
else:
    raise AssertionError('Se aceptó un valor no finito.')
print('Verificaciones superadas con los tres registros reales.')
