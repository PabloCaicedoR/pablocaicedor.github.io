import numpy as np
import pytest

from ictus_semg import metricas as m
from ictus_semg.procesamiento import envolvente, notch, pasabanda, procesar
from ictus_semg.sintesis import CANALES, ConfigSintesis, generar_semg

FS = 2000.0
T = np.arange(int(5 * FS)) / FS


def potencia_en(x, f0):
    f, p = m.espectro(x, FS, nperseg=4096)
    return p[np.argmin(np.abs(f - f0))]


def test_notch_atenua_60hz_al_menos_20db():
    x = np.sin(2 * np.pi * 60 * T) + 0.1 * np.sin(2 * np.pi * 150 * T)
    y = notch(x, FS, f0=60)
    atenuacion_db = 10 * np.log10(potencia_en(x, 60) / potencia_en(y, 60))
    assert atenuacion_db > 20


def test_pasabanda_conserva_banda_util_y_elimina_deriva():
    x = np.sin(2 * np.pi * 100 * T) + 2 * np.sin(2 * np.pi * 0.5 * T)
    y = pasabanda(x, FS, 20, 450)
    assert abs(m.rms(y) - 1 / np.sqrt(2)) < 0.05


def test_mdf_y_mnf_de_una_senoide():
    x = np.sin(2 * np.pi * 120 * T)
    f, p = m.espectro(x, FS, nperseg=2048)
    assert m.frecuencia_mediana(f, p) == pytest.approx(120, abs=2)
    assert m.frecuencia_media(f, p) == pytest.approx(120, abs=5)


def test_indice_coactivacion_limites():
    a = np.abs(np.sin(2 * np.pi * T))
    assert m.indice_coactivacion(a, a) == pytest.approx(100)
    assert m.indice_coactivacion(a, np.zeros_like(a)) == pytest.approx(0)


def test_envolvente_no_negativa():
    assert np.all(envolvente(np.random.default_rng(0).standard_normal(4000), FS) >= 0)


def test_severidad_aumenta_coactivacion_paretica():
    ci = []
    for sev in (0.0, 1.0):
        cfg = ConfigSintesis(severidad=sev, duracion_s=12)
        env = procesar(generar_semg(cfg), cfg.fs)["envolvente"]
        ci.append(m.indice_coactivacion(env[CANALES[0]], env[CANALES[1]]))
    assert ci[1] > ci[0] + 20


def test_procesar_no_modifica_la_entrada():
    df = generar_semg(ConfigSintesis(duracion_s=5))
    copia = df.copy()
    procesar(df, 2000.0)
    assert df.equals(copia)
