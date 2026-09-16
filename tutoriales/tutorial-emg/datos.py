"""Acceso a EMGDB 1.0.0. Lector limitado a sus archivos WFDB 16."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import numpy as np

BASE = 'https://physionet.org/files/emgdb/1.0.0/'
DOI = 'https://doi.org/10.13026/C24S3D'
CARPETA = Path(__file__).resolve().parent / 'datos'
ETIQUETAS = {
    'emg_neuropathy': 'Neuropatía por radiculopatía L5',
    'emg_healthy': 'Sin historia de enfermedad neuromuscular',
    'emg_myopathy': 'Miopatía por polimiositis',
}
# Valores publicados en SHA256SUMS.txt de EMGDB 1.0.0.
HASHES = {
 'emg_healthy.dat':
 '4990f34f0ffb314f6ce70c1d2089b7ce41fa96d19fcd9fb334815ca5033966fd',
 'emg_healthy.hea':
 '61221ce33e167d246ec2ea8984ce2f25882b32b4d90bd7ef785c7b56ef95bac5',
 'emg_myopathy.dat':
 '2c310a0f61843a5a5e906d413b6ff7f51370a91e516344af511c7fe6da9eaf7f',
 'emg_myopathy.hea':
 '752093a1e3af87cd87e39f553b8725133f57a75413661208e5d93c8d2c60e73b',
 'emg_neuropathy.dat':
 '108c6c71d3792f3dba645539bd9241119fb4c6a3a0f3e515399694b072375dd5',
 'emg_neuropathy.hea':
 'a424c80b6a5cc5733456f0848c2650321cac3658d9b02b845e08b619b2807ce0',
}


def asegurar(nombre):
    if nombre not in HASHES:
        raise ValueError('Archivo fuera del dataset permitido.')
    ruta = CARPETA / nombre
    CARPETA.mkdir(exist_ok=True)
    if ruta.exists():
        contenido = ruta.read_bytes()
    else:
        with urlopen(BASE + nombre, timeout=60) as respuesta:
            contenido = respuesta.read()
    real = hashlib.sha256(contenido).hexdigest()
    if real != HASHES[nombre]:
        raise ValueError(f'Integridad incorrecta: {nombre}')
    if not ruta.exists():
        ruta.write_bytes(contenido)
    return contenido


def cargar(registro):
    if registro not in ETIQUETAS:
        raise ValueError('Registro desconocido.')
    texto = asegurar(registro + '.hea').decode('ascii')
    lineas = texto.splitlines()
    cabecera, canal = lineas[0].split(), lineas[1].split()
    if cabecera[1] != '1' or canal[1] != '16':
        raise ValueError('Este lector requiere un canal WFDB 16.')
    fs, n = float(cabecera[2]), int(cabecera[3])
    if fs != 4000 or canal[2].lower() != '10000/mv':
        raise ValueError('Muestreo o ganancia inesperados.')
    contenido = asegurar(registro + '.dat')
    cuentas = np.frombuffer(contenido, dtype='<i2')
    if len(cuentas) != n or np.any(cuentas == -32768):
        raise ValueError('Longitud incorrecta o muestras faltantes.')
    cero = int(canal[4])
    x = (cuentas.astype(float) - cero) / 10000 * 1000
    t = np.arange(n) / fs
    meta = dict(registro=registro, etiqueta=ETIQUETAS[registro],
                fs_hz=fs, n=n, duracion_s=n/fs, unidades='uV',
                modalidad='EMG intramuscular', musculo='Tibial anterior',
                doi=DOI, licencia='ODC-By 1.0',
                sha256_dat=HASHES[registro + '.dat'],
                sha256_hea=HASHES[registro + '.hea'])
    return t, x, meta


if __name__ == '__main__':
    for registro in ETIQUETAS:
        _, _, meta = cargar(registro)
        print(registro, meta['n'], meta['duracion_s'])
