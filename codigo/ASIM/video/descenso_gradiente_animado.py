"""Genera un GIF didáctico sobre descenso de gradiente.

El video compara, iteración por iteración:

1. Regresión lineal optimizada con error cuadrático medio.
2. Regresión logística optimizada con entropía cruzada binaria.

Los modelos se implementan directamente con NumPy. No se emplean estimadores
de scikit-learn, de modo que el cálculo del gradiente queda visible en el código.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


# -----------------------------------------------------------------------------
# 1. Carga de los dos conjuntos de datos artificiales
# -----------------------------------------------------------------------------
DIRECTORIO = Path(__file__).resolve().parent
RUTA_LINEAL = DIRECTORIO / "datos_regresion_lineal.csv"
RUTA_LOGISTICA = DIRECTORIO / "datos_regresion_logistica.csv"


def cargar_datos() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida los dos CSV que acompañan al programa."""
    datos_lineales = pd.read_csv(RUTA_LINEAL)
    datos_logisticos = pd.read_csv(RUTA_LOGISTICA)

    if list(datos_lineales.columns) != ["x", "y"]:
        raise ValueError("El CSV lineal debe contener exactamente las columnas x,y.")
    if list(datos_logisticos.columns) != ["x", "clase"]:
        raise ValueError(
            "El CSV logístico debe contener exactamente las columnas x,clase."
        )

    x_lineal = datos_lineales["x"].to_numpy(dtype=float)
    y_lineal = datos_lineales["y"].to_numpy(dtype=float)
    x_logistica = datos_logisticos["x"].to_numpy(dtype=float)
    y_logistica = datos_logisticos["clase"].to_numpy(dtype=float)

    if not all(
        np.isfinite(v).all()
        for v in (x_lineal, y_lineal, x_logistica, y_logistica)
    ):
        raise ValueError("Los conjuntos de datos contienen valores no finitos.")
    if set(np.unique(y_logistica)) != {0.0, 1.0}:
        raise ValueError("La variable clase solo puede contener 0 y 1.")

    return x_lineal, y_lineal, x_logistica, y_logistica


# Los datos se cargan al inicio, antes de entrenar o dibujar cualquier modelo.
X_LINEAL, Y_LINEAL, X_LOGISTICA, Y_LOGISTICA = cargar_datos()


# -----------------------------------------------------------------------------
# 2. Funciones matemáticas y descenso de gradiente
# -----------------------------------------------------------------------------
def sigmoide(z: np.ndarray) -> np.ndarray:
    """Calcula la función sigmoide de manera numéricamente estable."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def entrenar_regresion_lineal(
    x: np.ndarray,
    y: np.ndarray,
    iteraciones: int,
    tasa_aprendizaje: float = 0.055,
) -> dict[str, np.ndarray]:
    """Optimiza y_hat = w*x + b mediante descenso de gradiente por lotes.

    Función de costo:
        J(w,b) = (1 / 2n) * sum((y_hat - y)^2)

    Gradientes:
        dJ/dw = promedio((y_hat - y) * x)
        dJ/db = promedio(y_hat - y)
    """
    w, b = -1.8, -1.0
    historial_w, historial_b, historial_costo = [], [], []

    for iteracion in range(iteraciones + 1):
        prediccion = w * x + b
        error = prediccion - y
        costo = 0.5 * np.mean(error**2)

        historial_w.append(w)
        historial_b.append(b)
        historial_costo.append(costo)

        if iteracion < iteraciones:
            gradiente_w = np.mean(error * x)
            gradiente_b = np.mean(error)
            w -= tasa_aprendizaje * gradiente_w
            b -= tasa_aprendizaje * gradiente_b

    return {
        "w": np.asarray(historial_w),
        "b": np.asarray(historial_b),
        "costo": np.asarray(historial_costo),
    }


def entrenar_regresion_logistica(
    x: np.ndarray,
    y: np.ndarray,
    iteraciones: int,
    tasa_aprendizaje: float = 0.8,
) -> dict[str, np.ndarray]:
    """Optimiza p(y=1|x) = sigmoide(w*x + b) con descenso de gradiente.

    Función de costo:
        J(w,b) = -promedio(y*log(p) + (1-y)*log(1-p))

    Gradientes:
        dJ/dw = promedio((p-y) * x)
        dJ/db = promedio(p-y)
    """
    w, b = -1.3, 0.9
    historial_w, historial_b, historial_costo, historial_exactitud = [], [], [], []
    epsilon = 1e-12

    for iteracion in range(iteraciones + 1):
        probabilidad = sigmoide(w * x + b)
        p_segura = np.clip(probabilidad, epsilon, 1.0 - epsilon)
        costo = -np.mean(y * np.log(p_segura) + (1.0 - y) * np.log(1.0 - p_segura))
        clase_predicha = (probabilidad >= 0.5).astype(float)

        historial_w.append(w)
        historial_b.append(b)
        historial_costo.append(costo)
        historial_exactitud.append(np.mean(clase_predicha == y))

        if iteracion < iteraciones:
            error = probabilidad - y
            gradiente_w = np.mean(error * x)
            gradiente_b = np.mean(error)
            w -= tasa_aprendizaje * gradiente_w
            b -= tasa_aprendizaje * gradiente_b

    return {
        "w": np.asarray(historial_w),
        "b": np.asarray(historial_b),
        "costo": np.asarray(historial_costo),
        "exactitud": np.asarray(historial_exactitud),
    }


# -----------------------------------------------------------------------------
# 3. Construcción de la animación
# -----------------------------------------------------------------------------
def crear_animacion(
    salida: Path,
    iteraciones: int = 100,
    fps: int = 12,
    dpi: int = 100,
) -> None:
    """Crea un GIF 16:9 con ambos procesos de optimización."""
    if iteraciones < 10:
        raise ValueError("Use al menos 10 iteraciones para apreciar la convergencia.")
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps y dpi deben ser valores positivos.")

    historia_lineal = entrenar_regresion_lineal(
        X_LINEAL, Y_LINEAL, iteraciones=iteraciones
    )
    historia_logistica = entrenar_regresion_logistica(
        X_LOGISTICA, Y_LOGISTICA, iteraciones=iteraciones
    )

    color_actual = "#E4572E"
    color_final = "#198754"
    color_datos = "#315B7D"
    color_costo = "#6F42C1"
    color_rejilla = "#D7DEE7"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "axes.edgecolor": "#A8B3C1",
            "text.color": "#243447",
            "xtick.color": "#52606D",
            "ytick.color": "#52606D",
        }
    )

    figura, ejes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.2),
        gridspec_kw={"height_ratios": [2.15, 1.0]},
        constrained_layout=False,
    )
    figura.patch.set_facecolor("#F7F9FC")
    figura.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.87, hspace=0.36, wspace=0.22)

    ax_lineal, ax_logistica = ejes[0]
    ax_costo_lineal, ax_costo_logistico = ejes[1]
    for ax in ejes.flat:
        ax.set_facecolor("white")
        ax.grid(True, color=color_rejilla, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)

    titulo_general = figura.suptitle(
        "Descenso de gradiente · iteración 0",
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )
    subtitulo = figura.text(
        0.5,
        0.915,
        "Los parámetros se actualizan en la dirección que reduce la función de costo",
        ha="center",
        va="center",
        fontsize=11,
        color="#52606D",
    )

    # Panel de regresión lineal.
    x_linea = np.linspace(X_LINEAL.min() - 0.15, X_LINEAL.max() + 0.15, 300)
    ax_lineal.scatter(
        X_LINEAL,
        Y_LINEAL,
        s=31,
        color=color_datos,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        label="Datos observados",
        zorder=3,
    )
    w_final_lineal = historia_lineal["w"][-1]
    b_final_lineal = historia_lineal["b"][-1]
    ax_lineal.plot(
        x_linea,
        w_final_lineal * x_linea + b_final_lineal,
        color=color_final,
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="Modelo al converger",
    )
    (linea_lineal,) = ax_lineal.plot(
        [], [], color=color_actual, linewidth=3.0, label="Modelo en esta iteración"
    )
    indices_residuos = np.linspace(0, len(X_LINEAL) - 1, 10, dtype=int)
    residuos = LineCollection([], colors=color_actual, linewidths=1.1, alpha=0.35)
    ax_lineal.add_collection(residuos)
    texto_lineal = ax_lineal.text(
        0.025,
        0.955,
        "",
        transform=ax_lineal.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#D7DEE7", "alpha": 0.94},
    )
    ax_lineal.set_title("A · Regresión lineal: ajustar una recta", fontsize=13)
    ax_lineal.set_xlabel("Variable explicativa, x")
    ax_lineal.set_ylabel("Respuesta continua, y")
    ax_lineal.set_xlim(x_linea.min(), x_linea.max())
    margen_y = 0.7
    ax_lineal.set_ylim(Y_LINEAL.min() - margen_y, Y_LINEAL.max() + margen_y)
    ax_lineal.legend(loc="lower right", fontsize=8.5, framealpha=0.94)

    # Panel de regresión logística.
    x_curva = np.linspace(X_LOGISTICA.min() - 0.15, X_LOGISTICA.max() + 0.15, 400)
    mascara_cero = Y_LOGISTICA == 0
    mascara_uno = Y_LOGISTICA == 1
    ax_logistica.scatter(
        X_LOGISTICA[mascara_cero],
        Y_LOGISTICA[mascara_cero],
        s=34,
        color="#277DA1",
        edgecolor="white",
        linewidth=0.6,
        label="Clase 0",
        zorder=3,
    )
    ax_logistica.scatter(
        X_LOGISTICA[mascara_uno],
        Y_LOGISTICA[mascara_uno],
        s=34,
        color="#F8961E",
        edgecolor="white",
        linewidth=0.6,
        label="Clase 1",
        zorder=3,
    )
    w_final_log = historia_logistica["w"][-1]
    b_final_log = historia_logistica["b"][-1]
    ax_logistica.plot(
        x_curva,
        sigmoide(w_final_log * x_curva + b_final_log),
        color=color_final,
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="Probabilidad al converger",
    )
    (curva_logistica,) = ax_logistica.plot(
        [], [], color=color_actual, linewidth=3.0, label="Probabilidad actual"
    )
    frontera = ax_logistica.axvline(
        0.0, color=color_actual, linestyle=":", linewidth=1.8, alpha=0.85
    )
    ax_logistica.axhline(
        0.5, color="#6C757D", linestyle=":", linewidth=1.2, alpha=0.85
    )
    ax_logistica.text(
        x_curva.min() + 0.08,
        0.53,
        "umbral p = 0.5",
        fontsize=8.5,
        color="#6C757D",
    )
    texto_logistico = ax_logistica.text(
        0.025,
        0.955,
        "",
        transform=ax_logistica.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#D7DEE7", "alpha": 0.94},
    )
    ax_logistica.set_title("B · Regresión logística: estimar una probabilidad", fontsize=13)
    ax_logistica.set_xlabel("Variable explicativa, x")
    ax_logistica.set_ylabel("Probabilidad estimada de la clase 1")
    ax_logistica.set_xlim(x_curva.min(), x_curva.max())
    ax_logistica.set_ylim(-0.10, 1.10)
    ax_logistica.set_yticks([0.0, 0.5, 1.0])
    ax_logistica.legend(loc="lower right", fontsize=8.2, framealpha=0.94, ncol=2)

    # Paneles de costo: la escala logarítmica permite ver bien toda la caída.
    pasos = np.arange(iteraciones + 1)
    (linea_costo_lineal,) = ax_costo_lineal.plot([], [], color=color_costo, linewidth=2.6)
    (punto_costo_lineal,) = ax_costo_lineal.plot([], [], "o", color=color_actual, markersize=6)
    ax_costo_lineal.axhline(
        historia_lineal["costo"][-1], color=color_final, linestyle="--", linewidth=1.3, alpha=0.75
    )
    ax_costo_lineal.set_title("Costo lineal: error cuadrático medio / 2", fontsize=11)
    ax_costo_lineal.set_xlabel("Iteración")
    ax_costo_lineal.set_ylabel("J(w, b)")
    ax_costo_lineal.set_xlim(0, iteraciones)
    ax_costo_lineal.set_yscale("log")
    ax_costo_lineal.set_ylim(historia_lineal["costo"].min() * 0.75, historia_lineal["costo"].max() * 1.25)

    (linea_costo_log,) = ax_costo_logistico.plot([], [], color=color_costo, linewidth=2.6)
    (punto_costo_log,) = ax_costo_logistico.plot([], [], "o", color=color_actual, markersize=6)
    ax_costo_logistico.axhline(
        historia_logistica["costo"][-1], color=color_final, linestyle="--", linewidth=1.3, alpha=0.75
    )
    ax_costo_logistico.set_title("Costo logístico: entropía cruzada binaria", fontsize=11)
    ax_costo_logistico.set_xlabel("Iteración")
    ax_costo_logistico.set_ylabel("J(w, b)")
    ax_costo_logistico.set_xlim(0, iteraciones)
    ax_costo_logistico.set_yscale("log")
    ax_costo_logistico.set_ylim(
        historia_logistica["costo"].min() * 0.82,
        historia_logistica["costo"].max() * 1.22,
    )

    def actualizar(iteracion: int):
        """Actualiza todos los artistas para una iteración del algoritmo."""
        w_lin = historia_lineal["w"][iteracion]
        b_lin = historia_lineal["b"][iteracion]
        costo_lin = historia_lineal["costo"][iteracion]
        y_actual_lineal = w_lin * x_linea + b_lin
        linea_lineal.set_data(x_linea, y_actual_lineal)

        segmentos = [
            [(X_LINEAL[j], Y_LINEAL[j]), (X_LINEAL[j], w_lin * X_LINEAL[j] + b_lin)]
            for j in indices_residuos
        ]
        residuos.set_segments(segmentos)
        texto_lineal.set_text(
            rf"$\hat{{y}} = {w_lin:.3f}x {b_lin:+.3f}$" + "\n" + rf"$J = {costo_lin:.4f}$"
        )

        w_log = historia_logistica["w"][iteracion]
        b_log = historia_logistica["b"][iteracion]
        costo_log = historia_logistica["costo"][iteracion]
        exactitud = historia_logistica["exactitud"][iteracion]
        curva_logistica.set_data(x_curva, sigmoide(w_log * x_curva + b_log))

        if abs(w_log) > 1e-10:
            x_frontera = -b_log / w_log
            frontera.set_xdata([x_frontera, x_frontera])
            frontera.set_visible(x_curva.min() <= x_frontera <= x_curva.max())
        else:
            frontera.set_visible(False)

        texto_logistico.set_text(
            rf"$p = \sigma({w_log:.3f}x {b_log:+.3f})$"
            + "\n"
            + rf"$J = {costo_log:.4f}$ · exactitud = {exactitud:.1%}"
        )

        hasta = iteracion + 1
        linea_costo_lineal.set_data(pasos[:hasta], historia_lineal["costo"][:hasta])
        punto_costo_lineal.set_data([iteracion], [costo_lin])
        linea_costo_log.set_data(pasos[:hasta], historia_logistica["costo"][:hasta])
        punto_costo_log.set_data([iteracion], [costo_log])

        if iteracion == iteraciones:
            titulo_general.set_text(
                f"Descenso de gradiente · iteración {iteracion} de {iteraciones} · modelo ajustado"
            )
            subtitulo.set_text(
                "Al disminuir el costo, la recta y la curva sigmoide se estabilizan cerca de la solución"
            )
        else:
            titulo_general.set_text(
                f"Descenso de gradiente · iteración {iteracion} de {iteraciones}"
            )
            subtitulo.set_text(
                "Los parámetros se actualizan en la dirección que reduce la función de costo"
            )

        return (
            linea_lineal,
            residuos,
            texto_lineal,
            curva_logistica,
            frontera,
            texto_logistico,
            linea_costo_lineal,
            punto_costo_lineal,
            linea_costo_log,
            punto_costo_log,
            titulo_general,
            subtitulo,
        )

    # Cada valor de 0 a iteraciones genera un fotograma. Se repiten algunos
    # fotogramas iniciales y finales para facilitar la explicación en clase.
    fotogramas = [0] * max(1, fps // 2)
    fotogramas += list(range(iteraciones + 1))
    fotogramas += [iteraciones] * max(1, fps * 2)

    animacion = FuncAnimation(
        figura,
        actualizar,
        frames=fotogramas,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    salida = salida.expanduser().resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)
    animacion.save(salida, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figura)

    print(f"GIF generado: {salida}")
    print(
        "Parámetros finales lineales: "
        f"w={historia_lineal['w'][-1]:.4f}, b={historia_lineal['b'][-1]:.4f}, "
        f"J={historia_lineal['costo'][-1]:.6f}"
    )
    print(
        "Parámetros finales logísticos: "
        f"w={historia_logistica['w'][-1]:.4f}, b={historia_logistica['b'][-1]:.4f}, "
        f"J={historia_logistica['costo'][-1]:.6f}, "
        f"exactitud={historia_logistica['exactitud'][-1]:.2%}"
    )


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un GIF didáctico de descenso de gradiente."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=DIRECTORIO / "descenso_gradiente_regresiones.gif",
        help="Ruta del GIF de salida.",
    )
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=100,
        help="Número de actualizaciones de cada modelo (predeterminado: 100).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Fotogramas por segundo (predeterminado: 12).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Resolución del GIF; 100 produce 1280x720 px.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = leer_argumentos()
    crear_animacion(
        salida=argumentos.salida,
        iteraciones=argumentos.iteraciones,
        fps=argumentos.fps,
        dpi=argumentos.dpi,
    )
