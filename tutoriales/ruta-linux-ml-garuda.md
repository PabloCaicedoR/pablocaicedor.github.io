# Ruta: Linux para ML/DL (IA causal y explicable) en Garuda Linux

Supuesto de partida: equipo con **GPU NVIDIA** (el caso más común para deep learning). Si tu GPU es AMD o no tienes GPU dedicada, la sección 1.3 cambia; avísame.

Todos los comandos son para que los ejecutes tú. Garuda usa **fish** como shell por defecto; los comandos de abajo funcionan igual en fish y en bash salvo donde se indique.

---

## Parte 1 · Primer día en Garuda (lo esencial para ML)

### 1.1 Lo que cambia respecto a Ubuntu / Pop!_OS

| Ubuntu / Pop!_OS | Garuda (Arch) |
|---|---|
| `apt install x` | `sudo pacman -S x` |
| `apt update && apt upgrade` | `garuda-update` (actualiza todo y crea snapshot antes) |
| PPAs | AUR, vía `paru -S x` (revisa el PKGBUILD antes de instalar) |
| `apt search x` | `pacman -Ss x` / `paru -Ss x` |
| `dpkg -L x` | `pacman -Ql x` |
| Versiones fijas por release | **Rolling release**: todo se actualiza continuamente |

La consecuencia práctica del rolling release: el Python y el CUDA del sistema pueden ir **por delante** de lo que soportan PyTorch o TensorFlow. Por eso, en ML nunca dependas del Python del sistema (ver 1.4).

### 1.2 Actualizar y proteger el sistema con snapshots

```bash
garuda-update                     # primera actualización completa
sudo snapper list                 # ver snapshots existentes (Btrfs)
```

Garuda viene con Btrfs + Snapper y crea un snapshot antes de cada actualización. Si algo se rompe, en el menú de GRUB eliges un snapshot y arrancas desde él; luego lo restauras con **Btrfs Assistant**. Esto es tu red de seguridad para experimentar: aprende a usarlo el primer día.

Buena práctica: excluye de los snapshots las carpetas con datasets y checkpoints grandes (ponlas en un subvolumen propio, p. ej. `/data`), o tus snapshots crecerán muchísimo.

### 1.3 Driver NVIDIA

Garuda detecta el hardware al instalar. Compruébalo:

```bash
nvidia-smi                        # debe mostrar la GPU, driver y versión CUDA máxima soportada
```

Si no aparece, abre **Garuda Settings Manager → Hardware Configuration** e instala el perfil NVIDIA propietario. No mezcles drivers instalados a mano con los del gestor.

### 1.4 Python para ML: entornos aislados, nunca el Python del sistema

Recomendación: **uv** (rápido, fija la versión de Python por proyecto).

```bash
sudo pacman -S uv
mkdir -p ~/proyectos/xai-demo && cd ~/proyectos/xai-demo
uv init --python 3.12
uv add torch torchvision            # las ruedas de PyTorch traen su propio CUDA
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Punto clave: **no necesitas instalar CUDA del sistema** para PyTorch; basta con el driver. Instala `cuda` con pacman solo si vas a compilar extensiones CUDA propias.

Alternativa si prefieres conda: **miniforge** o **pixi** (útiles cuando necesitas paquetes no-Python como librerías de C).

### 1.5 Contenedores (reproducibilidad)

```bash
sudo pacman -S docker docker-compose nvidia-container-toolkit
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # cierra sesión y vuelve a entrar
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Si el último comando muestra tu GPU, ya puedes correr experimentos en contenedores idénticos a los de un servidor o la nube.

### 1.6 Herramientas de monitoreo que usarás a diario

```bash
sudo pacman -S nvtop btop tmux git-lfs
```

- `nvtop`: uso de GPU/VRAM por proceso.
- `btop`: CPU, RAM, disco, red.
- `tmux`: entrenamientos largos que sobreviven si cierras la terminal o la conexión SSH.

---

## Parte 2 · Ruta de aprendizaje por fases

Cada fase tiene un **entregable práctico** para comprobar que la dominas.

### Fase 1 · Fundamentos del sistema (semanas 1–2)
- Jerarquía de archivos (`/etc`, `/var`, `/home`, `/opt`), permisos y propietarios (`chmod`, `chown`, `umask`), usuarios y grupos.
- Procesos y señales: `ps`, `kill`, `nice`, `renice`.
- Gestión de paquetes: pacman, AUR, `pacman -Qdt` (huérfanos), limpieza de caché con `paccache`.
- **Entregable**: crea un grupo `ml`, una carpeta `/data` compartida con permisos correctos y documenta qué hiciste.

### Fase 2 · systemd y registros (semanas 3–4)
- `systemctl` (servicios), `journalctl` (logs), unidades de usuario (`systemctl --user`).
- **Timers** de systemd como reemplazo moderno de cron.
- **Entregable**: un servicio de usuario que lance Jupyter Lab al iniciar sesión, y un timer que respalde tus notebooks cada noche.

### Fase 3 · Shell y automatización (semanas 5–6)
- Bash scripting sólido (aunque uses fish de forma interactiva, los scripts van en bash): variables, bucles, `set -euo pipefail`, argumentos.
- Herramientas de texto: `grep`/`ripgrep`, `sed`, `awk`, `jq`, `find`, `xargs`.
- **Entregable**: un script que lance un barrido de hiperparámetros, guarde logs por ejecución y te resuma los resultados.

### Fase 4 · Almacenamiento y rendimiento (semanas 7–8)
- Btrfs a fondo: subvolúmenes, snapshots, compresión (`zstd`).
- Discos y montajes: `lsblk`, `df`, `du`, `/etc/fstab`; swap y zram.
- Cuellos de botella en entrenamiento: I/O vs CPU vs GPU (`iostat`, `nvtop`, `py-spy`, profiler de PyTorch).
- **Entregable**: diagnostica si un entrenamiento real está limitado por la carga de datos o por la GPU, y corrígelo.

### Fase 5 · GPU y stack de cómputo (semanas 9–10)
- Relación driver ↔ CUDA ↔ cuDNN ↔ PyTorch; qué significa la "CUDA Version" de `nvidia-smi`.
- Gestión de VRAM, precisión mixta, múltiples GPU (`CUDA_VISIBLE_DEVICES`).
- Qué hacer cuando una actualización del kernel rompe el driver (snapshot + kernel LTS como respaldo: `linux-lts`).
- **Entregable**: tener `linux-lts` instalado como opción de arranque y probar que la GPU funciona con ambos kernels.

### Fase 6 · Reproducibilidad (semanas 11–12)
- Entornos bloqueados (`uv.lock`, `pixi.lock`), Dockerfiles para experimentos, semillas y determinismo.
- Versionado de datos y experimentos: DVC, MLflow o Weights & Biases.
- **Entregable**: un proyecto que cualquiera pueda clonar y reproducir con un solo comando.

### Fase 7 · Trabajo remoto y servidores (semanas 13–14)
- SSH con llaves, `~/.ssh/config`, túneles para Jupyter/TensorBoard, `rsync`.
- Firewall (`ufw` o `firewalld`), usuarios en servidores compartidos.
- Introducción a gestores de colas (Slurm) si usarás clústeres universitarios.
- **Entregable**: entrenar en una máquina remota desde tu Garuda, con la sesión en tmux y los resultados sincronizados.

### Fase 8 · El stack de IA causal y explicable
Con la base anterior, instala y usa en entornos aislados:
- **Causalidad**: DoWhy, EconML, causal-learn, CausalNex.
- **Explicabilidad**: SHAP, Captum (PyTorch), LIME, InterpretML.
- Algunas de estas librerías fijan versiones antiguas de numpy/scikit-learn: aquí es donde los entornos separados por proyecto te ahorran problemas.
- **Entregable**: un proyecto que estime un efecto causal con DoWhy y explique un modelo con SHAP, empaquetado de forma reproducible (Fase 6).

---

## Referencias
- Arch Wiki (la mejor documentación de Linux que existe, útil también para Garuda): https://wiki.archlinux.org
- Wiki y foro de Garuda: https://wiki.garudalinux.org · https://forum.garudalinux.org
- PyTorch, selector de instalación: https://pytorch.org/get-started/locally/
