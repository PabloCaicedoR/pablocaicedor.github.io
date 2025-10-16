"""Entry point for the Accelerolog mobile application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from kivy.app import App
from kivy.logger import Logger
from kivy.utils import platform

from calibration import CalibrationController
from export import ExportManager
from sensors import SensorManager, SensorSample
from storage import CSVStorageManager
from ui import SensorOption, build_ui
from utils.time import TimeProvider


APP_VERSION = "1.0.0"
CALIBRATION_DURATION_SECONDS = 6.0


class AppController:
    """High-level coordinator for services, UI, sensors, and storage."""

    def __init__(self, root_dir: Path) -> None:
        self._time = TimeProvider()
        self._root_dir = root_dir
        self._records_dir = self._root_dir / "records"
        self._calibration_dir = self._root_dir / "calibration"
        self._logs_dir = self._root_dir / "logs"
        self._records_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path: Optional[Path] = None
        self._active_sensors: List[str] = []
        self._demo_mode = False
        self._latest_samples: Dict[str, SensorSample] = {}
        self._calibration_target: Optional[str] = None
        self._calibration_samples: List[SensorSample] = []
        self._calibration_event = None
        self._calibration_progress_callback = None
        self._calibration_elapsed = 0.0

        self._calibration = CalibrationController(storage_dir=self._calibration_dir, time_provider=self._time)
        self._sensor_manager = SensorManager(calibration=self._calibration, time_provider=self._time)
        self._sensor_manager.set_callbacks([self._on_sample])
        self._storage = CSVStorageManager(output_dir=self._records_dir, time_provider=self._time)
        self._export = ExportManager()

        sensor_options = self._build_sensor_options()
        rates = self._sensor_manager.get_supported_rates()
        callbacks = {
            "start": self._handle_start,
            "stop": self._handle_stop,
            "save": self._handle_save,
            "share": self._handle_share,
            "calibrate_start": self._handle_calibration_start,
            "calibrate_cancel": self._handle_calibration_cancel,
        }
        self.ui = build_ui(sensor_options=sensor_options, rates=rates, callbacks=callbacks)

    # Sensor option helpers -------------------------------------------------------------

    def _build_sensor_options(self) -> List[SensorOption]:
        capabilities = self._sensor_manager.list_capabilities()
        options: List[SensorOption] = []
        for key, info in capabilities.items():
            options.append(SensorOption(key=key, label=info["name"], available=info["available"] == "yes"))
        return options

    # Callbacks wired to UI -------------------------------------------------------------

    def _handle_start(self, sensors: Iterable[str], target_rate: int, demo: bool) -> None:
        self._active_sensors = list(sensors)
        self._demo_mode = demo
        metadata = self._collect_metadata(target_rate, demo)
        file_name = None
        if demo:
            file_name = "accelerolog_demo.csv"
        self._csv_path = self._storage.start_session(metadata=metadata, filename=file_name)
        self._sensor_manager.start(sensors=self._active_sensors, target_rate=target_rate, demo_mode=demo)

    def _handle_stop(self) -> None:
        self._sensor_manager.stop()
        self._storage.close()
        self._active_sensors.clear()
        self._latest_samples.clear()

    def _handle_save(self) -> None:
        self._storage.flush(force=True)

    def _handle_share(self) -> None:
        self._storage.flush(force=True)
        if not self._csv_path:
            Logger.warning("main: No CSV generated yet")
            self.ui.notify_share_result(False)
            return
        shared = self._export.share_csv(self._csv_path)
        self.ui.notify_share_result(shared)

    def _handle_calibration_start(self, progress_callback) -> None:
        if not self._active_sensors:
            progress_callback("Inicia una captura antes de calibrar.", 0)
            return
        self._calibration_target = "accelerometer" if "accelerometer" in self._active_sensors else self._active_sensors[0]
        self._calibration_samples = []
        self._calibration_progress_callback = progress_callback
        self._calibration_elapsed = 0.0
        self._update_calibration_progress("Mantén el dispositivo quieto...", 0)
        from kivy.clock import Clock

        if self._calibration_event:
            self._calibration_event.cancel()
        self._calibration_event = Clock.schedule_interval(self._on_calibration_tick, 0.5)

    def _handle_calibration_cancel(self) -> None:
        if self._calibration_event:
            self._calibration_event.cancel()
            self._calibration_event = None
        self._calibration_target = None
        self._calibration_samples = []
        self._calibration_progress_callback = None
        self._calibration_elapsed = 0.0

    # Internal operations ---------------------------------------------------------------

    def _on_sample(self, sample: SensorSample) -> None:
        self._latest_samples[sample.sensor_name] = sample
        self._storage.store(sample)
        self.ui.update_acquisition_sample(sample)
        if self._calibration_target and sample.sensor_name == self._calibration_target:
            self._calibration_samples.append(sample)

    def _on_calibration_tick(self, dt) -> None:
        self._calibration_elapsed += dt
        progress = min(100.0, (self._calibration_elapsed / CALIBRATION_DURATION_SECONDS) * 100.0)
        self._update_calibration_progress("Midiendo...", progress)
        if self._calibration_elapsed >= CALIBRATION_DURATION_SECONDS:
            if self._calibration_event:
                self._calibration_event.cancel()
                self._calibration_event = None
            self._finalise_calibration()

    def _finalise_calibration(self) -> None:
        if not self._calibration_target or not self._calibration_samples:
            self._update_calibration_progress("No se obtuvieron lecturas suficientes.", 0)
            return
        vectors = [(sample.ax, sample.ay, sample.az) for sample in self._calibration_samples]
        record = self._calibration.calibrate(self._calibration_target, vectors)
        if record:
            message = (
                f"Calibración completada para {self._calibration_target}.\n"
                f"Offsets: {record.offset}\nDesviación σ: {record.deviation}"
            )
            self._update_calibration_progress("Calibración completada", 100)
            self.ui.calibration_completed(message)
        else:
            self._update_calibration_progress("Calibración incompleta, intenta de nuevo.", 0)
        self._calibration_target = None
        self._calibration_samples = []
        self._calibration_progress_callback = None
        self._calibration_elapsed = 0.0

    def _update_calibration_progress(self, message: str, progress: float) -> None:
        if self._calibration_progress_callback:
            self._calibration_progress_callback(message, progress)

    def _collect_metadata(self, target_rate: int, demo: bool) -> Dict[str, str]:
        version_data = self._retrieve_android_info()
        metadata = {
            "device_platform": platform,
            "android_release": version_data.get("release", "unknown"),
            "android_api": version_data.get("api", "unknown"),
            "app_version": APP_VERSION,
            "target_rate_hz": str(target_rate),
            "demo_mode": "yes" if demo else "no",
        }
        return metadata

    def _retrieve_android_info(self) -> Dict[str, str]:
        if platform != "android":
            return {}
        try:
            from jnius import autoclass  # type: ignore
        except Exception:
            return {}
        try:
            Build = autoclass("android.os.Build")  # pragma: no cover
            VERSION = autoclass("android.os.Build$VERSION")  # pragma: no cover
            return {
                "brand": Build.BRAND,
                "model": Build.MODEL,
                "release": VERSION.RELEASE,
                "api": str(VERSION.SDK_INT),
            }
        except Exception:
            Logger.warning("main: Unable to query android Build information")
            return {}


class AccelerologApp(App):
    """Application bootstrap class."""

    def build(self):
        user_dir = Path(self.user_data_dir)
        controller = AppController(root_dir=user_dir)
        self._configure_logging(controller)
        self.controller = controller
        return controller.ui.root

    def _configure_logging(self, controller: AppController) -> None:
        log_file = controller._logs_dir / "app.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        Logger.addHandler(file_handler)
        Logger.info("main: Logging initialised at %s", log_file)


if __name__ == "__main__":
    AccelerologApp().run()
