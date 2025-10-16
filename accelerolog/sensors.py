"""Sensor management built on top of Plyer."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from kivy.clock import Clock, ClockEvent
from kivy.logger import Logger
from kivy.utils import platform

from calibration import CalibrationController
from tests.mock_stream import MockSensorStream
from utils.time import TimeProvider


try:
    from plyer import accelerometer, gravity, gyroscope, magnetometer, orientation, proximity
except ImportError:  # pragma: no cover - Plyer is provided at runtime
    accelerometer = gravity = gyroscope = magnetometer = orientation = proximity = None  # type: ignore


Vector3 = Tuple[float, float, float]


@dataclass
class SensorSample:
    """Unified representation of a sensor sample."""

    timestamp_ms: int
    timestamp_iso: str
    sensor_name: str
    ax: float
    ay: float
    az: float
    extra1: float | None = None
    extra2: float | None = None


@dataclass
class SensorSpec:
    """Metadata describing how to interact with a Plyer sensor facade."""

    name: str
    display_name: str
    attr: str
    fallback: Vector3 = (0.0, 0.0, 0.0)


SENSOR_SPECS: Sequence[SensorSpec] = (
    SensorSpec("accelerometer", "Acelerómetro", "acceleration"),
    SensorSpec("gyroscope", "Giroscopio", "orientation"),
    SensorSpec("magnetometer", "Magnetómetro", "magnetic_field"),
    SensorSpec("gravity", "Gravedad", "gravity"),
    SensorSpec("linearacceleration", "Aceleración lineal", "linear_acceleration"),
    SensorSpec("orientation", "Orientación", "orientation"),
    SensorSpec("proximity", "Proximidad", "proximity"),
)


PLYER_BACKEND = {
    "accelerometer": accelerometer,
    "gyroscope": gyroscope,
    "magnetometer": magnetometer,
    "gravity": gravity,
    "linearacceleration": gravity,  # some backends expose linear acceleration in gravity module
    "orientation": orientation,
    "proximity": proximity,
}


class SensorRuntime:
    """Hold runtime state for an active sensor."""

    def __init__(self, spec: SensorSpec, backend, calibration: CalibrationController, time_provider: TimeProvider) -> None:
        self.spec = spec
        self.backend = backend
        self.calibration = calibration
        self.time = time_provider

    def enable(self) -> bool:
        if self.backend is None:
            return False
        try:
            if hasattr(self.backend, "enable"):
                self.backend.enable()
            Logger.info("sensors: %s enabled", self.spec.name)
            return True
        except Exception as exc:  # pragma: no cover - depends on platform backend
            Logger.warning("sensors: Failed to enable %s (%s)", self.spec.name, exc)
            return False

    def disable(self) -> None:
        if self.backend is None:
            return
        with contextlib.suppress(Exception):
            if hasattr(self.backend, "disable"):
                self.backend.disable()
        Logger.info("sensors: %s disabled", self.spec.name)

    def read(self, applied_offset: bool = True) -> Optional[SensorSample]:
        raw = self._read_raw()
        if raw is None:
            return None
        ax, ay, az, extra1, extra2 = raw
        if applied_offset:
            ax, ay, az = self.calibration.apply(self.spec.name, (ax, ay, az))
        timestamp_ms, timestamp_iso = self.time.now_pair()
        return SensorSample(
            timestamp_ms=timestamp_ms,
            timestamp_iso=timestamp_iso,
            sensor_name=self.spec.name,
            ax=ax,
            ay=ay,
            az=az,
            extra1=extra1,
            extra2=extra2,
        )

    def _read_raw(self) -> Optional[Tuple[float, float, float, Optional[float], Optional[float]]]:
        if self.backend is None:
            return None
        value = getattr(self.backend, self.spec.attr, None)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            series: List[float] = [float(value)]
        elif isinstance(value, dict):
            series = [float(v) for v in value.values() if v is not None]
        else:
            try:
                series = [float(v) if v is not None else 0.0 for v in value]
            except TypeError:
                series = [float(value)]
        while len(series) < 3:
            series.append(0.0)
        extra1 = series[3] if len(series) > 3 else None
        extra2 = series[4] if len(series) > 4 else None
        return series[0], series[1], series[2], extra1, extra2


class SensorManager:
    """Coordinate acquisition for multiple sensors."""

    SUPPORTED_RATES = [10, 25, 50, 100, 200]

    def __init__(self, calibration: CalibrationController, time_provider: TimeProvider) -> None:
        self._calibration = calibration
        self._time = time_provider
        self._runtimes: Dict[str, SensorRuntime] = {}
        self._active: Dict[str, SensorRuntime] = {}
        self._callbacks: List[Callable[[SensorSample], None]] = []
        self._poll_event: Optional[ClockEvent] = None
        self._target_rate = 25
        self._demo_stream: Optional[MockSensorStream] = None
        self._load_runtimes()

    def _load_runtimes(self) -> None:
        for spec in SENSOR_SPECS:
            backend = PLYER_BACKEND.get(spec.name)
            runtime = SensorRuntime(spec=spec, backend=backend, calibration=self._calibration, time_provider=self._time)
            self._runtimes[spec.name] = runtime

    def list_capabilities(self) -> Dict[str, Dict[str, str]]:
        """Return metadata about all sensors and their availability."""
        capabilities: Dict[str, Dict[str, str]] = {}
        for spec in SENSOR_SPECS:
            backend = self._runtimes[spec.name].backend
            capabilities[spec.name] = {
                "name": spec.display_name,
                "available": "yes" if backend is not None else "no",
            }
        return capabilities

    def set_callbacks(self, callbacks: Iterable[Callable[[SensorSample], None]]) -> None:
        """Replace callbacks invoked when samples are available."""
        self._callbacks = list(callbacks)

    def start(self, sensors: Sequence[str], target_rate: int, demo_mode: bool = False) -> None:
        """Start polling the provided sensors at the requested rate."""
        self.stop()
        supported_rate = self.select_supported_rate(target_rate)
        self._target_rate = supported_rate
        for name in sensors:
            runtime = self._runtimes.get(name)
            if not runtime:
                Logger.warning("sensors: Unknown sensor %s", name)
                continue
            if runtime.backend is None and not demo_mode:
                Logger.warning("sensors: Sensor %s not available on this device", name)
                continue
            if not demo_mode:
                runtime.enable()
            self._active[name] = runtime
        if not self._active and not demo_mode:
            Logger.error("sensors: No sensors active after start request")
            return
        if demo_mode:
            self._demo_stream = MockSensorStream(sensors=self._active.keys(), time_provider=self._time)
        interval = 1.0 / supported_rate
        self._poll_event = Clock.schedule_interval(self._poll_sensors, interval)
        Logger.info("sensors: Acquisition started at %s Hz", supported_rate)

    def stop(self) -> None:
        """Stop all sensors and cancel the poller."""
        if self._poll_event is not None:
            self._poll_event.cancel()
            self._poll_event = None
        for runtime in self._active.values():
            runtime.disable()
        self._active.clear()
        self._demo_stream = None
        Logger.info("sensors: Acquisition stopped")

    def select_supported_rate(self, requested: int) -> int:
        """Return the closest supported rate not exceeding hardware capabilities."""
        sorted_rates = sorted(self.SUPPORTED_RATES)
        fallback = sorted_rates[0]
        for rate in sorted_rates:
            if requested <= rate:
                return rate
            fallback = rate
        return fallback

    def get_supported_rates(self) -> Sequence[int]:
        """Return supported rates for UI selectors."""
        return self.SUPPORTED_RATES

    def _poll_sensors(self, *_args) -> None:
        for name, runtime in self._active.items():
            if self._demo_stream:
                sample = self._build_demo_sample(name)
            else:
                sample = runtime.read()
            if not sample:
                continue
            for callback in self._callbacks:
                callback(sample)

    def _build_demo_sample(self, name: str) -> SensorSample:
        assert self._demo_stream is not None
        vector = self._demo_stream.sample(name)
        timestamp_ms, timestamp_iso = self._time.now_pair()
        return SensorSample(
            timestamp_ms=timestamp_ms,
            timestamp_iso=timestamp_iso,
            sensor_name=name,
            ax=vector[0],
            ay=vector[1],
            az=vector[2],
        )

