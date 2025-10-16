"""Synthetic data stream used for testing or demo mode."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

from utils.time import TimeProvider


Vector3 = Tuple[float, float, float]


@dataclass
class MockSensorConfig:
    """Configuration for a synthetic sensor."""

    amplitude: float
    frequency: float
    noise: float


class MockSensorStream:
    """Generate deterministic but noisy sensor readings for demos/tests."""

    def __init__(self, sensors: Iterable[str], time_provider: TimeProvider | None = None) -> None:
        self._time = time_provider or TimeProvider()
        self._configs: Dict[str, MockSensorConfig] = {
            name: MockSensorConfig(amplitude=9.81 if "accelerometer" in name else 1.0, frequency=0.5, noise=0.05)
            for name in sensors
        }
        self._phase: Dict[str, float] = {name: 0.0 for name in sensors}

    def sample(self, sensor_name: str) -> Vector3:
        """Return a synthetic triple (x, y, z)."""
        config = self._configs[sensor_name]
        self._phase[sensor_name] += config.frequency
        base = math.sin(self._phase[sensor_name])
        noise = lambda: random.uniform(-config.noise, config.noise)
        return (
            config.amplitude * base + noise(),
            config.amplitude * math.cos(self._phase[sensor_name]) + noise(),
            config.amplitude * math.sin(self._phase[sensor_name] / 2) + noise(),
        )

    def iter_samples(self, sensor_name: str) -> Iterator[Vector3]:
        """Infinite generator for sensor data."""
        while True:
            yield self.sample(sensor_name)


if __name__ == "__main__":
    stream = MockSensorStream(["accelerometer", "gyroscope"])
    for idx, values in zip(range(5), stream.iter_samples("accelerometer")):
        print(f"sample {idx}: ax={values[0]:.3f} ay={values[1]:.3f} az={values[2]:.3f}")
