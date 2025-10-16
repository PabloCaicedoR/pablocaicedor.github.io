"""Calibration helpers for Accelerolog."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

from utils.time import TimeProvider


Vector3 = Tuple[float, float, float]


@dataclass
class CalibrationRecord:
    """Represents the calibration data for a single sensor."""

    offset: Vector3
    deviation: Vector3
    timestamp_ms: int
    timestamp_iso: str


class CalibrationController:
    """Manage calibration results and persistence."""

    def __init__(self, storage_dir: Path, time_provider: TimeProvider) -> None:
        self._storage_dir = storage_dir
        self._time = time_provider
        self._file_path = self._storage_dir / "calibration.json"
        self._records: Dict[str, CalibrationRecord] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._file_path.exists():
            return
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for sensor, raw in data.items():
            self._records[sensor] = CalibrationRecord(
                offset=tuple(raw["offset"]),  # type: ignore[arg-type]
                deviation=tuple(raw["deviation"]),  # type: ignore[arg-type]
                timestamp_ms=raw["timestamp_ms"],
                timestamp_iso=raw["timestamp_iso"],
            )

    def calibrate(self, sensor_name: str, samples: Iterable[Vector3]) -> Optional[CalibrationRecord]:
        """Compute offsets from a sequence of samples and persist to storage."""
        sample_list: List[Vector3] = list(samples)
        if len(sample_list) < 5:
            return None
        xs, ys, zs = zip(*sample_list)
        offset = (mean(xs), mean(ys), mean(zs))
        deviation = (
            pstdev(xs) if len(xs) > 1 else 0.0,
            pstdev(ys) if len(ys) > 1 else 0.0,
            pstdev(zs) if len(zs) > 1 else 0.0,
        )
        timestamp_ms, timestamp_iso = self._time.now_pair()
        record = CalibrationRecord(offset=offset, deviation=deviation, timestamp_ms=timestamp_ms, timestamp_iso=timestamp_iso)
        self._records[sensor_name] = record
        self._write_to_disk()
        return record

    def apply(self, sensor_name: str, values: Vector3) -> Vector3:
        """Apply stored offsets to a raw reading."""
        record = self._records.get(sensor_name)
        if not record:
            return values
        return tuple(value - offset for value, offset in zip(values, record.offset))  # type: ignore[return-value]

    def get_record(self, sensor_name: str) -> Optional[CalibrationRecord]:
        """Return the calibration record for a sensor if it exists."""
        return self._records.get(sensor_name)

    def remove(self, sensor_name: str) -> None:
        """Delete calibration data for a sensor."""
        if sensor_name in self._records:
            del self._records[sensor_name]
            self._write_to_disk()

    def _write_to_disk(self) -> None:
        payload = {sensor: asdict(record) for sensor, record in self._records.items()}
        self._file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

