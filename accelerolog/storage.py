"""Data buffering and CSV persistence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from kivy.logger import Logger

from utils.time import TimeProvider

try:
    from sensors import SensorSample  # type: ignore circular import guard
except ImportError:  # pragma: no cover - runtime only
    @dataclass
    class SensorSample:  # type: ignore[used-before-assignment]
        timestamp_ms: int
        timestamp_iso: str
        sensor_name: str
        ax: float
        ay: float
        az: float
        extra1: float | None = None
        extra2: float | None = None


CSV_HEADER = ["timestamp_unix_ms", "timestamp_iso", "sensor", "ax", "ay", "az", "extra1", "extra2"]


class CSVStorageManager:
    """Persist sensor samples to CSV with buffered writes."""

    def __init__(self, output_dir: Path, time_provider: TimeProvider, flush_threshold: int = 128) -> None:
        self._output_dir = output_dir
        self._time = time_provider
        self._flush_threshold = flush_threshold
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._active_file: Optional[Path] = None
        self._buffer: List[SensorSample] = []
        self._pending_flush: List[SensorSample] = []

    def start_session(self, metadata: Dict[str, str], filename: Optional[str] = None) -> Path:
        """Initialise a CSV file and write metadata/header rows."""
        timestamp_ms, timestamp_iso = self._time.now_pair()
        if filename:
            file_name = filename
        else:
            file_name = f"accelerolog_{timestamp_ms}.csv"
        self._active_file = self._output_dir / file_name
        Logger.info("storage: Starting new session file=%s", self._active_file)
        with self._active_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            metadata_row = ["metadata"]
            metadata_row.extend(f"{key}={value}" for key, value in metadata.items())
            metadata_row.append(f"created_at_ms={timestamp_ms}")
            metadata_row.append(f"created_at_iso={timestamp_iso}")
            writer.writerow(metadata_row)
            writer.writerow(CSV_HEADER)
        self._buffer.clear()
        self._pending_flush.clear()
        return self._active_file

    def store(self, sample: SensorSample) -> None:
        """Append a sample to the active buffer and flush if required."""
        if not self._active_file:
            Logger.warning("storage: Ignoring sample while no session active")
            return
        self._buffer.append(sample)
        if len(self._buffer) >= self._flush_threshold:
            self._swap_and_flush()

    def flush(self, force: bool = False) -> None:
        """Flush buffers to disk."""
        if force and self._buffer:
            self._swap_and_flush()
        if not self._pending_flush:
            return
        if not self._active_file:
            Logger.warning("storage: Pending flush but no file attached")
            return
        with self._active_file.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            row_count = 0
            for sample in self._pending_flush:
                row = [
                    sample.timestamp_ms,
                    sample.timestamp_iso,
                    sample.sensor_name,
                    sample.ax,
                    sample.ay,
                    sample.az,
                    sample.extra1 if sample.extra1 is not None else "",
                    sample.extra2 if sample.extra2 is not None else "",
                ]
                writer.writerow(row)
                row_count += 1
        Logger.info("storage: Flushed %d rows to %s", len(self._pending_flush), self._active_file)
        self._pending_flush.clear()

    def close(self) -> None:
        """Ensure all data is written to disk."""
        self.flush(force=True)
        self._pending_flush.clear()
        self._buffer.clear()

    def _swap_and_flush(self) -> None:
        """Swap the live buffer into the pending flush queue and flush."""
        if not self._buffer:
            return
        self._pending_flush.extend(self._buffer)
        self._buffer = []
        self.flush()

