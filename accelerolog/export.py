"""Sharing helpers built on top of Plyer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kivy.logger import Logger
from kivy.utils import platform

try:
    from plyer import share
except ImportError:  # pragma: no cover - Plyer is expected in runtime
    share = None  # type: ignore


class ExportManager:
    """Expose high-level helpers to share generated CSV files."""

    def __init__(self) -> None:
        self._last_shared: Optional[Path] = None

    def share_csv(self, file_path: Path, title: str = "Compartir CSV de sensores") -> bool:
        """Trigger Android share intent for the provided CSV file."""
        if not file_path.exists():
            Logger.error("export: CSV file not found at %s", file_path)
            return False
        if platform != "android":
            Logger.info("export: Sharing is simulated on non-Android platforms (%s)", platform)
            self._last_shared = file_path
            return True
        if share is None:
            Logger.error("export: Plyer share backend is not available")
            return False
        try:
            share.share(
                title=title,
                text="Datos capturados por Accelerolog",
                filepath=str(file_path),
            )
            self._last_shared = file_path
            Logger.info("export: Share intent triggered for %s", file_path)
            return True
        except Exception as exc:  # pragma: no cover - depends on platform state
            Logger.exception("export: Failed to share CSV: %s", exc)
            return False

    @property
    def last_shared(self) -> Optional[Path]:
        """Return the last shared file path."""
        return self._last_shared

