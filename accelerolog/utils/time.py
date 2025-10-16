"""Time utilities for timestamp management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple


@dataclass
class TimeProvider:
    """Utility class that centralises time related helpers."""

    def now(self) -> datetime:
        """Return the current timezone-aware datetime."""
        return datetime.now(timezone.utc).astimezone()

    def now_pair(self) -> Tuple[int, str]:
        """Return both UNIX timestamp in milliseconds and ISO string."""
        current = self.now()
        return int(current.timestamp() * 1000), current.isoformat()

    def to_iso(self, timestamp_ms: int) -> str:
        """Convert a UNIX timestamp in milliseconds to ISO-8601."""
        seconds, millis = divmod(timestamp_ms, 1000)
        base = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone()
        return base.replace(microsecond=millis * 1000).isoformat()

