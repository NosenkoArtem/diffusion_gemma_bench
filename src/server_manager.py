"""Sequential server lifecycle helpers.

The benchmark loads one model at a time. This wrapper gives long phases a small
context-manager surface while the concrete process logic remains in adapters.
"""

from __future__ import annotations

from typing import Any


class ManagedServer:
    """Context manager for adapters with `start` and `unload` methods."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __enter__(self) -> Any:
        self.adapter.start()
        return self.adapter

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.adapter.unload()
