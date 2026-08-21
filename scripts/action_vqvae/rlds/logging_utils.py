"""Small logging shim used by the vendored VQ-VLA RLDS pipeline."""

from __future__ import annotations

import logging
from typing import Any


class _Logger:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)

    @staticmethod
    def _without_context(kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.pop("ctx_level", None)
        return kwargs

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(message, *args, **self._without_context(kwargs))

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(message, *args, **self._without_context(kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(message, *args, **self._without_context(kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(message, *args, **self._without_context(kwargs))


def initialize_overwatch(name: str) -> _Logger:
    """Return the subset of VQ-VLA's Overwatch interface used by this pipeline."""
    return _Logger(name)
