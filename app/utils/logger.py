"""Backward-compatible logger helper."""

from __future__ import annotations

import logging

from app.utils.logging import configure_logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for older imports."""

    configure_logging(level)
    return logging.getLogger(name)
