"""Logging setup: concise console output plus a full log file per run."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "wdbc_pipeline"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def setup_logging(log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(fh)
    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    log = logger or get_logger()
    t0 = time.perf_counter()
    log.info("▶ %s", label)
    try:
        yield
    except BaseException:
        log.error("✖ %s failed after %.1f s", label, time.perf_counter() - t0)
        raise
    log.info("✔ %s (%.1f s)", label, time.perf_counter() - t0)
