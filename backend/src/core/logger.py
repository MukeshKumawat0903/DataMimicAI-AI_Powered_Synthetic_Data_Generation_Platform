import logging
import os
from logging.handlers import RotatingFileHandler

__all__ = ["get_logger", "logger"]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str = "backend") -> logging.Logger:
    """Return a configured logger. Safe to call multiple times (won't duplicate handlers).

    Configurable via environment variables:
    - LOG_LEVEL: default INFO
    - BACKEND_LOG_FILE: optional path to enable rotating file logging
    """
    logger = logging.getLogger(name)

    # If handlers already configured, assume initialization was done elsewhere.
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Optional rotating file handler
    log_file = os.getenv("BACKEND_LOG_FILE")
    if log_file:
        try:
            fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # Avoid raising if file handler can't be created; keep console logging.
            logger.exception("Failed to create file log handler for %s", log_file)

    return logger


# Module-level logger for convenient imports: `from src.core.logger import logger`
logger = get_logger("data_mimic_backend")

# Small usage note for maintainers:
# - Use get_logger(__name__) in modules that want module-specific loggers.
# - Configure LOG_LEVEL and BACKEND_LOG_FILE via environment as needed.
