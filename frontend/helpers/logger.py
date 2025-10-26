import logging
import os

__all__ = ["get_logger", "logger"]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str = "frontend") -> logging.Logger:
    """Return a configured logger for frontend code.

    Keeps a single initialization (no duplicate handlers). Configure via LOG_LEVEL env var.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# Module-level logger for easy imports in frontend modules:
logger = get_logger("data_mimic_frontend")

# Usage:
# from frontend.helpers.logger import logger
# logger.info("message")
