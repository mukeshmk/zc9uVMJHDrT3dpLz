import sys
import logging
from typing import Optional

from convai.utils.config import settings


def _get_log_level(level: Optional[str]):
    if level is None:
        return level

    if level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'debug':
        level = logging.DEBUG
    else:
        level = logging.INFO
    return level


def setup_logs(logger):
    """
    Configure and return a logger with console and optional file handlers.
    """
    cli_level = settings.LOG_LEVEL
    log_file = settings.LOG_FILE
    file_level = settings.LOG_FILE_LEVEL

    cli_level = _get_log_level(cli_level)
    file_level = _get_log_level(file_level)

    formatter = logging.Formatter("%(asctime)s : [%(levelname)s] %(message)s")

    # Set the log level on the application logger
    logger.setLevel(cli_level)
    logger.propagate = False
    
    # Console / Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(cli_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# Create and configure the application-level logger
logger = logging.getLogger("convai")
if not logger.handlers:
    setup_logs(logger)
