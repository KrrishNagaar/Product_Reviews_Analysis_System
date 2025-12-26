# utils/logger.py
# Centralized logging utility for the entire project

import logging
import os
from logging.handlers import RotatingFileHandler


# ===================== LOG DIRECTORY SETUP =====================
# Create logs directory if it does not already exist

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ===================== LOGGER FACTORY =====================
# Returns a configured logger instance for a given module

def get_logger(name, level=logging.INFO):
    """
    Returns a logger that writes to logs/<name>.log

    Parameters:
    name (str): Name of the logger (usually module name)
    level (int): Logging level (default = INFO)

    The logger outputs logs to:
    - A rotating file (logs/<name>.log)
    - The console (stdout)
    """

    # Get (or create) a logger with the given name
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Set logging level
    logger.setLevel(level)

    # Define log file path
    log_file = os.path.join(LOG_DIR, f"{name}.log")

    # Define log message format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # -------------------- FILE HANDLER --------------------
    # Rotating file handler prevents log files from growing too large
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,  # 2MB per log file
        backupCount=3,            # Keep last 3 backups
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # -------------------- CONSOLE HANDLER --------------------
    # Outputs logs to terminal for real-time debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger