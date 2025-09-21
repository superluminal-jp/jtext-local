"""
Structured logging configuration for jtext system.

This module provides centralized logging configuration using loguru
with structured JSON output and appropriate log levels.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Setup structured logging for the jtext system.

    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional log file path
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    log_level = "DEBUG" if verbose else "INFO"

    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=verbose,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON format for file logging
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_path,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=verbose,
        )

    logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(component=name)
