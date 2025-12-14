"""Structured JSON logging for production observability.

This module provides a JSON formatter and configuration utilities
for structured logging suitable for log aggregation systems like
Datadog, Splunk, or ELK stack.

Example:
    from catfacialid.logging import setup_logging, get_logger

    setup_logging(level="INFO", json_output=True)
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"batch_size": 32, "model": "v1.0"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as JSON objects with consistent schema,
    suitable for ingestion by log aggregation systems.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_path: bool = True,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize JSON formatter.

        Args:
            include_timestamp: Include ISO8601 timestamp.
            include_level: Include log level.
            include_logger: Include logger name.
            include_path: Include file path and line number.
            extra_fields: Static fields to include in every log entry.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_path = include_path
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Source location
        if self.include_path:
            log_data["path"] = f"{record.pathname}:{record.lineno}"
            log_data["function"] = record.funcName

        # Exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }:
                log_data[key] = value

        # Static extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str)


class StandardFormatter(logging.Formatter):
    """Standard text formatter with colors for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string.
        """
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{timestamp} | {color}{record.levelname:8}{reset} | "
            f"{record.name} | {record.getMessage()}"
        )


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    extra_fields: Optional[dict[str, Any]] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: Use JSON formatter instead of text.
        extra_fields: Static fields to include in JSON logs.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_output:
        formatter = JSONFormatter(extra_fields=extra_fields)
    else:
        formatter = StandardFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


__all__ = ["JSONFormatter", "StandardFormatter", "setup_logging", "get_logger"]
