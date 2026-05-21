"""Structured logging for NeuroGuard.

Configures structlog for machine-parseable, human-readable output.
JSON format in production/CI; coloured console output in development.

Usage:
    from neuroguard.logging_setup import get_logger

    logger = get_logger(__name__)
    logger.info("experiment started", session_id="abc", pressure="deadline")
"""

import logging
import sys

import structlog

_configured: bool = False


def setup_logging(level: str = "INFO", force_json: bool = False) -> None:
    """Configure structlog and stdlib logging.

    Safe to call multiple times; only the first call takes effect.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        force_json: If True, always use JSON renderer (useful in CI).
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Stdlib logging (catches third-party lib logs)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Choose renderer based on terminal vs pipe/CI
    use_json = force_json or not sys.stderr.isatty()
    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger.

    Ensures logging is configured on first call.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger instance.
    """
    setup_logging()
    return structlog.get_logger(name)
