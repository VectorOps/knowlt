import logging
import sys
from typing import Any

import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
    processors=[
        # enrich
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
)

# Ensure the stdlib logger named "knowlt" inherits the root logger configuration.
_std_logger = logging.getLogger("knowlt")
_std_logger.setLevel(logging.NOTSET)
_std_logger.propagate = True

logger: structlog.BoundLogger = structlog.get_logger("knowlt")
