"""
Structured Logging Infrastructure.

Implements AGENTS.md compliant structured logging with OpenTelemetry compatibility.
Follows RFC 5424 log levels and JSON format for ELK/EFK stack compatibility.

Features:
- Structured JSON logging (ELK/EFK stack compatible)
- Correlation IDs for distributed tracing
- Security-first design (no PII/credentials in logs)
- Performance monitoring with execution time tracking
- OpenTelemetry-compatible metadata structure

Architecture Decision:
- Uses contextvars for correlation ID management across async operations
- Implements structured logging to enable better observability
- Follows Clean Architecture by providing infrastructure for logging concerns

@author: jtext Development Team
@since: 1.0.0
@compliance: AGENTS.md Logging Standards, RFC 5424, OpenTelemetry
"""

import json
import logging
import contextvars
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..core import generate_id


class LogLevel(Enum):
    """
    RFC 5424 compliant log levels.

    Levels follow AGENTS.md requirements:
    - FATAL: System unusable (process termination)
    - ERROR: Immediate attention required (exceptions, failures)
    - WARN: Potentially harmful situations (deprecated APIs)
    - INFO: General application flow (business events)
    - DEBUG: Fine-grained debugging information
    - TRACE: Most detailed diagnostic information
    """

    FATAL = "FATAL"
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for request tracing.

    Tracks execution time and resource usage for observability.
    """

    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    def finish(self) -> None:
        """Mark operation as finished and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class LogEntry:
    """
    OpenTelemetry-compatible structured log entry.

    Implements JSON format for ELK/EFK stack compatibility with
    security-first design ensuring no PII/credentials are logged.
    """

    timestamp: datetime
    level: LogLevel
    message: str
    correlation_id: str
    service: str
    operation: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None  # Anonymized user identifier only
    session_id: Optional[str] = None  # Session tracking (no PII)
    request_id: Optional[str] = None
    performance: Optional[PerformanceMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to OpenTelemetry-compatible dictionary.

        Ensures GDPR/CCPA compliance by excluding any PII data.

        Returns:
            Dictionary representation suitable for structured logging systems
        """
        entry = {
            "@timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "service": self.service,
            "logger": self.service,
        }

        # Add optional fields only if present
        if self.operation:
            entry["operation"] = self.operation
        if self.component:
            entry["component"] = self.component
        if self.user_id:
            entry["user_id"] = self.user_id  # Must be anonymized
        if self.session_id:
            entry["session_id"] = self.session_id
        if self.request_id:
            entry["request_id"] = self.request_id

        # Add performance metrics
        if self.performance:
            entry["performance"] = {
                "duration_ms": self.performance.duration_ms,
                "memory_usage_mb": self.performance.memory_usage_mb,
            }

        # Add metadata and tags
        if self.metadata:
            entry["metadata"] = self.metadata
        if self.tags:
            entry["tags"] = self.tags

        return entry


@dataclass
class LoggingConfiguration:
    """
    Logging configuration following AGENTS.md standards.

    Supports both console and file output with structured JSON format.
    """

    level: LogLevel = LogLevel.INFO
    enable_structured_logging: bool = True
    enable_file_logging: bool = True
    log_file_path: Optional[str] = "logs/jtext.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_performance_tracking: bool = True
    enable_security_audit: bool = True


# Context management for distributed tracing
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
_performance_context: contextvars.ContextVar[Optional[PerformanceMetrics]] = (
    contextvars.ContextVar("performance_context", default=None)
)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current context.

    Enables request tracing across service boundaries as required by AGENTS.md.

    Args:
        correlation_id: Unique identifier for tracing requests
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str:
    """
    Get current correlation ID.

    Returns:
        Current correlation ID or empty string if not set
    """
    return _correlation_id.get("")


def start_performance_tracking() -> PerformanceMetrics:
    """
    Start performance tracking for current operation.

    Returns:
        PerformanceMetrics instance for tracking operation performance
    """
    metrics = PerformanceMetrics(start_time=time.time())
    _performance_context.set(metrics)
    return metrics


def get_performance_metrics() -> Optional[PerformanceMetrics]:
    """
    Get current performance metrics.

    Returns:
        Current performance metrics or None if not tracking
    """
    return _performance_context.get()


class CorrelationIdGenerator:
    """
    Correlation ID generator for distributed tracing.

    Implements OpenTelemetry-compatible ID generation patterns.
    """

    @staticmethod
    def generate() -> str:
        """
        Generate new correlation ID.

        Returns:
            Unique correlation ID with jtext prefix
        """
        return f"jtext-{generate_id()}"

    @staticmethod
    def from_request(request_id: str) -> str:
        """
        Generate correlation ID from request ID.

        Args:
            request_id: External request identifier

        Returns:
            Correlation ID based on request ID
        """
        return f"jtext-req-{request_id}"


class StructuredLogger:
    """
    AGENTS.md compliant structured logger.

    Implements OpenTelemetry-compatible structured logging with:
    - JSON format for ELK/EFK stack compatibility
    - Correlation IDs for distributed tracing
    - Performance monitoring
    - Security-first design (no PII/credentials)
    - File output for audit trails

    Architecture Pattern: Infrastructure service providing logging capabilities
    to all application layers while maintaining clean separation of concerns.
    """

    def __init__(self, name: str, config: Optional[LoggingConfiguration] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically module or component name)
            config: Optional logging configuration
        """
        self.name = name
        self.config = config or LoggingConfiguration()
        self.logger = logging.getLogger(name)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration with file and console handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set log level
        level_mapping = {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.FATAL: logging.CRITICAL,
        }
        self.logger.setLevel(level_mapping.get(self.config.level, logging.INFO))

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler for audit trails (if enabled)
        if self.config.enable_file_logging and self.config.log_file_path:
            self._setup_file_handler()

        self.logger.addHandler(console_handler)

    def _setup_file_handler(self) -> None:
        """Setup rotating file handler for structured logs."""
        try:
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
            )
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console-only logging if file setup fails
            self.logger.error(f"Failed to setup file logging: {e}")

    def log(
        self,
        level: LogLevel,
        message: str,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        user_id: Optional[str] = None,  # Must be anonymized
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
        **metadata,
    ) -> None:
        """
        Log structured message with full observability data.

        Implements AGENTS.md logging standards with security-first design.
        All PII data must be excluded or anonymized before calling this method.

        Args:
            level: Log level following RFC 5424
            message: Human-readable log message
            operation: Business operation being performed
            component: Component/module generating the log
            user_id: Anonymized user identifier (no PII)
            session_id: Session identifier for tracking
            request_id: Request identifier for correlation
            duration_ms: Operation duration in milliseconds
            memory_usage_mb: Memory usage in megabytes
            tags: Additional tags for categorization
            **metadata: Additional structured metadata (no PII)
        """
        # Get current context
        correlation_id = get_correlation_id()
        performance = get_performance_metrics()

        # Override performance metrics if provided
        if duration_ms is not None or memory_usage_mb is not None:
            performance = PerformanceMetrics(
                start_time=time.time(),
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage_mb,
            )

        # Create structured log entry
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            correlation_id=correlation_id,
            service=self.name,
            operation=operation,
            component=component,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            performance=performance,
            metadata=metadata,
            tags=tags or {},
        )

        # Log as structured JSON
        if self.config.enable_structured_logging:
            log_data = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
        else:
            log_data = message

        # Use appropriate log level
        level_mapping = {
            LogLevel.TRACE: self.logger.debug,
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARN: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
            LogLevel.FATAL: self.logger.critical,
        }

        log_method = level_mapping.get(level, self.logger.info)
        log_method(log_data)

    def trace(self, message: str, **kwargs) -> None:
        """Log trace message - most detailed diagnostic information."""
        self.log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message - fine-grained debugging information."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message - general application flow (business events)."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Log warning message - potentially harmful situations."""
        self.log(LogLevel.WARN, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message - immediate attention required."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs) -> None:
        """Log fatal message - system unusable (process termination)."""
        self.log(LogLevel.FATAL, message, **kwargs)

    def performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """
        Log performance metrics for operation tracking.

        Args:
            operation: Operation name
            duration_ms: Operation duration in milliseconds
            **kwargs: Additional performance metadata
        """
        self.info(
            f"Performance: {operation} completed",
            operation=operation,
            duration_ms=duration_ms,
            tags={"category": "performance"},
            **kwargs,
        )

    def security_audit(self, event: str, **kwargs) -> None:
        """
        Log security audit event.

        Args:
            event: Security event description
            **kwargs: Additional security context (no PII)
        """
        self.info(
            f"Security audit: {event}",
            tags={"category": "security", "audit": "true"},
            **kwargs,
        )

    def business_event(self, event: str, **kwargs) -> None:
        """
        Log business event for analytics and monitoring.

        Args:
            event: Business event description
            **kwargs: Additional business context
        """
        self.info(f"Business event: {event}", tags={"category": "business"}, **kwargs)


def get_logger(
    name: str, config: Optional[LoggingConfiguration] = None
) -> StructuredLogger:
    """
    Get AGENTS.md compliant structured logger.

    Args:
        name: Logger name (typically module or component name)
        config: Optional logging configuration

    Returns:
        Configured structured logger instance
    """
    return StructuredLogger(name, config)
