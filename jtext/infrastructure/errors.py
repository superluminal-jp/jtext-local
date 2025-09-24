"""
Error Handling Infrastructure.

Comprehensive error handling with different error types and circuit breaker pattern.
"""

import threading
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import logging

from ..core import Result, Ok, Err


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories."""

    VALIDATION = "VALIDATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    DOMAIN = "DOMAIN"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    SYSTEM = "SYSTEM"


@dataclass
class ErrorContext:
    """Error context information."""

    correlation_id: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingError(Exception):
    """Base processing error."""

    def __init__(
        self,
        message: str,
        error_code: str = "PROCESSING_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.DOMAIN,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.__dict__ if self.context else None,
            "cause": str(self.cause) if self.cause else None,
        }


class ValidationError(ProcessingError):
    """Validation error."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            **kwargs,
        )
        self.field = field


class InfrastructureError(ProcessingError):
    """Infrastructure error."""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="INFRASTRUCTURE_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INFRASTRUCTURE,
            **kwargs,
        )
        self.service = service


class DomainError(ProcessingError):
    """Domain error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DOMAIN_ERROR",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DOMAIN,
            **kwargs,
        )


class ExternalServiceError(ProcessingError):
    """External service error."""

    def __init__(
        self, message: str, service: str, status_code: Optional[int] = None, **kwargs
    ):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs,
        )
        self.service = service
        self.status_code = status_code


class SystemError(ProcessingError):
    """System error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            **kwargs,
        )


class ErrorHandler:
    """Error handler for processing errors."""

    def __init__(self):
        self.logger = logging.getLogger("jtext.error_handler")

    def handle_error(self, error: ProcessingError) -> Result[None, str]:
        """Handle processing error."""
        try:
            # Log error
            self.logger.error(f"Processing error: {error.to_dict()}")

            # Handle based on severity
            if error.severity == ErrorSeverity.CRITICAL:
                self._handle_critical_error(error)
            elif error.severity == ErrorSeverity.HIGH:
                self._handle_high_error(error)
            else:
                self._handle_standard_error(error)

            return Ok(None)

        except Exception as e:
            return Err(f"Failed to handle error: {str(e)}")

    def _handle_critical_error(self, error: ProcessingError) -> None:
        """Handle critical error."""
        # Send alert, stop processing, etc.
        self.logger.critical(f"CRITICAL ERROR: {error.message}")

    def _handle_high_error(self, error: ProcessingError) -> None:
        """Handle high severity error."""
        # Log, potentially retry, etc.
        self.logger.error(f"HIGH SEVERITY ERROR: {error.message}")

    def _handle_standard_error(self, error: ProcessingError) -> None:
        """Handle standard error."""
        # Log and continue
        self.logger.warning(f"ERROR: {error.message}")


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Result[Any, str]:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    return Err("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return Ok(result)

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

                return Err(f"Circuit breaker failure: {str(e)}")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True

        return (datetime.now() - self.last_failure_time).total_seconds() > self.timeout
