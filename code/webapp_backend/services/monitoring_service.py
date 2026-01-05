"""
Logging and monitoring service for enhanced error tracking and observability.
Implements structured logging with file rotation and performance metrics.
"""

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


class MonitoringService:
    """Handles application logging and monitoring."""

    def __init__(self, app_name: str = "lol-draft-predictor"):
        self.app_name = app_name
        self.logger = logging.getLogger(app_name)
        self.metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_predictions": 0,
            "batch_predictions": 0,
        }
        self.start_time = time.time()

    def setup(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Setup logging configuration."""
        # Set level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB per file
            )
            file_handler.setLevel(level)
            json_formatter = jsonlogger.JsonFormatter(
                "%(timestamp)s %(level)s %(name)s %(message)s"
            )
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

        self.logger.info(f"{self.app_name} logger initialized")

    def log_request(
        self,
        endpoint: str,
        method: str,
        client_ip: str,
        status_code: int,
        duration_ms: float,
        error: Optional[str] = None,
    ):
        """Log API request with metrics."""
        self.metrics["requests_total"] += 1
        if status_code >= 400:
            self.metrics["requests_errors"] += 1

        self.logger.info(
            f"Request: {method} {endpoint} | Client: {client_ip} | "
            f"Status: {status_code} | Duration: {duration_ms:.2f}ms",
            extra={
                "endpoint": endpoint,
                "method": method,
                "client_ip": client_ip,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "error": error,
            },
        )

    def log_prediction(
        self,
        is_batch: bool,
        num_samples: int,
        duration_ms: float,
        confidence: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Log prediction event."""
        if is_batch:
            self.metrics["batch_predictions"] += 1
        else:
            self.metrics["model_predictions"] += 1

        level = "ERROR" if error else "INFO"
        self.logger.log(
            getattr(logging, level),
            f"Prediction: {'Batch' if is_batch else 'Single'} | "
            f"Samples: {num_samples} | Duration: {duration_ms:.2f}ms "
            f"{'| Confidence: ' + f'{confidence:.2f}' if confidence else ''}",
            extra={
                "prediction_type": "batch" if is_batch else "single",
                "num_samples": num_samples,
                "duration_ms": duration_ms,
                "confidence": confidence,
                "error": error,
            },
        )

    def log_cache(self, event: str, key: str, ttl: Optional[int] = None):
        """Log cache operations."""
        if event == "hit":
            self.metrics["cache_hits"] += 1
        elif event == "miss":
            self.metrics["cache_misses"] += 1

        self.logger.debug(
            f"Cache {event}: {key}" + (f" (TTL: {ttl}s)" if ttl else ""),
            extra={"cache_event": event, "cache_key": key, "ttl": ttl},
        )

    def log_error(self, error_type: str, message: str, **kwargs):
        """Log application errors."""
        self.metrics["requests_errors"] += 1
        self.logger.error(
            f"{error_type}: {message}",
            extra={"error_type": error_type, **kwargs},
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime_seconds = time.time() - self.start_time
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total if cache_total > 0 else 0
        )

        return {
            "uptime_seconds": uptime_seconds,
            "requests_total": self.metrics["requests_total"],
            "requests_errors": self.metrics["requests_errors"],
            "error_rate": (
                self.metrics["requests_errors"] / self.metrics["requests_total"]
                if self.metrics["requests_total"] > 0
                else 0
            ),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "model_predictions": self.metrics["model_predictions"],
            "batch_predictions": self.metrics["batch_predictions"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
