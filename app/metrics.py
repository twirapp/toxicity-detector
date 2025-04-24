from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    disable_created_metrics,
    generate_latest,
)

from .config import metrics_prefix

__all__ = [
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "ACTIVE_REQUESTS",
    "MODEL_ERRORS",
    "MODEL_DURATION",
    "LOGITS_DISTRIBUTION",
    "generate_latest",
]

generate_latest = generate_latest

disable_created_metrics()

REQUEST_COUNT = Counter(
    f"{metrics_prefix}_http_requests",
    "Total number of HTTP requests",
    ["endpoint", "result"],
)

REQUEST_DURATION = Summary(
    f"{metrics_prefix}_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
)

ACTIVE_REQUESTS = Gauge(
    f"{metrics_prefix}_active_http_requests", "Current number of active HTTP requests"
)

MODEL_ERRORS = Counter(
    f"{metrics_prefix}_model_errors_total", "Total number of model errors"
)

MODEL_DURATION = Summary(
    f"{metrics_prefix}_model_execution_duration_seconds",
    "Model execution duration in seconds",
)

# Define buckets for logits distribution ranging from -10 to 10 with a step of 0.5
# 41 buckets to cover -10 to 10 inclusive
logits_buckets = [round(-10.0 + i * 0.5, 1) for i in range(41)]

LOGITS_DISTRIBUTION = Histogram(
    f"{metrics_prefix}_logits_distribution",
    "Distribution of the first logit value",
    buckets=logits_buckets,
)
