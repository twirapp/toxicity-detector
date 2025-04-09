import time

from prometheus_client import (
    Counter,
    Gauge,
    Summary,
    disable_created_metrics,
    generate_latest,
)
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route, Router

from .model import async_predict as model_predict
from .model import metrics_prefix

# Initialize Prometheus metrics
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


async def predict(request: Request):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    endpoint = request.url.path

    try:
        text = request.query_params.get("text")
        result = await model_predict(text) if text else False

        label = "toxic" if result else "non_toxic"
        REQUEST_COUNT.labels(endpoint=endpoint, result=label).inc()

        return PlainTextResponse("1" if result else "0")
    finally:
        duration = time.time() - start_time

        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        ACTIVE_REQUESTS.dec()


async def metrics(request: Request):
    return Response(generate_latest(), media_type="text/plain")


app = Router(
    (
        Route("/", predict),
        Route("/predict", predict),
        Route("/metrics", metrics),
    ),
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"]),
    ],
)
