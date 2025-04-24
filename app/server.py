import time

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route, Router

from .metrics import *
from .model import async_predict as model_predict


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
