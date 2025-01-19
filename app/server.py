from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route, Router

from .model import predict as call_model


async def predict(request: Request):
    text = request.query_params.get("text")
    result = call_model(text) if text else False
    return PlainTextResponse("1" if result else "0")


app = Router(
    (
        Route("/", predict),
        Route("/predict", predict),
    ),
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"]),
    ],
)
