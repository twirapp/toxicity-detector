# Install all dependencies from pyproject.toml
FROM python:3.11-slim-bookworm AS dependencies-installer
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.6.11 /uv /bin/
COPY uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache uv sync --no-dev --frozen --no-install-project --link-mode=copy

# Final State
# Distroless is a small image with only python, providing a non-root user
FROM gcr.io/distroless/python3-debian12:nonroot
WORKDIR /app

LABEL org.opencontainers.image.authors="MrPandir <MrPandir@users.noreply.github.com>"
LABEL org.opencontainers.image.source="https://github.com/twirapp/toxicity-detector"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.title="Toxicity Detector"
LABEL org.opencontainers.image.description="Simple Multi-Language HTTP Server Text Toxicity Detector"
LABEL org.opencontainers.image.vendor="TwirApp"

# This is necessary for python to understand where to look for libraries
ENV PYTHONPATH="/app/.venv/lib/python3.11/site-packages/:${PYTHONPATH:-}"
USER nonroot
ADD --chmod=005 https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/config.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/pytorch_model.bin https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/special_tokens_map.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer_config.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/vocab.txt model/
COPY --from=dependencies-installer /app/.venv .venv
COPY ./app app
CMD ["/app/.venv/bin/uvicorn", "--host", "0.0.0.0", "app.server:app"]
