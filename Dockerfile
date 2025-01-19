FROM python:3.11-slim-bookworm AS python-and-curl
RUN apt-get update && apt-get -y --no-install-recommends install curl

# Install all dependencies from pyproject.toml
# NOTE: The problem with uv is that it does not read the rye.excluded-dependencies metadata
FROM python-and-curl AS dependencies-installer
WORKDIR /app
RUN <<EOF
    curl -LsSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" RYE_TOOLCHAIN=/usr/local/bin/python3 bash
    ln -s /root/.rye/shims/rye /usr/local/bin/rye
    rye pin 3.11
EOF
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache rye sync --no-dev

# Final State
# Distroless is a small image with only python, providing a non-root user
FROM gcr.io/distroless/python3-debian12:nonroot
LABEL org.opencontainers.image.authors="MrPandir <MrPandir@users.noreply.github.com>"
LABEL org.opencontainers.image.source="https://github.com/twirapp/toxicity-detector"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.title="Toxicity Detector"
LABEL org.opencontainers.image.description="Simple Multi-Language HTTP Server Text Toxicity Detector"
LABEL org.opencontainers.image.vendor="TwirApp"

WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH
# This is necessary for python to understand where to look for libraries
ENV PYTHONPATH="/app/.venv/lib/python3.11/site-packages/:${PYTHONPATH:-}"
USER nonroot
ADD --chmod=004 https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/config.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/pytorch_model.bin https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/special_tokens_map.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer_config.json https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/vocab.txt model/
COPY --from=dependencies-installer /app/.venv .venv
COPY ./app app
CMD ["/app/.venv/bin/uvicorn", "--host", "0.0.0.0", "app.server:app"]
