FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Pass --build-arg UV_SYNC_EXTRA_ARGS="--extra qdrant" to bake in vector-store support.
ARG UV_SYNC_EXTRA_ARGS=""
RUN uv sync --locked --no-install-project --no-dev ${UV_SYNC_EXTRA_ARGS}

COPY app.py backends.py vector_store.py .

EXPOSE 8081
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
