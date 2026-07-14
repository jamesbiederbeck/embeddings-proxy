FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY app.py backends.py .

EXPOSE 8081
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
