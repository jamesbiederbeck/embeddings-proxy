# app.py reads its config at import time, so the required env vars must exist
# before any test imports it.
import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("EMBEDDINGS_URL", "http://embeddings:8080")
os.environ.setdefault("MODEL_NAME", "test-model")
