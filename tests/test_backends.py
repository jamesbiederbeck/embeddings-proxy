import json

import httpx

from backends import LlamaCppBackend, OpenAIBackend, TransformersInferenceBackend


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_transformers_inference_backend():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "http://embeddings:8080/vectors"
        assert json.loads(request.content) == {"text": "hello"}
        return httpx.Response(200, json={"text": "hello", "vector": [0.1, 0.2, 0.3]})

    backend = TransformersInferenceBackend(make_client(handler), "http://embeddings:8080", "mini")
    assert await backend.embed("hello") == [0.1, 0.2, 0.3]


async def test_openai_backend():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.openai.com/v1/embeddings"
        assert json.loads(request.content) == {"input": "hello", "model": "text-embedding-3-small"}
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    backend = OpenAIBackend(
        make_client(handler), "https://api.openai.com", "text-embedding-3-small", "sk-test"
    )
    assert await backend.embed("hello") == [1.0, 2.0]


async def test_llamacpp_backend_no_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://llamacpp:8080/v1/embeddings"
        assert json.loads(request.content) == {"input": "hello", "model": "nomic-embed"}
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"embedding": [3.0, 4.0]}]})

    backend = LlamaCppBackend(make_client(handler), "http://llamacpp:8080", "nomic-embed")
    assert await backend.embed("hello") == [3.0, 4.0]
    assert backend.name == "llamacpp"


async def test_openai_backend_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    backend = OpenAIBackend(make_client(handler), "https://api.openai.com", "m", "sk-test")
    try:
        await backend.embed("hello")
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected HTTPStatusError")
