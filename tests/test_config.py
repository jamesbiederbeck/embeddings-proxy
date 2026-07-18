import pytest
from pydantic import ValidationError

from config import AppConfig, load_config

MULTI_BACKEND_YAML = """
health_check_interval: 10
backends:
  - name: llama-local
    type: llamacpp
    url: http://llama:8080
    model: nomic-embed-text-v1.5
    embedding_task: code
  - type: openai
    model: text-embedding-3-small
    api_key_env: TEST_OPENAI_KEY
vector_stores:
  - name: qdrant-code
    type: qdrant
    url: http://qdrant:6333
    collection: code-nomic
    model: nomic-embed-text-v1.5
    metadata:
      corpus: code
"""


@pytest.fixture
def file_mode(monkeypatch, tmp_path):
    """Clear legacy env vars so load_config falls through to the YAML file."""

    def _write(content: str):
        for var in ("MODEL_NAME", "BACKEND", "EMBEDDINGS_URL", "VECTOR_STORE"):
            monkeypatch.delenv(var, raising=False)
        path = tmp_path / "config.yaml"
        path.write_text(content)
        monkeypatch.setenv("CONFIG_FILE", str(path))
        return path

    return _write


def test_legacy_env_synthesizes_single_backend(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "mini")
    monkeypatch.setenv("EMBEDDINGS_URL", "http://embeddings:8080")
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.delenv("VECTOR_STORE", raising=False)

    config = load_config()
    assert len(config.backends) == 1
    backend = config.backends[0]
    assert backend.type == "transformers-inference"
    assert backend.name == "transformers-inference"
    assert backend.model == "mini"
    assert backend.url == "http://embeddings:8080"
    assert config.vector_stores == []


def test_legacy_env_with_qdrant_store(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "mini")
    monkeypatch.setenv("EMBEDDINGS_URL", "http://embeddings:8080")
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "embeddings")

    config = load_config()
    store = config.vector_stores[0]
    assert store.type == "qdrant"
    assert store.collection == "embeddings"
    # The legacy store is tied to the one configured model — the routing key.
    assert store.model == "mini"


def test_legacy_env_missing_required_var_fails_fast(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "mini")
    monkeypatch.delenv("EMBEDDINGS_URL", raising=False)
    with pytest.raises(KeyError):
        load_config()


def test_yaml_file_multi_backend(monkeypatch, file_mode):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    file_mode(MULTI_BACKEND_YAML)

    config = load_config()
    assert config.health_check_interval == 10
    assert [b.name for b in config.backends] == ["llama-local", "openai"]
    assert config.backends[0].embedding_task == "code"
    assert config.backends[1].api_key == "sk-test"
    assert config.backends[1].url == "https://api.openai.com"
    store = config.vector_stores[0]
    assert store.metadata == {"corpus": "code"}


def test_health_check_interval_env_overrides_file(monkeypatch, file_mode):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    monkeypatch.setenv("HEALTH_CHECK_INTERVAL", "5")
    file_mode(MULTI_BACKEND_YAML)

    assert load_config().health_check_interval == 5.0


def test_missing_config_file_raises_informative_error(monkeypatch, file_mode):
    path = file_mode("")
    path.unlink()
    with pytest.raises(FileNotFoundError, match="MODEL_NAME"):
        load_config()


def test_missing_api_key_env_var_is_rejected(monkeypatch, file_mode):
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    file_mode(MULTI_BACKEND_YAML)
    with pytest.raises(ValidationError, match="TEST_OPENAI_KEY"):
        load_config()


def test_duplicate_names_are_rejected():
    with pytest.raises(ValidationError, match="duplicate backend names"):
        AppConfig.model_validate(
            {
                "backends": [
                    {"name": "a", "type": "llamacpp", "url": "http://x", "model": "m1"},
                    {"name": "a", "type": "llamacpp", "url": "http://y", "model": "m2"},
                ]
            }
        )


def test_two_stores_for_one_model_are_rejected():
    with pytest.raises(ValidationError, match="same model"):
        AppConfig.model_validate(
            {
                "backends": [{"type": "llamacpp", "url": "http://x", "model": "m"}],
                "vector_stores": [
                    {"name": "s1", "type": "qdrant", "url": "http://q", "collection": "c1", "model": "m"},
                    {"name": "s2", "type": "qdrant", "url": "http://q", "collection": "c2", "model": "m"},
                ],
            }
        )


def test_unknown_backend_type_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"backends": [{"type": "carrier-pigeon", "model": "m"}]})


def test_url_required_for_non_openai_backends():
    with pytest.raises(ValidationError, match="url is required"):
        AppConfig.model_validate({"backends": [{"type": "llamacpp", "model": "m"}]})
