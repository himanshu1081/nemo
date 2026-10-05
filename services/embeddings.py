"""Embedding providers for conversation memory.

EMBEDDING_PROVIDER picks the backend:
  fastembed (default)  local ONNX model, no API key, ~15ms per query on CPU
  openai               any OpenAI-compatible /embeddings endpoint (OpenAI, Jina, Mistral, Ollama...)

The vector size must match conversation_chunks.embedding (384 in the migration).
Switching to a model with another size needs a migration and re-embedding the summaries.
"""
import logging
import os
import threading
from typing import Protocol

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))


class Embedder(Protocol):
    model_name: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_document(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        # loaded lazily: the first load downloads the model (~70MB)
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding
                    self._model = TextEmbedding(self.model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._get_model().query_embed([text]))).tolist()

    def embed_document(self, text: str) -> list[float]:
        return next(iter(self._get_model().passage_embed([text]))).tolist()


class OpenAICompatibleEmbedder:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(self, text: str) -> list[float]:
        response = httpx.post(
            self._url,
            headers=self._headers,
            json={"model": self.model_name, "input": text, "dimensions": EMBEDDING_DIMENSIONS},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)


def _build_embedder() -> Embedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    if provider == "fastembed":
        return FastEmbedEmbedder(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            os.getenv("EMBEDDING_API_KEY"),
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


embedder = _build_embedder()


def warm_up():
    """Load the model ahead of the first request so it doesn't eat Alexa's 8s budget."""
    try:
        vector = embedder.embed_query("warm up")
        if len(vector) != EMBEDDING_DIMENSIONS:
            logger.error("Embedding size %s doesn't match EMBEDDING_DIMENSIONS %s", len(vector), EMBEDDING_DIMENSIONS)
    except Exception:
        logger.exception("Embedding warm-up failed")
