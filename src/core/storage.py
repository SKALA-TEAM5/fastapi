# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-05-29
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. LocalJSONCache              : TTL 기반 로컬 JSON 캐시 클래스
# 2. load_vectorstore()          : Qdrant 컬렉션 벡터스토어 로드 (read)
# 3. load_collection_documents() : 컬렉션 전체 문서 로드 (read)
# --------------------------------------------------------------------------
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

DEFAULT_TTL = 7 * 24 * 3600
DEFAULT_DIR = Path(".cache")
DEFAULT_EMBED_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_documents"

_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}
_vectorstore_cache: dict[tuple, Any] = {}
_collection_docs_cache: dict[tuple[str, str], list[Document]] = {}
_embeddings_lock = Lock()
_vectorstore_lock = Lock()
_collection_docs_lock = Lock()


class LocalJSONCache:
    def __init__(self, ttl: int = DEFAULT_TTL, cache_dir: Path = DEFAULT_DIR) -> None:
        self.ttl = ttl
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in key)
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - payload["_ts"]
            if age > self.ttl:
                log.info(f"캐시 만료: {key} (경과 {age / 3600:.1f}h)")
                return None
            log.info(f"캐시 히트: {key} (경과 {age / 3600:.1f}h)")
            return payload["value"]
        except Exception as e:
            log.warning(f"캐시 읽기 오류 ({key}): {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(
                json.dumps({"_ts": time.time(), "value": value}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"캐시 저장: {key}")
        except Exception as e:
            log.warning(f"캐시 쓰기 오류 ({key}): {e}")

    def invalidate(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            log.info(f"캐시 무효화: {key}")


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_-")
    if len(sanitized) < 3:
        sanitized = f"col_{sanitized}"
    return sanitized[:63]


def _get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    if model_name not in _embeddings_cache:
        with _embeddings_lock:
            if model_name not in _embeddings_cache:
                _embeddings_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings_cache[model_name]


def _get_qdrant_url(qdrant_url: str | None = None) -> str:
    return qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)


def _get_qdrant_client(qdrant_url: str | None = None):
    from qdrant_client import QdrantClient
    return QdrantClient(url=_get_qdrant_url(qdrant_url))


def load_vectorstore(
    collection_name: str,
    *,
    qdrant_url: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Any:
    from langchain_qdrant import QdrantVectorStore

    collection = _sanitize_name(collection_name)
    url = _get_qdrant_url(qdrant_url)
    key = (collection, url, embed_model)
    if key not in _vectorstore_cache:
        with _vectorstore_lock:
            if key not in _vectorstore_cache:
                embeddings = _get_embeddings(embed_model)
                _vectorstore_cache[key] = QdrantVectorStore(
                    client=_get_qdrant_client(url),
                    collection_name=collection,
                    embedding=embeddings,
                )
    return _vectorstore_cache[key]


def load_collection_documents(
    collection_name: str,
    *,
    qdrant_url: str | None = None,
) -> list[Document]:
    collection = _sanitize_name(collection_name)
    url = _get_qdrant_url(qdrant_url)
    cache_key = (collection, url)
    docs = _collection_docs_cache.get(cache_key)
    if docs is None:
        with _collection_docs_lock:
            docs = _collection_docs_cache.get(cache_key)
            if docs is None:
                client = _get_qdrant_client(url)
                offset = None
                docs = []
                while True:
                    points, next_offset = client.scroll(
                        collection_name=collection,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=offset,
                    )
                    for point in points:
                        payload = point.payload or {}
                        metadata = dict(payload.get("metadata") or {})
                        page_content = payload.get("page_content") or ""
                        docs.append(Document(page_content=page_content, metadata=metadata))
                    if next_offset is None:
                        break
                    offset = next_offset
                _collection_docs_cache[cache_key] = docs
    return docs


def _invalidate_collection_cache(collection: str) -> None:
    target_keys = [key for key in _collection_docs_cache if key[0] == collection]
    for key in target_keys:
        _collection_docs_cache.pop(key, None)
