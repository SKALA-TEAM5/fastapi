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
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

DEFAULT_TTL = 7 * 24 * 3600
DEFAULT_DIR = Path(".cache")
DEFAULT_EMBED_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_EMBED_LOCAL_PATH = "/app/models/ko-sroberta-multitask"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_documents"
DEFAULT_EMBEDDING_CACHE_TTL = 7 * 24 * 3600
DEFAULT_COLLECTION_DOCS_CACHE_TTL = 24 * 3600

_embeddings_cache: dict[str, Any] = {}
_vectorstore_cache: dict[tuple, Any] = {}
_collection_docs_cache: dict[tuple[str, str], list[Document]] = {}
_redis_client: Any | None = None
_embeddings_lock = Lock()
_vectorstore_lock = Lock()
_collection_docs_lock = Lock()
_redis_lock = Lock()


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


def _resolve_embed_model(model_name: str) -> str:
    if model_name != DEFAULT_EMBED_MODEL:
        return model_name
    local_path = os.getenv("EMBED_MODEL_PATH", DEFAULT_EMBED_LOCAL_PATH).strip()
    return local_path if local_path and Path(local_path).exists() else model_name


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("정수 환경변수 파싱 실패: %s=%s", name, raw)
        return default


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _get_redis_client() -> Any | None:
    global _redis_client

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                try:
                    from redis import Redis

                    client = Redis.from_url(redis_url, decode_responses=True)
                    client.ping()
                    _redis_client = client
                    log.info("Redis 캐시 연결 완료: %s", redis_url)
                except Exception as e:
                    log.warning("Redis 캐시 연결 실패: %s", e)
                    return None
    return _redis_client


def _redis_get_json(key: str) -> Any | None:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("Redis 캐시 읽기 오류 (%s): %s", key, e)
        return None


def _redis_set_json(key: str, value: Any, ttl: int) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception as e:
        log.warning("Redis 캐시 쓰기 오류 (%s): %s", key, e)


def _documents_to_cache_payload(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def _documents_from_cache_payload(payload: Any) -> list[Document] | None:
    if not isinstance(payload, list):
        return None
    try:
        return [
            Document(
                page_content=str(item.get("page_content") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload
            if isinstance(item, dict)
        ]
    except Exception as e:
        log.warning("문서 캐시 역직렬화 실패: %s", e)
        return None


class CachedEmbeddings:
    def __init__(self, embeddings: HuggingFaceEmbeddings, model_name: str) -> None:
        self._embeddings = embeddings
        self._model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        cache_key = f"embedding_query:{_hash_text(self._model_name)}:{_hash_text(text)}"
        cached = _redis_get_json(cache_key)
        if isinstance(cached, list):
            log.info("Redis 임베딩 캐시 히트: %s", cache_key)
            return [float(value) for value in cached]

        vector = self._embeddings.embed_query(text)
        ttl = _env_int("EMBEDDING_CACHE_TTL_SECONDS", DEFAULT_EMBEDDING_CACHE_TTL)
        _redis_set_json(cache_key, vector, ttl)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


def _get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> Any:
    resolved_model = _resolve_embed_model(model_name)
    if resolved_model not in _embeddings_cache:
        with _embeddings_lock:
            if resolved_model not in _embeddings_cache:
                log.info("Embedding 모델 로드 중: %s", resolved_model)
                embeddings = HuggingFaceEmbeddings(model_name=resolved_model)
                _embeddings_cache[resolved_model] = CachedEmbeddings(embeddings, resolved_model)
    return _embeddings_cache[resolved_model]


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
                    validate_collection_config=False,
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
                redis_key = f"collection_docs:{collection}:{_hash_text(url)}"
                cached_payload = _redis_get_json(redis_key)
                docs = _documents_from_cache_payload(cached_payload)
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
                redis_key = f"collection_docs:{collection}:{_hash_text(url)}"
                ttl = _env_int("COLLECTION_DOCS_CACHE_TTL_SECONDS", DEFAULT_COLLECTION_DOCS_CACHE_TTL)
                _redis_set_json(redis_key, _documents_to_cache_payload(docs), ttl)
            _collection_docs_cache[cache_key] = docs
    return docs


def _invalidate_collection_cache(collection: str) -> None:
    target_keys = [key for key in _collection_docs_cache if key[0] == collection]
    for key in target_keys:
        _collection_docs_cache.pop(key, None)
    client = _get_redis_client()
    if client is None:
        return
    try:
        for key in client.scan_iter(f"collection_docs:{collection}:*"):
            client.delete(key)
    except Exception as e:
        log.warning("Redis 컬렉션 캐시 무효화 실패 (%s): %s", collection, e)
