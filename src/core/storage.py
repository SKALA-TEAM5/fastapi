import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

DEFAULT_TTL = 7 * 24 * 3600
DEFAULT_DIR = Path(".cache")
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHROMA_DIR = "chroma_db"

_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}
_vectorstore_cache: dict[tuple, Chroma] = {}


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
        _embeddings_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings_cache[model_name]


def reset_chroma_collection(
    collection_name: str,
    persist_dir: str = DEFAULT_CHROMA_DIR,
) -> None:
    collection = _sanitize_name(collection_name)
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection)
        log.info(f"ChromaDB 컬렉션 삭제: {collection}")
    except Exception as e:
        log.info(f"컬렉션 삭제 스킵 (없거나 오류): {e}")


def save_chunks_to_chroma(
    chunks: list[Document],
    collection_name: str,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Chroma:
    collection = _sanitize_name(collection_name)
    embeddings = _get_embeddings(embed_model)
    print(f"ChromaDB 임베딩 및 저장 중 (컬렉션: {collection}, 청크 수: {len(chunks)})...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    print(f"✅ ChromaDB 저장 완료 → {Path(persist_dir).resolve() / collection}")
    return vectorstore


def load_vectorstore(
    collection_name: str,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Chroma:
    collection = _sanitize_name(collection_name)
    key = (collection, persist_dir, embed_model)
    if key not in _vectorstore_cache:
        embeddings = _get_embeddings(embed_model)
        _vectorstore_cache[key] = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    return _vectorstore_cache[key]
