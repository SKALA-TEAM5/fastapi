from langchain_core.documents import Document

from src.tools import safety_doc_reference_vector


class FakeVectorStore:
    def similarity_search_with_score(self, query: str, k: int):
        assert query == "안전모 구입"
        assert k == 3
        return [
            (
                Document(
                    page_content="안전모 구입 시 거래명세서와 영수증을 확인합니다.",
                    metadata={
                        "_id": "point-1",
                        "source": "safety-guide.md",
                        "section_title": "개인보호구",
                    },
                ),
                0.91,
            )
        ]


def test_search_reference_vector_db_uses_batch_embedding_and_normalizes_payload(
    monkeypatch,
):
    captured = {}

    def fake_load_vectorstore(collection_name, *, qdrant_url, embed_model):
        captured.update(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            embed_model=embed_model,
        )
        return FakeVectorStore()

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SAFETY_DOC_EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
    monkeypatch.setattr(
        safety_doc_reference_vector,
        "load_vectorstore",
        fake_load_vectorstore,
    )

    results = safety_doc_reference_vector.search_reference_vector_db(
        query="안전모 구입",
        collection_name="safety-guide",
        top_k=3,
    )

    assert captured == {
        "collection_name": "safety-guide",
        "qdrant_url": "http://localhost:6333",
        "embed_model": "jhgan/ko-sroberta-multitask",
    }
    assert results == [
        {
            "id": "point-1",
            "score": 0.91,
            "payload": {
                "text": "안전모 구입 시 거래명세서와 영수증을 확인합니다.",
                "source": "safety-guide.md",
                "section": "개인보호구",
                "metadata": {
                    "_id": "point-1",
                    "source": "safety-guide.md",
                    "section_title": "개인보호구",
                },
            },
        }
    ]
