from retrieval.models import RetrievalChunk
from retrieval.vector_store import ChromaVectorStore


class FakeEmbeddingProvider:

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


def create_store(tmp_path):

    return ChromaVectorStore(
        embedding_provider=FakeEmbeddingProvider(),
        persist_directory=str(
            tmp_path / "chroma"
        ),
        collection_name="test_collection",
    )


def test_add_chunks(tmp_path):

    store = create_store(tmp_path)

    chunks = [
        RetrievalChunk(
            content="Acme revenue was 12 crore.",
            source="test.pdf",
            page_number=1,
            element_type="text",
            chunk_id="chunk-1",
        ),
        RetrievalChunk(
            content="Automation growth was 32%.",
            source="test.pdf",
            page_number=2,
            element_type="table",
            chunk_id="chunk-2",
        ),
    ]

    store.add_chunks(chunks)

    assert store.collection.count() == 2


def test_similarity_search(tmp_path):

    store = create_store(tmp_path)

    chunks = [
        RetrievalChunk(
            content="Acme revenue was 12 crore.",
            source="test.pdf",
            page_number=1,
            element_type="text",
            chunk_id="chunk-1",
        ),
        RetrievalChunk(
            content="Automation growth was 32%.",
            source="test.pdf",
            page_number=2,
            element_type="table",
            chunk_id="chunk-2",
        ),
    ]

    store.add_chunks(chunks)

    results = store.similarity_search(
        "What was the revenue?",
        k=2,
    )

    assert len(results) == 2

    assert results[0].source == "test.pdf"
    assert results[0].chunk_id == "chunk-1"


def test_empty_query_fails(tmp_path):

    store = create_store(tmp_path)

    try:
        store.similarity_search("   ")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "Query cannot be empty" in str(error)