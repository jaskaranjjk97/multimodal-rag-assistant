from llm.llm_provider import LLMProvider
from rag.rag_service import RAGService
from retrieval.models import RetrievalChunk


class FakeVectorStore:

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ):
        return [
            RetrievalChunk(
                content="Revenue in 2024 was 10 crore.",
                source="acme.pdf",
                page_number=3,
                element_type="image",
                chunk_id="chunk-001",
            )
        ]


from llm.models.rag_generation import RAGGeneration


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        question: str,
        context: str,
    ) -> RAGGeneration:

        assert "Revenue in 2024 was 10 crore." in context

        return RAGGeneration(
            answer="Revenue in 2024 was 10 crore.",
            citation_chunk_ids=["chunk-001"],
        )

def test_rag_service_returns_answer_and_citations():

    service = RAGService(
        vector_store=FakeVectorStore(),
        llm_provider=FakeLLMProvider(),
    )

    response = service.ask(
        question="What was revenue in 2024?"
    )

    assert response.answer == (
        "Revenue in 2024 was 10 crore."
    )

    assert len(response.citations) == 1

    assert response.citations[0].page_number == 3
    assert response.citations[0].element_type == "image"
    assert response.citations[0].chunk_id == "chunk-001"


def test_rag_service_rejects_empty_question():

    service = RAGService(
        vector_store=FakeVectorStore(),
        llm_provider=FakeLLMProvider(),
    )

    try:
        service.ask("")
        assert False
    except ValueError as exc:
        assert str(exc) == "Question cannot be empty."


def test_rag_service_only_cites_supporting_chunks():

    class SelectiveFakeLLMProvider(LLMProvider):

        def generate(
            self,
            question: str,
            context: str,
        ) -> RAGGeneration:

            return RAGGeneration(
                answer="Revenue in 2024 was 10 crore.",
                citation_chunk_ids=["chunk-001"],
            )

    class MultiChunkVectorStore:

        def similarity_search(
            self,
            query: str,
            k: int = 5,
        ):
            return [
                RetrievalChunk(
                    content="Revenue in 2024 was 10 crore.",
                    source="acme.pdf",
                    page_number=3,
                    element_type="image",
                    chunk_id="chunk-001",
                ),
                RetrievalChunk(
                    content="Company was founded in 2018.",
                    source="acme.pdf",
                    page_number=1,
                    element_type="text",
                    chunk_id="chunk-002",
                ),
                RetrievalChunk(
                    content="Automation grew by 32%.",
                    source="acme.pdf",
                    page_number=2,
                    element_type="table",
                    chunk_id="chunk-003",
                ),
            ]

    service = RAGService(
        vector_store=MultiChunkVectorStore(),
        llm_provider=SelectiveFakeLLMProvider(),
    )

    response = service.ask(
        question="What was revenue in 2024?"
    )

    assert len(response.citations) == 1

    assert response.citations[0].chunk_id == "chunk-001"