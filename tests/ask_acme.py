from llm.openai_llm_provider import OpenAILLMProvider
from rag.rag_service import RAGService
from retrieval.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from retrieval.vector_store.chroma_store import ChromaVectorStore


def main():

    question = (
        "What was Acme Technologies' annual revenue trend "
        "from 2022 to 2025?"
    )

    print("=" * 70)
    print("ACME RAG ASSISTANT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Create embedding provider
    # --------------------------------------------------

    embedding_provider = OpenAIEmbeddingProvider()

    # --------------------------------------------------
    # 2. Connect to Chroma
    # --------------------------------------------------

    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory="data/chroma",
        collection_name="acme_documents",
    )

    # --------------------------------------------------
    # 3. Create LLM provider
    # --------------------------------------------------

    llm = OpenAILLMProvider(
        model="gpt-4o-mini"
    )

    # --------------------------------------------------
    # 4. Create RAG service
    # --------------------------------------------------

    rag_service = RAGService(
        vector_store=vector_store,
        llm_provider=llm,
    )

    # --------------------------------------------------
    # 5. Ask question
    # --------------------------------------------------

    response = rag_service.ask(
        question=question,
        k=3,
    )

    # --------------------------------------------------
    # 6. Display answer
    # --------------------------------------------------

    print("\nAnswer:")
    print("-" * 70)
    print(response.answer)

    # --------------------------------------------------
    # 7. Display verified citations
    # --------------------------------------------------

    print("\nCitations:")
    print("-" * 70)

    for index, citation in enumerate(
        response.citations,
        start=1,
    ):
        source_name = (
            citation.source
            .replace("\\", "/")
            .split("/")[-1]
        )

        print(
            f"[{index}] "
            f"{source_name} | "
            f"Page {citation.page_number} | "
            f"Type: {citation.element_type}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()