from retrieval.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from retrieval.reranking.lexical_reranker import LexicalReranker
from retrieval.vector_store.chroma_store import ChromaVectorStore


def inspect_query(query: str) -> None:

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    embedding_provider = OpenAIEmbeddingProvider()

    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory="data/chroma",
        collection_name="acme_documents",
    )

    # Retrieve a broad candidate set first.
    candidates = vector_store.similarity_search(
        query=query,
        k=10,
    )

    reranker = LexicalReranker()

    reranked = reranker.rerank(
        query=query,
        chunks=candidates,
        top_k=3,
    )

    print("\n" + "-" * 70)
    print("RERANKED RESULTS")
    print("-" * 70)

    for index, chunk in enumerate(reranked, start=1):

        print(f"\nResult #{index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Type: {chunk.element_type}")
        print(f"Page: {chunk.page_number}")
        print(f"Content: {chunk.content}")


if __name__ == "__main__":

    inspect_query(
        "What was Acme Technologies' revenue in 2024?"
    )

    inspect_query(
        "Which product had the highest growth rate?"
    )

    inspect_query(
        "What was Acme Technologies' annual revenue trend from 2022 to 2025?"
    )