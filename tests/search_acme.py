from retrieval.embeddings import OpenAIEmbeddingProvider
from retrieval.vector_store import ChromaVectorStore


def main():

    embedding_provider = OpenAIEmbeddingProvider()

    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory="data/chroma",
        collection_name="acme_documents",
    )

    queries = [
        "What was Acme Technologies' revenue in 2024?",
        "Which product had the highest growth rate?",
        "What was Acme Technologies' annual revenue trend from 2022 to 2025?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = vector_store.similarity_search(
            query=query,
            k=10,
        )

        print(f"\nRetrieved chunks: {len(results)}")

        for index, chunk in enumerate(results, start=1):

            print("\n" + "-" * 70)
            print(f"Result #{index}")

            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Type: {chunk.element_type}")
            print(f"Page: {chunk.page_number}")
            print(f"Source: {chunk.source}")

            print("\nContent:")
            print(chunk.content)


if __name__ == "__main__":
    main()