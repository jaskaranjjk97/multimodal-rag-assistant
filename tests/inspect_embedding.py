from retrieval.embeddings import OpenAIEmbeddingProvider


def main():
    provider = OpenAIEmbeddingProvider()

    text = """
    Acme Technologies generated 12 crore rupees
    in revenue in 2025.
    """

    embedding = provider.embed_text(text)

    print("\n" + "=" * 70)
    print("EMBEDDING INSPECTION")
    print("=" * 70)

    print(f"\nModel: {provider.model}")
    print(f"Vector dimensions: {len(embedding)}")

    print("\nFirst 10 values:")
    for value in embedding[:10]:
        print(f"  {value}")

    print("\nLast 10 values:")
    for value in embedding[-10:]:
        print(f"  {value}")


if __name__ == "__main__":
    main()