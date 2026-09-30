from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.processors.document_enricher import DocumentEnricher
from ingestion.processors.pdf_processor import PDFProcessor

from llm.vision.openai_vision_provider import OpenAIVisionProvider

from retrieval.chunker import RetrievalChunker
from retrieval.embeddings import OpenAIEmbeddingProvider
from retrieval.representation import RetrievalRepresentationBuilder
from retrieval.vector_store import ChromaVectorStore


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def main():

    print("\n" + "=" * 70)
    print("INDEXING ACME DOCUMENT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Extract document elements
    # --------------------------------------------------

    processor = PDFProcessor()

    elements = processor.process(PDF_PATH)

    print(f"\nExtracted elements: {len(elements)}")

    # --------------------------------------------------
    # 2. Enrich images with vision
    # --------------------------------------------------

    vision_provider = OpenAIVisionProvider()

    image_enricher = ImageEnricher(
        vision_provider=vision_provider
    )

    document_enricher = DocumentEnricher(
        image_enricher=image_enricher
    )

    enriched_elements = document_enricher.enrich(
        elements
    )

    # --------------------------------------------------
    # 3. Build retrieval representations
    # --------------------------------------------------

    representation_builder = (
        RetrievalRepresentationBuilder()
    )

    retrieval_documents = [
        representation_builder.build(element)
        for element in enriched_elements
    ]

    print(
        f"Retrieval documents: "
        f"{len(retrieval_documents)}"
    )

    # --------------------------------------------------
    # 4. Chunk
    # --------------------------------------------------

    chunker = RetrievalChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = []

    for document in retrieval_documents:
        chunks.extend(
            chunker.chunk(document)
        )

    print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------
    # 5. Create embedding provider
    # --------------------------------------------------

    embedding_provider = (
        OpenAIEmbeddingProvider()
    )

    # --------------------------------------------------
    # 6. Create Chroma vector store
    # --------------------------------------------------

    vector_store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        persist_directory="data/chroma",
        collection_name="acme_documents",
    )

    # --------------------------------------------------
    # 7. Delete old collection
    # --------------------------------------------------

    try:
        vector_store.client.delete_collection(name="acme_documents")

        print("\n Deleted the Existing collection")

    except Exception:
        print("\n No existing collection to delete.")


    #---------------------------------------------------
    # 8. Create a Fresh collection 
    #---------------------------------------------------

    vector_store.collection = (vector_store.client.get_or_create_collection(name="acme_documents"))

    #---------------------------------------------------
    # 9. Adding chunks
    #---------------------------------------------------

    print("\nGenerating embeddings and indexing...")

    vector_store.add_chunks(chunks)

    print(
        f"Documents in Chroma: "
        f"{vector_store.collection.count()}"
    )

    print("\n" + "=" * 70)
    print("REINDEXING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()