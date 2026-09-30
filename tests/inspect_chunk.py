from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.processors.pdf_processor import PDFProcessor
from ingestion.processors.document_enricher import DocumentEnricher
from ingestion.enrichers.image_enricher import ImageEnricher
from llm.vision.openai_vision_provider import OpenAIVisionProvider
from retrieval.representation import RetrievalRepresentationBuilder
from retrieval.chunker import RetrievalChunker


pdf_path = "data/raw/acme_multimodal_test.pdf"

def main():

    processor = PDFProcessor()
    elements = processor.process(pdf_path)

    vision_provider = OpenAIVisionProvider()
    image_enricher = ImageEnricher(vision_provider)
    document_enricher = DocumentEnricher(image_enricher)

    enriched_elements = document_enricher.enrich(elements)

    builder=RetrievalRepresentationBuilder()

    retrieved_documents = [builder.build(element) for element in enriched_elements]

    chunker = RetrievalChunker(chunk_size=500,chunk_overlap=50) 

    all_chunks=[] 

    for document in retrieved_documents:
        chunks=chunker.chunk(document)
        all_chunks.extend(chunks)

    print("\n" + "=" * 70)
    print("RETRIEVAL CHUNKS")
    print("=" * 70)

    print(f"\nTotal chunks: {len(all_chunks)}")

    for index, chunk in enumerate(all_chunks, start=1):
        print("\n" + "-" * 70)
        print(f"Chunk #{index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Type: {chunk.element_type}")
        print(f"Page: {chunk.page_number}")
        print(f"Source: {chunk.source}")

        print("\nContent:")
        print(chunk.content)


if __name__ == "__main__":
    main()