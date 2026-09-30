from ingestion.processors.document_enricher import DocumentEnricher
from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.processors.pdf_processor import PDFProcessor
from llm.vision.openai_vision_provider import OpenAIVisionProvider
from retrieval.representation import RetrievalRepresentationBuilder


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def main():
    processor = PDFProcessor()
    elements = processor.process(PDF_PATH)

    vision_provider = OpenAIVisionProvider()
    image_enricher = ImageEnricher(vision_provider)
    document_enricher = DocumentEnricher(image_enricher)

    enriched_elements = document_enricher.enrich(elements)

    builder = RetrievalRepresentationBuilder()

    retrieval_documents = [
        builder.build(element)
        for element in enriched_elements
    ]

    print("\n" + "=" * 70)
    print("RETRIEVAL REPRESENTATIONS")
    print("=" * 70)

    for index, document in enumerate(retrieval_documents, start=1):
        print("\n" + "-" * 70)
        print(f"Document #{index}")
        print(f"Type: {document.element_type}")
        print(f"Page: {document.page_number}")
        print(f"Source: {document.source}")
        print("\nContent:")
        print(document.content)


if __name__ == "__main__":
    main()