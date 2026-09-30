from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.models.document_element import ElementType
from ingestion.processors.document_enricher import DocumentEnricher
from ingestion.processors.pdf_processor import PDFProcessor
from llm.vision.openai_vision_provider import OpenAIVisionProvider


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def main():

    print("\n" + "=" * 70)
    print("MULTIMODAL INGESTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Extract document elements
    # --------------------------------------------------

    processor = PDFProcessor()

    elements = processor.process(PDF_PATH)

    print(f"\nExtracted elements: {len(elements)}")

    for element in elements:
        print(
            f"  Page {element.page_number} "
            f"→ {element.element_type.value}"
        )

    # --------------------------------------------------
    # 2. Create vision components
    # --------------------------------------------------

    vision_provider = OpenAIVisionProvider()

    image_enricher = ImageEnricher(
        vision_provider=vision_provider
    )

    document_enricher = DocumentEnricher(
        image_enricher=image_enricher
    )

    # --------------------------------------------------
    # 3. Enrich document
    # --------------------------------------------------

    enriched_elements = document_enricher.enrich(elements)

    # --------------------------------------------------
    # 4. Inspect results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ENRICHED DOCUMENT")
    print("=" * 70)

    for element in enriched_elements:

        print("\n" + "-" * 70)

        print(f"Type: {element.element_type.value}")
        print(f"Page: {element.page_number}")

        if element.element_type == ElementType.IMAGE:

            vision = element.metadata.get("vision")

            print("\nIMAGE:")
            print(f"Path: {element.content}")

            print("\nVISION:")
            print(f"Type: {vision['image_type']}")
            print(f"Description: {vision['description']}")

            print("\nData points:")

            for point in vision["data_points"]:
                print(
                    f"  {point['label']} → "
                    f"{point['value']}"
                )

            print("\nEntities:")

            for entity in vision["entities"]:
                print(f"  - {entity}")

            print("\nKeywords:")

            for keyword in vision["keywords"]:
                print(f"  - {keyword}")

        else:
            print(f"\nContent:\n{element.content[:500]}")


if __name__ == "__main__":
    main()