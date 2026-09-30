from ingestion.processors.pdf_processor import PDFProcessor


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


processor = PDFProcessor()

elements = processor.process(PDF_PATH)

print(f"\nTotal elements: {len(elements)}\n")

for index, element in enumerate(elements, start=1):
    print("=" * 60)
    print(f"Element #{index}")
    print(f"Type: {element.element_type}")
    print(f"Page: {element.page_number}")
    print(f"Source: {element.source}")
    print(f"Content:")
    print(element.content[:500])
    print(f"Metadata: {element.metadata}")