from ingestion.processors.pdf_processor import PDFProcessor
from ingestion.models.document_element import ElementType


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def test_pdf_processor():
    processor = PDFProcessor()

    elements = processor.process(PDF_PATH)

    assert len(elements) >= 5

    element_types = [element.element_type for element in elements]

    assert ElementType.TEXT in element_types
    assert ElementType.TABLE in element_types
    assert ElementType.IMAGE in element_types