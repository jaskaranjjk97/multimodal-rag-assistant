from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.extractors.text_extractor import TextExtractor
from ingestion.models.document_element import ElementType


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def test_text_extractor():

    loader = PDFLoader()
    document = loader.load(PDF_PATH)

    extractor = TextExtractor()

    elements = extractor.extract(
        document=document,
        source=PDF_PATH,
    )

    assert len(elements) > 0

    # We expect text from all three pages.
    assert len(elements) == 3

    first_element = elements[0]

    assert first_element.element_type == ElementType.TEXT
    assert first_element.page_number == 1

    assert "Acme Technologies" in first_element.content
    assert "2018" in first_element.content
    assert "12 crore" in first_element.content