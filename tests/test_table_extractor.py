import pymupdf

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.extractors.table_extractor import TableExtractor
from ingestion.models.document_element import ElementType

PDF_PATH = "data/raw/acme_multimodal_test.pdf"

def test_table_extractor():
    loader=PDFLoader()
    document=loader.load(PDF_PATH)

    extractor=TableExtractor()

    elements=extractor.extract(
        document=document,
        source=PDF_PATH,
    )

    assert len(elements) == 1

    table = elements[0]

    assert table.element_type == ElementType.TABLE
    assert table.page_number == 2

    assert "Product" in table.content
    assert "AI Assistant" in table.content
    assert "Automation" in table.content
    assert "32%" in table.content

    assert table.metadata["rows"] == 4
    assert table.metadata["columns"] == 3


def test_table_extraction_normalizes_rupee_symbol():

    rows = [
        ["Product", "Revenue", "Growth"],
        ["AI Assistant", "I5 Cr", "25%"],
        ["Analytics", "I4 Cr", "18%"],
        ["Automation", "I3 Cr", "32%"],
    ]

    result = TableExtractor._table_to_text(rows)

    assert "AI Assistant | ₹5 Cr | 25%" in result
    assert "Analytics | ₹4 Cr | 18%" in result
    assert "Automation | ₹3 Cr | 32%" in result