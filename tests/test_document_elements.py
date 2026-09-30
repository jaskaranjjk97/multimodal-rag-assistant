from ingestion.models.document_element import (ElementType, DocumentElement,)


def test_text_element():
    element= DocumentElement(
        element_type=ElementType.TEXT,
        content="The company was founded in 2015.",
        source="sample.pdf",
        page_number=1
        )
     
    assert element.element_type == ElementType.TEXT
    assert element.page_number == 1
    assert "2015" in element.content
    assert element.bbox is None
