from ingestion.models.document_element import (
    DocumentElement,
    ElementType,
)

from retrieval.representation import RetrievalRepresentationBuilder


def test_text_representation():
    element = DocumentElement(
        element_type=ElementType.TEXT,
        content="Acme Technologies was founded in 2018.",
        source="test.pdf",
        page_number=1,
    )

    builder = RetrievalRepresentationBuilder()

    result = builder.build(element)

    assert result.content == (
        "Acme Technologies was founded in 2018."
    )
    assert result.element_type == "text"
    assert result.page_number == 1
    assert result.source == "test.pdf"


def test_table_representation():
    element = DocumentElement(
        element_type=ElementType.TABLE,
        content=(
            "Product | Revenue | Growth\n"
            "AI Assistant | ₹5 Cr | 25%"
        ),
        source="test.pdf",
        page_number=2,
    )

    builder = RetrievalRepresentationBuilder()

    result = builder.build(element)

    assert "AI Assistant" in result.content
    assert "₹5 Cr" in result.content
    assert result.element_type == "table"


def test_image_representation():
    element = DocumentElement(
        element_type=ElementType.IMAGE,
        content="data/processed/chart.png",
        source="test.pdf",
        page_number=3,
        metadata={
            "vision": {
                "image_type": "bar chart",
                "description": (
                    "Revenue of Acme Technologies "
                    "from 2022 to 2025."
                ),
                "data_points": [
                    {
                        "label": "2025 Revenue",
                        "value": "12 Crore",
                    }
                ],
                "entities": [
                    "Acme Technologies"
                ],
                "keywords": [
                    "revenue",
                    "bar chart",
                ],
            }
        },
    )

    builder = RetrievalRepresentationBuilder()

    result = builder.build(element)

    assert "bar chart" in result.content
    assert "Acme Technologies" in result.content
    assert "2025 Revenue: 12 Crore" in result.content
    assert result.element_type == "image"


def test_image_without_vision_enrichment_fails():
    element = DocumentElement(
        element_type=ElementType.IMAGE,
        content="data/processed/chart.png",
        source="test.pdf",
        page_number=3,
    )

    builder = RetrievalRepresentationBuilder()

    try:
        builder.build(element)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "vision enrichment" in str(error)