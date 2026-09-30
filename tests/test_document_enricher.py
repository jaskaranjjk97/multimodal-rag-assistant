from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.models.document_element import (
    DocumentElement,
    ElementType,
)
from ingestion.processors.document_enricher import DocumentEnricher
from llm.vision.models import DataPoint, VisionResult
from llm.vision.vision_provider import VisionProvider


class FakeVisionProvider(VisionProvider):

    def describe_image(self, image_path: str) -> VisionResult:

        return VisionResult(
            image_type="bar_chart",
            description="Acme revenue chart.",
            data_points=[
                DataPoint(
                    label="2025",
                    value="₹12 Cr",
                )
            ],
            entities=["Acme Technologies"],
            keywords=["revenue"],
        )


def test_document_enricher():

    elements = [
        DocumentElement(
            element_type=ElementType.TEXT,
            content="Acme Technologies was founded in 2018.",
            source="test.pdf",
            page_number=1,
        ),
        DocumentElement(
            element_type=ElementType.IMAGE,
            content="test.png",
            source="test.pdf",
            page_number=3,
        ),
        DocumentElement(
            element_type=ElementType.TABLE,
            content="Product | Revenue | Growth",
            source="test.pdf",
            page_number=2,
        ),
    ]

    vision_provider = FakeVisionProvider()
    image_enricher = ImageEnricher(vision_provider)
    document_enricher = DocumentEnricher(image_enricher)

    enriched_elements = document_enricher.enrich(elements)

    assert len(enriched_elements) == 3

    # TEXT remains unchanged
    assert enriched_elements[0].element_type == ElementType.TEXT
    assert "founded in 2018" in enriched_elements[0].content

    # IMAGE gets vision enrichment
    image = enriched_elements[1]

    assert image.element_type == ElementType.IMAGE
    assert "vision" in image.metadata

    assert image.metadata["vision"]["image_type"] == "bar_chart"

    assert (
        image.metadata["vision"]["data_points"][0]["value"]
        == "₹12 Cr"
    )

    # TABLE remains unchanged
    assert enriched_elements[2].element_type == ElementType.TABLE
    assert "Product" in enriched_elements[2].content

# Checking whether it works for an empty list of elements    

def test_document_enricher_with_empty_elements():

    vision_provider = FakeVisionProvider()
    image_enricher = ImageEnricher(vision_provider)
    document_enricher = DocumentEnricher(image_enricher)

    enriched_elements = document_enricher.enrich([])

    assert enriched_elements == []