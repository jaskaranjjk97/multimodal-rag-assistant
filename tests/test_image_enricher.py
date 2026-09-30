from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.models.document_element import (
    DocumentElement,
    ElementType,
)
from llm.vision.models import DataPoint, VisionResult
from llm.vision.vision_provider import VisionProvider


class FakeVisionProvider(VisionProvider):

    def describe_image(self, image_path: str) -> VisionResult:

        return VisionResult(
            image_type="bar_chart",
            description="Test revenue chart.",
            data_points=[
                DataPoint(
                    label="2025",
                    value="₹12 Cr",
                )
            ],
            entities=["Acme Technologies"],
            keywords=["revenue", "chart"],
        )


def test_image_enricher():

    element = DocumentElement(
        element_type=ElementType.IMAGE,
        content="data/processed/images/test.png",
        source="test.pdf",
        page_number=3,
    )

    provider = FakeVisionProvider()
    enricher = ImageEnricher(provider)

    enriched_element = enricher.enrich(element)

    assert "vision" in enriched_element.metadata

    vision = enriched_element.metadata["vision"]

    assert vision["image_type"] == "bar_chart"
    assert vision["description"] == "Test revenue chart."

    assert vision["data_points"][0]["label"] == "2025"
    assert vision["data_points"][0]["value"] == "₹12 Cr"

    assert "Acme Technologies" in vision["entities"]
    assert "revenue" in vision["keywords"]

    def test_image_enricher_rejects_non_image():

        element = DocumentElement(
            element_type=ElementType.TEXT,
            content="Some text",
            source="test.pdf",
            page_number=1,
        )

        provider = FakeVisionProvider()
        enricher = ImageEnricher(provider)

        try:
            enricher.enrich(element)
            assert False, "Expected ValueError"
        except ValueError as error:
            assert "IMAGE" in str(error)