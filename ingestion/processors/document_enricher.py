from ingestion.enrichers.image_enricher import ImageEnricher
from ingestion.models.document_element import DocumentElement


class DocumentEnricher:

    def __init__(self, image_enricher: ImageEnricher):
        self.image_enricher = image_enricher

    def enrich(
        self,
        elements: list[DocumentElement],
    ) -> list[DocumentElement]:

        enriched_elements = []

        for element in elements:

            if element.element_type.value == "image":
                enriched_element = self.image_enricher.enrich(element)
                enriched_elements.append(enriched_element)

            else:
                enriched_elements.append(element)

        return enriched_elements