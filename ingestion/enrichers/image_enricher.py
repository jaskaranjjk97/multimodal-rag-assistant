from ingestion.models.document_element import DocumentElement, ElementType
from llm.vision.vision_provider import VisionProvider

class ImageEnricher:
    def __init__(self, vision_provider: VisionProvider):
        self.vision_provider = vision_provider

    def enrich(self, element: DocumentElement) -> DocumentElement:
        if element.element_type != ElementType.IMAGE:
            raise ValueError(  "ImageEnricher can only enrich IMAGE elements.")

        vision_result = self.vision_provider.describe_image(element.content) # element.content is the image path

        element.metadata["vision"] = vision_result.model_dump()  # converts pydantic model to a dictionary

        return element