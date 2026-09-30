from ingestion.models.document_element import (
    DocumentElement,
    ElementType,
)

from retrieval.models.retrieval_document import RetrievalDocument


class RetrievalRepresentationBuilder:

    def build(
        self,
        element: DocumentElement,
    ) -> RetrievalDocument:

        if element.element_type == ElementType.TEXT:
            content = self._build_text_representation(element)

        elif element.element_type == ElementType.TABLE:
            content = self._build_table_representation(element)

        elif element.element_type == ElementType.IMAGE:
            content = self._build_image_representation(element)

        else:
            raise ValueError(
                f"Unsupported element type: {element.element_type}"
            )

        return RetrievalDocument(
            content=content,
            source=element.source,
            page_number=element.page_number,
            element_type=element.element_type.value,
            metadata=element.metadata.copy(),
        )

    @staticmethod
    def _build_text_representation(
        element: DocumentElement,
    ) -> str:
        return element.content

    @staticmethod
    def _build_table_representation(
        element: DocumentElement,
    ) -> str:
        return element.content

    @staticmethod
    def _build_image_representation(
        element: DocumentElement,
    ) -> str:

        vision = element.metadata.get("vision")

        if not vision:
            raise ValueError(
                "Image element does not contain vision enrichment."
            )

        lines = [
            f"Image Type: {vision['image_type']}",
            "",
            f"Description: {vision['description']}",
        ]

        data_points = vision.get("data_points", [])

        if data_points:
            lines.append("")
            lines.append("Data Points:")

            for point in data_points:
                lines.append(
                    f"{point['label']}: {point['value']}"
                )

        entities = vision.get("entities", [])

        if entities:
            lines.append("")
            lines.append("Entities:")

            for entity in entities:
                lines.append(entity)

        keywords = vision.get("keywords", [])

        if keywords:
            lines.append("")
            lines.append("Keywords:")

            for keyword in keywords:
                lines.append(keyword)

        return "\n".join(lines)