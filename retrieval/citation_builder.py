from retrieval.models import RetrievalChunk
from retrieval.models.citation import Citation


class CitationBuilder:

    def build(
        self,
        chunks: list[RetrievalChunk],
    ) -> list[Citation]:

        citations = []

        for chunk in chunks:
            citations.append(
                Citation(
                    source=chunk.source,
                    page_number=chunk.page_number,
                    element_type=chunk.element_type,
                    chunk_id=chunk.chunk_id,
                )
            )

        return citations