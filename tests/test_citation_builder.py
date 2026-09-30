from retrieval.citation_builder import CitationBuilder
from retrieval.models import RetrievalChunk


def test_citation_builder_creates_citations():

    chunks = [
        RetrievalChunk(
            content="Revenue was 10 crore.",
            source="acme.pdf",
            page_number=3,
            element_type="image",
            chunk_id="chunk-001",
        ),
        RetrievalChunk(
            content="Company information.",
            source="acme.pdf",
            page_number=1,
            element_type="text",
            chunk_id="chunk-002",
        ),
    ]

    builder = CitationBuilder()

    citations = builder.build(chunks)

    assert len(citations) == 2

    assert citations[0].source == "acme.pdf"
    assert citations[0].page_number == 3
    assert citations[0].element_type == "image"
    assert citations[0].chunk_id == "chunk-001"

    assert citations[1].page_number == 1


def test_citation_builder_returns_empty_list_for_no_chunks():

    builder = CitationBuilder()

    citations = builder.build([])

    assert citations == []