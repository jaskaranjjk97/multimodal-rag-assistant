from retrieval.context_builder import ContextBuilder
from retrieval.models import RetrievalChunk


def test_context_builder_formats_chunk_with_metadata():
    chunk = RetrievalChunk(
        content="Acme Technologies generated ₹12 crore revenue in 2025.",
        source="data/raw/acme_multimodal_test.pdf",
        page_number=1,
        element_type="text",
        chunk_id="chunk-001",
    )

    builder = ContextBuilder()

    context = builder.build([chunk])

    assert "acme_multimodal_test.pdf" in context
    assert "Page: 1" in context
    assert "Type: text" in context
    assert "Chunk ID: chunk-001" in context
    assert "₹12 crore" in context


def test_context_builder_formats_multiple_chunks():
    chunks = [
        RetrievalChunk(
            content="First chunk",
            source="document.pdf",
            page_number=1,
            element_type="text",
            chunk_id="chunk-001",
        ),
        RetrievalChunk(
            content="Second chunk",
            source="document.pdf",
            page_number=2,
            element_type="table",
            chunk_id="chunk-002",
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(chunks)

    assert "First chunk" in context
    assert "Second chunk" in context
    assert "Page: 1" in context
    assert "Page: 2" in context
    assert "---" in context


def test_context_builder_returns_empty_string_for_no_chunks():
    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""