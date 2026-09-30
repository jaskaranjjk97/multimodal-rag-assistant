from retrieval.chunker import RetrievalChunker
from retrieval.models import RetrievalDocument


def test_text_document_is_split_into_chunks():

    document = RetrievalDocument(
        content="A " * 300,
        source="test.pdf",
        page_number=1,
        element_type="text",
    )

    chunker = RetrievalChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.element_type == "text"
        assert chunk.source == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_id


def test_table_is_kept_as_one_chunk():

    document = RetrievalDocument(
        content=(
            "Product | Revenue | Growth\n"
            "AI Assistant | I5 Cr | 25%\n"
            "Analytics | I4 Cr | 18%"
        ),
        source="test.pdf",
        page_number=2,
        element_type="table",
    )

    chunker = RetrievalChunker()

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert "AI Assistant" in chunks[0].content
    assert "Analytics" in chunks[0].content


def test_image_representation_is_kept_as_one_coherent_chunk():

    document = RetrievalDocument(
        content=(
            "Image Type: bar chart\n"
            "\n"
            "Description: Revenue chart\n"
            "\n"
            "Data Points:\n"
            "2022 Revenue: 6\n"
            "2025 Revenue: 12"
        ),
        source="test.pdf",
        page_number=3,
        element_type="image",
    )

    chunker = RetrievalChunker()

    chunks = chunker.chunk(document)

    assert len(chunks) == 1

    content = chunks[0].content

    assert "Image Type: bar chart" in content
    assert "Description: Revenue chart" in content
    assert "2022 Revenue: 6" in content
    assert "2025 Revenue: 12" in content