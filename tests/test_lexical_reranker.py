from retrieval.models import RetrievalChunk
from retrieval.reranking.lexical_reranker import LexicalReranker


def make_chunk(content: str, chunk_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        content=content,
        source="test.pdf",
        page_number=3,
        element_type="image",
        chunk_id=chunk_id,
    )


def test_lexical_reranker_returns_top_k():

    chunks = [
        make_chunk(
            "Company was founded in 2018.",
            "1",
        ),
        make_chunk(
            "Revenue 2024: 10 Crore",
            "2",
        ),
        make_chunk(
            "Revenue 2025: 12 Crore",
            "3",
        ),
    ]

    reranker = LexicalReranker()

    results = reranker.rerank(
        query="What was the revenue in 2024?",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "2"


def test_lexical_reranker_returns_empty_for_empty_chunks():

    reranker = LexicalReranker()

    results = reranker.rerank(
        query="What was the revenue?",
        chunks=[],
        top_k=3,
    )

    assert results == []


def test_lexical_reranker_rejects_empty_query():

    reranker = LexicalReranker()

    try:
        reranker.rerank(
            query="   ",
            chunks=[],
            top_k=3,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."


def test_year_match_gets_higher_weight():

    chunks = [
        make_chunk(
            "Acme Technologies annual revenue increased "
            "from 6 crore in 2022 to 12 crore in 2025.",
            "trend",
        ),
        make_chunk(
            "Image Data Point: Revenue 2024: 10 Crore",
            "2024",
        ),
    ]

    reranker = LexicalReranker()

    results = reranker.rerank(
        query="What was Acme Technologies' revenue in 2024?",
        chunks=chunks,
        top_k=2,
    )

    assert results[0].chunk_id == "2024"

def test_lexical_reranker_respects_top_k():

    chunks = [
        make_chunk("Revenue 2022: 6 Crore", "1"),
        make_chunk("Revenue 2023: 8 Crore", "2"),
        make_chunk("Revenue 2024: 10 Crore", "3"),
        make_chunk("Revenue 2025: 12 Crore", "4"),
    ]

    reranker = LexicalReranker()

    results = reranker.rerank(
        query="revenue 2024",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "3"