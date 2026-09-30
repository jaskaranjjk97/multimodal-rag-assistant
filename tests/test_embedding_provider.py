# pytest → testing framework
# monkeypatch → temporarily change configuration
# MagicMock → fake the OpenAI API response


from unittest.mock import MagicMock

import pytest

from retrieval.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider


def test_provider_requires_api_key(monkeypatch):

    monkeypatch.setattr(
        "retrieval.embeddings.openai_embedding_provider.settings.openai_api_key",
        None,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


def test_embed_text(monkeypatch):

    monkeypatch.setattr(
        "retrieval.embeddings.openai_embedding_provider.settings.openai_api_key",
        "test-key",
    )

    provider = OpenAIEmbeddingProvider()

    mock_response = MagicMock()

    mock_response.data = [
        MagicMock(
            embedding=[0.1, 0.2, 0.3]
        )
    ]

    provider.client.embeddings.create = MagicMock(
        return_value=mock_response
    )

    result = provider.embed_text(
        "Acme Technologies revenue"
    )

    assert result == [0.1, 0.2, 0.3]

    provider.client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="Acme Technologies revenue",
    )


def test_embed_documents(monkeypatch):

    monkeypatch.setattr(
        "retrieval.embeddings.openai_embedding_provider.settings.openai_api_key",
        "test-key",
    )

    provider = OpenAIEmbeddingProvider()

    mock_response = MagicMock()

    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    provider.client.embeddings.create = MagicMock(
        return_value=mock_response
    )

    result = provider.embed_documents(
        [
            "Acme revenue",
            "Acme products",
        ]
    )

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    provider.client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=[
            "Acme revenue",
            "Acme products",
        ],
    )


def test_empty_text_fails(monkeypatch):

    monkeypatch.setattr(
        "retrieval.embeddings.openai_embedding_provider.settings.openai_api_key",
        "test-key",
    )

    provider = OpenAIEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed_text("   ")


def test_empty_document_list_returns_empty(monkeypatch):

    monkeypatch.setattr(
        "retrieval.embeddings.openai_embedding_provider.settings.openai_api_key",
        "test-key",
    )

    provider = OpenAIEmbeddingProvider()

    result = provider.embed_documents([])

    assert result == []