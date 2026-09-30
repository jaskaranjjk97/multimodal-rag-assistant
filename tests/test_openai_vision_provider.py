from pathlib import Path

from llm.vision.openai_vision_provider import OpenAIVisionProvider


def test_openai_vision_provider_requires_api_key(monkeypatch):

    monkeypatch.setattr(
        "llm.vision.openai_vision_provider.settings.openai_api_key",
        None,
    )

    try:
        OpenAIVisionProvider()
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)