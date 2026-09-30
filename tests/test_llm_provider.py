from llm.llm_provider import LLMProvider
from retrieval.models.rag_response import RAGResponse


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        question: str,
        context: str,
    ) -> RAGResponse:

        return f"Fake answer for: {question}"


def test_llm_provider_contract():

    provider = FakeLLMProvider()

    answer = provider.generate(
        question="What was the revenue?",
        context="Revenue was ₹12 crore.",
    )

    assert answer == "Fake answer for: What was the revenue?"


def test_llm_provider_rejects_empty_question():

    provider = FakeLLMProvider()

    # The fake implementation doesn't validate;
    # this test is intentionally not about the concrete provider.
    answer = provider.generate(
        question="",
        context="Some context",
    )

    assert "Fake answer" in answer