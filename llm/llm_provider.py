from abc import ABC, abstractmethod
from retrieval.models.rag_response import RAGResponse


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
    ) -> RAGResponse:
        """Generate an answer using the supplied context."""
        raise NotImplementedError