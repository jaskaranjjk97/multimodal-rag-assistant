from abc import ABC, abstractmethod

from retrieval.models import RetrievalChunk


class Reranker(ABC):
    """
    Abstract interface for reranking retrieved chunks.

    The vector store performs broad semantic retrieval.
    The reranker then reorders those candidates according
    to their relevance to the user's query.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievalChunk],
        top_k: int,
    ) -> list[RetrievalChunk]:
        """
        Rerank candidate chunks and return the most relevant ones.

        Args:
            query: User's question.
            chunks: Candidate chunks returned by initial retrieval.
            top_k: Number of chunks to return after reranking.

        Returns:
            Reranked list of RetrievalChunk objects.
        """
        raise NotImplementedError