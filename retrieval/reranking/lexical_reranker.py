import re

from retrieval.models import RetrievalChunk
from retrieval.reranking.reranker import Reranker


class LexicalReranker(Reranker):
    """
    Reranks retrieved chunks using weighted lexical overlap.

    Important query terms such as numbers, years, and longer
    words receive higher weights than common words.
    """

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "was",
        "what",
        "which",
        "with",
    }

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalChunk],
        top_k: int,
    ) -> list[RetrievalChunk]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not chunks:
            return []

        query_tokens = self._tokenize(query)

        scored_chunks = [
            (
                chunk,
                self._score(
                    query_tokens=query_tokens,
                    content=chunk.content,
                ),
            )
            for chunk in chunks
        ]

        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            chunk
            for chunk, _score in scored_chunks[:top_k]
        ]

    def _score(
        self,
        query_tokens: list[str],
        content: str,
    ) -> float:

        if not query_tokens:
            return 0.0

        content_tokens = set(
            self._tokenize(content)
        )

        if not content_tokens:
            return 0.0

        score = 0.0
        total_weight = 0.0

        for token in query_tokens:

            weight = self._token_weight(token)

            total_weight += weight

            if token in content_tokens:
                score += weight

        if total_weight == 0:
            return 0.0

        return score / total_weight

    def _tokenize(self, text: str) -> list[str]:

        tokens = re.findall(
            r"\b[\w₹]+\b",
            text.lower(),
        )

        return [
            token
            for token in tokens
            if token not in self.STOP_WORDS
        ]

    @staticmethod
    def _token_weight(token: str) -> float:
        """
        Give more importance to specific tokens.

        Numbers and years are highly discriminative in
        factual/document questions.
        """

        if token.isdigit():
            return 3.0

        if len(token) >= 8:
            return 1.5

        return 1.0