from openai import OpenAI

from app.config import settings
from retrieval.embeddings.embeddings_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.model = model

    def embed_text(
        self,
        text: str,
    ) -> list[float]:

        if not text.strip():
            raise ValueError(
                "Cannot create an embedding for empty text."
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Cannot create embeddings for empty text."
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]