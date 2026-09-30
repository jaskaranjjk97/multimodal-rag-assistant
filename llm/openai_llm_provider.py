from openai import OpenAI

from app.config import settings
from llm.llm_provider import LLMProvider
from llm.models.rag_generation import RAGGeneration


class OpenAILLMProvider(LLMProvider):

    def __init__(self, model: str = "gpt-4o-mini"):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.model = model

    def generate(
        self,
        question: str,
        context: str,
    ) -> RAGGeneration:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not context.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Preserve important source formatting such as currency symbols, units, percentages, dates, and numerical values when answering.

The context contains retrieved chunks. Each chunk has a Chunk ID.

Rules:

1. Do not use outside knowledge.
2. If the answer cannot be found in the context, say:
   "I could not find the answer in the provided documents."
3. Be concise and factual.
4. Do not invent numbers or facts.
5. citation_chunk_ids MUST contain only Chunk IDs that appear in the provided context.
6. Include a chunk ID only if that chunk contains evidence directly supporting
   one or more claims in your answer.
7. Prefer the smallest set of chunks that fully supports the answer.
8. If one chunk fully supports the answer, cite only that chunk.
9. Do not cite a chunk merely because it is related to the topic.
10. Do not invent Chunk IDs.
11. Return the answer and the supporting Chunk IDs.
""" 

        user_prompt = f"""
Context:

{context}

Question:

{question}
"""

        response = self.client.responses.parse(
        model=self.model,
        instructions=system_prompt,
        input=user_prompt,
        text_format=RAGGeneration,
        )

        return response.output_parsed


 # we are using structured parsing instead of self.client.response.create()