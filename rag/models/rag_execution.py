from pydantic import BaseModel, Field

from retrieval.models.rag_response import RAGResponse
from retrieval.models.chunk import RetrievalChunk


class RAGExecution(BaseModel):
    response: RAGResponse
    retrieved_chunks: list[RetrievalChunk] = Field(default_factory=list)
    context: str = ""