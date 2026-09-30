from pydantic import BaseModel, Field


class RAGGeneration(BaseModel):
    answer: str
    citation_chunk_ids: list[str] = Field(default_factory=list)