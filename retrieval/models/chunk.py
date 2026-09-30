from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    content: str
    source: str
    page_number: int | None = None
    element_type: str
    chunk_id: str
    metadata: dict = Field(default_factory=dict)