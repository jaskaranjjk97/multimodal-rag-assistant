from pydantic import BaseModel,Field

from retrieval.models import Citation

class RAGResponse(BaseModel):
    answer:str
    citations : list[Citation] = Field(default_factory=list)