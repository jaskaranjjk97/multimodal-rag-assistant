#A DocumentElement represents what we extracted from the document.

# A RetrievalDocument represents what we want the retrieval system to search.

from pydantic import BaseModel, Field

class RetrievalDocument(BaseModel):
    content:str
    source:str
    page_number:int | None = None
    element_type :str | None = None
    metadata: dict = Field(default_factory=dict)