from pydantic import BaseModel


class Citation(BaseModel):
    source: str
    page_number: int | None = None
    element_type: str
    chunk_id: str