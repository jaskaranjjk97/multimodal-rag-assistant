# NORMALIZATION : Now our downstream system doesn't care where the information came from.

from enum import Enum
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

class DocumentElement(BaseModel):
    element_type: ElementType
    content: str
    source:str
    page_number: int | None = None 
    element_id : str | None = None 
    bbox: tuple[float, float, float, float] | None = None  # Bounding box for positioning (x0, y0, x1, y1)
    metadata: dict = Field(default_factory=dict)



