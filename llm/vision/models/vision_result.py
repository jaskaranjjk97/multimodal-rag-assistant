from pydantic import BaseModel, Field

class DataPoint(BaseModel):
    label: str # Label for the data point, e.g., "cat", "dog", "car", etc.
    value: str # value associated with the label, e.g., "A small domesticated carnivorous mammal", "A domesticated carnivorous mammal", etc.

class VisionResult(BaseModel):
    image_type: str # Type of the image, e.g., "JPEG", "PNG", etc.

    description: str # Description of the image content, e.g., "A cat sitting on a mat."

    data_points: list[DataPoint] = Field(default_factory=list) # List of data points extracted from the image, each containing a label and its associated value.

    entities: list[str] = Field(default_factory=list) # List of entities detected in the image, e.g., ["cat", "mat"].
    
    keywords: list[str] = Field(default_factory=list) # List of keywords associated with the image, e.g., ["pet", "animal", "feline"].