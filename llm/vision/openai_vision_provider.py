import base64
from pathlib import Path

from openai import OpenAI

from app.config import settings
from llm.vision.models.vision_result import VisionResult
from llm.vision.vision_provider import VisionProvider

import mimetypes


class OpenAIVisionProvider(VisionProvider):

    def __init__(self, model: str = "gpt-4o-mini"):

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def describe_image(self, image_path: str) -> VisionResult:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image_bytes = path.read_bytes()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(path.name)

        if mime_type is None:
          raise ValueError(
             f"Unsupported image format: {path.suffix}"
            )

        prompt = """
Analyze this image for a multimodal RAG system.

Return ONLY valid JSON with exactly these fields:

{
    "image_type": "string",
    "description": "string",
    "data_points": [
        {
            "label": "string",
            "value": "string"
        }
    ],
    "entities": ["string"],
    "keywords": ["string"]
}

Instructions:

1. Identify the type of image.
2. Describe the important information shown.
3. Extract numerical data points when they can be read reliably.
4. Do NOT invent or estimate numerical values.
5. If an exact value cannot be determined, omit that data point.
6. Identify important entities such as company names.
7. Provide useful keywords for retrieval.
"""

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime_type};base64,{base64_image}"
                            ),
                        },
                    ],
                }
            ],
            text_format=VisionResult,
        )

        return response.output_parsed