from abc import ABC, abstractmethod
from llm.vision.models import VisionResult

class VisionProvider(ABC):
     @abstractmethod
     def describe_image(self, image_path: str) -> VisionResult:
        """
        Generate a semantic description of an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Textual description of the image.
        """
        raise NotImplementedError