from pathlib import Path

from llm.vision.openai_vision_provider import OpenAIVisionProvider


IMAGE_PATH = (
    "data/processed/images/"
    "acme_multimodal_test_page_3_image_1.png"
)


def main():
    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    provider = OpenAIVisionProvider()

    result = provider.describe_image(IMAGE_PATH)

    print("\n" + "=" * 60)
    print("VISION RESULT")
    print("=" * 60)
    
    print(f"Image type: {result.image_type}")
    print(f"Description: {result.description}")
    
    print("\nData points:")
    
    for point in result.data_points:
        print(f"  {point.label} → {point.value}")
    
    print("\nEntities:")
    for entity in result.entities:
        print(f"  - {entity}")
    
    print("\nKeywords:")
    for keyword in result.keywords:
        print(f"  - {keyword}")
if __name__ == "__main__":
    main()