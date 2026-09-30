from pathlib import Path

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.extractors.image_extractor import ImageExtractor
from ingestion.models.document_element import ElementType


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


def test_image_extractor():

    loader = PDFLoader()
    document = loader.load(PDF_PATH)

    extractor = ImageExtractor()

    elements = extractor.extract(
        document=document,
        source=PDF_PATH,
    )

    assert len(elements) > 0

    image = elements[0]

    assert image.element_type == ElementType.IMAGE
    assert image.page_number == 3

    image_path = Path(image.content)

    assert image_path.exists()
    assert image_path.is_file()

    assert image.metadata["image_index"] == 1
    assert image.metadata["extension"] in {
        "png",
        "jpeg",
        "jpg",
    }