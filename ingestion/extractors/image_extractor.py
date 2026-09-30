import pymupdf
from ingestion.models.document_element import DocumentElement, ElementType
from pathlib import Path

class ImageExtractor:
    def extract(self,
                document:pymupdf.Document,
                source:str,
                output_directory: str ="data/processed/images") -> list[DocumentElement]:

        output_path=Path(output_directory)
        output_path.mkdir(parents=True,exist_ok=True)

        elements=[]

        for page_number,page in enumerate(document,start=1):
            images=page.get_images(full=True)

            for image_index,image in enumerate(images,start=1):
                xref=image[0]

                image_data=document.extract_image(xref)

                image_bytes =image_data["image"]
                image_extension =image_data["ext"]

                image_filename = (
                    f"{Path(source).stem}"
                    f"_page_{page_number}"
                    f"_image_{image_index}"
                    f".{image_extension}"
                )
                image_file = (output_path / image_filename)

                image_file.write_bytes(image_bytes)

                element=DocumentElement(
                    element_type=ElementType.IMAGE,
                    content=str(image_file),
                    source=source,
                    page_number=page_number,
                    metadata={"image_index": image_index,
                              "extension": image_extension,
                              "xref": xref,
                              },
                )
                elements.append(element)
        return elements
    




