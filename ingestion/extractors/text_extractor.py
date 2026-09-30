import pymupdf # importing the PyMuPDF library for PDF processing 

from ingestion.models.document_element import DocumentElement, ElementType 
from ingestion.extractors.text_normalizer import TextNormalizer

class TextExtractor:

    def extract(self,document:pymupdf.Document,source:str)-> list[DocumentElement]:
        elements=[]

        for page_number,page in enumerate(document,start=1):
            text=page.get_text("text").strip()
            text=TextNormalizer.normalize(text)

            if not text:
                continue
            element=DocumentElement(
                element_type=ElementType.TEXT,
                content=text,
                source=source,
                page_number=page_number
            )
            elements.append(element)
        return elements