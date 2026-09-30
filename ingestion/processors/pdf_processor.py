from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.extractors.text_extractor import TextExtractor
from ingestion.extractors.image_extractor import ImageExtractor
from ingestion.extractors.table_extractor import TableExtractor 
from ingestion.models.document_element import DocumentElement

class PDFProcessor:
    def __init__(self):
        self.loader = PDFLoader()
        self.text_extractor = TextExtractor()
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()

    def process(self,file_path:str)->list[DocumentElement]:
        document = self.loader.load(file_path)

        elements = []

        text_elements = self.text_extractor.extract(document=document, source=file_path)
        image_elements= self.image_extractor.extract(document=document, source=file_path)
        table_elements = self.table_extractor.extract(document=document, source=file_path)

        document.close()

        elements=(text_elements + table_elements + image_elements)

        elements.sort(key=lambda element: (element.page_number or 0, element.element_type.value))
        return elements