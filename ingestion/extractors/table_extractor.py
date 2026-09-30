import pymupdf

from ingestion.models.document_element import DocumentElement, ElementType
from ingestion.extractors.text_normalizer import TextNormalizer

class TableExtractor:
    def extract(self, document: pymupdf.Document, source: str) -> list[DocumentElement]:
        elements=[]

        for page_number, page in enumerate(document, start=1):
            tables=page.find_tables()
            for table in tables.tables:

                row = table.extract()
                if not row:
                    continue
                content = self._table_to_text(row)

                element = DocumentElement(
                    element_type=ElementType.TABLE,
                    content=content,
                    source=source,
                    page_number=page_number,
                    metadata={"rows": len(row),
                              "columns": len(row[0]),
                              },
                )
                elements.append(element)
        return elements
    
    @staticmethod
    def _table_to_text(rows: list[list[str | None]]) -> str:
        cleaned_rows = []

        for row in rows:
            cleaned_row=[TextNormalizer.normalize(cell.strip()) if cell else"" for cell in row]
            cleaned_rows.append(
                " | ".join(cleaned_row)
            )

        return "\n".join(cleaned_rows)