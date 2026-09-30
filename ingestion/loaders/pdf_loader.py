from pathlib import Path

import pymupdf

class PDFLoader:
    def  load(self,file_path:str):
        path=Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"The Provided file is not a PDF : {file_path}. Please provide a PDF file.")
        
        document=pymupdf.open(file_path)
        return document
