from ingestion.loaders.pdf_loader import PDFLoader 

def test_pdf_loader():

    loader=PDFLoader()

    document=loader.load("data/raw/acme_multimodal_test.pdf")

    assert len(document) == 3
