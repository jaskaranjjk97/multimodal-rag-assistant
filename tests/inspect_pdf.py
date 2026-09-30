import pymupdf


PDF_PATH = "data/raw/acme_multimodal_test.pdf"


document = pymupdf.open(PDF_PATH)

page = document[1]

tables = page.find_tables()

print(f"Number of tables: {len(tables.tables)}")

for table_number, table in enumerate(tables.tables, start=1):

    print(f"\nTABLE {table_number}")

    extracted = table.extract()

    for row in extracted:
        print(row)