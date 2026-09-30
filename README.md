# Multimodal AI Knowledge Assistant

A production-oriented multimodal Retrieval-Augmented Generation (RAG) system for querying PDF documents containing text, tables, and images.

The system extracts and normalizes document content, enriches images using a vision model, converts heterogeneous document elements into retrieval representations, indexes them in Chroma, retrieves and reranks relevant evidence, and generates grounded answers with source citations.

---

## Features

- Multimodal PDF ingestion
  - Text extraction
  - Table extraction
  - Image extraction
- Image understanding using an OpenAI vision model
- Structured document representations
- Type-aware chunking
- OpenAI embeddings
- Persistent Chroma vector store
- Similarity retrieval
- Lexical reranking
- Evidence/context selection
- Citation-aware RAG generation
- Structured LLM output using Pydantic
- FastAPI REST API
- Request IDs and latency logging
- DeepEval-based RAG evaluation
- Dockerized API
- Persistent Chroma storage using a host-mounted directory
- Automated Docker API smoke tests

---

## Architecture

```text
                         ┌──────────────────────┐
                         │       PDF Input      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    PDF Processing    │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
              Text               Tables             Images
                 │                  │                  │
                 │                  │                  ▼
                 │                  │          Vision Enrichment
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                         Document Elements
                                    │
                                    ▼
                    Retrieval Representations
                                    │
                                    ▼
                              Chunking
                                    │
                                    ▼
                         OpenAI Embeddings
                                    │
                                    ▼
                         Persistent Chroma
                                    │
                           User Question
                                    │
                                    ▼
                              Retrieval
                                    │
                                    ▼
                              Reranking
                                    │
                                    ▼
                         Context Construction
                                    │
                                    ▼
                            LLM Generation
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                       Answer               Citations

## Project Overview

This project demonstrates an end-to-end multimodal RAG pipeline designed to answer questions from structured and unstructured information contained in PDF documents.

The system does not treat every PDF element as plain text. Text, tables, and images are extracted as different document element types and processed according to their characteristics.

Images are enriched using a vision model before being converted into retrieval representations. This allows visual information such as charts and their data points to participate in semantic retrieval.

The retrieved evidence is then reranked before being passed to the language model. The generated response contains both the answer and references to the supporting retrieval chunks.

---

## Project Structure

```text
Project-01-rag-assistant/
│
├── app/
│   ├── config.py
│   └── main.py
│
├── ingestion/
│   ├── models/
│   │   └── document_element.py
│   ├── loaders/
│   │   └── pdf_loader.py
│   ├── extractors/
│   │   ├── text_extractor.py
│   │   ├── table_extractor.py
│   │   ├── image_extractor.py
│   │   └── text_normalizer.py
│   ├── enrichers/
│   │   └── image_enricher.py
│   └── processors/
│       ├── pdf_processor.py
│       └── document_enricher.py
│
├── retrieval/
│   ├── representation.py
│   ├── chunker.py
│   ├── context_builder.py
│   ├── citation_builder.py
│   ├── models/
│   ├── embeddings/
│   ├── reranking/
│   ├── evidence/
│   └── vector_store/
│
├── llm/
│   ├── llm_provider.py
│   ├── openai_llm_provider.py
│   ├── models/
│   └── vision/
│
├── rag/
│   ├── rag_service.py
│   └── models/
│
├── evaluation/
│   ├── dataset.py
│   └── deepeval_runner.py
│
├── tests/
│   ├── test_*.py
│   └── smoke_test_api.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
│
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── .dockerignore
```

---

## Tech Stack

| Component        | Technology                      |
| ---------------- | ------------------------------- |
| Language         | Python 3.12                     |
| API              | FastAPI                         |
| PDF Processing   | PyMuPDF                         |
| LLM              | OpenAI                          |
| Vision Model     | OpenAI GPT-4o-mini              |
| Embeddings       | OpenAI `text-embedding-3-small` |
| Vector Database  | Chroma                          |
| Validation       | Pydantic                        |
| RAG Evaluation   | DeepEval                        |
| Testing          | pytest                          |
| Containerization | Docker                          |

---

## Setup

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

The API key should not be committed to source control.

---

## Running the Project Locally

### Start the FastAPI server

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Health endpoint:

```text
GET /health
```

Query endpoint:

```text
POST /query
```

Example request:

```json
{
  "question": "What was Acme Technologies' revenue in 2024?"
}
```

The API returns the generated answer together with the supporting citations.

---

## Document Ingestion and Indexing

The project includes a controlled multimodal test document:

```text
data/raw/acme_multimodal_test.pdf
```

The document contains:

* Text describing Acme Technologies
* A product revenue and growth table
* A revenue trend chart

The indexing pipeline processes these elements separately, creates retrieval representations, chunks the content, generates embeddings, and stores the resulting chunks in the Chroma collection:

```text
acme_documents
```

The current test document produces **5 retrieval chunks**.

---

## RAG Retrieval Pipeline

For each user question, the system performs the following steps:

```text
User Question
      │
      ▼
Vector Similarity Retrieval
      │
      ▼
Retrieved Chunks
      │
      ▼
Lexical Reranking
      │
      ▼
Final Evidence Selection
      │
      ▼
Context Construction
      │
      ▼
LLM Generation
      │
      ▼
Answer + Citations
```

The system separates initial retrieval from final evidence selection so that the language model receives a smaller, more relevant context rather than the entire retrieved result set.

---

## Citations

Each generated citation contains:

* Source document
* Page number
* Element type
* Retrieval chunk ID

Example:

```json
{
  "source": "acme_multimodal_test.pdf",
  "page_number": 3,
  "element_type": "image",
  "chunk_id": "..."
}
```

The LLM is instructed to return only citation chunk IDs that exist in the supplied context. The RAG service then resolves those IDs against the reranked chunks before constructing the final citation response.

## Docker

The API can be packaged and run as a Docker container.

### Build the Docker image

The project includes a separate runtime dependency file for the container:

```powershell
docker build -t rag-assistant:1.5 .
```

### Run the container

The application requires the OpenAI API key from the `.env` file and uses the host-mounted Chroma directory for persistent vector storage.

```powershell
docker run --rm `
  -p 8000:8000 `
  --env-file .env `
  -v "${PWD}\data\chroma:/app/data/chroma" `
  --name rag-assistant-api `
  rag-assistant:1.5
```

The API is then available at:

```text
http://localhost:8000
```

### Chroma Persistence

Chroma is persisted outside the container using a Docker bind mount:

```text
Host:
data/chroma/

        │
        │ Docker bind mount
        ▼

Container:
/app/data/chroma/
```

This keeps the vector database available across container restarts and avoids rebuilding the indexed collection every time the API container starts.

The current Chroma collection used by the API is:

```text
acme_documents
```

---

## API Smoke Tests

The project includes a small API smoke-test script:

```text
tests/smoke_test_api.py
```

The smoke tests verify:

1. The `/health` endpoint is available.
2. The `/query` endpoint successfully performs a RAG query.
3. The response contains an answer.
4. The response contains citations.
5. A known answer can be retrieved from the indexed test document.

Run the smoke tests while the Docker container is running:

```powershell
python tests/smoke_test_api.py
```

Expected output:

```text
Docker smoke tests passed.
```

---

## Evaluation

The RAG pipeline is evaluated using DeepEval.

The current evaluation dataset contains three representative questions covering:

* Text-based factual retrieval
* Table-based retrieval
* Image/chart-based retrieval

The evaluation uses the following metrics:

| Metric               | Purpose                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| Faithfulness         | Measures whether the answer is supported by the retrieved context                             |
| Answer Relevancy     | Measures whether the answer addresses the question                                            |
| Contextual Relevancy | Measures whether retrieved context is relevant to the question                                |
| Contextual Recall    | Measures whether the retrieved context contains the information needed to answer the question |

Each metric currently uses a threshold of:

```text
0.70
```

### Latest Evaluation Result

The current evaluation run produced:

```text
Evaluation cases passed: 3 / 3
Metrics passed:          12 / 12
```

All three evaluation cases passed all four configured metrics.

The evaluation model and RAG generation model are separate:

```text
RAG generation model:
gpt-4o-mini

Evaluation model:
gpt-5.4
```

This separation allows the evaluation process to assess the generated RAG responses independently.

---

## Testing

The project uses pytest for automated testing.

Run the test suite with:

```powershell
pytest -q
```

The tests cover components including:

* PDF loading
* Text extraction
* Table extraction
* Image extraction
* Document processing
* Retrieval representations
* Chunking
* Embeddings
* Chroma vector storage
* Reranking
* Context construction
* Citation construction
* Evaluation dataset handling

The API smoke test is run separately because it requires the Dockerized API to be running.

---

## Observability

The API includes lightweight application observability without introducing a separate monitoring platform.

Each request receives a request ID, which is returned through the:

```text
X-Request-ID
```

response header.

The RAG service also records latency for major stages:

```text
Request
   │
   ├── Retrieval latency
   │
   ├── Reranking
   │
   ├── LLM generation latency
   │
   └── Total execution latency
```

These logs provide basic visibility into where time is being spent during RAG execution.

## Engineering Decisions

The project intentionally focuses on components that solve concrete RAG problems rather than adding infrastructure for its own sake.

### Type-aware document processing

Text, tables, and images are represented as different document element types. This prevents the ingestion pipeline from treating visually or structurally different content as identical plain text.

### Retrieval representations

Document elements are converted into retrieval-oriented representations before chunking and embedding.

For example, image content is enriched with:

* Image type
* Description
* Extracted data points
* Entities
* Keywords

Only the information useful for retrieval is included in the final image retrieval chunk.

### Type-aware chunking

Different document types use different chunking strategies.

* Text uses recursive character-based chunking.
* Tables remain coherent as complete table chunks.
* Images keep their description and data points together as a single retrieval unit.

This avoids arbitrarily splitting structured information that should remain together.

### Retrieval and reranking

The system uses a two-stage retrieval approach:

```text
Vector Similarity Retrieval
            │
            ▼
       Candidate Set
            │
            ▼
      Lexical Reranking
            │
            ▼
     Final Evidence Set
```

Initial retrieval provides a broader candidate set, while reranking selects the most relevant chunks for the generation step.

### Structured generation

The LLM response is parsed into a Pydantic model containing:

* Generated answer
* Citation chunk IDs

This provides a structured interface between the LLM and the application layer instead of relying on unstructured text parsing.

### Citation validation

Citation IDs returned by the LLM are resolved against the chunks actually supplied to the model.

Invalid or unsupported citation IDs are not converted into citations.

---

## Current Test Results

The current controlled test document is:

```text id="4my51f"
data/raw/acme_multimodal_test.pdf
```

It contains information about:

* Company revenue
* Product revenue and growth
* Annual revenue trends
* A revenue chart

Current indexing state:

```text id="yk7pca"
Retrieval chunks: 5
Chroma collection: acme_documents
```

Current DeepEval result:

```text id="l7hy7t"
Evaluation cases: 3 / 3 passed
Metrics:         12 / 12 passed
Threshold:       0.70
```

The Dockerized API has also been verified using the API smoke tests.

---

## Current Limitations

This project is intentionally scoped as a production-oriented portfolio implementation rather than a fully deployed multi-tenant production service.

Current limitations include:

* Chroma is running as a local persistent vector store.
* Vector persistence currently uses a host-mounted Docker directory.
* Authentication and authorization are not implemented.
* There is no multi-user or tenant isolation layer.
* Document indexing is currently a separate workflow from the API.
* The evaluation dataset is intentionally small.
* The system currently uses a lexical reranker rather than a dedicated learned reranking model.
* There is no cloud deployment configuration.
* The current API does not provide document upload and indexing endpoints.

These are deliberate scope boundaries rather than hidden dependencies.

---

## Potential Future Improvements

Possible extensions include:

* Add document upload and asynchronous indexing.
* Introduce authentication and tenant isolation.
* Move vector storage to a managed vector database when scale requires it.
* Add a dedicated cross-encoder or API-based reranker.
* Expand the evaluation dataset with more complex and adversarial questions.
* Add automated evaluation to CI/CD.
* Add tracing and metrics through an observability platform when operational scale justifies it.
* Add document and citation previews to the client application.
* Add support for additional document formats beyond PDF.

---

## Project Goal

The goal of this project is to demonstrate the engineering required to build a reliable multimodal RAG system beyond a basic "PDF → embeddings → LLM" implementation.

The project focuses on:

* Structured multimodal ingestion
* Retrieval-aware document representation
* Type-aware chunking
* Retrieval and reranking
* Evidence selection
* Grounded generation
* Citation handling
* Evaluation
* API design
* Dockerization
* Basic observability
* Persistent vector storage

The result is a modular RAG foundation that can be extended toward larger document collections and production deployment as additional requirements emerge.

---

## Project Status

**Status: Functional end-to-end prototype**

The current implementation successfully supports:

```text id="7c9y2a"
PDF
 │
 ├── Text
 ├── Tables
 └── Images
       │
       ▼
Vision Enrichment
       │
       ▼
Retrieval Representations
       │
       ▼
Chunking
       │
       ▼
Embeddings
       │
       ▼
Chroma
       │
       ▼
Retrieval + Reranking
       │
       ▼
Context
       │
       ▼
LLM
       │
       ▼
Answer + Citations
       │
       ▼
DeepEval
```

The project is suitable as a portfolio demonstration of a complete multimodal RAG architecture and provides a foundation for further production hardening.
