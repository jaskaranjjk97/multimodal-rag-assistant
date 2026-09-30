import logging
import time
import uuid
from fastapi import FastAPI,Request
from pydantic import BaseModel, Field

from llm.openai_llm_provider import OpenAILLMProvider
from retrieval.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from retrieval.vector_store.chroma_store import ChromaVectorStore
from rag.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger=logging.getLogger(__name__)


app = FastAPI(
    title="Multimodal AI Knowledge Assistant",
    description="RAG-based document question answering API",
    version="1.0.0",
)

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "request_started | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "request_completed | request_id=%s | status=%s | latency_ms=%.2f",
            request_id,
            response.status_code,
            elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "request_failed | request_id=%s | latency_ms=%.2f",
            request_id,
            elapsed_ms,
        )

        raise


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the indexed documents.",
    )


class CitationResponse(BaseModel):
    source: str
    page_number: int | None
    element_type: str
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


# Initialize RAG dependencies
embedding_provider = OpenAIEmbeddingProvider()

vector_store = ChromaVectorStore(
    embedding_provider=embedding_provider,
    persist_directory="data/chroma",
    collection_name="acme_documents",
)

llm_provider = OpenAILLMProvider()

rag_service = RAGService(
    vector_store=vector_store,
    llm_provider=llm_provider,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "rag-assistant",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    result = rag_service.ask(
        question=request.question,
        retrieval_k=10,
        final_k=2
    )

    response = QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                source=citation.source,
                page_number=citation.page_number,
                element_type=citation.element_type,
                chunk_id=citation.chunk_id,
            )
            for citation in result.citations
        ],
    )
    return response
