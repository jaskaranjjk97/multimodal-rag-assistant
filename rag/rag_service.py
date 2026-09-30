import logging
import time

from rag.models.rag_execution import RAGExecution
from retrieval.citation_builder import CitationBuilder
from retrieval.context_builder import ContextBuilder
from retrieval.models.rag_response import RAGResponse
from retrieval.reranking.lexical_reranker import LexicalReranker


logger=logging.getLogger(__name__)

class RAGService:

    def __init__(
        self,
        vector_store,
        llm_provider,
        reranker=None,
    ):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.reranker = reranker or LexicalReranker()

        self.context_builder = ContextBuilder()
        self.citation_builder = CitationBuilder()

    def execute(
        self,
        question: str,
        retrieval_k: int = 10,
        final_k: int = 3,
    ) -> RAGExecution:

        execution_start=time.perf_counter()

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if retrieval_k <= 0:
            raise ValueError(
                "retrieval_k must be greater than zero."
            )

        if final_k <= 0:
            raise ValueError(
                "final_k must be greater than zero."
            )

        retrieval_start=time.perf_counter()
        
        chunks = self.vector_store.similarity_search(
            query=question,
            k=retrieval_k,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start)*1000

        logger.info("rag_retrieval_completed | retrieved_chunks=%s | latency_ms= %.2f",
                    len(chunks),
                     retrieval_latency_ms )


        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=chunks,
            top_k=final_k,
        )

        logger.info("rag_reranking_completed | selected_chunks=%s",
                    len(reranked_chunks))


        context = self.context_builder.build(
            reranked_chunks
        )

        llm_started=time.perf_counter()

        generation = self.llm_provider.generate(
            question=question,
            context=context,
        )

        llm_latency_ms=(time.perf_counter() - llm_started)*1000

        logger.info("rag_llm_completed | latency_ms=%.2f",
                    llm_latency_ms)

        retrieved_chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in reranked_chunks
        }

        supported_chunks = []

        for chunk_id in generation.citation_chunk_ids:

            chunk = retrieved_chunks_by_id.get(
                chunk_id
            )

            if chunk is not None:
                supported_chunks.append(chunk)

        citations = self.citation_builder.build(
            supported_chunks
        )

        response = RAGResponse(
            answer=generation.answer,
            citations=citations,
        )

        execution_latency_ms =(time.perf_counter() - execution_start)*1000

        logger.info("rag_execution_completed | latency_ms=%.2f",
                    execution_latency_ms)

        return RAGExecution(
            response=response,
            retrieved_chunks=reranked_chunks,
            context=context,
        )

    def ask(
        self,
        question: str,
        retrieval_k: int = 10,
        final_k: int = 3,
    ) -> RAGResponse:

        execution = self.execute(
            question=question,
            retrieval_k=retrieval_k,
            final_k=final_k,
        )

        return execution.response