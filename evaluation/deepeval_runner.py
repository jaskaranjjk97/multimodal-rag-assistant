from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evaluation.dataset import EVALUATION_CASES
from llm.openai_llm_provider import OpenAILLMProvider
from rag.rag_service import RAGService
from retrieval.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from retrieval.reranking.lexical_reranker import LexicalReranker
from retrieval.vector_store.chroma_store import ChromaVectorStore


class DeepEvalRunner:

    def __init__(self):

        embedding_provider = OpenAIEmbeddingProvider()

        vector_store = ChromaVectorStore(
            embedding_provider=embedding_provider,
            persist_directory="data/chroma",
            collection_name="acme_documents",
        )

        llm_provider = OpenAILLMProvider(
            model="gpt-4o-mini"
        )

        reranker = LexicalReranker()

        self.rag_service = RAGService(
            vector_store=vector_store,
            llm_provider=llm_provider,
            reranker=reranker,
        )

    def build_test_case(
        self,
        question: str,
        expected_answer: str | None = None,
    ) -> LLMTestCase:

        execution = self.rag_service.execute(
            question=question,
            retrieval_k=10,
            final_k=2,
        )

        retrieval_context = [
            chunk.content
            for chunk in execution.retrieved_chunks
        ]

        return LLMTestCase(
            input=question,
            actual_output=execution.response.answer,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

    def evaluate_case(self, case):

        test_case = self.build_test_case(
            question=case.question,
            expected_answer=case.expected_answer,
        )

        metrics = [
            FaithfulnessMetric(
                threshold=0.7,
                include_reason=True,
            ),
            AnswerRelevancyMetric(
                threshold=0.7,
                include_reason=True,
            ),
            ContextualRelevancyMetric(
                threshold=0.7,
                include_reason=True,
            ),
            ContextualRecallMetric(
                threshold=0.7,
                include_reason=True,
            ),
        ]

        return evaluate(
            test_cases=[test_case],
            metrics=metrics,
        )

    def evaluate_dataset(self):

        results = []

        for case in EVALUATION_CASES:

            print("\n" + "=" * 70)
            print(f"QUESTION: {case.question}")
            print("=" * 70)

            result = self.evaluate_case(case)

            print(result)

            results.append(result)

        return results

if __name__ == "__main__":
    runner = DeepEvalRunner()
    runner.evaluate_dataset()