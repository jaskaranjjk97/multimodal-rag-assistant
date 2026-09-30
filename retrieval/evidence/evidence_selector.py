from retrieval.chunker import RetrievalChunk

class EvidenceSelector:

    def select(
        self,
        chunks: list[RetrievalChunk],
        top_k: int,
    ) -> list[RetrievalChunk]:

        return chunks[:top_k]