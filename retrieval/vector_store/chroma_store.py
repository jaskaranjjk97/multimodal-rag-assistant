from pathlib import Path

import chromadb

from retrieval.embeddings.embeddings_provider import EmbeddingProvider
from retrieval.models import RetrievalChunk


class ChromaVectorStore:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_directory: str = "data/chroma",  # Answers:Where should Chroma store its persistent database files? It's a physical filesystem location.

        collection_name: str = "rag_documents", # Which logical collection inside that Chroma database should I use? It's a logical name, not normally a filesystem directo
    ):
        self.embedding_provider = embedding_provider

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory    
        )  # basically telling the chroma db where to store the things

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        ) # what logic shoukd i use to systematically store in chromadb

    def add_chunks(
        self,
        chunks: list[RetrievalChunk], 
    ) -> None:

        if not chunks:
            return   

        texts = [
            chunk.content   
            for chunk in chunks
        ]  

        embeddings = (
            self.embedding_provider.embed_documents(texts)
        )

        self.collection.upsert(
            ids=[
                chunk.chunk_id
                for chunk in chunks
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk.source,
                    "page_number": (
                        chunk.page_number
                        if chunk.page_number is not None
                        else -1
                    ),
                    "element_type": chunk.element_type,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )



    def similarity_search(
        self,
        query: str,
        k: int = 5,
            ) -> list[RetrievalChunk]:

        if not query.strip():
            raise ValueError(
            "Query cannot be empty."
            )

        query_embedding = (
            self.embedding_provider.embed_text(query)
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
                ],
            )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []

        for document, metadata, chunk_id, distance in zip(
            documents,
            metadatas,
            ids,
            distances,
            ):
            page_number = metadata.get("page_number")

            if page_number == -1:
                page_number = None

            print(
                f"\nChunk: {chunk_id}"
                f"\nDistance: {distance}"
                f"\nPage: {page_number}"
                f"\nType: {metadata.get('element_type')}"
                f"\nContent: {document[:200]}"
                )

            chunks.append(
                RetrievalChunk(
                    content=document,
                    source=metadata["source"],
                    page_number=page_number,
                    element_type=metadata["element_type"],
                    chunk_id=chunk_id,
                    metadata=metadata,
                    )
                )

        return chunks